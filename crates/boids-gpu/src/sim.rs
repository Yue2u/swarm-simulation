//! Simulation resources and compute pipelines: the ping-pong agent buffers, the spatial grid, the
//! parameter uniforms and the passes that drive them.
//!
//! # Frame order
//!
//! ```text
//! P0 clear_cells    cell_start[c] = EMPTY_CELL
//! P1 hash           keys[dst][i] = {cell_index(pos_i), i}, padding -> PAD_KEY
//! P2 sort           153 bitonic stages, one compute pass each, stage parameters in immediates
//! P3 build_ranges   cell_start/cell_end from the runs of equal keys
//! P4 integrate      one invocation per agent; the grid variant searches the 27 cells around it
//! ```
//!
//! P0..P3 are [`SimPipelines::record_grid_prep`], and P0..P4 together are
//! [`SimPipelines::record_step`]. The naive strategy skips P0..P3 entirely and runs its own P4.
//!
//! # The sorted keys always end up in `keys[0]`
//!
//! The bitonic sort ping-pongs between the two key buffers, one buffer per stage, because a stage
//! reads what the previous stage wrote and the two dispatches must not overlap. Whether the result
//! lands in `keys[0]` or `keys[1]` therefore depends on the parity of the stage count, which is
//! `m * (m + 1) / 2` for `padded_n = 2^m` - odd for 100k agents (m = 17), even for 4096 (m = 12).
//!
//! Rather than carry that parity through the frame and pick a bind group for every consumer, the
//! host removes it at startup: [`KeyPlan`] aims the hash pass at whichever buffer leaves the last
//! stage writing `keys[0]`. Both consumers of the sorted order - `build_ranges` and `integrate_grid`
//! - then read a fixed buffer through the fixed bind group 0, which is what keeps group 0 bound once
//! for the lifetime of the simulation.
//!
//! # Buffer contract
//!
//! | buffer        | size                        | usage                              |
//! |---------------|-----------------------------|------------------------------------|
//! | `boids[2]`    | `padded_n * 48` bytes       | STORAGE, COPY_SRC, COPY_DST        |
//! | `keys[2]`     | `padded_n * 8` bytes        | STORAGE, COPY_SRC                  |
//! | `cell_start`  | `num_cells * 4` bytes       | STORAGE, COPY_DST                  |
//! | `cell_end`    | `num_cells * 4` bytes       | STORAGE                            |
//! | `params`      | 144 bytes                   | UNIFORM, COPY_DST                  |
//! | `interaction` | 48 bytes                    | UNIFORM, COPY_DST                  |
//!
//! `padded_n` is `num_boids` rounded up to a power of two, because the bitonic sort can only sort a
//! power-of-two count. The padding agents have undefined positions and are excluded from the
//! neighbour search by the sort (their keys are pushed to `PAD_KEY`), and never integrated because
//! every pass gates on `i < num_boids`.
//!
//! # Why two bind groups
//!
//! Group 0 holds everything that is stable for the lifetime of the simulation. Group 1 holds the
//! pass's working pair, which changes shape between passes (agents for `integrate`, agents-to-keys
//! for `hash`, keys for the sort) and swaps every step, so exactly one bind group exists per shape
//! per parity and the host selects between them. Rebuilding a bind group per frame would work but
//! allocates, and at 60 fps that is a slow leak of driver objects.

use boids_core::config::{GridDims, SimConfig};
use boids_core::layout::{
    Boid, InteractionUniforms, KeyVal, SimParams, SortParams, EMPTY_CELL,
};

use crate::context::GpuContext;
use crate::profile::GpuProfiler;
use crate::transfer::upload_uniform;

/// Bytes per boid, asserted against the layout contract.
const BOID_SIZE: u64 = core::mem::size_of::<Boid>() as u64;
const KEYVAL_SIZE: u64 = core::mem::size_of::<KeyVal>() as u64;

/// Which neighbour search strategy a frame uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Strategy {
    /// All-pairs search on the GPU. Exact but O(N^2): intended for validation and for small swarms.
    /// The app uses this below a few thousand agents, where it is faster than the grid because it
    /// needs no sort.
    #[default]
    Naive,
    /// Spatial grid search over the 27 cells around each agent. Requires the hash, sort and range
    /// passes to have run for this frame; see [`SimPipelines::record_grid_prep`].
    Grid,
}

/// All GPU buffers that hold simulation state.
#[derive(Debug)]
pub struct SimResources {
    /// Ping-pong agent buffers.
    pub boids: [wgpu::Buffer; 2],
    /// Ping-pong sort key buffers.
    pub keys: [wgpu::Buffer; 2],
    /// First index of each cell's run in the sorted key array, or [`EMPTY_CELL`].
    pub cell_start: wgpu::Buffer,
    /// One past the last index of each cell's run.
    pub cell_end: wgpu::Buffer,
    /// Simulation parameters.
    pub params: wgpu::Buffer,
    /// Cursor interaction state.
    pub interaction: wgpu::Buffer,
    /// Number of live agents.
    pub num_boids: u32,
    /// `num_boids` rounded up to a power of two.
    pub padded_n: u32,
    /// Grid geometry, mirrored into `SimParams`.
    pub grid: GridDims,
    /// Index into `boids` / `keys` holding the current state.
    read: usize,
    /// The grid geometry the buffers were sized for, kept for the resize check.
    allocated_cells: u32,
}

impl SimResources {
    /// Allocates every buffer for a configuration.
    ///
    /// # Panics
    /// Panics if `num_boids` is zero or does not fit in a `u32`.
    #[must_use]
    pub fn new(ctx: &GpuContext, config: &SimConfig) -> Self {
        let num_boids = u32::try_from(config.num_boids).expect("num_boids must fit in u32");
        assert!(num_boids > 0, "a simulation needs at least one agent");
        let padded_n = num_boids.next_power_of_two();
        let num_cells = config.grid.num_cells();

        let storage = wgpu::BufferUsages::STORAGE;
        let make = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let boid_bytes = u64::from(padded_n) * BOID_SIZE;
        let key_bytes = u64::from(padded_n) * KEYVAL_SIZE;
        let cell_bytes = u64::from(num_cells) * 4;

        log::info!(
            "sim buffers: {num_boids} agents (padded {padded_n}), grid {:?} = {num_cells} cells, \
             boids {:.1} MiB x2, keys {:.1} MiB x2, cells {:.1} MiB x2",
            config.grid.dim,
            boid_bytes as f64 / (1u64 << 20) as f64,
            key_bytes as f64 / (1u64 << 20) as f64,
            cell_bytes as f64 / (1u64 << 20) as f64,
        );

        let resources = Self {
            boids: [
                make("boids[0]", boid_bytes, storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST),
                make("boids[1]", boid_bytes, storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST),
            ],
            keys: [
                make("keys[0]", key_bytes, storage | wgpu::BufferUsages::COPY_SRC),
                make("keys[1]", key_bytes, storage | wgpu::BufferUsages::COPY_SRC),
            ],
            cell_start: make(
                "cell_start",
                cell_bytes,
                storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            ),
            // `COPY_SRC` on both cell arrays is for the test suite, which reads the grid back to
            // check the invariants `integrate_grid` searches under. Nothing in the frame copies them.
            cell_end: make(
                "cell_end",
                cell_bytes,
                storage | wgpu::BufferUsages::COPY_SRC,
            ),
            // Sizes come from the structs, rounded up to the 16-byte multiple the uniform address
            // space requires. Hardcoding them would go stale on the next field addition, and the
            // symptom would be a bind group validation error rather than anything readable.
            params: make(
                "sim params",
                (core::mem::size_of::<SimParams>() as u64).next_multiple_of(16),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            interaction: make(
                "interaction",
                (core::mem::size_of::<InteractionUniforms>() as u64).next_multiple_of(16),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            num_boids,
            padded_n,
            grid: config.grid,
            read: 0,
            allocated_cells: num_cells,
        };

        // Mark every cell empty up front. This is not just tidiness: until the range-building pass
        // runs (day 2), a grid search must find no neighbours rather than read uninitialised memory.
        // With `cell_start` at zero, `integrate_grid` would happily treat keys[0..cell_end] as a
        // cell's contents, which is garbage rather than a missing flock.
        resources.clear_cell_starts(&ctx.queue);
        resources
    }

    /// Fills `cell_start` with [`EMPTY_CELL`].
    pub fn clear_cell_starts(&self, queue: &wgpu::Queue) {
        let n = self.allocated_cells as usize;
        // Chunked so that a large grid does not require one giant temporary allocation.
        const CHUNK: usize = 1 << 16;
        let chunk = vec![EMPTY_CELL; CHUNK.min(n)];
        let mut written = 0usize;
        while written < n {
            let count = CHUNK.min(n - written);
            queue.write_buffer(
                &self.cell_start,
                (written * 4) as u64,
                bytemuck::cast_slice(&chunk[..count]),
            );
            written += count;
        }
        // `cell_end` only has to be sane for cells that `cell_start` marks as non-empty, and the
        // range-building pass fully overwrites it, so it needs no initial value.
    }

    /// Uploads the per-frame simulation parameters.
    pub fn write_params(&self, queue: &wgpu::Queue, params: &SimParams) {
        upload_uniform(queue, &self.params, params);
    }

    /// Uploads the per-frame cursor interaction state.
    pub fn write_interaction(&self, queue: &wgpu::Queue, interaction: &InteractionUniforms) {
        upload_uniform(queue, &self.interaction, interaction);
    }

    /// The agent buffer holding the current state.
    #[must_use]
    pub fn read_buffer(&self) -> &wgpu::Buffer {
        &self.boids[self.read]
    }

    /// The agent buffer being written this step.
    #[must_use]
    pub fn write_buffer(&self) -> &wgpu::Buffer {
        &self.boids[1 - self.read]
    }

    /// The key buffer holding the sorted keys, i.e. the one bind group 0 binds.
    ///
    /// [`KeyPlan`] is what guarantees the sort leaves the result here, so this is a constant and not
    /// a function of the frame's parity. It exists for tests and for the profiler; the passes read
    /// the buffer through their bind group.
    #[must_use]
    pub fn sorted_keys(&self) -> &wgpu::Buffer {
        &self.keys[0]
    }

    /// Index of the current read buffer, needed to select the matching agent bind group.
    #[must_use]
    pub const fn read_index(&self) -> usize {
        self.read
    }

    /// Flips the ping-pong parity after a completed step.
    pub const fn swap(&mut self) {
        self.read = 1 - self.read;
    }

    /// Number of grid cells.
    #[must_use]
    pub const fn num_cells(&self) -> u32 {
        self.allocated_cells
    }
}

/// Bind group layouts and compute pipelines for the simulation.
#[derive(Debug)]
pub struct SimPipelines {
    /// Layout of group 0: parameters, interaction and the spatial grid.
    pub sim_layout: wgpu::BindGroupLayout,
    /// Layout of group 1: the pass's read/read-write working pair.
    ///
    /// One layout for all three shapes the pair takes (agents, agents-to-keys, keys). They differ
    /// only in element type, which the layout does not describe.
    pub agent_layout: wgpu::BindGroupLayout,
    /// Pipeline layout shared by every simulation pass.
    pub pipeline_layout: wgpu::PipelineLayout,

    /// All-pairs integration.
    pub integrate_naive: wgpu::ComputePipeline,
    /// Spatial grid integration.
    pub integrate_grid: wgpu::ComputePipeline,

    /// Fills `cell_start` with [`EMPTY_CELL`].
    pub clear_cells: wgpu::ComputePipeline,
    /// Writes the per-agent cell key into the hash destination key buffer.
    pub hash: wgpu::ComputePipeline,
    /// One bitonic stage; its parameters arrive through immediates.
    pub sort: wgpu::ComputePipeline,
    /// Derives `cell_start` / `cell_end` from the sorted keys.
    pub build_ranges: wgpu::ComputePipeline,

    /// Group 0, bound once: everything in it is stable for the simulation's lifetime.
    sim_bind_group: wgpu::BindGroup,
    /// Group 1 for the integration passes, indexed by the read buffer: `agent_bind_groups[k]` reads
    /// `boids[k]` and writes `boids[1 - k]`.
    agent_bind_groups: [wgpu::BindGroup; 2],
    /// Group 1 for the sort: `key_bind_groups[k]` reads `keys[k]` and writes `keys[1 - k]`.
    key_bind_groups: [wgpu::BindGroup; 2],
    /// Group 1 for the hash pass, indexed by the read buffer: reads `boids[k]` and writes the key
    /// buffer [`KeyPlan::hash_dst`] names.
    hash_bind_groups: [wgpu::BindGroup; 2],
    /// Which key buffer the hash pass writes so that the sort ends in `keys[0]`.
    key_plan: KeyPlan,
    /// Number of live agents, cached for the dispatch size.
    num_boids: u32,
    /// `padded_n`, cached for the dispatch sizes.
    padded_n: u32,
}

/// Which physical key buffer each grid-preparation pass reads and writes.
///
/// This type exists to hold one argument: the bitonic sort ping-pongs between `keys[0]` and
/// `keys[1]`, so the buffer the sorted result lands in depends on the parity of the stage count.
/// Aiming the hash pass at the right buffer makes that land in `keys[0]` unconditionally, which is
/// what lets `build_ranges` and `integrate_grid` read one fixed buffer through the fixed group 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyPlan {
    /// Key buffer the hash pass writes, and therefore the one the first sort stage reads.
    pub hash_dst: usize,
    /// Number of bitonic stages a frame runs.
    pub stages: u32,
}

/// The key-buffer plan for a padded agent count.
///
/// `padded_n` must be a power of two (the sort requires it, and `SimResources` guarantees it). The
/// stage count for `n = 2^m` is `m * (m + 1) / 2`: every `k` contributes `log2(k)` stages.
///
/// Stage `t` reads `keys[s ^ (t & 1)]` and writes the other one, so after `stages` stages the result
/// is in `keys[hash_dst]` when the count is odd and in `keys[1 - hash_dst]` when it is even. Setting
/// `hash_dst` to the count's own parity makes `keys[0]` the answer either way.
#[must_use]
pub fn key_plan(padded_n: u32) -> KeyPlan {
    debug_assert!(
        padded_n.is_power_of_two(),
        "the bitonic sort needs a power-of-two count, got {padded_n}"
    );
    let m = padded_n.trailing_zeros();
    let stages = m * (m + 1) / 2;
    KeyPlan {
        hash_dst: (stages % 2) as usize,
        stages,
    }
}

/// One bitonic stage: the merge block size `k` and the compare distance `j`.
///
/// Yields `key_plan(n).stages` pairs in the order the sort needs them: `k` ascending, `j` halving
/// from `k / 2` down to 1 within each `k`.
pub fn sort_stages(padded_n: u32) -> impl Iterator<Item = (u32, u32)> {
    let m = padded_n.trailing_zeros();
    (1..=m).flat_map(move |exp| {
        let k = 1u32 << exp;
        (0..exp).rev().map(move |half| (k, 1u32 << half))
    })
}

impl SimPipelines {
    /// Builds the layouts and compiles every simulation shader.
    ///
    /// # Panics
    /// Panics if a shader fails to compile or a pipeline fails validation. Both are programming
    /// errors and should stop startup with the WGSL error text rather than render a black screen.
    #[must_use]
    pub fn new(ctx: &GpuContext, res: &SimResources) -> Self {
        let sim_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sim group 0"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                core::mem::size_of::<SimParams>() as u64,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                core::mem::size_of::<InteractionUniforms>() as u64,
                            ),
                        },
                        count: None,
                    },
                    storage_read_write_entry(2),
                    storage_read_write_entry(3),
                ],
            });

        let agent_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sim group 1"),
                // read pair, read-write pair, and the second read slot the grid integrator needs for
                // the sorted keys. See `shaders/sim/bindings.wgsl` for why the keys are not in group 0.
                entries: &[
                    storage_read_entry(0),
                    storage_read_write_entry(1),
                    storage_read_entry(2),
                ],
            });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sim pipeline layout"),
                bind_group_layouts: &[Some(&sim_layout), Some(&agent_layout)],
                // Enough for the sort's `SortParams` and no more. One immediate block per shader is
                // the limit, so this is also the largest block any simulation shader can have;
                // growing it later means growing this number to match.
                immediate_size: core::mem::size_of::<SortParams>() as u32,
            });

        let integrate_module = ctx.shader_module("sim/integrate", "sim/integrate.wgsl");
        let hash_module = ctx.shader_module("sim/hash", "sim/hash.wgsl");
        let sort_module = ctx.shader_module("sim/sort", "sim/sort.wgsl");
        let ranges_module = ctx.shader_module("sim/ranges", "sim/ranges.wgsl");

        let make = |label: &str, entry: &str, module: &wgpu::ShaderModule| -> wgpu::ComputePipeline {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        };
        let integrate_naive = make("integrate_naive", "integrate_naive", &integrate_module);
        let integrate_grid = make("integrate_grid", "integrate_grid", &integrate_module);
        let clear_cells = make("clear_cells", "clear_cells", &hash_module);
        let hash = make("hash", "hash", &hash_module);
        let sort = make("sort_step", "sort_step", &sort_module);
        let build_ranges = make("build_ranges", "build_ranges", &ranges_module);

        let sim_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim group 0"),
            layout: &sim_layout,
            entries: &[
                bind(0, &res.params),
                bind(1, &res.interaction),
                bind(2, &res.cell_start),
                bind(3, &res.cell_end),
            ],
        });

        let key_plan = key_plan(res.padded_n);
        // Group 1 always has the same three slots, and every bind group below fills the third with
        // the key array the sorted order lives in: `pair_bind_group` is the shape of all of them.
        let agent_bind_groups = [
            pair_bind_group(ctx, &agent_layout, &res.boids[0], &res.boids[1], &res.keys[0]),
            pair_bind_group(ctx, &agent_layout, &res.boids[1], &res.boids[0], &res.keys[0]),
        ];
        // The sort's pair, and the pair `build_ranges` reads the result through: a stage writes the
        // buffer it is not reading, and both slots follow `key_bind_groups[k] = keys[k] -> keys[1-k]`.
        let key_bind_groups = [
            pair_bind_group(ctx, &agent_layout, &res.keys[0], &res.keys[1], &res.keys[0]),
            pair_bind_group(ctx, &agent_layout, &res.keys[1], &res.keys[0], &res.keys[1]),
        ];
        // The hash pass does not ping-pong its own output: every frame it writes the same key buffer,
        // the one `key_plan` says leaves the sort's result in `keys[0]`. The third slot is inert and
        // has to be the agent array again, because the key buffer is already the read-write one.
        let hash_bind_groups = [
            pair_bind_group(ctx, &agent_layout, &res.boids[0], &res.keys[key_plan.hash_dst], &res.boids[0]),
            pair_bind_group(ctx, &agent_layout, &res.boids[1], &res.keys[key_plan.hash_dst], &res.boids[1]),
        ];

        log::debug!(
            "grid prep: {padded_n} keys padded, {} sort stages, hash writes keys[{}]",
            key_plan.stages,
            key_plan.hash_dst,
            padded_n = res.padded_n
        );

        Self {
            sim_layout,
            agent_layout,
            pipeline_layout,
            integrate_naive,
            integrate_grid,
            clear_cells,
            hash,
            sort,
            build_ranges,
            sim_bind_group,
            agent_bind_groups,
            key_bind_groups,
            hash_bind_groups,
            key_plan,
            num_boids: res.num_boids,
            padded_n: res.padded_n,
        }
    }

    /// Records one complete simulation step: the grid, then the integration.
    ///
    /// Reads `res.read_buffer()`, writes `res.write_buffer()`. The caller must call
    /// [`SimResources::swap`] after submitting, once the step is complete.
    ///
    /// `profiler` collects per-pass GPU timings; pass `&mut None` to record without timestamps.
    ///
    /// Every pass here is a separate compute pass, which is what makes the sequence correct: a stage
    /// reads what the previous stage wrote, and `wgpu` only inserts the memory barrier between
    /// passes, never between two dispatches inside one pass.
    pub fn record_step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        res: &SimResources,
        strategy: Strategy,
        profiler: &mut Option<GpuProfiler>,
    ) {
        if strategy == Strategy::Grid {
            self.record_grid_prep(encoder, res, profiler);
        }
        self.record_integrate(encoder, res, strategy, profiler);
    }

    /// Records the passes that build the spatial grid: `clear_cells`, `hash`, the bitonic stages and
    /// `build_ranges`, in that order.
    ///
    /// The grid is rebuilt from scratch every frame rather than updated incrementally. An
    /// incremental grid would have to move agents between cells as they fly, which means a variable
    /// amount of work per agent per frame, an atomic insertion order that changes the sorted result,
    /// and a whole class of bugs where a cell's range and its contents disagree. Rebuilding costs
    /// one pass over the keys per stage and no state at all.
    ///
    /// After this returns, `keys[0]` holds every live agent's key in ascending cell order and
    /// `cell_start` / `cell_end` describe each cell's run in it.
    pub fn record_grid_prep(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        res: &SimResources,
        profiler: &mut Option<GpuProfiler>,
    ) {
        // P0. Writes only `cell_start`, and every invocation writes a distinct element.
        //
        // Group 1 is bound but unused: `wgpu` requires every bind group the pipeline layout declares
        // to be set before a dispatch, whether or not the shader declares anything in it. The sort's
        // key pair is the honest thing to put there, and binding it costs a barrier.
        compute_pass(encoder, profiler, "clear_cells", |pass| {
            pass.set_bind_group(0, &self.sim_bind_group, &[]);
            pass.set_bind_group(1, &self.key_bind_groups[0], &[]);
            pass.set_pipeline(&self.clear_cells);
            pass.dispatch_workgroups(dispatch_size(res.num_cells()), 1, 1);
        });

        // P1. Reads the live agents, writes the hash destination key buffer.
        compute_pass(encoder, profiler, "hash", |pass| {
            pass.set_bind_group(0, &self.sim_bind_group, &[]);
            pass.set_bind_group(1, &self.hash_bind_groups[res.read_index()], &[]);
            pass.set_pipeline(&self.hash);
            pass.dispatch_workgroups(dispatch_size(self.padded_n), 1, 1);
        });

        // P2. One pass per bitonic stage, all of them with the same pipeline, immediate block and
        // *shape* of bind group; only the two integers and which of the two key pairs changes.
        //
        // The ping-pong is the easy thing to get wrong here: a stage reads what the stage before it
        // wrote, so the read buffer and the write buffer alternate, and the first stage must read the
        // buffer the hash just filled. `key_bind_groups[k]` is the pair `keys[k] -> keys[1 - k]`, so
        // stage `t` wants pair `hash_dst + t (mod 2)`.
        for (stage, (k, j)) in sort_stages(self.padded_n).enumerate() {
            let bind_group = &self.key_bind_groups[(self.key_plan.hash_dst + stage) % 2];
            let params = SortParams {
                j,
                k,
                n_padded: self.padded_n,
                _pad: 0,
            };
            compute_pass(encoder, profiler, "sort", |pass| {
                pass.set_bind_group(0, &self.sim_bind_group, &[]);
                pass.set_bind_group(1, bind_group, &[]);
                pass.set_pipeline(&self.sort);
                pass.set_immediates(0, bytemuck::bytes_of(&params));
                pass.dispatch_workgroups(dispatch_size(self.padded_n), 1, 1);
            });
        }

        // P3. Reads the sorted keys through group 0 and writes both range arrays. Group 1 is inert
        // here for the same reason as in P0; the grid lives entirely in group 0.
        compute_pass(encoder, profiler, "build_ranges", |pass| {
            pass.set_bind_group(0, &self.sim_bind_group, &[]);
            pass.set_bind_group(1, &self.key_bind_groups[0], &[]);
            pass.set_pipeline(&self.build_ranges);
            // One invocation per key, plus the terminator that closes the last run.
            pass.dispatch_workgroups(dispatch_size(self.padded_n + 1), 1, 1);
        });
    }

    /// Records the integration pass.
    ///
    /// [`Strategy::Grid`] requires [`Self::record_grid_prep`] to have been recorded into the same
    /// command buffer first. Nothing here can detect whether it was: an unprepared grid finds *no*
    /// neighbours rather than wrong ones, because `cell_start` is `EMPTY_CELL` everywhere until the
    /// range builder runs, so a missing preparation shows up as a swarm that flies in straight lines.
    pub fn record_integrate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        res: &SimResources,
        strategy: Strategy,
        profiler: &mut Option<GpuProfiler>,
    ) {
        compute_pass(encoder, profiler, "integrate", |pass| {
            pass.set_bind_group(0, &self.sim_bind_group, &[]);
            pass.set_bind_group(1, &self.agent_bind_groups[res.read_index()], &[]);
            pass.set_pipeline(match strategy {
                Strategy::Naive => &self.integrate_naive,
                Strategy::Grid => &self.integrate_grid,
            });
            pass.dispatch_workgroups(dispatch_size(self.num_boids), 1, 1);
        });
    }

    /// Number of agents this pipeline set was built for.
    #[must_use]
    pub const fn num_boids(&self) -> u32 {
        self.num_boids
    }

    /// Agent count the buffers and the sort were sized for.
    #[must_use]
    pub const fn padded_n(&self) -> u32 {
        self.padded_n
    }

    /// Which key buffer the hash pass writes, and how many sort stages a frame runs.
    #[must_use]
    pub const fn key_plan(&self) -> KeyPlan {
        self.key_plan
    }
}

/// Records one compute pass, with a timestamp pair around it when a profiler is attached.
///
/// `profiler.pass_writes` borrows the profiler for as long as the timestamp writes live, which is
/// exactly as long as the pass does. The borrow ends with `tw`, before the next call reuses the
/// profiler, so the pass-recording code does not have to thread any state between passes.
fn compute_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    profiler: &'a mut Option<GpuProfiler>,
    label: &'static str,
    body: impl FnOnce(&mut wgpu::ComputePass<'a>),
) {
    let timestamp_writes = match profiler {
        Some(profiler) => profiler.pass_writes(label),
        None => None,
    };
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes,
    });
    body(&mut pass);
}

/// Workgroups needed to cover `count` invocations at 256 per group.
#[must_use]
pub const fn dispatch_size(count: u32) -> u32 {
    count.div_ceil(256)
}

fn storage_read_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_read_write_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Builds a group-1 bind group: a read/write pair plus the key array the sorted order lives in.
///
/// All three slots are always filled, because a bind group has to cover every entry in its layout
/// even when the shader that will use it declares only some of them.
fn pair_bind_group(
    ctx: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    keys: &wgpu::Buffer,
) -> wgpu::BindGroup {
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sim group 1"),
        layout,
        entries: &[bind(0, src), bind(1, dst), bind(2, keys)],
    })
}

//! Simulation resources and compute pipelines: the ping-pong agent buffers, the spatial grid, the
//! parameter uniforms and the passes that drive them.
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
//! neighbour search by the sort (their keys are pushed to `u32::MAX`), and never integrated because
//! every pass gates on `i < num_boids`.
//!
//! # Why two bind groups
//!
//! Group 0 holds everything that is stable for the lifetime of the simulation. Group 1 holds the
//! `(source, destination)` agent pair, which swaps every step, so exactly two such bind groups exist
//! and the host alternates between them. Rebuilding a bind group per frame would work but allocates,
//! and at 60 fps that is a slow leak of driver objects.

use boids_core::config::{GridDims, SimConfig};
use boids_core::layout::{
    Boid, InteractionUniforms, KeyVal, SimParams, EMPTY_CELL,
};

use crate::context::GpuContext;
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
            cell_start: make("cell_start", cell_bytes, storage | wgpu::BufferUsages::COPY_DST),
            cell_end: make("cell_end", cell_bytes, storage),
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

    /// The sort key buffer holding the current state.
    #[must_use]
    pub fn read_keys(&self) -> &wgpu::Buffer {
        &self.keys[self.read]
    }

    /// The sort key buffer being written this step.
    #[must_use]
    pub fn write_keys(&self) -> &wgpu::Buffer {
        &self.keys[1 - self.read]
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
    /// Layout of group 1: the source/destination agent pair.
    pub agent_layout: wgpu::BindGroupLayout,
    /// Pipeline layout shared by every simulation pass.
    pub pipeline_layout: wgpu::PipelineLayout,

    /// All-pairs integration.
    pub integrate_naive: wgpu::ComputePipeline,
    /// Spatial grid integration.
    pub integrate_grid: wgpu::ComputePipeline,

    /// Group 0, bound once: everything in it is stable for the simulation's lifetime.
    sim_bind_group: wgpu::BindGroup,
    /// Group 1, indexed by the read buffer: `agent_bind_groups[k]` reads `boids[k]` and writes
    /// `boids[1 - k]`.
    agent_bind_groups: [wgpu::BindGroup; 2],
    /// The same, for the sort key ping-pong. Not read yet: the bitonic sort and the hash and
    /// range-building passes that use it land in the next batch of work.
    #[allow(dead_code)]
    key_bind_groups: [wgpu::BindGroup; 2],
    /// Number of live agents, cached for the dispatch size.
    num_boids: u32,
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
                    storage_read_entry(2),
                    storage_read_entry(3),
                    storage_read_entry(4),
                ],
            });

        let agent_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sim group 1"),
                entries: &[storage_read_entry(0), storage_read_write_entry(1)],
            });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sim pipeline layout"),
                bind_group_layouts: &[Some(&sim_layout), Some(&agent_layout)],
                // No immediates yet. Day 2 uses them for the bitonic sort stage parameters, which
                // replaces the dynamic-offset uniform array the naive implementation would need.
                immediate_size: 0,
            });

        let module = ctx.shader_module("sim", "sim/integrate.wgsl");
        let make =
            |label: &str, entry: &str| -> wgpu::ComputePipeline {
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    })
            };
        let integrate_naive = make("integrate_naive", "integrate_naive");
        let integrate_grid = make("integrate_grid", "integrate_grid");

        let sim_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim group 0"),
            layout: &sim_layout,
            entries: &[
                bind(0, &res.params),
                bind(1, &res.interaction),
                bind(2, &res.keys[0]),
                bind(3, &res.cell_start),
                bind(4, &res.cell_end),
            ],
        });

        let agent_bind_groups = [
            make_agent_bind_group(ctx, &agent_layout, &res.boids[0], &res.boids[1]),
            make_agent_bind_group(ctx, &agent_layout, &res.boids[1], &res.boids[0]),
        ];
        let key_bind_groups = [
            make_agent_bind_group(ctx, &agent_layout, &res.keys[0], &res.keys[1]),
            make_agent_bind_group(ctx, &agent_layout, &res.keys[1], &res.keys[0]),
        ];

        Self {
            sim_layout,
            agent_layout,
            pipeline_layout,
            integrate_naive,
            integrate_grid,
            sim_bind_group,
            agent_bind_groups,
            key_bind_groups,
            num_boids: res.num_boids,
        }
    }

    /// Records the integration pass for one simulation step.
    ///
    /// Reads `res.read_buffer()`, writes `res.write_buffer()`. The caller must call
    /// [`SimResources::swap`] after submitting, once the step is complete.
    ///
    /// [`Strategy::Grid`] additionally requires the spatial grid to have been built earlier in the
    /// same command buffer by the hash, sort and range passes. Those passes are not implemented yet,
    /// so today the grid path finds no neighbours: `cell_start` stays uniformly [`EMPTY_CELL`] and
    /// every agent behaves as if it were alone. That is a visible, honest failure rather than a
    /// silently wrong result.
    pub fn record_integrate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        res: &SimResources,
        strategy: Strategy,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("integrate"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.sim_bind_group, &[]);
        pass.set_bind_group(1, &self.agent_bind_groups[res.read_index()], &[]);
        pass.set_pipeline(match strategy {
            Strategy::Naive => &self.integrate_naive,
            Strategy::Grid => &self.integrate_grid,
        });
        pass.dispatch_workgroups(dispatch_size(self.num_boids), 1, 1);
    }

    /// Number of agents this pipeline set was built for.
    #[must_use]
    pub const fn num_boids(&self) -> u32 {
        self.num_boids
    }
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

fn make_agent_bind_group(
    ctx: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
) -> wgpu::BindGroup {
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sim group 1"),
        layout,
        entries: &[bind(0, src), bind(1, dst)],
    })
}

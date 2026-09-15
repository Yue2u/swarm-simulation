//! Shared helpers for the GPU test suite.

#![allow(dead_code)]

use boids_core::layout::{Boid, KeyVal};
use boids_gpu::context::{GpuContext, GpuContextDescriptor};
use boids_gpu::sim::{SimPipelines, SimResources, Strategy};

/// A check that either passes or reports why it did not.
pub type Check = Result<(), String>;

/// Creates the single device the whole suite shares.
///
/// Prefers a real GPU but accepts a software adapter so the suite runs on a machine with no usable
/// GPU (CI containers, headless servers). Set `BOIDS_TEST_REQUIRE_GPU=1` to refuse the fallback,
/// which is what a developer on their own machine wants: silently validating on llvmpipe while
/// believing the GPU path is covered is a good way to ship a driver-specific bug.
pub fn test_context() -> GpuContext {
    let require_gpu = std::env::var("BOIDS_TEST_REQUIRE_GPU").is_ok();
    match GpuContext::new(&GpuContextDescriptor {
        high_performance: true,
        want_timestamps: true,
        force_backends: None,
        force_fallback: false,
    }) {
        Ok(ctx) => {
            if require_gpu && is_software(&ctx) {
                panic!(
                    "BOIDS_TEST_REQUIRE_GPU is set but the adapter is {}",
                    ctx.info.name
                );
            }
            ctx
        }
        Err(e) if require_gpu => {
            panic!("BOIDS_TEST_REQUIRE_GPU is set but no device is available: {e}")
        }
        Err(e) => {
            eprintln!("note: no hardware adapter ({e}); falling back to a software adapter");
            GpuContext::new(&GpuContextDescriptor {
                high_performance: false,
                want_timestamps: true,
                force_backends: None,
                force_fallback: true,
            })
            .unwrap_or_else(|e| panic!("no usable GPU device, hardware or software: {e}"))
        }
    }
}

/// Whether the adapter is a software rasteriser.
pub fn is_software(ctx: &GpuContext) -> bool {
    let name = ctx.info.name.to_ascii_lowercase();
    ctx.info.device_type == wgpu::DeviceType::Cpu
        || name.contains("llvmpipe")
        || name.contains("lavapipe")
        || name.contains("swiftshader")
        || name.contains("software")
}

/// Relative difference between two floats, treating a near-zero pair as equal.
pub fn rel_err(a: f32, b: f32) -> f32 {
    let scale = a.abs().max(b.abs()).max(1e-6);
    (a - b).abs() / scale
}

/// Runs `steps` simulation steps on the GPU and returns the resulting agent state.
///
/// All steps are recorded into one command buffer, so the whole run costs one submission. The
/// ping-pong parity is flipped between steps exactly as the frame loop does it, and the returned
/// vector is the state of the buffer the simulation would render from.
pub fn run_gpu_steps(
    ctx: &GpuContext,
    res: &mut SimResources,
    pipes: &SimPipelines,
    strategy: Strategy,
    steps: usize,
) -> Vec<Boid> {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu test steps"),
        });
    for _ in 0..steps {
        pipes.record_step(&mut encoder, res, strategy, &mut None);
        res.swap();
    }
    ctx.queue.submit(Some(encoder.finish()));

    let mut out: Vec<Boid> = boids_gpu::transfer::read_buffer(ctx, res.read_buffer());
    out.truncate(res.num_boids as usize);
    out
}

/// Builds resources and pipelines for a configuration, with the swarm uploaded.
pub fn setup(
    ctx: &GpuContext,
    config: &boids_core::config::SimConfig,
    seed: u64,
) -> (SimResources, SimPipelines, Vec<Boid>) {
    let swarm = boids_core::spawn::spawn_swarm(config, seed);
    let res = SimResources::new(ctx, config);
    boids_gpu::transfer::upload_boids(&ctx.queue, &res.boids[0], &swarm);
    let pipes = SimPipelines::new(ctx, &res);
    (res, pipes, swarm)
}

/// The grid after a preparation pass, as the CPU can see it.
pub struct GridState {
    /// Every sort key, sorted by cell index, `padded_n` entries long.
    pub keys: Vec<KeyVal>,
    /// First index of each cell's run, or `boids_core::layout::EMPTY_CELL`.
    pub cell_start: Vec<u32>,
    /// One past the last index of each cell's run.
    pub cell_end: Vec<u32>,
}

/// Runs the grid preparation passes once and reads the result back.
///
/// The caller must have uploaded `SimParams` first: the hash pass reads the grid geometry from them.
/// This is the same sequence `SimPipelines::record_step` records for a grid frame, minus the
/// integration, so what it returns is what the integration pass would have searched.
pub fn run_grid_prep(
    ctx: &GpuContext,
    res: &SimResources,
    pipes: &SimPipelines,
) -> GridState {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu test grid prep"),
        });
    pipes.record_grid_prep(&mut encoder, res, &mut None);
    ctx.queue.submit(Some(encoder.finish()));

    GridState {
        keys: boids_gpu::transfer::read_buffer(ctx, res.sorted_keys()),
        cell_start: boids_gpu::transfer::read_buffer(ctx, &res.cell_start),
        cell_end: boids_gpu::transfer::read_buffer(ctx, &res.cell_end),
    }
}

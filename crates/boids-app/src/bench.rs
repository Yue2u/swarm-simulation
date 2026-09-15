//! Headless benchmark: runs the simulation for N frames and reports the per-pass GPU cost.
//!
//! This exists so that the numbers in `docs/perf.md` can be regenerated instead of remembered, and so
//! that a change to the sort or the grid can be compared against the previous one on the same machine
//! without a window, a display server or a person watching.
//!
//! What it measures is the *simulation*: the grid preparation and the integration, one `queue.submit`
//! per frame, exactly as the app records it. Rendering and presentation are not included, and on a
//! machine where the GPU cannot present (which is where this was developed) they cannot be. That is a
//! real limitation of the numbers and it is why the table says so rather than reporting a "frame
//! time" that excludes half the frame.
//!
//! The timings come from GPU timestamp queries, so they measure the passes and not the gaps between
//! them. The readback blocks the CPU once per frame, which inflates the wall-clock rate and does not
//! affect the pass times.

use boids_core::config::SimConfig;
use boids_core::layout::{SimMode, SimParams};
use boids_gpu::context::{GpuContext, GpuContextDescriptor};
use boids_gpu::profile::{GpuProfiler, PassAverages, MAX_TIMED_PASSES};
use boids_gpu::sim::{SimPipelines, SimResources, Strategy};

/// What to benchmark.
#[derive(Debug, Clone)]
pub struct BenchRequest {
    /// World to simulate.
    pub mode: SimMode,
    /// Number of agents.
    pub num_agents: usize,
    /// Frames to time, after the warmup.
    pub frames: usize,
    /// Frames to run untimed, so pipeline compilation and the first touches of each buffer do not
    /// land in the average.
    pub warmup_frames: usize,
    /// World seed.
    pub seed: u64,
    /// Strategy to use, or `None` to let the agent count decide.
    pub strategy: Option<Strategy>,
}

impl Default for BenchRequest {
    fn default() -> Self {
        Self {
            mode: SimMode::Fish,
            num_agents: 100_000,
            frames: 120,
            warmup_frames: 20,
            seed: 1,
            strategy: None,
        }
    }
}

/// Runs the benchmark and prints the table.
///
/// # Errors
/// Returns a description when there is no device, and reports the absence of timestamp support as a
/// failure: a benchmark that silently reports nothing is worse than one that says why.
pub fn run(request: &BenchRequest) -> Result<(), String> {
    let ctx = GpuContext::new(&GpuContextDescriptor::default())?;
    let config = SimConfig::for_mode(request.mode, request.num_agents);
    let grid = config.grid;
    #[allow(clippy::cast_possible_truncation)]
    let strategy = request
        .strategy
        .unwrap_or_else(|| Strategy::for_count(config.num_boids as u32));

    let swarm = boids_core::spawn::spawn_swarm(&config, request.seed);
    let mut sim = SimResources::new(&ctx, &config);
    boids_gpu::transfer::upload_boids(&ctx.queue, &sim.boids[0], &swarm);
    boids_gpu::transfer::upload_boids(&ctx.queue, &sim.boids[1], &swarm);
    let pipes = SimPipelines::new(&ctx, &sim);
    let plan = pipes.key_plan();

    let profiler = GpuProfiler::new(&ctx, MAX_TIMED_PASSES, 1)
        .ok_or_else(|| "no timestamp support on this device: nothing to measure".to_string())?;

    log::info!(
        "bench: {} agents ({:?}), {strategy:?} search, grid {:?} = {} cells, {} sort stages",
        config.num_boids,
        request.mode,
        grid.dim,
        grid.num_cells(),
        plan.stages
    );

    let mut averages = PassAverages::default();
    // Held in an `Option` because that is the shape `record_step` takes: it either times the passes
    // into this profiler or records without one.
    let mut profiler_slot: Option<GpuProfiler> = Some(profiler);
    let total = request.warmup_frames + request.frames;
    let started = std::time::Instant::now();

    for frame in 0..total {
        #[allow(clippy::cast_precision_loss)]
        let time = frame as f32 * config.dt;
        let params: SimParams = config.to_params(time, config.dt);
        sim.write_params(&ctx.queue, &params);
        sim.write_interaction(&ctx.queue, &SimConfig::idle_interaction());

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bench frame"),
            });
        pipes.record_step(&mut encoder, &sim, strategy, &mut profiler_slot);
        if let Some(profiler) = &mut profiler_slot {
            profiler.resolve(&mut encoder);
        }
        ctx.queue.submit(Some(encoder.finish()));
        sim.swap();

        // `read` blocks until the GPU has finished this frame and returns the timings on every frame
        // because the profiler's interval is 1. The first `warmup_frames` are dropped so that
        // pipeline compilation and the first touch of each buffer do not land in the average.
        if let Some(profiler) = &mut profiler_slot {
            if let Some(timings) = profiler.read(&ctx) {
                if frame >= request.warmup_frames {
                    averages.add(&timings);
                }
            }
        }
    }

    let elapsed = started.elapsed();
    #[allow(clippy::cast_precision_loss)]
    let rate = total as f64 / elapsed.as_secs_f64().max(1e-6);
    println!(
        "{}",
        averages.format_table("GPU compute, per pass, mean per frame")
    );
    println!(
        "    {} frames timed of {total} in {elapsed:.2?} ({rate:.1} frames/s; the readback blocks the \
         CPU, so this is a floor on the wall-clock rate, not a measurement of it)",
        averages.frames()
    );
    Ok(())
}

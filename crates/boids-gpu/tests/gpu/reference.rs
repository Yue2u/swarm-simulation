//! Checks that the GPU simulation reproduces the CPU reference.
//!
//! `boids_core::reference::step_cpu` is the specification; the shaders are the implementation. This
//! module is the only place where the two are compared, and it compares them at two different
//! granularities on purpose:
//!
//! * **One step, exact state.** With a single step there is no chance for chaotic divergence to
//!   amplify a small difference, so the comparison can be tight. If this fails, the force model
//!   differs from the reference somewhere specific: a wrong weight, a missing term, a flipped sign.
//! * **Many steps, aggregate state.** Over hundreds of steps an N-body system is chaotic and
//!   per-agent comparison is meaningless, but the *aggregates* must still agree. If this fails, the
//!   two implementations have systematically different dynamics even though each step looks right,
//!   which is what a neglected mode-specific term looks like.
//!
//! The comparison uses `Strategy::Naive`, whose neighbour set is unambiguous, so a failure here is
//! the force model and not the neighbour search. The grid strategy is validated against the same
//! all-pairs path in the `grid` module: first that it finds the same neighbours, then that the two
//! stay in step over a long run.

use boids_core::config::SimConfig;
use boids_core::layout::{EnvironmentKind, InteractionUniforms, SimParams};
use boids_core::math::{mean_speed, order_parameter};
use boids_gpu::context::GpuContext;
use boids_gpu::sim::Strategy;
use glam::Vec3;

use crate::common::{rel_err, run_gpu_steps, setup, Check};

/// Builds a dense world that the CPU reference can also handle.
///
/// Dense because flocking is density-driven: see [`SimConfig::dense`]. No environment, because the
/// CPU reference would need the identical field function to compare against, and the environment
/// force is validated separately by the SDF checks in `boids-core`.
fn comparison_config(n: usize, r_percept: f32, neighbours: f32) -> SimConfig {
    let mut cfg = SimConfig::dense(n, r_percept, neighbours);
    cfg.env = EnvironmentKind::None;
    // The app spawns a coherent flock, but these checks need the initial state to be *disordered*:
    // the aggregate check asserts the run reorganises, and comparing two identical "already
    // organised" runs would exercise nothing. `dense` inherits the coherent spawn, so undo it here.
    cfg.spawn_spread = 1.0;
    cfg
}

/// Checks a single step against the reference, agent by agent.
///
/// The tolerance is not zero even though both sides run the same arithmetic in the same order,
/// because the GPU compiler is free to contract `a * b + c` into a fused multiply-add and the drag
/// term uses the driver's `exp`. Those differences are around 1e-7 relative per operation, so a
/// 1e-3 metre tolerance on a world measured in hundreds of metres is a genuine equality check
/// rather than a rubber stamp.
pub fn one_step_matches_cpu(ctx: &GpuContext) -> Check {
    const N: usize = 1024;
    const SEED: u64 = 2024;

    // A world sized for ~15 neighbours each, so the comparison exercises separation, alignment and
    // cohesion rather than a sparse swarm drifting on its own.
    let cfg = comparison_config(N, 8.0, 15.0);
    let params: SimParams = cfg.to_params(0.0, cfg.dt);
    let interaction: InteractionUniforms = SimConfig::idle_interaction();

    let (mut res, pipes, swarm) = setup(ctx, &cfg, SEED);
    res.write_params(&ctx.queue, &params);
    res.write_interaction(&ctx.queue, &interaction);

    let gpu = run_gpu_steps(ctx, &mut res, &pipes, Strategy::Naive, 1);

    let mut cpu = swarm.clone();
    boids_core::reference::step_cpu(&mut cpu, &params, &interaction, None::<&fn(Vec3) -> f32>, 0.5);

    if gpu.len() != cpu.len() {
        return Err(format!("gpu returned {} agents, cpu has {}", gpu.len(), cpu.len()));
    }

    let mut worst_pos = 0.0f32;
    let mut worst_vel = 0.0f32;
    let mut worst_index = 0usize;
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let dp = (Vec3::from(g.pos) - Vec3::from(c.pos)).abs().max_element();
        let dv = (Vec3::from(g.vel) - Vec3::from(c.vel)).abs().max_element();
        if dp > worst_pos {
            worst_pos = dp;
            worst_index = i;
        }
        worst_vel = worst_vel.max(dv);
    }

    if worst_pos > 1e-3 || worst_vel > 1e-3 {
        let g = &gpu[worst_index];
        let c = &cpu[worst_index];
        return Err(format!(
            "after one step the GPU and the reference differ by {worst_pos:.3e} m and \
             {worst_vel:.3e} m/s (worst at agent {worst_index})\n      \
             gpu pos {:?} vel {:?}\n      cpu pos {:?} vel {:?}\n      \
             The force model in shaders/sim/forces.wgsl has drifted from \
             crates/boids-core/src/reference.rs.",
            g.pos, g.vel, c.pos, c.vel
        ));
    }
    Ok(())
}

/// Checks that a long run keeps the same aggregate behaviour.
pub fn many_steps_match_cpu_aggregates(ctx: &GpuContext) -> Check {
    // 512 agents for 40 steps is 10 million pair evaluations: enough for the flock to reorganise
    // several times, cheap enough to run on a software adapter.
    const N: usize = 512;
    const STEPS: usize = 40;
    const SEED: u64 = 77;

    let cfg = comparison_config(N, 6.0, 18.0);
    // A constant `time` for the whole run: the GPU gets one parameter upload for all steps, so the
    // CPU must see the same value or it would be comparing a different simulation.
    let params = cfg.to_params(0.0, cfg.dt);
    let interaction = SimConfig::idle_interaction();

    let (mut res, pipes, swarm) = setup(ctx, &cfg, SEED);
    res.write_params(&ctx.queue, &params);
    res.write_interaction(&ctx.queue, &interaction);

    let gpu = run_gpu_steps(ctx, &mut res, &pipes, Strategy::Naive, STEPS);

    let mut cpu = swarm.clone();
    for _ in 0..STEPS {
        boids_core::reference::step_cpu(
            &mut cpu,
            &params,
            &interaction,
            None::<&fn(Vec3) -> f32>,
            0.5,
        );
    }

    let gpu_vel: Vec<Vec3> = gpu.iter().map(|b| Vec3::from(b.vel)).collect();
    let cpu_vel: Vec<Vec3> = cpu.iter().map(|b| Vec3::from(b.vel)).collect();

    let gpu_points: Vec<Vec3> = gpu.iter().map(|b| Vec3::from(b.pos)).collect();
    let cpu_points: Vec<Vec3> = cpu.iter().map(|b| Vec3::from(b.pos)).collect();

    let (gpu_order, cpu_order) = (order_parameter(&gpu_vel), order_parameter(&cpu_vel));
    let (gpu_speed, cpu_speed) = (mean_speed(&gpu_vel), mean_speed(&cpu_vel));
    let centroid_gpu = boids_core::math::centroid(&gpu_points);
    let centroid_cpu = boids_core::math::centroid(&cpu_points);
    let centroid_err = (centroid_gpu - centroid_cpu).length();

    // Nothing may have escaped or gone NaN in either implementation.
    for (label, set) in [("gpu", &gpu), ("cpu", &cpu)] {
        let issues = boids_core::spawn::validation_issues(set, cfg.bounds_half * 3.0);
        if !issues.is_empty() {
            return Err(format!("{label} produced invalid agents:\n      {}", issues.join("\n      ")));
        }
    }

    let mut problems = Vec::new();
    // The order parameter is the sharpest aggregate: it is a single number describing the whole
    // flock's alignment, so a systematic force difference moves it immediately.
    if (gpu_order - cpu_order).abs() > 0.05 {
        problems.push(format!(
            "order parameter differs by {:.4}: gpu {gpu_order:.4} cpu {cpu_order:.4}",
            (gpu_order - cpu_order).abs()
        ));
    }
    if rel_err(gpu_speed, cpu_speed) > 0.02 {
        problems.push(format!("mean speed differs: gpu {gpu_speed:.4} cpu {cpu_speed:.4}"));
    }
    if centroid_err > 2.0 {
        problems.push(format!("centroid differs by {centroid_err:.4} m"));
    }
    // A swarm that never organised would satisfy the comparisons above trivially, since two
    // identical "nothing happened" results agree perfectly. Require that the run actually changed the
    // state it is being judged on, so this check cannot silently become a no-op.
    let initial_order = order_parameter(&swarm.iter().map(|b| Vec3::from(b.vel)).collect::<Vec<_>>());
    let reorganised = (cpu_order - initial_order).abs();
    if reorganised < 0.05 {
        problems.push(format!(
            "the swarm did not reorganise in {STEPS} steps (order {initial_order:.4} -> \
             {cpu_order:.4}); the comparison is not exercising anything"
        ));
    }

    if problems.is_empty() {
        println!(
            "\n    order {initial_order:.3} -> gpu {gpu_order:.3} / cpu {cpu_order:.3} | \
             speed {gpu_speed:.3}/{cpu_speed:.3} | centroid delta {centroid_err:.3} m"
        );
        Ok(())
    } else {
        Err(problems.join("\n    "))
    }
}

/// Checks that the GPU swarm actually flocks, and that it flocks like the CPU reference does.
///
/// The CPU version of the polarisation assertion lives in `boids-core`; running both here means the
/// GPU cannot regress to "agents move but never organise" without a failure. Comparing the two order
/// parameters on top of that catches a GPU-only dynamic difference that a per-agent comparison over a
/// short run could miss.
pub fn swarm_polarises_on_gpu(ctx: &GpuContext) -> Check {
    // The same world, agent count and step count as `boids-core`'s polarisation test, so the two
    // numbers are directly comparable rather than merely both "greater than".
    const N: usize = 500;
    const STEPS: usize = 600;
    const SEED: u64 = 11;

    let mut cfg = comparison_config(N, 6.0, 18.0);
    cfg.wander = 0.0;
    cfg.buoyancy = 0.0;

    // A constant `time` for the whole run: the GPU receives one parameter upload for all steps.
    let params = cfg.to_params(0.0, cfg.dt);
    let interaction = SimConfig::idle_interaction();

    let (mut res, pipes, swarm) = setup(ctx, &cfg, SEED);
    res.write_params(&ctx.queue, &params);
    res.write_interaction(&ctx.queue, &interaction);

    let initial = order_parameter(&swarm.iter().map(|b| Vec3::from(b.vel)).collect::<Vec<_>>());

    let gpu = run_gpu_steps(ctx, &mut res, &pipes, Strategy::Naive, STEPS);
    let mut cpu = swarm.clone();
    for _ in 0..STEPS {
        boids_core::reference::step_cpu(&mut cpu, &params, &interaction, None::<&fn(Vec3) -> f32>, 1.0);
    }

    let gpu_order = order_parameter(&gpu.iter().map(|b| Vec3::from(b.vel)).collect::<Vec<_>>());
    let cpu_order = order_parameter(&cpu.iter().map(|b| Vec3::from(b.vel)).collect::<Vec<_>>());

    let mut problems = Vec::new();
    if gpu_order < 0.5 {
        problems.push(format!(
            "the GPU swarm did not polarise in {STEPS} steps: order {initial:.4} -> {gpu_order:.4}"
        ));
    }
    if (gpu_order - cpu_order).abs() > 0.15 {
        problems.push(format!(
            "the GPU and the CPU reference polarised differently: gpu {gpu_order:.4} cpu {cpu_order:.4}"
        ));
    }
    if problems.is_empty() {
        println!("\n    order {initial:.3} -> gpu {gpu_order:.3} / cpu {cpu_order:.3}");
        Ok(())
    } else {
        Err(problems.join("\n    "))
    }
}

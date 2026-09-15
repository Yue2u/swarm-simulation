//! Initial swarm placement.
//!
//! One implementation, used by the app, by the CPU reference tests and by the GPU validation tests.
//! That matters: the GPU-vs-CPU comparison is only meaningful if both sides start from byte-identical
//! state, and two spawn routines that "obviously do the same thing" would quietly invalidate it.
//!
//! Placement is a uniform box fill, deliberately *not* an interesting distribution. Uniform noise
//! makes a flock form from nothing, which is the behaviour the simulation is supposed to demonstrate;
//! seeding a pre-formed flock would hide a broken alignment force.

use glam::Vec3;

use crate::config::SimConfig;
use crate::layout::Boid;
use crate::rng::Pcg32;

/// Places `config.num_boids` agents uniformly in a box 60% of the world bounds, with random
/// directions and speeds near `max_speed`.
///
/// The 60% factor keeps agents clear of the soft boundary ramp at spawn, so the first few hundred
/// steps show flocking rather than a boundary bounce.
#[must_use]
pub fn spawn_swarm(config: &SimConfig, seed: u64) -> Vec<Boid> {
    let mut rng = Pcg32::new(seed, 1);
    let half = config.bounds_half * 0.6;
    let (min_speed, max_speed) = (config.min_speed, config.max_speed);
    (0..config.num_boids)
        .map(|_| {
            let dir = rng.unit_vector();
            let speed = rng.range(min_speed, max_speed);
            Boid {
                pos: rng.in_box(half).to_array(),
                species: rng.next_f32(),
                vel: (dir * speed).to_array(),
                phase: rng.range(0.0, core::f32::consts::TAU),
                prev_dir: dir.to_array(),
                color_seed: rng.next_f32(),
            }
        })
        .collect()
}

/// Places agents in a shell between `inner` and `outer` metres from the origin.
///
/// Used by the obstacle tests, where a uniform box fill would place a random subset of agents
/// *inside* the obstacle and make any penetration assertion meaningless.
#[must_use]
pub fn spawn_shell(
    count: usize,
    seed: u64,
    inner: f32,
    outer: f32,
    min_speed: f32,
    max_speed: f32,
) -> Vec<Boid> {
    let mut rng = Pcg32::new(seed, 2);
    (0..count)
        .map(|_| {
            let dir = rng.unit_vector();
            let radius = rng.range(inner, outer);
            let vel_dir = rng.unit_vector();
            Boid {
                pos: (dir * radius).to_array(),
                species: rng.next_f32(),
                vel: (vel_dir * rng.range(min_speed, max_speed)).to_array(),
                phase: rng.range(0.0, core::f32::consts::TAU),
                prev_dir: vel_dir.to_array(),
                color_seed: rng.next_f32(),
            }
        })
        .collect()
}

/// Checks that every agent is finite and inside a sane multiple of the world bounds.
///
/// Used as an assertion by both the CPU and the GPU tests: a NaN position is the most common symptom
/// of a force-model bug, and it is much easier to diagnose here than from a screen full of nothing.
#[must_use]
pub fn validation_issues(boids: &[Boid], limit: Vec3) -> Vec<String> {
    let mut issues = Vec::new();
    for (i, b) in boids.iter().enumerate() {
        let p = Vec3::from(b.pos);
        let v = Vec3::from(b.vel);
        if !p.is_finite() || !v.is_finite() {
            issues.push(format!("agent {i} is not finite: pos {:?} vel {:?}", b.pos, b.vel));
        } else if !p.abs().cmple(limit).all() {
            issues.push(format!(
                "agent {i} left the world: pos {p:?} exceeds {limit:?}"
            ));
        }
        if issues.len() > 8 {
            issues.push("...".to_string());
            break;
        }
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::SimMode;

    #[test]
    fn spawn_is_deterministic_and_inside_the_world() {
        let cfg = SimConfig::for_mode(SimMode::Fish, 500);
        let a = spawn_swarm(&cfg, 99);
        let b = spawn_swarm(&cfg, 99);
        assert_eq!(a, b, "the same seed must produce the same swarm");
        let c = spawn_swarm(&cfg, 100);
        assert_ne!(a, c, "a different seed must produce a different swarm");

        assert_eq!(a.len(), 500);
        let limit = cfg.bounds_half * 0.61;
        assert!(
            validation_issues(&a, limit).is_empty(),
            "spawn left agents outside the world: {:?}",
            validation_issues(&a, limit)
        );
        // Speeds must be inside the clamp range, otherwise the first step silently corrects them
        // and the "initial state" the tests compare against is not the state they think it is.
        for b in &a {
            let speed = Vec3::from(b.vel).length();
            assert!(
                (cfg.min_speed * 0.999..=cfg.max_speed * 1.001).contains(&speed),
                "spawned speed {speed} outside [{}, {}]",
                cfg.min_speed,
                cfg.max_speed
            );
        }
    }

    #[test]
    fn shell_spawn_respects_the_radii() {
        let s = spawn_shell(200, 7, 30.0, 60.0, 5.0, 10.0);
        for b in &s {
            let r = Vec3::from(b.pos).length();
            assert!((30.0..=60.0).contains(&r), "radius {r} outside the shell");
        }
    }
}

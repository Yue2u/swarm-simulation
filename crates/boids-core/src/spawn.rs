//! Initial swarm placement.
//!
//! One implementation, used by the app, by the CPU reference tests and by the GPU validation tests.
//! That matters: the GPU-vs-CPU comparison is only meaningful if both sides start from byte-identical
//! state, and two spawn routines that "obviously do the same thing" would quietly invalidate it.
//!
//! # Why a compact cluster instead of a box fill
//!
//! The obvious placement is a uniform fill of the world box. It is also the placement that shows the
//! least: at the density a 100k swarm produces in a world sized for it, an agent sees five to ten
//! neighbours in its perception sphere, the alignment and cohesion terms are weak, and the swarm
//! spends its first seconds dissolving into hundreds of micro-flocks before the survivors find each
//! other. What the demo is supposed to show - a single flock forming, moving and reacting as one
//! body - is a property of the *dense* regime, and the initial condition has to start there.
//!
//! So the swarm is spawned as one ellipsoid centred on [`SimConfig::spawn_center`], sized so that an
//! interior agent has about [`SimConfig::density_ref`] neighbours inside `r_percept`. That number is
//! not invented here: it is the density the adaptive weights in `sim/forces.wgsl` are calibrated
//! against, so the swarm starts exactly at the density where separation and cohesion balance, grows
//! outward from there, and reaches the world's own density as a flock rather than as a spray.
//!
//! The ellipsoid's axes follow the world's own aspect ratio, and it is shrunk uniformly if it would
//! not fit inside [`MAX_FILL_FRACTION`] of `bounds_half`. Both are structural rather than cosmetic: a
//! sphere in a wide, shallow world would push its own edge through the soft boundary ramp within the
//! first seconds, and the resulting "flocking" would actually be the boundary force.
//!
//! # Why a minimum separation
//!
//! Uniform sampling puts two agents arbitrarily close together, and the first step is then spent on
//! the separation force blowing that pair apart - a visible jolt at t = 0 whose size depends on the
//! luck of the seed. Candidates are therefore rejection-sampled against a minimum distance of
//! `r_sep`. The requirement is *best effort*: after [`SEPARATION_TRIES`] attempts the last candidate
//! is accepted. That bound is what keeps the routine linear rather than potentially unbounded, and it
//! is only ever reached in the artificially dense `SimConfig::dense` test worlds, where the requested
//! separation is geometrically impossible (more exclusion volume than world volume).

use std::collections::HashMap;

use glam::Vec3;

use crate::config::SimConfig;
use crate::layout::Boid;
use crate::rng::Pcg32;

/// Largest fraction of `bounds_half` the spawn cluster may occupy on any axis.
///
/// `bounds_force` in `shaders/common/math_common.wgsl` starts pushing inward at 85% of `bounds_half`,
/// so 80% leaves the swarm a margin of one ramp width: the flock is free to expand at its own pace
/// instead of being compressed by the boundary from the very first step.
const MAX_FILL_FRACTION: f32 = 0.8;

/// Candidate positions tried per agent before the minimum separation is abandoned.
///
/// At the density the cluster is sized for, the exclusion volume is about 10% of the cluster's, so a
/// single attempt succeeds almost always and this bound is never approached: it exists to make the
/// routine's cost bounded rather than to be used.
const SEPARATION_TRIES: usize = 24;

/// Semi-axes of the ellipsoid the swarm is spawned into.
///
/// Sized so that an agent in the interior has about `density_ref` neighbours inside `r_percept`:
/// with `volume = n * (4/3 pi r^3) / density_ref`, the density is exactly the reference density by
/// construction. The shape is the world's aspect ratio scaled by a single factor, and that factor is
/// reduced if the resulting ellipsoid would cross into the boundary ramp or leave the world, which
/// makes the cluster tighter (and its neighbour count higher) rather than degenerate.
#[must_use]
pub fn swarm_extent(config: &SimConfig) -> Vec3 {
    #[allow(clippy::cast_precision_loss)]
    let n = (config.num_boids.max(2) - 1) as f32;
    let target = config.density_ref.max(1e-3);
    let sphere = 4.0 / 3.0 * core::f32::consts::PI * config.r_percept.max(1e-3).powi(3);
    let volume = n * sphere / target;

    // `max(1e-3)` rather than trusting the config: a zero on any axis would make the scale factor
    // infinite, and a degenerate world is a caller error that should not turn into NaN positions.
    let world = config.bounds_half.max(Vec3::splat(1e-3));
    let unit_volume = 4.0 / 3.0 * core::f32::consts::PI * world.x * world.y * world.z;
    let scaled = world * (volume / unit_volume).cbrt();

    // Uniform shrink, applied only when the cluster as a whole does not fit. Per-axis clamping would
    // preserve the volume and distort the shape into a slab that no longer matches the world.
    let limit = (world * MAX_FILL_FRACTION - config.spawn_center.abs()).max(Vec3::splat(1e-4));
    let scale = (limit / scaled).min_element().min(1.0);
    scaled * scale.max(1e-3)
}

/// Places `config.num_boids` agents in one dense ellipsoid around `config.spawn_center`, with random
/// directions, speeds near `max_speed` and no pair closer than `config.r_sep`.
#[must_use]
pub fn spawn_swarm(config: &SimConfig, seed: u64) -> Vec<Boid> {
    let mut rng = Pcg32::new(seed, 1);
    let axes = swarm_extent(config);
    let center = config.spawn_center;
    let min_distance = config.r_sep.max(1e-3);
    let mut grid = SeparationGrid::new(min_distance);

    let mut out = Vec::with_capacity(config.num_boids);
    for index in 0..config.num_boids {
        // Drawn before the position so that the number of random values consumed per agent does not
        // depend on how many placement attempts it took. Deterministic either way, but a fixed
        // per-agent draw count makes a future per-agent field addition a local change.
        let dir = rng.unit_vector();
        let speed = rng.range(config.min_speed, config.max_speed);
        let phase = rng.range(0.0, core::f32::consts::TAU);
        let species = rng.next_f32();
        let color_seed = rng.next_f32();

        let mut placed = None;
        for _ in 0..SEPARATION_TRIES {
            let candidate = sample_in_ellipsoid(&mut rng, center, axes);
            if grid.is_clear(candidate, min_distance) {
                placed = Some(candidate);
                break;
            }
        }
        // Best effort: see the module comment. The candidate is still drawn from the cluster, so
        // even the saturated case stays inside the world.
        let pos = placed.unwrap_or_else(|| sample_in_ellipsoid(&mut rng, center, axes));
        #[allow(clippy::cast_possible_truncation)]
        grid.insert(pos, index as u32);

        out.push(Boid {
            pos: pos.to_array(),
            species,
            vel: (dir * speed).to_array(),
            phase,
            prev_dir: dir.to_array(),
            color_seed,
        });
    }
    out
}

/// Uniform sample inside an ellipsoid.
///
/// `u^(1/3)` rather than `u`: a uniform radius would crowd the centre, because the volume of a shell
/// grows with the square of the radius.
fn sample_in_ellipsoid(rng: &mut Pcg32, center: Vec3, axes: Vec3) -> Vec3 {
    let dir = rng.unit_vector();
    let radius = rng.next_f32().cbrt();
    center + dir * (radius * axes)
}

/// A uniform hash grid over the placed agents, used to test a candidate position against its
/// neighbours instead of against every agent placed so far.
///
/// The cell edge is the minimum distance, which is what makes the 27-cell neighbourhood complete:
/// any agent closer than `min_distance` lies in a cell whose integer coordinate differs by at most
/// one on each axis. A brute-force check would be O(N^2), and at 100k agents that is the difference
/// between a startup that is instant and one that takes minutes.
struct SeparationGrid {
    cell: f32,
    cells: HashMap<[i32; 3], Vec<u32>>,
    points: Vec<Vec3>,
}

impl SeparationGrid {
    fn new(cell: f32) -> Self {
        Self {
            cell: cell.max(1e-3),
            cells: HashMap::new(),
            points: Vec::new(),
        }
    }

    fn key(&self, p: Vec3) -> [i32; 3] {
        [
            (p.x / self.cell).floor() as i32,
            (p.y / self.cell).floor() as i32,
            (p.z / self.cell).floor() as i32,
        ]
    }

    /// Whether no stored agent is closer than `min_distance` to `p`.
    fn is_clear(&self, p: Vec3, min_distance: f32) -> bool {
        let limit = min_distance * min_distance;
        let base = self.key(p);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let Some(bucket) = self.cells.get(&[base[0] + dx, base[1] + dy, base[2] + dz])
                    else {
                        continue;
                    };
                    for &index in bucket {
                        let other = self.points[index as usize];
                        if (other - p).length_squared() < limit {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    fn insert(&mut self, p: Vec3, index: u32) {
        let key = self.key(p);
        self.points.push(p);
        self.cells.entry(key).or_default().push(index);
    }
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

    /// Number of other agents within `radius` of each agent.
    ///
    /// Same hash-grid trick as the spawner, for the same reason: the property being checked is
    /// pairwise, and 20k agents make the direct form of the check 4 * 10^8 distance tests.
    fn neighbours_within(boids: &[Boid], radius: f32) -> Vec<u32> {
        let cell = radius.max(1e-3);
        let key = |p: Vec3| {
            [
                (p.x / cell).floor() as i32,
                (p.y / cell).floor() as i32,
                (p.z / cell).floor() as i32,
            ]
        };
        let mut grid: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        for (i, b) in boids.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            grid.entry(key(Vec3::from(b.pos))).or_default().push(i as u32);
        }

        let limit = radius * radius;
        let mut counts = vec![0u32; boids.len()];
        for (i, b) in boids.iter().enumerate() {
            let p = Vec3::from(b.pos);
            let base = key(p);
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let Some(bucket) =
                            grid.get(&[base[0] + dx, base[1] + dy, base[2] + dz])
                        else {
                            continue;
                        };
                        for &j in bucket {
                            if j as usize == i {
                                continue;
                            }
                            if (Vec3::from(boids[j as usize].pos) - p).length_squared() < limit {
                                counts[i] += 1;
                            }
                        }
                    }
                }
            }
        }
        counts
    }

    /// Smallest distance between any two agents.
    ///
    /// Direct O(N^2) here: the swarm this is checked against is 4,000 agents, and the point of the
    /// check is to be independent of the spawner's own grid rather than to be fast.
    fn min_pairwise_distance(boids: &[Boid]) -> f32 {
        let mut best = f32::INFINITY;
        for (i, b) in boids.iter().enumerate() {
            let p = Vec3::from(b.pos);
            for other in &boids[i + 1..] {
                best = best.min((Vec3::from(other.pos) - p).length());
            }
        }
        best
    }

    #[test]
    fn spawn_is_deterministic_and_inside_the_world() {
        let cfg = SimConfig::for_mode(SimMode::Fish, 500);
        let a = spawn_swarm(&cfg, 99);
        let b = spawn_swarm(&cfg, 99);
        assert_eq!(a, b, "the same seed must produce the same swarm");
        let c = spawn_swarm(&cfg, 100);
        assert_ne!(a, c, "a different seed must produce a different swarm");

        assert_eq!(a.len(), 500);
        let limit = cfg.bounds_half * (MAX_FILL_FRACTION + 1e-3);
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

    /// The point of the cluster: an agent must start with neighbours, not with an empty sphere.
    #[test]
    fn spawn_forms_one_dense_swarm() {
        for (mode, count) in [(SimMode::Fish, 20_000usize), (SimMode::Birds, 20_000)] {
            let cfg = SimConfig::for_mode(mode, count);
            let swarm = spawn_swarm(&cfg, 4);
            let counts = neighbours_within(&swarm, cfg.r_percept);
            let lonely = counts.iter().filter(|c| **c == 0).count();
            #[allow(clippy::cast_precision_loss)]
            let mean = counts.iter().sum::<u32>() as f32 / counts.len() as f32;

            assert!(
                lonely <= swarm.len() / 100,
                "{mode:?}: {lonely} of {} agents start with no neighbour inside r_percept={}",
                swarm.len(),
                cfg.r_percept
            );
            assert!(
                (0.4..=3.0).contains(&(mean / cfg.density_ref)),
                "{mode:?}: mean neighbour count {mean} is not near the reference density {}",
                cfg.density_ref
            );

            // And the swarm must be compact: a fill of the whole world would pass the two checks
            // above only by accident, and it is exactly the layout this test exists to prevent.
            let center = cfg.spawn_center;
            #[allow(clippy::cast_precision_loss)]
            let rms = (swarm
                .iter()
                .map(|b| (Vec3::from(b.pos) - center).length_squared())
                .sum::<f32>()
                / swarm.len() as f32)
                .sqrt();
            assert!(
                rms < 0.6 * cfg.bounds_half.length(),
                "{mode:?}: swarm is spread out (rms radius {rms} against world radius {})",
                cfg.bounds_half.length()
            );
        }
    }

    /// No pair closer than `r_sep`, which is what keeps the first step from being a jolt.
    #[test]
    fn spawn_keeps_a_minimum_separation() {
        let cfg = SimConfig::for_mode(SimMode::Fish, 4_000);
        let swarm = spawn_swarm(&cfg, 17);
        let closest = min_pairwise_distance(&swarm);
        assert!(
            closest >= cfg.r_sep * 0.999,
            "two agents spawned {closest} apart, which is inside r_sep={}",
            cfg.r_sep
        );
    }

    /// The extent has to be monotone in the agent count and never leave the fill fraction.
    #[test]
    fn swarm_extent_scales_with_count_and_fits_the_world() {
        let mut previous = Vec3::ZERO;
        for count in [1_000usize, 10_000, 100_000] {
            let cfg = SimConfig::for_mode(SimMode::Fish, count);
            let axes = swarm_extent(&cfg);
            assert!(
                axes.min_element() > 0.0 && axes.is_finite(),
                "degenerate extent {axes:?} for {count} agents"
            );
            assert!(
                (axes + cfg.spawn_center.abs()).cmple(cfg.bounds_half * MAX_FILL_FRACTION).all(),
                "{count} agents: extent {axes:?} from centre {:?} leaves the world",
                cfg.spawn_center
            );
            assert!(
                axes.cmpge(previous).all(),
                "extent shrank from {previous:?} to {axes:?} when the count grew"
            );
            previous = axes;
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

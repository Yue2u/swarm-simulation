//! Naive O(N^2) CPU reference implementation of one simulation step.
//!
//! This is **the specification**: it is written to be obviously correct rather than fast, and
//! `shaders/sim/integrate.wgsl` is expected to reproduce it. The GPU validation test runs both on
//! the same input and compares the `order_parameter`, centroid and mean speed, which are the
//! aggregates that survive floating-point reassociation.
//!
//! ## Exact step semantics (mirrored by the shader, in this order)
//!
//! 1. neighbour pass over the snapshot: separation / alignment / cohesion accumulators,
//! 2. density-adaptive weights,
//! 3. force accumulation: boids + wander + world bounds + SDF avoidance + cursor,
//! 4. `a = clamp_len(sum, max_force)`,
//! 5. `v += a * dt`,
//! 6. `v *= exp(-drag * dt)`,
//! 7. `v = clamp_len_range(v, min_speed, max_speed)`,
//! 8. `p += v * dt`, `prev_dir = normalize(v)`.
//!
//! Steps 6 and 7 are in this order on purpose: clamping first and applying drag afterwards would
//! let drag pull agents below `min_speed` forever, which looks like a swarm slowly going to sleep.
//!
//! All reads happen from the snapshot taken at the start of the step. The GPU achieves the same by
//! ping-ponging between two buffers, so the reference must not read partially updated state.

use glam::Vec3;

use crate::layout::{
    Boid, InteractionMode, InteractionUniforms, KeyVal, SimMode, SimParams,
};
use crate::math::{clamp_len, clamp_len_range};

/// A `[0,1)` value derived from an integer hash, matching `hash_to_unit` in the shaders.
///
/// Integer multiplication wraps in WGSL for unsigned types, and `f32(h >> 8)` is exact because the
/// result is below 2^24, so this is bit-identical on both sides.
#[inline]
#[must_use]
pub fn hash_unit(mut h: u32) -> f32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    #[allow(clippy::cast_precision_loss)]
    let v = (h >> 8) as f32;
    v * (1.0 / 16_777_216.0)
}

/// Wander tick: how often the wander direction changes, in Hz. Shared with the shader.
pub const WANDER_RATE: f32 = 0.7;

/// Per-agent wander force for a given tick, matching `wander_force` in the shaders.
///
/// Deliberately *not* normalized: a variable-magnitude nudge is visually indistinguishable from a
/// normalized one, and skipping the normalization removes a `normalize(0)` NaN case that would be
/// a divergence risk between the CPU and the GPU.
#[inline]
#[must_use]
pub fn wander_force(index: u32, tick: u32, scale: f32) -> Vec3 {
    let base = index.wrapping_mul(0x9e37_79b9) ^ tick.wrapping_mul(0x85eb_ca6b);
    Vec3::new(
        hash_unit(base) * 2.0 - 1.0,
        hash_unit(base ^ 0x27d4_eb2d) * 2.0 - 1.0,
        hash_unit(base ^ 0x1656_67b1) * 2.0 - 1.0,
    ) * scale
}

/// Soft containment force pushing an agent back inside the box `[-half, half]`.
///
/// The force ramps up over the outer 15% of the box, so agents turn smoothly instead of
/// reflecting off an invisible wall. Force units are m/s^2, scaled by the agent's own cruise speed
/// so that the same code works for slow fish and fast birds.
#[inline]
#[must_use]
pub fn bounds_force(pos: Vec3, half: Vec3, speed_scale: f32) -> Vec3 {
    let mut f = Vec3::ZERO;
    for axis in 0..3 {
        let margin = (0.15 * half[axis]).max(1e-3);
        let to_pos_wall = half[axis] - pos[axis];
        let to_neg_wall = pos[axis] + half[axis];
        if to_pos_wall < margin {
            f[axis] -= 1.0 - (to_pos_wall / margin).max(0.0);
        }
        if to_neg_wall < margin {
            f[axis] += 1.0 - (to_neg_wall / margin).max(0.0);
        }
    }
    f * (speed_scale * 4.0)
}

/// SDF avoidance, including the forward probe that produces tangential sliding.
///
/// The probe is what makes agents *go around* a rock instead of pressing into it: when the point
/// ahead is already inside the safety margin, the force is redirected along the component of
/// velocity that is tangent to the surface. A pure repulsion force has no way to break the
/// symmetry that keeps an agent pinned against a wall.
#[inline]
#[must_use]
pub fn sdf_avoidance<F: Fn(Vec3) -> f32>(
    pos: Vec3,
    vel: Vec3,
    params: &SimParams,
    field: &F,
    eps: f32,
) -> Vec3 {
    let d = field(pos);
    let n = gradient(field, pos, eps);
    let len = n.length();
    let n = if len > 1e-6 { n / len } else { Vec3::ZERO };

    let mut force = crate::sdf::avoid_force(d, n, params.r_safe, params.sdf_strength);

    let ahead = pos + vel * (params.sdf_probe / vel.length().max(1e-3));
    if field(ahead) < params.r_safe {
        let v_tangent = vel - n * vel.dot(n);
        let len_t = v_tangent.length();
        if len_t > 1e-6 {
            force += (v_tangent / len_t) * params.sdf_strength;
        }
    }
    force
}

/// Central-difference gradient, mirroring `sdf_gradient` in `shaders/common/sdf.wgsl`.
#[inline]
#[must_use]
pub fn gradient<F: Fn(Vec3) -> f32>(field: &F, p: Vec3, eps: f32) -> Vec3 {
    crate::sdf::gradient(field, p, eps)
}

/// Cursor influence at a point, mirroring `interaction_force` in the shaders.
///
/// `w(r) = clamp(1 - r/R, 0, 1)^falloff`, then a radial push and, in repel mode, a tangential
/// swirl. The swirl is what makes a repelled swarm look like it is *avoiding* the cursor rather
/// than merely being blown away from it.
#[inline]
#[must_use]
pub fn interaction_force(pos: Vec3, vel: Vec3, inter: &InteractionUniforms) -> Vec3 {
    let mode = inter.mode;
    if mode == InteractionMode::Off.as_u32() || inter.strength == 0.0 || inter.radius <= 0.0 {
        return Vec3::ZERO;
    }
    let focus = Vec3::from(inter.focus_point);
    let g = focus - pos;
    let r = g.length();
    if r > inter.radius || r < 1e-6 {
        return Vec3::ZERO;
    }
    let t = (1.0 - r / inter.radius).clamp(0.0, 1.0);
    let w = t.powf(inter.falloff);
    let dir = g / r;

    let mut f = if mode == InteractionMode::Attract.as_u32() {
        dir * (w * inter.strength)
    } else {
        -dir * (w.powf(1.5) * inter.strength)
    };

    if inter.tangent != 0.0 {
        // Swirl around the world up axis so the swarm curves around the cursor instead of
        // forming a straight radial burst.
        let tang = Vec3::Y.cross(dir);
        let len_t = tang.length();
        if len_t > 1e-6 {
            let tang = tang / len_t;
            // Only swirl agents that are actually moving, otherwise the sign is noise.
            let sense = if vel.dot(tang) >= 0.0 { 1.0 } else { -1.0 };
            f += tang * (sense * w * inter.tangent);
        }
    }
    f
}

/// Performs one full simulation step on the CPU, in the exact order documented in the module docs.
///
/// `env` is the environment SDF, or `None` for free space. `eps` is the gradient probe distance.
pub fn step_cpu<F: Fn(Vec3) -> f32>(
    boids: &mut [Boid],
    params: &SimParams,
    inter: &InteractionUniforms,
    env: Option<&F>,
    eps: f32,
) {
    let snapshot: Vec<Boid> = boids.to_vec();
    let n = snapshot.len();
    #[allow(clippy::cast_precision_loss)]
    let density_ref_inv = params.sep_boost.max(1e-6);
    let is_fish = params.mode == SimMode::Fish.as_u32();
    let tick = tick_of(params.time);

    let mut out = Vec::with_capacity(n);
    for (i, b) in snapshot.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let index = i as u32;
        out.push(step_one(
            b,
            index,
            &snapshot,
            params,
            inter,
            env,
            eps,
            density_ref_inv,
            is_fish,
            tick,
        ));
    }
    boids.copy_from_slice(&out);
}

/// Computes the wander tick for a simulation time. Shared by the CPU reference and the shader.
#[inline]
#[must_use]
pub fn tick_of(time: f32) -> u32 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let t = (time * WANDER_RATE) as u32;
    t
}

#[allow(clippy::too_many_arguments)]
fn step_one<F: Fn(Vec3) -> f32>(
    b: &Boid,
    index: u32,
    snapshot: &[Boid],
    params: &SimParams,
    inter: &InteractionUniforms,
    env: Option<&F>,
    eps: f32,
    density_ref_inv: f32,
    is_fish: bool,
    tick: u32,
) -> Boid {
    let pos = Vec3::from(b.pos);
    let vel = Vec3::from(b.vel);
    let r_percept_sq = params.r_percept * params.r_percept;
    let r_sep_sq = params.r_sep * params.r_sep;

    let mut sep = Vec3::ZERO;
    let mut ali_sum = Vec3::ZERO;
    let mut coh_sum = Vec3::ZERO;
    let mut count = 0.0f32;

    for (j, o) in snapshot.iter().enumerate() {
        if j == index as usize {
            continue;
        }
        let d = Vec3::from(o.pos) - pos;
        let d2 = d.length_squared();
        if d2 > r_percept_sq || d2 < 1e-8 {
            continue;
        }
        if d2 < r_sep_sq {
            // 1/r falloff: dividing by d2 after the unit vector yields 1/|d|.
            sep -= d / d2;
        }
        ali_sum += Vec3::from(o.vel);
        coh_sum += Vec3::from(o.pos);
        count += 1.0;
    }

    let density = count * density_ref_inv;
    // Symmetric density feedback, neutral at `density_ref` (both multipliers 1). The exponent form
    // makes `gain == 0` mean fixed weights, `gain == 1` the proportional feedback `push == density`
    // and larger gains swing harder. The former `1 + d` / `exp(-d)` pair was already ~10x
    // separation-dominant at the reference density, so a cluster spawned there was pre-loaded to
    // explode and broke into micro-flocks.
    let gain = params.coh_falloff.max(0.0);
    let push = if count > 0.0 { density.max(1e-6).powf(gain) } else { 1.0 };
    let w_sep = params.w_sep * push;
    let w_coh = params.w_coh / push;

    let mut acc = sep * w_sep;

    if count > 0.0 {
        let ali_dir = ali_sum.normalize_or_zero();
        if ali_dir.length_squared() > 0.5 {
            acc += (ali_dir * params.max_speed - vel) * params.w_ali;
        }
        let coh_center = coh_sum / count;
        let coh_dir = (coh_center - pos).normalize_or_zero();
        acc += (coh_dir * params.max_speed - vel) * w_coh;
    }

    acc += wander_force(index, tick, params.wander);
    acc += bounds_force(pos, Vec3::from(params.bounds_half), params.max_speed);

    if let Some(field) = env {
        acc += sdf_avoidance(pos, vel, params, field, eps);
    }

    acc += interaction_force(pos, vel, inter);

    // Mode-specific vertical term. Fish are slightly buoyant, birds glide on thermals; both are
    // expressed as a constant acceleration so the two modes differ by a single uniform.
    acc.y += params.buoyancy * params.max_speed;

    let acc = clamp_len(acc, params.max_force);
    let mut new_vel = vel + acc * params.dt;
    new_vel *= (-params.drag * params.dt).exp();
    let new_vel = clamp_len_range(new_vel, params.min_speed, params.max_speed);
    let new_pos = pos + new_vel * params.dt;

    let mut out = *b;
    out.pos = new_pos.to_array();
    out.vel = new_vel.to_array();
    out.prev_dir = new_vel.normalize_or_zero().to_array();
    if is_fish {
        // Fish tails beat faster when swimming faster; keeping the phase integrated here means the
        // GPU only has to advance it, which keeps the animation in sync with the motion.
        out.phase = (b.phase + params.dt * (2.0 + 3.0 * new_vel.length() / params.max_speed)).rem_euclid(core::f32::consts::TAU);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimConfig;
    use crate::layout::Boid;

    /// Spawns `n` agents inside a world of half-extent `half`, delegating to the shared spawn so that
    /// the reference tests and the GPU tests start from identical state.
    ///
    /// The world is set to exactly `half` rather than to a multiple of it: the spawner keeps its
    /// cluster inside `MAX_FILL_FRACTION` of the bounds on its own, so scaling the world here would
    /// only make the swarm denser than the density it is sized for.
    fn spawn(n: usize, seed: u64, half: Vec3, speed: f32) -> Vec<Boid> {
        let mut cfg = SimConfig::for_mode(SimMode::Fish, n);
        cfg.bounds_half = half;
        cfg.spawn_center = Vec3::ZERO;
        cfg.min_speed = speed * 0.5;
        cfg.max_speed = speed;
        crate::spawn::spawn_swarm(&cfg, seed)
    }

    /// Spawns agents in a shell between `inner` and `outer` metres from the origin.
    ///
    /// Used by the obstacle tests: spawning uniformly in a box would put a random subset of agents
    /// *inside* the obstacle, which makes a penetration assertion meaningless.
    fn spawn_shell(n: usize, seed: u64, inner: f32, outer: f32, speed: f32) -> Vec<Boid> {
        crate::spawn::spawn_shell(n, seed, inner, outer, speed * 0.5, speed)
    }

    #[test]
    fn step_keeps_agents_inside_bounds() {
        let cfg = SimConfig::for_mode(SimMode::Fish, 400);
        let inter = SimConfig::idle_interaction();
        let mut boids = spawn(400, 7, cfg.bounds_half, cfg.max_speed);
        for step in 0..600 {
            let params = cfg.to_params(step as f32 * cfg.dt, cfg.dt);
            step_cpu(&mut boids, &params, &inter, None::<&fn(Vec3) -> f32>, 1.0);
        }
        for b in &boids {
            let p = Vec3::from(b.pos);
            assert!(
                p.abs().cmple(cfg.bounds_half * 1.05).all(),
                "agent escaped to {p:?}"
            );
            let speed = Vec3::from(b.vel).length();
            assert!(
                (cfg.min_speed * 0.99..=cfg.max_speed * 1.01).contains(&speed),
                "speed {speed} out of [{} , {}]",
                cfg.min_speed,
                cfg.max_speed
            );
        }
    }

    #[test]
    fn swarm_polarises_without_perturbation() {
        // The core emergent behaviour: a set of agents that can only see each other should
        // spontaneously align. If this fails, the force signs are wrong.
        let mut cfg = SimConfig::dense(500, 6.0, 18.0);
        cfg.wander = 0.0;
        cfg.buoyancy = 0.0;
        let inter = SimConfig::idle_interaction();
        let mut boids = spawn(500, 11, cfg.bounds_half, cfg.max_speed);
        // The spawner launches a coherent flock; spontaneous alignment has to start from disorder,
        // so scatter the initial headings again.
        let mut rng = crate::rng::Pcg32::new(11, 5);
        for b in boids.iter_mut() {
            let dir = rng.unit_vector();
            b.vel = (dir * cfg.max_speed).to_array();
            b.prev_dir = dir.to_array();
        }
        let initial = crate::math::order_parameter(
            &boids.iter().map(|b| Vec3::from(b.vel)).collect::<Vec<_>>(),
        );
        for step in 0..600 {
            let params = cfg.to_params(step as f32 * cfg.dt, cfg.dt);
            step_cpu(&mut boids, &params, &inter, None::<&fn(Vec3) -> f32>, 1.0);
        }
        let final_order = crate::math::order_parameter(
            &boids.iter().map(|b| Vec3::from(b.vel)).collect::<Vec<_>>(),
        );
        assert!(
            final_order > 0.5,
            "swarm did not polarise: order {initial} -> {final_order}"
        );
    }

    #[test]
    fn separation_prevents_collapse() {
        // With cohesion on and separation working, agents must not pile into a single point. The
        // threshold is the separation radius rather than a fraction of the perception radius: at the
        // reference density the flock is compact (its natural spacing is below `r_sep`), so a
        // healthy flock sits just above it while a collapsed one falls through.
        let mut cfg = SimConfig::for_mode(SimMode::Birds, 150);
        cfg.wander = 0.0;
        cfg.buoyancy = 0.0;
        let inter = SimConfig::idle_interaction();
        let mut boids = spawn(150, 3, Vec3::splat(60.0), cfg.max_speed * 0.5);
        for step in 0..300 {
            let params = cfg.to_params(step as f32 * cfg.dt, cfg.dt);
            step_cpu(&mut boids, &params, &inter, None::<&fn(Vec3) -> f32>, 1.0);
        }
        let points: Vec<Vec3> = boids.iter().map(|b| Vec3::from(b.pos)).collect();
        let c = crate::math::centroid(&points);
        let mean_dist = points.iter().map(|p| (*p - c).length()).sum::<f32>() / points.len() as f32;
        assert!(
            mean_dist > cfg.r_sep,
            "swarm collapsed: mean distance from centroid {mean_dist} is inside r_sep={}",
            cfg.r_sep
        );
    }

    #[test]
    fn sdf_avoidance_keeps_agents_out_of_a_sphere() {
        // Agents are launched at the obstacle from outside; none of them may end up inside it.
        let cfg = SimConfig::for_mode(SimMode::Fish, 200);
        let inter = SimConfig::idle_interaction();
        let obstacle = |p: Vec3| crate::sdf::sphere(p, Vec3::ZERO, 25.0);
        let mut boids = spawn_shell(200, 5, 30.0, 60.0, cfg.max_speed);
        let params = cfg.to_params(0.0, cfg.dt);
        let mut worst_penetration = f32::INFINITY;
        for _ in 0..600 {
            step_cpu(&mut boids, &params, &inter, Some(&obstacle), 0.5);
            let min_d = boids
                .iter()
                .map(|b| obstacle(Vec3::from(b.pos)))
                .fold(f32::INFINITY, f32::min);
            worst_penetration = worst_penetration.min(min_d);
        }
        assert!(
            worst_penetration > -1.0,
            "agents sank into the obstacle by {} m",
            -worst_penetration
        );
        // And the avoidance must not simply freeze them: some agents should have flown past it.
        let beyond = boids
            .iter()
            .filter(|b| Vec3::from(b.pos).length() > 40.0)
            .count();
        assert!(beyond > 20, "only {beyond} agents got past the obstacle");
    }

    #[test]
    fn agents_inside_an_obstacle_escape() {
        // If an agent does end up inside a closed obstacle, the field must still push it out, which
        // requires the gradient to be well defined on the inside too.
        let cfg = SimConfig::for_mode(SimMode::Fish, 64);
        let inter = SimConfig::idle_interaction();
        let obstacle = |p: Vec3| crate::sdf::sphere(p, Vec3::ZERO, 25.0);
        let mut boids = spawn(64, 9, Vec3::splat(10.0), cfg.max_speed);
        let params = cfg.to_params(0.0, cfg.dt);
        for _ in 0..900 {
            step_cpu(&mut boids, &params, &inter, Some(&obstacle), 0.5);
        }
        let outside = boids
            .iter()
            .filter(|b| obstacle(Vec3::from(b.pos)) > 0.0)
            .count();
        assert_eq!(
            outside,
            boids.len(),
            "{} of {} agents stayed inside the obstacle",
            boids.len() - outside,
            boids.len()
        );
    }

    #[test]
    fn attractor_pulls_and_repeller_pushes() {
        let cfg = SimConfig::for_mode(SimMode::Fish, 100);
        let params = cfg.to_params(0.0, cfg.dt);
        let focus = Vec3::new(50.0, 0.0, 0.0);
        let mut boids = spawn(100, 21, Vec3::splat(40.0), cfg.max_speed * 0.4);

        let mut attract = SimConfig::idle_interaction();
        attract.mode = InteractionMode::Attract.as_u32();
        attract.focus_point = focus.to_array();
        attract.radius = 200.0;
        attract.strength = 30.0;
        attract.falloff = 1.0;

        let before = crate::math::mean_distance_to(&boids, focus);
        for _ in 0..240 {
            step_cpu(&mut boids, &params, &attract, None::<&fn(Vec3) -> f32>, 1.0);
        }
        let after = crate::math::mean_distance_to(&boids, focus);
        assert!(after < before, "attractor did not pull: {before} -> {after}");

        let mut repel = SimConfig::idle_interaction();
        repel.mode = InteractionMode::Repel.as_u32();
        repel.focus_point = focus.to_array();
        repel.radius = 120.0;
        repel.strength = 40.0;
        repel.falloff = 1.0;
        repel.tangent = 0.0;

        let mut boids = spawn(100, 21, Vec3::splat(40.0), cfg.max_speed * 0.4);
        let before = crate::math::mean_distance_to(&boids, focus);
        for _ in 0..240 {
            step_cpu(&mut boids, &params, &repel, None::<&fn(Vec3) -> f32>, 1.0);
        }
        let after = crate::math::mean_distance_to(&boids, focus);
        assert!(after > before, "repeller did not push: {before} -> {after}");
    }

    /// Connected components at `r_percept`, and the fraction in the largest, via union-find.
    fn connectivity(boids: &[Boid], radius: f32) -> (usize, f32) {
        let n = boids.len();
        let r2 = radius * radius;
        let mut parent: Vec<usize> = (0..n).collect();
        fn root(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for (i, b) in boids.iter().enumerate() {
            let pi = Vec3::from(b.pos);
            for (j, other) in boids.iter().enumerate().skip(i + 1) {
                if (Vec3::from(other.pos) - pi).length_squared() <= r2 {
                    let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                    if a != b {
                        parent[a] = b;
                    }
                }
            }
        }
        let mut sizes: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for i in 0..n {
            *sizes.entry(root(&mut parent, i)).or_default() += 1;
        }
        let largest = sizes.values().copied().max().unwrap_or(0);
        (sizes.len(), largest as f32 / n.max(1) as f32)
    }

    /// Advances the app world from a normal coherent spawn, with the real environment field, and
    /// returns the settled flock.
    fn settle(cfg: &SimConfig, seed: u64, steps: usize, env: bool) -> Vec<Boid> {
        let inter = SimConfig::idle_interaction();
        let mut boids = crate::spawn::spawn_swarm(cfg, seed);
        let eps = 0.5 * cfg.grid.cell_size;
        for step in 0..steps {
            let params = cfg.to_params(step as f32 * cfg.dt, cfg.dt);
            match (cfg.mode, env) {
                (SimMode::Fish, true) => {
                    let field = |p: Vec3| crate::sdf::reef_field(p, cfg.env_scale, cfg.env_floor_y);
                    step_cpu(&mut boids, &params, &inter, Some(&field), eps);
                }
                (SimMode::Birds, true) => {
                    let field = |p: Vec3| {
                        p.y - crate::terrain::height_at(
                            glam::Vec2::new(p.x, p.z),
                            cfg.env_scale,
                            cfg.env_freq,
                        )
                    };
                    step_cpu(&mut boids, &params, &inter, Some(&field), eps);
                }
                _ => step_cpu(&mut boids, &params, &inter, None::<&fn(Vec3) -> f32>, eps),
            }
        }
        boids
    }

    /// The flock must survive contact with the world as one body.
    ///
    /// Regression for the report that the swarm "starts as one cloud then splits into 5-10 boid
    /// swarms". Two causes compounded: the spawn gave every agent a random heading (so the dense
    /// cluster flew apart ballistically) and the adaptive weights were already ~10x
    /// separation-dominant at `density_ref` (so the cluster was pre-loaded to explode). With both
    /// fixed this measures one group; before, it measured ~50 groups with the largest holding 14%.
    #[test]
    fn spawned_flock_stays_one_group() {
        for mode in [SimMode::Fish, SimMode::Birds] {
            for seed in [4u64, 11] {
                let cfg = SimConfig::for_mode(mode, 800);
                let boids = settle(&cfg, seed, 600, true);
                let (groups, largest) = connectivity(&boids, cfg.r_percept);
                assert!(
                    largest >= 0.8,
                    "{mode:?} seed {seed}: flock shattered into {groups} groups, largest {:.0}%",
                    largest * 100.0
                );
            }
        }
    }

    /// Mean nearest-neighbour distance and the closest pair, the collision/spacing signal.
    fn nn_stats(boids: &[Boid]) -> (f32, f32) {
        let n = boids.len();
        let mut sum = 0.0f32;
        let mut closest = f32::INFINITY;
        for (i, b) in boids.iter().enumerate() {
            let pi = Vec3::from(b.pos);
            let mut best = f32::INFINITY;
            for (j, other) in boids.iter().enumerate() {
                if i == j {
                    continue;
                }
                let d = (Vec3::from(other.pos) - pi).length();
                best = best.min(d);
                closest = closest.min(d);
            }
            sum += best;
        }
        (sum / n as f32, closest)
    }

    #[test]
    #[ignore = "manual diagnostic, run with --ignored --nocapture"]
    fn tune_swarm() {
        for mode in [SimMode::Fish, SimMode::Birds] {
            for density_ref in [18.0f32, 24.0, 30.0, 36.0] {
                for env in [false, true] {
                    let mut cfg = SimConfig::for_mode(mode, 1000);
                    cfg.density_ref = density_ref;
                    cfg.density_gain = 1.0;
                    let boids = settle(&cfg, 5, 600, env);
                    let (groups, largest) = connectivity(&boids, cfg.r_percept);
                    let (mean_nn, closest) = nn_stats(&boids);
                    println!(
                        "{mode:?} dref {density_ref:>4} env {env:>5}: groups {groups:>3}, largest \
                         {:>3.0}%, nn {mean_nn:5.2} (r_sep {:.2}), closest {closest:.2}",
                        largest * 100.0,
                        cfg.r_sep,
                    );
                }
            }
        }
    }
}

/// Ascending sort of a key array by [`KeyVal::key`], the specification for the GPU bitonic sort.
///
/// A comparison sort from the standard library rather than a hand-written bitonic sort, on purpose:
/// reproducing the GPU's algorithm on the CPU would reproduce its bugs too, and the only thing this
/// function is for is being independently right.
///
/// Ties on `key` - two agents in the same cell - are ordered by `val` here and arbitrarily on the
/// GPU, because the bitonic network is not stable. Callers must not depend on the order *within* a
/// cell, and a comparison against the GPU result has to allow for it. Breaking ties by `val` is what
/// makes this side of that comparison deterministic, so a failure is reproducible rather than
/// dependent on the driver's scheduling.
pub fn sort_keys(keys: &mut [KeyVal]) {
    keys.sort_unstable_by_key(|k| (k.key, k.val));
}

#[cfg(test)]
mod debug_probe {
    use super::*;
    use crate::config::SimConfig;

    #[test]
    #[ignore = "manual diagnostic, run with --ignored --nocapture"]
    fn trace_worst_penetration() {
        let cfg = SimConfig::for_mode(SimMode::Fish, 200);
        let inter = SimConfig::idle_interaction();
        let obstacle = |p: Vec3| crate::sdf::sphere(p, Vec3::ZERO, 25.0);
        let mut boids = crate::spawn::spawn_shell(200, 5, 30.0, 60.0, cfg.min_speed, cfg.max_speed);
        let params = cfg.to_params(0.0, cfg.dt);
        println!("r_safe={} probe={} strength={} max_force={} r_percept={}",
            params.r_safe, params.sdf_probe, params.sdf_strength, params.max_force, params.r_percept);
        let mut worst = f32::INFINITY;
        let mut worst_i = 0usize;
        for step in 0..600 {
            step_cpu(&mut boids, &params, &inter, Some(&obstacle), 0.5);
            for (i, b) in boids.iter().enumerate() {
                let d = obstacle(Vec3::from(b.pos));
                if d < worst {
                    worst = d;
                    worst_i = i;
                    if step % 40 == 0 || d < 0.0 {
                        println!("step {step} worst {d:.3} agent {i} pos {:?} vel {:?}",
                            Vec3::from(b.pos), Vec3::from(b.vel));
                    }
                }
            }
        }
        println!("final worst {worst:.4} at agent {worst_i}");
    }
}

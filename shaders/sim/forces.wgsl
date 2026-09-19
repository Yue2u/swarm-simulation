// === file: sim/forces.wgsl ===================================================================
// The Reynolds force accumulation, shared by every integration strategy.
//
// The accumulation is split from the neighbour *search* on purpose. `integrate_naive` walks every
// other agent, `integrate_grid` walks the 27 cells around the agent: both must produce exactly the
// same forces for the same neighbour set. Splitting them means the force model can be validated
// against the CPU reference with the O(N^2) shader (where the neighbour set is unambiguous) and
// then trusted when the grid version replaces the search, with a single test asserting that the
// two shaders agree.
//
// Force order is normative and mirrors `boids_core::reference::step_cpu`. Any change here must be
// mirrored there in the same commit.
//
// Header only: declares helpers, no entry points.

//#include "sim/bindings.wgsl"

// Running neighbour statistics for one agent.
//
// Packed into a struct rather than passed as six parameters: shader function argument lists are
// positional and long argument lists are where force-model bugs hide.
struct NeighbourAccum {
    // Sum of (p_self - p_other) / |p_self - p_other|^2 over separation neighbours.
    separation: vec3<f32>,
    // Sum of neighbour velocities, over all perception neighbours.
    align_sum: vec3<f32>,
    // Sum of neighbour positions, over all perception neighbours.
    cohesion_sum: vec3<f32>,
    // Number of perception neighbours, as f32 because it is consumed as a weight.
    count: f32,
    // Number of separation neighbours, only used by the debug overlay.
    separation_count: f32,
}

fn accum_init() -> NeighbourAccum {
    var a: NeighbourAccum;
    a.separation = vec3<f32>(0.0);
    a.align_sum = vec3<f32>(0.0);
    a.cohesion_sum = vec3<f32>(0.0);
    a.count = 0.0;
    a.separation_count = 0.0;
    return a;
}

// Folds one neighbour into the accumulators.
//
// `d2` is passed in rather than recomputed because both search strategies already need it for their
// own distance test, and the squared form avoids a square root on the separation path.
fn accum_add(acc: ptr<function, NeighbourAccum>, pos: vec3<f32>, other: Boid, d2: f32) {
    if (d2 < params.r_sep * params.r_sep) {
        // 1/r falloff: dividing by d2 (already squared) yields 1/|d|.
        (*acc).separation = (*acc).separation - (other.pos - pos) / d2;
        (*acc).separation_count = (*acc).separation_count + 1.0;
    }
    (*acc).align_sum = (*acc).align_sum + other.vel;
    (*acc).cohesion_sum = (*acc).cohesion_sum + other.pos;
    (*acc).count = (*acc).count + 1.0;
}

// Converts the neighbour statistics into a steering acceleration.
//
// Reynolds' "steer = desired - current" form is used for alignment and cohesion rather than summing
// raw offset vectors. The difference matters: the raw form grows without bound as a flock
// compresses, while the steering form saturates at `max_speed`, so a dense flock is stable instead
// of exploding.
//
// Density-adaptive weights, a symmetric feedback around the reference density:
//   push      = density^gain                       (= 1 at density == 1 for any gain)
//   w_sep_eff = w_sep * push
//   w_coh_eff = w_coh / push
// `gain == 0` is fixed weights, `gain == 1` the proportional feedback `push == density`. Over-dense
// regions push apart and under-dense ones pull together. The neutral point matters: a flock spawned
// at the reference density must start at equilibrium, otherwise the initial separation impulse blows
// the cluster apart before cohesion can ever act.
fn neighbour_force(acc: NeighbourAccum, pos: vec3<f32>, vel: vec3<f32>) -> vec3<f32> {
    let density = acc.count * params.sep_boost;
    let gain = max(params.coh_falloff, 0.0);
    var push = 1.0;
    if (acc.count > 0.0) {
        push = pow(max(density, 1e-6), gain);
    }
    let w_sep_eff = params.w_sep * push;
    let w_coh_eff = params.w_coh / push;

    var force = acc.separation * w_sep_eff;

    if (acc.count > 0.0) {
        let align_dir = safe_normalize(acc.align_sum);
        if (dot(align_dir, align_dir) > 0.5) {
            force = force + (align_dir * params.max_speed - vel) * params.w_ali;
        }
        let center = acc.cohesion_sum / acc.count;
        let cohesion_dir = safe_normalize(center - pos);
        force = force + (cohesion_dir * params.max_speed - vel) * w_coh_eff;
    }
    return force;
}

// ---------------------------------------------------------------------------------------------
// Environment and world forces
// ---------------------------------------------------------------------------------------------

// SDF avoidance, including the forward probe that produces tangential sliding.
//
// The probe is what makes agents *go around* a rock instead of pressing into it. A pure repulsion
// force has no way to break the symmetry that keeps an agent pinned against a wall: the agent
// pushes in, is pushed out along the same line, and oscillates. Redirecting along the tangential
// component of velocity gives it a direction to escape in.
fn environment_force(pos: vec3<f32>, vel: vec3<f32>, env_id: u32, env_scale: f32, env_freq: f32, env_floor_y: f32) -> vec3<f32> {
    if (env_id == ENV_NONE) {
        return vec3<f32>(0.0);
    }
    let field = FieldArgs(env_id, env_scale, env_freq, env_floor_y);
    let d = eval_field(pos, field);
    let grad = sdf_gradient_at(pos, 0.5 * params.cell_size, field);
    let normal = safe_normalize(grad);

    var force = avoid_force_from(d, normal, params.r_safe, params.sdf_strength);

    let speed = length(vel);
    if (speed > 1e-3) {
        let ahead = pos + vel * (params.sdf_probe / speed);
        if (eval_field(ahead, field) < params.r_safe) {
            let v_tangent = vel - normal * dot(vel, normal);
            let tangent_dir = safe_normalize(v_tangent);
            if (dot(tangent_dir, tangent_dir) > 0.5) {
                force = force + tangent_dir * params.sdf_strength;
            }
        }
    }
    return force;
}

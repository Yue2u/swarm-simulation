// === file: sim/integrate.wgsl ================================================================
// Pass: integration and force application. One invocation per agent.
//
// bindings: @group(0) 0:SimParams(uniform) 1:InteractionUniforms(uniform)
//           2:cell_start(read) 3:cell_end(read)
//           @group(1) 0:boid_src(read) 1:boid_dst(read_write) 2:keys(read, sorted)
// workgroup: 256 x 1 x 1
// dispatch:  ceil(num_boids / 256)
//
// INVARIANTS
//   * exactly one invocation per live agent, gated by `i < params.num_boids`,
//   * reads only `boid_src`, writes only `boid_dst` (distinct buffers bound by the host), so no
//     invocation can observe another invocation's write and there is no need for any barrier,
//   * all reads of `boid_src` happen before any write to `boid_dst` by construction of the host
//     command buffer: the two passes are separate dispatches,
//   * `keys` is only read by the grid variant. It must hold the current sorted key array, which
//     means `clear_cells`, `hash`, the bitonic stages and `build_ranges` all ran earlier in the same
//     frame (`SimPipelines::record_step` records them in that order). Nothing in this pass can tell
//     whether that happened: an unprepared grid finds no neighbours at all rather than wrong ones,
//     because an uncleared `cell_start` is `EMPTY_CELL` and every cell is skipped.
//
// ENTRY POINTS
//   `integrate_naive` - O(N^2) neighbour search, exact and slow. Used for GPU-vs-CPU validation
//                       and for the app when the agent count is small.
//   `integrate_grid`  - 27-cell spatial hash search. The production path.
//
// Both entry points call the same force model in `sim/forces.wgsl`, so the only thing that can
// differ between them is which neighbours they find. `grid/matches_naive` asserts they agree.

//#include "sim/forces.wgsl"
//#include "sim/agents.wgsl"

// ---------------------------------------------------------------------------------------------
// Shared step
// ---------------------------------------------------------------------------------------------

// Applies the force model to one agent and writes the result.
//
// `acc` must already contain the neighbour statistics from whichever search strategy was used.
fn integrate_agent(i: u32, acc: NeighbourAccum) {
    let src = boid_src[i];
    let pos = src.pos;
    let vel = src.vel;

    // Step order is normative; see docs/math.md section 5 and reference.rs for the derivation.
    var force = neighbour_force(acc, pos, vel);
    force = force + wander_force(i, wander_tick(params.time), params.wander);
    force = force + bounds_force(pos, params.bounds_half, params.max_speed);
    force = force + environment_force(
        pos,
        vel,
        params.env_id,
        params.env_scale,
        params.env_freq,
        params.env_floor_y,
    );
    force = force + interaction_force(pos, vel, interaction);

    // Mode-specific vertical term. Fish are slightly buoyant, birds glide on thermals: a single
    // uniform differs between the two modes.
    force.y = force.y + params.buoyancy * params.max_speed;

    let accel = clamp_len(force, params.max_force);
    var new_vel = vel + accel * params.dt;
    new_vel = new_vel * exp(-params.drag * params.dt);
    new_vel = clamp_len_range(new_vel, params.min_speed, params.max_speed);
    let new_pos = pos + new_vel * params.dt;

    store_boid(i, src, new_pos, new_vel);
}

// ---------------------------------------------------------------------------------------------
// Naive (all-pairs) neighbour search
// ---------------------------------------------------------------------------------------------

// Adds every other live agent within the perception radius.
//
// The squared-distance early-out is the whole optimisation: a `length()` call here would make the
// all-pairs path roughly twice as expensive for no benefit.
fn gather_all_pairs(i: u32, pos: vec3<f32>) -> NeighbourAccum {
    var acc = accum_init();
    let r_percept_sq = params.r_percept * params.r_percept;
    let n = params.num_boids;
    for (var j = 0u; j < n; j = j + 1u) {
        if (j == i) {
            continue;
        }
        let other = boid_src[j];
        let d = other.pos - pos;
        let d2 = dot(d, d);
        if (d2 > r_percept_sq || d2 < 1e-8) {
            continue;
        }
        accum_add(&acc, pos, other, d2);
    }
    return acc;
}

@compute @workgroup_size(256, 1, 1)
fn integrate_naive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_boids) {
        return;
    }
    let acc = gather_all_pairs(i, boid_src[i].pos);
    integrate_agent(i, acc);
}

// ---------------------------------------------------------------------------------------------
// Spatial grid neighbour search
// ---------------------------------------------------------------------------------------------

// Adds every agent found in the 27 cells surrounding `pos`.
//
// Three guards, all necessary:
//   * `start == EMPTY_CELL` skips cells with no agents. Without it, the range [EMPTY_CELL, end)
//     would be read as a gigantic bogus range.
//   * the distance test still has to be applied: cells are cubes, so most agents in the 27 cells
//     are further away than the perception radius. The grid only removes the need to *scan* the
//     whole population, it does not make the neighbour test free.
//   * `j == i` must be skipped: an agent is inside its own cell and would otherwise align with
//     itself and separate from itself.
fn gather_grid(i: u32, pos: vec3<f32>) -> NeighbourAccum {
    var acc = accum_init();
    let r_percept_sq = params.r_percept * params.r_percept;
    let base = cell_coord(pos);
    let max_coord = params.grid_dim - vec3<u32>(1u);

    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let coord = clamp(
                    vec3<i32>(base) + vec3<i32>(dx, dy, dz),
                    vec3<i32>(0),
                    vec3<i32>(max_coord),
                );
                let ci = flatten_cell(vec3<u32>(coord));
                let start = cell_start[ci];
                if (start == EMPTY_CELL) {
                    continue;
                }
                let end = cell_end[ci];
                for (var t = start; t < end; t = t + 1u) {
                    let j = keys[t].val;
                    if (j == i) {
                        continue;
                    }
                    let other = boid_src[j];
                    let d = other.pos - pos;
                    let d2 = dot(d, d);
                    if (d2 > r_percept_sq || d2 < 1e-8) {
                        continue;
                    }
                    accum_add(&acc, pos, other, d2);
                }
            }
        }
    }
    return acc;
}

@compute @workgroup_size(256, 1, 1)
fn integrate_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_boids) {
        return;
    }
    let acc = gather_grid(i, boid_src[i].pos);
    integrate_agent(i, acc);
}

// === file: sim/agents.wgsl ==================================================================
// The integration view of bind group 1: the ping-pong pair of agent buffers, the sorted key array,
// and the one helper that writes the destination agent.
//
// BIND GROUP 1 - agent storage. The pair is swapped every step by the host:
//   @binding(0) boid_src storage array<Boid>    (read)
//   @binding(1) boid_dst storage array<Boid>    (read_write)
//   @binding(2) keys     storage array<KeyVal>  (read; the *sorted* keys)
//
// `boid_src` and `boid_dst` are never the same buffer. That is the entire reason the integration
// pass is race-free: it reads only `src` and writes only `dst`, so the order in which invocations
// run does not matter and no barrier is needed inside the pass.
//
// `keys` is binding 2 rather than part of group 0 because the sort writes it (see
// `sim/bindings.wgsl`): a pass may not bind one buffer read-only and read-write at the same time, so
// the key array lives with the buffers whose read/write roles change per pass. Only `integrate_grid`
// reads it; `integrate_naive` declares it because both entry points share this header, and pays
// nothing for the unused binding.
//
// Header only: declarations and one helper, no entry points.

//#include "sim/bindings.wgsl"

@group(1) @binding(0) var<storage, read> boid_src: array<Boid>;
@group(1) @binding(1) var<storage, read_write> boid_dst: array<Boid>;
@group(1) @binding(2) var<storage, read> keys: array<KeyVal>;

// Writes the destination boid and advances its animation phase.
//
// The phase is integrated here rather than on the host so that the tail beat rate stays coupled to
// the actual swim speed, and so that no per-agent CPU work is needed at all.
fn store_boid(index: u32, src: Boid, pos: vec3<f32>, vel: vec3<f32>) {
    var out = src;
    out.pos = pos;
    out.vel = vel;
    out.prev_dir = safe_normalize(vel);
    if (params.mode == MODE_FISH) {
        let speed_fraction = length(vel) / max(params.max_speed, 1e-6);
        out.phase = fract((src.phase + params.dt * (2.0 + 3.0 * speed_fraction)) / TAU) * TAU;
    }
    boid_dst[index] = out;
}

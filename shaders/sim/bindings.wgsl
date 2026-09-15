// === file: sim/bindings.wgsl ================================================================
// Bind group declarations for every simulation compute pass.
//
// Keeping all simulation bindings in one header is deliberate: the pipelines for both worlds
// (fish and birds) must expose *identical* bind group layouts so that switching modes reuses the
// same buffers and bind groups without reallocation. If a future pass needs an extra resource, add
// it here so both modes get it at once rather than diverging.
//
// BIND GROUP 0 - simulation state (same layout for hash, sort, ranges and integrate):
//   @binding(0) params      uniform  SimParams            (written by the host once per frame)
//   @binding(1) interaction uniform  InteractionUniforms  (written by the host once per frame)
//   @binding(2) keys        storage  array<KeyVal, N>     (read)
//   @binding(3) cell_start  storage  array<u32, CELLS>    (read)
//   @binding(4) cell_end    storage  array<u32, CELLS>    (read)
//
// BIND GROUP 1 - agent storage:
//   @binding(0) boid_src    storage  array<Boid, N>       (read)
//   @binding(1) boid_dst    storage  array<Boid, N>       (read_write)
//
// The ping-pong pair lives in its own group because the host swaps which physical buffer is bound
// as src/dst every frame. Everything else is stable for the lifetime of the simulation, so it
// stays in group 0 and is bound once.
//
// `boid_src` and `boid_dst` are distinct buffers, never the same buffer bound twice. That is what
// makes the whole simulation race-free: pass P only ever reads `src` and only ever writes `dst`,
// so ordering between invocations within a pass is irrelevant.
//
// Header only: declares bindings, no entry points.

//#include "common/sdf.wgsl"

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<uniform> interaction: InteractionUniforms;
@group(0) @binding(2) var<storage, read> keys: array<KeyVal>;
@group(0) @binding(3) var<storage, read> cell_start: array<u32>;
@group(0) @binding(4) var<storage, read> cell_end: array<u32>;

@group(1) @binding(0) var<storage, read> boid_src: array<Boid>;
@group(1) @binding(1) var<storage, read_write> boid_dst: array<Boid>;

// ---------------------------------------------------------------------------------------------
// Shared helpers over the simulation state
// ---------------------------------------------------------------------------------------------

// Grid cell coordinate containing `p`, clamped to the grid. Clamping rather than rejecting keeps
// agents that have been pushed outside the domain (by an extreme avoidance force, or by a resize of
// the bounds) in the border cells where they can still be found by a neighbour search, instead of
// falling into an out-of-range hole and never being seen again.
fn cell_coord(p: vec3<f32>) -> vec3<u32> {
    let local = (p - params.grid_min) / params.cell_size;
    return clamp(
        vec3<u32>(max(local, vec3<f32>(0.0))),
        vec3<u32>(0u),
        params.grid_dim - vec3<u32>(1u),
    );
}

// Linear cell index. Must match `GridDims::flatten`: x varies fastest.
fn flatten_cell(c: vec3<u32>) -> u32 {
    return (c.z * params.grid_dim.y + c.y) * params.grid_dim.x + c.x;
}

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

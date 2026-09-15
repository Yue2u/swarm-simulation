// === file: sim/bindings.wgsl ================================================================
// Bind group 0 for every simulation compute pass: the grid's state and the two uniforms.
//
// Keeping the simulation's stable bindings in one header is deliberate: the pipelines for both
// worlds (fish and birds) must expose *identical* bind group layouts so that switching modes reuses
// the same buffers and bind groups without reallocation. If a future pass needs an extra resource,
// add it here so both modes get it at once rather than diverging.
//
// BIND GROUP 0 - the grid and the parameters. Stable for the lifetime of the simulation, bound once:
//   @binding(0) params      uniform  SimParams            (written by the host once per frame)
//   @binding(1) interaction uniform  InteractionUniforms  (written by the host once per frame)
//   @binding(2) cell_start  storage  array<u32>           (read_write; written by `build_ranges`)
//   @binding(3) cell_end    storage  array<u32>           (read_write; written by `build_ranges`)
//
// The cell ranges are declared `read_write` rather than `read` because the range builder writes them
// through this group; the passes that only read them (`integrate_grid`) pay nothing for that. What a
// declaration describes is what the *group* may do, not what each pass does.
//
// BIND GROUP 1 - the pass's working pair, declared by each pass rather than here:
//   sim/agents.wgsl     0: boid_src read   1: boid_dst read_write   2: keys read
//   sim/hash.wgsl       0: boid_src read   1: key_dst  read_write
//   sim/sort.wgsl       0: key_src  read   1: key_dst  read_write
//   sim/ranges.wgsl     0: key_src  read
// It is not declared here because those pairs have different element types, and one declaration
// would force every pass to agree on a type it does not use. All of them have the same
// read/read-write *shape*, so a single bind group layout still covers every pass and the pipeline
// layout stays single, which is what keeps the fish/bird mode switch free of resource churn.
//
// WHY `keys` IS NOT IN GROUP 0
//   It was, until the sort needed to write it. `wgpu` forbids a buffer from being bound read-only and
//   read-write within one dispatch, which makes sense: the two bindings would describe incompatible
//   hazards for the same memory. Group 0 is bound by *every* pass, including the 153 stages that
//   write one key buffer while reading the other, so a read-only `keys` there made the sort
//   impossible. The sorted array is read through group 1 instead, where each pass can say which
//   physical buffer it means.
//
//   The consequence to keep in mind: the array bound to binding 0 of group 1 is *the* key array a
//   pass reads, and it must be the one holding the sorted keys (`keys[0]`), because that is what
//   `build_ranges` and `integrate_grid` derive the grid from. See `sim.rs::KeyPlan`.
//
// `boid_src` and `boid_dst` are always distinct buffers, never the same buffer bound twice. That is
// what makes the simulation race-free: the integration pass only reads `src` and only writes `dst`,
// so the order in which invocations run within a pass does not matter.
//
// The same is true across passes only because every pass is a *separate* compute pass: `wgpu` inserts
// the memory barrier between passes, and there is none between two dispatches recorded into one
// pass. The bitonic sort relies on exactly that, one compute pass per stage.
//
// Header only: declarations and pure helpers, no entry points.

//#include "common/sdf.wgsl"

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<uniform> interaction: InteractionUniforms;
@group(0) @binding(2) var<storage, read_write> cell_start: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_end: array<u32>;

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

// Number of cells the grid actually has, taken from the binding rather than from `params.grid_dim`.
//
// The two agree by construction (`SimResources` sizes the buffers from the same `GridDims` that the
// host packs into `SimParams`), and taking the count from the buffer means a dispatch can never walk
// off the end of the range arrays even if that ever stopped being true.
fn grid_cell_count() -> u32 {
    return arrayLength(&cell_start);
}

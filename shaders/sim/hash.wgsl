// === file: sim/hash.wgsl ====================================================================
// Passes P0 `clear_cells` and P1 `hash`: the two passes that turn agent positions into the input of
// the sort. Together with P2 (`sort_step`) and P3 (`build_ranges`) they build the spatial grid that
// `integrate_grid` searches.
//
// bindings: @group(0) 0:SimParams(uniform) 1:InteractionUniforms(uniform)
//           2:cell_start(read_write) 3:cell_end(read_write)
//           @group(1) 0:boid_src(read, agents) 1:key_dst(read_write, keys)
// workgroup: 256 x 1 x 1
// dispatch:  `clear_cells` ceil(num_cells / 256), `hash` ceil(padded_n / 256)
//
// ENTRY POINTS
//   `clear_cells` - cell_start[c] = EMPTY_CELL for every cell.
//   `hash`        - key_dst[i] = {cell_index(pos_i), i}, and {PAD_KEY, i} for the padding agents.
//
// INVARIANTS
//   * `clear_cells` runs before `hash` in the frame, and `hash` before every reader of `keys`. The
//     passes are separate compute passes, so `wgpu` inserts the barrier between them.
//   * `clear_cells` writes only `cell_start`, `hash` writes only `key_dst`. Neither reads what the
//     other writes, and within each pass every invocation writes a distinct element, so neither pass
//     has a race to guard against.
//   * every invocation is gated on the *array length* of the binding it writes, not on a count from
//     `SimParams`. A dispatch is a multiple of 256 invocations and the grid is not, so the guard is
//     what keeps the tail workgroup inside the buffer.
//   * after `hash`, `key_dst[i].val == i` for every i, so the payload survives the sort and the
//     sorted array is a permutation of the agent indices.
//
// UNUSED BINDINGS
//   Group 1 has three entries in its layout and this pass declares two of them. `wgpu` fills the
//   third with a buffer it never reads; the host binds the agent array the hash itself reads, which
//   is the one buffer in this dispatch that is already marked read-only.
//
// WHY THE CLEAR LIVES HERE
//   A cell that becomes empty has to stop being findable, and `build_ranges` cannot express that: it
//   only ever writes ranges for cells that *do* contain agents, so a cell that was populated last
//   frame and is empty now would keep last frame's range and its stale key entries would be scanned
//   as if they were neighbours. Clearing first is what makes the range build a from-scratch
//   construction rather than an incremental update.
//
//   The alternative, tracking which cells changed, needs a second buffer and a compaction pass to
//   pay for a 4.3 MB memset per frame on the default grid. At 60 fps that memset is 258 MB/s against
//   a memory system that moves tens of GB/s, so it is not worth the complexity.

//#include "sim/bindings.wgsl"

@group(1) @binding(0) var<storage, read> boid_src_hash: array<Boid>;
@group(1) @binding(1) var<storage, read_write> key_dst: array<KeyVal>;

// Sort key for the padding agents: `num_boids` is almost never a power of two, so
// `padded_n - num_boids` entries hold no agent and must not name a cell. `PAD_KEY` is larger than
// any real cell index, which sorts them past every live agent, and `build_ranges` drops any key that
// is not a cell index, which keeps them out of the range arrays entirely.
//
// It deliberately equals `EMPTY_CELL`. The two are different statements about the same value - "this
// key is not a cell" versus "this cell has no agents" - and both have to survive a comparison
// against any real cell index, so the same sentinel is the honest encoding.
const PAD_KEY: u32 = EMPTY_CELL;

@compute @workgroup_size(256, 1, 1)
fn clear_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= grid_cell_count()) {
        return;
    }
    cell_start[c] = EMPTY_CELL;
}

@compute @workgroup_size(256, 1, 1)
fn hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&key_dst)) {
        return;
    }
    // The padding tail of the agent buffer is never written by the host, so its positions are
    // uninitialised memory: it must not be hashed. `val` still carries `i` so that the entry is a
    // valid index into the agent array; nothing reads it, but a garbage index would be an
    // out-of-bounds read the moment anything did.
    if (i >= params.num_boids) {
        key_dst[i] = KeyVal(PAD_KEY, i);
        return;
    }
    key_dst[i] = KeyVal(flatten_cell(cell_coord(boid_src_hash[i].pos)), i);
}

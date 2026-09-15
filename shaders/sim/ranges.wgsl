// === file: sim/ranges.wgsl ==================================================================
// Pass P3 `build_ranges`: turns the sorted key array into the two per-cell range arrays that
// `integrate_grid` searches.
//
// bindings: @group(0) 0:SimParams(uniform) 1:InteractionUniforms(uniform)
//           2:cell_start(read_write) 3:cell_end(read_write)
//           @group(1) 0:key_src(read, sorted)
// workgroup: 256 x 1 x 1
// dispatch:  ceil((padded_n + 1) / 256)     <- one extra invocation, see TERMINATOR
//
// INVARIANTS (what `integrate_grid` is allowed to assume once this pass has run)
//   * `keys` is sorted ascending by `key` and `key_start[s] <= key_start[s+1]`,
//   * for a cell `c` that contains agents: `cell_start[c] < cell_end[c] <= padded_n`, every entry in
//     `keys[cell_start[c] .. cell_end[c]]` has `key == c` and a `val < num_boids`,
//   * for a cell with no agents: `cell_start[c] == EMPTY_CELL`, so the search skips it instead of
//     reading a range left over from a previous frame (`clear_cells` is what makes that true),
//   * the ranges partition the live agents: every padding entry (key == PAD_KEY) is inside no range.
//
// CONCURRENT ACCESS
//   Invocation `i` reads `key_src[i]` and `key_src[i-1]` - both written by the sort in an earlier pass -
//   and writes `cell_start[key(i)]` and `cell_end[key(i-1)]`. Two invocations can write the same
//   element only if they border two runs with the same pair of keys, which cannot happen: a run
//   boundary is unique, so every write here is to a distinct element and the pass has no race.
//
// WHY ONE EXTRA INVOCATION
//   A run is closed by the first entry that belongs to a *different* key, so the run that ends at
//   the last element has no successor to close it. The extra invocation at `i == n` is that
//   successor: it closes the last run and nothing else.

//#include "sim/bindings.wgsl"

// The sorted keys, read through group 1 binding 0. The host binds the key pair whose read slot is
// `keys[0]`: whatever the sort wrote, the sorted array is always there (see `sim.rs::KeyPlan`).
@group(1) @binding(0) var<storage, read> key_src: array<KeyVal>;

@compute @workgroup_size(256, 1, 1)
fn build_ranges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = arrayLength(&key_src);
    let i = gid.x;
    // `i == n` is the terminator; anything past it is the tail of a workgroup that has nothing to do.
    if (i > n) {
        return;
    }
    let cell_count = grid_cell_count();

    if (i == n) {
        let last = key_src[n - 1u].key;
        if (last < cell_count) {
            cell_end[last] = n;
        }
        return;
    }

    let key = key_src[i].key;
    if (i == 0u) {
        if (key < cell_count) {
            cell_start[key] = 0u;
        }
        return;
    }

    let prev = key_src[i - 1u].key;
    if (key == prev) {
        return;
    }
    // The run of `prev` ends here. The guard matters because the padding entries have
    // `key == PAD_KEY`, which is far outside the range arrays: without it, the first padding entry
    // would write `cell_end[PAD_KEY]`, an out-of-bounds write into whatever follows the buffer.
    if (prev < cell_count) {
        cell_end[prev] = i;
    }
    // ...and the run of `key` starts here, unless this entry is itself padding.
    if (key < cell_count) {
        cell_start[key] = i;
    }
}

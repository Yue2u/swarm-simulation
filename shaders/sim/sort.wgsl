// === file: sim/sort.wgsl ====================================================================
// Pass P2 `sort_step`: one stage of a bitonic sort of the key array, by cell index.
//
// bindings: @group(1) 0:key_src(read) 1:key_dst(read_write)
//           @group(0) - bound but not declared: the sort needs nothing but the keys and its stage
//           parameters. The key array cannot live in group 0 because a stage writes one key buffer
//           while the group would bind the other, and a buffer may not be read-only and read-write in
//           the same dispatch (see `sim/bindings.wgsl`).
// immediate: SortParams { j: u32, k: u32, n_padded: u32, pad: u32 }  (16 bytes)
// workgroup: 256 x 1 x 1
// dispatch:  ceil(padded_n / 256), once per stage
//
// WHY IMMEDIATES RATHER THAN A UNIFORM
//   The stage parameters are two integers that change between every one of the 153 dispatches in a
//   frame. Carrying them in a uniform buffer means either 153 buffer writes per frame from the host
//   (`write_buffer` is cheap but not free, and it serialises against the queue) or a dynamic-offset
//   uniform array with one `set_bind_group` per stage. `var<immediate>` + `ComputePass::set_immediates`
//   puts them directly in the command stream: one pipeline, one bind group, 153 dispatches, and the
//   parameters cost nothing but the 8 bytes of payload the hardware needs anyway.
//
// STAGE SEQUENCE (the host walks it; see `sim.rs::sort_stages`)
//   for k in 2, 4, ... n_padded:  for j in k/2, k/4, ... 1:  one dispatch of this entry point
//   `k` is the size of the merge block and decides the direction, `j` is the compare distance.
//   After the last stage the array is ascending by `key`.
//
// BITONIC IS NOT STABLE
//   `key` alone is compared, so agents in the same cell come out in an arbitrary order. Nothing
//   downstream may depend on the order *within* a cell: `build_ranges` only looks at run boundaries,
//   and the neighbour search sums over a cell in whatever order it finds it. The sort's contract is
//   therefore "the array is partitioned by key, ascending", not "this is a stable sort".
//
// CONCURRENT ACCESS
//   Invocation `i` returns immediately unless `i < (i ^ j)`, so exactly one invocation of each pair
//   reads and writes both slots. Every element of `key_dst` is written exactly once, by exactly one
//   invocation, and every read is of `key_src`. That is the whole correctness argument for the
//   pass: it is a permutation written by disjoint invocations, so invocation order does not matter.
//
// WHY THIS IS A PASS PER STAGE AND NOT A LOOP
//   Stage `t+1` reads what stage `t` wrote, and `key_src`/`key_dst` swap between them, so the two
//   dispatches must not overlap. `wgpu` synchronises *passes* - it inserts the memory barrier when a
//   pass ends - but not two dispatches inside one pass. One compute pass per stage is therefore not
//   a style choice, it is the synchronisation.

//#include "common/layout.wgsl"

// The one immediate block of this shader. A module may declare at most one, and every slot it uses
// must be written by `set_immediates` before the dispatch or the dispatch is a validation error
// (`DispatchError::MissingImmediateData`), which is the failure this file is most likely to produce
// while it is being edited.
var<immediate> sort_params: SortParams;

@group(1) @binding(0) var<storage, read> key_src: array<KeyVal>;
@group(1) @binding(1) var<storage, read_write> key_dst: array<KeyVal>;

@compute @workgroup_size(256, 1, 1)
fn sort_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    // `n_padded` is the element count the host sized the buffers with, and it is a power of two. The
    // guard is for the tail workgroup: the dispatch is a multiple of 256 and `n_padded` is not.
    if (i >= sort_params.n_padded) {
        return;
    }
    let partner = i ^ sort_params.j;
    if (partner <= i) {
        return;
    }

    var a = key_src[i];
    var b = key_src[partner];

    // Ascending blocks have their bit at `k` clear, descending blocks have it set. Writing the
    // comparison as "swap when the direction and the current order agree" keeps the two branches
    // from diverging, and the `sort/matches_cpu` check in the test suite is what says it is right.
    let ascending = (i & sort_params.k) == 0u;
    if (ascending == (a.key > b.key)) {
        let swap = a;
        a = b;
        b = swap;
    }
    key_dst[i] = a;
    key_dst[partner] = b;
}

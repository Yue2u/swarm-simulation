# ADR 0003: a direct-index grid and a bitonic sort with immediates

* Status: accepted, day 2
* Date: day 2 of the sprint

## Context

Day 1 left the neighbour search all-pairs, which is exact but O(N^2) and cannot reach the 100k target
(`ADR-0002`). Day 2 has to replace it with a spatial search that fits the frame, on a GPU, without a
CPU loop over agents. Three questions decided the shape:

1. **How is a cell found from a position?** A hash table is the usual answer, but it needs collision
   handling: either atomics that make insertion order non-deterministic, or a counting pass and a
   prefix sum.
2. **How are the agents brought into cell order?** The search walks contiguous per-cell runs, so the
   agents must be sorted by cell index every frame. A bitonic network sorts a power-of-two count in
   `O(n log^2 n)` with no atomics; a radix sort is `O(n)` but needs histogram passes and a scatter.
3. **How do the sort's stage parameters reach 153 dispatches?** They change between every dispatch, so
   either the host writes them per stage, or each stage is its own bind group.

## Decision

**A direct-index grid.** The domain is a fixed axis-aligned box and the agent count does not change, so
`GridDims::for_domain` divides it into cells once, at startup. `cell_of` is a subtract, a divide and a
clamp; the cell index is the array index, with no collision possible. The price is two flat `u32`
arrays (`cell_start`, `cell_end`) sized `grid_dim.x * y * z` - 331,240 entries, 1.3 MB each, for the
default fish world. A cell's agents are the contiguous run `keys[cell_start[c] .. cell_end[c]]`.

**Cell size is pinned to `r_percept`,** not tuned. The search visits the 27 cells around an agent; that
box contains a ball of radius equal to one cell whichever face the agent is nearest. A cell smaller
than the perception radius silently loses every neighbour between `cell_size` and `r_percept` that lies
across a cell face. `config.rs::cell_size_is_at_least_the_perception_radius` is the guard.

**A bitonic sort, one compute pass per stage, parameters in immediates.** `padded_n` is the next power
of two; for `2^m` agents there are `m(m+1)/2` stages (153 at 131,072, one dispatch each). The `(j, k,
n_padded)` triple goes through a `var<immediate>` block and `ComputePass::set_immediates`, so one
pipeline and one bind-group shape cover all 153 dispatches. Because a stage reads what the previous
stage wrote and `wgpu` only barriers *between* passes, the pass-per-stage structure is the
synchronisation, not a style choice.

**The sorted result is pinned to `keys[0]` by the host.** The stage count's parity decides which buffer
the ping-pong ends in, so `KeyPlan` sets the hash pass's destination buffer to that parity. Both
consumers (`build_ranges`, `integrate_grid`) then read one fixed buffer through one fixed bind group,
and no per-frame choice leaks into the frame.

**The grid is rebuilt from scratch every frame.** `clear_cells` (0.12 ms at 100k, measured) marks every
cell empty, which is what stops a cell that emptied this frame from keeping last frame's range.

## Consequences

**Positive.**

* Cell lookup cannot collide and cannot fail, so the search has no probe loop and no worst case.
* No atomics anywhere in the grid build, so the sort and the ranges are deterministic given the agent
  state. A tie within a cell is unordered, but nothing downstream depends on the order within a cell.
* The grid costs 0.36 ms of non-integration work at 100k in the measured run (`docs/perf.md`), and the
  whole simulation cost is dominated by the integration itself.
* Two strategies share the entire force model, so `grid/matches_naive` can compare them directly.

**Negative.**

* The range arrays are allocated for the whole grid whether or not it is populated. At the current
  world size that is 2.6 MB, which is not worth optimising; a much larger domain would want a hash.
* The bitonic sort is not the cheapest sort: at 100k it is 153 dispatches and 5.7 ms in the measured
  run, about 21% of the frame. A radix sort is the obvious follow-up, recorded under alternatives.
* Rebuilding the grid every frame costs a `clear_cells` and a full sort per frame, rather than moving
  only the agents that changed cells. The alternative has a per-agent variable work load and needs
  atomic insertion, which trades a fixed cost for a data-dependent one.

## Alternatives considered

**A hash table (uniform grid over a hash).** Rejected. It needs either atomics - which make the key
order non-deterministic and the ranges order-dependent - or a counting pass and a prefix sum. The
domain here is bounded and known at startup, so the direct index is strictly simpler.

**A Morton-code sort instead of cell index.** Rejected for now. It gives better memory locality in the
27-cell scan at the cost of a more expensive key and a harder argument that the 27 cells are the right
set. Worth revisiting if the integration turns out to be memory-bound on the target.

**A radix sort.** Deferred, not rejected. It is the obvious optimisation if the sort stays at ~20% of
the target frame; bitonic was chosen first because it is one small deterministic network, is far easier
to validate against a CPU reference, and `sort/matches_cpu` checks it on non-power-of-two counts and
both stage parities.

**A dynamic-offset uniform array for the stage parameters.** Rejected. It needs a `set_bind_group` per
stage and a buffer for 12 bytes of data. `sort/immediates_reach_the_shader` is the device check that the
immediates really change per dispatch, because if they did not the sort would silently run its first
stage 153 times.

**Incremental grid maintenance.** Rejected. See the negative consequences: a per-agent variable work
load and an atomic insertion order in exchange for avoiding a 0.12 ms clear and a rebuild that has no
state to get out of step.

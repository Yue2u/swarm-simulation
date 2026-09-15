# ADR 0002: all-pairs first, spatial grid second

* Status: accepted, day 1; the grid lands on day 2
* Date: day 1 of the sprint

## Context

The neighbour search is the only part of the simulation whose cost is not linear in the agent count.
Two implementations are available:

* **All-pairs**: every agent tests every other agent. Exact, no supporting data structures, O(N^2), and
  a very small constant per pair.
* **Spatial grid**: agents are bucketed into cells, sorted by cell, and each agent searches the 27 cells
  around it. O(Nk) with k the number of neighbours, but it needs three passes and a parallel sort, and a
  much higher constant.

The target is 100k agents, where all-pairs is hopeless and the grid is mandatory. The question is only
what to build first, in a sprint where every day has to end with something that runs.

## Decision

Build all-pairs first, use it for the CPU reference test and for small swarms, and add the grid as a
second strategy that shares the entire force model.

The force model lives in `shaders/sim/forces.wgsl` and is called by both entry points. The two
strategies differ *only* in which neighbours they find, so a test can compare them against each other
directly once the grid exists.

## Consequences

**Positive.**

* The force model was validated against the CPU reference on day 1, on the real GPU, to 1e-3 metres per
  agent after one step and to 0.01 in the order parameter after 600. That validation does not depend on
  the sort being correct, so when the grid lands, a failure is unambiguously the grid's fault.
* Small swarms (`--agents 4096`, and every demo scene) run without the sort, which is genuinely faster
  than paying for a sort that a few thousand agents do not need.
* The grid's supporting passes (`clear_cells`, `hash`, `build_ranges`) can be added and tested one at a
  time, because the integration pass does not change when they arrive.

**Negative.**

* The 100k target is not reachable until day 2. The app's default agent count is 4096 with an explicit
  warning above that, rather than a default that produces an unusable frame rate.
* Two integration entry points must be kept in step. This is mitigated by having them share the force
  model: the only duplicated code is the neighbour loop, which is 20 lines each.
* An unprepared grid is a live hazard: with `cell_start` uninitialised, the grid pass would read
  `keys[0..cell_end]` for an arbitrary cell index and consume garbage. `cell_start` is therefore filled
  with `EMPTY_CELL` at allocation, and `grid/unprepared_finds_no_neighbours` asserts that an unprepared
  grid finds nothing.

## Alternatives considered

**Grid only.** Rejected for the sprint: it makes day 1 depend on a correct parallel sort, so a bug in
the sort and a bug in the force model would be indistinguishable, and the whole day could be lost.

**A uniform grid over a hash table.** Rejected. A hash table needs either atomics or a second counting
pass, and the domain here is bounded and known, so a direct index has no collisions and no table. The
price is a preallocated `cell_start`/`cell_end` pair, 4.3 MB each at the default grid. On a desktop GPU
that is not a price worth optimising away.

**Sorting by Morton code instead of by cell index.** Rejected for now. It gives better memory locality
for the neighbour scan at the cost of a more expensive key computation and a slightly harder argument
that the 27 cells are the right set. Worth revisiting if the neighbour scan turns out to be
memory-bound, which the day-2 profiler will show.

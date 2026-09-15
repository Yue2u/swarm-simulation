# Performance

Measured numbers, what limits them, and how to reproduce them. Anything not measured on this page is
marked as such.

## How to measure

```bash
# interactive, logs a line every 120 frames
RUST_LOG=info cargo run --release -p boids-app -- --agents 4096

# one frame, no window
cargo run --release -p boids-app -- --agents 3000 --screenshot /tmp/frame.png
```

The periodic log line reports the smoothed frame time (an exponential average with a short time
constant: a raw per-frame number is unreadable, and a long average hides exactly the hitches worth
seeing). Per-pass GPU timing via timestamp queries lands on day 2, along with the passes that make it
worth having.

## Development machine

WSL2, AMD Ryzen 7 9800X3D, GeForce RTX 4060 Ti. The GPU is *not* reachable from WSL here: the only
Vulkan adapter is `llvmpipe`, a software rasteriser, and the D3D12-backed GL adapter can compute but
cannot present. So all measurements below are CPU software rendering, on a 2560x1440 window.

That makes them pessimistic by a large and unknown factor, and it makes them useful in exactly one way:
they bound the *simulation* cost, which is where the algorithm lives. The numbers the target machine
will produce are not known yet and are not guessed at here.

## Measurements

| agents | neighbour search | resolution | frame time | fps | limiting factor |
|---|---|---|---|---|---|
| 4,096 | all-pairs | 2560x1440 | ~20.5 ms | ~48 | the O(N^2) search |
| 16,384 | all-pairs | 2560x1440 | >130 ms | <8 | the O(N^2) search |
| 3,000 | all-pairs | 400x300 | ~150 ms per *batch of 30 steps* | - | the O(N^2) search |

The 16,384-agent figure is an extrapolation from "120 frames did not complete in 16 seconds" rather
than a measured frame time: at 16x the pair count of the 4,096 case, ~3 fps is what the scaling
predicts, and the run was consistent with that.

The pair counts are the whole story: 4,096 agents is 16.8M pairs per step, and 16,384 is 268M. Nothing
else in the frame comes close. Rendering 4,096 agents at 1440p is 90k triangles in a single instanced
draw call, and the backdrop is one triangle.

**Interpretation.** There is no useful optimisation to make before the spatial grid exists, and the
grid is day 2. The measured crossover matches the estimate in the plan: the all-pairs path is the right
answer below a few thousand agents and the wrong answer above it.

## Known budget for the 50k-100k target

From the plan, to be validated on the target hardware:

| item | budget at 100k agents | notes |
|---|---|---|
| `integrate` with the grid | ~3-5 ms | the dominant compute cost; ~10-25 neighbours per agent |
| bitonic sort of 131,072 keys | ~2-4 ms | 153 dispatches |
| hash + build_ranges + clear | ~1 ms | bandwidth-bound, easy to overlap |
| agent rendering | ~2-3 ms | 100k instances, 66 vertices each |
| background | <0.5 ms | one full-screen triangle |
| post-processing | ~2-3 ms | bloom chain at half resolution |

## Things that were measurably worth doing

* **One instanced draw for the whole swarm, with the mesh and the transform generated in the vertex
  shader.** The alternative, uploading a transform or instance buffer per frame, moves 4.8 MB per frame
  over the bus for no benefit and adds a frame of latency between the simulation and the picture.
* **Ping-pong buffers with two pre-built bind groups.** Rebuilding a bind group per frame allocates a
  driver object sixty times a second.
* **Pre-clearing `cell_start` once at allocation** rather than clearing 4.3 MB every frame. The
  per-frame clear becomes a compute pass on day 2; until then, one `write_buffer` at startup is both
  correct and free.
* **Padding the agent buffers to a power of two** rather than handling a non-power-of-two sort. It
  costs at most 2x memory in the worst case (100,000 pads to 131,072, so 1.3x) and removes an entire
  class of index arithmetic from the sort.

## Things measured and deliberately not optimised

* **The device limits are requested from the adapter** rather than the defaults, even though the
  defaults are lower and "safer". The lower limits silently cap `max_texture_dimension_2d` at 2048,
  which rejects a 1440p window. Correctness first.
* **`Limits::default()` rather than `downlevel_defaults()`** as the base, for the same reason. This
  project does not target WebGL, and paying for that portability with a 2048-pixel texture limit would
  be paying for nothing.
* **Surface reconfiguration failures are logged, not fatal.** A resize races with the compositor;
  crashing on a lost race would make the app unusable while dragging a window edge.

## Not yet measured

* Anything on the target GPU (Vulkan, RTX 4060 Ti). The `WGPU_BACKEND=vulkan` path exists but selects
  `llvmpipe` here.
* Per-pass GPU timings. `TIMESTAMP_QUERY` is requested and granted on both available adapters, and the
  plumbing lands with the sort on day 2.
* Memory bandwidth and occupancy. Neither is a limit at the current agent counts.

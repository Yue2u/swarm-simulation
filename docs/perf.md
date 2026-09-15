# Performance

Measured numbers, what limits them, and how to reproduce them. Anything not measured on this page is
marked as such. The simulation numbers come from `--bench`, which drives the same compute passes the
frame loop does and times each one with GPU timestamp queries.

## How to measure

```bash
# interactive, logs a line every 120 frames, per-pass table every 60 with --profile
RUST_LOG=info cargo run --release -p boids-app -- --agents 4096 --profile

# one frame, no window
cargo run --release -p boids-app -- --agents 3000 --screenshot /tmp/frame.png

# headless simulation timing: 30 timed frames after 20 warmup frames
cargo run --release -p boids-app -- --bench 30 --agents 100000
cargo run --release -p boids-app -- --bench 30 --agents 4096 --strategy naive
```

`--bench` prints the mean per-pass GPU cost over the timed frames, plus a total. It measures the
*simulation* only: no rendering, no presentation. On a machine where the GPU cannot present (which is
where this was developed) that is the only honest thing to report, and it is also the part that matters
for the agent-count target. The readback blocks the CPU once per frame, so the wall-clock frame rate it
prints is a floor, not a measurement; the pass times are unaffected.

`--strategy naive|grid` forces a strategy so the two can be compared on the same machine, same swarm
and same frame count. Without it the choice comes from `Strategy::for_count` (see the crossover below).

## Development machine

WSL2, AMD Ryzen 7 9800X3D, GeForce RTX 4060 Ti, 32 GB. Both GPUs are passed through to WSL, but the two
paths `wgpu` can take here are both wrong for the target:

* **Vulkan** enumerates `llvmpipe` only. The NVIDIA Vulkan ICD is not installed in the distro, so
  `WGPU_BACKEND=vulkan` selects a CPU software rasteriser even though `nvidia-smi` sees the 4060 Ti.
* **GL** goes through Mesa's `d3d12` driver and picks the **integrated** AMD Radeon, not the 4060 Ti,
  and adds a GL-over-D3D12 translation layer on top.

So every number below is the integrated Radeon via D3D12→GL. That is hardware (the test suite's
`is_software` does not flag it), it is not the target 4060 Ti, and the translation layer cost is
unknown. Treat the numbers as *bounded from below in usefulness*: they show the shape of the cost
(sort versus integration, linear versus quadratic) and they are the same adapter across runs, which is
what a comparison needs. They are not a prediction of the target frame rate, and the target is not
guessed at here.

## The crossover: all-pairs versus grid

`Strategy::for_count` chooses between the two searches using the measurement below, and
`NAIVE_AGENT_LIMIT` in `boids-gpu/src/sim.rs` is the pinned result. Mean total GPU time per frame over
30 timed frames, fish world, both strategies on the identical swarm:

| agents | all-pairs | grid |
|---|---|---|
| 512 | 0.23 ms | 0.44 ms |
| 1,024 | 0.42 ms | 0.52 ms |
| 1,536 | 0.61 ms | 0.62 ms |
| 2,048 | 0.81 ms | 0.61 ms |
| 4,096 | 1.80 ms | 0.78 ms |
| 16,384 | 25.5 ms | 2.18 ms |

The two cross just under 1,536 agents. Below it the grid wastes a roughly constant 0.2-0.3 ms on the
sort and its extra dispatches; above it all-pairs loses quadratically (1.8 ms at 4,096, 25.5 ms at
16,384). The constant is pinned at 1,024 rather than at the crossover itself, because being one step
early is cheap and bounded while being one step late is not. A discrete GPU shifts the crossover up -
the all-pairs kernel is the more parallel-friendly shape - which is a reason to re-measure on the target
and not a reason to guess.

## Where the time goes at 100k

Fish world (91 x 40 x 91 grid = 331,240 cells), 100,000 agents, padded to 131,072, 153 sort stages, mean
over 30 frames:

| pass | dispatches | time | share |
|---|---|---|---|
| `clear_cells` | 1 | 0.12 ms | 0.5% |
| `hash` | 1 | 0.20 ms | 0.7% |
| `sort` | 153 | 5.72 ms | 21.4% |
| `build_ranges` | 1 | 0.04 ms | 0.2% |
| `integrate` (grid) | 1 | 20.66 ms | 77.2% |
| **total** | 157 | **26.74 ms** | |

The same table at 4,096 and 1,024 agents, where the sort is 55-78 stages:

| agents | clear | hash | sort | ranges | integrate | total |
|---|---|---|---|---|---|---|
| 1,024 | 0.12 ms | 0.008 ms | 0.33 ms | 0.006 ms | 0.09 ms | 0.55 ms |
| 4,096 | 0.12 ms | 0.014 ms | 0.46 ms | 0.009 ms | 0.20 ms | 0.81 ms |
| 100,000 | 0.12 ms | 0.20 ms | 5.72 ms | 0.04 ms | 20.66 ms | 26.74 ms |

What these say:

* `clear_cells` is flat at ~0.12 ms regardless of agent count, because the grid is sized by the *world*
  (perception radius and bounds) and not by the flock. It is one pass over 331,240 `u32`s, and at 100k
  it is 0.5% of the frame.
* The sort is real money at 100k (5.7 ms, 153 dispatches of a fully random access pattern), which is the
  strongest argument for the deferred radix sort in `ADR-0003`.
* `integrate` dominates and grows super-linearly with density: 0.09 ms at 1,024 is 3.6 ns/agent, 20.7 ms
  at 100k is 207 ns/agent, because the same world at 100x the agents puts far more neighbours in each
  27-cell box. This is the expected shape for a spatial-hash search and it is why the grid exists.
* 26.7 ms of *simulation* is far past a 16.6 ms frame budget on this adapter. None of that number
  applies to the 4060 Ti, which is the point of the caveat above.

## Known budget for the 50k-100k target

From the plan, still to be validated on the target hardware. The measured column is the 100k row above,
for scale only:

| item | planned budget at 100k | measured here (iGPU/D3D12-GL) |
|---|---|---|
| `integrate` with the grid | ~3-5 ms | 20.7 ms |
| bitonic sort of 131,072 keys | ~2-4 ms | 5.7 ms |
| hash + build_ranges + clear | ~1 ms | 0.36 ms |
| agent rendering | ~2-3 ms | not measured (no present path) |
| sky backdrop | <0.5 ms | not measured |
| underwater raymarch | ? | not measured; heaviest render pass (up to 80 SDF steps/pixel) |
| bloom + composite | ~1-2 ms | not measured |

The last two rows are built as of day 3. The post chain is fixed-cost and independent of the agent
count: six full-screen passes at half, quarter and eighth resolution (a bright pass, two downsamples,
two additive upsamples, and the composite), plus one `Rgba16Float` scene target. The underwater
raymarch is the one render pass whose cost scales with the *world*, not the swarm, because every pixel
marches the reef field; `render/ocean.wgsl` is written so that the sample count and the step cap are
the two levers, and half-resolution rendering of that pass alone is the documented fallback
(`ADR-0005`). Both are waiting on the target machine, for the reason in the next section.

## Things that were measurably worth doing

* **One instanced draw for the whole swarm, with the mesh and the transform generated in the vertex
  shader.** The alternative, uploading a transform or instance buffer per frame, moves 6 MB per frame
  over the bus for no benefit and adds a frame of latency between the simulation and the picture.
* **Ping-pong buffers with pre-built bind groups, one per parity per pass shape.** Rebuilding a bind
  group per frame allocates a driver object sixty times a second.
* **Sort stage parameters through immediates rather than a uniform.** One pipeline and one bind group
  cover all 153 stages; the dynamic-offset alternative needs a `set_bind_group` per stage and a buffer
  for 12 bytes. `sort/immediates_reach_the_shader` is the device check that the value really changes per
  dispatch, because the failure would be silent.
* **`KeyPlan` pins the sorted keys to `keys[0]`.** Without it, the hash pass would have to write a
  buffer chosen by the parity of the stage count, and every consumer would need a matching bind group.
* **Padding the agent buffers to a power of two** rather than handling a non-power-of-two sort. At 100k
  that is 1.3x memory and it removes an entire class of index arithmetic from the sort.
* **Rebuilding the grid from scratch every frame** rather than updating it incrementally. The clear is
  0.12 ms and a passes-per-stage rebuild has no state to get out of step; incremental insertion needs
  atomics, order-dependent output and a class of bugs where a cell's range and its contents disagree.

## Things measured and deliberately not optimised

* **The device limits are requested from the adapter** rather than the defaults, even though the
  defaults are lower and "safer". The lower limits silently cap `max_texture_dimension_2d` at 2048,
  which rejects a 1440p window. Correctness first.
* **`Limits::default()` rather than `downlevel_defaults()`** as the base, for the same reason. This
  project does not target WebGL.
* **Surface reconfiguration failures are logged, not fatal.** A resize races with the compositor;
  crashing on a lost race would make the app unusable while dragging a window edge.

## Not yet measured

* Anything on the target GPU (Vulkan, RTX 4060 Ti over its own Vulkan driver). This is the single most
  important number in the project and it needs the target machine.
* Rendering and presentation cost at 100k, which cannot be measured where the GL adapter cannot present
  a swapchain. That includes the underwater raymarch and the whole post chain: both are built and
  covered by the render suite (which checks their output, not their cost), and neither has a measured
  millisecond figure yet.
* Memory bandwidth and occupancy per pass. Neither is a limit at the tested counts, and the profiling
  tools that would report them (`nvidia-smi`, Nsight) target the 4060 Ti, which the GL path here does
  not use.
* The radix sort that would replace the 153-stage bitonic pass, if the sort stays at 20% of the frame on
  the target.

# Architecture

## Crate responsibilities

Five crates, one direction of dependency. Nothing depends on `boids-app`; `boids-core` depends on
nothing but `glam` and `bytemuck`.

```
                 boids-app
                     |
        +------------+------------+
        |            |            |
   boids-render  boids-gpu   boids-scene
        |            |            |
        +------------+------------+
                     |
                 boids-core
```

| Crate | Owns | Does not own |
|---|---|---|
| `boids-core` | every `#[repr(C)]` struct that WGSL also declares, the CPU reference implementation of a step, camera and cursor-ray math, the SDF and terrain fields with their CPU twins, the deterministic RNG and swarm spawn, the `//#include` WGSL preprocessor | any `wgpu` type at all. It builds and tests without a GPU, which is why the force model can be developed and refuted in seconds |
| `boids-gpu` | adapter/device/surface acquisition, the ping-pong agent buffers, the spatial grid buffers, the parameter uniforms, the compute pipelines and their bind groups | rendering, biomes, input |
| `boids-scene` | the per-world numbers behind the look: water medium and reef geometry, post-processing parameters, and (day 4) heightfields, biomes, palettes and mesh prototypes | computation or presentation |
| `boids-render` | the frame graph: HDR and depth targets, scene uniform and its parity bind groups, the sky backdrop, the underwater raymarch, the agent pass, the bloom pyramid and the composite, texture readback, PNG output | simulation state or input |
| `boids-app` | the window, input accumulation, the camera, the frame loop, the mode switch, the CLI, headless screenshots | anything that knows how a pass works |

The split exists so that a change to the force model cannot require a GPU, and a change to a shader
cannot require a window. Both of those paid off within the first day.

## Frame data flow

A frame moves data in one direction. The CPU writes four small uniform structs and nothing else.

```
                     +------------------------------- CPU (once per frame) ---
  input events  -->  | OrbitCamera                   |
                     |  -> CameraUniform             |
                     |  -> SimParams        (144 B)  |
                     |  -> SceneUniform     (304 B)  |
                     |  -> WaterParams       (48 B)  |
                     |  -> InteractionUniforms (48 B)|  <- cursor ray vs plane
                     |  -> PostParams        (48 B)  |
                     +-------------------------------+
                                     |
  ------------------- GPU ------------------------------------------------
  1. clear_cells        cell_start[c] = EMPTY_CELL
  2. hash               keys[i] = {cell_index(pos_i), i}, padding -> PAD_KEY
  3. bitonic sort       153 stages, one compute pass each, keys ascending by cell
  4. build_ranges       cell_start / cell_end per cell
  5. integrate          boids[read] -> boids[write]
                                      |
                     swap ping-pong parity
                                      |
  6. ocean raymarch     Fish: half-res SDF sphere trace -> ocean target (alpha = hit metres)
  7. environment        sky: full-screen atmosphere, no depth write
                        sea: full-res resolve into the scene target, writes depth
  8. ground + trees     sky world only: vertex-pulled terrain, instanced trees, depth write
  9. agent pass         one instanced draw, depth test LessEqual, depth write
                                       |
 10. bloom              bright, two downsamples, two additive upsamples
 11. composite          aberration, exposure, ACES, grade -> swapchain
                                       |
 12. present
```

Passes 7, 8 and 9 share one render pass and one depth attachment, which is what puts a fish behind a
column and a bird behind a ridge: the resolve and the terrain write real distances and the agent pass
tests against them. The underwater raymarch itself runs one pass earlier, at half resolution, and hands
its colour and hit distance to the resolve through one `Rgba16Float` target. Passes 10 and 11 run on the
`Rgba16Float` intermediate and a three-level bloom pyramid, with no depth; the composite owns the
transfer function. The render graph and why the intermediate is HDR rather than 8-bit are `ADR-0004`;
why the underwater environment is a raymarch that ends in a depth-writing resolve is `ADR-0005`.

Passes 1-4 build the spatial grid and are what `Strategy::Grid` records; `Strategy::Naive` skips them
and runs an all-pairs search in pass 5 instead, which is exact and faster below the crossover in
`boids-gpu::sim::NAIVE_AGENT_LIMIT`. The sorted keys always land in `keys[0]` (`KeyPlan`), so both grid
consumers read one fixed group-1 binding. An unprepared grid finds no neighbours rather than wrong ones,
because every cell starts as `EMPTY_CELL`; `grid/unprepared_finds_no_neighbours` asserts that, and the
four `grid/*` device checks are what say the prepared one is right.

## Data ownership and the one rule

**No CPU code iterates over agents.** Not in the frame loop, not in the mode switch, not in the
resize path. The only per-frame CPU work that scales with anything is one `write_buffer` of 144 bytes
and one of 48.

The corollary is that anything that needs per-agent information must read it from the GPU or ask for
it in advance. The rendered mesh is the clearest example: rather than uploading transforms, the vertex
shader derives the orientation basis from the agent's velocity, which it reads from the buffer the
simulation just wrote. There is no transform data to keep in sync because there is no transform data.

## The mode switch

Both worlds' pipelines, buffers and bind groups are created at startup and stay alive. Switching modes
changes:

* `SimMode` inside `SimParams`,
* which `SimConfig` preset drives the parameters,
* `MeshParams` inside `SceneUniform`.

Nothing is reallocated, no pipeline is recompiled, and no `wgpu` object is created or dropped.
`tab` therefore costs one frame, which is what makes the two worlds comparable side by side instead of
a five-second hitch.

The reason this is possible is that the bind group layouts for the fish and bird integration pipelines
are identical by construction: they are two entry points in one shader module sharing one set of
bindings. A separate pipeline per mode with its own bindings would have forced two sets of buffers.

## Comment standard

Comments explain *why*, and every claim stated in a comment must be checkable. Three rules:

**WGSL files.** Every file begins with a contract block:

```wgsl
// === file: sim/integrate.wgsl ================================================
// Pass: integration and force application. One invocation per agent.
//
// bindings: @group(0) 0:SimParams(uniform) 1:InteractionUniforms(uniform) ...
// workgroup: 256 x 1 x 1
// dispatch:  ceil(num_boids / 256)
//
// INVARIANTS
//   * reads only `boid_src`, writes only `boid_dst`, so no invocation can observe another's write
//   * ...
```

Every entry point says what it reads, what it writes and why there is no race. A formula that has a
derivation says where it comes from (`// math.md §4.2`). A non-obvious constant says what it is for and
what happens at the wrong value.

**Rust GPU structs.** Every struct that WGSL mirrors carries a byte-offset table, a `const` assertion
on its size, and a pointer to `gpu-pipeline.md`. Padding fields are explicit and named `_pad`, because
`bytemuck` refuses to derive `Pod` for a struct with implicit padding, and that refusal is a feature: it
turns "this struct has a hole nobody noticed" into a compile error.

**`unsafe`.** Only `derive(Pod, Zeroable)`. A hand-written `unsafe impl` needs a `// SAFETY:` block
listing the invariants it relies on. `#![deny(unsafe_op_in_unsafe_fn)]` is set on every crate.

The standard exists because of a specific failure this codebase is prone to. A layout mistake, a sign
flip in a distance field, or an inverted Y in a ray reconstruction all produce something that runs,
looks plausible, and is wrong. Comments are the only place to record the reasoning that distinguishes
"correct" from "looks fine", and tests are the only place to check it.

## Testing strategy

Three layers, each catching what the others cannot.

**CPU unit tests** (`boids-core`, `boids-render`). Fast, no GPU, and where the force model is
actually developed. The spawn, the SDF primitives, the ray math, the input state machine and the PNG
encoder all have direct tests. Flocking assertions use a *dense* world (`SimConfig::dense`) because
flocking is density-driven: against a world sized for 100k agents, a few hundred test agents would
never see each other and every assertion about flocking would pass or fail for the wrong reason.

**Device checks** (`boids-gpu/tests/gpu`). One device, main thread, sequential. Shader compilation,
struct layout validated against `offset_of!`, GPU-versus-CPU equality after one step and after 600
steps, the SDF and terrain fields against their Rust twins on a 32^3 grid, and the grid: the sorted
keys against a CPU bitonic sort, the range invariants, an unprepared grid finding no neighbours, and the
grid search agreeing with all-pairs after one step and over a long run.

**Render checks** (`boids-render/tests/render_suite`). Renders real frames off screen and asserts
properties of the pixels and the depth buffer: structure in the backdrop, agents contributing pixels,
depth written and plausible, the two worlds differing, the ocean raymarch writing real distances, the
water being blue-dominant, bloom adding light, and the cursor marker being visible. This is the only
automated way to know a picture was produced at all.

Capability gaps are reported as *skipped*, with the reason, rather than failed: a suite that is red
because a backend cannot copy depth to a buffer is a suite people stop reading.

## Environment notes

This was developed under WSL, which shaped several decisions:

* Vulkan enumerates only `llvmpipe` (no NVIDIA Vulkan ICD is installed even though the 4060 Ti is
  passed through), and the GL path goes through Mesa's `d3d12` driver to the integrated Radeon but
  cannot present. Adapter selection therefore happens *after* surface creation, and device limits come
  from the adapter rather than from `downlevel_defaults()`, whose `max_texture_dimension_2d` of 2048
  rejects a 1440p window,
* `libtest`'s per-test threads crash the GL driver on device drop, hence `harness = false`,
* GL cannot copy depth textures to buffers, hence the depth check's skip path.

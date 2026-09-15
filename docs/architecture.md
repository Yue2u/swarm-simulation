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
| `boids-scene` | procedural environments and art content: SDF fields, heightfields, biomes, palettes, mesh prototypes | computation or presentation |
| `boids-render` | the frame graph: depth target, scene uniform and its parity bind groups, the background and agent passes, texture readback, PNG output | simulation state or input |
| `boids-app` | the window, input accumulation, the camera, the frame loop, the mode switch, the CLI, headless screenshots | anything that knows how a pass works |

The split exists so that a change to the force model cannot require a GPU, and a change to a shader
cannot require a window. Both of those paid off within the first day.

## Frame data flow

A frame moves data in one direction. The CPU writes three small uniform structs and nothing else.

```
                     +------------------------------- CPU (once per frame) ---
  input events  -->  | OrbitCamera                   |
                     |  -> CameraUniform             |
                     |  -> SimParams        (144 B)  |
                     |  -> InteractionUniforms (48 B)|  <- cursor ray vs plane
                     +-------------------------------+
                                     |
  ------------------- GPU ------------------------------------------------
  1. clear_cells        cell_start[i] = EMPTY_CELL            (day 2)
  2. hash               keys[i] = {cell_index(pos_i), i}      (day 2)
  3. bitonic sort       keys sorted by cell index             (day 2)
  4. build_ranges       cell_start/cell_end per cell          (day 2)
  5. integrate          boids[read] -> boids[write]           <- today: all-pairs
                                     |
                     swap ping-pong parity
                                     |
  6. background pass    one full-screen triangle, no depth write
  7. agent pass         one instanced draw, depth test + write
                                     |
  8. present
```

Passes 1-4 are recorded but not yet implemented: `Strategy::Naive` skips them, and the grid buffers
stay in their allocated state. That is deliberate rather than a stub, and
`grid/unprepared_finds_no_neighbours` asserts it: an unprepared grid must find no neighbours rather
than read uninitialised memory.

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
struct layout validated against `offset_of!`, and GPU-versus-CPU equality after one step and after
600 steps.

**Render checks** (`boids-render/tests/render_suite`). Renders real frames off screen and asserts
properties of the pixels and the depth buffer. This is the only automated way to know a picture was
produced at all.

Capability gaps are reported as *skipped*, with the reason, rather than failed: a suite that is red
because a backend cannot copy depth to a buffer is a suite people stop reading.

## Environment notes

This was developed under WSL, which shaped several decisions:

* the only Vulkan adapter is `llvmpipe`, and the D3D12-backed GL adapter can compute but cannot
  present. Adapter selection therefore happens *after* surface creation, and device limits come from
  the adapter rather than from `downlevel_defaults()`, whose `max_texture_dimension_2d` of 2048 rejects
  a 1440p window,
* `libtest`'s per-test threads crash the GL driver on device drop, hence `harness = false`,
* GL cannot copy depth textures to buffers, hence the depth check's skip path.

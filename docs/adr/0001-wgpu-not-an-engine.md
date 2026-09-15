# ADR 0001: build on `wgpu` directly, not on a game engine

* Status: accepted
* Date: day 1 of the sprint

## Context

The project needs 50k-100k agents simulated and drawn in real time, with two procedurally generated
worlds and cursor interaction. The obvious paths are Bevy (an engine with a render graph and compute
support), raw `wgpu`, or raw Vulkan/`ash`.

The work is unusual for an engine in three specific ways:

* there is exactly one draw call for the entire simulation, and the scene is generated rather than
  loaded. An engine's asset pipeline, scene graph and entity hierarchy would all be unused,
* the simulation must be expressed as a hand-written compute pipeline with an explicit ping-pong buffer
  discipline and a specific pass order. Engine compute abstractions exist, but they are built to hide
  exactly the buffer layout that this project is *about*,
* the interesting failure modes are numerical, not structural: a struct at offset 12 in Rust and 16 in
  WGSL, a sign flip in a distance field. An engine adds a large layer between the writer and those.

The sprint is three to four days.

## Decision

Build on `wgpu`, `winit`, `glam` and `bytemuck` directly. No engine.

The dependency set is deliberately tiny, and every dependency is one whose behaviour the project
depends on and can therefore reason about:

| crate | why |
|---|---|
| `wgpu` | the only cross-platform way to get compute shaders plus a swapchain in Rust without owning a Vulkan backend |
| `winit` | window and input, nothing else |
| `glam` | vector and matrix types with a `bytemuck` feature, so no conversion glue |
| `bytemuck` | the `Pod` derive *and* its refusal to derive for a struct with implicit padding, which turns a silent layout hole into a compile error |

## Consequences

**Positive.**

* The buffer layouts, the pass order and the residency strategy are all explicit and reviewable. They
  are the subject of the work, so making them visible is a feature.
* Shader changes are visible immediately, and the whole simulation compiles and runs as five small
  crates. The GPU test suite runs in seconds.
* A layout mistake is caught by a device test that reports both byte offsets, not by a black screen.

**Negative.**

* Nothing comes for free: no asset loading, no UI, no debugging overlay, no cross-platform audio. Any
  of those that turn out to be needed is a day of work that an engine would have provided.
* `wgpu` version churn is real. This project is pinned against `wgpu` 30, whose API differs from the
  0.x line in ways that invalidate most examples and tutorials.
* The app must own details an engine hides: surface reconfiguration on resize, adapter selection
  ordering, device-limit negotiation, and driver-specific capability gaps. Three of those produced
  concrete bugs during day 1.

## Alternatives considered

**Bevy.** Rejected. Its render graph and material abstractions would have to be bypassed for the
simulation and for the agent pass, and the parts that remained would be an entity hierarchy with one
entity. The build time alone (several minutes for the first build) is a real cost in a four-day sprint.

**Raw Vulkan via `ash`.** Rejected. It offers more control over memory and barriers, which the project
does not need (it has a handful of buffers and no aliasing), at the cost of writing a swapchain layer,
a shader compiler interface and a validation layer setup.

**A 2D library (`macroquad`, `pixels`).** Rejected. Neither exposes compute shaders, which is the whole
premise.

## Notes

`boids-scene` exists as a crate even though day 1 barely uses it. That is intentional: the procedural
environment code (SDF fields, heightfields, biomes, palettes, mesh prototypes) is the part of the
project with the most churn and the least interdependence, and giving it a crate boundary now means
environment work on days 3 and 4 cannot accidentally acquire a dependency on the frame loop.

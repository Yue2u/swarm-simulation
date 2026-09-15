# ADR 0005: the underwater environment is an SDF raymarch that writes depth

* Status: accepted, day 3
* Date: day 3 of the sprint

## Context

The sky world's backdrop is a full-screen gradient (`render/background.wgsl`), and that is sufficient
because nothing in it has to interact with the agents: a bird in front of a cloud and a bird behind it
look the same. The underwater world is not like that. The reef's columns have to occlude fish and be
occluded by them, and the swarm has to steer around the *same* rock the camera draws. Day 3 therefore
needed an environment that is both a render surface and the collision field, and three questions
followed:

1. **What is the reef made of?** Enumerable geometry (columns placed per world cell) or a procedural
   field evaluated per pixel.
2. **How do fish end up behind a column?** A separate depth prepass, painter's-order sorting, or a pass
   that writes the depth attachment itself.
3. **How is the medium between the camera and the surface shaded?** Per-channel extinction with
   in-scatter, a single-coefficient fog, or a full volumetric integral.

## Decision

**The reef is the simulation's own distance field, evaluated per pixel.** `shaders/common/sdf.wgsl`
declares `reef_field`: a domain-repeated family of tapered columns (plus its offset second family and
the seafloor, fused with `smin`) that is unbounded and procedural. The identical function is used by
`integrate_grid`'s avoidance term and by the render pass, so there is one rock, not two. The CPU twin in
`boids-core/src/sdf.rs` is kept in step by the `sdf/wgsl_matches_rust` device test.

**A full-screen sphere trace writes `frag_depth`.** `render/ocean.wgsl` runs before the agents, marches
the field at up to 80 steps, and writes the clip-space depth of the hit by hand through
`@builtin(frag_depth)`. The agent pipeline's `LessEqual` test then does the rest: a fish behind a column
is rejected, a fish in front of the seafloor is kept. A separate depth prepass would rasterise the same
field twice; painter's-order sorting does not work for a field with no enumerable primitives. Rays that
hit nothing write 1.0, the far plane, rather than writing nothing, so a caller-provided depth target
cannot keep the previous frame's value.

**Sphere tracing is damped because the field repeats.** A plain `t += sdf(p)` is only safe while the
field's Lipschitz constant is at most 1; at a domain-repetition boundary the value can jump upward, so
an unclamped step can tunnel through the edge of a column. The march uses a 0.6 safety factor and a
0.3 m floor, with an acceptance radius that grows with distance (`max(0.02, 0.0025 t)`) so a grazing ray
terminates instead of stepping forever below f32 precision.

**The medium is per-channel Beer-Lambert with in-scatter.** `render/water.wgsl` shades the hit with
`L0 exp(-sigma_e d) + scatter (1 - exp(-sigma_s d))`, where `sigma_e` is
`(0.45, 0.12, 0.06)` scaled per world in `boids_scene::water_params`. Red dying within metres and blue
surviving tens of them is the one property that makes depth read as depth rather than as darkness, and
it is why a single-coefficient fog (which the sky world keeps, for air) is not enough underwater.
Caustics are two layers of animated Worley noise projected along the sun, god rays are a short ray
march with a cheap shadow test, and the fish's own bioluminescence is emissive into the HDR target that
ADR-0004's bloom then spreads.

## Consequences

**Positive.**

* The render surface and the collision surface are the same expression, so they cannot drift. The
  `sdf/wgsl_matches_rust` check compares the CPU twin on a 32^3 grid at `1e-4`.
* No reef geometry is stored: the world is unbounded and costs arithmetic per pixel and no VRAM.
* Real distances in the depth attachment make occlusion between agents and rock a hardware depth test,
  including the agent-to-seafloor relationship, with no sorting.
* The same pass can put the cursor's influence marker behind the rock it hits, because the march already
  knows the hit distance.

**Negative.**

* The pass is the most expensive in the frame: up to 80 field evaluations for the march, six more for
  the shading normal, plus the shaft samples. `docs/perf.md` tracks it separately, and half-resolution
  rendering is the planned lever if the target frame budget needs it.
* The damped marching rule is a heuristic. A field with a sharper repetition would need a smaller
  safety factor, and at some point sphere tracing is the wrong tool and a different representation is.
* The medium model is an approximation of a volumetric integral, shaded at the hit only. It is
  convincing at these extinction values but it is not a participating-medium solve.

## Alternatives considered

**Rasterise explicit column meshes.** Rejected. The domain repetition exists precisely to avoid
enumerating the reef; generating columns per world cell reintroduces the enumeration and a mesh whose
vertices have to match the field the agents use, which is a second source of truth.

**A voxel surface of the reef.** Rejected. It would cost VRAM proportional to the world and need a
rebuild when the field parameters change, to replace arithmetic that the GPU is good at.

**A separate depth prepass.** Rejected. It would draw the field twice per frame to produce what the
colour pass can write on the way past, and the two passes would have to agree on the hit distance.

**A single-coefficient fog underwater.** Rejected. Fog blends toward one colour; water has to blend
toward one colour *per channel* and also absorb the surface behind it, so the same coefficient cannot
express both "the far side is blue" and "the far side is gone".

**Full volumetric ray marching of the medium.** Deferred, not rejected. It is the honest model and it
would give shafts for free, but at 80 steps already spent on the surface it is a substantially larger
per-pixel budget. The current model buys most of the look with a bounded cost.

# ADR 0004: an HDR intermediate and a physically-shaped post chain

* Status: accepted, day 3
* Date: day 3 of the sprint

## Context

Day 1 and 2 drew the scene straight into the swapchain, which was correct while the picture was a
backdrop, a swarm and a distance fog: every value was already a display value in `[0, 1]`. The
underwater world breaks that assumption. The water surface seen from below, the caustics projected onto
the reef, and the bioluminescent fish are all *lights*: they are deliberately brighter than the medium
around them, and the glow that should spread from them only exists if the renderer keeps the ratio
between a caustic and the water next to it. An 8-bit target clamps the caustic at white, and a bloom
pass fed clamped values produces a halo made of clipped pixels rather than of light.

Three decisions followed from that:

1. **What does the geometry pass write?** The swapchain directly, or an intermediate.
2. **How is a glow around a small bright feature produced?** A single blur, or a pyramid.
3. **How does the linear scene reach a display-referred image?** Which tone curve, and where.

## Decision

**Every geometry pass writes an `Rgba16Float` intermediate; only the composite writes the swapchain.**
The format is chosen in `boids-render/src/targets.rs` (`HDR_FORMAT`) and allocated and reallocated with
the depth attachment in `FrameTargets`. Half-float keeps the *exponent* as well as the range, so the
ratio between a caustic at 3.0 and the water at 0.05 survives to the tone map. `Rgba16Float` is
renderable and filterable on every backend this project targets, and it is a fraction of the bandwidth
of `Rgba32Float`.

**Bloom is a three-level pyramid, built once per size.** `PostChain` thresholds the scene with a
soft-knee luminance test into level 0 at half resolution, box-downsamples into a quarter and then an
eighth, and then upsamples back down with a nine-tap tent and additive blending. The additive upsample
is what makes the result a *sum of octaves*: a fish contributes a tight halo from level 0 and a wide one
from level 2, which is what a real lens does, for the cost of one full-resolution pass and a handful of
small ones. The level textures and every bind group that references them are rebuilt as one unit on
resize, so a pyramid can never point half at the previous size.

**The composite owns the transfer function.** Exposure, bloom addition and the Narkowicz ACES fit all
run in linear light, then the display-referred terms (vignette, saturation/contrast/lift, grain) run
after the curve. ACES rather than Reinhard because Reinhard desaturates highlights toward white, which
turns a bright cyan fish into a grey blob. Two composite pipelines exist (`fs_composite` and
`fs_composite_encoded`) because the final transfer function belongs to whoever owns the target: an sRGB
target is encoded by the hardware on write, so the shader emits linear, while a linear target needs the
shader to encode. `SurfaceState::is_srgb` picks.

**The two worlds differ by exposure and white point, not by shader.** `boids_scene::post_params` returns
a `PostParams` per mode. The composite applies `exposure / tonemap_white`, so the white point decides
how much of the HDR range sits below white and the exposure decides apparent brightness; the underwater
white point is more than twice the sky's to keep caustics off the clip, and the exposure is lifted to
match.

## Consequences

**Positive.**

* A caustic, the sun disc through Snell's window, and a bioluminescent fish all keep their relative
  brightness through the frame, which is the difference between "bright" and "glowing".
* Both worlds share one post chain and one set of pipelines. Switching worlds changes one uniform
  buffer's contents, not the render graph.
* The bright pass, the pyramid and the composite are independent render passes over textures, so the
  ordering is enforced by `wgpu`'s barriers between passes rather than by hand.

**Negative.**

* The scene target is 4x the bytes of an 8-bit one, and every geometry pass writes it. At 1440p that is
  about 15 MB for the intermediate plus the pyramid levels, which is a real cost on integrated graphics.
* Two composite pipelines and one bind group layout cover the sRGB and linear cases; a target format
  that is neither would need a third.
* Bloom is a fixed three-level pyramid. A world needing a much larger halo would either raise the level
  count or render bloom at half resolution; `docs/perf.md` records the post chain as not yet measured on
  the target.

## Alternatives considered

**Render straight to the swapchain and fake the glow with a screen-space blur of the clipped frame.**
Rejected. Thresholding an 8-bit frame in display space blooms whatever is already white, which is the
sky and the seafloor as much as the fish. The whole point of the pass is to isolate values that are
genuinely above the scene's midtones, and that information is destroyed by the clamp.

**A single full-resolution blur instead of a pyramid.** Rejected. One blur has one radius; making it
wide enough to read as a lens costs a large kernel at full resolution. The pyramid gets a wide falloff
for the price of a few small passes, and the additive upsample gives several radii at once.

**Bloom in display space after the tone map.** Rejected. Adding light after the curve is not adding
light; it lifts mid-greys rather than bright features, and it makes the exposure and white-point knobs
interact with the glow in a way that is impossible to reason about.

**`Rgba32Float` for the intermediate.** Rejected. The extra precision is not needed for display and it
doubles the bandwidth of the most-written texture in the frame. Half-float's 10-bit mantissa is ample
above the bloom threshold, which is where the range actually matters.

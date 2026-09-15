// === file: render/post_bindings.wgsl ========================================================
// Bind group 0 for the HDR post-processing passes, plus the full-screen vertex shader they share.
//
// BIND GROUP 0 - post state, one bind group per pass:
//   @binding(0) post         uniform PostParams        (exposure, bloom, lens terms)
//   @binding(1) post_sampler sampler                   (linear, clamp-to-edge)
//   @binding(2) post_src     texture_2d<f32>           (the pass's input)
//   @binding(3) post_bloom   texture_2d<f32>           (second input; the composite's bloom texture)
//
// The bloom passes bind their source to both texture slots, which costs nothing and keeps one bind
// group layout for all six post passes. The alternative - a second layout with three bindings - would
// mean two pipeline layouts and two sets of pipelines for a pass that differs from the first by one
// unused binding.
//
// The inert slot must *not* name the pass's own destination: a texture cannot be a colour target and
// a bound resource within one pass, and the pyramid's passes read one level while writing another, so
// naming the wrong one is a validation error rather than a subtle bug.
//
// WHY THE VERTEX SHADER EMITS UV INSTEAD OF DERIVING IT
//   The pass does not know its target size; the texture does. `textureDimensions` on the input gives
//   the texel size for the blur kernel, so the passes need no size uniform at all, and the UV is
//   interpolated from the clip-space triangle exactly as the rasteriser sees it. A size uniform would
//   have to be rebuilt (or written) per level per resize, which is one more piece of state that can
//   silently disagree with the texture it describes.
//
// Header only: declares bindings and the shared vertex stage, no fragment entry points.

//#include "common/math_common.wgsl"

@group(0) @binding(0) var<uniform> post: PostParams;
@group(0) @binding(1) var post_sampler: sampler;
@group(0) @binding(2) var post_src: texture_2d<f32>;
@group(0) @binding(3) var post_bloom: texture_2d<f32>;

/// Vertex output: clip position plus the texture coordinate of the same point.
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // (0, 0) at the top-left of the target, matching `textureSample`'s convention in `wgpu`.
    @location(0) uv: vec2<f32>,
}

// Full-screen triangle, with UV.
//
// One triangle rather than two: its vertices sit outside the clip volume so the interpolated UV
// covers [0, 1] exactly, which removes the diagonal seam a quad can show along the shared edge.
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    var out: VsOut;
    out.clip = vec4<f32>(x, y, 1.0, 1.0);
    // The Y flip happens here: NDC +1 is the top of the screen and texture v = 0 is the top of the
    // texture, so v is built as `0.5 - 0.5 * y`.
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

// Texel size of the input, in UV units.
fn src_texel() -> vec2<f32> {
    let dims = vec2<f32>(textureDimensions(post_src));
    return vec2<f32>(1.0, 1.0) / max(dims, vec2<f32>(1.0));
}

// sRGB encoding, for the case where the target format is *not* an sRGB one and the hardware therefore
// will not encode on write. `SurfaceState::is_srgb` on the host decides which composite pipeline runs.
fn srgb_encode(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

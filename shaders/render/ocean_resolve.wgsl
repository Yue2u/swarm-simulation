// === file: render/ocean_resolve.wgsl ========================================================
// Pass: puts the half-resolution underwater raymarch back at full resolution and writes its depth.
//
// bindings: @group(0)   scene state, shared with every other scene pass (see render/bindings.wgsl)
//           @group(1) 0:color   texture_2d<f32>  the half-size raymarch result, alpha = hit metres
//                     1:sampler   filtering      bilinear for the colour, which hides the half-res
//           draw:      draw(0..3, 0..1)
// depth:     compare Always, write enabled. This is where the raymarch's distance enters the shared
//            depth buffer, before the terrain, the trees and the agents, so all of them occlude
//            against rock correctly.
//
// The colour is sampled with a bilinear tap; the distance is fetched nearest, from the texel that
// contains the fragment, so a depth discontinuity does not blend into a value that belongs to
// neither the rock nor the water behind it. Both come from the same texture in one binding: the
// alpha channel is the distance (see `render/ocean.wgsl` for why it is not a depth texture).
//
// The NDC depth is rebuilt from the metric distance here rather than carried through the half-float
// alpha, because the NDC of a 200 m hit sits within a few thousandths of 1.0 and a half-float would
// quantise it to tens of metres. A metric distance keeps about 0.1 m of resolution up to 200 m.

//#include "render/bindings.wgsl"

@group(1) @binding(0) var ocean_color: texture_2d<f32>;
@group(1) @binding(1) var ocean_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
}

// Full-screen triangle, as in the raymarch: the three vertices lie outside the clip volume so their
// interpolation covers [-1, 1] with no seam.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    var out: VsOut;
    out.clip = vec4<f32>(x, y, 1.0, 1.0);
    return out;
}

struct ResolveOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> ResolveOut {
    // `frag` is the framebuffer coordinate: half the ocean target's dimensions.
    let half_dims = vec2<f32>(textureDimensions(ocean_color));
    let full_dims = half_dims * 2.0;
    let uv = frag.xy / full_dims;
    let max_texel = vec2<i32>(textureDimensions(ocean_color)) - vec2<i32>(1);
    let half_coord = clamp(vec2<i32>(floor(frag.xy * 0.5)), vec2<i32>(0), max_texel);
    let distance = textureLoad(ocean_color, half_coord, 0).a;

    // Bilinear upsample of the shaded medium; the alpha channel is the distance and is not colour.
    var color = textureSample(ocean_color, ocean_sampler, uv);
    color.a = 1.0;

    var depth = 1.0;
    if (distance > 0.0) {
        let near = scene.camera.near;
        let far = scene.camera.far;
        // DirectX-convention NDC depth for a metric distance, the inverse of the test suite's
        // `depth_metres`: z = far / (far - near) * (1 - near / d).
        depth = clamp(far * (1.0 - near / distance) / max(far - near, 1e-6), 0.0, 1.0);
    }

    var out: ResolveOut;
    out.color = color;
    out.depth = depth;
    return out;
}

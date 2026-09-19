// === file: common/heightfield.wgsl ==========================================================
// Sampling helpers for the baked terrain map, shared by the terrain mesh and the tree scatter.
//
// The texture and the parameters are passed in rather than read from module-scope globals: the two
// consumers declare their own bindings in different groups, and a header that silently depends on
// names it does not declare is a header that breaks when a third consumer includes it first.
//
// There is no sampler. The map is not filterable without `FLOAT32_FILTERABLE`, and requiring a device
// feature for the ground would be a poor trade when the interpolation is four lines of arithmetic.
// `textureLoad` also gives exactly what the generating pass stored, which the map comparison test
// relies on.
//
// Header only: no bindings, no entry points.

//#include "common/math_common.wgsl"

// Dimensions of the map, in texels.
//
// Taken from the uniform rather than from `textureDimensions`: the generating pass writes a storage
// texture, and sizing one of those needs the IMAGE_SIZE feature. Reading it from the same place the
// generator did also means the two cannot disagree about the map's size.
fn heightfield_dims(params: TerrainParams) -> vec2<u32> {
    return vec2<u32>(params.resolution);
}

// One texel of the map, clamped to the edge. Returns `(height, biome mask)` in metres and [0, 2).
fn terrain_texel(hf: texture_2d<f32>, dims: vec2<u32>, c: vec2<i32>) -> vec2<f32> {
    let max_texel = vec2<i32>(dims) - vec2<i32>(1);
    return textureLoad(hf, clamp(c, vec2<i32>(0), max_texel), 0).xy;
}

// World `xz` of a texel centre: the inverse of the generating pass's mapping.
fn terrain_texel_centre(params: TerrainParams, dims: vec2<u32>, c: vec2<f32>) -> vec2<f32> {
    return params.min_xz + (c + vec2<f32>(0.5)) * (params.size_xz / vec2<f32>(dims));
}

// Texel-space coordinate of a world `xz` position, continuous: an integer value is a texel centre.
fn terrain_texel_coord(params: TerrainParams, dims: vec2<u32>, world_xz: vec2<f32>) -> vec2<f32> {
    return (world_xz - params.min_xz) / (params.size_xz / vec2<f32>(dims)) - vec2<f32>(0.5);
}

// A height, the surface slope under it and the biome mask there: metres, metres per metre, [0, 2).
struct SurfaceSample {
    height: f32,
    slope: vec2<f32>,
    mask: f32,
}

// Bilinear height at an arbitrary world `xz`, with the slope over a baseline of `slope_texels`.
// Used by the scatter pass, which queries arbitrary positions; the terrain mesh reads texels directly
// because its grid is aligned to them and every one of its samples lands on a texel centre.
//
// The baseline is a parameter because a slope is only meaningful together with the distance it was
// measured over: over one texel the estimate is dominated by the noise's own roughness, over ten it
// averages a hillside. A caller that asks for a footing asks over metres.
fn terrain_sample(
    hf: texture_2d<f32>,
    params: TerrainParams,
    world_xz: vec2<f32>,
    slope_texels: f32,
) -> SurfaceSample {
    let dims = heightfield_dims(params);
    let texel = params.size_xz / vec2<f32>(dims);
    let g = terrain_texel_coord(params, dims, world_xz);
    let base = vec2<i32>(floor(g));
    let f = g - floor(g);

    let c00 = terrain_texel(hf, dims, base);
    let c10 = terrain_texel(hf, dims, base + vec2<i32>(1, 0));
    let c01 = terrain_texel(hf, dims, base + vec2<i32>(0, 1));
    let c11 = terrain_texel(hf, dims, base + vec2<i32>(1, 1));
    // Both channels interpolate together: r is the height, g the biome mask.
    let blended = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);

    // Central difference over the requested baseline, clamped to the map at the border. Each
    // difference carries both channels, so the component that matters is picked out below.
    let step = vec2<i32>(max(i32(slope_texels), 1));
    let hx = terrain_texel(hf, dims, base + vec2<i32>(step.x, 0)).x
        - terrain_texel(hf, dims, base - vec2<i32>(step.x, 0)).x;
    let hz = terrain_texel(hf, dims, base + vec2<i32>(0, step.y)).x
        - terrain_texel(hf, dims, base - vec2<i32>(0, step.y)).x;

    var out: SurfaceSample;
    out.height = blended.x;
    out.slope = vec2<f32>(hx, hz) / (2.0 * slope_texels * texel);
    out.mask = blended.y;
    return out;
}

// Surface normal from a slope, in world space (+Y up).
fn slope_normal(slope: vec2<f32>) -> vec3<f32> {
    return safe_normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
}

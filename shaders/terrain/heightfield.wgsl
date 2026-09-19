// === file: terrain/heightfield.wgsl =========================================================
// Pass: bakes the sky world's heightfield into an RGBA32Float map.
//
// bindings: @group(0) 0:TerrainParams(uniform) 1:height_out(texture_storage_2d<rgba32float, write>)
// workgroup: 16 x 16 x 1
// dispatch:  ceil(resolution / 16) per axis, once at startup
//
// CHANNELS
//   r: terrain height in metres, from `terrain_height` - the same expression the simulation
//      evaluates for its collision field, with the same amplitude and the same frequency.
//   g: biome mask in [0, 2), see `biome_mask`.
//   b, a: unused, left at zero.
//
// WHY RGBA32FLOAT AND NOT RG32FLOAT
//   Two channels of f32 would be the obvious choice, but `rg32float` has no storage support on the
//   GL adapter this is developed against (`adapter.get_texture_format_features` reports no
//   STORAGE_BINDING for it, and half the world's drivers agree), while `rgba32float` is in the
//   WebGPU core storage set and is supported everywhere. The two wasted channels cost 8 MiB at
//   1024^2, which is less than the agent buffers, and exact f32 means the map comparison test can
//   keep the same 1e-4 tolerance the SDF probe uses instead of arguing about half-float rounding.
//
// WHY A MAP AT ALL
//   The mesh and the scatter pass both need a height and its neighbours' heights to build a normal.
//   Evaluating the fbm per query would cost four octaves of value noise per sample per frame;
//   evaluating it once into a map costs it once at startup and turns every later query into a texel
//   load. The map is also what makes the mesh's normals *consistent*: a central difference over the
//   map sees the same surface the mesh displaces to, which is not true of a normal derived from a
//   differently tessellated analytic field.
//
// SAMPLING CONVENTION
//   Texel centres: `world = min_xz + (texel + 0.5) * size_xz / resolution`. The consumers invert
//   exactly that, so a query at a texel centre reads that texel's own value rather than a blend.
//   There is no sampler anywhere: a 32-bit float texture is not filterable without a device
//   feature, and the two places that need interpolation do it explicitly.

//#include "common/sdf.wgsl"

@group(0) @binding(0) var<uniform> terrain: TerrainParams;
@group(0) @binding(1) var height_out: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16, 16, 1)
fn generate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = terrain.resolution;
    if (gid.x >= resolution || gid.y >= resolution) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / f32(resolution);
    let world = terrain.min_xz + uv * terrain.size_xz;

    let height = terrain_height(world, terrain.amplitude, terrain.frequency);
    let mask = biome_mask(world, terrain.biome_frequency);
    textureStore(height_out, vec2<i32>(gid.xy), vec4<f32>(height, mask, 0.0, 0.0));
}

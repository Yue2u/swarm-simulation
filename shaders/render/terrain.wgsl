// === file: render/terrain.wgsl ==============================================================
// Pass: the sky world's ground. One draw call, vertex pulling, no vertex buffer.
//
// bindings: @group(0) 0:SceneUniform(uniform) 1:array<Boid>(storage, read) 2:WaterParams(uniform)
//                     3:InteractionUniforms(uniform) 4:PostParams(uniform) 5:TerrainParams(uniform)
//                     6:heightfield(texture_2d<f32>, read) 7:array<TreeInstance>(storage, read)
//                     8:SkyParams(uniform)
// draw:     draw(0..segments^2 * 2 * 3, 0..1)
// depth:    compare LessEqual, write enabled. Drawn after the backdrop (which writes no depth) and
//           before the trees and the agents, so both are occluded by the ground.
//
// GEOMETRY
//   A regular `segments x segments` grid over the map's extent, generated from `vertex_index`:
//
//       tri  = vi / 3, quad = tri / 2, second = tri % 2
//       cell = (quad % segments, quad / segments)
//       corner offsets: first triangle (0,0) (1,1) (1,0), second (1,0) (1,1) (0,1)
//
//   The winding is counter-clockwise seen from above, which is what makes the ground face the sky
//   with back-face culling on: a terrain that is invisible from above and solid from below is the
//   classic symptom of getting this backwards.
//
//   No clipmap. The camera in both worlds orbits at 0.8-1.15 times the world's diagonal, so a
//   ring-based LOD would spend its finest level on ground that is a few pixels wide. The grid's cell
//   size is instead tied to the map's texel size (four texels per cell), which keeps the mesh and the
//   data it samples in the same resolution regime: see `docs/gpu-pipeline.md`.
//
// SAMPLING
//   Every vertex samples texel *centres*, so no interpolation is involved and the mesh's height is
//   exactly what the generating pass wrote. The normal is a central difference over one mesh cell
//   (± one texel step), which is the same footprint as the drawn triangles, so the shading matches
//   the geometry that produced it.

//#include "render/water.wgsl"

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    /// Biome coordinate in [0, 2): 0 = forest, 1 = dunes, 2 = canyon.
    @location(2) biome: f32,
}

// Texel index of a grid line. `dims / segments` is an integer by construction (the host picks
// `segments = resolution / MESH_TEXELS_PER_CELL`), so interior lines land exactly on texel centres.
fn grid_texel(i: u32, dims: u32, segments: u32) -> u32 {
    return min(i * (dims / segments), dims - 1u);
}

// Everything a vertex needs from the map: where it is, which way the ground faces, and its biome.
struct GroundVertex {
    world: vec3<f32>,
    normal: vec3<f32>,
    biome: f32,
}

// Samples the map at one grid vertex: its own texel, and the four neighbours one mesh cell away for
// the normal's central difference.
fn ground_vertex(cell: vec2<u32>, corner: vec2<u32>, dims: vec2<u32>) -> GroundVertex {
    let segments = max(terrain.segments, 1u);
    let step = max(dims / segments, vec2<u32>(1u));
    let texel_size = terrain.size_xz / vec2<f32>(dims);
    let index = vec2<i32>(
        i32(grid_texel(cell.x + corner.x, dims.x, segments)),
        i32(grid_texel(cell.y + corner.y, dims.y, segments)),
    );

    let centre = terrain_texel(heightfield, dims, index);
    let step_x = vec2<i32>(i32(step.x), 0);
    let step_z = vec2<i32>(0, i32(step.y));
    // The step is clamped inside `terrain_texel`, so the border ring sees a one-sided difference
    // rather than a wrap around the map.
    let hx = terrain_texel(heightfield, dims, index + step_x).r
        - terrain_texel(heightfield, dims, index - step_x).r;
    let hz = terrain_texel(heightfield, dims, index + step_z).r
        - terrain_texel(heightfield, dims, index - step_z).r;

    let xz = terrain_texel_centre(terrain, dims, vec2<f32>(index));
    var out: GroundVertex;
    out.world = vec3<f32>(xz.x, centre.r, xz.y);
    out.normal = slope_normal(vec2<f32>(hx, hz) / (2.0 * vec2<f32>(step) * texel_size));
    out.biome = centre.g;
    return out;
}

// Corner offsets of the two triangles of a quad, both counter-clockwise seen from above (+Y).
//
//   first  triangle: (0,0) -> (1,1) -> (1,0)
//   second triangle: (0,0) -> (0,1) -> (1,1)
//
// The two triangles share the (0,0)-(1,1) diagonal, and the second lists the *same* three corners as
// the naive `(1,0) -> (1,1) -> (0,1)` in the opposite order. That reversal is the whole reason this
// is written out: the naive order winds the second triangle clockwise, back-face culling then removes
// exactly half the ground, and the result is a see-through net that looks like a terrain smaller
// than the world. It is invisible in code review, so the winding is spelled out.
fn quad_corner(second: u32, corner_index: u32) -> vec2<u32> {
    if (second == 0u) {
        if (corner_index == 0u) { return vec2<u32>(0u, 0u); }
        if (corner_index == 1u) { return vec2<u32>(1u, 1u); }
        return vec2<u32>(1u, 0u);
    }
    if (corner_index == 0u) { return vec2<u32>(0u, 0u); }
    if (corner_index == 1u) { return vec2<u32>(0u, 1u); }
    return vec2<u32>(1u, 1u);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let dims = heightfield_dims(terrain);
    let segments = max(terrain.segments, 1u);

    let tri = vi / 3u;
    let quad = tri / 2u;
    let cell = vec2<u32>(quad % segments, quad / segments);
    let corner = quad_corner(tri % 2u, vi % 3u);

    let vertex = ground_vertex(cell, corner, dims);

    var out: VsOut;
    out.clip = scene.camera.view_proj * vec4<f32>(vertex.world, 1.0);
    out.world = vertex.world;
    out.normal = vertex.normal;
    out.biome = vertex.biome;
    return out;
}

// ---------------------------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------------------------

// Ground albedo from the biome mask and the slope.
//
// Three palettes mixed by the mask, then pulled toward bare rock on steep ground: a vertical cliff
// has no soil on it whatever the biome says, and the slope term is what stops a forest from looking
// like a painted hillside in the places the trees cannot stand.
fn ground_albedo(biome: f32, normal: vec3<f32>, grain: f32) -> vec3<f32> {
    let forest = vec3<f32>(0.075, 0.13, 0.048);
    let dunes = vec3<f32>(0.45, 0.38, 0.235);
    let canyon = vec3<f32>(0.38, 0.175, 0.095);
    let rock = vec3<f32>(0.145, 0.135, 0.125);

    let t = clamp(biome, 0.0, 2.0);
    var albedo = mix(forest, dunes, clamp(t, 0.0, 1.0));
    albedo = mix(albedo, canyon, clamp(t - 1.0, 0.0, 1.0));

    // Grain: two lattice hashes at different scales, so the ground has texture at both the metre and
    // the ten-metre scale instead of reading as a flat colour between the trees.
    let variation = 0.82 + 0.36 * grain;
    // `smoothstep01(0.55, 0.2, n.y)` is 1 on a steep face and 0 on flat ground, which is the
    // direction `steep` is meant to run: a vertical cliff has no soil on it whatever the biome says.
    let steep = smoothstep01(0.55, 0.2, normal.y);
    return mix(albedo * variation, rock, steep * 0.75);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = safe_normalize(in.normal);
    let sun = safe_normalize(-scene.light_dir);
    let view = scene.camera.eye - in.world;
    let distance = length(view);
    let dir = view / max(distance, 1e-5);

    let grain = hash21(floor(in.world.xz * 0.9)) * 0.6 + hash21(floor(in.world.xz * 4.3)) * 0.4;
    let albedo = ground_albedo(in.biome, n, grain);

    // Two lights, from the same atmosphere the sky is drawn with: the sun's direct beam, reddened
    // by its own path through the air, and the sky's dome. Both are divided by pi because that is
    // what turns an irradiance into the radiance of a Lambertian surface, and both are in the same
    // units as the sky itself, so a sunlit slope and the sky above it are directly comparable. The
    // ratio between them - roughly 8:1 at normal incidence - is what gives the frame its shape: with
    // the sky doing most of the work the ground reads as flat, because a slope away from the sun is
    // lit almost as much as one facing it.
    let sun_light = sun_irradiance(sun, sky) * max(dot(n, sun), 0.0) / PI;
    let sky_light = sky_radiance(vec3<f32>(0.0, 1.0, 0.0), sun, sky) * (0.3 + 0.7 * max(n.y, 0.0));
    var color = albedo * (sun_light + sky_light);

    color = aerial_perspective(color, dir, distance, scene, sky);
    color = color + focus_marker(scene.camera.eye, dir, distance);
    return vec4<f32>(color, 1.0);
}

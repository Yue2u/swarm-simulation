// === file: render/bindings.wgsl ==============================================================
// Bind group 0 for every scene render pass: what the frame is, and the swarm in it.
//
// BIND GROUP 0 - scene state, stable for the lifetime of the renderer:
//   @binding(0) scene       uniform SceneUniform        (camera + mesh + lighting + fog, per frame)
//   @binding(1) boids       storage array<Boid, N>      (read, the simulation's current read buffer)
//   @binding(2) water       uniform WaterParams         (underwater medium and reef geometry)
//   @binding(3) interaction uniform InteractionUniforms (cursor attractor/repeller, for the marker)
//   @binding(4) post        uniform PostParams          (exposure, bloom, lens terms)
//   @binding(5) terrain     uniform TerrainParams       (heightfield geometry and vegetation)
//   @binding(6) heightfield texture_2d<f32>             (baked height in r, biome mask in g)
//   @binding(7) trees       storage array<StaticInstance> (read, scattered by the terrain scatter pass)
//   @binding(8) sky         uniform SkyParams           (atmospheric scattering coefficients)
//   @binding(9) landmarks   storage array<StaticInstance> (read, the castle, placed on the host)
//
// The boid array is bound as a *storage buffer* rather than a vertex buffer, and the vertex shader
// indexes it with `instance_index`. That is what makes instanced drawing of 100k agents possible
// without a per-frame upload or an instancing vertex buffer: the simulation already produced the data
// on the GPU and the draw call reads it where it lies.
//
// `water`, `interaction`, `post`, `terrain`, `heightfield`, `trees`, `sky` and `landmarks` are here
// rather than in pass-specific groups for the same reason the simulation keeps its uniforms in one
// group: several passes need some of them, and one group means one bind group set per pass instead
// of several. A pass that needs none of them (the bloom pyramid) still binds the group, which costs a
// bind and no bandwidth. The post passes (bloom, composite) do *not* use this group; they sample
// textures and have their own.
//
// Header only: declares bindings and pure helpers, no entry points.

//#include "common/sdf.wgsl"
//#include "common/heightfield.wgsl"
//#include "common/atmosphere.wgsl"

@group(0) @binding(0) var<uniform> scene: SceneUniform;
@group(0) @binding(1) var<storage, read> boids: array<Boid>;
@group(0) @binding(2) var<uniform> water: WaterParams;
@group(0) @binding(3) var<uniform> interaction: InteractionUniforms;
@group(0) @binding(4) var<uniform> post: PostParams;
@group(0) @binding(5) var<uniform> terrain: TerrainParams;
@group(0) @binding(6) var heightfield: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> trees: array<StaticInstance>;
@group(0) @binding(8) var<uniform> sky: SkyParams;
// The castle, in both worlds. Placed on the host rather than scattered on the GPU, because there is
// one per world and its position is a search (`boids_scene::landmark`), not a sampling.
@group(0) @binding(9) var<storage, read> landmarks: array<StaticInstance>;

// ---------------------------------------------------------------------------------------------
// Shared scene helpers
// ---------------------------------------------------------------------------------------------

// The reef field's arguments, for every pass that marches or shades the underwater environment.
//
// The reef's scale parameters live in the water uniform, which the host fills from the same
// `SimConfig` the simulation's `SimParams` come from, so the rock an agent avoids and the rock the
// camera draws stay one field. `freq` is the terrain's, and is unused by the reef.
fn reef_args() -> FieldArgs {
    return FieldArgs(ENV_REEF, water.reef_period, 0.0, water.floor_y);
}

// Depth-cue / medium extinction for the *air* worlds: `1 - exp(-density * distance)`.
//
// Only the sky world uses this. The underwater world attenuates with the per-channel extinction in
// `render/water.wgsl`, because water's absorption is strongly wavelength-dependent and a single
// coefficient cannot express that. Keeping both here, named, is what makes the asymmetry a decision
// rather than an oversight.
fn medium_blend(distance: f32) -> f32 {
    return 1.0 - exp(-scene.fog_density * distance);
}

// Converts an RGB hue triplet to linear RGB. Used to colour agents from their species index so that
// the palette is continuous rather than a set of discrete labels.
fn hue_to_rgb(h: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h) + k) * 6.0 - vec3<f32>(3.0));
    return clamp(p - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Palette entry for an agent.
//
// `color_seed` jitters the hue within the species band so that a school reads as many individuals
// rather than as one flat mass, which is the single biggest visual contributor to "this looks alive"
// in a large swarm.
fn agent_color(species: f32, color_seed: f32, alpha: f32) -> vec3<f32> {
    let h = scene.mesh.hue_base
        + species * scene.mesh.hue_range
        + (color_seed - 0.5) * 0.045;
    return hue_to_rgb(fract(h + alpha));
}

// HSV to RGB with explicit saturation and value, for pulling bioluminescent colours down in value
// while keeping them saturated.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let hp = fract(h) * 6.0;
    let x = c * (1.0 - abs(fract(hp / 2.0) * 2.0 - 1.0));
    var rgb = vec3<f32>(0.0);
    let sector = u32(hp);
    if (sector == 0u) { rgb = vec3<f32>(c, x, 0.0); }
    else if (sector == 1u) { rgb = vec3<f32>(x, c, 0.0); }
    else if (sector == 2u) { rgb = vec3<f32>(0.0, c, x); }
    else if (sector == 3u) { rgb = vec3<f32>(0.0, x, c); }
    else if (sector == 4u) { rgb = vec3<f32>(x, 0.0, c); }
    else { rgb = vec3<f32>(c, 0.0, x); }
    return rgb + vec3<f32>(v - c);
}

// World-space ray through a clip-space position.
//
// Reconstructed from the inverse view-projection rather than from basis vectors, because the same
// reconstruction is used on the CPU for the cursor ray (see `boids_core::camera::ray_from_ndc`). One
// derivation, two consumers, so the mouse and the backdrop cannot disagree about where a pixel points.
fn ray_from_clip(clip: vec2<f32>) -> vec3<f32> {
    let inv = scene.camera.inv_view_proj;
    let near = inv * vec4<f32>(clip, 0.0, 1.0);
    let far = inv * vec4<f32>(clip, 1.0, 1.0);
    return safe_normalize(far.xyz / far.w - near.xyz / near.w);
}

// Clip-space position of a world-space point, for a pass that has to write depth by hand.
//
// `wgpu` clip space has depth in [0, 1] and this project builds its projection with the `directx`
// convention (`OrbitCamera::projection`), so the division below is the whole conversion.
fn clip_of(world: vec3<f32>) -> vec4<f32> {
    return scene.camera.view_proj * vec4<f32>(world, 1.0);
}

// NDC of the pixel `frag` is the centre of, with the Y flip applied exactly once. `frag` is
// `@builtin(position)`: pixel coordinates with the origin at the top-left of the target.
fn pixel_ndc(frag: vec4<f32>) -> vec2<f32> {
    let viewport = scene.camera.viewport;
    return vec2<f32>(
        2.0 * (frag.x) / viewport.x - 1.0,
        1.0 - 2.0 * (frag.y) / viewport.y,
    );
}

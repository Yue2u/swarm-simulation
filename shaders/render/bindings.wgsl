// === file: render/bindings.wgsl ==============================================================
// Bind group declarations shared by every scene render pass.
//
// BIND GROUP 0 - scene state:
//   @binding(0) scene  uniform SceneUniform      (camera + mesh + lighting + fog, once per frame)
//   @binding(1) boids  storage array<Boid, N>    (read, the simulation's current read buffer)
//
// The boid array is bound as a *storage buffer* rather than a vertex buffer, and the vertex shader
// indexes it with `instance_index`. This is what makes instanced drawing of 100k agents possible
// without a per-frame upload or an instancing vertex buffer: the simulation already produced the
// data on the GPU and the draw call reads it where it lies.
//
// Header only: declares bindings, no entry points.

//#include "common/sdf.wgsl"

@group(0) @binding(0) var<uniform> scene: SceneUniform;
@group(0) @binding(1) var<storage, read> boids: array<Boid>;

// ---------------------------------------------------------------------------------------------
// Shared scene helpers
// ---------------------------------------------------------------------------------------------

// Depth-cue / medium extinction: `exp(-density * distance)`.
//
// For the underwater world this is the Beer-Lambert extinction `exp(-sigma_e * d)` with `density`
// standing in for the extinction coefficient, and `fog_color` for the in-scattered light. The same
// expression with a small density gives the birds a thin aerial-perspective haze, so one shader
// serves both worlds; only the uniform changes.
fn medium_blend(distance: f32) -> f32 {
    return 1.0 - exp(-scene.fog_density * distance);
}

// Converts an RGB hue triplet to linear RGB. Used to colour agents from their species index so
// that the palette is continuous rather than a set of discrete labels.
fn hue_to_rgb(h: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h) + k) * 6.0 - vec3<f32>(3.0));
    return clamp(p - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Palette entry for an agent.
//
// `color_seed` jitters the hue within the species band so that a school reads as many individuals
// rather than as one flat mass, which is the single biggest visual contributor to "this looks
// alive" in a large swarm.
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

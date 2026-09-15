// === file: render/background.wgsl ============================================================
// Pass: the procedural sky backdrop of the open-air world. One triangle, no textures, no vertex
// buffer.
//
// bindings: @group(0) 0:SceneUniform(uniform) 1:array<Boid>(storage, read) 2:WaterParams(uniform)
//                     3:InteractionUniforms(uniform) 4:PostParams(uniform)
// draw:     draw(0..3, 0..1)
// depth:    compare Always, write disabled -- this pass fills every pixel and leaves the far plane
//           clear for the agents to depth-test against
//
// Only the sky world uses this pass. The underwater world has its own backdrop in `render/ocean.wgsl`,
// a raymarch that writes real depth so the fish swim behind the reef instead of over it; a gradient
// cannot do that. The two share `render/bindings.wgsl`, so the camera reconstruction, the focus marker
// and the palette helpers cannot drift apart between them.

//#include "render/water.wgsl"

// Full-screen triangle: the three vertices are placed outside the clip volume so their interpolation
// covers [-1, 1] exactly, which avoids the diagonal seam a two-triangle quad can show.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 1.0, 1.0);
}

// Sun glow: a broad halo plus a tight core, both cheap polynomials. A real atmospheric scattering
// integral replaces this when the terrain lands; this is the version that has to look acceptable with
// three arithmetic operations.
fn sun_glow(dir: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let c = max(dot(dir, sun), 0.0);
    let halo = pow(c, 8.0) * 0.35;
    let core = pow(c, 900.0) * 6.0;
    return vec3<f32>(1.0, 0.92, 0.78) * (halo + core);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let dir = ray_from_clip(pixel_ndc(frag));
    let sun = safe_normalize(-scene.light_dir);

    // --- Sky -----------------------------------------------------------------------------------
    // Zenith-to-horizon gradient. The exponent controls how quickly the sky darkens overhead; 0.45
    // keeps a broad bright band near the horizon, which is where a flock silhouettes best.
    let up = clamp(dir.y, -1.0, 1.0);
    let zenith = vec3<f32>(0.16, 0.34, 0.72);
    let horizon = vec3<f32>(0.72, 0.80, 0.88);
    let ground = vec3<f32>(0.22, 0.20, 0.17);

    var color = scene.fog_color;
    if (up >= 0.0) {
        color = mix(horizon, zenith, pow(up, 0.45));
    } else {
        // Below the horizon: a darkening haze rather than a hard edge, so a terrain silhouette blends
        // into the backdrop instead of sitting on a line.
        color = mix(horizon, ground, clamp(-up * 3.0, 0.0, 1.0));
    }
    color = color + sun_glow(dir, sun);

    // The backdrop is at the far plane: nothing can hide the cursor marker here, which is why the
    // limit is the camera's own far distance.
    color = color + focus_marker(scene.camera.eye, dir, scene.camera.far);

    return vec4<f32>(color, 1.0);
}

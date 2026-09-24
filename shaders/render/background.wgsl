// === file: render/background.wgsl ============================================================
// Pass: the sky of the open-air world. One triangle, no textures, no vertex buffer.
//
// bindings: @group(0) 0:SceneUniform(uniform) 1:array<Boid>(storage, read) 2:WaterParams(uniform)
//                     3:InteractionUniforms(uniform) 4:PostParams(uniform) 5:TerrainParams(uniform)
//                     6:heightfield(texture_2d<f32>, read) 7:array<StaticInstance>(storage, read)
//                     8:SkyParams(uniform)
// draw:     draw(0..3, 0..1)
// depth:    compare Always, write disabled -- this pass fills every pixel and leaves the far plane
//           clear for the terrain and the agents to depth-test against
//
// The colour is `common/atmosphere.wgsl`'s single-scattering sky, not a painted gradient: the zenith
// is blue and the horizon is pale because that is what one scattering order through an exponential
// slab of air does, and a low sun is red because its light crossed fifty airmasses to get here. The
// terrain and the agents fade into the same function (`aerial_perspective`), so a ridge line
// dissolving into the sky is not a coincidence of two constants that happen to match.
//
// Below the horizon the same function is evaluated with the elevation clamped, then dimmed: the
// terrain covers most of that region, and where it ends the correct thing to see is haze, not a void.

//#include "render/water.wgsl"

// Full-screen triangle: the three vertices are placed outside the clip volume so their interpolation
// covers [-1, 1] exactly, which avoids the diagonal seam a two-triangle quad can show.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 1.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let dir = ray_from_clip(pixel_ndc(frag));
    let sun = safe_normalize(-scene.light_dir);

    // Above the horizon the sky; below it, the same air seen through the last few hundred metres of
    // haze, dimmed because the light reaching the eye there has already scattered off the ground.
    let height = max(dir.y, 0.0);
    var color = sky_radiance(vec3<f32>(dir.x, height, dir.z), sun, sky);
    if (dir.y < 0.0) {
        color = color * mix(1.0, 0.45, smoothstep01(0.0, -0.25, dir.y));
    }
    color = color + sun_disc(dir, sun, sky);

    // The backdrop is at the far plane: nothing can hide the cursor marker here, which is why the
    // limit is the camera's own far distance.
    color = color + focus_marker(scene.camera.eye, dir, scene.camera.far);

    return vec4<f32>(color, 1.0);
}

// === file: render/background.wgsl ============================================================
// Pass: full-screen procedural backdrop. One triangle, no textures, no vertex buffer.
//
// bindings: @group(0) 0:SceneUniform(uniform) 1:array<Boid>(storage, read)
// draw:     draw(0..3, 0..1)
// depth:    compare Always, write disabled -- this pass runs first and fills every pixel
//
// The two worlds share this shader; only the uniforms differ. That is deliberate: the horizon logic,
// the sun glow and the medium falloff are the same shapes in both, and keeping them in one place means
// the underwater and sky backdrops cannot drift apart stylistically.
//
// Rays are reconstructed from the inverse view-projection rather than from basis vectors, because the
// same reconstruction is used on the CPU for the cursor ray (see camera.rs). One derivation, two
// consumers, no chance of the mouse and the backdrop disagreeing about where a pixel points.

//#include "render/bindings.wgsl"

// Full-screen triangle: the three vertices are placed outside the clip volume so their interpolation
// covers [-1, 1] exactly, which avoids the diagonal seam a two-triangle quad can show.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 1.0, 1.0);
}

// World-space ray through a clip-space position.
fn ray_from_clip(clip: vec2<f32>) -> vec3<f32> {
    let inv = scene.camera.inv_view_proj;
    let near = inv * vec4<f32>(clip, 0.0, 1.0);
    let far = inv * vec4<f32>(clip, 1.0, 1.0);
    return safe_normalize(far.xyz / far.w - near.xyz / near.w);
}

// Sun glow: a broad halo plus a tight core, both cheap polynomials. A real atmospheric scattering
// integral replaces this in the bird world; this is the version that has to look acceptable with three
// arithmetic operations so that it also works as the underwater god-ray source term.
fn sun_glow(dir: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let c = max(dot(dir, sun), 0.0);
    let halo = pow(c, 8.0) * 0.35;
    let core = pow(c, 900.0) * 6.0;
    return vec3<f32>(1.0, 0.92, 0.78) * (halo + core);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let viewport = scene.camera.viewport;
    // Pixel centre, converted to NDC with the Y flip applied exactly once.
    let ndc = vec2<f32>(
        2.0 * (frag.x + 0.5) / viewport.x - 1.0,
        1.0 - 2.0 * (frag.y + 0.5) / viewport.y,
    );
    let dir = ray_from_clip(ndc);
    let sun = safe_normalize(-scene.light_dir);

    var color = scene.fog_color;

    if (scene.mesh.variant > 0.5) {
        // --- Sky world -----------------------------------------------------------------------
        // Zenith-to-horizon gradient. The exponent controls how quickly the sky darkens overhead;
        // 0.45 keeps a broad bright band near the horizon, which is where a flock silhouettes best.
        let up = clamp(dir.y, -1.0, 1.0);
        let zenith = vec3<f32>(0.16, 0.34, 0.72);
        let horizon = vec3<f32>(0.72, 0.80, 0.88);
        let ground = vec3<f32>(0.22, 0.20, 0.17);

        if (up >= 0.0) {
            color = mix(horizon, zenith, pow(up, 0.45));
        } else {
            // Below the horizon: a darkening haze rather than a hard edge, so the terrain silhouette
            // blends into the backdrop instead of sitting on a line.
            color = mix(horizon, ground, clamp(-up * 3.0, 0.0, 1.0));
        }
        color = color + sun_glow(dir, sun);
    } else {
        // --- Underwater ----------------------------------------------------------------------
        // Light entering from above is attenuated with depth, and the two colour bands are the
        // classic ones: warm green near the surface, deep blue further down. The transition is what
        // makes the camera's altitude readable.
        let up = clamp(dir.y, -1.0, 1.0);
        let surface = vec3<f32>(0.20, 0.52, 0.48);
        let deep = vec3<f32>(0.01, 0.06, 0.13);
        let depth_blend = clamp(0.5 - 0.5 * up, 0.0, 1.0);
        color = mix(surface, deep, pow(depth_blend, 0.7));
        // Shafts of light from the surface, faked as a strongly forward-biased lobe. The real
        // volumetric pass replaces this with a SDF shadow-marched integral.
        color = color + sun_glow(dir, sun) * 0.5;
    }

    return vec4<f32>(color, 1.0);
}

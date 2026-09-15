// === file: render/water.wgsl ================================================================
// The underwater medium: Beer-Lambert extinction, the caustic pattern, the water surface and the
// cursor marker. Shared by the ocean raymarch pass and the agent shader so that a fish and the water
// it swims in are shaded by one model.
//
// Header only: declares helpers, no bindings of its own. It reads `scene`, `water` and `interaction`,
// which come from `render/bindings.wgsl`, so it must be included after that file.
//
// Everything here is procedural. There is no water texture anywhere in the project: the medium is
// three exponentials, the caustics are a Worley pattern built from the integer hash shared with the
// CPU (`hash21` in `common/sdf.wgsl`), and the surface is a plane. That is what keeps the underwater
// scene free of assets, and why the pass costs arithmetic instead of bandwidth.
//
// COLOR SPACE: everything returned here is linear HDR. The surface is two orders of magnitude
// brighter than the reef, and that ratio has to survive into the bloom pass, which is the whole reason
// the frame is rendered into a half-float target and tone mapped at the end.

//#include "render/bindings.wgsl"

// Per-channel transmittance over `distance` metres: `exp(-sigma * d)`. math.md section 4.
//
// A vector rather than a scalar because the extinction of water is wavelength-dependent: red is gone
// within about three metres, blue at sixty. Shading with one coefficient makes deep water dark instead
// of blue, which reads as "night" rather than as "underwater".
fn water_transmittance(distance: f32) -> vec3<f32> {
    return exp(-water.extinction * max(distance, 0.0));
}

// Beer-Lambert with in-scatter: what an object at `distance` looks like through the medium.
//
// `scene.fog_color` is the in-scattered light, i.e. what the medium itself glows with once it is
// optically thick. Without that term distant water tends to black and the frame loses the only cue
// that the medium has a colour of its own.
fn water_medium(color: vec3<f32>, distance: f32) -> vec3<f32> {
    let transmit = water_transmittance(distance);
    return color * transmit + scene.fog_color * (1.0 - transmit);
}

// Mean extinction, for the few places a single scalar weight is needed (the shaft integral).
fn water_extinction_mean() -> f32 {
    return (water.extinction.x + water.extinction.y + water.extinction.z) / 3.0;
}

// ---------------------------------------------------------------------------------------------
// Caustics
// ---------------------------------------------------------------------------------------------

// Distance to the nearest feature point of a jittered Worley cell pattern, in cell units.
fn worley(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = fract(p);
    var best = 8.0;
    for (var j = -1; j <= 1; j = j + 1) {
        for (var i = -1; i <= 1; i = i + 1) {
            let c = cell + vec2<f32>(f32(i), f32(j));
            // Jitter inside the cell, so the pattern is irregular rather than a grid.
            let offset = vec2<f32>(hash21(c), hash21(c + vec2<f32>(31.0, 57.0)));
            best = min(best, length(c + offset - p));
        }
    }
    return best;
}

// The caustic pattern: thin bright ridges where the surface's waves focus sunlight.
//
// Two Worley layers at different scales, drifting at different rates. One layer reads as a regular
// grid; the interference of two is what makes the pattern look pushed around by a moving surface.
// `1 - d` raised to a high power keeps only the bright ridges, which is what a caustic is: a fold in
// the light field, not a smooth blob.
fn caustic_pattern(p: vec2<f32>, time: f32) -> f32 {
    let scale = max(water.caustic_scale, 1e-3);
    let drift = time * water.caustic_drift;
    let a = 1.0 - clamp(worley(p * scale + vec2<f32>(drift * 0.35, drift * 0.21)), 0.0, 1.0);
    let b = 1.0
        - clamp(
            worley(p * scale * 1.9 - vec2<f32>(drift * 0.23, drift * 0.4)),
            0.0,
            1.0,
        );
    return pow(a, 9.0) + pow(b, 9.0) * 0.7;
}

// ---------------------------------------------------------------------------------------------
// The water surface, seen from below
// ---------------------------------------------------------------------------------------------

// Cosine of the critical angle of the water-air interface, `cos(asin(1 / 1.333))`.
//
// Inside it the surface is a window onto the sky (Snell's window); outside it, total internal
// reflection makes it a mirror of the water below. That boundary is the most recognisable feature of
// looking up from underwater, and it is one `smoothstep` to place exactly.
const SNELL_COS: f32 = 0.7494;

// Radiance of the surface for a ray that reaches the surface plane at `p`.
fn water_surface(dir: vec3<f32>, sun: vec3<f32>, p: vec3<f32>, time: f32) -> vec3<f32> {
    let elevation = clamp(dir.y, -1.0, 1.0);

    // The window: bright sky and a sun disc. Outside it: a dim reflection of the water below.
    let sky = scene.fog_color * 2.4 + vec3<f32>(0.30, 0.50, 0.58) * water.surface_glow;
    let outside = scene.fog_color * 1.1 + vec3<f32>(0.02, 0.05, 0.07) * water.surface_glow;
    let window = smoothstep(SNELL_COS - 0.06, SNELL_COS + 0.02, elevation);

    // Wave perturbation, evaluated on the surface itself: the pattern that throws the shafts and the
    // pattern on the floor come from one field, so they move together.
    let waves = caustic_pattern(p.xz, time) * 0.35;
    // The sun is the one thing up there bright enough to survive the interface, and it has to keep
    // enough energy to bloom.
    let to_sun = max(dot(dir, -sun), 0.0);
    let disc = pow(to_sun, 220.0) * 40.0 * window;

    return mix(outside, sky, window) * (1.0 + waves) + vec3<f32>(1.0, 0.95, 0.85) * disc;
}

// ---------------------------------------------------------------------------------------------
// God rays
// ---------------------------------------------------------------------------------------------

// Visibility toward the sun from `p`, as a product of four samples.
//
// Four samples rather than a real shadow march: a shaft is a wide, soft feature and this is paid per
// step of the shaft integral, i.e. twelve times per pixel. Four is enough to give a column a hard
// enough shadow edge to read as a silhouette without turning the pass into a second raymarch.
fn sun_visibility(p: vec3<f32>, sun: vec3<f32>, distance: f32) -> f32 {
    var vis = 1.0;
    for (var i = 1u; i <= 4u; i = i + 1u) {
        let q = p - sun * (distance * (f32(i) / 4.0));
        vis = vis * smoothstep(0.0, 6.0, eval_field(q, ENV_REEF, water.reef_period, water.floor_y));
    }
    return vis;
}

// In-scattered sunlight along a view ray, weighted by the moving surface pattern.
//
// This is what produces the shafts: light is only collected where the view ray passes below an
// unobstructed patch of the *moving* caustic field, so as the pattern drifts the shafts sweep through
// the water. Sampling the pattern where the sample projects back onto the surface plane, along the
// sun direction, is what makes the shafts line up with the pattern on the floor instead of being an
// independent noise field.
//
// The result is the in-scatter integral `sigma_s * integral(source * visibility * exp(-sigma_e t) dt)`
// with `sigma_s = sigma_e` (a particle that absorbs light also scatters it, in the single-scattering
// approximation this pass makes). Writing the `sigma` factor explicitly is what keeps the shafts at
// the same order of magnitude as the surfaces they shine on: without it the sum over twelve samples of
// several hundred metres is a large number with no physical units, and the frame saturates.
// `godray_strength` is then a plain multiplier around 1, not a fudge factor.
fn water_shafts(ro: vec3<f32>, rd: vec3<f32>, t_end: f32, sun: vec3<f32>, time: f32) -> vec3<f32> {
    if (water.godray_strength <= 0.0) {
        return vec3<f32>(0.0);
    }
    let steps = 12u;
    let span = min(t_end, 320.0);
    let ds = span / f32(steps);
    let sigma = water_extinction_mean();

    var acc = 0.0;
    for (var i = 0u; i < steps; i = i + 1u) {
        let t = (f32(i) + 0.5) * ds;
        let p = ro + rd * t;

        var entry = p;
        if (abs(sun.y) > 1e-3) {
            entry = p - sun * ((water.surface_y - p.y) / sun.y);
        }
        let source = 0.4 + caustic_pattern(entry.xz, time);
        let to_surface = max(water.surface_y - p.y, 0.0);
        let vis = sun_visibility(p, sun, min(to_surface, 60.0));
        acc = acc + source * vis * exp(-sigma * t);
    }
    let in_scatter = sigma * acc * ds;
    return vec3<f32>(0.72, 0.86, 0.92) * (in_scatter * water.godray_strength);
}

// ---------------------------------------------------------------------------------------------
// The cursor's influence point, visualised
// ---------------------------------------------------------------------------------------------

// Adds a soft marker where the cursor's influence is centred, which is the only feedback the user
// gets about where that is.
//
// The marker is a sphere intersected analytically by the view ray: drawing it as geometry would mean
// a second pipeline and a per-frame instance buffer for one object, and the ray is already in hand.
// `t_limit` is the distance to the nearest environment geometry, so the marker is correctly hidden
// behind rock without a depth test.
fn focus_marker(ro: vec3<f32>, rd: vec3<f32>, t_limit: f32) -> vec3<f32> {
    if (interaction.mode == INTERACTION_OFF || interaction.radius <= 0.0) {
        return vec3<f32>(0.0);
    }
    // A fixed fraction of the influence radius: the influence is tens of metres across in a world
    // hundreds of metres wide, and a marker drawn at that size would fill the frame.
    let r = interaction.radius * 0.05;
    let to_focus = interaction.focus_point - ro;
    let t = dot(to_focus, rd);
    if (t <= 0.0 || t > t_limit) {
        return vec3<f32>(0.0);
    }
    let d = length(to_focus - rd * t);
    if (d > r) {
        return vec3<f32>(0.0);
    }
    // A ring with a soft centre rather than a disc: a disc hides the agents behind it, and the point
    // of the marker is to say where the centre is, not to cover it.
    let ring = smoothstep(r, r * 0.62, d) * smoothstep(r * 0.35, r * 0.62, d);
    let glow = 1.0 - smoothstep(0.0, r, d);
    var tint = vec3<f32>(0.35, 0.75, 1.0);
    if (interaction.mode == INTERACTION_REPEL) {
        tint = vec3<f32>(1.0, 0.45, 0.3);
    }
    return tint * (ring * 1.6 + glow * 0.35);
}

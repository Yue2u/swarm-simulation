// === file: common/atmosphere.wgsl ===========================================================
// Analytic single-scattering sky, and the aerial perspective that goes with it.
//
// Everything it needs is passed in: `render/bindings.wgsl` includes this file, so it cannot read the
// bindings that file declares, and a header that referenced them would fail the standalone
// compilation check in `shaders/compile_all` as well.
//
// THE MODEL (docs/math.md section 6)
//   The atmosphere is a flat, exponential, two-component slab: Rayleigh (molecular, blue, 8 km scale
//   height) and Mie (aerosol, grey and forward-scattering, 1.2 km). For a view direction `dir` the
//   optical depth is
//
//       tau(dir) = beta_r * H_r * m + beta_m * H_m * m,        m = 1 / max(dir.y, EPS)
//
//   where `m` is the airmass, i.e. the slant path over the vertical one. Light scattered once into
//   the view is
//
//       L = I_sun * exp(-tau_sun) * (beta_r * P_r + beta_m * P_m) / (beta_r + beta_m) * (1 - exp(-tau))
//
//   which is Beer-Lambert again, with a direction-dependent in-scatter colour. Two of the frame's
//   most recognisable features fall out of it rather than being painted in: `(1 - exp(-tau))`
//   saturates toward the horizon, where the path is longest, so the horizon is pale; and
//   `exp(-tau_sun)` reddens a low sun, so a sunset is red because the sunlight crossed 50 airmasses
//   of blue-scattering air to reach the eye.
//
// WHY NO PLANET CURVATURE
//   The world is 1.5 km across and the scale heights are kilometres, so a horizon is flat to well
//   inside the noise floor of an f32. Modelling a 6.4 Mm planet would make the sky a single colour
//   at this scale and put the horizon in a place the camera never reaches.
//
// Header only: no bindings, no entry points.

//#include "common/math_common.wgsl"

// Airmass cap at and below the horizon. 0.02 is about 1.1 degrees of elevation: without a cap the
// airmass diverges and the horizon becomes an infinite white line one pixel wide.
const ATMOSPHERE_MIN_ELEVATION: f32 = 0.02;

// Rayleigh phase function, normalized so its integral over the sphere is 1.
fn rayleigh_phase(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

// Henyey-Greenstein phase function. `g` is the anisotropy: 0.76 is the usual forward-scattering
// haze, which is what puts a bright halo around the sun and leaves the anti-solar sky clean.
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4);
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

// Ratio of a slant path to the vertical one.
fn airmass(dir_y: f32) -> f32 {
    return 1.0 / max(dir_y, ATMOSPHERE_MIN_ELEVATION);
}

// Optical depth of the whole slab along `dir`, per channel.
fn atmosphere_tau(dir: vec3<f32>, p: SkyParams) -> vec3<f32> {
    let m = airmass(dir.y);
    return p.beta_rayleigh * (p.ray_scale_height * m) + p.beta_mie * (p.mie_scale_height * m);
}

// Radiance of the sky in direction `dir`, in linear HDR units.
fn sky_radiance(dir: vec3<f32>, sun: vec3<f32>, p: SkyParams) -> vec3<f32> {
    let tau = atmosphere_tau(dir, p);
    let transmittance = exp(-tau);
    let cos_theta = dot(dir, sun);

    let beta_total = p.beta_rayleigh + p.beta_mie;
    let scatter_colour = p.beta_rayleigh * rayleigh_phase(cos_theta)
        + p.beta_mie * henyey_greenstein(cos_theta, p.mie_g);

    // The sun's own path, which is what reddens it near the horizon.
    let sun_transmittance = exp(-atmosphere_tau(sun, p));
    let boost = 1.0 + p.horizon_boost * smoothstep01(0.25, 0.0, dir.y);

    return p.sun_intensity * sun_transmittance * (scatter_colour / max(beta_total, vec3<f32>(1e-8)))
        * (1.0 - transmittance) * boost;
}

// Illuminance of the sun's direct beam, in units of the sky's own radiance.
//
// Not derived from `sun_intensity`, which scales the *sky's* in-scatter: a single-scattering model
// returns a fraction of the incident light, so the two are only related through the exposure the
// frame is tone mapped at. This is that exposure, stated once: at 3.2 a white surface facing the sun
// reflects about 1.0, and everything darker than white lands inside the tone curve instead of on top
// of it. The real ratio of sunlight to skylight is nearer 10; the sky is deliberately lifted here,
// because a frame in which the ground is eight times the sky is a frame of blown highlights.
const SUN_ILLUMINANCE: f32 = 3.2;

// Irradiance of the sun's direct beam on a surface facing it, per channel.
//
// The sun's own path through the air is what reddens it: the same `exp(-tau_sun)` that colours the sky
// near the horizon colours the light landing on the ground, so a sunset lights the terrain from the
// side in the same colour the sky is glowing with. Shading with a sun colour from anywhere else
// produces a landscape lit by light that does not match its own sky.
fn sun_irradiance(sun: vec3<f32>, p: SkyParams) -> vec3<f32> {
    return vec3<f32>(SUN_ILLUMINANCE) * exp(-atmosphere_tau(sun, p));
}

// The sun's disc, which single scattering does not produce: the model above is the halo, this is
// the source. `power` is high enough that the disc is a few pixels across at a 55 degree field of
// view, which is wider than the real sun and reads better against a procedural sky.
fn sun_disc(dir: vec3<f32>, sun: vec3<f32>, p: SkyParams) -> vec3<f32> {
    let c = clamp(dot(dir, sun), 0.0, 1.0);
    let disc = pow(c, 4000.0);
    return vec3<f32>(1.0, 0.94, 0.85) * (disc * p.sun_intensity * 40.0 * exp(-atmosphere_tau(sun, p)));
}

// Aerial perspective: what distance does to a surface's radiance in air.
//
// `scene.fog_density` is the art-directed haze, *not* the physical extinction above: at this world's
// scale the physical term is under 3% per kilometre and would leave the hills perfectly crisp, which
// is not what a hazy valley looks like. The haze's *colour* is the sky in the view direction rather
// than a constant, which is what makes a ridge line disappear into the sky it stands against instead
// of into a grey rectangle.
fn aerial_perspective(
    color: vec3<f32>,
    dir: vec3<f32>,
    distance: f32,
    scene: SceneUniform,
    sky: SkyParams,
) -> vec3<f32> {
    let sun = safe_normalize(-scene.light_dir);
    let haze = mix(scene.fog_color, sky_radiance(dir, sun, sky), sky.aerial_boost);
    let t = exp(-scene.fog_density * distance);
    return mix(haze, color, t);
}

//! The sky world's air, as the numbers the render passes read.
//!
//! # Why these numbers are physical
//!
//! The atmosphere in `shaders/common/atmosphere.wgsl` is a single-scattering integral, so its inputs
//! are not "sky brightness" knobs but scattering coefficients in metres^-1 and scale heights in
//! metres. Handing it the real values for air means the *relationships* come out right for free: the
//! sky is blue because Rayleigh scattering goes as `lambda^-4`, the horizon is pale because the sight
//! line there crosses tens of airmasses, and a low sun is red because its own light crossed them
//! first. Tuning those three things by hand would be three constants that have to be kept consistent
//! with each other; here they are consequences of four numbers that describe air.
//!
//! The one artistic value is [`SUN_INTENSITY`], which is the exposure of the sky relative to the
//! rest of the frame. Everything else is either physics or a stated approximation.

use boids_core::layout::{SimMode, SkyParams};

/// Rayleigh scattering coefficient of air at sea level, m^-1, per channel.
///
/// The real coefficient for 680/550/440 nm light. The ratio between the channels is what makes the
/// sky blue and the setting sun red; the absolute value sets how much air is enough to matter, and
/// 8 km of scale height gives an optical depth of about 0.1 vertically.
const BETA_RAYLEIGH: [f32; 3] = [5.8e-6, 13.5e-6, 33.1e-6];

/// Mie (aerosol) scattering coefficient, m^-1. Nearly grey, and much more forward-scattering.
const BETA_MIE: [f32; 3] = [1.5e-5, 1.5e-5, 1.5e-5];

/// Rayleigh scale height, metres. Molecular air thins slowly with altitude.
const RAY_SCALE_HEIGHT: f32 = 8000.0;

/// Mie scale height, metres. Aerosols hug the ground, which is why haze sits in valleys.
const MIE_SCALE_HEIGHT: f32 = 1200.0;

/// Henyey-Greenstein anisotropy of the Mie term. 0.76 is the standard figure for haze: strongly
/// forward-scattering, so a sun halo is bright and the rest of the sky is clean.
const MIE_G: f32 = 0.76;

/// Sun radiance scale: the only artistic value in the struct.
///
/// The single-scattering integral returns the *fraction* of the sun's light that scattered once into
/// the view, which for a vertical sight line is a few percent. A tropical noon sky is about
/// 1/5th as bright as the sun's disc, and the frame's tone curve puts the sky's zenith near 0.35
/// after exposure, so this is where the two meet.
const SUN_INTENSITY: f32 = 26.0;

/// Extra brightness near the horizon.
///
/// The single-scattering term already brightens toward the horizon (`1 - exp(-tau)` saturates), but
/// real horizons are brighter still: light that scattered *twice* arrives there in quantity, and
/// multiple scattering is precisely what this model does not integrate. This is the stand-in for it,
/// and it is the reason a sunset has a glow under the cloud line instead of a sharp edge.
const HORIZON_BOOST: f32 = 0.7;

/// How much of the aerial-perspective haze takes its colour from the sky rather than from
/// `SceneUniform::fog_color`.
///
/// At 1.0, distant terrain dissolves exactly into the sky it stands against, which is what real
/// aerial perspective does. Slightly under it keeps a hint of the world's own haze colour, which
/// reads as weather rather than as an exact optical model.
const AERIAL_BOOST: f32 = 0.85;

/// Sky parameters for a world.
///
/// The underwater world has no sky, but the boid shader and the shared headers still read the
/// struct; it gets the same air so that nothing has to branch, and the ocean pass simply never asks
/// for it.
#[must_use]
pub fn sky_params(mode: SimMode) -> SkyParams {
    SkyParams {
        beta_rayleigh: BETA_RAYLEIGH,
        // The sun is dimmer underwater, but nothing underwater reads this: the ocean pass has its
        // own surface and shafts. Kept at the same value so a switch of worlds cannot change the
        // sky that frame's history was built from.
        sun_intensity: SUN_INTENSITY,
        beta_mie: BETA_MIE,
        mie_g: MIE_G,
        ray_scale_height: RAY_SCALE_HEIGHT,
        mie_scale_height: MIE_SCALE_HEIGHT,
        horizon_boost: match mode {
            SimMode::Fish => 0.0,
            SimMode::Birds => HORIZON_BOOST,
        },
        aerial_boost: AERIAL_BOOST,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rayleigh scattering must fall off toward red, or the sky is not blue.
    ///
    /// This is the one relationship the whole model rests on, and it is also the one a careless edit
    /// to the coefficients can quietly destroy: three equal channels give a perfectly plausible grey
    /// sky.
    #[test]
    fn rayleigh_is_blue_dominant_and_mie_is_grey() {
        let p = sky_params(SimMode::Birds);
        assert!(
            p.beta_rayleigh[2] > p.beta_rayleigh[1] && p.beta_rayleigh[1] > p.beta_rayleigh[0],
            "Rayleigh scattering must grow toward blue, got {:?}",
            p.beta_rayleigh
        );
        // 440 nm scatters about 5.7 times as much as 680 nm in real air; the constant here is the
        // measured value, so anything close to that ratio is right and anything near 1 is not.
        let ratio = p.beta_rayleigh[2] / p.beta_rayleigh[0];
        assert!(
            (3.5..8.0).contains(&ratio),
            "blue/red Rayleigh ratio is {ratio}, expected the ~5.7 of real air"
        );
        assert!(p.beta_mie.iter().all(|c| *c == p.beta_mie[0]));
    }

    /// Optical depth vertically must be small, or the zenith is washed out instead of blue.
    #[test]
    fn vertical_optical_depth_is_thin() {
        let p = sky_params(SimMode::Birds);
        for (i, beta) in p.beta_rayleigh.iter().enumerate() {
            let tau = beta * p.ray_scale_height;
            assert!(
                tau < 0.5,
                "channel {i} has a vertical optical depth of {tau}: the sky above would be opaque"
            );
        }
    }

    #[test]
    fn underwater_has_no_horizon_glow() {
        assert_eq!(sky_params(SimMode::Fish).horizon_boost, 0.0);
        assert!(sky_params(SimMode::Birds).horizon_boost > 0.0);
    }
}

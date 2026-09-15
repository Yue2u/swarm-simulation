//! The underwater environment, expressed as the numbers the render passes read.
//!
//! # Why the numbers live here and not in the shaders
//!
//! The ocean's look is a handful of physical quantities - extinction per channel, caustic gain, the
//! shaft integral's weight - and exactly one of them (the reef field) is shared with the simulation.
//! Keeping them in one Rust table means "how clear is the water" is a readable decision with units,
//! that the same numbers are available to tests, and that tuning the visual does not mean editing
//! WGSL. It also means the values are *checkable*: the tests here assert the relationships that make
//! the scene read as water at all (red is absorbed faster than blue, the surface is above the floor,
//! the reef is inside the column) rather than the particular numbers.

use boids_core::config::SimConfig;
use boids_core::layout::{PostParams, WaterParams};
use boids_core::layout::SimMode;

/// The water column's height above the seafloor, as a fraction of the world's half-height.
///
/// The camera orbits at 0.8 of the world's diagonal, so it usually sits *above* the water surface
/// looking down. Putting the surface at the top of the world (`bounds_half.y`) means the camera
/// crosses it while orbiting, and the surface is the most recognisable feature of the underwater
/// world: the frame should show it, not sit safely below it.
const SURFACE_FRACTION: f32 = 1.0;

/// Underwater parameters for a world.
///
/// `config` supplies the three values that are shared with the simulation: the reef's repetition
/// period (`env_scale`), the seafloor height (`env_floor_y`) and the world's extent. Those are not
/// duplicated here because they *must* agree: the rock the fish avoid and the rock the camera sees
/// are the same field, and a second copy of the period is a second chance to disagree.
#[must_use]
pub fn water_params(config: &SimConfig) -> WaterParams {
    match config.mode {
        SimMode::Fish => WaterParams {
            surface_y: config.bounds_half.y * SURFACE_FRACTION,
            floor_y: config.env_floor_y,
            reef_period: config.env_scale,
            // Caustics are the brightest thing in the scene after the surface itself: they are the
            // reason the exposure and white point below are as large as they are.
            caustic_strength: 1.9,
            // Per-channel extinction in metres^-1, red to blue. Clear seawater, scaled so that the
            // reef is legible at the far end of this world (320 m across): red is gone by 30 m,
            // green by 150 m, blue carries. All three being equal would make distance read as
            // darkness instead of as blue.
            extinction: [0.062, 0.022, 0.0115],
            scatter: 1.0,
            godray_strength: 0.35,
            surface_glow: 0.35,
            // Cell size ~6 m: larger than the fish, smaller than a reef column, so the pattern reads
            // as light on rock rather than as noise.
            caustic_scale: 0.16,
            caustic_drift: 0.35,
        },
        SimMode::Birds => WaterParams {
            // The sky world has no water pass, but the boid shader and the shared headers still read
            // this struct. A near-zero extinction is the honest "no medium" value here: the aerial
            // haze comes from `SceneUniform::fog_density`, which is where the single-coefficient
            // model lives.
            surface_y: config.bounds_half.y,
            floor_y: config.spawn_center.y,
            reef_period: config.env_scale,
            caustic_strength: 0.0,
            extinction: [0.004, 0.0035, 0.003],
            scatter: 0.0,
            godray_strength: 0.0,
            surface_glow: 0.0,
            caustic_scale: 0.0,
            caustic_drift: 0.0,
        },
    }
}

/// Post-processing parameters for a world.
///
/// The two worlds need different *exposure* rather than different shaders: sunlight that has crossed
/// thirty metres of water arrives about two stops dimmer than sunlight in air, and the bioluminescent
/// fish are additive highlights on top of that. Everything else differs by taste, and each value is
/// documented with what it controls.
#[must_use]
pub fn post_params(mode: SimMode, time: f32) -> PostParams {
    match mode {
        SimMode::Fish => PostParams {
            // Higher than the sky's on purpose: the composite divides by `tonemap_white`, and the
            // underwater white point is more than twice the sky's to keep caustics off the clip.
            // Without the matching lift the medium's in-scatter would sit too far down the curve and
            // the reef would read as a flat haze, which is the "sunlight arrives dimmer" note below
            // made real.
            exposure: 1.5,
            // The fish and the caustics both sit above this, so both bloom; the water column's
            // in-scatter does not, which is what keeps the medium from washing the frame out.
            bloom_threshold: 0.75,
            bloom_knee: 0.6,
            bloom_strength: 0.7,
            vignette: 0.4,
            grain: 0.018,
            // Chromatic aberration is subtle by design: enough to be felt at the edges of a 1440p
            // frame, not enough to fringe the agents in the middle.
            aberration: 0.0016,
            // Large: the surface, the sun disc through it and the caustics are all well above the
            // mid-tones, and a smaller white point clips them into flat white patches.
            tonemap_white: 4.0,
            time,
            saturation: 1.06,
            contrast: 1.04,
            // Slightly negative: it crushes the deepest blue-black, which is what gives the deep
            // background its weight after the exposure lift.
            lift: -0.012,
        },
        SimMode::Birds => PostParams {
            exposure: 1.15,
            bloom_threshold: 1.1,
            bloom_knee: 0.5,
            bloom_strength: 0.5,
            vignette: 0.28,
            grain: 0.012,
            aberration: 0.0012,
            tonemap_white: 1.7,
            time,
            saturation: 1.04,
            contrast: 1.02,
            lift: 0.0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_is_above_the_seafloor_and_red_is_absorbed_first() {
        let cfg = SimConfig::for_mode(SimMode::Fish, 1000);
        let w = water_params(&cfg);
        assert!(w.surface_y > w.floor_y, "the surface must be above the seafloor");
        assert!(
            w.extinction[0] > w.extinction[1] && w.extinction[1] > w.extinction[2],
            "water must absorb red first and blue last, got {:?}",
            w.extinction
        );
        // The reef field's geometry is the simulation's, not a second copy.
        assert_eq!(w.reef_period, cfg.env_scale);
        assert_eq!(w.floor_y, cfg.env_floor_y);
    }

    /// The extinction has to leave the reef visible across the world and still tint it blue.
    #[test]
    fn extinction_leaves_the_world_legible() {
        let cfg = SimConfig::for_mode(SimMode::Fish, 1000);
        let w = water_params(&cfg);
        let span = 2.0 * cfg.bounds_half.length();
        for (i, sigma) in w.extinction.iter().enumerate() {
            let transmit = (-sigma * span).exp();
            assert!(
                transmit < 0.02,
                "channel {i} transmits {transmit} across the whole world: the far side must be fog, \
                 otherwise the world has a visible wall"
            );
        }
        // But the colour shift must be real: at a third of the world, blue has to survive far better
        // than red, or the water reads as a grey haze rather than as water.
        let d = span / 3.0;
        let red = (-w.extinction[0] * d).exp();
        let blue = (-w.extinction[2] * d).exp();
        assert!(
            blue > red * 4.0,
            "blue/red contrast at {d} m is only {blue}/{red}"
        );
    }

    #[test]
    fn fish_needs_more_light_than_the_sky() {
        let fish = post_params(SimMode::Fish, 0.0);
        let birds = post_params(SimMode::Birds, 0.0);
        assert!(
            fish.tonemap_white > birds.tonemap_white,
            "the underwater world has to keep its caustics below clipping"
        );
        assert!(fish.exposure > birds.exposure);
        // And the birds' manifold must be a no-op medium.
        let cfg = SimConfig::for_mode(SimMode::Birds, 1000);
        assert_eq!(water_params(&cfg).godray_strength, 0.0);
    }
}

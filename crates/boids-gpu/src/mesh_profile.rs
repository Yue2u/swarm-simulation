//! Per-mode mesh and scene parameters.
//!
//! The two worlds differ in a handful of numbers, not in code paths. Keeping those numbers in one
//! function means "what makes a bird look like a bird" is a single readable table, and it means the
//! render pipeline can be shared without any branching on the mode.

use boids_core::layout::{MeshParams, SceneUniform, SimMode};

/// Fraction of the perception radius that a drawn agent occupies.
///
/// This is the single number that decides whether a swarm reads as a flock or as a cloud. Too small and
/// each agent is a sub-pixel speck that aliases into noise; too large and the agents merge into one
/// solid mass with no visible individual motion. Roughly a quarter of the perception radius leaves
/// several body lengths of clear space between neighbours at typical flock density, which is what makes
/// both the individuals and the formation legible at once.
const AGENT_SIZE_FRACTION: f32 = 0.25;
#[cfg(test)]
use bytemuck::Zeroable;

/// Builds the mesh parameters for a world.
///
/// `speed_ref` is the simulation's `max_speed`, used to scale the motion stretch so that a fast
/// agent is drawn slightly elongated. Passing it in rather than hardcoding it keeps the visual
/// stretch tied to the physics that produced the motion.
///
/// Fish: laterally compressed body (tall and narrow, like a real fish cross-section), a large
/// swept tail fin, a lateral travelling wave, and bioluminescent emission.
/// Birds: a compact body, a wide swept wing, a flap animation, no emission.
#[must_use]
pub fn mesh_profile(mode: SimMode, speed_ref: f32, perception_radius: f32) -> MeshParams {
    let scale = perception_radius * AGENT_SIZE_FRACTION;
    match mode {
        SimMode::Fish => MeshParams {
            // Tall and narrow: the cross-section reads as a fish rather than as a dart.
            body_w: 0.16,
            body_h: 0.30,
            nose: 0.95,
            tail: 0.55,
            fin_size: 0.45,
            fin_z: 0.22,
            wave_amp: 0.16,
            // A fast tail beat: roughly three beats per second.
            wave_freq: 18.0,
            // Fish have no wings; the wing triangles are degenerate.
            wing_span: 0.0,
            wing_sweep: 0.0,
            // Strong enough that a fish is a light source in the HDR frame and the bloom pass picks
            // it out of the water. The underwater world is dim and hazy by construction, and a fish
            // that only reflected light with its flanks would be a teal silhouette against teal; the
            // emission is what keeps the swarm the subject of its own frame.
            emissive: 1.2,
            variant: 0.0,
            // Cyan through violet: the bioluminescent band, which is where the bloom pass will
            // later pick the strongest highlights.
            hue_base: 0.48,
            hue_range: 0.22,
            saturation: 0.85,
            value: 0.9,
            speed_ref,
            scale,
            _pad_a: 0.0,
            _pad_b: 0.0,
        },
        SimMode::Birds => MeshParams {
            // Rounder and wider: a bird body seen from above is broader than it is deep.
            body_w: 0.34,
            body_h: 0.26,
            nose: 0.85,
            tail: 0.5,
            fin_size: 0.32,
            fin_z: 0.18,
            wave_amp: 0.5,
            // A slower, larger flap: about two and a half flaps per second.
            wave_freq: 16.0,
            wing_span: 1.5,
            wing_sweep: 0.55,
            emissive: 0.0,
            variant: 1.0,
            // A wide hue range across warm colours, so a flock is visibly many species.
            hue_base: 0.02,
            hue_range: 0.45,
            saturation: 0.8,
            value: 1.0,
            speed_ref,
            scale,
            _pad_a: 0.0,
            _pad_b: 0.0,
        },
    }
}

/// Builds the scene uniform for one frame.
///
/// The medium settings are what carry most of the "which world am I in" feeling:
///
/// * fish: a strong, blue-shifted extinction with a short range. Blue light travels farthest in
///   water, which is why everything below about 20 metres reads as teal and then as black. The
///   density here stands in for the extinction coefficient of clear seawater.
/// * birds: a very weak, slightly warm haze that only shows up on distant terrain.
#[must_use]
pub fn scene_uniform(
    camera: boids_core::layout::CameraUniform,
    mode: SimMode,
    speed_ref: f32,
    perception_radius: f32,
) -> SceneUniform {
    let (light_dir, ambient, fog_color, fog_density) = match mode {
        SimMode::Fish => (
            // Sunlight from above and slightly to one side, so the reef casts readable columns of
            // light and the fish are lit from above the way they would be underwater. The ambient
            // term is higher than the sky world's because water scatters light in from every
            // direction: a fish whose flanks face away from the sun is still lit, which is not true
            // in air.
            [0.25, -1.0, 0.15],
            0.26,
            [0.015, 0.115, 0.16],
            0.022,
        ),
        SimMode::Birds => (
            [0.4, -0.85, -0.3],
            0.32,
            [0.45, 0.6, 0.78],
            0.0012,
        ),
    };
    SceneUniform {
        camera,
        mesh: mesh_profile(mode, speed_ref, perception_radius),
        light_dir,
        ambient,
        fog_color,
        fog_density,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fish_and_bird_profiles_differ_in_the_expected_ways() {
        let fish = mesh_profile(SimMode::Fish, 14.0, 3.5);
        let bird = mesh_profile(SimMode::Birds, 32.0, 14.0);

        // The fish cross-section is taller than it is wide; the bird's is the other way around.
        assert!(fish.body_h > fish.body_w, "fish should be laterally compressed");
        assert!(bird.body_w > bird.body_h, "bird should be broader than deep");

        // Only birds have wings, and only fish emit light.
        assert!(bird.wing_span > 0.0);
        assert_eq!(fish.wing_span, 0.0);
        assert!(fish.emissive > 0.0);
        assert_eq!(bird.emissive, 0.0);

        assert_eq!(fish.variant, 0.0);
        assert_eq!(bird.variant, 1.0);
    }

    #[test]
    fn fish_medium_is_much_denser_than_air() {
        let cam = boids_core::layout::CameraUniform::zeroed();
        let fish = scene_uniform(cam, SimMode::Fish, 14.0, 3.5);
        let bird = scene_uniform(cam, SimMode::Birds, 32.0, 14.0);
        assert!(
            fish.fog_density > bird.fog_density * 10.0,
            "water should extinguish far faster than air"
        );
        // Blue must survive longer than red in water, so the fog colour is blue-dominant.
        assert!(fish.fog_color[2] > fish.fog_color[0]);
    }

    #[test]
    fn profile_carries_the_speed_reference() {
        assert_eq!(mesh_profile(SimMode::Fish, 14.0, 3.5).speed_ref, 14.0);
        assert_eq!(mesh_profile(SimMode::Birds, 32.0, 14.0).speed_ref, 32.0);
    }
}

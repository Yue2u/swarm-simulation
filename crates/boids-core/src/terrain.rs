//! Procedural terrain height and biome weights: the CPU twin of `shaders/common/sdf.wgsl`.
//!
//! Used for three things:
//!
//! * the collision field for birds (`EnvironmentKind::Terrain` avoidance on the GPU),
//! * scatter placement for trees and rocks, which needs a height query per instance,
//! * the reference side of the terrain comparison test.
//!
//! ## Precision note
//!
//! Unlike the SDF primitives, terrain uses a *floating-point* hash (`fract(sin`-free but still
//! `fract` of a large product), so the CPU and the GPU are not required to agree bit for bit.
//! The comparison test for terrain therefore uses a loose tolerance and checks structure
//! (height range, biome partition of unity) rather than exact values. The strict 1e-4 comparison
//! is reserved for the analytic primitives, where exactness is a real invariant.
//!
//! Both implementations follow the same recipe: domain-warped value noise, four octaves,
//! amplitude halving and frequency slightly-more-than doubling per octave.

use glam::{Vec2, Vec3};

/// Number of octaves in the height field. Must match the loop bound in `terrain_height`.
pub const OCTAVES: u32 = 4;

/// Low-bias 32-bit integer finaliser, identical to `hash_u32` in `shaders/common/math_common.wgsl`.
///
/// Integer mixing rather than the usual `fract(sin(dot(p, k)) * n)` float hash, because that hash
/// silently degenerates: for inputs above roughly `1e4` the product's integer part exceeds the
/// 24-bit mantissa, `fract` returns exactly 0, and the terrain collapses to a flat plane. That
/// failure is silent, deterministic, and only appears once the world is large enough, which is
/// exactly the kind of bug this codebase cannot afford to debug on day four.
#[inline]
#[must_use]
pub fn hash_u32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

/// Integer-lattice coordinate hash in `[0, 1)`, matching `hash21` in `shaders/common/sdf.wgsl`.
///
/// Inputs are floored, so the value is constant across a unit cell and the result is exact on both
/// the CPU and the GPU. Callers must pass coordinates small enough to be exactly representable as
/// `i32`, which holds for any position inside the world bounds.
#[inline]
#[must_use]
pub fn hash21(p: Vec2) -> f32 {
    #[allow(clippy::cast_possible_truncation)]
    let xi = p.x.floor() as i32 as u32;
    #[allow(clippy::cast_possible_truncation)]
    let yi = p.y.floor() as i32 as u32;
    let h = hash_u32(xi.wrapping_mul(0x9e37_79b9) ^ hash_u32(yi.wrapping_mul(0x85eb_ca6b)));
    // `h >> 8` is below 2^24, so the conversion is exact.
    #[allow(clippy::cast_precision_loss)]
    let v = (h >> 8) as f32;
    v * (1.0 / 16_777_216.0)
}

/// Value noise at `p` built from `hash21` on the integer lattice, with Hermite interpolation.
#[inline]
#[must_use]
pub fn value_noise(p: Vec2) -> f32 {
    let cell = p.floor();
    let f = p - cell;
    let w = f * f * (3.0 - 2.0 * f);
    let c00 = hash21(cell);
    let c10 = hash21(cell + Vec2::new(1.0, 0.0));
    let c01 = hash21(cell + Vec2::new(0.0, 1.0));
    let c11 = hash21(cell + Vec2::new(1.0, 1.0));
    let a = c00 + (c10 - c00) * w.x;
    let b = c01 + (c11 - c01) * w.x;
    a + (b - a) * w.y
}

/// Frequency of the domain warp relative to the terrain's own, and how far it displaces the field, in
/// lattice cells of the base octave.
///
/// The warp is what turns a sum of smooth bumps into ridges and valleys, and it has to be *smooth* to
/// do that: an earlier version hashed the cell containing the sample, which made the warp jump six
/// lattice cells every 1.4 metres and turned the whole field into bounded white noise. It looked
/// plausible in a boundedness test and was unmistakable the first time the field was shaded, which is
/// what the render checks exist to catch.
const WARP_FREQUENCY: f32 = 0.5;
/// See [`WARP_FREQUENCY`].
const WARP_AMOUNT: f32 = 2.0;

/// Domain-warped fractal noise in `[0, 1]`.
#[must_use]
pub fn fbm_warped(p: Vec2, frequency: f32) -> f32 {
    // Two smooth low-frequency noises offset the sampling position. Both sides evaluate the same two
    // `value_noise` calls, so the CPU and the GPU see the same field.
    let warp_scale = frequency * WARP_FREQUENCY;
    let warp = Vec2::new(
        value_noise(p * warp_scale + Vec2::new(11.3, 4.7)),
        value_noise(p * warp_scale + Vec2::new(3.1, 19.9)),
    ) * 2.0
        - Vec2::ONE;
    let q = p * frequency + warp * WARP_AMOUNT;

    let mut h = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for _ in 0..OCTAVES {
        h += value_noise(q * freq) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.03;
    }
    h / norm.max(1e-6)
}

/// Terrain height in metres at a world `xz` position.
///
/// `amplitude` is the peak-to-trough scale and `frequency` the base frequency. Birds collide with
/// the surface through this field; the terrain mesh reads the same values from a heightmap
/// texture generated by the same expression in a compute pass.
#[inline]
#[must_use]
pub fn height_at(p: Vec2, amplitude: f32, frequency: f32) -> f32 {
    fbm_warped(p, frequency) * amplitude
}

/// Vertical gradient of the terrain height, i.e. the `xz` slope. Used for surface normals in the
/// scatter pass and by the CPU-side validation of bird avoidance.
#[must_use]
pub fn slope_at(p: Vec2, amplitude: f32, frequency: f32, eps: f32) -> Vec2 {
    let dx = height_at(p + Vec2::X * eps, amplitude, frequency)
        - height_at(p - Vec2::X * eps, amplitude, frequency);
    let dz = height_at(p + Vec2::Y * eps, amplitude, frequency)
        - height_at(p - Vec2::Y * eps, amplitude, frequency);
    Vec2::new(dx, dz) / (2.0 * eps)
}

/// Surface normal of the terrain at a world `xz` position, in world space.
#[must_use]
pub fn normal_at(p: Vec2, amplitude: f32, frequency: f32, eps: f32) -> Vec3 {
    let s = slope_at(p, amplitude, frequency, eps);
    Vec3::new(-s.x, 1.0, -s.y).normalize()
}

/// Biome weights as `(forest, dunes, canyon)`, summing to 1.
///
/// A partition of unity rather than a hard switch: a hard biome boundary shows up as a straight
/// seam across the terrain and as a palette discontinuity on the birds.
#[inline]
#[must_use]
pub fn biome_weights(p: Vec2, frequency: f32) -> Vec3 {
    let cell = (p * frequency).floor();
    let a = hash21(cell);
    let b = hash21(cell + Vec2::new(41.0, 7.0));
    let c = hash21(cell + Vec2::new(13.0, 53.0));
    let total = (a + b + c).max(1e-5);
    Vec3::new(a, b, c) / total
}

/// Index of the dominant biome at `p`: 0 = forest, 1 = dunes, 2 = canyon.
#[must_use]
pub fn dominant_biome(p: Vec2, frequency: f32) -> usize {
    let w = biome_weights(p, frequency);
    if w.x >= w.y && w.x >= w.z {
        0
    } else if w.y >= w.z {
        1
    } else {
        2
    }
}

/// Biome coordinate in `[0, 2)`: 0 = forest, 1 = dunes, 2 = canyon, continuous across a transition.
///
/// Derived from [`biome_weights`] rather than hashed on its own so that the heightfield's mask
/// channel and the weights cannot disagree. Because the three weights sum to 1, the map is a sweep:
/// at a forest/dunes boundary `w.z` is 0 and the mask runs 0 -> 1, and at a dunes/canyon boundary
/// `w.y` is 0 and it runs 1 -> 2.
///
/// One scalar is all the heightfield's second channel has room for, and one scalar is all a
/// three-stop palette ramp needs. Mirrors `biome_mask` in `shaders/common/sdf.wgsl`.
#[must_use]
pub fn biome_mask(p: Vec2, frequency: f32) -> f32 {
    let w = biome_weights(p, frequency);
    w.y + 2.0 * w.z
}

/// Base noise frequency, m^-1, for a world of half-extent `half_extent`.
///
/// The terrain is *world-relative* rather than absolute: the same expression has to produce
/// mountains across a 1.2 km world and hills across a 250 m test world, and a fixed frequency would
/// give one of them a single featureless slope. `TERRAIN_FEATURES` is how many base-wavelength
/// features span the world, which is a property of the look rather than of the world's size.
///
/// Mirrors `boids_core`'s use on the host: the value ends up in `SimParams::env_freq` and
/// `TerrainParams::frequency`, and both sides call [`height_at`] with it.
#[must_use]
pub fn frequency_for_extent(half_extent: f32) -> f32 {
    const TERRAIN_FEATURES: f32 = 3.0;
    TERRAIN_FEATURES / (2.0 * half_extent).max(1.0)
}

/// Height amplitude, metres, for a world of half-height `half_height`.
///
/// A third of the world's half-height. That leaves the top two thirds of the box as open air, which
/// is what the spawn cluster needs: the swarm is placed above the *maximum* terrain height with a
/// clearance, and a taller ridge would leave too little air for a cluster at the reference density
/// and force it to compress (see `spawn::swarm_extent`). A ridge that reaches half the world, which
/// this used to be, left the spawn cluster with a quarter of the box and squashed it into a dense
/// knot. Mirrors what `SimConfig::for_mode` puts in `env_scale` for the terrain field.
#[must_use]
pub fn amplitude_for_extent(half_height: f32) -> f32 {
    half_height * 0.3
}

#[cfg(test)]
mod tests {
    use super::*;
/// The field has to be *smooth*, not merely bounded.
///
/// This catches the failure mode a boundedness test cannot: a domain warp that is not smooth. Hashing
/// the sample's own cell makes the warp jump by whole lattice cells from one metre to the next, so the
/// field becomes bounded white noise with the terrain's full amplitude. Every boundedness assertion
/// passes, and the moment it is shaded it is a field of needles.
#[test]
fn height_field_is_lipschitz() {
    for (half, label) in [(600.0f32, "app"), (126.0, "test")] {
        let amplitude = amplitude_for_extent(half * 1.83);
        let frequency = frequency_for_extent(half);
        let mut worst = 0.0f32;
        let mut worst_at = Vec2::ZERO;
        for i in 0..4000 {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 * 0.37;
            let p = Vec2::new((t * 1.7).sin() * half, (t * 0.9).cos() * half);
            let e = Vec2::new(0.5, -0.5);
            let slope = (height_at(p + e, amplitude, frequency) - height_at(p, amplitude, frequency))
                .abs()
                / e.length();
            if slope > worst {
                worst = slope;
                worst_at = p;
            }
        }
        eprintln!("{label} world: steepest half-metre slope {worst:.2} at {worst_at:?}");
        assert!(
            worst < 4.0,
            "{label} world: a half-metre step changes the height by {worst} metres, so the field is \
             not smooth. A domain warp that is not smooth is the usual cause."
        );
    }
}


    #[test]
    fn height_is_bounded_and_varies_across_the_world() {
        let (half, amplitude) = (600.0f32, 110.0f32);
        let frequency = frequency_for_extent(half);
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        // A grid rather than a line: a single line can miss every feature of a field whose features
        // are hundreds of metres across, and then the check passes while measuring nothing.
        for i in 0..40 {
            for j in 0..40 {
                #[allow(clippy::cast_precision_loss)]
                let p = Vec2::new(
                    -half + 2.0 * half * (i as f32) / 39.0,
                    -half + 2.0 * half * (j as f32) / 39.0,
                );
                let h = height_at(p, amplitude, frequency);
                assert!(h.is_finite(), "height not finite at {p:?}");
                min = min.min(h);
                max = max.max(h);
            }
        }
        assert!(
            min >= 0.0 && max <= amplitude,
            "height range [{min}, {max}] escaped [0, {amplitude}]"
        );
        // It must actually vary, otherwise the terrain is a plane and every scatter test passes for
        // the wrong reason.
        assert!(
            max - min > amplitude * 0.4,
            "height field spans only {min}..{max} across the world"
        );
    }

    #[test]
    fn biome_weights_form_a_partition_of_unity() {
        for i in 0..500 {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 * 1.9;
            let w = biome_weights(Vec2::new(t, -t * 1.3), 0.004);
            assert!((w.x + w.y + w.z - 1.0).abs() < 1e-4, "weights {w:?} do not sum to 1");
            assert!(w.min_element() >= 0.0);
        }
    }

    #[test]
    fn normal_points_up_and_matches_slope() {
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 * 3.1;
            let p = Vec2::new(t, t * 0.4);
            let n = normal_at(p, 90.0, 0.0025, 1.0);
            assert!((n.length() - 1.0).abs() < 1e-4);
            assert!(n.y > 0.0, "terrain normal should point up, got {n:?}");
        }
    }
}

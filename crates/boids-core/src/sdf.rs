//! Signed distance fields used for environment collision avoidance and ocean raymarching.
//!
//! Every function here is mirrored in `shaders/common/sdf.wgsl`. The mirrored design is
//! deliberate: the CPU version exists so that (`sdf_wgsl_matches_rust` test) a GPU run of the same
//! field can be compared against it. A sign convention flip in the shader is otherwise very hard
//! to notice visually, because both signs "look like avoidance" until agents start flying inside
//! rocks.
//!
//! Sign convention: **negative inside** the solid, positive outside, magnitude approximately the
//! distance to the surface.

use glam::{Vec2, Vec3, Vec3Swizzles};

use crate::math::smoothstep;

/// SDF of a sphere centred at `center`.
#[inline]
#[must_use]
pub fn sphere(p: Vec3, center: Vec3, radius: f32) -> f32 {
    (p - center).length() - radius
}

/// SDF of an axis-aligned box.
#[inline]
#[must_use]
pub fn box_sdf(p: Vec3, center: Vec3, half: Vec3) -> f32 {
    let q = (p - center).abs() - half;
    q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
}

/// SDF of an infinite ground plane at height `y`, solid below. Positive above the plane.
#[inline]
#[must_use]
pub fn plane_y(p: Vec3, y: f32) -> f32 {
    p.y - y
}

/// Polynomial smooth minimum, matching the shader implementation.
///
/// `k` controls the blend radius. Used to fuse the columns and arches of the reef so that there
/// are no hard creases for agents to snag on.
#[inline]
#[must_use]
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b * (1.0 - h) + a * h - k * h * (1.0 - h)
}

/// SDF of a vertical capped column of radius `r` centred on the `y` axis at `p.xz`, from
/// `base_y` to `top_y`.
///
/// This is the primitive the reef is built from. Written as a 2D radial distance in `xz` fused
/// with a vertical slab so that the result is a genuine distance field (within the usual
/// approximation of a capped cylinder), which sphere tracing needs.
#[inline]
#[must_use]
pub fn column(p: Vec3, center_xz: glam::Vec2, radius: f32, base_y: f32, top_y: f32) -> f32 {
    let d_radial = (p.xz() - center_xz).length() - radius;
    let d_vertical = (p.y - 0.5 * (base_y + top_y)).abs() - 0.5 * (top_y - base_y);
    let outside = d_radial.max(0.0).hypot(d_vertical.max(0.0));
    outside + d_radial.max(d_vertical).min(0.0)
}

/// Radius profile of a reef column as a function of normalized height `t` in `[0, 1]`.
///
/// Columns thin toward the top with a slightly irregular flare, which reads as coral rather than
/// as extruded cylinders.
#[inline]
#[must_use]
pub fn column_radius_profile(t: f32, base_radius: f32, taper: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    base_radius * (1.0 - taper * t * t) * (1.0 + 0.12 * (t * 9.0).sin())
}

/// The reef field: two domain-repeated families of tapered columns plus a seafloor.
///
/// `period` is the repetition spacing in `xz`. Domain repetition is what keeps the field cheap:
/// the same handful of arithmetic operations describe an unbounded reef.
///
/// The second family, offset by half a period, is what fills the gaps of the first and gives the
/// reef its arches without a separate primitive. Both blend radii and the family parameters must
/// match `reef_field` in `shaders/common/sdf.wgsl` exactly; `sdf_wgsl_matches_rust` compares the two
/// on a grid.
#[inline]
#[must_use]
pub fn reef_field(p: Vec3, period: f32, seafloor_y: f32) -> f32 {
    let cell = (p.xz() / period).floor();
    // Hash the cell to vary radius and height per column without a texture lookup.
    let h = hash2(cell);
    let center = (cell + Vec2::splat(0.5)) * period;
    let height = 20.0 + 26.0 * h;
    let radius = 3.0 + 3.5 * hash2(cell + Vec2::splat(17.0));
    let base = seafloor_y;
    let top = seafloor_y + height;
    let t = ((p.y - base) / (top - base)).clamp(0.0, 1.0);
    let r = column_radius_profile(t, radius, 0.45);
    let a = column(p, center, r, base, top);

    let center2 = center + Vec2::new(0.55 * period, 0.1 * period);
    let height2 = 12.0 + 30.0 * hash2(cell + Vec2::new(5.0, 9.0));
    let r2 = 2.0 + 3.0 * hash2(cell + Vec2::new(31.0, 3.0));
    let t2 = ((p.y - base) / height2).clamp(0.0, 1.0);
    let b = column(
        p,
        center2,
        column_radius_profile(t2, r2, 0.5),
        base,
        base + height2,
    );

    let columns = smin(a, b, 3.0);
    // The floor uses a larger blend radius so columns visibly grow out of it.
    smin(columns, plane_y(p, seafloor_y), 6.0)
}

/// Cell-coordinate hash in `[0, 1)`, matching `hash21` in `shaders/common/sdf.wgsl`.
///
/// Delegates to [`crate::terrain::hash21`] so there is exactly one hash implementation in the
/// CPU code and one in the shaders. A second, subtly different hash would make the reef field on
/// the CPU disagree with the reef field in the shader, and the disagreement would look like a
/// physics bug rather than a hashing bug.
#[inline]
#[must_use]
pub fn hash2(p: glam::Vec2) -> f32 {
    crate::terrain::hash21(p)
}

/// Central-difference gradient of a scalar field, the basis of SDF avoidance.
///
/// `eps` should be about half a grid cell: much smaller and the field's own noise dominates, much
/// larger and thin obstacles get smoothed away.
pub fn gradient<F: Fn(Vec3) -> f32>(field: F, p: Vec3, eps: f32) -> Vec3 {
    let dx = field(p + Vec3::X * eps) - field(p - Vec3::X * eps);
    let dy = field(p + Vec3::Y * eps) - field(p - Vec3::Y * eps);
    let dz = field(p + Vec3::Z * eps) - field(p - Vec3::Z * eps);
    Vec3::new(dx, dy, dz) / (2.0 * eps)
}

/// Avoidance force from an SDF: zero far away, growing to `strength` as the surface is approached.
///
/// `smoothstep(r_safe, 0, d)` is used rather than `d.recip()` because the reciprocal blows up
/// inside the solid, which launches agents across the map when they clip a corner during a
/// frame hitch. The smoothstep form saturates instead.
#[inline]
#[must_use]
pub fn avoid_force(sdf: f32, normal: Vec3, r_safe: f32, strength: f32) -> Vec3 {
    let w = smoothstep(r_safe, 0.0, sdf);
    normal * (w * strength)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_signs_and_magnitude() {
        assert!((sphere(Vec3::ZERO, Vec3::ZERO, 2.0) + 2.0).abs() < 1e-6);
        assert!((sphere(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO, 2.0) - 1.0).abs() < 1e-6);
        assert!((sphere(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO, 2.0) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn box_sdf_outside_is_euclidean_distance_to_corner() {
        let d = box_sdf(Vec3::new(3.0, 3.0, 0.0), Vec3::ZERO, Vec3::splat(1.0));
        assert!((d - 2f32.hypot(2.0)).abs() < 1e-5, "got {d}");
        assert!(box_sdf(Vec3::ZERO, Vec3::ZERO, Vec3::splat(1.0)) < 0.0);
    }

    #[test]
    fn column_field_is_zero_on_the_surface() {
        let radius = 4.0;
        let p = Vec3::new(radius, 10.0, 0.0);
        let d = column(p, glam::Vec2::ZERO, radius, 0.0, 20.0);
        assert!(d.abs() < 1e-4, "surface point gave {d}");
        assert!(column(Vec3::new(1.0, 10.0, 0.0), glam::Vec2::ZERO, radius, 0.0, 20.0) < 0.0);
        assert!(column(Vec3::new(0.0, 30.0, 0.0), glam::Vec2::ZERO, radius, 0.0, 20.0) > 0.0);
    }

    #[test]
    fn gradient_points_away_from_the_solid() {
        let field = |p: Vec3| sphere(p, Vec3::ZERO, 5.0);
        let g = gradient(field, Vec3::new(10.0, 0.0, 0.0), 0.1);
        assert!(g.normalize().dot(Vec3::X) > 0.99, "gradient {g:?}");
    }

    #[test]
    fn gradient_of_reef_is_finite_everywhere() {
        let field = |p: Vec3| reef_field(p, 40.0, -30.0);
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 * 1.7;
            let p = Vec3::new(t.sin() * 60.0, t.cos() * 40.0, t * 0.5 - 40.0);
            assert!(field(p).is_finite(), "reef field not finite at {p:?}");
            assert!(gradient(field, p, 1.0).is_finite());
        }
    }

    #[test]
    fn avoid_force_saturates_and_vanishes() {
        let far = avoid_force(100.0, Vec3::X, 3.0, 50.0);
        assert!(far.length() < 1e-6, "force should vanish far away");
        let deep = avoid_force(-5.0, Vec3::X, 3.0, 50.0);
        assert!((deep.length() - 50.0).abs() < 1e-4, "force should saturate inside");
        let mid = avoid_force(1.5, Vec3::X, 3.0, 50.0);
        assert!(mid.x > 0.0 && mid.x < 50.0);
    }

    #[test]
    fn smin_never_exceeds_min_and_is_smooth() {
        assert!(smin(1.0, 2.0, 0.5) <= 1.0 + 1e-6);
        assert!(smin(-1.0, -2.0, 0.5) <= -1.0 + 1e-6);
        // Continuity: a small step in the input must not produce a jump.
        let a = smin(1.0, 1.0001, 0.5);
        let b = smin(1.0, 1.0002, 0.5);
        assert!((a - b).abs() < 1e-3);
    }
}

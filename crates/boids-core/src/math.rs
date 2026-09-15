//! Small numeric helpers shared by the CPU reference and the WGSL shaders.
//!
//! Each function here has an exact counterpart in `shaders/common/math_common.wgsl`; the
//! `sdf` and `math` GPU-vs-CPU comparison tests keep the two implementations honest.

use glam::Vec3;

/// Hermite smoothstep, matching WGSL's `smoothstep(edge0, edge1, x)` for `edge0 < edge1`.
#[inline]
#[must_use]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < f32::EPSILON {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Clamps a vector to at most `max_len`, preserving direction. Zero-length input stays zero.
#[inline]
#[must_use]
pub fn clamp_len(v: Vec3, max_len: f32) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > max_len * max_len && len_sq > 0.0 {
        v * (max_len / len_sq.sqrt())
    } else {
        v
    }
}

/// Clamps a vector's length into `[min_len, max_len]`, preserving direction.
///
/// A zero vector is *not* given a direction here: the caller decides which way a stalled agent
/// should be nudged, because that choice is mode-specific.
#[inline]
#[must_use]
pub fn clamp_len_range(v: Vec3, min_len: f32, max_len: f32) -> Vec3 {
    let len = v.length();
    if len <= f32::EPSILON {
        return v;
    }
    v * (len.clamp(min_len, max_len) / len)
}

/// Polarisation order parameter: `|mean(v_hat)|`, in `[0, 1]`.
///
/// `0` means fully disordered, `1` means every agent flies the same way. This is the primary
/// scalar used to compare the GPU simulation against the CPU reference, because it is invariant
/// under translation and rotation and therefore robust to floating-point drift.
#[must_use]
pub fn order_parameter(velocities: &[Vec3]) -> f32 {
    if velocities.is_empty() {
        return 0.0;
    }
    let mut sum = Vec3::ZERO;
    for v in velocities {
        let len = v.length();
        if len > f32::EPSILON {
            sum += *v / len;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let n = velocities.len() as f32;
    (sum / n).length()
}

/// Centroid of a point set. Returns zero for an empty slice.
#[must_use]
pub fn centroid(points: &[Vec3]) -> Vec3 {
    if points.is_empty() {
        return Vec3::ZERO;
    }
    let sum: Vec3 = points.iter().copied().sum();
    #[allow(clippy::cast_precision_loss)]
    let n = points.len() as f32;
    sum / n
}

/// Mean speed of a velocity set. Returns zero for an empty slice.
#[must_use]
pub fn mean_speed(velocities: &[Vec3]) -> f32 {
    if velocities.is_empty() {
        return 0.0;
    }
    let sum: f32 = velocities.iter().map(|v| v.length()).sum();
    #[allow(clippy::cast_precision_loss)]
    let n = velocities.len() as f32;
    sum / n
}

/// Mean distance from a point set to a reference point. Returns zero for an empty slice.
///
/// Used by the interaction tests to assert that the attractor actually pulls the swarm in and the
/// repeller actually pushes it away, rather than just moving it around.
#[must_use]
pub fn mean_distance_to(boids: &[crate::layout::Boid], point: Vec3) -> f32 {
    if boids.is_empty() {
        return 0.0;
    }
    let sum: f32 = boids
        .iter()
        .map(|b| (Vec3::from(b.pos) - point).length())
        .sum();
    #[allow(clippy::cast_precision_loss)]
    let n = boids.len() as f32;
    sum / n
}

/// Builds an orthonormal basis whose third axis is `fwd`.
///
/// This mirrors the vertex shader's TBN construction exactly, including the pivot choice: a
/// reference axis nearly parallel to `fwd` is swapped out to avoid a degenerate cross product.
#[inline]
#[must_use]
pub fn basis_from_forward(fwd: Vec3) -> (Vec3, Vec3, Vec3) {
    let fwd = fwd.normalize_or_zero();
    let reference = if fwd.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let right = reference.cross(fwd).normalize_or_zero();
    let up = fwd.cross(right);
    (right, up, fwd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_len_preserves_direction() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let c = clamp_len(v, 2.0);
        assert!((c.length() - 2.0).abs() < 1e-5);
        assert!((c.normalize() - v.normalize()).length() < 1e-5);
        assert_eq!(clamp_len(Vec3::ZERO, 2.0), Vec3::ZERO);
        assert_eq!(clamp_len(Vec3::new(0.5, 0.0, 0.0), 2.0), Vec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn order_parameter_extremes() {
        assert!(order_parameter(&[]).abs() < 1e-6);
        let aligned = vec![Vec3::X; 32];
        assert!((order_parameter(&aligned) - 1.0).abs() < 1e-5);
        let opposed = vec![Vec3::X, -Vec3::X];
        assert!(order_parameter(&opposed).abs() < 1e-6);
    }

    #[test]
    fn basis_is_orthonormal_even_when_forward_is_vertical() {
        for fwd in [
            Vec3::Y,
            -Vec3::Y,
            Vec3::X,
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 0.999_999, 0.001),
        ] {
            let (r, u, f) = basis_from_forward(fwd);
            assert!((r.length() - 1.0).abs() < 1e-4, "right not unit for {fwd:?}");
            assert!((u.length() - 1.0).abs() < 1e-4, "up not unit for {fwd:?}");
            assert!((f.length() - 1.0).abs() < 1e-4, "fwd not unit for {fwd:?}");
            assert!(r.dot(u).abs() < 1e-4, "right/up not orthogonal for {fwd:?}");
            assert!(r.dot(f).abs() < 1e-4, "right/fwd not orthogonal for {fwd:?}");
        }
    }

    #[test]
    fn smoothstep_matches_reference() {
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        assert!(smoothstep(0.0, 1.0, -1.0).abs() < 1e-6);
        assert!((smoothstep(0.0, 1.0, 2.0) - 1.0).abs() < 1e-6);
    }
}

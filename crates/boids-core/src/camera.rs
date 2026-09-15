//! Camera and cursor-ray math.
//!
//! Everything here runs **once per frame on the CPU**, never per agent, so it is written for
//! clarity rather than throughput. Each function has a unit test pinning the convention, because a
//! silently flipped Y axis in NDC handling is the classic way to end up with a cursor that
//! attracts agents on the opposite side of the screen.

use glam::{Mat4, Vec2, Vec3};

use crate::layout::CameraUniform;

/// A world-space ray with a normalized direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// Start of the ray.
    pub origin: Vec3,
    /// Unit direction.
    pub dir: Vec3,
}

impl Ray {
    /// Point at parameter `t`.
    #[inline]
    #[must_use]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// Turntable camera controlled by the mouse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    /// Point the camera orbits and looks at.
    pub target: Vec3,
    /// Distance from target.
    pub distance: f32,
    /// Azimuth in radians.
    pub yaw: f32,
    /// Elevation in radians, clamped to just short of the poles.
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near plane.
    pub near: f32,
    /// Far plane.
    pub far: f32,
    /// Viewport aspect ratio.
    pub aspect: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 120.0,
            yaw: 0.6,
            pitch: 0.35,
            fov_y: 55f32.to_radians(),
            near: 0.5,
            far: 6000.0,
            aspect: 16.0 / 9.0,
        }
    }
}

/// Pitch clamp in radians, keeping the up vector well conditioned.
const PITCH_LIMIT: f32 = 1.553_343; // 89 degrees

impl OrbitCamera {
    /// Camera position derived from the orbit parameters.
    #[must_use]
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(cp * sy, sp, cp * cy) * self.distance
    }

    /// Unit vector from the camera toward the target.
    #[must_use]
    pub fn forward(&self) -> Vec3 {
        (self.target - self.eye()).normalize_or_zero()
    }

    /// View matrix.
    #[must_use]
    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    /// Projection matrix.
    ///
    /// Uses the `directx` convention (`rh::proj::directx::perspective`): right-handed Y-up view
    /// space with NDC depth in `[0, 1]`, which is what `wgpu` expects. Do not switch this to the
    /// `opengl` variant, whose `[-1, 1]` depth range would silently break depth testing against
    /// the ocean raymarch pass, nor to `vulkan`, which also flips Y and would mirror the scene.
    #[must_use]
    pub fn projection(&self) -> Mat4 {
        glam::camera::rh::proj::directx::perspective(self.fov_y, self.aspect, self.near, self.far)
    }

    /// Combined view-projection matrix.
    #[must_use]
    pub fn view_proj(&self) -> Mat4 {
        self.projection() * self.view()
    }

    /// Writes the uniform uploaded once per frame.
    #[must_use]
    pub fn to_uniform(&self, time: f32, viewport: Vec2) -> CameraUniform {
        let vp = self.view_proj();
        CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            eye: self.eye().to_array(),
            time,
            forward: self.forward().to_array(),
            near: self.near,
            up: Vec3::Y.to_array(),
            far: self.far,
            viewport: viewport.to_array(),
            tan_half_fovy: (self.fov_y * 0.5).tan(),
            aspect: self.aspect,
        }
    }

    /// Applies a mouse drag in pixels: horizontal drag orbits, vertical drag pitches.
    pub fn orbit(&mut self, delta: Vec2) {
        const SENSITIVITY: f32 = 0.006;
        self.yaw -= delta.x * SENSITIVITY;
        self.pitch = (self.pitch + delta.y * SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Applies a scroll-wheel zoom, exponential so it feels uniform at any distance.
    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (1.0 - scroll * 0.1).max(0.05)).clamp(1.0, self.far * 0.9);
    }

    /// Applies a pan in camera space, for middle-drag.
    pub fn pan(&mut self, delta: Vec2) {
        let fwd = self.forward();
        let right = fwd.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(fwd);
        let scale = self.distance * 0.0015;
        self.target += (right * -delta.x + up * delta.y) * scale;
    }

    /// Updates the aspect ratio after a resize.
    pub fn set_viewport(&mut self, width: u32, height: u32) {
        #[allow(clippy::cast_precision_loss)]
        let aspect = if height == 0 {
            1.0
        } else {
            width as f32 / height as f32
        };
        self.aspect = aspect;
    }
}

/// Builds a world-space ray through a pixel.
///
/// `cursor_px` is the pointer position in physical pixels with the origin at the window's
/// top-left, which is what `winit` reports. NDC is derived as
/// `(2*mx/W - 1, 1 - 2*my/H)`, so the Y flip happens exactly here.
///
/// The ray is reconstructed from the inverse view-projection rather than from basis vectors, so
/// it stays correct if a non-symmetric projection is ever introduced.
#[must_use]
pub fn ray_from_ndc(inv_view_proj: Mat4, cursor_px: Vec2, viewport: Vec2) -> Ray {
    let ndc = Vec2::new(
        2.0 * cursor_px.x / viewport.x - 1.0,
        1.0 - 2.0 * cursor_px.y / viewport.y,
    );
    let near = inv_view_proj * glam::Vec4::new(ndc.x, ndc.y, 0.0, 1.0);
    let far = inv_view_proj * glam::Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
    // Perspective divide: w is not 1 after an inverse perspective transform.
    let near = near.truncate() / near.w;
    let far = far.truncate() / far.w;
    Ray {
        origin: near,
        dir: (far - near).normalize_or_zero(),
    }
}

/// Intersects a ray with the infinite plane through `point` with normal `normal`.
///
/// Returns `None` when the ray is parallel to the plane or when the hit is behind the origin,
/// which is the case that must be handled gracefully: a cursor pointing at the horizon should
/// simply stop influencing the swarm rather than teleport the focus point.
#[must_use]
pub fn ray_plane_intersect(ray: Ray, point: Vec3, normal: Vec3) -> Option<Vec3> {
    let denom = ray.dir.dot(normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (point - ray.origin).dot(normal) / denom;
    if t <= 0.0 {
        return None;
    }
    Some(ray.at(t))
}

/// Intersects a ray with an axis-aligned box, returning entry distance when it hits.
///
/// Used to keep the interaction focus point inside the simulation domain, so that a cursor aimed
/// at the sky does not put the attractor kilometers away where it does nothing useful.
#[must_use]
pub fn ray_aabb_intersect(ray: Ray, min: Vec3, max: Vec3) -> Option<f32> {
    let mut t_near = f32::NEG_INFINITY;
    let mut t_far = f32::INFINITY;
    for axis in 0..3 {
        let o = ray.origin[axis];
        let d = ray.dir[axis];
        let (lo, hi) = (min[axis], max[axis]);
        if d.abs() < 1e-8 {
            if o < lo || o > hi {
                return None;
            }
        } else {
            let inv = 1.0 / d;
            let mut t0 = (lo - o) * inv;
            let mut t1 = (hi - o) * inv;
            if t0 > t1 {
                core::mem::swap(&mut t0, &mut t1);
            }
            t_near = t_near.max(t0);
            t_far = t_far.min(t1);
            if t_near > t_far {
                return None;
            }
        }
    }
    if t_far < 0.0 {
        None
    } else {
        Some(t_near.max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera() -> OrbitCamera {
        OrbitCamera {
            target: Vec3::ZERO,
            distance: 50.0,
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 90f32.to_radians(),
            near: 0.1,
            far: 1000.0,
            aspect: 1.0,
        }
    }

    #[test]
    fn centre_pixel_ray_points_at_target() {
        let cam = test_camera();
        let inv = cam.view_proj().inverse();
        let viewport = Vec2::new(800.0, 600.0);
        let ray = ray_from_ndc(inv, viewport * 0.5, viewport);
        let to_target = (cam.target - ray.origin).normalize();
        assert!(
            ray.dir.dot(to_target) > 0.9999,
            "centre ray {:?} should look at target",
            ray.dir
        );
    }

    #[test]
    fn y_axis_is_flipped_once() {
        let cam = test_camera();
        let inv = cam.view_proj().inverse();
        let viewport = Vec2::new(800.0, 600.0);
        // Top-left pixel must aim above the centre ray, bottom-left below it.
        let top = ray_from_ndc(inv, Vec2::new(1.0, 1.0), viewport);
        let bottom = ray_from_ndc(inv, Vec2::new(1.0, 599.0), viewport);
        assert!(
            top.dir.y > 0.0,
            "top-left ray should point upward, got {:?}",
            top.dir
        );
        assert!(
            bottom.dir.y < 0.0,
            "bottom-left ray should point downward, got {:?}",
            bottom.dir
        );
        // And the x axis must not be flipped: left pixel aims to the left.
        let left = ray_from_ndc(inv, Vec2::new(1.0, 300.0), viewport);
        assert!(left.dir.x < 0.0, "left ray should point -x, got {:?}", left.dir);
    }

    #[test]
    fn rays_are_normalized() {
        let cam = test_camera();
        let inv = cam.view_proj().inverse();
        let viewport = Vec2::new(1920.0, 1080.0);
        for px in [
            Vec2::ZERO,
            Vec2::new(1919.0, 0.0),
            Vec2::new(0.0, 1079.0),
            viewport - Vec2::ONE,
        ] {
            let ray = ray_from_ndc(inv, px, viewport);
            assert!((ray.dir.length() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn plane_intersection_hits_expected_point() {
        let ray = Ray {
            origin: Vec3::new(0.0, 10.0, 0.0),
            dir: Vec3::NEG_Y,
        };
        let hit = ray_plane_intersect(ray, Vec3::ZERO, Vec3::Y).expect("should hit");
        assert!(hit.abs_diff_eq(Vec3::ZERO, 1e-5));
        // Parallel: no intersection.
        let parallel = Ray {
            origin: Vec3::new(1.0, 10.0, 0.0),
            dir: Vec3::X,
        };
        assert!(ray_plane_intersect(parallel, Vec3::ZERO, Vec3::Y).is_none());
        // Behind the origin: rejected.
        let away = Ray {
            origin: Vec3::new(0.0, 10.0, 0.0),
            dir: Vec3::Y,
        };
        assert!(ray_plane_intersect(away, Vec3::ZERO, Vec3::Y).is_none());
    }

    #[test]
    fn aabb_intersection_handles_inside_and_miss() {
        let inside = Ray {
            origin: Vec3::ZERO,
            dir: Vec3::X,
        };
        assert_eq!(ray_aabb_intersect(inside, Vec3::splat(-1.0), Vec3::splat(1.0)), Some(0.0));
        let outside = Ray {
            origin: Vec3::new(-5.0, 0.0, 0.0),
            dir: Vec3::X,
        };
        let t = ray_aabb_intersect(outside, Vec3::splat(-1.0), Vec3::splat(1.0)).unwrap();
        assert!((t - 4.0).abs() < 1e-5);
        let miss = Ray {
            origin: Vec3::new(-5.0, 10.0, 0.0),
            dir: Vec3::X,
        };
        assert!(ray_aabb_intersect(miss, Vec3::splat(-1.0), Vec3::splat(1.0)).is_none());
    }

    #[test]
    fn orbit_pitch_is_clamped() {
        let mut cam = test_camera();
        for _ in 0..1000 {
            cam.orbit(Vec2::new(0.0, 1000.0));
        }
        assert!(cam.pitch <= PITCH_LIMIT + 1e-6);
        assert!(cam.eye().is_finite());
    }
}

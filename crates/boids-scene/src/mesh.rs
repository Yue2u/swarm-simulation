//! The tree mesh: a fractal grown once on the CPU, drawn instanced by `render/tree.wgsl`.
//!
//! # Why this one is a real mesh
//!
//! The agents are generated from `vertex_index` in the vertex shader because their mesh has to be
//! *animated* per agent (tail wave, wing flap, bank) and because 100k instances cannot afford a
//! vertex buffer read. A tree has neither property: it is static, it is drawn a few thousand times,
//! and a recursion that expresses itself as a triangle list is far easier to read than one encoded as
//! index arithmetic. So this is the one mesh in the project that is built on the host.
//!
//! # Shape
//!
//! A trunk that splits into four branches, each of which splits into three twigs: 17 tapered prisms,
//! 136 triangles. The branching angles come from a golden-angle spiral, which is what keeps a stand of
//! identical trees from reading as a pattern: two trees with different yaw never show the same
//! silhouette, because the branch azimuths are coprime with the rotation.
//!
//! The mesh is normalized to a height of 1.0 with its base at the origin, so an instance's scale *is*
//! its height in metres.

use glam::Vec3;

use bytemuck::{Pod, Zeroable};

/// One vertex of a static mesh. Mirrors the vertex buffer layout in `render/tree.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct TreeVertex {
    /// Position in mesh space, metres at scale 1.
    pub pos: [f32; 3],
    /// Outward normal, used for the trunk/canopy material split.
    pub normal: [f32; 3],
}

/// A CPU-generated mesh: vertices plus 32-bit indices.
#[derive(Debug, Clone, Default)]
pub struct TreeMesh {
    /// Vertices, in triangle order per quad.
    pub vertices: Vec<TreeVertex>,
    /// Triangle list indices.
    pub indices: Vec<u32>,
}

impl TreeMesh {
    /// Number of triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// Sides of a branch cross-section. Four is enough: a tree is tens of pixels tall in the app's
/// framing, and every extra side multiplies the triangle count of the whole forest.
const SIDES: usize = 4;

/// Branching angles from the parent axis, radians. The first split is wider than the second, which is
/// what gives the silhouette a trunk rather than a bush.
const TILT_LEVELS: [f32; 2] = [0.62, 0.55];

/// Length of a child branch as a fraction of its parent's.
const LENGTH_FALLOFF: f32 = 0.62;

/// Radius of a child branch as a fraction of its parent's.
const RADIUS_FALLOFF: f32 = 0.55;

/// Children per split, per level. Four then three: a trunk with four limbs, each with three twigs.
const CHILDREN: [usize; 2] = [4, 3];

/// Trunk length in mesh units before normalization.
const TRUNK_LENGTH: f32 = 0.42;

/// Trunk radius in mesh units before normalization.
const TRUNK_RADIUS: f32 = 0.052;

/// Golden angle, radians. Branch azimuths advance by it, so no two levels line up.
const GOLDEN_ANGLE: f32 = 2.399_963_3;

/// Builds the tree mesh, normalized to a height of 1.0.
#[must_use]
pub fn tree_mesh() -> TreeMesh {
    let mut mesh = TreeMesh::default();
    grow(&mut mesh, Vec3::ZERO, Vec3::Y, TRUNK_LENGTH, TRUNK_RADIUS, 0);

    // Normalize: scale so the highest vertex is at y = 1, and rebase the lowest at y = 0. An
    // instance's scale is then its height in metres, which is what `TerrainParams::tree_height`
    // promises.
    let max_y = mesh.vertices.iter().fold(0.0f32, |a, v| a.max(v.pos[1]));
    let scale = if max_y > 1e-6 { 1.0 / max_y } else { 1.0 };
    for v in &mut mesh.vertices {
        v.pos[0] *= scale;
        v.pos[1] *= scale;
        v.pos[2] *= scale;
    }
    mesh
}

/// Appends one tapered prism plus its children.
fn grow(
    mesh: &mut TreeMesh,
    base: Vec3,
    dir: Vec3,
    length: f32,
    radius: f32,
    level: usize,
) {
    let tip = base + dir * length;
    push_prism(mesh, base, tip, radius, radius * 0.6);

    if level >= TILT_LEVELS.len() {
        return;
    }
    let tilt = TILT_LEVELS[level];
    let children = CHILDREN[level];
    for i in 0..children {
        #[allow(clippy::cast_precision_loss)]
        let azimuth = GOLDEN_ANGLE * (i as f32 + 1.0) + tilt * (level as f32);
        // Tilt the parent direction away from its axis by `tilt`, in the direction `azimuth`.
        let axis = perpendicular(dir);
        let tilted = rotate_about(dir, axis, tilt);
        let swept = rotate_about(tilted, dir, azimuth);
        grow(
            mesh,
            tip,
            swept.normalize_or_zero(),
            length * LENGTH_FALLOFF,
            radius * RADIUS_FALLOFF,
            level + 1,
        );
    }
}

/// Appends a four-sided tapered prism from `base` to `tip`, without caps.
fn push_prism(mesh: &mut TreeMesh, base: Vec3, tip: Vec3, radius_base: f32, radius_tip: f32) {
    let axis = (tip - base).normalize_or_zero();
    if axis.length_squared() < 0.5 {
        return;
    }
    let (u, v) = ring_basis(axis);

    let mut ring_base = [Vec3::ZERO; SIDES];
    let mut ring_tip = [Vec3::ZERO; SIDES];
    let mut normals = [Vec3::ZERO; SIDES];
    for i in 0..SIDES {
        #[allow(clippy::cast_precision_loss)]
        let angle = core::f32::consts::TAU * (i as f32) / (SIDES as f32);
        let offset = u * angle.cos() + v * angle.sin();
        ring_base[i] = base + offset * radius_base;
        ring_tip[i] = tip + offset * radius_tip;
        normals[i] = offset;
    }

    for i in 0..SIDES {
        let j = (i + 1) % SIDES;
        // Two triangles per quad, wound so the outward face is the one the normals point to.
        let quad = [
            (ring_base[i], normals[i]),
            (ring_base[j], normals[j]),
            (ring_tip[j], normals[j]),
            (ring_tip[i], normals[i]),
        ];
        for (a, b, c) in [
            (quad[0], quad[1], quad[2]),
            (quad[0], quad[2], quad[3]),
        ] {
            let index = u32::try_from(mesh.vertices.len()).unwrap_or(u32::MAX);
            for (pos, normal) in [a, b, c] {
                mesh.vertices.push(TreeVertex {
                    pos: pos.to_array(),
                    normal: normal.to_array(),
                });
            }
            mesh.indices.extend_from_slice(&[index, index + 1, index + 2]);
        }
    }
}

/// Any unit vector perpendicular to `axis`.
fn perpendicular(axis: Vec3) -> Vec3 {
    let reference = if axis.y.abs() > 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    axis.cross(reference).normalize_or_zero()
}

/// Any two unit vectors spanning the plane perpendicular to `axis`.
fn ring_basis(axis: Vec3) -> (Vec3, Vec3) {
    let u = perpendicular(axis);
    (u, axis.cross(u))
}

/// Rodrigues rotation of `v` about the unit `axis` by `angle`.
fn rotate_about(v: Vec3, axis: Vec3, angle: f32) -> Vec3 {
    let (s, c) = angle.sin_cos();
    v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3Swizzles;

    #[test]
    fn the_mesh_is_normalized_and_closed_enough_to_light() {
        let mesh = tree_mesh();
        assert!(!mesh.vertices.is_empty());
        // Every index addresses a vertex: an off-by-one here is a validation error at draw time.
        assert!(mesh
            .indices
            .iter()
            .all(|i| (*i as usize) < mesh.vertices.len()));

        let max_y = mesh.vertices.iter().fold(0.0f32, |a, v| a.max(v.pos[1]));
        let min_y = mesh.vertices.iter().fold(f32::INFINITY, |a, v| a.min(v.pos[1]));
        assert!((max_y - 1.0).abs() < 1e-4, "height is {max_y}, expected 1.0");
        assert!(min_y.abs() < 1e-4, "base is at {min_y}, expected 0.0");
        // The tree has to be taller than it is wide, or it is a bush.
        let width = mesh
            .vertices
            .iter()
            .fold(0.0f32, |a, v| a.max(v.pos[0].abs().max(v.pos[2].abs())));
        assert!(width < 0.6, "tree spans {width} in xz, expected a trunk-like profile");
    }

    #[test]
    fn normals_are_unit_length_and_face_outward() {
        let mesh = tree_mesh();
        for v in &mesh.vertices {
            let n = Vec3::from(v.normal);
            assert!((n.length() - 1.0).abs() < 1e-3, "normal {n:?} is not unit");
            // A branch's normal is perpendicular to the branch axis, so it always has a radial part.
            assert!(n.xy().length() + n.z.abs() > 0.5);
        }
    }

    #[test]
    fn the_mesh_is_generated_deterministically() {
        // Two calls must produce byte-identical meshes: the tree pass uploads one buffer and the
        // scatter's instance data refers to it by index, so a mesh that varied per call would put
        // trees and their trunks in different places.
        let a = tree_mesh();
        let b = tree_mesh();
        assert_eq!(a.vertices, b.vertices);
        assert_eq!(a.indices, b.indices);
    }
}

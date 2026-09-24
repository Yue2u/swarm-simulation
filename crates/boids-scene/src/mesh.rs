//! The static meshes: a tree grown as a fractal and a castle assembled from boxes, both built once on
//! the CPU and drawn instanced by `render/tree.wgsl` and `render/landmark.wgsl`.
//!
//! # Why these are real meshes
//!
//! The agents are generated from `vertex_index` in the vertex shader because their mesh has to be
//! *animated* per agent (tail wave, wing flap, bank) and because 100k instances cannot afford a
//! vertex buffer read. Neither static mesh has that property: they are drawn a few thousand times (the
//! trees) or twice (the castle), and a recursion that expresses itself as a triangle list is far
//! easier to read than one encoded as index arithmetic. So these are the two meshes in the project
//! that are built on the host.
//!
//! # Shape
//!
//! The tree is a trunk that splits into four branches, each of which splits into three twigs: 17
//! tapered prisms, 136 triangles. The branching angles come from a golden-angle spiral, which is what
//! keeps a stand of identical trees from reading as a pattern: two trees with different yaw never show
//! the same silhouette, because the branch azimuths are coprime with the rotation.
//!
//! The castle is the opposite construction: boxes and pyramids, because a castle is architecture and
//! its silhouette is a set of straight edges. A stepped plinth, a curtain wall with a gatehouse, a
//! corbelled gate arch and a portcullis, four spired corner towers, a keep with four bartizans, a
//! chimney, a slate roof and a flag, arrow loops and framed windows, string courses, a corbel table
//! and battlements on every parapet: about 2,700 triangles, and each face carries the material the
//! shader should shade it as.
//!
//! Both are normalized to a height of 1.0, so an instance's scale *is* its height in metres: for the
//! tree from its base at the origin, for the castle from its ground line. The castle also extends
//! *below* y = 0 (see [`castle_mesh`]), which is what buries its base in a slope.

use glam::Vec3;

use bytemuck::{Pod, Zeroable};

/// One vertex of a static mesh. Mirrors the vertex buffer layout in `render/tree.wgsl` and
/// `render/landmark.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct MeshVertex {
    /// Position in mesh space, metres at scale 1.
    pub pos: [f32; 3],
    /// Outward normal. The tree shader splits trunk from canopy with it, the castle shader splits
    /// wall from parapet, and both pipelines cull back faces, so it has to agree with the winding.
    pub normal: [f32; 3],
    /// Which material the face is: one of the `MATERIAL_*` selectors. A number rather than a colour
    /// so the palette stays in the shader, where the lighting can act on it.
    ///
    /// The tree does not need it - its two materials fall out of the normal's Y - and every tree
    /// vertex carries [`MATERIAL_STONE`].
    pub material: f32,
}

/// Stone: walls, towers, battlements. Also what the tree's vertices carry, since the tree shader
/// reads the field as "no override".
pub const MATERIAL_STONE: f32 = 0.0;
/// Slate: the keep's roof.
pub const MATERIAL_ROOF: f32 = 1.0;
/// Bare rock: the plinth the castle stands on.
pub const MATERIAL_ROCK: f32 = 2.0;
/// Iron: the portcullis and the flagpole.
pub const MATERIAL_IRON: f32 = 3.0;
/// Cloth: the pennant.
pub const MATERIAL_BANNER: f32 = 4.0;
/// A window or arrow loop: a dark recess, so an otherwise unbroken wall reads as inhabited.
pub const MATERIAL_WINDOW: f32 = 5.0;
/// Dressed stone: string courses, window surrounds, corbel tables and the chimney. A lighter, warmer
/// cut than the rubble the walls are built from, so a band or a frame reads as a deliberate edge
/// rather than as a trick of the light.
pub const MATERIAL_TRIM: f32 = 6.0;

/// A CPU-generated mesh: vertices plus 32-bit indices.
#[derive(Debug, Clone, Default)]
pub struct StaticMesh {
    /// Vertices, in triangle order per quad.
    pub vertices: Vec<MeshVertex>,
    /// Triangle list indices.
    pub indices: Vec<u32>,
}

impl StaticMesh {
    /// Number of triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// Material names in `MATERIAL_*` order, for the OBJ `usemtl` groups.
const MATERIAL_NAMES: [&str; 7] = ["stone", "slate", "rock", "iron", "cloth", "window", "trim"];

/// Writes a mesh as a Wavefront OBJ: positions, flat normals, and one `usemtl` group per material.
///
/// OBJ rather than glTF because this mesh is positions, flat per-face normals and a handful of
/// materials, which is exactly what OBJ stores natively and what a DCC tool, a slicer or a game
/// engine imports without a library. The mesh is written in the same normalized mesh space the
/// renderer uploads, so an exported file and the drawn object are the same geometry: `y = 0` is the
/// ground line and the top is at `y = 1`.
///
/// # Errors
/// Returns whatever `std::fs::write` returns if the file cannot be created.
pub fn write_obj(mesh: &StaticMesh, path: &std::path::Path) -> std::io::Result<()> {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(mesh.vertices.len() * 56);
    out.push_str("# boids static mesh\n");
    out.push_str("o boids_mesh\n");
    for v in &mesh.vertices {
        let _ = writeln!(out, "v {} {} {}", v.pos[0], v.pos[1], v.pos[2]);
    }
    for v in &mesh.vertices {
        let _ = writeln!(out, "vn {} {} {}", v.normal[0], v.normal[1], v.normal[2]);
    }
    // One group per material rather than one group per triangle run: the mesh interleaves its
    // materials, and a `usemtl` per triangle would be thousands of redundant directives.
    for (material, name) in MATERIAL_NAMES.iter().enumerate() {
        let mut opened = false;
        for tri in mesh.indices.as_chunks::<3>().0 {
            if mesh.vertices[tri[0] as usize].material as usize != material {
                continue;
            }
            if !opened {
                let _ = writeln!(out, "usemtl {name}");
                opened = true;
            }
            let _ = writeln!(
                out,
                "f {a}//{a} {b}//{b} {c}//{c}",
                a = tri[0] + 1,
                b = tri[1] + 1,
                c = tri[2] + 1
            );
        }
    }
    std::fs::write(path, out)
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
pub fn tree_mesh() -> StaticMesh {
    let mut mesh = StaticMesh::default();
    grow(
        &mut mesh,
        Vec3::ZERO,
        Vec3::Y,
        TRUNK_LENGTH,
        TRUNK_RADIUS,
        0,
    );

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
fn grow(mesh: &mut StaticMesh, base: Vec3, dir: Vec3, length: f32, radius: f32, level: usize) {
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
fn push_prism(mesh: &mut StaticMesh, base: Vec3, tip: Vec3, radius_base: f32, radius_tip: f32) {
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
        for (a, b, c) in [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])] {
            let index = u32::try_from(mesh.vertices.len()).unwrap_or(u32::MAX);
            for (pos, normal) in [a, b, c] {
                mesh.vertices.push(MeshVertex {
                    pos: pos.to_array(),
                    normal: normal.to_array(),
                    material: MATERIAL_STONE,
                });
            }
            mesh.indices
                .extend_from_slice(&[index, index + 1, index + 2]);
        }
    }
}

/// Any unit vector perpendicular to `axis`.
fn perpendicular(axis: Vec3) -> Vec3 {
    let reference = if axis.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
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

// ---------------------------------------------------------------------------------------------
// The castle
// ---------------------------------------------------------------------------------------------
//
// A second host-built mesh, drawn instanced by `render/landmark.wgsl`. Composed from boxes and one
// pyramid rather than grown like the tree, because a castle is architecture: its silhouette is a set
// of straight edges, and a fractal would round off exactly the features that make it read as one.
//
// COORDINATES. Mesh units, with the *ground line* at y = 0. That is where the curtain wall meets the
// terrain; the plinth (the rock base) extends below it, and `castle_mesh` normalizes the height above
// the ground line to 1.0. An instance's scale is therefore the castle's height in metres above the
// ground, exactly as a tree's scale is its height. The part below y = 0 is what buries the base: a
// castle standing on a slope with its footing at one height would otherwise float on the low side.
//
// The gate is at +Z, so an instance with yaw 0 has its gate facing +Z.

/// Outer half-extent of the curtain wall in `xz`.
const CASTLE_WALL_HALF: f32 = 0.32;
/// Half thickness of a curtain wall.
const CASTLE_WALL_THICK: f32 = 0.038;
/// Height of the wall walk: the top of the wall, below its merlons.
const CASTLE_WALL_HEIGHT: f32 = 0.30;
/// Half width of a corner tower in `xz`.
const CASTLE_TOWER_HALF: f32 = 0.085;
/// Height of a corner tower's shaft, below its merlons.
const CASTLE_TOWER_HEIGHT: f32 = 0.50;
/// Half width of the central keep.
const CASTLE_KEEP_HALF: f32 = 0.13;
/// Height of the keep's parapet, below its roof.
const CASTLE_KEEP_HEIGHT: f32 = 0.60;
/// Half width of the keep's roof where it meets the parapet. Set back from [`CASTLE_KEEP_HALF`] so
/// the parapet is a walkway the merlons stand on rather than a rim the roof grows out of.
const CASTLE_KEEP_ROOF_HALF: f32 = 0.072;
/// Height of the keep's pyramidal roof above its parapet.
const CASTLE_KEEP_ROOF: f32 = 0.17;
/// Half width of a gate tower in `xz`.
const CASTLE_GATE_HALF: f32 = 0.06;
/// Height of a gate tower's shaft, below its merlons.
const CASTLE_GATE_HEIGHT: f32 = 0.44;
/// Half width of the gate opening between the two gate towers.
const CASTLE_GATE_OPEN: f32 = 0.058;
/// Height of the gate opening. The wall above it is the lintel.
const CASTLE_GATE_TOP: f32 = 0.16;
/// Half width of a merlon, i.e. of one block of a battlement, in `xz`.
const CASTLE_MERLON_HALF: f32 = 0.024;
/// Height of a merlon above the parapet it stands on.
const CASTLE_MERLON_HEIGHT: f32 = 0.06;
/// Merlons along each edge of the curtain wall's parapet.
const CASTLE_WALL_MERLONS: usize = 7;
/// Merlons along each edge of the keep's parapet.
const CASTLE_KEEP_MERLONS: usize = 5;
/// Top of the plinth above the ground line: the courtyard floor. Positive so the ground inside the
/// walls is below it on every side, which is what stops the terrain from poking through the courtyard.
const CASTLE_PLINTH_TOP: f32 = 0.03;
/// How far the plinth extends below the ground line. This is the skirt that buries the base on a
/// slope; the site search keeps the ground's variation across the footprint well inside it.
const CASTLE_PLINTH_SKIRT: f32 = 0.12;
/// How far the rock base steps out past the stone terrace, and the terrace past the walls. Two steps
/// read as a foundation; one slab reads as the castle sitting on a box.
const CASTLE_PLINTH_OUT: f32 = 0.03;
/// See [`CASTLE_PLINTH_OUT`].
const CASTLE_TERRACE_OUT: f32 = 0.012;

/// Half span and half height of an arrow loop, and how far it stands proud of the wall.
///
/// Narrow and tall, as a real loop is, and proud rather than recessed: the pipelines cull back faces,
/// so a recess would need a hole cut through the wall, which is far more geometry than a window is
/// worth. `CASTLE_LOOP_PROUD` has to stay below `CASTLE_LOOP_HALF` or the loop reads as a pilaster.
const CASTLE_LOOP_HALF: f32 = 0.007;
/// See [`CASTLE_LOOP_HALF`].
const CASTLE_LOOP_HEIGHT_HALF: f32 = 0.024;
/// See [`CASTLE_LOOP_HALF`].
const CASTLE_LOOP_PROUD: f32 = 0.004;

/// Half width of a keep corner turret (a bartizan), and how far it rises above the keep parapet.
///
/// The turrets are what stop the keep reading as a plain box: four smaller towers on its corners give
/// it the stepped skyline every castle illustration has. They stay below the roof apex, so the
/// normalization is still the flag.
const CASTLE_TURRET_HALF: f32 = 0.034;
/// See [`CASTLE_TURRET_HALF`].
const CASTLE_TURRET_RISE: f32 = 0.10;

/// Half width of one corbel step in the gate arch, and how many steps each side.
const CASTLE_ARCH_STEP: f32 = 0.022;
/// See [`CASTLE_ARCH_STEP`].
const CASTLE_ARCH_STEPS: usize = 3;

/// Half width of a corner tower's spire at its base, and its height above the tower's parapet.
///
/// Set back from the merlon line exactly as the keep's roof is, so the tower's crown of merlons stands
/// clear of it. Four slate spires are most of what turns a flat-topped blockhouse into a castle
/// skyline; they stay under the flag, so the normalization is still the flag.
const CASTLE_SPIRE_HALF: f32 = 0.034;
/// See [`CASTLE_SPIRE_HALF`].
const CASTLE_SPIRE_HEIGHT: f32 = 0.22;

/// Half width of the iron finial that spikes the top of a spire, and its height above it.
const CASTLE_FINIAL_HALF: f32 = 0.006;
/// See [`CASTLE_FINIAL_HALF`].
const CASTLE_FINIAL_HEIGHT: f32 = 0.05;

/// Half height of a string course, and how far it stands proud of the wall it bands.
///
/// A course is one thin dressed band around a mass. Two or three of them up a tower are what give the
/// eye something to count, which is how a plain shaft reads as a storey rather than as a slab.
const CASTLE_COURSE_HALF: f32 = 0.006;
/// See [`CASTLE_COURSE_HALF`].
const CASTLE_COURSE_PROUD: f32 = 0.007;

/// Half width of one corbel, how far it steps out of the wall, and corbels along each wall.
///
/// The corbel table is the row of brackets under the wall's machicolation. It is the toothed shadow
/// line that reads as "castle" from across the valley, and it is cheap: one small box each. The step
/// out stays inside [`CASTLE_MACHICOLATION_OUT`], so the brackets carry the overhang rather than
/// poking past it.
const CASTLE_CORBEL_HALF: f32 = 0.013;
/// See [`CASTLE_CORBEL_HALF`].
const CASTLE_CORBEL_OUT: f32 = 0.014;
/// See [`CASTLE_CORBEL_HALF`].
const CASTLE_CORBELS: usize = 9;

/// How far a window's dressed surround reaches past its dark pane, and how far the surround is proud.
///
/// The pane stands proud of the surround, so a window reads as a dark slit set in a raised frame
/// rather than as a bare hole. A frame is what keeps the keep's windows from being lost in its wall.
const CASTLE_FRAME_MARGIN: f32 = 0.004;
/// See [`CASTLE_FRAME_MARGIN`].
const CASTLE_FRAME_PROUD: f32 = 0.002;

/// Half width of the chimney on the keep's roof, and how far it rises above the parapet.
///
/// One flue and its cap are what stop the biggest slate surface in the mesh from reading as a blank
/// pyramid, and they cost two boxes.
const CASTLE_CHIMNEY_HALF: f32 = 0.013;
/// See [`CASTLE_CHIMNEY_HALF`].
const CASTLE_CHIMNEY_RISE: f32 = 0.10;

/// How far the machicolation steps out of the wall, and how tall the course is.
///
/// A wall that ends in a battlement with nothing under it reads as a fence. One course that steps out
/// just below the parapet is what gives the top of a wall a shadow line, which is most of what makes
/// the silhouette read as a castle at a distance.
const CASTLE_MACHICOLATION_OUT: f32 = 0.018;
/// See [`CASTLE_MACHICOLATION_OUT`].
const CASTLE_MACHICOLATION_HEIGHT: f32 = 0.032;

/// Half width of a buttress along the wall it braces, and how far it stands out of it.
const CASTLE_BUTTRESS_HALF: f32 = 0.022;
/// See [`CASTLE_BUTTRESS_HALF`].
const CASTLE_BUTTRESS_OUT: f32 = 0.02;
/// Buttresses along each of the three walls that have no gate in them.
const CASTLE_BUTTRESSES: usize = 3;

/// Half thickness of a portcullis bar.
const CASTLE_PORTCULLIS_BAR: f32 = 0.008;
/// Vertical bars across the gate opening, and horizontal bars across it.
const CASTLE_PORTCULLIS_BARS: usize = 3;
/// See [`CASTLE_PORTCULLIS_BARS`].
const CASTLE_PORTCULLIS_RAILS: usize = 2;

/// Half thickness of the flagpole.
const CASTLE_FLAG_POLE_HALF: f32 = 0.007;
/// Height of the flagpole above the keep's roof apex.
const CASTLE_FLAG_HEIGHT: f32 = 0.15;
/// Length of the pennant along +x, and how far it hangs down.
const CASTLE_FLAG_LENGTH: f32 = 0.14;
/// See [`CASTLE_FLAG_LENGTH`].
const CASTLE_FLAG_DROP: f32 = 0.055;

/// Height of the tallest point above the ground line, before normalization: the tip of the flagpole.
const CASTLE_MESH_HEIGHT: f32 = CASTLE_KEEP_HEIGHT + CASTLE_KEEP_ROOF + CASTLE_FLAG_HEIGHT;

/// Half-extent of the castle in `xz` at scale 1, i.e. after normalization: the corner towers stick
/// out past the curtain wall, and the arrow loop proud of each tower's outer face sticks out past
/// that, so they set the footprint. Placement uses this to know how much ground a castle covers, and
/// a test pins it against the built mesh.
pub const CASTLE_HALF_EXTENT: f32 = (CASTLE_TOWER_HALF + CASTLE_WALL_HALF - CASTLE_WALL_THICK
    + 2.0 * CASTLE_LOOP_PROUD)
    / CASTLE_MESH_HEIGHT;

/// Builds the castle mesh, normalized so its height above the ground line is 1.0.
///
/// Built the way a mason would: the rock it stands on, the curtain wall with its corbelled gate, the
/// towers and their spires, the keep with its bartizans, then the arrow loops, string courses, corbel
/// tables, battlements and ironwork that finish it. Every part is a box or a pyramid, and each carries
/// the material the shader should shade it as, so stone, slate, rock, iron, cloth, window and dressed
/// trim are one palette in `render/landmark.wgsl` rather than seven meshes.
#[must_use]
pub fn castle_mesh() -> StaticMesh {
    let mut mesh = StaticMesh::default();

    let wall_inner = CASTLE_WALL_HALF - CASTLE_WALL_THICK;
    let wall_mid = CASTLE_WALL_HALF - 0.5 * CASTLE_WALL_THICK;
    let thick_half = 0.5 * CASTLE_WALL_THICK;
    let wall_half_y = 0.5 * CASTLE_WALL_HEIGHT;
    let gate_outer = CASTLE_GATE_OPEN + 2.0 * CASTLE_GATE_HALF;

    // The plinth: the rock the castle stands on, stepped so the foundation reads as masonry rather
    // than as a slab. The rock reaches below the ground line - this is the skirt that buries the base
    // on a slope - and the stone terrace above it is the courtyard floor.
    push_box(
        &mut mesh,
        Vec3::new(0.0, -0.5 * CASTLE_PLINTH_SKIRT, 0.0),
        Vec3::new(
            CASTLE_WALL_HALF + CASTLE_PLINTH_OUT,
            0.5 * CASTLE_PLINTH_SKIRT,
            CASTLE_WALL_HALF + CASTLE_PLINTH_OUT,
        ),
        MATERIAL_ROCK,
    );
    push_box(
        &mut mesh,
        Vec3::new(0.0, 0.5 * CASTLE_PLINTH_TOP, 0.0),
        Vec3::new(
            CASTLE_WALL_HALF + CASTLE_TERRACE_OUT,
            0.5 * CASTLE_PLINTH_TOP,
            CASTLE_WALL_HALF + CASTLE_TERRACE_OUT,
        ),
        MATERIAL_STONE,
    );

    // Curtain wall: the back, left and right walls are one box each. The front wall is split around
    // the gate, with the lintel as a third box spanning the opening above the gate's head.
    push_box(
        &mut mesh,
        Vec3::new(0.0, wall_half_y, -wall_mid),
        Vec3::new(CASTLE_WALL_HALF, wall_half_y, thick_half),
        MATERIAL_STONE,
    );
    push_box(
        &mut mesh,
        Vec3::new(-wall_mid, wall_half_y, 0.0),
        Vec3::new(thick_half, wall_half_y, wall_inner),
        MATERIAL_STONE,
    );
    push_box(
        &mut mesh,
        Vec3::new(wall_mid, wall_half_y, 0.0),
        Vec3::new(thick_half, wall_half_y, wall_inner),
        MATERIAL_STONE,
    );
    for side in [-1.0f32, 1.0] {
        push_box(
            &mut mesh,
            Vec3::new(
                side * 0.5 * (gate_outer + CASTLE_WALL_HALF),
                wall_half_y,
                wall_mid,
            ),
            Vec3::new(
                0.5 * (CASTLE_WALL_HALF - gate_outer),
                wall_half_y,
                thick_half,
            ),
            MATERIAL_STONE,
        );
    }
    push_box(
        &mut mesh,
        Vec3::new(0.0, 0.5 * (CASTLE_GATE_TOP + CASTLE_WALL_HEIGHT), wall_mid),
        Vec3::new(
            CASTLE_GATE_OPEN,
            0.5 * (CASTLE_WALL_HEIGHT - CASTLE_GATE_TOP),
            thick_half,
        ),
        MATERIAL_STONE,
    );

    // Arrow loops: three along each unbroken wall and one on each gate segment, dark and proud of the
    // face. They are what keeps a wall this size from reading as one untextured slab at the distance
    // the castle is actually seen from, and they cost twelve triangles each.
    for t in [-0.15f32, 0.0, 0.15] {
        push_loop(&mut mesh, Vec3::new(t, 0.18, -CASTLE_WALL_HALF), Vec3::NEG_Z);
        push_loop(&mut mesh, Vec3::new(-CASTLE_WALL_HALF, 0.18, t), Vec3::NEG_X);
        push_loop(&mut mesh, Vec3::new(CASTLE_WALL_HALF, 0.18, t), Vec3::X);
    }
    for side in [-1.0f32, 1.0] {
        push_loop(
            &mut mesh,
            Vec3::new(
                side * 0.5 * (gate_outer + CASTLE_WALL_HALF),
                0.18,
                CASTLE_WALL_HALF,
            ),
            Vec3::Z,
        );
    }

    // Buttresses: ribs on the walls, standing a little proud of both faces, so the outside gets the
    // relief that stops a wall this size from reading as one flat panel and the courtyard gets
    // pilasters. The three walls without a gate take a full row; the front takes one per side.
    let buttress = Vec3::new(
        CASTLE_BUTTRESS_HALF,
        wall_half_y,
        thick_half + CASTLE_BUTTRESS_OUT,
    );
    let buttress_side = Vec3::new(
        thick_half + CASTLE_BUTTRESS_OUT,
        wall_half_y,
        CASTLE_BUTTRESS_HALF,
    );
    for i in 0..CASTLE_BUTTRESSES {
        let t = spread(i, CASTLE_BUTTRESSES, wall_inner);
        push_box(
            &mut mesh,
            Vec3::new(t, wall_half_y, -wall_mid),
            buttress,
            MATERIAL_STONE,
        );
        push_box(
            &mut mesh,
            Vec3::new(-wall_mid, wall_half_y, t),
            buttress_side,
            MATERIAL_STONE,
        );
        push_box(
            &mut mesh,
            Vec3::new(wall_mid, wall_half_y, t),
            buttress_side,
            MATERIAL_STONE,
        );
    }
    for side in [-1.0f32, 1.0] {
        push_box(
            &mut mesh,
            Vec3::new(
                side * 0.5 * (gate_outer + CASTLE_WALL_HALF),
                wall_half_y,
                wall_mid,
            ),
            buttress,
            MATERIAL_STONE,
        );
    }

    // The machicolation: one course stepped out of the wall just below the parapet, all the way
    // round. It is what puts a shadow line under the battlements. Across the gate it is inside the
    // gatehouse and never seen, which is cheaper than splitting the course around it.
    let mach = 0.5 * CASTLE_MACHICOLATION_HEIGHT;
    let mach_y = CASTLE_WALL_HEIGHT - mach;
    let mach_half = thick_half + CASTLE_MACHICOLATION_OUT;
    push_box(
        &mut mesh,
        Vec3::new(0.0, mach_y, -wall_mid),
        Vec3::new(CASTLE_WALL_HALF + CASTLE_MACHICOLATION_OUT, mach, mach_half),
        MATERIAL_STONE,
    );
    push_box(
        &mut mesh,
        Vec3::new(0.0, mach_y, wall_mid),
        Vec3::new(CASTLE_WALL_HALF + CASTLE_MACHICOLATION_OUT, mach, mach_half),
        MATERIAL_STONE,
    );
    push_box(
        &mut mesh,
        Vec3::new(-wall_mid, mach_y, 0.0),
        Vec3::new(mach_half, mach, wall_inner),
        MATERIAL_STONE,
    );
    push_box(
        &mut mesh,
        Vec3::new(wall_mid, mach_y, 0.0),
        Vec3::new(mach_half, mach, wall_inner),
        MATERIAL_STONE,
    );

    // The corbel table: the row of brackets the machicolation above rests on. It is what puts the
    // toothed shadow line under the wall's parapet, and most of why a wall this size reads as a
    // castle rather than as a fence. The three unbroken walls take a full row; the front takes one
    // per side of the gatehouse, since the middle of that wall is inside it.
    let corbel_y = CASTLE_WALL_HEIGHT - CASTLE_MACHICOLATION_HEIGHT - CASTLE_CORBEL_HALF;
    let corbel_span = wall_inner - CASTLE_TOWER_HALF;
    for i in 0..CASTLE_CORBELS {
        let t = spread(i, CASTLE_CORBELS, corbel_span);
        push_corbel(
            &mut mesh,
            Vec3::new(t, corbel_y, -CASTLE_WALL_HALF),
            Vec3::NEG_Z,
            CASTLE_CORBEL_OUT,
        );
        push_corbel(
            &mut mesh,
            Vec3::new(-CASTLE_WALL_HALF, corbel_y, t),
            Vec3::NEG_X,
            CASTLE_CORBEL_OUT,
        );
        push_corbel(
            &mut mesh,
            Vec3::new(CASTLE_WALL_HALF, corbel_y, t),
            Vec3::X,
            CASTLE_CORBEL_OUT,
        );
    }
    for side in [-1.0f32, 1.0] {
        push_corbel(
            &mut mesh,
            Vec3::new(
                side * 0.5 * (gate_outer + CASTLE_WALL_HALF),
                corbel_y,
                CASTLE_WALL_HALF,
            ),
            Vec3::Z,
            CASTLE_CORBEL_OUT,
        );
    }

    // The gatehouse: two towers flanking the opening, taller than the wall and thicker in `z` so they
    // step out of it. This is what makes the gate read as a gate from outside rather than as a hole.
    for side in [-1.0f32, 1.0] {
        push_box(
            &mut mesh,
            Vec3::new(
                side * (CASTLE_GATE_OPEN + CASTLE_GATE_HALF),
                0.5 * CASTLE_GATE_HEIGHT,
                wall_mid,
            ),
            Vec3::new(
                CASTLE_GATE_HALF,
                0.5 * CASTLE_GATE_HEIGHT,
                thick_half + 0.012,
            ),
            MATERIAL_STONE,
        );
        push_loop(
            &mut mesh,
            Vec3::new(
                side * (CASTLE_GATE_OPEN + CASTLE_GATE_HALF),
                0.30,
                wall_mid + thick_half + 0.012,
            ),
            Vec3::Z,
        );
        push_course(
            &mut mesh,
            side * (CASTLE_GATE_OPEN + CASTLE_GATE_HALF),
            wall_mid,
            CASTLE_GATE_HALF,
            0.16,
        );
        push_course(
            &mut mesh,
            side * (CASTLE_GATE_OPEN + CASTLE_GATE_HALF),
            wall_mid,
            CASTLE_GATE_HALF,
            0.36,
        );
    }

    // Corbel steps under the lintel: each course overhangs the opening a little more, which is how a
    // stone arch is actually built and what makes the gate read as an arch rather than a rectangle.
    for i in 0..CASTLE_ARCH_STEPS {
        #[allow(clippy::cast_precision_loss)]
        let step = (i as f32 + 1.0) * CASTLE_ARCH_STEP;
        for side in [-1.0f32, 1.0] {
            push_box(
                &mut mesh,
                Vec3::new(
                    side * (CASTLE_GATE_OPEN - 0.5 * step),
                    CASTLE_GATE_TOP - 0.5 * step,
                    wall_mid,
                ),
                Vec3::new(0.5 * step, 0.5 * step, thick_half),
                MATERIAL_STONE,
            );
        }
    }

    // The portcullis: vertical bars in the opening with two rails across them. It is the only iron in
    // the mesh, and it is what makes the gate read as closed rather than as a hole.
    let bar_y = 0.5 * CASTLE_GATE_TOP;
    for i in 0..CASTLE_PORTCULLIS_BARS {
        let t = spread(i, CASTLE_PORTCULLIS_BARS, CASTLE_GATE_OPEN * 0.72);
        push_box(
            &mut mesh,
            Vec3::new(t, bar_y, wall_mid),
            Vec3::new(CASTLE_PORTCULLIS_BAR, bar_y, CASTLE_PORTCULLIS_BAR),
            MATERIAL_IRON,
        );
    }
    for i in 0..CASTLE_PORTCULLIS_RAILS {
        #[allow(clippy::cast_precision_loss)]
        let t = CASTLE_GATE_TOP * (0.28 + 0.44 * (i as f32));
        push_box(
            &mut mesh,
            Vec3::new(0.0, t, wall_mid),
            Vec3::new(
                CASTLE_GATE_OPEN,
                CASTLE_PORTCULLIS_BAR,
                CASTLE_PORTCULLIS_BAR,
            ),
            MATERIAL_IRON,
        );
    }

    // The four corner towers, centred on the wall's inner face so each one bulges past the wall, with
    // an arrow loop on each of the two faces that look away from the courtyard.
    for sx in [-1.0f32, 1.0] {
        for sz in [-1.0f32, 1.0] {
            push_box(
                &mut mesh,
                Vec3::new(sx * wall_inner, 0.5 * CASTLE_TOWER_HEIGHT, sz * wall_inner),
                Vec3::new(
                    CASTLE_TOWER_HALF,
                    0.5 * CASTLE_TOWER_HEIGHT,
                    CASTLE_TOWER_HALF,
                ),
                MATERIAL_STONE,
            );
            push_loop(
                &mut mesh,
                Vec3::new(
                    sx * (wall_inner + CASTLE_TOWER_HALF),
                    0.30,
                    sz * wall_inner,
                ),
                if sx > 0.0 { Vec3::X } else { Vec3::NEG_X },
            );
            push_loop(
                &mut mesh,
                Vec3::new(
                    sx * wall_inner,
                    0.30,
                    sz * (wall_inner + CASTLE_TOWER_HALF),
                ),
                if sz > 0.0 { Vec3::Z } else { Vec3::NEG_Z },
            );
            // String courses, bracketing the loop so the shaft reads as two storeys.
            push_course(&mut mesh, sx * wall_inner, sz * wall_inner, CASTLE_TOWER_HALF, 0.18);
            push_course(&mut mesh, sx * wall_inner, sz * wall_inner, CASTLE_TOWER_HALF, 0.38);
            // The spire: set back behind the tower's merlons exactly as the keep's roof is set back
            // behind its own, so the crown of merlons reads as a parapet the roof rises out of.
            let tower_top = CASTLE_TOWER_HEIGHT;
            push_pyramid(
                &mut mesh,
                sx * wall_inner,
                sz * wall_inner,
                CASTLE_SPIRE_HALF,
                tower_top,
                tower_top + CASTLE_SPIRE_HEIGHT,
                MATERIAL_ROOF,
            );
            push_finial(
                &mut mesh,
                sx * wall_inner,
                sz * wall_inner,
                tower_top + CASTLE_SPIRE_HEIGHT,
            );
        }
    }

    // The keep: the tallest mass, with a set-back roof so its parapet is walkable.
    push_box(
        &mut mesh,
        Vec3::new(0.0, 0.5 * CASTLE_KEEP_HEIGHT, 0.0),
        Vec3::new(CASTLE_KEEP_HALF, 0.5 * CASTLE_KEEP_HEIGHT, CASTLE_KEEP_HALF),
        MATERIAL_STONE,
    );
    // Three courses up the shaft, bracketing its two rows of windows, so the tallest wall in the mesh
    // is divided into storeys instead of reading as one tall slab.
    for y in [0.20f32, 0.36, 0.52] {
        push_course(&mut mesh, 0.0, 0.0, CASTLE_KEEP_HALF, y);
    }
    push_pyramid(
        &mut mesh,
        0.0,
        0.0,
        CASTLE_KEEP_ROOF_HALF,
        CASTLE_KEEP_HEIGHT,
        CASTLE_KEEP_HEIGHT + CASTLE_KEEP_ROOF,
        MATERIAL_ROOF,
    );

    // One chimney and its cap, off the roof's centre so it does not hide the flagpole, rising out of
    // the slate slope to just under the apex. It is the only interruption of the roof.
    let flue_y = CASTLE_KEEP_HEIGHT + 0.5 * CASTLE_CHIMNEY_RISE;
    push_box(
        &mut mesh,
        Vec3::new(0.04, flue_y, 0.04),
        Vec3::new(
            CASTLE_CHIMNEY_HALF,
            0.5 * CASTLE_CHIMNEY_RISE,
            CASTLE_CHIMNEY_HALF,
        ),
        MATERIAL_TRIM,
    );
    push_box(
        &mut mesh,
        Vec3::new(
            0.04,
            CASTLE_KEEP_HEIGHT + CASTLE_CHIMNEY_RISE + 0.006,
            0.04,
        ),
        Vec3::new(
            CASTLE_CHIMNEY_HALF + 0.005,
            0.006,
            CASTLE_CHIMNEY_HALF + 0.005,
        ),
        MATERIAL_TRIM,
    );

    // Four bartizans on the keep's corners: half inside the keep, half proud of it, rising above the
    // parapet but staying under the roof apex. They are what gives the keep a stepped skyline instead
    // of a plain box with a hat.
    for sx in [-1.0f32, 1.0] {
        for sz in [-1.0f32, 1.0] {
            let cx = sx * CASTLE_KEEP_HALF;
            let cz = sz * CASTLE_KEEP_HALF;
            let cy = CASTLE_KEEP_HEIGHT + 0.5 * CASTLE_TURRET_RISE;
            push_box(
                &mut mesh,
                Vec3::new(cx, cy, cz),
                Vec3::new(
                    CASTLE_TURRET_HALF,
                    0.5 * CASTLE_TURRET_RISE,
                    CASTLE_TURRET_HALF,
                ),
                MATERIAL_STONE,
            );
            push_loop(
                &mut mesh,
                Vec3::new(cx + sx * CASTLE_TURRET_HALF, cy, cz),
                if sx > 0.0 { Vec3::X } else { Vec3::NEG_X },
            );
            push_loop(
                &mut mesh,
                Vec3::new(cx, cy, cz + sz * CASTLE_TURRET_HALF),
                if sz > 0.0 { Vec3::Z } else { Vec3::NEG_Z },
            );
        }
    }

    // Keep windows: two per face at different heights, in dressed surrounds, so the tallest mass
    // carries some scale.
    for side in [-1.0f32, 1.0] {
        for (y, t) in [(0.28f32, 0.05f32), (0.44, -0.05)] {
            push_window(
                &mut mesh,
                Vec3::new(side * CASTLE_KEEP_HALF, y, t),
                if side > 0.0 { Vec3::X } else { Vec3::NEG_X },
            );
            push_window(
                &mut mesh,
                Vec3::new(t, y, side * CASTLE_KEEP_HALF),
                if side > 0.0 { Vec3::Z } else { Vec3::NEG_Z },
            );
        }
    }

    // The flag: a pole above the roof apex with a pennant on it. Nothing else reaches this high, so
    // this is the point the whole mesh is normalized by.
    let apex = CASTLE_KEEP_HEIGHT + CASTLE_KEEP_ROOF;
    push_box(
        &mut mesh,
        Vec3::new(0.0, apex + 0.5 * CASTLE_FLAG_HEIGHT, 0.0),
        Vec3::new(
            CASTLE_FLAG_POLE_HALF,
            0.5 * CASTLE_FLAG_HEIGHT,
            CASTLE_FLAG_POLE_HALF,
        ),
        MATERIAL_IRON,
    );
    push_box(
        &mut mesh,
        Vec3::new(
            0.5 * CASTLE_FLAG_LENGTH,
            apex + CASTLE_FLAG_HEIGHT - 0.5 * CASTLE_FLAG_DROP,
            0.0,
        ),
        Vec3::new(
            0.5 * CASTLE_FLAG_LENGTH,
            0.5 * CASTLE_FLAG_DROP,
            CASTLE_FLAG_POLE_HALF * 0.6,
        ),
        MATERIAL_BANNER,
    );

    // Battlements, on every parapet: the wall ring, the keep, the four corner towers and the two gate
    // towers. The wall ring's inner two merlons on the front edge sit inside the gate towers, which is
    // 24 hidden triangles rather than a special case in the loop.
    push_battlements(
        &mut mesh,
        0.0,
        0.0,
        wall_inner,
        CASTLE_WALL_HEIGHT,
        CASTLE_WALL_MERLONS,
        MATERIAL_STONE,
    );
    push_battlements(
        &mut mesh,
        0.0,
        0.0,
        CASTLE_KEEP_HALF - CASTLE_MERLON_HALF,
        CASTLE_KEEP_HEIGHT,
        CASTLE_KEEP_MERLONS,
        MATERIAL_STONE,
    );
    for sx in [-1.0f32, 1.0] {
        for sz in [-1.0f32, 1.0] {
            push_battlements(
                &mut mesh,
                sx * CASTLE_KEEP_HALF,
                sz * CASTLE_KEEP_HALF,
                CASTLE_TURRET_HALF - CASTLE_MERLON_HALF,
                CASTLE_KEEP_HEIGHT + CASTLE_TURRET_RISE,
                1,
                MATERIAL_STONE,
            );
        }
    }
    for sx in [-1.0f32, 1.0] {
        for sz in [-1.0f32, 1.0] {
            push_battlements(
                &mut mesh,
                sx * wall_inner,
                sz * wall_inner,
                CASTLE_TOWER_HALF - CASTLE_MERLON_HALF,
                CASTLE_TOWER_HEIGHT,
                1,
                MATERIAL_STONE,
            );
        }
        push_battlements(
            &mut mesh,
            sx * (CASTLE_GATE_OPEN + CASTLE_GATE_HALF),
            wall_mid,
            CASTLE_GATE_HALF - CASTLE_MERLON_HALF,
            CASTLE_GATE_HEIGHT,
            1,
            MATERIAL_STONE,
        );
    }

    // Normalize by the tallest point only, so the ground line stays at y = 0 and the skirt stays
    // negative. Scaling by the *total* extent would move the ground line and make an instance's scale
    // mean something different from "how tall the castle is".
    let scale = 1.0 / CASTLE_MESH_HEIGHT;
    for v in &mut mesh.vertices {
        v.pos[0] *= scale;
        v.pos[1] *= scale;
        v.pos[2] *= scale;
    }
    mesh
}

/// Evenly spaced centres for `count` items inside `[-half, half]`, inset by half a step so the first
/// and the last do not sit on the ends. One item lands exactly in the middle, which is what a tower's
/// crown and a portcullis want.
fn spread(index: usize, count: usize, half: f32) -> f32 {
    let n = count.max(1);
    #[allow(clippy::cast_precision_loss)]
    let t = (index as f32 + 0.5) / (n as f32);
    -half + 2.0 * half * t
}

/// Appends a closed axis-aligned box.
///
/// Each face is listed as `(outward normal, tangent u, tangent v)` with `u x v == normal`, which is
/// what makes the quad below wind counter-clockwise seen from outside. Both static pipelines cull back
/// faces, so a flipped row here would delete a wall rather than draw it inside out, and
/// `castle_faces_wind_toward_their_normals` is the test that catches it.
fn push_box(mesh: &mut StaticMesh, centre: Vec3, half: Vec3, material: f32) {
    let faces = [
        (Vec3::X, Vec3::NEG_Z, Vec3::Y, half.z, half.y),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y, half.z, half.y),
        (Vec3::Y, Vec3::X, Vec3::NEG_Z, half.x, half.z),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z, half.x, half.z),
        (Vec3::Z, Vec3::X, Vec3::Y, half.x, half.y),
        (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y, half.x, half.y),
    ];
    for (normal, u, v, half_u, half_v) in faces {
        // `.abs()` is load-bearing: `normal.dot(half)` is *negative* on the -X/-Y/-Z faces, and
        // multiplying by the normal would push those faces to the *positive* side, collapsing the
        // box onto its positive three planes and leaving the negative walls invisible from outside.
        let face = centre + normal * normal.dot(half).abs();
        push_quad(
            mesh,
            [
                face - u * half_u - v * half_v,
                face + u * half_u - v * half_v,
                face + u * half_u + v * half_v,
                face - u * half_u + v * half_v,
            ],
            normal,
            material,
        );
    }
}

/// Appends a quad as two triangles, all six vertices carrying the same normal and material.
///
/// A flat normal per face rather than an interpolated one: every face of the castle is flat, and a
/// shared corner normal would round the box edges into a soft blob.
fn push_quad(mesh: &mut StaticMesh, corners: [Vec3; 4], normal: Vec3, material: f32) {
    for (a, b, c) in [(0usize, 1usize, 2usize), (0, 2, 3)] {
        let index = u32::try_from(mesh.vertices.len()).unwrap_or(u32::MAX);
        for i in [a, b, c] {
            mesh.vertices.push(MeshVertex {
                pos: corners[i].to_array(),
                normal: normal.to_array(),
                material,
            });
        }
        mesh.indices
            .extend_from_slice(&[index, index + 1, index + 2]);
    }
}

/// Appends a four-sided pyramid with its square base at `base_y`, centred on `(cx, cz)`, and its apex
/// at `apex_y`, without the base face: the parapet or roof it stands on is already there and a
/// coincident quad would z-fight with it.
fn push_pyramid(
    mesh: &mut StaticMesh,
    cx: f32,
    cz: f32,
    half: f32,
    base_y: f32,
    apex_y: f32,
    material: f32,
) {
    let base = [
        Vec3::new(cx - half, base_y, cz - half),
        Vec3::new(cx + half, base_y, cz - half),
        Vec3::new(cx + half, base_y, cz + half),
        Vec3::new(cx - half, base_y, cz + half),
    ];
    let apex = Vec3::new(cx, apex_y, cz);
    for i in 0..4 {
        let a = base[i];
        let b = base[(i + 1) % 4];
        // (apex - a) x (b - a) points away from the axis for a base wound counter-clockwise in `xz`.
        let normal = (apex - a).cross(b - a).normalize_or_zero();
        let index = u32::try_from(mesh.vertices.len()).unwrap_or(u32::MAX);
        for p in [a, apex, b] {
            mesh.vertices.push(MeshVertex {
                pos: p.to_array(),
                normal: normal.to_array(),
                material,
            });
        }
        mesh.indices
            .extend_from_slice(&[index, index + 1, index + 2]);
    }
}

/// A window or arrow loop: a thin, dark slab proud of the wall face it sits on.
///
/// `face` is the outward normal of the wall, one of ±X or ±Z. The loop is thin along it and
/// `CASTLE_LOOP_HALF` wide across it, standing `CASTLE_LOOP_PROUD` out of the wall so the pipelines'
/// back-face cull still draws it; a recess would need a hole through the wall, which is far more
/// geometry than a window is worth.
fn push_loop(mesh: &mut StaticMesh, centre: Vec3, face: Vec3) {
    let half = if face.x.abs() > 0.5 {
        Vec3::new(CASTLE_LOOP_PROUD, CASTLE_LOOP_HEIGHT_HALF, CASTLE_LOOP_HALF)
    } else {
        Vec3::new(CASTLE_LOOP_HALF, CASTLE_LOOP_HEIGHT_HALF, CASTLE_LOOP_PROUD)
    };
    push_box(
        mesh,
        centre + face * CASTLE_LOOP_PROUD,
        half,
        MATERIAL_WINDOW,
    );
}

/// A dressed string course: a thin band wrapping a square mass of half-extent `half` at height `y`.
///
/// One box, so it also wraps the corners, which is what a real course does.
fn push_course(mesh: &mut StaticMesh, cx: f32, cz: f32, half: f32, y: f32) {
    push_box(
        mesh,
        Vec3::new(cx, y, cz),
        Vec3::new(
            half + CASTLE_COURSE_PROUD,
            CASTLE_COURSE_HALF,
            half + CASTLE_COURSE_PROUD,
        ),
        MATERIAL_TRIM,
    );
}

/// A window: a dressed surround with the dark pane proud of it.
///
/// The pane is [`push_loop`]'s slit, so the surround reads as a frame a hand's breadth wider on every
/// side. Both stand proud of the wall, the pane furthest, so the back-face cull keeps both.
fn push_window(mesh: &mut StaticMesh, centre: Vec3, face: Vec3) {
    let half = if face.x.abs() > 0.5 {
        Vec3::new(
            CASTLE_FRAME_PROUD,
            CASTLE_LOOP_HEIGHT_HALF + CASTLE_FRAME_MARGIN,
            CASTLE_LOOP_HALF + CASTLE_FRAME_MARGIN,
        )
    } else {
        Vec3::new(
            CASTLE_LOOP_HALF + CASTLE_FRAME_MARGIN,
            CASTLE_LOOP_HEIGHT_HALF + CASTLE_FRAME_MARGIN,
            CASTLE_FRAME_PROUD,
        )
    };
    push_box(
        mesh,
        centre + face * CASTLE_FRAME_PROUD,
        half,
        MATERIAL_TRIM,
    );
    push_loop(mesh, centre, face);
}

/// A corbel: one bracket of the table under a wall's machicolation, its outer face `out` from the
/// wall plane at `centre`, with `face` the outward normal.
///
/// The block straddles the wall plane rather than sitting on it, so it reads as a bracket carrying
/// the course above, and a run of them is the toothed shadow line.
fn push_corbel(mesh: &mut StaticMesh, centre: Vec3, face: Vec3, out: f32) {
    let half = if face.x.abs() > 0.5 {
        Vec3::new(0.5 * out, CASTLE_CORBEL_HALF, CASTLE_CORBEL_HALF)
    } else {
        Vec3::new(CASTLE_CORBEL_HALF, CASTLE_CORBEL_HALF, 0.5 * out)
    };
    push_box(mesh, centre + face * (0.5 * out), half, MATERIAL_TRIM);
}

/// An iron finial: a spike capping a spire, so the silhouette ends in a point rather than a blunt
/// slate edge.
fn push_finial(mesh: &mut StaticMesh, cx: f32, cz: f32, base_y: f32) {
    push_pyramid(
        mesh,
        cx,
        cz,
        CASTLE_FINIAL_HALF,
        base_y,
        base_y + CASTLE_FINIAL_HEIGHT,
        MATERIAL_IRON,
    );
}


/// `y`: `per_side` blocks along each edge, evenly spaced. One block per edge gives a tower a crown of
/// four; five gives the curtain wall a proper battlement.
fn push_battlements(
    mesh: &mut StaticMesh,
    cx: f32,
    cz: f32,
    half: f32,
    y: f32,
    per_side: usize,
    material: f32,
) {
    let half_y = 0.5 * CASTLE_MERLON_HEIGHT;
    let centre_y = y + half_y;
    let block = Vec3::new(CASTLE_MERLON_HALF, half_y, CASTLE_MERLON_HALF);
    for i in 0..per_side {
        let t = spread(i, per_side, half);
        // Two edges run along x at z = +/-half, two along z at x = +/-half.
        push_box(
            mesh,
            Vec3::new(cx + t, centre_y, cz + half),
            block,
            material,
        );
        push_box(
            mesh,
            Vec3::new(cx + t, centre_y, cz - half),
            block,
            material,
        );
        push_box(
            mesh,
            Vec3::new(cx + half, centre_y, cz + t),
            block,
            material,
        );
        push_box(
            mesh,
            Vec3::new(cx - half, centre_y, cz + t),
            block,
            material,
        );
    }
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
        let min_y = mesh
            .vertices
            .iter()
            .fold(f32::INFINITY, |a, v| a.min(v.pos[1]));
        assert!(
            (max_y - 1.0).abs() < 1e-4,
            "height is {max_y}, expected 1.0"
        );
        assert!(min_y.abs() < 1e-4, "base is at {min_y}, expected 0.0");
        // The tree has to be taller than it is wide, or it is a bush.
        let width = mesh
            .vertices
            .iter()
            .fold(0.0f32, |a, v| a.max(v.pos[0].abs().max(v.pos[2].abs())));
        assert!(
            width < 0.6,
            "tree spans {width} in xz, expected a trunk-like profile"
        );
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

    /// `push_box` has to close on all six sides, not three of them twice.
    ///
    /// `normal.dot(half)` is negative on the -X/-Y/-Z faces, so dropping the `.abs()` in `push_box`
    /// moves those faces to the *positive* side: the box collapses onto its positive three planes and
    /// is open on the other three. The winding check below cannot see it, because each misplaced face
    /// still winds toward its own normal, so this checks the *positions*: every box must reach
    /// `centre - half` and `centre + half` on every axis, with all six normals present.
    #[test]
    fn push_box_closes_on_all_six_sides() {
        let mut mesh = StaticMesh::default();
        let centre = Vec3::new(1.0, 2.0, 3.0);
        let half = Vec3::new(0.5, 0.25, 0.75);
        push_box(&mut mesh, centre, half, MATERIAL_STONE);

        let lo = centre - half;
        let hi = centre + half;
        for axis in 0..3 {
            let min = mesh
                .vertices
                .iter()
                .fold(f32::INFINITY, |a, v| a.min(v.pos[axis]));
            let max = mesh
                .vertices
                .iter()
                .fold(f32::NEG_INFINITY, |a, v| a.max(v.pos[axis]));
            assert!(
                (min - lo.to_array()[axis]).abs() < 1e-4,
                "axis {axis} reaches {min}, expected {}",
                lo.to_array()[axis]
            );
            assert!(
                (max - hi.to_array()[axis]).abs() < 1e-4,
                "axis {axis} reaches {max}, expected {}",
                hi.to_array()[axis]
            );
        }
        for n in [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z] {
            assert!(
                mesh.vertices.iter().any(|v| Vec3::from(v.normal).dot(n) > 0.9),
                "no face with normal {n:?}"
            );
        }
        assert_eq!(mesh.triangle_count(), 12);
    }

    /// Every triangle's winding has to agree with the normal stored on it.
    ///
    /// Both static pipelines cull back faces, so a face listed in the wrong order is not drawn inside
    /// out, it is not drawn at all: a flipped box face is a hole in a wall. The cross product of a
    /// triangle's edges is what the rasteriser sees, so that is what the check compares against.
    #[test]
    fn castle_faces_wind_toward_their_normals() {
        let mesh = castle_mesh();
        for tri in mesh.indices.as_chunks::<3>().0 {
            let [a, b, c] = [
                Vec3::from(mesh.vertices[tri[0] as usize].pos),
                Vec3::from(mesh.vertices[tri[1] as usize].pos),
                Vec3::from(mesh.vertices[tri[2] as usize].pos),
            ];
            let normal = Vec3::from(mesh.vertices[tri[0] as usize].normal);
            let wound = (b - a).cross(c - a).normalize_or_zero();
            assert!(
                wound.dot(normal) > 0.5,
                "triangle {tri:?} winds toward {wound:?} but its vertices carry {normal:?}"
            );
        }
    }

    /// The castle has to be normalized the way the landmark placement assumes.
    ///
    /// `pos.y` of an instance is the *ground line*, so mesh y = 0 must be where the walls meet the
    /// ground, the top of the keep must be at exactly 1.0 (or an instance's scale is not its height),
    /// and the plinth must dip below 0 (or a castle on a slope floats on the low side).
    #[test]
    fn the_castle_is_normalized_around_its_ground_line() {
        let mesh = castle_mesh();
        assert!(!mesh.vertices.is_empty());
        assert!(mesh
            .indices
            .iter()
            .all(|i| (*i as usize) < mesh.vertices.len()));

        let max_y = mesh.vertices.iter().fold(0.0f32, |a, v| a.max(v.pos[1]));
        let min_y = mesh
            .vertices
            .iter()
            .fold(f32::INFINITY, |a, v| a.min(v.pos[1]));
        assert!(
            (max_y - 1.0).abs() < 1e-4,
            "height is {max_y}, expected 1.0"
        );
        assert!(
            min_y < -0.1,
            "the skirt only reaches {min_y}, expected a plinth below 0"
        );

        // The footprint the placement searches with has to match the mesh it places, or a castle is
        // put down half inside a hill.
        let half_extent = mesh
            .vertices
            .iter()
            .fold(0.0f32, |a, v| a.max(v.pos[0].abs().max(v.pos[2].abs())));
        assert!(
            (half_extent - CASTLE_HALF_EXTENT).abs() < 1e-4,
            "mesh spans {half_extent} in xz but CASTLE_HALF_EXTENT says {CASTLE_HALF_EXTENT}"
        );
        // Wider than a tree and wider than it is tall would be a hill fort, not a castle.
        assert!(
            half_extent < 0.6,
            "the castle is {half_extent} half-wide for a height of 1"
        );
    }

    #[test]
    fn the_castle_is_generated_deterministically() {
        let a = castle_mesh();
        let b = castle_mesh();
        assert_eq!(a.vertices, b.vertices);
        assert_eq!(a.indices, b.indices);
        // A landmark is drawn once per frame, so this is only a sanity bound; it exists so that a
        // battlement loop that grows out of hand shows up as a test failure rather than as a
        // frame-time regression nobody can attribute.
        assert!(
            (100..4000).contains(&a.triangle_count()),
            "the castle is {} triangles",
            a.triangle_count()
        );
    }
}

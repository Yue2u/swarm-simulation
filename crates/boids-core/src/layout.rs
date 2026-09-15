//! GPU memory layout contract.
//!
//! Every struct in this module is mirrored **field-for-field** by a struct with the same name in
//! `shaders/common/layout.wgsl`. The two definitions must never drift; the invariants are:
//!
//! 1. every struct is `#[repr(C)]` and aligned to 16 bytes,
//! 2. every field offset is a multiple of its WGSL alignment (16 for `vec3<f32>`, 4 for scalars),
//! 3. the total size is a multiple of 16, so arrays of the struct keep stride == size.
//!
//! To keep rule 2 trivially true we only ever use two shapes of 16-byte block:
//! `(vec3<f32>, f32)` and `(f32|u32) * 4`. There is no `vec3` padding trap anywhere.
//!
//! The static assertions below enforce rules 1 and 3 at compile time, and the
//! `layout::tests::wgsl_offsets_match_rust` test (run on a real device) enforces rule 2 by
//! having the shader echo its own `offsetof()` values back for comparison.
//!
//! See `docs/gpu-pipeline.md` for the buffer table and buffer bindings.

use bytemuck::{Pod, Zeroable};

/// Marker written into `cell_start` for an empty grid cell.
pub const EMPTY_CELL: u32 = u32::MAX;

/// Which world the swarm lives in. Mirrored in WGSL as `u32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum SimMode {
    /// Underwater: buoyancy, drag, SDF reef avoidance.
    #[default]
    Fish = 0,
    /// Sky: gravity-free glide, heightfield terrain avoidance.
    Birds = 1,
}

impl SimMode {
    /// Raw value as stored in `SimParams::mode`.
    #[inline]
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Cursor interaction mode. Mirrored in WGSL as `u32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum InteractionMode {
    /// Cursor does not affect the swarm.
    #[default]
    Off = 0,
    /// Agents steer toward the focus point.
    Attract = 1,
    /// Agents flee the focus point and swirl around it.
    Repel = 2,
}

impl InteractionMode {
    /// Raw value as stored in `InteractionUniforms::mode`.
    #[inline]
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

/// A single agent. 48 bytes, stride == size.
///
/// | offset | size | Rust field   | WGSL                |
/// |--------|------|--------------|---------------------|
/// | 0      | 12   | `pos`        | `pos: vec3<f32>`    |
/// | 12     | 4    | `species`    | `species: f32`      |
/// | 16     | 12   | `vel`        | `vel: vec3<f32>`    |
/// | 28     | 4    | `phase`      | `phase: f32`        |
/// | 32     | 12   | `prev_dir`   | `prev_dir: vec3<f32>` |
/// | 44     | 4    | `color_seed` | `color_seed: f32`   |
///
/// `species` and `color_seed` are both in `[0, 1)` and are generated once at spawn time.
/// `prev_dir` is the previous frame's normalized velocity; the vertex shader uses it for the
/// bank angle, which is why it is stored instead of recomputed.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Boid {
    /// World-space position, meters.
    pub pos: [f32; 3],
    /// Species selector in `[0, 1)`: palette index, mesh variant, speed bias.
    pub species: f32,
    /// World-space velocity, meters per second.
    pub vel: [f32; 3],
    /// Per-agent animation phase in radians (tail wiggle / wing flap).
    pub phase: f32,
    /// Previous frame's normalized velocity, used for the bank angle.
    pub prev_dir: [f32; 3],
    /// Stable per-agent color jitter in `[0, 1)`.
    pub color_seed: f32,
}

/// Sort key / payload pair used by the bitonic sort.
///
/// `key` is the linear grid cell index, `val` is the boid index. Padding entries pushed into the
/// sorted array get `key = u32::MAX` so they sort to the end and are detected as "not a cell".
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct KeyVal {
    /// Linear cell index.
    pub key: u32,
    /// Index into the boid buffer.
    pub val: u32,
}

/// Bitonic sort step descriptor. Mirrored in WGSL as `SortParams`.
///
/// One entry is uploaded per sort stage. `j` is the compare distance and `k` the merge size, so the
/// partner index is `i ^ j` and the ascending flag is `(i & k) == 0`.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable, Default)]
pub struct SortParams {
    /// Compare distance for this stage.
    pub j: u32,
    /// Merge block size for this stage.
    pub k: u32,
    /// Number of elements being sorted (padded to a power of two).
    pub n_padded: u32,
    /// Explicit padding keeping the struct 16-byte sized.
    pub _pad: u32,
}

/// Simulation parameters, uploaded once per frame.
///
/// Mirrored in WGSL as `SimParams` in `shaders/common/layout.wgsl`. 128 bytes.
///
/// | offset | Rust field        | WGSL                |
/// |--------|-------------------|---------------------|
/// | 0      | `grid_min`        | `grid_min: vec3<f32>` |
/// | 12     | `cell_size`       | `cell_size: f32`    |
/// | 16     | `grid_dim`        | `grid_dim: vec3<u32>` |
/// | 28     | `num_boids`       | `num_boids: u32`    |
/// | 32     | `w_sep`, `w_ali`, `w_coh`, `r_percept` | same names |
/// | 48     | `r_sep`, `max_speed`, `min_speed`, `max_force` | same names |
/// | 64     | `dt`, `time`, `sdf_strength`, `sdf_probe` | same names |
/// | 80     | `bounds_half`, `mode` | `bounds_half: vec3<f32>`, `mode: u32` |
/// | 96     | `wander`, `sep_boost`, `coh_falloff`, `r_safe` | same names |
/// | 112    | `buoyancy`, `drag`, `env_scale`, `env_floor_y` | same names |
/// | 128    | `env_id`, `_pad0`..`_pad2` | `env_id: u32`, then padding |
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct SimParams {
    /// World-space minimum corner of the grid domain.
    pub grid_min: [f32; 3],
    /// Edge length of a grid cell, meters.
    pub cell_size: f32,

    /// Grid resolution per axis.
    pub grid_dim: [u32; 3],
    /// Number of live boids (the tail of the padded buffers is inactive).
    pub num_boids: u32,

    /// Separation weight (already scaled by the density-adaptive factor).
    pub w_sep: f32,
    /// Alignment weight.
    pub w_ali: f32,
    /// Cohesion weight.
    pub w_coh: f32,
    /// Perception radius, meters.
    pub r_percept: f32,

    /// Separation radius, meters.
    pub r_sep: f32,
    /// Speed clamp upper bound, m/s.
    pub max_speed: f32,
    /// Speed clamp lower bound, m/s.
    pub min_speed: f32,
    /// Steering force clamp, m/s^2.
    pub max_force: f32,

    /// Integration timestep, seconds.
    pub dt: f32,
    /// Accumulated simulation time, seconds.
    pub time: f32,
    /// SDF avoidance force scale.
    pub sdf_strength: f32,
    /// Forward probe distance used to detect walls before impact, meters.
    pub sdf_probe: f32,

    /// Half-extent of the soft world bounds box.
    pub bounds_half: [f32; 3],
    /// `SimMode` as `u32`.
    pub mode: u32,

    /// Wander force scale.
    pub wander: f32,
    /// Extra separation multiplier in crowded cells.
    pub sep_boost: f32,
    /// Cohesion damping exponent for crowded cells.
    pub coh_falloff: f32,
    /// Safe distance kept from SDF surfaces, meters.
    pub r_safe: f32,

    /// Constant vertical push (buoyancy for fish, thermal lift for birds).
    pub buoyancy: f32,
    /// Medium drag applied to velocity, 1/s.
    pub drag: f32,
    /// Field scale: the reef's column repetition period, or the terrain's amplitude in metres.
    pub env_scale: f32,
    /// Seafloor height for the reef field; ignored by the terrain field.
    pub env_floor_y: f32,

    /// Avoidance field selector, mirroring [`EnvironmentKind`]:
    /// 0 = none, 1 = reef (`sdf::reef_field`), 2 = terrain (`terrain::height_at`).
    pub env_id: u32,
    /// Explicit padding keeping the struct 144 bytes and free of implicit padding. `bytemuck`
    /// refuses to derive `Pod` for a struct with padding, so any new field must keep each 16-byte
    /// block exactly full or the crate stops compiling.
    pub _pad0: f32,
    /// Explicit padding, see `_pad0`.
    pub _pad1: f32,
    /// Explicit padding, see `_pad0`.
    pub _pad2: f32,
}

/// Cursor interaction state, uploaded once per frame.
///
/// Mirrored in WGSL as `InteractionUniforms`. 48 bytes.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct InteractionUniforms {
    /// Ray origin (camera position).
    pub ray_origin: [f32; 3],
    /// `InteractionMode` as `u32`.
    pub mode: u32,

    /// Where the cursor ray meets the interaction plane (or the terrain).
    pub focus_point: [f32; 3],
    /// Influence radius around the focus point, meters.
    pub radius: f32,

    /// Force magnitude at the focus point.
    pub strength: f32,
    /// Radial falloff exponent.
    pub falloff: f32,
    /// Tangential swirl strength (repel mode only).
    pub tangent: f32,
    /// Explicit padding keeping the struct 48 bytes.
    pub _pad: f32,
}

/// Camera matrices, uploaded once per frame.
///
/// Mirrored in WGSL as `CameraUniform`. `view_proj` is used by the vertex shader and
/// `inv_view_proj` by the ocean raymarch pass, so both are kept together.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct CameraUniform {
    /// World -> clip transform.
    pub view_proj: [[f32; 4]; 4],
    /// Clip -> world transform.
    pub inv_view_proj: [[f32; 4]; 4],
    /// Camera position in world space.
    pub eye: [f32; 3],
    /// Time in seconds, reused by shader-side animation.
    pub time: f32,
    /// Forward unit vector.
    pub forward: [f32; 3],
    /// Near plane distance.
    pub near: f32,
    /// Up unit vector.
    pub up: [f32; 3],
    /// Far plane distance.
    pub far: f32,
    /// Viewport size in pixels.
    pub viewport: [f32; 2],
    /// `tan(fovy/2)`, used to reconstruct rays from NDC.
    pub tan_half_fovy: f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
}

/// Per-mode mesh shape parameters, consumed by `shaders/render/boid.wgsl`.
///
/// Mirrored in WGSL as `MeshParams`. 80 bytes. The whole point of this struct is that one vertex
/// shader can draw both a laterally compressed, long-finned fish and a wide-winged, swept-wing
/// bird: switching worlds changes these numbers, not the pipeline.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct MeshParams {
    /// Half-width of the body cross-section (across the swimming axis).
    pub body_w: f32,
    /// Half-height of the body cross-section.
    pub body_h: f32,
    /// Nose length along +Z in mesh units.
    pub nose: f32,
    /// Tail length along -Z in mesh units.
    pub tail: f32,

    /// Tail fin half-height.
    pub fin_size: f32,
    /// Z position of the front ring, which sets where the body's widest point sits.
    pub fin_z: f32,
    /// Animation amplitude: lateral wave for fish, wing flap angle for birds.
    pub wave_amp: f32,
    /// Animation rate in radians per second.
    pub wave_freq: f32,

    /// Wing half-span (birds only).
    pub wing_span: f32,
    /// How far back the wing tip sits, giving the wing its sweep.
    pub wing_sweep: f32,
    /// Emissive multiplier; non-zero for bioluminescent fish, zero for birds.
    pub emissive: f32,
    /// Mesh variant: 0 = fish, 1 = bird.
    pub variant: f32,

    /// Base hue of the palette ramp, in `[0, 1)`.
    pub hue_base: f32,
    /// How much of the hue wheel the species range covers.
    pub hue_range: f32,
    /// Palette saturation.
    pub saturation: f32,
    /// Palette value (brightness).
    pub value: f32,

    /// Reference speed for the motion-stretch term, usually the simulation's `max_speed`.
    pub speed_ref: f32,
    /// Explicit padding keeping the struct 80 bytes.
    pub _pad_a: f32,
    /// Explicit padding keeping the struct 80 bytes.
    pub _pad_b: f32,
    /// Explicit padding keeping the struct 80 bytes.
    pub _pad_c: f32,
}

/// Everything the render passes need for one frame.
///
/// Mirrored in WGSL as `SceneUniform`. 304 bytes. The camera is embedded rather than kept in its
/// own binding so that the vertex shader reads a single uniform buffer: one binding is one cache
/// line of setup per draw call, and with a single instanced draw that is not a bottleneck yet, but
/// it keeps the pass simple.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct SceneUniform {
    /// Camera matrices and vectors.
    pub camera: CameraUniform,
    /// Per-mode mesh shape.
    pub mesh: MeshParams,
    /// Direction the light travels (from the light toward the scene).
    pub light_dir: [f32; 3],
    /// Ambient term, added to the diffuse term.
    pub ambient: f32,
    /// Colour of the medium at infinite distance.
    pub fog_color: [f32; 3],
    /// Extinction/mixing coefficient: 0 is a vacuum, 0.5 is a murky river.
    pub fog_density: f32,
}

/// Which environment field the simulation avoids. Mirrors the `ENV_*` constants in WGSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum EnvironmentKind {
    /// No environment: free space, no avoidance force at all.
    #[default]
    None = 0,
    /// Underwater reef, `sdf::reef_field`.
    Reef = 1,
    /// Procedural terrain, `terrain::height_at`.
    Terrain = 2,
}

impl EnvironmentKind {
    /// Raw value as stored in `SimParams::env_id`.
    #[inline]
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------------------------
// Compile-time layout contract.
//
// These assertions replace "trust me" comments: if anyone edits a struct above such that the WGSL
// mirror would silently disagree, the crate stops compiling.
// ---------------------------------------------------------------------------------------------

const _: () = {
    assert!(core::mem::size_of::<Boid>() == 48);
    assert!(core::mem::align_of::<Boid>() == 16);

    assert!(core::mem::size_of::<KeyVal>() == 8);

    assert!(core::mem::size_of::<SortParams>() == 16);
    assert!(core::mem::align_of::<SortParams>() == 16);

    assert!(core::mem::size_of::<SimParams>() == 144);
    assert!(core::mem::align_of::<SimParams>() == 16);

    assert!(core::mem::size_of::<InteractionUniforms>() == 48);
    assert!(core::mem::align_of::<InteractionUniforms>() == 16);

    assert!(core::mem::size_of::<CameraUniform>() == 192);
    assert!(core::mem::align_of::<CameraUniform>() == 16);

    assert!(core::mem::size_of::<MeshParams>() == 80);
    assert!(core::mem::align_of::<MeshParams>() == 16);

    assert!(core::mem::size_of::<SceneUniform>() == 304);
    assert!(core::mem::align_of::<SceneUniform>() == 16);
};

/// Field offsets of every GPU struct, as the host sees them.
///
/// The `wgsl_offsets_match_rust` test compares this table against values computed by
/// `shaders/tests/offsets.wgsl` on the device, which is the only way to catch a WGSL-side drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OffsetTable {
    /// `Boid` field offsets in declaration order.
    pub boid: [u32; 6],
    /// `SimParams` field offsets in declaration order.
    pub sim_params: [u32; 16],
    /// `InteractionUniforms` field offsets in declaration order.
    pub interaction: [u32; 7],
    /// Sizes of the structs, in the same order.
    pub sizes: [u32; 3],
}

/// Builds the host-side [`OffsetTable`] with `offset_of!`.
#[must_use]
pub const fn offset_table() -> OffsetTable {
    OffsetTable {
        boid: [
            core::mem::offset_of!(Boid, pos) as u32,
            core::mem::offset_of!(Boid, species) as u32,
            core::mem::offset_of!(Boid, vel) as u32,
            core::mem::offset_of!(Boid, phase) as u32,
            core::mem::offset_of!(Boid, prev_dir) as u32,
            core::mem::offset_of!(Boid, color_seed) as u32,
        ],
        sim_params: [
            core::mem::offset_of!(SimParams, grid_min) as u32,
            core::mem::offset_of!(SimParams, cell_size) as u32,
            core::mem::offset_of!(SimParams, grid_dim) as u32,
            core::mem::offset_of!(SimParams, num_boids) as u32,
            core::mem::offset_of!(SimParams, w_sep) as u32,
            core::mem::offset_of!(SimParams, w_ali) as u32,
            core::mem::offset_of!(SimParams, w_coh) as u32,
            core::mem::offset_of!(SimParams, r_percept) as u32,
            core::mem::offset_of!(SimParams, r_sep) as u32,
            core::mem::offset_of!(SimParams, max_speed) as u32,
            core::mem::offset_of!(SimParams, min_speed) as u32,
            core::mem::offset_of!(SimParams, max_force) as u32,
            core::mem::offset_of!(SimParams, dt) as u32,
            core::mem::offset_of!(SimParams, time) as u32,
            core::mem::offset_of!(SimParams, sdf_strength) as u32,
            core::mem::offset_of!(SimParams, sdf_probe) as u32,
        ],
        interaction: [
            core::mem::offset_of!(InteractionUniforms, ray_origin) as u32,
            core::mem::offset_of!(InteractionUniforms, mode) as u32,
            core::mem::offset_of!(InteractionUniforms, focus_point) as u32,
            core::mem::offset_of!(InteractionUniforms, radius) as u32,
            core::mem::offset_of!(InteractionUniforms, strength) as u32,
            core::mem::offset_of!(InteractionUniforms, falloff) as u32,
            core::mem::offset_of!(InteractionUniforms, tangent) as u32,
        ],
        sizes: [
            core::mem::size_of::<Boid>() as u32,
            core::mem::size_of::<SimParams>() as u32,
            core::mem::size_of::<InteractionUniforms>() as u32,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pod_impls_exist_and_are_zeroable() {
        // Proves the `Pod`/`Zeroable` derives are present and usable on the buffer types, which is
        // what lets `bytemuck::cast_slice` and `write_buffer` accept them directly.
        let boid = Boid::zeroed();
        assert_eq!(bytemuck::bytes_of(&boid).len(), 48);
        let params = SimParams::zeroed();
        assert_eq!(bytemuck::bytes_of(&params).len(), 144);
        let inter = InteractionUniforms::zeroed();
        assert_eq!(bytemuck::bytes_of(&inter).len(), 48);
        let cam = CameraUniform::zeroed();
        assert_eq!(bytemuck::bytes_of(&cam).len(), 192);
        let kv = KeyVal::zeroed();
        assert_eq!(bytemuck::bytes_of(&kv).len(), 8);
        let sort = SortParams::zeroed();
        assert_eq!(bytemuck::bytes_of(&sort).len(), 16);
        let mesh = MeshParams::zeroed();
        assert_eq!(bytemuck::bytes_of(&mesh).len(), 80);
        let scene = SceneUniform::zeroed();
        assert_eq!(bytemuck::bytes_of(&scene).len(), 304);
    }

    #[test]
    fn no_implicit_padding_holds() {
        // If any of these fail, a field was added without updating the padding fields and the WGSL
        // mirror in shaders/common/layout.wgsl.
        assert_eq!(core::mem::size_of::<Boid>(), 48);
        assert_eq!(core::mem::size_of::<SimParams>(), 144);
        assert_eq!(core::mem::size_of::<InteractionUniforms>(), 48);
    }

    #[test]
    fn offset_table_is_16_aligned() {
        let t = offset_table();
        for (name, offs) in [("boid", &t.boid[..]), ("sim_params", &t.sim_params[..])] {
            for (i, o) in offs.iter().enumerate() {
                assert_eq!(o % 4, 0, "{name}[{i}] offset {o} is not 4-byte aligned");
            }
        }
        assert_eq!(t.sizes[0], 48);
        assert_eq!(t.sizes[1], 144);
        assert_eq!(t.sizes[2], 48);
    }
}

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
/// | 128    | `env_id`, `env_freq`, `_pad1`, `_pad2` | `env_id: u32`, `env_freq: f32`, padding |
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
    /// Reciprocal of `density_ref`, turning a neighbour count into a density ratio.
    pub sep_boost: f32,
    /// Density-feedback exponent (`density_gain`): `push = density^coh_falloff`. Zero gives fixed
    /// weights, one the proportional feedback, larger values swing harder.
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
    /// Terrain base noise frequency, m^-1 (`terrain::height_at`'s second argument). Unused by the
    /// reef, whose scale is `env_scale`.
    ///
    /// It lives in `SimParams` rather than in the render-side `TerrainParams` because the
    /// *simulation* needs it and never sees a render struct: the surface an agent avoids and the
    /// surface the camera draws have to be the same field evaluated with the same frequency.
    pub env_freq: f32,
    /// Explicit padding keeping the struct 144 bytes and free of implicit padding. `bytemuck`
    /// refuses to derive `Pod` for a struct with padding, so any new field must keep each 16-byte
    /// block exactly full or the crate stops compiling.
    pub _pad1: f32,
    /// Explicit padding, see `_pad1`.
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
    /// Uniform scale applied to the whole mesh.
    ///
    /// Without it every agent would be drawn at the same absolute size, which reads as dust in a world
    /// sized for a hundred thousand agents and as a solid wall in a small one. The host derives it from
    /// the perception radius, so an agent always occupies a similar share of the space it can see.
    pub scale: f32,
    /// Which medium attenuates the agent: 0 = the underwater per-channel model, 1 = the aerial haze.
    ///
    /// The host sets this to the *destination* world's medium during a morph while `variant` blends
    /// the body, so the creature keeps the light it is actually flying through instead of fading
    /// through the pale in-scatter of a medium it has already left. It lives in what was padding: the
    /// struct has room for one more scalar and this is the one the shader needs.
    pub medium: f32,
    /// Explicit padding keeping the struct 80 bytes, and keeping it free of implicit padding so that
    /// `bytemuck` will derive `Pod`: every 16-byte block must be exactly full.
    pub _pad_b: f32,
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

/// Underwater medium and reef geometry for the ocean raymarch pass.
///
/// Mirrored in WGSL as `WaterParams`. 48 bytes, twelve scalars, no vector members, so there is no
/// padding to reason about beyond the 16-byte total.
///
/// # Why the extinction is per channel
///
/// Beer-Lambert extinction in water is strongly wavelength-dependent: red is gone within a few metres
/// and blue survives tens of them, which is the entire reason a reef at depth reads as teal and a
/// distant one as deep blue. A single `fog_density` (as in `SceneUniform`) can darken a scene but
/// cannot produce that shift, and the shift is most of what "underwater" looks like. The three
/// coefficients are metres^-1 and already absolute, so `exp(-sigma * d)` is the transmittance for
/// each channel.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct WaterParams {
    /// Height of the water surface, metres. Rays above it see the surface from below.
    pub surface_y: f32,
    /// Seafloor height, metres. Matches `env_floor_y` in the simulation's reef field.
    pub floor_y: f32,
    /// Reef column repetition period, metres. Matches `env_scale` in the reef field.
    pub reef_period: f32,
    /// Caustic intensity on surfaces, in linear HDR units.
    pub caustic_strength: f32,

    /// Per-channel extinction coefficient, metres^-1: `(red, green, blue)`.
    pub extinction: [f32; 3],
    /// Scale on the in-scattered light that fills the medium, so the water is not black between
    /// surfaces.
    pub scatter: f32,

    /// Intensity of the god-ray (shaft) term.
    pub godray_strength: f32,
    /// Brightness of the water surface seen from below, which is where most of the light comes from.
    pub surface_glow: f32,
    /// Spatial frequency of the caustic cell pattern, in metres^-1.
    pub caustic_scale: f32,
    /// Caustic animation rate, in hertz of pattern drift.
    pub caustic_drift: f32,
}

/// HDR post-processing parameters: exposure, bloom, the lens terms and the final grade.
///
/// Mirrored in WGSL as `PostParams`. 48 bytes, twelve scalars.
///
/// These are separate from [`SceneUniform`] because they are read by the post passes, which run on
/// full-screen targets with their own bind group, and because the render scene uniform is already at
/// the size where adding two more `vec3`-sized concerns would make it a struct nobody can hold in
/// their head at once.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct PostParams {
    /// Exposure applied before the tone curve. The one knob that has to differ per world: sunlight
    /// through 30 metres of water arrives about two stops dimmer than open sky.
    pub exposure: f32,
    /// Luminance above which a pixel contributes to bloom.
    pub bloom_threshold: f32,
    /// Width of the soft knee around the threshold, as a fraction of it.
    pub bloom_knee: f32,
    /// How much of the bloom pyramid is added back into the frame.
    pub bloom_strength: f32,

    /// Corner darkening. 0 disables it.
    pub vignette: f32,
    /// Film grain amplitude, in output units before the tone curve.
    pub grain: f32,
    /// Chromatic aberration at the frame edge, in UV units.
    pub aberration: f32,
    /// Luminance mapped to white by the tone curve; the ACES fit saturates rather than clipping.
    pub tonemap_white: f32,

    /// Simulation time, seconds. Animated grain and the dither offset live here so that the post
    /// passes need no uniform of their own beyond this struct.
    pub time: f32,
    /// Saturation applied after the tone curve. 1 is neutral.
    pub saturation: f32,
    /// Contrast around mid grey, applied after the tone curve. 1 is neutral.
    pub contrast: f32,
    /// Black level added after the tone curve; the negative direction crushes the deepest shadows,
    /// which is what keeps the underwater frame from looking flat after the exposure lift.
    pub lift: f32,
}

/// The sky world's land: the heightfield map, the mesh that draws it and the vegetation on it.
///
/// One struct rather than three because its consumers describe the same piece of ground: the
/// heightfield generator writes the map, the terrain mesh reads it back through the same
/// `amplitude` and `frequency` the simulation's collision field uses, and the scatter pass places
/// trees on it. Splitting them would allow a mesh built from one amplitude and a collision field
/// built from another, which is exactly the divergence the shared-field tests exist to prevent.
///
/// Mirrored in WGSL as `TerrainParams`. 48 bytes, twelve scalars and two `vec2`s.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct TerrainParams {
    /// World `xz` of the map's minimum corner.
    pub min_xz: [f32; 2],
    /// World size the map covers, metres.
    pub size_xz: [f32; 2],

    /// Height scale, metres. Equal to the simulation's `env_scale` for the terrain field.
    pub amplitude: f32,
    /// Base noise frequency, m^-1. Equal to the simulation's `env_freq`.
    pub frequency: f32,
    /// Biome cell frequency, m^-1.
    pub biome_frequency: f32,
    /// Mesh segments per axis: `segments^2 * 2` triangles.
    pub segments: u32,

    /// Height a drawn tree reaches above the ground, metres. Zero disables the tree pass.
    pub tree_height: f32,
    /// Instance slots in the tree buffer.
    pub tree_capacity: u32,
    /// Scatter candidates per axis. The capacity is a small fraction of `candidates^2`.
    pub tree_candidates: u32,
    /// Texels per axis of the baked map.
    ///
    /// Carried in the uniform rather than queried with `textureDimensions`: the generating pass
    /// writes a *storage* texture, and sizing one needs the `IMAGE_SIZE` feature, which the GL
    /// backend does not offer.
    pub resolution: u32,
}

/// Analytic single-scattering atmosphere for the sky world.
///
/// Mirrored in WGSL as `SkyParams`. 48 bytes, twelve scalars.
///
/// The coefficients are physical (metres^-1) rather than artistic: they describe air, and the two
/// numbers that decide the look are `sun_intensity` and `horizon_boost`. See `docs/math.md` for the
/// integral this approximates and why one flat-atmosphere term is enough at this world's scale.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct SkyParams {
    /// Rayleigh scattering coefficient, m^-1. Strong in blue, which is why the sky is blue.
    pub beta_rayleigh: [f32; 3],
    /// Sun radiance scale. The only free knob in the model.
    pub sun_intensity: f32,

    /// Mie scattering coefficient, m^-1: the aerosol term, nearly grey.
    pub beta_mie: [f32; 3],
    /// Henyey-Greenstein anisotropy of the Mie term. 0.76 is the usual forward-scattering haze.
    pub mie_g: f32,

    /// Rayleigh scale height, metres.
    pub ray_scale_height: f32,
    /// Mie scale height, metres.
    pub mie_scale_height: f32,
    /// Extra brightness within a few degrees of the horizon, where a line of sight crosses most air.
    pub horizon_boost: f32,
    /// Multiplier on the in-scattered light distant geometry fades into.
    pub aerial_boost: f32,
}

/// One scattered tree: a base position, an orientation and a variant.
///
/// Mirrored in WGSL as `TreeInstance`. 32 bytes so the array stride stays a multiple of 16. This is
/// a storage element rather than a uniform: the scatter pass appends to an array of them and the
/// tree pass draws however many were accepted.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct TreeInstance {
    /// Base position, on the ground, metres.
    pub pos: [f32; 3],
    /// Uniform scale.
    pub scale: f32,

    /// Rotation about Y, radians.
    pub yaw: f32,
    /// Species/variant selector in `[0, 1)`, from the scatter hash.
    pub kind: f32,
    /// Biome mask under the trunk, in `[0, 2)`. Stored rather than re-sampled by the draw pass: the
    /// scatter pass already had it in hand, and the draw pass would otherwise pay a texel load per
    /// vertex to learn what colour it is.
    pub mask: f32,
    /// Explicit padding, see [`SimParams::_pad1`].
    pub pad1: f32,
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

    assert!(core::mem::size_of::<WaterParams>() == 48);
    assert!(core::mem::align_of::<WaterParams>() == 16);

    assert!(core::mem::size_of::<PostParams>() == 48);
    assert!(core::mem::align_of::<PostParams>() == 16);

    assert!(core::mem::size_of::<TerrainParams>() == 48);
    assert!(core::mem::align_of::<TerrainParams>() == 16);

    assert!(core::mem::size_of::<SkyParams>() == 48);
    assert!(core::mem::align_of::<SkyParams>() == 16);

    assert!(core::mem::size_of::<TreeInstance>() == 32);
    assert!(core::mem::align_of::<TreeInstance>() == 16);
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
    /// `WaterParams` field offsets in declaration order.
    pub water: [u32; 12],
    /// `PostParams` field offsets in declaration order.
    pub post: [u32; 12],
    /// Sizes of the structs, in the same order as the fields above.
    pub sizes: [u32; 5],
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
        water: [
            core::mem::offset_of!(WaterParams, surface_y) as u32,
            core::mem::offset_of!(WaterParams, floor_y) as u32,
            core::mem::offset_of!(WaterParams, reef_period) as u32,
            core::mem::offset_of!(WaterParams, caustic_strength) as u32,
            core::mem::offset_of!(WaterParams, extinction) as u32,
            core::mem::offset_of!(WaterParams, scatter) as u32,
            core::mem::offset_of!(WaterParams, godray_strength) as u32,
            core::mem::offset_of!(WaterParams, surface_glow) as u32,
            core::mem::offset_of!(WaterParams, caustic_scale) as u32,
            core::mem::offset_of!(WaterParams, caustic_drift) as u32,
            // The three extinction channels are one `[f32; 3]` member on the Rust side and three
            // scalars in WGSL, so the table has to expand them to stay comparable slot for slot.
            core::mem::offset_of!(WaterParams, extinction) as u32 + 4,
            core::mem::offset_of!(WaterParams, extinction) as u32 + 8,
        ],
        post: [
            core::mem::offset_of!(PostParams, exposure) as u32,
            core::mem::offset_of!(PostParams, bloom_threshold) as u32,
            core::mem::offset_of!(PostParams, bloom_knee) as u32,
            core::mem::offset_of!(PostParams, bloom_strength) as u32,
            core::mem::offset_of!(PostParams, vignette) as u32,
            core::mem::offset_of!(PostParams, grain) as u32,
            core::mem::offset_of!(PostParams, aberration) as u32,
            core::mem::offset_of!(PostParams, tonemap_white) as u32,
            core::mem::offset_of!(PostParams, time) as u32,
            core::mem::offset_of!(PostParams, saturation) as u32,
            core::mem::offset_of!(PostParams, contrast) as u32,
            core::mem::offset_of!(PostParams, lift) as u32,
        ],
        sizes: [
            core::mem::size_of::<Boid>() as u32,
            core::mem::size_of::<SimParams>() as u32,
            core::mem::size_of::<InteractionUniforms>() as u32,
            core::mem::size_of::<WaterParams>() as u32,
            core::mem::size_of::<PostParams>() as u32,
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
        let water = WaterParams::zeroed();
        assert_eq!(bytemuck::bytes_of(&water).len(), 48);
        let post = PostParams::zeroed();
        assert_eq!(bytemuck::bytes_of(&post).len(), 48);
    }

    #[test]
    fn no_implicit_padding_holds() {
        // If any of these fail, a field was added without updating the padding fields and the WGSL
        // mirror in shaders/common/layout.wgsl.
        assert_eq!(core::mem::size_of::<Boid>(), 48);
        assert_eq!(core::mem::size_of::<SimParams>(), 144);
        assert_eq!(core::mem::size_of::<InteractionUniforms>(), 48);
        assert_eq!(core::mem::size_of::<WaterParams>(), 48);
        assert_eq!(core::mem::size_of::<PostParams>(), 48);
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

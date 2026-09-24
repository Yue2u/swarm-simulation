// === file: common/layout.wgsl ================================================================
// GPU memory layout contract, mirroring `crates/boids-core/src/layout.rs` field for field.
//
// OWNERSHIP: if you change a struct here you must change the Rust twin in the same commit, and
// the `layout_probe` test (`shaders/tests/layout_probe.wgsl`) will fail on the device if the two
// disagree. That test is the only thing standing between you and a silently corrupted simulation.
//
// LAYOUT RULES (enforced by construction):
//   * every member is 4, 8 or 16-byte aligned and no member is misaligned,
//   * `vec3<f32>` always has alignment 16 and size 12, so each `vec3` is followed by exactly one
//     `f32` scalar to fill its 16-byte block. There is no implicit padding anywhere,
//   * every struct size is a multiple of 16.
//
// See docs/gpu-pipeline.md for the buffer table and binding assignments.

struct Boid {
    // byte 0
    pos: vec3<f32>,
    // byte 12
    species: f32,
    // byte 16
    vel: vec3<f32>,
    // byte 28
    phase: f32,
    // byte 32
    prev_dir: vec3<f32>,
    // byte 44
    color_seed: f32,
    // byte 48 == size
}

struct SimParams {
    // 0
    grid_min: vec3<f32>,
    cell_size: f32,
    // 16
    grid_dim: vec3<u32>,
    num_boids: u32,
    // 32
    w_sep: f32,
    w_ali: f32,
    w_coh: f32,
    r_percept: f32,
    // 48
    r_sep: f32,
    max_speed: f32,
    min_speed: f32,
    max_force: f32,
    // 64
    dt: f32,
    time: f32,
    sdf_strength: f32,
    sdf_probe: f32,
    // 80
    bounds_half: vec3<f32>,
    mode: u32,
    // 96
    wander: f32,
    sep_boost: f32,
    coh_falloff: f32,
    r_safe: f32,
    // 112
    buoyancy: f32,
    drag: f32,
    env_scale: f32,
    env_floor_y: f32,
    // 128
    env_id: u32,
    // Terrain base noise frequency, m^-1. Unused by the reef, whose scale is `env_scale` (its
    // repetition period). Lives here rather than in `TerrainParams` because the simulation's
    // collision field needs it and the simulation never sees the render-side structs.
    env_freq: f32,
    pad1: f32,
    pad2: f32,
    // 144 == size
}

struct InteractionUniforms {
    // 0
    ray_origin: vec3<f32>,
    mode: u32,
    // 16
    focus_point: vec3<f32>,
    radius: f32,
    // 32
    strength: f32,
    falloff: f32,
    tangent: f32,
    pad: f32,
    // 48 == size
}

struct CameraUniform {
    // 0
    view_proj: mat4x4<f32>,
    // 64
    inv_view_proj: mat4x4<f32>,
    // 128
    eye: vec3<f32>,
    time: f32,
    // 144
    forward: vec3<f32>,
    near: f32,
    // 160
    up: vec3<f32>,
    far: f32,
    // 176
    viewport: vec2<f32>,
    tan_half_fovy: f32,
    aspect: f32,
    // 192 == size
}

// Per-mode mesh shape parameters. Produced on the host from `boids_gpu::mesh_profile` so that a
// single vertex shader can draw both a laterally compressed fish and a wide-winged bird.
struct MeshParams {
    // 0
    body_w: f32,
    body_h: f32,
    nose: f32,
    tail: f32,
    // 16
    fin_size: f32,
    fin_z: f32,
    wave_amp: f32,
    wave_freq: f32,
    // 32
    wing_span: f32,
    wing_sweep: f32,
    emissive: f32,
    variant: f32,
    // 48
    hue_base: f32,
    hue_range: f32,
    saturation: f32,
    value: f32,
    // 64
    speed_ref: f32,
    // Uniform scale applied to the whole mesh. Without it every agent would be drawn at the same
    // absolute size (~1.5 world units), which is invisible dust in a world sized for 100k agents and
    // an impenetrable wall in a small one. The host sets this to a fraction of the perception radius so
    // that an agent always occupies a similar share of the space it can see, which is what keeps the
    // swarm readable across both worlds and both scales.
    scale: f32,
    // Which medium attenuates the agent: 0 = the underwater per-channel model, 1 = the aerial haze.
    // Set from the destination world during a morph so the creature keeps the light it is flying
    // through; `variant` blends the body independently of this.
    medium: f32,
    pad_b: f32,
    // 80 == size
}

struct SceneUniform {
    // 0
    camera: CameraUniform,
    // 192
    mesh: MeshParams,
    // 272
    light_dir: vec3<f32>,
    ambient: f32,
    // 288
    fog_color: vec3<f32>,
    fog_density: f32,
    // 304 == size
}

struct KeyVal {
    key: u32,
    val: u32,
}

struct SortParams {
    j: u32,
    k: u32,
    n_padded: u32,
    pad: u32,
}

// Underwater medium and reef geometry, read by the ocean raymarch pass. Mirrors `WaterParams` in
// `crates/boids-core/src/layout.rs`, field for field; the `layout_probe` test on the device fails if
// the two disagree. Twelve scalars, so no member needs padding.
struct WaterParams {
    // 0
    surface_y: f32,
    floor_y: f32,
    reef_period: f32,
    caustic_strength: f32,
    // 16
    // Per-channel extinction, metres^-1. Red is absorbed within metres, blue survives tens of them:
    // that ratio is what makes depth read as depth instead of as darkness.
    extinction: vec3<f32>,
    scatter: f32,
    // 32
    godray_strength: f32,
    surface_glow: f32,
    caustic_scale: f32,
    caustic_drift: f32,
    // 48 == size
}

// HDR post-processing parameters, read by the bloom and composite passes. Mirrors `PostParams`.
struct PostParams {
    // 0
    exposure: f32,
    bloom_threshold: f32,
    bloom_knee: f32,
    bloom_strength: f32,
    // 16
    vignette: f32,
    grain: f32,
    aberration: f32,
    tonemap_white: f32,
    // 32
    time: f32,
    saturation: f32,
    contrast: f32,
    lift: f32,
    // 48 == size
}

// The sky world's land: the heightfield map, the mesh that draws it and the vegetation on it.
//
// One struct rather than three because the three consumers describe the same piece of ground: the
// heightfield generator writes the map, the terrain mesh reads it back through the same
// `amplitude`/`frequency` that the simulation's collision field uses, and the scatter pass places
// trees on it. Splitting them would allow a mesh built from one amplitude and a collision field
// built from another, which is exactly the class of bug the shared-field tests exist to prevent.
//
// Mirrors `TerrainParams` in `crates/boids-core/src/layout.rs`.
struct TerrainParams {
    // 0
    // World xz of the map's minimum corner.
    min_xz: vec2<f32>,
    // 8
    // World size the map covers, metres.
    size_xz: vec2<f32>,
    // 16
    // Height scale, metres. Must equal the simulation's `env_scale` for the terrain field.
    amplitude: f32,
    // 20
    // Base noise frequency, m^-1. Must equal the simulation's `env_freq`.
    frequency: f32,
    // 24
    // Biome cell frequency, m^-1.
    biome_frequency: f32,
    // 28
    // Mesh segments per axis. The grid has (segments + 1)^2 vertices and segments^2 * 2 triangles.
    segments: u32,
    // 32
    // Height a drawn tree reaches above the ground, metres.
    tree_height: f32,
    // 36
    // Instance slots in the tree buffer.
    tree_capacity: u32,
    // 40
    // Scatter candidates per axis: capacity must be a small fraction of candidates^2.
    tree_candidates: u32,
    // 44
    // Texels per axis of the baked map. Carried here rather than read with `textureDimensions`:
    // the generating pass writes a *storage* texture, and asking a storage texture for its size
    // needs the IMAGE_SIZE feature, which the GL adapter does not offer.
    resolution: u32,
    // 48 == size
}

// Analytic single-scattering atmosphere. Mirrors `SkyParams`, 48 bytes, twelve scalars.
//
// The coefficients are physical (metres^-1) rather than artistic, so the two numbers that matter to
// the look are `sun_intensity` and `horizon_boost`. See `docs/math.md` for the integral this
// approximates and why a single flat-atmosphere term is enough at this world's scale.
struct SkyParams {
    // 0
    // Rayleigh scattering coefficient, m^-1: strong in blue, and the reason the sky is blue.
    beta_rayleigh: vec3<f32>,
    // 12
    // Sun radiance scale. The only free knob in the model.
    sun_intensity: f32,
    // 16
    // Mie scattering coefficient, m^-1: the aerosol term, nearly grey.
    beta_mie: vec3<f32>,
    // 28
    // Henyey-Greenstein anisotropy of the Mie term. 0.76 is the usual forward-scattering haze.
    mie_g: f32,
    // 32
    // Rayleigh scale height, metres.
    ray_scale_height: f32,
    // 36
    // Mie scale height, metres.
    mie_scale_height: f32,
    // 40
    // Extra brightness within a few degrees of the horizon, where the line of sight crosses the
    // most air.
    horizon_boost: f32,
    // 44
    // Multiplier on the in-scattered light that distant geometry fades into.
    aerial_boost: f32,
    // 48 == size
}

// One scattered tree. Storage buffer element, not a uniform: the scatter pass appends and the tree
// pass draws `count` of them. Mirrors `StaticInstance` in `crates/boids-core/src/layout.rs`.
//
// 32 bytes so the array stride is a multiple of 16.
struct StaticInstance {
    // 0
    // Base position (on the ground), metres.
    pos: vec3<f32>,
    // 12
    // Uniform scale.
    scale: f32,
    // 16
    // Rotation about Y, radians.
    yaw: f32,
    // 20
    // Species/variant selector in [0, 1): picked from the scatter hash.
    kind: f32,
    // 24
    // Biome mask under the trunk, in [0, 2). Stored rather than re-sampled by the draw pass: the
    // scatter pass already had it in hand, and the tree pass would otherwise pay a texel load per
    // vertex to learn what colour it is.
    mask: f32,
    // 28
    pad1: f32,
    // 32 == size
}

// Sentinel stored in `cell_start` for a cell that contains no boids.
const EMPTY_CELL: u32 = 0xFFFFFFFFu;

const MODE_FISH: u32 = 0u;
const MODE_BIRDS: u32 = 1u;

const INTERACTION_OFF: u32 = 0u;
const INTERACTION_ATTRACT: u32 = 1u;
const INTERACTION_REPEL: u32 = 2u;

// Which avoidance field the environment exposes, mirroring `EnvironmentKind` on the host. Declared
// here rather than in `sim/forces.wgsl` because the render passes need the same selectors: the rock
// the fish steer around and the rock the camera sees have to be the same rock.
const ENV_NONE: u32 = 0u;
const ENV_REEF: u32 = 1u;
const ENV_TERRAIN: u32 = 2u;

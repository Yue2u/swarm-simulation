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
    pad0: f32,
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
    pad_a: f32,
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

// Sentinel stored in `cell_start` for a cell that contains no boids.
const EMPTY_CELL: u32 = 0xFFFFFFFFu;

const MODE_FISH: u32 = 0u;
const MODE_BIRDS: u32 = 1u;

const INTERACTION_OFF: u32 = 0u;
const INTERACTION_ATTRACT: u32 = 1u;
const INTERACTION_REPEL: u32 = 2u;

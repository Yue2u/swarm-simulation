// === file: tests/sdf_probe.wgsl ==============================================================
// Device-side probe of the SDF and terrain fields that the simulation and the renderer share.
//
// bindings: @group(0) 0:points(read) 1:probe(uniform) 2:values(read_write)
// workgroup: 64 x 1 x 1
// dispatch:  ceil(N / 64)
//
// HOW THE TEST WORKS
//   The host fills `points` with N sample positions and `probe` with the reef period and seafloor
//   height, dispatches this shader, and reads back `values`. For each sample it recomputes the same
//   fields in Rust (`crates/boids-core/src/sdf.rs`, `terrain.rs`) and compares them element by
//   element. The field order below is normative and must match `Field` in `tests/gpu/sdf.rs`.
//
//   The comparison is the only thing that can catch a CPU/GPU divergence in the fields, and a
//   divergence is invisible until it is not: agents steer around a rock that the camera does not
//   draw, or the camera draws one the agents ignore. Primitive sign conventions are checked just as
//   strictly as the composed reef, because a flipped sign still "looks like avoidance".
//
//   `common/sdf.wgsl` declares no bindings and no entry points, so including it here composes
//   cleanly with the probe's own.

//#include "common/sdf.wgsl"

// Reef geometry, passed from the host so the probe and the Rust side cannot drift on the constants.
struct SdfProbeParams {
    period: f32,
    floor_y: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> probe: SdfProbeParams;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;

// Number of fields evaluated per point. Must match `FIELDS` on the host.
const FIELDS: u32 = 11u;

// Biome mask frequency: a low-frequency field on purpose, see `terrain::biome_weights`.
const BIOME_FREQUENCY: f32 = 0.004;

// Terrain frequency used by `eval_field` for the sky world.
const TERRAIN_FREQUENCY: f32 = 0.0025;

@compute @workgroup_size(64, 1, 1)
fn probe_sdf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&points)) {
        return;
    }
    let p = points[i].xyz;
    // Radius-profile parameter, a pure function of the sample so both sides agree exactly.
    let t = clamp((p.y + 5.0) / 10.0, 0.0, 1.0);

    let sphere = sd_sphere(p, vec3<f32>(0.0, 0.0, 0.0), 5.0);
    let box = sd_box(p, vec3<f32>(2.0, 1.0, -3.0), vec3<f32>(4.0, 2.0, 1.0));
    let biome = biome_weights(p.xz, BIOME_FREQUENCY);

    let base = i * FIELDS;
    values[base + 0u] = sphere;
    values[base + 1u] = box;
    values[base + 2u] = sd_plane_y(p, 1.5);
    values[base + 3u] = smin(sphere, box, 0.75);
    values[base + 4u] = sd_column(p, vec2<f32>(3.0, -2.0), 2.5, -4.0, 6.0);
    values[base + 5u] = column_radius_profile(t, 3.0, 0.45);
    values[base + 6u] = reef_field(p, probe.period, probe.floor_y);
    values[base + 7u] = terrain_height(p.xz, probe.period, TERRAIN_FREQUENCY);
    values[base + 8u] = biome.x;
    values[base + 9u] = biome.y;
    values[base + 10u] = biome.z;
}

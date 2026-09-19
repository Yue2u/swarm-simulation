// === file: tests/layout_probe.wgsl ===========================================================
// Device-side probe of the CPU/WGSL struct layout contract.
//
// bindings: @group(0) 0:boid_in(read) 1:params_in(read) 2:interaction_in(read) 3:values_out(rw)
//                     4:water_in(read) 5:post_in(read)
// workgroup: 1 x 1 x 1
// dispatch:  1
//
// HOW THE TEST WORKS
//   The host fills `boid_in`, `params_in`, `interaction_in`, `water_in` and `post_in` with a byte
//   ramp: byte offset 4k holds the f32 value `k + 1`. It then dispatches this shader on a 1x1x1
//   workgroup and reads back `values_out`.
//
//   This shader copies each struct field, in declaration order, into `values_out`. If the WGSL
//   member offsets match the Rust `offset_of!` offsets, then `values_out[i] == offset_i / 4 + 1`.
//   If they do not match, the value at that index is whatever lives at the offset WGSL actually
//   used, and the test fails with the two numbers side by side.
//
//   This is the only mechanism that can catch a WGSL-side layout drift. Compile-time asserts in
//   Rust cannot see the shader, and the shader's own layout rules are not visible in Rust. Without
//   this test, a struct edit that is correct in Rust and wrong in WGSL produces a simulation that
//   runs, does not crash, and is subtly wrong.
//
//   The ramp values are all integers below 2^24, so they are exactly representable as f32 and the
//   comparison is exact: no tolerance is needed or wanted here.

//#include "common/layout.wgsl"

@group(0) @binding(0) var<storage, read> boid_in: Boid;
@group(0) @binding(1) var<storage, read> params_in: SimParams;
@group(0) @binding(2) var<storage, read> interaction_in: InteractionUniforms;
@group(0) @binding(3) var<storage, read_write> values_out: array<f32>;
// The render-side uniforms are declared as storage here rather than as uniforms: the probe only
// compares member *offsets*, and for structs made of scalars and one `vec3` the two address spaces
// agree member for member. Binding them as uniforms would need separate hosts for each, which is
// exactly the kind of duplication this probe exists to avoid.
@group(0) @binding(4) var<storage, read> water_in: WaterParams;
@group(0) @binding(5) var<storage, read> post_in: PostParams;
@group(0) @binding(6) var<storage, read> terrain_in: TerrainParams;
@group(0) @binding(7) var<storage, read> sky_in: SkyParams;
@group(0) @binding(8) var<storage, read> tree_in: TreeInstance;

// Number of scalar slots the probe writes. Must match PROBE_SLOTS on the host.
const PROBE_SLOTS: u32 = 120u;

@compute @workgroup_size(1, 1, 1)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    var out: array<f32, PROBE_SLOTS>;
    for (var i = 0u; i < PROBE_SLOTS; i = i + 1u) {
        out[i] = -1.0;
    }

    // --- Boid: 12 scalars, indices 0..12 ---
    let b = boid_in;
    out[0] = b.pos.x;
    out[1] = b.pos.y;
    out[2] = b.pos.z;
    out[3] = b.species;
    out[4] = b.vel.x;
    out[5] = b.vel.y;
    out[6] = b.vel.z;
    out[7] = b.phase;
    out[8] = b.prev_dir.x;
    out[9] = b.prev_dir.y;
    out[10] = b.prev_dir.z;
    out[11] = b.color_seed;

    // --- SimParams: 36 scalars, indices 12..48 ---
    let p = params_in;
    out[12] = p.grid_min.x;
    out[13] = p.grid_min.y;
    out[14] = p.grid_min.z;
    out[15] = p.cell_size;
    out[16] = f32(p.grid_dim.x);
    out[17] = f32(p.grid_dim.y);
    out[18] = f32(p.grid_dim.z);
    out[19] = f32(p.num_boids);
    out[20] = p.w_sep;
    out[21] = p.w_ali;
    out[22] = p.w_coh;
    out[23] = p.r_percept;
    out[24] = p.r_sep;
    out[25] = p.max_speed;
    out[26] = p.min_speed;
    out[27] = p.max_force;
    out[28] = p.dt;
    out[29] = p.time;
    out[30] = p.sdf_strength;
    out[31] = p.sdf_probe;
    out[32] = p.bounds_half.x;
    out[33] = p.bounds_half.y;
    out[34] = p.bounds_half.z;
    out[35] = f32(p.mode);
    out[36] = p.wander;
    out[37] = p.sep_boost;
    out[38] = p.coh_falloff;
    out[39] = p.r_safe;
    out[40] = p.buoyancy;
    out[41] = p.drag;
    out[42] = p.env_scale;
    out[43] = p.env_floor_y;
    out[44] = f32(p.env_id);
    out[45] = p.env_freq;
    // Slots 46..48 are the two explicit padding scalars of SimParams; they are reported as -1 and are
    // deliberately not compared, because padding has no defined value.

    // --- InteractionUniforms: 12 scalars, indices 48..60 ---
    let it = interaction_in;
    out[48] = it.ray_origin.x;
    out[49] = it.ray_origin.y;
    out[50] = it.ray_origin.z;
    out[51] = f32(it.mode);
    out[52] = it.focus_point.x;
    out[53] = it.focus_point.y;
    out[54] = it.focus_point.z;
    out[55] = it.radius;
    out[56] = it.strength;
    out[57] = it.falloff;
    out[58] = it.tangent;
    out[59] = it.pad;

    // --- WaterParams: 12 scalars, indices 60..72 ---
    let w = water_in;
    out[60] = w.surface_y;
    out[61] = w.floor_y;
    out[62] = w.reef_period;
    out[63] = w.caustic_strength;
    out[64] = w.extinction.x;
    out[65] = w.extinction.y;
    out[66] = w.extinction.z;
    out[67] = w.scatter;
    out[68] = w.godray_strength;
    out[69] = w.surface_glow;
    out[70] = w.caustic_scale;
    out[71] = w.caustic_drift;

    // --- PostParams: 12 scalars, indices 72..84 ---
    let q = post_in;
    out[72] = q.exposure;
    out[73] = q.bloom_threshold;
    out[74] = q.bloom_knee;
    out[75] = q.bloom_strength;
    out[76] = q.vignette;
    out[77] = q.grain;
    out[78] = q.aberration;
    out[79] = q.tonemap_white;
    out[80] = q.time;
    out[81] = q.saturation;
    out[82] = q.contrast;
    out[83] = q.lift;

    // --- TerrainParams: 12 scalars, indices 84..96 ---
    let t = terrain_in;
    out[84] = t.min_xz.x;
    out[85] = t.min_xz.y;
    out[86] = t.size_xz.x;
    out[87] = t.size_xz.y;
    out[88] = t.amplitude;
    out[89] = t.frequency;
    out[90] = t.biome_frequency;
    out[91] = f32(t.segments);
    out[92] = t.tree_height;
    out[93] = f32(t.tree_capacity);
    out[94] = f32(t.tree_candidates);
    out[95] = f32(t.resolution);

    // --- SkyParams: 12 scalars, indices 96..108 ---
    let s = sky_in;
    out[96] = s.beta_rayleigh.x;
    out[97] = s.beta_rayleigh.y;
    out[98] = s.beta_rayleigh.z;
    out[99] = s.sun_intensity;
    out[100] = s.beta_mie.x;
    out[101] = s.beta_mie.y;
    out[102] = s.beta_mie.z;
    out[103] = s.mie_g;
    out[104] = s.ray_scale_height;
    out[105] = s.mie_scale_height;
    out[106] = s.horizon_boost;
    out[107] = s.aerial_boost;

    // --- TreeInstance: 8 scalars, indices 108..116 ---
    let tr = tree_in;
    out[108] = tr.pos.x;
    out[109] = tr.pos.y;
    out[110] = tr.pos.z;
    out[111] = tr.scale;
    out[112] = tr.yaw;
    out[113] = tr.kind;
    out[114] = tr.mask;
    out[115] = tr.pad1;
    // Slots 116..120 are spare: the probe allocates a round number of slots so that adding a member
    // to a struct does not silently overrun the array.

    for (var i = 0u; i < PROBE_SLOTS; i = i + 1u) {
        values_out[i] = out[i];
    }
}

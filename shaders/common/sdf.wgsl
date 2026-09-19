// === file: common/sdf.wgsl ==================================================================
// Signed distance fields: the reef used by the underwater world, the terrain used by the sky
// world, and the gradient machinery both use for collision avoidance.
//
// Sign convention: **negative inside** the solid, positive outside, magnitude approximately the
// distance to the nearest surface. This matches `crates/boids-core/src/sdf.rs`, and the
// `sdf_wgsl_matches_rust` device test compares the two on a grid of sample points. A sign flip is
// otherwise very hard to notice visually: both signs "look like avoidance" until agents start
// flying inside rocks.
//
// Every field function is *deterministic and cheap*: no textures, no precomputed volumes. The
// reef is a domain-repeated family of tapered columns, which is why an unbounded environment
// costs a handful of arithmetic operations instead of a mesh.
//
// Header only: declares no bindings and no entry points.

//#include "common/math_common.wgsl"

// ---------------------------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------------------------

fn sd_sphere(p: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    return length(p - center) - radius;
}

fn sd_box(p: vec3<f32>, center: vec3<f32>, half: vec3<f32>) -> f32 {
    let q = abs(p - center) - half;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Infinite plane at height `y`, solid below. Positive above the plane.
fn sd_plane_y(p: vec3<f32>, y: f32) -> f32 {
    return p.y - y;
}

// Polynomial smooth minimum. Fuses primitives without hard creases, which matters because a hard
// crease in a distance field produces a discontinuity in the avoidance gradient, and agents snag
// on it instead of sliding past.
fn smin(a: f32, b: f32, k: f32) -> f32 {
    if (k <= 0.0) {
        return min(a, b);
    }
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Vertical capped column of radius `r` centred at `center_xz`, spanning [base_y, top_y].
//
// Written as a 2D radial term fused with a vertical slab so the result is a genuine distance
// field, which sphere tracing requires. A naive union of a cylinder and two caps is not.
fn sd_column(p: vec3<f32>, center_xz: vec2<f32>, radius: f32, base_y: f32, top_y: f32) -> f32 {
    let d_radial = length(p.xz - center_xz) - radius;
    let d_vertical = abs(p.y - 0.5 * (base_y + top_y)) - 0.5 * (top_y - base_y);
    let outside = length(max(vec2<f32>(d_radial, d_vertical), vec2<f32>(0.0)));
    return outside + min(max(d_radial, d_vertical), 0.0);
}

// Radius of a reef column as a function of normalized height `t` in [0, 1].
//
// Columns thin toward the top with a slight irregular flare so the silhouette reads as coral
// rather than as an extruded cylinder. Mirrors `sdf::column_radius_profile`.
fn column_radius_profile(t_in: f32, base_radius: f32, taper: f32) -> f32 {
    let t = clamp(t_in, 0.0, 1.0);
    return base_radius * (1.0 - taper * t * t) * (1.0 + 0.12 * sin(t * 9.0));
}

// Integer-lattice coordinate hash in [0, 1), matching `terrain::hash21` on the CPU.
//
// Integer mixing, not `fract(sin(dot(p, k)) * n)`: the float form silently returns 0 once the
// product's integer part exceeds the 24-bit mantissa, which flattens the terrain and freezes the
// biome mask. Integer arithmetic wraps identically on both sides, so this is also bit-exact
// between the CPU and the GPU.
fn hash21(p: vec2<f32>) -> f32 {
    let xi = bitcast<u32>(i32(floor(p.x)));
    let yi = bitcast<u32>(i32(floor(p.y)));
    let h = hash_u32(xi * 0x9e3779b9u ^ hash_u32(yi * 0x85ebca6bu));
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

// ---------------------------------------------------------------------------------------------
// Environments
// ---------------------------------------------------------------------------------------------

// Bias added to the cell coordinate before the `floor`, so that a sample sitting exactly on a cell
// boundary lands in the same cell on the CPU and the GPU.
//
// This is not paranoia, it is a real portability hazard that this repository already hit: a driver
// is free to rewrite `p / period` as `p * (1 / period)`, and the reciprocal of 48 is not exactly
// representable, so `96.0 / 48.0` evaluates to 1.9999999 on one side and 2.0 on the other. The two
// sides then sample different columns and disagree by metres on a field that is supposed to agree
// to 1e-4. `CELL_BIAS` in `crates/boids-core/src/sdf.rs` must hold the same value: it is a million
// times larger than the rounding a reciprocal introduces and four orders of magnitude smaller than
// anything the field's users can see (0.004 m on a 48 m period).
const REEF_CELL_BIAS: f32 = 1e-4;

// The reef: domain-repeated tapered columns rising from a seafloor.
//
// `period` is the repetition spacing in world metres and `floor_y` the seafloor height. Two
// column families at different phases are blended so the reef does not read as a regular lattice.
fn reef_field(p: vec3<f32>, period: f32, floor_y: f32) -> f32 {
    let cell = floor(p.xz / period + vec2<f32>(REEF_CELL_BIAS));
    let h = hash21(cell);
    let center = (cell + vec2<f32>(0.5)) * period;
    let height = 20.0 + 26.0 * h;
    let radius = 3.0 + 3.5 * hash21(cell + vec2<f32>(17.0));
    let base = floor_y;
    let top = floor_y + height;
    let t = clamp((p.y - base) / (top - base), 0.0, 1.0);
    let r = column_radius_profile(t, radius, 0.45);
    let a = sd_column(p, center, r, base, top);

    // Second family, offset by half a period: fills the gaps of the first and gives the reef
    // its arches without needing a separate primitive.
    let center2 = center + vec2<f32>(0.55 * period, 0.1 * period);
    let height2 = 12.0 + 30.0 * hash21(cell + vec2<f32>(5.0, 9.0));
    let r2 = 2.0 + 3.0 * hash21(cell + vec2<f32>(31.0, 3.0));
    let t2 = clamp((p.y - base) / height2, 0.0, 1.0);
    let b = sd_column(p, center2, column_radius_profile(t2, r2, 0.5), base, base + height2);

    let columns = smin(a, b, 3.0);
    // The floor uses a larger blend radius so columns visibly grow out of it.
    return smin(columns, sd_plane_y(p, floor_y), 6.0);
}

// Frequency of the domain warp relative to the terrain's own, and how far it displaces the field, in
// lattice cells of the base octave. Must match `WARP_FREQUENCY`/`WARP_AMOUNT` in
// `crates/boids-core/src/terrain.rs`.
//
// The warp is what turns a sum of smooth bumps into ridges and valleys, and it has to be *smooth* to
// do that: hashing the cell containing the sample makes the warp jump six lattice cells every 1.4
// metres, which is white noise with the terrain's amplitude rather than a landscape. That mistake was
// made once, survived a boundedness test, and was unmistakable the first time the field was shaded.
const WARP_FREQUENCY: f32 = 0.5;
const WARP_AMOUNT: f32 = 2.0;

// Procedural terrain height, matching `boids_core::terrain::height_at`.
//
// Four octaves of value noise with domain warping. The biome mask is a separate low-frequency field
// so that the terrain and the biome boundaries are correlated but not identical, which avoids
// suspiciously rectangular forests.
fn terrain_height(p: vec2<f32>, amplitude: f32, frequency: f32) -> f32 {
    let warp_scale = frequency * WARP_FREQUENCY;
    let warp = vec2<f32>(
        value_noise(p * warp_scale + vec2<f32>(11.3, 4.7)),
        value_noise(p * warp_scale + vec2<f32>(3.1, 19.9)),
    ) * 2.0 - vec2<f32>(1.0);
    let q = p * frequency + warp * WARP_AMOUNT;

    var h = 0.0;
    var amp = 1.0;
    var freq = 1.0;
    var norm = 0.0;
    for (var octave = 0u; octave < 4u; octave = octave + 1u) {
        h = h + value_noise(q * freq) * amp;
        norm = norm + amp;
        amp = amp * 0.5;
        freq = freq * 2.03;
    }
    return (h / max(norm, 1e-6)) * amplitude;
}

// Value noise on the integer lattice with Hermite interpolation, matching
// `boids_core::terrain::value_noise`.
//
// From the integer hash rather than a texture: no bindings, no precomputed permutation table, and
// the same lattice on the CPU and the GPU.
fn value_noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    let c00 = hash21(cell);
    let c10 = hash21(cell + vec2<f32>(1.0, 0.0));
    let c01 = hash21(cell + vec2<f32>(0.0, 1.0));
    let c11 = hash21(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(c00, c10, w.x), mix(c01, c11, w.x), w.y);
}

// Biome weights as (forest, dunes, canyon), summing to 1.
//
// A hard biome switch would show as a straight seam across the terrain; a partition of unity keeps
// the transition gradual and, more importantly, keeps the palette blend continuous.
fn biome_weights(p: vec2<f32>, frequency: f32) -> vec3<f32> {
    let a = hash21(floor(p * frequency));
    let b = hash21(floor(p * frequency) + vec2<f32>(41.0, 7.0));
    let c = hash21(floor(p * frequency) + vec2<f32>(13.0, 53.0));
    let total = max(a + b + c, 1e-5);
    return vec3<f32>(a, b, c) / total;
}

// Biome coordinate in [0, 2): 0 = forest, 1 = dunes, 2 = canyon, continuous across a transition.
//
// Derived from `biome_weights` rather than hashed on its own so that the palette ramp and the
// weights cannot disagree. Because the weights sum to 1, the map is a sweep: at a forest/dunes
// boundary `w.z` is 0 and the mask runs 0 -> 1, and at a dunes/canyon boundary `w.y` is 0 and it
// runs 1 -> 2. One scalar is all the heightfield's second channel has room for, and one scalar is
// all a three-stop palette ramp needs.
fn biome_mask(p: vec2<f32>, frequency: f32) -> f32 {
    let w = biome_weights(p, frequency);
    return w.y + 2.0 * w.z;
}

// ---------------------------------------------------------------------------------------------
// Gradient-based avoidance
// ---------------------------------------------------------------------------------------------

// Everything the field functions need: which field, and its scale parameters.
//
// Bundled rather than passed as four positional scalars because three of them are `f32` and used
// adjacently: `eval_field(p, ENV_TERRAIN, amplitude, frequency)` and
// `eval_field(p, ENV_TERRAIN, frequency, amplitude)` both compile, and only one of them is the
// terrain. `scale` means the reef's repetition period or the terrain's amplitude, depending on the
// field, and `freq` is the terrain's base noise frequency (unused by the reef).
struct FieldArgs {
    id: u32,
    scale: f32,
    freq: f32,
    floor_y: f32,
}

// The field the simulation should avoid, from its own parameters.
fn sim_field_args(params: SimParams) -> FieldArgs {
    return FieldArgs(params.env_id, params.env_scale, params.env_freq, params.env_floor_y);
}

// Central-difference gradient of an arbitrary scalar field.
//
// `eps` should be about half a cell: much smaller and the field's own noise dominates the
// difference, much larger and thin obstacles are smoothed out of existence.
fn sdf_gradient_at(p: vec3<f32>, eps: f32, f: FieldArgs) -> vec3<f32> {
    let ex = vec3<f32>(eps, 0.0, 0.0);
    let ey = vec3<f32>(0.0, eps, 0.0);
    let ez = vec3<f32>(0.0, 0.0, eps);
    let dx = eval_field(p + ex, f) - eval_field(p - ex, f);
    let dy = eval_field(p + ey, f) - eval_field(p - ey, f);
    let dz = eval_field(p + ez, f) - eval_field(p - ez, f);
    return vec3<f32>(dx, dy, dz) / (2.0 * eps);
}

// Evaluates whichever environment field is active. `f.id` mirrors `EnvironmentKind` on the
// host: 0 = empty space (no avoidance at all), 1 = reef, 2 = terrain.
//
// The terrain branch is the *same* expression the heightfield map is baked with
// (`terrain/heightfield.wgsl`), evaluated with the same `scale` and `freq` that the host puts in
// `SimParams`, so the surface the agents avoid and the surface the camera draws are one field
// evaluated twice rather than two fields that happen to look alike.
fn eval_field(p: vec3<f32>, f: FieldArgs) -> f32 {
    if (f.id == ENV_NONE) {
        // A large positive constant means "infinitely far from any surface", so the avoidance
        // term vanishes without a branch in the caller.
        return 1e6;
    }
    if (f.id == ENV_REEF) {
        return reef_field(p, f.scale, f.floor_y);
    }
    return p.y - terrain_height(p.xz, f.scale, f.freq);
}

// Avoidance force from a distance value and a surface normal.
//
// `smoothstep(r_safe, 0, d)` rather than `d.recip()`: the reciprocal blows up inside the solid,
// which launches agents across the map when one clips a corner during a frame hitch. The
// smoothstep form saturates instead.
fn avoid_force_from(d: f32, normal: vec3<f32>, r_safe: f32, strength: f32) -> vec3<f32> {
    return normal * (smoothstep01(r_safe, 0.0, d) * strength);
}

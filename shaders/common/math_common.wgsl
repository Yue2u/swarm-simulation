// === file: common/math_common.wgsl ==========================================================
// Host-independent helpers shared by the simulation and the render passes.
//
// Every function here has a CPU twin in `crates/boids-core/src/{math,reference}.rs`. Where a
// function is used to produce simulation forces, the two implementations must agree bit for bit,
// which is why they are written with integer hashing rather than `sin()` of a float, and why no
// function here relies on transcendental functions that differ between drivers.
//
// Header only: this file declares no bindings, no entry points and no globals.

//#include "common/layout.wgsl"

const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;

// ---------------------------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------------------------

// Bit-reinterpretation of a float, used for hash inputs derived from positions. Named rather than
// inlined so the intent is obvious at call sites.
fn as_u32(v: f32) -> u32 {
    return bitcast<u32>(v);
}

// Integer hash, mixing a u32 into a well-distributed u32.
//
// Unsigned integer arithmetic wraps in WGSL (unlike float arithmetic, which is exact only under
// rounding rules the driver may choose), so this is the only kind of hashing in the codebase that
// is guaranteed identical on the CPU and the GPU. The constants are the "lowbias32" finaliser.
fn hash_u32(input: u32) -> u32 {
    var h = input;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}

// Uniform f32 in [0, 1) derived from an integer hash.
// `h >> 8` is below 2^24, so the conversion is exact and matches the CPU twin.
fn hash_unit(input: u32) -> f32 {
    let h = hash_u32(input);
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

// Uniform f32 in [-1, 1).
fn hash_signed(input: u32) -> f32 {
    return hash_unit(input) * 2.0 - 1.0;
}

// Smooth Hermite interpolation. Mirrors `boids_core::math::smoothstep`.
fn smoothstep01(edge0: f32, edge1: f32, x: f32) -> f32 {
    let d = edge1 - edge0;
    if (abs(d) < 1e-7) {
        if (x < edge0) { return 0.0; }
        return 1.0;
    }
    let t = clamp((x - edge0) / d, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// ---------------------------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------------------------

// Clamps a vector to at most `max_len`, preserving direction. Zero stays zero.
fn clamp_len(v: vec3<f32>, max_len: f32) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > max_len * max_len && len_sq > 0.0) {
        return v * (max_len / sqrt(len_sq));
    }
    return v;
}

// Clamps a vector's length into [min_len, max_len], preserving direction.
//
// A zero vector is returned unchanged: the caller decides which way a stalled agent should be
// nudged, because that is mode-specific.
fn clamp_len_range(v: vec3<f32>, min_len: f32, max_len: f32) -> vec3<f32> {
    let len = length(v);
    if (len <= 1e-7) {
        return v;
    }
    return v * (clamp(len, min_len, max_len) / len);
}

// Normalizes, returning the input when it is too short to have a reliable direction.
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq < 1e-12) {
        return vec3<f32>(0.0);
    }
    return v * inverseSqrt(len_sq);
}

// Builds an orthonormal basis whose third axis is `fwd`.
//
// The reference axis is swapped when `fwd` is nearly vertical, otherwise the cross product is
// ~zero and the basis degenerates into NaNs. Mirrors `boids_core::math::basis_from_forward`.
fn basis_from_forward(fwd_in: vec3<f32>) -> mat3x3<f32> {
    let fwd = safe_normalize(fwd_in);
    var reference = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(fwd.y) > 0.99) {
        reference = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = safe_normalize(cross(reference, fwd));
    let up = cross(fwd, right);
    return mat3x3<f32>(right, up, fwd);
}

// Rotation of `v` around a unit `axis` by `angle` (Rodrigues). Cheaper and more stable than
// building a quaternion for a single-axis animation tilt.
fn rotate_axis(v: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return v * c + cross(axis, v) * s + axis * (dot(axis, v) * (1.0 - c));
}

// ---------------------------------------------------------------------------------------------
// Simulation forces, mirrored exactly by `boids_core::reference`
// ---------------------------------------------------------------------------------------------

// How often the wander direction changes, in Hz. Must match WANDER_RATE in reference.rs.
const WANDER_RATE: f32 = 0.7;

// The wander tick for a simulation time. Integer-typed on purpose so the CPU and the GPU agree:
// a float tick would flip by one step whenever the two disagree in the last mantissa bit.
fn wander_tick(time: f32) -> u32 {
    return u32(max(time, 0.0) * WANDER_RATE);
}

// Per-agent wander force. Deliberately not normalized: a variable-magnitude nudge is
// indistinguishable visually from a normalized one, and skipping normalization removes a
// `normalize(0)` NaN case that would otherwise be a CPU/GPU divergence risk.
fn wander_force(index: u32, tick: u32, scale: f32) -> vec3<f32> {
    let base = index * 0x9e3779b9u ^ tick * 0x85ebca6bu;
    return vec3<f32>(
        hash_signed(base),
        hash_signed(base ^ 0x27d4eb2du),
        hash_signed(base ^ 0x165667b1u),
    ) * scale;
}

// Soft containment pushing an agent back inside the box [-half, half].
//
// The force ramps up over the outer 15% of each axis, so agents bank away from the boundary
// instead of reflecting off an invisible wall. Scaled by the agent's own cruise speed so the same
// expression works for slow fish and fast birds.
fn bounds_force(pos: vec3<f32>, half: vec3<f32>, speed_scale: f32) -> vec3<f32> {
    var f = vec3<f32>(0.0);
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        let h = half[axis];
        let margin = max(0.15 * h, 1e-3);
        let p = pos[axis];
        let to_pos_wall = h - p;
        let to_neg_wall = p + h;
        if (to_pos_wall < margin) {
            f[axis] = f[axis] - (1.0 - max(to_pos_wall / margin, 0.0));
        }
        if (to_neg_wall < margin) {
            f[axis] = f[axis] + (1.0 - max(to_neg_wall / margin, 0.0));
        }
    }
    return f * (speed_scale * 4.0);
}

// Cursor influence at a point, mirroring `boids_core::reference::interaction_force`.
//
// `w(r) = clamp(1 - r/R, 0, 1)^falloff`, then a radial push and, in repel mode, a tangential
// swirl. The swirl is what makes a repelled swarm look like it is threading around the cursor
// rather than being blown radially away from it.
fn interaction_force(pos: vec3<f32>, vel: vec3<f32>, inter: InteractionUniforms) -> vec3<f32> {
    if (inter.mode == INTERACTION_OFF || inter.strength == 0.0 || inter.radius <= 0.0) {
        return vec3<f32>(0.0);
    }
    let g = inter.focus_point - pos;
    let r = length(g);
    if (r > inter.radius || r < 1e-6) {
        return vec3<f32>(0.0);
    }
    let t = clamp(1.0 - r / inter.radius, 0.0, 1.0);
    let w = pow(t, inter.falloff);
    let dir = g / r;

    var f = vec3<f32>(0.0);
    if (inter.mode == INTERACTION_ATTRACT) {
        f = dir * (w * inter.strength);
    } else {
        f = -dir * (pow(w, 1.5) * inter.strength);
    }

    if (inter.tangent != 0.0) {
        let tang_raw = cross(vec3<f32>(0.0, 1.0, 0.0), dir);
        let tang = safe_normalize(tang_raw);
        // Only swirl agents that are already moving along the tangent, otherwise the sign is noise
        // and the swarm jitters instead of orbiting.
        if (length(tang_raw) > 1e-6) {
            var sense = 1.0;
            if (dot(vel, tang) < 0.0) {
                sense = -1.0;
            }
            f = f + tang * (sense * w * inter.tangent);
        }
    }
    return f;
}

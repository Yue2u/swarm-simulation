// === file: render/boid.wgsl ==================================================================
// Pass: instanced agent rendering. Non-indexed triangle list, vertex pulling.
//
// bindings: @group(0) 0:SceneUniform(uniform) 1:array<Boid>(storage, read)
// draw:     draw(0..MESH_VERTICES, 0..num_boids)   one instanced call for the whole swarm
// cull:     none. Agents are drawn with both faces because the wing and fin triangles are
//           single-sided by construction, and a back-facing tail fin on a distant fish should
//           still be visible rather than popping out of existence.
//
// ORIENTATION
//   There is no vertex buffer and no per-instance transform buffer. The mesh is a pure function of
//   `vertex_index`, and the orientation basis is derived from the agent's velocity *inside the
//   vertex shader*. That keeps the per-frame CPU cost at zero for any agent count and removes an
//   entire class of bugs where instance data and simulation data drift out of sync.
//
//   fwd   = normalize(vel)
//   right = normalize(cross(reference, fwd)), with the reference axis swapped when fwd is vertical
//   up    = cross(fwd, right)
//   world = pos + M * (local + animation_offset)
//
//   The bank angle comes from the change in heading since the previous frame (`prev_dir`, stored by
//   the simulation). Banking is what separates "a swarm" from "a cloud of darts": it reads as the
//   agents reacting to their own turns.
//
// NORMALS
//   Computed in the fragment shader from screen-space derivatives of the world position rather than
//   interpolated from per-vertex normals. This is exact for the actual deformed triangle, needs no
//   normal data in the mesh function, and cannot produce a wrong normal on a degenerate wing
//   triangle because such a triangle has no visible pixels to shade.

//#include "render/bindings.wgsl"

// Per-mode shape parameters, set from `boids_render::MeshProfile`.
const VARIANT_FISH: f32 = 0.0;
const VARIANT_BIRD: f32 = 1.0;

// Vertex budget: 18 body triangles (54 verts) plus 4 wing triangles (12 verts).
const BODY_TRIANGLES: u32 = 18u;
const WING_TRIANGLES: u32 = 4u;
const MESH_VERTICES: u32 = (BODY_TRIANGLES + WING_TRIANGLES) * 3u;

const RING_SIDES: u32 = 4u;

// A point on the 4-sided cross-section ring at the front of the body, `i` wrapping.
fn front_ring(i: u32, m: MeshParams) -> vec3<f32> {
    let a = f32(i % RING_SIDES) * (TAU / f32(RING_SIDES));
    return vec3<f32>(cos(a) * m.body_w, sin(a) * m.body_h, m.fin_z);
}

// A point on the ring at the back of the body.
fn back_ring(i: u32, m: MeshParams) -> vec3<f32> {
    let a = f32(i % RING_SIDES) * (TAU / f32(RING_SIDES));
    return vec3<f32>(cos(a) * m.body_w * 0.45, sin(a) * m.body_h * 0.55, -0.45);
}

// Builds one vertex of the tail fin.
//
// A single flat triangle pair, drawn without culling. `s` selects between the two triangles and
// `c` the corner, mirroring the body's triangle layout so the same vertex-count arithmetic works.
fn tail_fin_vertex(s: u32, c: u32, m: MeshParams) -> vec3<f32> {
    let root = vec3<f32>(0.0, 0.0, -0.55);
    let upper = vec3<f32>(0.0, m.fin_size, -0.95 - m.tail * 0.3);
    let lower = vec3<f32>(0.0, -m.fin_size * 0.8, -0.9 - m.tail * 0.3);
    if (s == 0u) {
        if (c == 0u) { return root; }
        if (c == 1u) { return upper; }
        return lower;
    }
    // Second triangle doubles back so the fin has a swept trailing edge.
    if (c == 0u) { return upper; }
    if (c == 1u) { return vec3<f32>(0.0, m.fin_size * 0.35, -1.05 - m.tail * 0.3); }
    return lower;
}

// Builds one vertex of a wing. `side` is -1 for the left wing and +1 for the right.
fn wing_vertex(s: u32, c: u32, side: f32, m: MeshParams) -> vec3<f32> {
    let span = m.wing_span * side;
    let root_front = vec3<f32>(0.0, 0.05, 0.12);
    let root_back = vec3<f32>(0.0, 0.02, -0.30);
    let tip_back = vec3<f32>(span * 0.85, -0.02, -m.wing_sweep - 0.30);
    let tip_front = vec3<f32>(span, 0.06, -m.wing_sweep + 0.18);
    // Each wing is two triangles: root->tip is the leading edge, sweep controls how far back the
    // tip sits. A swept wing reads as a bird; a straight one reads as a paper aeroplane.
    if (s == 0u) {
        if (c == 0u) { return root_front; }
        if (c == 1u) { return root_back; }
        return tip_back;
    }
    if (c == 0u) { return root_front; }
    if (c == 1u) { return tip_back; }
    return tip_front;
}

// The full procedural mesh. Returns the local-space position for vertex `vi`.
//
// Triangle layout (all non-indexed, 3 vertices per triangle):
//   [0, 4)    nose cap      : nose, front_ring[c], front_ring[c+1]
//   [4, 12)   body sides    : two triangles per ring quadrant
//   [12, 16)  tail cap      : tail, back_ring[c+1], back_ring[c]
//   [16, 18)  tail fin
//   [18, 22)  wings (birds only; collapsed to a point for fish)
//
// For fish the wing triangles are degenerate (all three corners identical), which costs a handful
// of vertex shader invocations and no visible pixels: cheaper than a second pipeline, and it keeps
// both worlds on one mesh function.
fn mesh_vertex(vi: u32, m: MeshParams) -> vec3<f32> {
    let tri = vi / 3u;
    let corner = vi - tri * 3u;

    if (tri < 4u) {
        let c = tri;
        if (corner == 0u) { return vec3<f32>(0.0, 0.0, m.nose); }
        if (corner == 1u) { return front_ring(c, m); }
        return front_ring(c + 1u, m);
    }

    if (tri < 12u) {
        let s = tri - 4u;
        let q = s / 2u;
        let second = s - q * 2u;
        if (second == 0u) {
            if (corner == 0u) { return front_ring(q, m); }
            if (corner == 1u) { return back_ring(q, m); }
            return back_ring(q + 1u, m);
        }
        if (corner == 0u) { return front_ring(q, m); }
        if (corner == 1u) { return back_ring(q + 1u, m); }
        return front_ring(q + 1u, m);
    }

    if (tri < 16u) {
        let c = tri - 12u;
        if (corner == 0u) { return vec3<f32>(0.0, 0.0, -1.0 - m.tail * 0.2); }
        if (corner == 1u) { return back_ring(c + 1u, m); }
        return back_ring(c, m);
    }

    if (tri < 18u) {
        return tail_fin_vertex(tri - 16u, corner, m);
    }

    if (m.variant < 0.5) {
        // Degenerate: fish have no wings.
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let s = (tri - 18u) / 2u;
    let side = select(1.0, -1.0, (tri - 18u) % 2u == 0u);
    return wing_vertex(s, corner, side, m);
}

// Applies the swimming / flapping deformation in local space.
//
// Fish: a lateral wave running from nose to tail whose amplitude grows quadratically toward the
// tail, which is what an actual swimming body does.
// Birds: the wings rotate about the body axis by a single flapping sine; the body itself stays
// rigid so the silhouette does not wobble.
fn animate_local(p_in: vec3<f32>, phase: f32, m: MeshParams) -> vec3<f32> {
    var p = p_in;
    let t = scene.camera.time * m.wave_freq + phase;
    if (m.variant < 0.5) {
        let along = clamp(-p.z, 0.0, 1.2);
        p.x = p.x + sin(t - along * 2.0) * m.wave_amp * along * along;
        return p;
    }
    if (abs(p.x) > 1e-5) {
        let side = sign(p.x);
        let flap = sin(t) * m.wave_amp;
        p = rotate_axis(p, vec3<f32>(0.0, 0.0, 1.0), side * flap);
    }
    return p;
}

// Bank angle from the change of heading, in radians.
//
// `prev_dir` is stored by the simulation. Using the stored value rather than a second velocity
// read keeps the memory traffic at one Boid load per instance, and it also means the bank freezes
// correctly if the simulation is paused.
fn bank_angle(prev_dir: vec3<f32>, fwd: vec3<f32>, right: vec3<f32>) -> f32 {
    let turn = cross(safe_normalize(prev_dir), fwd);
    let rate = dot(turn, vec3<f32>(0.0, 1.0, 0.0));
    let lateral = dot(turn, right);
    return clamp(lateral * 3.0 + rate * 1.5, -0.7, 0.7);
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) emissive: f32,
    @location(3) speed: f32,
    @location(4) species: f32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) inst: u32,
) -> VsOut {
    let b = boids[inst];
    let m = scene.mesh;

    let fwd = safe_normalize(b.vel);
    let basis = basis_from_forward(fwd);
    let right = basis[0];
    let up = basis[1];

    var local = animate_local(mesh_vertex(vi, m), b.phase, m);

    let bank = bank_angle(b.prev_dir, fwd, right);
    if (abs(bank) > 1e-4) {
        local = rotate_axis(local, fwd, bank);
    }

    // Scale the mesh by the agent's speed so that a fast agent is drawn slightly stretched, which
    // reads as motion without needing any motion blur.
    let speed = length(b.vel);
    let stretch = 1.0 + 0.18 * clamp(speed / max(params_max_speed_proxy(), 1e-3), 0.0, 1.5);
    local.z = local.z * stretch;

    let world = b.pos + basis * local;

    var out: VsOut;
    out.clip = scene.camera.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.color = agent_color(b.species, b.color_seed, 0.0);
    out.emissive = m.emissive * (0.5 + 0.5 * sin(scene.camera.time * 2.0 + b.phase));
    out.speed = speed;
    out.species = b.species;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let view_dir = scene.camera.eye - in.world;
    let distance = length(view_dir);
    let to_eye = view_dir / max(distance, 1e-5);

    // Geometric normal from screen-space derivatives: exact for the triangle actually rasterised,
    // including the animated deformation, and free of any per-vertex normal data.
    var n = normalize(cross(dpdx(in.world), dpdy(in.world)));
    if (dot(n, to_eye) < 0.0) {
        n = -n;
    }

    let light = safe_normalize(-scene.light_dir);
    let diffuse = max(dot(n, light), 0.0);
    let rim = pow(1.0 - max(dot(n, to_eye), 0.0), 3.0);

    var color = in.color * (scene.ambient + diffuse) + in.color * rim * 0.6;
    // Bioluminescence: fish light up, birds do not (their emissive is 0).
    color = color + in.color * in.emissive * 2.5;

    let blend = medium_blend(distance);
    color = mix(color, scene.fog_color, blend);

    return vec4<f32>(color, 1.0);
}

// === file: render/ocean.wgsl ================================================================
// Pass: the underwater environment. One full-screen triangle, raymarched against the reef SDF, at
// half resolution. `render/ocean_resolve.wgsl` upsamples it and writes depth.
//
// bindings: @group(0) 0:SceneUniform(uniform) 1:array<Boid>(storage, read) 2:WaterParams(uniform)
//                     3:InteractionUniforms(uniform) 4:PostParams(uniform)
// draw:     draw(0..3, 0..1)
// target:   half-size colour. The alpha channel carries the ray's hit distance in metres, or 0 for a
//           miss, which is how the distance reaches the resolve without a depth texture the GL
//           backend could not sample.
//
// WHY A RAYMARCH AND NOT GEOMETRY
//   The reef is a domain-repeated family of tapered columns: unbounded, procedural, and cheap to
//   evaluate but impossible to enumerate. Rasterising it would mean either generating distinct columns
//   per world cell (which the domain repetition exists to avoid) or a voxel surface (which costs VRAM
//   proportional to the world). Marching a distance field costs arithmetic per pixel and no memory,
//   and it gives the resolve a real distance so the agents occlude against it correctly. That last
//   point is the reason the pass is not simply a background gradient: the fish have to swim *behind*
//   the columns, not be drawn over them.
//
// STEPPING RULE
//   Sphere tracing with a 0.6 safety factor and a hard minimum step, not the textbook
//   `t += sdf(p)`. The field is a *domain-repeated* SDF: at a repetition boundary the value can jump
//   upward, so the Lipschitz bound that makes full sphere tracing safe does not hold across the
//   boundary, and an unclamped step can tunnel through the edge of a column. The 0.6 factor and the
//   0.3 m floor keep every step inside the region where the field is well behaved, and 80 steps at
//   that rate still cover the whole world from any camera position inside it.
//
// INVARIANTS
//   * the alpha channel is written for every fragment: the hit distance in metres, or 0 when the ray
//     hit nothing, so a caller-provided target is never left with the previous frame's value.
//   * colour is linear HDR: the surface is far above 1.0 and the bloom pass depends on that.

//#include "render/water.wgsl"

const MARCH_STEPS: u32 = 80u;
/// Hard cap on the march distance, as a multiple of the water column's height. A ray that has crossed
/// the whole world without hitting anything is looking at open water, and shading it as medium from
/// here is indistinguishable from shading it after another thousand metres.
const MARCH_RANGE: f32 = 3.0;

/// The raymarch renders into a target this fraction of the scene's size per axis, so the ray for a
/// fragment is reconstructed against the *scaled* viewport rather than the full one. Must match the
/// half-size target the host allocates in `targets.rs`.
const OCEAN_DOWNSCALE: f32 = 0.5;

// NDC of the pixel `frag` is the centre of, for the half-size target.
fn ocean_ndc(frag: vec4<f32>) -> vec2<f32> {
    let viewport = scene.camera.viewport * OCEAN_DOWNSCALE;
    return vec2<f32>(
        2.0 * (frag.x) / viewport.x - 1.0,
        1.0 - 2.0 * (frag.y) / viewport.y,
    );
}

// What a ray hit.
const KIND_NONE: u32 = 0u;
const KIND_SOLID: u32 = 1u;
const KIND_SURFACE: u32 = 2u;

struct MarchHit {
    t: f32,
    kind: u32,
}

// Full-screen triangle. The three vertices are placed outside the clip volume so their interpolation
// covers [-1, 1] exactly, which avoids the diagonal seam a two-triangle quad can show.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 1.0, 1.0);
}

// Distance along `rd` at which the ray crosses the horizontal plane `y = height`, or -1 when it never
// does in front of the origin.
fn plane_t(ro: vec3<f32>, rd: vec3<f32>, height: f32) -> f32 {
    if (abs(rd.y) < 1e-6) {
        return -1.0;
    }
    let t = (height - ro.y) / rd.y;
    if (t <= 0.0) {
        return -1.0;
    }
    return t;
}

// Marches the reef field up to `t_max`, treating the surface plane as the far end of the water column.
fn march(ro: vec3<f32>, rd: vec3<f32>, t_max: f32) -> MarchHit {
    var t = 0.05;
    for (var i = 0u; i < MARCH_STEPS; i = i + 1u) {
        if (t > t_max) {
            break;
        }
        let p = ro + rd * t;
        let d = eval_field(p, reef_args());
        // Acceptance radius grows with distance: at 300 m a 2 cm threshold is below the precision of
        // an f32 position, and the march would never terminate on a surface it is grazing.
        if (d < max(0.02, 0.0025 * t)) {
            var hit: MarchHit;
            hit.t = t;
            hit.kind = KIND_SOLID;
            return hit;
        }
        t = t + clamp(d * 0.6, 0.3, 24.0);
    }
    var miss: MarchHit;
    miss.t = t_max;
    miss.kind = KIND_NONE;
    return miss;
}

// Surface colour of the rock: procedural, from height above the seafloor, surface normal and noise.
//
// Coral grows with light, so the tops of the columns are the saturated colour and the bases are bare
// olive rock; the flat seafloor gets sand. Writing it as three albedos blended by where the point is
// is what keeps the reef from reading as one flat material under the caustics.
fn rock_albedo(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let column_height = max(water.surface_y - water.floor_y, 1.0);
    let t = clamp((p.y - water.floor_y) / (column_height * 0.5), 0.0, 1.0);
    let grain = hash21(floor(p.xz * 0.9)) * 0.6 + hash21(floor(p.xz * 3.7)) * 0.4;

    let bare = vec3<f32>(0.17, 0.16, 0.14);
    let coral = vec3<f32>(0.34, 0.15, 0.13) * (0.7 + 0.7 * grain);
    let sand = vec3<f32>(0.44, 0.39, 0.29);
    // Flat, upward-facing rock is sand; anything steep is colonised.
    let flat = smoothstep(0.82, 0.99, n.y);
    return mix(mix(bare, coral, smoothstep(0.0, 0.5, t)), sand, flat * 0.85);
}

// Shades a ray that reached the surface plane.
fn shade_surface(ro: vec3<f32>, rd: vec3<f32>, t: f32, sun: vec3<f32>, time: f32) -> vec3<f32> {
    let p = ro + rd * t;
    let radiance = water_surface(rd, sun, p, time);
    return water_medium(radiance, t) + water_shafts(ro, rd, t, sun, time);
}

// Shades a ray that hit rock.
fn shade_rock(ro: vec3<f32>, rd: vec3<f32>, t: f32, sun: vec3<f32>, time: f32) -> vec3<f32> {
    let p = ro + rd * t;
    let normal = safe_normalize(
        sdf_gradient_at(p, 0.4, reef_args()),
    );
    let albedo = rock_albedo(p, normal);

    // Caustics are projected along the sun onto whatever surface the ray landed on, so a column's
    // side is striped the same way the floor is: the pattern is a property of the light, not of the
    // surface it lands on.
    var projected = p.xz;
    if (abs(sun.y) > 1e-3) {
        projected = p.xz - sun.xz * ((water.floor_y - p.y) / sun.y);
    }
    let caustic = caustic_pattern(projected, time) * water.caustic_strength;

    let diffuse = max(dot(normal, -sun), 0.0);
    let ambient = 0.25 + 0.35 * smoothstep(-0.3, 0.9, normal.y);
    var lit = albedo * (ambient + diffuse * 0.75);
    // Caustics only land on surfaces that face the surface, and are strongest where the sun hits.
    lit = lit + albedo * caustic * clamp(normal.y, 0.0, 1.0) * (0.35 + diffuse);
    // Rim of bioluminescent haze along the silhouette: the reef is covered in the same organisms the
    // fish carry, and it is what keeps the rock readable where the light does not reach.
    let rim = pow(1.0 - clamp(dot(normal, -rd), 0.0, 1.0), 3.0);
    lit = lit + vec3<f32>(0.05, 0.16, 0.2) * rim;

    return water_medium(lit, t) + water_shafts(ro, rd, t, sun, time);
}

// Shades a ray that never hit anything: open water all the way to the fog distance.
fn shade_open_water(ro: vec3<f32>, rd: vec3<f32>, sun: vec3<f32>, time: f32) -> vec3<f32> {
    let far = max(water.surface_y - water.floor_y, 1.0) * MARCH_RANGE;
    let depth_cue = clamp(-rd.y, 0.0, 1.0);
    let base = scene.fog_color * (0.55 + 0.9 * depth_cue) + vec3<f32>(0.01, 0.03, 0.05);
    return water_medium(base, far) + water_shafts(ro, rd, far, sun, time);
}

struct OceanOut {
    // rgb is the shaded medium, a is the hit distance in metres or 0 for a miss. The resolve reads the
    // distance back out of alpha and turns it into `frag_depth` at full resolution.
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> OceanOut {
    let ndc = ocean_ndc(frag);
    let rd = ray_from_clip(ndc);
    let ro = scene.camera.eye;
    let sun = safe_normalize(-scene.light_dir);
    let time = scene.camera.time;

    // The water column ends at the surface plane, so the surface is the march's horizon: anything
    // past it would be above the water.
    let t_surface = plane_t(ro, rd, water.surface_y);
    var t_max = max(water.surface_y - water.floor_y, 1.0) * MARCH_RANGE;
    if (t_surface > 0.0) {
        t_max = min(t_max, t_surface);
    }

    var hit = march(ro, rd, t_max);
    if (hit.kind == KIND_NONE && t_surface > 0.0) {
        hit.t = t_surface;
        hit.kind = KIND_SURFACE;
    }

    var color = vec3<f32>(0.0);
    if (hit.kind == KIND_SOLID) {
        color = shade_rock(ro, rd, hit.t, sun, time);
    } else if (hit.kind == KIND_SURFACE) {
        color = shade_surface(ro, rd, hit.t, sun, time);
    } else {
        color = shade_open_water(ro, rd, sun, time);
    }

    // The marker is hidden by anything the ray hit first, which is what `hit.t` gives exactly.
    color = color + focus_marker(ro, rd, hit.t);

    // The ray's metric distance, or 0 for a miss. `hit.t` is measured along the unit ray, so it is
    // already metres from the eye; the resolve turns it into NDC depth at full resolution.
    var distance = 0.0;
    if (hit.kind != KIND_NONE) {
        distance = hit.t;
    }

    var out: OceanOut;
    out.color = vec4<f32>(color, distance);
    return out;
}

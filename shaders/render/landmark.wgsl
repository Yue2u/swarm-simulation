// === file: render/landmark.wgsl ==============================================================
// Pass: the castle, in both worlds. One instanced draw over the host-placed landmark buffer.
//
// bindings: @group(0) 0:SceneUniform(uniform) 1:array<Boid>(storage, read) 2:WaterParams(uniform)
//                     3:InteractionUniforms(uniform) 4:PostParams(uniform) 5:TerrainParams(uniform)
//                     6:heightfield(texture_2d<f32>, read) 7:array<StaticInstance>(storage, read)
//                     8:SkyParams(uniform) 9:array<StaticInstance>(storage, read, the landmarks)
// buffers:  @location(0) position: vec3<f32>  @location(1) normal: vec3<f32>
//           @location(2) material: f32                                  (the castle mesh)
// draw:     draw_indexed(0..index_count, 0, 0..landmark_count)
// cull:     back faces, like the trees. The mesh is closed, so nothing is lost, and a castle is the
//           other place in the frame where the fill rate is worth a cull.
// depth:    compare LessEqual, write enabled. Drawn after the environment, so the seafloor hides the
//           buried part of a sunken castle and the terrain hides one behind a ridge, and before the
//           agents, so a bird disappears behind a tower.
//
// The mesh is built on the host (`boids_scene::mesh::castle_mesh`) and the instance comes from a
// storage buffer, exactly as the trees do. What is different is the material: the castle's parts are
// told apart by the `material` attribute the mesh carries rather than by a texture or by the normal,
// because a castle is seven materials - stone, slate, rock, iron, cloth, window, trim - and one number
// per face is the cheapest way to say which.
//
// The instance buffer holds one world's landmarks at a time: `renderer.rs` uploads the world's list
// when the world it draws changes, so the same 32-byte write that a key press already costs covers
// this too.

//#include "render/water.wgsl"

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Mesh space, so the fragment shader can band the masonry by height without knowing the scale.
    @location(2) local: vec3<f32>,
    @location(3) material: f32,
}

// The albedo of one of the castle's seven materials, weathered by where the face is.
//
// `local.y` is the height above the ground line in units of the castle's own height, so the same
// expression darkens the foot of a 25 m sunken castle and a 75 m one on a hill.
fn castle_albedo(material: f32, local: vec3<f32>) -> vec3<f32> {
    let stone = vec3<f32>(0.355, 0.340, 0.310);
    let slate = vec3<f32>(0.165, 0.180, 0.205);
    let rock = vec3<f32>(0.235, 0.220, 0.195);
    let iron = vec3<f32>(0.090, 0.095, 0.100);
    let cloth = vec3<f32>(0.430, 0.065, 0.070);
    // Near-black: a window is a hole into the keep, so it stays dark whatever the light does.
    let window = vec3<f32>(0.020, 0.024, 0.030);
    // Dressed stone, a shade lighter and warmer than the wall's rubble, so a course or a surround
    // reads as cut stone laid against rough stone.
    let trim = vec3<f32>(0.470, 0.452, 0.402);

    let m = i32(material + 0.5);
    var color = stone;
    if (m == 1) { color = slate; }
    else if (m == 2) { color = rock; }
    else if (m == 3) { color = iron; }
    else if (m == 4) { color = cloth; }
    else if (m == 5) { color = window; }
    else if (m == 6) { color = trim; }

    // Coursing: a band every few courses of stone, taken from the mesh's own height. It is what keeps
    // a wall this size from reading as one untextured quad, and it costs a `fract`. Only the masonry
    // has it: slate, iron and cloth are smooth.
    let course = 0.9 + 0.1 * smoothstep01(0.2, 0.5, fract(local.y * 26.0));
    // Weathering: the foot of a wall is darker than its parapet, as if the rain had run off it. The
    // ironwork and the pennant are not masonry and do not weather.
    let damp = 0.70 + 0.30 * smoothstep01(0.0, 0.45, local.y);
    let masonry = (m == 0) || (m == 6);
    return color * select(1.0, course, masonry) * select(1.0, damp, m <= 2 || m == 6);
}

@vertex
fn vs_main(
    @location(0) local_pos: vec3<f32>,
    @location(1) local_normal: vec3<f32>,
    @location(2) material: f32,
    @builtin(instance_index) inst: u32,
) -> VsOut {
    let landmark = landmarks[inst];
    let c = cos(landmark.yaw);
    let s = sin(landmark.yaw);
    let rotated = vec3<f32>(
        local_pos.x * c - local_pos.z * s,
        local_pos.y,
        local_pos.x * s + local_pos.z * c,
    ) * landmark.scale;
    let world = landmark.pos + rotated;
    let normal = vec3<f32>(
        local_normal.x * c - local_normal.z * s,
        local_normal.y,
        local_normal.x * s + local_normal.z * c,
    );

    var out: VsOut;
    out.clip = scene.camera.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.normal = normal;
    out.local = local_pos;
    out.material = material;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = safe_normalize(in.normal);
    let sun = safe_normalize(-scene.light_dir);
    let view = scene.camera.eye - in.world;
    let distance = length(view);
    let dir = view / max(distance, 1e-5);

    // The same two lights the ground and the trees use, so a castle and the hill it stands on are lit
    // alike. The `0.9` on the sun term matches `render/tree.wgsl`: stone is not quite as bright as the
    // foliage next to it, and the two passes have to agree about that.
    let albedo = castle_albedo(in.material, in.local);
    let sun_light = sun_irradiance(sun, sky) * max(dot(n, sun), 0.0) / PI;
    let sky_light = sky_radiance(vec3<f32>(0.0, 1.0, 0.0), sun, sky) * (0.35 + 0.65 * max(n.y, 0.0));
    var color = albedo * (sun_light * 0.9 + sky_light);

    // Underwater, the caustics the ocean pass paints on the reef land on the castle too. The pattern
    // is projected along the sun onto the seafloor exactly as `render/ocean.wgsl` projects it, so the
    // stripes on a battlement are the continuation of the stripes on the sand next to it.
    let underwater = 1.0 - clamp(scene.mesh.medium, 0.0, 1.0);
    if (underwater > 0.0) {
        var projected = in.world.xz;
        if (abs(sun.y) > 1e-3) {
            projected = in.world.xz - sun.xz * ((water.floor_y - in.world.y) / sun.y);
        }
        let caustic = caustic_pattern(projected, scene.camera.time) * water.caustic_strength;
        color = color + albedo * caustic * clamp(n.y, 0.0, 1.0) * 0.6 * underwater;
    }

    // The two worlds attenuate differently and the choice is the destination world's, exactly as for
    // the agents: per-channel Beer-Lambert underwater, the aerial haze in the sky.
    let through_water = water_medium(color, distance);
    let through_air = aerial_perspective(color, dir, distance, scene, sky);
    color = mix(through_water, through_air, clamp(scene.mesh.medium, 0.0, 1.0));

    return vec4<f32>(color, 1.0);
}

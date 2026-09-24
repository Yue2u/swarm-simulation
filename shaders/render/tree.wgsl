// === file: render/tree.wgsl ================================================================
// Pass: the sky world's trees. One instanced draw over the scatter pass's instance buffer.
//
// bindings: @group(0) 0:SceneUniform(uniform) 1:array<Boid>(storage, read) 2:WaterParams(uniform)
//                     3:InteractionUniforms(uniform) 4:PostParams(uniform) 5:TerrainParams(uniform)
//                     6:heightfield(texture_2d<f32>, read) 7:array<StaticInstance>(storage, read)
//                     8:SkyParams(uniform)
// buffers:  @location(0) position: vec3<f32>  @location(1) normal: vec3<f32>
//           @location(2) material: f32    (unused here: a tree's two materials are its normals)
// draw:     draw_indexed(0..index_count, 0..tree_count)
// cull:     back faces. The mesh is closed enough for it, and a forest is the one place in the frame
//           where the fill rate is worth spending a cull on.
//
// Unlike the agents, a tree has a real vertex buffer: the mesh is generated once on the CPU as a
// fractal (`boids_scene::mesh::tree_mesh`) and never changes, so there is nothing to gain from
// generating it per vertex and something to lose in readability.
//
// The instance data comes from the *storage* buffer the scatter pass wrote, indexed by
// `instance_index`, exactly as the agents do: the CPU never touches a per-instance value.

//#include "render/bindings.wgsl"

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tint: vec3<f32>,
}

// Colour of a tree from its variant and the biome it stands on.
//
// The biome term is what keeps a forest from being one flat green: trees on the forest floor are
// deep and cold, the ones straggling onto the dunes are dry and yellow, and a canyon edge leans red.
// `kind` covers the rest: a small per-tree hue and value jitter so a stand reads as individuals.
fn tree_tint(kind: f32, mask: f32, normal_y: f32) -> vec3<f32> {
    let conifer = vec3<f32>(0.045, 0.085, 0.038);
    let broadleaf = vec3<f32>(0.105, 0.16, 0.045);
    let dry = vec3<f32>(0.24, 0.175, 0.07);

    let t = clamp(mask, 0.0, 2.0);
    var color = mix(conifer, broadleaf, kind);
    color = mix(color, dry, clamp(t, 0.0, 1.0) * 0.85);
    // Trunks are a different material, and this is the cheapest honest way to say so: the mesh's
    // normals are horizontal on a trunk and (near) vertical on a canopy, and bark is brown.
    let trunk = 1.0 - smoothstep01(0.35, 0.75, abs(normal_y));
    let bark = vec3<f32>(0.105, 0.075, 0.05) * (0.7 + 0.6 * kind);
    color = mix(color, bark, trunk * 0.9);
    return color;
}

@vertex
fn vs_main(
    @location(0) local_pos: vec3<f32>,
    @location(1) local_normal: vec3<f32>,
    // The static mesh vertex carries a material selector for the castle, which has several materials
    // that its normals cannot separate. A tree's two are its normals, so this is ignored rather than
    // branched on: declaring it keeps the vertex layout shared.
    @location(2) _material: f32,
    @builtin(instance_index) inst: u32,
) -> VsOut {
    let tree = trees[inst];
    let c = cos(tree.yaw);
    let s = sin(tree.yaw);
    let rotated = vec3<f32>(
        local_pos.x * c - local_pos.z * s,
        local_pos.y,
        local_pos.x * s + local_pos.z * c,
    ) * tree.scale;
    let world = tree.pos + rotated;
    let normal = vec3<f32>(
        local_normal.x * c - local_normal.z * s,
        local_normal.y,
        local_normal.x * s + local_normal.z * c,
    );

    var out: VsOut;
    out.clip = scene.camera.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.normal = normal;
    out.tint = tree_tint(tree.kind, tree.mask, normal.y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = safe_normalize(in.normal);
    let sun = safe_normalize(-scene.light_dir);
    let view = scene.camera.eye - in.world;
    let distance = length(view);
    let dir = view / max(distance, 1e-5);

    // The same two lights the ground uses, so a tree and the slope it grows on are lit alike.
    let sun_light = sun_irradiance(sun, sky) * max(dot(n, sun), 0.0) / PI;
    let sky_light = sky_radiance(vec3<f32>(0.0, 1.0, 0.0), sun, sky) * (0.35 + 0.65 * max(n.y, 0.0));
    var color = in.tint * (sun_light * 0.9 + sky_light);

    color = aerial_perspective(color, dir, distance, scene, sky);
    return vec4<f32>(color, 1.0);
}

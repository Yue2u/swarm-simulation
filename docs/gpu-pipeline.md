# GPU pipeline

Everything that crosses the CPU/GPU boundary, and every pass that touches it.

## The layout contract

Every struct here has a Rust twin in `crates/boids-core/src/layout.rs` and a WGSL twin in
`shaders/common/layout.wgsl`. They must agree field for field, byte for byte.
`boids-gpu/tests/gpu/layout.rs` checks that they do, on the device, and is the only thing that can:
Rust's compile-time assertions cannot see the shader, and the shader's layout rules are invisible in
Rust.

The two shapes of 16-byte block used everywhere, so no `vec3` padding trap can exist:

```
(vec3<f32>, f32)      // 16 bytes: a vec3 plus the scalar that fills its alignment
(u32 | f32) * 4       // 16 bytes: four scalars
```

### `Boid` - 48 bytes, array stride 48

| offset | Rust | WGSL |
|---|---|---|
| 0 | `pos: [f32; 3]` | `pos: vec3<f32>` |
| 12 | `species: f32` | `species: f32` |
| 16 | `vel: [f32; 3]` | `vel: vec3<f32>` |
| 28 | `phase: f32` | `phase: f32` |
| 32 | `prev_dir: [f32; 3]` | `prev_dir: vec3<f32>` |
| 44 | `color_seed: f32` | `color_seed: f32` |

`prev_dir` is stored rather than recomputed because the vertex shader needs the *change* in heading to
compute the bank angle, and reading the previous frame's velocity would need either a second buffer or
a longer history.

### `SimParams` - 144 bytes, uniform

| offset | fields |
|---|---|
| 0 | `grid_min: vec3<f32>`, `cell_size: f32` |
| 16 | `grid_dim: vec3<u32>`, `num_boids: u32` |
| 32 | `w_sep`, `w_ali`, `w_coh`, `r_percept` |
| 48 | `r_sep`, `max_speed`, `min_speed`, `max_force` |
| 64 | `dt`, `time`, `sdf_strength`, `sdf_probe` |
| 80 | `bounds_half: vec3<f32>`, `mode: u32` |
| 96 | `wander`, `sep_boost`, `coh_falloff`, `r_safe` |
| 112 | `buoyancy`, `drag`, `env_scale`, `env_floor_y` |
| 128 | `env_id: u32`, three explicit padding scalars |

`sep_boost` and `coh_falloff` are reciprocals of `density_ref` rather than the value itself, because
the shader multiplies by them once per agent per frame and a division there would be paid 100k times.

### `InteractionUniforms` - 48 bytes, uniform

| offset | fields |
|---|---|
| 0 | `ray_origin: vec3<f32>`, `mode: u32` |
| 16 | `focus_point: vec3<f32>`, `radius: f32` |
| 32 | `strength`, `falloff`, `tangent`, `_pad` |

`mode` is `0` off, `1` attract, `2` repel.

### `CameraUniform` - 192 bytes, `MeshParams` - 80, `SceneUniform` - 304

`SceneUniform` embeds the camera and the mesh parameters so the vertex shader reads one uniform
binding. The camera carries both `view_proj` and `inv_view_proj` because the cursor ray on the CPU and
the backdrop ray reconstruction in the shader both need the inverse, and deriving it twice from
different code is how the cursor ends up pointing somewhere else than the sky.

`MeshParams::scale` is applied to the whole mesh in the vertex shader. Without it every agent is drawn
at the same absolute size, which is dust in a world sized for 100k agents and a solid wall in a small
one. The host sets it to a quarter of the perception radius, so an agent always occupies a similar share
of the space it can see.

## Buffers

| buffer | size | usage |
|---|---|---|
| `boids[2]` | `padded_n * 48` | STORAGE, COPY_SRC, COPY_DST |
| `keys[2]` | `padded_n * 8` | STORAGE, COPY_SRC |
| `cell_start` | `num_cells * 4` | STORAGE, COPY_DST |
| `cell_end` | `num_cells * 4` | STORAGE |
| `params` | 144 | UNIFORM, COPY_DST |
| `interaction` | 48 | UNIFORM, COPY_DST |
| `scene` | 304 | UNIFORM, COPY_DST |

`padded_n` is `num_boids` rounded up to a power of two, because the bitonic sort can only sort a
power-of-two count. The padding agents are never integrated (every pass gates on `i < num_boids`) and
are pushed out of the neighbour search by the sort, which gives them the maximum key.

`num_cells` is `grid_dim.x * y * z`, 1,126,140 for the default fish world. The grid is a *direct*
lookup rather than a hash, so there are no collisions and no hash table; the price is two flat arrays
of 4.3 MB, which is a good trade on a desktop GPU.

`cell_start` is filled with `EMPTY_CELL` (`u32::MAX`) at allocation. That is not tidiness: until the
range-building pass exists, a grid search has to find no neighbours rather than read `keys[0..cell_end]`
for an arbitrary cell.

## Bind groups

**Group 0, simulation state.** Stable for the lifetime of the simulation, bound once.

| binding | resource |
|---|---|
| 0 | `params` uniform |
| 1 | `interaction` uniform |
| 2 | `keys` read |
| 3 | `cell_start` read |
| 4 | `cell_end` read |

**Group 1, agent storage.** Swaps every step.

| binding | resource |
|---|---|
| 0 | `boid_src` read |
| 1 | `boid_dst` read_write |

Two bind groups exist, one per parity, created once at startup. Rebuilding a bind group per frame
would work and would allocate a driver object sixty times a second for no reason.

`boid_src` and `boid_dst` are always distinct buffers. That is the entire reason the simulation is
race-free: a pass only reads `src` and only writes `dst`, so the order in which invocations run does not
matter and no barrier is needed.

**Group 0, rendering.** A separate layout for the render passes: `SceneUniform` plus the agent array,
with the agent buffer bound as a storage buffer rather than a vertex buffer. Both parities are bound up
front for the same reason as above.

## Passes

### Today: `integrate_naive` (all-pairs)

```
bindings: group 0 (all), group 1 (src, dst)
workgroup: 256 x 1 x 1
dispatch:  ceil(num_boids / 256)
```

Each invocation walks every other agent, applies the squared-distance reject, and accumulates. Exact,
and O(N^2): it exists to validate the force model against the CPU reference and to run small swarms
where it is genuinely faster than a grid plus its sort.

### Day 2: the grid

```
clear_cells    dispatch ceil(num_cells / 256)   cell_start[i] = EMPTY_CELL
hash           dispatch ceil(padded_n / 256)    keys[i] = {cell_index(pos_i), i}
bitonic sort   one dispatch per (k, j) stage    sort keys by key
build_ranges   dispatch ceil(padded_n / 256)    derive cell_start / cell_end from runs of equal keys
integrate_grid dispatch ceil(num_boids / 256)   27-cell neighbour search
```

The sort's stage parameters go through a `var<immediate>` block and `ComputePass::set_immediates`,
which `wgpu` 30 provides and which replaces the dynamic-offset uniform array the classic
implementation needs. One pipeline, one bind group, 153 dispatches with different constants.

`integrate_grid` and `integrate_naive` share the entire force model (`shaders/sim/forces.wgsl`), so the
only thing that can differ between them is which neighbours they find, and a test can compare the two
directly.

### Render passes

```
1. background   draw(0..3, 0..1)          no vertex buffer, no depth write, compare Always
2. agents       draw(0..66, 0..num_boids) mesh from vertex_index, transform from the agent buffer
```

Neither pass has a vertex buffer. The agent mesh is a pure function of `vertex_index` (22 triangles: a
body, a tail fin, and two wing triangles that are degenerate for fish), and the per-agent transform
comes from the storage buffer indexed by `instance_index`.

The vertex shader builds the orientation basis from the agent's velocity:

```
fwd   = normalize(vel)
right = normalize(cross(reference, fwd))    reference is Y, or X when fwd is nearly vertical
up    = cross(fwd, right)
world = pos + mat3(right, up, fwd) * (animated_local * scale)
```

Fragment shading derives the normal from screen-space derivatives of the world position rather than
interpolating per-vertex normals. It is exact for the triangle actually rasterised, including the
animated deformation, needs no normal data in the mesh function, and cannot produce a wrong normal on a
degenerate triangle because such a triangle has no pixels.

## Reference-space conventions

| thing | convention |
|---|---|
| clip space | `wgpu`, so NDC depth in `[0, 1]`, Y up. `glam::camera::rh::proj::directx::perspective` |
| world | right-handed, Y up, metres |
| camera | orbit around a target; `u32` viewport in physical pixels |
| the screen-space Y flip | applied exactly once, in `ray_from_ndc` on the CPU and in the backdrop's deprojection on the GPU |
| blend | none. The backdrop and the agents both write opaque |
| winding | no culling; the fins and wings are single-sided by construction |

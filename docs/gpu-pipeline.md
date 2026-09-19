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

`sep_boost` is stored as the reciprocal of `density_ref` because the shader multiplies by it once per
agent per frame and a division there would be paid 100k times. `coh_falloff` is the density-feedback
exponent `density_gain`: 0 disables the feedback, 1 is proportional, and larger values swing harder.

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

| buffer | size @100k | usage |
|---|---|---|
| `boids[2]` | `padded_n * 48` (2 x 6.0 MB) | STORAGE, COPY_SRC, COPY_DST |
| `keys[2]` | `padded_n * 8` (2 x 1.0 MB) | STORAGE, COPY_SRC |
| `cell_start` | `num_cells * 4` (1.3 MB) | STORAGE, COPY_SRC, COPY_DST |
| `cell_end` | `num_cells * 4` (1.3 MB) | STORAGE, COPY_SRC |
| `params` | 144 | UNIFORM, COPY_DST |
| `interaction` | 48 | UNIFORM, COPY_DST |
| `scene` | 304 | UNIFORM, COPY_DST |
| `water` | 48 | UNIFORM, COPY_DST |
| `post` | 48 | UNIFORM, COPY_DST |

`padded_n` is `num_boids` rounded up to a power of two, because the bitonic sort can only sort a
power-of-two count. `cell_start` is filled with `EMPTY_CELL` (`u32::MAX`) at allocation, and
`clear_cells` re-fills it every grid frame. An empty cell must read as empty rather than as the range
left over from the frame before, or the search would scan a previous frame's keys as neighbours.
`cell_end` is only meaningful where `cell_start` is not `EMPTY_CELL`, and `build_ranges` overwrites
it from scratch, so it needs no initial value.

`cell_start` and `cell_end` carry `COPY_SRC` for the test suite, which reads the grid back to check
the invariants `integrate_grid` searches under. Nothing in a frame copies either buffer.

`num_cells` is `grid_dim.x * y * z`, derived from the world's bounds and perception radius by
`GridDims::for_domain` (cell size is pinned to `r_percept`, capped at 256 cells per axis). The default
fish world is 91 x 40 x 91 = 331,240 cells, 1.3 MB per range array. The grid is a *direct* index
rather than a hash, so there are no collisions and no hash table, which `ADR-0003` records as a
deliberate trade.

## Render targets

The scene is drawn into attachments the renderer owns, not into the swapchain; only the composite
writes the swapchain (`ADR-0004`).

| target | format | size @1440p | usage |
|---|---|---|---|
| HDR scene | `Rgba16Float` | 12.6 MB | RENDER_ATTACHMENT, TEXTURE_BINDING |
| depth | `Depth32Float` | 7.9 MB | RENDER_ATTACHMENT (`frag_depth` written by the ocean resolve) |
| ocean half-res | `Rgba16Float` | 720p | RENDER_ATTACHMENT, TEXTURE_BINDING (alpha = hit metres) |
| bloom level 0..2 | `Rgba16Float` | 720p / 360p / 180p | RENDER_ATTACHMENT, TEXTURE_BINDING |

`Rgba16Float` rather than `Rgba32Float`: the scene contains values far above 1.0 (caustics, the surface,
bioluminescence) and half-float keeps both the range and the *exponent*, at half the bandwidth. The
format is chosen once, in `targets.rs`, and every geometry pipeline is built against it. `Depth32Float`
rather than `Depth24PlusStencil8`: nothing uses stencil, and the ocean raymarch writes real distances
in the hundreds of metres into this buffer, so the extra precision matters. The ocean target is half
the scene per axis: the raymarch is the most expensive pass and its result is smooth, so its alpha
channel carries the hit distance in metres and `render/ocean_resolve.wgsl` turns that into full-size
colour and `frag_depth`. The bloom pyramid is recreated whenever the target size changes, with every
bind group that references a level, so a resize cannot leave half the chain pointing at a previous
size; the ocean resolve's bind group is rebuilt the same way.

## Bind groups

**Group 0, simulation state.** Stable for the lifetime of the simulation, bound once.

| binding | resource |
|---|---|
| 0 | `params` uniform |
| 1 | `interaction` uniform |
| 2 | `cell_start` read_write |
| 3 | `cell_end` read_write |

Both range arrays are declared `read_write` because `build_ranges` writes them through this group; the
passes that only read them (`integrate_grid`) pay nothing for that. A declaration describes what the
group *may* do, not what each pass does.

**Group 1, the working pair.** Changed shape per pass, swaps every step.

| binding | resource |
|---|---|
| 0 | `src` read |
| 1 | `dst` read_write |
| 2 | `keys` read |

One layout serves three shapes, because they differ only in element type, which the layout does not
describe: agents for `integrate`, agents-to-keys for `hash`, keys for the sort. Bind groups are built
once per parity at startup, one set per shape, and the host selects between them. Rebuilding a bind
group per frame would work and would allocate a driver object sixty times a second for no reason.

`src` and `dst` are always distinct buffers. That is the entire reason the simulation is race-free: a
pass only reads `src` and only writes `dst`, so the order in which invocations run does not matter and
no barrier is needed inside a pass.

`keys` is binding 2 here rather than in group 0 because the sort writes it. A bind group may not bind
one buffer read-only and read-write at once, and group 0 is bound by every pass including the 153 sort
stages, so a read-only key array there would make the sort impossible. Pinning the sorted result to
`keys[0]` (see `KeyPlan` below) means every consumer binds the same key buffer, and the sort's
ping-pong stays entirely inside group 1.

**Group 0, rendering.** A separate layout for the scene passes:

| binding | resource | visible to |
|---|---|---|
| 0 | `scene` uniform (`SceneUniform`) | vertex + fragment |
| 1 | `boids` storage array, read | vertex + fragment |
| 2 | `water` uniform (`WaterParams`) | fragment |
| 3 | `interaction` uniform (`InteractionUniforms`) | fragment |
| 4 | `post` uniform (`PostParams`) | fragment |

The agent buffer is bound as a storage buffer rather than a vertex buffer. Both parities are bound up
front for the same reason as the simulation pair. `water`, `interaction` and `post` live here because
every pass that shades the medium needs some of them, and one group means one `set_bind_group` per pass;
`post` is the *same buffer* the post chain binds in its own group, so there is one upload per frame and
one struct that can be stale (`boids-render/src/scene.rs`).

**Group 0, post.** The bloom and composite passes use their own group, since they sample textures and
read only `PostParams`:

| binding | resource |
|---|---|
| 0 | `post` uniform (`PostParams`) |
| 1 | linear, clamp-to-edge sampler |
| 2 | `post_src` texture (`texture_2d<f32>`) |
| 3 | `post_bloom` texture (the composite's bloom level 0; the bloom passes repeat `post_src`) |

One layout covers all four post pipelines (`fs_bright`, `fs_down`, `fs_up`, `fs_composite`). The second
texture slot is inert for the bloom passes, which bind their source to both slots rather than introduce
a second three-binding layout for one unused field.

## The bitonic sort and `KeyPlan`

The sort is one entry point (`sort_step`) dispatched once per bitonic stage, with the stage parameters
`(j, k, n_padded)` carried in a `var<immediate>` block and `ComputePass::set_immediates`. One pipeline,
one bind group shape, 153 dispatches with different constants; the alternative, a dynamic-offset
uniform array, needs a `set_bind_group` per stage and a buffer for what is 12 bytes of data.

For `padded_n = 2^m` the stage count is `m * (m + 1) / 2`: `m = 17` for the 131,072-key target, so 153
stages, one compute pass each. The passes are separate because a stage reads what the previous stage
wrote; `wgpu` inserts the memory barrier between passes and none between two dispatches inside one.

Each stage ping-pongs `keys[0]` and `keys[1]`, so which buffer holds the sorted result depends on the
parity of the stage count: odd stages end in `keys[hash_dst]`, even in `keys[1 - hash_dst]`.
`KeyPlan` (in `boids-gpu/src/sim.rs`) sets `hash_dst` to that parity so the hash pass writes the buffer
that leaves the answer in `keys[0]` for either parity. This is what lets `build_ranges` and
`integrate_grid` read one fixed buffer with no per-frame choice; the unit tests assert the invariant for
every padded count the app can reach.

The bitonic network is not stable: agents within one cell come out in an arbitrary order. Nothing
downstream depends on the order *within* a cell - `build_ranges` only looks at run boundaries, and the
neighbour search sums over a cell in whatever order it finds it - so the sort's contract is "partitioned
by key, ascending" and not "stable".

## Passes

Two strategies share the whole force model (`shaders/sim/forces.wgsl`) and differ only in which
neighbours they find, so a device test can compare them directly. `Strategy::for_count` picks one from
the agent count; `--strategy` overrides it for an A/B comparison. Both are exact (up to floating-point
summation order); the grid is the one that scales.

### The grid (production path)

`record_grid_prep` records P0-P3, then `record_integrate` records P4. Every pass is a separate compute
pass, workgroup 256, so the barrier between passes is the synchronisation.

```
P0 clear_cells    dispatch ceil(num_cells / 256)         cell_start[c] = EMPTY_CELL
P1 hash           dispatch ceil(padded_n / 256)          keys[dst][i] = {cell_index(pos_i), i}
P2 sort_step      dispatch ceil(padded_n / 256), once    one bitonic stage per (k, j)
                  per stage (m*(m+1)/2 = 153 for 131,072)
P3 build_ranges   dispatch ceil((padded_n + 1) / 256)    cell_start / cell_end from equal-key runs
P4 integrate_grid dispatch ceil(num_boids / 256)         27-cell neighbour search
```

The guards and the one-extra-invocation in P3 are the parts worth stating, because each exists to
prevent a read of memory that is out of range or out of date:

* `clear_cells` runs first because a cell that was populated last frame and is empty now must stop
  being findable; `build_ranges` only writes ranges for cells that contain agents, so without the
  clear the search would use last frame's range.
* `hash` writes `PAD_KEY` (the `EMPTY_CELL` value, larger than any cell index) for the padding tail,
  and `build_ranges` ignores any key `>= num_cells`. That is what keeps both the padding and any
  stale entry out of the range arrays.
* `build_ranges` dispatches one invocation more than it has keys: a run is closed by the first entry
  of a *different* key, so the run that ends at the last element needs a terminator.
* `integrate_grid` skips a cell whose `cell_start == EMPTY_CELL`; without that guard the range
  `[EMPTY_CELL, cell_end)` would be walked as a real range.

The integration pass clamps each agent's cell coordinate into the grid, so an agent pushed outside the
domain is found by the border cells instead of falling into an out-of-range hole. That is deliberate
and is why the grid and all-pairs comparisons in the test suite spawn away from the border.

### `integrate_naive` (all-pairs)

```
bindings: group 0 (all), group 1 (src, dst, keys)
workgroup: 256 x 1 x 1
dispatch:  ceil(num_boids / 256)
```

Each invocation walks every other agent, applies the squared-distance reject, and accumulates. Exact
and O(N^2). It exists to validate the force model against the CPU reference, to run small swarms where
it is genuinely faster than a grid plus its sort, and as the reference the grid is compared against.

### Render passes

One frame is one half-size ocean pre-pass, one scene render pass, and the post chain. The environment
and the agents share the scene pass and its depth attachment, which is what lets a fish sit behind a
column:

```
ocean pre-pass (half-res HDR, no depth, Fish only)
  0. raymarch      ocean.wgsl           draw(0..3, 0..1)   rgb = medium, a = hit metres
scene pass (full HDR target + depth, depth cleared to 1.0)
  1. environment   Fish:  ocean_resolve.wgsl draw(0..3, 0..1)  compare Always, frag_depth written
                   Birds: background.wgsl    draw(0..3, 0..1)  compare Always, no depth write
  2. ground        terrain.wgsl              draw(0..segments^2*6, 0..1)  Birds only
  3. trees         tree.wgsl                 draw(0..tree_count*idx, 0..1)  Birds only
  4. agents        boid.wgsl                 draw(0..66, 0..num_boids)  compare LessEqual, depth write
post chain (no depth)
  5. bloom bright  draw(0..3, 0..1)   hdr -> bloom[0]
  6. bloom down    draw(0..3, 0..1)   bloom[i-1] -> bloom[i]
  7. bloom up      draw(0..3, 0..1)   bloom[i] -> bloom[i-1], additive
  8. composite     draw(0..3, 0..1)   hdr + bloom[0] -> swapchain
```

The underwater environment writes `frag_depth` from the raymarched hit (`ADR-0005`), so the agent
pipeline's `LessEqual` test against the 1.0 clear rejects a fish behind rock and keeps one in front of
the seafloor. The distance travels through the half-size pre-pass in the colour target's alpha channel
because a depth texture cannot be sampled portably on every backend, and the resolve derives NDC depth
from the metric distance at full precision. It runs first in the scene pass and with
`CompareFunction::Always`, so it overwrites the clear rather than testing against it. The sky backdrop
writes no depth; the terrain and the trees do, so a bird depth-tests against the ridge and the forest as
well as against the flock.

No pass has a vertex buffer. The agent mesh is a pure function of `vertex_index` (22 triangles: a
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
| blend | none in the scene pass; the bloom upsample adds (`One`/`One`), the composite replaces |
| winding | no culling; the fins and wings are single-sided by construction |

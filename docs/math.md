# The mathematics

Every formula here has an implementation on both sides: `crates/boids-core/src/reference.rs` (the
specification) and `shaders/sim/*.wgsl` (the implementation). Where a formula has a non-obvious form,
the reason is given, because in every case the obvious form is wrong in a way that looks fine.

Units are SI: metres, seconds, m/s, m/s^2.

## 1. Neighbour search

An agent sees every other agent within `r_percept`, and separates from those within `r_sep`.

The grid cell size is `2 * r_percept / 3`, which makes the 27 cells around an agent sufficient: a cell
is one third of a perception diameter across in every direction, so anything within `r_percept` is at
most one cell away. That relationship is the entire point of the grid, and it is why cell size is
*derived* from the perception radius rather than configured independently.

Within a cell, the squared-distance test is still applied. The grid removes the need to scan the whole
population; it does not make the neighbour test free, because a cube of 27 cells contains far more
agents than the perception sphere inside it.

## 2. The three Reynolds forces

Accumulated in one pass over the neighbours:

**Separation.** `F_sep = sum_j (p_i - p_j) / |p_i - p_j|^2` for `|p_i - p_j| < r_sep`.

The `1/r^2` denominator is deliberate. Summed over neighbours, the *unit* repulsion vector times
`1/|d|` gives a force that grows as neighbours get closer, without the singularity of a bare
`1/|d|^2` repulsion. The squared distance is already computed for the range test, so the division is
free.

**Alignment.** `F_ali = normalize(mean_j v_j) * v_max - v_i`.

**Cohesion.** `F_coh = normalize(mean_j p_j - p_i) * v_max - v_i`.

Both use Reynolds' *steering* form: a desired velocity minus the current velocity. The obvious
alternative, summing raw offsets, grows without bound as a flock compresses and makes a dense flock
explode. The steering form saturates at `v_max`, so the flock is stable at any density.

If the neighbour set is empty, alignment and cohesion are skipped entirely: `normalize(0)` is
undefined, and an agent with no neighbours has nothing to align with.

## 3. Density-adaptive weights

```
density   = count * sep_boost          where sep_boost = 1 / density_ref
w_sep_eff = w_sep * (1 + density)
w_coh_eff = w_coh * exp(-density)
```

These two together are what prevent the classic boids failure mode, where cohesion wins inside a
crowded core, the flock collapses to a point, and then separation throws it apart again in a cycle.
Pushing apart faster than linearly while pulling together exponentially decays means the equilibrium
density is stable rather than oscillating.

`sep_boost` is stored as a reciprocal because the shader multiplies by it once per agent per frame.

## 4. Environment avoidance

The field is a signed distance: negative inside the solid, positive outside, `|SDF(p)|` approximately
the distance to the nearest surface.

**Gradient by central differences:**

```
grad SDF(p) ~ ( SDF(p + e_k) - SDF(p - e_k) ) / (2*eps)   for k in {x, y, z}
```

with `eps = 0.5 * cell_size`. Smaller and the field's own high-frequency content dominates the
difference; larger and thin obstacles are smoothed out of existence. Half a cell is the largest scale
that still resolves anything the grid can distinguish.

**Avoidance force:**

```
n      = normalize(grad SDF(p))
d      = SDF(p)
F_sdf  = k_avoid * smoothstep(r_safe, 0, d) * n
```

`smoothstep(r_safe, 0, d)` rather than `k / d`: a reciprocal blows up inside the solid, so an agent
that clips a corner during a frame hitch is launched across the world. The smoothstep saturates at
`k_avoid` instead.

**The safety margin is derived, not chosen:**

```
stopping_distance = v_max^2 / (2 * k_avoid)
r_safe            = max(0.4 * r_percept, 1.25 * stopping_distance)
```

An agent entering the smoothstep band already travelling faster than the band can arrest will pass
through the surface. Sizing `r_safe` from the stopping distance is the only way to make "agents do not
enter terrain" a property of the model rather than a hope. On the fish preset this is the difference
between `r_safe = 1.4` (tunnels) and `r_safe = 2.9` (stops).

**The forward probe, which is what makes agents go around obstacles:**

```
p_ahead = p + v * (sdf_probe / |v|)
if SDF(p_ahead) < r_safe:
    v_tangent = v - n * dot(v, n)
    F_slide   = k_avoid * normalize(v_tangent)
```

A pure repulsion force has no way to break the symmetry that keeps an agent pinned against a wall: it
pushes in, is pushed out along the same line, and oscillates. Redirecting along the component of
velocity tangential to the surface gives it a direction to escape in. Without this term, a swarm
pressed against a reef by cohesion stays there.

## 5. The step, in order

```
1. neighbour pass over the snapshot        separation / alignment / cohesion
2. density-adaptive weights
3. force accumulation                      boids + wander + bounds + SDF + cursor
4. a  = clamp_len(F, max_force)
5. v' = v + a * dt
6. v' = v' * exp(-drag * dt)
7. v' = clamp_len_range(v', min_speed, max_speed)
8. p' = p + v' * dt,  prev_dir = normalize(v')
```

Steps 6 and 7 are in this order on purpose. Clamping first and applying drag afterwards lets drag pull
agents below `min_speed` permanently, which looks like a swarm slowly going to sleep.

All reads come from a snapshot taken at the start of the step. On the GPU that is achieved by
ping-ponging between two buffers, which is why the buffers are separate rather than one buffer with
in-place updates: in-place would make the result depend on evaluation order.

**Soft world bounds.** The force ramps up over the outer 15% of each axis of the box:

```
to_wall      = half[a] - |p[a]|
F_bounds[a]  = -sign(p[a]) * (1 - to_wall / margin)   when to_wall < margin,  margin = 0.15 * half[a]
F_bounds    *= 4 * v_max
```

An invisible wall is what makes a swarm look like a screensaver; a ramp lets agents bank away from the
boundary.

**Wander** is a per-agent pseudo-random nudge, resampled at 0.7 Hz:

```
tick        = u32(time * 0.7)
base        = index * 0x9e3779b9 XOR tick * 0x85ebca6b
F_wander    = vec3(hash_signed(base), hash_signed(base ^ 0x27d4eb2d), hash_signed(base ^ 0x165667b1)) * w_wander
```

Deliberately *not* normalized. A variable-magnitude nudge is visually indistinguishable from a
normalized one, and skipping the normalization removes a `normalize(0)` NaN case that would otherwise
be a CPU/GPU divergence risk. The hash is integer, not `fract(sin(x))`, because integer arithmetic
wraps identically on both sides and `sin` does not.

## 6. The cursor ray

**Screen to world.** The pixel is converted to NDC with the Y flip applied here and nowhere else:

```
ndc = (2 * mouse_x / width - 1,  1 - 2 * mouse_y / height)
```

Then the ray is reconstructed from the inverse view-projection rather than from camera basis vectors:

```
p_near = inv_view_proj * vec4(ndc, 0, 1)
p_far  = inv_view_proj * vec4(ndc, 1, 1)
p_near = p_near.xyz / p_near.w          <- the perspective divide is required
p_far  = p_far.xyz / p_far.w
ray    = { origin: p_near, dir: normalize(p_far - p_near) }
```

The divide is the step that is easy to omit, and omitting it produces a ray that is wrong by a
position-dependent factor rather than obviously broken.

**Ray against the interaction plane.** The plane faces the camera and sits at
`p0 = eye + forward * d_focus`:

```
t     = dot(p0 - ray.origin, n) / dot(ray.dir, n),   n = -forward
focus = ray.origin + t * ray.dir
```

Rejected when `|dot(ray.dir, n)| < 1e-6` (parallel) or when `t <= 0` (the plane is behind the camera).
Both cases mean "the cursor points at nothing", and the correct response is to stop influencing the
swarm rather than to place the focus point at an absurd distance, which would drag the whole swarm into
a line.

The focus point is finally clamped to the world bounds. A cursor aimed at the horizon otherwise places
the attractor kilometres away.

## 7. Cursor influence

```
g   = focus - p_i
r   = |g|
w(r) = clamp(1 - r / R, 0, 1)^falloff
attract: F = +k_a * w(r) * normalize(g)
repel:   F = -k_r * w(r)^1.5 * normalize(g) + k_t * w(r) * normalize(cross(Y, g))
```

The `w^1.5` in the repel term makes the falloff steeper than the attract term, so the boundary of the
influence region is sharp: agents outside are unaffected, agents inside are thrown out decisively.
A softer falloff produces a slow drift that reads as the swarm being nudged rather than repelled.

The tangential term is the difference between "the swarm is blown away from the cursor" and "the swarm
curves around the cursor". Its sign is taken from `dot(v, tangent)` so it always reinforces the
direction the agent is already turning in; without that, half the swarm swirls each way and the
result is jitter instead of an orbit.

## 8. Choice of quantities for validation

The `order_parameter` is the primary scalar for comparing two implementations:

```
order = | mean_i ( v_i / |v_i| ) |
```

`0` is fully disordered, `1` is a single heading. It is invariant under translation and rotation, so
floating-point differences in position do not move it, and it changes immediately when a force
direction is wrong. Comparing raw positions after a long run would be meaningless, because an N-body
system is chaotic: after 600 steps the GPU and CPU agree on the order parameter to within 0.01 while
individual agents have diverged.

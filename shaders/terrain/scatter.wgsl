// === file: terrain/scatter.wgsl ==============================================================
// Pass: places the trees of the sky world on the terrain. One dispatch at startup.
//
// bindings: @group(0) 0:TerrainParams(uniform) 1:heightfield(texture_2d<f32>, read)
//                     2:trees(storage, read_write array<TreeInstance>)
//                     3:counters(storage, read_write Counters)
// workgroup: 64 x 1 x 1
// dispatch:  ceil(candidates^2 / 64)
//
// WHAT IT PRODUCES
//   Up to `tree_capacity` instances, appended in completion order, each with a position on the
//   ground, a scale, a yaw and the biome mask it stands on. The host reads `counters.count` back once
//   after the dispatch and draws exactly that many.
//
// WHY A JITTERED LATTICE RATHER THAN TRUE BLUE NOISE
//   Blue noise needs either a precomputed texture or a void-and-cluster pass, and both are more
//   machinery than a forest needs. What a plain lattice gets wrong is the *alignment*: trees in rows.
//   Jittering each candidate inside its own cell removes the rows at no cost, and the density
//   modulation below removes the rest of the regularity by making acceptance a function of the biome
//   field. The remaining artefact - a minimum spacing equal to the cell size - is smaller than a
//   tree.
//
// COMPETITION: every invocation writes at most one instance, to the slot `atomicAdd` hands it, and
// reads only the map. There is no ordering between invocations, which is why the acceptance test is
// per-candidate rather than a neighbour query: an order-dependent rule would make the forest depend
// on scheduling.

//#include "common/sdf.wgsl"
//#include "common/heightfield.wgsl"

@group(0) @binding(0) var<uniform> terrain: TerrainParams;
@group(0) @binding(1) var heightfield: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> trees: array<TreeInstance>;
@group(0) @binding(3) var<storage, read_write> counters: Counters;

struct Counters {
    count: atomic<u32>,
}

// Fraction of the forest that actually gets a tree. The candidates are ~11 metres apart at the
// default world size and a tree is ~13 metres tall, so a high rate here would produce a solid canopy.
const TREE_COVERAGE: f32 = 0.35;

// Minimum cosine of the slope a tree will stand on: 0.72 is about 44 degrees, past which the ground
// is bare rock in every biome. Real treelines sit near 40.
const TREE_MIN_NORMAL_Y: f32 = 0.72;

// Baseline of the slope estimate, in map texels. A tree is metres across, so its footing is judged
// over metres: one texel is closer to the noise's own roughness than to a slope a tree would feel.
const TREE_SLOPE_TEXELS: f32 = 4.0;

// Hash of a scatter candidate and a stream selector, in [0, 1).
fn scatter_hash(index: u32, stream: u32) -> f32 {
    return hash_unit(index * 0x9e3779b9u ^ stream * 0x85ebca6bu ^ 0x27d4eb2du);
}

@compute @workgroup_size(64, 1, 1)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let candidates = terrain.tree_candidates;
    let i = gid.x;
    if (candidates == 0u || i >= candidates * candidates) {
        return;
    }

    // Jittered lattice: one candidate per cell, placed uniformly inside it.
    let cell = vec2<u32>(i % candidates, i / candidates);
    let jitter = vec2<f32>(scatter_hash(i, 1u), scatter_hash(i, 2u));
    let uv = (vec2<f32>(cell) + jitter) / f32(candidates);
    let world = terrain.min_xz + uv * terrain.size_xz;

    let sample = terrain_sample(heightfield, terrain, world, TREE_SLOPE_TEXELS);
    let normal = slope_normal(sample.slope);
    if (normal.y < TREE_MIN_NORMAL_Y) {
        return;
    }

    // Forest is mask near 0, dunes is around 1 and canyon around 2. The mask's distribution is not
    // uniform - the three normalised hashes put the median at about 1.0, i.e. half the world is
    // dunes - so the ramp is positioned where the *biome* boundary is rather than at the middle of
    // the range: full density inside the forest band, nothing by the time the ground is sand. The
    // ground shader's palette uses the same mask, so a tree cannot stand on ground that is drawn as
    // bare rock.
    let density = smoothstep01(1.05, 0.45, sample.mask);
    if (scatter_hash(i, 3u) > density * TREE_COVERAGE) {
        return;
    }

    let slot = atomicAdd(&counters.count, 1u);
    if (slot >= arrayLength(&trees)) {
        return;
    }

    var tree: TreeInstance;
    tree.pos = vec3<f32>(world.x, sample.height, world.y);
    tree.scale = terrain.tree_height * (0.72 + 0.65 * scatter_hash(i, 4u));
    tree.yaw = scatter_hash(i, 5u) * TAU;
    tree.kind = scatter_hash(i, 6u);
    tree.mask = sample.mask;
    tree.pad1 = 0.0;
    trees[slot] = tree;
}

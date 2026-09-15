//! Checks the spatial grid: the sort it is built on, the ranges it exposes, and whether the
//! neighbour search that consumes it finds what the all-pairs search finds.
//!
//! The three checks are deliberately at three different levels, because a grid failure can be in any
//! of them and they fail very differently:
//!
//! * **`sort/matches_cpu`** compares the sorted key array against `boids_core::reference::sort_keys`
//!   on the same input. A failure here is in `hash`, in the bitonic network, or in the host's stage
//!   sequence, and it names the first index that differs.
//! * **`grid/ranges_are_consistent`** checks the invariants `integrate_grid` is allowed to assume:
//!   ranges partition exactly the live agents, empty cells are `EMPTY_CELL`, and a range that claims
//!   to hold a cell holds only agents of that cell. A failure here is in `build_ranges` or in the
//!   clearing between frames.
//! * **`grid/matches_naive`** is the one that matters: the grid and all-pairs integrations run the
//!   same force model on the same swarm, so any difference is a difference in the neighbour set. It
//!   is the only check that would catch a grid that is internally consistent and still finds the
//!   wrong agents.
//!
//! The bitonic network is not stable, so nothing here depends on the order *within* a cell: keys are
//! compared element by element, and the agents inside one cell are compared as sets.

use boids_core::config::SimConfig;
use boids_core::layout::{InteractionUniforms, KeyVal, SimParams, EMPTY_CELL};
use boids_core::math::{mean_speed, order_parameter};
use boids_gpu::context::GpuContext;
use boids_gpu::sim::Strategy;
use glam::Vec3;

use crate::common::{run_gpu_steps, run_grid_prep, setup, Check};

/// A world whose grid is larger than the box the agents live in.
///
/// This is what makes the grid-vs-naive comparison exact. `integrate_grid` clamps an agent's cell
/// coordinate into the grid and searches the 27 cells around the clamped one, so an agent in a border
/// cell searches a *smaller* neighbourhood than all-pairs does. That is intended behaviour - the
/// alternative is agents that leave the domain and are never seen again - but it means a border agent
/// would legitimately find different neighbours and the comparison would fail for the wrong reason.
///
/// Spawning inside the inner 60% of a grid that extends twice as far puts every agent at least one
/// cell away from the border, so both strategies see the same neighbourhood and the only thing that
/// can differ is the correctness of the grid itself.
fn roomy_config(n: usize, r_percept: f32, neighbours: f32) -> SimConfig {
    let mut cfg = SimConfig::dense(n, r_percept, neighbours);
    cfg.env = boids_core::layout::EnvironmentKind::None;
    let spawn_half = cfg.bounds_half * 0.6;
    cfg.bounds_half = spawn_half;
    cfg.grid = boids_core::config::GridDims::for_domain(
        spawn_half * 2.0,
        cfg.grid.cell_size,
        64,
    );
    cfg
}

/// The keys the CPU expects for a swarm, padded to `padded_n` exactly as the hash pass pads them.
fn expected_keys(cfg: &SimConfig, swarm: &[boids_core::layout::Boid], padded_n: u32) -> Vec<KeyVal> {
    let mut keys: Vec<KeyVal> = (0..padded_n)
        .map(|i| KeyVal {
            // The padding entries carry `PAD_KEY`, which is `EMPTY_CELL`'s value: larger than any cell
            // index, so they sort past every live agent and `build_ranges` ignores them.
            key: EMPTY_CELL,
            val: i,
        })
        .collect();
    for (i, boid) in swarm.iter().enumerate() {
        keys[i].key = cfg.grid.cell_of(Vec3::from(boid.pos));
    }
    boids_core::reference::sort_keys(&mut keys);
    keys
}

/// Compares the GPU's sorted key array against the CPU's, for several agent counts.
///
/// The counts are chosen for what they exercise rather than for coverage of the sort alone:
///
/// * 1024 and 4096 are powers of two, so `padded_n == num_boids` and there is no padding at all,
/// * 1000, 3000 and 5000 pad to 1024, 4096 and 8192, so the padded tail - the entries the sort must
///   push to the end and the range builder must ignore - is exercised,
/// * 1000 pads to `m = 10` (55 stages, odd) and 3000 to `m = 12` (78 stages, even), which are the two
///   parities of the stage count. The host aims the hash pass at a different key buffer for each, and
///   a mistake there would write the sorted result into the buffer nothing reads.
pub fn sort_matches_cpu(ctx: &GpuContext) -> Check {
    let mut problems = Vec::new();
    for (n, seed) in [(1024usize, 1u64), (1000, 2), (4096, 3), (3000, 4), (5000, 5)] {
        let cfg = roomy_config(n, 6.0, 18.0);
        let params: SimParams = cfg.to_params(0.0, cfg.dt);
        let (res, pipes, swarm) = setup(ctx, &cfg, seed);
        res.write_params(&ctx.queue, &params);
        res.write_interaction(&ctx.queue, &SimConfig::idle_interaction());

        let padded_n = pipes.padded_n();
        let expected = expected_keys(&cfg, &swarm, padded_n);
        let got = run_grid_prep(ctx, &res, &pipes).keys;

        let plan = pipes.key_plan();
        let context = format!(
            "N={n} padded={padded_n} stages={} hash_dst=keys[{}]",
            plan.stages, plan.hash_dst
        );

        if got.len() != expected.len() {
            problems.push(format!(
                "{context}: read back {} keys, expected {padded_n}",
                got.len()
            ));
            continue;
        }

        // 1. The key sequence must match element by element. Ties within a cell are allowed to be in
        //    any order, and equal keys cannot be distinguished by this test at all.
        if let Some(i) = (0..got.len()).find(|&i| got[i].key != expected[i].key) {
            let lo = i.saturating_sub(4);
            let hi = (i + 5).min(got.len());
            problems.push(format!(
                "{context}: key mismatch at {i}: gpu {} expected {}\n      \
                 gpu      {:?}\n      expected {:?}",
                got[i].key,
                expected[i].key,
                &got[lo..hi],
                &expected[lo..hi]
            ));
            continue;
        }

        // 2. Within a run of equal keys the payloads must be the same set: the sort permutes the
        //    agents of a cell but may not lose or duplicate one.
        let mut run_start = 0usize;
        while run_start < got.len() {
            let key = got[run_start].key;
            let mut run_end = run_start;
            while run_end < got.len() && got[run_end].key == key {
                run_end += 1;
            }
            let mut gpu_vals: Vec<u32> = got[run_start..run_end].iter().map(|k| k.val).collect();
            let mut cpu_vals: Vec<u32> = expected[run_start..run_end]
                .iter()
                .map(|k| k.val)
                .collect();
            gpu_vals.sort_unstable();
            cpu_vals.sort_unstable();
            if gpu_vals != cpu_vals {
                problems.push(format!(
                    "{context}: cell {key} holds the wrong agents at [{run_start}, {run_end}):\n      \
                     gpu      {gpu_vals:?}\n      expected {cpu_vals:?}"
                ));
                break;
            }
            run_start = run_end;
        }

        // 3. The padding must be at the end and must not name a cell, because `build_ranges` derives
        //    cell ranges from keys and a padding entry that named a cell would look like an agent.
        let live = swarm.len();
        if got[..live].iter().any(|k| k.key >= res.num_cells()) {
            problems.push(format!(
                "{context}: a live agent has a key outside the grid"
            ));
        }
        if let Some(i) = (live..got.len()).find(|&i| got[i].key != EMPTY_CELL) {
            problems.push(format!(
                "{context}: padding entry {i} has key {} instead of PAD_KEY",
                got[i].key
            ));
        }
    }

    if problems.is_empty() {
        println!("\n    5 agent counts, including non-power-of-two and both stage parities");
        Ok(())
    } else {
        Err(problems.join("\n    "))
    }
}

/// Checks the invariants `integrate_grid` searches under, over the whole grid.
///
/// Read as a contract rather than as a smoke test: every clause here is something the integration
/// pass relies on and cannot check, which is exactly the kind of assumption that silently produces a
/// swarm that is merely *slightly* wrong.
pub fn ranges_are_consistent(ctx: &GpuContext) -> Check {
    const N: usize = 1500;
    let cfg = roomy_config(N, 6.0, 18.0);
    let params: SimParams = cfg.to_params(0.0, cfg.dt);
    let (res, pipes, swarm) = setup(ctx, &cfg, 21);
    res.write_params(&ctx.queue, &params);
    res.write_interaction(&ctx.queue, &SimConfig::idle_interaction());

    let state = run_grid_prep(ctx, &res, &pipes);
    let padded_n = pipes.padded_n();
    let cell_count = res.num_cells();
    let live = swarm.len() as u32;
    let mut problems = Vec::new();
    let mut covered = 0u64;
    let mut non_empty = 0u32;
    let mut shared = 0u64;
    let mut previous_start = 0u32;

    for cell in 0..cell_count {
        let start = state.cell_start[cell as usize];
        let end = state.cell_end[cell as usize];

        // A cell with no agents is marked as such and its `cell_end` is never read by the search.
        if start == EMPTY_CELL {
            continue;
        }
        non_empty += 1;

        if start >= end || end > padded_n {
            problems.push(format!(
                "cell {cell} has range [{start}, {end}) for {padded_n} keys: the search would read \
                 {} entries",
                end.saturating_sub(start)
            ));
            continue;
        }
        // Ranges are laid out in cell order, so a cell's start can never be before the previous
        // non-empty cell's start. A violation means two cells claim overlapping runs, and the search
        // of one of them would return the other's agents.
        if start < previous_start {
            problems.push(format!(
                "cell {cell} starts at {start}, before the previous non-empty cell's {previous_start}"
            ));
        }
        previous_start = start;

        covered += u64::from(end - start);
        if end - start > 1 {
            shared += u64::from(end - start);
        }
        for index in start..end {
            let entry = state.keys[index as usize];
            if entry.key != cell {
                problems.push(format!(
                    "cell {cell}'s range [{start}, {end}) contains key {} at {index}",
                    entry.key
                ));
                break;
            }
            if entry.val >= live {
                problems.push(format!(
                    "cell {cell} contains val {} at {index}, which is a padding agent (num_boids \
                     {live})",
                    entry.val
                ));
                break;
            }
        }
    }

    if covered != u64::from(live) {
        problems.push(format!(
            "the ranges cover {covered} agents, but {live} were spawned: {} are in no cell",
            u64::from(live).abs_diff(covered)
        ));
    }
    // A cell holding two or more agents is the case the ranges exist for: a swarm where every agent
    // landed in a cell of its own would make `cell_start < cell_end` trivially true and never
    // exercise a run with more than one entry.
    if shared * 5 < u64::from(live) {
        problems.push(format!(
            "only {shared} of {live} agents share a cell with another agent: this world is too \
             sparse for the ranges to be exercised"
        ));
    }

    if problems.is_empty() {
        println!(
            "\n    {live} agents in {non_empty} cells of {cell_count}, ranges cover {covered}",
        );
        Ok(())
    } else {
        Err(problems.join("\n    "))
    }
}

/// Runs the grid and the all-pairs integration on the same swarm and compares the result agent by
/// agent.
///
/// One step, so there is no chaos to amplify a difference: both sides run the same force model over
/// the same neighbours, and the only source of divergence is floating-point reassociation from
/// summing the neighbours in a different order. That is why the tolerance is 1e-3 m rather than zero,
/// and why a missing or duplicated neighbour - which moves an agent by centimetres or metres in one
/// step at these weights - cannot hide inside it.
pub fn matches_naive(ctx: &GpuContext) -> Check {
    const N: usize = 1024;
    const SEED: u64 = 9;

    let cfg = roomy_config(N, 6.0, 18.0);
    let params: SimParams = cfg.to_params(0.0, cfg.dt);
    let interaction: InteractionUniforms = SimConfig::idle_interaction();

    let run = |strategy: Strategy| -> Vec<boids_core::layout::Boid> {
        let (mut res, pipes, _) = setup(ctx, &cfg, SEED);
        res.write_params(&ctx.queue, &params);
        res.write_interaction(&ctx.queue, &interaction);
        run_gpu_steps(ctx, &mut res, &pipes, strategy, 1)
    };

    let naive = run(Strategy::Naive);
    let grid = run(Strategy::Grid);

    if naive.len() != grid.len() {
        return Err(format!(
            "the grid run returned {} agents and the all-pairs run {}",
            grid.len(),
            naive.len()
        ));
    }

    let mut worst = 0.0f32;
    let mut worst_index = 0usize;
    let mut differing = 0usize;
    for (i, (g, n)) in grid.iter().zip(naive.iter()).enumerate() {
        let dv = (Vec3::from(g.vel) - Vec3::from(n.vel)).abs().max_element();
        if dv > 1e-3 {
            differing += 1;
        }
        if dv > worst {
            worst = dv;
            worst_index = i;
        }
    }

    if differing > 0 {
        let g = &grid[worst_index];
        let n = &naive[worst_index];
        return Err(format!(
            "{differing} of {N} agents moved differently after one step; worst {worst:.4e} m/s at \
             agent {worst_index}\n      \
             grid vel {:?}\n      naive vel {:?}\n      \
             A difference this size after one step is a different neighbour set, not a different \
             summation order. Check the cell size (it must be at least r_percept) and the range \
             building in shaders/sim/ranges.wgsl.",
            g.vel, n.vel
        ));
    }
    println!("\n    all {N} agents agree with all-pairs after one step (worst {worst:.2e} m/s)");
    Ok(())
}

/// Checks that the two strategies stay in step over a run long enough for the swarm to reorganise.
///
/// The exact comparison above says the neighbour *sets* agree at spawn. This says the grid keeps
/// finding them as the swarm moves: a range that is rebuilt from a stale `cell_start`, or a cell that
/// stops being cleared once it is empty, would show up here and not in a single step. The comparison
/// is on aggregates, because after a few hundred steps the two runs are chaotic and agent-by-agent
/// agreement is meaningless.
pub fn long_run_matches_naive(ctx: &GpuContext) -> Check {
    const N: usize = 1024;
    const STEPS: usize = 40;
    const SEED: u64 = 5;

    let mut cfg = roomy_config(N, 6.0, 18.0);
    cfg.wander = 0.0;
    cfg.buoyancy = 0.0;
    let params: SimParams = cfg.to_params(0.0, cfg.dt);
    let interaction = SimConfig::idle_interaction();

    let run = |strategy: Strategy| -> Vec<boids_core::layout::Boid> {
        let (mut res, pipes, _) = setup(ctx, &cfg, SEED);
        res.write_params(&ctx.queue, &params);
        res.write_interaction(&ctx.queue, &interaction);
        run_gpu_steps(ctx, &mut res, &pipes, strategy, STEPS)
    };

    let naive = run(Strategy::Naive);
    let grid = run(Strategy::Grid);

    let order_of = |set: &[boids_core::layout::Boid]| {
        order_parameter(&set.iter().map(|b| Vec3::from(b.vel)).collect::<Vec<_>>())
    };
    let speed_of = |set: &[boids_core::layout::Boid]| {
        mean_speed(&set.iter().map(|b| Vec3::from(b.vel)).collect::<Vec<_>>())
    };

    let (grid_order, naive_order) = (order_of(&grid), order_of(&naive));
    let (grid_speed, naive_speed) = (speed_of(&grid), speed_of(&naive));

    let mut problems = Vec::new();
    if (grid_order - naive_order).abs() > 0.1 {
        problems.push(format!(
            "the grid and all-pairs polarised differently over {STEPS} steps: {grid_order:.4} vs \
             {naive_order:.4}"
        ));
    }
    if (grid_speed - naive_speed).abs() > 0.1 * naive_speed.max(1e-3) {
        problems.push(format!(
            "mean speed differs: grid {grid_speed:.4} naive {naive_speed:.4}"
        ));
    }
    for (label, set) in [("grid", &grid), ("naive", &naive)] {
        let issues = boids_core::spawn::validation_issues(set, cfg.bounds_half * 6.0);
        if !issues.is_empty() {
            problems.push(format!(
                "{label} produced invalid agents:\n      {}",
                issues.join("\n      ")
            ));
        }
    }

    if problems.is_empty() {
        println!(
            "\n    after {STEPS} steps: order {grid_order:.3}/{naive_order:.3}, speed \
             {grid_speed:.3}/{naive_speed:.3} (grid/naive)"
        );
        Ok(())
    } else {
        Err(problems.join("\n    "))
    }
}

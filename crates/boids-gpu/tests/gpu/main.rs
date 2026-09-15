//! GPU test suite entry point.
//!
//! # Why this is a hand-rolled runner instead of `#[test]`
//!
//! Two reasons, both practical rather than stylistic:
//!
//! 1. **Thread affinity.** `libtest` runs each `#[test]` in its own thread. Creating and then
//!    dropping a GL-backed `wgpu` device from a non-main thread reliably crashes the process at
//!    exit on the WSL D3D12-though-GL driver, and it is fragile even where it works because GL
//!    contexts are per-thread. Running the checks on the main thread removes the whole class of
//!    problem, and lets the device be created once and dropped once.
//! 2. **One device for the suite.** Every `#[test]` would otherwise create its own adapter and
//!    device. On a software adapter that dominates the runtime, and on a real one it obscures which
//!    check is slow.
//!
//! The tradeoff is that checks run sequentially and that a panic in one check aborts the suite.
//! Both are acceptable here: the checks are independent, and a panic means a driver or API misuse
//! that invalidates the rest of the run anyway.
//!
//! Usage:
//! ```text
//! cargo test -p boids-gpu                 # whole suite
//! cargo test -p boids-gpu -- --list       # names only
//! cargo test -p boids-gpu -- layout       # filter by substring
//! BOIDS_TEST_REQUIRE_GPU=1 cargo test -p boids-gpu
//! ```

mod common;
mod grid;
mod immediates;
mod layout;
mod reference;
mod sdf;
mod shaders;

use std::time::Instant;

use boids_gpu::context::GpuContext;
use common::Check;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .parse_env("RUST_LOG")
        .try_init()
        .ok();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let filter = args.iter().find(|a| !a.starts_with('-')).cloned();

    if args.iter().any(|a| a == "--list") {
        for (name, _) in cases() {
            println!("{name}");
        }
        return;
    }

    let ctx = common::test_context();
    println!(
        "adapter: {} (backend {:?}, type {:?}){}",
        ctx.info.name,
        ctx.info.backend,
        ctx.info.device_type,
        if common::is_software(&ctx) {
            " [SOFTWARE]"
        } else {
            ""
        }
    );
    println!("timestamps: {}\n", ctx.timestamps_enabled);

    let mut passed = 0usize;
    let mut failures: Vec<(String, String)> = Vec::new();
    let mut skipped = 0usize;

    for (name, check) in cases() {
        if let Some(f) = &filter {
            if !name.contains(f.as_str()) {
                skipped += 1;
                continue;
            }
        }
        let start = Instant::now();
        print!("{name} ... ");
        let result: Check = check(&ctx);
        let elapsed = start.elapsed();
        match result {
            Ok(()) => {
                passed += 1;
                println!("ok ({:.2?})", elapsed);
            }
            Err(message) => {
                println!("FAILED ({:.2?})", elapsed);
                println!("    {message}\n");
                failures.push((name.to_string(), message));
            }
        }
    }

    println!(
        "\n{passed} passed, {} failed, {skipped} filtered out",
        failures.len()
    );

    // Dropping the device on the main thread is the supported path; doing it before exiting rather
    // than relying on the process teardown also makes a driver crash here clearly attributable.
    drop(ctx);

    if failures.is_empty() {
        return;
    }
    for (name, message) in &failures {
        println!("FAILED {name}\n    {message}");
    }
    std::process::exit(1);
}

/// A named GPU check.
type NamedCheck = (&'static str, fn(&GpuContext) -> Check);

/// The ordered list of checks.
fn cases() -> Vec<NamedCheck> {
    vec![
        ("shaders/compile_all", shaders::compile_all),
        (
            "layout/wgsl_offsets_match_rust",
            layout::wgsl_offsets_match_rust,
        ),
        (
            "layout/struct_sizes_are_exact",
            layout::struct_sizes_are_exact,
        ),
        ("sdf/wgsl_matches_rust", sdf::wgsl_matches_rust),
        (
            "reference/one_step_matches_cpu",
            reference::one_step_matches_cpu,
        ),
        (
            "reference/many_steps_match_cpu_aggregates",
            reference::many_steps_match_cpu_aggregates,
        ),
        (
            "reference/swarm_polarises_on_gpu",
            reference::swarm_polarises_on_gpu,
        ),
        (
            "sort/immediates_reach_the_shader",
            immediates::immediates_reach_the_shader,
        ),
        ("grid/sort_matches_cpu", grid::sort_matches_cpu),
        (
            "grid/unprepared_finds_no_neighbours",
            grid::unprepared_finds_no_neighbours,
        ),
        ("grid/ranges_are_consistent", grid::ranges_are_consistent),
        ("grid/matches_naive", grid::matches_naive),
        ("grid/long_run_matches_naive", grid::long_run_matches_naive),
    ]
}

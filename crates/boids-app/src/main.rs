//! The runtime: window, input, frame loop.
//!
//! # Frame structure
//!
//! ```text
//! 1. drain window events into `InputState` and `OrbitCamera`
//! 2. build the cursor ray on the CPU (once per frame, never per agent)
//! 3. upload SimParams and InteractionUniforms
//! 4. record the simulation compute passes, swap the ping-pong parity
//! 5. record the render passes against the current parity
//! 6. present
//! ```
//!
//! Nothing in this loop iterates over agents. The only per-frame CPU work proportional to anything is
//! the single `write_buffer` of two small uniform structs.
//!
//! # Cursor interaction
//!
//! The cursor's world position is the intersection of the ray through the mouse pixel with a plane
//! placed at the camera's focus distance. That is exact, cheap, and behaves the way users expect:
//! the influence point stays under the cursor as the camera orbits. Once the underwater reef exists,
//! the same ray is raymarched against the SDF first, so clicking on a rock puts the attractor on the
//! rock rather than behind it; the plane is the fallback when the ray hits nothing.

mod app;
mod bench;
mod input;
mod screenshot;

use app::BoidsApp;
use boids_core::layout::SimMode;
use boids_gpu::sim::Strategy;
use boids_render::Model;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_env("RUST_LOG")
        .init();

    let invocation = match parse_args(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(invocation) => invocation,
        Err(message) => {
            eprintln!("boids: {message}");
            eprintln!("try --help");
            std::process::exit(2);
        }
    };

    match invocation {
        Invocation::Help => println!("{}", usage()),
        // Benchmark mode is headless like the screenshot: it needs a device, not a window.
        Invocation::Bench(request) => match bench::run(&request) {
            Ok(()) => {}
            Err(e) => {
                log::error!("benchmark failed: {e}");
                std::process::exit(1);
            }
        },
        // Screenshot mode bypasses the event loop entirely: it needs no window, no input and no
        // display server, and going through `winit` would require a display connection to render one
        // image.
        Invocation::Screenshot(request) => match screenshot::capture(&request) {
            Ok(()) => {}
            Err(e) => {
                log::error!("screenshot failed: {e}");
                std::process::exit(1);
            }
        },
        // The OBJ export is pure CPU: it builds the same mesh the renderer uploads and writes it, so
        // it runs on a machine with no GPU at all.
        Invocation::Export(path) => {
            let mesh = boids_scene::mesh::castle_mesh();
            match boids_scene::mesh::write_obj(&mesh, &path) {
                Ok(()) => log::info!(
                    "wrote {} ({} vertices, {} triangles)",
                    path.display(),
                    mesh.vertices.len(),
                    mesh.triangle_count()
                ),
                Err(e) => {
                    log::error!("export failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Invocation::Window(startup) => run_window(startup),
    }
}

/// Opens a window and runs until it closes.
fn run_window(startup: app::StartupConfig) {
    let event_loop = EventLoop::new().expect("creating the event loop");
    // `Poll` keeps the frame loop running when nothing is happening, which is required: the
    // simulation is animated and must advance every frame whether or not the window has events.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = BoidsApp::new(startup);
    if let Err(e) = event_loop.run_app(&mut app) {
        log::error!("event loop exited with an error: {e}");
        std::process::exit(1);
    }
}

/// What the command line asked the program to do.
///
/// Parsing produces this rather than starting work directly, which is what makes the arguments
/// testable and what keeps `--help` from being a `process::exit` buried in a `match` arm.
#[derive(Debug)]
enum Invocation {
    /// Open a window and run the interactive loop.
    Window(app::StartupConfig),
    /// Print the usage text and exit successfully.
    Help,
    /// Render one frame headlessly.
    Screenshot(Box<screenshot::ScreenshotRequest>),
    /// Time a headless simulation and print the per-pass breakdown.
    Bench(Box<bench::BenchRequest>),
    /// Write a static mesh to a Wavefront OBJ and exit.
    Export(std::path::PathBuf),
}

/// The flags that describe the world, whichever mode consumes them.
///
/// Every field is optional so that a mode can fall back to *its own* default rather than to the
/// window's. That distinction matters for the screenshot, whose default of 2,000 agents is chosen so
/// that the picture shows a flock: inheriting the window's 100,000 would render the whole target world
/// at 720p, where every agent is smaller than a pixel and the frame is uniform speckle.
#[derive(Debug, Default)]
struct WorldFlags {
    num_agents: Option<usize>,
    seed: Option<u64>,
    mode: Option<SimMode>,
    deterministic: bool,
    profile: bool,
    strategy: Option<Strategy>,
    /// The mesh the model viewer opens on, from `--model`.
    model: Option<Model>,
}

/// Which headless mode was asked for, if any.
#[derive(Debug)]
enum Headless {
    Screenshot(std::path::PathBuf),
    Bench(usize),
    /// Write a static mesh to a Wavefront OBJ and exit. Needs no device at all.
    Export(std::path::PathBuf),
}

/// Parses the command line.
///
/// Hand parsed rather than pulled from a CLI crate: there are a handful of options, and adding a
/// dependency tree to read a handful of strings is not a trade worth making.
///
/// # Errors
/// Returns a message for a flag that needs a value and has none, a value that does not parse, and a
/// request for both headless modes at once. Every one of those is a mistake worth refusing to run
/// on: `--agents 10000x` quietly running 100,000 agents instead is how an evening disappears.
fn parse_args(args: &[String]) -> Result<Invocation, String> {
    let mut world = WorldFlags::default();
    let mut headless: Option<Headless> = None;

    let mut cursor = 0usize;
    while let Some(arg) = args.get(cursor).map(String::as_str) {
        cursor += 1;
        match arg {
            "--agents" | "-n" => {
                let value = required(args, &mut cursor, arg)?;
                world.num_agents = Some(
                    value
                        .parse()
                        .map_err(|e| format!("--agents {value}: {e}"))?,
                );
            }
            "--seed" => {
                let value = required(args, &mut cursor, arg)?;
                world.seed = Some(value.parse().map_err(|e| format!("--seed {value}: {e}"))?);
            }
            "--strategy" => {
                let value = required(args, &mut cursor, arg)?;
                // `auto` is the default and means "no override", so it is accepted here and clears
                // any earlier override rather than being an error.
                world.strategy = if value == "auto" {
                    None
                } else {
                    Some(value.parse()?)
                };
            }
            "--fish" => world.mode = Some(SimMode::Fish),
            "--birds" => world.mode = Some(SimMode::Birds),
            "--model" => {
                let value = required(args, &mut cursor, arg)?;
                world.model = Some(Model::from_name(value).ok_or_else(|| {
                    let names: Vec<&str> = Model::ALL.iter().map(|m| m.label()).collect();
                    format!("--model {value}: expected one of {}", names.join(", "))
                })?);
            }
            "--deterministic" => world.deterministic = true,
            "--profile" => world.profile = true,
            "--screenshot" => {
                let path = required(args, &mut cursor, arg)?.into();
                set_headless(&mut headless, Headless::Screenshot(path))?;
            }
            "--bench" => {
                // The frame count is optional, which is why the value is taken only if the next
                // argument is a number: `--bench --birds` has to mean "120 frames in the sky", not
                // "parse `--birds` as a frame count and then wonder where the flag went".
                let frames = match args.get(cursor).and_then(|v| v.parse().ok()) {
                    Some(frames) => {
                        cursor += 1;
                        frames
                    }
                    None => DEFAULT_BENCH_FRAMES,
                };
                set_headless(&mut headless, Headless::Bench(frames))?;
            }
            "--export-castle" => {
                let path = required(args, &mut cursor, arg)?.into();
                set_headless(&mut headless, Headless::Export(path))?;
            }
            "--help" | "-h" => return Ok(Invocation::Help),
            other => return Err(format!("unknown argument {other}")),
        }
    }

    match headless {
        Some(Headless::Screenshot(path)) => {
            let defaults = screenshot::ScreenshotRequest::default();
            Ok(Invocation::Screenshot(Box::new(
                screenshot::ScreenshotRequest {
                    path,
                    mode: world.mode.unwrap_or(defaults.mode),
                    num_agents: world.num_agents.unwrap_or(defaults.num_agents),
                    seed: world.seed.unwrap_or(defaults.seed),
                    strategy: world.strategy,
                    model: world.model,
                    ..defaults
                },
            )))
        }
        // The benchmark inherits the flags that describe the world, the same way the screenshot
        // does, so `--birds --agents 50000 --bench 200` does what it reads like. Its own defaults are
        // the target: 100,000 agents, which is the number worth timing.
        Some(Headless::Bench(frames)) => {
            let defaults = bench::BenchRequest::default();
            Ok(Invocation::Bench(Box::new(bench::BenchRequest {
                mode: world.mode.unwrap_or(defaults.mode),
                num_agents: world.num_agents.unwrap_or(defaults.num_agents),
                frames,
                seed: world.seed.unwrap_or(defaults.seed),
                strategy: world.strategy,
                ..defaults
            })))
        }
        Some(Headless::Export(path)) => Ok(Invocation::Export(path)),
        None => {
            let defaults = app::StartupConfig::default();
            Ok(Invocation::Window(app::StartupConfig {
                num_agents: world.num_agents.unwrap_or(defaults.num_agents),
                seed: world.seed.unwrap_or(defaults.seed),
                start_in_birds: world.mode.unwrap_or(SimMode::Fish) == SimMode::Birds,
                deterministic: world.deterministic,
                profile: world.profile,
                strategy: world.strategy,
                viewer: world.model,
            }))
        }
    }
}

/// Frames a bare `--bench` runs.
const DEFAULT_BENCH_FRAMES: usize = 120;

/// The next argument, which the caller needs.
fn required<'a>(args: &'a [String], cursor: &mut usize, flag: &str) -> Result<&'a str, String> {
    let value = args
        .get(*cursor)
        .ok_or_else(|| format!("{flag} needs a value"))?;
    *cursor += 1;
    Ok(value)
}

/// Records the requested headless mode, refusing to guess between two.
fn set_headless(slot: &mut Option<Headless>, request: Headless) -> Result<(), String> {
    if slot.is_some() {
        return Err(
            "--bench, --screenshot and --export-castle each run headlessly: ask for one of them"
                .into(),
        );
    }
    *slot = Some(request);
    Ok(())
}

/// The usage text.
fn usage() -> &'static str {
    "boids - GPU flocking simulation\n\n\
     USAGE:\n    boids [OPTIONS]\n\n\
     OPTIONS:\n    \
     -n, --agents <N>   number of agents (default 100000)\n    \
     --seed <N>         world seed (default 1)\n    \
     --strategy <S>     neighbour search: naive, grid, or auto by agent count (default auto)\n    \
     --fish             start in the underwater world (default)\n    \
     --birds            start in the sky world\n    \
     --model <M>        open the model viewer on fish, bird, tree or castle\n    \
     --deterministic    fixed timestep, for screenshot comparison\n    \
     --profile          log per-pass GPU timings once a second\n    \
     --bench <N>        time N headless simulation frames and print a per-pass\n    \
                        breakdown, then exit\n    \
     --screenshot <f>   render one frame headlessly to a PNG and exit; with\n    \
                        --model, render that model alone\n    \
     --export-castle <f>  write the castle mesh as a Wavefront OBJ and exit\n    \
     -h, --help         print this help\n\n\
     CONTROLS:\n    \
     left drag          orbit the camera\n    \
     right drag         pan\n    \
     wheel              zoom\n    \
     middle drag / 1    attractor at the cursor\n    \
     2                  repeller at the cursor\n    \
     0                  cursor influence off\n    \
     tab                switch between fish and birds\n    \
     space              pause\n    \
     r                  reset the swarm\n    \
     esc                quit\n\n\
     MODEL VIEWER (v):\n    \
     left drag          rotate the model\n    \
     wheel              zoom\n    \
     1..4 / [ ]         select fish, bird, tree, castle\n    \
     r                  reframe the camera\n    \
     v                  back to the swarm"
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses a command line written the way it would be typed.
    fn parse(args: &[&str]) -> Result<Invocation, String> {
        let owned: Vec<String> = args.iter().map(|a| (*a).to_string()).collect();
        parse_args(&owned)
    }

    /// The window configuration, or a panic naming what was parsed instead.
    fn window(args: &[&str]) -> app::StartupConfig {
        match parse(args) {
            Ok(Invocation::Window(config)) => config,
            other => panic!("expected a window run, got {other:?}"),
        }
    }

    #[test]
    fn no_arguments_runs_the_window_on_the_target_agent_count() {
        let config = window(&[]);
        assert_eq!(config.num_agents, 100_000);
        assert_eq!(config.seed, 1);
        assert!(!config.start_in_birds);
        assert!(!config.deterministic);
        assert!(!config.profile);
        assert_eq!(config.strategy, None);
    }

    #[test]
    fn world_flags_apply_to_the_window() {
        let config = window(&[
            "--agents",
            "5000",
            "--seed",
            "7",
            "--birds",
            "--deterministic",
            "--profile",
        ]);
        assert_eq!(config.num_agents, 5000);
        assert_eq!(config.seed, 7);
        assert!(config.start_in_birds);
        assert!(config.deterministic);
        assert!(config.profile);
    }

    #[test]
    fn the_benchmark_inherits_the_world_flags() {
        let request = match parse(&[
            "--birds",
            "--agents",
            "20000",
            "--seed",
            "3",
            "--strategy",
            "grid",
            "--bench",
            "30",
        ]) {
            Ok(Invocation::Bench(request)) => request,
            other => panic!("expected a benchmark, got {other:?}"),
        };
        assert_eq!(request.frames, 30);
        assert_eq!(request.num_agents, 20_000);
        assert_eq!(request.seed, 3);
        assert_eq!(request.mode, SimMode::Birds);
        assert_eq!(request.strategy, Some(Strategy::Grid));
    }

    #[test]
    fn a_benchmark_without_a_frame_count_takes_the_default_and_not_the_next_flag() {
        let request = match parse(&["--bench", "--birds"]) {
            Ok(Invocation::Bench(request)) => request,
            other => panic!("expected a benchmark, got {other:?}"),
        };
        assert_eq!(request.frames, DEFAULT_BENCH_FRAMES);
        assert_eq!(request.mode, SimMode::Birds);
    }

    #[test]
    fn the_screenshot_inherits_the_world_flags_it_was_given() {
        let request = match parse(&["--agents", "3000", "--birds", "--screenshot", "out.png"]) {
            Ok(Invocation::Screenshot(request)) => request,
            other => panic!("expected a screenshot, got {other:?}"),
        };
        assert_eq!(request.path, std::path::Path::new("out.png"));
        assert_eq!(request.num_agents, 3000);
        assert_eq!(request.mode, SimMode::Birds);
        assert_eq!(request.strategy, None);
        // The screenshot's own defaults survive: only the world flags are inherited.
        assert_eq!(request.width, 1280);
        assert_eq!(request.height, 720);
    }

    #[test]
    fn a_screenshot_without_world_flags_keeps_the_agent_count_it_was_sized_for() {
        let defaults = screenshot::ScreenshotRequest::default();
        let request = match parse(&["--screenshot", "out.png"]) {
            Ok(Invocation::Screenshot(request)) => request,
            other => panic!("expected a screenshot, got {other:?}"),
        };
        // Not the window's 100,000: at 720p that is a frame of uniform speckle rather than a flock.
        assert_eq!(request.num_agents, defaults.num_agents);
        assert_eq!(request.warmup_steps, defaults.warmup_steps);
        assert_eq!(request.mode, SimMode::Fish);
        assert_eq!(request.seed, 1);
    }

    #[test]
    fn a_benchmark_without_world_flags_times_the_target_agent_count() {
        let request = match parse(&["--bench", "5"]) {
            Ok(Invocation::Bench(request)) => request,
            other => panic!("expected a benchmark, got {other:?}"),
        };
        assert_eq!(request.num_agents, 100_000);
        assert_eq!(request.frames, 5);
    }

    #[test]
    fn strategy_auto_clears_an_earlier_override() {
        assert_eq!(
            window(&["--strategy", "naive"]).strategy,
            Some(Strategy::Naive)
        );
        assert_eq!(
            window(&["--strategy", "grid"]).strategy,
            Some(Strategy::Grid)
        );
        assert_eq!(window(&["--strategy", "auto"]).strategy, None);
        assert_eq!(
            window(&["--strategy", "naive", "--strategy", "auto"]).strategy,
            None
        );
    }

    #[test]
    fn two_headless_modes_at_once_are_refused() {
        let error = parse(&["--screenshot", "out.png", "--bench", "5"]).unwrap_err();
        assert!(error.contains("--bench"), "{error}");
        assert!(parse(&["--bench", "5", "--screenshot", "out.png"]).is_err());
    }

    #[test]
    fn bad_arguments_are_errors_rather_than_silent_defaults() {
        assert!(parse(&["--agents", "10000x"]).is_err());
        assert!(parse(&["--agents"]).is_err());
        assert!(parse(&["--seed"]).is_err());
        assert!(parse(&["--strategy", "radix"]).is_err());
        assert!(parse(&["--strategy"]).is_err());
        assert!(parse(&["--screenshot"]).is_err());
        assert!(parse(&["--nonsense"]).is_err());
    }

    #[test]
    fn help_wins_over_everything_after_it() {
        assert!(matches!(parse(&["--help"]), Ok(Invocation::Help)));
        assert!(matches!(parse(&["-h"]), Ok(Invocation::Help)));
        assert!(matches!(parse(&["-h", "--nonsense"]), Ok(Invocation::Help)));
        // ...but a bad argument before it is still a bad argument.
        assert!(parse(&["--nonsense", "-h"]).is_err());
    }
}

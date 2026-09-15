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
mod input;
mod screenshot;

use app::BoidsApp;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_env("RUST_LOG")
        .init();

    let (startup, screenshot_request) = cli_config();

    // Screenshot mode bypasses the event loop entirely: it needs no window, no input and no display
    // server, and going through `winit` would require a display connection to render one image.
    if let Some(request) = screenshot_request {
        match screenshot::capture(&request) {
            Ok(()) => return,
            Err(e) => {
                log::error!("screenshot failed: {e}");
                std::process::exit(1);
            }
        }
    }

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

/// Command line configuration.
///
/// Hand parsed rather than pulled from a CLI crate: there are four options, and adding a dependency
/// tree to read four strings is not a trade worth making.
fn cli_config() -> (app::StartupConfig, Option<screenshot::ScreenshotRequest>) {
    let mut config = app::StartupConfig::default();
    let mut screenshot: Option<screenshot::ScreenshotRequest> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--agents" | "-n" => {
                if let Some(value) = args.next() {
                    match value.parse() {
                        Ok(n) => config.num_agents = n,
                        Err(e) => log::warn!("--agents {value}: {e}; keeping {}", config.num_agents),
                    }
                }
            }
            "--seed" => {
                if let Some(value) = args.next() {
                    match value.parse() {
                        Ok(s) => config.seed = s,
                        Err(e) => log::warn!("--seed {value}: {e}"),
                    }
                }
            }
            "--screenshot" => {
                let Some(path) = args.next() else {
                    log::error!("--screenshot needs a file path");
                    std::process::exit(2);
                };
                screenshot = Some(screenshot::ScreenshotRequest {
                    path: path.into(),
                    ..Default::default()
                });
            }
            "--birds" => config.start_in_birds = true,
            "--fish" => config.start_in_birds = false,
            "--deterministic" => config.deterministic = true,
            "--profile" => config.profile = true,
            "--help" | "-h" => {
                println!(
                    "boids - GPU flocking simulation\n\n\
                     USAGE:\n    boids [OPTIONS]\n\n\
                     OPTIONS:\n    \
                     -n, --agents <N>   number of agents (default 100000)\n    \
                     --seed <N>         world seed (default 1)\n    \
                     --fish             start in the underwater world (default)\n    \
                     --birds            start in the sky world\n    \
                     --deterministic    fixed timestep, for screenshot comparison\n    \
                     --profile          log per-pass GPU timings once a second\n    \
                     --screenshot <f>   render one frame headlessly to a PNG and exit\n    \
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
                     esc                quit"
                );
                std::process::exit(0);
            }
            other => log::warn!("ignoring unknown argument {other} (try --help)"),
        }
    }

    // Let the screenshot inherit whatever the flags configured, so `--fish --agents 5000 --screenshot
    // a.png` does what it reads like.
    if let Some(request) = &mut screenshot {
        request.mode = if config.start_in_birds {
            boids_core::layout::SimMode::Birds
        } else {
            boids_core::layout::SimMode::Fish
        };
        request.num_agents = config.num_agents;
        request.seed = config.seed;
    }

    (config, screenshot)
}

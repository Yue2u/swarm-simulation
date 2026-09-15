//! The application: window lifecycle, per-frame orchestration, and the mode switch.
//!
//! # Why the world switch does not rebuild the context
//!
//! The two worlds differ in uniforms and in which compute pipeline runs, not in resources. Both
//! worlds' pipelines, buffers and bind groups are created once, at startup, and switching modes swaps
//! a `SimMode`, a `SimConfig` and a `MeshParams`. Nothing is reallocated, no pipeline is recompiled,
//! and the GPU state the driver has already warmed up stays warm. A mode switch therefore costs one
//! frame of latency, which is what makes the transition usable as a live A/B comparison instead of a
//! multi-second hitch.

use std::sync::Arc;

use boids_core::camera::OrbitCamera;
use boids_core::config::SimConfig;
use boids_core::layout::{
    CameraUniform, InteractionMode, InteractionUniforms, SceneUniform, SimMode, SimParams,
};
use boids_gpu::context::{GpuContext, GpuContextDescriptor, SurfaceState};
use boids_gpu::sim::{SimPipelines, SimResources, Strategy};
use boids_render::renderer::{FrameInput, Renderer};
use boids_render::SceneBinding;
use glam::{Vec2, Vec3};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

use crate::input::InputState;

/// How the app should start.
#[derive(Debug, Clone)]
pub struct StartupConfig {
    /// Number of agents to simulate.
    pub num_agents: usize,
    /// World seed. Fixed by default so that a run is reproducible.
    pub seed: u64,
    /// Start in the sky world rather than the underwater one.
    pub start_in_birds: bool,
    /// Use a fixed timestep regardless of real elapsed time.
    ///
    /// Without this the simulation advances by the measured frame time, which makes a run
    /// unreproducible: the same seed produces a different world at a different frame rate. Fixed
    /// timesteps cost a little visual smoothness and buy reproducibility, which is what makes
    /// screenshot comparison possible.
    pub deterministic: bool,
}

impl Default for StartupConfig {
    fn default() -> Self {
        Self {
            // 4096 rather than the 100k target: until the spatial grid is wired up the simulation
            // runs the all-pairs path, which is O(N^2) per step and cannot hold a frame rate above
            // roughly this many agents. The default is set to what actually runs; the flag still
            // accepts anything, and `strategy` says what will happen.
            num_agents: 4_096,
            seed: 1,
            start_in_birds: false,
            deterministic: false,
        }
    }
}

/// Above this agent count the spatial grid is the faster strategy.
///
/// The crossover is not a clean analytic number: the all-pairs path does O(N) work per agent with a
/// very low constant, while the grid does O(k) with a much higher constant plus the cost of the sort
/// that feeds it. A few thousand agents is where they meet on the target hardware.
///
/// The grid path is not selectable yet: its hash, sort and range passes do not exist, so the grid
/// would find no neighbours at all. Until they land, this constant only controls when the app warns
/// that the simulation is about to be slow.
const NAIVE_AGENT_LIMIT: u32 = 4096;

/// Distance in front of the camera where the cursor's interaction plane is placed, as a fraction of
/// the orbit distance. Close enough that the influence point feels attached to the cursor, far enough
/// that it is not inside the near plane.
const INTERACTION_PLANE_FRACTION: f32 = 0.6;

/// The application.
pub struct BoidsApp {
    startup: StartupConfig,
    // Everything below is created in `resumed`, once a window exists.
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    surface: Option<SurfaceState>,
    sim: Option<(SimResources, SimPipelines)>,
    renderer: Option<Renderer>,
    camera: OrbitCamera,
    input: InputState,
    sim_config: SimConfig,
    /// Simulation clock, advanced by the fixed or measured timestep.
    sim_time: f32,
    /// Smoothed frame time, for the title bar and the variable timestep.
    smoothed_frame_time: f32,
    /// Last frame's timestamp, for the measured timestep.
    last_frame: Option<std::time::Instant>,
    /// Frames rendered since startup, for the HUD.
    frames: u64,
}

impl core::fmt::Debug for BoidsApp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BoidsApp")
            .field("startup", &self.startup)
            .field("has_window", &self.window.is_some())
            .field("frames", &self.frames)
            .finish_non_exhaustive()
    }
}

impl BoidsApp {
    /// Creates the app. No GPU work happens until the window exists.
    #[must_use]
    pub fn new(startup: StartupConfig) -> Self {
        let mode = if startup.start_in_birds {
            SimMode::Birds
        } else {
            SimMode::Fish
        };
        let sim_config = SimConfig::for_mode(mode, startup.num_agents);
        let camera = OrbitCamera {
            distance: sim_config.bounds_half.length() * 0.8,
            ..Default::default()
        };
        Self {
            startup,
            window: None,
            gpu: None,
            surface: None,
            sim: None,
            renderer: None,
            camera,
            input: InputState::default(),
            sim_config,
            sim_time: 0.0,
            smoothed_frame_time: 1.0 / 60.0,
            last_frame: None,
            frames: 0,
        }
    }

    /// Whether the GPU side is up.
    #[must_use]
    pub const fn is_initialised(&self) -> bool {
        self.renderer.is_some()
    }

    /// Builds the device, surface, simulation and renderer, and spawns the swarm.
    ///
    /// Returns a description on failure rather than panicking, so that a machine without a usable
    /// adapter reports what is missing instead of aborting inside a driver call.
    pub fn initialise(&mut self, window: Arc<Window>) -> Result<(), String> {
        let size = window.inner_size();
        // Surface first, adapter second: the adapter must be one that can present to this window.
        let (gpu, surface) = GpuContext::new_with_window(
            &GpuContextDescriptor::default(),
            window.clone(),
            size.width,
            size.height,
        )?;
        log::info!("adapter: {}", gpu.adapter_summary());

        let mut camera = self.camera;
        camera.set_viewport(size.width, size.height);

        let swarm = boids_core::spawn::spawn_swarm(&self.sim_config, self.startup.seed);
        let resources = SimResources::new(&gpu, &self.sim_config);
        boids_gpu::transfer::upload_boids(&gpu.queue, &resources.boids[0], &swarm);
        // Prime both parities with the same state: the first frame draws from parity 0 and the first
        // simulation step writes parity 1, so the renderer never sees an uninitialised buffer.
        boids_gpu::transfer::upload_boids(&gpu.queue, &resources.boids[1], &swarm);
        let pipelines = SimPipelines::new(&gpu, &resources);

        let renderer = Renderer::new(
            &gpu,
            [&resources.boids[0], &resources.boids[1]],
            surface.config.format,
            surface.is_srgb,
            size.width,
            size.height,
        );

        log::info!(
            "simulation: {} agents, mode {:?}, grid {:?}, strategy {:?}",
            self.sim_config.num_boids,
            self.sim_config.mode,
            self.sim_config.grid.dim,
            self.strategy()
        );

        self.camera = camera;
        self.gpu = Some(gpu);
        self.surface = Some(surface);
        self.sim = Some((resources, pipelines));
        self.renderer = Some(renderer);
        self.window = Some(window);
        Ok(())
    }

    /// The neighbour-search strategy for the current agent count.
    ///
    /// Always all-pairs for now, with a warning above the point where the grid would be the right
    /// answer. Reporting the honest limitation is better than silently running an unusable frame rate
    /// or silently producing a swarm with no interactions.
    fn strategy(&self) -> Strategy {
        #[allow(clippy::cast_possible_truncation)]
        let count = self.sim_config.num_boids as u32;
        if count > NAIVE_AGENT_LIMIT {
            log::warn!(
                "{count} agents exceeds the all-pairs limit of {NAIVE_AGENT_LIMIT}: the spatial grid \
                 that would make this real-time is not wired up yet, so the frame rate will be low"
            );
        }
        Strategy::Naive
    }

    /// Rebuilds the cursor interaction uniform for this frame.
    ///
    /// The focus point is where the ray through the cursor pixel meets a plane placed
    /// [`INTERACTION_PLANE_FRACTION`] of the orbit distance in front of the camera. If the ray misses
    /// (the camera looking away from the plane's hemisphere) the interaction is reported as off rather
    /// than placed at an absurd distance, which is what keeps the swarm from lurching when the cursor
    /// leaves the window.
    fn interaction(&self, viewport: Vec2) -> InteractionUniforms {
        let idle = SimConfig::idle_interaction();
        if !self.input.interaction_enabled() {
            return idle;
        }
        let Some(cursor) = self.input.cursor else {
            return idle;
        };

        let camera = self.camera;
        let inv = camera.view_proj().inverse();
        let ray = boids_core::camera::ray_from_ndc(inv, cursor, viewport);
        let plane_point = camera.eye() + camera.forward() * (camera.distance * INTERACTION_PLANE_FRACTION);
        let plane_normal = -camera.forward();

        let Some(focus) = boids_core::camera::ray_plane_intersect(ray, plane_point, plane_normal)
        else {
            return idle;
        };

        // Keep the influence point inside the world: a cursor aimed at the horizon otherwise places
        // the attractor thousands of metres away, where it pulls the whole swarm into a line.
        let extent = self.sim_config.bounds_half;
        let focus = focus.clamp(-extent, extent);

        let (strength, tangent) = match self.input.mode {
            InteractionMode::Attract => (self.sim_config.max_speed * 6.0, 0.0),
            InteractionMode::Repel => (
                self.sim_config.max_speed * 8.0,
                self.sim_config.max_speed * 2.5,
            ),
            InteractionMode::Off => return idle,
        };

        InteractionUniforms {
            ray_origin: camera.eye().to_array(),
            mode: self.input.mode.as_u32(),
            focus_point: focus.to_array(),
            radius: self.sim_config.r_percept * 6.0,
            strength,
            falloff: 1.6,
            tangent,
            _pad: 0.0,
        }
    }

    /// Builds the scene uniform for this frame.
    fn scene_uniform(&self, viewport: Vec2) -> SceneUniform {
        let camera: CameraUniform = self.camera.to_uniform(self.sim_time, viewport);
        boids_gpu::mesh_profile::scene_uniform(
            camera,
            self.sim_config.mode,
            self.sim_config.max_speed,
            self.sim_config.r_percept,
        )
    }

    /// The timestep for this frame.
    fn timestep(&mut self, now: std::time::Instant) -> f32 {
        let dt = match self.last_frame {
            Some(previous) => now.duration_since(previous).as_secs_f32(),
            None => 1.0 / 60.0,
        };
        self.last_frame = Some(now);

        if self.startup.deterministic {
            return self.sim_config.dt;
        }
        // Clamp the measured step: a hitch (a shader compile, a window drag, the machine waking from
        // sleep) would otherwise advance the simulation by a huge step and throw the swarm apart.
        dt.clamp(1.0 / 480.0, 1.0 / 20.0)
    }

    /// Switches worlds without touching the device.
    fn toggle_world(&mut self) {
        let mode = match self.sim_config.mode {
            SimMode::Fish => SimMode::Birds,
            SimMode::Birds => SimMode::Fish,
        };
        self.sim_config = SimConfig::for_mode(mode, self.startup.num_agents);
        // Respawn: the two worlds have different bounds and speeds, and carrying agents across would
        // leave them outside the new world's grid and immediately clamped into the border cells.
        self.respawn();
        self.camera.target = Vec3::ZERO;
        self.camera.distance = self.sim_config.bounds_half.length() * 0.8;
        log::info!("switched to {:?}", mode);
    }

    /// Respawns the swarm in the current world.
    fn respawn(&mut self) {
        let Some(gpu) = &self.gpu else { return };
        let Some((resources, _)) = &mut self.sim else {
            return;
        };
        let swarm = boids_core::spawn::spawn_swarm(&self.sim_config, self.startup.seed);
        boids_gpu::transfer::upload_boids(&gpu.queue, &resources.boids[0], &swarm);
        boids_gpu::transfer::upload_boids(&gpu.queue, &resources.boids[1], &swarm);
        self.sim_time = 0.0;
        self.last_frame = None;
    }

    /// Advances the simulation by one step and returns the parity to render.
    fn step_simulation(&mut self, dt: f32) -> usize {
        // Everything that needs an immutable view of `self` is computed first: `self.sim` has to be
        // borrowed mutably to ping-pong, and holding that borrow while calling back into `self` would
        // not compile. Ordering the work this way is clearer than threading the borrows through.
        let Some(renderer) = &self.renderer else { return 0 };
        let (width, height) = renderer.size();
        #[allow(clippy::cast_precision_loss)]
        let viewport = Vec2::new(width as f32, height as f32);
        let params: SimParams = self.sim_config.to_params(self.sim_time, dt);
        let interaction = self.interaction(viewport);
        let strategy = self.strategy();

        let Some(gpu) = &self.gpu else { return 0 };
        let Some((resources, pipelines)) = &mut self.sim else {
            return 0;
        };

        resources.write_params(&gpu.queue, &params);
        resources.write_interaction(&gpu.queue, &interaction);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("simulation"),
            });
        pipelines.record_integrate(&mut encoder, resources, strategy);
        gpu.queue.submit(Some(encoder.finish()));
        resources.swap();

        self.sim_time += dt;
        resources.read_index()
    }

    /// Records and presents one frame.
    fn render_frame(&mut self) {
        // As in `step_simulation`: gather what needs an immutable view of `self` up front, then take
        // the mutable borrows for the rest of the frame.
        let Some(gpu) = &self.gpu else { return };
        let Some(surface) = &self.surface else { return };

        let (width, height) = surface.size();
        #[allow(clippy::cast_precision_loss)]
        let viewport = Vec2::new(width as f32, height as f32);
        let scene = self.scene_uniform(viewport);
        let parity = self
            .sim
            .as_ref()
            .map_or(0, |(resources, _)| resources.read_index());
        #[allow(clippy::cast_possible_truncation)]
        let num_agents = self.sim_config.num_boids as u32;

        let Some(surface) = &mut self.surface else {
            return;
        };
        let Some(frame) = surface.acquire(gpu, width, height) else {
            // Nothing to present: a timeout, an occluded window, or a surface that needed
            // reconfiguring. Skipping is the correct response to all of them.
            return;
        };
        let Some(renderer) = &mut self.renderer else {
            return;
        };
        renderer.resize(gpu, width, height);

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let stats = renderer.render(
            gpu,
            &view,
            &scene,
            FrameInput {
                binding: SceneBinding::new(parity),
                num_agents,
            },
        );

        gpu.queue.present(frame);
        self.frames += 1;
        log::trace!(
            "frame {}: {} draws, {} instances, {} triangles",
            self.frames,
            stats.draw_calls,
            stats.instances,
            stats.triangles
        );
    }

    /// Updates the window title with the frame rate and the current world.
    ///
    /// Also prints a periodic line to the log. A title bar is unreadable over a remote session and
    /// invisible in a screenshot, and the frame time is the number that decides whether a change was an
    /// improvement, so it has to reach the log as well.
    fn update_title(&self) {
        let Some(window) = &self.window else { return };
        // The title is updated four times a second at 60 fps: often enough to feel live, rare
        // enough that the window system is not being hammered by a string that mostly does not change.
        if !self.frames.is_multiple_of(15) {
            return;
        }
        if self.frames.is_multiple_of(120) {
            log::info!(
                "frame {}: {:.1} fps ({:.2} ms) | {} agents | {} | {}",
                self.frames,
                if self.smoothed_frame_time > 0.0 {
                    1.0 / self.smoothed_frame_time
                } else {
                    0.0
                },
                self.smoothed_frame_time * 1000.0,
                self.sim_config.num_boids,
                self.gpu
                    .as_ref()
                    .map_or("no device".to_string(), GpuContext::adapter_summary),
                match self.input.mode {
                    InteractionMode::Attract => "attract",
                    InteractionMode::Repel => "repel",
                    InteractionMode::Off => "no cursor",
                },
            );
        }
        let fps = if self.smoothed_frame_time > 0.0 {
            1.0 / self.smoothed_frame_time
        } else {
            0.0
        };
        window.set_title(&format!(
            "boids | {} agents | {} | {:.0} fps | {:.1} ms | {}",
            self.sim_config.num_boids,
            match self.sim_config.mode {
                SimMode::Fish => "underwater",
                SimMode::Birds => "sky",
            },
            fps,
            self.smoothed_frame_time * 1000.0,
            match self.input.mode {
                InteractionMode::Attract => "attract",
                InteractionMode::Repel => "repel",
                InteractionMode::Off => "no cursor",
            },
        ));
    }
}

impl ApplicationHandler for BoidsApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            // `resumed` fires again when the platform re-activates the window; re-creating the
            // surface and the whole GPU context would discard the running simulation.
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("boids")
            .with_inner_size(PhysicalSize::new(2560, 1440));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(e) => {
                log::error!("creating the window failed: {e}");
                event_loop.exit();
                return;
            }
        };
        if let Err(e) = self.initialise(window) {
            log::error!("initialisation failed: {e}");
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                return;
            }
            WindowEvent::Resized(size) => {
                if let (Some(gpu), Some(surface)) = (&self.gpu, &mut self.surface) {
                    surface.resize(gpu, size.width, size.height);
                }
                self.camera.set_viewport(size.width, size.height);
                return;
            }
            WindowEvent::RedrawRequested => {
                self.run_frame();
                return;
            }
            _ => {}
        }

        self.input.handle(&event);
        let mut camera = self.camera;
        let actions = self.input.consume(&mut camera);
        // `set_viewport` is not touched by `consume`, so the camera can be written back wholesale.
        self.camera = camera;

        if actions.quit {
            event_loop.exit();
            return;
        }
        if actions.toggle_world {
            self.toggle_world();
        }
        if actions.reset {
            self.respawn();
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Poll-driven frame loop: draw as fast as the display allows, with no event needed.
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

/// Frames the simulation by advancing it when a redraw is requested.
///
/// `winit` delivers `RedrawRequested` inside `window_event`; handling it there rather than in
/// `about_to_wait` keeps the simulation step and the render in one place, which matters because they
/// share the ping-pong parity.
impl BoidsApp {
    /// Runs one frame: advance the simulation, then draw it.
    pub fn run_frame(&mut self) {
        if !self.is_initialised() {
            return;
        }
        let now = std::time::Instant::now();
        let dt = self.timestep(now);
        // Exponential smoothing with a short time constant: a raw per-frame number is unreadable, and
        // a long average hides the hitches worth seeing.
        self.smoothed_frame_time = self.smoothed_frame_time * 0.9 + dt * 0.1;

        if !self.input.paused {
            self.step_simulation(dt);
        }
        self.render_frame();
        self.update_title();
    }
}

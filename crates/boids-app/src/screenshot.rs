//! Headless screenshot mode.
//!
//! Renders one frame of the simulation's steady state to a PNG with no window and no event loop.
//!
//! This exists for three reasons, in order of importance:
//!
//! * it makes the visual result reviewable without running the app, which matters when the machine
//!   running the simulation and the person looking at it are not the same,
//! * it is the basis of visual regression: with a fixed seed and a fixed number of steps, the output is
//!   deterministic, so two runs can be compared byte for byte,
//! * it works on a machine with no display server at all, which is where most of this development
//!   happened.
//!
//! The frame is rendered into an off-screen colour target with the same `Renderer` the window path uses,
//! so nothing here can drift from what the app draws.

use boids_core::camera::OrbitCamera;
use boids_core::config::SimConfig;
use boids_core::layout::{SceneUniform, SimMode};
use boids_gpu::context::{GpuContext, GpuContextDescriptor};
use boids_gpu::sim::{SimPipelines, SimResources, Strategy};
use boids_render::renderer::{FrameInput, Renderer};
use boids_render::targets::DEPTH_FORMAT;
use boids_render::{png, readback, SceneBinding};
use glam::Vec2;

/// Everything a screenshot needs.
#[derive(Debug, Clone)]
pub struct ScreenshotRequest {
    /// Where to write the PNG.
    pub path: std::path::PathBuf,
    /// Which world to draw.
    pub mode: SimMode,
    /// Number of agents.
    pub num_agents: usize,
    /// Steps to simulate before drawing, so the swarm has had time to form.
    pub warmup_steps: usize,
    /// Output size in pixels.
    pub width: u32,
    /// Output size in pixels.
    pub height: u32,
    /// World seed, so a screenshot is reproducible.
    pub seed: u64,
    /// Camera distance as a multiple of the world's half extent.
    pub camera_distance: f32,
    /// Camera elevation in radians.
    pub camera_pitch: f32,
}

impl Default for ScreenshotRequest {
    fn default() -> Self {
        Self {
            path: std::path::PathBuf::from("boids.png"),
            mode: SimMode::Fish,
            num_agents: 2_000,
            // Enough steps for the initial random velocities to organise into flocks. Fewer and the
            // screenshot shows the spawn noise; many more and a slow machine waits for nothing.
            warmup_steps: 240,
            width: 1_280,
            height: 720,
            seed: 1,
            camera_distance: 1.1,
            camera_pitch: 0.22,
        }
    }
}

/// Renders one frame and writes it to a PNG.
///
/// # Errors
/// Returns a description when the device is unavailable, a buffer cannot be read back, or the file
/// cannot be written.
pub fn capture(request: &ScreenshotRequest) -> Result<(), String> {
    let ctx = GpuContext::new(&GpuContextDescriptor::default())?;
    log::info!("adapter: {}", ctx.adapter_summary());

    // The world is sized for a screenshot rather than for the app: the camera is placed relative to the
    // world, and a default world sized for 100k agents would put 2000 of them at a speck in the middle.
    let mut config = SimConfig::for_mode(request.mode, request.num_agents);
    config.bounds_half = glam::Vec3::splat(config.r_percept * 9.0);

    let swarm = boids_core::spawn::spawn_swarm(&config, request.seed);
    let mut sim = SimResources::new(&ctx, &config);
    boids_gpu::transfer::upload_boids(&ctx.queue, &sim.boids[0], &swarm);
    boids_gpu::transfer::upload_boids(&ctx.queue, &sim.boids[1], &swarm);
    let pipes = SimPipelines::new(&ctx, &sim);

    let params = config.to_params(0.0, config.dt);
    let interaction = SimConfig::idle_interaction();
    sim.write_params(&ctx.queue, &params);
    sim.write_interaction(&ctx.queue, &interaction);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("screenshot warmup"),
        });
    for _ in 0..request.warmup_steps {
        pipes.record_step(&mut encoder, &sim, Strategy::Naive, &mut None);
        sim.swap();
    }
    ctx.queue.submit(Some(encoder.finish()));

    let mut renderer = Renderer::new(
        &ctx,
        [&sim.boids[0], &sim.boids[1]],
        wgpu::TextureFormat::Rgba8UnormSrgb,
        true,
        request.width,
        request.height,
    );

    let color = create_render_target(&ctx, request.width, request.height);
    let depth = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot depth"),
        size: wgpu::Extent3d {
            width: request.width,
            height: request.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    #[allow(clippy::cast_precision_loss)]
    let (width_f, height_f) = (request.width as f32, request.height as f32);
    let camera = OrbitCamera {
        target: glam::Vec3::ZERO,
        distance: config.bounds_half.length() * request.camera_distance,
        pitch: request.camera_pitch,
        yaw: 0.7,
        fov_y: 52f32.to_radians(),
        near: 0.5,
        far: config.bounds_half.length() * 12.0,
        aspect: width_f / height_f,
    };
    let scene: SceneUniform = boids_gpu::mesh_profile::scene_uniform(
        camera.to_uniform(0.0, Vec2::new(width_f, height_f)),
        config.mode,
        config.max_speed,
        config.r_percept,
    );

    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    #[allow(clippy::cast_possible_truncation)]
    let num_agents = config.num_boids as u32;
    renderer.render_into(
        &ctx,
        &color_view,
        &depth_view,
        &scene,
        FrameInput {
            binding: SceneBinding::new(sim.read_index()),
            num_agents,
        },
    );

    let pixels = readback::read_texture_rgba(&ctx, &color, request.width, request.height)?;
    png::write_png_rgba(&request.path, request.width, request.height, &pixels)?;
    log::info!(
        "wrote {} ({}x{}, {} agents, mode {:?}, {} warmup steps)",
        request.path.display(),
        request.width,
        request.height,
        config.num_boids,
        config.mode,
        request.warmup_steps
    );
    Ok(())
}

fn create_render_target(ctx: &GpuContext, width: u32, height: u32) -> wgpu::Texture {
    ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

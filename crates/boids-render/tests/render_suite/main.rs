//! Off-screen render checks: renders real frames and inspects the pixels.
//!
//! This is the only automated way to know that the render path produced a picture at all. Every other
//! check in the project would pass just as happily if the vertex shader emitted degenerate triangles,
//! the depth test rejected everything, or the instance count was zero.
//!
//! Each check fails for a different reason:
//!
//! * `background_has_structure` - shading runs, the gradient exists, and the sky is oriented the way
//!   up is, which catches a flipped Y in the ray reconstruction.
//! * `agents_contribute_pixels` - the instanced draw contributes geometry, which catches a mesh
//!   function that returns nothing or a draw with zero instances.
//! * `depth_is_written` - geometry survived the depth test and is within the world's distance range,
//!   which catches a wrong winding, a wrong compare function, and a camera pointed the wrong way.
//! * `worlds_look_different` - the mode reaches the shaders, which catches a stale scene uniform.
//!
//! Run as a hand-rolled main: the device must be created and dropped on the main thread, for the reason
//! documented in `boids-gpu/tests/gpu/main.rs`.

mod common;

use std::time::Instant;

use boids_core::config::SimConfig;
use boids_core::camera::OrbitCamera;
use boids_core::layout::{SceneUniform, SimMode};
use boids_gpu::context::GpuContext;
use boids_gpu::sim::{SimPipelines, SimResources, Strategy};
use boids_render::renderer::{FrameInput, Renderer};
use boids_render::targets::DEPTH_FORMAT;
use boids_render::SceneBinding;
use common::{COLOR_FORMAT, HEIGHT, WIDTH};
use glam::{Vec2, Vec3};

/// A named check.
type NamedCheck = (&'static str, fn(&GpuContext) -> CheckResult);

/// The result of a check that ran to completion.
///
/// A capability gap on the current backend is reported as a skip rather than a failure. The
/// alternative, failing, would make the suite red on a backend that simply cannot do the thing being
/// checked, and a suite that is red for a reason nobody can fix is a suite nobody reads.
enum Outcome {
    /// The property held.
    Pass,
    /// The property could not be checked here, with the reason.
    Skipped(String),
}

/// A check: either an outcome or a failure description.
type CheckResult = Result<Outcome, String>;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .parse_env("RUST_LOG")
        .try_init()
        .ok();

    let ctx = common::test_context();
    println!(
        "adapter: {} (backend {:?}, type {:?}){}",
        ctx.info.name,
        ctx.info.backend,
        ctx.info.device_type,
        if common::is_software(&ctx) { " [SOFTWARE]" } else { "" }
    );
    println!("target: {WIDTH}x{HEIGHT} {COLOR_FORMAT:?}\n");

    let cases: Vec<NamedCheck> = vec![
        ("background_has_structure", background_has_structure),
        ("agents_contribute_pixels", agents_contribute_pixels),
        ("depth_is_written", depth_is_written),
        ("worlds_look_different", worlds_look_different),
    ];

    let mut passed = 0usize;
    let mut skipped = 0usize;
    let mut failures: Vec<(&str, String)> = Vec::new();
    for (name, check) in cases {
        let start = Instant::now();
        print!("{name} ... ");
        match check(&ctx) {
            Ok(Outcome::Pass) => {
                passed += 1;
                println!("ok ({:.2?})", start.elapsed());
            }
            Ok(Outcome::Skipped(reason)) => {
                skipped += 1;
                println!("skipped ({:.2?})", start.elapsed());
                println!("    {reason}\n");
            }
            Err(message) => {
                println!("FAILED ({:.2?})", start.elapsed());
                println!("    {message}\n");
                failures.push((name, message));
            }
        }
    }

    println!("{passed} passed, {skipped} skipped, {} failed", failures.len());
    drop(ctx);
    if failures.is_empty() {
        return;
    }
    for (name, message) in failures {
        println!("FAILED {name}\n    {message}");
    }
    std::process::exit(1);
}

/// A rendered frame plus the depth values written alongside it.
struct Captured {
    color: Vec<u8>,
    /// Depth samples, or `None` when the backend cannot copy depth to a buffer. Reading depth back
    /// requires `DEPTH_TEXTURE_AND_BUFFER_COPIES`, which GL does not provide; see `depth_is_written`.
    depth: Option<Vec<f32>>,
}

impl Captured {
    fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * WIDTH + x) * 4) as usize;
        [
            self.color[i],
            self.color[i + 1],
            self.color[i + 2],
            self.color[i + 3],
        ]
    }

    /// Pixels where the two frames differ by more than `threshold` in any channel.
    fn differing_pixels(&self, other: &Self, threshold: u8) -> usize {
        self.color
            .chunks_exact(4)
            .zip(other.color.chunks_exact(4))
            .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.abs_diff(*y) > threshold))
            .count()
    }

    /// Mean absolute difference between horizontally adjacent pixels: a cheap "is there any
    /// structure at all" measure that a flat fill fails.
    fn horizontal_detail(&self) -> f32 {
        let mut sum = 0u64;
        for y in 0..HEIGHT {
            for x in 1..WIDTH {
                let a = self.pixel(x, y);
                let b = self.pixel(x - 1, y);
                for c in 0..3 {
                    sum += u64::from(a[c].abs_diff(b[c]));
                }
            }
        }
        sum as f32 / ((WIDTH - 1) * HEIGHT * 3) as f32
    }

    /// Depth samples, or a panic if the backend could not produce them. Only call after checking the
    /// capability.
    fn depth(&self) -> &[f32] {
        self.depth
            .as_deref()
            .expect("depth readback was requested but the backend does not support it")
    }

    /// Fraction of depth samples closer than the far plane, i.e. covered by some geometry.
    fn depth_coverage(&self) -> f32 {
        let written = self.depth().iter().filter(|d| **d < 0.999_9).count();
        written as f32 / self.depth().len() as f32
    }

    /// Whether every written depth sample is at a plausible distance.
    fn depth_is_plausible(&self) -> bool {
        self.depth()
            .iter()
            .filter(|d| **d < 0.999_9)
            .all(|d| (0.0..1.0).contains(d))
    }

    /// The closest written depth sample.
    fn depth_nearest(&self) -> f32 {
        self.depth()
            .iter()
            .filter(|d| **d < 0.999_9)
            .fold(f32::INFINITY, |a, b| a.min(*b))
    }
}

/// One simulation plus one renderer, drawing into attachments that can be read back.
struct Harness<'a> {
    ctx: &'a GpuContext,
    sim: SimResources,
    pipes: SimPipelines,
    renderer: Renderer,
    config: SimConfig,
    /// Viewpoint for the next capture. Checks adjust the pitch to look at the horizon from above and
    /// from below.
    camera: OrbitCamera,
    color: wgpu::Texture,
    depth: wgpu::Texture,
}

impl<'a> Harness<'a> {
    fn new(ctx: &'a GpuContext, mode: SimMode, agents: usize) -> Self {
        let mut config = SimConfig::for_mode(mode, agents);
        // Shrink the world so the agents are near the camera. The default world is sized for a hundred
        // thousand agents and would leave a few thousand as an invisible speck.
        config.bounds_half = Vec3::splat(config.r_percept * 8.0);

        let swarm = boids_core::spawn::spawn_swarm(&config, 4);
        let sim = SimResources::new(ctx, &config);
        boids_gpu::transfer::upload_boids(&ctx.queue, &sim.boids[0], &swarm);
        boids_gpu::transfer::upload_boids(&ctx.queue, &sim.boids[1], &swarm);
        let pipes = SimPipelines::new(ctx, &sim);

        let renderer = Renderer::new(
            ctx,
            [&sim.boids[0], &sim.boids[1]],
            COLOR_FORMAT,
            true,
            WIDTH,
            HEIGHT,
        );

        let color = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen color"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("readable depth"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let camera = OrbitCamera {
            distance: config.bounds_half.length() * 1.15,
            aspect: WIDTH as f32 / HEIGHT as f32,
            fov_y: 55f32.to_radians(),
            near: 0.5,
            far: 10_000.0,
            ..Default::default()
        };

        Self {
            ctx,
            sim,
            pipes,
            renderer,
            config,
            camera,
            color,
            depth,
        }
    }

    /// Advances the swarm by `steps`, then draws a frame and reads both attachments back.
    fn capture(&mut self, steps: usize, agents: u32) -> Captured {
        let params = self.config.to_params(0.5, self.config.dt);
        let interaction = SimConfig::idle_interaction();
        self.sim.write_params(&self.ctx.queue, &params);
        self.sim.write_interaction(&self.ctx.queue, &interaction);

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..steps {
            self.pipes
                .record_integrate(&mut encoder, &self.sim, Strategy::Naive);
            self.sim.swap();
        }
        self.ctx.queue.submit(Some(encoder.finish()));

        let camera = self.camera;
        let viewport = Vec2::new(WIDTH as f32, HEIGHT as f32);
        let scene: SceneUniform = boids_gpu::mesh_profile::scene_uniform(
            camera.to_uniform(0.5, viewport),
            self.config.mode,
            self.config.max_speed,
            self.config.r_percept,
        );

        let color_view = self.color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = self.depth.create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render_into(
            self.ctx,
            &color_view,
            &depth_view,
            &scene,
            FrameInput {
                binding: SceneBinding::new(self.sim.read_index()),
                num_agents: agents,
            },
        );

        let color = read_color(self.ctx, &self.color);
        let depth = read_depth(self.ctx, &self.depth);
        Captured { color, depth }
    }
}

fn read_color(ctx: &GpuContext, texture: &wgpu::Texture) -> Vec<u8> {
    let bytes = read_texture(ctx, texture, 4, wgpu::TextureAspect::All);
    assert_eq!(bytes.len(), (WIDTH * HEIGHT * 4) as usize);
    bytes
}

/// Reads the depth attachment, or returns `None` when the backend cannot copy depth to a buffer.
///
/// The capability is checked here rather than at the call site because every check renders a frame, and
/// a copy that the backend rejects is a validation error that aborts the process through the
/// uncaptured-error handler. Guarding it in one place is what keeps the other checks runnable on a
/// backend with the gap.
fn read_depth(ctx: &GpuContext, texture: &wgpu::Texture) -> Option<Vec<f32>> {
    if !supports_depth_copy(ctx) {
        return None;
    }
    let bytes = read_texture(ctx, texture, 4, wgpu::TextureAspect::DepthOnly);
    Some(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

/// Whether this backend can copy a depth texture into a buffer.
fn supports_depth_copy(ctx: &GpuContext) -> bool {
    ctx.adapter
        .get_downlevel_capabilities()
        .flags
        .contains(wgpu::DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES)
}

/// Copies a texture into a staging buffer and maps it, with the tight row pitch the targets use.
fn read_texture(
    ctx: &GpuContext,
    texture: &wgpu::Texture,
    bytes_per_texel: u32,
    aspect: wgpu::TextureAspect,
) -> Vec<u8> {
    let row_bytes = WIDTH * bytes_per_texel;
    // `copy_texture_to_buffer` requires the row pitch to be a multiple of 256 for a non-trivial copy
    // on some backends; `WIDTH = 400` at 4 bytes is 1600, which is not. Padding the staging buffer and
    // stripping the padding on the way out keeps the texture layout simple.
    let padded_row = row_bytes.next_multiple_of(256);
    let size = u64::from(padded_row) * u64::from(HEIGHT);

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("mapping {size} bytes failed: {e}"),
        Err(e) => panic!("readback channel closed: {e}"),
    }

    let view = slice.get_mapped_range().expect("mapped range");
    let mut out = Vec::with_capacity((WIDTH * HEIGHT * bytes_per_texel) as usize);
    for row in 0..HEIGHT as usize {
        let start = row * padded_row as usize;
        out.extend_from_slice(&view[start..start + row_bytes as usize]);
    }
    drop(view);
    staging.unmap();
    out
}

// -------------------------------------------------------------------------------------------
// Checks
// -------------------------------------------------------------------------------------------

/// Luma of a pixel, as a plain sum of the colour channels. Only used for comparisons.
fn luma(p: [u8; 4]) -> u32 {
    u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])
}

/// Verifies that "up" in the world maps to up on the screen.
///
/// The sky world shades a bright band at the horizon, a darker zenith above it, and a dark ground
/// below. A flipped Y in the ray reconstruction in `render/background.wgsl` inverts that, and nothing
/// else in the suite would notice: the picture still looks like a plausible gradient, it is just upside
/// down.
///
/// The camera's own pitch decides where the horizon lands in the frame, so the check looks at the
/// horizon from both sides rather than assuming a position. That independence matters: with the
/// default orbit pitch the horizon sits near the top of the frame, and a check that assumed otherwise
/// would be testing the camera, not the shader.
fn background_has_structure(ctx: &GpuContext) -> CheckResult {
    let mut harness = Harness::new(ctx, SimMode::Birds, 64);

    // Looking down: the frame is mostly ground, with sky only above the horizon near the top.
    harness.camera.pitch = 0.6;
    let looking_down = harness.capture(0, 64);
    let detail = looking_down.horizontal_detail();
    if detail < 0.05 {
        return Err(format!(
            "the frame has no horizontal structure (mean adjacent-pixel difference {detail:.4}); \
             the background pass is not running, or every pixel is the same colour"
        ));
    }
    let top = looking_down.pixel(WIDTH / 2, 2);
    let bottom = looking_down.pixel(WIDTH / 2, HEIGHT - 2);
    if luma(top) <= luma(bottom) {
        return Err(format!(
            "looking down, the top of the frame ({top:?}, luma {}) should be brighter than the \
             bottom ({bottom:?}, luma {}): the sky belongs above the ground. The ray reconstruction \
             or the Y flip in render/background.wgsl is inverted.",
            luma(top),
            luma(bottom)
        ));
    }

    // Looking up: the frame is mostly sky, and the zenith above must be darker than the horizon below.
    harness.camera.pitch = -0.6;
    let looking_up = harness.capture(0, 64);
    let zenith = looking_up.pixel(WIDTH / 2, 2);
    let near_horizon = looking_up.pixel(WIDTH / 2, HEIGHT - 2);
    if luma(zenith) >= luma(near_horizon) {
        return Err(format!(
            "looking up, the zenith at the top of the frame ({zenith:?}, luma {}) should be darker \
             than the horizon at the bottom ({near_horizon:?}, luma {}); the vertical gradient in \
             render/background.wgsl is inverted",
            luma(zenith),
            luma(near_horizon)
        ));
    }

    println!(
        "\n    detail {detail:.3}; down: top {} > bottom {}; up: zenith {} < horizon {}",
        luma(top),
        luma(bottom),
        luma(zenith),
        luma(near_horizon)
    );
    Ok(Outcome::Pass)
}

fn agents_contribute_pixels(ctx: &GpuContext) -> CheckResult {
    let mut with = Harness::new(ctx, SimMode::Birds, 3000);
    let with_agents = with.capture(30, 3000);

    let mut without = Harness::new(ctx, SimMode::Birds, 3000);
    let without_agents = without.capture(30, 0);

    let differing = with_agents.differing_pixels(&without_agents, 6);
    let fraction = differing as f32 / (WIDTH * HEIGHT) as f32;
    if fraction < 0.005 {
        return Err(format!(
            "only {differing} of {} pixels changed when the agents were removed ({:.3}%); the \
             instanced draw produces nothing. Check MESH_VERTICES against the triangle budget in \
             render/boid.wgsl, the instance count, and whether the mesh function returns degenerate \
             positions.",
            WIDTH * HEIGHT,
            fraction * 100.0
        ));
    }
    println!(
        "\n    {differing} pixels ({:.2}%) changed when the agents were removed",
        fraction * 100.0
    );
    Ok(Outcome::Pass)
}

fn depth_is_written(ctx: &GpuContext) -> CheckResult {
    // Reading depth back needs `DEPTH_TEXTURE_AND_BUFFER_COPIES`, which the GL backend does not
    // provide. The property is still enforced indirectly on such backends: the agent pipeline compares
    // `LessEqual` against a clear of 1.0, so any agent pixel that appears at all proved the depth test
    // against a correctly cleared buffer. The direct check below adds the value-range assertion.
    if !supports_depth_copy(ctx) {
        return Ok(Outcome::Skipped(format!(
            "backend {:?} does not support depth texture-to-buffer copies, so the depth attachment \
             cannot be inspected directly. On this backend the depth test is still exercised \
             indirectly by 'agents_contribute_pixels': the agent pipeline compares LessEqual against \
             a clear of 1.0, so a visible agent pixel proves the clear and the comparison are right.",
            ctx.info.backend
        )));
    }

    let mut harness = Harness::new(ctx, SimMode::Birds, 3000);
    let frame = harness.capture(30, 3000);
    let coverage = frame.depth_coverage();

    if coverage <= 0.0 {
        return Err(
            "the depth attachment was never written closer than the far plane, so nothing was \
             rasterised. Check the winding, the depth compare function (it must be LessEqual against \
             a clear of 1.0), and that the camera is not looking away from the swarm."
                .to_string(),
        );
    }
    // The background writes no depth, so anything past a modest fraction of the frame would mean the
    // depth clear is not happening and the pass is testing against last frame's values.
    if coverage > 0.9 {
        return Err(format!(
            "{:.1}% of depth samples were written, which is far more than the swarm can cover. The \
             depth attachment is probably not being cleared between frames.",
            coverage * 100.0
        ));
    }
    if !frame.depth_is_plausible() {
        return Err(
            "the depth attachment contains values outside [0, 1), which means NaN or inf positions \
             reached the vertex shader"
                .to_string(),
        );
    }

    // The agent positions are finite and inside the world, so the depth range they occupy must be
    // small: every agent is within the world bounds of the camera, well inside the far plane.
    let min_depth = frame.depth_nearest();
    if min_depth < 0.001 {
        return Err(format!(
            "the closest depth sample is {min_depth:.5}, which is essentially on the near plane; \
             positions are probably in the wrong units or the projection is wrong"
        ));
    }

    println!(
        "\n    depth coverage {:.2}%, closest sample {min_depth:.4}",
        coverage * 100.0
    );
    Ok(Outcome::Pass)
}

fn worlds_look_different(ctx: &GpuContext) -> CheckResult {
    let mut sky = Harness::new(ctx, SimMode::Birds, 1500);
    let sky_frame = sky.capture(30, 1500);
    let mut sea = Harness::new(ctx, SimMode::Fish, 1500);
    let sea_frame = sea.capture(30, 1500);

    let differing = sky_frame.differing_pixels(&sea_frame, 24);
    let fraction = differing as f32 / (WIDTH * HEIGHT) as f32;
    if fraction < 0.5 {
        return Err(format!(
            "only {:.1}% of pixels differ between the two worlds; the mode is not reaching the \
             shaders. Check that SceneUniform::mesh.variant, fog_color and fog_density are rebuilt \
             when the mode changes, and that the scene uniform is uploaded every frame.",
            fraction * 100.0
        ));
    }
    println!("\n    {:.1}% of pixels differ between sky and sea", fraction * 100.0);
    Ok(Outcome::Pass)
}

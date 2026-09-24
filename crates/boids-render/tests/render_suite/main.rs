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
//! * `ocean_writes_depth` - the underwater raymarch found geometry and reported its distance, which
//!   catches a march that never hits, a wrong `frag_depth`, and a surface plane in the wrong place.
//! * `underwater_is_blue_and_lit` - the frame is neither black nor grey: the medium scatters light and
//!   water extinguishes red before blue, which is the one property that makes it read as water.
//! * `bloom_adds_light` - the pyramid contributes light to the composite, which catches a bright pass
//!   that thresholds everything away, a pyramid that never gets read, and a composite that ignores it.
//! * `focus_marker_is_visible` - the cursor's influence point is drawn, which catches the marker being
//!   culled by the environment depth or the interaction uniform never reaching the shader.
//! * `landmark_is_drawn` - the castle mesh reaches the screen through the landmark pass, which catches
//!   a mesh builder that returns nothing, a zero instance count, or binding 9 never reaching the
//!   vertex shader. The placement itself is checked on the CPU in `boids-scene`.
//!
//! Run as a hand-rolled main: the device must be created and dropped on the main thread, for the reason
//! documented in `boids-gpu/tests/gpu/main.rs`.

mod common;

use std::time::Instant;

use boids_core::camera::OrbitCamera;
use boids_core::config::SimConfig;
use boids_core::layout::{
    InteractionMode, InteractionUniforms, PostParams, SceneUniform, SimMode, WaterParams,
};
use boids_gpu::context::GpuContext;
use boids_gpu::sim::{SimPipelines, SimResources, Strategy};
use boids_render::renderer::{FrameInput, Renderer};
use boids_render::targets::DEPTH_FORMAT;
use boids_render::{Model, SceneBinding};
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
        if common::is_software(&ctx) {
            " [SOFTWARE]"
        } else {
            ""
        }
    );
    println!("target: {WIDTH}x{HEIGHT} {COLOR_FORMAT:?}\n");

    let cases: Vec<NamedCheck> = vec![
        ("background_has_structure", background_has_structure),
        (
            "terrain_grounds_the_sky_world",
            terrain_grounds_the_sky_world,
        ),
        ("landmark_is_drawn", landmark_is_drawn),
        ("agents_contribute_pixels", agents_contribute_pixels),
        ("depth_is_written", depth_is_written),
        ("worlds_look_different", worlds_look_different),
        ("ocean_writes_depth", ocean_writes_depth),
        ("underwater_is_blue_and_lit", underwater_is_blue_and_lit),
        ("bloom_adds_light", bloom_adds_light),
        ("focus_marker_is_visible", focus_marker_is_visible),
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

    println!(
        "{passed} passed, {skipped} skipped, {} failed",
        failures.len()
    );
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
            .as_chunks::<4>()
            .0
            .iter()
            .zip(other.color.as_chunks::<4>().0.iter())
            .filter(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .any(|(x, y)| x.abs_diff(*y) > threshold)
            })
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

    /// Every written depth sample, converted back to metres from the camera.
    ///
    /// The attachment holds NDC depth, which is `far / (far - near) * (1 - near / d)` for a
    /// `directx`-convention projection. Reading it as a distance is what makes an assertion about the
    /// environment's *relief* meaningful: in NDC, a seafloor at 150 m and a surface at 200 m differ in
    /// the fourth decimal place, so a threshold on the raw values would either be vacuous or
    /// arbitrary.
    fn depth_metres(&self, near: f32, far: f32) -> Vec<f32> {
        self.depth()
            .iter()
            .filter(|z| **z < 0.999_9)
            .map(|z| near * far / (far - z * (far - near)))
            .collect()
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
    water: WaterParams,
    sky: boids_core::layout::SkyParams,
    post: PostParams,
    /// Cursor state for the next capture. Checks that do not care leave it idle.
    interaction: InteractionUniforms,
}

impl<'a> Harness<'a> {
    fn new(ctx: &'a GpuContext, mode: SimMode, agents: usize) -> Self {
        let mut config = SimConfig::for_mode(mode, agents);
        // Shrink the world so the agents are near the camera. The default world is sized for a hundred
        // thousand agents and would leave a few thousand as an invisible speck. `resize_world` and not
        // a bare assignment: the terrain's amplitude, the seafloor and the spawn centre are all
        // derived from the extent, and the app world's 110 m mountains in a 250 m box are a wall.
        config.resize_world(Vec3::splat(config.r_percept * 8.0));

        let swarm = boids_core::spawn::spawn_swarm(&config, 4);
        let sim = SimResources::new(ctx, &config);
        boids_gpu::transfer::upload_boids(&ctx.queue, &sim.boids[0], &swarm);
        boids_gpu::transfer::upload_boids(&ctx.queue, &sim.boids[1], &swarm);
        let pipes = SimPipelines::new(ctx, &sim);

        let renderer = Renderer::new(
            ctx,
            [&sim.boids[0], &sim.boids[1]],
            &config,
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
            water: boids_scene::water_params(&config),
            sky: boids_scene::sky_params(config.mode),
            post: boids_scene::post_params(config.mode, 0.5),
            interaction: SimConfig::idle_interaction(),
            config,
            camera,
            color,
            depth,
        }
    }

    /// Advances the swarm by `steps`, then draws a frame and reads both attachments back.
    fn capture(&mut self, steps: usize, agents: u32) -> Captured {
        let params = self.config.to_params(0.5, self.config.dt);
        self.sim.write_params(&self.ctx.queue, &params);
        self.sim
            .write_interaction(&self.ctx.queue, &self.interaction);

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..steps {
            self.pipes
                .record_step(&mut encoder, &self.sim, Strategy::Naive, &mut None);
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

        let color_view = self
            .color
            .create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = self
            .depth
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render_into(
            self.ctx,
            &color_view,
            &depth_view,
            &scene,
            FrameInput {
                binding: SceneBinding::new(self.sim.read_index()),
                num_agents: agents,
                world: self.config.mode,
                water: self.water,
                interaction: self.interaction,
                post: self.post,
                sky: self.sky,
            },
        );

        let color = read_color(self.ctx, &self.color);
        let depth = read_depth(self.ctx, &self.depth);
        Captured { color, depth }
    }

    /// Draws one model in the viewer's studio and reads the colour target back.
    ///
    /// The same `render_model` the interactive viewer and `--model` use, into the same off-screen
    /// target, so a model checked here is the model the app shows. No depth is read: the studio has
    /// nothing behind the model to test against, so the attachment is written and discarded.
    fn capture_model(&mut self, model: Model, camera: &OrbitCamera) -> Captured {
        let color_view = self
            .color
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render_model(self.ctx, &color_view, model, camera, 0.0);
        let color = read_color(self.ctx, &self.color);
        Captured { color, depth: None }
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
///
/// It samples a column off the frame's centre, not the centre itself: the landmark is placed at the
/// world's centre, so the middle column is castle wall rather than background, and a sky check that
/// read a battlement would fail for the wrong reason.
fn background_has_structure(ctx: &GpuContext) -> CheckResult {
    let mut harness = Harness::new(ctx, SimMode::Birds, 64);
    let column = WIDTH / 4;

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
    let top = looking_down.pixel(column, 2);
    let bottom = looking_down.pixel(column, HEIGHT - 2);
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
    let zenith = looking_up.pixel(column, 2);
    let near_horizon = looking_up.pixel(column, HEIGHT - 2);
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

/// The ground has to be a surface, not a net.
///
/// The grid's two triangles per quad must wind the same way; if the second is listed in the order
/// that flips it, back-face culling removes exactly half the ground and the sky shows through the
/// holes. The frame still has structure and still looks plausible at a glance, so this is checked by
/// colour: the ground palettes (forest, dunes, canyon, rock) are all red-dominant or neutral, while
/// the sky and its below-horizon haze are blue-dominant. Looking down, the bottom of the frame must
/// be ground.
fn terrain_grounds_the_sky_world(ctx: &GpuContext) -> CheckResult {
    let mut harness = Harness::new(ctx, SimMode::Birds, 64);
    harness.camera.pitch = 0.6;
    let frame = harness.capture(0, 64);

    let rows = 20.min(HEIGHT);
    let mut sky_px = 0usize;
    let mut total = 0usize;
    for y in (HEIGHT - rows)..HEIGHT {
        for x in 0..WIDTH {
            let p = frame.pixel(x, y);
            total += 1;
            if i32::from(p[2]) > i32::from(p[0]) + 12 {
                sky_px += 1;
            }
        }
    }
    let fraction = sky_px as f32 / total as f32;
    if fraction > 0.25 {
        return Err(format!(
            "{:.0}% of the bottom {rows} rows are blue-dominant, so the sky is showing through the \
             ground and the terrain mesh has holes. Check the winding in `quad_corner` in \
             render/terrain.wgsl: half the grid triangles facing the wrong way are culled and leave \
             a see-through net.",
            fraction * 100.0
        ));
    }
    println!(
        "\n    ground covers the lower frame (sky through it {:.1}%)",
        fraction * 100.0
    );
    Ok(Outcome::Pass)
}

/// The castle has to be drawn, not only placed.
///
/// The placement is a CPU search over the terrain and the reef, checked in `boids-scene`. What no CPU
/// test can see is whether the landmark pass turns that placement into geometry: a mesh builder that
/// returns nothing, a zero instance count, a winding culled to nothing, or binding 9 never reaching
/// the vertex shader would each leave a frame with no castle in it while every other check stayed
/// green. The viewer's studio isolates the pass - the castle is the only thing drawn against a flat
/// background - and comparing two framings of it fails for exactly one reason: a pass that draws
/// nothing produces two identical frames.
fn landmark_is_drawn(ctx: &GpuContext) -> CheckResult {
    let mut harness = Harness::new(ctx, SimMode::Birds, 64);
    let framed = |distance: f32| OrbitCamera {
        target: Model::Castle.target(),
        distance,
        yaw: 0.7,
        pitch: Model::Castle.pitch(),
        fov_y: 45f32.to_radians(),
        near: 0.02,
        far: 100.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
    };
    // Two distances: the silhouette is a different size, so a pass that draws the mesh changes many
    // pixels and a pass that draws nothing changes none. The post chain is screen-space, so the
    // background, its vignette and its grain are identical between the two and cancel.
    let near = harness.capture_model(Model::Castle, &framed(Model::Castle.distance() * 0.8));
    let far = harness.capture_model(Model::Castle, &framed(Model::Castle.distance() * 1.2));

    let differing = near.differing_pixels(&far, 6);
    let fraction = differing as f32 / (WIDTH * HEIGHT) as f32;
    if fraction < 0.01 {
        return Err(format!(
            "only {differing} of {} pixels changed when the castle was reframed ({:.3}%); the \
             landmark pass draws no geometry. Check `castle_mesh` in boids-scene/src/mesh.rs, the \
             instance count from `LandmarkGpu::upload`, and that binding 9 reaches the vertex shader \
             in render/landmark.wgsl.",
            WIDTH * HEIGHT,
            fraction * 100.0
        ));
    }
    println!(
        "\n    reframing the castle changed {differing} pixels ({:.2}%)",
        fraction * 100.0
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
    // The terrain writes depth across the lower frame now, so a large coverage is expected; what must
    // still be true is that the sky above it left the far plane clear, i.e. that the attachment was
    // cleared rather than left holding last frame's values. A completely written frame would mean the
    // clear is not happening.
    if coverage > 0.999 {
        return Err(format!(
            "{:.1}% of depth samples were written, leaving no sky at the far plane. The depth \
             attachment is probably not being cleared between frames.",
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
    println!(
        "\n    {:.1}% of pixels differ between sky and sea",
        fraction * 100.0
    );
    Ok(Outcome::Pass)
}

// -------------------------------------------------------------------------------------------
// HDR pass checks
// -------------------------------------------------------------------------------------------

/// The underwater environment has to produce real distances into the depth attachment.
///
/// This is what makes the fish occlude against the reef at all: the pass writes depth by hand from the
/// raymarched hit, so a march that never hits anything, a `frag_depth` that is not in [0, 1], or a
/// surface plane below the camera all show up here rather than as agents floating in front of rock.
fn ocean_writes_depth(ctx: &GpuContext) -> CheckResult {
    if !supports_depth_copy(ctx) {
        return Ok(Outcome::Skipped(format!(
            "backend {:?} cannot copy depth to a buffer, so the raymarch's distance cannot be \
             inspected. The pass is still exercised: its output is what `underwater_is_blue_and_lit` \
             reads from the colour target.",
            ctx.info.backend
        )));
    }

    let mut harness = Harness::new(ctx, SimMode::Fish, 1500);
    // Looking down at the seafloor from above the reef: the column here is short, so the whole frame
    // must be either rock or water, and both are at finite distances.
    harness.camera.pitch = 0.45;
    let frame = harness.capture(20, 1500);

    let coverage = frame.depth_coverage();
    if coverage < 0.5 {
        return Err(format!(
            "only {:.1}% of the frame has a distance written by the ocean raymarch. The march is \
             probably stepping past the seafloor (check the maximum step against the depth of the \
             world) or `frag_depth` is being left at the far plane.",
            coverage * 100.0
        ));
    }
    if !frame.depth_is_plausible() {
        return Err(
            "the ocean wrote depth outside [0, 1), which means the clip-space conversion in \
             render/ocean.wgsl is wrong rather than the march"
                .to_string(),
        );
    }

    // And the distances must have real relief: a single constant distance would mean every ray hit
    // the same plane (a broken surface test, for instance), which a coverage check alone would pass.
    let distances = frame.depth_metres(harness.camera.near, harness.camera.far);
    let nearest = distances.iter().copied().fold(f32::INFINITY, f32::min);
    let furthest = distances.iter().copied().fold(0.0, f32::max);
    let spread = furthest - nearest;
    if spread < 1.0 {
        return Err(format!(
            "every raymarched distance is between {nearest:.1} m and {furthest:.1} m, so the \
             environment has no relief in the frame. Check that the reef field is evaluated rather \
             than the seafloor plane alone."
        ));
    }

    println!(
        "\n    ocean depth coverage {:.1}%, distances {nearest:.0}..{furthest:.0} m",
        coverage * 100.0
    );
    Ok(Outcome::Pass)
}

/// The underwater frame must be lit and blue-dominant.
///
/// Two properties in one check because they fail for the same reason - the medium model - and because
/// a frame that is black satisfies "blue-dominant" vacuously while a frame that is grey does not mean
/// the extinction is running per channel. Blue surviving longer than red is the entire visual
/// signature of water, and it is the one thing a single-coefficient fog cannot express.
fn underwater_is_blue_and_lit(ctx: &GpuContext) -> CheckResult {
    let mut harness = Harness::new(ctx, SimMode::Fish, 3000);
    let frame = harness.capture(30, 3000);

    let mut sum = [0u64; 3];
    for px in frame.color.as_chunks::<4>().0 {
        for c in 0..3 {
            sum[c] += u64::from(px[c]);
        }
    }
    let total = (WIDTH * HEIGHT) as f64;
    let mean = [
        sum[0] as f64 / total,
        sum[1] as f64 / total,
        sum[2] as f64 / total,
    ];

    if mean[2] < 4.0 {
        return Err(format!(
            "the underwater frame is essentially black (mean blue {:.2}); the medium scatters no \
             light. Check `WaterParams::scatter`, the exposure in `boids_scene::post_params`, and \
             whether the ocean pass's colour reaches the composite.",
            mean[2]
        ));
    }
    if mean[2] <= mean[0] {
        return Err(format!(
            "the underwater frame is not blue-dominant (mean RGB {mean:.2?}). Water must extinguish \
             red before blue: check the per-channel extinction in `boids_scene::water_params` and \
             that `render/water.wgsl` is the model the agents and the medium both use."
        ));
    }

    println!(
        "\n    underwater mean RGB {:.1}/{:.1}/{:.1}",
        mean[0], mean[1], mean[2]
    );
    Ok(Outcome::Pass)
}

/// Bloom has to add light to the composite.
///
/// The comparison is deliberately between two frames of the *same* scene with only `bloom_strength`
/// changed: any difference is the pyramid's contribution, so the check cannot pass because the scene
/// got brighter for an unrelated reason.
fn bloom_adds_light(ctx: &GpuContext) -> CheckResult {
    let mut with = Harness::new(ctx, SimMode::Fish, 2000);
    with.post.bloom_strength = 1.2;
    let with_bloom = with.capture(30, 2000);

    let mut without = Harness::new(ctx, SimMode::Fish, 2000);
    without.post.bloom_strength = 0.0;
    let without_bloom = without.capture(30, 2000);

    let sum = |frame: &Captured| -> u64 {
        frame
            .color
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| u64::from(px[0]) + u64::from(px[1]) + u64::from(px[2]))
            .sum()
    };
    let a = sum(&with_bloom);
    let b = sum(&without_bloom);
    if b == 0 {
        return Err("the frame without bloom is completely black; nothing to compare".to_string());
    }
    let gain = (a as f64 - b as f64) / b as f64;
    if gain < 0.005 {
        return Err(format!(
            "enabling bloom changed the frame by only {:.3}%; the pyramid contributes nothing. \
             Check the bright pass's threshold against the scene's actual HDR range (a threshold \
             above every pixel empties the pyramid), and that the composite samples level 0.",
            gain * 100.0
        ));
    }
    println!("\n    bloom raised total luma by {:.2}%", gain * 100.0);
    Ok(Outcome::Pass)
}

/// The cursor's influence point has to be visible where it is.
///
/// The marker is drawn analytically in the environment shaders rather than as geometry, which means a
/// broken interaction uniform, a marker placed behind the reef's depth, or a wrong ray reconstruction
/// would leave the frame identical. Comparing the idle frame against one with an active attractor over
/// the centre of the view isolates exactly that.
fn focus_marker_is_visible(ctx: &GpuContext) -> CheckResult {
    let mut idle = Harness::new(ctx, SimMode::Fish, 1000);
    let without = idle.capture(10, 1000);

    let mut active = Harness::new(ctx, SimMode::Fish, 1000);
    let focus = active.camera.target;
    active.interaction = InteractionUniforms {
        ray_origin: active.camera.eye().to_array(),
        mode: InteractionMode::Attract.as_u32(),
        focus_point: focus.to_array(),
        radius: active.config.r_percept * 6.0,
        strength: 10.0,
        falloff: 1.6,
        tangent: 0.0,
        _pad: 0.0,
    };
    let with = active.capture(10, 1000);

    let differing = with.differing_pixels(&without, 8);
    if differing < 8 {
        return Err(format!(
            "only {differing} pixels changed when the cursor attractor was placed at the centre of \
             the frame; the focus marker is not being drawn. Check that `render/water.wgsl`'s \
             `focus_marker` is called by the ocean pass, that the interaction uniform reaches \
             `SceneLayout`, and that the marker is not behind the environment's depth."
        ));
    }
    println!("\n    focus marker changed {differing} pixels");
    Ok(Outcome::Pass)
}

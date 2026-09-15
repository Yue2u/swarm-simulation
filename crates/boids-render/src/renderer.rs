//! The frame graph: one render pass over the scene, then the HDR post chain.
//!
//! # What one frame does
//!
//! ```text
//! 1. uniforms   SceneUniform, WaterParams, InteractionUniforms, PostParams   (four write_buffer)
//! 2. environment
//!      underwater  render/ocean.wgsl       full-screen SDF raymarch, writes depth
//!      sky         render/background.wgsl  full-screen gradient, no depth
//! 3. agents     render/boid.wgsl           one instanced draw, depth tested and writing
//! 4. bloom      bright, downsample, additive upsample into the pyramid
//! 5. composite  aberration, exposure, ACES tone map, grade -> the target
//! ```
//!
//! Steps 2 and 3 share one render pass and one depth attachment: the environment writes depth and the
//! agents compare against it, which is what puts a fish behind a column and in front of the seafloor.
//! Steps 4 and 5 are separate passes with no depth at all.
//!
//! # Why both worlds are built up front
//!
//! Switching worlds changes one `SimMode` and selects between two pipelines that were compiled at
//! startup. Nothing is allocated, no pipeline is recompiled and the driver state the GPU has already
//! warmed up stays warm, so the switch costs one frame and can be used as a live A/B comparison
//! instead of a multi-second hitch. That is also why the post chain is shared rather than owned per
//! world: the exposure differs, the chain does not.

use boids_core::layout::{InteractionUniforms, PostParams, SceneUniform, WaterParams};
use boids_core::layout::SimMode;
use boids_gpu::context::GpuContext;

use crate::background::BackgroundPass;
use crate::boid_pass::BoidPass;
use crate::ocean::OceanPass;
use crate::post::PostChain;
use crate::scene::{SceneBinding, SceneLayout};
use crate::targets::FrameTargets;

/// What the renderer needs to know about this frame that is not in the scene uniform.
#[derive(Debug, Clone, Copy)]
pub struct FrameInput {
    /// Which agent buffer parity to draw, i.e. what the last simulation step wrote.
    pub binding: SceneBinding,
    /// Number of live agents, so the instanced draw can be sized without reading a uniform.
    pub num_agents: u32,
    /// Which world's environment to draw.
    pub world: SimMode,
    /// Underwater medium and reef geometry.
    pub water: WaterParams,
    /// Cursor interaction state, for the focus marker.
    pub interaction: InteractionUniforms,
    /// Post-processing parameters.
    pub post: PostParams,
}

/// Draw calls, instances and triangles recorded for one frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// Explicit draw calls recorded.
    pub draw_calls: u32,
    /// Instance count submitted to the agent draw.
    pub instances: u64,
    /// Triangles submitted to the agent draw.
    pub triangles: u64,
}

/// The frame renderer for both worlds.
#[derive(Debug)]
pub struct Renderer {
    targets: FrameTargets,
    scene: SceneLayout,
    background: BackgroundPass,
    ocean: OceanPass,
    boids: BoidPass,
    post: PostChain,
    format: wgpu::TextureFormat,
    is_srgb: bool,
    last_size: (u32, u32),
}

impl Renderer {
    /// Builds every pipeline and target for a swapchain format and size.
    ///
    /// `boids` is the pair of ping-pong agent buffers. Both are bound up front; the renderer picks one
    /// per frame rather than rebuilding a bind group.
    #[must_use]
    pub fn new(
        ctx: &GpuContext,
        boids: [&wgpu::Buffer; 2],
        format: wgpu::TextureFormat,
        is_srgb: bool,
        width: u32,
        height: u32,
    ) -> Self {
        let scene = SceneLayout::new(ctx, boids);
        let targets = FrameTargets::new(&ctx.device, width, height);
        // The sky backdrop needs the format of the *scene* target now, not the swapchain: it draws
        // into the HDR intermediate like everything else, and the composite owns the swapchain.
        let background = BackgroundPass::new(ctx, &scene);
        let ocean = OceanPass::new(ctx, &scene);
        let boids_pass = BoidPass::new(ctx, &scene);
        let post = PostChain::new(
            ctx,
            format,
            is_srgb,
            &scene.post,
            targets.hdr_view(),
            width,
            height,
        );
        log::info!(
            "renderer: {}x{} swapchain {format:?} (srgb {is_srgb}), bloom {:?}",
            width.max(1),
            height.max(1),
            post.bloom_size()
        );
        Self {
            targets,
            scene,
            background,
            ocean,
            boids: boids_pass,
            post,
            format,
            is_srgb,
            last_size: (width.max(1), height.max(1)),
        }
    }

    /// The swapchain format every pipeline was built for.
    #[must_use]
    pub const fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Whether the swapchain format is sRGB.
    #[must_use]
    pub const fn is_srgb(&self) -> bool {
        self.is_srgb
    }

    /// Recreates size-dependent targets when the viewport changed.
    ///
    /// Called every frame rather than only on a window resize event: the surface can be reconfigured
    /// by the compositor without the app being told, and comparing sizes is cheaper than tracking
    /// whether every resize notification actually arrived.
    pub fn resize(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        if self.targets.resize(&ctx.device, width, height) {
            log::debug!(
                "render targets resized to {}x{}",
                self.targets.size.0,
                self.targets.size.1
            );
        }
        // The post chain is rebuilt whenever the size *or* the HDR view changed, which the chain
        // decides for itself by comparing sizes; after a resize both have.
        self.post
            .resize(ctx, &self.scene.post, self.targets.hdr_view(), width, height);
        self.last_size = self.targets.size;
    }

    /// Current render size in physical pixels.
    #[must_use]
    pub const fn size(&self) -> (u32, u32) {
        self.last_size
    }

    /// Draws one frame into the renderer's own depth attachment and tonemaps into `target`.
    ///
    /// The normal path for presenting to a window.
    pub fn render(
        &mut self,
        ctx: &GpuContext,
        target: &wgpu::TextureView,
        scene: &SceneUniform,
        input: FrameInput,
    ) -> FrameStats {
        let depth = self.targets.depth_view().clone();
        self.render_into(ctx, target, &depth, scene, input)
    }

    /// Draws one frame into caller-provided attachments.
    ///
    /// This is the real implementation; [`Renderer::render`] is the convenience wrapper around it.
    /// Taking the depth attachment as a parameter is what makes the scene reproducible off screen: the
    /// tests draw into colour and depth textures they can read back, which is the only way to verify
    /// that geometry was rasterised rather than merely that the code ran.
    ///
    /// The target must have the size the renderer was built for, because the colour half of the frame
    /// goes through the renderer's own HDR intermediate and its bloom pyramid; only the depth and the
    /// final composited image are caller-provided.
    pub fn render_into(
        &mut self,
        ctx: &GpuContext,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        scene: &SceneUniform,
        input: FrameInput,
    ) -> FrameStats {
        self.scene.write(
            &ctx.queue,
            scene,
            &input.water,
            &input.interaction,
            &input.post,
        );

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        let mut stats = FrameStats::default();
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.targets.hdr_view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Both the backdrop and the ocean cover every pixel, so loading the previous
                        // contents would be pure bandwidth. `Clear` also means the target is valid
                        // even if a future backdrop is restricted to part of the viewport.
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        // Cleared to 1.0, the far plane, because the agents use `LessEqual`: anything
                        // an agent writes must be closer than the clear value or it would be rejected.
                        // The ocean pass then overwrites this with its real hit distances, and the
                        // agents are rejected behind rock exactly as they are behind each other.
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // 1. Environment. One of the two backdrops; only the underwater one writes depth.
            match input.world {
                SimMode::Fish => self.ocean.draw(&mut pass, &self.scene, input.binding),
                SimMode::Birds => self.background.draw(&mut pass, &self.scene, input.binding),
            }
            stats.draw_calls += 1;

            // 2. Agents: one instanced draw for the whole swarm, depth tested against whatever the
            //    environment wrote.
            self.boids
                .draw(&mut pass, &self.scene, input.binding, input.num_agents);
            if input.num_agents > 0 {
                stats.draw_calls += 1;
                stats.instances += u64::from(input.num_agents);
                stats.triangles += u64::from(input.num_agents)
                    * u64::from(crate::boid_pass::MESH_TRIANGLES);
            }
        }

        // 3. Post: bloom pyramid, then the tone map into the caller's target.
        self.post.record(&mut encoder, target);

        ctx.queue.submit(Some(encoder.finish()));
        stats
    }
}

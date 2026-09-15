//! Frame orchestration: owns the render passes, the targets and the per-frame uniform upload.
//!
//! The renderer is deliberately the only place that knows the pass order. Passes do not call each
//! other; each is handed an open render pass and a scene binding. That keeps the frame graph readable
//! in one screen and makes adding a pass a matter of adding one call in [`Renderer::render`].

use boids_core::layout::SceneUniform;
use boids_gpu::context::GpuContext;

use crate::background::BackgroundPass;
use crate::boid_pass::BoidPass;
use crate::scene::{SceneBinding, SceneLayout};
use crate::targets::FrameTargets;

/// What the renderer needs to know about the simulation's buffer parity this frame.
#[derive(Debug, Clone, Copy)]
pub struct FrameInput {
    /// Which agent buffer holds the state to draw.
    pub binding: SceneBinding,
    /// Number of live agents.
    pub num_agents: u32,
}

/// Counters the app shows in the HUD. Cheap to produce and the fastest way to notice that a pass
/// started doing something it should not.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    /// Draw calls recorded this frame.
    pub draw_calls: u32,
    /// Instances submitted (one per agent).
    pub instances: u32,
    /// Triangles submitted.
    pub triangles: u64,
}

/// The whole presentation side of the frame.
#[derive(Debug)]
pub struct Renderer {
    targets: FrameTargets,
    scene: SceneLayout,
    background: BackgroundPass,
    boids: BoidPass,
    format: wgpu::TextureFormat,
    /// Whether the swapchain format is sRGB, i.e. whether the hardware applies the transfer function.
    is_srgb: bool,
    /// Size the last frame was rendered at, for resize detection without a resize event.
    last_size: (u32, u32),
}

impl Renderer {
    /// Builds every pipeline and allocates the depth target.
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
        let background = BackgroundPass::new(ctx, &scene, format);
        let boids_pass = BoidPass::new(ctx, &scene, format);
        Self {
            targets: FrameTargets::new(&ctx.device, width, height),
            scene,
            background,
            boids: boids_pass,
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
        self.last_size = self.targets.size;
    }

    /// Current render size in physical pixels.
    #[must_use]
    pub const fn size(&self) -> (u32, u32) {
        self.last_size
    }

    /// Draws one frame into the renderer's own depth attachment.
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
    /// tests draw into a colour texture and a depth texture they can read back, which is the only way
    /// to verify that geometry was rasterised rather than merely that the code ran. The same hook is
    /// what the HDR and bloom chain will use when the passes land, since those need the scene rendered
    /// into an `Rgba16Float` target instead of the swapchain.
    pub fn render_into(
        &mut self,
        ctx: &GpuContext,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        scene: &SceneUniform,
        input: FrameInput,
    ) -> FrameStats {
        self.scene.write(&ctx.queue, scene);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        let mut stats = FrameStats::default();
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame passes"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // The background pass covers every pixel, so loading the previous contents
                        // would be pure bandwidth. `Clear` also means the target is valid even if the
                        // background pass is later restricted to part of the viewport.
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        // Cleared to 1.0, the far plane, because the pass uses `LessEqual`: anything an
                        // agent writes must be closer than the clear value or it would be rejected.
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // 1. Backdrop: fills the frame and establishes the horizon. Writes no depth.
            self.background.draw(&mut pass, &self.scene, input.binding);
            stats.draw_calls += 1;

            // 2. Agents: one instanced draw for the whole swarm, depth tested against the backdrop's
            //    far plane (and, later today, against the raymarched reef).
            self.boids
                .draw(&mut pass, &self.scene, input.binding, input.num_agents);
            if input.num_agents > 0 {
                stats.draw_calls += 1;
                stats.instances += input.num_agents;
                stats.triangles += u64::from(input.num_agents)
                    * u64::from(crate::boid_pass::MESH_TRIANGLES);
            }
        }

        ctx.queue.submit(Some(encoder.finish()));
        stats
    }
}

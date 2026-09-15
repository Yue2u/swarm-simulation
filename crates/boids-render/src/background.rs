//! The background pass: a procedural sky or water backdrop for the whole swath.
//!
//! # Why a pass instead of a clear colour
//!
//! A flat clear colour is the single fastest way to make a 100k-agent simulation look like a
//! tech demo. A gradient costs one full-screen triangle and no texture memory, and it does three
//! useful jobs at once:
//!
//! * it establishes the horizon, which is what makes the swarm's scale readable,
//! * it carries the medium (water extinction with depth, aerial haze toward the horizon), which is
//!   the same Beer-Lambert falloff the agents and terrain are shaded with, so the whole frame agrees,
//! * it gives the sun a position that matches `SceneUniform::light_dir`, so the lighting on the agents
//!   and the glow in the sky come from the same source.
//!
//! The pass writes no depth: it is the backdrop, and leaving depth untouched means the agents' depth
//! test compares against the cleared far plane rather than against the sky.

use crate::scene::SceneLayout;

/// The background pipeline plus its bind group.
#[derive(Debug)]
pub struct BackgroundPass {
    pipeline: wgpu::RenderPipeline,
}

impl BackgroundPass {
    /// Builds the pipeline.
    ///
    /// # Panics
    /// Panics if the shader fails to compile. That is a programming error and belongs at startup.
    #[must_use]
    pub fn new(
        ctx: &boids_gpu::context::GpuContext,
        scene: &SceneLayout,
        format: wgpu::TextureFormat,
    ) -> Self {
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("background pipeline layout"),
                bind_group_layouts: &[Some(&scene.layout)],
                immediate_size: 0,
            });
        let module = ctx.shader_module("background", "render/background.wgsl");
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("background"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    // No vertex buffer at all: the triangle is generated from `vertex_index`.
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        // The tone curve lives in the fragment shader for now, so blending is off.
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // The backdrop triangle is wound clockwise when viewed with the Y axis pointing
                    // down, and a culling decision on a full-screen triangle is pure risk.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::targets::DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        Self { pipeline }
    }

    /// Records the pass into an already-open render pass.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, scene: &'a SceneLayout, binding: crate::scene::SceneBinding) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene.bind_group(binding), &[]);
        // One triangle covering the whole target: three vertices, no index buffer.
        pass.draw(0..3, 0..1);
    }
}

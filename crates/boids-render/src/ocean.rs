//! The underwater environment pass: a full-screen raymarch of the reef distance field.
//!
//! # Why a pass of its own
//!
//! The sky world's backdrop can be a full-screen gradient (`background.rs`), because nothing in it
//! needs to interact with the agents' depth. The reef cannot: the fish have to swim *behind* the
//! columns, which means the environment has to write real depth before the agents are drawn. So this
//! pass writes `frag_depth` from the raymarched hit distance and runs first, and the agent pipeline's
//! `LessEqual` test then does the rest.
//!
//! # Cost
//!
//! Per pixel: up to 80 field evaluations for the march, six more for the shading gradient, and twelve
//! shaft samples with four shadow taps each. That is the most expensive pass in the frame by a wide
//! margin, and it is why `docs/perf.md` tracks it separately. The levers, in order, are the shaft
//! sample count, half-resolution rendering of this pass alone, and the march step cap.
//!
//! The pass is created at startup and never rebuilt: switching worlds selects between this pipeline
//! and the sky backdrop's, which is the whole reason both exist in one renderer.

use crate::scene::SceneLayout;
use crate::targets::HDR_FORMAT;

/// The underwater environment pipeline plus its bind group layout.
#[derive(Debug)]
pub struct OceanPass {
    pipeline: wgpu::RenderPipeline,
}

impl OceanPass {
    /// Builds the pipeline.
    ///
    /// # Panics
    /// Panics if the shader fails to compile. That is a programming error and belongs at startup,
    /// with the WGSL error text, rather than a black frame.
    #[must_use]
    pub fn new(ctx: &boids_gpu::context::GpuContext, scene: &SceneLayout) -> Self {
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ocean pipeline layout"),
                bind_group_layouts: &[Some(&scene.layout)],
                immediate_size: 0,
            });
        let module = ctx.shader_module("ocean", "render/ocean.wgsl");
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ocean"),
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
                        // HDR: the surface and the caustics are far above 1.0 and the bloom pass
                        // needs that range to have anything to spread.
                        format: HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::targets::DEPTH_FORMAT,
                    // The one pass in the frame that writes depth without rasterising anything. The
                    // value comes from `@builtin(frag_depth)`, computed from the raymarched hit, so
                    // the agents can be occluded by rock.
                    depth_write_enabled: Some(true),
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
    pub fn draw<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a SceneLayout,
        binding: crate::scene::SceneBinding,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene.bind_group(binding), &[]);
        // One triangle covering the whole target: three vertices, no index buffer.
        pass.draw(0..3, 0..1);
    }
}

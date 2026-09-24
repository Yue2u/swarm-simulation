//! The agent pass: one instanced draw call for the entire swarm.
//!
//! # Vertex pulling
//!
//! There is no vertex buffer and no instance buffer. The mesh is generated from `vertex_index` and the
//! per-agent transform comes from the simulation's own storage buffer indexed by `instance_index`.
//! That has three consequences worth stating explicitly, because they are the reason for the design:
//!
//! * the CPU cost of drawing N agents is one `draw` call, independent of N,
//! * there is no upload step, so the drawn positions cannot lag the simulated ones by a frame,
//! * there is no second copy of the agent state to keep in sync, which removes an entire class of
//!   "the render and the physics disagree" bugs.
//!
//! # Mesh vertex count
//!
//! `MESH_VERTICES` must match the triangle budget in `shaders/render/boid.wgsl`. The shader is written
//! to be *total*: any vertex index beyond its triangle list produces a degenerate triangle rather than
//! reading out of bounds. So a mismatch between these two numbers is a silent visual change (a missing
//! wing) rather than a crash, and the count is asserted against the shader's own budget in the GPU test
//! suite by rendering the full range and checking that the last triangle is the one expected.

use crate::scene::{SceneBinding, SceneLayout};
use crate::targets::DEPTH_FORMAT;

/// Triangles in the procedural agent mesh: 18 body/tail triangles plus 4 wing triangles.
pub const MESH_TRIANGLES: u32 = 22;

/// Vertices drawn per agent. Must match `MESH_VERTICES` in `shaders/render/boid.wgsl`.
pub const MESH_VERTICES: u32 = MESH_TRIANGLES * 3;

/// The instanced agent pipeline.
#[derive(Debug)]
pub struct BoidPass {
    pipeline: wgpu::RenderPipeline,
}

impl BoidPass {
    /// Builds the pipeline.
    ///
    /// # Panics
    /// Panics if the shader fails to compile, which is a programming error and belongs at startup.
    #[must_use]
    pub fn new(ctx: &boids_gpu::context::GpuContext, scene: &SceneLayout) -> Self {
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("boid pipeline layout"),
                bind_group_layouts: &[Some(&scene.layout)],
                immediate_size: 0,
            });
        let module = ctx.shader_module("boid", "render/boid.wgsl");
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("boid"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        // The HDR scene target: a bioluminescent fish is an additive light above
                        // 1.0, and the bloom pass needs that value rather than a clipped white.
                        format: crate::targets::HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // No culling: the tail fin and the wings are single-sided by construction, and a
                    // distant agent whose fin happens to face away should still be drawn.
                    cull_mode: None,
                    front_face: wgpu::FrontFace::Ccw,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    // Against the environment's own distances now: the ocean pass writes the reef's
                    // real hit depth, so an agent behind a column is rejected by the same test that
                    // rejects it behind another agent. See `post.rs`'s sibling `ocean.rs` for why the
                    // environment writes depth by hand rather than rasterising geometry.
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        Self { pipeline }
    }

    /// Records the swarm draw.
    pub fn draw<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a SceneLayout,
        binding: SceneBinding,
        num_agents: u32,
    ) {
        self.draw_bound(pass, scene.bind_group(binding), num_agents);
    }

    /// Records the swarm draw over an explicit group 0.
    ///
    /// [`BoidPass::draw`] is the scene's path and binds the parity group the simulation just wrote.
    /// This one exists for the model viewer, which binds the same layout over a single synthetic
    /// agent: one agent, the same mesh, the same shader, so the viewer cannot show a different
    /// creature from the one in the scene.
    pub fn draw_bound<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        group: &'a wgpu::BindGroup,
        num_agents: u32,
    ) {
        if num_agents == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..MESH_VERTICES, 0..num_agents);
    }
}

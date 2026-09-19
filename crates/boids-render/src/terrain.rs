//! The sky world's ground: one vertex-pulled grid draw, depth tested and depth writing.
//!
//! # Where it sits in the frame
//!
//! After the sky backdrop (which writes no depth) and before the trees and the agents, so that a bird
//! flying behind a ridge is hidden by it and a tree is occluded by the ground it stands on. The
//! pipeline writes depth, which is what makes that ordering mean anything.
//!
//! # Cost
//!
//! `segments^2 * 2` triangles per frame, all generated from `vertex_index`, each vertex costing five
//! texel loads. At the app's world size that is 131k triangles and 2M texel loads: two orders of
//! magnitude below the agent draw, and the cheapest ground a 1.5 km world can be given. The levers, if
//! it ever matters, are the mesh's tessellation ratio and half-resolution shading, in that order.
//!
//! # Why there is no clipmap here
//!
//! A ring-based LOD spends its finest level on whatever is nearest the camera. In this project the
//! camera orbits at 0.8-1.15 times the world's diagonal in both worlds, so the ground's nearest point
//! is at 300-900 m and a clipmap's inner ring would cover a few pixels. What the mesh's resolution is
//! instead tied to is the *map's* texel size: see
//! [`boids_scene::terrain::mesh_segments`], and `docs/gpu-pipeline.md` for the numbers.

use crate::scene::SceneLayout;
use crate::targets::HDR_FORMAT;

/// Triangles per map texel column, i.e. the ratio the host sized the grid with. Used only for the
/// vertex-count arithmetic below and for the log line.
const MESH_TEXELS_PER_CELL: u32 = 4;

/// The ground pipeline.
#[derive(Debug)]
pub struct TerrainPass {
    pipeline: wgpu::RenderPipeline,
    /// Triangles in the grid, from the world's own terrain parameters.
    triangles: u32,
}

impl TerrainPass {
    /// Builds the pipeline for a world's terrain.
    ///
    /// # Panics
    /// Panics if the shader fails to compile. That is a programming error and belongs at startup,
    /// with the WGSL error text, rather than a hole in the world.
    #[must_use]
    pub fn new(ctx: &boids_gpu::context::GpuContext, scene: &SceneLayout, config: &boids_core::config::SimConfig) -> Self {
        let params = boids_scene::terrain_params(config);
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain pipeline layout"),
                bind_group_layouts: &[Some(&scene.layout)],
                immediate_size: 0,
            });
        let module = ctx.shader_module("terrain", "render/terrain.wgsl");
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("terrain"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    // No vertex buffer: the grid is a function of `vertex_index`, and the height is a
                    // texel load.
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // The grid is wound counter-clockwise seen from above, so the ground faces the
                    // sky and nothing has to be drawn twice.
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::targets::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let segments = params.segments.max(1);
        Self {
            pipeline,
            triangles: segments * segments * 2,
        }
    }

    /// Triangles the grid submits.
    #[must_use]
    pub const fn triangles(&self) -> u32 {
        self.triangles
    }

    /// Records the ground into an already-open render pass.
    pub fn draw<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a SceneLayout,
        binding: crate::scene::SceneBinding,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene.bind_group(binding), &[]);
        // Three vertices per triangle, no index buffer, no vertex buffer.
        pass.draw(0..self.triangles * 3, 0..1);
    }

    /// Ratio the mesh is sized with, for the pipeline docs and the log line.
    #[must_use]
    pub const fn texels_per_cell() -> u32 {
        MESH_TEXELS_PER_CELL
    }
}

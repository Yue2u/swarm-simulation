//! The sky world's trees: one indexed instanced draw over the scatter pass's instance buffer.
//!
//! # Why this pass has a vertex buffer and the agent pass does not
//!
//! An agent's mesh is animated per agent and generated from `vertex_index` because 100k instances
//! cannot afford to read a vertex buffer. A tree is static and is drawn a few thousand times, so the
//! mesh is built once on the host ([`boids_scene::mesh::tree_mesh`]), uploaded once, and read by the
//! GPU as an ordinary vertex buffer. The *instances* come from a storage buffer the scatter pass
//! filled, exactly as the agents do, so the per-frame CPU cost is still a single draw call.
//!
//! # Cost
//!
//! Up to `tree_capacity` instances of 136 triangles. At the app's world that is about 1,900 trees and
//! 260k triangles, which is more than the terrain and a fifth of the agent draw. The lever is the
//! scatter's coverage constant, not the mesh.

use boids_scene::mesh::{tree_mesh, TreeVertex};

use crate::scene::SceneLayout;
use crate::targets::HDR_FORMAT;

/// Vertex attributes of a tree vertex: position then normal, matching `TreeVertex`.
const TREE_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

/// The tree pipeline plus its mesh and instance count.
#[derive(Debug)]
pub struct TreePass {
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

impl TreePass {
    /// Builds the mesh and the pipeline.
    ///
    /// # Panics
    /// Panics if the shader fails to compile. That is a programming error and belongs at startup.
    #[must_use]
    pub fn new(ctx: &boids_gpu::context::GpuContext, scene: &SceneLayout) -> Self {
        let mesh = tree_mesh();
        let vertices = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tree vertices"),
            size: (mesh.vertices.len() * core::mem::size_of::<TreeVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&vertices, 0, bytemuck::cast_slice(&mesh.vertices));
        let indices = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tree indices"),
            size: (mesh.indices.len() * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&indices, 0, bytemuck::cast_slice(&mesh.indices));

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tree pipeline layout"),
                bind_group_layouts: &[Some(&scene.layout)],
                immediate_size: 0,
            });
        let module = ctx.shader_module("tree", "render/tree.wgsl");
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("tree"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: core::mem::size_of::<TreeVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &TREE_ATTRIBUTES,
                    })],
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

        #[allow(clippy::cast_possible_truncation)]
        let index_count = mesh.indices.len() as u32;
        log::debug!(
            "tree mesh: {} vertices, {} triangles",
            mesh.vertices.len(),
            mesh.triangle_count()
        );
        Self {
            pipeline,
            vertices,
            indices,
            index_count,
        }
    }

    /// Triangles one tree contributes.
    #[must_use]
    pub const fn triangles_per_tree(&self) -> u32 {
        self.index_count / 3
    }

    /// Records the trees into an already-open render pass.
    ///
    /// `count` comes from the scatter pass and is zero for a world with no vegetation, in which case
    /// nothing is recorded at all: an instanced draw with zero instances is legal, but skipping it is
    /// clearer in a capture and costs nothing.
    pub fn draw<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a SceneLayout,
        binding: crate::scene::SceneBinding,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene.bind_group(binding), &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..count);
    }
}

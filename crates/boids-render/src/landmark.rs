//! The castle: one indexed instanced draw over the host-placed landmark instance buffer.
//!
//! # Where it sits in the frame
//!
//! In both worlds, after the environment and before the agents. That order is what makes the depth
//! test mean something in each: underwater the ocean resolve has already written the seafloor's
//! distance, so the buried part of a sunken castle is hidden by the sand it stands in; in the sky
//! world the terrain and the trees have written theirs, so a castle behind a ridge is hidden by it.
//! Drawing it before the agents is what lets a bird disappear behind a tower.
//!
//! # Cost
//!
//! `index_count` triangles per landmark, one instance each: about 1,100 triangles per world per
//! frame, which is under a hundredth of the agent draw and about 1/250th of the forest. The mesh is
//! built once on the host and uploaded once, exactly like the tree mesh, and the instances are the
//! same storage-buffer trick, so the per-frame CPU cost is a single draw call.
//!
//! # Why the material is a vertex attribute
//!
//! The tree separates trunk from canopy with the sign of the normal's Y, which works because a tree
//! has exactly two materials and one of them faces up. A castle has five - stone, slate, rock, iron,
//! cloth - and they are not separable by orientation: a battlement cap and a courtyard floor both
//! face up and are both stone, while the keep's roof faces up and is not. So the mesh carries the
//! material per face and the shader looks it up in a palette.

use boids_scene::mesh::{castle_mesh, MeshVertex};

use crate::scene::SceneLayout;
use crate::targets::HDR_FORMAT;
use crate::tree::STATIC_MESH_ATTRIBUTES;

/// The castle pipeline plus its mesh.
#[derive(Debug)]
pub struct LandmarkPass {
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

impl LandmarkPass {
    /// Builds the mesh and the pipeline.
    ///
    /// # Panics
    /// Panics if the shader fails to compile. That is a programming error and belongs at startup.
    #[must_use]
    pub fn new(ctx: &boids_gpu::context::GpuContext, scene: &SceneLayout) -> Self {
        let mesh = castle_mesh();
        let vertices = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("landmark vertices"),
            size: (mesh.vertices.len() * core::mem::size_of::<MeshVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&vertices, 0, bytemuck::cast_slice(&mesh.vertices));
        let indices = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("landmark indices"),
            size: (mesh.indices.len() * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&indices, 0, bytemuck::cast_slice(&mesh.indices));

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("landmark pipeline layout"),
                bind_group_layouts: &[Some(&scene.layout)],
                immediate_size: 0,
            });
        let module = ctx.shader_module("landmark", "render/landmark.wgsl");
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("landmark"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: core::mem::size_of::<MeshVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &STATIC_MESH_ATTRIBUTES,
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
            "castle mesh: {} vertices, {} triangles",
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

    /// Triangles one landmark contributes.
    #[must_use]
    pub const fn triangles_per_landmark(&self) -> u32 {
        self.index_count / 3
    }

    /// Records the landmarks into an already-open render pass.
    ///
    /// `count` is the number of instances the renderer uploaded for the world being drawn, and is
    /// zero for a world that places none, in which case nothing is recorded at all.
    pub fn draw<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a SceneLayout,
        binding: crate::scene::SceneBinding,
        count: u32,
    ) {
        self.draw_bound(pass, scene.bind_group(binding), count);
    }

    /// Records the landmarks over an explicit group 0. See [`LandmarkPass::draw`]; the model viewer is
    /// the caller that needs it, binding the same layout over a single castle at the origin.
    pub fn draw_bound<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        group: &'a wgpu::BindGroup,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..count);
    }
}

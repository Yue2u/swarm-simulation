//! The HDR post-processing chain: bloom pyramid, tone map, grade.
//!
//! # Frame position
//!
//! Runs after every geometry pass, on the HDR intermediate, and writes the swapchain:
//!
//! ```text
//! bright   hdr            -> bloom[0]        threshold + box downsample
//! down     bloom[i-1]     -> bloom[i]        box downsample, i = 1..n-1
//! up       bloom[i]       -> bloom[i-1]      tent upsample, additive, i = n-1..1
//! composite hdr + bloom[0] -> swapchain      aberration, exposure, ACES, grade
//! ```
//!
//! The pyramid is built once, at startup, and only recreated on resize. This matters more than it
//! looks: the chain has one bind group per level per pass, and rebuilding them per frame would
//! allocate driver objects 60 times a second for a topology that never changes.
//!
//! # Why the up passes blend instead of writing
//!
//! An upsample that overwrites would replace the finer level's halo with the coarser one. Blending
//! additively makes the pyramid a *sum* of octaves: each level contributes glow at its own scale, so
//! a bright fish has a tight halo from level 0 and a wide one from level 2, which is what a real lens
//! does. The level-0 result is what the composite samples.
//!
//! # Format handling
//!
//! Two composite pipelines exist because the final transfer function belongs to whoever writes the
//! last pixel: an sRGB target is encoded by the hardware on write, so the shader must emit linear,
//! while a linear target needs the shader to encode. `SurfaceState::is_srgb` decides which one the
//! renderer builds, and getting it wrong is the classic "everything is washed out" bug.

use boids_core::layout::PostParams;
use boids_gpu::context::GpuContext;

use crate::targets::HDR_FORMAT;

/// Bloom pyramid depth. Three levels cover a halo from roughly the size of an agent to a quarter of
/// the frame; more levels cost memory for glow that reads as a haze rather than as a lens.
const LEVELS: usize = 3;

/// One level of the bloom pyramid: its texture, its view and its size.
#[derive(Debug)]
struct Level {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

/// The size-dependent half of the chain: the levels and every bind group that references them.
#[derive(Debug)]
struct Pyramid {
    levels: Vec<Level>,
    /// Bright pass: samples the HDR scene, writes level 0.
    bright_bg: wgpu::BindGroup,
    /// Downsample passes, indexed by destination level minus one.
    down_bgs: Vec<wgpu::BindGroup>,
    /// Upsample passes, indexed by source level minus one. Each adds into the level below.
    up_bgs: Vec<wgpu::BindGroup>,
    /// Composite: samples the HDR scene and bloom level 0.
    composite_bg: wgpu::BindGroup,
}

/// The post-processing pipeline set.
#[derive(Debug)]
pub struct PostChain {
    /// Bind group layout shared by all six passes.
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    bright: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    /// Everything that depends on the target size: the block below is rebuilt as one unit on resize,
    /// because a pyramid whose bind groups refer to textures from a previous size is not a state
    /// worth being able to represent.
    pyramid: Pyramid,
    /// Target format the composite writes, kept for the resize check.
    format: wgpu::TextureFormat,
}

impl PostChain {
    /// Builds the pipelines and the pyramid for a target size.
    ///
    /// `post` is the uniform buffer the renderer already owns: the same buffer is bound in the scene
    /// group (where the geometry passes read it) and here (where the post passes read it), so there is
    /// one upload per frame and one struct that can be stale, instead of two.
    #[must_use]
    pub fn new(
        ctx: &GpuContext,
        format: wgpu::TextureFormat,
        is_srgb: bool,
        post: &wgpu::Buffer,
        hdr: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("post layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                core::mem::size_of::<PostParams>() as u64,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    texture_entry(2),
                    texture_entry(3),
                ],
            });

        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post sampler"),
            // Linear: every post pass resamples at a different resolution than its source, and a
            // nearest-neighbour tap would make the bloom pyramid visible as blocks.
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            // Clamp rather than repeat: a wrap would fold the opposite edge of the frame into the
            // bloom of an agent at the frame edge.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("post pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });

        let bloom_module = ctx.shader_module("bloom", "render/bloom.wgsl");
        let composite_module = ctx.shader_module("composite", "render/composite.wgsl");

        let bloom_pipeline = |label: &str, entry: &str, blend: Option<wgpu::BlendState>| {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &bloom_module,
                        entry_point: Some("vs_fullscreen"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &bloom_module,
                        entry_point: Some(entry),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: HDR_FORMAT,
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: no_cull_triangle(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
        };

        // The upsample adds into the level below, so its pipeline blends. Both factors are `One`, so
        // the alpha channel of the source is irrelevant: this is a sum of light, not a composite.
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };

        let bright = bloom_pipeline("bloom bright", "fs_bright", None);
        let down = bloom_pipeline("bloom down", "fs_down", None);
        let up = bloom_pipeline("bloom up", "fs_up", Some(additive));

        // The composite shader has one entry point per output encoding; the target format decides
        // which is correct, and `is_srgb` is the only thing that can tell us.
        let composite_entry = if is_srgb {
            "fs_composite"
        } else {
            "fs_composite_encoded"
        };
        let composite = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("composite"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &composite_module,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &composite_module,
                    entry_point: Some(composite_entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: no_cull_triangle(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let pyramid = build_pyramid(ctx, &layout, &sampler, post, hdr, width, height);
        Self {
            layout,
            sampler,
            bright,
            down,
            up,
            composite,
            pyramid,
            format,
        }
    }

    /// Recreates the pyramid when the target size changed.
    ///
    /// `hdr` is the scene view the bright pass and the composite read; it changes identity on resize,
    /// which is why the bind groups are rebuilt here rather than only at startup.
    pub fn resize(
        &mut self,
        ctx: &GpuContext,
        post: &wgpu::Buffer,
        hdr: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        let want = level_sizes(width, height);
        let unchanged = self.pyramid.levels.len() == want.len()
            && self
                .pyramid
                .levels
                .iter()
                .zip(&want)
                .all(|(level, size)| level.size == *size);
        if unchanged {
            return;
        }
        self.pyramid = build_pyramid(ctx, &self.layout, &self.sampler, post, hdr, width, height);
    }

    /// Records the whole chain: bloom into the pyramid, then the composite into `target`.
    ///
    /// `encoder` must not have any pass open, and `target` must have the size the pyramid was built
    /// for. Everything here is a separate render pass, which is what makes the ordering safe: `wgpu`
    /// inserts the memory barrier between passes, and every pass reads a texture the previous one
    /// finished writing.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        // Bright pass into level 0. Cleared rather than loaded: level 0 is fully rewritten and its
        // previous contents are last frame's glow.
        {
            let mut pass = post_pass(
                encoder,
                "bloom bright",
                &self.pyramid.levels[0].view,
                clear(),
            );
            fullscreen(&mut pass, &self.bright, &self.pyramid.bright_bg);
        }

        for i in 1..self.pyramid.levels.len() {
            let mut pass = post_pass(
                encoder,
                "bloom down",
                &self.pyramid.levels[i].view,
                clear(),
            );
            fullscreen(&mut pass, &self.down, &self.pyramid.down_bgs[i - 1]);
        }

        // Upsample from the coarsest level back down, adding into each level as it goes. Recorded in
        // reverse, and the destination is a *different* texture from the source at every step.
        for i in (1..self.pyramid.levels.len()).rev() {
            let mut pass = post_pass(
                encoder,
                "bloom up",
                &self.pyramid.levels[i - 1].view,
                wgpu::LoadOp::Load,
            );
            fullscreen(&mut pass, &self.up, &self.pyramid.up_bgs[i - 1]);
        }

        {
            let mut pass = post_pass(encoder, "composite", target, clear());
            fullscreen(&mut pass, &self.composite, &self.pyramid.composite_bg);
        }
    }

    /// Size of the bloom pyramid level the composite samples, for tests.
    #[must_use]
    pub fn bloom_size(&self) -> (u32, u32) {
        self.pyramid.levels[0].size
    }

    /// Target format the composite writes.
    #[must_use]
    pub const fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Keeps the level textures alive, which matters because the bind groups reference their views.
    pub fn level_textures(&self) -> impl Iterator<Item = &wgpu::Texture> {
        self.pyramid.levels.iter().map(|level| &level.texture)
    }
}

/// Sizes of the bloom pyramid levels for a target size: half, quarter, eighth.
///
/// Levels stop before they would collapse to a single texel: a 1x1 level would blur nothing and its
/// upsample would still cost a full-screen pass.
fn level_sizes(width: u32, height: u32) -> Vec<(u32, u32)> {
    let mut sizes = Vec::with_capacity(LEVELS);
    let (mut w, mut h) = (width.max(1), height.max(1));
    for _ in 0..LEVELS {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        if w <= 2 || h <= 2 {
            break;
        }
        sizes.push((w, h));
    }
    // Always at least one level: an empty pyramid is a state every recording site would have to
    // check for, and the bright pass has to write somewhere.
    if sizes.is_empty() {
        sizes.push(((width / 2).max(1), (height / 2).max(1)));
    }
    sizes
}

/// A bloom clear: the pyramid levels hold light, and every level is fully rewritten before it is
/// read, so the cleared value itself is never visible. Black is the honest "no glow yet".
fn clear() -> wgpu::LoadOp<wgpu::Color> {
    wgpu::LoadOp::Clear(wgpu::Color::BLACK)
}

/// Begins one post pass writing `view`, with the given load operation.
fn post_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    label: &'static str,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn no_cull_triangle() -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        // The full-screen triangle is wound either way depending on the backend's Y convention, and
        // a culling decision on it is pure risk with no benefit.
        cull_mode: None,
        ..Default::default()
    }
}

/// Builds every size-dependent object in the chain, in one place, so resize cannot leave half the
/// chain pointing at textures from the previous size.
fn build_pyramid(
    ctx: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    post: &wgpu::Buffer,
    hdr: &wgpu::TextureView,
    width: u32,
    height: u32,
) -> Pyramid {
    let levels: Vec<Level> = level_sizes(width, height)
        .into_iter()
        .map(|(w, h)| {
            let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("bloom level"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            Level {
                texture,
                view,
                size: (w, h),
            }
        })
        .collect();

    let bg = |src: &wgpu::TextureView, second: &wgpu::TextureView| {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: post.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(second),
                },
            ],
        })
    };

    Pyramid {
        // The second texture slot is inert for these passes, but it must still be bound, and binding
        // the pass's *destination* would be a validation error: a texture may not be a colour target
        // and a resource in the same pass. So the inert slot repeats the source.
        bright_bg: bg(hdr, hdr),
        down_bgs: (1..levels.len())
            .map(|i| bg(&levels[i - 1].view, &levels[i - 1].view))
            .collect(),
        // Upsample level `i` reads level `i`; the second slot is the same view.
        up_bgs: (1..levels.len())
            .map(|i| bg(&levels[i].view, &levels[i].view))
            .collect(),
        composite_bg: bg(hdr, &levels[0].view),
        levels,
    }
}

/// One full-screen triangle with the pass's pipeline and bind group.
fn fullscreen(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_halves_and_never_collapses() {
        assert_eq!(level_sizes(2560, 1440), vec![(1280, 720), (640, 360), (320, 180)]);
        // A tiny target keeps at least one level rather than an empty chain.
        assert_eq!(level_sizes(4, 4), vec![(2, 2)]);
        assert_eq!(level_sizes(1, 1), vec![(1, 1)]);
        // And no level is degenerate, which would make the tent kernel sample its own texel nine
        // times and produce a black pyramid level.
        for size in [(1920, 1080), (400, 300), (17, 33)] {
            for (w, h) in level_sizes(size.0, size.1) {
                assert!(w >= 1 && h >= 1);
            }
        }
    }
}

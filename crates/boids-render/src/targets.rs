//! Depth target management.
//!
//! Only the depth buffer lives here for now. When the HDR intermediate and the bloom chain land they
//! join it, because they share the same property that makes them worth isolating: their size is a
//! function of the window and they must be recreated together on resize, exactly once, rather than by
//! each pass that happens to notice the size changed.

/// Depth format. `Depth32Float` rather than `Depth24PlusStencil8`: nothing here uses stencil, and the
/// extra depth precision matters once the underwater raymarch writes real distances into this buffer
/// and the agents are depth-tested against terrain at a range of hundreds of metres.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The frame's depth attachment.
#[derive(Debug)]
pub struct FrameTargets {
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// Current size in physical pixels.
    pub size: (u32, u32),
}

impl FrameTargets {
    /// Allocates the depth target at a given size.
    #[must_use]
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            depth,
            depth_view,
            size: (width, height),
        }
    }

    /// Recreates the target if the size changed. Returns whether anything was reallocated, so the
    /// caller can log resize churn instead of guessing whether a resize happened.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        let (width, height) = (width.max(1), height.max(1));
        if self.size == (width, height) {
            return false;
        }
        *self = Self::new(device, width, height);
        true
    }

    /// View to bind as the depth attachment.
    #[must_use]
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// The depth texture, exposed for passes that need its size.
    #[must_use]
    pub fn depth_texture(&self) -> &wgpu::Texture {
        &self.depth
    }
}

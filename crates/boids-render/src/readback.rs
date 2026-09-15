//! Texture readback for screenshots and tests.
//!
//! Lives in the render crate because both the screenshot path and the off-screen render tests need it,
//! and because the one thing that is easy to get wrong here is specific to render targets: a
//! `copy_texture_to_buffer` row pitch must be a multiple of 256 unless the copy is trivially whole, so
//! a 400-pixel-wide RGBA target needs padding that the caller never sees. Reading a target with the
//! wrong pitch produces a sheared image, which is easy to produce and annoying to diagnose.

/// Copies an RGBA8 texture into a buffer and returns its rows, tightly packed.
///
/// # Errors
/// Returns a message if the copy or the mapping fails.
pub fn read_texture_rgba(
    ctx: &boids_gpu::context::GpuContext,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    read_texture(
        ctx,
        texture,
        width,
        height,
        4,
        wgpu::TextureAspect::All,
    )
}

/// Copies the depth aspect of a depth texture into a buffer as `f32` samples.
///
/// # Errors
/// Returns a message if the backend cannot copy depth textures, or if the copy or mapping fails. The
/// capability check is explicit rather than left to a validation error: on a backend without
/// `DEPTH_TEXTURE_AND_BUFFER_COPIES` the error would otherwise arrive asynchronously through the
/// uncaptured-error handler, far from the call that caused it.
pub fn read_texture_depth_f32(
    ctx: &boids_gpu::context::GpuContext,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, String> {
    if !supports_depth_copy(ctx) {
        return Err(format!(
            "backend {:?} does not support depth texture-to-buffer copies \
             (DEPTH_TEXTURE_AND_BUFFER_COPIES)",
            ctx.info.backend
        ));
    }
    let bytes = read_texture(ctx, texture, width, height, 4, wgpu::TextureAspect::DepthOnly)?;
    Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

/// Whether this device can copy a depth texture into a buffer.
#[must_use]
pub fn supports_depth_copy(ctx: &boids_gpu::context::GpuContext) -> bool {
    ctx.adapter
        .get_downlevel_capabilities()
        .flags
        .contains(wgpu::DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES)
}

/// The shared implementation: copy, map, and strip the row padding.
fn read_texture(
    ctx: &boids_gpu::context::GpuContext,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    bytes_per_texel: u32,
    aspect: wgpu::TextureAspect,
) -> Result<Vec<u8>, String> {
    let row_bytes = width * bytes_per_texel;
    let padded_row = row_bytes.next_multiple_of(256);
    let size = u64::from(padded_row) * u64::from(height);

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    ctx.wait_idle();

    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(format!("mapping the readback buffer failed: {e}")),
        Err(e) => return Err(format!("readback channel closed before mapping: {e}")),
    }

    let view = slice
        .get_mapped_range()
        .map_err(|e| format!("getting the mapped range failed: {e}"))?;
    let mut out = Vec::with_capacity((row_bytes * height) as usize);
    for row in 0..height as usize {
        let start = row * padded_row as usize;
        out.extend_from_slice(&view[start..start + row_bytes as usize]);
    }
    drop(view);
    staging.unmap();
    Ok(out)
}

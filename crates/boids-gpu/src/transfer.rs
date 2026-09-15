//! Buffer upload and readback helpers.
//!
//! Readback is only used by tests and by the deterministic screenshot mode: the frame loop never
//! synchronises with the GPU. Keeping it in one small module makes the synchronisation cost obvious
//! and keeps it out of the hot path.

use boids_core::layout::Boid;

/// Uploads a boid slice into a buffer.
///
/// # Panics
/// Panics if the slice does not fit in the buffer. A too-small agent buffer is a configuration bug
/// that must be caught at startup, not silently truncated into a half-simulated swarm.
pub fn upload_boids(queue: &wgpu::Queue, buffer: &wgpu::Buffer, boids: &[Boid]) {
    let bytes = bytemuck::cast_slice(boids);
    assert!(
        bytes.len() as u64 <= buffer.size(),
        "uploading {} bytes into a {} byte buffer",
        bytes.len(),
        buffer.size()
    );
    queue.write_buffer(buffer, 0, bytes);
}

/// Uploads a `Pod` uniform value.
pub fn upload_uniform<T: bytemuck::Pod>(queue: &wgpu::Queue, buffer: &wgpu::Buffer, value: &T) {
    queue.write_buffer(buffer, 0, bytemuck::bytes_of(value));
}

/// Reads a buffer back into a typed vector, blocking until the GPU has finished.
///
/// # Panics
/// Panics if the buffer size is not a multiple of the element size, which would mean the caller
/// asked for the wrong type.
pub fn read_buffer<T: bytemuck::Pod>(
    ctx: &crate::context::GpuContext,
    buffer: &wgpu::Buffer,
) -> Vec<T> {
    assert_eq!(
        buffer.size() % core::mem::size_of::<T>() as u64,
        0,
        "buffer size {} is not a multiple of element size {}",
        buffer.size(),
        core::mem::size_of::<T>()
    );
    let bytes = read_raw(ctx, buffer);
    bytemuck::cast_slice::<u8, T>(&bytes).to_vec()
}

/// Reads a buffer's raw bytes back, blocking until the GPU has finished.
///
/// The staging buffer is created and destroyed per call: this is a test/diagnostic path, and reusing
/// staging buffers would only add lifetime management with no benefit.
pub fn read_raw(ctx: &crate::context::GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let size = buffer.size();
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback encoder"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        // The receiver may be gone if the caller panicked; that is fine, the send just fails.
        let _ = tx.send(result);
    });
    ctx.wait_idle();

    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("mapping readback buffer failed: {e}"),
        Err(e) => panic!("readback channel closed before mapping completed: {e}"),
    }

    let view = match slice.get_mapped_range() {
        Ok(view) => view,
        Err(e) => panic!("mapping readback buffer failed: {e}"),
    };
    let data = view.to_vec();
    drop(view);
    staging.unmap();
    data
}

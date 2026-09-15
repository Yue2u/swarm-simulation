//! Per-pass GPU timing with timestamp queries.
//!
//! `wgpu` offers no CPU-side view of where a frame's GPU time goes, so the only way to answer "is
//! the sort or the neighbour search the bottleneck" is to ask the GPU. This module wraps that: a
//! query set sized for one frame's worth of passes, a resolve step at the end of the frame, and a
//! readback that is deliberately synchronous and deliberately rare.
//!
//! # Why the readback blocks
//!
//! Mapping a buffer can only complete once the GPU has finished the work that wrote it, so reading
//! the timings of frame N stalls the CPU until frame N is done. That is exactly the definition of a
//! profiler that costs you the thing it measures, which is why the profiler is opt-in, why it
//! reports once every `interval` frames rather than every frame, and why the numbers below are read
//! as "the frame was this long on the GPU" rather than "the frame rate is this".
//!
//! The timings themselves are unaffected: they are written by the GPU as it executes the passes.
//! What the stall distorts is throughput, not the breakdown.

use crate::context::GpuContext;

/// How many passes one frame can be timed for.
///
/// The sort is one pass per stage and there are `m * (m + 1) / 2` stages for `2^m` keys: 78 at 4096
/// keys, 153 at 131,072 (the default 100k-agent world). Four more passes - `clear_cells`, `hash`,
/// `build_ranges`, `integrate` - put the ceiling at 157 for the target agent count, and 256 covers
/// every swarm up to 2^21 keys. A larger one needs a larger profiler, and says so rather than
/// silently reporting a partial frame.
pub const MAX_TIMED_PASSES: usize = 256;

/// Accumulated GPU time for one pass name within one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassTotal {
    /// The pass's label, as passed to [`GpuProfiler::pass_writes`].
    pub label: &'static str,
    /// Total time in milliseconds across every pass with this label.
    pub millis: f32,
    /// How many passes carried this label.
    pub passes: u32,
}

/// One frame's pass timings, aggregated by label.
#[derive(Debug, Clone, Default)]
pub struct FrameTimings {
    /// One entry per distinct label, in the order the labels first appear in the frame.
    pub totals: Vec<PassTotal>,
    /// Sum of every timed pass in the frame.
    pub total_millis: f32,
}

impl FrameTimings {
    /// The table this exists for: one line per pass, plus the frame's total.
    ///
    /// The total is not the frame time - it excludes everything not run through the profiler, and
    /// excludes the gaps between passes, which on a memory-latency-bound sort are not nothing. It is
    /// the simulation's GPU cost, which is the number a change to the sort or the grid moves.
    #[must_use]
    pub fn format_table(&self, title: &str) -> String {
        let mut out = format!("{title}\n");
        for total in &self.totals {
            let calls = if total.passes == 1 {
                "1 pass".to_string()
            } else {
                format!("{:>3} passes", total.passes)
            };
            out.push_str(&format!(
                "    {:<14} {calls}   {:>7.3} ms\n",
                total.label, total.millis
            ));
        }
        out.push_str(&format!(
            "    {:<14}            {:>7.3} ms\n",
            "total", self.total_millis
        ));
        out
    }
}

/// A timestamp profiler for the simulation's compute passes.
///
/// Created once and reused every frame. When the adapter has no timestamp support this type is never
/// constructed: [`GpuProfiler::new`] returns `None` and the frame records as it would without one.
#[derive(Debug)]
pub struct GpuProfiler {
    query_set: wgpu::QuerySet,
    /// Query results, resolved to real memory by the GPU.
    resolve: wgpu::Buffer,
    /// The copy that gets mapped, so the resolve buffer can be written to again next frame.
    readback: wgpu::Buffer,
    /// Labels of the passes recorded this frame, in slot order.
    labels: Vec<&'static str>,
    /// Passes one frame can be timed for.
    capacity: usize,
    /// Frames recorded since creation.
    frames: u64,
    /// Report every this many frames.
    interval: u64,
    /// Nanoseconds per timestamp tick, from the queue.
    period_ns: f32,
    /// Set once if a frame overflowed the query set, so the warning is printed once.
    overflowed: bool,
}

impl GpuProfiler {
    /// Creates a profiler, or returns `None` when the device cannot write timestamps.
    ///
    /// `interval` is the number of frames between readbacks; 1 means every frame, which stalls the
    /// CPU every frame and is only useful for a single-frame measurement.
    #[must_use]
    pub fn new(ctx: &GpuContext, capacity: usize, interval: u64) -> Option<Self> {
        if !ctx.timestamps_enabled {
            log::warn!("no GPU profiler: the device was created without TIMESTAMP_QUERY");
            return None;
        }
        let capacity = capacity.clamp(1, MAX_TIMED_PASSES);
        let count = (capacity * 2) as u32;
        let bytes = u64::from(count) * 8;
        let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("pass timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let period_ns = ctx.queue.get_timestamp_period();
        log::info!(
            "GPU profiler: {capacity} passes per frame, a report every {interval} frame(s), \
             {period_ns:.3} ns per tick"
        );
        Some(Self {
            query_set,
            resolve,
            readback,
            labels: Vec::with_capacity(capacity),
            capacity,
            frames: 0,
            interval: interval.max(1),
            period_ns,
            overflowed: false,
        })
    }

    /// Starts a timed pass: records `label` and returns the timestamp writes for it.
    ///
    /// The returned value borrows the profiler, which is what stops a caller from starting the next
    /// pass before this one has ended. Returns `None` once the frame's slots are used up, and the
    /// pass records untimed: a partial profile beats a corrupted query set.
    pub fn pass_writes(
        &mut self,
        label: &'static str,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let slot = self.labels.len();
        if slot >= self.capacity {
            if !self.overflowed {
                self.overflowed = true;
                log::warn!(
                    "GPU profiler is full at {} passes: the rest of the frame is not timed. Raise \
                     MAX_TIMED_PASSES to cover this agent count.",
                    self.capacity
                );
            }
            return None;
        }
        self.labels.push(label);
        #[allow(clippy::cast_possible_truncation)]
        let index = (slot * 2) as u32;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: Some(index + 1),
        })
    }

    /// How many passes have been recorded this frame. For tests and for the profiling table's title.
    #[must_use]
    pub fn recorded_passes(&self) -> usize {
        self.labels.len()
    }

    /// Resolves the frame's timestamps into real memory. Records into `encoder`, so it must be called
    /// between the last timed pass and the submit.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let count = (self.labels.len() * 2) as u32;
        if count == 0 {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..count, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, u64::from(count) * 8);
    }

    /// Reads the timings of the frame just submitted, if this frame is a reporting frame.
    ///
    /// Must be called after the submit that includes [`Self::resolve`], and it blocks until the GPU
    /// has finished that submission.
    pub fn read(&mut self, ctx: &GpuContext) -> Option<FrameTimings> {
        let labels = core::mem::take(&mut self.labels);
        self.frames += 1;
        if labels.is_empty() {
            return None;
        }
        if !self.frames.is_multiple_of(self.interval) {
            return None;
        }

        let bytes = (labels.len() * 16) as u64;
        let slice = self.readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // The receiver is only gone if the caller panicked; the send failing is not an error.
            let _ = tx.send(result);
        });
        ctx.wait_idle();
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                log::error!("mapping the timestamp buffer failed: {e}");
                return None;
            }
            Err(e) => {
                log::error!("timestamp mapping channel closed: {e}");
                return None;
            }
        }

        let view = match slice.get_mapped_range() {
            Ok(view) => view,
            Err(e) => {
                log::error!("reading the timestamp buffer failed: {e}");
                return None;
            }
        };
        // Decoded by hand rather than with `bytemuck::cast_slice`: the mapped bytes are not
        // guaranteed to be 8-byte aligned, and the timestamps are native-endian, which is what the
        // GPU wrote.
        let ticks: Vec<u64> = view
            .chunks_exact(8)
            .map(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap_or([0; 8])))
            .collect();
        drop(view);
        self.readback.unmap();

        let mut totals: Vec<PassTotal> = Vec::new();
        let mut total_millis = 0.0f32;
        for (i, label) in labels.iter().enumerate() {
            let (Some(begin), Some(end)) = (ticks.get(i * 2), ticks.get(i * 2 + 1)) else {
                break;
            };
            let ticks = end.saturating_sub(*begin);
            #[allow(clippy::cast_precision_loss)]
            let millis = ticks as f32 * self.period_ns / 1.0e6;
            total_millis += millis;
            match totals.iter_mut().find(|t| t.label == *label) {
                // Aggregated by name because the sort is 153 passes of the same pipeline: a table
                // with 153 "sort" rows is a wall of numbers nobody reads.
                Some(existing) => {
                    existing.millis += millis;
                    existing.passes += 1;
                }
                None => totals.push(PassTotal {
                    label,
                    millis,
                    passes: 1,
                }),
            }
        }

        Some(FrameTimings {
            totals,
            total_millis,
        })
    }
}

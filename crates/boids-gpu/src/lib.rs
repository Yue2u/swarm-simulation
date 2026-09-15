//! GPU device setup, buffer management and compute pipelines for the simulation.
//!
//! This crate owns everything that talks to `wgpu` about *simulation state*: the ping-pong agent
//! buffers, the spatial grid, the sort, and the compute pipelines. It knows nothing about biomes,
//! rendering or input; those live in `boids-scene`, `boids-render` and `boids-app`.
//!
//! # The one rule
//!
//! Nothing here may touch the CPU per agent. Every buffer is filled once at startup and then only
//! ever written by compute shaders. The only per-frame CPU work is `write_buffer` on a handful of
//! uniform structs, which is why the simulation cost is independent of the agent count in the app's
//! frame budget.
//!
//! # Frame structure
//!
//! ```text
//! P0 clear_cells    cell_start[i] = EMPTY_CELL
//! P1 hash           keys[i] = {cell_index(pos_i), i}
//! P2 sort           bitonic sort of keys by cell index
//! P3 build_ranges   cell_start/cell_end from the sorted keys
//! P4 integrate      neighbour search over 27 cells -> forces -> integration
//! ```
//!
//! `record_frame` records P0..P4 and returns the index of the agent buffer that now holds the
//! updated state. The renderer draws from that buffer while the next frame writes into the other
//! one, so a frame in flight never reads a buffer that is being written.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod context;
pub mod mesh_profile;
pub mod sim;
pub mod transfer;

pub use context::{GpuContext, GpuContextDescriptor, SurfaceState};
pub use mesh_profile::{mesh_profile, scene_uniform};
pub use sim::{dispatch_size, SimPipelines, SimResources, Strategy};
pub use transfer::{read_buffer, read_raw, upload_boids, upload_uniform};

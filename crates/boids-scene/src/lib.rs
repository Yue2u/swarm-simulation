//! Procedural environment and art content for both worlds.
//!
//! This crate owns *what the two worlds look like*: the underwater medium, the palettes, the biome
//! parameters and, as they land, the heightfield and the meshes the render passes draw. It owns no
//! pipelines and no device: every module here produces POD uniforms and pure functions, which is what
//! keeps the look tunable and testable without a GPU.
//!
//! The split with its neighbours is deliberate:
//!
//! * `boids-core` defines the *layout* of every struct that crosses to the GPU, including these,
//!   because two independent mirrors of one WGSL struct is one mirror too many,
//! * `boids-gpu` runs the simulation and knows nothing about water or palettes,
//! * `boids-render` owns the passes, and asks this crate for the numbers they need.
//!
//! Today that is [`water`] (the underwater medium and the post-processing grade), [`sky`] (the
//! atmosphere) and [`terrain`] (the sky world's ground, its baked heightfield and the meshes that go
//! on it).

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod mesh;
pub mod sky;
pub mod terrain;
pub mod water;

pub use mesh::{tree_mesh, TreeMesh, TreeVertex};
pub use sky::sky_params;
pub use terrain::{map_resolution, mesh_segments, terrain_params, TerrainGpu};
pub use water::{post_params, water_params};

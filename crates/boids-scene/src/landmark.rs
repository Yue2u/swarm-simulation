//! The castle: where it stands in each world, and the instance buffer the landmark pass reads.
//!
//! # Why the placement is on the host
//!
//! The tree scatter runs on the GPU because there are thousands of trees and their positions are a
//! pure function of the heightfield. A landmark is the opposite case: there are one or two, and their
//! placement is a *search* - the flattest ground in the sky world, the clearest water on the seafloor -
//! which is a few thousand field evaluations at startup. Doing it on the host also means the CPU twins
//! of the two fields (`terrain::height_at`, `sdf::reef_field`) are what place it, so the drawn castle
//! cannot end up standing inside the ground it is drawn on or inside a reef column drawn through it.
//!
//! # Why the buffer holds one world at a time
//!
//! One renderer draws both worlds, and there is one instance buffer in one bind group. The renderer
//! uploads a world's instances when the world it draws changes, which is a 32-byte `write_buffer` on a
//! key press and no allocation, so the TAB switch keeps its "nothing is reallocated" property.

use boids_core::config::SimConfig;
use boids_core::layout::{SimMode, StaticInstance};
use glam::{Vec2, Vec3};

use crate::mesh::CASTLE_HALF_EXTENT;

/// Instance slots in the buffer.
///
/// Sized for a couple of landmarks per world so that placing a second one is a data change rather
/// than a reallocation. The upload clamps to it, and a unit test asserts the placement fits.
pub const LANDMARK_CAPACITY: u32 = 4;

/// Height of a castle as a fraction of the world's half-height.
///
/// A fraction of the world rather than a fixed number of metres: the same expression has to be a
/// landmark in the app's 1.2 km world and in the 250 m world the render suite builds, where a fixed
/// 75 m castle would not fit in the frame at all. `bounds_half.y` is the axis that scales with how
/// much air or water the world has, which is what the castle stands in.
///
/// `1.02` is three times the 0.34 the castle first shipped at: it is meant to be the landmark the eye
/// lands on, not a prop beside the flock, so it stands most of the world's half-height tall.
const CASTLE_HEIGHT_FRACTION: f32 = 1.02;

/// Where a castle would like to stand, as a fraction of the world's half-extents in `xz`.
///
/// The world centre, so the castle is the thing the frame is built around. The candidate search still
/// nudges it off a reef column or a ridge, but only within [`SITE_SPREAD`] of here.
const CASTLE_SITE: [f32; 2] = [0.0, 0.0];

/// Candidate sites tested around [`CASTLE_SITE`], per axis. An odd count puts one candidate exactly
/// on the preferred site.
const SITE_CANDIDATES: i32 = 7;

/// Radius of the candidate grid, as a fraction of the world's half-extent in `x`.
///
/// Small enough that the castle stays where the framing expects it, large enough to step off a reef
/// column (the widest is about 6.5 m) or a ridge.
const SITE_SPREAD: f32 = 0.15;

/// Samples across a candidate's footprint when scoring the ground under it, per axis.
const FOOTPRINT_SAMPLES: i32 = 5;

/// Samples up the castle's height when scoring its clearance from the reef, including both ends.
const CLEARANCE_SAMPLES_Y: i32 = 5;

/// Yaw of every castle, radians.
///
/// A fixed yaw rather than a hash: a landmark is placed once and should look deliberate. This one
/// turns the gate about 25 degrees off the default orbit's view axis, so the opening frame shows two
/// faces of the castle instead of one flat wall.
const CASTLE_YAW: f32 = -0.2;

/// Height of a world's castle, metres.
#[must_use]
pub fn castle_height(config: &SimConfig) -> f32 {
    CASTLE_HEIGHT_FRACTION * config.bounds_half.y
}

/// Half the width of the ground a castle covers, metres.
#[must_use]
pub fn castle_half_extent(config: &SimConfig) -> f32 {
    castle_height(config) * CASTLE_HALF_EXTENT
}

/// The castle instance for a world: placed on the terrain in the sky world, on the seafloor
/// underwater.
///
/// `pos.y` is the *ground line* the mesh is built around, so the curtain wall meets the ground where
/// the instance is placed and the plinth extends below it.
#[must_use]
pub fn castle_instance(config: &SimConfig) -> StaticInstance {
    let (site, ground_y) = match config.mode {
        SimMode::Birds => sky_site(config),
        SimMode::Fish => sea_site(config),
    };
    StaticInstance {
        pos: [site.x, ground_y, site.y],
        scale: castle_height(config),
        yaw: CASTLE_YAW,
        kind: 0.0,
        mask: 0.0,
        pad1: 0.0,
    }
}

/// The sky world's site: the flattest ground near the preferred offset, at the highest point of its
/// own footprint.
///
/// The ground line is the *highest* ground under the footprint rather than the height at the centre.
/// On a slope the two differ, and placing at the centre would leave the uphill half of the plinth
/// buried and the courtyard floor below the terrain on that side, where the ground would show through
/// the floor. Placing at the highest point means the plinth's skirt has to cover only the drop.
fn sky_site(config: &SimConfig) -> (Vec2, f32) {
    let half_extent = castle_half_extent(config);
    let mut best_site = Vec2::ZERO;
    let mut best_slope = f32::INFINITY;
    for site in candidate_sites(config) {
        let slope = footprint_max_slope(site, half_extent, config.env_scale, config.env_freq);
        if slope < best_slope {
            best_slope = slope;
            best_site = site;
        }
    }
    let ground = footprint_max_height(best_site, half_extent, config.env_scale, config.env_freq);
    (best_site, ground)
}

/// The underwater site: the clearest water near the preferred offset, standing on the seafloor.
///
/// The score is the *smallest* distance from the castle's volume to the reef, so a site with a column
/// touching one corner of the castle loses to a site with no column anywhere near it. The SDF is the
/// same field the ocean pass raymarches and the simulation's fish steer around, so "clear" here means
/// clear in both of those.
fn sea_site(config: &SimConfig) -> (Vec2, f32) {
    let height = castle_height(config);
    let half_extent = castle_half_extent(config);
    let mut best_site = Vec2::ZERO;
    let mut best_clearance = f32::NEG_INFINITY;
    for site in candidate_sites(config) {
        let clearance = footprint_min_clearance(
            site,
            half_extent,
            height,
            config.env_scale,
            config.env_floor_y,
        );
        if clearance > best_clearance {
            best_clearance = clearance;
            best_site = site;
        }
    }
    (best_site, config.env_floor_y)
}

/// The candidate sites, in world `xz`.
fn candidate_sites(config: &SimConfig) -> Vec<Vec2> {
    let centre = Vec2::new(
        CASTLE_SITE[0] * config.bounds_half.x,
        CASTLE_SITE[1] * config.bounds_half.z,
    );
    let spread = SITE_SPREAD * config.bounds_half.x;
    let steps = SITE_CANDIDATES.max(2);
    let mut out = Vec::with_capacity((steps * steps) as usize);
    for i in 0..steps {
        for j in 0..steps {
            #[allow(clippy::cast_precision_loss)]
            let fx = 2.0 * (i as f32) / ((steps - 1) as f32) - 1.0;
            #[allow(clippy::cast_precision_loss)]
            let fz = 2.0 * (j as f32) / ((steps - 1) as f32) - 1.0;
            out.push(centre + Vec2::new(fx, fz) * spread);
        }
    }
    out
}

/// The `FOOTPRINT_SAMPLES^2` offsets across a footprint, as fractions of its half-extent in `[-1, 1]`.
fn footprint_grid() -> Vec<Vec2> {
    let steps = FOOTPRINT_SAMPLES.max(2);
    let mut out = Vec::with_capacity((steps * steps) as usize);
    for i in 0..steps {
        for j in 0..steps {
            #[allow(clippy::cast_precision_loss)]
            let fx = 2.0 * (i as f32) / ((steps - 1) as f32) - 1.0;
            #[allow(clippy::cast_precision_loss)]
            let fz = 2.0 * (j as f32) / ((steps - 1) as f32) - 1.0;
            out.push(Vec2::new(fx, fz));
        }
    }
    out
}

/// The steepest ground inside a castle's footprint centred on `site`, as a gradient magnitude.
///
/// The steepest point rather than the mean: it is one corner of the plinth that floats if the ground
/// falls away there, so the worst sample is what decides whether the site is flat enough.
fn footprint_max_slope(site: Vec2, half_extent: f32, amplitude: f32, frequency: f32) -> f32 {
    let mut worst = 0.0f32;
    // The baseline the slope is measured over is a fraction of the footprint, not the whole of it:
    // the castle's own width is the scale at which a step under it matters.
    let eps = (half_extent * 0.25).max(0.5);
    for offset in footprint_grid() {
        let p = site + offset * half_extent;
        worst = worst.max(boids_core::terrain::slope_at(p, amplitude, frequency, eps).length());
    }
    worst
}

/// The highest ground inside a castle's footprint centred on `site`.
fn footprint_max_height(site: Vec2, half_extent: f32, amplitude: f32, frequency: f32) -> f32 {
    let mut highest = f32::NEG_INFINITY;
    for offset in footprint_grid() {
        let p = site + offset * half_extent;
        highest = highest.max(boids_core::terrain::height_at(p, amplitude, frequency));
    }
    highest
}

/// How far the nearest reef rock comes to the castle's volume, in metres. Positive means the whole
/// castle stands clear of the reef.
fn footprint_min_clearance(
    site: Vec2,
    half_extent: f32,
    height: f32,
    period: f32,
    floor_y: f32,
) -> f32 {
    let steps = CLEARANCE_SAMPLES_Y.max(2);
    let mut closest = f32::INFINITY;
    for offset in footprint_grid() {
        let xz = site + offset * half_extent;
        for k in 0..steps {
            // From just above the seafloor, not from it: the plinth is *meant* to be buried, and a
            // sample sitting on the floor plane would read the floor's own distance, which is zero,
            // as a collision.
            #[allow(clippy::cast_precision_loss)]
            let y = floor_y + height * (0.08 + 0.92 * (k as f32) / ((steps - 1) as f32));
            closest = closest.min(boids_core::sdf::reef_field(
                Vec3::new(xz.x, y, xz.y),
                period,
                floor_y,
            ));
        }
    }
    closest
}

/// The landmark instance buffer, and the per-world instance lists behind it.
#[derive(Debug)]
pub struct LandmarkGpu {
    buffer: wgpu::Buffer,
    /// Instances per world, indexed by `SimMode as usize`. Both worlds are placed at startup, so a
    /// world switch is a write rather than a site search.
    per_world: [Vec<StaticInstance>; 2],
    /// Which world's instances the buffer holds, so a frame drawing the same world as the last one
    /// skips the write.
    uploaded: Option<SimMode>,
}

impl LandmarkGpu {
    /// Places the landmarks for both worlds and allocates the buffer.
    ///
    /// Both worlds are placed here rather than lazily because placing is a search, and a search on the
    /// frame the world changes is a hitch in the middle of the transition. It costs a few thousand
    /// field evaluations once, at startup.
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        active: &SimConfig,
        other: &SimConfig,
    ) -> Self {
        let mut per_world: [Vec<StaticInstance>; 2] = [Vec::new(), Vec::new()];
        per_world[active.mode as usize].push(castle_instance(active));
        per_world[other.mode as usize].push(castle_instance(other));

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("landmark instances"),
            size: u64::from(LANDMARK_CAPACITY) * core::mem::size_of::<StaticInstance>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut landmarks = Self {
            buffer,
            per_world,
            uploaded: None,
        };
        landmarks.upload(queue, active.mode);
        log::info!(
            "landmarks: {} underwater, {} in the sky",
            landmarks.count(SimMode::Fish),
            landmarks.count(SimMode::Birds),
        );
        landmarks
    }

    /// Uploads a world's instances unless they are already in the buffer, and returns how many there
    /// are.
    pub fn upload(&mut self, queue: &wgpu::Queue, mode: SimMode) -> u32 {
        let index = mode as usize;
        let count = u32::try_from(self.per_world[index].len())
            .unwrap_or(LANDMARK_CAPACITY)
            .min(LANDMARK_CAPACITY);
        if self.uploaded != Some(mode) {
            let bytes = bytemuck::cast_slice(&self.per_world[index][..count as usize]);
            queue.write_buffer(&self.buffer, 0, bytes);
            self.uploaded = Some(mode);
            log::debug!("landmarks: uploaded {count} for {mode:?}");
        }
        count
    }

    /// The instance buffer the scene bind group reads.
    #[must_use]
    pub const fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Landmarks placed in a world.
    #[must_use]
    pub fn count(&self, mode: SimMode) -> u32 {
        u32::try_from(self.per_world[mode as usize].len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::castle_mesh;
    use boids_core::layout::SimMode;

    /// `SimMode as usize` is the index into `per_world`, so the discriminants are load-bearing.
    #[test]
    fn the_mode_discriminants_are_the_array_indices() {
        assert_eq!(SimMode::Fish as usize, 0);
        assert_eq!(SimMode::Birds as usize, 1);
    }

    /// Every world has to fit the buffer, or the upload silently drops a landmark.
    #[test]
    fn each_world_fits_the_buffer() {
        for mode in [SimMode::Fish, SimMode::Birds] {
            let config = SimConfig::for_mode(mode, 100_000);
            let mut per_world: [Vec<StaticInstance>; 2] = [Vec::new(), Vec::new()];
            per_world[mode as usize].push(castle_instance(&config));
            assert!(
                per_world[mode as usize].len() as u32 <= LANDMARK_CAPACITY,
                "{mode:?} places more landmarks than the buffer holds"
            );
        }
    }

    /// The castle has to be inside the box the swarm is steered in, or it is a decoration in the
    /// distance rather than something the flock flies around.
    #[test]
    fn the_castle_stands_inside_the_world() {
        for mode in [SimMode::Fish, SimMode::Birds] {
            let config = SimConfig::for_mode(mode, 100_000);
            let instance = castle_instance(&config);
            let half = castle_half_extent(&config);
            assert!(
                instance.pos[0].abs() + half <= config.bounds_half.x,
                "{mode:?}: the castle at x = {} leaves the world",
                instance.pos[0]
            );
            assert!(
                instance.pos[2].abs() + half <= config.bounds_half.z,
                "{mode:?}: the castle at z = {} leaves the world",
                instance.pos[2]
            );
            assert!(instance.scale > 0.0);
        }
    }

    /// The drawn castle stands on the drawn ground: the placement samples the same field the
    /// heightfield is baked from, with the same amplitude and frequency.
    #[test]
    fn the_sky_castle_stands_on_the_terrain() {
        let config = SimConfig::for_mode(SimMode::Birds, 100_000);
        let instance = castle_instance(&config);
        let at_site = boids_core::terrain::height_at(
            Vec2::new(instance.pos[0], instance.pos[2]),
            config.env_scale,
            config.env_freq,
        );
        // The ground line is the highest point of the footprint, so it is at or above the centre's
        // height, and never more than the footprint's relief above it.
        let relief = castle_half_extent(&config) * 1.5;
        assert!(
            instance.pos[1] >= at_site - 1e-3 && instance.pos[1] <= at_site + relief,
            "ground line {} against a centre height of {at_site}",
            instance.pos[1]
        );
    }

    /// A sunken castle on the seafloor, not floating above it or buried to the battlements.
    #[test]
    fn the_sea_castle_stands_on_the_seafloor() {
        let config = SimConfig::for_mode(SimMode::Fish, 100_000);
        let instance = castle_instance(&config);
        assert!((instance.pos[1] - config.env_floor_y).abs() < 1e-3);
        assert!(instance.pos[1] < 0.0, "the seafloor is below the origin");
    }

    /// No reef column may pass through the castle.
    ///
    /// The ocean pass raymarches `reef_field` and knows nothing about the castle, so a column drawn
    /// through a wall would interpenetrate it: the site search is the only thing preventing that, and
    /// this is the check that it works. The metric is the placement's own, so the test cannot drift
    /// from what the search optimises.
    #[test]
    fn the_sea_castle_clears_the_reef() {
        let config = SimConfig::for_mode(SimMode::Fish, 100_000);
        let instance = castle_instance(&config);
        let site = Vec2::new(instance.pos[0], instance.pos[2]);
        let half = castle_half_extent(&config);
        let height = castle_height(&config);
        let chosen =
            footprint_min_clearance(site, half, height, config.env_scale, config.env_floor_y);
        assert!(
            chosen > 0.0,
            "the chosen site has {chosen:.2} m of clearance; a reef column is drawn through the castle"
        );

        // And the search has to be worth running: the preferred site is not allowed to be better.
        let preferred = Vec2::new(
            CASTLE_SITE[0] * config.bounds_half.x,
            CASTLE_SITE[1] * config.bounds_half.z,
        );
        let at_preferred = footprint_min_clearance(
            preferred,
            half,
            height,
            config.env_scale,
            config.env_floor_y,
        );
        assert!(
            chosen >= at_preferred,
            "the search picked {chosen:.2} m over {at_preferred:.2} m at the preferred site"
        );
    }

    /// The placement has to hold in the shrunken worlds the screenshots and the render suite build,
    /// not only in the app's world: a castle that leaves the world sideways is a wall of stone, and
    /// one taller than the whole box is a column.
    #[test]
    fn the_placement_scales_to_a_small_world() {
        for mode in [SimMode::Fish, SimMode::Birds] {
            let mut config = SimConfig::for_mode(mode, 3_000);
            config.resize_world(Vec3::splat(config.r_percept * 9.0));
            let instance = castle_instance(&config);
            let half = castle_half_extent(&config);
            assert!(instance.pos[0].abs() + half <= config.bounds_half.x);
            assert!(instance.pos[2].abs() + half <= config.bounds_half.z);
            // It stands most of the half-height tall on purpose (see `CASTLE_HEIGHT_FRACTION`), so the
            // bound is the whole box rather than half of it: taller than that and the castle is a
            // column, not a castle.
            assert!(
                instance.scale < 2.0 * config.bounds_half.y,
                "{mode:?}: a {} m castle in a {} m tall world",
                instance.scale,
                2.0 * config.bounds_half.y
            );
        }
    }

    /// The mesh the placement measures is the mesh the pass draws.
    #[test]
    fn the_footprint_matches_the_mesh() {
        let mesh = castle_mesh();
        let half_extent = mesh
            .vertices
            .iter()
            .fold(0.0f32, |a, v| a.max(v.pos[0].abs().max(v.pos[2].abs())));
        assert!((half_extent - CASTLE_HALF_EXTENT).abs() < 1e-4);
    }
}

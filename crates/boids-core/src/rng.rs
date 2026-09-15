//! Deterministic PCG32 random number generator.
//!
//! Used for agent spawn, tree scatter, and the `--deterministic` screenshot mode. It is a plain
//! integer LCG/permutation pair so that the same seed produces the same world on every machine,
//! which is what makes visual regressions comparable between runs.

use glam::Vec3;

/// PCG-XSH-RR 32-bit generator (O'Neill, 2014).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Creates a generator from a seed and a stream selector.
    #[must_use]
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (stream << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Advances the generator and returns the next 32-bit value.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        #[allow(clippy::cast_possible_truncation)]
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // 24 mantissa bits keeps the value strictly below 1.0.
        #[allow(clippy::cast_precision_loss)]
        let v = (self.next_u32() >> 8) as f32;
        v / (1u32 << 24) as f32
    }

    /// Uniform `f32` in `[min, max)`.
    pub fn range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    /// Uniform point in an axis-aligned box centred on the origin.
    pub fn in_box(&mut self, half: Vec3) -> Vec3 {
        Vec3::new(
            self.range(-half.x, half.x),
            self.range(-half.y, half.y),
            self.range(-half.z, half.z),
        )
    }

    /// Uniform point on the unit sphere (rejection sampling, no trig).
    pub fn unit_vector(&mut self) -> Vec3 {
        loop {
            let v = Vec3::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            let len_sq = v.length_squared();
            if (1e-6..=1.0).contains(&len_sq) {
                return v / len_sq.sqrt();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = Pcg32::new(42, 1);
        let mut b = Pcg32::new(42, 1);
        for _ in 0..64 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
        let mut c = Pcg32::new(43, 1);
        assert_ne!(Pcg32::new(42, 1).next_u32(), c.next_u32());
    }

    #[test]
    fn floats_stay_in_range_and_are_not_degenerate() {
        let mut rng = Pcg32::new(7, 99);
        let mut sum = 0.0f64;
        let n = 10_000;
        for _ in 0..n {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "value {v} out of range");
            sum += f64::from(v);
        }
        let mean = sum / f64::from(n);
        assert!((mean - 0.5).abs() < 0.02, "mean {mean} is biased");
    }

    #[test]
    fn unit_vectors_are_unit_length() {
        let mut rng = Pcg32::new(1, 2);
        for _ in 0..1000 {
            let v = rng.unit_vector();
            assert!((v.length() - 1.0).abs() < 1e-5);
        }
    }
}

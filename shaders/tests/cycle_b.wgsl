// === file: tests/cycle_b.wgsl ================================================================
// Fixture: the other half of a mutually recursive include pair.

//#include "tests/cycle_a.wgsl"

fn cycle_b() -> f32 {
    return 2.0;
}

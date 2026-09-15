// === file: tests/cycle_a.wgsl ================================================================
// Fixture: mutually recursive include. The preprocessor must report a cycle, not hang.

//#include "tests/cycle_b.wgsl"

fn cycle_a() -> f32 {
    return 1.0;
}

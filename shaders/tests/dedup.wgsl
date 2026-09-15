// === file: tests/dedup.wgsl ==================================================================
// Fixture: includes the same header twice. Used only by the preprocessor unit test.

//#include "tests/dedup_shared.wgsl"
//#include "tests/dedup_shared.wgsl"

fn dedup_entry(x: f32) -> f32 {
    return dedup_probe(x);
}

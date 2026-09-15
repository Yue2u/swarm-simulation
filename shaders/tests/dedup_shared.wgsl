// === file: tests/dedup_shared.wgsl ===========================================================
// Fixture for the preprocessor's include-deduplication test. Included twice on purpose by
// `tests/dedup.wgsl`; if the loader inlined it twice, the GPU would reject the compilation unit for
// a duplicate `dedup_probe` declaration. Kept deliberately trivial: it is never executed.

fn dedup_probe(x: f32) -> f32 {
    return x;
}

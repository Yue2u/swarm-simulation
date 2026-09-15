// === file: render/composite.wgsl ============================================================
// Pass: the final tone map and grade. Reads the HDR scene and the bloom pyramid, writes the
// swapchain.
//
// bindings: @group(0) 0:PostParams(uniform) 1:sampler 2:src=HDR scene 3:bloom=bloom level 0
// draw:     draw(0..3, 0..1)
// depth:    none
//
// ORDER OF OPERATIONS, and why each one is where it is:
//
//   1. chromatic aberration - a *sampling* effect, so it has to happen before anything that depends
//      on the sampled value. Applied radially: the offset grows with the square of the distance from
//      the centre, which is what a real lens does and what keeps the middle of the frame sharp.
//   2. exposure and bloom addition - both are in linear light, which is the only space where adding
//      light is meaningful.
//   3. ACES tone map - compresses the now-unbounded HDR range into [0, 1]. Nothing after this point
//      is linear light any more, so nothing that means "light" may be added after it.
//   4. vignette, grade, grain - display-referred terms. A vignette applied before the tone curve
//      would be lifted back up by it; grain applied before it would be smoothed away in the
//      highlights, which is precisely where film grain is most visible.
//
// WHY ACES AND NOT `x / (1 + x)`
//   A Reinhard curve desaturates highlights towards white, so a bloomed bioluminescent fish turns
//   into a grey blob. The ACES fit (Narkowicz's) preserves hue as it compresses, which is what lets a
//   bright cyan light stay cyan, and it has a shoulder steep enough that the reef's caustics can be
//   several times above 1.0 without clipping into flat white.

//#include "render/post_bindings.wgsl"

/// White point of the tone curve. ACES maps this luminance to 1.0 and everything above it into the
/// shoulder; the host writes it per world so that the underwater scene, which is lit through 30
/// metres of water, sits at the same apparent brightness as the sky.
const ACES_A: f32 = 2.51;
const ACES_B: f32 = 0.03;
const ACES_C: f32 = 2.43;
const ACES_D: f32 = 0.59;
const ACES_E: f32 = 0.14;

/// Narkowicz's ACES filmic approximation.
fn aces_film(x: vec3<f32>) -> vec3<f32> {
    let num = x * (ACES_A * x + vec3<f32>(ACES_B));
    let den = x * (ACES_C * x + vec3<f32>(ACES_D)) + vec3<f32>(ACES_E);
    return clamp(num / den, vec3<f32>(0.0), vec3<f32>(1.0));
}

/// Rec. 709 luminance, matching the bloom shader's threshold weights.
const LUMA_DISPLAY: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

/// Per-pixel noise in [0, 1), from the pixel coordinate and the frame time.
///
/// Integer hashing of the pixel index rather than a texture: the grain has to be re-randomised every
/// frame or it stops reading as grain and starts reading as a dirty lens, and a 2D hash costs nothing
/// next to the bandwidth of a noise texture. The Y coordinate is multiplied by a large odd constant so
/// that adjacent rows land far apart in the hash; without it, hashing `x ^ y` produces visible
/// diagonal structure.
fn grain_noise(frag: vec2<f32>) -> f32 {
    let x = u32(frag.x);
    let y = u32(frag.y);
    let frame = u32(post.time * 60.0);
    // Unsigned multiplication wraps in WGSL, which is what makes this hash well distributed and
    // identical on every driver.
    return hash_unit(x * 0x9e3779b9u ^ y * 0x85ebca6bu ^ frame);
}

/// The whole post chain, returning *linear* light. Both entry points below are this function plus the
/// output encoding the target format requires.
///
/// `frag` is the fragment coordinate: `VsOut::clip` carries the `position` builtin, which in a fragment
/// stage *is* the pixel centre, so the grain gets a stable per-pixel seed without a second varying.
fn compose(uv: vec2<f32>, frag: vec2<f32>) -> vec3<f32> {
    let centre = vec2<f32>(0.5);
    let d = uv - centre;

    // 1. Chromatic aberration: the channels are sampled with a radial offset, red out and blue in.
    let offset = d * (post.aberration * dot(d, d) * 4.0);
    var color = vec3<f32>(
        textureSample(post_src, post_sampler, uv + offset).r,
        textureSample(post_src, post_sampler, uv).g,
        textureSample(post_src, post_sampler, uv - offset).b,
    );

    // 2. Linear light: exposure and the white point, then the bloom pyramid's sum.
    //
    // `exposure / tonemap_white` puts a scene luminance of `tonemap_white` at the curve's own white
    // point, so the two knobs mean different things: exposure decides how bright the frame is, and
    // the white point decides how much of the HDR range is spent below white. The underwater world
    // needs a much larger white point than the sky, because its caustics are tens of times brighter
    // than its mid-tones and clipping them would erase the pattern that makes it read as water.
    color = color * (post.exposure / max(post.tonemap_white, 1e-3));
    color = color + textureSample(post_bloom, post_sampler, uv).rgb * post.bloom_strength;

    // 3. Tone map. `max` guards the fit against a negative value from the aberration offset reaching
    //    past the edge of the source.
    color = aces_film(max(color, vec3<f32>(0.0)));

    // 4. Display-referred grade.
    let vig = clamp(1.0 - post.vignette * dot(d, d) * 2.2, 0.0, 1.0);
    color = color * vig;
    let luma = dot(color, LUMA_DISPLAY);
    color = mix(vec3<f32>(luma), color, post.saturation);
    color = (color - vec3<f32>(0.5)) * post.contrast + vec3<f32>(0.5) + vec3<f32>(post.lift);
    color = color + vec3<f32>((grain_noise(frag) - 0.5) * post.grain);
    return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Linear output: correct when the target format is an sRGB one, because the hardware encodes on write.
@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(compose(in.uv, in.clip.xy), 1.0);
}

// Encoded output: used when the target format is *not* sRGB, so nothing downstream will encode. The
// host picks between this and `fs_composite` from the surface's own format, which is the only place
// that knows.
@fragment
fn fs_composite_encoded(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(srgb_encode(compose(in.uv, in.clip.xy)), 1.0);
}

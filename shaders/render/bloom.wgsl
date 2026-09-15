// === file: render/bloom.wgsl ===============================================================
// Passes: the bloom pyramid. Bright-pass, downsample, and additive upsample.
//
// bindings: @group(0) 0:PostParams(uniform) 1:sampler 2:src texture_2d<f32> 3:bloom texture_2d<f32>
// draw:     draw(0..3, 0..1)   one full-screen triangle per pass
// depth:    none. The post passes own no depth attachment; they run after the scene is complete.
//
// WHY A PYRAMID
//   A single blurred bright-pass gives a tight halo. A pyramid gives the wide, soft falloff that
//   reads as a lens: bright small features bleed into their surroundings across tens of pixels, and
//   the cost is bounded because each level blurs at a lower resolution. Three levels at half, quarter
//   and eighth resolution cover the range from a bioluminescent fish's outline to the glow that fills
//   a rock arch, for the price of one full-resolution pass and a handful of small ones.
//
// WHY THE UPSAMPLE ADDS RATHER THAN REPLACES
//   The upsample pass is recorded with additive blending into the level below. That is what makes the
//   chain a sum of octaves instead of a sequence of overwrites: each level contributes its own scale
//   of glow to the final image, which is what the composite samples from level 0.

//#include "render/post_bindings.wgsl"

// Luminance weights, Rec. 709. The threshold is applied to luminance rather than to the maximum
// channel so that a saturated blue highlights less than a white one of the same value, which is what
// the eye does and what keeps a school of cyan fish from blooming as if it were a light source.
const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Soft-knee threshold: everything above `bloom_threshold` contributes, with a quadratic ramp over the
// knee so the transition is continuous. A hard cut makes the bloom's edge pop as an agent crosses the
// threshold, which is visible as flicker on a moving swarm.
fn bright_pass(c: vec3<f32>) -> vec3<f32> {
    let luma = dot(c, LUMA);
    let knee = max(post.bloom_knee, 1e-4);
    let curve = clamp(luma - post.bloom_threshold + knee, 0.0, 2.0 * knee);
    let soft = curve * curve / (4.0 * knee);
    let hard = max(luma - post.bloom_threshold, 0.0);
    let weight = max(soft, hard) / max(luma, 1e-5);
    return c * weight;
}

// Four-tap box filter at the centre of the destination texel.
//
// Box rather than a wider tent: this is the *downsample* step, and the input is already band-limited
// by the level above it (or, for the bright pass, by the scene's own antialiasing). A wider kernel
// here would spend samples removing detail that the next level's upsample is going to blur anyway.
fn downsample(src: vec2<f32>) -> vec3<f32> {
    let e = src_texel() * 0.5;
    return 0.25 * (
        textureSample(post_src, post_sampler, src + vec2<f32>(-e.x, -e.y)).rgb
            + textureSample(post_src, post_sampler, src + vec2<f32>(e.x, -e.y)).rgb
            + textureSample(post_src, post_sampler, src + vec2<f32>(-e.x, e.y)).rgb
            + textureSample(post_src, post_sampler, src + vec2<f32>(e.x, e.y)).rgb
    );
}

// Nine-tap tent filter for the upsample. Wider than the downsample on purpose: the glow has to grow
// as it travels up the pyramid, otherwise every level would contribute a halo of the same size and
// the sum would look like a single blur with four times the cost.
fn upsample(src: vec2<f32>) -> vec3<f32> {
    let e = src_texel();
    var sum = textureSample(post_src, post_sampler, src).rgb * 4.0;
    sum = sum + textureSample(post_src, post_sampler, src + vec2<f32>(-e.x, 0.0)).rgb * 2.0;
    sum = sum + textureSample(post_src, post_sampler, src + vec2<f32>(e.x, 0.0)).rgb * 2.0;
    sum = sum + textureSample(post_src, post_sampler, src + vec2<f32>(0.0, -e.y)).rgb * 2.0;
    sum = sum + textureSample(post_src, post_sampler, src + vec2<f32>(0.0, e.y)).rgb * 2.0;
    sum = sum + textureSample(post_src, post_sampler, src + vec2<f32>(-e.x, -e.y)).rgb;
    sum = sum + textureSample(post_src, post_sampler, src + vec2<f32>(e.x, -e.y)).rgb;
    sum = sum + textureSample(post_src, post_sampler, src + vec2<f32>(-e.x, e.y)).rgb;
    sum = sum + textureSample(post_src, post_sampler, src + vec2<f32>(e.x, e.y)).rgb;
    return sum * (1.0 / 16.0);
}

// Bright pass: the scene, thresholded, at the first bloom level's resolution.
@fragment
fn fs_bright(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(bright_pass(downsample(in.uv)), 1.0);
}

// Downsample one level.
@fragment
fn fs_down(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(downsample(in.uv), 1.0);
}

// Upsample one level, added into the level below by the pass's blend state.
@fragment
fn fs_up(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(upsample(in.uv), 1.0);
}

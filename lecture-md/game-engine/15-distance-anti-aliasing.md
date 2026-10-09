# Chapter 15: Distance anti-aliasing

By the end of this chapter, detail that shrinks below a pixel stops misbehaving. A pixel shows the light reflected over its whole footprint on a surface, but the engine shades it at one point, and once the footprint covers more detail than that point can stand for, the point picks whichever piece it lands on. From far away, three things go wrong:
1. **Highlights on curved surfaces sparkle.** A sphere 40 pixels wide turns its normal by a few degrees from one pixel to the next. A sharp highlight, smaller than a pixel, lands on a pixel's shading point in one frame and between them in the next, and crawls over the surface as the camera moves.
2. **Normal-mapped surfaces sparkle too.** A brick wall's normal map has bumps much smaller than a pixel from across a courtyard. The mip chain averages them into a flatter normal, so the wall reflects like a smooth one, with highlights it shouldn't have.
3. **Foliage thins out and disappears.** A masked material keeps a texel only where its alpha reaches the cutoff. Each mip level averages alpha, blurring a leaf's edges into partly covered texels, and fewer of them reach the cutoff: far away, a plant is bare stalks.

The usual cure for the sparkle is temporal anti-aliasing, which averages many frames' worth of shading points. That trades the sparkle for blur and ghosting. The thinning it can't cure at all: every frame samples the same thinned mip, and the coverage is already gone. Instead, this chapter removes the detail the pixel can't show before shading, and keeps what the mips lose:
1. **Specular anti-aliasing for curvature** (Tokuyoshi and Kaplanyan, "Stable Geometric Specular Antialiasing with Projected-Space NDF Filtering", JCGT 2021). How fast the surface's normal changes between neighbouring pixels says how much it spreads within one. That spread is roughness, so it's added to the material's.
2. **Specular anti-aliasing for normal maps** (Toksvig, "Mipmapping Normal Maps", 2005). The average of normals that point in different directions is shorter than 1. Its length says how much they spread. The mips keep that spread, and it's added to the roughness too. The 2021 paper can't see it: it measures how the normal changes between pixels, and by the time a far wall is shaded, the mips have already averaged its bumps into a nearly flat normal.
3. **Mip levels that keep their coverage** (Castaño, "Computing Alpha Mipmaps", 2010). Each level's alpha is scaled so that the same share of texels reaches the cutoff as at full size.

The second and third need the mip chain to be built with them in mind, which the blits Chapter 5 made can't do. The textures' mips are now built by compute shaders, in one buffer, and copied into the images.

Together they cost nothing measurable per frame: the lighting pass took 1.351 ms with them and 1.345 ms without, at 1080p on an RTX 5070 Laptop.

What this chapter doesn't do is smooth the edges of geometry, the staircase along a roof line. That's a separate problem, for a later chapter.

This chapter builds on [Chapter 14](14-sky-and-air.md).

## 15.1 Mip chains in compute: `mips.slang`

### Why
A blit averages a 2 × 2 block of texels, with no idea what they hold. Averaging colors, normals and alpha each needs its own care, and the coverage needs to know the cutoff.

### How
- **The data:** every level of every texture lives in one buffer, as 8-bit RGBA texels packed into a `uint`, level after level. Each step reaches it through addresses in its push data; no descriptors.
- **`downsampleMain`: a box filter.** Each texel of the smaller level averages the texels its area covers. With an even size, that's a 2 × 2 block. With an odd one, the footprint is between 2 and 3 texels wide, and the texels it cuts count by the share it covers: a 5-texel row halves to 2 texels, each covering 2.5. Float rounding can put the last footprint's end a hair past the level's edge, so the loop stops at the edge.
- **What's averaged, by kind:**
  - **Colors (sRGB textures)** are averaged as linear light, as the sampler would read them, and encoded back to sRGB.
  - **See-through colors, weighted by alpha:** base colors whose every use takes alpha as coverage or opacity, for masked and blended materials. A texel that's fully transparent has no color to contribute. Leaf textures are often black where they're cut out, and a plain average darkens the leaves' edges at every level. Weighting by alpha leaves the transparent texels' color out. An opaque material shows every texel, whatever its alpha, so its colors are averaged plainly.
  - **Data** (metallic-roughness, occlusion) is averaged as it is.
  - **Normals** are averaged as vectors, and their spread goes into alpha: the measure 15.4 reads. The direction is renormalized, so its 8 bits aren't spent on a shorter vector.
- **The spread (Toksvig 2005):** normals whose angle from their average has a standard deviation `s` average to a vector about `1 / (1 + s²)` long, so the length gives the spread back: `s² = (1 − length) / length`.
  - **Stored as `1 − s`.** Storing the length itself would be too coarse: its first 8-bit step below 1, 254/255, already means roughness about 0.3. Storing `1 − s`, each step near no spread is 1/255 of a radian. And 1, which an 8-bit normal map's alpha usually is, means no spread.
  - **Each level** reads its source texels as vectors, each its direction times the length its spread gives, averages them, and stores the average's direction and spread.
  - **Rounding isn't spread.** Stored in 8 bits, a unit normal is only roughly unit length: each component rounds to within 1/255 of its value, so the length is off by about 0.2–0.3% typically, and up to 0.6%. Read as spread, that alone would be roughness about 0.3 on a mirror-smooth normal-mapped surface. So the direction is always renormalized as it's read, and the length comes only from the spread.
- **`prepareNormalsMain`: the top level of a normal map** has no spread of its own, so its alpha is set to 1. glTF ignores a normal map's alpha, so it's free to hold this.
- **Coverage (Castaño 2010):** for a masked material's base color:
  - **`histogramMain`** counts each level's 8-bit alpha values into 256 bins, with atomic adds.
  - **`coverageMain`,** one thread per level below the top, finds the 8-bit alpha `a` such that the share of the level's texels at or above `a` is closest to the top level's share at or above the cutoff. That's Castaño's idea: a new alpha reference for the level, then a scale that takes it to the cutoff. He finds the reference by bisection; with 8-bit alpha, the histogram gives the best of all 255 candidates directly.
  - **The scale,** `cutoff / ((a − 0.5) / 255)`, takes `a` to a little above the cutoff and `a − 1` to a little below: half a scaled step each way. When the coverage shrank down the chain, the usual case, the scale is above 1, that's more than half an 8-bit step, and rounding the scaled alpha to 8 bits keeps each on its side. When it grew, the scale is below 1, several alphas round to one 8-bit value, and the split is only close.
  - **`scaleAlphaMain`** multiplies the level's alpha by its scale.
  - **The order:** every level is made from the unscaled level above it first, and only then scaled, so a level's scale doesn't compound into the next.

### Code
`game-engine/shaders/mips.slang`:
```slang
// Mip chains for the scene's textures, in compute shaders. Every level of
// every texture lives in one buffer, as 8-bit RGBA texels packed in a uint,
// level after level; texture.cpp copies the finished levels into the images.
//   prepareNormalsMain  a normal map's top level: no spread, in alpha
//   downsampleMain      each smaller level, from the one above it
//   histogramMain       a level's alpha values, counted in 256 bins
//   coverageMain        per level of a masked texture: the alpha scale that
//                       keeps as many texels above the cutoff as the top level
//   scaleAlphaMain      a level's alpha, times that scale

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

struct MipPushData {
    uint *source;         // texels read: the level above, or the level itself
    uint *target;         // texels written, or the histograms and scales
    uint2 source_size;    // in texels
    uint2 target_size;
    uint kind;            // a mip_ kind; coverageMain: the level count
    float cutoff;         // coverageMain: the alpha a texel must reach
};

[[vk::push_constant]]
ConstantBuffer<MipPushData> push;

// What a texture's texels hold, which decides how they're averaged.
static const uint mip_data = 0;         // plain numbers: averaged as they are
static const uint mip_color = 1;        // sRGB colors: averaged as linear light
static const uint mip_see_through = 2;  // sRGB colors with coverage or opacity in alpha:
                                        // as linear light, weighted by alpha
static const uint mip_normal = 3;       // normals: averaged as vectors

// --- Texels --------------------------------------------------------------------

float4 unpack(uint texel) {
    return float4(texel & 0xFF, (texel >> 8) & 0xFF, (texel >> 16) & 0xFF, texel >> 24) / 255.0;
}

uint pack(float4 value) {
    const uint4 bytes = uint4(round(saturate(value) * 255.0));
    return bytes.x | (bytes.y << 8) | (bytes.z << 16) | (bytes.w << 24);
}

// The sRGB transfer function, both ways, as the Khronos Data Format
// Specification defines it for Vulkan's _SRGB formats: what the sampler does
// when it reads one.
float3 srgb_to_linear(float3 c) {
    return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4));
}

float3 linear_to_srgb(float3 c) {
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

// --- Normals and their spread ---------------------------------------------------

// Normals that spread around their average by an angle with standard
// deviation s average to a vector about 1 / (1 + s^2) long (Toksvig 2005).
// A normal map's alpha holds 1 - s: 1, as an 8-bit normal map's alpha
// usually is, means no spread, and near no spread, where it matters most,
// each 8-bit step is a tiny one.
//
// The direction is always renormalized as it's read: stored in 8 bits, a
// unit normal is only roughly unit length, up to about 0.6% short or long,
// and that rounding mustn't count as spread.

// A texel as a vector: its direction, unit length, times the length its
// spread gives.
float3 unpack_normal(float4 texel) {
    const float3 normal = texel.xyz * 2.0 - 1.0;
    const float length_squared = dot(normal, normal);
    const float3 direction = length_squared > 0.0 ? normal / sqrt(length_squared) : float3(0.0, 0.0, 1.0);
    const float spread = 1.0 - texel.w;
    return direction / (1.0 + spread * spread);
}

// The average of such vectors, as a texel: its direction, and the spread its
// length gives back, s^2 = (1 - length) / length.
float4 pack_normal(float3 average) {
    const float length = sqrt(dot(average, average));
    const float3 direction = length > 0.0 ? average / length : float3(0.0, 0.0, 1.0);
    const float spread = length > 0.0 ? sqrt(max(1.0 - length, 0.0) / length) : 1.0;
    return float4(direction * 0.5 + 0.5, 1.0 - min(spread, 1.0));
}

// A normal map's top level has no spread of its own: alpha 1. glTF ignores a
// normal map's alpha, so it's free to hold this.
[shader("compute")]
[numthreads(8, 8, 1)]
void prepareNormalsMain(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= push.source_size)) {
        return;
    }

    const uint index = id.y * push.source_size.x + id.x;
    push.source[index] |= 0xFF000000;
}

// --- Downsampling ----------------------------------------------------------------

// The share of source texel `i` (covering i..i+1) inside the footprint
// start..end, along one axis.
float overlap(float i, float start, float end) {
    return max(min(i + 1.0, end) - max(i, start), 0.0);
}

// One texel of the smaller level: the average of the source texels its area
// covers, a box filter. With an even size, that's a 2 x 2 block; with an odd
// one, the footprint is between 2 and 3 texels wide, and the texels it cuts
// get their share. The averages are of what the texels stand for:
//   colors       in linear light
//   see-through  in linear light, weighted by alpha: a texel that's
//                transparent has no color to contribute, so the cut-out
//                parts of a leaf texture don't darken its edges
//   normals      as vectors, with the average's spread in alpha
[shader("compute")]
[numthreads(8, 8, 1)]
void downsampleMain(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= push.target_size)) {
        return;
    }

    const float2 ratio = float2(push.source_size) / float2(push.target_size);
    const float2 start = float2(id.xy) * ratio;
    const float2 end = start + ratio;

    // Rounding can put the last footprint's end a hair past the level's edge.
    const uint2 first = uint2(start);
    const uint2 last = min(uint2(ceil(end)), push.source_size);

    float4 sum = 0.0;          // see-through: rgb weighted by alpha, and alpha
    float3 normal_sum = 0.0;
    float weight_sum = 0.0;

    for (uint y = first.y; y < last.y; ++y) {
        for (uint x = first.x; x < last.x; ++x) {
            const float weight = overlap(float(x), start.x, end.x) * overlap(float(y), start.y, end.y);
            const float4 texel = unpack(push.source[y * push.source_size.x + x]);

            if (push.kind == mip_color) {
                sum += weight * float4(srgb_to_linear(texel.rgb), texel.a);
            } else if (push.kind == mip_see_through) {
                sum += weight * float4(srgb_to_linear(texel.rgb) * texel.a, texel.a);
            } else if (push.kind == mip_normal) {
                normal_sum += weight * unpack_normal(texel);
            } else {
                sum += weight * texel;
            }

            weight_sum += weight;
        }
    }

    float4 result;

    if (push.kind == mip_color) {
        const float4 average = sum / weight_sum;
        result = float4(linear_to_srgb(average.rgb), average.a);
    } else if (push.kind == mip_see_through) {
        // Fully transparent everywhere: no color to weight, so keep black.
        const float3 color = sum.a > 0.0 ? sum.rgb / sum.a : 0.0;
        result = float4(linear_to_srgb(color), sum.a / weight_sum);
    } else if (push.kind == mip_normal) {
        result = pack_normal(normal_sum / weight_sum);
    } else {
        result = sum / weight_sum;
    }

    push.target[id.y * push.target_size.x + id.x] = pack(result);
}

// --- Alpha coverage (Castaño 2010) --------------------------------------------------

// A masked material cuts out every texel whose alpha is below its cutoff.
// Averaging alpha, the levels below blur the edges of the shapes; once a
// level's texels are mostly partly covered, fewer of them reach the cutoff,
// and the shape thins out, until far away a tree is bare branches. Scaling
// each level's alpha so that the same share of texels reaches the cutoff as
// on the top level keeps the coverage, and the shape.

// The 8-bit alpha of every texel of a level, counted into 256 bins.
[shader("compute")]
[numthreads(8, 8, 1)]
void histogramMain(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= push.source_size)) {
        return;
    }

    InterlockedAdd(push.target[push.source[id.y * push.source_size.x + id.x] >> 24], 1);
}

// One thread per level below the top: the 8-bit alpha `a` whose share of
// texels at or above it is closest to the top level's share at or above the
// cutoff. The scale takes alpha a to a little above the cutoff, and a - 1 to
// a little below: half of a scaled step. When coverage shrank down the
// chain, the scale is above 1 and that's more than half an 8-bit step, so
// rounding the scaled alpha keeps them on their sides. Below 1, several
// alphas round to one 8-bit value, and the split is only close. The
// histograms are level after level, 256 bins each; the scales follow them,
// one float per level.
[shader("compute")]
[numthreads(64, 1, 1)]
void coverageMain(uint3 id : SV_DispatchThreadID) {
    const uint levels = push.kind;
    const uint level = id.x + 1;

    if (level >= levels) {
        return;
    }

    uint *top = push.source;
    uint *histogram = push.source + 256 * level;
    float *scales = (float*)push.target;

    // The top level's covered share: texels at or above the cutoff.
    const uint first_covered = uint(ceil(push.cutoff * 255.0 - 1e-3));
    uint top_total = 0;
    uint top_covered = 0;
    uint total = 0;

    for (uint bin = 0; bin < 256; ++bin) {
        top_total += top[bin];
        top_covered += bin >= first_covered ? top[bin] : 0;
        total += histogram[bin];
    }

    const float target = float(top_covered) / float(top_total);

    // Going down from the most opaque bin, the share at or above it only
    // grows; stop where it's closest to the target.
    uint best = 255;
    float best_error = 2.0;
    uint above = 0;

    for (int bin = 255; bin >= 1; --bin) {
        above += histogram[bin];
        const float error = abs(float(above) / float(total) - target);

        if (error < best_error) {
            best = uint(bin);
            best_error = error;
        }
    }

    scales[level] = push.cutoff / ((float(best) - 0.5) / 255.0);
}

// A level's alpha, times its scale.
[shader("compute")]
[numthreads(8, 8, 1)]
void scaleAlphaMain(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= push.source_size)) {
        return;
    }

    const uint index = id.y * push.source_size.x + id.x;
    const float scale = ((float*)push.target)[0];
    float4 texel = unpack(push.source[index]);
    texel.a *= scale;
    push.source[index] = pack(texel);
}
```

## 15.2 The push data: `shader_types.h`

### Why
Every step of `mips.slang` takes the same push data.

### How
- **`MipPushData`:** the source and target addresses, their sizes in texels, the texture's kind, and the cutoff. `coverageMain` reuses `kind` for the level count. 40 bytes, no padding.
- **`MipKind`:** data, color, see-through color or normal map.

### Code
`game-engine/src/includes/shader_types.h`:
```cpp
#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

// C++ mirrors of the structs in the shaders (shaders/*.slang).
// The GPU reads these bytes as they are, so the two sides must agree on every
// size and offset; the static_asserts catch a mismatch at compile time.

// --- Vertex ------------------------------------------------------------------

// Slang lays out data behind a pointer like C: each member aligned only to
// the size of its scalar type. Every member here is made of 4-byte floats,
// so nothing needs padding, and glm agrees member for member. All of these
// structs are packed tight like this: no padding anywhere.
//   - A normal of (0, 0, 0) means the file had none (see mesh.slang).
//   - A tangent of (0, 0, 0, 0) means the file had none; the shader then
//     works the tangent out from the texture coordinates.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;  // xyz: the direction of +u on the surface; w: +1 or -1, the bitangent's sign
    glm::vec2 uv0;      // TEXCOORD_0
    glm::vec2 uv1;      // TEXCOORD_1
    glm::vec4 color;    // COLOR_0, linear RGBA; white when the file has none
};

static_assert(sizeof(Vertex) == 72);
static_assert(offsetof(Vertex, tangent) == 24);
static_assert(offsetof(Vertex, uv0) == 40);
static_assert(offsetof(Vertex, uv1) == 48);
static_assert(offsetof(Vertex, color) == 56);

// --- Per-draw data -----------------------------------------------------------

// One per draw, in a GPU buffer the shaders index. A draw is placed in a
// world cell (cells.h): its model matrix moves the primitive into the cell,
// measured from the cell's corner. A shadow ray that hits a draw's triangle
// finds the triangle's vertices through first_index and vertex_offset, as
// drawIndexed does. The cull tests the draw's box.
struct DrawData {
    glm::mat4 model;            // this primitive's space -> its cell, from the cell's corner
    glm::mat4 normal_matrix;    // transposed inverse of model: keeps normals perpendicular under any scale
    std::uint32_t material;     // index into the material buffer
    std::uint32_t first_index;  // where the primitive's indices start in the index buffer
    std::int32_t vertex_offset; // added to each index: where its vertices start in the vertex buffer
    glm::ivec3 cell;            // the world cell the draw is placed in
    glm::vec3 bounds_min;       // the draw's box, in its cell, which holds all of its triangles
    glm::vec3 bounds_max;
};

static_assert(sizeof(DrawData) == 176);
static_assert(offsetof(DrawData, normal_matrix) == 64);
static_assert(offsetof(DrawData, material) == 128);
static_assert(offsetof(DrawData, cell) == 140);
static_assert(offsetof(DrawData, bounds_min) == 152);

// --- Materials ---------------------------------------------------------------

// glTF's three ways of using a material's alpha. Each gets its own pipeline,
// and the shader reads the mode as a specialization constant.
enum class AlphaMode : std::uint32_t {
    opaque,  // alpha is ignored
    mask,    // fully opaque or fully transparent: cut out below alpha_cutoff
    blend,   // see-through: blended over what's behind it
};

// Which texture a material slot samples, with which sampler and which set of
// texture coordinates. Heap indices: texture 0 is a 1x1 white texture and
// sampler 0 the default sampler, for slots the file leaves empty.
struct TextureSlot {
    std::uint32_t texture = 0;  // resource heap index
    std::uint32_t sampler = 0;  // sampler heap index
    std::uint32_t uv_set = 0;   // 0: TEXCOORD_0, 1: TEXCOORD_1
};

static_assert(sizeof(TextureSlot) == 12);

// A glTF metallic-roughness material: every factor and texture of the core
// spec. Each texture is multiplied by its factor; see mesh.slang for how.
struct Material {
    glm::vec4 base_color_factor;     // linear RGBA
    glm::vec3 emissive_factor;       // linear RGB light the surface gives off
    float metallic_factor;           // 1: metal, 0: not
    float roughness_factor;          // 1: fully rough, 0: mirror smooth
    float normal_scale;              // how strongly the normal map tilts the normal
    float occlusion_strength;        // 0: ignore the occlusion map, 1: use it fully
    float alpha_cutoff;              // AlphaMode::mask: alpha below this is cut out
    std::uint32_t double_sided;      // 1: both sides are drawn and lit
    AlphaMode alpha_mode;            // for shadow rays, which meet every kind of material
    TextureSlot base_color;          // RGBA, sRGB
    TextureSlot metallic_roughness;  // G: roughness, B: metallic
    TextureSlot normal;              // tangent-space normal
    TextureSlot occlusion;           // R: how much ambient light reaches the surface
    TextureSlot emissive;            // RGB, sRGB
};

static_assert(sizeof(Material) == 116);
static_assert(offsetof(Material, emissive_factor) == 16);
static_assert(offsetof(Material, metallic_factor) == 28);
static_assert(offsetof(Material, double_sided) == 48);
static_assert(offsetof(Material, alpha_mode) == 52);
static_assert(offsetof(Material, base_color) == 56);
static_assert(offsetof(Material, emissive) == 104);

// --- Lights ------------------------------------------------------------------

// KHR_lights_punctual's three kinds of light. "Punctual" means infinitely
// small: all of a light's power comes from one point, or one direction.
enum class LightType : std::uint32_t {
    directional,  // like the sun: parallel rays, intensity in lux
    point,        // shines in every direction, intensity in candela
    spot,         // a point light limited to a cone, intensity in candela
};

// One light from the file, placed in a world cell (cells.h); its direction is
// along the world's axes.
struct Light {
    glm::vec3 offset;     // point and spot lights: where in `cell` the light is (cells.h)
    float range;          // distance where the light fades to nothing; 0 for no limit
    glm::vec3 direction;  // spot and directional lights: the way the light shines
    float spot_scale;     // spot cone falloff: 1 / (cos(inner) - cos(outer))
    glm::vec3 intensity;  // color times intensity
    float spot_offset;    // spot cone falloff: -cos(outer) * spot_scale
    glm::ivec3 cell;      // point and spot lights: the world cell the light is in
    LightType type;
};

static_assert(sizeof(Light) == 64);
static_assert(offsetof(Light, direction) == 16);
static_assert(offsetof(Light, intensity) == 32);
static_assert(offsetof(Light, cell) == 48);
static_assert(offsetof(Light, type) == 60);

// --- Views -------------------------------------------------------------------

// What the fragment shader outputs: the shaded scene, one material input on
// its own, for checking that each one loaded correctly, the ambient
// occlusion, or how much of the sun's light reaches each point. Keys 1-9
// and 0 pick one.
enum class View : std::uint32_t {
    lit,
    base_color,
    normal,         // the final normal, normal map included
    vertex_normal,  // the interpolated vertex normal, without the normal map
    metallic,
    roughness,
    occlusion,
    emissive,
    ambient_occlusion,  // GTAO's visibility: white open, black occluded
    shadow,             // the sun's light that gets through: white all of it, black none
};

// --- The environment -----------------------------------------------------------

// What the environment's compute shaders tell the CPU and the scene shader
// about the sky, in host-visible memory both can read.
//   - irradiance_sh: the light falling on a surface from the whole sky, as 9
//     spherical harmonics coefficients per color channel (see environment.slang).
//   - sun_illuminance: the sun's light at the ground after the atmosphere, in
//     lux on a surface facing it; 0 when the sun is down or the sky is an image.
struct EnvironmentInfo {
    std::array<glm::vec3, 9> irradiance_sh;
    glm::vec3 sun_illuminance;
};

static_assert(sizeof(EnvironmentInfo) == 120);

// --- Per-frame data -----------------------------------------------------------

// Everything the shaders need that's the same for every draw in a frame. Each
// frame in flight has its own copy in host-visible memory, rewritten by the
// CPU before the frame is recorded. Push data points at it.
//   - Lighting values are physical: lux for illuminance, nits (candela per
//     square meter) for the brightness of the sky.
//   - Pointers come right after the matrices, so all of them land on 8-byte
//     boundaries with no padding.
// Shaders work in camera-relative space: the world's axes, with the camera
// at the origin. A position given by a cell and an offset (cells.h) is moved
// into it by subtracting the camera's cell and offset.
struct FrameData {
    glm::mat4 view_projection;          // camera-relative space -> clip space
    glm::mat4 inverse_view_projection;  // clip space -> camera-relative space
    vk::DeviceAddress vertices;         // the scene's vertices
    vk::DeviceAddress indices;          // the scene's indices, for shadow rays' alpha tests
    vk::DeviceAddress draws;            // one DrawData per draw
    vk::DeviceAddress instances;        // the cull's visible draws: what each instance draws
    vk::DeviceAddress materials;        // the scene's materials
    vk::DeviceAddress lights;           // the file's lights
    vk::DeviceAddress environment;      // the EnvironmentInfo
    vk::DeviceAddress scene_tlas;       // the top-level acceleration structure, for ray queries
    glm::ivec3 camera_cell;             // the world cell the camera is in
    float exposure;                     // scales light into the 0..1 range the tone mapper expects
    glm::vec3 sun_direction;            // unit vector pointing toward the sun
    std::uint32_t light_count;          // how many Lights `lights` holds
    glm::vec3 sun_illuminance;          // lux, per color channel, on a surface facing the sun
    View view;                          // what the fragment shader outputs
    std::uint32_t sky_cube;             // resource heap slot: the sky, full detail
    std::uint32_t specular_cube;        // resource heap slot: the sky prefiltered per roughness
    std::uint32_t brdf_lut;             // resource heap slot: the split-sum BRDF table
    std::uint32_t clamp_sampler;        // sampler heap index: trilinear, clamped to the edge
    std::uint32_t specular_mips;        // mip levels of specular_cube: roughness 0 to 1
    float sun_angular_radius;           // radians: half the sun's apparent width
    std::uint32_t ambient_occlusion;    // resource heap slot: the GTAO image, full resolution
    std::uint32_t ao_enabled;           // 0: ignore it, to compare
    glm::vec3 camera_offset;            // where in its cell the camera is
    glm::vec3 tlas_offset;              // the camera, measured from the TLAS's origin (acceleration.h)
    std::uint32_t sky_view;             // resource heap slot: the sky-view table
    std::uint32_t aerial_inscatter;     // resource heap slot: the air's light, as a volume over the view
    std::uint32_t aerial_transmittance; // resource heap slot: the air's transmittance, likewise
    std::uint32_t atmosphere;           // 1: the simulated sky, with its haze; 0: the photograph, without
};

static_assert(sizeof(FrameData) == 312);
static_assert(offsetof(FrameData, vertices) == 128);
static_assert(offsetof(FrameData, instances) == 152);
static_assert(offsetof(FrameData, scene_tlas) == 184);
static_assert(offsetof(FrameData, camera_cell) == 192);
static_assert(offsetof(FrameData, sun_direction) == 208);
static_assert(offsetof(FrameData, sun_illuminance) == 224);
static_assert(offsetof(FrameData, sky_cube) == 240);
static_assert(offsetof(FrameData, sun_angular_radius) == 260);
static_assert(offsetof(FrameData, ambient_occlusion) == 264);
static_assert(offsetof(FrameData, camera_offset) == 272);
static_assert(offsetof(FrameData, tlas_offset) == 284);
static_assert(offsetof(FrameData, sky_view) == 296);

// --- Push data ---------------------------------------------------------------

// Written with vkCmdPushDataEXT before each pipeline's draws: where this
// frame's data is. Which DrawData an instance draws, the vertex shader looks
// up among the cull's instances.
struct PushData {
    vk::DeviceAddress frame;
};

static_assert(sizeof(PushData) == 8);

// --- GPU culling (culling.h, cull.slang) ---------------------------------------

// Draws of one primitive in one draw list, a run of the cull's order: drawn
// as one instanced command, of as many instances as are in view.
struct DrawGroup {
    std::uint32_t first;          // where its draws start in the order
    std::uint32_t count;          // how many draws
    std::uint32_t list;           // its draw list
    std::uint32_t index_count;    // the primitive's drawIndexed arguments
    std::uint32_t first_index;
    std::int32_t vertex_offset;
};

static_assert(sizeof(DrawGroup) == 24);

// A draw list's groups, a run of the group table. Each group has one command
// slot, so it's the list's run of commands too.
struct DrawListRange {
    std::uint32_t first_group = 0;
    std::uint32_t group_count = 0;
};

static_assert(sizeof(DrawListRange) == 8);

// Where all of the cull's buffers are, in one table the steps read.
struct CullTables {
    vk::DeviceAddress order;        // draw indices, by list, then primitive
    vk::DeviceAddress groups;       // one DrawGroup per group
    vk::DeviceAddress lists;        // one DrawListRange per list
    vk::DeviceAddress visible;      // per draw in the order: 1 if in view
    vk::DeviceAddress draw_slots;   // prefix sums of visible, then their total
    vk::DeviceAddress group_flags;  // per group: 1 if any of its draws is in view
    vk::DeviceAddress group_slots;  // prefix sums of group_flags, then their total
    vk::DeviceAddress instances;    // the visible draws' indices
    vk::DeviceAddress commands;     // one VkDrawIndexedIndirectCommand per group
    vk::DeviceAddress counts;       // per list, how many commands to draw
    std::uint32_t draw_count;
    std::uint32_t group_count;
};

static_assert(sizeof(CullTables) == 88);

// Every cull step's push data: the frame (the draws and the view) and the tables.
struct CullPushData {
    vk::DeviceAddress frame;
    vk::DeviceAddress tables;
};

static_assert(sizeof(CullPushData) == 16);

// The tone-mapping pass's push data: which resource heap slot holds the HDR
// image, and the view, so material views can skip tone mapping.
struct TonemapPushData {
    std::uint32_t hdr_image;
    View view;
};

// The transparency composite's push data (composite.slang): the resource heap
// slots of the transparency pass's two sums.
struct CompositePushData {
    std::uint32_t accum;
    std::uint32_t reveal;
};

// The environment compute shaders' push data. Each dispatch sets only what
// its shader reads, so every member has a default.
struct EnvironmentPushData {
    vk::DeviceAddress info = 0;     // where the EnvironmentInfo goes
    std::uint32_t source = 0;       // resource heap slot to read
    std::uint32_t target = 0;       // resource heap slot to write (a storage image)
    std::uint32_t size = 0;         // the target's width and height in texels
    float roughness = 0.0f;         // prefiltering: the roughness this mip level is for
    std::uint32_t sampler = 0;      // sampler heap index: the clamp sampler
    std::uint32_t source_size = 0;  // the source cube's face size at mip 0
};

static_assert(sizeof(EnvironmentPushData) == 32);
static_assert(offsetof(EnvironmentPushData, size) == 16);

// The atmosphere's compute shaders' push data (atmosphere.slang). Each step
// reads what it needs: the per-frame ones the camera's height and the sun,
// the aerial perspective the frame's view. Push data follows std430 rules,
// where a vec3 starts on a 16-byte boundary: the two pointers fill the
// first 16 bytes, so sun_direction lands on one.
struct AtmospherePushData {
    vk::DeviceAddress frame = 0;       // aerial perspective: this frame's FrameData
    vk::DeviceAddress info = 0;        // the sky cube: where the sunlight at the camera goes
    glm::vec3 sun_direction{0.0f};     // toward the sun
    float altitude = 0.0f;             // the camera's height above the ground, in metres
    std::uint32_t transmittance = 0;   // resource heap slot: the transmittance table, sampled
    std::uint32_t multiscatter = 0;    // resource heap slot: the multiple scattering table, sampled
    std::uint32_t target = 0;          // resource heap slot this step writes (storage)
    std::uint32_t second_target = 0;   // aerial perspective: the transmittance volume (storage)
    std::uint32_t size = 0;            // the sky cube's face size
    std::uint32_t sampler = 0;         // sampler heap index: the clamp sampler
};

static_assert(sizeof(AtmospherePushData) == 56);
static_assert(offsetof(AtmospherePushData, sun_direction) == 16);
static_assert(offsetof(AtmospherePushData, transmittance) == 32);

// The mip chain compute shaders' push data (mips.slang). Every level of
// every texture is in one buffer, reached through addresses.
struct MipPushData {
    vk::DeviceAddress source = 0;      // texels read: the level above, or the level itself
    vk::DeviceAddress target = 0;      // texels written, or the histograms and scales
    glm::uvec2 source_size{0};         // in texels
    glm::uvec2 target_size{0};
    std::uint32_t kind = 0;            // a MipKind; coverageMain: the level count
    float cutoff = 0.0f;               // coverageMain: the alpha a texel must reach
};

static_assert(sizeof(MipPushData) == 40);

// What a texture's texels hold, which decides how mips average them.
enum class MipKind : std::uint32_t {
    data = 0,         // plain numbers: averaged as they are
    color = 1,        // sRGB colors: averaged as linear light
    see_through = 2,  // sRGB colors with coverage or opacity in alpha: weighted by it too
    normal = 3,       // normal map: averaged as vectors, their spread in alpha
};

// The ambient occlusion compute shaders' push data (ao.slang), the same for
// all four steps. The half-resolution images and `source` and `target` are
// storage images; each step reads and writes the ones it needs.
struct AoPushData {
    vk::DeviceAddress frame = 0;   // this frame's FrameData
    std::uint32_t depth = 0;       // resource heap slot: the depth buffer, sampled
    std::uint32_t normals = 0;     // resource heap slot: the prepass's normals, sampled
    std::uint32_t ao_depth = 0;    // resource heap slot: half resolution, nearest distance (storage)
    std::uint32_t ao_normals = 0;  // resource heap slot: half resolution, its normal (storage)
    std::uint32_t source = 0;      // resource heap slot: what this step reads (storage)
    std::uint32_t target = 0;      // resource heap slot: what this step writes (storage)
    std::uint32_t width = 0;       // the full-resolution images' size in pixels
    std::uint32_t height = 0;
    float radius = 0.0f;           // meters: how far around a point occluders are looked for
    std::uint32_t slices = 0;      // directions around the view vector
    std::uint32_t steps = 0;       // samples along each direction, each way
    std::uint32_t blur_axis = 0;   // the blur's direction: 0 across, 1 down
};

static_assert(sizeof(AoPushData) == 56);
```

## 15.3 Building the chains: `texture.cpp`

### Why
`create_scene_textures` now plans each texture's chain from what the materials use it for, runs the compute steps, and copies the levels into the images.

### How
- **`plan_mips`:** for each image:
  - **Its kind:** sRGB images are colors, see-through if every use is a masked or blended material's base color. An image only ever used as a normal map is a normal map. Anything else is data, including an image that's a normal map for one material and something else for another: its alpha may mean something, so it's left alone. `mesh.slang` still reads that alpha as spread, which is fine for the usual alpha of 1, and wrong for the rare file that keeps something else there.
  - **Its cutoff:** a masked material keeps a pixel where its base color factor's alpha, times the texture's, times the vertex color's, reaches `alpha_cutoff`. With vertex color 1, the texture's alpha must reach `alpha_cutoff / factor`. That's the cutoff coverage is kept for. A texture shared by masked materials with different cutoffs keeps the first one's.
  - **Its levels:** each level's size, halving down to 1 × 1, and its offset in the mip buffer.
- **The buffers:** the mip buffer, device-local, for every level of every texture; and the coverage buffer, for each masked texture's histograms, 256 counts per level, then its scales, one per level. The coverage buffer starts zeroed, for the histograms' atomic adds.
- **One submission:**
  1. Each image's pixels are copied from staging into its top level; the coverage buffer is cleared.
  2. Normal maps' top levels are prepared.
  3. Level by level, every texture's next level is made from the one above. Textures don't depend on each other, so a single barrier per level covers all of them.
  4. Masked base colors: histograms, scales, then scaling.
  5. Each image gets all its levels in one copy, with one region per level, and is moved to `eShaderReadOnlyOptimal`.
- **Barriers:** every step reads what the step before wrote, through buffer addresses, so a global memory barrier between them is all it takes: no image layouts until the copies.
- **Workgroups of 8 × 8,** over a level's width and height, for every per-texel step. One row of 64-thread groups would need 65,536 of them for a 2048 × 2048 texture, one more than the 65,535 Vulkan guarantees along a dimension.
- **The images** are now only copied into and sampled: no more `eTransferSrc`. The blit check goes too. Vulkan requires `R8G8B8A8_UNORM` and `_SRGB` to support linear filtering, so every GPU can sample them smoothly.
- **The cost:** Sponza's 70 textures load in about 240 ms in release, mostly decoding.

### Code
`game-engine/src/texture.cpp`:
```cpp
#include "includes/texture.h"

#include "includes/buffer.h"
#include "includes/pipeline.h"
#include "includes/shader_types.h"

#include <jpgd.h>
#include <spng.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <print>
#include <thread>

namespace {

// --- Decoding ----------------------------------------------------------------

// What a failed image becomes: one magenta pixel, impossible to miss.
DecodedImage missing_image(std::string error) {
    return DecodedImage{.width = 1, .height = 1, .rgba = {255, 0, 255, 255}, .error = std::move(error)};
}

DecodedImage decode_jpeg(std::span<const std::uint8_t> encoded) {
    int width = 0;
    int height = 0;
    int components = 0;

    // Asking for 4 components gives RGBA, with alpha 255. jpgd allocates the
    // pixels with malloc, so free() releases them.
    const std::unique_ptr<unsigned char, decltype(&std::free)> pixels(
        jpgd::decompress_jpeg_image_from_memory(encoded.data(), static_cast<int>(encoded.size()), &width, &height, &components, 4),
        &std::free
    );

    if (!pixels) {
        return missing_image("not a JPEG jpgd can decode");
    }

    const std::size_t size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4;

    return DecodedImage{
        .width = static_cast<std::uint32_t>(width),
        .height = static_cast<std::uint32_t>(height),
        .rgba = std::vector<std::uint8_t>(pixels.get(), pixels.get() + size),
    };
}

DecodedImage decode_png(std::span<const std::uint8_t> encoded) {
    const std::unique_ptr<spng_ctx, decltype(&spng_ctx_free)> context(spng_ctx_new(0), &spng_ctx_free);

    spng_ihdr header{};
    std::size_t size = 0;

    // SPNG_FMT_RGBA8 converts any PNG (grey, palette, 16-bit, ...) to 8-bit
    // RGBA; SPNG_DECODE_TRNS turns a transparency chunk into real alpha.
    const bool decoded = context
        && spng_set_png_buffer(context.get(), encoded.data(), encoded.size()) == 0
        && spng_get_ihdr(context.get(), &header) == 0
        && spng_decoded_image_size(context.get(), SPNG_FMT_RGBA8, &size) == 0;

    if (!decoded) {
        return missing_image("not a PNG libspng can read");
    }

    DecodedImage image{.width = header.width, .height = header.height, .rgba = std::vector<std::uint8_t>(size)};

    if (spng_decode_image(context.get(), image.rgba.data(), size, SPNG_FMT_RGBA8, SPNG_DECODE_TRNS) != 0) {
        return missing_image("libspng failed to decode it");
    }

    return image;
}

// Chooses the decoder from the file's first bytes, its "magic number", which
// is more reliable than the file name or the glTF mimeType.
DecodedImage decode_image(std::span<const std::uint8_t> encoded) {
    constexpr std::uint8_t png[] = {0x89, 'P', 'N', 'G'};
    constexpr std::uint8_t jpeg[] = {0xFF, 0xD8, 0xFF};

    if (encoded.size() >= 4 && std::equal(std::begin(png), std::end(png), encoded.begin())) {
        return decode_png(encoded);
    }

    if (encoded.size() >= 3 && std::equal(std::begin(jpeg), std::end(jpeg), encoded.begin())) {
        return decode_jpeg(encoded);
    }

    return missing_image(encoded.empty() ? "no image data" : "not a PNG or JPEG");
}

// --- Uploading ---------------------------------------------------------------

// Moves mip levels [base, base + count) of `image` between layouts.
void transition_mips(
    const vk::raii::CommandBuffer &commands,
    vk::Image image,
    std::uint32_t base,
    std::uint32_t count,
    vk::ImageLayout from,
    vk::ImageLayout to,
    vk::PipelineStageFlags2 src_stage,
    vk::AccessFlags2 src_access,
    vk::PipelineStageFlags2 dst_stage,
    vk::AccessFlags2 dst_access
) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = base,
            .levelCount = count,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    commands.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}

// Makes writes by `src_stage` visible to `dst_stage`, for every buffer.
void memory_barrier(
    const vk::raii::CommandBuffer &commands,
    vk::PipelineStageFlags2 src_stage,
    vk::AccessFlags2 src_access,
    vk::PipelineStageFlags2 dst_stage,
    vk::AccessFlags2 dst_access
) {
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
    };

    commands.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

// An image for `decoded` with room for every mip level, in device-local memory.
Texture create_texture(const vk::raii::Device &device, const GpuChoice &gpu, const DecodedImage &decoded, vk::Format format) {
    Texture texture;
    texture.format = format;
    texture.extent = vk::Extent2D{.width = decoded.width, .height = decoded.height};

    // Halving until 1x1: a 1024x1024 image has 11 levels (1024, 512, ..., 1).
    texture.mip_levels = std::bit_width(std::max(decoded.width, decoded.height));

    texture.handle = vk::raii::Image(device, vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {decoded.width, decoded.height, 1},
        .mipLevels = texture.mip_levels,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        // Every level is copied in from the mip buffer, then sampled.
        .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    });

    const vk::MemoryRequirements requirements = texture.handle.getMemoryRequirements();

    texture.memory = vk::raii::DeviceMemory(device, vk::MemoryAllocateInfo{
        .allocationSize = requirements.size,
        .memoryTypeIndex = find_memory_type(gpu, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
    });

    texture.handle.bindMemory(*texture.memory, 0);
    return texture;
}

// --- Mip chains --------------------------------------------------------------

// One texture's levels in the mip buffer, and how they're made.
struct MipChain {
    std::vector<glm::uvec2> sizes;          // each level's, in texels
    std::vector<vk::DeviceSize> offsets;    // each level's, in bytes into the mip buffer
    MipKind kind = MipKind::data;
    float cutoff = 0.0f;                    // a masked base color's: the texture alpha that's kept; 0 for none
    vk::DeviceSize coverage = 0;            // masked: where its histograms, then its scales, start in the coverage buffer
};

// What each image is used for, from the materials: how its mips average
// (MipKind), and, for a masked material's base color, the alpha its texels
// must reach to be kept. Texture i + 1 is image i; texture 0 is white.
std::vector<MipChain> plan_mips(const Scene &scene, std::span<const DecodedImage> images) {
    std::vector<MipChain> chains(images.size());
    std::vector<bool> as_normal(images.size(), false);
    std::vector<bool> as_other(images.size(), false);        // anything but a normal map
    std::vector<bool> as_see_through(images.size(), false);  // a masked or blended base color
    std::vector<bool> as_opaque_color(images.size(), false); // any other color: its alpha means nothing

    for (const SceneMaterial &material : scene.materials) {
        for (const TextureRef *ref : {&material.base_color, &material.metallic_roughness, &material.occlusion, &material.emissive}) {
            if (ref->image >= 0) {
                as_other[static_cast<std::size_t>(ref->image) + 1] = true;
            }
        }

        if (material.normal.image >= 0) {
            as_normal[static_cast<std::size_t>(material.normal.image) + 1] = true;
        }

        if (material.base_color.image >= 0) {
            const auto image = static_cast<std::size_t>(material.base_color.image) + 1;
            (material.alpha_mode == AlphaMode::opaque ? as_opaque_color : as_see_through)[image] = true;
        }

        if (material.emissive.image >= 0) {
            as_opaque_color[static_cast<std::size_t>(material.emissive.image) + 1] = true;
        }

        // The material keeps a pixel where factor x texture x vertex color
        // reaches the cutoff; with vertex color 1, the texture's alpha must
        // reach cutoff / factor. Above 1, nothing is kept at any level.
        const float factor = material.base_color_factor.a;
        if (material.alpha_mode == AlphaMode::mask && material.base_color.image >= 0 && factor > 0.0f) {
            MipChain &chain = chains[static_cast<std::size_t>(material.base_color.image) + 1];
            const float cutoff = material.alpha_cutoff / factor;

            // A texture shared by masked materials with different cutoffs
            // keeps the first one's coverage.
            if (chain.cutoff == 0.0f && cutoff > 0.0f && cutoff <= 1.0f) {
                chain.cutoff = cutoff;
            }
        }
    }

    vk::DeviceSize offset = 0;

    for (std::size_t i = 0; i < images.size(); ++i) {
        MipChain &chain = chains[i];
        const bool srgb = i == 0 || scene.images[i - 1].srgb;

        // Colors weight by alpha only where every use takes alpha as
        // coverage or opacity: an opaque material shows its texels whatever
        // their alpha. An image that's a normal map and something else too
        // is averaged as plain data: its alpha may mean something.
        if (srgb) {
            chain.kind = as_see_through[i] && !as_opaque_color[i] ? MipKind::see_through : MipKind::color;
        } else {
            chain.kind = as_normal[i] && !as_other[i] ? MipKind::normal : MipKind::data;
        }

        glm::uvec2 size{images[i].width, images[i].height};
        const std::uint32_t levels = std::bit_width(std::max(size.x, size.y));

        for (std::uint32_t level = 0; level < levels; ++level) {
            chain.sizes.push_back(size);
            chain.offsets.push_back(offset);
            offset += vk::DeviceSize{size.x} * size.y * 4;
            size = glm::max(size / 2u, glm::uvec2{1});
        }
    }

    return chains;
}

// Runs one of mips.slang's steps, over `x` x `y` threads in workgroups of
// `group_x` x `group_y`.
void dispatch_mips(
    const vk::raii::CommandBuffer &commands,
    const MipPushData &push,
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t group_x,
    std::uint32_t group_y
) {
    commands.pushDataEXT(vk::PushDataInfoEXT{
        .offset = 0,
        .data = {.address = &push, .size = sizeof(push)},
    });
    commands.dispatch((x + group_x - 1) / group_x, (y + group_y - 1) / group_y, 1);
}

}  // namespace

// --- Decoding ----------------------------------------------------------------

std::vector<DecodedImage> decode_images(std::span<const SceneImage> images) {
    std::vector<DecodedImage> decoded(images.size());

    // Each worker thread takes the next undecoded image until none are left.
    // Images are independent, so the threads never touch the same one.
    std::atomic<std::size_t> next{0};
    const unsigned workers = std::max(1u, std::thread::hardware_concurrency());

    {
        std::vector<std::jthread> threads;

        for (unsigned t = 0; t < workers; ++t) {
            threads.emplace_back([&] {
                for (std::size_t i = next++; i < images.size(); i = next++) {
                    decoded[i] = decode_image(std::span(images[i].encoded.data(), images[i].encoded.size()));
                }
            });
        }
    }  // jthreads join here, when they go out of scope

    for (std::size_t i = 0; i < images.size(); ++i) {
        if (!decoded[i].error.empty()) {
            std::println(stderr, "Image {} ({}): {}; using magenta", i, images[i].name, decoded[i].error);
        }
    }

    return decoded;
}

// --- GPU textures ------------------------------------------------------------

std::vector<Texture> create_scene_textures(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    const Scene &scene
) {
    // Index 0: white, so "no texture" can sample like any other.
    std::vector<DecodedImage> images{DecodedImage{.width = 1, .height = 1, .rgba = {255, 255, 255, 255}}};
    std::vector<vk::Format> formats{vk::Format::eR8G8B8A8Srgb};

    for (DecodedImage &image : decode_images(scene.images)) {
        images.push_back(std::move(image));
    }

    // sRGB formats make the GPU decode colors to linear when sampling; data
    // textures stay as they are. Vulkan requires both formats to support
    // linear filtering, so every GPU can sample them smoothly.
    for (const SceneImage &image : scene.images) {
        formats.push_back(image.srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm);
    }

    std::vector<MipChain> chains = plan_mips(scene, images);

    // One staging buffer holds every image's pixels back to back.
    vk::DeviceSize total = 0;
    for (const DecodedImage &image : images) {
        total += image.rgba.size();
    }

    const Buffer staging = create_buffer(device, gpu, total, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    auto *mapped = static_cast<std::uint8_t*>(staging.memory.mapMemory(0, total));
    std::vector<vk::DeviceSize> offsets;
    vk::DeviceSize offset = 0;

    for (const DecodedImage &image : images) {
        std::memcpy(mapped + offset, image.rgba.data(), image.rgba.size());
        offsets.push_back(offset);
        offset += image.rgba.size();
    }

    staging.memory.unmapMemory();

    // Every level of every texture, in device memory, where compute shaders
    // make the levels below the top through its address.
    const MipChain &last = chains.back();
    const vk::DeviceSize mip_bytes = last.offsets.back() + vk::DeviceSize{last.sizes.back().x} * last.sizes.back().y * 4;
    const Buffer mips = create_buffer(device, gpu, mip_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress
            | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Masked base colors also need, per level, a 256-bin alpha histogram,
    // then, per level, a scale: in a buffer of their own, zeroed first.
    vk::DeviceSize coverage_bytes = 4;  // never empty
    for (MipChain &chain : chains) {
        if (chain.cutoff > 0.0f) {
            chain.coverage = coverage_bytes;
            coverage_bytes += chain.sizes.size() * (256 + 1) * 4;
        }
    }

    const Buffer coverage = create_buffer(device, gpu, coverage_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress
            | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    const vk::raii::Pipeline prepare_normals = create_compute_pipeline(device, "mips", "prepareNormalsMain");
    const vk::raii::Pipeline downsample = create_compute_pipeline(device, "mips", "downsampleMain");
    const vk::raii::Pipeline histogram = create_compute_pipeline(device, "mips", "histogramMain");
    const vk::raii::Pipeline solve_coverage = create_compute_pipeline(device, "mips", "coverageMain");
    const vk::raii::Pipeline scale_alpha = create_compute_pipeline(device, "mips", "scaleAlphaMain");

    std::vector<Texture> textures;
    for (std::size_t i = 0; i < images.size(); ++i) {
        textures.push_back(create_texture(device, gpu, images[i], formats[i]));
    }

    const auto level_address = [&](const MipChain &chain, std::size_t level) { return mips.address + chain.offsets[level]; };
    const auto compute_to_compute = [](const vk::raii::CommandBuffer &commands) {
        memory_barrier(commands,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
    };

    // Everything in one command buffer and one submission.
    submit_and_wait(device, queue, pool, [&](const vk::raii::CommandBuffer &commands) {
        // 1. Each image's pixels into its top level; the coverage buffer zeroed.
        for (std::size_t i = 0; i < images.size(); ++i) {
            commands.copyBuffer(*staging.handle, *mips.handle, vk::BufferCopy{
                .srcOffset = offsets[i],
                .dstOffset = chains[i].offsets[0],
                .size = images[i].rgba.size(),
            });
        }

        commands.fillBuffer(*coverage.handle, 0, vk::WholeSize, 0);

        memory_barrier(commands,
            vk::PipelineStageFlagBits2::eCopy | vk::PipelineStageFlagBits2::eClear, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

        // 2. Normal maps' top levels: no spread, in alpha.
        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *prepare_normals);
        for (const MipChain &chain : chains) {
            if (chain.kind == MipKind::normal) {
                const MipPushData push{.source = level_address(chain, 0), .source_size = chain.sizes[0]};
                dispatch_mips(commands, push, chain.sizes[0].x, chain.sizes[0].y, 8, 8);
            }
        }

        compute_to_compute(commands);

        // 3. Level by level, every texture's next level from the one above.
        // Textures are independent, so one barrier per level covers them all.
        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *downsample);
        std::size_t most_levels = 0;
        for (const MipChain &chain : chains) {
            most_levels = std::max(most_levels, chain.sizes.size());
        }

        for (std::size_t level = 1; level < most_levels; ++level) {
            for (const MipChain &chain : chains) {
                if (level < chain.sizes.size()) {
                    const MipPushData push{
                        .source = level_address(chain, level - 1),
                        .target = level_address(chain, level),
                        .source_size = chain.sizes[level - 1],
                        .target_size = chain.sizes[level],
                        .kind = static_cast<std::uint32_t>(chain.kind),
                    };
                    dispatch_mips(commands, push, chain.sizes[level].x, chain.sizes[level].y, 8, 8);
                }
            }

            compute_to_compute(commands);
        }

        // 4. Masked base colors: count every level's alpha, work out each
        // level's scale, then apply it below the top.
        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *histogram);
        for (const MipChain &chain : chains) {
            for (std::size_t level = 0; chain.cutoff > 0.0f && level < chain.sizes.size(); ++level) {
                const MipPushData push{
                    .source = level_address(chain, level),
                    .target = coverage.address + chain.coverage + level * 256 * 4,
                    .source_size = chain.sizes[level],
                };
                dispatch_mips(commands, push, chain.sizes[level].x, chain.sizes[level].y, 8, 8);
            }
        }

        compute_to_compute(commands);

        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *solve_coverage);
        for (const MipChain &chain : chains) {
            if (chain.cutoff > 0.0f) {
                const MipPushData push{
                    .source = coverage.address + chain.coverage,
                    .target = coverage.address + chain.coverage + chain.sizes.size() * 256 * 4,
                    .kind = static_cast<std::uint32_t>(chain.sizes.size()),
                    .cutoff = chain.cutoff,
                };
                dispatch_mips(commands, push, static_cast<std::uint32_t>(chain.sizes.size()), 1, 64, 1);
            }
        }

        compute_to_compute(commands);

        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *scale_alpha);
        for (const MipChain &chain : chains) {
            for (std::size_t level = 1; chain.cutoff > 0.0f && level < chain.sizes.size(); ++level) {
                const MipPushData push{
                    .source = level_address(chain, level),
                    .target = coverage.address + chain.coverage + chain.sizes.size() * 256 * 4 + level * 4,
                    .source_size = chain.sizes[level],
                };
                dispatch_mips(commands, push, chain.sizes[level].x, chain.sizes[level].y, 8, 8);
            }
        }

        memory_barrier(commands,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);

        // 5. Every level into its image, then ready for the fragment shaders.
        for (std::size_t i = 0; i < textures.size(); ++i) {
            const Texture &texture = textures[i];
            const MipChain &chain = chains[i];

            transition_mips(commands, *texture.handle, 0, texture.mip_levels,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);

            std::vector<vk::BufferImageCopy> regions;
            for (std::uint32_t level = 0; level < texture.mip_levels; ++level) {
                regions.push_back(vk::BufferImageCopy{
                    .bufferOffset = chain.offsets[level],
                    .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = level, .baseArrayLayer = 0, .layerCount = 1},
                    .imageExtent = {chain.sizes[level].x, chain.sizes[level].y, 1},
                });
            }

            commands.copyBufferToImage(*mips.handle, *texture.handle, vk::ImageLayout::eTransferDstOptimal, regions);

            transition_mips(commands, *texture.handle, 0, texture.mip_levels,
                vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead);
        }
    });

    return textures;
}
```

## 15.4 Specular anti-aliasing: `mesh.slang`

### Why
With the normal map's spread in its alpha, the fragment shader has both spreads to add to the roughness.

### How
- **Spread is roughness.** Roughness describes how the microfacets' normals spread. Kaplanyan et al. ("Filtering Distributions of Normals for Shading Antialiasing", 2016) model it as a distribution of slopes: alpha squared is twice their variance, exactly for Beckmann's distribution, and closely enough to use for GGX's. Normals that spread across the pixel are, to the pixel, more microfacets, so their variance adds: alpha squared plus twice the pixel's variance.
- **Curvature (Tokuyoshi and Kaplanyan 2021):**
  - **The variance:** `ddx` and `ddy` of the normal give how much it changes from one pixel to the next, both ways. Times the pixel filter's variance, `σ² = 1 / (2π)` pixels squared, they give the variances of the normals over the pixel's footprint along the two screen axes. Their sum, `σ² (|∂n/∂x|² + |∂n/∂y|²)`, is at least the largest variance in any direction, so a round highlight widened by it is wide enough.
  - **Twice that** is added to alpha squared, the conversion from variance. That's their conservative form, their equation 13. Their equation 14 uses the mean of the two variances instead, half as much. They found it closest to the reference overall, but it lets slightly more aliasing through. Without temporal anti-aliasing to hide that, we take the conservative form.
  - **Capped at 0.18:** the difference between two pixels estimates the change only roughly. The cap, from Kaplanyan et al., keeps a bad estimate from turning a mirror into chalk.
  - **Which normal:** the vertex normal, the shape of the surface, as Filament and Unity's HDRP use it. The normal map's spread is already measured; its pixel-to-pixel differences would add a noisy, flickering estimate on top.
  - **Why this form:** the paper's main method filters by how the half vector changes between pixels, so it's redone for every light, and gives an anisotropic roughness, which needs an anisotropic BRDF. Their isotropic form, their Section 5, uses the normal instead. They derive it for deferred rendering, and note it suits forward rendering too: it works with the BRDF we have, for every light and the sky alike, and it doesn't depend on the light at all, so it's computed once per pixel.
- **The normal map (Toksvig 2005):** the spread `s` its mips measured, read from alpha as `1 − alpha`. Twice its variance, `2s²`, is added to alpha squared, like the curvature's, uncapped: the spread is measured, not estimated.
- **`surface_normal`** also returns `map_spread`: 0 without a map.
- **Where it's used:** the widened roughness goes everywhere the material's roughness went: the BRDF's alpha for the sun and the lights, the specular cube's mip and the BRDF table for the sky, and the energy compensation.
- **The roughness view** (key 6) now shows the roughness as shaded, widened, rather than the material's.

### Code
`game-engine/shaders/mesh.slang`:
```slang
// Draws one glTF primitive: its vertices come from the scene's vertex buffer,
// its place in the world from its DrawData, and its surface from its glTF
// material, whose textures are read from the descriptor heap. Shaded with
// glTF's physically based BRDF, lit by the sun and the file's lights, with
// ray-traced shadows, and by the sky around the scene, already exposed.
// Three fragment shaders:
//   prepassMain      the depth prepass's vertex normal
//   fragmentMain     opaque and masked surfaces, into the HDR image
//   transparentMain  blended surfaces, into weighted blended transparency's sums

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

#include "shared.slangh"
#include "atmosphere.slangh"

// In a descriptor heap pipeline, the push_constant block is where push data lands.
[[vk::push_constant]]
ConstantBuffer<PushData> push;

// The alpha mode this pipeline was built for (AlphaMode in C++):
// 0 opaque, 1 mask, 2 blend. A specialization constant: its value is
// fixed when the pipeline is created, so each pipeline's fragment shader
// keeps only the code its mode needs.
[vk::constant_id(0)]
const uint alpha_mode = 0;

static const uint alpha_opaque = 0;
static const uint alpha_mask = 1;
static const uint alpha_blend = 2;

// --- Stage interface ---------------------------------------------------------

// What the vertex shader hands to the rasterizer. SV_Position is the
// clip-space position; every other field but draw_index is interpolated
// across the triangle. Vulkan requires integer fields to be flat, which
// nointerpolation makes them.
struct VertexOutput {
    float4 position : SV_Position;
    float3 relative_position : POSITION;  // camera-relative: the world's axes, the camera at the origin
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float3 bitangent : BINORMAL;
    float2 uv0 : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
    float4 color : COLOR;
    nointerpolation uint draw_index : DRAW_INDEX;  // the same for a whole triangle, so never interpolated
};

// --- Vertex shader -----------------------------------------------------------

// SV_VulkanVertexID is Vulkan's own gl_VertexIndex, which includes the draw's
// vertexOffset: each primitive's indices start at 0, and the draw adds where
// that primitive's vertices begin in the shared buffer. SV_VulkanInstanceID
// is gl_InstanceIndex, which likewise counts from the command's
// firstInstance: the cull points that at the command's run of visible draws
// in `instances`, so each instance finds its draw there.
[shader("vertex")]
VertexOutput vertexMain(uint vertex_id : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    FrameData *frame = push.frame;
    const Vertex vertex = frame.vertices[vertex_id];
    const uint draw_index = frame.instances[instance];
    const DrawData draw = frame.draws[draw_index];

    // The vertex in its draw's cell, then relative to the camera.
    const float3 in_cell = mul(draw.model, float4(vertex.position, 1.0)).xyz;
    const float3 relative_position = camera_relative(frame, draw.cell, in_cell);

    // Tangent and bitangent lie along the surface, so they move with the
    // model matrix, like positions; only the normal needs the normal matrix.
    // The bitangent is built before the transform, from glTF's rule
    // B = cross(N, T) * w: a mirroring transform then mirrors it too.
    const float3 bitangent = cross(vertex.normal, vertex.tangent.xyz) * vertex.tangent.w;

    VertexOutput output;
    output.position = mul(frame.view_projection, float4(relative_position, 1.0));
    output.relative_position = relative_position;
    output.normal = mul((float3x3)draw.normal_matrix, vertex.normal);
    output.tangent = mul((float3x3)draw.model, vertex.tangent.xyz);
    output.bitangent = mul((float3x3)draw.model, bitangent);
    output.uv0 = vertex.uv0;
    output.uv1 = vertex.uv1;
    output.color = vertex.color;
    output.draw_index = draw_index;
    return output;
}

// --- Material textures -------------------------------------------------------

// Samples a material slot: its texture, with its sampler, at its set of
// texture coordinates. Descriptor heap access: a handle made from an index
// reads that descriptor from the bound heap.
float4 sample_slot(TextureSlot slot, VertexOutput input) {
    const Texture2D texture = Texture2D.Handle(uint2(slot.texture, 0));
    const SamplerState sampler = SamplerState.Handle(uint2(slot.sampler, 0));
    const float2 uv = slot.uv_set == 0 ? input.uv0 : input.uv1;
    return texture.Sample(sampler, uv);
}

// --- Normals -----------------------------------------------------------------

// The direction the surface faces at this pixel, for lighting.
//   1. The interpolated vertex normal. Without normals in the file, glTF asks
//      for flat shading: the triangle's own normal is the cross product of
//      how the position changes across neighbouring pixels (ddx, ddy).
//   2. A normal map tilts it, per texel, within the surface's tangent frame.
//   3. On a double-sided material's back face, the surface faces the other way.
// `map_spread`: how much the normal map's normals spread here, as the
// standard deviation of their angle, which the mips keep in its alpha as
// 1 - spread (mips.slang); 0 without a map.
float3 surface_normal(VertexOutput input, Material material, bool front_face, bool apply_normal_map, out float map_spread) {
    float3 normal = input.normal;
    map_spread = 0.0;

    // cross(ddy, ddx), not cross(ddx, ddy): Vulkan's screen Y points down,
    // so this order is the one that points toward the camera.
    if (all(normal == 0.0)) {
        normal = cross(ddy(input.relative_position), ddx(input.relative_position));
    }

    normal = normalize(normal);

    // Texture 0 is white: no normal map.
    const bool mapped = apply_normal_map && material.normal.texture != 0;

    float3 tangent = input.tangent;
    float3 bitangent = input.bitangent;

    // Without tangents in the file, work the frame out from how position and
    // texture coordinates change between neighbouring pixels (ddx, ddy):
    //     dp/dx = P_u * du/dx + P_v * dv/dx
    //     dp/dy = P_u * du/dy + P_v * dv/dy
    // Solving these for P_u and P_v, how position changes per unit of u and v,
    // gives the tangent (+u) and bitangent. glTF's v runs down the image while
    // a normal map's +Y points up, so the bitangent is -P_v. Only the
    // directions matter, so the determinant's sign stands in for dividing by
    // it. This can differ slightly from the MikkTSpace tangents glTF
    // specifies, but needs no precomputation.
    if (mapped && all(tangent == 0.0)) {
        const float2 uv = material.normal.uv_set == 0 ? input.uv0 : input.uv1;
        const float3 dp_dx = ddx(input.relative_position);
        const float3 dp_dy = ddy(input.relative_position);
        const float2 duv_dx = ddx(uv);
        const float2 duv_dy = ddy(uv);

        const float determinant = duv_dx.x * duv_dy.y - duv_dy.x * duv_dx.y;
        const float orientation = determinant < 0.0 ? -1.0 : 1.0;

        tangent = (dp_dx * duv_dy.y - dp_dy * duv_dx.y) * orientation;
        bitangent = -(dp_dy * duv_dx.x - dp_dx * duv_dy.x) * orientation;
    }

    // The frame's three axes flip together, keeping the map's tilt correct.
    if (material.double_sided != 0 && !front_face) {
        normal = -normal;
        tangent = -tangent;
        bitangent = -bitangent;
    }

    // Texture coordinates that don't change across the triangle give no frame
    // at all; the plain normal is all we have then.
    if (!mapped || all(tangent == 0.0) || all(bitangent == 0.0)) {
        return normal;
    }

    // All three axes must be unit length, or the map's tilt is scaled with
    // them. The normal already is; the other two grow and shrink with the
    // model matrix, and interpolation shortens them between vertices.
    tangent = normalize(tangent);
    bitangent = normalize(bitangent);

    // The map stores each component in 0..1; unpack to -1..1. normal_scale
    // scales the tilt: X and Y only, as glTF specifies.
    const float4 texel = sample_slot(material.normal, input);
    float3 tangent_space = texel.xyz * 2.0 - 1.0;
    tangent_space.xy *= material.normal_scale;
    map_spread = 1.0 - texel.w;

    return normalize(tangent * tangent_space.x + bitangent * tangent_space.y + normal * tangent_space.z);
}

// --- The glTF BRDF -----------------------------------------------------------

// glTF's metallic-roughness model, as its specification's Appendix B writes
// it. A BRDF says how much of the light arriving from one direction leaves
// toward another: here from the light (l) toward the viewer (v), around the
// half vector h between them.

static const float pi = 3.14159265;

// What shading needs to know about the surface at this pixel.
struct Surface {
    float3 base_color;
    float metallic;
    float alpha;   // roughness squared: the "alpha" of GGX
    float3 normal;
    float3 view;   // unit vector toward the camera
    float3 energy_compensation;  // what the specular reflection is scaled by (energy_compensation)
};

// D: the GGX (Trowbridge-Reitz) distribution of microfacet normals. Smooth
// surfaces have nearly all their tiny facets aligned with the normal, so D
// is a tall, narrow peak around h = n; rough ones spread it out.
float distribution_ggx(float n_dot_h, float alpha) {
    const float alpha2 = alpha * alpha;
    const float f = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (pi * f * f);
}

// V: Smith's height-correlated visibility, the share of facets neither in
// shadow nor hidden, with the BRDF's 1 / (4 n.l n.v) folded in.
float visibility_smith(float n_dot_l, float n_dot_v, float alpha) {
    const float alpha2 = alpha * alpha;
    const float from_view = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2);
    const float from_light = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2);
    const float sum = from_view + from_light;
    return sum > 0.0 ? 0.5 / sum : 0.0;
}

// The light leaving toward the viewer, in nits, from light arriving from
// direction `l` with illuminance `illuminance` (lux, on a surface facing it).
float3 shade(Surface surface, float3 l, float3 illuminance) {
    const float n_dot_l = dot(surface.normal, l);

    if (n_dot_l <= 0.0) {
        return float3(0.0);
    }

    const float3 h = normalize(l + surface.view);
    const float n_dot_v = max(dot(surface.normal, surface.view), 1e-4);
    const float n_dot_h = max(dot(surface.normal, h), 0.0);
    const float v_dot_h = max(dot(surface.view, h), 0.0);

    // Mirror-like reflection off the facets: white, for every material.
    const float specular = visibility_smith(n_dot_l, n_dot_v, surface.alpha) * distribution_ggx(n_dot_h, surface.alpha);

    // Schlick's Fresnel: every surface reflects more at grazing angles.
    const float fresnel = pow(1.0 - v_dot_h, 5.0);

    // Metals tint their reflection with the base color and have no diffuse
    // part. Dielectrics (everything else) reflect 4% head-on, rising to 100%
    // at grazing angles, and the rest enters the surface and scatters back
    // out as Lambertian diffuse light, colored by the base color.
    const float3 compensated = specular * surface.energy_compensation;
    const float3 metal = compensated * (surface.base_color + (1.0 - surface.base_color) * fresnel);
    const float3 dielectric = lerp(surface.base_color / pi, compensated, 0.04 + 0.96 * fresnel);
    const float3 brdf = lerp(dielectric, metal, surface.metallic);

    // Light falling at an angle spreads over more surface: the n.l factor.
    return brdf * illuminance * n_dot_l;
}

// --- Shadows -------------------------------------------------------------------

// How far a ray toward the sun, or another light infinitely far away, may go.
static const float infinite_distance = 1e9;

// A ray starting exactly on a surface can hit that same surface: the hit
// point's rounding puts it a hair below. This moves the origin off the
// surface along its geometric normal by up to 256 units in the last place
// (ULPs) of each coordinate: an offset that grows with the coordinates, so
// it suits any distance from the origin, where a fixed distance would be too
// much near it and too little far away. The constants are the authors',
// found by experiment. From "A Fast and Robust Method for Avoiding
// Self-Intersection" (Wachter and Binder, Ray Tracing Gems, 2019).
float3 offset_ray_origin(float3 position, float3 normal) {
    const float near_origin = 1.0 / 32.0;
    const float float_scale = 1.0 / 65536.0;
    const float int_scale = 256.0;

    float3 offset;
    for (int axis = 0; axis < 3; ++axis) {
        // Step the float's bits, as an integer, away from the surface.
        const int step = int(int_scale * normal[axis]);
        const float stepped = asfloat(asint(position[axis]) + (position[axis] < 0.0 ? -step : step));

        // Close to 0 a few units in the last place are tiny, so add a small
        // fixed distance there instead.
        offset[axis] = abs(position[axis]) < near_origin ? position[axis] + float_scale * normal[axis] : stepped;
    }

    return offset;
}

// The alpha of a masked or blended triangle a ray met, at the hit point. The
// hit's barycentric coordinates weight the triangle's three vertices; the
// texture is read at full resolution, since there are no neighbouring pixels
// to pick a mip level from. Rays from neighbouring pixels can hit different
// materials, so the texture's heap index differs between them: that's fine,
// since descriptor heap access is non-uniform unless the SPIR-V marks it
// uniform, and Slang doesn't.
float candidate_alpha(FrameData *frame, DrawData draw, Material material, uint triangle, float2 barycentrics) {
    const uint first = draw.first_index + triangle * 3;
    const Vertex v0 = frame.vertices[int(frame.indices[first]) + draw.vertex_offset];
    const Vertex v1 = frame.vertices[int(frame.indices[first + 1]) + draw.vertex_offset];
    const Vertex v2 = frame.vertices[int(frame.indices[first + 2]) + draw.vertex_offset];
    const float3 weight = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);

    const float2 uv = material.base_color.uv_set == 0
        ? v0.uv0 * weight.x + v1.uv0 * weight.y + v2.uv0 * weight.z
        : v0.uv1 * weight.x + v1.uv1 * weight.y + v2.uv1 * weight.z;
    const float vertex_alpha = v0.color.a * weight.x + v1.color.a * weight.y + v2.color.a * weight.z;

    const Texture2D texture = Texture2D.Handle(uint2(material.base_color.texture, 0));
    const SamplerState sampler = SamplerState.Handle(uint2(material.base_color.sampler, 0));
    return material.base_color_factor.a * vertex_alpha * texture.SampleLevel(sampler, uv, 0.0).a;
}

// How much of a light gets from `origin` to `distance` along `direction`: 0
// when something solid is in the way, otherwise the share every see-through
// layer on the way lets through. A ray query walks the TLAS and BLASes:
//   - an opaque triangle ends it at once: any blocking hit will do,
//   - a masked one comes back as a candidate, which blocks where its alpha
//     reaches the cutoff, and lets the light through its cut-out texels,
//   - a blended one comes back as a candidate that lets 1 - alpha of the
//     light through, as the transparency pass's reveal sum does. It never
//     ends the ray: the light goes on, dimmed, to whatever is behind.
float light_visibility(FrameData *frame, float3 origin, float3 direction, float distance) {
    const RaytracingAccelerationStructure scene = RaytracingAccelerationStructure(frame.scene_tlas);

    RayDesc ray;
    ray.Origin = origin;
    ray.TMin = 0.0;
    ray.Direction = direction;
    ray.TMax = distance;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);

    float transmittance = 1.0;

    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
            continue;
        }

        const DrawData draw = frame.draws[query.CandidateInstanceID()];
        const Material material = frame.materials[draw.material];
        const float alpha = candidate_alpha(frame, draw, material, query.CandidatePrimitiveIndex(),
            query.CandidateTriangleBarycentrics());

        if (material.alpha_mode == alpha_blend) {
            transmittance *= 1.0 - saturate(alpha);
        } else if (alpha >= material.alpha_cutoff) {
            query.CommitNonOpaqueTriangleHit();
        }
    }

    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0 : transmittance;
}

// Where a ray toward the light `l` starts: off the surface on the light's
// side, along `face_normal`, the triangle's own flat normal. An interpolated
// or normal-mapped normal can disagree about which side the light is on.
//
// The TLAS's space is measured from its origin cell, near the camera, not
// from the camera itself (acceleration.h): tlas_offset, the camera's
// position in it, moves the camera-relative position there first, so the
// offset grows with the coordinates the ray is really traced at.
float3 shadow_ray_origin(FrameData *frame, float3 position, float3 face_normal, float3 l) {
    return offset_ray_origin(position + frame.tlas_offset, dot(face_normal, l) >= 0.0 ? face_normal : -face_normal);
}

// shade(), times how much of the light gets through. A ray is only traced
// when the light could reach the surface at all.
float3 shade_shadowed(
    Surface surface, FrameData *frame, float3 position, float3 face_normal,
    float3 l, float3 illuminance, float distance
) {
    if (dot(surface.normal, l) <= 0.0 || all(illuminance == 0.0)) {
        return float3(0.0);
    }

    const float reaching = light_visibility(frame, shadow_ray_origin(frame, position, face_normal, l), l, distance);
    return reaching > 0.0 ? shade(surface, l, illuminance) * reaching : float3(0.0);
}

// --- Lights --------------------------------------------------------------------

// The direction toward a light, how far away it is, and the illuminance it
// gives here, following KHR_lights_punctual. Point and spot lights fade with
// the square of the distance, then smoothly to nothing at `range`; spot
// lights also fade from the inner cone to the outer one. A directional
// light is infinitely far away.
float3 punctual_light(FrameData *frame, Light light, float3 position, out float3 l, out float distance) {
    if (light.type == light_directional) {
        l = -light.direction;
        distance = infinite_distance;
        return light.intensity;
    }

    const float3 to_light = camera_relative(frame, light.cell, light.offset) - position;
    const float distance2 = max(dot(to_light, to_light), 1e-8);
    distance = sqrt(distance2);
    l = to_light / distance;

    float attenuation = 1.0 / distance2;

    if (light.range > 0.0) {
        const float ratio = distance / light.range;
        attenuation *= saturate(1.0 - ratio * ratio * ratio * ratio);
    }

    if (light.type == light_spot) {
        const float cone = saturate(dot(light.direction, -l) * light.spot_scale + light.spot_offset);
        attenuation *= cone * cone;
    }

    return light.intensity * attenuation;
}

// --- Image-based lighting ------------------------------------------------------

// Light from the whole sky at once, from what environment.slang prepared.

// The sky's irradiance on a surface facing `n`, in lux: its nine spherical
// harmonics coefficients, each weighted by its basis function at `n`.
float3 sky_irradiance(EnvironmentInfo *environment, float3 n) {
    const float basis[9] = {
        0.282095,
        0.488603 * n.y,
        0.488603 * n.z,
        0.488603 * n.x,
        1.092548 * n.x * n.y,
        1.092548 * n.y * n.z,
        0.315392 * (3.0 * n.z * n.z - 1.0),
        1.092548 * n.x * n.z,
        0.546274 * (n.x * n.x - n.y * n.y),
    };

    float3 irradiance = 0.0;
    for (uint k = 0; k < 9; ++k) {
        irradiance += environment.irradiance_sh[k] * basis[k];
    }

    return max(irradiance, 0.0);
}

// --- Ambient occlusion -------------------------------------------------------------

// The rotation that turns `from` into `to`, applied to `v`: Rodrigues'
// formula rewritten without angles (Moller and Hughes 1999), from the cross
// and dot products alone. `from` and `to` are never opposite here: the bent
// normal averages directions in the hemisphere around the normal.
float3 rotate_from_to(float3 from, float3 to, float3 v) {
    const float3 axis = cross(from, to);
    const float c = dot(from, to);

    if (c > 0.9999) {
        return v;
    }

    return v * c + cross(axis, v) + axis * (dot(axis, v) / (1.0 + c));
}

// Ambient occlusion counts light that's blocked, but light also bounces off
// the occluders, and more so the brighter they are. Jimenez et al.'s fit,
// from the same GTAO paper, brightens the visibility by the surface's own
// albedo, standing in for its surroundings'.
float3 multi_bounce(float visibility, float3 albedo) {
    const float3 a = 2.0404 * albedo - 0.3324;
    const float3 b = -4.7951 * albedo + 0.6417;
    const float3 c = 2.7552 * albedo + 0.6903;
    return max(float3(visibility), ((visibility * a + b) * visibility + c) * visibility);
}

// How much of the sky's reflection a partly occluded point still sees
// (Lagarde and de Rousiers 2014): smooth surfaces, looking straight on, keep
// more of it than occlusion alone suggests; rough ones lose about as much.
// Its roughness is GGX's alpha, roughness squared.
float specular_occlusion(float n_dot_v, float visibility, float alpha) {
    return saturate(pow(n_dot_v + visibility, exp2(-16.0 * alpha - 1.0)) - 1.0 + visibility);
}

// The sky's light reflected toward the viewer.
//   - Diffuse: a Lambertian surface reflects base color / pi of the
//     irradiance falling on it, read along `irradiance_normal` (the bent
//     normal: the direction the open sky lies in), dimmed by the visibility
//     and brightened again by multiple bounces.
//   - Specular, the "split sum": the light (the prefiltered sky along the
//     reflected ray, at the mip level for this roughness) times how much
//     the BRDF reflects overall (the table, as a scale and bias on F0),
//     dimmed by the specular occlusion.
//   - A roughness-aware Fresnel term splits the light between the two.
float3 shade_environment(Surface surface, FrameData *frame, float roughness, float visibility, float3 irradiance_normal) {
    const float n_dot_v = max(dot(surface.normal, surface.view), 1e-4);
    const float3 f0 = lerp(float3(0.04), surface.base_color, surface.metallic);
    const float3 fresnel = f0 + (max(float3(1.0 - roughness), f0) - f0) * pow(1.0 - n_dot_v, 5.0);

    const SamplerState clamped = SamplerState.Handle(uint2(frame.clamp_sampler, 0));

    const float3 diffuse_color = surface.base_color * (1.0 - surface.metallic);
    const float3 diffuse = (1.0 - fresnel) * diffuse_color * sky_irradiance(frame.environment, irradiance_normal) / pi
        * multi_bounce(visibility, diffuse_color);

    const TextureCube specular_cube = TextureCube.Handle(uint2(frame.specular_cube, 0));
    const float3 reflected = reflect(-surface.view, surface.normal);
    const float lod = roughness * float(frame.specular_mips - 1);
    const float3 prefiltered = specular_cube.SampleLevel(clamped, reflected, lod).rgb;

    const Texture2D brdf_lut = Texture2D.Handle(uint2(frame.brdf_lut, 0));
    const float2 brdf = brdf_lut.SampleLevel(clamped, float2(n_dot_v, roughness), 0.0).rg;
    const float3 specular = prefiltered * (f0 * brdf.x + brdf.y) * specular_occlusion(n_dot_v, visibility, surface.alpha)
        * surface.energy_compensation;

    return diffuse + specular;
}

// --- Energy compensation -------------------------------------------------------------

// The GGX specular reflection counts light that bounces once off the
// microfacets; on a rough surface, much of it bounces again, between them,
// and still leaves. Without it, rough metals come out too dark: a fully
// rough white metal reflects only about 40% of the light. The BRDF
// table's two numbers, at f0 = 1, add up to E, the share that one bounce
// leaves with; scaling the reflection by 1 + f0 (1 / E - 1) puts the rest
// back, in proportion to how much the surface reflects at all. Kulla and
// Conty's (2017) idea, in Turquin's (2019) simpler scaled form, which
// Filament uses.
float3 energy_compensation(FrameData *frame, float3 base_color, float metallic, float roughness, float n_dot_v) {
    const SamplerState clamped = SamplerState.Handle(uint2(frame.clamp_sampler, 0));
    const Texture2D brdf_lut = Texture2D.Handle(uint2(frame.brdf_lut, 0));
    const float2 brdf = brdf_lut.SampleLevel(clamped, float2(n_dot_v, roughness), 0.0).rg;
    const float3 f0 = lerp(float3(0.04), base_color, metallic);
    return 1.0 + f0 * (1.0 / max(brdf.x + brdf.y, 1e-3) - 1.0);
}

// --- Specular antialiasing -----------------------------------------------------------

// A pixel shows the average of the light reflected over its whole footprint
// on the surface, but it's shaded at one point. Where the normal changes
// faster than a pixel can follow, a highlight smaller than the pixel lands
// on the sample in one frame and misses it in the next: sparkles that crawl
// as the camera moves. That happens in two places:
//   - a surface curving, or seen from far away, faster than the pixels
//   - a normal map's bumps, smaller than a texel of the mip being read
// Both mean the normals within the pixel spread out. To the BRDF, that
// spread is roughness: alpha squared is twice the variance of the
// microfacet slopes, exactly for Beckmann's distribution and closely enough
// for GGX's (Kaplanyan et al. 2016), so the spread's variance adds to it. A
// rougher, wider highlight that every pixel catches part of.

// The variance of the pixel filter, in pixels squared, and the most the
// curvature may add: differences between neighbouring pixels estimate the
// change only roughly, and the cap keeps a bad estimate from turning a
// mirror into chalk (Kaplanyan et al. 2016; Tokuyoshi and Kaplanyan 2021).
static const float pixel_filter_variance = 0.15915494;  // 1 / (2 pi)
static const float curvature_limit = 0.18;

// GGX's alpha, widened by both spreads.
//   Curvature: how fast the vertex normal changes from pixel to pixel,
//      both ways, gives the variances of the normals across the pixel's
//      footprint along the two screen axes. Their sum bounds the largest
//      variance in any direction, so a round highlight widened by it is
//      wide enough (Tokuyoshi and Kaplanyan 2021, equation 13, their
//      conservative isotropic form). The vertex normal, as Filament and
//      Unity's HDRP use: the surface's shape. The map's bumps are the
//      other spread; their pixel-to-pixel differences would add a noisy,
//      flickering estimate on top.
//   Normal map: the spread its mips measured (Toksvig 2005, mips.slang),
//      as the standard deviation s of the normals' angle; its variance s^2,
//      twice, like the curvature's.
float antialiased_alpha(float alpha, float3 geometric_normal, float map_spread) {
    const float3 dn_dx = ddx(geometric_normal);
    const float3 dn_dy = ddy(geometric_normal);
    const float curvature = 2.0 * pixel_filter_variance * (dot(dn_dx, dn_dx) + dot(dn_dy, dn_dy));
    const float bumps = 2.0 * map_spread * map_spread;

    return sqrt(saturate(alpha * alpha + min(curvature, curvature_limit) + bumps));
}

// --- Aerial perspective --------------------------------------------------------------

// The light the air between the camera and `position` (camera-relative)
// adds, and, in `transmittance`, the share of the surface's light it lets
// through: the aerial perspective volumes, at the point's place on the
// screen and its distance. Closer than the first slice's far edge, a share
// of the first slice's air.
float3 aerial_perspective(FrameData *frame, float3 position, out float3 transmittance) {
    const float4 clip = mul(frame.view_projection, float4(position, 1.0));
    const float2 screen = clip.xy / clip.w * 0.5 + 0.5;

    float first_slice_share;
    const float depth = aerial_depth(length(position), first_slice_share);
    const float3 coordinate = float3(screen, depth);

    const SamplerState clamped = SamplerState.Handle(uint2(frame.clamp_sampler, 0));
    const float3 inscatter = Texture3D.Handle(uint2(frame.aerial_inscatter, 0)).SampleLevel(clamped, coordinate, 0.0).rgb;
    const float3 through = Texture3D.Handle(uint2(frame.aerial_transmittance, 0)).SampleLevel(clamped, coordinate, 0.0).rgb;

    transmittance = lerp(float3(1.0), through, first_slice_share);
    return inscatter * first_slice_share;
}

// --- Depth and normal prepass ------------------------------------------------------

// The interpolated vertex normal, facing the viewer on a double-sided
// material's back face; flat when the file has no normals. This is the
// surface at the scale the mesh describes it, which ambient occlusion
// searches against: a normal map's detail isn't in the depth buffer.
float3 vertex_normal(VertexOutput input, Material material, bool front_face) {
    float3 normal = input.normal;

    if (all(normal == 0.0)) {
        normal = cross(ddy(input.relative_position), ddx(input.relative_position));
    }

    normal = normalize(normal);
    return material.double_sided != 0 && !front_face ? -normal : normal;
}

// The prepass draws every opaque and masked surface first, writing only its
// depth and its vertex normal, octahedrally encoded. Masked surfaces cut out
// their transparent texels here too, so the depth buffer holds exactly the
// surfaces the lighting pass will shade.
[shader("fragment")]
float2 prepassMain(VertexOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    FrameData *frame = push.frame;
    const Material material = frame.materials[frame.draws[input.draw_index].material];

    if (alpha_mode == alpha_mask) {
        const float alpha = material.base_color_factor.a * sample_slot(material.base_color, input).a * input.color.a;
        if (alpha < material.alpha_cutoff) {
            discard;
        }
    }

    return encode_octahedral(vertex_normal(input, material, front_face));
}

// --- Shading a fragment ------------------------------------------------------

// The surface at this fragment: its exposed radiance (or one input, in a
// debug view), and its alpha. The lighting pass writes it as it is; the
// transparency pass adds it into its sums.
// `front_face`: whether this triangle faces the camera.
float4 shade_fragment(VertexOutput input, bool front_face) {
    FrameData *frame = push.frame;
    const Material material = frame.materials[frame.draws[input.draw_index].material];

    // Base color: factor x texture x vertex color. sRGB textures are decoded
    // to linear by the sampler, so all three are linear.
    float4 base_color = material.base_color_factor * sample_slot(material.base_color, input) * input.color;

    if (alpha_mode == alpha_opaque) {
        base_color.a = 1.0;
    } else if (alpha_mode == alpha_mask) {
        // Below the cutoff the pixel is cut out entirely: no color, no depth.
        if (base_color.a < material.alpha_cutoff) {
            discard;
        }
        base_color.a = 1.0;
    }

    // Metallic and roughness share one texture: blue and green.
    const float4 metallic_roughness = sample_slot(material.metallic_roughness, input);
    const float metallic = material.metallic_factor * metallic_roughness.b;
    const float roughness = material.roughness_factor * metallic_roughness.g;

    // Occlusion darkens creases that ambient light can't reach. Strength
    // blends between no effect (0) and the full map (1).
    const float occlusion = 1.0 + material.occlusion_strength * (sample_slot(material.occlusion, input).r - 1.0);

    const float3 emissive = material.emissive_factor * sample_slot(material.emissive, input).rgb;

    float map_spread;
    const float3 normal = surface_normal(input, material, front_face, frame.view != view_vertex_normal, map_spread);

    // Roughness, widened where the normals spread within the pixel. A
    // perfectly smooth surface would reflect a punctual light from a single
    // point, too small for any pixel to catch; a floor on roughness keeps
    // highlights visible.
    const float floored = max(roughness, 0.045);
    const float alpha = antialiased_alpha(floored * floored, vertex_normal(input, material, front_face), map_spread);
    const float shading_roughness = sqrt(alpha);

    // Ambient occlusion, from the AO pass's image at this pixel. It combines
    // with the occlusion map by min, not product: both estimate the same
    // thing, at two scales. The bent normal is a deflection from the vertex
    // normal; turning the shading normal by the same deflection keeps the
    // normal map's detail. See-through surfaces aren't in the prepass, so
    // the image there holds whatever is behind them: they use the map alone.
    const float4 gtao = Texture2D.Handle(uint2(frame.ambient_occlusion, 0)).Load(int3(int2(input.position.xy), 0));
    float visibility = occlusion;
    float3 irradiance_normal = normal;

    if (frame.ao_enabled != 0 && alpha_mode != alpha_blend) {
        visibility = min(occlusion, gtao.w);
        irradiance_normal = normalize(rotate_from_to(vertex_normal(input, material, front_face), gtao.xyz, normal));
    }

    // The debug views show one input each. Directions are shown as colors:
    // each component's -1..1 mapped to 0..1.
    switch (frame.view) {
        case view_base_color: return base_color;
        case view_normal:
        case view_vertex_normal: return float4(normal * 0.5 + 0.5, 1.0);
        case view_metallic: return float4(metallic.xxx, 1.0);
        case view_roughness: return float4(shading_roughness.xxx, 1.0);  // as shaded: widened
        case view_occlusion: return float4(occlusion.xxx, 1.0);
        case view_emissive: return float4(emissive, 1.0);
        case view_ambient_occlusion: return float4(gtao.www, 1.0);
        default: break;
    }

    // The triangle's flat normal, from how the position changes across
    // neighbouring pixels: exact up to rounding, since a triangle is flat.
    // Shadow rays start off the surface along it. A triangle seen exactly
    // edge-on has no area on screen, and no such normal: then the shading
    // normal stands in, rather than a division by zero.
    const float3 face_cross = cross(ddy(input.relative_position), ddx(input.relative_position));
    const float3 face_normal = dot(face_cross, face_cross) > 1e-24 ? normalize(face_cross) : normal;

    // The shadow view: how much of the sun's light gets through to each point.
    if (frame.view == view_shadow) {
        const float sun = frame.sun_direction.y > 0.0
            ? light_visibility(frame, shadow_ray_origin(frame, input.relative_position, face_normal, frame.sun_direction),
                frame.sun_direction, infinite_distance)
            : 0.0;
        return float4(sun.xxx, 1.0);
    }

    const float3 view = normalize(-input.relative_position);  // toward the camera, at the origin
    const Surface surface = {
        base_color.rgb,
        metallic,
        alpha,
        normal,
        view,
        energy_compensation(frame, base_color.rgb, metallic, shading_roughness, max(dot(normal, view), 1e-4)),
    };

    // Direct light, shadowed: the sun, then every light in the file.
    float3 radiance = shade_shadowed(surface, frame, input.relative_position, face_normal,
        frame.sun_direction, frame.sun_illuminance, infinite_distance);

    for (uint i = 0; i < frame.light_count; ++i) {
        float3 l;
        float distance;
        const float3 illuminance = punctual_light(frame, frame.lights[i], input.relative_position, l, distance);
        radiance += shade_shadowed(surface, frame, input.relative_position, face_normal, l, illuminance, distance);
    }

    // Indirect light from the sky, darkened by occlusion. Ambient occlusion
    // only ever reaches this indirect light: the sun and the lights are
    // direct, and only a shadow can block them.
    radiance += shade_environment(surface, frame, shading_roughness, visibility, irradiance_normal);

    // The air between the camera and the surface, with the simulated sky:
    // it dims the surface's light, emission included, and adds its own. The
    // photograph has no air to go with it.
    float3 transmittance = 1.0;
    float3 inscatter = 0.0;

    if (frame.atmosphere != 0) {
        inscatter = aerial_perspective(frame, input.relative_position, transmittance);
    }

    // Exposure scales nits into the tone mapper's range here, before the
    // 16-bit HDR image could overflow. glTF defines emission in nits, but, as
    // its spec notes many engines do, we take it as already exposed: an
    // emissive value of 1 shows as near-white, whatever the exposure.
    return float4((radiance * transmittance + inscatter) * frame.exposure + emissive * transmittance, base_color.a);
}

// --- Fragment shaders ----------------------------------------------------------

// The lighting pass, for opaque and masked surfaces.
// SV_Target: the value written to color attachment 0.
// SV_IsFrontFace: whether this triangle faces the camera.
[shader("fragment")]
float4 fragmentMain(VertexOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    return shade_fragment(input, front_face);
}

// --- Weighted blended transparency ---------------------------------------------

// The transparency pass, for blended surfaces (McGuire and Bavoil 2013,
// "Weighted Blended Order-Independent Transparency"). Blending one surface
// over another depends on which is in front, so blended surfaces would have
// to be sorted back to front, per pixel. Instead, every fragment adds into
// two sums, in any order:
//   accum   (premultiplied color, coverage) times a weight that falls with
//           distance, added up
//   reveal  the share of the scene that shows through: 1, times every
//           fragment's (1 - coverage)
// The composite (composite.slang) divides accum's color by its coverage,
// a weighted average of the layers, and lays it over the scene by
// 1 - reveal. One layer comes out as blending would draw it, to 16-bit
// precision; where layers overlap, the nearer one counts for more.
struct TransparentOutput {
    float4 accum : SV_Target0;
    float reveal : SV_Target1;
};

// The weight: McGuire and Bavoil's equation 7, tuned for 16-bit float sums
// and distances from 0.1 m to 500 m. Past a few hundred metres it bottoms
// out at its floor, so distant layers that overlap count equally. It falls
// steeply with the distance in front of the camera, so where layers overlap,
// the nearest dominates; the
// clamp keeps it between 1e-2 and 3e3. Colors are clamped to
// transparent_max, which tone mapping already shows as nearly white: one
// fragment then adds at most 4 x 3e3 = 12000, well below the 65504 a
// 16-bit float holds, so many layers can stack up before the sums overflow.
static const float transparent_max = 4.0;

float transparent_weight(float coverage, float view_depth) {
    const float a = view_depth / 5.0;
    const float b = view_depth / 200.0;
    return coverage * clamp(10.0 / (1e-5 + a * a + b * b * b * b * b * b), 1e-2, 3e3);
}

[shader("fragment")]
TransparentOutput transparentMain(VertexOutput input, bool front_face : SV_IsFrontFace) {
    const float4 color = shade_fragment(input, front_face);
    const float coverage = saturate(color.a);

    // The distance in front of the camera, along its view direction: the
    // clip-space w a perspective projection leaves.
    const float view_depth = mul(push.frame.view_projection, float4(input.relative_position, 1.0)).w;
    const float weight = transparent_weight(coverage, view_depth);

    TransparentOutput output;
    output.accum = float4(min(color.rgb, transparent_max) * coverage, coverage) * weight;
    output.reveal = coverage;
    return output;
}
```

## 15.5 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **Sponza** looks as it did in Chapter 14 close up. Its materials are mostly rough, so the widening barely shows: fewer than 1% of pixels change visibly.
- **The roughness view** (key 6): curved surfaces get lighter toward their silhouettes, where the normal turns fastest, and normal-mapped surfaces get lighter with distance, as their mips flatten their bumps.
- **Load `MetalRoughSpheres`** (change `scene_file` in `main.cpp`) and back away about 100 m, until each sphere is a few pixels wide. Moving slowly, the smooth spheres' highlights flicker much less than in Chapter 14. Measured over small camera steps, each highlight's frame-to-frame change in brightness drops to about 40% of what it was at 100 m, and to 60% at 200 m. Their highlights are a little wider, which is the price.
- **Sponza's plants,** seen from the far end of the courtyard, keep their leaves. Their textures' coverage now holds down the levels: one of them keeps 44% of its texels at 8 × 8 and at 4 × 4, where plain averaging kept 33% and 19%.
- **No `[validation …]` lines.**

Next, in Chapter 16, clustered lights: the view cut into a grid of clusters, each lit by only the lights that reach it, so a scene can hold thousands of lights.

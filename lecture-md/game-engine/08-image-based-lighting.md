# Chapter 8: Image-based lighting

By the end of this chapter the scene is lit by the whole sky around it, and the sky shows behind it. There are two skies to choose from with the `E` key:
- **a simulated atmosphere,** for any time of day: deep blue at noon, a reddened sun and an orange horizon at sunset, a pink-and-purple twilight, then a black night.
- **a photographed sky,** Poly Haven's `kloppenheim_06_puresky`, a real evening sky stored as an HDR image.

Every surface now reflects its surroundings. Metal spheres mirror the horizon and blur it as they get rougher, and the diffuse light on a wall comes from whichever part of the sky it faces. This is **image-based lighting** (IBL): the sky is treated as an image of light, which the GPU preprocesses into the two forms shading needs. Chapter 7's two-tone sky goes away.

The pieces, all computed on the GPU:
1. **a sky cube map,** 512 × 512 texels per face: the atmosphere rendered into it, or the photograph projected onto it,
2. **its diffuse light:** nine spherical-harmonics coefficients,
3. **its specular light:** a smaller cube map, blurred more for each mip level, one level per roughness,
4. **a BRDF table,** computed once, which scales the specular light.

This chapter builds on [Chapter 7](07-light-exposure-hdr.md).

## 8.1 The photographed sky: `assets/environments`, `radiance_hdr.h`, `radiance_hdr.cpp`

### Why
A photographed sky captures light no simple model produces: clouds, haze, the glow around a hidden sun. HDR sky photographs are made by combining many exposures of a full panorama, and are usually published as Radiance `.hdr` files. Poly Haven's skies are free (CC0), and its "puresky" versions have the ground removed, which suits a scene with its own floor.

### How
- **The file** is `kloppenheim_06_puresky_2k.hdr`, 2048 × 1024 pixels and 4.4 MB: a cloudy evening sky with the sun low and hidden in haze. Its brightest pixel is only about 60 times the average, so there's no hard sun to deal with: all of its light can come from the image.
- **The layout is equirectangular:** longitude runs across the image and latitude down it, like a world map. The top row is straight up, the bottom row straight down.
- **The Radiance format** is simple enough to read ourselves:
  - **The header:** text lines, ending at an empty line, then the size line `-Y 1024 +X 2048`, which means 1024 rows from the top, of 2048 pixels each.
  - **RGBE pixels:** each pixel is four bytes: three 8-bit mantissas sharing one 8-bit exponent, so `value = mantissa × 2^(E − 136)`. That spans a huge range in 32 bits, which is exactly what light needs.
  - **Run-length encoding:** each row starts with the bytes 2, 2 and its width. Then come the row's red bytes, its green bytes, its blue bytes and its exponents, one channel after another. Within a channel, a count above 128 repeats the next byte `count − 128` times, and a smaller count copies that many bytes as they are.
- **The loader** checks every count against the row's width and the file's size, so a damaged file throws instead of reading past the end.

### Code
Download `kloppenheim_06_puresky` from <https://polyhaven.com/a/kloppenheim_06_puresky> as **HDR, 2K**, and save it in a new directory as `lecture-md/game-engine/assets/environments/kloppenheim_06_puresky_2k.hdr`. If you already have Poly Haven skies locally, copy the file there instead.

`game-engine/src/includes/radiance_hdr.h`:
```cpp
#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

// An image of light: three floats per pixel, red, green and blue, row by row from the top. The values are radiance, not colors, so they go far past 1.
struct HdrImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> rgb;
};

// Reads a Radiance .hdr file (RGBE, run-length encoded), the format HDR sky images are usually published in. Throws if the file isn't one.
HdrImage load_radiance_hdr(const std::filesystem::path &path);
```

`game-engine/src/radiance_hdr.cpp`:
```cpp
#include "includes/radiance_hdr.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

// The format

// A Radiance file is a text header, a blank line, a size line such as "-Y 1024 +X 2048" (1024 rows from the top, 2048 pixels each, left to right), then the pixels. Each pixel is four bytes, RGBE: three mantissas sharing one exponent, so value = mantissa * 2^(E - 128) / 256. That spans an enormous range in 32 bits, which is what light needs.
//
// Each row is usually run-length encoded: a 4-byte row header (2, 2, then the width as a 16-bit number), then the row's R bytes, its G bytes, its B bytes and its E bytes, one channel after another. Within a channel, a count byte above 128 means "repeat the next byte count - 128 times", and a count of 128 or less means "copy the next count bytes as they are".

namespace {

    // Reads one line of the header, without its '\n'.
    std::string read_line(const std::vector<unsigned char> &bytes, std::size_t &at) {
        std::string line;

        while (at < bytes.size() && bytes[at] != '\n') {
            line += static_cast<char>(bytes[at++]);
        }

        ++at;  // the '\n'
        return line;
    }

    // Decodes one run-length encoded channel of a row into `channel`, `width` bytes.
    void read_channel(const std::vector<unsigned char> &bytes, std::size_t &at, std::uint32_t width, unsigned char *channel) {
        std::uint32_t x = 0;

        while (x < width) {
            if (at >= bytes.size()) {
                throw std::runtime_error("the file ends in the middle of a row");
            }

            std::uint32_t count = bytes[at++];
            const bool run = count > 128;
            count = run ? count - 128 : count;

            if (count == 0 || x + count > width || at + (run ? 1 : count) > bytes.size()) {
                throw std::runtime_error("a run doesn't fit in its row");
            }

            for (std::uint32_t i = 0; i < count; ++i) {
                channel[x++] = run ? bytes[at] : bytes[at + i];
            }

            at += run ? 1 : count;
        }
    }

}  // namespace

// Loading

HdrImage load_radiance_hdr(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("can't open " + path.string());
    }

    const std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    std::size_t at = 0;

    // The header: "#?RADIANCE" (or "#?RGBE"), then lines like FORMAT=32-bit_rle_rgbe, up to an empty line.
    if (!read_line(bytes, at).starts_with("#?")) {
        throw std::runtime_error(path.string() + " isn't a Radiance .hdr file");
    }

    while (at < bytes.size() && !read_line(bytes, at).empty()) {
    }

    HdrImage image;
    const std::string size = read_line(bytes, at);

    if (std::sscanf(size.c_str(), "-Y %u +X %u", &image.height, &image.width) != 2) {
        throw std::runtime_error(path.string() + ": only top-to-bottom, left-to-right images are supported");
    }

    image.rgb.resize(static_cast<std::size_t>(image.width) * image.height * 3);
    std::vector<unsigned char> row(static_cast<std::size_t>(image.width) * 4);

    for (std::uint32_t y = 0; y < image.height; ++y) {
        // Every row starts 2, 2, width (high byte, low byte). Older flat or differently encoded files exist, but sky images aren't written that way.
        if (at + 4 > bytes.size() || bytes[at] != 2 || bytes[at + 1] != 2
            || ((bytes[at + 2] << 8) | bytes[at + 3]) != static_cast<int>(image.width)) {
            throw std::runtime_error(path.string() + ": row " + std::to_string(y) + " isn't run-length encoded");
        }

        at += 4;

        // The channels arrive one after another; store them side by side.
        for (std::uint32_t channel = 0; channel < 4; ++channel) {
            read_channel(bytes, at, image.width, row.data() + static_cast<std::size_t>(channel) * image.width);
        }

        for (std::uint32_t x = 0; x < image.width; ++x) {
            const int exponent = row[3 * image.width + x];

            // ldexp(m, e) is m * 2^e. An exponent byte of 0 means black.
            const float scale = exponent == 0 ? 0.0f : std::ldexp(1.0f, exponent - 128 - 8);
            float *pixel = &image.rgb[(static_cast<std::size_t>(y) * image.width + x) * 3];

            for (std::uint32_t c = 0; c < 3; ++c) {
                pixel[c] = static_cast<float>(row[c * image.width + x]) * scale;
            }
        }
    }

    return image;
}
```

## 8.2 The environment's data: `shader_types.h`

### Why
The scene shader now needs the environment's heap slots and the sky's diffuse light. The sky background needs to turn pixels back into view directions. The compute shaders need their own push data.

### How
- **`EnvironmentInfo`:** what the compute shaders work out about the sky, in host-visible memory that the GPU writes and both the CPU and the scene shader read.
  - **`irradiance_sh`:** the sky's diffuse light as nine spherical-harmonics coefficients per color channel (8.4).
  - **`sun_illuminance`:** the sun's light at the ground, after the atmosphere. It's 0 when the sun is down, or when the sky is the photograph, whose sun is part of the image.
- **`FrameData` grows to 240 bytes:**
  - **The inverse view-projection matrix:** used by the sky background.
  - **The environment's address,** right after the other pointers.
  - **Slots:** the heap slots of the sky cube, the specular cube and the BRDF table, plus the clamp sampler's index and the specular cube's mip count.
  - **The sun's angular radius,** for drawing its disk.
  - **No padding:** the 4-byte members pair up exactly behind the 8-byte pointers.
  - **What's gone:** chapter 7's sky and ground colors.
- **`EnvironmentPushData`:** one push data struct for all five compute shaders, 48 bytes. Push data uses the std430 rules, where a `vec3` must start on a 16-byte boundary. The pointer and two slot indices fill the first 16 bytes, so `sun_direction` lands on one with no padding. Each dispatch fills in only what its shader reads, so every member has a default.

### Code
`game-engine/src/includes/shader_types.h`:
```cpp
#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

// C++ mirrors of the structs in the shaders (shaders/*.slang). The GPU reads these bytes as they are, so the two sides must agree on every size and offset; the static_asserts catch a mismatch at compile time.

// Vertex

// Slang lays out data behind a pointer like C: each member aligned only to the size of its scalar type. Every member here is made of 4-byte floats, so nothing needs padding, and glm agrees member for member. All of these structs are packed tight like this: no padding anywhere.
//   - A normal of (0, 0, 0) means the file had none (see mesh.slang).
//   - A tangent of (0, 0, 0, 0) means the file had none; the shader then works the tangent out from the texture coordinates.
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

// Per-draw data

// One per draw, in a GPU buffer the shader indexes.
struct DrawData {
    glm::mat4 model;          // this primitive's space -> world space
    glm::mat4 normal_matrix;  // transposed inverse of model: keeps normals perpendicular under any scale
    std::uint32_t material;   // index into the material buffer
};

static_assert(sizeof(DrawData) == 132);
static_assert(offsetof(DrawData, normal_matrix) == 64);
static_assert(offsetof(DrawData, material) == 128);

// Materials

// glTF's three ways of using a material's alpha. Each gets its own pipeline, and the shader reads the mode as a specialization constant.
enum class AlphaMode : std::uint32_t {
    opaque,  // alpha is ignored
    mask,    // fully opaque or fully transparent: cut out below alpha_cutoff
    blend,   // see-through: blended over what's behind it
};

// Which texture a material slot samples, with which sampler and which set of texture coordinates. Heap indices: texture 0 is a 1x1 white texture and sampler 0 the default sampler, for slots the file leaves empty.
struct TextureSlot {
    std::uint32_t texture = 0;  // resource heap index
    std::uint32_t sampler = 0;  // sampler heap index
    std::uint32_t uv_set = 0;   // 0: TEXCOORD_0, 1: TEXCOORD_1
};

static_assert(sizeof(TextureSlot) == 12);

// A glTF metallic-roughness material: every factor and texture of the core spec. Each texture is multiplied by its factor; see mesh.slang for how.
struct Material {
    glm::vec4 base_color_factor;     // linear RGBA
    glm::vec3 emissive_factor;       // linear RGB light the surface gives off
    float metallic_factor;           // 1: metal, 0: not
    float roughness_factor;          // 1: fully rough, 0: mirror smooth
    float normal_scale;              // how strongly the normal map tilts the normal
    float occlusion_strength;        // 0: ignore the occlusion map, 1: use it fully
    float alpha_cutoff;              // AlphaMode::mask: alpha below this is cut out
    std::uint32_t double_sided;      // 1: both sides are drawn and lit
    TextureSlot base_color;          // RGBA, sRGB
    TextureSlot metallic_roughness;  // G: roughness, B: metallic
    TextureSlot normal;              // tangent-space normal
    TextureSlot occlusion;           // R: how much ambient light reaches the surface
    TextureSlot emissive;            // RGB, sRGB
};

static_assert(sizeof(Material) == 112);
static_assert(offsetof(Material, emissive_factor) == 16);
static_assert(offsetof(Material, metallic_factor) == 28);
static_assert(offsetof(Material, double_sided) == 48);
static_assert(offsetof(Material, base_color) == 52);
static_assert(offsetof(Material, emissive) == 100);

// Lights

// KHR_lights_punctual's three kinds of light. "Punctual" means infinitely small: all of a light's power comes from one point, or one direction.
enum class LightType : std::uint32_t {
    directional,  // like the sun: parallel rays, intensity in lux
    point,        // shines in every direction, intensity in candela
    spot,         // a point light limited to a cone, intensity in candela
};

// One light from the file, in world space.
struct Light {
    glm::vec3 position;   // point and spot lights
    float range;          // distance where the light fades to nothing; 0 for no limit
    glm::vec3 direction;  // spot and directional lights: the way the light shines
    float spot_scale;     // spot cone falloff: 1 / (cos(inner) - cos(outer))
    glm::vec3 intensity;  // color times intensity
    float spot_offset;    // spot cone falloff: -cos(outer) * spot_scale
    LightType type;
};

static_assert(sizeof(Light) == 52);
static_assert(offsetof(Light, direction) == 16);
static_assert(offsetof(Light, intensity) == 32);
static_assert(offsetof(Light, type) == 48);

// Views

// What the fragment shader outputs: the shaded scene, or one material input on its own, for checking that each one loaded correctly. Keys 1-8 pick one.
enum class View : std::uint32_t {
    lit,
    base_color,
    normal,         // the final normal, normal map included
    vertex_normal,  // the interpolated vertex normal, without the normal map
    metallic,
    roughness,
    occlusion,
    emissive,
};

// The environment

// What the environment's compute shaders tell the CPU and the scene shader about the sky, in host-visible memory both can read.
//   - irradiance_sh: the light falling on a surface from the whole sky, as 9 spherical harmonics coefficients per color channel (see environment.slang).
//   - sun_illuminance: the sun's light at the ground after the atmosphere, in lux on a surface facing it; 0 when the sun is down or the sky is an image.
struct EnvironmentInfo {
    std::array<glm::vec3, 9> irradiance_sh;
    glm::vec3 sun_illuminance;
};

static_assert(sizeof(EnvironmentInfo) == 120);

// Per-frame data

// Everything the shaders need that's the same for every draw in a frame. Each frame in flight has its own copy in host-visible memory, rewritten by the CPU before the frame is recorded. Push data points at it.
//   - Lighting values are physical: lux for illuminance, nits (candela per square meter) for the brightness of the sky.
//   - Pointers come right after the matrices, so all of them land on 8-byte boundaries with no padding.
struct FrameData {
    glm::mat4 view_projection;          // world space -> clip space
    glm::mat4 inverse_view_projection;  // clip space -> world space, for the sky's view directions
    vk::DeviceAddress vertices;         // the scene's vertices
    vk::DeviceAddress draws;            // one DrawData per draw
    vk::DeviceAddress materials;        // the scene's materials
    vk::DeviceAddress lights;           // the file's lights
    vk::DeviceAddress environment;      // the EnvironmentInfo
    glm::vec3 camera_position;          // world space
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
};

static_assert(sizeof(FrameData) == 240);
static_assert(offsetof(FrameData, vertices) == 128);
static_assert(offsetof(FrameData, environment) == 160);
static_assert(offsetof(FrameData, camera_position) == 168);
static_assert(offsetof(FrameData, sun_direction) == 184);
static_assert(offsetof(FrameData, sun_illuminance) == 200);
static_assert(offsetof(FrameData, sky_cube) == 216);
static_assert(offsetof(FrameData, sun_angular_radius) == 236);

// Push data

// Written with vkCmdPushDataEXT before each draw: where this frame's data is, and which DrawData this draw uses. That's 12 bytes of data; the struct is 16, because a struct with an 8-byte member is padded to a multiple of 8. It's the one struct here with padding, and the shader just ignores it.
struct PushData {
    vk::DeviceAddress frame;
    std::uint32_t draw_index;
};

static_assert(offsetof(PushData, draw_index) == 8);
static_assert(sizeof(PushData) == 16);

// The tone-mapping pass's push data: which resource heap slot holds the HDR image, and the view, so material views can skip tone mapping.
struct TonemapPushData {
    std::uint32_t hdr_image;
    View view;
};

// The environment compute shaders' push data. Push data follows std430 rules, where a vec3 starts on a 16-byte boundary: the pointer and two indices fill the first 16 bytes, so sun_direction lands on one. Each dispatch sets only what its shader reads, so every member has a default.
struct EnvironmentPushData {
    vk::DeviceAddress info = 0;     // where the EnvironmentInfo goes
    std::uint32_t source = 0;       // resource heap slot to read
    std::uint32_t target = 0;       // resource heap slot to write (a storage image)
    glm::vec3 sun_direction{0.0f};  // the atmosphere's sun
    float source_scale = 0.0f;      // the HDR image's brightness: nits per stored 1.0
    std::uint32_t size = 0;         // the target's width and height in texels
    float roughness = 0.0f;         // prefiltering: the roughness this mip level is for
    std::uint32_t sampler = 0;      // sampler heap index: the clamp sampler
    std::uint32_t source_size = 0;  // the source cube's face size at mip 0
};

static_assert(sizeof(EnvironmentPushData) == 48);
static_assert(offsetof(EnvironmentPushData, sun_direction) == 16);
static_assert(offsetof(EnvironmentPushData, size) == 32);
```

## 8.3 Storage images and a clamp sampler: `descriptor_heap.h`, `descriptor_heap.cpp`

### Why
Compute shaders write the environment's images, and writing an image from a shader needs a **storage image** descriptor, not a sampled one. Two of the new images also mustn't wrap around at their edges: the BRDF table, and the photograph at its poles.

### How
- **`write_image_descriptor`** takes a descriptor type: sampled, or storage.
  - **Layouts:** a storage image is used in the `eGeneral` layout, the one layout that allows shader writes, so its descriptor names that layout.
  - **Formats:** Slang declares storage images without a format. Vulkan 1.3 allows that for formats that report support for it, and the validation layer checks this GPU's 16-bit float RGBA does.
- **The clamp sampler** is added after the scene's samplers, and `DescriptorHeaps::clamp_sampler` says where. It's a `SceneSampler` set to glTF's `CLAMP_TO_EDGE` in both directions, so `sampler_info` builds it like any other.

### Code
`game-engine/src/includes/descriptor_heap.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/texture.h"

#include <cstddef>
#include <cstdint>
#include <span>

// The two descriptor heaps shaders read: a resource heap of image descriptors and a sampler heap. Each is a plain buffer of descriptor bytes that we write ourselves, ending in a range reserved for the driver.
//
// The resource heap stays mapped in host-visible memory, so a descriptor can be written into it whenever the GPU isn't reading that slot: the HDR image's, for instance, after every resize. The sampler heap never changes, so it's uploaded once to device-local memory.
struct DescriptorHeaps {
    Buffer resources;
    std::byte *resource_bytes = nullptr;       // the resource heap, mapped
    vk::DeviceSize image_descriptor_size = 0;  // bytes per slot
    std::uint32_t resource_slots = 0;          // slots before the reserved range
    vk::DeviceSize resource_reserved_offset = 0;
    vk::DeviceSize resource_reserved_size = 0;

    Buffer samplers;
    vk::DeviceSize sampler_reserved_offset = 0;
    vk::DeviceSize sampler_reserved_size = 0;
    std::uint32_t clamp_sampler = 0;  // trilinear, clamped to the edge: the last sampler
};

// Creates both heaps.
//   - Resource heap: texture i at slot i, then `extra_slots` empty slots, which the program fills itself with write_image_descriptor.
//   - Sampler heap: index 0 is a default sampler; scene sampler i is at index i + 1; after them comes `clamp_sampler`, for images that mustn't wrap around at their edges.
DescriptorHeaps create_descriptor_heaps(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::span<const Texture> textures,
    std::span<const SceneSampler> samplers,
    std::uint32_t extra_slots
);

// Writes resource heap slot `slot`: an image described by `view`, either
//   - eSampledImage: read by shaders as a Texture2D, TextureCube, ... handle, in eShaderReadOnlyOptimal layout, or
//   - eStorageImage: written by compute shaders as a RWTexture2D, ... handle, in eGeneral layout.
// The GPU must not be reading the slot while it's written.
void write_image_descriptor(
    const vk::raii::Device &device,
    DescriptorHeaps &heaps,
    std::uint32_t slot,
    const vk::ImageViewCreateInfo &view,
    vk::DescriptorType type = vk::DescriptorType::eSampledImage
);

// Makes `heaps` the ones shaders read for the rest of `commands`.
void bind_descriptor_heaps(const vk::raii::CommandBuffer &commands, const DescriptorHeaps &heaps);
```

`game-engine/src/descriptor_heap.cpp`:
```cpp
#include "includes/descriptor_heap.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    vk::DeviceSize align_up(vk::DeviceSize value, vk::DeviceSize alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    // Heaps must start at a multiple of `alignment` in GPU memory.
    void check_heap_alignment(const Buffer &heap, vk::DeviceSize alignment) {
        if (heap.address % alignment != 0) {
            throw std::runtime_error("a descriptor heap's address isn't aligned to " + std::to_string(alignment) + " bytes");
        }
    }

    // Memory for the resource heap: the CPU writes it, the GPU reads it. Memory that's both device-local and host-visible (resizable BAR) is fastest for the GPU to read; without it, plain host-visible memory works too.
    vk::MemoryPropertyFlags heap_memory(const GpuChoice &gpu) {
        const vk::MemoryPropertyFlags best = vk::MemoryPropertyFlagBits::eDeviceLocal
            | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        const vk::PhysicalDeviceMemoryProperties memory = gpu.device.getMemoryProperties();

        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if ((memory.memoryTypes[i].propertyFlags & best) == best) {
                return best;
            }
        }

        return vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    }

    // Samplers

    // glTF's sampler settings as Vulkan sampler state. glTF uses OpenGL's enums, whose minifying filters name two things at once: how to filter within a mip level, and how to pick between levels. Filters the file leaves out get our best quality: trilinear and anisotropic.
    vk::SamplerCreateInfo sampler_info(const SceneSampler &sampler, float max_anisotropy) {
        constexpr int nearest = 9728;
        constexpr int linear = 9729;
        constexpr int nearest_mipmap_nearest = 9984;
        constexpr int linear_mipmap_nearest = 9985;
        constexpr int nearest_mipmap_linear = 9986;
        constexpr int clamp_to_edge = 33071;
        constexpr int mirrored_repeat = 33648;

        const auto address_mode = [](int wrap) {
            switch (wrap) {
                case clamp_to_edge: return vk::SamplerAddressMode::eClampToEdge;
                case mirrored_repeat: return vk::SamplerAddressMode::eMirroredRepeat;
                default: return vk::SamplerAddressMode::eRepeat;
            }
        };

        const int min = sampler.min_filter;
        const bool min_nearest = min == nearest || min == nearest_mipmap_nearest || min == nearest_mipmap_linear;

        // Plain NEAREST and LINEAR don't use mipmaps at all, only level 0. Vulkan has no such filter; the spec's recipe is nearest mipmapping with maxLod 0.25, so level 0 is always picked but the minifying filter is still used.
        const bool no_mipmaps = min == nearest || min == linear;
        const bool between_levels_nearest = no_mipmaps || min == nearest_mipmap_nearest || min == linear_mipmap_nearest;

        // Anisotropic filtering only makes sense on top of full trilinear filtering.
        const bool trilinear = min == -1 || min == 9987;  // 9987: LINEAR_MIPMAP_LINEAR

        return vk::SamplerCreateInfo{
            .magFilter = sampler.mag_filter == nearest ? vk::Filter::eNearest : vk::Filter::eLinear,
            .minFilter = min_nearest ? vk::Filter::eNearest : vk::Filter::eLinear,
            .mipmapMode = between_levels_nearest ? vk::SamplerMipmapMode::eNearest : vk::SamplerMipmapMode::eLinear,
            .addressModeU = address_mode(sampler.wrap_s),
            .addressModeV = address_mode(sampler.wrap_t),
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .anisotropyEnable = trilinear && sampler.mag_filter != nearest ? vk::True : vk::False,
            .maxAnisotropy = max_anisotropy,
            .maxLod = no_mipmaps ? 0.25f : vk::LodClampNone,
        };
    }

}  // namespace

// Writing descriptors

void write_image_descriptor(
    const vk::raii::Device &device,
    DescriptorHeaps &heaps,
    std::uint32_t slot,
    const vk::ImageViewCreateInfo &view,
    vk::DescriptorType type
) {
    if (slot >= heaps.resource_slots) {
        throw std::runtime_error("resource heap slot " + std::to_string(slot) + " is past the last one");
    }

    // A descriptor is written from a description of the image view, so no VkImageView object is needed. It also names the layout the image will be in while shaders use it: storage images are written in eGeneral.
    const vk::ImageDescriptorInfoEXT image{
        .pView = &view,
        .layout = type == vk::DescriptorType::eStorageImage
            ? vk::ImageLayout::eGeneral
            : vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    const vk::ResourceDescriptorInfoEXT descriptor{
        .type = type,
        .data = {.pImage = &image},
    };

    // Slot i sits at i * imageDescriptorSize, which is where a shader's Texture2D.Handle(i) reads it. The heap is host-coherent, so the GPU sees the bytes without a flush.
    const vk::HostAddressRangeEXT destination{
        .address = heaps.resource_bytes + slot * heaps.image_descriptor_size,
        .size = heaps.image_descriptor_size,
    };

    device.writeResourceDescriptorsEXT(descriptor, destination);
}

// Creating the heaps

DescriptorHeaps create_descriptor_heaps(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::span<const Texture> textures,
    std::span<const SceneSampler> samplers,
    std::uint32_t extra_slots
) {
    const auto properties = gpu.device.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceDescriptorHeapPropertiesEXT
    >();
    const auto &heap = properties.get<vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();
    const auto &limits = properties.get<vk::PhysicalDeviceProperties2>().properties.limits;

    DescriptorHeaps heaps;

    // Resource heap: the slots, then the driver's reserved range.
    heaps.image_descriptor_size = heap.imageDescriptorSize;
    heaps.resource_slots = static_cast<std::uint32_t>(textures.size()) + extra_slots;
    heaps.resource_reserved_offset = align_up(heap.imageDescriptorSize * heaps.resource_slots, heap.imageDescriptorAlignment);
    heaps.resource_reserved_size = heap.minResourceHeapReservedRange;

    const vk::DeviceSize resource_size =
        align_up(heaps.resource_reserved_offset + heaps.resource_reserved_size, heap.resourceHeapAlignment);

    heaps.resources = create_buffer(device, gpu, resource_size,
        vk::BufferUsageFlagBits::eDescriptorHeapEXT | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        heap_memory(gpu));
    check_heap_alignment(heaps.resources, heap.resourceHeapAlignment);

    // Mapped once, for as long as the buffer lives: freeing the memory unmaps it.
    heaps.resource_bytes = static_cast<std::byte*>(heaps.resources.memory.mapMemory(0, resource_size));

    for (std::uint32_t i = 0; i < textures.size(); ++i) {
        write_image_descriptor(device, heaps, i, vk::ImageViewCreateInfo{
            .image = *textures[i].handle,
            .viewType = vk::ImageViewType::e2D,
            .format = textures[i].format,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = textures[i].mip_levels,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        });
    }

    // Sampler heap: the default sampler at index 0, then one per glTF sampler, then the clamp sampler. The default is a SceneSampler with nothing set: trilinear, anisotropic and repeating.
    std::vector<vk::SamplerCreateInfo> sampler_infos{sampler_info(SceneSampler{}, limits.maxSamplerAnisotropy)};

    for (const SceneSampler &sampler : samplers) {
        sampler_infos.push_back(sampler_info(sampler, limits.maxSamplerAnisotropy));
    }

    // glTF's CLAMP_TO_EDGE (33071) in both directions, every other setting left to the default.
    heaps.clamp_sampler = static_cast<std::uint32_t>(sampler_infos.size());
    sampler_infos.push_back(sampler_info(SceneSampler{.wrap_s = 33071, .wrap_t = 33071}, limits.maxSamplerAnisotropy));

    heaps.sampler_reserved_offset = align_up(heap.samplerDescriptorSize * sampler_infos.size(), heap.samplerDescriptorAlignment);
    heaps.sampler_reserved_size = heap.minSamplerHeapReservedRange;

    std::vector<std::byte> sampler_bytes(
        align_up(heaps.sampler_reserved_offset + heaps.sampler_reserved_size, heap.samplerHeapAlignment));

    std::vector<vk::HostAddressRangeEXT> sampler_destinations(sampler_infos.size());

    for (std::size_t i = 0; i < sampler_infos.size(); ++i) {
        sampler_destinations[i] = vk::HostAddressRangeEXT{
            .address = sampler_bytes.data() + i * heap.samplerDescriptorSize,
            .size = heap.samplerDescriptorSize,
        };
    }

    device.writeSamplerDescriptorsEXT(sampler_infos, sampler_destinations);
    heaps.samplers = upload_buffer(device, gpu, queue, pool, sampler_bytes,
        vk::BufferUsageFlagBits::eDescriptorHeapEXT | vk::BufferUsageFlagBits::eShaderDeviceAddress);
    check_heap_alignment(heaps.samplers, heap.samplerHeapAlignment);

    return heaps;
}

// Binding

void bind_descriptor_heaps(const vk::raii::CommandBuffer &commands, const DescriptorHeaps &heaps) {
    commands.bindResourceHeapEXT(vk::BindHeapInfoEXT{
        .heapRange = {.address = heaps.resources.address, .size = heaps.resources.size},
        .reservedRangeOffset = heaps.resource_reserved_offset,
        .reservedRangeSize = heaps.resource_reserved_size,
    });

    commands.bindSamplerHeapEXT(vk::BindHeapInfoEXT{
        .heapRange = {.address = heaps.samplers.address, .size = heaps.samplers.size},
        .reservedRangeOffset = heaps.sampler_reserved_offset,
        .reservedRangeSize = heaps.sampler_reserved_size,
    });
}
```

## 8.4 The environment shaders: `shared.slangh`, `CMakeLists.txt`, `environment.slang`

### Why
Everything that turns a sky into light runs on the GPU, as five compute shaders in one file. They also need to share structs with the scene's shaders, which until now each declared their own copy.

### How
- **A shared header:**
  - **`shared.slangh`** holds the structs `mesh.slang` and the new `background.slang` both use, and both `#include` it.
  - **Building it:** CMake compiles only `.slang` files, so the header is never compiled on its own. A new glob collects the `.slangh` files, and every shader depends on them, so editing one recompiles the shaders.
- **Compute shaders** run one thread per texel, in 8 × 8 workgroups. A cube map is written as a 2D array of six layers, with the dispatch's z as the face. `cube_direction` turns a texel into its direction, following the face table in the Vulkan spec.
- **`skyMain`: the atmosphere.** Single scattering in a spherical atmosphere, with Earth's values as Bruneton's and Hillaire's atmosphere models use them:
  - **Air molecules** (Rayleigh scattering) scatter blue about six times more than red. That gives the blue sky and the red sunset.
  - **Haze** (Mie scattering) scatters every color alike, and mostly forward, which gives the bright glow around the sun.
  - **Ozone** absorbs red and green, which keeps the twilight sky blue and purple instead of grey.
  - **The march:** each texel's view ray is marched in 32 steps to the top of the atmosphere, or to the ground. At each step, sunlight arrives dimmed by the path from the sun (8 more steps), is scattered toward the camera by the air and the haze, and is dimmed again on the way back. Each scattering type has its own phase function, which says how much light goes each way.
  - **What's left out:** the sun's disk isn't in the cube. The scene gets direct sunlight from the sun as a light, and the background draws the disk.
  - **Single scattering only:** light scattered more than once is left out, so the sky is somewhat darker than a real one. The model also has no moon or stars, so the night is black.
  - **The sun's light:** thread 0 also works out the sunlight reaching the camera, `sun_illuminance`, for the scene's sun.
- **`equirectMain`: the photograph.**
  - **The mapping:** it projects the equirectangular image onto the cube, with longitude from `atan2` and latitude from `acos`.
  - **Units:** the file's values are relative, so `source_scale` turns them into nits.
  - **Clamping:** scaled to nits, this sky's brightest texels pass 65,504, the largest 16-bit float. What a GPU stores for a value too large isn't defined, and an infinity would spread through the mips and the lighting, so both sky shaders clamp to it.
- **`irradianceMain`: diffuse light.**
  - **Why so few numbers:** the light a diffuse surface receives changes slowly with its normal. Nine spherical harmonics, the 3D equivalent of the first few terms of a Fourier series, capture it to within a few percent (Ramamoorthi and Hanrahan, 2001).
  - **The projection:** one workgroup of 64 threads reads the sky at the mip that's 32 texels across, weights each texel by the solid angle it covers, and sums the nine products.
  - **The cosine lobe:** convolving with the cosine lobe a diffuse surface sees is then just one factor per band: π, 2π/3 and π/4.
- **`prefilterMain`: specular light, the "split sum" approximation (Karis, 2013).**
  - **The split:** the full integral of sky light times the BRDF is split into two parts, which are precomputed separately: the sky's light, blurred by the GGX lobe, here, and the BRDF's total, in the table below.
  - **One mip per roughness:** each mip level of the specular cube holds the blur for one roughness, from 0 at the top level (a mirror) to 1 at the bottom.
  - **The samples:** 128 directions per texel, chosen where GGX reflects most, using a low-discrepancy Hammersley sequence.
  - **Filtered importance sampling** (Křivánek and Colbert, 2008): each sample reads the sky at a mip level matching the solid angle it stands for, so 128 samples cover the lobe with little visible noise.
- **`brdfLutMain`: the BRDF table.** For each view angle (across) and roughness (down), it stores how much the specular BRDF reflects in total, as a scale and a bias on F0, the reflectance head-on: `reflected = F0 × scale + bias`. It uses the same GGX and Smith functions as `mesh.slang`, and it never changes, so it's computed once at startup.

### Code
`game-engine/shaders/shared.slangh`:
```slang
// The structs every scene shader shares with C++ (src/includes/shader_types.h), included by mesh.slang and background.slang. A .slangh file isn't compiled on its own: CMakeLists.txt only compiles .slang files.

// Data behind a pointer is laid out like C: each member aligned only to the size of its scalar type. Every member here is made of 4-byte floats, so there's no padding, and this matches the C++ Vertex exactly (72 bytes). The other structs follow the same rule and match theirs.
struct Vertex {
    float3 position;
    float3 normal;   // (0, 0, 0) when the file had no normals
    float4 tangent;  // (0, 0, 0, 0) when the file had no tangents
    float2 uv0;
    float2 uv1;
    float4 color;
};

struct DrawData {
    float4x4 model;          // this primitive's space -> world space
    float4x4 normal_matrix;  // transposed inverse of model
    uint material;           // index into the materials
};

// Which texture, sampler and texture coordinates a material slot uses.
struct TextureSlot {
    uint texture;  // resource heap index; 0 is plain white
    uint sampler;  // sampler heap index; 0 is the default sampler
    uint uv_set;   // 0: TEXCOORD_0, 1: TEXCOORD_1
};

struct Material {
    float4 base_color_factor;
    float3 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    float occlusion_strength;
    float alpha_cutoff;
    uint double_sided;
    TextureSlot base_color;          // RGBA
    TextureSlot metallic_roughness;  // G: roughness, B: metallic
    TextureSlot normal;              // tangent-space normal
    TextureSlot occlusion;           // R
    TextureSlot emissive;            // RGB
};

// KHR_lights_punctual light types (LightType in C++).
static const uint light_directional = 0;
static const uint light_point = 1;
static const uint light_spot = 2;

struct Light {
    float3 position;
    float range;       // 0: no limit
    float3 direction;  // the way the light shines
    float spot_scale;
    float3 intensity;  // lux (directional) or candela (point, spot), per channel
    float spot_offset;
    uint type;
};

// View: what the fragment shader outputs (keys 1-8).
static const uint view_lit = 0;
static const uint view_base_color = 1;
static const uint view_normal = 2;
static const uint view_vertex_normal = 3;
static const uint view_metallic = 4;
static const uint view_roughness = 5;
static const uint view_occlusion = 6;
static const uint view_emissive = 7;

// What the environment's compute shaders found out about the sky.
struct EnvironmentInfo {
    float3 irradiance_sh[9];  // diffuse light, as spherical harmonics
    float3 sun_illuminance;   // lux at the ground; 0 for a photographed sky
};

// The same for every draw in a frame. Natural layout, like the C++ struct: the pointers land on 8-byte boundaries, right after the matrices.
struct FrameData {
    float4x4 view_projection;          // world space -> clip space
    float4x4 inverse_view_projection;  // clip space -> world space
    Vertex *vertices;                  // the scene's vertices
    DrawData *draws;                   // one DrawData per draw
    Material *materials;               // the scene's materials
    Light *lights;                     // the file's lights
    EnvironmentInfo *environment;      // the sky's diffuse light and the sun
    float3 camera_position;
    float exposure;                    // scene nits -> tone mapper input
    float3 sun_direction;              // toward the sun
    uint light_count;
    float3 sun_illuminance;            // lux, facing the sun
    uint view;                         // what to output
    uint sky_cube;                     // resource heap slots: the sky in full detail,
    uint specular_cube;                //   prefiltered per roughness,
    uint brdf_lut;                     //   and the split-sum BRDF table
    uint clamp_sampler;                // sampler heap index
    uint specular_mips;                // mip levels of specular_cube
    float sun_angular_radius;          // radians
};

// Written with vkCmdPushDataEXT before each draw.
struct PushData {
    FrameData *frame;  // this frame's data
    uint draw_index;   // which DrawData this draw uses
};

// In a descriptor heap pipeline, the push_constant block is where push data lands.
[[vk::push_constant]]
ConstantBuffer<PushData> push;
```

In `game-engine/CMakeLists.txt`, replace the line `file(GLOB shader_sources CONFIGURE_DEPENDS shaders/*.slang)` with:
```cmake
file(GLOB shader_sources CONFIGURE_DEPENDS shaders/*.slang)

# Headers shared between shaders (#include "shared.slangh"). They aren't
# compiled on their own, but every shader depends on them: editing one
# recompiles the shaders.
file(GLOB shader_headers CONFIGURE_DEPENDS shaders/*.slangh)
```

In `game-engine/CMakeLists.txt`, replace the shader's `add_custom_command` and the comment above it with:
```cmake
    # -fvk-use-entrypoint-name: keep vertexMain/fragmentMain as the SPIR-V
    #   entry point names (a file with a single entry point would get "main").
    # -matrix-layout-column-major: read float4x4 data column by column, the way
    #   glm stores it, so mul(M, v) in a shader means M * v in C++.
    # -capability spvDescriptorHeapEXT: compile descriptor handles to direct
    #   descriptor heap access (SPV_EXT_descriptor_heap).
    # -g: keep names and source in the SPIR-V for debuggers like RenderDoc.
    add_custom_command(
        OUTPUT ${spirv}
        COMMAND ${SLANGC} ${source} -target spirv -profile spirv_1_6 -fvk-use-entrypoint-name
                -matrix-layout-column-major -capability spvDescriptorHeapEXT -g -o ${spirv}
        DEPENDS ${source} ${shader_headers}
        COMMENT "Compiling ${shader_name}.slang"
        VERBATIM
    )
```

`game-engine/shaders/environment.slang`:
```slang
// Compute shaders that build the environment the scene is lit by:
//   skyMain         a physically based sky, rendered into the sky cube
//   equirectMain    an HDR photograph of the sky, projected into the sky cube
//   irradianceMain  the sky's diffuse light, as spherical harmonics
//   prefilterMain   the sky's specular light, blurred per roughness
//   brdfLutMain     the BRDF table the specular light is scaled by

// Data shared with C++ (src/includes/shader_types.h)

struct EnvironmentInfo {
    float3 irradiance_sh[9];
    float3 sun_illuminance;
};

struct EnvironmentPushData {
    EnvironmentInfo *info;
    uint source;
    uint target;
    float3 sun_direction;
    float source_scale;
    uint size;
    float roughness;
    uint sampler;
    uint source_size;
};

[[vk::push_constant]]
ConstantBuffer<EnvironmentPushData> push;

static const float pi = 3.14159265;

// Cube maps

// The direction through texel `texel` of cube face `face`, `size` texels across. Vulkan's cube faces are +X, -X, +Y, -Y, +Z, -Z, and each face's u and v run along the axes its table in the Vulkan spec gives; this is that table read backwards. u and v are -1..1 across the face, through texel centers.
float3 cube_direction(uint3 texel, uint size) {
    const float2 uv = (float2(texel.xy) + 0.5) / float(size) * 2.0 - 1.0;

    float3 direction;
    switch (texel.z) {
        case 0: direction = float3(1.0, -uv.y, -uv.x); break;
        case 1: direction = float3(-1.0, -uv.y, uv.x); break;
        case 2: direction = float3(uv.x, 1.0, uv.y); break;
        case 3: direction = float3(uv.x, -1.0, -uv.y); break;
        case 4: direction = float3(uv.x, -uv.y, 1.0); break;
        default: direction = float3(-uv.x, -uv.y, -1.0); break;
    }

    return normalize(direction);
}

// A cube map is written as a 2D array of 6 layers, one per face.
RWTexture2DArray<float4> storage_target() {
    return RWTexture2DArray<float4>.Handle(uint2(push.target, 0));
}

TextureCube source_cube() {
    return TextureCube.Handle(uint2(push.source, 0));
}

SamplerState clamp_sampler() {
    return SamplerState.Handle(uint2(push.sampler, 0));
}

// The atmosphere

// Single scattering in a spherical atmosphere: the light of the sun, scattered once toward the camera by air molecules (Rayleigh) and haze (Mie), and dimmed along both paths by the air, the haze and ozone. The constants are Earth's, as used by Bruneton's and Hillaire's atmosphere models. Distances in meters.

static const float planet_radius = 6360e3;
static const float atmosphere_radius = 6460e3;
static const float camera_altitude = 100.0;

// Air molecules scatter blue far more than red: the blue sky, the red sunset.
static const float3 rayleigh_scattering = float3(5.802e-6, 13.558e-6, 33.1e-6);
static const float rayleigh_height = 8000.0;  // density falls to 1/e of its value every 8 km

// Haze scatters all colors alike, mostly forward: the bright glow around the sun.
static const float mie_scattering = 3.996e-6;
static const float mie_extinction = 4.440e-6;  // scattering plus absorption
static const float mie_height = 1200.0;
static const float mie_anisotropy = 0.8;

// Ozone absorbs red and green, which keeps the twilight sky blue. It sits in a layer peaking 25 km up, tapering to nothing 15 km above and below.
static const float3 ozone_absorption = float3(0.650e-6, 1.881e-6, 0.085e-6);

// The sun's light before the atmosphere, in lux. White here: the atmosphere gives it all of its color.
static const float sun_illuminance_in_space = 128000.0;

static const float ground_albedo = 0.2;

// Distance along a ray from `origin` (relative to the planet's center) in direction `direction` to where it leaves a sphere of `radius`, or -1 if it misses. With the origin inside, the far hit is the one we want.
float sphere_exit(float3 origin, float3 direction, float radius) {
    const float b = dot(origin, direction);
    const float c = dot(origin, origin) - radius * radius;
    const float discriminant = b * b - c;
    return discriminant < 0.0 ? -1.0 : -b + sqrt(discriminant);
}

// Distance to where a ray hits the ground, or -1 if it doesn't.
float ground_hit(float3 origin, float3 direction) {
    const float b = dot(origin, direction);
    const float c = dot(origin, origin) - planet_radius * planet_radius;
    const float discriminant = b * b - c;

    if (discriminant < 0.0) {
        return -1.0;
    }

    const float t = -b - sqrt(discriminant);
    return t > 0.0 ? t : -1.0;
}

// How dense air, haze and ozone are at `altitude`, relative to sea level (air, haze) or the layer's peak (ozone).
float3 densities(float altitude) {
    return float3(
        exp(-altitude / rayleigh_height),
        exp(-altitude / mie_height),
        max(0.0, 1.0 - abs(altitude - 25000.0) / 15000.0)
    );
}

// The fraction of light, per channel, that crosses `optical_depth`: the densities summed along the path, times each one's extinction.
float3 transmittance(float3 optical_depth) {
    const float3 extinction = rayleigh_scattering * optical_depth.x
        + mie_extinction * optical_depth.y
        + ozone_absorption * optical_depth.z;
    return exp(-extinction);
}

// Optical depth from `origin` toward the sun, to the top of the atmosphere. A huge value if the planet is in the way: then no sunlight arrives.
float3 sun_optical_depth(float3 origin, float3 sun) {
    if (ground_hit(origin, sun) > 0.0) {
        return float3(1e9);
    }

    const int steps = 8;
    const float distance = sphere_exit(origin, sun, atmosphere_radius);
    const float step = distance / steps;

    float3 depth = 0.0;
    for (int i = 0; i < steps; ++i) {
        const float3 point = origin + sun * (step * (i + 0.5));
        depth += densities(length(point) - planet_radius) * step;
    }

    return depth;
}

// Rayleigh's phase function: how much light scatters by angle (cosine mu); as much forward as back, least at right angles.
float rayleigh_phase(float mu) {
    return 3.0 / (16.0 * pi) * (1.0 + mu * mu);
}

// The Cornette-Shanks phase function for haze: strongly forward.
float mie_phase(float mu) {
    const float g = mie_anisotropy;
    const float g2 = g * g;
    return 3.0 / (8.0 * pi) * (1.0 - g2) * (1.0 + mu * mu)
        / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * mu, 1.5));
}

// The sky's brightness in nits, looking in `direction` from the camera. The sun's disk itself isn't included: the scene's direct sunlight comes from the sun as a light, and the sky background draws the disk.
float3 sky_radiance(float3 direction, float3 sun) {
    const float3 origin = float3(0.0, planet_radius + camera_altitude, 0.0);

    // March from the camera to the top of the atmosphere, or to the ground.
    const float ground = ground_hit(origin, direction);
    const float distance = ground > 0.0 ? ground : sphere_exit(origin, direction, atmosphere_radius);

    const int steps = 32;
    const float step = distance / steps;
    const float mu = dot(direction, sun);

    float3 view_depth = 0.0;
    float3 scattered = 0.0;

    for (int i = 0; i < steps; ++i) {
        const float3 point = origin + direction * (step * (i + 0.5));
        const float3 density = densities(length(point) - planet_radius);

        // Sunlight reaches the point dimmed by the path from the sun, and the light it scatters reaches the camera dimmed by the path back.
        view_depth += density * (step * 0.5);
        const float3 attenuation = transmittance(view_depth + sun_optical_depth(point, sun));
        view_depth += density * (step * 0.5);

        const float3 scattering = rayleigh_scattering * density.x * rayleigh_phase(mu)
            + mie_scattering * density.y * mie_phase(mu);

        scattered += attenuation * scattering * step;
    }

    float3 radiance = sun_illuminance_in_space * scattered;

    // Looking down: the ground, lit by the sun (not by the sky, for simplicity), seen through the air in between.
    if (ground > 0.0) {
        const float3 point = origin + direction * ground;
        const float3 up = normalize(point);
        const float3 sunlight = sun_illuminance_in_space * transmittance(sun_optical_depth(point, sun));
        radiance += ground_albedo / pi * sunlight * max(dot(up, sun), 0.0) * transmittance(view_depth);
    }

    return radiance;
}

// Renders the sky into mip 0 of the sky cube: one thread per texel, the six faces as the dispatch's z. Thread 0 also works out the sunlight at the camera, for the scene's sun.
[shader("compute")]
[numthreads(8, 8, 1)]
void skyMain(uint3 texel : SV_DispatchThreadID) {
    if (all(texel == 0)) {
        const float3 origin = float3(0.0, planet_radius + camera_altitude, 0.0);
        push.info->sun_illuminance = sun_illuminance_in_space * transmittance(sun_optical_depth(origin, push.sun_direction));
    }

    if (texel.x >= push.size || texel.y >= push.size) {
        return;
    }

    // Kept below 65504, the largest 16-bit float, like the photograph.
    const float3 direction = cube_direction(texel, push.size);
    storage_target()[texel] = float4(min(sky_radiance(direction, push.sun_direction), 65504.0), 1.0);
}

// An HDR photograph of the sky

// Projects an equirectangular image (longitude across, latitude down: the usual layout for sky photographs) into mip 0 of the sky cube. The file's values are relative, so source_scale turns them into nits.
[shader("compute")]
[numthreads(8, 8, 1)]
void equirectMain(uint3 texel : SV_DispatchThreadID) {
    if (texel.x >= push.size || texel.y >= push.size) {
        return;
    }

    const float3 d = cube_direction(texel, push.size);

    // Longitude 0 (the image's center) looks down -Z; latitude runs from straight up (top row) to straight down (bottom row).
    const float2 uv = float2(atan2(d.x, -d.z) / (2.0 * pi) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / pi);

    const Texture2D image = Texture2D.Handle(uint2(push.source, 0));
    const float3 radiance = image.SampleLevel(clamp_sampler(), uv, 0.0).rgb * push.source_scale;

    // In nits, the brightest texels can pass 65504, the largest 16-bit float the cube can hold; what the GPU stores instead isn't defined, and an infinity would spread into the mips and the lighting. Clamp it.
    storage_target()[texel] = float4(min(radiance, 65504.0), 1.0);
}

// Diffuse light: spherical harmonics

// A diffuse surface gathers light from its whole hemisphere, so what it receives (irradiance) changes slowly with its normal. Nine spherical harmonics, the 3D equivalent of a few Fourier terms, capture it to within a few percent (Ramamoorthi and Hanrahan, 2001).

// The nine real spherical harmonics, bands 0 to 2, at unit vector n.
void sh_basis(float3 n, out float basis[9]) {
    basis[0] = 0.282095;
    basis[1] = 0.488603 * n.y;
    basis[2] = 0.488603 * n.z;
    basis[3] = 0.488603 * n.x;
    basis[4] = 1.092548 * n.x * n.y;
    basis[5] = 1.092548 * n.y * n.z;
    basis[6] = 0.315392 * (3.0 * n.z * n.z - 1.0);
    basis[7] = 1.092548 * n.x * n.z;
    basis[8] = 0.546274 * (n.x * n.x - n.y * n.y);
}

static const uint irradiance_threads = 64;
static const uint irradiance_face_size = 32;

groupshared float3 partial_sh[irradiance_threads][9];

// One workgroup projects the sky onto the nine harmonics. It reads the sky cube at the mip level that's 32 texels across (a detailed sky only blurs into irradiance anyway), each texel weighted by the solid angle it covers. The sum is then convolved with the cosine lobe a diffuse surface sees, which in this basis is just a factor per band: pi, 2pi/3 and pi/4.
[shader("compute")]
[numthreads(irradiance_threads, 1, 1)]
void irradianceMain(uint thread : SV_GroupIndex) {
    const uint size = irradiance_face_size;
    const float lod = log2(float(push.source_size) / float(size));

    float3 sum[9];
    for (uint k = 0; k < 9; ++k) {
        sum[k] = 0.0;
    }

    for (uint i = thread; i < size * size * 6; i += irradiance_threads) {
        const uint3 texel = uint3(i % size, (i / size) % size, i / (size * size));
        const float3 direction = cube_direction(texel, size);

        // A texel at (u, v) on a face one unit away covers (2 / size)^2 of area, seen at a slant: its solid angle shrinks by (1 + u^2 + v^2)^1.5.
        const float2 uv = (float2(texel.xy) + 0.5) / float(size) * 2.0 - 1.0;
        const float solid_angle = 4.0 / (size * size) / pow(1.0 + dot(uv, uv), 1.5);

        const float3 radiance = source_cube().SampleLevel(clamp_sampler(), direction, lod).rgb;

        float basis[9];
        sh_basis(direction, basis);

        for (uint k = 0; k < 9; ++k) {
            sum[k] += radiance * basis[k] * solid_angle;
        }
    }

    for (uint k = 0; k < 9; ++k) {
        partial_sh[thread][k] = sum[k];
    }

    GroupMemoryBarrierWithGroupSync();

    if (thread == 0) {
        static const float band[9] = {
            pi,
            2.0 * pi / 3.0, 2.0 * pi / 3.0, 2.0 * pi / 3.0,
            pi / 4.0, pi / 4.0, pi / 4.0, pi / 4.0, pi / 4.0,
        };

        for (uint k = 0; k < 9; ++k) {
            float3 total = 0.0;
            for (uint t = 0; t < irradiance_threads; ++t) {
                total += partial_sh[t][k];
            }
            push.info->irradiance_sh[k] = total * band[k];
        }
    }
}

// Specular light: prefiltering

// The GGX distribution, as in mesh.slang.
float distribution_ggx(float n_dot_h, float alpha) {
    const float alpha2 = alpha * alpha;
    const float f = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (pi * f * f);
}

// A low-discrepancy sequence: point i of n, spread evenly over the unit square. Reversing the bits of i gives the second coordinate.
float2 hammersley(uint i, uint n) {
    return float2(float(i) / float(n), float(reversebits(i)) * 2.3283064365386963e-10);
}

// A half vector around `normal`, distributed like GGX's microfacets: more of them where D is high. `xi` picks which one.
float3 sample_ggx(float2 xi, float3 normal, float alpha) {
    const float phi = 2.0 * pi * xi.x;
    const float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
    const float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    const float3 h = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    // From the sample's frame, around +Z, into the normal's.
    const float3 helper = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    const float3 tangent = normalize(cross(helper, normal));
    const float3 bitangent = cross(normal, tangent);
    return normalize(tangent * h.x + bitangent * h.y + normal * h.z);
}

// One mip level of the specular cube: the sky as a surface of `roughness` reflects it, looking straight at it (n = v = r, the split-sum assumption). Rougher levels are smaller: their light is blurrier.
//   - Importance sampling: 128 directions, chosen where GGX reflects the most light, weighted by n.l.
//   - Filtered importance sampling (Krivanek and Colbert, 2008): each sample reads the sky at a mip level matching the solid angle it stands for, so a few samples cover the lobe with little visible noise.
[shader("compute")]
[numthreads(8, 8, 1)]
void prefilterMain(uint3 texel : SV_DispatchThreadID) {
    if (texel.x >= push.size || texel.y >= push.size) {
        return;
    }

    const float3 normal = cube_direction(texel, push.size);

    // Smooth: a mirror. Read the sky at the mip level this size matches.
    if (push.roughness == 0.0) {
        const float lod = log2(float(push.source_size) / float(push.size));
        storage_target()[texel] = source_cube().SampleLevel(clamp_sampler(), normal, lod);
        return;
    }

    const float alpha = push.roughness * push.roughness;
    const uint samples = 128;
    const float texel_solid_angle = 4.0 * pi / (6.0 * push.source_size * push.source_size);

    float3 sum = 0.0;
    float weight = 0.0;

    for (uint i = 0; i < samples; ++i) {
        const float3 h = sample_ggx(hammersley(i, samples), normal, alpha);
        const float3 l = 2.0 * dot(normal, h) * h - normal;
        const float n_dot_l = dot(normal, l);

        if (n_dot_l > 0.0) {
            // With n = v, the probability of this direction is D / 4.
            const float n_dot_h = max(dot(normal, h), 0.0);
            const float pdf = distribution_ggx(n_dot_h, alpha) / 4.0;
            const float sample_solid_angle = 1.0 / (samples * pdf + 1e-6);
            const float lod = max(0.5 * log2(sample_solid_angle / texel_solid_angle), 0.0);

            sum += source_cube().SampleLevel(clamp_sampler(), l, lod).rgb * n_dot_l;
            weight += n_dot_l;
        }
    }

    storage_target()[texel] = float4(sum / max(weight, 1e-6), 1.0);
}

// The BRDF table

// Smith's height-correlated visibility, as in mesh.slang.
float visibility_smith(float n_dot_l, float n_dot_v, float alpha) {
    const float alpha2 = alpha * alpha;
    const float from_view = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2);
    const float from_light = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2);
    const float sum = from_view + from_light;
    return sum > 0.0 ? 0.5 / sum : 0.0;
}

// The split sum's second half: how much light the specular BRDF reflects overall, for a view angle (n.v across) and a roughness (down), as a scale and a bias on the reflectance at normal incidence, F0:
//     reflected = F0 * scale + bias
// Computed once, by importance-sampling GGX like the prefilter.
[shader("compute")]
[numthreads(8, 8, 1)]
void brdfLutMain(uint3 texel : SV_DispatchThreadID) {
    if (texel.x >= push.size || texel.y >= push.size) {
        return;
    }

    const float n_dot_v = (texel.x + 0.5) / push.size;
    const float roughness = (texel.y + 0.5) / push.size;
    const float alpha = roughness * roughness;

    const float3 normal = float3(0.0, 0.0, 1.0);
    const float3 view = float3(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);

    const uint samples = 512;
    float scale = 0.0;
    float bias = 0.0;

    for (uint i = 0; i < samples; ++i) {
        const float3 h = sample_ggx(hammersley(i, samples), normal, alpha);
        const float3 l = 2.0 * dot(view, h) * h - view;
        const float n_dot_l = l.z;

        if (n_dot_l > 0.0) {
            const float n_dot_h = max(h.z, 0.0);
            const float v_dot_h = max(dot(view, h), 0.0);

            // BRDF x n.l / pdf, with Fresnel left out: V * D * n.l over D * n.h / (4 v.h).
            const float reflected = visibility_smith(n_dot_l, n_dot_v, alpha) * 4.0 * n_dot_l * v_dot_h / max(n_dot_h, 1e-6);

            // Schlick's Fresnel, F0 + (1 - F0) * f, split into F0's share and the rest.
            const float fresnel = pow(1.0 - v_dot_h, 5.0);
            scale += (1.0 - fresnel) * reflected;
            bias += fresnel * reflected;
        }
    }

    RWTexture2D<float4> table = RWTexture2D<float4>.Handle(uint2(push.target, 0));
    table[texel.xy] = float4(scale / samples, bias / samples, 0.0, 1.0);
}
```

## 8.5 Compute and full-screen pipelines: `pipeline.h`, `pipeline.cpp`

### Why
The environment needs compute pipelines. The sky background is a full-screen triangle like tone mapping, but depth-tested into the HDR image.

### How
- **`create_compute_pipeline`:** a compute pipeline is a single shader stage and nothing else. Like the graphics pipelines, it reads its resources from the descriptor heap, so it has the same `eDescriptorHeapEXT` flag and no layout.
- **`create_fullscreen_pipeline`** replaces Chapter 7's `create_tonemap_pipeline`. It takes the shader's name and an optional depth format.
  - **With a depth format,** the triangle is depth-tested with "greater or equal" and writes no depth. With reverse-Z, the depth buffer is cleared to 0, and the triangle sits at depth 0, so it passes only where nothing has been drawn: the sky lands behind the scene.
  - **Without one,** as for tone mapping, there's no depth test at all.

### Code
`game-engine/src/includes/pipeline.h`:
```cpp
#pragma once

#include "includes/shader_types.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <filesystem>
#include <vector>

// A .spv file as the 32-bit words SPIR-V is made of.
std::vector<std::uint32_t> read_spirv(const std::filesystem::path &path);

// Draws shaders/mesh.slang into a `color_format` image, depth-tested against a `depth_format` depth buffer, for materials with alpha mode `alpha_mode`. There is no pipeline layout: shaders find their resources in the descriptor heap. Cull mode and front face are set per draw.
vk::raii::Pipeline create_mesh_pipeline(
    const vk::raii::Device &device,
    vk::Format color_format,
    vk::Format depth_format,
    AlphaMode alpha_mode
);

// Draws shaders/<shader>.spv's vertexMain and fragmentMain as one full-screen triangle into a `color_format` image: no vertex data, nothing culled. With a `depth_format`, the triangle is depth-tested at depth 0, the far plane, without writing depth, so it only reaches pixels nothing else has been drawn on: that's how the sky goes behind the scene.
vk::raii::Pipeline create_fullscreen_pipeline(
    const vk::raii::Device &device,
    const char *shader,
    vk::Format color_format,
    vk::Format depth_format = vk::Format::eUndefined
);

// A compute pipeline running `entry_point` from shaders/<shader>.spv.
vk::raii::Pipeline create_compute_pipeline(const vk::raii::Device &device, const char *shader, const char *entry_point);
```

`game-engine/src/pipeline.cpp`:
```cpp
#include "includes/pipeline.h"

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>

// Loading SPIR-V

std::vector<std::uint32_t> read_spirv(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("can't open " + path.string());
    }

    const std::uintmax_t size = std::filesystem::file_size(path);

    if (size % sizeof(std::uint32_t) != 0) {
        throw std::runtime_error(path.string() + " isn't SPIR-V: its size isn't a whole number of words");
    }

    // A vector of uint32_t, not of char: Vulkan wants the code 4-byte aligned.
    std::vector<std::uint32_t> words(size / sizeof(std::uint32_t));
    file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(size));

    if (!file) {
        throw std::runtime_error("can't read " + path.string());
    }

    return words;
}

// The mesh pipeline

vk::raii::Pipeline create_mesh_pipeline(
    const vk::raii::Device &device,
    vk::Format color_format,
    vk::Format depth_format,
    AlphaMode alpha_mode
) {
    // Shaders: one module, two entry points picked by name. The module is only needed while the pipeline is built, so it's destroyed on return.
    const std::vector<std::uint32_t> spirv = read_spirv(std::filesystem::path(SHADER_DIR) / "mesh.spv");

    const vk::raii::ShaderModule module(device, vk::ShaderModuleCreateInfo{
        .codeSize = spirv.size() * sizeof(std::uint32_t),
        .pCode = spirv.data(),
    });

    // The fragment shader's alpha mode is a specialization constant: a constant whose value is filled in now, when the pipeline is built. The compiler then removes the code the other modes need, so opaque surfaces never pay for the alpha test.
    const vk::SpecializationMapEntry alpha_mode_entry{
        .constantID = 0,  // [vk::constant_id(0)] in mesh.slang
        .offset = 0,
        .size = sizeof(AlphaMode),
    };

    const vk::SpecializationInfo specialization{
        .mapEntryCount = 1,
        .pMapEntries = &alpha_mode_entry,
        .dataSize = sizeof(alpha_mode),
        .pData = &alpha_mode,
    };

    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *module,
            .pName = "vertexMain",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *module,
            .pName = "fragmentMain",
            .pSpecializationInfo = &specialization,
        },
    };

    // Vertex input and assembly: no vertex attributes, the vertex shader reads its vertex through a pointer. Every 3 vertices form a triangle.
    const vk::PipelineVertexInputStateCreateInfo vertex_input{};

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };

    // Viewport: counts only. The viewport and scissor rectangles are set while recording, so a resized window doesn't need a new pipeline.
    const vk::PipelineViewportStateCreateInfo viewport{
        .viewportCount = 1,
        .scissorCount = 1,
    };

    // Which side of a triangle is culled depends on the draw: its material may be double-sided, and its transform may mirror it. Both states are dynamic, so one pipeline serves every draw.
    const std::array dynamic_states{
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
        vk::DynamicState::eCullMode,
        vk::DynamicState::eFrontFace,
    };

    const vk::PipelineDynamicStateCreateInfo dynamic{
        .dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    // Rasterization: filled triangles. cullMode and frontFace are dynamic, set before each draw.
    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .polygonMode = vk::PolygonMode::eFill,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    const bool blend = alpha_mode == AlphaMode::blend;

    // Depth: keep a fragment only if it's nearer than what's already there. With reverse-Z (see camera.cpp) nearer means a *greater* depth value, and the buffer is cleared to 0, the far plane. Solid surfaces then record their depth. See-through ones don't: something drawn behind them later must still show through.
    const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = blend ? vk::False : vk::True,
        .depthCompareOp = vk::CompareOp::eGreater,
    };

    // Color output. Opaque and masked surfaces replace what's there. Blended ones mix with it, weighted by their alpha:
    //     color = source.rgb * source.a + destination.rgb * (1 - source.a)
    const vk::PipelineColorBlendAttachmentState blend_attachment{
        .blendEnable = blend ? vk::True : vk::False,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                        | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    const vk::PipelineColorBlendStateCreateInfo color_blend{
        .attachmentCount = 1,
        .pAttachments = &blend_attachment,
    };

    // Dynamic rendering: instead of a VkRenderPass, the pipeline names the formats of the images it will draw into.
    const vk::PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &color_format,
        .depthAttachmentFormat = depth_format,
    };

    // Descriptor heap mode is what makes `layout = nullptr` legal: shaders will reach resources through the heap and push data, not descriptor sets and push constants declared in a VkPipelineLayout.
    const vk::PipelineCreateFlags2CreateInfo flags{
        .pNext = &rendering,
        .flags = vk::PipelineCreateFlagBits2::eDescriptorHeapEXT,
    };

    // pNext chain: create info -> flags -> rendering. Everything it points at lives until the end of this function, past the pipeline's creation.
    return vk::raii::Pipeline(device, nullptr, vk::GraphicsPipelineCreateInfo{
        .pNext = &flags,
        .stageCount = static_cast<std::uint32_t>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic,
        .layout = nullptr,
    });
}

// Full-screen pipelines

vk::raii::Pipeline create_fullscreen_pipeline(
    const vk::raii::Device &device,
    const char *shader,
    vk::Format color_format,
    vk::Format depth_format
) {
    const std::vector<std::uint32_t> spirv = read_spirv(std::filesystem::path(SHADER_DIR) / (std::string(shader) + ".spv"));

    const vk::raii::ShaderModule module(device, vk::ShaderModuleCreateInfo{
        .codeSize = spirv.size() * sizeof(std::uint32_t),
        .pCode = spirv.data(),
    });

    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *module,
            .pName = "vertexMain",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *module,
            .pName = "fragmentMain",
        },
    };

    // The vertex shader makes the triangle's three corners from the vertex index alone.
    const vk::PipelineVertexInputStateCreateInfo vertex_input{};

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };

    const vk::PipelineViewportStateCreateInfo viewport{
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    const vk::PipelineDynamicStateCreateInfo dynamic{
        .dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    // One triangle that covers the screen: culling it would only ever hide it.
    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    // With reverse-Z the buffer is cleared to 0, and the triangle sits at 0: it passes "greater or equal" only where the depth is still 0, where nothing has been drawn. Without a depth buffer, every pixel passes.
    const bool depth = depth_format != vk::Format::eUndefined;

    const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
        .depthTestEnable = depth ? vk::True : vk::False,
        .depthWriteEnable = vk::False,
        .depthCompareOp = vk::CompareOp::eGreaterOrEqual,
    };

    // Every pixel the triangle reaches is written exactly once: no blending.
    const vk::PipelineColorBlendAttachmentState blend_attachment{
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                        | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    const vk::PipelineColorBlendStateCreateInfo color_blend{
        .attachmentCount = 1,
        .pAttachments = &blend_attachment,
    };

    const vk::PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &color_format,
        .depthAttachmentFormat = depth_format,
    };

    const vk::PipelineCreateFlags2CreateInfo flags{
        .pNext = &rendering,
        .flags = vk::PipelineCreateFlagBits2::eDescriptorHeapEXT,
    };

    return vk::raii::Pipeline(device, nullptr, vk::GraphicsPipelineCreateInfo{
        .pNext = &flags,
        .stageCount = static_cast<std::uint32_t>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic,
        .layout = nullptr,
    });
}

// Compute pipelines

vk::raii::Pipeline create_compute_pipeline(const vk::raii::Device &device, const char *shader, const char *entry_point) {
    const std::vector<std::uint32_t> spirv = read_spirv(std::filesystem::path(SHADER_DIR) / (std::string(shader) + ".spv"));

    const vk::raii::ShaderModule module(device, vk::ShaderModuleCreateInfo{
        .codeSize = spirv.size() * sizeof(std::uint32_t),
        .pCode = spirv.data(),
    });

    // A compute pipeline is a single shader stage, and nothing else. Like the graphics pipelines, it reads its resources from the descriptor heap.
    const vk::PipelineCreateFlags2CreateInfo flags{
        .flags = vk::PipelineCreateFlagBits2::eDescriptorHeapEXT,
    };

    return vk::raii::Pipeline(device, nullptr, vk::ComputePipelineCreateInfo{
        .pNext = &flags,
        .stage = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = *module,
            .pName = entry_point,
        },
        .layout = nullptr,
    });
}
```

## 8.6 The environment on the CPU: `environment.h`, `environment.cpp`

### Why
Something has to create the environment's images and pipelines, upload the photograph, write all the descriptors, and run the compute shaders in the right order, with the right barriers, whenever the sky changes.

### How
- **The images,** all 16-bit float RGBA:
  - **The sky cube:** 512 per face, with 10 mips. It's written by a compute shader, shrunk into its mips by blits, and sampled.
  - **The specular cube:** 128 per face, with 6 mips, for roughness 0, 0.2, … 1.
  - **The BRDF table:** 128 × 128.
  - **The photograph,** as loaded. `glm::packHalf4x16` packs each RGBA texel into four 16-bit halves; the file's relative values, at most about 33, fit easily.
  - **No image views:** `GpuImage` is just an image and its memory. Shaders reach the images through descriptor heap slots, written from view descriptions.
- **Twelve resource heap slots** follow the HDR image's.
  - **Reading:** the cubes are sampled as cube views, with all of their mips.
  - **Writing:** they're written as 2D-array views of their six layers. A storage descriptor covers one mip level, so the specular cube has one storage slot per mip.
- **`update_environment`,** in one submission:
  1. the atmosphere or the photograph into mip 0 of the sky cube,
  2. its mip chain, each level blitted from the one above, all six faces at once,
  3. the spherical harmonics, from one workgroup,
  4. each mip of the specular cube, for its roughness.

  Barriers between the steps make each one wait for the writes it reads. A last memory barrier makes the compute shaders' writes to the info buffer visible to the host, which reads it once the fence says the work is done, and to the scene's fragment shaders. Waiting for a fence orders the work, but doesn't by itself make one shader's writes visible to another's reads.
- **Its own command buffer:**
  - **The design:** the environment keeps one command buffer, reset and re-recorded for each update, instead of allocating and freeing one per update like `submit_and_wait` does.
  - **The reason:** these commands bind the descriptor heaps, and the validation layer (version 1.4.341) wrongly reports "conflicting reserved ranges" once a command buffer that bound them has been freed.
- **Waiting:** `update_environment` waits for the GPU before and after, because frames in flight may still be sampling the cubes it rewrites. An update takes about 2 ms, which is fine for a key press.
- **`sky_irradiance`** evaluates the spherical harmonics on the CPU, for the light meter (8.9).

### Code
`game-engine/src/includes/environment.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/descriptor_heap.h"
#include "includes/shader_types.h"
#include "includes/vulkan_setup.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>

// The environment

// Where the scene's sky comes from.
enum class SkySource {
    atmosphere,  // simulated, for the time of day, with the sun as a light
    photograph,  // an HDR image of a real sky; its sun is already in the image
};

// An image and its memory. There's no VkImageView: shaders reach it through descriptor heap slots, written from view descriptions.
struct GpuImage {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image handle = nullptr;
};

// Every image is 16-bit float RGBA: sky radiance needs the range, up to 65504 nits.
constexpr vk::Format environment_format = vk::Format::eR16G16B16A16Sfloat;

constexpr std::uint32_t sky_cube_size = 512;      // per face, at mip 0
constexpr std::uint32_t sky_cube_mips = 10;       // 512 down to 1
constexpr std::uint32_t specular_cube_size = 128;
constexpr std::uint32_t specular_mips = 6;        // roughness 0, 0.2, ... 1
constexpr std::uint32_t brdf_lut_size = 128;

// The resource heap slots the environment uses, consecutive from `first`: sampled descriptors for reading, storage descriptors for compute shaders to write.
struct EnvironmentSlots {
    std::uint32_t sky_cube;          // sampled, every mip
    std::uint32_t sky_target;        // storage, mip 0
    std::uint32_t specular_cube;     // sampled, every mip
    std::uint32_t specular_targets;  // storage, one per mip: this slot and the next specular_mips - 1
    std::uint32_t brdf_lut;          // sampled
    std::uint32_t brdf_target;       // storage
    std::uint32_t photograph;        // sampled: the HDR image as loaded
};

constexpr std::uint32_t environment_slot_count = 6 + specular_mips;

// The sky in its forms for lighting: a full-detail cube map, its diffuse light as spherical harmonics (in `info`), a cube prefiltered for specular light, and the BRDF table that goes with it.
struct Environment {
    GpuImage sky_cube;
    GpuImage specular_cube;
    GpuImage brdf_lut;
    GpuImage photograph;
    std::uint32_t photograph_width = 0;
    std::uint32_t photograph_height = 0;

    Buffer info;                       // one EnvironmentInfo, host-visible
    EnvironmentInfo *mapped = nullptr;

    vk::raii::Pipeline sky = nullptr;
    vk::raii::Pipeline equirect = nullptr;
    vk::raii::Pipeline irradiance = nullptr;
    vk::raii::Pipeline prefilter = nullptr;
    vk::raii::Pipeline brdf = nullptr;

    EnvironmentSlots slots;
    std::uint32_t clamp_sampler = 0;

    // The environment's own command buffer, re-recorded for each update.
    vk::raii::CommandBuffer commands = nullptr;
};

// Creates the images and pipelines, writes their descriptors from slot `first_slot` on, loads the HDR photograph at `photograph`, and computes the BRDF table, which never changes. Call update_environment before drawing.
Environment create_environment(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    DescriptorHeaps &heaps,
    std::uint32_t first_slot,
    const std::filesystem::path &photograph
);

// Rebuilds the sky from `source`, with the sun toward `sun_direction`, then its diffuse and specular light. Waits for the GPU before and after, so it must not be called while a frame is being recorded.
void update_environment(
    Environment &environment,
    const vk::raii::Device &device,
    const vk::raii::Queue &queue,
    const DescriptorHeaps &heaps,
    SkySource source,
    glm::vec3 sun_direction
);

// The sky's irradiance, in lux, on a surface facing `normal`: the spherical harmonics evaluated on the CPU, for the light meter.
glm::vec3 sky_irradiance(const EnvironmentInfo &info, glm::vec3 normal);
```

`game-engine/src/environment.cpp`:
```cpp
#include "includes/environment.h"

#include "includes/pipeline.h"
#include "includes/radiance_hdr.h"

#include <glm/gtc/packing.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <limits>

namespace {

    // The photograph's values are relative: a camera records light, not its units. This sets how bright a stored 1.0 is, in nits; with it, the kloppenheim sky's average is about that of an overcast evening.
    constexpr float photograph_nits = 2000.0f;

    // Images

    GpuImage create_gpu_image(const vk::raii::Device &device, const GpuChoice &gpu, const vk::ImageCreateInfo &info) {
        GpuImage image;
        image.handle = vk::raii::Image(device, info);

        const vk::MemoryRequirements requirements = image.handle.getMemoryRequirements();

        image.memory = vk::raii::DeviceMemory(device, vk::MemoryAllocateInfo{
            .allocationSize = requirements.size,
            .memoryTypeIndex = find_memory_type(gpu, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
        });

        image.handle.bindMemory(*image.memory, 0);
        return image;
    }

    // A cube map: a 2D image with 6 array layers that may be viewed as a cube.
    GpuImage create_cube(const vk::raii::Device &device, const GpuChoice &gpu, std::uint32_t size, std::uint32_t mips, vk::ImageUsageFlags usage) {
        return create_gpu_image(device, gpu, vk::ImageCreateInfo{
            .flags = vk::ImageCreateFlagBits::eCubeCompatible,
            .imageType = vk::ImageType::e2D,
            .format = environment_format,
            .extent = {size, size, 1},
            .mipLevels = mips,
            .arrayLayers = 6,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
    }

    GpuImage create_flat(const vk::raii::Device &device, const GpuChoice &gpu, std::uint32_t width, std::uint32_t height, vk::ImageUsageFlags usage) {
        return create_gpu_image(device, gpu, vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = environment_format,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
    }

    // How shaders see an image: `type` (2D, 2D array, cube), from mip `base_mip`, `mips` levels, all of its layers.
    vk::ImageViewCreateInfo view_of(const GpuImage &image, vk::ImageViewType type, std::uint32_t base_mip, std::uint32_t mips, std::uint32_t layers) {
        return vk::ImageViewCreateInfo{
            .image = *image.handle,
            .viewType = type,
            .format = environment_format,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = base_mip,
                .levelCount = mips,
                .baseArrayLayer = 0,
                .layerCount = layers,
            },
        };
    }

    // Barriers

    // Moves mips [base_mip, base_mip + mips) of every layer of `image` between layouts, after `src` work and before `dst` work.
    void barrier(
        const vk::raii::CommandBuffer &commands,
        const GpuImage &image,
        vk::ImageLayout from,
        vk::ImageLayout to,
        vk::PipelineStageFlags2 src_stage,
        vk::AccessFlags2 src_access,
        vk::PipelineStageFlags2 dst_stage,
        vk::AccessFlags2 dst_access,
        std::uint32_t base_mip = 0,
        std::uint32_t mips = vk::RemainingMipLevels
    ) {
        const vk::ImageMemoryBarrier2 image_barrier{
            .srcStageMask = src_stage,
            .srcAccessMask = src_access,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = from,
            .newLayout = to,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = *image.handle,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = base_mip,
                .levelCount = mips,
                .baseArrayLayer = 0,
                .layerCount = vk::RemainingArrayLayers,
            },
        };

        commands.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &image_barrier});
    }

    // Makes compute shaders' writes to memory, through pointers, visible to the `dst` work: the CPU once the submission is done, or later shaders.
    void memory_barrier(const vk::raii::CommandBuffer &commands, vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
        const vk::MemoryBarrier2 memory{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
        };

        commands.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &memory});
    }

    // Submitting

    // Records `record` into the environment's command buffer, submits it and waits. The same command buffer is reset and reused every time rather than allocated and freed like submit_and_wait's: these commands bind the descriptor heaps, and the validation layer (1.4.341) wrongly reports conflicting heap ranges after a command buffer that bound them is freed.
    void run(
        const Environment &environment,
        const vk::raii::Device &device,
        const vk::raii::Queue &queue,
        const std::function<void(const vk::raii::CommandBuffer&)> &record
    ) {
        const vk::raii::CommandBuffer &commands = environment.commands;

        commands.reset();
        commands.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        record(commands);
        commands.end();

        const vk::raii::Fence done(device, vk::FenceCreateInfo{});
        const vk::CommandBufferSubmitInfo command_info{.commandBuffer = *commands};

        queue.submit2(vk::SubmitInfo2{
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &command_info,
        }, *done);

        (void)device.waitForFences(*done, vk::True, std::numeric_limits<std::uint64_t>::max());
    }

    // Dispatching

    // Runs `pipeline` over a `size` x `size` image with `layers` layers, one thread per texel, in the shaders' 8 x 8 workgroups.
    void dispatch(
        const vk::raii::CommandBuffer &commands,
        const vk::raii::Pipeline &pipeline,
        const EnvironmentPushData &push,
        std::uint32_t layers
    ) {
        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        commands.pushDataEXT(vk::PushDataInfoEXT{
            .offset = 0,
            .data = {.address = &push, .size = sizeof(push)},
        });

        const std::uint32_t groups = (push.size + 7) / 8;
        commands.dispatch(groups, groups, layers);
    }

    // Fills mips 1 and up of the sky cube from mip 0. Each level is the one above shrunk to half size by a linearly filtered blit, all six faces at once. On entry mip 0 is eTransferSrcOptimal and the rest eUndefined; on return every mip is eShaderReadOnlyOptimal.
    void generate_sky_mips(const vk::raii::CommandBuffer &commands, const GpuImage &cube) {
        barrier(commands, cube, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
            vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite, 1);

        for (std::uint32_t mip = 1; mip < sky_cube_mips; ++mip) {
            const auto from = static_cast<std::int32_t>(sky_cube_size >> (mip - 1));
            const auto to = static_cast<std::int32_t>(sky_cube_size >> mip);

            commands.blitImage(*cube.handle, vk::ImageLayout::eTransferSrcOptimal,
                *cube.handle, vk::ImageLayout::eTransferDstOptimal,
                vk::ImageBlit{
                    .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = mip - 1, .baseArrayLayer = 0, .layerCount = 6},
                    .srcOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{from, from, 1}},
                    .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = mip, .baseArrayLayer = 0, .layerCount = 6},
                    .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{to, to, 1}},
                },
                vk::Filter::eLinear);

            // This level is the next blit's source.
            barrier(commands, cube, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
                vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead, mip, 1);
        }

        barrier(commands, cube, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader,
            vk::AccessFlagBits2::eShaderSampledRead);
    }

    // The photograph

    // Loads the HDR file and uploads it as 16-bit floats. glm::packHalf4x16 packs four floats into four halves, 8 bytes: one RGBA texel. Halves top out at 65504: fine for the file's relative values (this sky's peak is about 33), though once scaled to nits its brightest texels pass it, and equirectMain clamps them.
    void upload_photograph(
        Environment &environment,
        const vk::raii::Device &device,
        const GpuChoice &gpu,
        const vk::raii::Queue &queue,
        const vk::raii::CommandPool &pool,
        const std::filesystem::path &path
    ) {
        const HdrImage image = load_radiance_hdr(path);
        const std::size_t texels = static_cast<std::size_t>(image.width) * image.height;

        const Buffer staging = create_buffer(device, gpu, texels * sizeof(std::uint64_t),
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        auto *halves = static_cast<std::uint64_t*>(staging.memory.mapMemory(0, staging.size));

        for (std::size_t i = 0; i < texels; ++i) {
            const glm::vec4 rgba{image.rgb[i * 3], image.rgb[i * 3 + 1], image.rgb[i * 3 + 2], 1.0f};
            halves[i] = glm::packHalf4x16(rgba);
        }

        staging.memory.unmapMemory();

        environment.photograph_width = image.width;
        environment.photograph_height = image.height;
        environment.photograph = create_flat(device, gpu, image.width, image.height,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);

        submit_and_wait(device, queue, pool, [&](const vk::raii::CommandBuffer &commands) {
            barrier(commands, environment.photograph, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);

            commands.copyBufferToImage(*staging.handle, *environment.photograph.handle, vk::ImageLayout::eTransferDstOptimal,
                vk::BufferImageCopy{
                    .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                    .imageExtent = {image.width, image.height, 1},
                });

            barrier(commands, environment.photograph, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead);
        });
    }

}  // namespace

// Creating

Environment create_environment(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    DescriptorHeaps &heaps,
    std::uint32_t first_slot,
    const std::filesystem::path &photograph
) {
    Environment environment;
    environment.clamp_sampler = heaps.clamp_sampler;

    // Images

    // The sky cube is written by a compute shader (storage), shrunk into its mips by blits (transfer), and sampled.
    environment.sky_cube = create_cube(device, gpu, sky_cube_size, sky_cube_mips,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled
        | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);

    environment.specular_cube = create_cube(device, gpu, specular_cube_size, specular_mips,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);

    environment.brdf_lut = create_flat(device, gpu, brdf_lut_size, brdf_lut_size,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);

    upload_photograph(environment, device, gpu, queue, pool, photograph);

    // Descriptors

    // Cubes are sampled as cubes, but written as 2D arrays of six layers, one mip at a time: a storage descriptor covers one mip level.
    EnvironmentSlots &slots = environment.slots;
    slots.sky_cube = first_slot;
    slots.sky_target = first_slot + 1;
    slots.specular_cube = first_slot + 2;
    slots.specular_targets = first_slot + 3;
    slots.brdf_lut = slots.specular_targets + specular_mips;
    slots.brdf_target = slots.brdf_lut + 1;
    slots.photograph = slots.brdf_target + 1;

    constexpr auto storage = vk::DescriptorType::eStorageImage;

    write_image_descriptor(device, heaps, slots.sky_cube, view_of(environment.sky_cube, vk::ImageViewType::eCube, 0, sky_cube_mips, 6));
    write_image_descriptor(device, heaps, slots.sky_target, view_of(environment.sky_cube, vk::ImageViewType::e2DArray, 0, 1, 6), storage);
    write_image_descriptor(device, heaps, slots.specular_cube, view_of(environment.specular_cube, vk::ImageViewType::eCube, 0, specular_mips, 6));

    for (std::uint32_t mip = 0; mip < specular_mips; ++mip) {
        write_image_descriptor(device, heaps, slots.specular_targets + mip,
            view_of(environment.specular_cube, vk::ImageViewType::e2DArray, mip, 1, 6), storage);
    }

    write_image_descriptor(device, heaps, slots.brdf_lut, view_of(environment.brdf_lut, vk::ImageViewType::e2D, 0, 1, 1));
    write_image_descriptor(device, heaps, slots.brdf_target, view_of(environment.brdf_lut, vk::ImageViewType::e2D, 0, 1, 1), storage);
    write_image_descriptor(device, heaps, slots.photograph, view_of(environment.photograph, vk::ImageViewType::e2D, 0, 1, 1));

    // The info buffer and pipelines

    // Written by compute shaders through its address, read by the CPU and the scene shader.
    environment.info = create_buffer(device, gpu, sizeof(EnvironmentInfo),
        vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    environment.mapped = static_cast<EnvironmentInfo*>(environment.info.memory.mapMemory(0, sizeof(EnvironmentInfo)));
    *environment.mapped = EnvironmentInfo{};

    environment.sky = create_compute_pipeline(device, "environment", "skyMain");
    environment.equirect = create_compute_pipeline(device, "environment", "equirectMain");
    environment.irradiance = create_compute_pipeline(device, "environment", "irradianceMain");
    environment.prefilter = create_compute_pipeline(device, "environment", "prefilterMain");
    environment.brdf = create_compute_pipeline(device, "environment", "brdfLutMain");

    // The BRDF table, once

    // The pool was created with eResetCommandBuffer, so this one can be re-recorded for every update.
    environment.commands = std::move(vk::raii::CommandBuffers(device, vk::CommandBufferAllocateInfo{
        .commandPool = *pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    })[0]);

    run(environment, device, queue, [&](const vk::raii::CommandBuffer &commands) {
        bind_descriptor_heaps(commands, heaps);

        barrier(commands, environment.brdf_lut, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);

        dispatch(commands, environment.brdf, EnvironmentPushData{.target = slots.brdf_target, .size = brdf_lut_size}, 1);

        barrier(commands, environment.brdf_lut, vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead);
    });

    return environment;
}

// Updating

void update_environment(
    Environment &environment,
    const vk::raii::Device &device,
    const vk::raii::Queue &queue,
    const DescriptorHeaps &heaps,
    SkySource source,
    glm::vec3 sun_direction
) {
    // Frames in flight may still be sampling the cubes this rewrites.
    device.waitIdle();

    // A photographed sky has its sun in the picture, so there's no separate sun light. The atmosphere's compute shader writes this itself.
    if (source == SkySource::photograph) {
        environment.mapped->sun_illuminance = glm::vec3{0.0f};
    }

    const EnvironmentSlots &slots = environment.slots;

    const EnvironmentPushData push{
        .info = environment.info.address,
        .source = slots.sky_cube,
        .sun_direction = sun_direction,
        .source_scale = photograph_nits,
        .size = sky_cube_size,
        .sampler = environment.clamp_sampler,
        .source_size = sky_cube_size,
    };

    run(environment, device, queue, [&](const vk::raii::CommandBuffer &commands) {
        bind_descriptor_heaps(commands, heaps);

        // 1. The sky into mip 0 of the sky cube. Its old contents don't matter.
        barrier(commands, environment.sky_cube, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, 0, 1);

        EnvironmentPushData sky = push;
        sky.target = slots.sky_target;

        if (source == SkySource::photograph) {
            sky.source = slots.photograph;
            dispatch(commands, environment.equirect, sky, 6);
        } else {
            dispatch(commands, environment.sky, sky, 6);
        }

        // 2. Its mip chain, by blits.
        barrier(commands, environment.sky_cube, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead, 0, 1);

        generate_sky_mips(commands, environment.sky_cube);

        // 3. Diffuse light: one workgroup sums the sky into nine coefficients.
        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *environment.irradiance);
        commands.pushDataEXT(vk::PushDataInfoEXT{.offset = 0, .data = {.address = &push, .size = sizeof(push)}});
        commands.dispatch(1, 1, 1);

        // 4. Specular light: each mip level of the specular cube for its roughness.
        barrier(commands, environment.specular_cube, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);

        for (std::uint32_t mip = 0; mip < specular_mips; ++mip) {
            EnvironmentPushData level = push;
            level.target = slots.specular_targets + mip;
            level.size = specular_cube_size >> mip;
            level.roughness = static_cast<float>(mip) / static_cast<float>(specular_mips - 1);
            dispatch(commands, environment.prefilter, level, 6);
        }

        barrier(commands, environment.specular_cube, vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead);

        // The coefficients and the sun went into the info buffer: make them visible to the CPU, which reads them once this submission is done, and to the scene's fragment shaders, which read them every frame. Waiting for the fence orders the work, but doesn't by itself make one shader's writes visible to another's reads.
        memory_barrier(commands,
            vk::PipelineStageFlagBits2::eHost | vk::PipelineStageFlagBits2::eFragmentShader,
            vk::AccessFlagBits2::eHostRead | vk::AccessFlagBits2::eShaderStorageRead);
    });
}

// Reading the irradiance on the CPU

glm::vec3 sky_irradiance(const EnvironmentInfo &info, glm::vec3 n) {
    // The same nine basis functions as environment.slang.
    const float basis[9] = {
        0.282095f,
        0.488603f * n.y,
        0.488603f * n.z,
        0.488603f * n.x,
        1.092548f * n.x * n.y,
        1.092548f * n.y * n.z,
        0.315392f * (3.0f * n.z * n.z - 1.0f),
        1.092548f * n.x * n.z,
        0.546274f * (n.x * n.x - n.y * n.y),
    };

    glm::vec3 irradiance{0.0f};
    for (std::size_t k = 0; k < 9; ++k) {
        irradiance += info.irradiance_sh[k] * basis[k];
    }

    return glm::max(irradiance, glm::vec3{0.0f});
}
```

## 8.7 Lighting from the sky: `mesh.slang`

### Why
With the sky's light prepared, the scene shader replaces Chapter 7's two-tone ambient term with the real thing.

### How
- **The shared structs** now come from `shared.slangh`.
- **Diffuse light:** `sky_irradiance` sums the nine coefficients weighted by their basis functions at the normal. That gives the irradiance in lux, and a Lambertian surface reflects base color / π of it.
- **Specular light, the split sum:** the prefiltered sky along the reflected ray, read at the mip level for the surface's roughness, times the BRDF table's `F0 × scale + bias`, looked up by view angle and roughness. Both lookups use the clamp sampler; a repeating one would wrap the table's edges into each other.
- **Splitting light between the two:** a roughness-aware Fresnel term divides the light between specular and diffuse, as in Chapter 7, and occlusion darkens both.

### Code
`game-engine/shaders/mesh.slang`:
```slang
// Draws one glTF primitive: its vertices come from the scene's vertex buffer, its place in the world from its DrawData, and its surface from its glTF material, whose textures are read from the descriptor heap. Shaded with glTF's physically based BRDF, lit by the sun, the file's lights and the sky around the scene, and written to the HDR image already exposed.

// Data shared with C++ (src/includes/shader_types.h)

#include "shared.slangh"

// The alpha mode this pipeline was built for (AlphaMode in C++): 0 opaque, 1 mask, 2 blend. A specialization constant: its value is fixed when the pipeline is created, so each pipeline's fragment shader keeps only the code its mode needs.
[vk::constant_id(0)]
const uint alpha_mode = 0;

static const uint alpha_opaque = 0;
static const uint alpha_mask = 1;

// Stage interface

// What the vertex shader hands to the rasterizer. SV_Position is the clip-space position; every other field is interpolated across the triangle.
struct VertexOutput {
    float4 position : SV_Position;
    float3 world_position : POSITION;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float3 bitangent : BINORMAL;
    float2 uv0 : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
    float4 color : COLOR;
};

// Vertex shader

// SV_VulkanVertexID is Vulkan's own gl_VertexIndex, which includes the draw's vertexOffset: each primitive's indices start at 0, and the draw adds where that primitive's vertices begin in the shared buffer.
[shader("vertex")]
VertexOutput vertexMain(uint vertex_id : SV_VulkanVertexID) {
    FrameData *frame = push.frame;
    const Vertex vertex = frame.vertices[vertex_id];
    const DrawData draw = frame.draws[push.draw_index];

    const float4 world = mul(draw.model, float4(vertex.position, 1.0));

    // Tangent and bitangent lie along the surface, so they move with the model matrix, like positions; only the normal needs the normal matrix. The bitangent is built before the transform, from glTF's rule B = cross(N, T) * w: a mirroring transform then mirrors it too.
    const float3 bitangent = cross(vertex.normal, vertex.tangent.xyz) * vertex.tangent.w;

    VertexOutput output;
    output.position = mul(frame.view_projection, world);
    output.world_position = world.xyz;
    output.normal = mul((float3x3)draw.normal_matrix, vertex.normal);
    output.tangent = mul((float3x3)draw.model, vertex.tangent.xyz);
    output.bitangent = mul((float3x3)draw.model, bitangent);
    output.uv0 = vertex.uv0;
    output.uv1 = vertex.uv1;
    output.color = vertex.color;
    return output;
}

// Material textures

// Samples a material slot: its texture, with its sampler, at its set of texture coordinates. Descriptor heap access: a handle made from an index reads that descriptor from the bound heap.
float4 sample_slot(TextureSlot slot, VertexOutput input) {
    const Texture2D texture = Texture2D.Handle(uint2(slot.texture, 0));
    const SamplerState sampler = SamplerState.Handle(uint2(slot.sampler, 0));
    const float2 uv = slot.uv_set == 0 ? input.uv0 : input.uv1;
    return texture.Sample(sampler, uv);
}

// Normals

// The direction the surface faces at this pixel, for lighting.
//   1. The interpolated vertex normal. Without normals in the file, glTF asks for flat shading: the triangle's own normal is the cross product of how the position changes across neighbouring pixels (ddx, ddy).
//   2. A normal map tilts it, per texel, within the surface's tangent frame.
//   3. On a double-sided material's back face, the surface faces the other way.
float3 surface_normal(VertexOutput input, Material material, bool front_face, bool apply_normal_map) {
    float3 normal = input.normal;

    // cross(ddy, ddx), not cross(ddx, ddy): Vulkan's screen Y points down, so this order is the one that points toward the camera.
    if (all(normal == 0.0)) {
        normal = cross(ddy(input.world_position), ddx(input.world_position));
    }

    normal = normalize(normal);

    // Texture 0 is white: no normal map.
    const bool mapped = apply_normal_map && material.normal.texture != 0;

    float3 tangent = input.tangent;
    float3 bitangent = input.bitangent;

    // Without tangents in the file, work the frame out from how position and texture coordinates change between neighbouring pixels (ddx, ddy):
    //     dp/dx = P_u * du/dx + P_v * dv/dx
    //     dp/dy = P_u * du/dy + P_v * dv/dy
    // Solving these for P_u and P_v, how position changes per unit of u and v, gives the tangent (+u) and bitangent. glTF's v runs down the image while a normal map's +Y points up, so the bitangent is -P_v. Only the directions matter, so the determinant's sign stands in for dividing by it. This can differ slightly from the MikkTSpace tangents glTF specifies, but needs no precomputation.
    if (mapped && all(tangent == 0.0)) {
        const float2 uv = material.normal.uv_set == 0 ? input.uv0 : input.uv1;
        const float3 dp_dx = ddx(input.world_position);
        const float3 dp_dy = ddy(input.world_position);
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

    // Texture coordinates that don't change across the triangle give no frame at all; the plain normal is all we have then.
    if (!mapped || all(tangent == 0.0) || all(bitangent == 0.0)) {
        return normal;
    }

    // All three axes must be unit length, or the map's tilt is scaled with them. The normal already is; the other two grow and shrink with the model matrix, and interpolation shortens them between vertices.
    tangent = normalize(tangent);
    bitangent = normalize(bitangent);

    // The map stores each component in 0..1; unpack to -1..1. normal_scale scales the tilt: X and Y only, as glTF specifies.
    float3 tangent_space = sample_slot(material.normal, input).xyz * 2.0 - 1.0;
    tangent_space.xy *= material.normal_scale;

    return normalize(tangent * tangent_space.x + bitangent * tangent_space.y + normal * tangent_space.z);
}

// The glTF BRDF

// glTF's metallic-roughness model, as its specification's Appendix B writes it. A BRDF says how much of the light arriving from one direction leaves toward another: here from the light (l) toward the viewer (v), around the half vector h between them.

static const float pi = 3.14159265;

// What shading needs to know about the surface at this pixel.
struct Surface {
    float3 base_color;
    float metallic;
    float alpha;   // roughness squared: the "alpha" of GGX
    float3 normal;
    float3 view;   // unit vector toward the camera
};

// D: the GGX (Trowbridge-Reitz) distribution of microfacet normals. Smooth surfaces have nearly all their tiny facets aligned with the normal, so D is a tall, narrow peak around h = n; rough ones spread it out.
float distribution_ggx(float n_dot_h, float alpha) {
    const float alpha2 = alpha * alpha;
    const float f = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (pi * f * f);
}

// V: Smith's height-correlated visibility, the share of facets neither in shadow nor hidden, with the BRDF's 1 / (4 n.l n.v) folded in.
float visibility_smith(float n_dot_l, float n_dot_v, float alpha) {
    const float alpha2 = alpha * alpha;
    const float from_view = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2);
    const float from_light = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2);
    const float sum = from_view + from_light;
    return sum > 0.0 ? 0.5 / sum : 0.0;
}

// The light leaving toward the viewer, in nits, from light arriving from direction `l` with illuminance `illuminance` (lux, on a surface facing it).
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

    // Metals tint their reflection with the base color and have no diffuse part. Dielectrics (everything else) reflect 4% head-on, rising to 100% at grazing angles, and the rest enters the surface and scatters back out as Lambertian diffuse light, colored by the base color.
    const float3 metal = specular * (surface.base_color + (1.0 - surface.base_color) * fresnel);
    const float3 dielectric = lerp(surface.base_color / pi, float3(specular), 0.04 + 0.96 * fresnel);
    const float3 brdf = lerp(dielectric, metal, surface.metallic);

    // Light falling at an angle spreads over more surface: the n.l factor.
    return brdf * illuminance * n_dot_l;
}

// Lights

// The direction toward a light and the illuminance it gives here, following KHR_lights_punctual. Point and spot lights fade with the square of the distance, then smoothly to nothing at `range`; spot lights also fade from the inner cone to the outer one.
float3 punctual_light(Light light, float3 position, out float3 l) {
    if (light.type == light_directional) {
        l = -light.direction;
        return light.intensity;
    }

    const float3 to_light = light.position - position;
    const float distance2 = max(dot(to_light, to_light), 1e-8);
    l = to_light * rsqrt(distance2);

    float attenuation = 1.0 / distance2;

    if (light.range > 0.0) {
        const float ratio = sqrt(distance2) / light.range;
        attenuation *= saturate(1.0 - ratio * ratio * ratio * ratio);
    }

    if (light.type == light_spot) {
        const float cone = saturate(dot(light.direction, -l) * light.spot_scale + light.spot_offset);
        attenuation *= cone * cone;
    }

    return light.intensity * attenuation;
}

// Image-based lighting

// Light from the whole sky at once, from what environment.slang prepared.

// The sky's irradiance on a surface facing `n`, in lux: its nine spherical harmonics coefficients, each weighted by its basis function at `n`.
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

// The sky's light reflected toward the viewer.
//   - Diffuse: a Lambertian surface reflects base color / pi of the irradiance falling on it.
//   - Specular, the "split sum": the light (the prefiltered sky along the reflected ray, at the mip level for this roughness) times how much the BRDF reflects overall (the table, as a scale and bias on F0).
//   - A roughness-aware Fresnel term splits the light between the two.
float3 shade_environment(Surface surface, FrameData *frame, float roughness) {
    const float n_dot_v = max(dot(surface.normal, surface.view), 1e-4);
    const float3 f0 = lerp(float3(0.04), surface.base_color, surface.metallic);
    const float3 fresnel = f0 + (max(float3(1.0 - roughness), f0) - f0) * pow(1.0 - n_dot_v, 5.0);

    const SamplerState clamped = SamplerState.Handle(uint2(frame.clamp_sampler, 0));

    const float3 diffuse_color = surface.base_color * (1.0 - surface.metallic);
    const float3 diffuse = (1.0 - fresnel) * diffuse_color * sky_irradiance(frame.environment, surface.normal) / pi;

    const TextureCube specular_cube = TextureCube.Handle(uint2(frame.specular_cube, 0));
    const float3 reflected = reflect(-surface.view, surface.normal);
    const float lod = roughness * float(frame.specular_mips - 1);
    const float3 prefiltered = specular_cube.SampleLevel(clamped, reflected, lod).rgb;

    const Texture2D brdf_lut = Texture2D.Handle(uint2(frame.brdf_lut, 0));
    const float2 brdf = brdf_lut.SampleLevel(clamped, float2(n_dot_v, roughness), 0.0).rg;
    const float3 specular = prefiltered * (f0 * brdf.x + brdf.y);

    return diffuse + specular;
}

// Fragment shader

// SV_Target: the value written to color attachment 0. SV_IsFrontFace: whether this triangle faces the camera.
[shader("fragment")]
float4 fragmentMain(VertexOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    FrameData *frame = push.frame;
    const Material material = frame.materials[frame.draws[push.draw_index].material];

    // Base color: factor x texture x vertex color. sRGB textures are decoded to linear by the sampler, so all three are linear.
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

    // Occlusion darkens creases that ambient light can't reach. Strength blends between no effect (0) and the full map (1).
    const float occlusion = 1.0 + material.occlusion_strength * (sample_slot(material.occlusion, input).r - 1.0);

    const float3 emissive = material.emissive_factor * sample_slot(material.emissive, input).rgb;

    const float3 normal = surface_normal(input, material, front_face, frame.view != view_vertex_normal);

    // The debug views show one input each. Directions are shown as colors: each component's -1..1 mapped to 0..1.
    switch (frame.view) {
        case view_base_color: return base_color;
        case view_normal:
        case view_vertex_normal: return float4(normal * 0.5 + 0.5, 1.0);
        case view_metallic: return float4(metallic.xxx, 1.0);
        case view_roughness: return float4(roughness.xxx, 1.0);
        case view_occlusion: return float4(occlusion.xxx, 1.0);
        case view_emissive: return float4(emissive, 1.0);
        default: break;
    }

    // A perfectly smooth surface would reflect a punctual light from a single point, too small for any pixel to catch; a floor on roughness keeps highlights visible.
    const Surface surface = {
        base_color.rgb,
        metallic,
        max(roughness, 0.045) * max(roughness, 0.045),
        normal,
        normalize(frame.camera_position - input.world_position),
    };

    // Direct light: the sun, then every light in the file.
    float3 radiance = shade(surface, frame.sun_direction, frame.sun_illuminance);

    for (uint i = 0; i < frame.light_count; ++i) {
        float3 l;
        const float3 illuminance = punctual_light(frame.lights[i], input.world_position, l);
        radiance += shade(surface, l, illuminance);
    }

    // Indirect light from the sky, darkened by occlusion.
    radiance += shade_environment(surface, frame, roughness) * occlusion;

    // Exposure scales nits into the tone mapper's range here, before the 16-bit HDR image could overflow. glTF defines emission in nits, but, as its spec notes many engines do, we take it as already exposed: an emissive value of 1 shows as near-white, whatever the exposure.
    return float4(radiance * frame.exposure + emissive, base_color.a);
}
```

## 8.8 The sky behind the scene: `background.slang`

### Why
Until now the sky above Sponza was a flat clear color. The sky cube holds the real one.

### How
- **Drawn between the batches:** it's a full-screen triangle at depth 0, drawn after the opaque and masked batches and before the blended one. It only covers pixels still at the far plane, and see-through surfaces then blend over it like over anything else.
- **The view direction:** the vertex shader passes each corner's clip-space position on, so each pixel receives its own. The inverse view-projection matrix turns that position, at the far plane, back into a world-space point, and the direction is from the camera to it.
- **The sun's disk** is added on top of the cube's sky. The sun's illuminance is spread over the disk's solid angle, π r² for its angular radius r, 0.27°, which makes it over a billion nits. Exposed, it's clamped below the 16-bit float limit.

### Code
`game-engine/shaders/background.slang`:
```slang
// Draws the sky behind the scene: one full-screen triangle at the far plane, depth-tested so it only covers pixels nothing else has drawn on.

#include "shared.slangh"

// Vertex shader

struct VertexOutput {
    float4 position : SV_Position;
    float2 clip : TEXCOORD0;  // this point's clip-space x and y
};

// The full-screen triangle of tonemap.slang, at depth 0: with reverse-Z, that's the far plane, so the depth test lets it through only where the depth buffer still holds its cleared 0. Its clip-space corners are also passed on: interpolated across the triangle, they arrive at each pixel as that pixel's own clip-space position.
[shader("vertex")]
VertexOutput vertexMain(uint vertex_id : SV_VulkanVertexID) {
    const float2 corner = float2((vertex_id << 1) & 2, vertex_id & 2) * 2.0 - 1.0;

    VertexOutput output;
    output.position = float4(corner, 0.0, 1.0);
    output.clip = corner;
    return output;
}

// Fragment shader

// Each pixel looks along the ray from the camera through it. Turning the pixel's clip-space position at the far plane (depth 0) back into world space gives a point on that ray; the direction is from the camera to it.
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    FrameData *frame = push.frame;

    const float4 far_point = mul(frame.inverse_view_projection, float4(input.clip, 0.0, 1.0));
    const float3 direction = normalize(far_point.xyz / far_point.w - frame.camera_position);

    const TextureCube sky = TextureCube.Handle(uint2(frame.sky_cube, 0));
    const SamplerState clamped = SamplerState.Handle(uint2(frame.clamp_sampler, 0));
    float3 radiance = sky.SampleLevel(clamped, direction, 0.0).rgb;

    // The sun's disk, which the sky cube leaves out: its illuminance spread over the tiny solid angle it covers, pi r^2 for an angular radius r.
    if (dot(direction, frame.sun_direction) > cos(frame.sun_angular_radius)) {
        const float solid_angle = 3.14159265 * frame.sun_angular_radius * frame.sun_angular_radius;
        radiance += frame.sun_illuminance / solid_angle;
    }

    // Exposed like the scene, and kept below the 16-bit float limit: the sun's disk is over a billion nits.
    return float4(min(radiance * frame.exposure, 60000.0), 1.0);
}
```

## 8.9 The sun and the meter: `daylight.h`, `daylight.cpp`, `main.cpp`

### Why
The atmosphere now works out the sun's color and the sky's light, so Chapter 7's daylight model shrinks to the sun's direction. `main` creates the environment, rebuilds it when the sky changes, and meters the exposure from it.

### How
- **`daylight.h`** keeps the solar geometry, `sun_direction_at`, and the exposure functions. The air-mass formula and the sky colors are gone.
- **Heap slots:** the resource heap gets `1 + environment_slot_count` extra slots: the HDR image's, then the environment's.
- **Rebuilding the sky:** `update_environment` runs when the sky source changes, or when the time of day changes in the simulated sky. `sky_settings` remembers what the environment was last built for.
- **The light meter** reads the light on flat ground, as in Chapter 7:
  - **the sky's part:** its irradiance on an upward-facing surface, evaluated on the CPU from the spherical harmonics,
  - **the sun's part:** the sun's illuminance times the sine of its elevation.
  - **The result:** noon gives about 117,000 lux and EV 15.5; the photographed sky about 5,000 lux and EV 11.
- **Keys:** `E` switches between the two skies. The window title shows which one is in use, with the time for the simulated one.
- **Drawing the background:** `record_frame` draws it just before the blended batch. The HDR image is cleared to black, since every pixel is now drawn over.

### Code
`game-engine/src/includes/daylight.h`:
```cpp
#pragma once

#include <glm/glm.hpp>

// The sun

// The direction toward the sun at `hours` (0 to 24, local solar time) on a midsummer day at latitude 35 degrees north, in world space: -Z is north, +X east, +Y up. The atmosphere (environment.slang) works out how much of its light reaches the ground.
glm::vec3 sun_direction_at(float hours);

// The sun's angular radius as seen from Earth: half a degree across.
constexpr float sun_angular_radius = 0.004654f;

// Exposure

// A camera's exposure as one number, EV100: the exposure value at ISO 100. Each step of 1 halves the light that reaches the sensor. Sunny noon is about 15, a lit room about 7, moonlight about -2.

// The EV100 an incident light meter suggests for `illuminance` lux. Photo meters use the calibration constant C = 250: EV100 = log2(E * 100 / C).
float metered_ev100(float illuminance);

// The factor that scales scene brightness (nits) into the tone mapper's range: brightness 1.2 * 2^EV100 maps to 1, where a sensor saturates.
float exposure_from_ev100(float ev100);
```

`game-engine/src/daylight.cpp`:
```cpp
#include "includes/daylight.h"

#include <algorithm>
#include <cmath>

namespace {

    // The place and the date

    constexpr float latitude = glm::radians(35.0f);     // north of the equator
    constexpr float declination = glm::radians(23.4f);  // the sun's, at the June solstice

}  // namespace

// The sun

// Solar geometry. The hour angle is how far the earth has turned since solar noon, 15 degrees per hour. With latitude phi and declination delta, the direction toward the sun in (east, north, up) coordinates is
//     east  = -cos(delta) sin(h)
//     north =  cos(phi) sin(delta) - sin(phi) cos(delta) cos(h)
//     up    =  sin(phi) sin(delta) + cos(phi) cos(delta) cos(h)
glm::vec3 sun_direction_at(float hours) {
    const float hour_angle = glm::radians(15.0f * (hours - 12.0f));

    const float east = -std::cos(declination) * std::sin(hour_angle);
    const float north = std::cos(latitude) * std::sin(declination)
        - std::sin(latitude) * std::cos(declination) * std::cos(hour_angle);
    const float up = std::sin(latitude) * std::sin(declination)
        + std::cos(latitude) * std::cos(declination) * std::cos(hour_angle);

    // World axes: +X east, +Y up, -Z north.
    return glm::normalize(glm::vec3{east, up, -north});
}

// Exposure

float metered_ev100(float illuminance) {
    // A tiny floor keeps log2 finite in total darkness.
    return std::log2(std::max(illuminance, 1e-4f) * 100.0f / 250.0f);
}

float exposure_from_ev100(float ev100) {
    return 1.0f / (1.2f * std::exp2(ev100));
}
```

`game-engine/src/main.cpp`:
```cpp
#include "includes/buffer.h"
#include "includes/camera.h"
#include "includes/daylight.h"
#include "includes/descriptor_heap.h"
#include "includes/environment.h"
#include "includes/pipeline.h"
#include "includes/scene.h"
#include "includes/sdl.h"
#include "includes/shader_types.h"
#include "includes/swapchain.h"
#include "includes/texture.h"
#include "includes/vulkan_setup.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <limits>
#include <print>
#include <span>
#include <string>
#include <vector>

namespace {

    // Frames in flight

    // How many frames the CPU may record ahead of the GPU.
    constexpr std::size_t frames_in_flight = 2;

    constexpr std::uint64_t no_timeout = std::numeric_limits<std::uint64_t>::max();

    // What each in-flight frame needs for itself. `data` holds this frame's FrameData; the GPU may still be reading the other frame's while the CPU writes this one.
    struct Frame {
        vk::raii::CommandBuffer commands = nullptr;
        vk::raii::Semaphore image_acquired = nullptr;  // swapchain image is ready to draw into
        vk::raii::Fence done = nullptr;                // GPU finished this frame's commands
        Buffer data;                                   // one FrameData, host-visible
        FrameData *mapped = nullptr;                   // `data`, mapped for the CPU to write
    };

    // Recording a frame

    // Moves `image` between layouts, and makes the `dst` work wait for the `src` work. `aspect` is which part of the image: its color, or its depth.
    void transition(
        const vk::raii::CommandBuffer &commands,
        vk::Image image,
        vk::ImageLayout from,
        vk::ImageLayout to,
        vk::PipelineStageFlags2 src_stage,
        vk::AccessFlags2 src_access,
        vk::PipelineStageFlags2 dst_stage,
        vk::AccessFlags2 dst_access,
        vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor
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
                .aspectMask = aspect,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        commands.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier,
        });
    }

    // The three alpha modes, in the order they're drawn: solid surfaces first, so see-through ones blend over everything behind them.
    constexpr std::array alpha_modes{AlphaMode::opaque, AlphaMode::mask, AlphaMode::blend};

    // glTF's front faces wind counter-clockwise, seen from the front. Our projection's Y flip (see camera.cpp) only undoes the difference between OpenGL's upward Y and Vulkan's downward one, so on screen they still wind counter-clockwise. A mirroring transform reverses that.
    constexpr vk::FrontFace front_face = vk::FrontFace::eCounterClockwise;
    constexpr vk::FrontFace mirrored_front_face = vk::FrontFace::eClockwise;

    // What to draw: every primitive draw in a scene, the frame's data, and how to finish the frame. `batches` lists draw indices per alpha mode, in drawing order.
    struct DrawList {
        vk::Buffer index_buffer;
        vk::DeviceAddress frame = 0;   // this frame's FrameData
        std::uint32_t hdr_slot = 0;    // resource heap slot of swapchain.hdr
        View view = View::lit;
        std::span<const Primitive> primitives;
        std::span<const MeshDraw> mesh_draws;
        std::span<const SceneMaterial> scene_materials;
        std::array<std::span<const std::uint32_t>, alpha_modes.size()> batches;
    };

    // Records a frame in two passes:
    //   1. the scene, into the HDR image: clear color and depth, then draw each alpha mode's batch with that mode's pipeline, with the sky drawn behind everything solid before the see-through batch,
    //   2. tone mapping, from the HDR image into the swapchain image, which is then ready to present.
    void record_frame(
        const vk::raii::CommandBuffer &commands,
        const Swapchain &swapchain,
        std::uint32_t image_index,
        std::span<const vk::raii::Pipeline> pipelines,
        const vk::raii::Pipeline &background_pipeline,
        const vk::raii::Pipeline &tonemap_pipeline,
        const DescriptorHeaps &heaps,
        const DrawList &draws
    ) {
        const vk::Image image = swapchain.images[image_index];
        const vk::Image hdr = *swapchain.hdr.handle;

        commands.reset();
        commands.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        // Pass 1: the scene

        // The HDR image is shared by the frames in flight, like the depth buffer, so this also waits for the previous frame's tone mapping to finish reading it before this frame clears it.
        transition(commands, hdr,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite
        );

        transition(commands, *swapchain.depth.handle,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal,
            vk::PipelineStageFlagBits2::eLateFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::ImageAspectFlagBits::eDepth
        );

        // Every pixel is drawn over, by the scene or the sky; clearing is just cheaper than loading what was there.
        const vk::RenderingAttachmentInfo hdr_attachment{
            .imageView = *swapchain.hdr.view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = std::array{0.0f, 0.0f, 0.0f, 1.0f}}},
        };

        // Reverse-Z: 0 is the far plane. Depth is only needed while drawing this frame, so it isn't stored afterwards.
        const vk::RenderingAttachmentInfo depth_attachment{
            .imageView = *swapchain.depth.view,
            .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .clearValue = vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = 0.0f}},
        };

        const vk::Rect2D whole_image{.offset = {0, 0}, .extent = swapchain.extent};

        commands.beginRendering(vk::RenderingInfo{
            .renderArea = whole_image,
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &hdr_attachment,
            .pDepthAttachment = &depth_attachment,
        });

        // Every texture and sampler the shaders read comes from these two heaps. They stay bound when the pipeline changes, and for the second pass.
        bind_descriptor_heaps(commands, heaps);

        // The pipelines leave these dynamic; they cover the whole image.
        commands.setViewport(0, vk::Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(swapchain.extent.width),
            .height = static_cast<float>(swapchain.extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        commands.setScissor(0, whole_image);

        // One index buffer for the whole scene. Indices go through the GPU's fixed-function index fetch, which also lets it reuse vertices shared between neighbouring triangles.
        commands.bindIndexBuffer(draws.index_buffer, 0, vk::IndexType::eUint32);

        // One draw per primitive per node, batch by batch. Push data says where the frame's data is and which DrawData to use; the primitive's index range and vertex offset go to drawIndexed.
        for (std::size_t mode = 0; mode < alpha_modes.size(); ++mode) {
            // The sky goes in once everything solid is drawn: it only covers pixels still at the far plane. See-through surfaces then blend over it like over anything else.
            if (alpha_modes[mode] == AlphaMode::blend) {
                commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *background_pipeline);

                const PushData push{.frame = draws.frame, .draw_index = 0};
                commands.pushDataEXT(vk::PushDataInfoEXT{
                    .offset = 0,
                    .data = {.address = &push, .size = sizeof(push)},
                });

                commands.draw(3, 1, 0, 0);
            }

            commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines[mode]);

            for (const std::uint32_t i : draws.batches[mode]) {
                const MeshDraw &mesh_draw = draws.mesh_draws[i];
                const Primitive &primitive = draws.primitives[mesh_draw.primitive];
                const SceneMaterial &material = draws.scene_materials[primitive.material];

                // Single-sided surfaces are invisible from behind, so the GPU can skip their back faces before running the fragment shader.
                commands.setCullMode(material.double_sided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack);
                commands.setFrontFace(mesh_draw.mirrored ? mirrored_front_face : front_face);

                const PushData push{.frame = draws.frame, .draw_index = i};

                commands.pushDataEXT(vk::PushDataInfoEXT{
                    .offset = 0,
                    .data = {.address = &push, .size = sizeof(push)},
                });

                commands.drawIndexed(primitive.index_count, 1, primitive.first_index, primitive.vertex_offset, 0);
            }
        }

        commands.endRendering();

        // Pass 2: tone mapping

        // The scene is finished: the tone-mapping shader may read it now.
        transition(commands, hdr,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead
        );

        // Undefined: every pixel is about to be overwritten.
        transition(commands, image,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite
        );

        // The full-screen triangle writes every pixel, so there's nothing to clear or load first.
        const vk::RenderingAttachmentInfo color_attachment{
            .imageView = *swapchain.views[image_index],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eDontCare,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };

        commands.beginRendering(vk::RenderingInfo{
            .renderArea = whole_image,
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment,
        });

        commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *tonemap_pipeline);

        const TonemapPushData push{.hdr_image = draws.hdr_slot, .view = draws.view};

        commands.pushDataEXT(vk::PushDataInfoEXT{
            .offset = 0,
            .data = {.address = &push, .size = sizeof(push)},
        });

        commands.draw(3, 1, 0, 0);
        commands.endRendering();

        transition(commands, image,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone
        );

        commands.end();
    }

    // Events

    // What the keyboard controls, besides the camera.
    struct Settings {
        View view = View::lit;
        SkySource sky = SkySource::atmosphere;
        float hours = 10.0f;                    // time of day, 0 to 24
        float exposure_compensation = 0.0f;     // stops brighter (+) or darker (-) than metered
    };

    // The views' names, in View's order, for the window title.
    constexpr std::array view_names{
        "Lit", "Base color", "Normal", "Vertex normal", "Metallic", "Roughness", "Occlusion", "Emissive",
    };

    // Handles every pending event and fills in `input` for this frame. False once the window was closed or Escape pressed.
    //   1-8   pick the view
    //   e     switch between the simulated sky and the photographed one
    //   [ ]   time of day, a quarter of an hour earlier or later
    //   - =   exposure, half a stop darker or brighter: like a camera's exposure compensation, + is brighter
    // Holding a key repeats it.
    bool poll_events(SDL_Window *window, CameraInput &input, Settings &settings) {
        input = CameraInput{};
        SDL_Event event;

        while (SDL_PollEvent(&event)) {
            const bool escape = event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE;

            if (event.type == SDL_EVENT_QUIT || escape) {
                return false;
            }

            if (event.type == SDL_EVENT_KEY_DOWN) {
                const SDL_Keycode key = event.key.key;

                // SDLK_1 to SDLK_8 are consecutive key codes.
                if (key >= SDLK_1 && key < SDLK_1 + view_names.size()) {
                    settings.view = static_cast<View>(key - SDLK_1);
                } else if (key == SDLK_E) {
                    settings.sky = settings.sky == SkySource::atmosphere ? SkySource::photograph : SkySource::atmosphere;
                } else if (key == SDLK_LEFTBRACKET) {
                    settings.hours = std::fmod(settings.hours + 23.75f, 24.0f);
                } else if (key == SDLK_RIGHTBRACKET) {
                    settings.hours = std::fmod(settings.hours + 0.25f, 24.0f);
                } else if (key == SDLK_MINUS) {
                    settings.exposure_compensation -= 0.5f;
                } else if (key == SDLK_EQUALS) {
                    settings.exposure_compensation += 0.5f;
                }
            }

            if (event.type == SDL_EVENT_MOUSE_MOTION) {
                input.mouse_delta += glm::vec2{event.motion.xrel, event.motion.yrel};
            } else if (event.type == SDL_EVENT_MOUSE_WHEEL) {
                input.wheel += event.wheel.y;
            }
        }

        // Which buttons are held right now.
        const SDL_MouseButtonFlags buttons = SDL_GetMouseState(nullptr, nullptr);
        input.right_button = (buttons & SDL_BUTTON_RMASK) != 0;
        input.left_button = (buttons & SDL_BUTTON_LMASK) != 0;
        input.middle_button = (buttons & SDL_BUTTON_MMASK) != 0;

        // While a button is held, relative mode hides the cursor and keeps reporting movement, so a drag can't run into the edge of the screen.
        const bool dragging = input.right_button || input.left_button || input.middle_button;

        if (dragging != SDL_GetWindowRelativeMouseMode(window)) {
            SDL_SetWindowRelativeMouseMode(window, dragging);
        }

        return true;
    }

}  // namespace

int main() {
    try {
        // Window and instance

        SdlContext sdl;
        const int version = SDL_GetVersion();
        std::println("SDL {}.{}.{} on {}", SDL_VERSIONNUM_MAJOR(version), SDL_VERSIONNUM_MINOR(version),
            SDL_VERSIONNUM_MICRO(version), SdlContext::video_driver());

        // Loads libvulkan at runtime, so nothing has to link against it.
        vk::raii::Context context;

#ifdef NDEBUG
        const bool validation = false;
#else
        const bool validation = validation_layer_available(context);
#endif
        std::println("Validation layer {}", validation ? "on" : "off");

        // Declaration order matters: each object is destroyed before the ones above it. The window comes first: creating it loads Vulkan into SDL, which required_vulkan_extensions() needs.
        Window window = make_vulkan_window(1920, 1080, "game-engine", true);

        vk::raii::Instance instance = create_instance(context, SdlContext::required_vulkan_extensions(), validation);
        vk::raii::DebugUtilsMessengerEXT messenger = validation
            ? create_debug_messenger(instance)
            : vk::raii::DebugUtilsMessengerEXT(nullptr);

        vk::raii::SurfaceKHR surface = create_surface(instance, window.get());

        // GPU, device and swapchain

        std::println("GPUs:");
        std::optional<GpuChoice> gpu = pick_gpu(instance, surface);

        if (!gpu) {
            std::println(stderr, "No GPU has Vulkan 1.4 and the descriptor heap, and can present to this window");
            return EXIT_FAILURE;
        }

        std::println("Using {}", gpu->device.getProperties().deviceName.data());
        print_descriptor_heap_properties(*gpu);

        vk::raii::Device device = create_device(*gpu);
        vk::raii::Queue queue = device.getQueue(gpu->queue_family, 0);
        Swapchain swapchain = create_swapchain(device, *gpu, surface, window.get());

        // Pipelines

        // One per alpha mode, in alpha_modes' order. recreate_swapchain() picks the same formats again, so the pipelines stay valid across resizes. The mesh pipelines draw into the HDR image; tone mapping writes the swapchain image.
        std::vector<vk::raii::Pipeline> pipelines;

        for (const AlphaMode mode : alpha_modes) {
            pipelines.push_back(create_mesh_pipeline(device, hdr_format, depth_format, mode));
        }

        // The sky draws into the HDR image, behind the scene; tone mapping writes the swapchain image.
        const vk::raii::Pipeline background_pipeline = create_fullscreen_pipeline(device, "background", hdr_format, depth_format);
        const vk::raii::Pipeline tonemap_pipeline = create_fullscreen_pipeline(device, "tonemap", swapchain.format);

        // Per-frame resources

        // eResetCommandBuffer lets us re-record each frame's command buffer.
        vk::raii::CommandPool command_pool(device, vk::CommandPoolCreateInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = gpu->queue_family,
        });

        vk::raii::CommandBuffers command_buffers(device, vk::CommandBufferAllocateInfo{
            .commandPool = *command_pool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = frames_in_flight,
        });

        std::vector<Frame> frames;
        for (vk::raii::CommandBuffer &commands : command_buffers) {
            // Host-coherent: the CPU's writes reach the GPU without a flush.
            Buffer data = create_buffer(device, *gpu, sizeof(FrameData),
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            auto *mapped = static_cast<FrameData*>(data.memory.mapMemory(0, sizeof(FrameData)));

            frames.push_back(Frame{
                .commands = std::move(commands),
                .image_acquired = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
                // Start signalled, so the first wait on each frame returns straight away.
                .done = vk::raii::Fence(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}),
                .data = std::move(data),
                .mapped = mapped,
            });
        }

        // Scene

        // The glTF file to draw, under lecture-md/game-engine/assets.
        const std::filesystem::path scene_file = std::filesystem::path(ASSET_DIR) / "Sponza/Sponza.gltf";

        const std::uint64_t load_start = SDL_GetTicksNS();
        const Scene scene = load_gltf(scene_file);
        std::println("Loaded {}: {} vertices, {} triangles, {} primitives, {} draws, {} materials, {} images",
            scene_file.filename().string(), scene.vertices.size(), scene.indices.size() / 3,
            scene.primitives.size(), scene.draws.size(), scene.materials.size(), scene.images.size());

        // Each draw's matrices. The normal matrix is the transposed inverse of the model matrix: under non-uniform scale, transforming a normal by the model matrix itself would tilt it off the surface.
        std::vector<DrawData> draw_data;

        for (const MeshDraw &draw : scene.draws) {
            draw_data.push_back(DrawData{
                .model = draw.model,
                .normal_matrix = glm::transpose(glm::inverse(draw.model)),
                .material = scene.primitives[draw.primitive].material,
            });
        }

        // Vertices and draw data are read through pointers; indices go to the GPU's index fetch, so that buffer is an index buffer.
        const Buffer vertex_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.vertices)), vk::BufferUsageFlagBits::eShaderDeviceAddress);
        const Buffer index_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.indices)), vk::BufferUsageFlagBits::eIndexBuffer);
        const Buffer draw_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(draw_data)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        // Most files have no lights, and a buffer can't be empty: then there's no buffer, and the shader's light count is 0.
        const Buffer light_buffer = scene.lights.empty() ? Buffer{} : upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.lights)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        std::println("Lights: {} from the file, plus the sun", scene.lights.size());

        // Textures and materials

        // Decode every image, upload them with mipmaps, and describe them in the descriptor heap. Texture 0 is white; scene image i is texture i + 1.
        const std::uint64_t texture_start = SDL_GetTicksNS();
        const std::vector<Texture> textures = create_scene_textures(device, *gpu, queue, command_pool, scene);
        // After the textures: one slot for the HDR image, then the environment's.
        DescriptorHeaps heaps = create_descriptor_heaps(device, *gpu, queue, command_pool, textures, scene.samplers,
            1 + environment_slot_count);
        const auto hdr_slot = static_cast<std::uint32_t>(textures.size());

        // The HDR image is recreated with the swapchain, so its descriptor is rewritten every time: after this, only while the GPU is idle.
        const auto describe_hdr = [&] {
            write_image_descriptor(device, heaps, hdr_slot, vk::ImageViewCreateInfo{
                .image = *swapchain.hdr.handle,
                .viewType = vk::ImageViewType::e2D,
                .format = hdr_format,
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            });
        };

        describe_hdr();

        // recreate_swapchain() waits for the GPU to go idle, so the slot is free to rewrite straight afterwards.
        const auto resize = [&] {
            recreate_swapchain(swapchain, device, *gpu, surface, window.get());
            describe_hdr();
        };

        // The environment

        const std::uint64_t environment_start = SDL_GetTicksNS();
        Environment environment = create_environment(device, *gpu, queue, command_pool, heaps, hdr_slot + 1,
            std::filesystem::path(ASSET_DIR) / "environments/kloppenheim_06_puresky_2k.hdr");

        std::println("Environment: {:.0f} ms", static_cast<double>(SDL_GetTicksNS() - environment_start) * 1e-6);

        std::println("Textures: {} in {:.0f} ms (whole load {:.0f} ms)", textures.size(),
            static_cast<double>(SDL_GetTicksNS() - texture_start) * 1e-6,
            static_cast<double>(SDL_GetTicksNS() - load_start) * 1e-6);

        // Heap indices are one past the scene's: image i is texture i + 1 and sampler i is sampler i + 1, so "none" (-1) becomes 0, the white texture or the default sampler.
        const auto slot = [](const TextureRef &ref) {
            return TextureSlot{
                .texture = static_cast<std::uint32_t>(ref.image + 1),
                .sampler = static_cast<std::uint32_t>(ref.sampler + 1),
                .uv_set = ref.uv_set,
            };
        };

        std::vector<Material> materials;

        for (const SceneMaterial &material : scene.materials) {
            materials.push_back(Material{
                .base_color_factor = material.base_color_factor,
                .emissive_factor = material.emissive_factor,
                .metallic_factor = material.metallic_factor,
                .roughness_factor = material.roughness_factor,
                .normal_scale = material.normal_scale,
                .occlusion_strength = material.occlusion_strength,
                .alpha_cutoff = material.alpha_cutoff,
                .double_sided = material.double_sided ? 1u : 0u,
                .base_color = slot(material.base_color),
                .metallic_roughness = slot(material.metallic_roughness),
                .normal = slot(material.normal),
                .occlusion = slot(material.occlusion),
                .emissive = slot(material.emissive),
            });
        }

        const Buffer material_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(materials)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        // Draw indices by alpha mode. Opaque and masked draws can go in any order, so their batches are fixed; blended ones are sorted by distance every frame, below.
        std::array<std::vector<std::uint32_t>, alpha_modes.size()> batches;

        for (std::uint32_t i = 0; i < scene.draws.size(); ++i) {
            const AlphaMode mode = scene.materials[scene.primitives[scene.draws[i].primitive].material].alpha_mode;
            batches[static_cast<std::size_t>(mode)].push_back(i);
        }

        std::println("Draws: {} opaque, {} masked, {} blended", batches[0].size(), batches[1].size(), batches[2].size());

        // Spawns at the origin, looking down -Z.
        FlyCamera camera;
        CameraInput input;
        Settings settings;
        Settings shown_settings{.hours = -1.0f};  // what the title shows; differs at first
        Settings sky_settings{.hours = -1.0f};    // what the environment was built for

        std::uint64_t previous_ticks = SDL_GetTicksNS();

        // Frame loop

        std::uint64_t frame_count = 0;

        while (poll_events(window.get(), input, settings)) {
            // Minimised: nothing to draw into, so sleep until something happens.
            int width = 0;
            int height = 0;
            SDL_GetWindowSizeInPixels(window.get(), &width, &height);

            if (width == 0 || height == 0 || (SDL_GetWindowFlags(window.get()) & SDL_WINDOW_MINIMIZED)) {
                SDL_WaitEvent(nullptr);
                continue;
            }

            if (width != swapchain.window_width || height != swapchain.window_height) {
                resize();
            }

            // Update

            // Seconds since the last frame, so movement doesn't depend on frame rate.
            const std::uint64_t ticks = SDL_GetTicksNS();
            const float seconds = static_cast<float>(ticks - previous_ticks) * 1e-9f;
            previous_ticks = ticks;

            update_camera(camera, input, seconds);

            const float aspect = static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height);

            // Blending mixes with what's already drawn, so see-through draws go back to front: farthest from the camera first. Sorting by each draw's center is approximate, but right for separate objects.
            std::vector<std::uint32_t> &blended = batches[static_cast<std::size_t>(AlphaMode::blend)];

            std::ranges::sort(blended, std::ranges::greater{}, [&](std::uint32_t i) {
                const glm::vec3 offset = scene.draws[i].center - camera.position;
                return glm::dot(offset, offset);
            });

            // The sky: rebuilt whenever its source changes, or the time of day moves the sun in the simulated one. That takes a few milliseconds and waits for the GPU, which is fine for a key press.
            const glm::vec3 sun_direction = sun_direction_at(settings.hours);
            const bool sky_moved = settings.sky == SkySource::atmosphere && settings.hours != sky_settings.hours;

            if (settings.sky != sky_settings.sky || sky_moved) {
                update_environment(environment, device, queue, heaps, settings.sky, sun_direction);
                sky_settings = settings;
            }

            // Light and exposure. The meter reads the light falling on flat ground: the sky's irradiance on an upward-facing surface, plus the sun's share at its angle (Rec. 709 luminance of each). Compensation works like a camera's: +1 is a stop brighter, which means a lower EV (EV measures the light the camera expects).
            const auto luminance = [](glm::vec3 c) { return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b; };
            const glm::vec3 sun_illuminance = environment.mapped->sun_illuminance;
            const float ground_illuminance = luminance(sky_irradiance(*environment.mapped, {0.0f, 1.0f, 0.0f}))
                + luminance(sun_illuminance) * std::max(sun_direction.y, 0.0f);

            const float ev100 = std::clamp(metered_ev100(ground_illuminance), -2.0f, 16.0f) - settings.exposure_compensation;
            const float exposure = exposure_from_ev100(ev100);

            // The title shows the view, the sky, the time and the exposure, whenever one changes.
            if (settings.view != shown_settings.view || settings.sky != shown_settings.sky
                || settings.hours != shown_settings.hours
                || settings.exposure_compensation != shown_settings.exposure_compensation) {
                const auto minutes = static_cast<int>(std::lround(settings.hours * 60.0f));
                const std::string sky = settings.sky == SkySource::atmosphere
                    ? std::format("sky {:02}:{:02}", minutes / 60, minutes % 60)
                    : std::string("photographed sky");
                const std::string title = std::format("game-engine: {}, {}, EV {:.1f}",
                    view_names[static_cast<std::size_t>(settings.view)], sky, ev100);

                SDL_SetWindowTitle(window.get(), title.c_str());
                shown_settings = settings;
            }

            // Render

            Frame &frame = frames[frame_count % frames_in_flight];

            // 1. Wait until the GPU is done with this frame's command buffer and
            //    data from last time, then write this frame's data.
            (void)device.waitForFences(*frame.done, vk::True, no_timeout);

            const glm::mat4 view_projection = camera.projection(aspect) * camera.view();

            *frame.mapped = FrameData{
                .view_projection = view_projection,
                .inverse_view_projection = glm::inverse(view_projection),
                .vertices = vertex_buffer.address,
                .draws = draw_buffer.address,
                .materials = material_buffer.address,
                .lights = light_buffer.address,
                .environment = environment.info.address,
                .camera_position = camera.position,
                .exposure = exposure,
                .sun_direction = sun_direction,
                .light_count = static_cast<std::uint32_t>(scene.lights.size()),
                .sun_illuminance = sun_illuminance,
                .view = settings.view,
                .sky_cube = environment.slots.sky_cube,
                .specular_cube = environment.slots.specular_cube,
                .brdf_lut = environment.slots.brdf_lut,
                .clamp_sampler = environment.clamp_sampler,
                .specular_mips = specular_mips,
                .sun_angular_radius = sun_angular_radius,
            };

            const DrawList draws{
                .index_buffer = *index_buffer.handle,
                .frame = frame.data.address,
                .hdr_slot = hdr_slot,
                .view = settings.view,
                .primitives = scene.primitives,
                .mesh_draws = scene.draws,
                .scene_materials = scene.materials,
                .batches = {batches[0], batches[1], batches[2]},
            };

            // 2. Ask the swapchain for an image; `image_acquired` is signalled once it's free.
            const auto acquired = swapchain.handle.acquireNextImage(no_timeout, *frame.image_acquired);

            if (acquired.result == vk::Result::eErrorOutOfDateKHR) {
                resize();
                continue;
            }

            const std::uint32_t image_index = acquired.value;
            device.resetFences(*frame.done);

            // 3. Record and submit: wait for the image, draw, signal `rendered` and `done`.
            record_frame(frame.commands, swapchain, image_index, pipelines, background_pipeline, tonemap_pipeline, heaps, draws);

            const vk::SemaphoreSubmitInfo wait_info{
                .semaphore = *frame.image_acquired,
                .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            };
            const vk::CommandBufferSubmitInfo command_info{.commandBuffer = *frame.commands};
            const vk::SemaphoreSubmitInfo signal_info{
                .semaphore = *swapchain.rendered[image_index],
                .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            };

            queue.submit2(vk::SubmitInfo2{
                .waitSemaphoreInfoCount = 1,
                .pWaitSemaphoreInfos = &wait_info,
                .commandBufferInfoCount = 1,
                .pCommandBufferInfos = &command_info,
                .signalSemaphoreInfoCount = 1,
                .pSignalSemaphoreInfos = &signal_info,
            }, *frame.done);

            // 4. Present once `rendered` is signalled.
            const vk::Semaphore rendered = *swapchain.rendered[image_index];
            const vk::SwapchainKHR swapchain_handle = *swapchain.handle;

            const vk::Result presented = queue.presentKHR(vk::PresentInfoKHR{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &rendered,
                .swapchainCount = 1,
                .pSwapchains = &swapchain_handle,
                .pImageIndices = &image_index,
            });

            if (presented == vk::Result::eErrorOutOfDateKHR || presented == vk::Result::eSuboptimalKHR) {
                resize();
            }

            ++frame_count;
        }

        // Shutdown

        // Everything above is destroyed on the way out of this scope; the GPU must be idle first.
        device.waitIdle();
        std::println("Presented {} frames", frame_count);
    } catch (const std::exception &e) {
        std::println(stderr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

## 8.10 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The terminal** shows a new `Environment: …` line after loading; building the environment takes about 40 ms in release, loading the HDR file included.
- **The window** opens at 10:00 in the simulated sky: a deep blue sky above the courtyard, and soft blue light in the shade.
- **`]` and `[`** move through the day, and the whole sky follows:

  | Time | Sun's light (R, G, B) | Light on the ground | EV |
  |---|---|---|---|
  | 06:00 | 103,000 / 75,000 / 45,000 lux | 22,000 lux | 13.1 |
  | 10:00 | 121,000 / 111,000 / 98,000 lux | 104,000 lux | 15.3 |
  | 12:00 | 121,000 / 113,000 / 100,000 lux | 117,000 lux | 15.5 |
  | 19:00 | 45,000 / 12,000 / 850 lux | 2,300 lux | 9.9 |
  | 19:30 | below the horizon | 88 lux (twilight) | 5.1 |
  | 20:00 | below the horizon | none | −2.0 |

  The sun is nearly white at noon, and turns orange, then deep red, as it sinks through more air. After sunset, the twilight sky is pink and purple, and the night is black.
- **Looking at the sun** shows its small, bright disk, with the haze's glow around it.
- **`E`** switches to the photographed sky: clouds above the courtyard, and a cooler, softer light with no direct sun.
- **Reflections:** in MetalRoughSpheres the smooth metal spheres mirror the horizon, and each column to the right blurs it more.
- **No `[validation …]` lines.**

Next, in Chapter 9, the sky's light stops reaching into every crease: ambient occlusion, computed from a depth prepass, darkens corners, gaps and contact points by how much of the sky they can see.

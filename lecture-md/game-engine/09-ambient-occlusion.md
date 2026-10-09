# Chapter 9: Ambient occlusion

By the end of this chapter, the sky's light no longer reaches into places it can't. The insides of corners, the gaps behind curtains, the undersides of arches and the ground where a column meets it all darken by how much of the sky they can see. This is **ambient occlusion** (AO), and it's what makes the image-based lighting from Chapter 8 look grounded instead of glowing.

The method is **ground-truth ambient occlusion** (GTAO, Jimenez et al. 2016). It works from the depth buffer and the surfaces' normals, in a compute shader, every frame. For each pixel it searches the depth buffer around it for the **horizon** in a few directions: the highest point that blocks the sky. The visible part of the sky between the horizons is then integrated exactly, with the same cosine weighting the irradiance uses. The result is a **visibility**, from 0 (fully enclosed) to 1 (open sky), and a **bent normal**: the average direction of the open sky.

The frame now runs in four passes:
1. **a depth prepass:** every solid surface's depth and vertex normal, with no shading,
2. **ambient occlusion:** the horizon search, then a blur, in compute shaders,
3. **the lighting:** the full shading from Chapter 8, against the prepass's depth, now reading the AO,
4. **tone mapping,** as before.

Two new keys: `O` switches ambient occlusion off and on to compare, and `9` shows the AO on its own.

This chapter builds on [Chapter 8](08-image-based-lighting.md).

## 9.1 The new images: `swapchain.h`, `swapchain.cpp`

### Why
AO is computed from the depth of the whole frame, so the depth has to exist before any surface is shaded. Until now, depth was only a by-product of drawing. It was written and tested while shading, then thrown away. It now gets its own pass, and four images to go with it.

### How
- **The depth buffer** is now also **sampled**: the AO shader reads it like a texture. Its usage gains `eSampled`.
- **A normals image,** `R16G16Sfloat`. The search needs each pixel's surface direction as well as its depth. A unit vector fits in two 16-bit numbers with **octahedral encoding** (9.3), half the size of three.
- **Two AO images,** `R16G16B16A16Sfloat`: the bent normal in RGB, the visibility in A. Compute shaders write them as **storage images** and the lighting pass samples them, so both usages are set. Two are needed because the blur reads one while writing the other.
- **All are the window's size** and are rebuilt with the swapchain. Each is written from scratch every frame before anything reads it, so, like the HDR image, frames in flight can share them.

### Code
`game-engine/src/includes/swapchain.h`:
```cpp
#pragma once

#include "includes/image.h"
#include "includes/vulkan_setup.h"

#include <vector>

// 32-bit float depth: the precision reverse-Z depth needs (see camera.cpp).
constexpr vk::Format depth_format = vk::Format::eD32Sfloat;

// 16-bit floats per channel for the scene before tone mapping: enough range
// for sunlit highlights many times brighter than white (up to 65504).
constexpr vk::Format hdr_format = vk::Format::eR16G16B16A16Sfloat;

// The prepass's normals: two 16-bit floats, a unit vector in octahedral
// encoding (see mesh.slang).
constexpr vk::Format normal_format = vk::Format::eR16G16Sfloat;

// Ambient occlusion: the bent normal in RGB and the visibility in A.
constexpr vk::Format ao_format = vk::Format::eR16G16B16A16Sfloat;

// The window's images, plus what we need per image to draw into them.
// Members are destroyed bottom-up, so the views and semaphores go before
// the swapchain that owns the images.
struct Swapchain {
    vk::raii::SwapchainKHR handle = nullptr;
    vk::Format format = vk::Format::eUndefined;
    vk::Extent2D extent;

    // Framebuffer size this swapchain was made for; a change means rebuild.
    int window_width = 0;
    int window_height = 0;

    std::vector<vk::Image> images;               // owned by `handle`
    std::vector<vk::raii::ImageView> views;      // one per image
    std::vector<vk::raii::Semaphore> rendered;   // one per image, signalled when drawing is done

    // Images the size of the swapchain images, which every frame writes from
    // scratch before reading, so frames in flight can share them:
    //   depth    the depth prepass writes it; ambient occlusion and the
    //            lighting pass then read it
    //   normals  the prepass's normals, for ambient occlusion
    //   ao       ambient occlusion, written and blurred by compute shaders
    //   ao_blur  the blur's halfway point
    //   hdr      the lit scene, which tone mapping writes to the swapchain image
    Image depth;
    Image normals;
    Image ao;
    Image ao_blur;
    Image hdr;
};

Swapchain create_swapchain(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::SurfaceKHR& surface,
    SDL_Window* window
);

// Rebuilds `swapchain` for the window's current size (after a resize).
// Waits for the GPU to go idle first.
void recreate_swapchain(
    Swapchain& swapchain,
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::SurfaceKHR& surface,
    SDL_Window* window
);
```

In `game-engine/src/swapchain.cpp`, replace `build` with:
```cpp
Swapchain build(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::SurfaceKHR& surface,
    SDL_Window* window,
    vk::SwapchainKHR old_swapchain
) {
    Swapchain swapchain;
    SDL_GetWindowSizeInPixels(window, &swapchain.window_width, &swapchain.window_height);

    const auto capabilities = gpu.device.getSurfaceCapabilitiesKHR(*surface);
    const auto format = choose_format(gpu.device.getSurfaceFormatsKHR(*surface));

    swapchain.format = format.format;
    swapchain.extent = choose_extent(capabilities, swapchain.window_width, swapchain.window_height);

    // One more than the minimum, so we rarely wait on the driver for an image.
    // A maxImageCount of 0 means no limit.
    std::uint32_t image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        image_count = std::min(image_count, capabilities.maxImageCount);
    }

    const vk::SwapchainCreateInfoKHR create_info{
        .surface = *surface,
        .minImageCount = image_count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = swapchain.extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = choose_composite_alpha(capabilities.supportedCompositeAlpha),
        .presentMode = vk::PresentModeKHR::eFifo,  // vsync; the only mode every driver must support
        .clipped = vk::True,
        .oldSwapchain = old_swapchain,              // lets the driver reuse resources on resize
    };

    swapchain.handle = vk::raii::SwapchainKHR(device, create_info);
    swapchain.images = swapchain.handle.getImages();

    for (vk::Image image : swapchain.images) {
        const vk::ImageViewCreateInfo view_info{
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = swapchain.format,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        swapchain.views.emplace_back(device, view_info);
        swapchain.rendered.emplace_back(device, vk::SemaphoreCreateInfo{});
    }

    // The depth buffer is also sampled now: ambient occlusion reads it.
    swapchain.depth = create_image(device, gpu, swapchain.extent, depth_format,
        vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eDepth);

    swapchain.normals = create_image(device, gpu, swapchain.extent, normal_format,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor);

    // Written by compute shaders (storage), then read by the lighting pass.
    for (Image* image : {&swapchain.ao, &swapchain.ao_blur}) {
        *image = create_image(device, gpu, swapchain.extent, ao_format,
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor);
    }

    // Drawn into, then read by the tone-mapping shader.
    swapchain.hdr = create_image(device, gpu, swapchain.extent, hdr_format,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor);

    return swapchain;
}
```

Then replace `recreate_swapchain` with:
```cpp
void recreate_swapchain(
    Swapchain& swapchain,
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::SurfaceKHR& surface,
    SDL_Window* window
) {
    // Nothing may still be using the old images, views or semaphores.
    device.waitIdle();

    Swapchain next = build(device, gpu, surface, window, *swapchain.handle);

    // Destroy the old views and semaphores while their images still exist,
    // then the old swapchain itself and the images that go with it.
    swapchain.views.clear();
    swapchain.rendered.clear();
    swapchain = std::move(next);
}
```

## 9.2 Sampling the depth buffer: `descriptor_heap.h`, `descriptor_heap.cpp`

### Why
A descriptor names the layout the image will be in while shaders read it. Color images are read in `eShaderReadOnlyOptimal`. A depth buffer has a better choice.

### How
- **`eDepthReadOnlyOptimal`** allows both kinds of read: sampling the depth in a shader, and depth testing against it. It would even allow both in one render pass. Here the AO pass samples the depth, then the lighting pass tests against it, with no layout change in between.
- **`write_image_descriptor`** picks this layout for any view whose aspect is depth.

### Code
`game-engine/src/includes/descriptor_heap.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/texture.h"

#include <cstddef>
#include <cstdint>
#include <span>

// The two descriptor heaps shaders read: a resource heap of image
// descriptors and a sampler heap. Each is a plain buffer of descriptor bytes
// that we write ourselves, ending in a range reserved for the driver.
//
// The resource heap stays mapped in host-visible memory, so a descriptor can
// be written into it whenever the GPU isn't reading that slot: the HDR
// image's, for instance, after every resize. The sampler heap never changes,
// so it's uploaded once to device-local memory.
struct DescriptorHeaps {
    Buffer resources;
    std::byte* resource_bytes = nullptr;       // the resource heap, mapped
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
//   - Resource heap: texture i at slot i, then `extra_slots` empty slots,
//     which the program fills itself with write_image_descriptor.
//   - Sampler heap: index 0 is a default sampler; scene sampler i is at
//     index i + 1; after them comes `clamp_sampler`, for images that mustn't
//     wrap around at their edges.
DescriptorHeaps create_descriptor_heaps(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    std::span<const Texture> textures,
    std::span<const SceneSampler> samplers,
    std::uint32_t extra_slots
);

// Writes resource heap slot `slot`: an image described by `view`, either
//   - eSampledImage: read by shaders as a Texture2D, TextureCube, ... handle,
//     in eShaderReadOnlyOptimal layout (eDepthReadOnlyOptimal for depth), or
//   - eStorageImage: written by compute shaders as a RWTexture2D, ...
//     handle, in eGeneral layout.
// The GPU must not be reading the slot while it's written.
void write_image_descriptor(
    const vk::raii::Device& device,
    DescriptorHeaps& heaps,
    std::uint32_t slot,
    const vk::ImageViewCreateInfo& view,
    vk::DescriptorType type = vk::DescriptorType::eSampledImage
);

// Makes `heaps` the ones shaders read for the rest of `commands`.
void bind_descriptor_heaps(const vk::raii::CommandBuffer& commands, const DescriptorHeaps& heaps);
```

In `game-engine/src/descriptor_heap.cpp`, replace `write_image_descriptor` with:
```cpp
void write_image_descriptor(
    const vk::raii::Device& device,
    DescriptorHeaps& heaps,
    std::uint32_t slot,
    const vk::ImageViewCreateInfo& view,
    vk::DescriptorType type
) {
    if (slot >= heaps.resource_slots) {
        throw std::runtime_error("resource heap slot " + std::to_string(slot) + " is past the last one");
    }

    // A descriptor is written from a description of the image view, so no
    // VkImageView object is needed. It also names the layout the image will
    // be in while shaders use it:
    //   - storage images are written in eGeneral,
    //   - a depth buffer is sampled in eDepthReadOnlyOptimal, which also lets
    //     the same pass depth-test against it,
    //   - every other sampled image in eShaderReadOnlyOptimal.
    const bool depth = static_cast<bool>(view.subresourceRange.aspectMask & vk::ImageAspectFlagBits::eDepth);

    const vk::ImageDescriptorInfoEXT image{
        .pView = &view,
        .layout = type == vk::DescriptorType::eStorageImage ? vk::ImageLayout::eGeneral
            : depth ? vk::ImageLayout::eDepthReadOnlyOptimal
            : vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    const vk::ResourceDescriptorInfoEXT descriptor{
        .type = type,
        .data = {.pImage = &image},
    };

    // Slot i sits at i * imageDescriptorSize, which is where a shader's
    // Texture2D.Handle(i) reads it. The heap is host-coherent, so the GPU
    // sees the bytes without a flush.
    const vk::HostAddressRangeEXT destination{
        .address = heaps.resource_bytes + slot * heaps.image_descriptor_size,
        .size = heaps.image_descriptor_size,
    };

    device.writeResourceDescriptorsEXT(descriptor, destination);
}
```

## 9.3 The shared data: `shader_types.h`, `shared.slangh`

### Why
The lighting pass needs to find the AO image, and the AO shaders need their own push data: which images to read and write, and the search's settings.

### How
- **`FrameData`** gains the AO image's heap slot and an on/off flag, at offset 240. `FrameData` grows to 248 bytes, still with no padding.
- **`View::ambient_occlusion`** is the ninth view: the visibility as gray.
- **`AoPushData`**, 48 bytes, serves all three AO dispatches:
  - the frame's address, for the camera matrices,
  - the depth and normals slots,
  - `source` and `target`, the storage images a step reads and writes,
  - the image size, the search radius in meters, the slice and step counts, and the blur's axis.
- **One push block per shader.** `shared.slangh` declared the scene shaders' push block, `ConstantBuffer<PushData> push`. Now `ao.slang` includes the file too, with its own push data, and a shader can only have one push block. So each shader declares its own: `mesh.slang` and `background.slang` declare `PushData`, and `ao.slang` declares `AoPushData`.
- **Octahedral encoding** (Meyer et al. 2010) stores a unit vector in two numbers:
  - Divide the vector by |x| + |y| + |z|. It now lies on an **octahedron**, a diamond with its six points on the axes.
  - The upper half, z ≥ 0, seen from above, is a square standing on one corner. Its x and y are the encoding.
  - The lower half is folded over the upper half's edges, filling the four corners of the square left empty.
  - Every direction now has a place in the square -1..1 × -1..1, spread evenly enough that 16 bits per number lose almost nothing.
  - Decoding unfolds the corners and normalizes.

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

// One per draw, in a GPU buffer the shader indexes.
struct DrawData {
    glm::mat4 model;          // this primitive's space -> world space
    glm::mat4 normal_matrix;  // transposed inverse of model: keeps normals perpendicular under any scale
    std::uint32_t material;   // index into the material buffer
};

static_assert(sizeof(DrawData) == 132);
static_assert(offsetof(DrawData, normal_matrix) == 64);
static_assert(offsetof(DrawData, material) == 128);

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

// --- Lights ------------------------------------------------------------------

// KHR_lights_punctual's three kinds of light. "Punctual" means infinitely
// small: all of a light's power comes from one point, or one direction.
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

// --- Views -------------------------------------------------------------------

// What the fragment shader outputs: the shaded scene, one material input on
// its own, for checking that each one loaded correctly, or the ambient
// occlusion. Keys 1-9 pick one.
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
    std::uint32_t ambient_occlusion;    // resource heap slot: the blurred GTAO image
    std::uint32_t ao_enabled;           // 0: ignore it, to compare
};

static_assert(sizeof(FrameData) == 248);
static_assert(offsetof(FrameData, vertices) == 128);
static_assert(offsetof(FrameData, environment) == 160);
static_assert(offsetof(FrameData, camera_position) == 168);
static_assert(offsetof(FrameData, sun_direction) == 184);
static_assert(offsetof(FrameData, sun_illuminance) == 200);
static_assert(offsetof(FrameData, sky_cube) == 216);
static_assert(offsetof(FrameData, sun_angular_radius) == 236);
static_assert(offsetof(FrameData, ambient_occlusion) == 240);

// --- Push data ---------------------------------------------------------------

// Written with vkCmdPushDataEXT before each draw: where this frame's data is,
// and which DrawData this draw uses. That's 12 bytes of data; the struct
// is 16, because a struct with an 8-byte member is padded to a multiple of 8.
// It's the one struct here with padding, and the shader just ignores it.
struct PushData {
    vk::DeviceAddress frame;
    std::uint32_t draw_index;
};

static_assert(offsetof(PushData, draw_index) == 8);
static_assert(sizeof(PushData) == 16);

// The tone-mapping pass's push data: which resource heap slot holds the HDR
// image, and the view, so material views can skip tone mapping.
struct TonemapPushData {
    std::uint32_t hdr_image;
    View view;
};

// The environment compute shaders' push data. Push data follows std430
// rules, where a vec3 starts on a 16-byte boundary: the pointer and two
// indices fill the first 16 bytes, so sun_direction lands on one. Each
// dispatch sets only what its shader reads, so every member has a default.
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

// The ambient occlusion compute shaders' push data (ao.slang). Each dispatch
// reads `source` and writes `target`, both storage images, except the GTAO
// pass, which reads the depth and normals and only writes.
struct AoPushData {
    vk::DeviceAddress frame = 0;   // this frame's FrameData
    std::uint32_t depth = 0;       // resource heap slot: the depth buffer, sampled
    std::uint32_t normals = 0;     // resource heap slot: the prepass's normals, sampled
    std::uint32_t source = 0;      // resource heap slot: what the blur reads (storage)
    std::uint32_t target = 0;      // resource heap slot: what this step writes (storage)
    std::uint32_t width = 0;       // the images' size in pixels
    std::uint32_t height = 0;
    float radius = 0.0f;           // meters: how far around a point occluders are looked for
    std::uint32_t slices = 0;      // directions around the view vector
    std::uint32_t steps = 0;       // samples along each direction, each way
    std::uint32_t blur_axis = 0;   // the blur's direction: 0 across, 1 down
};

static_assert(sizeof(AoPushData) == 48);
```

`game-engine/shaders/shared.slangh`:
```slang
// The structs every scene shader shares with C++ (src/includes/shader_types.h),
// and a few helpers, included by mesh.slang, background.slang and ao.slang.
// Each of those declares its own push data block. A .slangh file isn't
// compiled on its own: CMakeLists.txt only compiles .slang files.

// Data behind a pointer is laid out like C: each member aligned only to the
// size of its scalar type. Every member here is made of 4-byte floats, so
// there's no padding, and this matches the C++ Vertex exactly (72 bytes).
// The other structs follow the same rule and match theirs.
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

// View: what the fragment shader outputs (keys 1-9).
static const uint view_lit = 0;
static const uint view_base_color = 1;
static const uint view_normal = 2;
static const uint view_vertex_normal = 3;
static const uint view_metallic = 4;
static const uint view_roughness = 5;
static const uint view_occlusion = 6;
static const uint view_emissive = 7;
static const uint view_ambient_occlusion = 8;

// What the environment's compute shaders found out about the sky.
struct EnvironmentInfo {
    float3 irradiance_sh[9];  // diffuse light, as spherical harmonics
    float3 sun_illuminance;   // lux at the ground; 0 for a photographed sky
};

// The same for every draw in a frame. Natural layout, like the C++ struct:
// the pointers land on 8-byte boundaries, right after the matrices.
struct FrameData {
    float4x4 view_projection;          // world space -> clip space
    float4x4 inverse_view_projection;  // clip space -> world space
    Vertex* vertices;                  // the scene's vertices
    DrawData* draws;                   // one DrawData per draw
    Material* materials;               // the scene's materials
    Light* lights;                     // the file's lights
    EnvironmentInfo* environment;      // the sky's diffuse light and the sun
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
    uint ambient_occlusion;            // resource heap slot: the GTAO image
    uint ao_enabled;                   // 0: ignore it
};

// Written with vkCmdPushDataEXT before each draw.
struct PushData {
    FrameData* frame;  // this frame's data
    uint draw_index;   // which DrawData this draw uses
};

// --- Normals in two numbers ------------------------------------------------------

// Octahedral encoding (Meyer et al. 2010): a unit vector is projected onto
// the octahedron |x| + |y| + |z| = 1, whose lower half is folded up over the
// upper; flattened, that's a square, so two numbers in -1..1 hold any
// direction, evenly enough for 16-bit floats.
float2 encode_octahedral(float3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    const float2 folded = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return n.z >= 0.0 ? n.xy : folded;
}

float3 decode_octahedral(float2 e) {
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    const float fold = saturate(-n.z);
    n.x += n.x >= 0.0 ? -fold : fold;
    n.y += n.y >= 0.0 ? -fold : fold;
    return normalize(n);
}
```

## 9.4 The prepass and the lit surface: `mesh.slang`, `background.slang`

### Why
`mesh.slang` gets a second fragment shader for the prepass. Its lighting then uses the AO in four ways, each a correction to the indirect light.

### How
- **`prepassMain`** writes the **vertex normal**, encoded, and nothing else. The rasterizer writes the depth.
  - **Masked materials** discard their transparent texels here too, so the depth buffer holds only what's actually solid.
  - **Why the vertex normal and not the normal map's:** the depth buffer only knows the mesh's shape. A normal-mapped brick tilts this way and that, but its depth is flat. Searching around a tilted normal would find the flat surface beside it above the horizon and darken it. The vertex normal agrees with the depth.
- **The lighting pass reads the AO** at its own pixel with `Load`, with no filtering. One pixel of the AO image belongs to one pixel of the screen.
- **Visibility combines with the occlusion map by `min`,** not by multiplying. The map is baked AO for detail inside one mesh. GTAO sees detail between meshes, at the depth buffer's scale. Both estimate the same thing, and where both see the same crease, multiplying would darken it twice.
- **The bent normal** is the direction the open sky lies in, so the diffuse sky light is read along it instead of the surface normal. A point in a corner then gets the light from the open side, not from the wall it faces.
  - The bent normal was found around the vertex normal, and has none of the normal map's detail.
  - **`rotate_from_to`** finds the rotation that turns the vertex normal into the bent normal. It applies the same rotation to the shading normal, which keeps the detail. It's Rodrigues' rotation formula, rewritten with the cross and dot products of the two vectors so that it needs no angles: `v·c + a × v + a·(a·v) / (1 + c)`, with `a = from × to` and `c = from · to`.
- **Multi-bounce** (Jimenez et al. 2016): AO counts light as blocked, but the occluders also reflect some of it on. White surfaces in a corner bounce most of it back, black ones almost none. The paper fits a cubic in the visibility to path-traced results, with the albedo as a parameter. The surface's own albedo stands in for its surroundings', which are usually similar. Without this, AO looks too dark on bright surfaces.
- **Specular occlusion** (Lagarde and de Rousiers 2014): the reflection of the sky is also blocked, but differently. A smooth surface reflects a narrow cone around the mirror direction, which sees past nearby occluders when you look at the surface straight on. A rough surface's wide lobe loses about as much as the diffuse light. The formula `saturate((n·v + visibility)^(2^(-16 α - 1)) - 1 + visibility)` gives both. Its roughness is GGX's α, the roughness squared, which `Surface` already holds.
- **AO only darkens the indirect light.** The sun and the punctual lights are direct: a point either sees them or it doesn't, and only a shadow can say which. Darkening them by AO would put dark halos in sunlight.
- **Blended surfaces** aren't in the prepass, so the AO image under them holds whatever is behind. They use the occlusion map alone.
- **`background.slang`** only declares its own push block now.

### Code
`game-engine/shaders/mesh.slang`:
```slang
// Draws one glTF primitive: its vertices come from the scene's vertex buffer,
// its place in the world from its DrawData, and its surface from its glTF
// material, whose textures are read from the descriptor heap. Shaded with
// glTF's physically based BRDF, lit by the sun, the file's lights and the
// sky around the scene, and written to the HDR image already exposed.

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

#include "shared.slangh"

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
// clip-space position; every other field is interpolated across the triangle.
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

// --- Vertex shader -----------------------------------------------------------

// SV_VulkanVertexID is Vulkan's own gl_VertexIndex, which includes the draw's
// vertexOffset: each primitive's indices start at 0, and the draw adds where
// that primitive's vertices begin in the shared buffer.
[shader("vertex")]
VertexOutput vertexMain(uint vertex_id : SV_VulkanVertexID) {
    FrameData* frame = push.frame;
    const Vertex vertex = frame.vertices[vertex_id];
    const DrawData draw = frame.draws[push.draw_index];

    const float4 world = mul(draw.model, float4(vertex.position, 1.0));

    // Tangent and bitangent lie along the surface, so they move with the
    // model matrix, like positions; only the normal needs the normal matrix.
    // The bitangent is built before the transform, from glTF's rule
    // B = cross(N, T) * w: a mirroring transform then mirrors it too.
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
float3 surface_normal(VertexOutput input, Material material, bool front_face, bool apply_normal_map) {
    float3 normal = input.normal;

    // cross(ddy, ddx), not cross(ddx, ddy): Vulkan's screen Y points down,
    // so this order is the one that points toward the camera.
    if (all(normal == 0.0)) {
        normal = cross(ddy(input.world_position), ddx(input.world_position));
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
    float3 tangent_space = sample_slot(material.normal, input).xyz * 2.0 - 1.0;
    tangent_space.xy *= material.normal_scale;

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
    const float3 metal = specular * (surface.base_color + (1.0 - surface.base_color) * fresnel);
    const float3 dielectric = lerp(surface.base_color / pi, float3(specular), 0.04 + 0.96 * fresnel);
    const float3 brdf = lerp(dielectric, metal, surface.metallic);

    // Light falling at an angle spreads over more surface: the n.l factor.
    return brdf * illuminance * n_dot_l;
}

// --- Lights --------------------------------------------------------------------

// The direction toward a light and the illuminance it gives here, following
// KHR_lights_punctual. Point and spot lights fade with the square of the
// distance, then smoothly to nothing at `range`; spot lights also fade from
// the inner cone to the outer one.
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

// --- Image-based lighting ------------------------------------------------------

// Light from the whole sky at once, from what environment.slang prepared.

// The sky's irradiance on a surface facing `n`, in lux: its nine spherical
// harmonics coefficients, each weighted by its basis function at `n`.
float3 sky_irradiance(EnvironmentInfo* environment, float3 n) {
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
float3 shade_environment(Surface surface, FrameData* frame, float roughness, float visibility, float3 irradiance_normal) {
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
    const float3 specular = prefiltered * (f0 * brdf.x + brdf.y) * specular_occlusion(n_dot_v, visibility, surface.alpha);

    return diffuse + specular;
}

// --- Depth and normal prepass ------------------------------------------------------

// The interpolated vertex normal, facing the viewer on a double-sided
// material's back face; flat when the file has no normals. This is the
// surface at the scale the mesh describes it, which ambient occlusion
// searches against: a normal map's detail isn't in the depth buffer.
float3 vertex_normal(VertexOutput input, Material material, bool front_face) {
    float3 normal = input.normal;

    if (all(normal == 0.0)) {
        normal = cross(ddy(input.world_position), ddx(input.world_position));
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
    FrameData* frame = push.frame;
    const Material material = frame.materials[frame.draws[push.draw_index].material];

    if (alpha_mode == alpha_mask) {
        const float alpha = material.base_color_factor.a * sample_slot(material.base_color, input).a * input.color.a;
        if (alpha < material.alpha_cutoff) {
            discard;
        }
    }

    return encode_octahedral(vertex_normal(input, material, front_face));
}

// --- Fragment shader ---------------------------------------------------------

// SV_Target: the value written to color attachment 0.
// SV_IsFrontFace: whether this triangle faces the camera.
[shader("fragment")]
float4 fragmentMain(VertexOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    FrameData* frame = push.frame;
    const Material material = frame.materials[frame.draws[push.draw_index].material];

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

    const float3 normal = surface_normal(input, material, front_face, frame.view != view_vertex_normal);

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
        case view_roughness: return float4(roughness.xxx, 1.0);
        case view_occlusion: return float4(occlusion.xxx, 1.0);
        case view_emissive: return float4(emissive, 1.0);
        case view_ambient_occlusion: return float4(gtao.www, 1.0);
        default: break;
    }

    // A perfectly smooth surface would reflect a punctual light from a single
    // point, too small for any pixel to catch; a floor on roughness keeps
    // highlights visible.
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

    // Indirect light from the sky, darkened by occlusion. Ambient occlusion
    // only ever reaches this indirect light: the sun and the lights are
    // direct, and only a shadow can block them.
    radiance += shade_environment(surface, frame, roughness, visibility, irradiance_normal);

    // Exposure scales nits into the tone mapper's range here, before the
    // 16-bit HDR image could overflow. glTF defines emission in nits, but, as
    // its spec notes many engines do, we take it as already exposed: an
    // emissive value of 1 shows as near-white, whatever the exposure.
    return float4(radiance * frame.exposure + emissive, base_color.a);
}
```

`game-engine/shaders/background.slang`:
```slang
// Draws the sky behind the scene: one full-screen triangle at the far plane,
// depth-tested so it only covers pixels nothing else has drawn on.

#include "shared.slangh"

// In a descriptor heap pipeline, the push_constant block is where push data lands.
[[vk::push_constant]]
ConstantBuffer<PushData> push;

// --- Vertex shader -----------------------------------------------------------

struct VertexOutput {
    float4 position : SV_Position;
    float2 clip : TEXCOORD0;  // this point's clip-space x and y
};

// The full-screen triangle of tonemap.slang, at depth 0: with reverse-Z,
// that's the far plane, so the depth test lets it through only where the
// depth buffer still holds its cleared 0. Its clip-space corners are also
// passed on: interpolated across the triangle, they arrive at each pixel as
// that pixel's own clip-space position.
[shader("vertex")]
VertexOutput vertexMain(uint vertex_id : SV_VulkanVertexID) {
    const float2 corner = float2((vertex_id << 1) & 2, vertex_id & 2) * 2.0 - 1.0;

    VertexOutput output;
    output.position = float4(corner, 0.0, 1.0);
    output.clip = corner;
    return output;
}

// --- Fragment shader ---------------------------------------------------------

// Each pixel looks along the ray from the camera through it. Turning the
// pixel's clip-space position at the far plane (depth 0) back into world
// space gives a point on that ray; the direction is from the camera to it.
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    FrameData* frame = push.frame;

    const float4 far_point = mul(frame.inverse_view_projection, float4(input.clip, 0.0, 1.0));
    const float3 direction = normalize(far_point.xyz / far_point.w - frame.camera_position);

    const TextureCube sky = TextureCube.Handle(uint2(frame.sky_cube, 0));
    const SamplerState clamped = SamplerState.Handle(uint2(frame.clamp_sampler, 0));
    float3 radiance = sky.SampleLevel(clamped, direction, 0.0).rgb;

    // The sun's disk, which the sky cube leaves out: its illuminance spread
    // over the tiny solid angle it covers, pi r^2 for an angular radius r.
    if (dot(direction, frame.sun_direction) > cos(frame.sun_angular_radius)) {
        const float solid_angle = 3.14159265 * frame.sun_angular_radius * frame.sun_angular_radius;
        radiance += frame.sun_illuminance / solid_angle;
    }

    // Exposed like the scene, and kept below the 16-bit float limit: the
    // sun's disk is over a billion nits.
    return float4(min(radiance * frame.exposure, 60000.0), 1.0);
}
```

## 9.5 Two passes per material: `pipeline.h`, `pipeline.cpp`

### Why
Opaque and masked materials are now drawn twice: once in the prepass and once in the lighting pass. Each pass needs its own pipelines.

### How
- **`MeshPass`** picks which: `depth_normals` for the prepass, `lighting` for the rest.
- **The prepass** runs `prepassMain` into the normals image, and writes depth with the `eGreater` test, as the scene did until now.
- **The lighting pass writes no depth.** It tests with **`eGreaterOrEqual`** against the prepass's depth:
  - **A solid surface** passes only where it *is* the nearest surface: its depth equals what the prepass stored. Everything hidden behind it fails before its fragment shader runs. Each pixel is shaded once, however many surfaces overlap there. That's a speed-up the prepass gives us for free: Sponza has plenty of overlapping surfaces.
  - **A see-through surface** passes wherever it's in front, as before, and blends.
- **Equal depths need identical math.** Both pipelines run the same vertex shader on the same vertices, so they compute the same depth, bit for bit, in practice. Strictly, Vulkan only promises this across pipelines for a position output decorated `Invariant`, and Slang offers no way to add that decoration to `SV_Position` yet. If a driver ever compiled the two differently, the symptom would be surfaces speckled with black: pixels where the lighting pass's depth came out a hair farther than the prepass's, so neither the surface nor the sky behind it passes, and the HDR image keeps its clear color.
- **No blended prepass pipeline:** see-through surfaces aren't in the prepass, so there's nothing for it to draw.

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
std::vector<std::uint32_t> read_spirv(const std::filesystem::path& path);

// The two passes the scene is drawn in.
enum class MeshPass {
    depth_normals,  // the prepass: depth and vertex normals, for opaque and masked materials
    lighting,       // the full shading, against the prepass's depth
};

// Draws shaders/mesh.slang into a `color_format` image, depth-tested against
// a `depth_format` depth buffer, for `pass` and materials with alpha mode
// `alpha_mode`. There is no pipeline layout: shaders find their resources in
// the descriptor heap. Cull mode and front face are set per draw.
vk::raii::Pipeline create_mesh_pipeline(
    const vk::raii::Device& device,
    vk::Format color_format,
    vk::Format depth_format,
    AlphaMode alpha_mode,
    MeshPass pass
);

// Draws shaders/<shader>.spv's vertexMain and fragmentMain as one full-screen
// triangle into a `color_format` image: no vertex data, nothing culled.
// With a `depth_format`, the triangle is depth-tested at depth 0, the far
// plane, without writing depth, so it only reaches pixels nothing else has
// been drawn on: that's how the sky goes behind the scene.
vk::raii::Pipeline create_fullscreen_pipeline(
    const vk::raii::Device& device,
    const char* shader,
    vk::Format color_format,
    vk::Format depth_format = vk::Format::eUndefined
);

// A compute pipeline running `entry_point` from shaders/<shader>.spv.
vk::raii::Pipeline create_compute_pipeline(const vk::raii::Device& device, const char* shader, const char* entry_point);
```

In `game-engine/src/pipeline.cpp`, replace `create_mesh_pipeline` with:
```cpp
vk::raii::Pipeline create_mesh_pipeline(
    const vk::raii::Device& device,
    vk::Format color_format,
    vk::Format depth_format,
    AlphaMode alpha_mode,
    MeshPass pass
) {
    const bool prepass = pass == MeshPass::depth_normals;

    // Shaders: one module, two entry points picked by name. The module is
    // only needed while the pipeline is built, so it's destroyed on return.
    const std::vector<std::uint32_t> spirv = read_spirv(std::filesystem::path(SHADER_DIR) / "mesh.spv");

    const vk::raii::ShaderModule module(device, vk::ShaderModuleCreateInfo{
        .codeSize = spirv.size() * sizeof(std::uint32_t),
        .pCode = spirv.data(),
    });

    // The fragment shader's alpha mode is a specialization constant: a
    // constant whose value is filled in now, when the pipeline is built. The
    // compiler then removes the code the other modes need, so opaque
    // surfaces never pay for the alpha test.
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
            .pName = prepass ? "prepassMain" : "fragmentMain",
            .pSpecializationInfo = &specialization,
        },
    };

    // Vertex input and assembly: no vertex attributes, the vertex shader reads
    // its vertex through a pointer. Every 3 vertices form a triangle.
    const vk::PipelineVertexInputStateCreateInfo vertex_input{};

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };

    // Viewport: counts only. The viewport and scissor rectangles are set
    // while recording, so a resized window doesn't need a new pipeline.
    const vk::PipelineViewportStateCreateInfo viewport{
        .viewportCount = 1,
        .scissorCount = 1,
    };

    // Which side of a triangle is culled depends on the draw: its material
    // may be double-sided, and its transform may mirror it. Both states are
    // dynamic, so one pipeline serves every draw.
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

    // Rasterization: filled triangles. cullMode and frontFace are dynamic,
    // set before each draw.
    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .polygonMode = vk::PolygonMode::eFill,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    const bool blend = alpha_mode == AlphaMode::blend && !prepass;

    // Depth. With reverse-Z (see camera.cpp) nearer means a *greater* depth
    // value, and the buffer is cleared to 0, the far plane.
    //   - The prepass keeps a fragment only if it's nearer than what's there,
    //     and records its depth: the depth buffer ends up holding the nearest
    //     solid surface at every pixel.
    //   - The lighting pass writes no depth. Its solid surfaces pass "greater
    //     or equal" only where they are that nearest surface, so each pixel is
    //     shaded once. See-through surfaces pass wherever they're in front.
    const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = prepass ? vk::True : vk::False,
        .depthCompareOp = prepass ? vk::CompareOp::eGreater : vk::CompareOp::eGreaterOrEqual,
    };

    // Color output. Opaque and masked surfaces replace what's there. Blended
    // ones mix with it, weighted by their alpha:
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

    // Dynamic rendering: instead of a VkRenderPass, the pipeline names the
    // formats of the images it will draw into.
    const vk::PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &color_format,
        .depthAttachmentFormat = depth_format,
    };

    // Descriptor heap mode is what makes `layout = nullptr` legal: shaders
    // will reach resources through the heap and push data, not descriptor
    // sets and push constants declared in a VkPipelineLayout.
    const vk::PipelineCreateFlags2CreateInfo flags{
        .pNext = &rendering,
        .flags = vk::PipelineCreateFlagBits2::eDescriptorHeapEXT,
    };

    // pNext chain: create info -> flags -> rendering. Everything it points at
    // lives until the end of this function, past the pipeline's creation.
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
```

## 9.6 Ground-truth ambient occlusion: `ao.slang`

### Why
Ambient occlusion at a point is the share of the sky it can see, weighted by the cosine to its normal, because that's how much each direction contributes to the diffuse light. A ray tracer would shoot hundreds of rays per pixel to measure this. GTAO gets close with a few dozen depth-buffer reads, by turning the problem into a few 2D slices that each have an exact answer.

Earlier screen-space methods sampled points around each pixel and counted how many were hidden (SSAO, Crytek 2007), or searched for horizons like GTAO but weighted the sky they found differently (HBAO, Bavoil et al. 2008). Neither result is quite the cosine-weighted visibility the lighting needs: SSAO, sampling a whole sphere, even turns open flat ground gray. GTAO's result matches path-traced AO, in its paper's own comparisons, wherever the depth buffer holds the occluders.

### How
- **Slices.** Pick a direction perpendicular to the view vector. That direction and the view vector together span a plane through the point: a **slice** of the hemisphere. We take 4 slices, spread around the view vector. (The paper takes one per pixel per frame and spreads the rest over neighbouring pixels and frames; that needs temporal accumulation, which we don't have.)
  - Directions are picked **in 3D, around the view vector,** not as angles on the screen. The integral weights every slice the same, and that's only true for directions uniform around the view vector. Angles uniform on the screen aren't, toward the screen's edges, where the view vector meets the image at a slant.
  - Each slice is a line on the screen, through the pixel. Projecting a point one radius along the slice's direction gives that line, and how many pixels the radius covers.
- **The horizon search.** Walk along the line on both sides, 6 steps each way.
  - At each step, rebuild the sample's world position from its depth, through the inverse view-projection.
  - The **horizon** is the highest occluder found on that side, kept as the cosine of its angle from the view vector.
  - Steps bunch up near the pixel (`t²`), where contact shadows are, and each is at least one pixel further out. The radius is capped at 16 pixels per step on average, so the steps never skip far over the surface.
- **What a sample can't be:**
  - **Too far:** only occluders within 0.8 m count. Approaching that radius, a sample's horizon fades back toward fully open, so occluders don't pop in and out as they cross it.
  - **The same surface:** a sample that doesn't rise above the point's tangent plane can't occlude it. On a flat floor seen at a slant, rounding the slice's line to whole pixels steps a little off the slice's plane, and the floor there would otherwise count as a horizon.
  - **The sky:** depth 0, which blocks nothing.
- **The exact integral.** Within a slice, measure angles from the view vector. The normal projects into the slice at some angle `n`, and its hemisphere spans `n − π/2` to `n + π/2`. The open sky in this slice runs from the negative side's horizon `h₀` to the positive side's `h₁`, clamped to that hemisphere. Integrating the cosine to the normal over that arc, weighted by `|sin h|`, each angle's share of the solid angle, has a closed form: `(cos n + 2h·sin n − cos(2h − n)) / 4` for each horizon `h`. That's the slice's visibility. The direction-weighted version of the same integral, along the view vector and along the slice, gives the arc's centroid: the slice's part of the **bent normal**.
- **Slices are weighted** by the length of the normal's projection into them. For any direction in a slice, the cosine to the real normal is that length times the cosine to the projected normal. A slice nearly perpendicular to the normal counts for little.
- **Noise.** A per-pixel hash offsets each pixel's slice directions and step positions. Four slices then sample different directions in neighbouring pixels, and their average approaches many slices. The noise is **fixed**: the same pixel always gets the same offsets. There's no frame-to-frame noise, which would flicker without temporal accumulation.
- **The blur** averages that noise away: 9 taps across, then 9 down, Gaussian-weighted with a standard deviation of 2 pixels.
  - **It respects edges:** each tap is weighted again by how close its surface is to the centre's in distance from the camera. A 5% difference is one standard deviation. A pillar's AO doesn't bleed onto the wall far behind it.
  - **It blurs only the visibility.** A bent normal averaged across an edge would point into whatever it was averaged with.
- **What it can't see:** whatever isn't in the depth buffer: occluders behind the camera, off screen, or hidden behind something nearer. That's the price of working in screen space, and why the radius is kept short.

### Code
`game-engine/shaders/ao.slang`:
```slang
// Ambient occlusion: how much of the sky each visible point can see, from the
// depth buffer and the prepass's normals. Ground-truth ambient occlusion (GTAO: Jimenez et al.
// 2016, "Practical Real-Time Strategies for Accurate Indirect Occlusion"), in
// the form Intel's XeGTAO gives it:
//   gtaoMain  for a few slices of the hemisphere around the view direction,
//             walk outward on both sides and keep the highest horizon found;
//             the visible arc between the two horizons, integrated against
//             the cosine, is the visibility, and its centroid the bent normal
//   blurMain  a separable, depth-aware blur of the visibility
// It's deterministic: a fixed per-pixel hash, no frame index, no accumulation.

#include "shared.slangh"

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

struct AoPushData {
    FrameData* frame;
    uint depth;
    uint normals;
    uint source;
    uint target;
    uint width;
    uint height;
    float radius;
    uint slices;
    uint steps;
    uint blur_axis;
};

[[vk::push_constant]]
ConstantBuffer<AoPushData> push;

static const float pi = 3.14159265;
static const float half_pi = 1.57079633;

// --- Positions from the depth buffer -------------------------------------------

// The world position of the surface at pixel position `pixel` with depth
// `depth`: its clip-space coordinates, back through the inverse
// view-projection, divided by w.
float3 world_position(FrameData* frame, float2 pixel, float depth) {
    const float2 ndc = pixel / float2(push.width, push.height) * 2.0 - 1.0;
    const float4 position = mul(frame.inverse_view_projection, float4(ndc, depth, 1.0));
    return position.xyz / position.w;
}

// Where a world position lands on screen, in pixels.
float2 to_pixels(FrameData* frame, float3 position) {
    const float4 clip = mul(frame.view_projection, float4(position, 1.0));
    return (clip.xy / clip.w * 0.5 + 0.5) * float2(push.width, push.height);
}

// --- Small helpers ---------------------------------------------------------------

// acos to within about 0.01 radians, for a fraction of the cost: it runs
// three times per slice. The polynomial is XeGTAO's.
float fast_acos(float x) {
    const float a = abs(x);
    const float r = (-0.156583 * a + half_pi) * sqrt(1.0 - a);
    return x >= 0.0 ? r : pi - r;
}

// A number in 0..1 per pixel, different for each `salt`. A hash rather than
// a regular pattern, so the blur that follows can smooth it away without
// leaving bands or diagonals.
float spatial_hash(uint2 pixel, uint salt) {
    uint h = pixel.x * 73856093u ^ pixel.y * 19349663u ^ salt * 83492791u;
    h = (h ^ (h >> 16)) * 0x45D9F3Bu;
    h = (h ^ (h >> 16)) * 0x45D9F3Bu;
    return float(h & 0xFFFFu) / 65535.0;
}

// --- The visible arc ---------------------------------------------------------------

// Within one slice, angles h are measured from the view direction, and n is
// the angle of the surface normal projected into the slice. The share of the
// slice's light arriving between the view direction and angle h, weighted by
// the cosine to the normal and by |sin h| (each angle's share of the solid
// angle), has a closed form; so do its components along the view direction
// and along the slice, which give the bent normal:
//   A(h)  = (cos n + 2 h sin n - cos(2h - n)) / 4
//   Cv(h) = cos n (-cos^3 h / 3) + sin n (sin^3 h / 3)
//   Co(h) = cos n (sin^3 h / 3) + sin n (cos^3 h / 3 - cos h)
// Cv and Co leave out their constants: over both sides of the slice, those
// add (2/3) cos n and (4/3) sin n.
void arc_terms(float h, float cos_n, float sin_n, out float a, out float along_view, out float along_slice) {
    const float c = cos(h);
    const float s = sin(h);
    const float c3 = c * c * c;
    const float s3 = s * s * s;

    // cos(2h - n), expanded so there's one cos and one sin to compute.
    const float cos_2h_n = (1.0 - 2.0 * s * s) * cos_n + 2.0 * s * c * sin_n;

    a = 0.25 * (cos_n + 2.0 * h * sin_n - cos_2h_n);
    along_view = cos_n * (-c3 / 3.0) + sin_n * (s3 / 3.0);
    along_slice = cos_n * (s3 / 3.0) + sin_n * (c3 / 3.0 - c);
}

// --- The horizon search ------------------------------------------------------------

[shader("compute")]
[numthreads(8, 8, 1)]
void gtaoMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= push.width || id.y >= push.height) {
        return;
    }

    FrameData* frame = push.frame;
    const int2 pixel = int2(id.xy);
    const Texture2D depth_buffer = Texture2D.Handle(uint2(push.depth, 0));
    const Texture2D normals = Texture2D.Handle(uint2(push.normals, 0));
    RWTexture2D<float4> target = RWTexture2D<float4>.Handle(uint2(push.target, 0));

    // Depth 0 is the far plane: the sky, which nothing occludes.
    const float depth = depth_buffer.Load(int3(pixel, 0)).r;
    if (depth == 0.0) {
        target[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float3 centre = world_position(frame, float2(pixel) + 0.5, depth);
    const float3 view_dir = normalize(frame.camera_position - centre);
    const float2 centre_pixels = float2(pixel) + 0.5;

    // The interpolated vertex normal: the surface at the scale the mesh
    // describes it. A normal facing away from the viewer has no arc the
    // integral can describe, so it's tilted just far enough to face it.
    float3 normal = decode_octahedral(normals.Load(int3(pixel, 0)).xy);
    normal = normalize(normal + max(0.0, 1e-3 - dot(normal, view_dir)) * view_dir);

    // Two axes perpendicular to the view direction. A slice's direction is
    // picked uniformly around the view direction, in 3D: the integral weighs
    // each slice for exactly that. Picking angles on the screen instead would
    // bias it toward the edges of the screen, where the view direction meets
    // the image at a slant.
    const float3 helper = abs(view_dir.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 axis_a = normalize(cross(helper, view_dir));
    const float3 axis_b = cross(view_dir, axis_a);

    const float rotation_noise = spatial_hash(id.xy, 0);
    const float step_noise = spatial_hash(id.xy, 1);

    float visibility = 0.0;
    float3 bent = 0.0;

    for (uint slice = 0; slice < push.slices; ++slice) {
        const float phi = pi * (float(slice) + rotation_noise) / float(push.slices);
        const float3 ortho = axis_a * cos(phi) + axis_b * sin(phi);
        const float3 slice_normal = cross(view_dir, ortho);

        // Where the slice runs on screen, and how many pixels the radius
        // covers along it: project a point one radius away. Capped at 16
        // pixels per step on average, so the steps never skip far over the
        // surface.
        const float2 reach = to_pixels(frame, centre + ortho * push.radius) - centre_pixels;
        const float reach_length = length(reach);
        if (reach_length < 1e-3) {
            continue;
        }
        const float2 omega = reach / reach_length;
        const float radius_pixels = min(reach_length, 16.0 * float(push.steps));

        // The normal projected into the slice's plane, and its angle n from
        // the view direction, signed toward `ortho`.
        float3 projected = normal - slice_normal * dot(normal, slice_normal);
        const float projected_length = length(projected);
        if (projected_length < 1e-4) {
            continue;
        }
        projected /= projected_length;

        const float cos_n = clamp(dot(projected, view_dir), -1.0, 1.0);
        const float n = sign(dot(projected, ortho)) * fast_acos(cos_n);

        // Both horizons start fully open, at the edge of the hemisphere
        // around the normal, kept as cosines from the view direction.
        float horizon_positive = cos(n + half_pi);
        float horizon_negative = cos(n - half_pi);

        for (uint step = 0; step < push.steps; ++step) {
            // Steps bunch up near the centre (t squared), where contact
            // shadows are, and each is at least one pixel further out.
            const float t = (float(step) + 0.5 + step_noise) / float(push.steps);
            const float distance_pixels = max(t * t * radius_pixels, 1.0 + float(step));
            const int2 offset = int2(round(omega * distance_pixels));

            for (int side = 0; side < 2; ++side) {
                const int2 sample_pixel = pixel + (side == 0 ? offset : -offset);
                if (any(sample_pixel < 0) || sample_pixel.x >= int(push.width) || sample_pixel.y >= int(push.height)) {
                    continue;
                }

                const float sample_depth = depth_buffer.Load(int3(sample_pixel, 0)).r;
                if (sample_depth == 0.0) {
                    continue;
                }

                const float3 delta = world_position(frame, float2(sample_pixel) + 0.5, sample_depth) - centre;
                const float distance = length(delta);
                if (distance <= 1e-4 || distance > push.radius) {
                    continue;
                }

                // A sample that doesn't rise above the surface's own tangent
                // plane is this same surface, so it can't occlude it.
                if (dot(delta, normal) <= 0.0) {
                    continue;
                }

                // The sample's horizon, faded back toward fully open as it
                // nears the radius, so occluders don't pop in and out.
                const float retract = 1.0 - smoothstep(0.6 * push.radius, push.radius, distance);
                const float horizon = dot(delta / distance, view_dir);

                if (side == 0) {
                    horizon_positive = max(horizon_positive, lerp(cos(n + half_pi), horizon, retract));
                } else {
                    horizon_negative = max(horizon_negative, lerp(cos(n - half_pi), horizon, retract));
                }
            }
        }

        // The visible arc runs from angle h0 (negative side) to h1 (positive
        // side), clamped to the hemisphere around the normal. Its value is the
        // slice's visibility; its centroid adds to the bent normal. Each slice
        // counts in proportion to the projected normal's length: the cosine
        // to the real normal is that length times the cosine within the slice.
        const float sin_n = sin(n);
        float h0 = -fast_acos(clamp(horizon_negative, -1.0, 1.0));
        float h1 = fast_acos(clamp(horizon_positive, -1.0, 1.0));
        h0 = n + max(h0 - n, -half_pi);
        h1 = n + min(h1 - n, half_pi);

        float a0, view0, slice0;
        float a1, view1, slice1;
        arc_terms(h0, cos_n, sin_n, a0, view0, slice0);
        arc_terms(h1, cos_n, sin_n, a1, view1, slice1);

        visibility += (a0 + a1) * projected_length;
        bent += (view_dir * (view0 + view1 + (2.0 / 3.0) * cos_n)
            + ortho * (slice0 + slice1 + (4.0 / 3.0) * sin_n)) * projected_length;
    }

    visibility = saturate(visibility / float(push.slices));
    const float3 bent_normal = dot(bent, bent) > 1e-8 ? normalize(bent) : normal;
    target[pixel] = float4(bent_normal, visibility);
}

// --- The blur ------------------------------------------------------------------------

// Nine taps along one axis, Gaussian-weighted, and weighted again by how close
// each tap's surface is to this one's in distance from the camera, so the
// blur never mixes a foreground edge with what's behind it. Only the
// visibility is blurred: a bent normal averaged across an edge would point
// into whatever it was averaged with.
float camera_distance(FrameData* frame, int2 pixel, float depth) {
    return length(world_position(frame, float2(pixel) + 0.5, depth) - frame.camera_position);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void blurMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= push.width || id.y >= push.height) {
        return;
    }

    FrameData* frame = push.frame;
    const int2 pixel = int2(id.xy);
    const Texture2D depth_buffer = Texture2D.Handle(uint2(push.depth, 0));
    RWTexture2D<float4> source = RWTexture2D<float4>.Handle(uint2(push.source, 0));
    RWTexture2D<float4> target = RWTexture2D<float4>.Handle(uint2(push.target, 0));

    const float4 centre = source[pixel];
    const float centre_depth = depth_buffer.Load(int3(pixel, 0)).r;

    if (centre_depth == 0.0) {
        target[pixel] = centre;
        return;
    }

    const float centre_distance = camera_distance(frame, pixel, centre_depth);
    const int2 axis = push.blur_axis == 0 ? int2(1, 0) : int2(0, 1);
    const float weights[5] = {0.20416, 0.18017, 0.12383, 0.06628, 0.02763};  // a Gaussian, sigma 2

    float sum = centre.w * weights[0];
    float total = weights[0];

    for (int i = 1; i <= 4; ++i) {
        for (int side = 0; side < 2; ++side) {
            const int2 tap = pixel + axis * (side == 0 ? i : -i);
            if (any(tap < 0) || tap.x >= int(push.width) || tap.y >= int(push.height)) {
                continue;
            }

            const float tap_depth = depth_buffer.Load(int3(tap, 0)).r;
            if (tap_depth == 0.0) {
                continue;
            }

            // A 5% difference in distance is one standard deviation.
            const float difference = (camera_distance(frame, tap, tap_depth) - centre_distance) / centre_distance;
            const float weight = weights[i] * exp(-difference * difference * 200.0);

            sum += source[tap].w * weight;
            total += weight;
        }
    }

    target[pixel] = float4(centre.xyz, sum / max(total, 1e-4));
}
```

## 9.7 Recording the AO pass: `ambient_occlusion.h`, `ambient_occlusion.cpp`

### Why
The AO pass is three compute dispatches with barriers between them. It's a self-contained step, so it gets its own small module, like the environment did.

### How
- **`AmbientOcclusion`** holds the two compute pipelines, `gtaoMain` and `blurMain` from the one module, and the settings: 0.8 m, 4 slices, 6 steps.
- **`record_ambient_occlusion`** records:
  1. **Both AO images to `eGeneral`,** from `eUndefined`: they're rewritten from scratch. The barrier also waits for the previous frame's lighting to finish reading the AO image.
  2. **The search,** into `ao`.
  3. **A barrier:** the search's writes must reach the blur's reads. A memory barrier is enough: the images stay in `eGeneral`.
  4. **The blur across,** from `ao` into `ao_blur`. Another barrier.
  5. **The blur down,** from `ao_blur` back into `ao`.
  6. **`ao` to `eShaderReadOnlyOptimal`,** ready for the lighting pass's fragment shaders.
- **One thread per pixel,** in 8 × 8 workgroups, rounded up to cover the image. Shaders skip the threads outside it.

### Code
`game-engine/src/includes/ambient_occlusion.h`:
```cpp
#pragma once

#include "includes/shader_types.h"
#include "includes/swapchain.h"

#include <vulkan/vulkan_raii.hpp>

// --- Ambient occlusion -------------------------------------------------------

// GTAO's compute pipelines (shaders/ao.slang) and its settings: occluders
// up to 0.8 m away count, found along 4 slices of 6 steps each way. Fewer
// slices leave noise the blur can't hide; more cost time for little change.
struct AmbientOcclusion {
    vk::raii::Pipeline search = nullptr;
    vk::raii::Pipeline blur = nullptr;
    float radius = 0.8f;
    std::uint32_t slices = 4;
    std::uint32_t steps = 6;
};

AmbientOcclusion create_ambient_occlusion(const vk::raii::Device& device);

// Records the whole pass: the search into swapchain.ao, then the blur, across
// into swapchain.ao_blur and down back into swapchain.ao. `push` names the
// frame and the slots: the depth and normals (sampled, in their read-only
// layouts by now) and the two AO images as storage (source and target are
// filled in here). On return swapchain.ao is ready for fragment shaders.
void record_ambient_occlusion(
    const vk::raii::CommandBuffer& commands,
    const AmbientOcclusion& ambient_occlusion,
    const Swapchain& swapchain,
    std::uint32_t ao_target,
    std::uint32_t ao_blur_target,
    AoPushData push
);
```

`game-engine/src/ambient_occlusion.cpp`:
```cpp
#include "includes/ambient_occlusion.h"

#include "includes/pipeline.h"

namespace {

// Moves a whole color image between layouts, after `src` work, before `dst` work.
void barrier(
    const vk::raii::CommandBuffer& commands,
    const Image& image,
    vk::ImageLayout from,
    vk::ImageLayout to,
    vk::PipelineStageFlags2 src_stage,
    vk::AccessFlags2 src_access,
    vk::PipelineStageFlags2 dst_stage,
    vk::AccessFlags2 dst_access
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
        .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1},
    };

    commands.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &image_barrier});
}

// One compute step over the whole image, in the shaders' 8 x 8 workgroups.
void dispatch(const vk::raii::CommandBuffer& commands, const vk::raii::Pipeline& pipeline, const AoPushData& push) {
    commands.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commands.pushDataEXT(vk::PushDataInfoEXT{
        .offset = 0,
        .data = {.address = &push, .size = sizeof(push)},
    });
    commands.dispatch((push.width + 7) / 8, (push.height + 7) / 8, 1);
}

// Between two compute steps: the first's storage writes, visible to the second.
void compute_to_compute(const vk::raii::CommandBuffer& commands) {
    const vk::MemoryBarrier2 memory{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
    };

    commands.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &memory});
}

}  // namespace

// --- Creating ----------------------------------------------------------------

AmbientOcclusion create_ambient_occlusion(const vk::raii::Device& device) {
    AmbientOcclusion ambient_occlusion;
    ambient_occlusion.search = create_compute_pipeline(device, "ao", "gtaoMain");
    ambient_occlusion.blur = create_compute_pipeline(device, "ao", "blurMain");
    return ambient_occlusion;
}

// --- Recording ---------------------------------------------------------------

void record_ambient_occlusion(
    const vk::raii::CommandBuffer& commands,
    const AmbientOcclusion& ambient_occlusion,
    const Swapchain& swapchain,
    std::uint32_t ao_target,
    std::uint32_t ao_blur_target,
    AoPushData push
) {
    push.width = swapchain.extent.width;
    push.height = swapchain.extent.height;
    push.radius = ambient_occlusion.radius;
    push.slices = ambient_occlusion.slices;
    push.steps = ambient_occlusion.steps;

    // Both images are rewritten from scratch. Frames in flight share them,
    // so this also waits for the previous frame's lighting to finish
    // reading the AO image.
    barrier(commands, swapchain.ao, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);
    barrier(commands, swapchain.ao_blur, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);

    // 1. The horizon search, into the AO image.
    push.target = ao_target;
    dispatch(commands, ambient_occlusion.search, push);
    compute_to_compute(commands);

    // 2. The blur across, into the blur image, then down, back into the AO image.
    push.source = ao_target;
    push.target = ao_blur_target;
    push.blur_axis = 0;
    dispatch(commands, ambient_occlusion.blur, push);
    compute_to_compute(commands);

    push.source = ao_blur_target;
    push.target = ao_target;
    push.blur_axis = 1;
    dispatch(commands, ambient_occlusion.blur, push);

    // The lighting pass samples it.
    barrier(commands, swapchain.ao, vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead);
}
```

CMake already picks up every `.cpp` under `src/` and every `.slang` under `shaders/`, so neither new file needs a build change.

## 9.8 Four passes a frame: `main.cpp`

### Why
`record_frame` grows from two passes to four, and the swapchain images it uses now need six heap slots instead of one.

### How
- **`ScreenSlots`** names the six slots the screen images use, right after the textures: the HDR image, depth, normals, and the AO image as sampled, plus the two AO images as storage. `describe_screen` writes all six, at startup and after every resize.
- **`ScenePipelines`** gathers the pipelines: the prepass's two (opaque, masked), the lighting pass's three, the sky and tone mapping.
- **`draw_batch`** draws one alpha mode's batch, the loop from Chapter 8, now shared by both passes. `set_viewport` likewise.
- **`record_frame`:**
  1. **The prepass.** Depth and normals go from `eUndefined` to attachment layouts. This waits for the previous frame's AO pass and lighting pass to finish reading them. Depth is now **stored**: two later passes read it. Pixels no surface covers keep depth 0, and the AO shader checks the depth before it reads a normal, so the normals' clear value is never used.
  2. **Depth to `eDepthReadOnlyOptimal`, normals to `eShaderReadOnlyOptimal`,** for the compute shaders, then the AO pass.
  3. **The lighting pass.** The depth attachment is **loaded** and its `storeOp` is `eNone`: nothing writes it, so nothing needs storing. The sky goes in before the blended batch, as before. It's drawn wherever the depth is still 0.
  4. **Tone mapping,** unchanged.
- **The heaps** are bound once at the start of the frame. They stay bound across graphics and compute.
- **`O`** flips `Settings::ambient_occlusion`, which becomes `FrameData::ao_enabled`. The AO pass still runs, so view `9` works either way. The title shows `AO on` or `AO off`.

### Code
`game-engine/src/main.cpp`:
```cpp
#include "includes/ambient_occlusion.h"
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

// --- Frames in flight --------------------------------------------------------

// How many frames the CPU may record ahead of the GPU.
constexpr std::size_t frames_in_flight = 2;

constexpr std::uint64_t no_timeout = std::numeric_limits<std::uint64_t>::max();

// What each in-flight frame needs for itself. `data` holds this frame's
// FrameData; the GPU may still be reading the other frame's while the CPU
// writes this one.
struct Frame {
    vk::raii::CommandBuffer commands = nullptr;
    vk::raii::Semaphore image_acquired = nullptr;  // swapchain image is ready to draw into
    vk::raii::Fence done = nullptr;                // GPU finished this frame's commands
    Buffer data;                                   // one FrameData, host-visible
    FrameData* mapped = nullptr;                   // `data`, mapped for the CPU to write
};

// --- Recording a frame -------------------------------------------------------

// Moves `image` between layouts, and makes the `dst` work wait for the `src` work.
// `aspect` is which part of the image: its color, or its depth.
void transition(
    const vk::raii::CommandBuffer& commands,
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

// The three alpha modes, in the order they're drawn: solid surfaces first,
// so see-through ones blend over everything behind them.
constexpr std::array alpha_modes{AlphaMode::opaque, AlphaMode::mask, AlphaMode::blend};

// glTF's front faces wind counter-clockwise, seen from the front. Our
// projection's Y flip (see camera.cpp) only undoes the difference between
// OpenGL's upward Y and Vulkan's downward one, so on screen they still wind
// counter-clockwise. A mirroring transform reverses that.
constexpr vk::FrontFace front_face = vk::FrontFace::eCounterClockwise;
constexpr vk::FrontFace mirrored_front_face = vk::FrontFace::eClockwise;

// Resource heap slots of the swapchain's images, written by describe_screen()
// in main and rewritten whenever the swapchain is rebuilt.
struct ScreenSlots {
    std::uint32_t hdr = 0;             // sampled, by tone mapping
    std::uint32_t depth = 0;           // sampled, by ambient occlusion
    std::uint32_t normals = 0;         // sampled, by ambient occlusion
    std::uint32_t ao = 0;              // sampled, by the lighting pass
    std::uint32_t ao_target = 0;       // storage, for the AO pass
    std::uint32_t ao_blur_target = 0;  // storage, for the AO pass
};

constexpr std::uint32_t screen_slot_count = 6;

// Every graphics pipeline a frame uses. The mesh pipelines are indexed by
// alpha mode, in alpha_modes' order; the prepass has no blended one, since
// see-through surfaces aren't part of it.
struct ScenePipelines {
    std::vector<vk::raii::Pipeline> prepass;
    std::vector<vk::raii::Pipeline> lighting;
    vk::raii::Pipeline background = nullptr;
    vk::raii::Pipeline tonemap = nullptr;
};

// What to draw: every primitive draw in a scene, the frame's data, and how
// to finish the frame. `batches` lists draw indices per alpha mode, in
// drawing order.
struct DrawList {
    vk::Buffer index_buffer;
    vk::DeviceAddress frame = 0;   // this frame's FrameData
    ScreenSlots screen;
    View view = View::lit;
    std::span<const Primitive> primitives;
    std::span<const MeshDraw> mesh_draws;
    std::span<const SceneMaterial> scene_materials;
    std::array<std::span<const std::uint32_t>, alpha_modes.size()> batches;
};

// Draws one alpha mode's batch with `pipeline`. Push data says where the
// frame's data is and which DrawData to use; the primitive's index range
// and vertex offset go to drawIndexed.
void draw_batch(const vk::raii::CommandBuffer& commands, const DrawList& draws, std::size_t mode, const vk::raii::Pipeline& pipeline) {
    commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

    for (const std::uint32_t i : draws.batches[mode]) {
        const MeshDraw& mesh_draw = draws.mesh_draws[i];
        const Primitive& primitive = draws.primitives[mesh_draw.primitive];
        const SceneMaterial& material = draws.scene_materials[primitive.material];

        // Single-sided surfaces are invisible from behind, so the GPU can
        // skip their back faces before running the fragment shader.
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

// The viewport and scissor every pass uses: the whole image. The pipelines
// leave both dynamic.
void set_viewport(const vk::raii::CommandBuffer& commands, vk::Extent2D extent) {
    commands.setViewport(0, vk::Viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(extent.width),
        .height = static_cast<float>(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    });
    commands.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = extent});
}

// Records a frame in four passes:
//   1. the depth prepass: every solid surface's depth and vertex normal,
//   2. ambient occlusion, from those, in compute shaders,
//   3. the lighting, into the HDR image: each alpha mode's batch with that
//      mode's pipeline, against the prepass's depth, with the sky drawn
//      behind everything solid before the see-through batch,
//   4. tone mapping, from the HDR image into the swapchain image, which is
//      then ready to present.
void record_frame(
    const vk::raii::CommandBuffer& commands,
    const Swapchain& swapchain,
    std::uint32_t image_index,
    const ScenePipelines& pipelines,
    const AmbientOcclusion& ambient_occlusion,
    const DescriptorHeaps& heaps,
    const DrawList& draws
) {
    const vk::Image image = swapchain.images[image_index];
    const vk::Image hdr = *swapchain.hdr.handle;
    const vk::Image depth = *swapchain.depth.handle;
    const vk::Image normals = *swapchain.normals.handle;

    commands.reset();
    commands.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Every texture and sampler the shaders read comes from these two heaps.
    // They stay bound for every pass, graphics and compute.
    bind_descriptor_heaps(commands, heaps);

    // One index buffer for the whole scene. Indices go through the GPU's
    // fixed-function index fetch, which also lets it reuse vertices shared
    // between neighbouring triangles.
    commands.bindIndexBuffer(draws.index_buffer, 0, vk::IndexType::eUint32);

    // --- Pass 1: the depth prepass -------------------------------------------

    // The depth buffer and the normals are shared by the frames in flight, so
    // these also wait for the previous frame to finish reading them: the AO
    // pass reads both, and the lighting pass depth-tests against the depth.
    transition(commands, depth,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::AccessFlagBits2::eShaderSampledRead | vk::AccessFlagBits2::eDepthStencilAttachmentRead,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::ImageAspectFlagBits::eDepth
    );

    transition(commands, normals,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite
    );

    // Reverse-Z: 0 is the far plane. The depth is stored this time: the AO
    // pass and the lighting pass both read it.
    const vk::RenderingAttachmentInfo normal_attachment{
        .imageView = *swapchain.normals.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = std::array{0.0f, 0.0f, 0.0f, 0.0f}}},
    };

    const vk::RenderingAttachmentInfo prepass_depth{
        .imageView = *swapchain.depth.view,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = 0.0f}},
    };

    const vk::Rect2D whole_image{.offset = {0, 0}, .extent = swapchain.extent};

    commands.beginRendering(vk::RenderingInfo{
        .renderArea = whole_image,
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &normal_attachment,
        .pDepthAttachment = &prepass_depth,
    });

    set_viewport(commands, swapchain.extent);

    // Solid surfaces only: opaque, then masked.
    for (std::size_t mode = 0; mode < pipelines.prepass.size(); ++mode) {
        draw_batch(commands, draws, mode, pipelines.prepass[mode]);
    }

    commands.endRendering();

    // --- Pass 2: ambient occlusion -------------------------------------------

    // From here on the depth is only read: by the AO pass, and as the
    // lighting pass's depth test, which eDepthReadOnlyOptimal allows at once.
    transition(commands, depth,
        vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthReadOnlyOptimal,
        vk::PipelineStageFlagBits2::eLateFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eEarlyFragmentTests
            | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::AccessFlagBits2::eShaderSampledRead | vk::AccessFlagBits2::eDepthStencilAttachmentRead,
        vk::ImageAspectFlagBits::eDepth
    );

    transition(commands, normals,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead
    );

    record_ambient_occlusion(commands, ambient_occlusion, swapchain, draws.screen.ao_target, draws.screen.ao_blur_target,
        AoPushData{.frame = draws.frame, .depth = draws.screen.depth, .normals = draws.screen.normals});

    // --- Pass 3: the lighting --------------------------------------------------

    // The HDR image is shared by the frames in flight too: this waits for
    // the previous frame's tone mapping to finish reading it.
    transition(commands, hdr,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite
    );

    // Every pixel is drawn over, by the scene or the sky; clearing is just
    // cheaper than loading what was there.
    const vk::RenderingAttachmentInfo hdr_attachment{
        .imageView = *swapchain.hdr.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = std::array{0.0f, 0.0f, 0.0f, 1.0f}}},
    };

    // The prepass's depth, loaded and tested against but not written.
    const vk::RenderingAttachmentInfo lighting_depth{
        .imageView = *swapchain.depth.view,
        .imageLayout = vk::ImageLayout::eDepthReadOnlyOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eNone,
    };

    commands.beginRendering(vk::RenderingInfo{
        .renderArea = whole_image,
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &hdr_attachment,
        .pDepthAttachment = &lighting_depth,
    });

    set_viewport(commands, swapchain.extent);

    for (std::size_t mode = 0; mode < alpha_modes.size(); ++mode) {
        // The sky goes in once everything solid is drawn: it only covers
        // pixels still at the far plane. See-through surfaces then blend over
        // it like over anything else.
        if (alpha_modes[mode] == AlphaMode::blend) {
            commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.background);

            const PushData push{.frame = draws.frame, .draw_index = 0};
            commands.pushDataEXT(vk::PushDataInfoEXT{
                .offset = 0,
                .data = {.address = &push, .size = sizeof(push)},
            });

            commands.draw(3, 1, 0, 0);
        }

        draw_batch(commands, draws, mode, pipelines.lighting[mode]);
    }

    commands.endRendering();

    // --- Pass 4: tone mapping ------------------------------------------------

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

    // The full-screen triangle writes every pixel, so there's nothing to
    // clear or load first.
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

    commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.tonemap);

    const TonemapPushData push{.hdr_image = draws.screen.hdr, .view = draws.view};

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

// --- Events ------------------------------------------------------------------

// What the keyboard controls, besides the camera.
struct Settings {
    View view = View::lit;
    SkySource sky = SkySource::atmosphere;
    float hours = 10.0f;                    // time of day, 0 to 24
    float exposure_compensation = 0.0f;     // stops brighter (+) or darker (-) than metered
    bool ambient_occlusion = true;
};

// The views' names, in View's order, for the window title.
constexpr std::array view_names{
    "Lit", "Base color", "Normal", "Vertex normal", "Metallic", "Roughness", "Occlusion", "Emissive", "Ambient occlusion",
};

// Handles every pending event and fills in `input` for this frame. False once
// the window was closed or Escape pressed.
//   1-9   pick the view
//   e     switch between the simulated sky and the photographed one
//   o     switch ambient occlusion off and on, to compare
//   [ ]   time of day, a quarter of an hour earlier or later
//   - =   exposure, half a stop darker or brighter: like a camera's
//         exposure compensation, + is brighter
// Holding a key repeats it.
bool poll_events(SDL_Window* window, CameraInput& input, Settings& settings) {
    input = CameraInput{};
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        const bool escape = event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE;

        if (event.type == SDL_EVENT_QUIT || escape) {
            return false;
        }

        if (event.type == SDL_EVENT_KEY_DOWN) {
            const SDL_Keycode key = event.key.key;

            // SDLK_1 to SDLK_9 are consecutive key codes.
            if (key >= SDLK_1 && key < SDLK_1 + view_names.size()) {
                settings.view = static_cast<View>(key - SDLK_1);
            } else if (key == SDLK_O) {
                settings.ambient_occlusion = !settings.ambient_occlusion;
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

    // While a button is held, relative mode hides the cursor and keeps
    // reporting movement, so a drag can't run into the edge of the screen.
    const bool dragging = input.right_button || input.left_button || input.middle_button;

    if (dragging != SDL_GetWindowRelativeMouseMode(window)) {
        SDL_SetWindowRelativeMouseMode(window, dragging);
    }

    return true;
}

}  // namespace

int main() {
    try {
        // --- Window and instance ---------------------------------------------

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

        // Declaration order matters: each object is destroyed before the ones above it.
        // The window comes first: creating it loads Vulkan into SDL, which
        // required_vulkan_extensions() needs.
        Window window = make_vulkan_window(1920, 1080, "game-engine", true);

        vk::raii::Instance instance = create_instance(context, SdlContext::required_vulkan_extensions(), validation);
        vk::raii::DebugUtilsMessengerEXT messenger = validation
            ? create_debug_messenger(instance)
            : vk::raii::DebugUtilsMessengerEXT(nullptr);

        vk::raii::SurfaceKHR surface = create_surface(instance, window.get());

        // --- GPU, device and swapchain ---------------------------------------

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

        // --- Pipelines -------------------------------------------------------

        // recreate_swapchain() picks the same formats again, so the pipelines
        // stay valid across resizes.
        //   - The prepass draws normals and depth, for opaque and masked
        //     materials; the lighting pass draws every alpha mode into the
        //     HDR image.
        //   - The sky draws into the HDR image, behind the scene; tone mapping
        //     writes the swapchain image.
        ScenePipelines pipelines;

        for (const AlphaMode mode : alpha_modes) {
            if (mode != AlphaMode::blend) {
                pipelines.prepass.push_back(create_mesh_pipeline(device, normal_format, depth_format, mode, MeshPass::depth_normals));
            }
            pipelines.lighting.push_back(create_mesh_pipeline(device, hdr_format, depth_format, mode, MeshPass::lighting));
        }

        pipelines.background = create_fullscreen_pipeline(device, "background", hdr_format, depth_format);
        pipelines.tonemap = create_fullscreen_pipeline(device, "tonemap", swapchain.format);

        const AmbientOcclusion ambient_occlusion = create_ambient_occlusion(device);

        // --- Per-frame resources ---------------------------------------------

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
        for (vk::raii::CommandBuffer& commands : command_buffers) {
            // Host-coherent: the CPU's writes reach the GPU without a flush.
            Buffer data = create_buffer(device, *gpu, sizeof(FrameData),
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            auto* mapped = static_cast<FrameData*>(data.memory.mapMemory(0, sizeof(FrameData)));

            frames.push_back(Frame{
                .commands = std::move(commands),
                .image_acquired = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
                // Start signalled, so the first wait on each frame returns straight away.
                .done = vk::raii::Fence(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}),
                .data = std::move(data),
                .mapped = mapped,
            });
        }

        // --- Scene -----------------------------------------------------------

        // The glTF file to draw, under lecture-md/game-engine/assets.
        const std::filesystem::path scene_file = std::filesystem::path(ASSET_DIR) / "Sponza/Sponza.gltf";

        const std::uint64_t load_start = SDL_GetTicksNS();
        const Scene scene = load_gltf(scene_file);
        std::println("Loaded {}: {} vertices, {} triangles, {} primitives, {} draws, {} materials, {} images",
            scene_file.filename().string(), scene.vertices.size(), scene.indices.size() / 3,
            scene.primitives.size(), scene.draws.size(), scene.materials.size(), scene.images.size());

        // Each draw's matrices. The normal matrix is the transposed inverse of
        // the model matrix: under non-uniform scale, transforming a normal by
        // the model matrix itself would tilt it off the surface.
        std::vector<DrawData> draw_data;

        for (const MeshDraw& draw : scene.draws) {
            draw_data.push_back(DrawData{
                .model = draw.model,
                .normal_matrix = glm::transpose(glm::inverse(draw.model)),
                .material = scene.primitives[draw.primitive].material,
            });
        }

        // Vertices and draw data are read through pointers; indices go to the
        // GPU's index fetch, so that buffer is an index buffer.
        const Buffer vertex_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.vertices)), vk::BufferUsageFlagBits::eShaderDeviceAddress);
        const Buffer index_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.indices)), vk::BufferUsageFlagBits::eIndexBuffer);
        const Buffer draw_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(draw_data)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        // Most files have no lights, and a buffer can't be empty: then there's
        // no buffer, and the shader's light count is 0.
        const Buffer light_buffer = scene.lights.empty() ? Buffer{} : upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.lights)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        std::println("Lights: {} from the file, plus the sun", scene.lights.size());

        // --- Textures and materials ------------------------------------------

        // Decode every image, upload them with mipmaps, and describe them in
        // the descriptor heap. Texture 0 is white; scene image i is texture i + 1.
        const std::uint64_t texture_start = SDL_GetTicksNS();
        const std::vector<Texture> textures = create_scene_textures(device, *gpu, queue, command_pool, scene);
        // After the textures: the swapchain images' slots, then the environment's.
        DescriptorHeaps heaps = create_descriptor_heaps(device, *gpu, queue, command_pool, textures, scene.samplers,
            screen_slot_count + environment_slot_count);

        const auto first_screen_slot = static_cast<std::uint32_t>(textures.size());
        const ScreenSlots screen{
            .hdr = first_screen_slot,
            .depth = first_screen_slot + 1,
            .normals = first_screen_slot + 2,
            .ao = first_screen_slot + 3,
            .ao_target = first_screen_slot + 4,
            .ao_blur_target = first_screen_slot + 5,
        };

        // The swapchain's images are recreated with it, so their descriptors
        // are rewritten every time: after this, only while the GPU is idle.
        const auto describe_screen = [&] {
            const auto whole = [](const Image& image, vk::ImageAspectFlags aspect) {
                return vk::ImageViewCreateInfo{
                    .image = *image.handle,
                    .viewType = vk::ImageViewType::e2D,
                    .format = image.format,
                    .subresourceRange = {.aspectMask = aspect, .levelCount = 1, .layerCount = 1},
                };
            };
            constexpr auto color = vk::ImageAspectFlagBits::eColor;
            constexpr auto storage = vk::DescriptorType::eStorageImage;

            write_image_descriptor(device, heaps, screen.hdr, whole(swapchain.hdr, color));
            write_image_descriptor(device, heaps, screen.depth, whole(swapchain.depth, vk::ImageAspectFlagBits::eDepth));
            write_image_descriptor(device, heaps, screen.normals, whole(swapchain.normals, color));
            write_image_descriptor(device, heaps, screen.ao, whole(swapchain.ao, color));
            write_image_descriptor(device, heaps, screen.ao_target, whole(swapchain.ao, color), storage);
            write_image_descriptor(device, heaps, screen.ao_blur_target, whole(swapchain.ao_blur, color), storage);
        };

        describe_screen();

        // recreate_swapchain() waits for the GPU to go idle, so the slots are
        // free to rewrite straight afterwards.
        const auto resize = [&] {
            recreate_swapchain(swapchain, device, *gpu, surface, window.get());
            describe_screen();
        };

        // --- The environment -------------------------------------------------

        const std::uint64_t environment_start = SDL_GetTicksNS();
        Environment environment = create_environment(device, *gpu, queue, command_pool, heaps, first_screen_slot + screen_slot_count,
            std::filesystem::path(ASSET_DIR) / "environments/kloppenheim_06_puresky_2k.hdr");

        std::println("Environment: {:.0f} ms", static_cast<double>(SDL_GetTicksNS() - environment_start) * 1e-6);

        std::println("Textures: {} in {:.0f} ms (whole load {:.0f} ms)", textures.size(),
            static_cast<double>(SDL_GetTicksNS() - texture_start) * 1e-6,
            static_cast<double>(SDL_GetTicksNS() - load_start) * 1e-6);

        // Heap indices are one past the scene's: image i is texture i + 1 and
        // sampler i is sampler i + 1, so "none" (-1) becomes 0, the white
        // texture or the default sampler.
        const auto slot = [](const TextureRef& ref) {
            return TextureSlot{
                .texture = static_cast<std::uint32_t>(ref.image + 1),
                .sampler = static_cast<std::uint32_t>(ref.sampler + 1),
                .uv_set = ref.uv_set,
            };
        };

        std::vector<Material> materials;

        for (const SceneMaterial& material : scene.materials) {
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

        // Draw indices by alpha mode. Opaque and masked draws can go in any
        // order, so their batches are fixed; blended ones are sorted by
        // distance every frame, below.
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

        // --- Frame loop ------------------------------------------------------

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

            // --- Update -----------------------------------------------------

            // Seconds since the last frame, so movement doesn't depend on frame rate.
            const std::uint64_t ticks = SDL_GetTicksNS();
            const float seconds = static_cast<float>(ticks - previous_ticks) * 1e-9f;
            previous_ticks = ticks;

            update_camera(camera, input, seconds);

            const float aspect = static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height);

            // Blending mixes with what's already drawn, so see-through draws go
            // back to front: farthest from the camera first. Sorting by each
            // draw's center is approximate, but right for separate objects.
            std::vector<std::uint32_t>& blended = batches[static_cast<std::size_t>(AlphaMode::blend)];

            std::ranges::sort(blended, std::ranges::greater{}, [&](std::uint32_t i) {
                const glm::vec3 offset = scene.draws[i].center - camera.position;
                return glm::dot(offset, offset);
            });

            // The sky: rebuilt whenever its source changes, or the time of day
            // moves the sun in the simulated one. That takes a few milliseconds
            // and waits for the GPU, which is fine for a key press.
            const glm::vec3 sun_direction = sun_direction_at(settings.hours);
            const bool sky_moved = settings.sky == SkySource::atmosphere && settings.hours != sky_settings.hours;

            if (settings.sky != sky_settings.sky || sky_moved) {
                update_environment(environment, device, queue, heaps, settings.sky, sun_direction);
                sky_settings = settings;
            }

            // Light and exposure. The meter reads the light falling on flat
            // ground: the sky's irradiance on an upward-facing surface, plus
            // the sun's share at its angle (Rec. 709 luminance of each).
            // Compensation works like a camera's: +1 is a stop brighter, which
            // means a lower EV (EV measures the light the camera expects).
            const auto luminance = [](glm::vec3 c) { return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b; };
            const glm::vec3 sun_illuminance = environment.mapped->sun_illuminance;
            const float ground_illuminance = luminance(sky_irradiance(*environment.mapped, {0.0f, 1.0f, 0.0f}))
                + luminance(sun_illuminance) * std::max(sun_direction.y, 0.0f);

            const float ev100 = std::clamp(metered_ev100(ground_illuminance), -2.0f, 16.0f) - settings.exposure_compensation;
            const float exposure = exposure_from_ev100(ev100);

            // The title shows the view, the sky, the time, the exposure and
            // whether ambient occlusion is on, whenever one changes.
            if (settings.view != shown_settings.view || settings.sky != shown_settings.sky
                || settings.hours != shown_settings.hours
                || settings.exposure_compensation != shown_settings.exposure_compensation
                || settings.ambient_occlusion != shown_settings.ambient_occlusion) {
                const auto minutes = static_cast<int>(std::lround(settings.hours * 60.0f));
                const std::string sky = settings.sky == SkySource::atmosphere
                    ? std::format("sky {:02}:{:02}", minutes / 60, minutes % 60)
                    : std::string("photographed sky");
                const std::string title = std::format("game-engine: {}, {}, EV {:.1f}, AO {}",
                    view_names[static_cast<std::size_t>(settings.view)], sky, ev100,
                    settings.ambient_occlusion ? "on" : "off");

                SDL_SetWindowTitle(window.get(), title.c_str());
                shown_settings = settings;
            }

            // --- Render -----------------------------------------------------

            Frame& frame = frames[frame_count % frames_in_flight];

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
                .ambient_occlusion = screen.ao,
                .ao_enabled = settings.ambient_occlusion ? 1u : 0u,
            };

            const DrawList draws{
                .index_buffer = *index_buffer.handle,
                .frame = frame.data.address,
                .screen = screen,
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
            record_frame(frame.commands, swapchain, image_index, pipelines, ambient_occlusion, heaps, draws);

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

        // --- Shutdown --------------------------------------------------------

        // Everything above is destroyed on the way out of this scope; the GPU must be idle first.
        device.waitIdle();
        std::println("Presented {} frames", frame_count);
    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

## 9.9 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The window** opens at 10:00 in the simulated sky, as before. The title now ends in `AO on`.
- **`9`** shows the ambient occlusion: white almost everywhere, smoothly darkening into the corners where walls meet floors, under the arches, behind the curtains and around the bases of the columns. There should be no speckle: the blur removes the per-pixel noise. The sky shows behind, untouched by tone mapping, as in the other debug views.
- **`O`** switches AO off and on in the lit view. The difference shows most in the shade: the courtyard's corners, the walkways behind the columns and the folds of the curtains darken, while sunlit surfaces barely change.
- **Toward evening** (`]`), when sky light is all there is, the AO matters most: the interior reads as a solid space instead of glowing evenly.
- **Resizing the window** keeps working: the new images are rebuilt with the swapchain.
- **No `[validation …]` lines.**

Next, in Chapter 10, see-through surfaces get drawn correctly whatever order they overlap in: weighted, blended order-independent transparency.

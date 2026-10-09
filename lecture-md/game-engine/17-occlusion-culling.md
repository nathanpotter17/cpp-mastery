# Chapter 17: Occlusion culling

By the end of this chapter, the cull drops draws hidden behind other draws, not just those outside the view. Chapter 12's cull tests each draw's box against the six planes of the view: a box behind the camera, or off to the side, is dropped. A box behind a wall survives, and the GPU draws every one of its triangles, finds each covered by the wall's depth, and throws the pixels away. Looking down a street, that's most of the city.

The fix is **hierarchical Z-buffer visibility**, after Greene, Kass and Miller (SIGGRAPH 1993): a **depth pyramid**, the depth buffer with a chain of smaller copies in which each texel holds the farthest depth of the texels above it. A draw's box projects to a rectangle on screen; in the GPU form of the test, at the pyramid level where that rectangle spans at most two texels each way, four reads give the farthest depth under the whole rectangle. If the box's nearest point is farther than that, the box is hidden, wherever in the rectangle it is. In games, the test is known as **Hi-Z occlusion culling**, and it runs where our cull already runs: in compute, one thread per draw.

Which depth buffer, though? The one this frame draws is what the cull decides. The usual way out is to test against the previous frame's depth, and accept that a draw which just became visible appears a frame late: fine while the camera is still, a flicker of popping when it isn't. We take the two-phase scheme that Sebastian Aaltonen described for RedLynx's engine in Haar and Aaltonen's "GPU-Driven Rendering Pipelines" (SIGGRAPH 2015, Advances in Real-Time Rendering course), a variant of which Unreal's Nanite uses: an **early phase** tests every draw against the previous frame's pyramid, and the depth prepass draws what it keeps. A new pyramid is built from that depth. A **late phase** tests the draws the early phase rejected against the new pyramid, and the prepass continues with whatever it adds. Every draw visible this frame is drawn this frame: the early phase is only a guess at what's hidden, and the late phase corrects it before anything is shaded. Nothing pops, however fast the camera moves or cuts.

Sponza has 103 draws, most of them big pieces of the building, so there's little for occlusion to drop. This chapter's test scene is a field of 16,379 utility boxes, each drawn from one mesh, on a plain 256 m across: standing among them, almost every box is hidden behind a nearer one, and from far away, each is a few pixels wide. The window title now says how many draws each phase kept.

This chapter builds on [Chapter 16](16-clustered-lights.md).

## 17.1 The depth pyramid: `image.h`, `image.cpp`, `swapchain.h`, `swapchain.cpp`

### Why
The pyramid is an image with mip levels, kept from one frame to the next, and `create_image` only makes single-level images.

### How
- **`create_image`** takes `mip_levels`, 1 unless given, makes the image with that many, and views all of them. `Image` remembers the count.
- **`depth_pyramid_format`** is `R32Sfloat`: the depth buffer's own 32-bit floats, since every level holds depths read from it. No depth format is required to support storage images, and in practice none does, which is why level 0 is a copy rather than the depth buffer itself.
- **`mip_level_count`:** how many levels an image has when each halves the one above, rounding down, until the larger side is one texel: `floor(log2(larger side)) + 1`. 1920 × 1080 has 11 levels, from 1920 × 1080 down to 1 × 1.
- **`max_depth_pyramid_levels`,** 16, is how many resource heap slots the pyramid reserves, one per level (17.6): enough for screens up to 65,535 pixels wide, a side of 32,768 needing all 16.
- **`Swapchain::depth_pyramid`** is made with the other screen-sized images, as storage, for the compute shaders that build and read it, and as a transfer destination, for the one clear it gets when made (17.4). Unlike the others, it's read before this frame writes it: the early cull reads what the previous frame built.

### Code
`game-engine/src/includes/image.h`:
```cpp
#pragma once

#include "includes/vulkan_setup.h"

#include <cstdint>

// A VkImage, the memory behind it, and a view of the whole image. Members are destroyed bottom-up: the view, then the image, then its memory.
struct Image {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image handle = nullptr;
    vk::raii::ImageView view = nullptr;
    vk::Format format = vk::Format::eUndefined;
    vk::Extent2D extent;
    std::uint32_t mip_levels = 1;
};

// A 2D image in device-local memory, with `mip_levels` levels, and a view of its `aspect` (color, or depth for a depth buffer) covering all of them.
Image create_image(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    vk::Extent2D extent,
    vk::Format format,
    vk::ImageUsageFlags usage,
    vk::ImageAspectFlags aspect,
    std::uint32_t mip_levels = 1
);
```

`game-engine/src/image.cpp`:
```cpp
#include "includes/image.h"

#include "includes/buffer.h"

#include <stdexcept>

Image create_image(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    vk::Extent2D extent,
    vk::Format format,
    vk::ImageUsageFlags usage,
    vk::ImageAspectFlags aspect,
    std::uint32_t mip_levels
) {
    // Not every format can be used for everything; ask before creating.
    const vk::FormatProperties support = gpu.device.getFormatProperties(format);
    const bool depth = static_cast<bool>(aspect & vk::ImageAspectFlagBits::eDepth);

    if (depth && !(support.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)) {
        throw std::runtime_error(vk::to_string(format) + " can't be a depth attachment on this GPU");
    }

    Image image;
    image.format = format;
    image.extent = extent;
    image.mip_levels = mip_levels;

    // 1. The image object. Optimal tiling lets the GPU arrange texels however
    //    is fastest for it; we never read them from the CPU.
    image.handle = vk::raii::Image(device, vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = mip_levels,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    });

    // 2. Device-local memory that fits it, bound at offset 0.
    const vk::MemoryRequirements requirements = image.handle.getMemoryRequirements();

    image.memory = vk::raii::DeviceMemory(device, vk::MemoryAllocateInfo{
        .allocationSize = requirements.size,
        .memoryTypeIndex = find_memory_type(gpu, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
    });

    image.handle.bindMemory(*image.memory, 0);

    // 3. A view of the whole image, every level, which is what rendering attaches to.
    image.view = vk::raii::ImageView(device, vk::ImageViewCreateInfo{
        .image = *image.handle,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = mip_levels,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });

    return image;
}
```

`game-engine/src/includes/swapchain.h`:
```cpp
#pragma once

#include "includes/image.h"
#include "includes/vulkan_setup.h"

#include <cstdint>
#include <vector>

// 32-bit float depth: the precision reverse-Z depth needs (see camera.cpp).
constexpr vk::Format depth_format = vk::Format::eD32Sfloat;

// 16-bit floats per channel for the scene before tone mapping: enough range for sunlit highlights many times brighter than white (up to 65504).
constexpr vk::Format hdr_format = vk::Format::eR16G16B16A16Sfloat;

// The prepass's normals: two 16-bit floats, a unit vector in octahedral encoding (see mesh.slang).
constexpr vk::Format normal_format = vk::Format::eR16G16Sfloat;

// Ambient occlusion: the bent normal in RGB and the visibility in A.
constexpr vk::Format ao_format = vk::Format::eR16G16B16A16Sfloat;

// Ambient occlusion's half-resolution input: each 2 x 2 block's nearest distance in front of the camera. Its normal uses normal_format.
constexpr vk::Format ao_depth_format = vk::Format::eR32Sfloat;

// The depth pyramid (culling.h): the depth buffer's values, so the same 32-bit floats, with a mip chain down to one texel.
constexpr vk::Format depth_pyramid_format = vk::Format::eR32Sfloat;

// The most levels the pyramid can have: enough for images up to 32,768 pixels wide. The resource heap reserves a slot per level.
constexpr std::uint32_t max_depth_pyramid_levels = 16;

// How many mip levels an image of `extent` has when every level halves the one above, rounding down, until one texel is left: floor(log2(the larger side)) + 1.
std::uint32_t mip_level_count(vk::Extent2D extent);

// Weighted blended transparency's two sums (see mesh.slang): the weighted color and coverage, and the share of the scene that still shows through.
constexpr vk::Format accum_format = vk::Format::eR16G16B16A16Sfloat;
constexpr vk::Format reveal_format = vk::Format::eR16Sfloat;

// The window's images, plus what we need per image to draw into them. Members are destroyed bottom-up, so the views and semaphores go before the swapchain that owns the images.
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

    // Images the size of the swapchain images, which every frame writes from scratch before reading, so frames in flight can share them:
    //   depth    the depth prepass writes it; ambient occlusion and the lighting pass then read it
    //   normals  the prepass's normals, for ambient occlusion
    //   ao       ambient occlusion, written by a compute shader, read by the lighting pass
    //   accum    see-through surfaces' weighted color and coverage, summed
    //   reveal   how much of the scene still shows through them
    //   hdr      the lit scene, which tone mapping writes to the swapchain image
    // the depth pyramid, the one image kept from frame to frame:
    //   depth_pyramid  level 0 is the depth buffer copied; each level below holds the farthest depth of the 2 x 2 texels above it (culling.h). The cull reads it, as last frame's and this frame's
    // and, at half the size, rounded up, ambient occlusion's working images:
    //   ao_depth    each 2 x 2 block's nearest distance in front of the camera
    //   ao_normals  that surface's normal
    //   ao_raw      the horizon search's result, and the blur's
    //   ao_blur     the blur's halfway point
    Image depth;
    Image normals;
    Image depth_pyramid;
    Image ao;
    Image ao_depth;
    Image ao_normals;
    Image ao_raw;
    Image ao_blur;
    Image accum;
    Image reveal;
    Image hdr;
};

Swapchain create_swapchain(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::SurfaceKHR &surface,
    SDL_Window *window
);

// Rebuilds `swapchain` for the window's current size (after a resize). Waits for the GPU to go idle first.
void recreate_swapchain(
    Swapchain &swapchain,
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::SurfaceKHR &surface,
    SDL_Window *window
);
```

`game-engine/src/swapchain.cpp`:
```cpp
#include "includes/swapchain.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

    // Choosing the swapchain's settings

    vk::SurfaceFormatKHR choose_format(const std::vector<vk::SurfaceFormatKHR> &formats) {
        // 8-bit BGRA with sRGB encoding: shaders write linear colors and the GPU encodes them to sRGB on the way out. Otherwise take what the surface offers.
        for (const vk::SurfaceFormatKHR &format : formats) {
            if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return format;
            }
        }

        return formats.front();
    }

    vk::Extent2D choose_extent(const vk::SurfaceCapabilitiesKHR &capabilities, int width, int height) {
        // Most platforms dictate the size. Wayland reports 0xFFFFFFFF and lets us pick.
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        return {
            std::clamp(static_cast<std::uint32_t>(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp(static_cast<std::uint32_t>(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
        };
    }

    vk::CompositeAlphaFlagBitsKHR choose_composite_alpha(vk::CompositeAlphaFlagsKHR supported) {
        for (auto mode : {vk::CompositeAlphaFlagBitsKHR::eOpaque, vk::CompositeAlphaFlagBitsKHR::eInherit,
                          vk::CompositeAlphaFlagBitsKHR::ePreMultiplied, vk::CompositeAlphaFlagBitsKHR::ePostMultiplied}) {
            if (supported & mode) {
                return mode;
            }
        }

        return vk::CompositeAlphaFlagBitsKHR::eOpaque;
    }

    // Building one

    Swapchain build(
        const vk::raii::Device &device,
        const GpuChoice &gpu,
        const vk::raii::SurfaceKHR &surface,
        SDL_Window *window,
        vk::SwapchainKHR old_swapchain
    ) {
        Swapchain swapchain;
        SDL_GetWindowSizeInPixels(window, &swapchain.window_width, &swapchain.window_height);

        const auto capabilities = gpu.device.getSurfaceCapabilitiesKHR(*surface);
        const auto format = choose_format(gpu.device.getSurfaceFormatsKHR(*surface));

        swapchain.format = format.format;
        swapchain.extent = choose_extent(capabilities, swapchain.window_width, swapchain.window_height);

        // One more than the minimum, so we rarely wait on the driver for an image. A maxImageCount of 0 means no limit.
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

        // Every level of the depth pyramid is written and read by compute shaders (storage); it's cleared once when made (transfer).
        swapchain.depth_pyramid = create_image(device, gpu, swapchain.extent, depth_pyramid_format,
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst, vk::ImageAspectFlagBits::eColor,
            mip_level_count(swapchain.extent));

        // Ambient occlusion works at half resolution, rounded up, and writes its result at full resolution. Compute shaders write all of them (storage); the lighting pass samples the full-resolution one.
        const vk::Extent2D half{(swapchain.extent.width + 1) / 2, (swapchain.extent.height + 1) / 2};
        swapchain.ao_depth = create_image(device, gpu, half, ao_depth_format,
            vk::ImageUsageFlagBits::eStorage, vk::ImageAspectFlagBits::eColor);
        swapchain.ao_normals = create_image(device, gpu, half, normal_format,
            vk::ImageUsageFlagBits::eStorage, vk::ImageAspectFlagBits::eColor);

        for (Image *image : {&swapchain.ao_raw, &swapchain.ao_blur}) {
            *image = create_image(device, gpu, half, ao_format,
                vk::ImageUsageFlagBits::eStorage, vk::ImageAspectFlagBits::eColor);
        }

        swapchain.ao = create_image(device, gpu, swapchain.extent, ao_format,
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor);

        // Drawn into by the transparency pass, then read by its composite.
        swapchain.accum = create_image(device, gpu, swapchain.extent, accum_format,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor);
        swapchain.reveal = create_image(device, gpu, swapchain.extent, reveal_format,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor);

        // Drawn into, then read by the tone-mapping shader.
        swapchain.hdr = create_image(device, gpu, swapchain.extent, hdr_format,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor);

        return swapchain;
    }

}  // namespace

// Create and recreate

std::uint32_t mip_level_count(vk::Extent2D extent) {
    return static_cast<std::uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
}

Swapchain create_swapchain(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::SurfaceKHR &surface,
    SDL_Window *window
) {
    return build(device, gpu, surface, window, nullptr);
}

void recreate_swapchain(
    Swapchain &swapchain,
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::SurfaceKHR &surface,
    SDL_Window *window
) {
    // Nothing may still be using the old images, views or semaphores.
    device.waitIdle();

    Swapchain next = build(device, gpu, surface, window, *swapchain.handle);

    // Destroy the old views and semaphores while their images still exist, then the old swapchain itself and the images that go with it.
    swapchain.views.clear();
    swapchain.rendered.clear();
    swapchain = std::move(next);
}
```

## 17.2 The cull's data: `shader_types.h`, `shared.slangh`

### Why
The cull needs the pyramid and the screen's size, each phase needs tables of its own, and the vertex shader needs to know which phase's instances it's drawing.

### How
- **`FrameData`** gains `depth_pyramid`, the resource heap slot of the pyramid's level 0, with level `i` at `depth_pyramid + i`; `depth_pyramid_levels`; and `screen`, the depth buffer's size in pixels. It loses `instances`: each phase has its own. 360 bytes, no padding.
- **`PushData`** gains `instances`: the phase's visible draws, which `draw_mode` pushes with the frame. 16 bytes.
- **`CullTables`** gains `early_visible`, the early phase's flags, which the late phase reads so as not to test a draw twice. Each phase has a table: the shared order, groups and lists, then its own buffers. 96 bytes.
- **`CullPushData`** also serves the two steps that build the pyramid: the slot read, the slot written, and their sizes in texels. Push data follows std430 rules, where a `uvec2` starts on an 8-byte boundary: two pointers and two slots fill the first 24 bytes, so `source_size` lands on one. 40 bytes.

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

// One per draw, in a GPU buffer the shaders index. A draw is placed in a world cell (cells.h): its model matrix moves the primitive into the cell, measured from the cell's corner. A shadow ray that hits a draw's triangle finds the triangle's vertices through first_index and vertex_offset, as drawIndexed does. The cull tests the draw's box.
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

// Lights

// KHR_lights_punctual's three kinds of light. "Punctual" means infinitely small: all of a light's power comes from one point, or one direction.
enum class LightType : std::uint32_t {
    directional,  // like the sun: parallel rays, intensity in lux
    point,        // shines in every direction, intensity in candela
    spot,         // a point light limited to a cone, intensity in candela
};

// One light from the file, placed in a world cell (cells.h); its direction is along the world's axes.
struct Light {
    glm::vec3 offset;     // point and spot lights: where in `cell` the light is (cells.h)
    float range;          // point and spot lights: distance where the light fades to nothing
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

// Views

// What the fragment shader outputs: the shaded scene, one material input on its own, for checking that each one loaded correctly, the ambient occlusion, how much of the sun's light reaches each point, or how many lights each pixel's cluster holds. Keys 1-9 and 0 pick one, L the last.
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
    light_count,        // how many lights the pixel's cluster holds, as a heat map
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
//   - Pointers come right after the matrices, and at the end, so all of them land on 8-byte boundaries with no padding.
// Shaders work in camera-relative space: the world's axes, with the camera at the origin. A position given by a cell and an offset (cells.h) is moved into it by subtracting the camera's cell and offset.
struct FrameData {
    glm::mat4 view_projection;          // camera-relative space -> clip space
    glm::mat4 inverse_view_projection;  // clip space -> camera-relative space
    vk::DeviceAddress vertices;         // the scene's vertices
    vk::DeviceAddress indices;          // the scene's indices, for shadow rays' alpha tests
    vk::DeviceAddress draws;            // one DrawData per draw
    vk::DeviceAddress materials;        // the scene's materials
    vk::DeviceAddress lights;           // the scene's lights: directional ones first
    vk::DeviceAddress environment;      // the EnvironmentInfo
    vk::DeviceAddress scene_tlas;       // the top-level acceleration structure, for ray queries
    glm::ivec3 camera_cell;             // the world cell the camera is in
    float exposure;                     // scales light into the 0..1 range the tone mapper expects
    glm::vec3 sun_direction;            // unit vector pointing toward the sun
    std::uint32_t directional_light_count;  // the first lights, which reach everywhere
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
    glm::vec3 camera_forward;           // the way the camera looks: view depth is distance along it
    std::uint32_t shadow_ray_budget;    // 0: every light traces its shadow ray; N: only the N strongest
    glm::uvec2 cluster_tiles;           // the light clusters' tiles across and down (light_clusters.h)
    std::uint32_t depth_pyramid;        // resource heap slot of the depth pyramid's level 0; level i is at depth_pyramid + i (storage)
    std::uint32_t depth_pyramid_levels; // how many levels it has
    glm::uvec2 screen;                  // the depth buffer's size in pixels: the pyramid's level 0
    vk::DeviceAddress light_clusters;   // per cluster, one bit per visible light
    vk::DeviceAddress visible_lights;   // how many local lights are in view, then their indices
};

static_assert(sizeof(FrameData) == 360);
static_assert(offsetof(FrameData, vertices) == 128);
static_assert(offsetof(FrameData, scene_tlas) == 176);
static_assert(offsetof(FrameData, camera_cell) == 184);
static_assert(offsetof(FrameData, sun_direction) == 200);
static_assert(offsetof(FrameData, sun_illuminance) == 216);
static_assert(offsetof(FrameData, sky_cube) == 232);
static_assert(offsetof(FrameData, sun_angular_radius) == 252);
static_assert(offsetof(FrameData, ambient_occlusion) == 256);
static_assert(offsetof(FrameData, camera_offset) == 264);
static_assert(offsetof(FrameData, tlas_offset) == 276);
static_assert(offsetof(FrameData, sky_view) == 288);
static_assert(offsetof(FrameData, camera_forward) == 304);
static_assert(offsetof(FrameData, depth_pyramid) == 328);
static_assert(offsetof(FrameData, screen) == 336);
static_assert(offsetof(FrameData, light_clusters) == 344);

// Push data

// Written with vkCmdPushDataEXT before each pipeline's draws: where this frame's data is, and the cull phase's instances, where the vertex shader looks up which DrawData an instance draws.
struct PushData {
    vk::DeviceAddress frame;
    vk::DeviceAddress instances;
};

static_assert(sizeof(PushData) == 16);

// GPU culling (culling.h, cull.slang)

// Draws of one primitive in one draw list, a run of the cull's order: drawn as one instanced command, of as many instances as are in view.
struct DrawGroup {
    std::uint32_t first;          // where its draws start in the order
    std::uint32_t count;          // how many draws
    std::uint32_t list;           // its draw list
    std::uint32_t index_count;    // the primitive's drawIndexed arguments
    std::uint32_t first_index;
    std::int32_t vertex_offset;
};

static_assert(sizeof(DrawGroup) == 24);

// A draw list's groups, a run of the group table. Each group has one command slot, so it's the list's run of commands too.
struct DrawListRange {
    std::uint32_t first_group = 0;
    std::uint32_t group_count = 0;
};

static_assert(sizeof(DrawListRange) == 8);

// Where all of one cull phase's buffers are, in one table its steps read. The two phases share the order, groups and lists, and have the rest each.
struct CullTables {
    vk::DeviceAddress order;          // draw indices, by list, then primitive
    vk::DeviceAddress groups;         // one DrawGroup per group
    vk::DeviceAddress lists;          // one DrawListRange per list
    vk::DeviceAddress visible;        // per draw in the order: 1 if this phase draws it
    vk::DeviceAddress draw_slots;     // prefix sums of visible, then their total
    vk::DeviceAddress group_flags;    // per group: 1 if any of its draws is visible
    vk::DeviceAddress group_slots;    // prefix sums of group_flags, then their total
    vk::DeviceAddress instances;      // the visible draws' indices
    vk::DeviceAddress commands;       // one VkDrawIndexedIndirectCommand per group
    vk::DeviceAddress counts;         // per list, how many commands to draw
    vk::DeviceAddress early_visible;  // the early phase's `visible`: what the late phase needn't test again
    std::uint32_t draw_count;
    std::uint32_t group_count;
};

static_assert(sizeof(CullTables) == 96);
static_assert(offsetof(CullTables, early_visible) == 80);

// The cull steps' push data (cull.slang). The cull phases read the frame (the draws, the view and the depth pyramid) and their tables; the depth pyramid steps read which levels to copy or reduce, and their sizes. Push data follows std430 rules, where a uvec2 starts on an 8-byte boundary: the two pointers and two slots fill the first 24 bytes, so source_size lands on one.
struct CullPushData {
    vk::DeviceAddress frame = 0;   // this frame's FrameData
    vk::DeviceAddress tables = 0;  // the phase's CullTables
    std::uint32_t source = 0;      // resource heap slot: the depth buffer (sampled), or the level above (storage)
    std::uint32_t target = 0;      // resource heap slot: the pyramid level written (storage)
    glm::uvec2 source_size{0};     // in texels
    glm::uvec2 target_size{0};
};

static_assert(sizeof(CullPushData) == 40);
static_assert(offsetof(CullPushData, source_size) == 24);

// The tone-mapping pass's push data: which resource heap slot holds the HDR image, and the view, so material views can skip tone mapping.
struct TonemapPushData {
    std::uint32_t hdr_image;
    View view;
};

// The transparency composite's push data (composite.slang): the resource heap slots of the transparency pass's two sums.
struct CompositePushData {
    std::uint32_t accum;
    std::uint32_t reveal;
};

// The environment compute shaders' push data. Each dispatch sets only what its shader reads, so every member has a default.
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

// The atmosphere's compute shaders' push data (atmosphere.slang). Each step reads what it needs: the per-frame ones the camera's height and the sun, the aerial perspective the frame's view. Push data follows std430 rules, where a vec3 starts on a 16-byte boundary: the two pointers fill the first 16 bytes, so sun_direction lands on one.
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

// Light clusters (light_clusters.h, lights.slang)

// Where the light cluster steps' buffers are, and which lights are local.
struct LightTables {
    vk::DeviceAddress flags;      // per local light: 1 if in view
    vk::DeviceAddress slots;      // prefix sums of flags, then their total
    vk::DeviceAddress spheres;    // per visible light: its centre in view space, and its range
    std::uint32_t first_local;    // the first point or spot light: after the directional ones
    std::uint32_t local_count;    // how many point and spot lights there are
};

static_assert(sizeof(LightTables) == 32);

// Every light cluster step's push data: the frame, the tables, and the screen's size, which sets the tiles.
struct LightPushData {
    vk::DeviceAddress frame;
    vk::DeviceAddress tables;
    glm::uvec2 extent;
};

static_assert(sizeof(LightPushData) == 24);

// The mip chain compute shaders' push data (mips.slang). Every level of every texture is in one buffer, reached through addresses.
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

// The ambient occlusion compute shaders' push data (ao.slang), the same for all four steps. The half-resolution images and `source` and `target` are storage images; each step reads and writes the ones it needs.
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

`game-engine/shaders/shared.slangh`:
```slang
// The structs every scene shader shares with C++ (src/includes/shader_types.h), and a few helpers, included by mesh.slang, background.slang, ao.slang, cull.slang, atmosphere.slang and lights.slang. Each of those declares its own push data block. A .slangh file isn't compiled on its own: CMakeLists.txt only compiles .slang files.

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
    float4x4 model;          // this primitive's space -> its cell, from the cell's corner
    float4x4 normal_matrix;  // transposed inverse of model
    uint material;           // index into the materials
    uint first_index;        // where the primitive's indices start
    int vertex_offset;       // added to each index
    int3 cell;               // the world cell the draw is placed in
    float3 bounds_min;       // the draw's box, in its cell
    float3 bounds_max;
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
    uint alpha_mode;                 // 0 opaque, 1 mask, 2 blend
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
    float3 offset;     // where in its cell the light is
    float range;       // point and spot: where the light fades to nothing
    float3 direction;  // the way the light shines
    float spot_scale;
    float3 intensity;  // lux (directional) or candela (point, spot), per channel
    float spot_offset;
    int3 cell;         // the world cell the light is in
    uint type;
};

// View: what the fragment shader outputs (keys 1-9, then 0, then L).
static const uint view_lit = 0;
static const uint view_base_color = 1;
static const uint view_normal = 2;
static const uint view_vertex_normal = 3;
static const uint view_metallic = 4;
static const uint view_roughness = 5;
static const uint view_occlusion = 6;
static const uint view_emissive = 7;
static const uint view_ambient_occlusion = 8;
static const uint view_shadow = 9;
static const uint view_light_count = 10;

// What the environment's compute shaders found out about the sky.
struct EnvironmentInfo {
    float3 irradiance_sh[9];  // diffuse light, as spherical harmonics
    float3 sun_illuminance;   // lux at the ground; 0 for a photographed sky
};

// The same for every draw in a frame. Natural layout, like the C++ struct: the pointers land on 8-byte boundaries, after the matrices and at the end.
struct FrameData {
    float4x4 view_projection;          // camera-relative space -> clip space
    float4x4 inverse_view_projection;  // clip space -> camera-relative space
    Vertex *vertices;                  // the scene's vertices
    uint *indices;                     // the scene's indices
    DrawData *draws;                   // one DrawData per draw
    Material *materials;               // the scene's materials
    Light *lights;                     // the scene's lights: directional ones first
    EnvironmentInfo *environment;      // the sky's diffuse light and the sun
    uint64_t scene_tlas;               // the top-level acceleration structure's address
    int3 camera_cell;                  // the world cell the camera is in
    float exposure;                    // scene nits -> tone mapper input
    float3 sun_direction;              // toward the sun
    uint directional_light_count;      // the first lights, which reach everywhere
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
    float3 camera_offset;              // where in its cell the camera is
    float3 tlas_offset;                // the camera, from the TLAS's origin
    uint sky_view;                     // resource heap slot: the sky-view table
    uint aerial_inscatter;             // resource heap slot: the air's light, over the view
    uint aerial_transmittance;         // resource heap slot: the air's transmittance, likewise
    uint atmosphere;                   // 1: the simulated sky and its haze; 0: the photograph
    float3 camera_forward;             // the way the camera looks: view depth is distance along it
    uint shadow_ray_budget;            // 0: every light traces its shadow ray; N: only the N strongest
    uint2 cluster_tiles;               // the light clusters' tiles across and down
    uint depth_pyramid;                // resource heap slot of the depth pyramid's level 0; level i is at depth_pyramid + i
    uint depth_pyramid_levels;         // how many levels it has
    uint2 screen;                      // the depth buffer's size in pixels: the pyramid's level 0
    uint *light_clusters;              // per cluster, one bit per visible light
    uint *visible_lights;              // how many local lights are in view, then their indices
};

// Light clusters (src/includes/light_clusters.h)

// The view, cut into clusters: tiles of 64 x 64 pixels across the screen, and 24 slices in depth, two per doubling of the distance from the near plane. Each slice is about 41% of its distance deep, so clusters keep the same shape at every distance; with these tiles, about six times deeper than wide. The last slice reaches on to infinity. Each cluster holds one bit per light in view: 4,096 lights, 128 words.
static const uint cluster_tile_size = 64;
static const uint cluster_slices = 24;
static const float cluster_near = 0.05;  // the camera's near plane, where slice 0 starts
static const uint max_visible_lights = 4096;
static const uint cluster_words = max_visible_lights / 32;

// The slice a view depth falls in: slice s starts at near x 2^(s / 2).
uint cluster_slice(float view_depth) {
    const float slice = floor(2.0 * log2(max(view_depth, cluster_near) / cluster_near));
    return min(uint(slice), cluster_slices - 1);
}

// The first word of the bits of the cluster holding `pixel` at `view_depth`.
uint *cluster_bits(FrameData *frame, float2 pixel, float view_depth) {
    const uint2 tile = min(uint2(pixel) / cluster_tile_size, frame.cluster_tiles - 1);
    const uint cluster = (cluster_slice(view_depth) * frame.cluster_tiles.y + tile.y) * frame.cluster_tiles.x + tile.x;
    return frame.light_clusters + cluster * cluster_words;
}

// Camera-relative positions

// The side of a world cell, in metres: cells.h's cell_size.
static const float cell_size = 64.0;

// A position given as a world cell and an offset in it (cells.h), relative to the camera. The cells are subtracted as integers, exactly; only their difference, and the offsets' difference, become floats. Near the camera, both are small, and keep a float's full precision anywhere in the world.
float3 camera_relative(FrameData *frame, int3 cell, float3 offset) {
    return float3(cell - frame.camera_cell) * cell_size + (offset - frame.camera_offset);
}

// Written with vkCmdPushDataEXT before each pipeline's draws.
struct PushData {
    FrameData *frame;  // this frame's data
    uint *instances;   // the cull phase's visible draws: what each instance draws
};

// Normals in two numbers

// Octahedral encoding (Meyer et al. 2010): a unit vector is projected onto the octahedron |x| + |y| + |z| = 1, whose lower half is folded up over the upper; flattened, that's a square, so two numbers in -1..1 hold any direction, evenly enough for 16-bit floats.
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

## 17.3 The box test and the pyramid's steps: `cull.slang`

### Why
One function decides whether a box is hidden; two entry points run it, one per phase; two more steps build the pyramid it reads.

### How
- **`occluded`:** whether the box is hidden behind what the pyramid holds.
  1. **The corners.** Each of the box's eight corners is multiplied by the view-projection matrix. If any has `w ≤ z` in clip space, it's at or in front of the near plane: our projection (Chapter 13) sets `z` to the near distance and `w` to the point's distance, so that's where depth would reach 1. Such a box reaches the camera, and part of it may be nearer than anything drawn, so it isn't tested: it stays visible. Otherwise the corners' screen positions, `xy / w` mapped to pixels, span the box's **rectangle**, and the greatest of their depths, `z / w`, is the box's **nearest depth**: with reverse-Z, nearer is greater. Depth is monotonic along any straight line, so the box's nearest point is at a corner.
  2. **The pixels.** The rectangle is widened to whole pixels and clamped to the screen: what's off screen can't be seen, and a box partly off screen is tested by its on-screen part.
  3. **The level.** The longer side of the rectangle, in pixels, picks the pyramid level whose texels are at least that large: `ceil(log2(side))`, which `firstbithigh` of `side − 1` gives, plus one. At that level the rectangle spans at most two texels each way, whatever its alignment.
  4. **The reads.** The four texels around the rectangle at that level, their coordinates clamped to the level's edge. Each holds the farthest depth of all the pixels under it, so between them they cover every pixel the rectangle touches. The least of the four is the farthest depth under the rectangle.
  5. **The verdict.** Hidden if the box's nearest depth is less than that. Everything on those pixels is nearer than any point of the box; equal counts as visible.
- **Why it's safe.** Every rounding keeps the box: the rectangle grows, never shrinks; the pyramid's texels grow to cover odd edges (below); clamped coordinates land on texels that cover more, never less. A box the test keeps may still be hidden, and costs what it did before. A box it drops is hidden, always.
- **`drawn`:** in view, by Chapter 12's test, and not occluded: 1 or 0.
- **`cullEarlyMain`** writes that for every draw. The pyramid it reads was built last frame, and is tested with this frame's view: wherever the view moved, the depths are a little off. That's the guess.
- **`cullLateMain`** writes it only for draws the early phase left out, and 0 for the rest: those are being drawn already. The pyramid it reads was built this frame, from the early draws, so a draw it drops is behind something this frame draws.
- **The other steps** are unchanged, and shared: each phase runs them on its own tables.
- **`copyDepthMain`:** level 0, the depth buffer copied texel for texel.
- **`reduceDepthMain`:** each level below, from the one above: texel `(x, y)` takes the least depth of the four texels `(2x, 2y)` to `(2x + 1, 2y + 1)`. When the level above has an odd width, its last column would be under no texel: the last texel of this level takes it too, three columns wide, and likewise an odd height. 1920 × 1080's fourth level is 240 × 135; the fifth, 120 × 67, whose last row covers three rows of the fourth. So every texel of every level covers all the pixels under it, and a box hidden by a level is hidden by the pixels. A level that halves to an odd size one way and to one texel the other reads no further than the edge.

### Code
`game-engine/shaders/cull.slang`:
```slang
// GPU culling: which draws are in view and not hidden behind others, turned into indirect draw commands. Two phases (culling.h, record_culling) of six steps each:
//   cullEarlyMain or cullLateMain  per draw: 1 if the phase draws it, else 0
//   scanDrawsMain       prefix sums of those: each visible draw's slot among the instances, and each group's first instance
//   markGroupsMain      per group: 1 if any of its draws is visible
//   scanGroupsMain      prefix sums of those: each group's command slot
//   writeInstancesMain  per visible draw: its index, into its slot
//   writeCommandsMain   per visible group: its instanced command; per list: how many commands it has
// and, between the phases, two steps that build the depth pyramid (record_depth_pyramid):
//   copyDepthMain       level 0: the depth buffer, copied
//   reduceDepthMain     each level below: the farthest depth of the texels above it
// Draws are visited in the cull's order (by list, then primitive), so a group's draws, and a list's groups, are runs. Every write goes to a place the prefix sums fix: the result doesn't depend on which thread runs first.

#include "shared.slangh"
#include "scan.slangh"

// Data shared with C++ (src/includes/shader_types.h)

// Vulkan's VkDrawIndexedIndirectCommand: what vkCmdDrawIndexed's arguments would be.
struct DrawCommand {
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance;
};

struct DrawGroup {
    uint first;          // where its draws start in the order
    uint count;          // how many draws
    uint list;           // its draw list
    uint index_count;    // the primitive's drawIndexed arguments
    uint first_index;
    int vertex_offset;
};

struct DrawListRange {
    uint first_group;    // where its groups, and its commands, start
    uint group_count;
};

struct CullTables {
    uint *order;            // draw indices, in the cull's order
    DrawGroup *groups;
    DrawListRange *lists;
    uint *visible;          // per position in the order: 1 if this phase draws it
    uint *draw_slots;       // prefix sums of visible, then their total
    uint *group_flags;      // per group
    uint *group_slots;      // prefix sums of group_flags, then their total
    uint *instances;        // the visible draws' indices
    DrawCommand *commands;  // per group
    uint *counts;           // per list
    uint *early_visible;    // the early phase's visible: what the late phase needn't test again
    uint draw_count;
    uint group_count;
};

struct CullPushData {
    FrameData *frame;
    CullTables *tables;
    uint source;        // the depth pyramid steps: the resource heap slot read, the depth buffer or the level above
    uint target;        // the level written
    uint2 source_size;  // in texels
    uint2 target_size;
};

[[vk::push_constant]]
ConstantBuffer<CullPushData> push;

// 1. The view

// Whether any of the box from `lo` to `hi` can be inside the view.
//
// A point p is on screen when its clip-space position c = M p, with M the view-projection matrix, has -w <= x <= w, -w <= y <= w and 0 <= z <= w. Each of those six inequalities is a plane in camera-relative space, read off M's rows (Gribb and Hartmann 2001): w - x >= 0 is (row 3 - row 0) . (p, 1) >= 0. With reverse-Z, z >= 0 is the far plane and z <= w the near one. Slang's matrix[i] is row i, whatever the layout in memory.
//
// A box is outside if it lies wholly behind one of the planes: if even its corner furthest along the plane's normal is behind it. That corner is the box's centre plus its half-size, each axis signed like the normal.
//
// This keeps every box that's really visible. It also keeps a few that aren't, near the frustum's corners, where a box can be in front of every plane but still outside: one wasted instance each, whose triangles the clipper drops.
bool in_view(float4x4 view_projection, float3 lo, float3 hi) {
    const float4 row0 = view_projection[0];
    const float4 row1 = view_projection[1];
    const float4 row2 = view_projection[2];
    const float4 row3 = view_projection[3];

    const float4 planes[6] = {
        row3 + row0,  // left:   x >= -w
        row3 - row0,  // right:  x <= w
        row3 + row1,  // top or bottom: y >= -w
        row3 - row1,  // the other:     y <= w
        row2,         // far, with reverse-Z: z >= 0; with no far plane, always true
        row3 - row2,  // near, with reverse-Z: z <= w
    };

    const float3 centre = (lo + hi) * 0.5;
    const float3 half_size = (hi - lo) * 0.5;

    for (int i = 0; i < 6; ++i) {
        const float4 plane = planes[i];
        const float reach = dot(abs(plane.xyz), half_size);

        if (dot(plane.xyz, centre) + plane.w + reach < 0.0) {
            return false;
        }
    }

    return true;
}

// 2. The depth pyramid

// Whether the box from `lo` to `hi` is hidden behind what the depth pyramid holds: whether its nearest point is farther than the farthest surface on every pixel it covers (Greene, Kass and Miller 1993).
//
// The box is projected corner by corner. If any corner is at or in front of the near plane, where w <= z in clip space with our projection, the box can't be tested: it reaches the camera, and part of it may be nearer than anything drawn. Otherwise its screen rectangle is the corners' extent in pixels, and its nearest depth the greatest of their depths, z / w, which reverse-Z makes the nearest. Depth varies monotonically along any straight line, so the box's nearest point is at a corner.
//
// The rectangle is tested at the pyramid level where it spans at most two texels each way: the level whose texels are at least as large as it. Each texel there holds the farthest depth of all the pixels under it (reduceDepthMain), so the four texels around the rectangle cover every pixel it touches. The box is hidden if its nearest depth is less than the least of the four: everything on those pixels is nearer than any of the box.
//
// Everything rounds the safe way, keeping the box: the rectangle grows to whole pixels, equal depths count as visible, and texel coordinates clamp to the level's edge, where the last texel also holds the leftover pixels of an odd-sized level above.
bool occluded(FrameData *frame, float3 lo, float3 hi) {
    float2 rect_min = float2(1.0e30);  // in pixels
    float2 rect_max = float2(-1.0e30);
    float nearest = 0.0;

    for (int i = 0; i < 8; ++i) {
        const float3 corner = float3((i & 1) != 0 ? hi.x : lo.x, (i & 2) != 0 ? hi.y : lo.y, (i & 4) != 0 ? hi.z : lo.z);
        const float4 clip = mul(frame.view_projection, float4(corner, 1.0));

        if (clip.w <= clip.z) {
            return false;
        }

        const float2 pixel = (clip.xy / clip.w * 0.5 + 0.5) * float2(frame.screen);
        rect_min = min(rect_min, pixel);
        rect_max = max(rect_max, pixel);
        nearest = max(nearest, clip.z / clip.w);
    }

    // The pixels the rectangle touches, within the screen: what's off screen can't be seen anyway.
    const int2 last_pixel = int2(frame.screen) - 1;
    const uint2 first = uint2(clamp(int2(floor(rect_min)), int2(0), last_pixel));
    const uint2 last = uint2(clamp(int2(floor(rect_max)), int2(0), last_pixel));

    // The level whose texels are at least as large as the rectangle's longer side: ceil(log2(side)), from the highest bit of side - 1. At that level the rectangle spans at most two texels each way.
    const uint side = max(last.x - first.x, last.y - first.y) + 1;
    const uint level = min(side <= 1 ? 0 : firstbithigh(side - 1) + 1, frame.depth_pyramid_levels - 1);
    const uint2 last_texel = max(frame.screen >> level, uint2(1)) - 1;
    const uint2 first_texel = min(first >> level, last_texel);
    const uint2 end_texel = min(last >> level, last_texel);

    RWTexture2D<float> pyramid = RWTexture2D<float>.Handle(uint2(frame.depth_pyramid + level, 0));
    const float farthest = min(
        min(pyramid[first_texel], pyramid[uint2(end_texel.x, first_texel.y)]),
        min(pyramid[uint2(first_texel.x, end_texel.y)], pyramid[end_texel]));

    return nearest < farthest;
}

// 3. Which draws each phase draws

// 1 if the draw at `position` in the order is in view and not hidden by the pyramid, else 0.
uint drawn(FrameData *frame, CullTables *tables, uint position) {
    // The draw's box, from its cell to the camera: both corners in the same cell.
    const DrawData draw = frame.draws[tables.order[position]];
    const float3 lo = camera_relative(frame, draw.cell, draw.bounds_min);
    const float3 hi = camera_relative(frame, draw.cell, draw.bounds_max);

    return in_view(frame.view_projection, lo, hi) && !occluded(frame, lo, hi) ? 1 : 0;
}

// The early phase tests every draw against the pyramid the previous frame built, under this frame's view: a guess at what's hidden, right wherever the view hasn't changed.
[shader("compute")]
[numthreads(64, 1, 1)]
void cullEarlyMain(uint3 id : SV_DispatchThreadID) {
    CullTables *tables = push.tables;
    const uint position = id.x;

    if (position >= tables.draw_count) {
        return;
    }

    tables.visible[position] = drawn(push.frame, tables, position);
}

// The late phase tests only what the early phase left out, against the pyramid built from the early draws: whatever the guess hid wrongly. A draw neither phase keeps is outside the view, or behind what the early phase drew.
[shader("compute")]
[numthreads(64, 1, 1)]
void cullLateMain(uint3 id : SV_DispatchThreadID) {
    CullTables *tables = push.tables;
    const uint position = id.x;

    if (position >= tables.draw_count) {
        return;
    }

    tables.visible[position] = tables.early_visible[position] == 0 ? drawn(push.frame, tables, position) : 0;
}

// 4 and 6. Prefix sums (scan.slangh)

[shader("compute")]
[numthreads(scan_size, 1, 1)]
void scanDrawsMain(uint3 id : SV_GroupThreadID) {
    CullTables *tables = push.tables;
    exclusive_scan(tables.visible, tables.draw_slots, tables.draw_count, id.x);
}

[shader("compute")]
[numthreads(scan_size, 1, 1)]
void scanGroupsMain(uint3 id : SV_GroupThreadID) {
    CullTables *tables = push.tables;
    exclusive_scan(tables.group_flags, tables.group_slots, tables.group_count, id.x);
}

// 5. Groups with something to draw

// A group's draws are a run of the order, so its visible draws are a run of the instances, from draw_slots[first] to draw_slots[first + count].
uint visible_in_group(CullTables *tables, DrawGroup group) {
    return tables.draw_slots[group.first + group.count] - tables.draw_slots[group.first];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void markGroupsMain(uint3 id : SV_DispatchThreadID) {
    CullTables *tables = push.tables;

    if (id.x >= tables.group_count) {
        return;
    }

    tables.group_flags[id.x] = visible_in_group(tables, tables.groups[id.x]) > 0 ? 1 : 0;
}

// 7 and 8. Writing the instances and the commands

[shader("compute")]
[numthreads(64, 1, 1)]
void writeInstancesMain(uint3 id : SV_DispatchThreadID) {
    CullTables *tables = push.tables;
    const uint position = id.x;

    if (position >= tables.draw_count || tables.visible[position] == 0) {
        return;
    }

    tables.instances[tables.draw_slots[position]] = tables.order[position];
}

// A visible group's command goes in its list's run, after the list's visible groups before it: group_slots counts those, from the list's first group. The command draws the group's visible draws as instances: its firstInstance is where they start among the instances, and the vertex shader looks each draw's index up there.
[shader("compute")]
[numthreads(64, 1, 1)]
void writeCommandsMain(uint3 id : SV_DispatchThreadID) {
    CullTables *tables = push.tables;
    const uint group_index = id.x;

    if (group_index >= tables.group_count) {
        return;
    }

    const DrawGroup group = tables.groups[group_index];
    const DrawListRange list = tables.lists[group.list];
    const uint list_start = tables.group_slots[list.first_group];

    // The list's first group also writes the list's count: its visible groups.
    if (group_index == list.first_group) {
        tables.counts[group.list] = tables.group_slots[list.first_group + list.group_count] - list_start;
    }

    const uint instance_count = visible_in_group(tables, group);

    if (instance_count == 0) {
        return;
    }

    const uint command = list.first_group + tables.group_slots[group_index] - list_start;
    tables.commands[command] = DrawCommand(
        group.index_count, instance_count, group.first_index, group.vertex_offset, tables.draw_slots[group.first]);
}

// 9. The depth pyramid's levels

// Level 0: the depth buffer, copied. No depth format need support storage images, and none does in practice, so the steps that reduce the pyramid, and the cull that reads it, get a copy that does.
[shader("compute")]
[numthreads(8, 8, 1)]
void copyDepthMain(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= push.target_size)) {
        return;
    }

    const Texture2D depth_buffer = Texture2D.Handle(uint2(push.source, 0));
    RWTexture2D<float> level = RWTexture2D<float>.Handle(uint2(push.target, 0));

    level[id.xy] = depth_buffer.Load(int3(id.xy, 0)).r;
}

// Each level below: texel (x, y) holds the least depth, the farthest with reverse-Z, of texels (2x, 2y) to (2x + 1, 2y + 1) of the level above. When the level above has an odd width, its last column would be under no texel: this level's last texel takes it too, three columns wide; likewise an odd height. So every texel of every level covers all the pixels under it, and a box hidden by a level is hidden by the pixels.
[shader("compute")]
[numthreads(8, 8, 1)]
void reduceDepthMain(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= push.target_size)) {
        return;
    }

    RWTexture2D<float> above = RWTexture2D<float>.Handle(uint2(push.source, 0));
    RWTexture2D<float> level = RWTexture2D<float>.Handle(uint2(push.target, 0));

    // Two texels each way, or three at the last texel when the level above is odd-sized that way; never past its edge.
    const uint2 first = id.xy * 2;
    const uint2 odd = push.source_size & 1;
    const uint2 last = push.target_size - 1;
    const uint2 count = min(uint2(id.x == last.x ? 2 + odd.x : 2, id.y == last.y ? 2 + odd.y : 2), push.source_size - first);

    float farthest = 1.0;

    for (uint y = 0; y < count.y; ++y) {
        for (uint x = 0; x < count.x; ++x) {
            farthest = min(farthest, above[first + uint2(x, y)]);
        }
    }

    level[id.xy] = farthest;
}
```

## 17.4 Two phases: `culling.h`, `culling.cpp`

### Why
Each phase needs the buffers the six steps rewrite, and the frame needs the pyramid built between them.

### How
- **`CullPhase`** names the two, and `cull_phases` lists them, for loops over both.
- **`CullPhaseBuffers`:** what Chapter 12's `DrawCulling` held per frame, now once per phase, plus the phase's `CullTables`. `DrawCulling` keeps the shared order, groups and lists, and gains the two cull pipelines, one per entry point, and the pyramid's two.
- **`create_draw_culling`** makes both phases' buffers, and each phase's table. The late phase's names the early phase's flags; the early phase's table names its own, which it never reads.
- **`CullTotals`** has a count of draws and of commands per phase, each phase copying its own pair out.
- **`clear_depth_pyramid`:** every level to 0, the far plane, through a one-off submission, so that a cull reading the pyramid before any frame has built it hides nothing. The image goes to `eGeneral`, where storage images are written and read, and stays there for good: every later barrier on it is a memory barrier, with no layout to change.
- **`record_culling`** records one phase: the wait for the previous frame's readers of the phase's buffers, the six steps with that phase's cull entry point and table, and the barrier that lets the draws read the results. That barrier now also covers compute shaders: the late cull reads the early phase's flags. The totals go to the phase's half of `CullTotals`.
- **`record_depth_pyramid`:** first a barrier, since this frame's early cull, and the previous frame's late cull before it, read the levels about to be rewritten. Then the copy, and one reduction per level below, each waiting for the one above. The last barrier makes every level visible to the late cull, and to the next frame's early cull after it. At 1080p: one copy and ten reductions.
- **`draw_list`** draws a list from one phase's commands and count.

### Code
`game-engine/src/includes/culling.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/image.h"
#include "includes/shader_types.h"
#include "includes/vulkan_setup.h"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

// Draw lists

// One indirect draw call draws a whole list of commands with one pipeline and one dynamic state, so draws are sorted into lists by what those fix:
//   - the alpha mode, which picks the pipeline,
//   - whether the material is double-sided, which sets the cull mode,
//   - whether the transform mirrors, which sets the front face.
// Three alpha modes times two times two: twelve lists, numbered alpha mode x 4 + double-sided x 2 + mirrored.
constexpr std::uint32_t draw_list_count = 12;

constexpr std::uint32_t draw_list_index(AlphaMode alpha_mode, bool double_sided, bool mirrored) {
    return static_cast<std::uint32_t>(alpha_mode) * 4 + (double_sided ? 2 : 0) + (mirrored ? 1 : 0);
}

// What the cull needs to know about a draw: its list, which primitive it draws (draws of one primitive in one list can share an instanced command), and that primitive's drawIndexed arguments.
struct CullDraw {
    std::uint32_t list;
    std::uint32_t primitive;
    std::uint32_t index_count;
    std::uint32_t first_index;
    std::int32_t vertex_offset;
};

// GPU culling

// The cull's compute pipelines and buffers, for a scene whose draws never change: the groups and lists are worked out once, here.
//   - Draws are put in an order: by list, then by primitive, then by draw index. A run of draws with the same list and primitive is a group, drawn as one instanced command; a list's groups are a run too.
//   - A draw is kept when its box is in view and not hidden: not wholly behind what the depth pyramid holds. The pyramid is the depth buffer with a mip chain where each texel holds the farthest depth of the texels above it, so a box can be tested against the depth under its whole screen rectangle in four reads.
//   - Every frame, the cull runs in two phases around the depth prepass, each of six compute steps (cull.slang). The early phase tests every draw against the pyramid the previous frame built, under this frame's view: a guess, right wherever the view hasn't changed. The prepass draws what it keeps, the pyramid is built from that depth, and the late phase tests what the early phase left out against it. Whatever the guess hid wrongly is drawn late; nothing visible is missed, and nothing is drawn twice. Each phase writes its own commands, instances and counts, and every pass draws both phases' lists.
// Everything a step writes goes to a place fixed by prefix sums over the previous steps' results, never by which thread got there first: the same view gives the same commands, in the same order, every frame.
enum class CullPhase : std::size_t {
    early,
    late,
};

constexpr std::array cull_phases{CullPhase::early, CullPhase::late};

// What one phase rewrites every frame, in this order, and the table that names all of it.
struct CullPhaseBuffers {
    Buffer visible;      // per draw in the order: 1 if this phase draws it, else 0
    Buffer draw_slots;   // prefix sums of `visible`, then their total
    Buffer group_flags;  // per group: 1 if any of its draws is visible
    Buffer group_slots;  // prefix sums of `group_flags`, then their total
    Buffer instances;    // the visible draws' indices, group after group
    Buffer commands;     // one VkDrawIndexedIndirectCommand per group: each list's run
    Buffer counts;       // per list, how many of its commands to draw
    Buffer tables;       // one CullTables: where all of these are, and the shared tables
};

struct DrawCulling {
    vk::raii::Pipeline cull_early = nullptr;
    vk::raii::Pipeline cull_late = nullptr;
    vk::raii::Pipeline scan_draws = nullptr;
    vk::raii::Pipeline mark_groups = nullptr;
    vk::raii::Pipeline scan_groups = nullptr;
    vk::raii::Pipeline write_instances = nullptr;
    vk::raii::Pipeline write_commands = nullptr;
    vk::raii::Pipeline copy_depth = nullptr;
    vk::raii::Pipeline reduce_depth = nullptr;

    // Written once, shared by the phases.
    Buffer order;   // draw indices, in the cull's order
    Buffer groups;  // one DrawGroup per group, in order
    Buffer lists;   // one DrawListRange per list

    std::array<CullPhaseBuffers, cull_phases.size()> phases;

    std::array<DrawListRange, draw_list_count> list_ranges{};  // the lists' runs, for the draw calls
    std::uint32_t draw_count = 0;
    std::uint32_t group_count = 0;
};

// `draws` has one CullDraw per draw, in draw order.
DrawCulling create_draw_culling(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::span<const CullDraw> draws
);

// The totals, as the cull copies them out: how many draws each phase kept, and in how many commands.
struct CullTotals {
    std::uint32_t early_draws;
    std::uint32_t late_draws;
    std::uint32_t early_commands;
    std::uint32_t late_commands;
};

// Clears every level of `pyramid` to 0, the far plane, so a cull that reads it before any frame has built it hides nothing. Once, after the swapchain that owns it is made or remade; waits for the GPU.
void clear_depth_pyramid(
    const vk::raii::Device &device,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    const Image &pyramid
);

// Records one phase of the cull for the frame whose FrameData is at `frame`: the late phase must come after the early one, and after record_depth_pyramid. Afterwards, indirect draws may read the phase's commands and counts, and vertex shaders its instances. Its totals are also copied into their half of `readback`, a host-visible buffer holding one CullTotals, for the CPU to read once the frame is done.
void record_culling(
    const vk::raii::CommandBuffer &commands,
    const DrawCulling &culling,
    CullPhase phase,
    vk::DeviceAddress frame,
    vk::Buffer readback
);

// Records the depth pyramid's build: level 0 copied from the depth buffer, which must be in eDepthReadOnlyOptimal, then each level below reduced from the one above. `depth_slot` is the depth buffer's resource heap slot, `pyramid_slot` that of the pyramid's level 0, with the other levels' slots after it. Afterwards compute shaders may read every level.
void record_depth_pyramid(
    const vk::raii::CommandBuffer &commands,
    const DrawCulling &culling,
    const Image &pyramid,
    std::uint32_t depth_slot,
    std::uint32_t pyramid_slot
);

// Draws list `list`'s commands from `phase` for this frame, in one indirect call. The pipeline, its push data with the phase's instances, and the list's dynamic state must already be set.
void draw_list(const vk::raii::CommandBuffer &commands, const DrawCulling &culling, CullPhase phase, std::uint32_t list);
```

`game-engine/src/culling.cpp`:
```cpp
#include "includes/culling.h"

#include "includes/pipeline.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace {

    // Threads per workgroup for the steps that take one draw or one group each.
    constexpr std::uint32_t workgroup_size = 64;

    constexpr vk::DeviceSize command_size = sizeof(vk::DrawIndexedIndirectCommand);

    std::uint32_t workgroups(std::uint32_t threads) {
        return (threads + workgroup_size - 1) / workgroup_size;
    }

    // A device-local buffer of `count` 32-bit numbers, written and read only by the GPU, at least one long: a buffer can't be empty.
    Buffer gpu_numbers(const vk::raii::Device &device, const GpuChoice &gpu, std::size_t count, vk::BufferUsageFlags usage = {}) {
        return create_buffer(device, gpu, std::max<std::size_t>(count, 1) * sizeof(std::uint32_t),
            vk::BufferUsageFlagBits::eShaderDeviceAddress | usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
    }

    // Makes the `dst` work wait for the `src` work, and the `src` writes visible to the `dst` accesses: for buffers, which need no layouts, and for the pyramid, which stays in one.
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

    // Between two compute steps: the first's writes, visible to the second.
    void compute_to_compute(const vk::raii::CommandBuffer &commands) {
        memory_barrier(commands,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);
    }

    // Binds a compute pipeline and pushes its data; the dispatch follows.
    void bind(const vk::raii::CommandBuffer &commands, const vk::raii::Pipeline &pipeline, const CullPushData &push) {
        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        commands.pushDataEXT(vk::PushDataInfoEXT{
            .offset = 0,
            .data = {.address = &push, .size = sizeof(push)},
        });
    }

    // The size of pyramid level `level`: each level halves the one above, rounding down, never below one texel.
    vk::Extent2D level_size(vk::Extent2D level0, std::uint32_t level) {
        return {std::max(level0.width >> level, 1u), std::max(level0.height >> level, 1u)};
    }

}  // namespace

// Creating

DrawCulling create_draw_culling(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::span<const CullDraw> draws
) {
    DrawCulling culling;
    culling.cull_early = create_compute_pipeline(device, "cull", "cullEarlyMain");
    culling.cull_late = create_compute_pipeline(device, "cull", "cullLateMain");
    culling.scan_draws = create_compute_pipeline(device, "cull", "scanDrawsMain");
    culling.mark_groups = create_compute_pipeline(device, "cull", "markGroupsMain");
    culling.scan_groups = create_compute_pipeline(device, "cull", "scanGroupsMain");
    culling.write_instances = create_compute_pipeline(device, "cull", "writeInstancesMain");
    culling.write_commands = create_compute_pipeline(device, "cull", "writeCommandsMain");
    culling.copy_depth = create_compute_pipeline(device, "cull", "copyDepthMain");
    culling.reduce_depth = create_compute_pipeline(device, "cull", "reduceDepthMain");
    culling.draw_count = static_cast<std::uint32_t>(draws.size());

    // The order: by list, then by primitive. stable_sort keeps draws that tie in draw order, so the order, and every frame's commands, are fixed.
    std::vector<std::uint32_t> order(draws.size());
    std::iota(order.begin(), order.end(), 0u);
    std::ranges::stable_sort(order, [&](std::uint32_t a, std::uint32_t b) {
        return draws[a].list != draws[b].list ? draws[a].list < draws[b].list : draws[a].primitive < draws[b].primitive;
    });

    // Groups: runs of the order with one list and one primitive. Lists: runs of groups, which start where the previous list's end.
    std::vector<DrawGroup> groups;

    for (std::uint32_t position = 0; position < order.size(); ++position) {
        const CullDraw &draw = draws[order[position]];

        if (groups.empty() || groups.back().list != draw.list
            || draws[order[groups.back().first]].primitive != draw.primitive) {
            groups.push_back(DrawGroup{
                .first = position,
                .count = 0,
                .list = draw.list,
                .index_count = draw.index_count,
                .first_index = draw.first_index,
                .vertex_offset = draw.vertex_offset,
            });
            ++culling.list_ranges[draw.list].group_count;
        }

        ++groups.back().count;
    }

    culling.group_count = static_cast<std::uint32_t>(groups.size());

    std::uint32_t next_group = 0;
    for (DrawListRange &range : culling.list_ranges) {
        range.first_group = next_group;
        next_group += range.group_count;
    }

    // The tables, uploaded once; a buffer can't be empty.
    const auto upload = [&](std::span<const std::byte> bytes) {
        const std::vector<std::byte> one(sizeof(std::uint32_t));
        return upload_buffer(device, gpu, queue, pool, bytes.empty() ? std::span(one) : bytes,
            vk::BufferUsageFlagBits::eShaderDeviceAddress);
    };
    culling.order = upload(std::as_bytes(std::span(order)));
    culling.groups = upload(std::as_bytes(std::span(groups)));
    culling.lists = upload(std::as_bytes(std::span(culling.list_ranges)));

    // What each phase's steps rewrite every frame. The draws read the commands and counts as indirect arguments, and the counts' totals are copied out.
    for (CullPhaseBuffers &phase : culling.phases) {
        phase.visible = gpu_numbers(device, gpu, culling.draw_count);
        phase.draw_slots = gpu_numbers(device, gpu, culling.draw_count + 1, vk::BufferUsageFlagBits::eTransferSrc);
        phase.group_flags = gpu_numbers(device, gpu, culling.group_count);
        phase.group_slots = gpu_numbers(device, gpu, culling.group_count + 1, vk::BufferUsageFlagBits::eTransferSrc);
        phase.instances = gpu_numbers(device, gpu, culling.draw_count);
        phase.commands = create_buffer(device, gpu, std::max<vk::DeviceSize>(culling.group_count, 1) * command_size,
            vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eIndirectBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        phase.counts = gpu_numbers(device, gpu, draw_list_count, vk::BufferUsageFlagBits::eIndirectBuffer);
    }

    // Each phase's table: its own buffers, the shared ones, and the early phase's flags, which the late phase skips the 1s of. The early phase's table names its own flags there, and never reads them.
    for (CullPhaseBuffers &phase : culling.phases) {
        const CullTables tables{
            .order = culling.order.address,
            .groups = culling.groups.address,
            .lists = culling.lists.address,
            .visible = phase.visible.address,
            .draw_slots = phase.draw_slots.address,
            .group_flags = phase.group_flags.address,
            .group_slots = phase.group_slots.address,
            .instances = phase.instances.address,
            .commands = phase.commands.address,
            .counts = phase.counts.address,
            .early_visible = culling.phases[static_cast<std::size_t>(CullPhase::early)].visible.address,
            .draw_count = culling.draw_count,
            .group_count = culling.group_count,
        };
        phase.tables = upload(std::as_bytes(std::span(&tables, 1)));
    }

    return culling;
}

void clear_depth_pyramid(
    const vk::raii::Device &device,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    const Image &pyramid
) {
    submit_and_wait(device, queue, pool, [&](const vk::raii::CommandBuffer &commands) {
        // Into eGeneral, the layout storage images are written and read in, which the pyramid then keeps for good: first for the clear, then for the compute shaders.
        vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eNone,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eClear,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = *pyramid.handle,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = pyramid.mip_levels,
                .layerCount = 1,
            },
        };
        commands.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});

        commands.clearColorImage(*pyramid.handle, vk::ImageLayout::eGeneral,
            vk::ClearColorValue{.float32 = std::array{0.0f, 0.0f, 0.0f, 0.0f}}, barrier.subresourceRange);

        barrier.srcStageMask = vk::PipelineStageFlagBits2::eClear;
        barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead;
        barrier.oldLayout = vk::ImageLayout::eGeneral;
        commands.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    });
}

// Recording

void record_culling(
    const vk::raii::CommandBuffer &commands,
    const DrawCulling &culling,
    CullPhase phase,
    vk::DeviceAddress frame,
    vk::Buffer readback
) {
    const CullPhaseBuffers &buffers = culling.phases[static_cast<std::size_t>(phase)];
    const bool early = phase == CullPhase::early;

    // The previous frame's draws read the phase's commands, counts and instances, and its copy read the totals: wait for them before rewriting any. Rewriting what was read only needs the wait, so no access is made visible.
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eVertexShader
            | vk::PipelineStageFlagBits2::eCopy,
        vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);

    // Every step reads its buffers through the phase's table.
    const CullPushData push{.frame = frame, .tables = buffers.tables.address};

    const auto step = [&](const vk::raii::Pipeline &pipeline, std::uint32_t workgroup_count) {
        bind(commands, pipeline, push);
        commands.dispatch(workgroup_count, 1, 1);
    };

    // 1. Which draws this phase draws.
    step(early ? culling.cull_early : culling.cull_late, workgroups(culling.draw_count));
    compute_to_compute(commands);

    // 2. Each visible draw's slot among the instances. One workgroup.
    step(culling.scan_draws, 1);
    compute_to_compute(commands);

    // 3. Which groups have a visible draw.
    step(culling.mark_groups, workgroups(culling.group_count));
    compute_to_compute(commands);

    // 4. Each such group's command slot. One workgroup.
    step(culling.scan_groups, 1);
    compute_to_compute(commands);

    // 5. The instances.
    step(culling.write_instances, workgroups(culling.draw_count));

    // 6. The commands and counts, which don't read step 5's results: no barrier between.
    step(culling.write_commands, workgroups(culling.group_count));

    // The draws read the commands and counts as indirect arguments, and the instances in their vertex shaders; the copy reads the totals; the late cull reads the early phase's flags.
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eVertexShader
            | vk::PipelineStageFlagBits2::eCopy | vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eShaderStorageRead
            | vk::AccessFlagBits2::eTransferRead);

    // The totals are the scans' last numbers.
    commands.copyBuffer(*buffers.draw_slots.handle, readback, vk::BufferCopy{
        .srcOffset = culling.draw_count * sizeof(std::uint32_t),
        .dstOffset = early ? offsetof(CullTotals, early_draws) : offsetof(CullTotals, late_draws),
        .size = sizeof(std::uint32_t),
    });
    commands.copyBuffer(*buffers.group_slots.handle, readback, vk::BufferCopy{
        .srcOffset = culling.group_count * sizeof(std::uint32_t),
        .dstOffset = early ? offsetof(CullTotals, early_commands) : offsetof(CullTotals, late_commands),
        .size = sizeof(std::uint32_t),
    });

    // Visible to the CPU once it has waited for this frame's fence.
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
}

void record_depth_pyramid(
    const vk::raii::CommandBuffer &commands,
    const DrawCulling &culling,
    const Image &pyramid,
    std::uint32_t depth_slot,
    std::uint32_t pyramid_slot
) {
    // The early cull just read every level, and the previous frame's late cull before it: both must finish before the levels are rewritten. The pyramid stays in eGeneral, so a memory barrier is all it takes.
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);

    // One step per level, in the shaders' 8 x 8 workgroups.
    const auto step = [&](const vk::raii::Pipeline &pipeline, const CullPushData &push) {
        bind(commands, pipeline, push);
        commands.dispatch((push.target_size.x + 7) / 8, (push.target_size.y + 7) / 8, 1);
    };

    // 1. Level 0: the depth buffer, copied.
    const glm::uvec2 level0{pyramid.extent.width, pyramid.extent.height};
    step(culling.copy_depth, CullPushData{
        .source = depth_slot,
        .target = pyramid_slot,
        .source_size = level0,
        .target_size = level0,
    });

    // 2. Each level below, from the one above.
    for (std::uint32_t level = 1; level < pyramid.mip_levels; ++level) {
        const vk::Extent2D above = level_size(pyramid.extent, level - 1);
        const vk::Extent2D size = level_size(pyramid.extent, level);

        compute_to_compute(commands);
        step(culling.reduce_depth, CullPushData{
            .source = pyramid_slot + level - 1,
            .target = pyramid_slot + level,
            .source_size = {above.width, above.height},
            .target_size = {size.width, size.height},
        });
    }

    // The late cull reads the levels, and the next frame's early cull after it.
    compute_to_compute(commands);
}

void draw_list(const vk::raii::CommandBuffer &commands, const DrawCulling &culling, CullPhase phase, std::uint32_t list) {
    // The GPU reads the list's count from the phase's `counts`, and draws that many of the phase's commands at the start of its run, never more than the run is long.
    const CullPhaseBuffers &buffers = culling.phases[static_cast<std::size_t>(phase)];
    const DrawListRange &range = culling.list_ranges[list];

    commands.drawIndexedIndirectCount(
        *buffers.commands.handle, range.first_group * command_size,
        *buffers.counts.handle, list * sizeof(std::uint32_t),
        range.group_count, static_cast<std::uint32_t>(command_size));
}
```

## 17.5 Instances per phase: `mesh.slang`

### Why
An instance's draw index is in the phase's instances, which the push data now names.

### How
- **`vertexMain`** reads `push.instances[instance]` instead of the frame's instances. Nothing else changes.

### Code
In `game-engine/shaders/mesh.slang`, replace `vertexMain` with:
```slang
// SV_VulkanVertexID is Vulkan's own gl_VertexIndex, which includes the draw's vertexOffset: each primitive's indices start at 0, and the draw adds where that primitive's vertices begin in the shared buffer. SV_VulkanInstanceID is gl_InstanceIndex, which likewise counts from the command's firstInstance: the cull points that at the command's run of visible draws in its phase's instances, which the push data names, so each instance finds its draw there.
[shader("vertex")]
VertexOutput vertexMain(uint vertex_id : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    FrameData *frame = push.frame;
    const Vertex vertex = frame.vertices[vertex_id];
    const uint draw_index = push.instances[instance];
    const DrawData draw = frame.draws[draw_index];

    // The vertex in its draw's cell, then relative to the camera.
    const float3 in_cell = mul(draw.model, float4(vertex.position, 1.0)).xyz;
    const float3 relative_position = camera_relative(frame, draw.cell, in_cell);

    // Tangent and bitangent lie along the surface, so they move with the model matrix, like positions; only the normal needs the normal matrix. The bitangent is built before the transform, from glTF's rule B = cross(N, T) * w: a mirroring transform then mirrors it too.
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
```

## 17.6 The frame: `main.cpp`

### Why
The prepass is split in two around the pyramid, every later pass draws both phases, and the pyramid's levels need their heap slots.

### How
- **`ScreenSlots::depth_pyramid`** is the first of `max_depth_pyramid_levels` slots, one per level; `screen_slot_count` grows by that many. `describe_screen` writes a storage descriptor per level the pyramid has, each a view of just that level, then clears the pyramid: after a resize too, since the image is new.
- **`draw_mode`** takes the phase, and pushes its instances with the frame.
- **`record_frame`,** in order:
  1. **The early cull,** against the pyramid the previous frame built.
  2. **The prepass, early draws:** depth and normals cleared, the early phase's solid lists drawn.
  3. **The depth pyramid.** The depth buffer goes to its read-only layout, the copy samples it, and the reductions follow.
  4. **The late cull,** against the new pyramid.
  5. **The prepass, late draws:** the depth buffer goes back to being drawn into, the normals' first half is made to land before the second begins, both attachments are loaded rather than cleared, and the late phase's lists are drawn.
  6. **The rest as before:** ambient occlusion, then the lighting, the transparency and the tone mapping, the lighting and transparency passes drawing each list from both phases.
- **`FrameData`** gets the pyramid's slot, its level count and the screen's size.
- **`scene_file`** is the box field (17.7), and `test_lights` is 0: the field is lit by the sun. Sponza, with `test_lights` back at 256, draws as it did in Chapter 16.
- **The title** says `drawn N of M (E early, L late) in K commands`, with the drawn draws and the commands summed over the phases.

### Code
`game-engine/src/main.cpp`:
```cpp
#include "includes/acceleration.h"
#include "includes/ambient_occlusion.h"
#include "includes/buffer.h"
#include "includes/camera.h"
#include "includes/cells.h"
#include "includes/culling.h"
#include "includes/daylight.h"
#include "includes/descriptor_heap.h"
#include "includes/environment.h"
#include "includes/light_clusters.h"
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

    // How far, in cells each way, the camera may stray from the TLAS's origin cell before the TLAS is rebuilt around it: 16 cells of 64 m, about 1 km. Within that, a ray near the camera is traced at coordinates under about 1.1 km, where floats are 0.06 to 0.12 mm apart.
    constexpr int tlas_reach_cells = 16;

    // What each in-flight frame needs for itself. `data` holds this frame's FrameData; the GPU may still be reading the other frame's while the CPU writes this one.
    struct Frame {
        vk::raii::CommandBuffer commands = nullptr;
        vk::raii::Semaphore image_acquired = nullptr;  // swapchain image is ready to draw into
        vk::raii::Fence done = nullptr;                // GPU finished this frame's commands
        Buffer data;                                   // one FrameData, host-visible
        FrameData *mapped = nullptr;                   // `data`, mapped for the CPU to write
        Buffer cull_totals;                            // the cull's totals, copied out, host-visible
        const CullTotals *totals = nullptr;            // `cull_totals`, mapped for the CPU to read
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

    // The alpha modes the prepass and the lighting pass draw, in the order of their pipelines. The see-through mode, AlphaMode::blend, has only the transparency pass's.
    constexpr std::array solid_modes{AlphaMode::opaque, AlphaMode::mask};

    // glTF's front faces wind counter-clockwise, seen from the front. Our projection's Y flip (see camera.cpp) only undoes the difference between OpenGL's upward Y and Vulkan's downward one, so on screen they still wind counter-clockwise. A mirroring transform reverses that.
    constexpr vk::FrontFace front_face = vk::FrontFace::eCounterClockwise;
    constexpr vk::FrontFace mirrored_front_face = vk::FrontFace::eClockwise;

    // Resource heap slots of the swapchain's images, written by describe_screen() in main and rewritten whenever the swapchain is rebuilt.
    struct ScreenSlots {
        std::uint32_t hdr = 0;             // sampled, by tone mapping
        std::uint32_t depth = 0;           // sampled, by ambient occlusion
        std::uint32_t normals = 0;         // sampled, by ambient occlusion
        std::uint32_t ao = 0;              // sampled, by the lighting pass
        AoTargets ao_targets;              // storage, for the AO pass
        std::uint32_t accum = 0;           // sampled, by the transparency composite
        std::uint32_t reveal = 0;          // sampled, by the transparency composite
        std::uint32_t depth_pyramid = 0;   // storage, one slot per level, for the pyramid's build and the cull
    };

    constexpr std::uint32_t screen_slot_count = 11 + max_depth_pyramid_levels;

    // Every graphics pipeline a frame uses. The prepass and the lighting pass have one per solid alpha mode, in solid_modes' order; see-through surfaces have only the transparency pass's.
    struct ScenePipelines {
        std::vector<vk::raii::Pipeline> prepass;
        std::vector<vk::raii::Pipeline> lighting;
        vk::raii::Pipeline transparency = nullptr;
        vk::raii::Pipeline background = nullptr;
        vk::raii::Pipeline composite = nullptr;
        vk::raii::Pipeline tonemap = nullptr;
    };

    // What a frame draws and where it finishes: the scene's index buffer, the frame's data, the screen images' slots, and where the cull's totals go.
    struct DrawList {
        vk::Buffer index_buffer;
        vk::DeviceAddress frame = 0;   // this frame's FrameData
        ScreenSlots screen;
        View view = View::lit;
        vk::Buffer readback;           // this frame's copy of the cull's totals
        bool see_through = false;      // whether the scene has blended draws at all
        glm::vec3 sun_direction{0.0f}; // toward the sun
        float altitude = 0.0f;         // the camera's height above the ground, in metres
    };

    // Draws every list of alpha mode `mode` that cull phase `phase` kept, with `pipeline`: one indirect call per list, after setting the list's cull mode and front face. Push data says where the frame's data is and the phase's instances are; each instance finds its DrawData through those.
    void draw_mode(
        const vk::raii::CommandBuffer &commands,
        const DrawCulling &culling,
        const DrawList &draws,
        CullPhase phase,
        AlphaMode mode,
        const vk::raii::Pipeline &pipeline
    ) {
        commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

        const PushData push{
            .frame = draws.frame,
            .instances = culling.phases[static_cast<std::size_t>(phase)].instances.address,
        };

        commands.pushDataEXT(vk::PushDataInfoEXT{
            .offset = 0,
            .data = {.address = &push, .size = sizeof(push)},
        });

        for (const bool double_sided : {false, true}) {
            for (const bool mirrored : {false, true}) {
                const std::uint32_t list = draw_list_index(mode, double_sided, mirrored);

                if (culling.list_ranges[list].group_count == 0) {
                    continue;
                }

                // Single-sided surfaces are invisible from behind, so the GPU can skip their back faces before running the fragment shader.
                commands.setCullMode(double_sided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack);
                commands.setFrontFace(mirrored ? mirrored_front_face : front_face);
                draw_list(commands, culling, phase, list);
            }
        }
    }

    // The viewport and scissor every pass uses: the whole image. The pipelines leave both dynamic.
    void set_viewport(const vk::raii::CommandBuffer &commands, vk::Extent2D extent) {
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

    // Records a frame: the atmosphere's tables and the light clusters in compute shaders, then five passes, with the cull's two phases around the first:
    //   1. the depth prepass, in two halves: the early cull, then every solid surface it keeps, its depth and vertex normal; the depth pyramid, from that depth; the late cull against the pyramid, then the surfaces it adds,
    //   2. ambient occlusion, from the depth and normals, in compute shaders,
    //   3. the lighting, into the HDR image: each solid alpha mode's lists, from both phases, with that mode's pipeline, against the prepass's depth, then the sky behind them,
    //   4. transparency, if anything is see-through: the blended lists into two sums, in any order, then those laid over the HDR image,
    //   5. tone mapping, from the HDR image into the swapchain image, which is then ready to present.
    void record_frame(
        const vk::raii::CommandBuffer &commands,
        const Swapchain &swapchain,
        std::uint32_t image_index,
        const ScenePipelines &pipelines,
        const AmbientOcclusion &ambient_occlusion,
        const DrawCulling &culling,
        const LightClusters &light_clusters,
        const Environment &environment,
        const DescriptorHeaps &heaps,
        const DrawList &draws
    ) {
        const vk::Image image = swapchain.images[image_index];
        const vk::Image hdr = *swapchain.hdr.handle;
        const vk::Image depth = *swapchain.depth.handle;
        const vk::Image normals = *swapchain.normals.handle;

        commands.reset();
        commands.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        // Every texture and sampler the shaders read comes from these two heaps. They stay bound for every pass, graphics and compute.
        bind_descriptor_heaps(commands, heaps);

        // One index buffer for the whole scene. Indices go through the GPU's fixed-function index fetch, which also lets it reuse vertices shared between neighbouring triangles.
        commands.bindIndexBuffer(draws.index_buffer, 0, vk::IndexType::eUint32);

        // The atmosphere

        // The sky around the camera and the air in front of it, for this frame's sun, height and view: the background and the lighting read them.
        record_atmosphere(commands, environment, draws.frame, draws.sun_direction, draws.altitude);

        // The light clusters

        // Which point and spot lights reach which part of the view: the lighting and transparency passes light each pixel with its cluster's lights.
        record_light_clusters(commands, light_clusters, draws.frame);

        // The early cull

        // Every pass below draws only what the cull keeps, from its two phases' lists. The early phase tests against the depth pyramid the previous frame built.
        record_culling(commands, culling, CullPhase::early, draws.frame, draws.readback);

        // Pass 1: the depth prepass, the early draws

        // The depth buffer and the normals are shared by the frames in flight, so these also wait for the previous frame to finish reading them: the AO pass reads both, and the lighting pass depth-tests against the depth.
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

        // Reverse-Z: 0 is infinitely far. The depth is stored this time: the pyramid, the AO pass and the lighting pass all read it.
        vk::RenderingAttachmentInfo normal_attachment{
            .imageView = *swapchain.normals.view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = std::array{0.0f, 0.0f, 0.0f, 0.0f}}},
        };

        vk::RenderingAttachmentInfo prepass_depth{
            .imageView = *swapchain.depth.view,
            .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = 0.0f}},
        };

        const vk::Rect2D whole_image{.offset = {0, 0}, .extent = swapchain.extent};

        const vk::RenderingInfo prepass_info{
            .renderArea = whole_image,
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &normal_attachment,
            .pDepthAttachment = &prepass_depth,
        };

        commands.beginRendering(prepass_info);
        set_viewport(commands, swapchain.extent);

        // Solid surfaces only: opaque, then masked.
        for (std::size_t i = 0; i < solid_modes.size(); ++i) {
            draw_mode(commands, culling, draws, CullPhase::early, solid_modes[i], pipelines.prepass[i]);
        }

        commands.endRendering();

        // The depth pyramid

        // From the early draws' depth: the late cull tests against it, and the next frame's early cull. Its first step samples the depth buffer, which goes to its read-only layout for that.
        transition(commands, depth,
            vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eLateFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead,
            vk::ImageAspectFlagBits::eDepth
        );

        record_depth_pyramid(commands, culling, swapchain.depth_pyramid, draws.screen.depth, draws.screen.depth_pyramid);

        // The late cull

        // What the early cull left out, against this frame's pyramid: the draws its guess hid wrongly, because the view moved, or because nothing had been drawn yet.
        record_culling(commands, culling, CullPhase::late, draws.frame, draws.readback);

        // Pass 1, continued: the late draws

        // Over what the early draws left: the depth and the normals are loaded, not cleared. The depth goes back to being drawn into, and the normals' first half must land before the second starts.
        transition(commands, depth,
            vk::ImageLayout::eDepthReadOnlyOptimal, vk::ImageLayout::eDepthAttachmentOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::ImageAspectFlagBits::eDepth
        );

        transition(commands, normals,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite
        );

        normal_attachment.loadOp = vk::AttachmentLoadOp::eLoad;
        prepass_depth.loadOp = vk::AttachmentLoadOp::eLoad;

        commands.beginRendering(prepass_info);

        for (std::size_t i = 0; i < solid_modes.size(); ++i) {
            draw_mode(commands, culling, draws, CullPhase::late, solid_modes[i], pipelines.prepass[i]);
        }

        commands.endRendering();

        // Pass 2: ambient occlusion

        // From here on the depth is only read: by the AO pass, and as the lighting pass's depth test, which eDepthReadOnlyOptimal allows at once.
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

        record_ambient_occlusion(commands, ambient_occlusion, swapchain, draws.screen.ao_targets,
            AoPushData{.frame = draws.frame, .depth = draws.screen.depth, .normals = draws.screen.normals});

        // Pass 3: the lighting

        // The HDR image is shared by the frames in flight too: this waits for the previous frame's tone mapping to finish reading it.
        transition(commands, hdr,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite
        );

        // Every pixel is drawn over, by the scene or the sky; clearing is just cheaper than loading what was there.
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

        for (std::size_t i = 0; i < solid_modes.size(); ++i) {
            for (const CullPhase phase : cull_phases) {
                draw_mode(commands, culling, draws, phase, solid_modes[i], pipelines.lighting[i]);
            }
        }

        // The sky goes in once everything solid is drawn: it only covers pixels still at depth 0, infinitely far.
        commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.background);

        const PushData sky_push{.frame = draws.frame, .instances = 0};  // one triangle, no instances to look up
        commands.pushDataEXT(vk::PushDataInfoEXT{
            .offset = 0,
            .data = {.address = &sky_push, .size = sizeof(sky_push)},
        });

        commands.draw(3, 1, 0, 0);
        commands.endRendering();

        // Pass 4: transparency

        if (draws.see_through) {
            const vk::Image accum = *swapchain.accum.handle;
            const vk::Image reveal = *swapchain.reveal.handle;

            // Both sums are shared by the frames in flight: this waits for the previous frame's composite to finish reading them.
            for (const vk::Image sum : {accum, reveal}) {
                transition(commands, sum,
                    vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                    vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
                    vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite
                );
            }

            // Before any layer: nothing summed, and the whole scene showing through.
            const std::array sum_attachments{
                vk::RenderingAttachmentInfo{
                    .imageView = *swapchain.accum.view,
                    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .loadOp = vk::AttachmentLoadOp::eClear,
                    .storeOp = vk::AttachmentStoreOp::eStore,
                    .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = std::array{0.0f, 0.0f, 0.0f, 0.0f}}},
                },
                vk::RenderingAttachmentInfo{
                    .imageView = *swapchain.reveal.view,
                    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .loadOp = vk::AttachmentLoadOp::eClear,
                    .storeOp = vk::AttachmentStoreOp::eStore,
                    .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = std::array{1.0f, 0.0f, 0.0f, 0.0f}}},
                },
            };

            // The same depth as the lighting pass: tested, not written, so a see-through surface behind a solid one is hidden, and one behind another see-through one still counts.
            commands.beginRendering(vk::RenderingInfo{
                .renderArea = whole_image,
                .layerCount = 1,
                .colorAttachmentCount = static_cast<std::uint32_t>(sum_attachments.size()),
                .pColorAttachments = sum_attachments.data(),
                .pDepthAttachment = &lighting_depth,
            });

            // The viewport and scissor set earlier still apply: dynamic state lasts for the whole command buffer.
            for (const CullPhase phase : cull_phases) {
                draw_mode(commands, culling, draws, phase, AlphaMode::blend, pipelines.transparency);
            }
            commands.endRendering();

            // The composite reads both sums, and blends into the HDR image, which the lighting pass just wrote: the same layout, but its writes must land before the blend reads them.
            for (const vk::Image sum : {accum, reveal}) {
                transition(commands, sum,
                    vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
                    vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead
                );
            }

            transition(commands, hdr,
                vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite
            );

            // The lit scene is kept and blended into.
            const vk::RenderingAttachmentInfo scene_attachment{
                .imageView = *swapchain.hdr.view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
            };

            commands.beginRendering(vk::RenderingInfo{
                .renderArea = whole_image,
                .layerCount = 1,
                .colorAttachmentCount = 1,
                .pColorAttachments = &scene_attachment,
            });

            commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.composite);

            const CompositePushData composite_push{.accum = draws.screen.accum, .reveal = draws.screen.reveal};
            commands.pushDataEXT(vk::PushDataInfoEXT{
                .offset = 0,
                .data = {.address = &composite_push, .size = sizeof(composite_push)},
            });

            commands.draw(3, 1, 0, 0);
            commands.endRendering();
        }

        // Pass 5: tone mapping

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

    // Events

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
        "Sun shadow", "Light count",
    };

    // Handles every pending event and fills in `input` for this frame. False once the window was closed or Escape pressed.
    //   1-9 0 pick the view: 0 is the tenth, as on the keyboard
    //   l     the light count view: how many lights each cluster holds
    //   e     switch between the simulated sky and the photographed one
    //   o     switch ambient occlusion off and on, to compare
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

                // SDLK_1 to SDLK_9 are consecutive key codes; SDLK_0 comes before them.
                if (key >= SDLK_1 && key <= SDLK_9) {
                    settings.view = static_cast<View>(key - SDLK_1);
                } else if (key == SDLK_0) {
                    settings.view = View::shadow;
                } else if (key == SDLK_L) {
                    settings.view = View::light_count;
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
            std::println(stderr, "No GPU has Vulkan 1.4, the descriptor heap and ray queries, and can present to this window");
            return EXIT_FAILURE;
        }

        std::println("Using {}", gpu->device.getProperties().deviceName.data());
        print_descriptor_heap_properties(*gpu);

        vk::raii::Device device = create_device(*gpu);
        vk::raii::Queue queue = device.getQueue(gpu->queue_family, 0);
        Swapchain swapchain = create_swapchain(device, *gpu, surface, window.get());

        // Pipelines

        // recreate_swapchain() picks the same formats again, so the pipelines stay valid across resizes.
        //   - The prepass draws normals and depth, and the lighting pass the HDR image, for opaque and masked materials. The transparency pass draws blended ones into its two sums.
        //   - The sky draws into the HDR image, behind the scene, and the composite over it; tone mapping writes the swapchain image.
        ScenePipelines pipelines;

        for (const AlphaMode mode : solid_modes) {
            pipelines.prepass.push_back(create_mesh_pipeline(device, std::array{normal_format},
                depth_format, mode, MeshPass::depth_normals));
            pipelines.lighting.push_back(create_mesh_pipeline(device, std::array{hdr_format},
                depth_format, mode, MeshPass::lighting));
        }

        pipelines.transparency = create_mesh_pipeline(device, std::array{accum_format, reveal_format},
            depth_format, AlphaMode::blend, MeshPass::transparency);

        pipelines.background = create_fullscreen_pipeline(device, "background", hdr_format, depth_format);
        pipelines.composite = create_fullscreen_pipeline(device, "composite", hdr_format, vk::Format::eUndefined, ColorBlend::over);
        pipelines.tonemap = create_fullscreen_pipeline(device, "tonemap", swapchain.format);

        const AmbientOcclusion ambient_occlusion = create_ambient_occlusion(device);

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

            // Starts zeroed: the first wait on each frame reads it before the GPU has written it.
            Buffer cull_totals = create_buffer(device, *gpu, sizeof(CullTotals), vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            auto *totals = static_cast<CullTotals*>(cull_totals.memory.mapMemory(0, sizeof(CullTotals)));
            *totals = CullTotals{};

            frames.push_back(Frame{
                .commands = std::move(commands),
                .image_acquired = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
                // Start signalled, so the first wait on each frame returns straight away.
                .done = vk::raii::Fence(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}),
                .data = std::move(data),
                .mapped = mapped,
                .cull_totals = std::move(cull_totals),
                .totals = totals,
            });
        }

        // Scene

        // The glTF file to draw, under lecture-md/game-engine/assets: a field of 16,379 utility boxes, this chapter's test of the cull. Sponza/Sponza.gltf draws as before.
        const std::filesystem::path scene_file = std::filesystem::path(ASSET_DIR) / "utility_box_02/BoxField.gltf";

        // Where in the world the scene is placed, in metres: its origin. Move it far away, to {100000.0, 0.0, 100000.0} say, 141 km out, and the image stays the same: everything is drawn relative to the camera.
        const glm::dvec3 scene_origin{0.0, 0.0, 0.0};

        const std::uint64_t load_start = SDL_GetTicksNS();
        Scene scene = load_gltf(scene_file, scene_origin);
        std::println("Loaded {}: {} vertices, {} triangles, {} primitives, {} draws, {} materials, {} images",
            scene_file.filename().string(), scene.vertices.size(), scene.indices.size() / 3,
            scene.primitives.size(), scene.draws.size(), scene.materials.size(), scene.images.size());

        // Ground under the scene, out to 32 km each way: something to see the air over, as far as the aerial perspective reaches. 1 cm below the scene's lowest point, so it never fights the scene's own floor.
        add_ground(scene, scene_origin, static_cast<double>(scene.bounds_min.y) - 0.01, 65536.0, 1024.0);

        // Lights to test the clusters with, through the scene's box: 0 for just the file's. Try 16, 256 and 1024.
        constexpr std::uint32_t test_lights = 0;
        add_test_lights(scene, test_lights);

        // Shadow rays for point and spot lights: 0 traces one for every light that reaches a pixel, exactly. 1 to 16 trace only that many, for the lights that give the pixel the most light; the rest light it unshadowed. Cheaper where many lights overlap, but light from the fainter ones can leak through walls.
        constexpr std::uint32_t shadow_ray_budget = 0;

        // Directional lights first: they reach everywhere, so they're not clustered, and the shader takes the first directional_count. The rest keep their order.
        const auto directional_end = std::ranges::stable_partition(scene.lights,
            [](const Light &light) { return light.type == LightType::directional; });
        const auto directional_count = static_cast<std::uint32_t>(directional_end.begin() - scene.lights.begin());

        // Each draw's matrices, triangles and box, and what the cull needs to group it. The normal matrix is the transposed inverse of the model matrix: under non-uniform scale, transforming a normal by the model matrix itself would tilt it off the surface.
        std::vector<DrawData> draw_data;
        std::vector<CullDraw> cull_draws;
        std::array<std::size_t, 3> mode_draws{};  // how many draws of each alpha mode

        for (const MeshDraw &draw : scene.draws) {
            const Primitive &primitive = scene.primitives[draw.primitive];
            const SceneMaterial &material = scene.materials[primitive.material];
            const std::uint32_t list = draw_list_index(material.alpha_mode, material.double_sided, draw.mirrored);

            draw_data.push_back(DrawData{
                .model = draw.model,
                .normal_matrix = glm::transpose(glm::inverse(draw.model)),
                .material = primitive.material,
                .first_index = primitive.first_index,
                .vertex_offset = primitive.vertex_offset,
                .cell = draw.cell,
                .bounds_min = draw.bounds_min,
                .bounds_max = draw.bounds_max,
            });
            cull_draws.push_back(CullDraw{
                .list = list,
                .primitive = draw.primitive,
                .index_count = primitive.index_count,
                .first_index = primitive.first_index,
                .vertex_offset = primitive.vertex_offset,
            });
            ++mode_draws[static_cast<std::size_t>(material.alpha_mode)];
        }

        std::println("Draws: {} opaque, {} masked, {} blended", mode_draws[0], mode_draws[1], mode_draws[2]);

        // Vertices and draw data are read through pointers; indices go to the GPU's index fetch, so that buffer is an index buffer. Vertices and indices are also what the acceleration structures are built from, and shadow rays read indices through a pointer too.
        const vk::BufferUsageFlags build_input = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
            | vk::BufferUsageFlagBits::eShaderDeviceAddress;
        const Buffer vertex_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.vertices)), build_input);
        const Buffer index_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.indices)), vk::BufferUsageFlagBits::eIndexBuffer | build_input);
        const Buffer draw_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(draw_data)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        // The cull, its groups and its draw lists, worked out for this scene's draws.
        const DrawCulling culling = create_draw_culling(device, *gpu, queue, command_pool, cull_draws);

        std::println("Culling: {} draws in {} groups", culling.draw_count, culling.group_count);

        // Most files have no lights, and a buffer can't be empty: then there's no buffer, and the shader's light count is 0.
        const Buffer light_buffer = scene.lights.empty() ? Buffer{} : upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.lights)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        std::println("Lights: {} ({} directional), plus the sun", scene.lights.size(), directional_count);

        // Acceleration structures

        const std::uint64_t acceleration_start = SDL_GetTicksNS();
        // Around the cell the camera starts in: the scene's origin.
        AccelerationStructures acceleration = build_acceleration_structures(device, *gpu, queue, command_pool,
            scene, vertex_buffer, index_buffer, to_cell(scene_origin).cell);

        std::println("Acceleration structures: {} BLAS, {} instances in {:.0f} ms", acceleration.blases.size(),
            scene.draws.size(), static_cast<double>(SDL_GetTicksNS() - acceleration_start) * 1e-6);

        // Textures and materials

        // Decode every image, upload them with mipmaps, and describe them in the descriptor heap. Texture 0 is white; scene image i is texture i + 1.
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
            .ao_targets = {
                .ao_depth = first_screen_slot + 4,
                .ao_normals = first_screen_slot + 5,
                .ao_raw = first_screen_slot + 6,
                .ao_blur = first_screen_slot + 7,
                .ao = first_screen_slot + 8,
            },
            .accum = first_screen_slot + 9,
            .reveal = first_screen_slot + 10,
            .depth_pyramid = first_screen_slot + 11,
        };

        // The swapchain's images are recreated with it, so their descriptors are rewritten every time: after this, only while the GPU is idle.
        const auto describe_screen = [&] {
            const auto whole = [](const Image &image, vk::ImageAspectFlags aspect) {
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
            write_image_descriptor(device, heaps, screen.ao_targets.ao_depth, whole(swapchain.ao_depth, color), storage);
            write_image_descriptor(device, heaps, screen.ao_targets.ao_normals, whole(swapchain.ao_normals, color), storage);
            write_image_descriptor(device, heaps, screen.ao_targets.ao_raw, whole(swapchain.ao_raw, color), storage);
            write_image_descriptor(device, heaps, screen.ao_targets.ao_blur, whole(swapchain.ao_blur, color), storage);
            write_image_descriptor(device, heaps, screen.ao_targets.ao, whole(swapchain.ao, color), storage);
            write_image_descriptor(device, heaps, screen.accum, whole(swapchain.accum, color));
            write_image_descriptor(device, heaps, screen.reveal, whole(swapchain.reveal, color));

            // The pyramid's levels, one slot each: a storage image is one level.
            for (std::uint32_t level = 0; level < swapchain.depth_pyramid.mip_levels; ++level) {
                vk::ImageViewCreateInfo view = whole(swapchain.depth_pyramid, color);
                view.subresourceRange.baseMipLevel = level;
                write_image_descriptor(device, heaps, screen.depth_pyramid + level, view, storage);
            }

            // Until a frame builds it, the pyramid is empty: the far plane everywhere.
            clear_depth_pyramid(device, queue, command_pool, swapchain.depth_pyramid);
        };

        describe_screen();

        // The light clusters, for these lights and this screen.
        LightClusters light_clusters = create_light_clusters(device, *gpu, queue, command_pool,
            static_cast<std::uint32_t>(scene.lights.size()), directional_count, swapchain.extent);

        // recreate_swapchain() waits for the GPU to go idle, so the slots are free to rewrite, and the clusters to remake, straight afterwards.
        const auto resize = [&] {
            recreate_swapchain(swapchain, device, *gpu, surface, window.get());
            describe_screen();
            resize_light_clusters(light_clusters, device, *gpu, swapchain.extent);
        };

        // The environment

        const std::uint64_t environment_start = SDL_GetTicksNS();
        Environment environment = create_environment(device, *gpu, queue, command_pool, heaps, first_screen_slot + screen_slot_count,
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
                .alpha_mode = material.alpha_mode,
                .base_color = slot(material.base_color),
                .metallic_roughness = slot(material.metallic_roughness),
                .normal = slot(material.normal),
                .occlusion = slot(material.occlusion),
                .emissive = slot(material.emissive),
            });
        }

        const Buffer material_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(materials)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        // Spawns at the scene's origin, looking down -Z.
        FlyCamera camera{.position = scene_origin};
        CameraInput input;
        Settings settings;
        Settings shown_settings{.hours = -1.0f};  // what the title shows; differs at first
        CullTotals shown_totals{};                // the cull's totals the title shows
        Settings sky_settings{.hours = -1.0f};    // what the environment was built for
        float sky_altitude = 0.0f;                // the camera's height it was built for

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

            // The sky: rebuilt whenever its source changes, or the time of day moves the sun in the simulated one. That takes a few milliseconds and waits for the GPU, which is fine for a key press.
            const glm::vec3 sun_direction = sun_direction_at(settings.hours);
            // The camera's height above the ground, which is at y = 0. The simulated sky is lit for it too, so climbing far enough, 100 m, rebuilds it.
            const auto altitude = static_cast<float>(std::max(camera.position.y, 1.0));
            const bool sky_moved = settings.sky == SkySource::atmosphere
                && (settings.hours != sky_settings.hours || std::abs(altitude - sky_altitude) > 100.0f);

            if (settings.sky != sky_settings.sky || sky_moved) {
                update_environment(environment, device, queue, heaps, settings.sky, sun_direction, altitude);
                sky_settings = settings;
                sky_altitude = altitude;
            }

            // Light and exposure. The meter reads the light falling on flat ground: the sky's irradiance on an upward-facing surface, plus the sun's share at its angle (Rec. 709 luminance of each). Compensation works like a camera's: +1 is a stop brighter, which means a lower EV (EV measures the light the camera expects).
            const auto luminance = [](glm::vec3 c) { return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b; };
            const glm::vec3 sun_illuminance = environment.mapped->sun_illuminance;
            const float ground_illuminance = luminance(sky_irradiance(*environment.mapped, {0.0f, 1.0f, 0.0f}))
                + luminance(sun_illuminance) * std::max(sun_direction.y, 0.0f);

            const float ev100 = std::clamp(metered_ev100(ground_illuminance), -2.0f, 16.0f) - settings.exposure_compensation;
            const float exposure = exposure_from_ev100(ev100);

            // Render

            Frame &frame = frames[frame_count % frames_in_flight];

            // 1. Wait until the GPU is done with this frame's command buffer and
            //    data from last time, then write this frame's data.
            (void)device.waitForFences(*frame.done, vk::True, no_timeout);

            // What the cull's phases kept the last time this frame's resources were used: two frames ago.
            const CullTotals totals = *frame.totals;

            // The title shows the view, the sky, the time, the exposure, whether ambient occlusion is on and what the cull kept, whenever one changes.
            if (settings.view != shown_settings.view || settings.sky != shown_settings.sky
                || settings.hours != shown_settings.hours
                || settings.exposure_compensation != shown_settings.exposure_compensation
                || settings.ambient_occlusion != shown_settings.ambient_occlusion
                || totals.early_draws != shown_totals.early_draws || totals.late_draws != shown_totals.late_draws
                || totals.early_commands != shown_totals.early_commands || totals.late_commands != shown_totals.late_commands) {
                const auto minutes = static_cast<int>(std::lround(settings.hours * 60.0f));
                const std::string sky = settings.sky == SkySource::atmosphere
                    ? std::format("sky {:02}:{:02}", minutes / 60, minutes % 60)
                    : std::string("photographed sky");
                const std::string title = std::format("game-engine: {}, {}, EV {:.1f}, AO {}, drawn {} of {} ({} early, {} late) in {} commands",
                    view_names[static_cast<std::size_t>(settings.view)], sky, ev100, settings.ambient_occlusion ? "on" : "off",
                    totals.early_draws + totals.late_draws, culling.draw_count, totals.early_draws, totals.late_draws,
                    totals.early_commands + totals.late_commands);

                SDL_SetWindowTitle(window.get(), title.c_str());
                shown_settings = settings;
                shown_totals = totals;
            }

            // Where the camera is: a cell and an offset, like everything the GPU places. Once it's far from the TLAS's origin cell, the TLAS is rebuilt around the camera's cell. That waits for the GPU to stop using the old one: a brief pause, once per kilometre or so.
            const CellPosition camera_at = to_cell(camera.position);

            if (glm::any(glm::greaterThan(glm::abs(camera_at.cell - acceleration.origin_cell), glm::ivec3(tlas_reach_cells)))) {
                device.waitIdle();
                build_tlas(device, *gpu, queue, command_pool, acceleration, scene, camera_at.cell);
                std::println("TLAS rebuilt around cell ({}, {}, {})", camera_at.cell.x, camera_at.cell.y, camera_at.cell.z);
            }

            // The view-projection matrix works in camera-relative space: the view only turns the world, the camera being at its origin.
            const glm::mat4 view_projection = camera.projection(aspect) * camera.view();

            *frame.mapped = FrameData{
                .view_projection = view_projection,
                .inverse_view_projection = glm::inverse(view_projection),
                .vertices = vertex_buffer.address,
                .indices = index_buffer.address,
                .draws = draw_buffer.address,
                .materials = material_buffer.address,
                .lights = light_buffer.address,
                .environment = environment.info.address,
                .scene_tlas = acceleration.tlas_address,
                .camera_cell = camera_at.cell,
                .exposure = exposure,
                .sun_direction = sun_direction,
                .directional_light_count = directional_count,
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
                .camera_offset = camera_at.offset,
                .tlas_offset = glm::vec3(camera.position - glm::dvec3(acceleration.origin_cell) * cell_size),
                .sky_view = environment.slots.sky_view,
                .aerial_inscatter = environment.slots.aerial_inscatter,
                .aerial_transmittance = environment.slots.aerial_transmittance,
                .atmosphere = settings.sky == SkySource::atmosphere ? 1u : 0u,
                .camera_forward = camera.forward(),
                .shadow_ray_budget = shadow_ray_budget,
                .cluster_tiles = light_clusters.tiles,
                .depth_pyramid = screen.depth_pyramid,
                .depth_pyramid_levels = swapchain.depth_pyramid.mip_levels,
                .screen = {swapchain.extent.width, swapchain.extent.height},
                .light_clusters = light_clusters.clusters.address,
                .visible_lights = light_clusters.visible.address,
            };

            const DrawList draws{
                .index_buffer = *index_buffer.handle,
                .frame = frame.data.address,
                .screen = screen,
                .view = settings.view,
                .readback = *frame.cull_totals.handle,
                .see_through = mode_draws[static_cast<std::size_t>(AlphaMode::blend)] > 0,
                .sun_direction = sun_direction,
                .altitude = altitude,
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
            record_frame(frame.commands, swapchain, image_index, pipelines, ambient_occlusion, culling, light_clusters, environment, heaps, draws);

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

## 17.7 The test field

### Why
Occlusion culling pays off in scenes with many draws hiding one another, and Sponza isn't one. A field of boxes, standing among them, is: whichever way the camera faces, nearly every box is behind a nearer one. It's a test of instancing too: 16,379 draws of one primitive, in one group, so the whole field is one command per pass, however many of its boxes are in view.

### How
- **The box** is Poly Haven's `utility_box_02`, a green electrical cabinet 0.92 m wide, 1.12 m tall and 0.43 m deep, with 6,268 triangles and three 1K textures: color, normals, and metal and roughness together. It's free (CC0). Download it from <https://polyhaven.com/a/utility_box_02> as **glTF, 1K**, and unpack it into `lecture-md/game-engine/assets/utility_box_02/`, so that `utility_box_02_1k.gltf`, its `.bin` and its `textures/` are there.
- **The field** is a glTF file of its own, `BoxField.gltf`, in the same directory: the box's mesh, material and textures, referred to by their indices and paths, and 16,384 nodes that each place the mesh once. Written by a script, since a file of that size is nothing to type:
  - **A 128 × 128 grid,** 2 m apart, 256 m across, with each box moved up to 0.4 m from its grid point either way, and turned its own way about the up axis. The amounts come from a hash of the box's number, so the field is the same every run.
  - **The floor is at y = −1:** the camera starts at the origin, a metre above it, between the boxes, which stand just taller than that.
  - **A clearing** 3 m around the origin, so the camera doesn't start inside a box. It takes 5 boxes: 16,379 remain.
- **The ground** is Chapter 14's, which `main` adds under any scene: here it lies a centimetre below the boxes' feet.

Save this as `lecture-md/game-engine/assets/box_field.py`, and run it from that directory with `python3 box_field.py utility_box_02`:
```python
# Writes BoxField.gltf next to Poly Haven's utility_box_02_1k.gltf: the same mesh, materials and textures, placed 16,384 times on a jittered 128 x 128 grid, 2 m apart, each box turned its own way about the up axis, 1 m below the origin, and none within 3 m of it. Run: python3 box_field.py assets/utility_box_02
import json, math, pathlib, sys

directory = pathlib.Path(sys.argv[1])
source = json.load(open(directory / "utility_box_02_1k.gltf"))

# A hash of the box's number and which value is wanted, as 0..1: the same field every run.
def random(box, value):
    x = (box * 0x9E3779B9 + value * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return (x >> 8) / float(1 << 24)

grid, pitch, jitter, clearing, floor = 128, 2.0, 0.4, 3.0, -1.0
nodes = []
for row in range(grid):
    for column in range(grid):
        i = row * grid + column
        x = (column - (grid - 1) / 2) * pitch + (random(i, 0) * 2 - 1) * jitter
        z = (row - (grid - 1) / 2) * pitch + (random(i, 1) * 2 - 1) * jitter
        if math.hypot(x, z) < clearing:
            continue
        yaw = random(i, 2) * 2 * math.pi
        nodes.append({"mesh": 0, "translation": [round(x, 3), floor, round(z, 3)], "rotation": [0.0, round(math.sin(yaw / 2), 6), 0.0, round(math.cos(yaw / 2), 6)]})

field = dict(source)
field["asset"] = {"version": "2.0", "generator": "box_field.py: Poly Haven utility_box_02 (CC0), placed 16,384 times"}
field["nodes"] = nodes
field["scenes"] = [{"nodes": list(range(len(nodes)))}]
field["scene"] = 0
json.dump(field, open(directory / "BoxField.gltf", "w"), separators=(",", ":"))
print(f"{len(nodes)} boxes over {grid * pitch:.0f} m")
```

It prints `16379 boxes over 256 m`.

## 17.8 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The terminal** shows `Loaded BoxField.gltf: 5068 vertices, 6268 triangles, 1 primitives, 16379 draws, 2 materials, 3 images`, then `Culling: 20475 draws in 2 groups`: the field's boxes and the ground's 4,096 tiles, one group each.
- **The image:** rows of green cabinets around the camera, under the midmorning sun, the plain and the sky beyond them.
- **The title** starts at `drawn 2312 of 20475 (2312 early, 0 late) in 2 commands`: Chapter 16's cull kept 5,367 draws from the same spot. With the camera still, every visible draw was visible last frame too, so the early phase keeps them all and the late phase adds none.
- **Turn** with the right mouse button held: the late count jumps as draws come into view that the previous frame's pyramid hid, and falls back to 0 when the camera stops. Nothing pops: the late draws are in the frame they're first visible in.
- **Fly up** (hold E) above the field: the counts rise toward Chapter 16's as the boxes stop hiding one another. From far away, the whole field is a few thousand pixels, and most of its boxes are still drawn: a box only a few pixels wide is dropped only if those pixels are nearer, and a field seen from above has few such pixels.
- **At night** (press `[` until about 22:00) with `test_lights` at 256, the field is dark at eye level: the lights sit low among the boxes, with ranges of 2 to 5 m. From above (hold E), each lights a few boxes and the ground between them in its own color.
- **Sponza** (`scene_file` back to `Sponza/Sponza.gltf`, `test_lights` to 256) is identical to Chapter 16's, pixel for pixel, by day and at night. Its title starts at `drawn 23 of 4199 (23 early, 0 late) in 20 commands`: Chapter 16 drew 1,137, most of them the ground's tiles, which the walls hide now.
- **No `[validation …]` lines.**

**What it costs.** Release, 1920 × 1080, RTX 5070 Laptop, the field by day:

| View | Draws kept, Chapter 16 | Draws kept, now | Frame, Chapter 16 | Frame, now | The pyramid |
|---|---|---|---|---|---|
| Among the boxes, at eye level | 5,367 | 2,312 | 10.3 ms | 4.3 ms | 0.3 ms |
| 40 m up, over the field | 15,354 | 15,354 | 33.0 ms | 33.1 ms | 0.8 ms |
| 600 m away, 10 m up | 17,529 | 16,777 | 33.1 ms | 31.4 ms | 0.8 ms |
| At eye level, at night, 256 test lights | 5,367 | 2,312 | 10.4 ms | 4.3 ms | 0.3 ms |
| 40 m up, at night, 256 test lights | 15,354 | 15,354 | 33.7 ms | 33.6 ms | 0.8 ms |

- **Among the boxes,** the cull drops 3,055 draws that Chapter 16 drew and the depth test threw away, and the frame takes less than half the time. The two phases together cost about what the one cull did, 0.13 ms, and the pyramid 0.3 ms: eleven dispatches with a barrier between each.
- **Over the field,** nothing hides anything, and every box costs what it did: 15,354 boxes of 6,268 triangles are 96 million triangles, drawn twice, in the prepass and in the lighting pass. That's what the next chapter is for.
- **Far away,** the test drops only 750 draws of a field that's a few thousand pixels tall. A box 3 pixels wide is tested at the level whose texels are 4 pixels wide, over a rectangle of up to 8 × 8 pixels, and the plain beyond the field shows somewhere in it. The test is conservative by design; distant draws are a job for level of detail, not for occlusion.
- **The test lights** change little: 256 lights over 256 m are far apart, and the cull and the clusters don't depend on each other.
- **The GPU slows down** under the heavier frames: tone mapping, the same work in every row, takes 0.14 ms in the first and 0.9 ms in the others, and the pyramid 0.3 and 0.8 ms. A laptop GPU at its power limit runs its clocks lower the more it has to do, so a pass's cost depends on what runs beside it.

Next, in Chapter 18, mesh shaders: the geometry in clusters, so the GPU can drop triangles facing away or too small to see, and pick each draw's level of detail itself.

# Chapter 3: Camera, matrices and depth

By the end of this chapter, nine tumbling cubes intersect each other in 3D, seen through a fly camera with the Unreal Editor's viewport controls. Three ideas make that work:
- **matrices** that place objects in the world and project the world onto the screen,
- a **depth buffer**, so nearer surfaces hide farther ones,
- **reverse-Z**, which keeps depth precise over long distances. Sponza in Chapter 4 needs that.

This chapter builds on [Chapter 2](02-buffers-device-addresses-push-data.md).

## 3.1 glm: `vendor/glm` and `CMakeLists.txt`

### Why
Placing and viewing objects in 3D takes 4×4 matrices and vectors on the CPU. **glm** is the standard C++ math library for graphics. Its types match the shader's `float3` and `float4x4`, and it provides the transforms we need: translate, rotate, look-at, perspective.

### How
- **Vendoring:** glm 1.0.3 is vendored like SDL. It ships an optional compiled library; `GLM_BUILD_LIBRARY OFF` skips that and leaves just the headers, as the `glm::glm` target.
- **`GLM_FORCE_DEPTH_ZERO_TO_ONE`:** glm was written for OpenGL, whose depth runs from −1 to 1 after projection. Vulkan's runs from 0 to 1. This define makes `glm::perspective` produce Vulkan's range. Without it, part of the depth range is wasted and the near plane lands in the wrong place.

### Code
From the repo root:
```bash
git clone --depth 1 --branch 1.0.3 https://github.com/g-truc/glm game-engine/vendor/glm
```

In `game-engine/CMakeLists.txt`, add this section after `add_subdirectory(vendor/Vulkan-Headers SYSTEM)`:
```cmake
# --- vendor/glm --------------------------------------------------------------
# Vectors and matrices for the CPU side. glm is header-only; GLM_BUILD_LIBRARY
# OFF skips its optional compiled library, leaving just the headers.

set(GLM_BUILD_LIBRARY OFF)
add_subdirectory(vendor/glm SYSTEM)
```

Then, after the `target_compile_definitions` block for vulkan.hpp, define the depth range and link glm. This replaces the `target_link_libraries` line:
```cmake
# glm follows OpenGL, whose clip-space depth runs from -1 to 1. Vulkan's runs
# from 0 to 1; this makes glm::perspective produce that range instead.
target_compile_definitions(game-engine PRIVATE GLM_FORCE_DEPTH_ZERO_TO_ONE)

target_link_libraries(game-engine PRIVATE SDL3::SDL3 Vulkan::Headers glm::glm)
```

**Build once and restart clangd.** Nothing uses glm yet, so the code from the previous chapter still builds. Run `./game-engine/build.bash` and pick option 1: CMake reconfigures and writes glm's include path into `compile_commands.json`. Then reload the editor window (**Developer: Reload Window**, or **clangd: Restart language server**), so clangd finds `<glm/glm.hpp>` when the code below includes it.

## 3.2 Matrices shared with the shader: `shader_types.h`, `mesh.slang`

### Why
Every vertex now goes through two matrices:
- the **model** matrix moves a cube from its own space (centered on the origin) to its place in the world,
- the **view-projection** matrix turns world positions into clip space: first relative to the camera (view), then onto the screen with perspective (projection).

Both have to reach the shader, and both sides must agree on how a matrix is laid out in memory.

### How
- **The push data grows to 136 bytes:** two `float4x4`s and the vertex pointer. That's well under the 256 bytes this GPU allows. The model matrix changes per draw. The view-projection matrix is the same for every draw; pushing it alongside keeps things simple while there are only a few draws.
- **Member order matters.** The push block uses std430 rules, where a matrix starts on a 16-byte boundary and a pointer on an 8-byte one. With the matrices first, every member lands exactly where the C++ struct puts it: 0, 64, 128. Pointer first would put the first matrix at 16 in the shader but at 8 in C++. The `static_assert`s would catch that.
- **Matrix layout:** glm stores a matrix column by column. `-matrix-layout-column-major` tells Slang to read it the same way, so `mul(M, v)` in the shader computes exactly glm's `M * v`. It's also Slang's default, but writing it in the build states what glm needs.
- **The vertex shader:** applies the model matrix, then the view-projection matrix. `Vertex` now uses `glm::vec3`, which is three floats, so the layout is unchanged.
- **Renaming:** the shader no longer draws one triangle, so `triangle.slang` becomes `mesh.slang`.

### Code
Rename the shader:
```bash
mv game-engine/shaders/triangle.slang game-engine/shaders/mesh.slang
```

`game-engine/src/includes/shader_types.h`:
```cpp
#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <cstddef>

// C++ mirrors of the structs in shaders/mesh.slang. The GPU reads these
// bytes as they are, so the two sides must agree on every size and offset;
// the static_asserts catch a mismatch at compile time.

// --- Vertex ------------------------------------------------------------------

// Slang lays out data behind a pointer like C: a float3 is 12 bytes, with no
// padding, so position is at byte 0, color at byte 12, and a vertex is 24.
// glm::vec3 is exactly three floats.
struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
};

static_assert(sizeof(Vertex) == 24);
static_assert(offsetof(Vertex, color) == 12);

// --- Push data ---------------------------------------------------------------

// Written with vkCmdPushDataEXT before each draw. The push block uses std430
// rules: each float4x4 is 64 bytes, and the matrices come first so the
// 8-byte pointer lands at 128 without padding.
struct PushData {
    glm::mat4 view_projection;   // world space -> clip space, the same for every draw
    glm::mat4 model;             // this object's space -> world space
    vk::DeviceAddress vertices;  // where the first Vertex is in GPU memory
};

static_assert(offsetof(PushData, model) == 64);
static_assert(offsetof(PushData, vertices) == 128);
static_assert(sizeof(PushData) == 136);
```

`game-engine/shaders/mesh.slang`:
```slang
// Draws a mesh whose vertices live in a GPU buffer, placed in the world by a
// model matrix and seen through a camera's view-projection matrix.

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

// Data behind a pointer is laid out like C: a float3 is 12 bytes and nothing
// is padded, so this matches the C++ Vertex exactly (24 bytes).
struct Vertex {
    float3 position;
    float3 color;
};

// Written with vkCmdPushDataEXT before each draw.
struct PushData {
    float4x4 view_projection;  // world space -> clip space
    float4x4 model;            // this object's space -> world space
    Vertex* vertices;          // device address of the first vertex
};

// In a descriptor heap pipeline, the push_constant block is where push data lands.
[[vk::push_constant]]
ConstantBuffer<PushData> push;

// --- Stage interface ---------------------------------------------------------

// What the vertex shader hands to the rasterizer. SV_Position is the
// clip-space position; every other field is interpolated across the triangle.
struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

// --- Vertex shader -----------------------------------------------------------

// SV_VulkanVertexID is Vulkan's own gl_VertexIndex. HLSL's SV_VertexID would
// subtract the draw's base vertex, which needs the DrawParameters capability.
[shader("vertex")]
VertexOutput vertexMain(uint vertex_id : SV_VulkanVertexID) {
    const Vertex vertex = push.vertices[vertex_id];

    // mul(M, v) treats v as a column vector, the same as glm's M * v.
    // Read right to left: object space -> world space -> clip space.
    const float4 world = mul(push.model, float4(vertex.position, 1.0));

    VertexOutput output;
    output.position = mul(push.view_projection, world);
    output.color = vertex.color;
    return output;
}

// --- Fragment shader ---------------------------------------------------------

// SV_Target: the value written to color attachment 0.
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    return float4(input.color, 1.0);
}
```

In `game-engine/CMakeLists.txt`, replace the shader's `add_custom_command` and the comment above it with:
```cmake
    # -fvk-use-entrypoint-name: keep vertexMain/fragmentMain as the SPIR-V
    #   entry point names (a file with a single entry point would get "main").
    # -matrix-layout-column-major: read float4x4 data column by column, the way
    #   glm stores it, so mul(M, v) in a shader means M * v in C++.
    # -g: keep names and source in the SPIR-V for debuggers like RenderDoc.
    add_custom_command(
        OUTPUT ${spirv}
        COMMAND ${SLANGC} ${source} -target spirv -profile spirv_1_6 -fvk-use-entrypoint-name
                -matrix-layout-column-major -g -o ${spirv}
        DEPENDS ${source}
        COMMENT "Compiling ${shader_name}.slang"
        VERBATIM
    )
```

## 3.3 Images and the depth buffer: `image.h`/`image.cpp`, `buffer.h`, `swapchain.h`/`swapchain.cpp`

### Why
With several objects on screen, a triangle drawn later can cover one that's nearer to the camera. A **depth buffer** fixes that. It's an image, the size of the screen, that stores how far away the nearest surface drawn so far is at each pixel. For each new fragment, the GPU compares its depth with the stored value and keeps whichever is nearer.

This is the first image we create ourselves; the swapchain's images belonged to the window system. Creating one works like a buffer:
1. describe the image,
2. allocate device-local memory for it and bind it,
3. create a **view**, which says which part of the image to use. For a depth buffer that's its depth aspect.

Chapter 5's textures will use the same helper.

### How
- **Sharing memory selection:** images need `find_memory_type` too, so it moves from `buffer.cpp`'s private namespace into `buffer.h`.
- **`create_image`** first checks that the GPU can use the format the way we intend; for a depth buffer, as a depth attachment. Then it does the three steps above.
- **Optimal tiling** lets the GPU arrange the texels however suits it, since the CPU never reads them.
- **The depth format is `D32_SFLOAT`,** 32-bit floating point. Reverse-Z (section 3.5) depends on floating-point depth.
- **The depth buffer belongs to the `Swapchain`,** because it must always match the swapchain images' size. `build()` creates it, so resizing the window rebuilds it with everything else.

### Code
In `game-engine/src/buffer.cpp`, make `find_memory_type` public. Delete the `namespace {` line above it and the `}  // namespace` line below it, along with its comment, which moves to the header. Then, in `game-engine/src/includes/buffer.h`, add `#include <cstdint>` to the includes, and this section before `// --- Creating buffers`:
```cpp
// --- Memory types ------------------------------------------------------------

// The GPU offers a few memory types, each a set of properties (device-local,
// host-visible, ...) in one of its heaps. `allowed` is the bitmask a buffer's
// or image's memory requirements permit; this returns the first allowed type
// that has every property in `required`.
std::uint32_t find_memory_type(const GpuChoice& gpu, std::uint32_t allowed, vk::MemoryPropertyFlags required);
```

`game-engine/src/includes/image.h`:
```cpp
#pragma once

#include "includes/vulkan_setup.h"

// A VkImage, the memory behind it, and a view of the whole image. Members are
// destroyed bottom-up: the view, then the image, then its memory.
struct Image {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image handle = nullptr;
    vk::raii::ImageView view = nullptr;
    vk::Format format = vk::Format::eUndefined;
    vk::Extent2D extent;
};

// A 2D image in device-local memory, with one mip level, and a view of its
// `aspect` (color, or depth for a depth buffer).
Image create_image(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    vk::Extent2D extent,
    vk::Format format,
    vk::ImageUsageFlags usage,
    vk::ImageAspectFlags aspect
);
```

`game-engine/src/image.cpp`:
```cpp
#include "includes/image.h"

#include "includes/buffer.h"

#include <stdexcept>

Image create_image(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    vk::Extent2D extent,
    vk::Format format,
    vk::ImageUsageFlags usage,
    vk::ImageAspectFlags aspect
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

    // 1. The image object. Optimal tiling lets the GPU arrange texels however
    //    is fastest for it; we never read them from the CPU.
    image.handle = vk::raii::Image(device, vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
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

    // 3. A view of the whole image, which is what rendering attaches to.
    image.view = vk::raii::ImageView(device, vk::ImageViewCreateInfo{
        .image = *image.handle,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = 1,
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

#include <vector>

// 32-bit float depth: the precision reverse-Z depth needs (see camera.cpp).
constexpr vk::Format depth_format = vk::Format::eD32Sfloat;

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

    // One depth buffer, the size of the images. Every frame clears it before
    // drawing, so frames in flight can share it.
    Image depth;
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

In `swapchain.cpp`, `build()` creates the depth buffer after the views and semaphores, and the comment in `recreate_swapchain` now mentions it.

`game-engine/src/swapchain.cpp`:
```cpp
#include "includes/swapchain.h"

#include <algorithm>
#include <limits>

namespace {

// --- Choosing the swapchain's settings ---------------------------------------

vk::SurfaceFormatKHR choose_format(const std::vector<vk::SurfaceFormatKHR>& formats) {
    // 8-bit BGRA with sRGB encoding: shaders write linear colors and the GPU
    // encodes them to sRGB on the way out. Otherwise take what the surface offers.
    for (const vk::SurfaceFormatKHR& format : formats) {
        if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return format;
        }
    }

    return formats.front();
}

vk::Extent2D choose_extent(const vk::SurfaceCapabilitiesKHR& capabilities, int width, int height) {
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

// --- Building one ------------------------------------------------------------

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

    swapchain.depth = create_image(device, gpu, swapchain.extent, depth_format,
        vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth);

    return swapchain;
}

}  // namespace

// --- Create and recreate -----------------------------------------------------

Swapchain create_swapchain(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::SurfaceKHR& surface,
    SDL_Window* window
) {
    return build(device, gpu, surface, window, nullptr);
}

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
    // then the old swapchain itself and its depth buffer.
    swapchain.views.clear();
    swapchain.rendered.clear();
    swapchain = std::move(next);
}
```

## 3.4 Depth testing in the pipeline: `pipeline.h` / `pipeline.cpp`

### Why
The depth buffer only does something if the pipeline uses it. The pipeline has to say how to test depth, and, since dynamic rendering has no render pass, which depth format it will draw into.

### How
- **Depth state:** test against the depth buffer, and write the depth of every fragment that passes. With reverse-Z, a *greater* depth means *nearer*, so the comparison is `eGreater`.
- **Formats:** `PipelineRenderingCreateInfo` gains the depth format.
- **The name:** the function becomes `create_mesh_pipeline` and loads `mesh.spv`.

### Code
`game-engine/src/includes/pipeline.h`:
```cpp
#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <filesystem>
#include <vector>

// A .spv file as the 32-bit words SPIR-V is made of.
std::vector<std::uint32_t> read_spirv(const std::filesystem::path& path);

// Draws shaders/mesh.slang into a `color_format` image, depth-tested against
// a `depth_format` depth buffer. There is no pipeline layout: shaders will
// find their resources in the descriptor heap.
vk::raii::Pipeline create_mesh_pipeline(const vk::raii::Device& device, vk::Format color_format, vk::Format depth_format);
```

`game-engine/src/pipeline.cpp`:
```cpp
#include "includes/pipeline.h"

#include <array>
#include <fstream>
#include <stdexcept>

// --- Loading SPIR-V ----------------------------------------------------------

std::vector<std::uint32_t> read_spirv(const std::filesystem::path& path) {
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

// --- The mesh pipeline -------------------------------------------------------

vk::raii::Pipeline create_mesh_pipeline(const vk::raii::Device& device, vk::Format color_format, vk::Format depth_format) {
    // Shaders: one module, two entry points picked by name. The module is
    // only needed while the pipeline is built, so it's destroyed on return.
    const std::vector<std::uint32_t> spirv = read_spirv(std::filesystem::path(SHADER_DIR) / "mesh.spv");

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

    const std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    const vk::PipelineDynamicStateCreateInfo dynamic{
        .dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    // Rasterization: filled triangles, both sides drawn for now.
    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    // Depth: keep a fragment only if it's nearer than what's already there,
    // then record its depth. With reverse-Z (see camera.cpp) nearer means a
    // *greater* depth value, and the buffer is cleared to 0, the far plane.
    const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eGreater,
    };

    // Color output: no blending, the fragment shader's color replaces what's there.
    const vk::PipelineColorBlendAttachmentState blend_attachment{
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

## 3.5 The camera: `camera.h` / `camera.cpp`

### Why
The camera turns "where am I and where am I looking" into the view and projection matrices. Being able to fly it around makes the 3D scene, and every later scene, easy to inspect. We copy the Unreal Editor's viewport controls, which many people already know:

| Input | Action |
|---|---|
| Right mouse button held | the mouse looks around; WASD moves, Q/E go down and up; scroll changes the flying speed |
| Left mouse button drag | left/right turns, up/down moves forward and back along the ground |
| Middle mouse button drag | pans sideways and up and down |
| Scroll wheel | moves forward and back |

### How
- **Where it starts:** `FlyCamera` spawns at the origin with yaw and pitch 0, looking "forward", and the player takes it from there.
- **Where it looks:** the camera stores a position plus two angles, **yaw** (turning left and right around +Y) and **pitch** (looking up and down). `forward()` turns them into a direction. At yaw 0 and pitch 0 that direction is −Z, because glm's view space looks down −Z.
- **`view()`:** `glm::lookAt` builds the view matrix from the position, a point straight ahead, and the world's up direction.
- **`projection()`:** `glm::perspective` with the vertical field of view and the image's aspect ratio. Two Vulkan-specific details:
  - **Reverse-Z:** passing the far plane *before* the near plane maps near to depth 1 and far to depth 0. Perspective crowds most of the depth range close to the camera, while floating-point numbers are most precise near 0. Reversing the range puts the float precision where perspective needs it, so distant surfaces don't flicker against each other. The depth buffer is cleared to 0, the far plane, and the test keeps greater values.
  - **Y flip:** glm's clip space has +Y up, like OpenGL. Vulkan's has +Y down. Negating `projection[1][1]` flips it.
- **`update_camera`:** the button held decides what the mouse does.
  - **Right button (fly):** mouse movement turns the camera, with pitch clamped just short of straight up or down, where forward and up would point the same way. `SDL_GetKeyboardState` reports which keys are held right now, and the camera moves along its forward, right and up directions. Scrolling multiplies the speed by 1.25 per step, between 0.05 and 500.
  - **Speed:** movement is multiplied by the seconds since the last frame, so it's the same at any frame rate.
  - **Left button (drag):** turning works as above, but moving the mouse up and down moves the camera along the ground. That uses forward with the pitch removed, so looking down doesn't send you into the floor.
  - **Middle button (pan):** moves the camera in the plane facing it.
  - **Scroll** without the right button moves forward and back.
  - **Drag distances** scale with the flying speed, so they suit the scene as you adjust it.

### Code
`game-engine/src/includes/camera.h`:
```cpp
#pragma once

#include <glm/glm.hpp>

// What the mouse did since the last frame. The keyboard is read directly
// from SDL in update_camera().
struct CameraInput {
    glm::vec2 mouse_delta{0.0f};  // pixels moved since the last frame
    float wheel = 0.0f;           // scroll steps; positive is away from you
    bool right_button = false;    // held: look around and fly
    bool left_button = false;     // held: turn and move along the ground
    bool middle_button = false;   // held: pan
};

// A fly camera with the Unreal Editor viewport's controls:
//   right button held   mouse looks around, WASD moves, Q/E go down/up,
//                       scroll changes the flying speed
//   left button drag    left/right turns, up/down moves forward/back
//   middle button drag  pans sideways and up/down
//   scroll              moves forward/back
struct FlyCamera {
    glm::vec3 position{0.0f};
    float yaw = 0.0f;    // radians around +Y; 0 looks down -Z, glm's "forward"
    float pitch = 0.0f;  // radians up (+) or down (-)
    float vertical_fov = glm::radians(60.0f);
    float near_plane = 0.05f;
    float far_plane = 500.0f;
    float speed = 3.0f;  // world units per second while flying

    // The unit vector the camera looks along.
    glm::vec3 forward() const;

    // World space -> view space: the world as seen from the camera.
    glm::mat4 view() const;

    // View space -> clip space, for an image `aspect` (width / height) wide.
    glm::mat4 projection(float aspect) const;
};

// Moves and turns `camera` from `input` and the keyboard, `seconds` after
// the last update.
void update_camera(FlyCamera& camera, const CameraInput& input, float seconds);
```

`game-engine/src/camera.cpp`:
```cpp
#include "includes/camera.h"

#include <SDL3/SDL.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace {

constexpr glm::vec3 world_up{0.0f, 1.0f, 0.0f};

constexpr float look_sensitivity = 0.003f;  // radians per pixel of mouse movement
constexpr float drag_sensitivity = 0.01f;   // fraction of `speed` moved per pixel of drag
constexpr float scroll_step = 0.2f;         // fraction of `speed` moved per scroll step
constexpr float speed_step = 1.25f;         // flying speed multiplier per scroll step
constexpr float min_speed = 0.05f;
constexpr float max_speed = 500.0f;

}  // namespace

// --- Matrices ----------------------------------------------------------------

glm::vec3 FlyCamera::forward() const {
    // Yaw turns around +Y, pitch tilts up and down. At yaw 0 and pitch 0 this
    // is (0, 0, -1): glm's view space looks down -Z.
    return {
        -std::sin(yaw) * std::cos(pitch),
        std::sin(pitch),
        -std::cos(yaw) * std::cos(pitch),
    };
}

glm::mat4 FlyCamera::view() const {
    return glm::lookAt(position, position + forward(), world_up);
}

glm::mat4 FlyCamera::projection(float aspect) const {
    // Reverse-Z: passing far before near maps the near plane to depth 1 and the
    // far plane to depth 0. Floats are most precise near 0, and perspective
    // crowds distant depths together; reversing puts the two in balance.
    // GLM_FORCE_DEPTH_ZERO_TO_ONE (in CMakeLists.txt) makes glm produce
    // Vulkan's 0..1 depth range instead of OpenGL's -1..1.
    glm::mat4 projection = glm::perspective(vertical_fov, aspect, far_plane, near_plane);

    // glm follows OpenGL, where clip-space +Y is up; in Vulkan it's down.
    projection[1][1] *= -1.0f;
    return projection;
}

// --- Controls ----------------------------------------------------------------

void update_camera(FlyCamera& camera, const CameraInput& input, float seconds) {
    const glm::vec3 forward = camera.forward();
    const glm::vec3 right = glm::normalize(glm::cross(forward, world_up));

    // Forward with the pitch removed, for moving along the ground.
    const glm::vec3 ground_forward{-std::sin(camera.yaw), 0.0f, -std::cos(camera.yaw)};

    const float drag_distance = camera.speed * drag_sensitivity;

    if (input.right_button) {
        // Look: each pixel of mouse movement is a small angle. Pitch stops just
        // short of straight up or down, where "forward" and "up" would coincide.
        camera.yaw -= input.mouse_delta.x * look_sensitivity;
        camera.pitch -= input.mouse_delta.y * look_sensitivity;
        camera.pitch = std::clamp(camera.pitch, glm::radians(-89.0f), glm::radians(89.0f));

        // Scrolling while flying changes the speed, 25% per step.
        camera.speed = std::clamp(camera.speed * std::pow(speed_step, input.wheel), min_speed, max_speed);

        // Fly: SDL keeps the current up/down state of every key.
        const bool* keys = SDL_GetKeyboardState(nullptr);

        glm::vec3 direction{0.0f};
        if (keys[SDL_SCANCODE_W]) direction += forward;
        if (keys[SDL_SCANCODE_S]) direction -= forward;
        if (keys[SDL_SCANCODE_D]) direction += right;
        if (keys[SDL_SCANCODE_A]) direction -= right;
        if (keys[SDL_SCANCODE_E]) direction += world_up;
        if (keys[SDL_SCANCODE_Q]) direction -= world_up;

        // Speed times seconds since the last frame: the same speed at any frame rate.
        if (direction != glm::vec3{0.0f}) {
            camera.position += glm::normalize(direction) * camera.speed * seconds;
        }

        return;
    }

    if (input.left_button) {
        // Left drag: sideways turns, up/down moves along the ground.
        camera.yaw -= input.mouse_delta.x * look_sensitivity;
        camera.position -= ground_forward * input.mouse_delta.y * drag_distance;
    } else if (input.middle_button) {
        // Middle drag: pan in the plane facing the camera.
        const glm::vec3 up = glm::cross(right, forward);
        camera.position += (right * input.mouse_delta.x - up * input.mouse_delta.y) * drag_distance;
    }

    // Scrolling without flying moves forward and back.
    camera.position += forward * input.wheel * camera.speed * scroll_step;
}
```

## 3.6 The scene: `main.cpp`

### Why
`main` now builds a cube and draws it nine times with different model matrices. Each frame it updates the camera from the mouse and keyboard, and clears and uses the depth buffer.

### How
- **`transition`** takes an `aspect` argument, defaulting to color, so it can also move the depth image between layouts.
- **`record_frame`:**
  - **Clearing depth:** before rendering, a barrier moves the depth image into depth-attachment layout. The image is shared by both frames in flight, so the barrier also waits for the previous frame's depth tests to finish (`eLateFragmentTests`) before this frame clears it.
  - **The depth attachment** is cleared to 0. Its store op is `eDontCare`, because nothing reads depth after the frame.
  - **Drawing:** a `DrawList` describes what to draw: one vertex buffer, and a model matrix per object. Each object gets its own push data and draw call.
- **`make_cube`:** builds 36 vertices, 2 triangles for each of 6 faces, each face its own color. Corners go counter-clockwise as seen from outside, which Chapter 7's back-face culling will rely on.
- **`poll_events`:**
  - adds up the mouse movement and scroll steps in `CameraInput`,
  - asks SDL which buttons are held after the events are handled,
  - while any of the three buttons is held, turns on SDL's *relative mouse mode*, which hides the cursor and keeps reporting movement without stopping at the screen edge.
- **The camera** spawns at the origin; the cube grid sits 6 units in front of it and slightly below.
- **Each frame:**
  - measure the seconds since the last frame with `SDL_GetTicksNS`,
  - update the camera,
  - rotate each cube about its own axis,
  - build the view-projection matrix for the current window aspect,
  - record and submit as before.

### Code
`game-engine/src/main.cpp`:
```cpp
#include "includes/buffer.h"
#include "includes/camera.h"
#include "includes/pipeline.h"
#include "includes/sdl.h"
#include "includes/shader_types.h"
#include "includes/swapchain.h"
#include "includes/vulkan_setup.h"

#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>
#include <print>
#include <span>
#include <vector>

namespace {

// --- Frames in flight --------------------------------------------------------

// How many frames the CPU may record ahead of the GPU.
constexpr std::size_t frames_in_flight = 2;

constexpr std::uint64_t no_timeout = std::numeric_limits<std::uint64_t>::max();

// What each in-flight frame needs for itself.
struct Frame {
    vk::raii::CommandBuffer commands = nullptr;
    vk::raii::Semaphore image_acquired = nullptr;  // swapchain image is ready to draw into
    vk::raii::Fence done = nullptr;                // GPU finished this frame's commands
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

// What to draw: one mesh, placed in the world by each of `models`.
struct DrawList {
    vk::DeviceAddress vertices = 0;
    std::uint32_t vertex_count = 0;
    glm::mat4 view_projection{1.0f};
    std::span<const glm::mat4> models;
};

// Records: swapchain image -> clear color and depth -> draw everything in
// `draws` with `pipeline` -> ready to present.
void record_frame(
    const vk::raii::CommandBuffer& commands,
    const Swapchain& swapchain,
    std::uint32_t image_index,
    std::array<float, 4> color,
    const vk::raii::Pipeline& pipeline,
    const DrawList& draws
) {
    const vk::Image image = swapchain.images[image_index];

    commands.reset();
    commands.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Undefined: we don't care what was in the image, we're about to clear it.
    transition(commands, image,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite
    );

    // The depth buffer is shared by the frames in flight, so this also waits
    // for the previous frame's depth tests before this frame clears it.
    transition(commands, *swapchain.depth.handle,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal,
        vk::PipelineStageFlagBits2::eLateFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::ImageAspectFlagBits::eDepth
    );

    // loadOp eClear does the clearing when rendering begins.
    const vk::RenderingAttachmentInfo color_attachment{
        .imageView = *swapchain.views[image_index],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = color}},
    };

    // Reverse-Z: 0 is the far plane. Depth is only needed while drawing this
    // frame, so it isn't stored afterwards.
    const vk::RenderingAttachmentInfo depth_attachment{
        .imageView = *swapchain.depth.view,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = 0.0f}},
    };

    commands.beginRendering(vk::RenderingInfo{
        .renderArea = {.offset = {0, 0}, .extent = swapchain.extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
        .pDepthAttachment = &depth_attachment,
    });

    commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

    // The pipeline left these dynamic; they cover the whole image.
    commands.setViewport(0, vk::Viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(swapchain.extent.width),
        .height = static_cast<float>(swapchain.extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    });
    commands.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = swapchain.extent});

    // One draw per object: the same vertices, a different model matrix. Push
    // data is copied into the command buffer, so each draw sees its own values.
    for (const glm::mat4& model : draws.models) {
        const PushData push{
            .view_projection = draws.view_projection,
            .model = model,
            .vertices = draws.vertices,
        };

        commands.pushDataEXT(vk::PushDataInfoEXT{
            .offset = 0,
            .data = {.address = &push, .size = sizeof(push)},
        });

        commands.draw(draws.vertex_count, 1, 0, 0);
    }

    commands.endRendering();

    transition(commands, image,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone
    );

    commands.end();
}

// --- Geometry ----------------------------------------------------------------

// A unit cube centred on the origin: 6 faces x 2 triangles x 3 vertices, each
// face one color. Corners go counter-clockwise as seen from outside the cube,
// which Chapter 7's back-face culling will rely on.
std::array<Vertex, 36> make_cube() {
    struct Face {
        std::array<glm::vec3, 4> corners;
        glm::vec3 color;
    };

    constexpr float h = 0.5f;
    const std::array<Face, 6> faces{{
        {{{{h, -h, h}, {h, -h, -h}, {h, h, -h}, {h, h, h}}}, {1.0f, 0.2f, 0.2f}},        // +X red
        {{{{-h, -h, -h}, {-h, -h, h}, {-h, h, h}, {-h, h, -h}}}, {0.2f, 1.0f, 1.0f}},    // -X cyan
        {{{{-h, h, h}, {h, h, h}, {h, h, -h}, {-h, h, -h}}}, {0.2f, 1.0f, 0.2f}},        // +Y green
        {{{{-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {-h, -h, h}}}, {1.0f, 0.2f, 1.0f}},    // -Y magenta
        {{{{-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}}}, {0.2f, 0.2f, 1.0f}},        // +Z blue
        {{{{h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h}}}, {1.0f, 1.0f, 0.2f}},    // -Z yellow
    }};

    std::array<Vertex, 36> vertices{};
    std::size_t next = 0;

    for (const Face& face : faces) {
        // Two triangles per face: corners 0-1-2 and 0-2-3.
        for (const std::size_t corner : {0, 1, 2, 0, 2, 3}) {
            vertices[next++] = Vertex{.position = face.corners[corner], .color = face.color};
        }
    }

    return vertices;
}

// --- Events ------------------------------------------------------------------

// Handles every pending event and fills in `input` for this frame. False once
// the window was closed or Escape pressed.
bool poll_events(SDL_Window* window, CameraInput& input) {
    input = CameraInput{};
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        const bool escape = event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE;

        if (event.type == SDL_EVENT_QUIT || escape) {
            return false;
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

        // Built for swapchain.format and depth_format. recreate_swapchain() picks
        // the same formats again, so the pipeline stays valid across resizes.
        vk::raii::Pipeline pipeline = create_mesh_pipeline(device, swapchain.format, depth_format);

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
            frames.push_back(Frame{
                .commands = std::move(commands),
                .image_acquired = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
                // Start signalled, so the first wait on each frame returns straight away.
                .done = vk::raii::Fence(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}),
            });
        }

        // --- Geometry --------------------------------------------------------

        // eShaderDeviceAddress: the shader reads it through a pointer, so it
        // needs a GPU address and no other usage.
        const std::array<Vertex, 36> cube = make_cube();
        const Buffer cube_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(cube)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        // --- Scene -----------------------------------------------------------

        // Spawns at the origin, looking down -Z.
        FlyCamera camera;
        CameraInput input;

        // A 3x3 grid of spinning cubes in front of the camera, close enough to
        // cut into each other. Where they intersect only looks right with a
        // depth buffer.
        constexpr int grid = 3;
        constexpr float spacing = 1.0f;
        std::vector<glm::mat4> models(grid * grid);
        std::uint64_t previous_ticks = SDL_GetTicksNS();

        // --- Frame loop ------------------------------------------------------

        const std::array black{0.0f, 0.0f, 0.0f, 1.0f};
        std::uint64_t frame_count = 0;

        while (poll_events(window.get(), input)) {
            // Minimised: nothing to draw into, so sleep until something happens.
            int width = 0;
            int height = 0;
            SDL_GetWindowSizeInPixels(window.get(), &width, &height);

            if (width == 0 || height == 0 || (SDL_GetWindowFlags(window.get()) & SDL_WINDOW_MINIMIZED)) {
                SDL_WaitEvent(nullptr);
                continue;
            }

            if (width != swapchain.window_width || height != swapchain.window_height) {
                recreate_swapchain(swapchain, device, *gpu, surface, window.get());
            }

            // --- Update -----------------------------------------------------

            // Seconds since the last frame, so movement doesn't depend on frame rate.
            const std::uint64_t ticks = SDL_GetTicksNS();
            const float seconds = static_cast<float>(ticks - previous_ticks) * 1e-9f;
            const float time = static_cast<float>(ticks) * 1e-9f;
            previous_ticks = ticks;

            update_camera(camera, input, seconds);

            // Each cube tumbles about its own mostly-horizontal axis, at its own speed.
            for (int z = 0; z < grid; ++z) {
                for (int x = 0; x < grid; ++x) {
                    const int i = z * grid + x;
                    const glm::vec3 offset{(x - 1) * spacing, -1.0f, -6.0f + (z - 1) * spacing};
                    const glm::vec3 axis = glm::normalize(glm::vec3{std::cos(1.7f * i), 0.5f, std::sin(1.7f * i)});
                    const float angle = time * (0.6f + 0.1f * i);

                    models[i] = glm::rotate(glm::translate(glm::mat4{1.0f}, offset), angle, axis);
                }
            }

            const float aspect = static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height);

            const DrawList draws{
                .vertices = cube_buffer.address,
                .vertex_count = static_cast<std::uint32_t>(cube.size()),
                .view_projection = camera.projection(aspect) * camera.view(),
                .models = models,
            };

            // --- Render -----------------------------------------------------

            Frame& frame = frames[frame_count % frames_in_flight];

            // 1. Wait until the GPU is done with this frame's command buffer from last time.
            (void)device.waitForFences(*frame.done, vk::True, no_timeout);

            // 2. Ask the swapchain for an image; `image_acquired` is signalled once it's free.
            const auto acquired = swapchain.handle.acquireNextImage(no_timeout, *frame.image_acquired);

            if (acquired.result == vk::Result::eErrorOutOfDateKHR) {
                recreate_swapchain(swapchain, device, *gpu, surface, window.get());
                continue;
            }

            const std::uint32_t image_index = acquired.value;
            device.resetFences(*frame.done);

            // 3. Record and submit: wait for the image, draw, signal `rendered` and `done`.
            record_frame(frame.commands, swapchain, image_index, black, pipeline, draws);

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
                recreate_swapchain(swapchain, device, *gpu, surface, window.get());
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

## 3.7 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The scene:** nine cubes tumbling in a 3×3 grid, every face its own color. Where two cubes pass through each other, the intersection is drawn correctly; without the depth buffer, whichever cube was drawn last would cover the rest.
- **Moving:** hold the right mouse button to look around and fly with WASD, Q and E; scroll while holding it to change speed. Left-drag, middle-drag and the scroll wheel work as in the table in section 3.5.
- **Resizing:** resize the window; the cubes keep their shape, because the projection uses the current aspect ratio, and the depth buffer is rebuilt with the swapchain.
- **No `[validation …]` lines.**

Next, in Chapter 4, we replace the cube with real geometry: we load glTF files with tinygltf, upload their vertices and indices, and walk the scene's node tree to place every mesh. Sponza becomes the first real scene.

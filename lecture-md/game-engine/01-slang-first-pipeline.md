# Chapter 1: Slang shaders and the first pipeline

By the end of this chapter the window shows a triangle with red, green and blue corners. It's drawn by a shader we compile at build time and a pipeline with **no pipeline layout**, the first visible piece of the descriptor-heap model.

This chapter builds on [Chapter 0](00-project-window-device.md), and needs a recent Slang compiler on your `PATH` (check with `slangc -v`; this was tested with 2026.19).

## 1.1 The shader: `shaders/triangle.slang`

### Why
Everything the GPU draws runs through two programs we write:
- the **vertex shader** runs once per vertex and decides where that vertex lands on screen,
- the **fragment shader** runs once per covered pixel and decides its color.

The GPU doesn't run source code. It runs **SPIR-V**, a compact binary format that Vulkan drivers turn into machine code. We write shaders in **Slang**, a modern shading language close to HLSL with modules and generics, and `slangc` compiles them to SPIR-V.

### How
- **The corners are hardcoded** in the shader as constant arrays, so nothing has to be uploaded to the GPU yet. That comes in Chapter 2. The vertex shader receives its vertex number (0, 1, 2) and uses it to pick a corner.
- **Semantics:** the `: NAME` after a parameter or field is a *semantic*. Names starting with `SV_` are values the GPU itself provides or consumes. `SV_Position` is where the vertex lands on screen. `COLOR` is just a label that matches the vertex output to the fragment input, which the GPU interpolates across the triangle.
- **`SV_VulkanVertexID`, not `SV_VertexID`.** Slang implements HLSL's `SV_VertexID` as "vertex index minus the draw's base vertex". Reading the base vertex needs an extra capability (`DrawParameters`) that our device doesn't enable, and validation reports an error. `SV_VulkanVertexID` is Vulkan's own vertex index, with no subtraction and no extra capability.
- **One file, two entry points:** `[shader("vertex")]` and `[shader("fragment")]` mark the entry points, and both end up in a single SPIR-V module.
- **Clip space:** Vulkan's x and y run from −1 to 1, and **+y points down**. So y = −0.5 is near the top.
- **Color space:** the swapchain is sRGB, so the GPU encodes the linear colors we return.

### Code
Create the directory:
```bash
mkdir -p game-engine/shaders
```

`game-engine/shaders/triangle.slang`:
```slang
// A triangle whose corners are baked into the shader, so nothing has to be
// uploaded yet: the vertex index (0, 1, 2) picks the corner.

// --- Data --------------------------------------------------------------------

// Vulkan clip space: x and y run from -1 to 1, with +y pointing down.
static const float2 positions[3] = {
    float2(0.0, -0.5),   // top
    float2(0.5, 0.5),    // bottom right
    float2(-0.5, 0.5),   // bottom left
};

static const float3 colors[3] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0),
};

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
    VertexOutput output;
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.color = colors[vertex_id];
    return output;
}

// --- Fragment shader ---------------------------------------------------------

// SV_Target: the value written to color attachment 0.
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    return float4(input.color, 1.0);
}
```

## 1.2 Compiling shaders in the build: `CMakeLists.txt`

### Why
Shaders are code, so they should be compiled by the build like the C++ is. When we edit a shader, rebuilding should recompile that shader and nothing else. The program also needs to know where the `.spv` files end up.

### How
- **One rule per shader:** for each `shaders/*.slang` file, `add_custom_command` declares "this `.spv` is produced by running slangc on this `.slang`". The rule only reruns when the shader changes.
- **Hooking into the program's build:** `add_custom_target(shaders)` collects the `.spv` files into something buildable, and `add_dependencies` makes sure they're built before the program.
- **Finding the files at runtime:** `SHADER_DIR` passes the output directory to the C++ code as a string.
- **slangc flags:**
  - `-target spirv -profile spirv_1_6`: Vulkan 1.3 and later accept SPIR-V 1.6.
  - `-fvk-use-entrypoint-name`: keeps our function names as the SPIR-V entry-point names. Without it, a file with a *single* entry point gets renamed to `main`. With it, the C++ side can always ask for `"vertexMain"`.
  - `-g`: keeps names and source in the SPIR-V for graphics debuggers like RenderDoc.

### Code
In `game-engine/CMakeLists.txt`, add this section directly after `target_link_libraries(game-engine PRIVATE SDL3::SDL3 Vulkan::Headers)`:
```cmake
# --- shaders/ ----------------------------------------------------------------
# Each shaders/<name>.slang becomes <build>/shaders/<name>.spv at build time.
# Only an edited shader is recompiled, and the program finds them in SHADER_DIR.

find_program(SLANGC slangc REQUIRED)

set(shader_dir ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${shader_dir})

file(GLOB shader_sources CONFIGURE_DEPENDS shaders/*.slang)
set(spirv_files "")

foreach(source IN LISTS shader_sources)
    get_filename_component(shader_name ${source} NAME_WE)
    set(spirv ${shader_dir}/${shader_name}.spv)

    # -fvk-use-entrypoint-name: keep vertexMain/fragmentMain as the SPIR-V
    #   entry point names (a file with a single entry point would get "main").
    # -g: keep names and source in the SPIR-V for debuggers like RenderDoc.
    add_custom_command(
        OUTPUT ${spirv}
        COMMAND ${SLANGC} ${source} -target spirv -profile spirv_1_6 -fvk-use-entrypoint-name -g -o ${spirv}
        DEPENDS ${source}
        COMMENT "Compiling ${shader_name}.slang"
        VERBATIM
    )

    list(APPEND spirv_files ${spirv})
endforeach()

add_custom_target(shaders DEPENDS ${spirv_files})
add_dependencies(game-engine shaders)
target_compile_definitions(game-engine PRIVATE SHADER_DIR="${shader_dir}")
```

## 1.3 The pipeline: `pipeline.h` / `pipeline.cpp`

### Why
A **graphics pipeline** bakes everything about *how* to draw into one object: which shaders run, what vertex data looks like, how triangles are filled, how colors are written, and what kind of image they're written into. Building it is expensive, because the driver compiles the SPIR-V into GPU machine code here. So we build it once at startup and bind it every frame.

Normally a pipeline also needs a `VkPipelineLayout`, which declares the descriptor sets and push constants the shaders use. With the descriptor heap there is no such layout. Shaders find resources in the heap and in *push data*. We tell Vulkan that by creating the pipeline in **descriptor heap mode**, which is what makes `layout = nullptr` legal. Without that flag, validation reports "layout is not a valid VkPipelineLayout".

### How
- **`read_spirv`:** loads a `.spv` file into a `std::vector<std::uint32_t>`. SPIR-V is a stream of 32-bit words, and a `uint32_t` vector guarantees the 4-byte alignment Vulkan requires.
- **The shader module:** wraps the SPIR-V. It's only needed while the pipeline is being built, so it's a local that's destroyed on return.
- **The pipeline's state, one struct per part:**
  - **stages:** one module, two entry points picked by name,
  - **vertex input:** empty, because there are no vertex buffers,
  - **input assembly:** every 3 vertices form a triangle,
  - **viewport and scissor:** declared as *dynamic*. The pipeline only records how many there are, and the actual rectangles are set while recording, so a resized window never needs a new pipeline,
  - **rasterization:** filled triangles, both sides drawn for now,
  - **multisampling:** off,
  - **color blending:** off, so the fragment shader's color replaces the pixel,
  - **`PipelineRenderingCreateInfo`:** dynamic rendering's replacement for a render pass. It names the format of the image we'll draw into,
  - **`PipelineCreateFlags2CreateInfo`:** carries the descriptor heap flag.
- **The `pNext` chain** runs create info → flags → rendering. Every struct it points at is a local that lives until the function returns, which is after the pipeline is created.
- **Field order:** designated initializers must follow the order of `GraphicsPipelineCreateInfo`'s fields. Anything we skip (tessellation, depth/stencil) is zero. Depth comes in Chapter 3.
- **The `nullptr` before the create info** is the *pipeline cache*, which we don't use.

### Code
`game-engine/src/includes/pipeline.h`:
```cpp
#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <filesystem>
#include <vector>

// A .spv file as the 32-bit words SPIR-V is made of.
std::vector<std::uint32_t> read_spirv(const std::filesystem::path &path);

// Draws shaders/triangle.slang into a `color_format` image. There is no
// pipeline layout: shaders will find their resources in the descriptor heap.
vk::raii::Pipeline create_triangle_pipeline(const vk::raii::Device &device, vk::Format color_format);
```

`game-engine/src/pipeline.cpp`:
```cpp
#include "includes/pipeline.h"

#include <array>
#include <fstream>
#include <stdexcept>

// --- Loading SPIR-V ----------------------------------------------------------

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

// --- The triangle pipeline ---------------------------------------------------

vk::raii::Pipeline create_triangle_pipeline(const vk::raii::Device &device, vk::Format color_format) {
    // Shaders: one module, two entry points picked by name. The module is
    // only needed while the pipeline is built, so it's destroyed on return.
    const std::vector<std::uint32_t> spirv = read_spirv(std::filesystem::path(SHADER_DIR) / "triangle.spv");

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

    // Vertex input and assembly: no vertex buffers, the vertex shader makes
    // its corners from SV_VulkanVertexID. Every 3 vertices form a triangle.
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
    // format of the image it will draw into.
    const vk::PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &color_format,
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
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic,
        .layout = nullptr,
    });
}
```

## 1.4 Drawing: `main.cpp`

### Why
The pipeline exists, but nothing uses it yet. During rendering, the frame has to bind it, fill in the two pieces of state we left dynamic, and issue the draw.

### How
- **Recording order:** binding a pipeline only *records* a command, like everything else in a command buffer. Inside `beginRendering`/`endRendering`, the order is: bind the pipeline, set the viewport and scissor to cover the whole image, then `draw(3, 1, 0, 0)`, which means 3 vertices, 1 instance, starting at vertex 0 and instance 0.
- **When the pipeline is created:** after the swapchain, because the pipeline needs `swapchain.format`. `recreate_swapchain` chooses the same format every time, so the pipeline stays valid across resizes.
- **Destruction:** the pipeline is declared after the device, so it's destroyed before it. The final `device.waitIdle()` makes sure the GPU has finished using it.

### Code
Four edits to `game-engine/src/main.cpp`.

**1.** Add the include at the top, keeping the includes in alphabetical order:
```cpp
#include "includes/pipeline.h"
#include "includes/sdl.h"
#include "includes/swapchain.h"
#include "includes/vulkan_setup.h"
```

**2.** Replace `record_frame` with this complete version. It takes the pipeline and draws where the "Draw calls go here" comment was:
```cpp
// Records: swapchain image -> clear to `color` -> draw with `pipeline` -> ready to present.
void record_frame(
    const vk::raii::CommandBuffer &commands,
    const Swapchain &swapchain,
    std::uint32_t image_index,
    std::array<float, 4> color,
    const vk::raii::Pipeline &pipeline
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

    // loadOp eClear does the clearing when rendering begins.
    const vk::RenderingAttachmentInfo color_attachment{
        .imageView = *swapchain.views[image_index],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = color}},
    };

    commands.beginRendering(vk::RenderingInfo{
        .renderArea = {.offset = {0, 0}, .extent = swapchain.extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
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

    // 3 vertices, 1 instance, starting at vertex 0 and instance 0.
    commands.draw(3, 1, 0, 0);

    commands.endRendering();

    transition(commands, image,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone
    );

    commands.end();
}
```

**3.** In `main`, directly after `Swapchain swapchain = create_swapchain(...);`, add a new section:
```cpp
        // --- Pipelines -------------------------------------------------------

        // Built for swapchain.format. recreate_swapchain() picks the same
        // format again, so the pipeline stays valid across resizes.
        vk::raii::Pipeline pipeline = create_triangle_pipeline(device, swapchain.format);
```

**4.** In the frame loop, pass the pipeline:
```cpp
            record_frame(frame.commands, swapchain, image_index, black, pipeline);
```

## 1.5 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **Shader compile:** the build log includes `Compiling triangle.slang`.
- **The triangle:** on black, red at the top, green bottom-right, blue bottom-left, blended in between.
- **Validation:** no `[validation …]` lines, including after resizing and minimizing the window.
- **Incremental rebuilds:** edit only `triangle.slang` (change a color, say) and build again. Only `Compiling triangle.slang` runs, with no C++ rebuild.

**Looking ahead:** in Chapter 6, shaders index the descriptor heap directly, and slangc needs `-capability spvDescriptorHeapEXT` to emit that SPIR-V. Older compilers, such as Ubuntu's `slang-compiler` 2026.1.1 package, don't support it, which is why this project needs a recent Slang.

The triangle's corners still live inside the shader. In Chapter 2 we'll move them into GPU memory. That means allocating a buffer ourselves, getting its 64-bit device address, and passing that address to the shader with `vkCmdPushDataEXT`, the first piece of the descriptor-heap API.

# Chapter 2: GPU buffers, device addresses and push data

The triangle on screen at the end of this chapter looks exactly like Chapter 1's. What changes is where it comes from: its corners now live in a buffer in GPU memory that we allocate and fill ourselves. The shader reads them through a 64-bit **device address**, which we hand it with `vkCmdPushDataEXT`. Every later chapter uses these three pieces: meshes, materials and the scene are all buffers read through addresses, and push data tells each draw where to look.

This chapter builds on [Chapter 1](01-slang-first-pipeline.md).

## 2.1 Data shared by C++ and the shader: `shader_types.h`

### Why
The GPU reads our buffers byte for byte. If the C++ side writes a vertex as 24 bytes and the shader expects 32, every vertex after the first is read from the wrong place, and nothing reports an error. Both sides have to agree on the size of every struct and the offset of every field.

### How
- **Slang's layout for pointer data:** a `Vertex` that Slang reads through a pointer is laid out like a C struct. A `float3` is 12 bytes with no padding, so `position` is at byte 0, `color` at byte 12, and one vertex takes 24 bytes. The C++ struct uses `std::array<float, 3>` for each field, which has exactly that layout.
- **Why Chapter 0 enabled `scalarBlockLayout`:** Vulkan's default rules for buffer data don't allow a 3-component vector to start at byte 12. Without the feature, validation rejects the shader:

  > member 1 is an improperly straddling vector at offset 12. This is may be allowed if you enable the scalarBlockLayout feature

  With it, buffer data may use exactly this C-like layout.
- **`static_assert` checks the layout at compile time.** If someone later adds or reorders a field on one side only, the build fails instead of the picture silently breaking.
- **`PushData`** holds one `vk::DeviceAddress`, a 64-bit number, matching the shader's `Vertex*` pointer.

### Code
`game-engine/src/includes/shader_types.h`:
```cpp
#pragma once

#include <vulkan/vulkan.hpp>

#include <array>
#include <cstddef>

// C++ mirrors of the structs in shaders/triangle.slang. The GPU reads these
// bytes as they are, so the two sides must agree on every size and offset;
// the static_asserts catch a mismatch at compile time.

// --- Vertex ------------------------------------------------------------------

// Slang lays out data behind a pointer like C: a float3 is 12 bytes, with no
// padding, so position is at byte 0, color at byte 12, and a vertex is 24.
struct Vertex {
    std::array<float, 3> position;
    std::array<float, 3> color;
};

static_assert(sizeof(Vertex) == 24);
static_assert(offsetof(Vertex, color) == 12);

// --- Push data ---------------------------------------------------------------

// Written with vkCmdPushDataEXT before each draw.
struct PushData {
    vk::DeviceAddress vertices;  // where the first Vertex is in GPU memory
};

static_assert(sizeof(PushData) == 8);
```

## 2.2 GPU memory: `buffer.h` / `buffer.cpp`

### Why
A `VkBuffer` is only a description: a size and a list of uses. It owns no memory. We have to allocate memory ourselves, choose what *kind* of memory it is, and bind the two together.

The kind matters. A GPU offers a handful of **memory types**, each a combination of properties:
- **device-local:** on the GPU itself, the fastest memory for the GPU to read, which the CPU usually can't write directly,
- **host-visible:** memory the CPU can map and write into,
- **host-coherent:** CPU writes become visible to the GPU without an explicit flush.

So getting data into fast GPU memory takes two steps. We write it into a host-visible **staging buffer**, then have the GPU copy it into a device-local buffer.

Finally, a shader can't just "use a buffer". It needs some way to find it. Instead of a descriptor, we give the shader the buffer's **device address**: its location in the GPU's address space, a plain 64-bit number that the shader treats as a pointer.

### How
- **`find_memory_type`:**
  - A buffer's memory requirements include `memoryTypeBits`, a bitmask of the memory types it may live in.
  - We walk the GPU's memory types and return the first one that is both allowed and has every property we asked for.
- **`create_buffer`** does four things, numbered in the code:
  1. creates the buffer object,
  2. allocates memory of a suitable type,
  3. binds the memory to the buffer,
  4. reads the buffer's device address.

  A buffer with a device address needs its memory allocated with `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` too. Leaving `VkMemoryAllocateFlagsInfo` out of the chain makes validation report "buffer was created with … SHADER_DEVICE_ADDRESS_BIT, but the memory was not allocated with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT".
- **`submit_and_wait`:** records commands through a callback, submits them, and waits on a fence for exactly that submission. It's for one-off work at load time, not for every frame.
- **`upload_buffer`:**
  - makes a staging buffer, maps it, copies the bytes in with `memcpy`, and unmaps it,
  - creates the device-local buffer,
  - records a `copyBuffer` from one to the other and waits.

  Once the copy has finished, the staging buffer is destroyed when the function returns.
- **The `Buffer` struct** declares `memory` before `handle`. Members are destroyed in reverse order, so the buffer goes before the memory it's bound to.
- **One allocation per buffer** is fine for now. A real engine suballocates from a few large allocations; GPUs limit how many allocations you can make, often to about 4096. We'll only ever make a few dozen.

### Code
`game-engine/src/includes/buffer.h`:
```cpp
#pragma once

#include "includes/vulkan_setup.h"

#include <cstddef>
#include <functional>
#include <span>

// A VkBuffer, the memory behind it, and its GPU address. Members are destroyed
// bottom-up, so the buffer goes before the memory it's bound to.
struct Buffer {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Buffer handle = nullptr;
    vk::DeviceSize size = 0;
    vk::DeviceAddress address = 0;  // 0 unless created with eShaderDeviceAddress
};

// --- Creating buffers --------------------------------------------------------

// A buffer of `size` bytes in memory with `properties`. With eShaderDeviceAddress
// in `usage`, `address` is filled in, so shaders can read it through a pointer.
Buffer create_buffer(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties
);

// --- Uploading ---------------------------------------------------------------

// Records commands with `record`, submits them to `queue`, and waits until
// the GPU has finished. For one-off work like uploads, not for every frame.
void submit_and_wait(
    const vk::raii::Device& device,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    const std::function<void(const vk::raii::CommandBuffer&)>& record
);

// A device-local buffer holding a copy of `bytes`, which the GPU reads at
// full speed and shaders can reach through `address`.
Buffer upload_buffer(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    std::span<const std::byte> bytes,
    vk::BufferUsageFlags usage
);
```

`game-engine/src/buffer.cpp`:
```cpp
#include "includes/buffer.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

// --- Memory types ------------------------------------------------------------

// The GPU offers a few memory types, each a set of properties (device-local,
// host-visible, ...) in one of its heaps. `allowed` is the bitmask a buffer's
// memory requirements permit; we take the first allowed type with `required`.
std::uint32_t find_memory_type(const GpuChoice& gpu, std::uint32_t allowed, vk::MemoryPropertyFlags required) {
    const vk::PhysicalDeviceMemoryProperties memory = gpu.device.getMemoryProperties();

    for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        const bool is_allowed = (allowed & (1u << i)) != 0;

        if (is_allowed && (memory.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }

    throw std::runtime_error("no memory type has " + vk::to_string(required));
}

}  // namespace

// --- Creating buffers --------------------------------------------------------

Buffer create_buffer(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties
) {
    Buffer buffer;
    buffer.size = size;

    // 1. The buffer object: a size and what it will be used for, but no memory yet.
    buffer.handle = vk::raii::Buffer(device, vk::BufferCreateInfo{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    });

    // 2. Memory that fits it. A buffer that has a device address needs memory
    //    allocated with the device-address flag too.
    const vk::MemoryRequirements requirements = buffer.handle.getMemoryRequirements();
    const bool has_address = static_cast<bool>(usage & vk::BufferUsageFlagBits::eShaderDeviceAddress);

    const vk::MemoryAllocateFlagsInfo allocate_flags{
        .flags = has_address ? vk::MemoryAllocateFlagBits::eDeviceAddress : vk::MemoryAllocateFlags{},
    };

    buffer.memory = vk::raii::DeviceMemory(device, vk::MemoryAllocateInfo{
        .pNext = &allocate_flags,
        .allocationSize = requirements.size,
        .memoryTypeIndex = find_memory_type(gpu, requirements.memoryTypeBits, properties),
    });

    // 3. Bind them together, at offset 0 of the allocation.
    buffer.handle.bindMemory(*buffer.memory, 0);

    // 4. The 64-bit GPU address shaders use to read it.
    if (has_address) {
        buffer.address = device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = *buffer.handle});
    }

    return buffer;
}

// --- Uploading ---------------------------------------------------------------

void submit_and_wait(
    const vk::raii::Device& device,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    const std::function<void(const vk::raii::CommandBuffer&)>& record
) {
    vk::raii::CommandBuffers commands(device, vk::CommandBufferAllocateInfo{
        .commandPool = *pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    });

    commands[0].begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    record(commands[0]);
    commands[0].end();

    // A fence lets us wait for exactly this submission, rather than the whole queue.
    const vk::raii::Fence done(device, vk::FenceCreateInfo{});
    const vk::CommandBufferSubmitInfo command_info{.commandBuffer = *commands[0]};

    queue.submit2(vk::SubmitInfo2{
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &command_info,
    }, *done);

    (void)device.waitForFences(*done, vk::True, std::numeric_limits<std::uint64_t>::max());
}

Buffer upload_buffer(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    std::span<const std::byte> bytes,
    vk::BufferUsageFlags usage
) {
    // The CPU can't write device-local memory directly, so the bytes go into a
    // host-visible "staging" buffer first. Host-coherent means our writes are
    // visible to the GPU without an explicit flush.
    const Buffer staging = create_buffer(device, gpu, bytes.size(),
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void* mapped = staging.memory.mapMemory(0, bytes.size());
    std::memcpy(mapped, bytes.data(), bytes.size());
    staging.memory.unmapMemory();

    // The real buffer lives in device-local memory and is filled by a GPU copy.
    Buffer buffer = create_buffer(device, gpu, bytes.size(),
        usage | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    submit_and_wait(device, queue, pool, [&](const vk::raii::CommandBuffer& commands) {
        commands.copyBuffer(*staging.handle, *buffer.handle, vk::BufferCopy{.size = bytes.size()});
    });

    // Waiting above means the copy is done, so `staging` can be destroyed on return.
    return buffer;
}
```

## 2.3 Reading through a pointer: `shaders/triangle.slang`

### Why
The classic way to feed vertices to a vertex shader is to describe their layout in the pipeline's *vertex input state*, and let fixed-function hardware fetch them. The modern alternative is **vertex pulling**: the shader receives a pointer and reads its vertex itself. The pipeline stays simple (no vertex input state at all), and any vertex format works. Later, the same technique reads materials and the whole scene description.

### How
- **The structs mirror `shader_types.h`:** `PushData` holds a `Vertex*`, which is Slang's syntax for a pointer to GPU memory.
- **Where push data lands:** `[[vk::push_constant]] ConstantBuffer<PushData> push;` declares the small block that push data is written into. In a pipeline created in descriptor heap mode, `vkCmdPushDataEXT` fills this block.
- **Reading a vertex:** `push.vertices[vertex_id]` indexes the pointer like an array. The GPU loads 24 bytes from `address + vertex_id × 24`.

### Code
`game-engine/shaders/triangle.slang`:
```slang
// A triangle whose corners live in a GPU buffer. The C++ side uploads them
// and pushes the buffer's address; the vertex shader reads its corner through
// that pointer.

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

// Data behind a pointer is laid out like C: a float3 is 12 bytes and nothing
// is padded, so this matches the C++ Vertex exactly (24 bytes).
struct Vertex {
    float3 position;
    float3 color;
};

// Written with vkCmdPushDataEXT before each draw.
struct PushData {
    Vertex* vertices;  // device address of the first vertex
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
    // "Vertex pulling": the shader indexes the buffer itself, so the pipeline
    // declares no vertex attributes.
    const Vertex vertex = push.vertices[vertex_id];

    VertexOutput output;
    output.position = float4(vertex.position, 1.0);
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

## 2.4 The pipeline: one comment

The pipeline itself doesn't change: it still declares no vertex input, which is exactly what vertex pulling needs. Only the comment above `vertex_input` in `game-engine/src/pipeline.cpp` is now out of date. Update it:
```cpp
    // Vertex input and assembly: no vertex attributes, the vertex shader reads
    // its vertex through a pointer. Every 3 vertices form a triangle.
    const vk::PipelineVertexInputStateCreateInfo vertex_input{};
```

## 2.5 Uploading and pushing: `main.cpp`

### Why
`main` creates the vertex buffer once, after it has a queue and a command pool to upload with. Each frame then tells the shader where the buffer is.

### How
- **The triangle data** is a `constexpr std::array<Vertex, 3>`: the same corners and colors Chapter 1 had in the shader.
  - `std::as_bytes(std::span(triangle))` views the array as raw bytes, which is what `upload_buffer` copies.
  - The buffer's only usage is `eShaderDeviceAddress`, because the shader reads it through a pointer and nothing else touches it.
- **When it's created:** after the command pool, because the upload records a command buffer from that pool. It's declared after the device, so it's destroyed before the device. `device.waitIdle()` at the end of `main` makes sure the GPU has stopped reading it first.
- **Push data is recorded into the command buffer.** `pushDataEXT` takes an offset and a `{pointer, size}` range of bytes. It copies the 8 bytes of `PushData` into the command buffer at recording time, so `push` can be a local variable. Each draw can push different values, which is how Chapter 4 will draw many meshes with one pipeline.

### Code
Four edits to `game-engine/src/main.cpp`.

**1.** Replace the includes at the top. The new lines are `buffer.h`, `shader_types.h` and `<span>`:
```cpp
#include "includes/buffer.h"
#include "includes/pipeline.h"
#include "includes/sdl.h"
#include "includes/shader_types.h"
#include "includes/swapchain.h"
#include "includes/vulkan_setup.h"

#include <array>
#include <cstdlib>
#include <exception>
#include <limits>
#include <print>
#include <span>
#include <vector>
```

**2.** Replace `record_frame` with this version. It takes the vertex buffer's address and pushes it before the draw:
```cpp
// Records: swapchain image -> clear to `color` -> draw the triangle at
// `vertices` with `pipeline` -> ready to present.
void record_frame(
    const vk::raii::CommandBuffer& commands,
    const Swapchain& swapchain,
    std::uint32_t image_index,
    std::array<float, 4> color,
    const vk::raii::Pipeline& pipeline,
    vk::DeviceAddress vertices
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

    // Hand the shader the vertex buffer's address. Push data is recorded into
    // the command buffer, so each draw can push different values.
    const PushData push{.vertices = vertices};

    commands.pushDataEXT(vk::PushDataInfoEXT{
        .offset = 0,
        .data = {.address = &push, .size = sizeof(push)},
    });

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

**3.** In `main`, directly before the `// --- Frame loop` section, add a new section:
```cpp
        // --- Geometry --------------------------------------------------------

        // Vulkan clip space: x and y run from -1 to 1, with +y pointing down.
        constexpr std::array<Vertex, 3> triangle{{
            {.position = {0.0f, -0.5f, 0.0f}, .color = {1.0f, 0.0f, 0.0f}},   // top
            {.position = {0.5f, 0.5f, 0.0f}, .color = {0.0f, 1.0f, 0.0f}},    // bottom right
            {.position = {-0.5f, 0.5f, 0.0f}, .color = {0.0f, 0.0f, 1.0f}},   // bottom left
        }};

        // eShaderDeviceAddress: the shader reads it through a pointer, so it
        // needs a GPU address and no other usage.
        const Buffer vertex_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(triangle)), vk::BufferUsageFlagBits::eShaderDeviceAddress);
```

**4.** In the frame loop, pass the address:
```cpp
            record_frame(frame.commands, swapchain, image_index, black, pipeline, vertex_buffer.address);
```

## 2.6 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The same triangle as Chapter 1:** red at the top, green bottom-right, blue bottom-left. The shader no longer contains any positions or colors, so everything on screen came out of the buffer.
- **No `[validation …]` lines.**
- **Try it:** change a color in the `triangle` array in `main.cpp` and build again. Only the C++ is recompiled, not the shader, because the data is no longer part of the shader.

We can now put arbitrary data in GPU memory and point shaders at it. In Chapter 3 we'll add a camera and a depth buffer, so the shader can draw in 3D.

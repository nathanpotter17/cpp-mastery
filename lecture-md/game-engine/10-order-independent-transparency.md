# Chapter 10: Order-independent transparency

By the end of this chapter, see-through surfaces are drawn correctly whatever order they overlap in. Since Chapter 6, blended draws have been sorted by the distance to their centers, farthest first, and blended over the scene one after another. That's right for separate small objects, and wrong whenever a center doesn't say which surface is in front:
- **Intersecting surfaces:** a glass pane through another, or water meeting glass. Each is partly in front of the other.
- **One mesh in front of and behind itself:** a double-sided bottle's back and front faces are one draw, blended in whatever order its triangles happen to be in.
- **Large and small together:** a window's center can be farther away than a small cup behind it.

Sorting per pixel would fix it, at a cost that grows with the overlap. Sorting per triangle fixes most cases, but not triangles that cut through each other. This chapter takes the other road: **weighted blended order-independent transparency** (WBOIT, McGuire and Bavoil 2013). Every see-through fragment adds into two sums, in any order, and one full-screen pass turns the sums into the final color. Nothing is sorted. A single layer comes out as blending would draw it, to 16-bit precision; where layers overlap, the result is a weighted average in which the nearer layer counts for more.

The frame now has five passes:
1. the depth prepass,
2. ambient occlusion,
3. the lighting, of solid surfaces only, then the sky,
4. **transparency:** the see-through surfaces into the two sums, then the sums laid over the lit scene,
5. tone mapping.

This chapter builds on [Chapter 9](09-ambient-occlusion.md).

## 10.1 A blend per attachment: `vulkan_setup.cpp`

### Why
The transparency pass draws into two images at once, and they need different blending: one adds, the other multiplies. Vulkan's core **`independentBlend`** feature allows a different blend state for each color attachment. Without it, every attachment of a pipeline must blend the same way.

### How
- **`has_features`** now also requires `independentBlend`. Every desktop GPU has it.
- **`create_device`** enables it, next to `samplerAnisotropy`.

### Code
In `game-engine/src/vulkan_setup.cpp`, replace `has_features` with:
```cpp
// Only valid once has_extensions() is true: the extension structs in the
// chain may not be queried on a device that lacks their extension.
bool has_features(const vk::raii::PhysicalDevice& device) {
    const Features supported = device.getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceDescriptorHeapFeaturesEXT,
        vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR
    >();

    const auto& vulkan10 = supported.get<vk::PhysicalDeviceFeatures2>().features;
    const auto& vulkan12 = supported.get<vk::PhysicalDeviceVulkan12Features>();
    const auto& vulkan13 = supported.get<vk::PhysicalDeviceVulkan13Features>();

    return vulkan10.independentBlend
        && vulkan10.samplerAnisotropy
        && vulkan12.bufferDeviceAddress
        && vulkan12.scalarBlockLayout
        && vulkan13.shaderDemoteToHelperInvocation
        && vulkan13.synchronization2
        && vulkan13.dynamicRendering
        && supported.get<vk::PhysicalDeviceDescriptorHeapFeaturesEXT>().descriptorHeap
        && supported.get<vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR>().shaderUntypedPointers;
}
```

Then replace `create_device` with:
```cpp
vk::raii::Device create_device(const GpuChoice& gpu) {
    const float priority = 1.0f;

    const vk::DeviceQueueCreateInfo queue_info{
        .queueFamilyIndex = gpu.queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };

    // StructureChain fills in each struct's pNext, so the order here is the
    // order of the chain. Features2 at the head stands in for pEnabledFeatures.
    const Features features{
        vk::PhysicalDeviceFeatures2{
            .features = {
                .independentBlend = vk::True,   // a different blend for each color attachment
                .samplerAnisotropy = vk::True,  // sharper textures seen at an angle
            },
        },
        vk::PhysicalDeviceVulkan12Features{
            .scalarBlockLayout = vk::True,    // shader structs laid out like C++ structs
            .bufferDeviceAddress = vk::True,  // buffers as 64-bit GPU pointers
        },
        vk::PhysicalDeviceVulkan13Features{
            .shaderDemoteToHelperInvocation = vk::True,  // discard in shaders (alpha masking)
            .synchronization2 = vk::True,     // vkCmdPipelineBarrier2, vkQueueSubmit2
            .dynamicRendering = vk::True,     // vkCmdBeginRendering, no VkRenderPass
        },
        vk::PhysicalDeviceDescriptorHeapFeaturesEXT{
            .descriptorHeap = vk::True,       // descriptors live in buffers we own
        },
        vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR{
            .shaderUntypedPointers = vk::True,
        },
    };

    const vk::DeviceCreateInfo create_info{
        .pNext = &features.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = static_cast<std::uint32_t>(device_extensions.size()),
        .ppEnabledExtensionNames = device_extensions.data(),
    };

    return vk::raii::Device(gpu.device, create_info);
}
```

## 10.2 The two sums: `swapchain.h`, `swapchain.cpp`

### Why
The transparency pass's two sums are images the size of the window, like the others.

### How
- **`accum`,** `R16G16B16A16Sfloat`: the sum of every see-through fragment's color times its coverage, in RGB, and of its coverage, in A, each times the fragment's weight (10.4). Colors are HDR, so it's floating point, like the HDR image.
- **`reveal`,** `R16Sfloat`, one channel: how much of the scene behind still shows through. It starts at 1, and each fragment multiplies it by `1 − coverage`.
- **Both are color attachments, then sampled** by the composite, which reads one texel of each per pixel.

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

// Ambient occlusion's half-resolution input: each 2 x 2 block's nearest
// distance in front of the camera. Its normal uses normal_format.
constexpr vk::Format ao_depth_format = vk::Format::eR32Sfloat;

// Weighted blended transparency's two sums (see mesh.slang): the weighted
// color and coverage, and the share of the scene that still shows through.
constexpr vk::Format accum_format = vk::Format::eR16G16B16A16Sfloat;
constexpr vk::Format reveal_format = vk::Format::eR16Sfloat;

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
    //   ao       ambient occlusion, written by a compute shader, read by the
    //            lighting pass
    //   accum    see-through surfaces' weighted color and coverage, summed
    //   reveal   how much of the scene still shows through them
    //   hdr      the lit scene, which tone mapping writes to the swapchain image
    // and, at half the size, rounded up, ambient occlusion's working images:
    //   ao_depth    each 2 x 2 block's nearest distance in front of the camera
    //   ao_normals  that surface's normal
    //   ao_raw      the horizon search's result, and the blur's
    //   ao_blur     the blur's halfway point
    Image depth;
    Image normals;
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

    // Ambient occlusion works at half resolution, rounded up, and writes its
    // result at full resolution. Compute shaders write all of them (storage);
    // the lighting pass samples the full-resolution one.
    const vk::Extent2D half{(swapchain.extent.width + 1) / 2, (swapchain.extent.height + 1) / 2};
    swapchain.ao_depth = create_image(device, gpu, half, ao_depth_format,
        vk::ImageUsageFlagBits::eStorage, vk::ImageAspectFlagBits::eColor);
    swapchain.ao_normals = create_image(device, gpu, half, normal_format,
        vk::ImageUsageFlagBits::eStorage, vk::ImageAspectFlagBits::eColor);

    for (Image* image : {&swapchain.ao_raw, &swapchain.ao_blur}) {
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
```

## 10.3 The composite's push data: `shader_types.h`

### Why
The composite shader needs to know which heap slots hold the two sums.

### How
**`CompositePushData`** holds the two slots, 8 bytes, as `TonemapPushData` does for the HDR image.

### Code
In `game-engine/src/includes/shader_types.h`, add this section before `// The environment compute shaders' push data. Push data follows std430`:
```cpp
// The transparency composite's push data (composite.slang): the resource heap
// slots of the transparency pass's two sums.
struct CompositePushData {
    std::uint32_t accum;
    std::uint32_t reveal;
};
```

## 10.4 Shading into sums: `mesh.slang`

### Why
A see-through surface is shaded exactly like a solid one; only what happens to the result changes. So the lighting moves out of `fragmentMain` into a function both passes call, and the transparency pass gets its own entry point.

### How
- **`shade_fragment`** is Chapter 9's `fragmentMain` body, unchanged: the shaded color, or a debug view's value, and the alpha. `fragmentMain` just returns it.
- **"Over" blending,** what glTF's `BLEND` mode means, puts a layer of color `c` and coverage `α` on top of what's behind, `b`: `c·α + b·(1 − α)`. With several layers, the result depends on their order, which is why it needed sorting.
- **Order-independent instead:** McGuire and Bavoil observed that the layers' colors can be averaged, weighted by coverage, and then laid over the background by how much all the layers together cover:
  - **The average color:** `Σ cᵢ·αᵢ·wᵢ / Σ αᵢ·wᵢ`.
  - **The total coverage:** `1 − Π(1 − αᵢ)`. That product is exactly how much of the background shows through, in any order.
  - Both are sums and products, which blending can build in any order. **`transparentMain`** writes the terms: `(c·α·w, α·w)` into `accum`, which adds them up, and `α` into `reveal`, which multiplies by `1 − α`.
- **One layer comes out right:** the average of one color is that color, and the coverage is its own. Only overlapping layers are approximated.
- **The weight `w`** makes the nearer layers count for more, as they would in front. The paper's equation 7 falls steeply with the distance `z` in front of the camera: `α · clamp(10 / (10⁻⁵ + (z/5)² + (z/200)⁶), 10⁻², 3·10³)`. The paper tuned it for 16-bit float sums and distances from 0.1 m to 500 m.
  - **The distance** is the clip-space `w` of the point: a perspective projection's `w` is exactly how far the point is in front of the camera, along its view direction.
  - **Colors are clamped to 4,** which tone mapping already shows as nearly white. One fragment then adds at most 4 × 3000 = 12000 to `accum`, against 65504, the largest 16-bit float. Only several bright layers within a few tens of centimeters of the camera could overflow the sum.
  - **The weights aren't scaled down.** Dividing them all by the same factor would cancel in the average, but not in 16-bit floats: a faint layer far away would sink below the smallest numbers they hold precisely, and lose its color.
- **Its limits:** two overlapping layers of very different opacity come out as a blend of both, even when the front one should hide most of the back one: two 90% layers close together show nearly half of each, instead of 90% of the front one. For glass, water and smoke, which are mostly faint and similar, it looks right.

### Code
`game-engine/shaders/mesh.slang`:
```slang
// Draws one glTF primitive: its vertices come from the scene's vertex buffer,
// its place in the world from its DrawData, and its surface from its glTF
// material, whose textures are read from the descriptor heap. Shaded with
// glTF's physically based BRDF, lit by the sun, the file's lights and the
// sky around the scene, already exposed. Three fragment shaders:
//   prepassMain      the depth prepass's vertex normal
//   fragmentMain     opaque and masked surfaces, into the HDR image
//   transparentMain  blended surfaces, into weighted blended transparency's sums

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

// --- Shading a fragment ------------------------------------------------------

// The surface at this fragment: its exposed radiance (or one input, in a
// debug view), and its alpha. The lighting pass writes it as it is; the
// transparency pass adds it into its sums.
// `front_face`: whether this triangle faces the camera.
float4 shade_fragment(VertexOutput input, bool front_face) {
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
// and distances from 0.1 m to 500 m. It falls steeply with the distance in
// front of the camera, so where layers overlap, the nearest dominates; the
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
    const float view_depth = mul(push.frame.view_projection, float4(input.world_position, 1.0)).w;
    const float weight = transparent_weight(coverage, view_depth);

    TransparentOutput output;
    output.accum = float4(min(color.rgb, transparent_max) * coverage, coverage) * weight;
    output.reveal = coverage;
    return output;
}
```

## 10.5 The composite: `composite.slang`

### Why
The sums aren't a color yet: the average has to be divided out, and the result laid over the lit scene.

### How
- **The average:** `accum.rgb / accum.a`. The weights cancel.
- **Laid over the scene** with ordinary "over" blending, with `1 − reveal` as the alpha: `average · (1 − reveal) + scene · reveal`.
- **Where nothing see-through was drawn,** `reveal` is still exactly 1. Those pixels are discarded, leaving the scene as it is.
- **One full-screen triangle,** like tone mapping.

### Code
`game-engine/shaders/composite.slang`:
```slang
// The transparency composite: the transparency pass's sums (transparentMain
// in mesh.slang) turned into one color, and laid over the lit scene. One
// full-screen triangle; the pipeline blends "over":
//     color = average * (1 - reveal) + scene * reveal
// with the fragment's alpha, 1 - reveal, as the weight.

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

struct CompositePushData {
    uint accum;   // resource heap slot: the weighted color and coverage
    uint reveal;  // resource heap slot: how much of the scene shows through
};

[[vk::push_constant]]
ConstantBuffer<CompositePushData> push;

// --- Vertex shader -----------------------------------------------------------

// The full-screen triangle, as in tonemap.slang.
[shader("vertex")]
float4 vertexMain(uint vertex_id : SV_VulkanVertexID) : SV_Position {
    const float2 corner = float2((vertex_id << 1) & 2, vertex_id & 2);
    return float4(corner * 2.0 - 1.0, 0.0, 1.0);
}

// --- Fragment shader -----------------------------------------------------------

[shader("fragment")]
float4 fragmentMain(float4 position : SV_Position) : SV_Target {
    const int3 pixel = int3(int2(position.xy), 0);
    const float reveal = Texture2D.Handle(uint2(push.reveal, 0)).Load(pixel).r;

    // Nothing see-through here: leave the scene as it is.
    if (reveal >= 1.0) {
        discard;
    }

    // The weighted average of the layers' colors: the weights cancel out.
    const float4 accum = Texture2D.Handle(uint2(push.accum, 0)).Load(pixel);
    return float4(accum.rgb / max(accum.a, 1e-7), 1.0 - reveal);
}
```

## 10.6 Pipelines for the sums: `pipeline.h`, `pipeline.cpp`

### Why
The transparency pass's pipeline draws into two images, with a blend state each, and the composite blends "over".

### How
- **`MeshPass::transparency`** runs `transparentMain`. It tests depth like the lighting pass, against the prepass's depth, without writing it.
- **`create_mesh_pipeline` takes a span of color formats,** one per attachment: one image for the prepass and the lighting pass, two for the transparency pass. It throws if the count doesn't fit the pass.
- **The sums' blend states:**
  - **`accum` adds:** source factor 1, destination factor 1.
  - **`reveal` multiplies:** source factor 0, destination factor `1 − source color`, so `reveal = reveal · (1 − α)`. It has only a red channel, so the alpha factors never apply.
- **Chapter 6's "over" blending moves** from the mesh pipelines to `create_fullscreen_pipeline`, behind a new **`ColorBlend`** parameter, for the composite.

### Code
`game-engine/src/includes/pipeline.h`:
```cpp
#pragma once

#include "includes/shader_types.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

// A .spv file as the 32-bit words SPIR-V is made of.
std::vector<std::uint32_t> read_spirv(const std::filesystem::path& path);

// The passes the scene is drawn in, and the images each draws into.
enum class MeshPass {
    depth_normals,  // the prepass, for opaque and masked materials: the normals
    lighting,       // the full shading of opaque and masked materials: the HDR image
    transparency,   // blended materials, into weighted blended transparency's two sums
};

// Draws shaders/mesh.slang into images of `color_formats`, one per
// attachment, depth-tested against a `depth_format` depth buffer, for `pass`
// and materials with alpha mode `alpha_mode`. There is no pipeline layout:
// shaders find their resources in the descriptor heap. Cull mode and front
// face are set per draw.
vk::raii::Pipeline create_mesh_pipeline(
    const vk::raii::Device& device,
    std::span<const vk::Format> color_formats,
    vk::Format depth_format,
    AlphaMode alpha_mode,
    MeshPass pass
);

// How a full-screen pipeline's output meets what's in the image:
//   replace  overwrites it
//   over     mixes with it by the output's alpha
enum class ColorBlend {
    replace,
    over,
};

// Draws shaders/<shader>.spv's vertexMain and fragmentMain as one full-screen
// triangle into a `color_format` image: no vertex data, nothing culled.
// With a `depth_format`, the triangle is depth-tested at depth 0, the far
// plane, without writing depth, so it only reaches pixels nothing else has
// been drawn on: that's how the sky goes behind the scene.
vk::raii::Pipeline create_fullscreen_pipeline(
    const vk::raii::Device& device,
    const char* shader,
    vk::Format color_format,
    vk::Format depth_format = vk::Format::eUndefined,
    ColorBlend blend = ColorBlend::replace
);

// A compute pipeline running `entry_point` from shaders/<shader>.spv.
vk::raii::Pipeline create_compute_pipeline(const vk::raii::Device& device, const char* shader, const char* entry_point);
```

In `game-engine/src/pipeline.cpp`, replace `create_mesh_pipeline` with:
```cpp
vk::raii::Pipeline create_mesh_pipeline(
    const vk::raii::Device& device,
    std::span<const vk::Format> color_formats,
    vk::Format depth_format,
    AlphaMode alpha_mode,
    MeshPass pass
) {
    const bool prepass = pass == MeshPass::depth_normals;
    const bool transparency = pass == MeshPass::transparency;

    // The transparency pass draws into its two sums, every other pass into one image.
    if (color_formats.size() != (transparency ? 2 : 1)) {
        throw std::invalid_argument("create_mesh_pipeline: wrong number of color formats for this pass");
    }

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
            .pName = prepass ? "prepassMain" : transparency ? "transparentMain" : "fragmentMain",
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

    // Depth. With reverse-Z (see camera.cpp) nearer means a *greater* depth
    // value, and the buffer is cleared to 0, the far plane.
    //   - The prepass keeps a fragment only if it's nearer than what's there,
    //     and records its depth: the depth buffer ends up holding the nearest
    //     solid surface at every pixel.
    //   - The lighting pass writes no depth. Its solid surfaces pass "greater
    //     or equal" only where they are that nearest surface, so each pixel is
    //     shaded once.
    //   - The transparency pass writes none either: its see-through surfaces
    //     pass wherever they're in front of the nearest solid one.
    const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = prepass ? vk::True : vk::False,
        .depthCompareOp = prepass ? vk::CompareOp::eGreater : vk::CompareOp::eGreaterOrEqual,
    };

    // Color output, one blend state per attachment.
    //   - The prepass and the lighting pass replace what's there.
    //   - The transparency pass sums. Its first image adds every fragment's
    //     output; its second keeps what each lets through:
    //         accum  = accum + source
    //         reveal = reveal * (1 - source)
    constexpr vk::ColorComponentFlags all_channels = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                                                   | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

    const vk::PipelineColorBlendAttachmentState replace{
        .blendEnable = vk::False,
        .colorWriteMask = all_channels,
    };

    const vk::PipelineColorBlendAttachmentState add{
        .blendEnable = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eOne,
        .dstColorBlendFactor = vk::BlendFactor::eOne,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = all_channels,
    };

    // The reveal image has only a red channel: the alpha factors never apply.
    const vk::PipelineColorBlendAttachmentState let_through{
        .blendEnable = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eZero,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcColor,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eZero,
        .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = all_channels,
    };

    const std::array sums{add, let_through};

    const vk::PipelineColorBlendStateCreateInfo color_blend{
        .attachmentCount = static_cast<std::uint32_t>(color_formats.size()),
        .pAttachments = transparency ? sums.data() : &replace,
    };

    // Dynamic rendering: instead of a VkRenderPass, the pipeline names the
    // formats of the images it will draw into.
    const vk::PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount = static_cast<std::uint32_t>(color_formats.size()),
        .pColorAttachmentFormats = color_formats.data(),
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

Then replace `create_fullscreen_pipeline` with:
```cpp
vk::raii::Pipeline create_fullscreen_pipeline(
    const vk::raii::Device& device,
    const char* shader,
    vk::Format color_format,
    vk::Format depth_format,
    ColorBlend blend
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

    // The vertex shader makes the triangle's three corners from the vertex
    // index alone.
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

    // With reverse-Z the buffer is cleared to 0, and the triangle sits at 0:
    // it passes "greater or equal" only where the depth is still 0, where
    // nothing has been drawn. Without a depth buffer, every pixel passes.
    const bool depth = depth_format != vk::Format::eUndefined;

    const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
        .depthTestEnable = depth ? vk::True : vk::False,
        .depthWriteEnable = vk::False,
        .depthCompareOp = vk::CompareOp::eGreaterOrEqual,
    };

    // Each pixel the triangle reaches gets one fragment. "Over" mixes it with
    // what's there, weighted by its alpha:
    //     color = source.rgb * source.a + destination.rgb * (1 - source.a)
    const vk::PipelineColorBlendAttachmentState blend_attachment{
        .blendEnable = blend == ColorBlend::over ? vk::True : vk::False,
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
```

## 10.7 No more sorting: `scene.h`, `scene.cpp`

### Why
Each draw's center was kept only to sort see-through draws by distance. The sums don't need that.

### How
`MeshDraw::center` goes, and `visit_node` no longer computes it.

### Code
`game-engine/src/includes/scene.h`:
```cpp
#pragma once

#include "includes/shader_types.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

// One glTF primitive: a run of indices in the scene's index buffer, drawn
// against the vertices starting at `vertex_offset` in the vertex buffer.
struct Primitive {
    std::uint32_t first_index = 0;
    std::uint32_t index_count = 0;
    std::int32_t vertex_offset = 0;
    std::uint32_t material = 0;  // index into Scene::materials
};

// glTF's texture filters and wrap modes, as the file stores them: OpenGL
// enum values (9728 GL_NEAREST, 10497 GL_REPEAT, ...). A filter of -1 means
// the file leaves it to the renderer.
struct SceneSampler {
    int mag_filter = -1;
    int min_filter = -1;
    int wrap_s = 10497;  // GL_REPEAT, glTF's default
    int wrap_t = 10497;
};

// A material's reference to one texture.
struct TextureRef {
    std::int32_t image = -1;    // index into Scene::images, or -1 for none
    std::int32_t sampler = -1;  // index into Scene::samplers, or -1 for the default
    std::uint32_t uv_set = 0;   // which texture coordinates: 0 or 1
};

// A glTF metallic-roughness material. The defaults are glTF's: a material
// that sets nothing is white, fully metallic and fully rough.
struct SceneMaterial {
    glm::vec4 base_color_factor{1.0f};
    TextureRef base_color;
    float metallic_factor = 1.0f;
    float roughness_factor = 1.0f;
    TextureRef metallic_roughness;
    TextureRef normal;
    float normal_scale = 1.0f;
    TextureRef occlusion;
    float occlusion_strength = 1.0f;
    glm::vec3 emissive_factor{0.0f};
    TextureRef emissive;
    AlphaMode alpha_mode = AlphaMode::opaque;
    float alpha_cutoff = 0.5f;
    bool double_sided = false;
};

// An image as the file stores it: still encoded as PNG, JPEG, ...
struct SceneImage {
    std::vector<unsigned char> encoded;
    std::string name;   // the file name or glTF name, for messages
    bool srgb = false;  // holds colors (base color, emissive) rather than data like normals
};

// One thing to draw: a primitive, placed in the world by a node's transform.
// A mesh used by several nodes is drawn once per node.
struct MeshDraw {
    glm::mat4 model{1.0f};
    std::uint32_t primitive = 0;

    // A transform that mirrors the primitive (a negative scale) reverses the
    // order its triangles' corners appear in, which decides which side is
    // the front.
    bool mirrored = false;
};

// Everything from a glTF file that drawing its geometry needs, flattened into
// arrays ready to upload: every primitive's vertices and indices back to back.
struct Scene {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<Primitive> primitives;
    std::vector<MeshDraw> draws;

    // The file's materials, plus a plain white one at the end for primitives
    // that don't name a material.
    std::vector<SceneMaterial> materials;
    std::vector<SceneImage> images;
    std::vector<SceneSampler> samplers;

    // KHR_lights_punctual lights, placed in the world by their nodes.
    std::vector<Light> lights;

    // World-space box around everything drawn, for placing the camera.
    glm::vec3 bounds_min{std::numeric_limits<float>::max()};
    glm::vec3 bounds_max{std::numeric_limits<float>::lowest()};
};

// Loads the default scene of a .gltf or .glb file, with its materials,
// samplers, lights and images, still encoded.
Scene load_gltf(const std::filesystem::path& path);
```

In `game-engine/src/scene.cpp`, replace `visit_node` with:
```cpp
// Walks the node tree. Each node's world transform is its parent's times its
// own; every primitive of a node's mesh becomes one draw, and a node's light
// is placed by the same transform.
void visit_node(
    const tinygltf::Model& model,
    int node_index,
    const glm::mat4& parent,
    const std::vector<std::vector<LoadedPrimitive>>& mesh_primitives,
    Scene& scene
) {
    const tinygltf::Node& node = model.nodes.at(node_index);
    const glm::mat4 world = parent * local_transform(node);

    if (node.mesh >= 0) {
        // A negative determinant means the transform mirrors space.
        const bool mirrored = glm::determinant(glm::mat3(world)) < 0.0f;

        for (const LoadedPrimitive& primitive : mesh_primitives.at(node.mesh)) {
            scene.draws.push_back(MeshDraw{
                .model = world,
                .primitive = primitive.index,
                .mirrored = mirrored,
            });

            // Grow the scene bounds by the 8 corners of the primitive's box,
            // moved into world space.
            for (int corner = 0; corner < 8; ++corner) {
                const glm::vec3 local{
                    corner & 1 ? primitive.local_max.x : primitive.local_min.x,
                    corner & 2 ? primitive.local_max.y : primitive.local_min.y,
                    corner & 4 ? primitive.local_max.z : primitive.local_min.z,
                };
                const glm::vec3 point = glm::vec3(world * glm::vec4(local, 1.0f));

                scene.bounds_min = glm::min(scene.bounds_min, point);
                scene.bounds_max = glm::max(scene.bounds_max, point);
            }
        }
    }

    if (node.light >= 0) {
        if (const auto light = world_light(model.lights.at(node.light), world)) {
            scene.lights.push_back(*light);
        }
    }

    for (const int child : node.children) {
        visit_node(model, child, world, mesh_primitives, scene);
    }
}
```

## 10.8 Five passes a frame: `main.cpp`

### Why
`record_frame` gains the transparency pass, and the per-frame sort goes.

### How
- **`solid_mode_count` and `blend_mode`** name the batches: the first two are solid, the third is see-through. A `static_assert` checks that the third is the blended one.
- **`ScreenSlots`** gains `accum` and `reveal`, sampled: eleven slots now.
- **`ScenePipelines`** has one pipeline per solid alpha mode in the prepass and the lighting pass, and one `transparency` pipeline. Plus the sky, the composite and tone mapping.
- **The lighting pass** draws the opaque and masked batches, then the sky.
- **The transparency pass** runs only if the scene has see-through draws. Sponza has none, and skipping it saves clearing two images and a full-screen pass.
  1. **Both sums to `eColorAttachmentOptimal`,** from `eUndefined`. This waits for the previous frame's composite to finish reading them.
  2. **Draw the blended batch** into them, `accum` cleared to 0 and `reveal` to 1, against the lighting pass's read-only depth. A see-through surface behind a solid one fails the depth test; one behind another see-through one doesn't, and adds in. The viewport and scissor set earlier still apply: dynamic state lasts the whole command buffer.
  3. **Both sums to `eShaderReadOnlyOptimal`** for the composite.
  4. **A barrier on the HDR image,** without a layout change. The composite blends into it, which reads what the lighting pass wrote, and two render passes aren't ordered by themselves.
  5. **The composite,** into the HDR image, which is loaded and stored.
- **The blended batch is no longer sorted.** The batches are built once and never change.
- **Debug views** (keys 2 to 9) go through the sums too. They give alpha 1, except the base color view, so a see-through surface hides what's behind it there. Where two overlap, the view shows a weighted average of both.

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

// The three alpha modes, in batch order: the solid ones, which the prepass
// and the lighting pass draw, then the see-through one, which only the
// transparency pass draws.
constexpr std::array alpha_modes{AlphaMode::opaque, AlphaMode::mask, AlphaMode::blend};
constexpr std::size_t solid_mode_count = 2;
constexpr std::size_t blend_mode = 2;

static_assert(alpha_modes[blend_mode] == AlphaMode::blend);

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
    AoTargets ao_targets;              // storage, for the AO pass
    std::uint32_t accum = 0;           // sampled, by the transparency composite
    std::uint32_t reveal = 0;          // sampled, by the transparency composite
};

constexpr std::uint32_t screen_slot_count = 11;

// Every graphics pipeline a frame uses. The prepass and the lighting pass
// have one per solid alpha mode, in alpha_modes' order; see-through surfaces
// have only the transparency pass's.
struct ScenePipelines {
    std::vector<vk::raii::Pipeline> prepass;
    std::vector<vk::raii::Pipeline> lighting;
    vk::raii::Pipeline transparency = nullptr;
    vk::raii::Pipeline background = nullptr;
    vk::raii::Pipeline composite = nullptr;
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

// Records a frame in five passes:
//   1. the depth prepass: every solid surface's depth and vertex normal,
//   2. ambient occlusion, from those, in compute shaders,
//   3. the lighting, into the HDR image: each solid alpha mode's batch with
//      that mode's pipeline, against the prepass's depth, then the sky
//      behind them,
//   4. transparency, if anything is see-through: the blended batch into two
//      sums, in any order, then those laid over the HDR image,
//   5. tone mapping, from the HDR image into the swapchain image, which is
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
    for (std::size_t mode = 0; mode < solid_mode_count; ++mode) {
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

    record_ambient_occlusion(commands, ambient_occlusion, swapchain, draws.screen.ao_targets,
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

    for (std::size_t mode = 0; mode < solid_mode_count; ++mode) {
        draw_batch(commands, draws, mode, pipelines.lighting[mode]);
    }

    // The sky goes in once everything solid is drawn: it only covers pixels
    // still at the far plane.
    commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.background);

    const PushData sky_push{.frame = draws.frame, .draw_index = 0};
    commands.pushDataEXT(vk::PushDataInfoEXT{
        .offset = 0,
        .data = {.address = &sky_push, .size = sizeof(sky_push)},
    });

    commands.draw(3, 1, 0, 0);
    commands.endRendering();

    // --- Pass 4: transparency ------------------------------------------------

    if (!draws.batches[blend_mode].empty()) {
        const vk::Image accum = *swapchain.accum.handle;
        const vk::Image reveal = *swapchain.reveal.handle;

        // Both sums are shared by the frames in flight: this waits for the
        // previous frame's composite to finish reading them.
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

        // The same depth as the lighting pass: tested, not written, so a
        // see-through surface behind a solid one is hidden, and one behind
        // another see-through one still counts.
        commands.beginRendering(vk::RenderingInfo{
            .renderArea = whole_image,
            .layerCount = 1,
            .colorAttachmentCount = static_cast<std::uint32_t>(sum_attachments.size()),
            .pColorAttachments = sum_attachments.data(),
            .pDepthAttachment = &lighting_depth,
        });

        // The viewport and scissor set earlier still apply: dynamic state
        // lasts for the whole command buffer.
        draw_batch(commands, draws, blend_mode, pipelines.transparency);
        commands.endRendering();

        // The composite reads both sums, and blends into the HDR image, which
        // the lighting pass just wrote: the same layout, but its writes must
        // land before the blend reads them.
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

    // --- Pass 5: tone mapping ------------------------------------------------

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
        //   - The prepass draws normals and depth, and the lighting pass the
        //     HDR image, for opaque and masked materials. The transparency
        //     pass draws blended ones into its two sums.
        //   - The sky draws into the HDR image, behind the scene, and the
        //     composite over it; tone mapping writes the swapchain image.
        ScenePipelines pipelines;

        for (const AlphaMode mode : alpha_modes) {
            if (mode == AlphaMode::blend) {
                pipelines.transparency = create_mesh_pipeline(device, std::array{accum_format, reveal_format},
                    depth_format, mode, MeshPass::transparency);
            } else {
                pipelines.prepass.push_back(create_mesh_pipeline(device, std::array{normal_format},
                    depth_format, mode, MeshPass::depth_normals));
                pipelines.lighting.push_back(create_mesh_pipeline(device, std::array{hdr_format},
                    depth_format, mode, MeshPass::lighting));
            }
        }

        pipelines.background = create_fullscreen_pipeline(device, "background", hdr_format, depth_format);
        pipelines.composite = create_fullscreen_pipeline(device, "composite", hdr_format, vk::Format::eUndefined, ColorBlend::over);
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
            .ao_targets = {
                .ao_depth = first_screen_slot + 4,
                .ao_normals = first_screen_slot + 5,
                .ao_raw = first_screen_slot + 6,
                .ao_blur = first_screen_slot + 7,
                .ao = first_screen_slot + 8,
            },
            .accum = first_screen_slot + 9,
            .reveal = first_screen_slot + 10,
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
            write_image_descriptor(device, heaps, screen.ao_targets.ao_depth, whole(swapchain.ao_depth, color), storage);
            write_image_descriptor(device, heaps, screen.ao_targets.ao_normals, whole(swapchain.ao_normals, color), storage);
            write_image_descriptor(device, heaps, screen.ao_targets.ao_raw, whole(swapchain.ao_raw, color), storage);
            write_image_descriptor(device, heaps, screen.ao_targets.ao_blur, whole(swapchain.ao_blur, color), storage);
            write_image_descriptor(device, heaps, screen.ao_targets.ao, whole(swapchain.ao, color), storage);
            write_image_descriptor(device, heaps, screen.accum, whole(swapchain.accum, color));
            write_image_descriptor(device, heaps, screen.reveal, whole(swapchain.reveal, color));
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

        // Draw indices by alpha mode. Every batch can be drawn in any order:
        // even the blended one, whose sums don't depend on it.
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

## 10.9 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **Sponza** looks exactly as in Chapter 9. The terminal shows `0 blended`, and the transparency pass is skipped.
- **AlphaBlendModeTest** (change `scene_file` in `main.cpp`): every panel shows its check mark, as in Chapter 6. The blended panel's gradient fades from opaque to clear, and looks exactly as it did with sorting: it's a single layer.
- **No `[validation …]` lines.**

**Overlapping layers.** None of the sample models overlaps see-through surfaces much, so here is one that does: three half-transparent squares, red, green and blue, 0.6 m apart, overlapping, and double-sided, 3 m in front of where the camera starts. Save it as `lecture-md/game-engine/assets/OitLayers/OitLayers.gltf`. Its geometry, 204 bytes, is inside the file, as a base64 data URI:
```json
{
  "asset": {"version": "2.0", "generator": "game-engine chapter 10"},
  "scene": 0,
  "scenes": [
    {"nodes": [0, 1, 2]}
  ],
  "nodes": [
    {"name": "Red", "mesh": 0, "translation": [0.0, 0.0, -3.0]},
    {"name": "Green", "mesh": 1, "translation": [0.0, 0.0, -3.0]},
    {"name": "Blue", "mesh": 2, "translation": [0.0, 0.0, -3.0]}
  ],
  "meshes": [
    {"name": "Red", "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 3}, "indices": 4, "material": 0}]},
    {"name": "Green", "primitives": [{"attributes": {"POSITION": 1, "NORMAL": 3}, "indices": 4, "material": 1}]},
    {"name": "Blue", "primitives": [{"attributes": {"POSITION": 2, "NORMAL": 3}, "indices": 4, "material": 2}]}
  ],
  "materials": [
    {"name": "Red", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.1, 0.1, 0.5], "metallicFactor": 0.0, "roughnessFactor": 0.8}, "alphaMode": "BLEND", "doubleSided": true},
    {"name": "Green", "pbrMetallicRoughness": {"baseColorFactor": [0.1, 1.0, 0.1, 0.5], "metallicFactor": 0.0, "roughnessFactor": 0.8}, "alphaMode": "BLEND", "doubleSided": true},
    {"name": "Blue", "pbrMetallicRoughness": {"baseColorFactor": [0.1, 0.2, 1.0, 0.5], "metallicFactor": 0.0, "roughnessFactor": 0.8}, "alphaMode": "BLEND", "doubleSided": true}
  ],
  "accessors": [
    {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-1.1, -0.35, 0.6], "max": [0.1, 0.85, 0.6]},
    {"bufferView": 0, "byteOffset": 48, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-0.6, -0.85, 0.0], "max": [0.6, 0.35, 0.0]},
    {"bufferView": 0, "byteOffset": 96, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-0.1, -0.35, -0.6], "max": [1.1, 0.85, -0.6]},
    {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
    {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 144, "target": 34962},
    {"buffer": 0, "byteOffset": 144, "byteLength": 48, "target": 34962},
    {"buffer": 0, "byteOffset": 192, "byteLength": 12, "target": 34963}
  ],
  "buffers": [
    {"byteLength": 204, "uri": "data:application/octet-stream;base64,zcyMvzMzs76amRk/zczMPTMzs76amRk/zczMPZqZWT+amRk/zcyMv5qZWT+amRk/mpkZv5qZWb8AAAAAmpkZP5qZWb8AAAAAmpkZPzMzsz4AAAAAmpkZvzMzsz4AAAAAzczMvTMzs76amRm/zcyMPzMzs76amRm/zcyMP5qZWT+amRm/zczMvZqZWT+amRm/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAABAAIAAAACAAMA"}
  ]
}
```

Load it: the red square is nearest. Fly around and look from behind, where the blue one is. Where only one square covers the sky, it looks just as it would with Chapter 9's sorting. Where squares overlap, the nearer one's color leads, from either side, and nothing flickers or jumps as you move around them. The lead is clearest up close: at 3 m each square weighs about 1.5 times the one behind it, at 10 m only about 1.1 times.

Next, in Chapter 11, the scene finally gets shadows, by tracing rays: acceleration structures over the whole scene, and a ray from every pixel toward the sun.

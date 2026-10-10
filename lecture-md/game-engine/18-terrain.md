# Chapter 18: Terrain

By the end of this chapter, the scene stands on a landscape: 8 km of hills with a valley through them, from a height field of 16-bit samples 4 m apart, repeated to the horizon. The flat ground of Chapter 14 is gone. The terrain is drawn by **mesh shaders**, a stage that replaces the vertex shader with a workgroup that makes its own vertices and triangles: each workgroup reads the heights of one patch of 8 × 8 quads and writes 81 vertices and 128 triangles, with no vertex or index buffer anywhere. Far patches use every second, fourth or thousandth sample, so the triangles drawn stay about the same size on screen wherever they are: **continuous distance-dependent level of detail**, CDLOD, after Strugar ("Continuous Distance-Dependent Level of Detail for Rendering Heightmaps", Journal of Graphics Tools 14(4), 2009), with vertices sliding between levels so that no crack ever opens between patches.

Which patches to draw is the cull's job. Each phase of Chapter 17's cull walks the terrain's quadtree from the whole landscape down to the patches the view needs, testing each node's box against the view and the depth pyramid, and lists the patches for a single indirect mesh dispatch. The terrain is culled, drawn and shaded like everything else: the same two phases, the same prepass, the same lighting, now shared in a header every surface shader includes.

Shadows need no acceleration structure for the terrain: a shadow ray that meets nothing in the TLAS goes on to **march the height field**, through a quadtree of each node's lowest and highest sample, stepping node by node along the ray with no stack, down to a quad only where it might hit. The sun and every lamp cast the hills' shadows exactly, and a hill blocks a lamp on its far side.

The field itself is generated: ridged fractal noise, a valley, and a flat plateau for the box field to stand on. Most large open-world engines keep heights as the source of their terrain too, and build its mesh at run time from them: a height field is small, exact for rays and physics, and the same data at every distance.

This chapter builds on [Chapter 17](17-occlusion-culling.md).

## 18.1 Mesh shaders on the device: `vulkan_setup.cpp`

### Why
Mesh shaders are an extension, `VK_EXT_mesh_shader`, with a feature to enable.

### How
- **`device_extensions`** gains the extension; **`Features`** and `has_features` the `VkPhysicalDeviceMeshShaderFeaturesEXT` struct, requiring `meshShader`. Its `taskShader`, a stage that runs before mesh shaders to decide how many to launch, isn't needed: the cull decides that.
- **`geometryShader`** is required and enabled too, though no geometry shader is ever used: for a fragment shader that reads its triangle's index, `SV_PrimitiveID`, Slang declares SPIR-V's Geometry capability (Vulkan would accept the mesh shading one as well), and Vulkan ties that capability to this feature (18.4).
- **`create_device`** enables both.

### Code
`game-engine/src/vulkan_setup.cpp`:
```cpp
#include "includes/vulkan_setup.h"

#include <algorithm>
#include <array>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

    constexpr const char *validation_layer = "VK_LAYER_KHRONOS_validation";

    // What the renderer needs from a GPU

    // Device creation doesn't need VK_KHR_shader_untyped_pointers, but shaders that index the descriptor heap compile to SPIR-V untyped pointers. Acceleration structures (the scene, organized for tracing rays) need deferred host operations, an extension they're built on, even though we build them on the GPU. Ray queries trace rays from any shader. Mesh shaders draw the terrain (terrain.slang).
    constexpr std::array device_extensions{
        vk::KHRSwapchainExtensionName,
        vk::EXTDescriptorHeapExtensionName,
        vk::KHRShaderUntypedPointersExtensionName,
        vk::KHRAccelerationStructureExtensionName,
        vk::KHRDeferredHostOperationsExtensionName,
        vk::KHRRayQueryExtensionName,
        vk::EXTMeshShaderExtensionName,
    };

    // Every feature struct we read in pick_gpu() and write in create_device(), linked through pNext by StructureChain.
    using Features = vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceDescriptorHeapFeaturesEXT,
        vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
        vk::PhysicalDeviceRayQueryFeaturesKHR,
        vk::PhysicalDeviceMeshShaderFeaturesEXT
    >;

    // Helpers

    VKAPI_ATTR vk::Bool32 VKAPI_CALL on_validation_message(
        vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
        vk::DebugUtilsMessageTypeFlagsEXT,
        const vk::DebugUtilsMessengerCallbackDataEXT *data,
        void*
    ) {
        std::println(stderr, "[validation {}] {}", vk::to_string(severity), data->pMessage);
        return vk::False;
    }

    int rank(vk::PhysicalDeviceType type) {
        switch (type) {
            case vk::PhysicalDeviceType::eDiscreteGpu: return 3;
            case vk::PhysicalDeviceType::eIntegratedGpu: return 2;
            case vk::PhysicalDeviceType::eVirtualGpu: return 1;
            default: return 0;
        }
    }

    bool has_extensions(const vk::raii::PhysicalDevice &device) {
        const std::vector<vk::ExtensionProperties> available = device.enumerateDeviceExtensionProperties();

        return std::ranges::all_of(device_extensions, [&](std::string_view name) {
            return std::ranges::any_of(available, [&](const vk::ExtensionProperties &extension) {
                return name == extension.extensionName.data();
            });
        });
    }

    // Only valid once has_extensions() is true: the extension structs in the chain may not be queried on a device that lacks their extension.
    bool has_features(const vk::raii::PhysicalDevice &device) {
        const Features supported = device.getFeatures2<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan12Features,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceDescriptorHeapFeaturesEXT,
            vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR,
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
            vk::PhysicalDeviceRayQueryFeaturesKHR,
            vk::PhysicalDeviceMeshShaderFeaturesEXT
        >();

        const auto &vulkan10 = supported.get<vk::PhysicalDeviceFeatures2>().features;
        const auto &vulkan12 = supported.get<vk::PhysicalDeviceVulkan12Features>();
        const auto &vulkan13 = supported.get<vk::PhysicalDeviceVulkan13Features>();

        return vulkan10.independentBlend
            && vulkan10.geometryShader
            && vulkan10.multiDrawIndirect
            && vulkan10.drawIndirectFirstInstance
            && vulkan10.samplerAnisotropy
            && vulkan10.shaderInt64
            && vulkan12.drawIndirectCount
            && vulkan12.bufferDeviceAddress
            && vulkan12.scalarBlockLayout
            && vulkan13.shaderDemoteToHelperInvocation
            && vulkan13.synchronization2
            && vulkan13.dynamicRendering
            && supported.get<vk::PhysicalDeviceDescriptorHeapFeaturesEXT>().descriptorHeap
            && supported.get<vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR>().shaderUntypedPointers
            && supported.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure
            && supported.get<vk::PhysicalDeviceRayQueryFeaturesKHR>().rayQuery
            && supported.get<vk::PhysicalDeviceMeshShaderFeaturesEXT>().meshShader;
    }

}  // namespace

// Instance

bool validation_layer_available(const vk::raii::Context &context) {
    return std::ranges::any_of(context.enumerateInstanceLayerProperties(), [](const vk::LayerProperties &layer) {
        return std::string_view(layer.layerName.data()) == validation_layer;
    });
}

vk::raii::Instance create_instance(
    const vk::raii::Context &context,
    std::span<const char *const> extensions,
    bool validation
) {
    std::vector<const char*> enabled_extensions(extensions.begin(), extensions.end());
    std::vector<const char*> enabled_layers;

    if (validation) {
        enabled_layers.push_back(validation_layer);
        enabled_extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    // apiVersion is the newest Vulkan the app will use. Each GPU reports its own version, which pick_gpu() checks.
    const vk::ApplicationInfo app_info{
        .pApplicationName = "game-engine",
        .applicationVersion = vk::makeApiVersion(0, 1, 0, 0),
        .pEngineName = "none",
        .engineVersion = 0,
        .apiVersion = vk::ApiVersion14,
    };

    const vk::InstanceCreateInfo create_info{
        .pApplicationInfo = &app_info,
        .enabledLayerCount = static_cast<std::uint32_t>(enabled_layers.size()),
        .ppEnabledLayerNames = enabled_layers.data(),
        .enabledExtensionCount = static_cast<std::uint32_t>(enabled_extensions.size()),
        .ppEnabledExtensionNames = enabled_extensions.data(),
    };

    return vk::raii::Instance(context, create_info);
}

vk::raii::DebugUtilsMessengerEXT create_debug_messenger(const vk::raii::Instance &instance) {
    using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;

    const vk::DebugUtilsMessengerCreateInfoEXT create_info{
        .messageSeverity = Severity::eWarning | Severity::eError,
        .messageType = Type::eGeneral | Type::eValidation | Type::ePerformance,
        .pfnUserCallback = &on_validation_message,
    };

    return vk::raii::DebugUtilsMessengerEXT(instance, create_info);
}

vk::raii::SurfaceKHR create_surface(const vk::raii::Instance &instance, SDL_Window *window) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    // SDL speaks the C API, so hand it the raw handle and wrap the result.
    if (!SDL_Vulkan_CreateSurface(window, static_cast<VkInstance>(*instance), nullptr, &surface)) {
        throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface failed (") + SDL_GetError() + ")");
    }

    return vk::raii::SurfaceKHR(instance, surface);
}

// Picking a GPU

std::optional<GpuChoice> pick_gpu(const vk::raii::Instance &instance, const vk::raii::SurfaceKHR &surface) {
    std::optional<GpuChoice> best;
    int best_rank = -1;

    for (const vk::raii::PhysicalDevice &device : instance.enumeratePhysicalDevices()) {
        const vk::PhysicalDeviceProperties properties = device.getProperties();
        const std::vector<vk::QueueFamilyProperties> families = device.getQueueFamilyProperties();

        // First queue family that can both draw and present to this surface.
        std::optional<std::uint32_t> family;

        for (std::uint32_t i = 0; i < families.size(); ++i) {
            if ((families[i].queueFlags & vk::QueueFlagBits::eGraphics) && device.getSurfaceSupportKHR(i, *surface)) {
                family = i;
                break;
            }
        }

        // Each check may only run once the one before it has passed.
        std::string_view verdict = "usable";

        if (!family) {
            verdict = "can't present";
        } else if (properties.apiVersion < vk::ApiVersion14) {
            verdict = "needs Vulkan 1.4";
        } else if (!has_extensions(device)) {
            verdict = "missing extensions";
        } else if (!has_features(device)) {
            verdict = "missing features";
        }

        std::println(
            "  {:<45} {:<14} Vulkan {}.{}.{}  {}",
            properties.deviceName.data(),
            vk::to_string(properties.deviceType),
            vk::apiVersionMajor(properties.apiVersion),
            vk::apiVersionMinor(properties.apiVersion),
            vk::apiVersionPatch(properties.apiVersion),
            verdict
        );

        if (verdict == "usable" && rank(properties.deviceType) > best_rank) {
            best = GpuChoice{device, *family};
            best_rank = rank(properties.deviceType);
        }
    }

    return best;
}

// Logical device

vk::raii::Device create_device(const GpuChoice &gpu) {
    const float priority = 1.0f;

    const vk::DeviceQueueCreateInfo queue_info{
        .queueFamilyIndex = gpu.queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };

    // StructureChain fills in each struct's pNext, so the order here is the order of the chain. Features2 at the head stands in for pEnabledFeatures.
    const Features features{
        vk::PhysicalDeviceFeatures2{
            .features = {
                .independentBlend = vk::True,           // a different blend for each color attachment
                .geometryShader = vk::True,             // no geometry shader, but fragment shaders reading their triangle's index need its capability
                .multiDrawIndirect = vk::True,          // indirect calls that draw more than one command
                .drawIndirectFirstInstance = vk::True,  // indirect draws that start past instance 0
                .samplerAnisotropy = vk::True,          // sharper textures seen at an angle
                .shaderInt64 = vk::True,                // 64-bit integers: the TLAS's address
            },
        },
        vk::PhysicalDeviceVulkan12Features{
            .drawIndirectCount = vk::True,    // indirect draws whose count the GPU reads
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
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR{
            .accelerationStructure = vk::True,  // build BLAS and TLAS
        },
        vk::PhysicalDeviceRayQueryFeaturesKHR{
            .rayQuery = vk::True,               // trace rays from fragment shaders
        },
        vk::PhysicalDeviceMeshShaderFeaturesEXT{
            .meshShader = vk::True,             // mesh shaders: the terrain's patches
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

// Descriptor heap limits

void print_descriptor_heap_properties(const GpuChoice &gpu) {
    const auto properties = gpu.device.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceDescriptorHeapPropertiesEXT
    >();
    const auto &heap = properties.get<vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();

    std::println("Descriptor heap:");
    std::println("  image descriptor    {:>4} bytes, {:>3}-byte aligned", heap.imageDescriptorSize, heap.imageDescriptorAlignment);
    std::println("  buffer descriptor   {:>4} bytes, {:>3}-byte aligned", heap.bufferDescriptorSize, heap.bufferDescriptorAlignment);
    std::println("  sampler descriptor  {:>4} bytes, {:>3}-byte aligned", heap.samplerDescriptorSize, heap.samplerDescriptorAlignment);
    std::println("  resource heap       up to {} bytes, {} reserved for the driver", heap.maxResourceHeapSize, heap.minResourceHeapReservedRange);
    std::println("  sampler heap        up to {} bytes, {} reserved for the driver", heap.maxSamplerHeapSize, heap.minSamplerHeapReservedRange);
    std::println("  push data           {} bytes", heap.maxPushDataSize);
}
```

## 18.2 The height field: `height_field.py`, `terrain.h`, `terrain.cpp`

### Why
The terrain needs data, and the engine needs it on the GPU in a form every shader can read: the samples, and the quadtree of bounds that the cull and the rays walk.

### How
- **The file:** `field.r16` is 2049 × 2049 unsigned 16-bit samples, row by row, little-endian: 8.4 MB. `field.json` says how many, how far apart (4 m), where the field's corner is in metres from the scene's origin (−4096, −4096), and what 0 and 65535 mean in metres. A side of 2^n + 1 samples is 2^n quads, which halve exactly down to one.
- **Generating it:** a script, since a landscape is nothing to type. Ridged multifractal noise (Musgrave, in Ebert et al., "Texturing and Modeling: A Procedural Approach", 1994), eight octaves of 1 − |gradient noise| from 2 km down to 16 m, each octave weighted by the one above so that ridges stay sharp and valleys broad; a valley pressed into it 600 m either side of x = 0; and a plateau at −1 m within 300 m of the origin, blended into the hills over the next 200 m, so the box field and Sponza stand on level ground and the camera starts a metre above it. The heights run from −21 m to 318 m. Save it as `lecture-md/game-engine/assets/height_field.py` and run it from that directory with `python3 height_field.py terrain`:
```python
# Writes assets/terrain/field.r16 and field.json: a height field of 2049 x 2049 16-bit samples, 4 m apart, 8,192 m across. Hills from ridged fractal noise, a valley through the middle, and a flat plateau at -1 m around the origin for the box field. Run: python3 height_field.py assets/terrain
import json, pathlib, sys
import numpy as np

directory = pathlib.Path(sys.argv[1])
directory.mkdir(parents=True, exist_ok=True)

samples, step = 2049, 4.0
extent = (samples - 1) * step  # 8,192 m
half = extent / 2

# Sample positions in metres, the origin at the centre.
x = np.linspace(-half, half, samples, dtype=np.float64)
z = x
X, Z = np.meshgrid(x, z)  # Z varies down the rows: row j is z = -half + j * step

# Gradient noise (Perlin 1985, with the smoother fade of Perlin 2002), from a hash of the lattice point: the same field every run.
def hash2(ix, iz, seed):
    h = (ix.astype(np.int64) * 0x9E3779B1 + iz.astype(np.int64) * 0x85EBCA77 + seed * 0xC2B2AE3D) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x2C1B3C6D) & 0xFFFFFFFF
    h ^= h >> 12
    h = (h * 0x297A2D39) & 0xFFFFFFFF
    h ^= h >> 15
    return h

def gradient_noise(px, pz, seed):
    ix, iz = np.floor(px), np.floor(pz)
    fx, fz = px - ix, pz - iz
    ix, iz = ix.astype(np.int64), iz.astype(np.int64)
    def dot_gradient(dx, dz):
        h = hash2(ix + dx, iz + dz, seed)
        angle = h.astype(np.float64) * (2 * np.pi / 4294967296.0)
        return np.cos(angle) * (fx - dx) + np.sin(angle) * (fz - dz)
    ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    uz = fz * fz * fz * (fz * (fz * 6 - 15) + 10)
    top = dot_gradient(0, 0) * (1 - ux) + dot_gradient(1, 0) * ux
    bottom = dot_gradient(0, 1) * (1 - ux) + dot_gradient(1, 1) * ux
    return top * (1 - uz) + bottom * uz  # about -0.7 to 0.7

# Ridged multifractal (Musgrave, in "Texturing and Modeling: A Procedural Approach", 1994): octaves of 1 - |noise|, squared, so crests are sharp and valleys broad, each octave half the size, its amplitude falling off by `falloff`, and its weight set by the octave above, so detail gathers on the ridges.
def ridged(px, pz, octaves, lacunarity=2.0, falloff=0.5, seed=1):
    total, amplitude, weight, frequency = np.zeros_like(px), 1.0, np.ones_like(px), 1.0
    for octave in range(octaves):
        signal = 1.0 - np.abs(gradient_noise(px * frequency, pz * frequency, seed + octave))
        signal = signal * signal * weight
        weight = np.clip(signal * 2.0, 0.0, 1.0)
        total += signal * amplitude
        amplitude *= falloff
        frequency *= lacunarity
    return total

# Hills: ridges 2 km apart, 220 m high, with eight octaves of detail down to 16 m.
hills = ridged(X / 2000.0, Z / 2000.0, 8) * 220.0 - 120.0

# A broad valley running north to south through the origin: the hills pressed down within 600 m of x = 0, the floor near -40 m.
valley = np.exp(-(X / 600.0) ** 2)
height = hills * (1.0 - 0.8 * valley) - 40.0 * valley

# The plateau: flat at -1 m within 300 m of the origin, blended into the hills over the next 200 m.
distance = np.sqrt(X * X + Z * Z)
blend = np.clip((distance - 300.0) / 200.0, 0.0, 1.0)
blend = blend * blend * (3.0 - 2.0 * blend)
height = -1.0 + (height + 1.0) * blend

# 16-bit samples over the field's own range.
low, high = float(height.min()), float(height.max())
samples16 = np.round((height - low) / (high - low) * 65535.0).astype(np.uint16)
samples16.tofile(directory / "field.r16")

json.dump({
    "heights": "field.r16",
    "samples": samples,
    "step": step,
    "origin": [-half, -half],
    "height_min": low,
    "height_max": high,
}, open(directory / "field.json", "w"), indent=2)
print(f"{samples} x {samples} samples at {step:g} m, heights {low:.1f} to {high:.1f} m")
```
  It prints `2049 x 2049 samples at 4 m, heights -21.5 to 318.3 m`, and takes a few seconds.
- **Repeating it:** an 8 km field ends 4 km from the camera, with nothing beyond. So the terrain the shaders see is the field repeated `tiles` times each way, 8 here, 64 km, mirrored in every other tile so that each edge meets its own reflection and there's no seam. Every position maps back to a sample of the one field (18.6); the field itself stands in tile 4 of 8 each way, unmirrored. The repeat shows from high up, as the valley recurring; a real world would stream more fields.
- **`Terrain`** holds the samples twice: on the GPU, two per word, and on the CPU, for `terrain_height_at`. And the **bounds**: for every node of the quadtree, its least and greatest sample, as two 16-bit numbers in a word. Level 0 has one node per quad, from its four corners; every level above one node per 2 × 2 below; levels stored finest first. 5.6 million nodes, 22 MB, built in a quarter of a second.
- **`load_terrain`** reads both files, builds the bounds, uploads everything, and fills in the `TerrainInfo` the shaders read: where the corner is, as a cell (it must lie on one, so that every sample's cell is exact), the heights' range, `height_offset`, which `main` sets to put the plateau 1 cm under the scene's floor, the two materials, and how many metres the textures repeat over.
- **`terrain_height_at`:** the drawn surface's height under a point, from the triangle the mesh shader draws there, for the camera's altitude.
- **`terrain_edge_pixels`,** 6: the finest level's samples project to no fewer than this many pixels apart; where they would come closer, the next level takes over. Within a level's range, its spacing runs from twelve pixels at the near end to six at the far. `main` turns it into `terrain_range`, the finest level's reach in metres, from the screen's height and the field of view, each frame.
- **`max_terrain_patches`,** 65,536: the most patches a frame can draw, and the most nodes a level of the selection can hold.

### Code
`game-engine/src/includes/terrain.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/shader_types.h"
#include "includes/vulkan_setup.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>
#include <vector>

// The terrain

// One height field: a square grid of 16-bit heights, a power of two plus one samples a side, `step` metres apart, repeated mirrored `tiles` times each way so that it reaches the horizon without an edge, the whole standing with its corner at the corner of a world cell. It's drawn as patches of 8 x 8 quads by a mesh shader (terrain.slang), at a level of detail that halves the samples used with every doubling of the distance (CDLOD, Strugar 2009); which patches, the cull decides (cull.slang). Shadow rays march it (shading.slangh). The quadtree of height bounds serves both: its level 0 has one node per quad of the finest grid, each level above one node per 2 x 2 below, up to one node for the whole field.
struct Terrain {
    Buffer heights;  // the samples, two per 32-bit word
    Buffer bounds;   // per quadtree node, its least and greatest sample, finest level first
    Buffer info;     // one TerrainInfo
    TerrainInfo data;
    std::vector<std::uint16_t> samples;  // the heights again, for the CPU
    std::uint32_t patch_levels = 0;      // levels of patches of the repeated terrain: its quadtree's, less the 3 that fit in a patch
    std::uint32_t node_count = 0;        // the field's quadtree nodes, all levels
    std::uint32_t patch_count = 0;       // every patch of the repeated terrain there could be
};

// A patch at level L reaches `range` x 2^L metres from the camera, before the next level takes over. The range is set per frame from the screen: the finest level's samples, `step` apart, project to terrain_edge_pixels apart at its range, and no fewer within it (main.cpp).
constexpr float terrain_edge_pixels = 6.0f;

// The most nodes a level of the selection can hold, and the most patches a frame can draw: the finest level's node count, when every one of its patches is drawn.
constexpr std::uint32_t max_terrain_patches = 65536;

// Loads the field described by `descriptor`, a JSON file naming the raw 16-bit samples next to it, their count a side, their spacing, the field's corner in metres from `origin` and what the samples' 0 and 65535 mean. The field repeats `tiles` times each way, a power of two, with the field itself in tile tiles / 2 each way, unmirrored. Every height is moved by `height_offset`, which puts the field under the scene. The corner must land on a cell corner. The materials are indices into the scene's materials, tiled every `uv_repeat` metres.
Terrain load_terrain(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    const std::filesystem::path &descriptor,
    const glm::dvec3 &origin,
    std::uint32_t tiles,
    float height_offset,
    std::uint32_t flat_material,
    std::uint32_t steep_material,
    float uv_repeat
);

// The terrain's height under `position`, in metres, from the triangles the terrain is drawn with, anywhere in the repeated terrain; the lowest height outside it.
double terrain_height_at(const Terrain &terrain, const glm::dvec3 &position);
```

`game-engine/src/terrain.cpp`:
```cpp
#include "includes/terrain.h"

#include "includes/cells.h"

#include <json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <span>
#include <stdexcept>

namespace {

    // Sample (x, z) of the field.
    std::uint16_t sample_at(const Terrain &terrain, std::uint32_t x, std::uint32_t z) {
        return terrain.samples[static_cast<std::size_t>(z) * terrain.data.samples + x];
    }

    // The field's sample at `sample` of the repeated terrain: flipped in every other tile (terrain.slangh's terrain_mirror).
    std::uint32_t mirrored(const Terrain &terrain, std::uint32_t sample) {
        const std::uint32_t quads = terrain.data.samples - 1;
        const std::uint32_t tile = std::min(sample / quads, terrain.data.tiles - 1);
        const std::uint32_t within = sample - tile * quads;
        return (tile & 1) != 0 ? quads - within : within;
    }

    // The quadtree's bounds, finest level first: level 0 takes each quad's four corners, every level above its four nodes below. Each node packs its least sample in the low half of a word and its greatest in the high half.
    std::vector<std::uint32_t> build_bounds(const Terrain &terrain) {
        std::vector<std::uint32_t> bounds;
        std::vector<std::uint32_t> level;

        for (std::uint32_t side = terrain.data.samples - 1, l = 0; l < terrain.data.quad_levels; side /= 2, ++l) {
            std::vector<std::uint32_t> next(static_cast<std::size_t>(side) * side);

            for (std::uint32_t z = 0; z < side; ++z) {
                for (std::uint32_t x = 0; x < side; ++x) {
                    std::uint32_t low = 0xFFFF;
                    std::uint32_t high = 0;

                    for (std::uint32_t k = 0; k < 4; ++k) {
                        const std::uint32_t cx = 2 * x + (k & 1);
                        const std::uint32_t cz = 2 * z + (k >> 1);
                        // Level 0 reads the quad's corners; the rest, the level below.
                        const std::uint32_t child = l == 0
                            ? sample_at(terrain, x + (k & 1), z + (k >> 1))
                            : level[static_cast<std::size_t>(cz) * side * 2 + cx];
                        low = std::min(low, l == 0 ? child : child & 0xFFFF);
                        high = std::max(high, l == 0 ? child : child >> 16);
                    }

                    next[static_cast<std::size_t>(z) * side + x] = low | (high << 16);
                }
            }

            bounds.insert(bounds.end(), next.begin(), next.end());
            level = std::move(next);
        }

        return bounds;
    }

}  // namespace

// Loading

Terrain load_terrain(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    const std::filesystem::path &descriptor,
    const glm::dvec3 &origin,
    std::uint32_t tiles,
    float height_offset,
    std::uint32_t flat_material,
    std::uint32_t steep_material,
    float uv_repeat
) {
    std::ifstream file(descriptor);

    if (!file) {
        throw std::runtime_error("can't open " + descriptor.string());
    }

    const nlohmann::json json = nlohmann::json::parse(file);

    Terrain terrain;
    terrain.data.samples = json.at("samples").get<std::uint32_t>();
    terrain.data.step = json.at("step").get<float>();
    terrain.data.height_min = json.at("height_min").get<float>();
    terrain.data.height_max = json.at("height_max").get<float>();
    terrain.data.height_offset = height_offset;
    terrain.data.flat_material = flat_material;
    terrain.data.steep_material = steep_material;
    terrain.data.uv_repeat = uv_repeat;

    // A side of 2^n + 1 samples halves exactly, level after level, down to one quad; repeated 2^k times, the terrain's quadtree has n + k + 1 levels.
    const std::uint32_t quads = terrain.data.samples - 1;

    if (quads == 0 || (quads & (quads - 1)) != 0 || quads < (1u << 3)) {
        throw std::runtime_error(descriptor.string() + ": a side must be a power of two plus one samples, at least 9");
    }

    if (tiles == 0 || (tiles & (tiles - 1)) != 0) {
        throw std::runtime_error("the terrain's tiles must be a power of two");
    }

    terrain.data.tiles = tiles;
    terrain.data.virtual_quads = quads * tiles;
    terrain.data.quad_levels = static_cast<std::uint32_t>(std::countr_zero(terrain.data.virtual_quads)) + 1;
    terrain.patch_levels = terrain.data.quad_levels - 3;

    // The corner: `origin` plus the field's own offset, less the tiles before the field's own, tiles / 2 each way, an even number, so the field stands unmirrored. It must be a cell's corner, so every sample's cell is exact.
    const auto corner = json.at("origin").get<std::array<double, 2>>();
    const double extent = static_cast<double>(quads) * terrain.data.step;
    const double before = static_cast<double>(tiles / 2) * extent;
    const CellPosition placed = to_cell(origin + glm::dvec3{corner[0] - before, 0.0, corner[1] - before});

    if (placed.offset != glm::vec3{0.0f}) {
        throw std::runtime_error(descriptor.string() + ": the terrain's corner must lie on a cell corner, a multiple of 64 m");
    }

    terrain.data.origin_cell = placed.cell;

    // The samples, as the file holds them: little-endian 16-bit, row by row.
    const std::filesystem::path heights_path = descriptor.parent_path() / json.at("heights").get<std::string>();
    std::ifstream heights(heights_path, std::ios::binary);
    terrain.samples.resize(static_cast<std::size_t>(terrain.data.samples) * terrain.data.samples);
    heights.read(reinterpret_cast<char*>(terrain.samples.data()), static_cast<std::streamsize>(terrain.samples.size() * sizeof(std::uint16_t)));

    if (!heights) {
        throw std::runtime_error("can't read " + heights_path.string());
    }

    // Two samples per word; an odd count leaves the last word half used.
    std::vector<std::uint16_t> padded = terrain.samples;
    padded.resize((padded.size() + 1) & ~std::size_t{1}, 0);
    terrain.heights = upload_buffer(device, gpu, queue, pool, std::as_bytes(std::span(padded)),
        vk::BufferUsageFlagBits::eShaderDeviceAddress);

    const std::vector<std::uint32_t> bounds = build_bounds(terrain);
    terrain.node_count = static_cast<std::uint32_t>(bounds.size());

    // Every patch of the repeated terrain: at the finest patch level, one per 8 x 8 quads, then a quarter as many per level up, to one.
    for (std::uint32_t side = terrain.data.virtual_quads / 8; side > 0; side /= 2) {
        terrain.patch_count += side * side;
    }
    terrain.bounds = upload_buffer(device, gpu, queue, pool, std::as_bytes(std::span(bounds)),
        vk::BufferUsageFlagBits::eShaderDeviceAddress);

    terrain.data.heights = terrain.heights.address;
    terrain.data.bounds = terrain.bounds.address;
    terrain.info = upload_buffer(device, gpu, queue, pool, std::as_bytes(std::span(&terrain.data, 1)),
        vk::BufferUsageFlagBits::eShaderDeviceAddress);

    return terrain;
}

double terrain_height_at(const Terrain &terrain, const glm::dvec3 &position) {
    const TerrainInfo &data = terrain.data;
    const glm::dvec3 corner = glm::dvec3(data.origin_cell) * cell_size;
    const double x = (position.x - corner.x) / data.step;
    const double z = (position.z - corner.z) / data.step;
    const double last = data.virtual_quads;

    if (x < 0.0 || z < 0.0 || x >= last || z >= last) {
        return data.height_min + data.height_offset;
    }

    // The quad under the point, and where in it; every quad is split from its (0, 0) corner to its (1, 1) corner, as the mesh shader draws it.
    const auto qx = static_cast<std::uint32_t>(x);
    const auto qz = static_cast<std::uint32_t>(z);
    const double fx = x - qx;
    const double fz = z - qz;
    const auto height = [&](std::uint32_t sx, std::uint32_t sz) {
        return data.height_min + sample_at(terrain, mirrored(terrain, sx), mirrored(terrain, sz)) * ((data.height_max - data.height_min) / 65535.0);
    };
    const double h00 = height(qx, qz);
    const double h11 = height(qx + 1, qz + 1);
    const double h = fx >= fz
        ? h00 + (height(qx + 1, qz) - h00) * fx + (h11 - height(qx + 1, qz)) * fz
        : h00 + (height(qx, qz + 1) - h00) * fz + (h11 - height(qx, qz + 1)) * fx;

    return h + data.height_offset;
}
```

## 18.3 The data: `shader_types.h`, `shared.slangh`

### Why
Every terrain shader needs the field; the cull needs lists for the patches; the frame needs the level ranges.

### How
- **`TerrainInfo`,** 72 bytes, pointers first: the samples and bounds, the corner's cell, the step, the heights' range and offset, the counts, the two materials, the texture repeat, the tiles, and the repeated terrain's quads per side.
- **`FrameData`** gains `terrain`, the `TerrainInfo`'s address; `terrain_range`, how far level 0 reaches; and `frame_index`, which frame this is, counting from 1. The cull marks each patch it draws with it (18.5). 376 bytes.
- **`CullTables`** gains four addresses: the phase's `patches`, its `patch_command` (a `VkDrawMeshTasksIndirectCommandEXT`: how many workgroups, then 1 and 1), the selection's working lists, and the shared `early_patches`. 128 bytes.
- **`alpha_opaque`, `alpha_mask` and `alpha_blend`** move to `shared.slangh`: the shadow code that reads them is shared now.

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
    vk::DeviceAddress terrain;          // the TerrainInfo
    float terrain_range;                // metres: how far level 0 of the terrain reaches; each level twice as far (terrain.h)
    std::uint32_t frame_index;          // which frame this is, counting from 1: the cull marks what it drew with it
};

static_assert(sizeof(FrameData) == 376);
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
static_assert(offsetof(FrameData, terrain) == 360);

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
    vk::DeviceAddress patches;        // the terrain patches this phase draws: quadtree node indices (terrain.h)
    vk::DeviceAddress patch_command;  // one VkDrawMeshTasksIndirectCommandEXT: how many patches, 1, 1
    vk::DeviceAddress nodes;          // the terrain selection's working lists: two runs of max_terrain_nodes_per_level
    vk::DeviceAddress early_patches;  // per terrain patch: the frame the early phase last drew it in
    std::uint32_t draw_count;
    std::uint32_t group_count;
};

static_assert(sizeof(CullTables) == 128);
static_assert(offsetof(CullTables, early_visible) == 80);
static_assert(offsetof(CullTables, draw_count) == 120);

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

// Terrain (terrain.h, terrain.slangh)

// One height field, which every terrain shader reads: its samples, its quadtree of height bounds, and where it stands in the world. The field repeats, mirrored at each edge, `tiles` times each way: the terrain the shaders see is that larger square, and every position in it maps back to a sample of the one field (terrain.slangh). Pointers first, so nothing needs padding.
//   - heights: samples x samples 16-bit heights, row by row, two per word; 0 is height_min and 65535 height_max, plus height_offset.
//   - bounds: per quadtree node of the one field, its least and greatest sample as two 16-bit numbers in one word, least in the low half. Level 0 is one node per quad, every level above one per 2 x 2 nodes below, coarsest last (terrain.h).
struct TerrainInfo {
    vk::DeviceAddress heights;
    vk::DeviceAddress bounds;
    glm::ivec3 origin_cell;        // the cell whose corner is the repeated terrain's corner
    float step;                    // metres between samples
    float height_min;              // metres, what a sample of 0 means
    float height_max;              // metres, what 65535 means
    float height_offset;           // metres added to every height: puts the field under the scene
    std::uint32_t samples;         // the field's samples per side: a power of two plus one
    std::uint32_t quad_levels;     // levels of the repeated terrain's quadtree: log2(virtual_quads) + 1
    std::uint32_t flat_material;   // index into the materials: level ground
    std::uint32_t steep_material;  // index into the materials: slopes
    float uv_repeat;               // metres per repeat of the materials' textures
    std::uint32_t tiles;           // how many times the field repeats each way: a power of two
    std::uint32_t virtual_quads;   // the repeated terrain's quads per side: (samples - 1) x tiles
};

static_assert(sizeof(TerrainInfo) == 72);
static_assert(offsetof(TerrainInfo, origin_cell) == 16);
static_assert(offsetof(TerrainInfo, samples) == 44);
static_assert(offsetof(TerrainInfo, tiles) == 64);
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

// glTF's three ways of using a material's alpha (AlphaMode in C++).
static const uint alpha_opaque = 0;
static const uint alpha_mask = 1;
static const uint alpha_blend = 2;

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

// One height field (src/includes/terrain.h), repeated mirrored `tiles` times each way: its samples, its quadtree of height bounds, and where the repeated terrain stands.
struct TerrainInfo {
    uint *heights;         // samples x samples 16-bit heights, two per word, row by row
    uint *bounds;          // per quadtree node of the field, least and greatest sample: low and high halves
    int3 origin_cell;      // the cell whose corner is the repeated terrain's corner
    float step;            // metres between samples
    float height_min;      // what a sample of 0 means, in metres
    float height_max;      // what 65535 means
    float height_offset;   // metres added to every height
    uint samples;          // the field's, per side
    uint quad_levels;      // levels of the repeated terrain's quadtree
    uint flat_material;    // materials index: level ground
    uint steep_material;   // materials index: slopes
    float uv_repeat;       // metres per repeat of the textures
    uint tiles;            // how many times the field repeats each way
    uint virtual_quads;    // the repeated terrain's quads per side
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
    TerrainInfo *terrain;              // the height field
    float terrain_range;               // metres: how far terrain level 0 reaches; each level twice as far
    uint frame_index;                  // which frame this is, from 1: the cull marks what it drew with it
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

## 18.4 Shading for every surface: `shading.slangh`, `mesh.slang`, `scan.slangh`

### Why
The terrain is lit, shadowed and hazed exactly like a glTF surface. The code for that was in `mesh.slang`, tied to its vertex layout; the terrain's fragment shader has another. And shadow rays have to march the terrain.

### How
- **`shading.slangh`** takes the BRDF, the shadows, the lights, the sky's light, ambient occlusion, energy compensation, specular antialiasing and aerial perspective out of `mesh.slang`, unchanged but for two functions that took a `VertexOutput`, which now take the position and the pixel. It ends with **`shade_surface`**: the sun, the directional lights, the cluster's lights, the sky and the air, exposed, for a `Surface` at a position. `mesh.slang`'s `shade_fragment` makes its surface as before and calls it.
- **`terrain_blocks`:** whether the terrain blocks a shadow ray, which `light_visibility` asks once the TLAS has let the ray through. The ray goes to sample space, x and z in samples from the terrain's corner and y in metres, and is clipped to the terrain's square and to `terrain_shadow_reach`, 4 km: a 300 m hill shadows ground 3.4 km away with the sun 5° up; what's farther is let through. Then the walk (Tevs et al. 2008; Dick et al. 2009), from the root node, with no stack:
  1. **Where the ray leaves** the node's square, through the nearer of the two sides ahead of it.
  2. **The node's heights.** If the ray's segment through the node is wholly above its greatest or below its least, the node is **passed**.
  3. **A quad** that isn't passed has its two triangles tested exactly (Möller and Trumbore 1997), the way the mesh shader splits it: a hit blocks the ray.
  4. **Otherwise descend** into the child the ray enters: the one a hair past the ray's current point, since that point can lie exactly on a boundary between children, having come from the node across it.
  5. **After a node is passed or finished,** step to the next node across the side the ray leaves by, at the coarsest level that has a node starting at that side: the new cell's coordinate across it, or one past it going the other way, has as many trailing zeros as there are levels to go up. A ray crossing a patch boundary rises to the patch level in one step; one crossing between two quads of a patch stays low.
  Both papers walk a pyramid of greatest heights only, and go up at most one level per step, when the coordinate crossed is even; Tevs et al. describe the full count of levels and set it aside for lack of a cheap way to compute it, which `firstbitlow` is now. The first version of step 5 went up one level after every quad; a grazing ray then revisited the same parent at every quad of a bumpy slope, and a low sun cost 36 ms a frame. This one costs 2.5. The least heights are ours too: a ray wholly under a node is passed as well.
- **A shadow bug, found from above.** Looking down on the box field, every roof had a band of dark speckles along its ridge, which flickered as the camera moved. The shadow view showed them to be shadow rays hitting the roof they started on. Since Chapter 11, a ray's origin has been moved off its surface by a few hundred units in the last place of its coordinates (Wächter and Binder 2019), measured in the TLAS's space, where the field's roofs are 12 cm up: an offset of 2 µm. But a ray meets an instance's triangles in that instance's own space, after the inverse of its transform, where the roof is 1.12 m up and 2 µm is sixteen units in the last place, too few to clear the triangle. So `shadow_ray_origin` now takes a **`ShadowSurface`**: the position, the flat normal, and, for a surface in the TLAS, the draw's model matrix, its inverse, and its cell's corner. An instanced surface is first moved off in its own space, through the inverse model matrix, which `DrawData` already holds as the normal matrix's transpose, then in the TLAS's, where the instance transform's rounding needs an offset of its own. `shade_shadowed`, `shade_light`, `shade_local_lights` and `shade_surface` take the `ShadowSurface` in place of a position and a normal.
- **The flat normal from the triangle itself.** The same look showed a second weakness: the flat normal came from derivatives of the position across neighbouring pixels, and along a silhouette the neighbouring pixels belong to whatever is behind. `mesh.slang`'s `triangle_normal` reads the triangle's three vertices instead, through `SV_PrimitiveID`, the triangle's index among the draw's, for which Slang declares the Geometry capability (18.1); its sign doesn't matter, since `shadow_ray_origin` picks the light's side.
- **`light_count_heat`** moves here too, taking the pixel and the position.
- **`group_exclusive_scan`,** in `scan.slangh`: prefix sums over one workgroup's threads, for the patch selection, with the same rounds as `exclusive_scan`.

### Code
`game-engine/shaders/shading.slangh`:
```slang
// Shading, shared by every surface shader: mesh.slang's glTF surfaces and terrain.slang's ground. The glTF BRDF, the shadows, the lights, the sky's light, ambient occlusion, energy compensation, specular antialiasing and the air in between. Each shader includes it after shared.slangh and atmosphere.slangh, and makes a Surface for its pixel.

#include "terrain.slangh"

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
    float3 energy_compensation;  // what the specular reflection is scaled by (energy_compensation)
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
    const float3 compensated = specular * surface.energy_compensation;
    const float3 metal = compensated * (surface.base_color + (1.0 - surface.base_color) * fresnel);
    const float3 dielectric = lerp(surface.base_color / pi, compensated, 0.04 + 0.96 * fresnel);
    const float3 brdf = lerp(dielectric, metal, surface.metallic);

    // Light falling at an angle spreads over more surface: the n.l factor.
    return brdf * illuminance * n_dot_l;
}

// Shadows

// How far a ray toward the sun, or another light infinitely far away, may go.
static const float infinite_distance = 1e9;

// A ray starting exactly on a surface can hit that same surface: the hit point's rounding puts it a hair below. This moves the origin off the surface along its geometric normal by up to 256 units in the last place (ULPs) of each coordinate: an offset that grows with the coordinates, so it suits any distance from the origin, where a fixed distance would be too much near it and too little far away. The constants are the authors', found by experiment. From "A Fast and Robust Method for Avoiding Self-Intersection" (Wachter and Binder, Ray Tracing Gems, 2019).
float3 offset_ray_origin(float3 position, float3 normal) {
    const float near_origin = 1.0 / 32.0;
    const float float_scale = 1.0 / 65536.0;
    const float int_scale = 256.0;

    float3 offset;
    for (int axis = 0; axis < 3; ++axis) {
        // Step the float's bits, as an integer, away from the surface.
        const int step = int(int_scale * normal[axis]);
        const float stepped = asfloat(asint(position[axis]) + (position[axis] < 0.0 ? -step : step));

        // Close to 0 a few units in the last place are tiny, so add a small fixed distance there instead.
        offset[axis] = abs(position[axis]) < near_origin ? position[axis] + float_scale * normal[axis] : stepped;
    }

    return offset;
}

// The alpha of a masked or blended triangle a ray met, at the hit point. The hit's barycentric coordinates weight the triangle's three vertices; the texture is read at full resolution, since there are no neighbouring pixels to pick a mip level from. Rays from neighbouring pixels can hit different materials, so the texture's heap index differs between them: that's fine, since descriptor heap access is non-uniform unless the SPIR-V marks it uniform, and Slang doesn't.
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

// Whether the ray from `o` along `d` meets the triangle a, b, c between t_min and t_max along it (Moller and Trumbore 1997): the ray's parameter and the hit's barycentric coordinates, solved together.
bool hits_triangle(float3 o, float3 d, float3 a, float3 b, float3 c, float t_min, float t_max) {
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 p = cross(d, ac);
    const float determinant = dot(ab, p);

    if (abs(determinant) < 1e-12) {
        return false;
    }

    const float inverse = 1.0 / determinant;
    const float3 s = o - a;
    const float u = dot(s, p) * inverse;

    if (u < 0.0 || u > 1.0) {
        return false;
    }

    const float3 q = cross(s, ab);
    const float v = dot(d, q) * inverse;

    if (v < 0.0 || u + v > 1.0) {
        return false;
    }

    const float t = dot(ac, q) * inverse;
    return t >= t_min && t <= t_max;
}

// How far a shadow ray marches the terrain at most, in metres. A 300 m hill shadows the ground about 3.4 km away with the sun 5 degrees up; what's farther is let through. The whole terrain, 64 km, would cost a grazing ray many times the steps for shadows a few minutes of twilight could show.
static const float terrain_shadow_reach = 4096.0;

// Whether the terrain blocks the ray from `origin` (camera-relative) along `direction` within `max_distance`, up to terrain_shadow_reach: the height field marched through its quadtree of bounds, one node at a time in the ray's order, with no stack, as Tevs et al. (2008) and Dick et al. (2009) march a pyramid of greatest heights. At each node, the ray's segment through the node's square is compared with the node's heights: wholly above its greatest or, our addition, below its least, and the node is passed over; otherwise the walk descends to the child the ray enters, down to a quad, whose two triangles are tested exactly. After passing or finishing a node, the walk steps to the next node across the side the ray leaves by, as coarse as that side allows: coarser where there may be less to test.
//
// The ray is traced in sample space: x and z in samples from the repeated terrain's corner, y in metres, so the quadtree's cells are unit squares at level 0, and `t` is still metres along the ray. The ray is first clipped to the terrain; what leaves it blocks nothing.
bool terrain_blocks(FrameData *frame, float3 origin, float3 direction, float max_distance) {
    TerrainInfo *terrain = frame.terrain;
    const float2 camera = terrain_camera(frame, terrain);
    const float3 o = float3(camera.x + origin.x / terrain.step, origin.y + terrain_camera_height(frame), camera.y + origin.z / terrain.step);
    const float3 d = float3(direction.x / terrain.step, direction.y, direction.z / terrain.step);
    const float side = float(terrain.virtual_quads);

    // Where the ray is inside the field's square: the overlap of its spans between the two x planes and the two z planes.
    const float3 inverse = 1.0 / d;
    const float2 t_low = (float2(0.0) - o.xz) * inverse.xz;
    const float2 t_high = (float2(side) - o.xz) * inverse.xz;
    const float2 t_near = min(t_low, t_high);
    const float2 t_far = max(t_low, t_high);
    float t = max(max(t_near.x, t_near.y), 0.0);
    const float t_end = min(min(t_far.x, t_far.y), min(max_distance, terrain_shadow_reach));

    if (!(t < t_end)) {
        return false;
    }

    const uint top = terrain.quad_levels - 1;
    uint level = top;
    int2 cell = int2(0);
    const int2 step = int2(d.x < 0.0 ? -1 : 1, d.z < 0.0 ? -1 : 1);

    for (uint iteration = 0; iteration < 512 && t < t_end; ++iteration) {
        const float size = float(1u << level);

        // Where the ray leaves this node's square: through the nearer of its x and z sides ahead.
        const float2 ahead = (float2(cell) + float2(step.x > 0 ? 1.0 : 0.0, step.y > 0 ? 1.0 : 0.0)) * size;
        const float2 t_sides = (ahead - o.xz) * inverse.xz;
        const float t_exit = min(min(t_sides.x, t_sides.y), t_end);
        const float2 heights = terrain_node_heights(terrain, level, uint2(cell));
        const float y_in = o.y + d.y * t;
        const float y_out = o.y + d.y * t_exit;
        bool passed = max(y_in, y_out) < heights.x || min(y_in, y_out) > heights.y;

        if (!passed && level == 0) {
            // The quad's two triangles, each from its (0, 0) corner to its (1, 1) corner.
            const float3 p00 = float3(cell.x, terrain_height(terrain, uint2(cell)), cell.y);
            const float3 p10 = float3(cell.x + 1, terrain_height(terrain, uint2(cell) + uint2(1, 0)), cell.y);
            const float3 p01 = float3(cell.x, terrain_height(terrain, uint2(cell) + uint2(0, 1)), cell.y + 1);
            const float3 p11 = float3(cell.x + 1, terrain_height(terrain, uint2(cell) + uint2(1, 1)), cell.y + 1);

            if (hits_triangle(o, d, p00, p01, p11, t, t_exit) || hits_triangle(o, d, p00, p11, p10, t, t_exit)) {
                return true;
            }

            passed = true;
        }

        if (passed) {
            // On to the next node across the side the ray leaves by, at the coarsest level that has a node starting at that side: the new cell's coordinate across it, or one past it when going the other way, has as many trailing zeros as there are levels to go up. Tevs et al. describe this count and settle for one level when the coordinate is even, for want of a cheap way to compute it.
            t = t_exit;
            const bool across_x = t_sides.x < t_sides.y;
            cell += across_x ? int2(step.x, 0) : int2(0, step.y);

            if (any(cell < 0) || any(cell >= int2(int(side) >> level))) {
                return false;
            }

            const int along = across_x ? cell.x : cell.y;
            const uint rise = min(firstbitlow(uint(along + ((across_x ? step.x : step.y) < 0 ? 1 : 0))), top - level);
            cell >>= rise;
            level += rise;
        } else {
            // Down a level, into the child the ray enters first: the one a hair past where the ray is now, since that point can lie exactly on a boundary between children, having come from the node across it.
            --level;
            const float2 p = o.xz + d.xz * (t + 1e-4 * (t_exit - t));
            cell = clamp(int2(floor(p / float(1u << level))), cell * 2, cell * 2 + 1);
        }
    }

    return false;
}

// How much of a light gets from `origin` to `distance` along `direction`: 0 when something solid is in the way, otherwise the share every see-through layer on the way lets through. A ray query walks the TLAS and BLASes:
//   - an opaque triangle ends it at once: any blocking hit will do,
//   - a masked one comes back as a candidate, which blocks where its alpha reaches the cutoff, and lets the light through its cut-out texels,
//   - a blended one comes back as a candidate that lets 1 - alpha of the light through, as the transparency pass's reveal sum does. It never ends the ray: the light goes on, dimmed, to whatever is behind.
// Then the terrain is marched, if nothing solid was met.
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

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        return 0.0;
    }

    // Then the terrain, which is in no acceleration structure. The ray goes back to camera-relative space: the TLAS's is offset from it (shadow_ray_origin).
    return terrain_blocks(frame, origin - frame.tlas_offset, direction, distance) ? 0.0 : transmittance;
}

// Where shadow rays start from: a surface's camera-relative position and its flat normal, the triangle's own (an interpolated or normal-mapped normal can disagree about which side a light is on), and, for a surface that's in the TLAS, the way into its instance's own space.
struct ShadowSurface {
    float3 position;
    float3 face_normal;
    bool instanced;      // in the TLAS: the rays meet its triangles in the instance's space, so the origin is moved off the surface there too
    float3 cell_corner;  // camera-relative: the corner of the draw's cell, where its model matrix measures from
    float4x4 model;      // the draw's model matrix
    float4x4 to_model;   // its inverse
};

// Where a ray toward the light `l` starts: off the surface on the light's side, along the flat normal, by offset_ray_origin.
//
// The offset is a few hundred units in the last place of the coordinates, so it must be measured where the ray is really intersected. The TLAS's space is measured from its origin cell, near the camera, not from the camera itself (acceleration.h): tlas_offset, the camera's position in it, moves the camera-relative position there first. But a ray meets an instance's triangles in that instance's own space, after the inverse of its transform: there a roof is a metre up, while in the TLAS's space it may be a few centimetres up, and an offset measured there is sixteen times too small to clear the triangle it starts on. So an instanced surface is first moved off in its own space, through the inverse model matrix, and then in the TLAS's, where the instance transform's rounding needs an offset of its own. The terrain is in no instance: its rays start from the exact height field, and meet only the TLAS.
float3 shadow_ray_origin(FrameData *frame, ShadowSurface surface, float3 l) {
    const float3 n = dot(surface.face_normal, l) >= 0.0 ? surface.face_normal : -surface.face_normal;
    float3 position = surface.position;

    if (surface.instanced) {
        // Normals go to the instance's space by the model matrix's transpose: the inverse of how they leave it.
        const float3 in_model = mul(surface.to_model, float4(position - surface.cell_corner, 1.0)).xyz;
        const float3 n_model = normalize(mul(transpose((float3x3)surface.model), n));
        position = surface.cell_corner + mul(surface.model, float4(offset_ray_origin(in_model, n_model), 1.0)).xyz;
    }

    return offset_ray_origin(position + frame.tlas_offset, n);
}

// shade(), times how much of the light gets through. A ray is only traced when the light could reach the surface at all.
float3 shade_shadowed(Surface surface, FrameData *frame, ShadowSurface from, float3 l, float3 illuminance, float distance) {
    if (dot(surface.normal, l) <= 0.0 || all(illuminance == 0.0)) {
        return float3(0.0);
    }

    const float reaching = light_visibility(frame, shadow_ray_origin(frame, from, l), l, distance);
    return reaching > 0.0 ? shade(surface, l, illuminance) * reaching : float3(0.0);
}

// Lights

// The direction toward a light, how far away it is, and the illuminance it gives here, following KHR_lights_punctual. Point and spot lights fade with the square of the distance, and smoothly to nothing at `range`, through Karis's window (2013), (1 - (d / range)^4)^2, which reaches zero with zero slope. (Karis also divides by d^2 + 1 rather than d^2, to keep a light finite at its centre; KHR_lights_punctual's inverse square has no + 1.) Spot lights also fade from the inner cone to the outer one. A directional light is infinitely far away.
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

    const float ratio = distance / light.range;
    const float window = saturate(1.0 - ratio * ratio * ratio * ratio);
    float attenuation = window * window / distance2;

    if (light.type == light_spot) {
        const float cone = saturate(dot(light.direction, -l) * light.spot_scale + light.spot_offset);
        attenuation *= cone * cone;
    }

    return light.intensity * attenuation;
}

// One light's shadowed contribution at the surface.
float3 shade_light(Surface surface, FrameData *frame, Light light, ShadowSurface from) {
    float3 l;
    float distance;
    const float3 illuminance = punctual_light(frame, light, from.position, l, distance);
    return shade_shadowed(surface, frame, from, l, illuminance, distance);
}

// Point and spot lights

// The most shadow rays a pixel's budget can hold (FrameData's shadow_ray_budget, 0 for no budget).
static const uint max_shadow_ray_budget = 16;

float luminance(float3 color) {
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

// How much light a light brings to the surface at `position`: its illuminance's luminance, times n.l, without its shadow. A light that brings none has no reason to trace a ray. Cheaper than the BRDF, and close enough to rank the lights by.
float arriving(Surface surface, Light light, float3 illuminance, float3 l) {
    return luminance(illuminance) * max(dot(surface.normal, l), 0.0);
}

// The next light in a cluster's bits: each word holds 32 of the visible lights' bits, and firstbitlow finds the lowest one still set. Returns the light's index among all the lights, and clears its bit in `remaining`.
uint next_light(FrameData *frame, uint word, inout uint remaining) {
    const uint bit = firstbitlow(remaining);
    remaining &= remaining - 1;  // clears that bit
    return frame.visible_lights[1 + word * 32 + bit];
}

// The point and spot lights whose range reaches the cluster of the pixel at `pixel`, in the lights' order, each with its shadow ray: a light that brings the surface no light at all traces none. With a budget of N rays, only the N lights that bring it the most light, without shadows, trace theirs; the rest light the pixel unshadowed.
float3 shade_local_lights(Surface surface, FrameData *frame, ShadowSurface from, float2 pixel) {
    const float3 position = from.position;
    uint *bits = cluster_bits(frame, pixel, dot(position, frame.camera_forward));
    const uint visible_count = frame.visible_lights[0];
    const uint budget = min(frame.shadow_ray_budget, max_shadow_ray_budget);

    // With a budget, a first pass finds the cutoff: what the Nth strongest light brings here. `strongest` holds the 16 strongest so far, in order; each light goes in by a fixed chain of compare-and-swaps, every index known when the shader is compiled, so the compiler can keep the array in registers.
    float cutoff = 0.0;

    // A cluster with no more lights than the budget traces them all anyway: counting its bits is far cheaper than the first pass.
    uint cluster_lights = 0;

    if (budget != 0) {
        for (uint word = 0; word * 32 < visible_count; ++word) {
            cluster_lights += countbits(bits[word]);
        }
    }

    if (cluster_lights > budget) {
        float strongest[max_shadow_ray_budget];

        [unroll]
        for (uint i = 0; i < max_shadow_ray_budget; ++i) {
            strongest[i] = 0.0;
        }

        for (uint word = 0; word * 32 < visible_count; ++word) {
            uint remaining = bits[word];

            while (remaining != 0) {
                const Light light = frame.lights[next_light(frame, word, remaining)];
                float3 l;
                float distance;
                const float3 illuminance = punctual_light(frame, light, position, l, distance);
                float value = arriving(surface, light, illuminance, l);

                [unroll]
                for (uint i = 0; i < max_shadow_ray_budget; ++i) {
                    const float larger = max(strongest[i], value);
                    value = min(strongest[i], value);
                    strongest[i] = larger;
                }
            }
        }

        [unroll]
        for (uint i = 0; i < max_shadow_ray_budget; ++i) {
            cutoff = i + 1 == budget ? strongest[i] : cutoff;
        }
    }

    // Every light, shaded once; a ray for those at or above the cutoff. With no budget the cutoff is 0, and every light that brings any light traces. Ties at the cutoff all trace, so a budget can be exceeded by a light or two that give exactly the same light. The two passes work the amounts out separately, and the compiler may round them a hair differently: the cutoff is lowered by a millionth of itself, so the Nth light always makes it.
    float3 radiance = 0.0;

    for (uint word = 0; word * 32 < visible_count; ++word) {
        uint remaining = bits[word];

        while (remaining != 0) {
            const Light light = frame.lights[next_light(frame, word, remaining)];
            float3 l;
            float distance;
            const float3 illuminance = punctual_light(frame, light, position, l, distance);
            const float value = arriving(surface, light, illuminance, l);

            if (value <= 0.0) {
                continue;
            }

            const float3 lit = shade(surface, l, illuminance);
            radiance += value >= cutoff * 0.999999
                ? lit * light_visibility(frame, shadow_ray_origin(frame, from, l), l, distance)
                : lit;
        }
    }

    return radiance;
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

// Ambient occlusion

// The rotation that turns `from` into `to`, applied to `v`: Rodrigues' formula rewritten without angles (Moller and Hughes 1999), from the cross and dot products alone. `from` and `to` are never opposite here: the bent normal averages directions in the hemisphere around the normal.
float3 rotate_from_to(float3 from, float3 to, float3 v) {
    const float3 axis = cross(from, to);
    const float c = dot(from, to);

    if (c > 0.9999) {
        return v;
    }

    return v * c + cross(axis, v) + axis * (dot(axis, v) / (1.0 + c));
}

// Ambient occlusion counts light that's blocked, but light also bounces off the occluders, and more so the brighter they are. Jimenez et al.'s fit, from the same GTAO paper, brightens the visibility by the surface's own albedo, standing in for its surroundings'.
float3 multi_bounce(float visibility, float3 albedo) {
    const float3 a = 2.0404 * albedo - 0.3324;
    const float3 b = -4.7951 * albedo + 0.6417;
    const float3 c = 2.7552 * albedo + 0.6903;
    return max(float3(visibility), ((visibility * a + b) * visibility + c) * visibility);
}

// How much of the sky's reflection a partly occluded point still sees (Lagarde and de Rousiers 2014): smooth surfaces, looking straight on, keep more of it than occlusion alone suggests; rough ones lose about as much. Its roughness is GGX's alpha, roughness squared.
float specular_occlusion(float n_dot_v, float visibility, float alpha) {
    return saturate(pow(n_dot_v + visibility, exp2(-16.0 * alpha - 1.0)) - 1.0 + visibility);
}

// The sky's light reflected toward the viewer.
//   - Diffuse: a Lambertian surface reflects base color / pi of the irradiance falling on it, read along `irradiance_normal` (the bent normal: the direction the open sky lies in), dimmed by the visibility and brightened again by multiple bounces.
//   - Specular, the "split sum": the light (the prefiltered sky along the reflected ray, at the mip level for this roughness) times how much the BRDF reflects overall (the table, as a scale and bias on F0), dimmed by the specular occlusion.
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

// Energy compensation

// The GGX specular reflection counts light that bounces once off the microfacets; on a rough surface, much of it bounces again, between them, and still leaves. Without it, rough metals come out too dark: a fully rough white metal reflects only about 40% of the light. The BRDF table's two numbers, at f0 = 1, add up to E, the share that one bounce leaves with; scaling the reflection by 1 + f0 (1 / E - 1) puts the rest back, in proportion to how much the surface reflects at all. Kulla and Conty's (2017) idea, in Turquin's (2019) simpler scaled form, which Filament uses.
float3 energy_compensation(FrameData *frame, float3 base_color, float metallic, float roughness, float n_dot_v) {
    const SamplerState clamped = SamplerState.Handle(uint2(frame.clamp_sampler, 0));
    const Texture2D brdf_lut = Texture2D.Handle(uint2(frame.brdf_lut, 0));
    const float2 brdf = brdf_lut.SampleLevel(clamped, float2(n_dot_v, roughness), 0.0).rg;
    const float3 f0 = lerp(float3(0.04), base_color, metallic);
    return 1.0 + f0 * (1.0 / max(brdf.x + brdf.y, 1e-3) - 1.0);
}

// Specular antialiasing

// A pixel shows the average of the light reflected over its whole footprint on the surface, but it's shaded at one point. Where the normal changes faster than a pixel can follow, a highlight smaller than the pixel lands on the sample in one frame and misses it in the next: sparkles that crawl as the camera moves. That happens in two places:
//   - a surface curving, or seen from far away, faster than the pixels
//   - a normal map's bumps, smaller than a texel of the mip being read
// Both mean the normals within the pixel spread out. To the BRDF, that spread is roughness: alpha squared is twice the variance of the microfacet slopes, exactly for Beckmann's distribution and closely enough for GGX's (Kaplanyan et al. 2016), so the spread's variance adds to it. A rougher, wider highlight that every pixel catches part of.

// The variance of the pixel filter, in pixels squared, and the most the curvature may add: differences between neighbouring pixels estimate the change only roughly, and the cap keeps a bad estimate from turning a mirror into chalk (Kaplanyan et al. 2016; Tokuyoshi and Kaplanyan 2021).
static const float pixel_filter_variance = 0.15915494;  // 1 / (2 pi)
static const float curvature_limit = 0.18;

// GGX's alpha, widened by both spreads.
//   Curvature: how fast the vertex normal changes from pixel to pixel, both ways, gives the variances of the normals across the pixel's footprint along the two screen axes. Their sum bounds the largest variance in any direction, so a round highlight widened by it is wide enough (Tokuyoshi and Kaplanyan 2021, equation 13, their conservative isotropic form). The vertex normal, as Filament and Unity's HDRP use: the surface's shape. The map's bumps are the other spread; their pixel-to-pixel differences would add a noisy, flickering estimate on top.
//   Normal map: the spread its mips measured (Toksvig 2005, mips.slang), as the standard deviation s of the normals' angle; its variance s^2, twice, like the curvature's.
float antialiased_alpha(float alpha, float3 geometric_normal, float map_spread) {
    const float3 dn_dx = ddx(geometric_normal);
    const float3 dn_dy = ddy(geometric_normal);
    const float curvature = 2.0 * pixel_filter_variance * (dot(dn_dx, dn_dx) + dot(dn_dy, dn_dy));
    const float bumps = 2.0 * map_spread * map_spread;

    return sqrt(saturate(alpha * alpha + min(curvature, curvature_limit) + bumps));
}

// Aerial perspective

// The light the air between the camera and `position` (camera-relative) adds, and, in `transmittance`, the share of the surface's light it lets through: the aerial perspective volumes, at the point's place on the screen and its distance. Closer than the first slice's far edge, a share of the first slice's air.
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

// The light count view

// How many lights the cluster of the pixel at `pixel`, at camera-relative `position`, holds, as a heat map: black for none, dark blue for 1 or 2, then blue, green, yellow and red at 4, 8, 16 and 32 or more.
float3 light_count_heat(FrameData *frame, float2 pixel, float3 position) {
    uint *bits = cluster_bits(frame, pixel, dot(position, frame.camera_forward));
    const uint visible_count = frame.visible_lights[0];
    uint count = 0;

    for (uint word = 0; word * 32 < visible_count; ++word) {
        count += countbits(bits[word]);
    }

    if (count == 0) {
        return float3(0.0);
    }

    const float3 colors[5] = {float3(0.0, 0.0, 0.5), float3(0.0, 0.3, 1.0), float3(0.0, 0.9, 0.2), float3(1.0, 0.9, 0.0), float3(1.0, 0.0, 0.0)};
    const float level = clamp(log2(float(count)), 0.0, 5.0) - 1.0;  // 2 -> 0, 4 -> 1, ..., 32 -> 4
    const float position = clamp(level, 0.0, 3.999);
    const uint i = uint(position);
    return lerp(colors[i], colors[i + 1], position - float(i));
}

// A surface's light

// The exposed radiance of `surface` at `from`, seen at `pixel`: the sun and the file's directional lights, then the point and spot lights that reach the pixel's cluster, all shadowed, with rays from `from`; the sky's light, darkened by `visibility` and read along `irradiance_normal`; then the air in between, which dims it all, emission included, and adds its own.
float3 shade_surface(
    Surface surface, FrameData *frame, ShadowSurface from, float2 pixel,
    float roughness, float visibility, float3 irradiance_normal, float3 emissive
) {
    const float3 position = from.position;
    float3 radiance = shade_shadowed(surface, frame, from, frame.sun_direction, frame.sun_illuminance, infinite_distance);

    for (uint i = 0; i < frame.directional_light_count; ++i) {
        radiance += shade_light(surface, frame, frame.lights[i], from);
    }

    radiance += shade_local_lights(surface, frame, from, pixel);

    // Indirect light from the sky, darkened by occlusion. Ambient occlusion only ever reaches this indirect light: the sun and the lights are direct, and only a shadow can block them.
    radiance += shade_environment(surface, frame, roughness, visibility, irradiance_normal);

    // The air between the camera and the surface, with the simulated sky: it dims the surface's light, emission included, and adds its own. The photograph has no air to go with it.
    float3 transmittance = 1.0;
    float3 inscatter = 0.0;

    if (frame.atmosphere != 0) {
        inscatter = aerial_perspective(frame, position, transmittance);
    }

    // Exposure scales nits into the tone mapper's range here, before the 16-bit HDR image could overflow. glTF defines emission in nits, but, as its spec notes many engines do, we take it as already exposed: an emissive value of 1 shows as near-white, whatever the exposure.
    return (radiance * transmittance + inscatter) * frame.exposure + emissive * transmittance;
}
```

`game-engine/shaders/mesh.slang`:
```slang
// Draws one glTF primitive: its vertices come from the scene's vertex buffer, its place in the world from its DrawData, and its surface from its glTF material, whose textures are read from the descriptor heap. Shaded by shading.slangh: glTF's physically based BRDF, lit by the sun and the file's lights, with ray-traced shadows, and by the sky around the scene, already exposed. Three fragment shaders:
//   prepassMain      the depth prepass's vertex normal
//   fragmentMain     opaque and masked surfaces, into the HDR image
//   transparentMain  blended surfaces, into weighted blended transparency's sums

// Data shared with C++ (src/includes/shader_types.h)

#include "shared.slangh"
#include "atmosphere.slangh"
#include "shading.slangh"

// In a descriptor heap pipeline, the push_constant block is where push data lands.
[[vk::push_constant]]
ConstantBuffer<PushData> push;

// The alpha mode this pipeline was built for (AlphaMode in C++): 0 opaque, 1 mask, 2 blend. A specialization constant: its value is fixed when the pipeline is created, so each pipeline's fragment shader keeps only the code its mode needs.
[vk::constant_id(0)]
const uint alpha_mode = 0;

// Stage interface

// What the vertex shader hands to the rasterizer. SV_Position is the clip-space position; every other field but draw_index is interpolated across the triangle. Vulkan requires integer fields to be flat, which nointerpolation makes them.
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

// Vertex shader

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
// `map_spread`: how much the normal map's normals spread here, as the standard deviation of their angle, which the mips keep in its alpha as 1 - spread (mips.slang); 0 without a map.
float3 surface_normal(VertexOutput input, Material material, bool front_face, bool apply_normal_map, out float map_spread) {
    float3 normal = input.normal;
    map_spread = 0.0;

    // cross(ddy, ddx), not cross(ddx, ddy): Vulkan's screen Y points down, so this order is the one that points toward the camera.
    if (all(normal == 0.0)) {
        normal = cross(ddy(input.relative_position), ddx(input.relative_position));
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

    // Texture coordinates that don't change across the triangle give no frame at all; the plain normal is all we have then.
    if (!mapped || all(tangent == 0.0) || all(bitangent == 0.0)) {
        return normal;
    }

    // All three axes must be unit length, or the map's tilt is scaled with them. The normal already is; the other two grow and shrink with the model matrix, and interpolation shortens them between vertices.
    tangent = normalize(tangent);
    bitangent = normalize(bitangent);

    // The map stores each component in 0..1; unpack to -1..1. normal_scale scales the tilt: X and Y only, as glTF specifies.
    const float4 texel = sample_slot(material.normal, input);
    float3 tangent_space = texel.xyz * 2.0 - 1.0;
    tangent_space.xy *= material.normal_scale;
    map_spread = 1.0 - texel.w;

    return normalize(tangent * tangent_space.x + bitangent * tangent_space.y + normal * tangent_space.z);
}

// Depth and normal prepass

// The interpolated vertex normal, facing the viewer on a double-sided material's back face; flat when the file has no normals. This is the surface at the scale the mesh describes it, which ambient occlusion searches against: a normal map's detail isn't in the depth buffer.
float3 vertex_normal(VertexOutput input, Material material, bool front_face) {
    float3 normal = input.normal;

    if (all(normal == 0.0)) {
        normal = cross(ddy(input.relative_position), ddx(input.relative_position));
    }

    normal = normalize(normal);
    return material.double_sided != 0 && !front_face ? -normal : normal;
}

// The prepass draws every opaque and masked surface first, writing only its depth and its vertex normal, octahedrally encoded. Masked surfaces cut out their transparent texels here too, so the depth buffer holds exactly the surfaces the lighting pass will shade.
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

// The triangle's normal

// The flat normal of triangle `primitive` of `draw`, from its three vertices: where shadow rays start, off the surface along it (shading.slangh's shadow_ray_origin picks the light's side, so its sign doesn't matter). Derivatives of the position across neighbouring pixels would give it more cheaply, but along a silhouette the neighbouring pixels belong to whatever is behind.
float3 triangle_normal(FrameData *frame, DrawData draw, uint primitive) {
    const uint first = draw.first_index + primitive * 3;
    const float3 p0 = mul(draw.model, float4(frame.vertices[int(frame.indices[first]) + draw.vertex_offset].position, 1.0)).xyz;
    const float3 p1 = mul(draw.model, float4(frame.vertices[int(frame.indices[first + 1]) + draw.vertex_offset].position, 1.0)).xyz;
    const float3 p2 = mul(draw.model, float4(frame.vertices[int(frame.indices[first + 2]) + draw.vertex_offset].position, 1.0)).xyz;
    const float3 n = cross(p1 - p0, p2 - p0);
    return dot(n, n) > 0.0 ? normalize(n) : float3(0.0, 1.0, 0.0);
}

// Shading a fragment

// The surface at this fragment: its exposed radiance (or one input, in a debug view), and its alpha. The lighting pass writes it as it is; the transparency pass adds it into its sums. `front_face`: whether this triangle faces the camera; `primitive`: which of the draw's triangles it is.
float4 shade_fragment(VertexOutput input, bool front_face, uint primitive) {
    FrameData *frame = push.frame;
    const Material material = frame.materials[frame.draws[input.draw_index].material];

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

    float map_spread;
    const float3 normal = surface_normal(input, material, front_face, frame.view != view_vertex_normal, map_spread);

    // Roughness, widened where the normals spread within the pixel. A perfectly smooth surface would reflect a punctual light from a single point, too small for any pixel to catch; a floor on roughness keeps highlights visible.
    const float floored = max(roughness, 0.045);
    const float alpha = antialiased_alpha(floored * floored, vertex_normal(input, material, front_face), map_spread);
    const float shading_roughness = sqrt(alpha);

    // Ambient occlusion, from the AO pass's image at this pixel. It combines with the occlusion map by min, not product: both estimate the same thing, at two scales. The bent normal is a deflection from the vertex normal; turning the shading normal by the same deflection keeps the normal map's detail. See-through surfaces aren't in the prepass, so the image there holds whatever is behind them: they use the map alone.
    const float4 gtao = Texture2D.Handle(uint2(frame.ambient_occlusion, 0)).Load(int3(int2(input.position.xy), 0));
    float visibility = occlusion;
    float3 irradiance_normal = normal;

    if (frame.ao_enabled != 0 && alpha_mode != alpha_blend) {
        visibility = min(occlusion, gtao.w);
        irradiance_normal = normalize(rotate_from_to(vertex_normal(input, material, front_face), gtao.xyz, normal));
    }

    // The debug views show one input each. Directions are shown as colors: each component's -1..1 mapped to 0..1.
    switch (frame.view) {
        case view_base_color: return base_color;
        case view_normal:
        case view_vertex_normal: return float4(normal * 0.5 + 0.5, 1.0);
        case view_metallic: return float4(metallic.xxx, 1.0);
        case view_roughness: return float4(shading_roughness.xxx, 1.0);  // as shaded: widened
        case view_occlusion: return float4(occlusion.xxx, 1.0);
        case view_emissive: return float4(emissive, 1.0);
        case view_ambient_occlusion: return float4(gtao.www, 1.0);
        case view_light_count: return float4(light_count_heat(frame, input.position.xy, input.relative_position), 1.0);
        default: break;
    }

    // Where shadow rays start: off the triangle, along its own flat normal, in the draw's space and the TLAS's (shading.slangh).
    const DrawData draw = frame.draws[input.draw_index];
    const ShadowSurface from = {
        input.relative_position,
        triangle_normal(frame, draw, primitive),
        true,
        camera_relative(frame, draw.cell, float3(0.0)),
        draw.model,
        transpose(draw.normal_matrix),
    };

    // The shadow view: how much of the sun's light gets through to each point.
    if (frame.view == view_shadow) {
        const float sun = frame.sun_direction.y > 0.0
            ? light_visibility(frame, shadow_ray_origin(frame, from, frame.sun_direction), frame.sun_direction, infinite_distance)
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

    // Direct light, shadowed, the sky's light, the air in between, and the exposure (shading.slangh).
    return float4(shade_surface(surface, frame, from, input.position.xy, shading_roughness, visibility, irradiance_normal, emissive), base_color.a);
}

// Fragment shaders

// The lighting pass, for opaque and masked surfaces. SV_Target: the value written to color attachment 0. SV_IsFrontFace: whether this triangle faces the camera; SV_PrimitiveID: which of the draw's triangles it is, in index-buffer order.
[shader("fragment")]
float4 fragmentMain(VertexOutput input, bool front_face : SV_IsFrontFace, uint primitive : SV_PrimitiveID) : SV_Target {
    return shade_fragment(input, front_face, primitive);
}

// Weighted blended transparency

// The transparency pass, for blended surfaces (McGuire and Bavoil 2013, "Weighted Blended Order-Independent Transparency"). Blending one surface over another depends on which is in front, so blended surfaces would have to be sorted back to front, per pixel. Instead, every fragment adds into two sums, in any order:
//   accum   (premultiplied color, coverage) times a weight that falls with distance, added up
//   reveal  the share of the scene that shows through: 1, times every fragment's (1 - coverage)
// The composite (composite.slang) divides accum's color by its coverage, a weighted average of the layers, and lays it over the scene by 1 - reveal. One layer comes out as blending would draw it, to 16-bit precision; where layers overlap, the nearer one counts for more.
struct TransparentOutput {
    float4 accum : SV_Target0;
    float reveal : SV_Target1;
};

// The weight: McGuire and Bavoil's equation 7, tuned for 16-bit float sums and distances from 0.1 m to 500 m. Past a few hundred metres it bottoms out at its floor, so distant layers that overlap count equally. It falls steeply with the distance in front of the camera, so where layers overlap, the nearest dominates; the clamp keeps it between 1e-2 and 3e3. Colors are clamped to transparent_max, which tone mapping already shows as nearly white: one fragment then adds at most 4 x 3e3 = 12000, well below the 65504 a 16-bit float holds, so many layers can stack up before the sums overflow.
static const float transparent_max = 4.0;

float transparent_weight(float coverage, float view_depth) {
    const float a = view_depth / 5.0;
    const float b = view_depth / 200.0;
    return coverage * clamp(10.0 / (1e-5 + a * a + b * b * b * b * b * b), 1e-2, 3e3);
}

[shader("fragment")]
TransparentOutput transparentMain(VertexOutput input, bool front_face : SV_IsFrontFace, uint primitive : SV_PrimitiveID) {
    const float4 color = shade_fragment(input, front_face, primitive);
    const float coverage = saturate(color.a);

    // The distance in front of the camera, along its view direction: the clip-space w a perspective projection leaves.
    const float view_depth = mul(push.frame.view_projection, float4(input.relative_position, 1.0)).w;
    const float weight = transparent_weight(coverage, view_depth);

    TransparentOutput output;
    output.accum = float4(min(color.rgb, transparent_max) * coverage, coverage) * weight;
    output.reveal = coverage;
    return output;
}
```

`game-engine/shaders/scan.slangh`:
```slang
// Prefix sums in one workgroup, shared by cull.slang and lights.slang. A shader that includes this runs exclusive_scan from a [numthreads(scan_size, 1, 1)] entry point, every thread of the group calling it.

// The exclusive prefix sums of `input`: output[i] is the sum of input[0] to input[i - 1], and output[count] the sum of all of them. Given 0 or 1 per item, output[i] is how many items before item i were 1: where item i goes when only the 1s are kept, in their order.
//
// One workgroup does all of it, scan_size numbers at a time. Within a chunk, the threads sum in log2(scan_size) rounds (Hillis and Steele 1986): in each round, every thread adds the number `offset` places before its own, and `offset` doubles. Each chunk's total carries over to the next. That's enough for tens of thousands of items; far more would call for a scan spread over many workgroups.
static const uint scan_size = 256;

groupshared uint scan_numbers[scan_size];
groupshared uint scan_carry;

void exclusive_scan(uint *input, uint *output, uint count, uint thread) {
    if (thread == 0) {
        scan_carry = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint chunk = 0; chunk < count; chunk += scan_size) {
        const uint i = chunk + thread;
        const uint value = i < count ? input[i] : 0;
        scan_numbers[thread] = value;
        GroupMemoryBarrierWithGroupSync();

        // Inclusive sums: scan_numbers[t] becomes the sum up to and with t.
        for (uint offset = 1; offset < scan_size; offset *= 2) {
            const uint before = thread >= offset ? scan_numbers[thread - offset] : 0;
            GroupMemoryBarrierWithGroupSync();
            scan_numbers[thread] += before;
            GroupMemoryBarrierWithGroupSync();
        }

        // Exclusive: without the thread's own number.
        if (i < count) {
            output[i] = scan_carry + scan_numbers[thread] - value;
        }
        GroupMemoryBarrierWithGroupSync();

        if (thread == scan_size - 1) {
            scan_carry += scan_numbers[thread];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (thread == 0) {
        output[count] = scan_carry;
    }
}

// Prefix sums over one workgroup's threads: `prefix` is the sum of the other threads' values before this one, in thread order, and the sum of all of them is returned. Every thread of the group calls it, together. The same rounds as exclusive_scan's, over one chunk.
uint group_exclusive_scan(uint value, uint thread, out uint prefix) {
    scan_numbers[thread] = value;
    GroupMemoryBarrierWithGroupSync();

    for (uint offset = 1; offset < scan_size; offset *= 2) {
        const uint before = thread >= offset ? scan_numbers[thread - offset] : 0;
        GroupMemoryBarrierWithGroupSync();
        scan_numbers[thread] += before;
        GroupMemoryBarrierWithGroupSync();
    }

    prefix = scan_numbers[thread] - value;
    const uint total = scan_numbers[scan_size - 1];
    GroupMemoryBarrierWithGroupSync();
    return total;
}
```

## 18.5 Which patches: `terrain.slangh`, `cull.slang`, `culling.h`, `culling.cpp`

### Why
Each cull phase has to turn the camera's position and the view into the list of patches to draw: coarse far away, fine nearby, none where nothing shows.

### How
- **`terrain.slangh`,** included by the cull, the terrain's own shaders and the shading header:
  - **`terrain_mirror`:** the field's sample at a sample of the repeated terrain: the tile it's in, and the position within the tile, flipped in every other tile.
  - **`terrain_height`** reads a sample, two per word, scaled and offset into metres.
  - **`terrain_node`** and **`terrain_patch_index`:** a node's place in the finest-first storage: with n nodes a side at level 0, level `l` starts after `(4 n² − 4 (n ≫ l)²) / 3` nodes.
  - **`terrain_node_heights`:** a node's least and greatest height, in metres. A node within one tile is that tile's node of the field, mirrored with the tile; a node spanning tiles has the whole field's bounds.
  - **`terrain_relative`:** where a sample is, relative to the camera: its metres from the corner split into whole cells and the rest, so the cells subtract exactly, like every other position (Chapter 13). A sample's metres are an exact float up to 16 million of them.
  - **`terrain_exact_height`:** the drawn triangle's height under any point of a quad, every quad split from its (0, 0) corner to its (1, 1) corner.
  - **`pack_patch`:** a patch in a word: its level and its column and row at that level.
- **The selection** (`select_patches`), in one workgroup of 256 threads, walking the quadtree level by level from its root:
  - **A node** at level L is a patch of 8 × 8 quads with 2^L samples between its vertices: the quadtree's node at level L + 3. Its box is its square across and its height bounds up, relative to the camera, and it's tested like a draw's: against the six planes, then the depth pyramid.
  - **Split or draw:** a node within the range of the level below, `terrain_range × 2^(L−1)` of the camera, is split into its four children for the next level; any other node is drawn at its level, and level 0 always. The nearest point of the box decides the distance. Simpler than Strugar's selection, which lets a parent cover a child that's outside the child's own range: here such a child is drawn at its level, morphed all the way to its parent's grid, which is the same shape with more triangles.
  - **In order:** each level's nodes are visited in order, in chunks of 256, and prefix sums over the chunk place each node's children in the next level's list and each patch in the patch list: the same lists every frame for the same view, with no atomics. The two lists alternate between the two halves of `nodes`.
  - **Early and late:** the early phase marks every patch it draws with the frame's number in `early_patches`, one word per patch there could be, 5.6 million; the late phase, walking the same tree against the new pyramid, draws only patches not so marked. The frame number means nothing needs clearing.
  - **The command:** the patch count, as the mesh dispatch's workgroup count, then 1 and 1.
- **Why the ranges work:** with level 0 reaching 623 m at 1080p, and each level twice as far, neighbouring patches never differ by more than one level, which is what the morphing (18.6) relies on. A level-L patch is drawn whole when it's outside the range of level L − 1; its neighbour at level L + 2 would have to be outside the range of level L + 1 while touching a node inside the range of level L, which needs the two ranges to differ by less than a level-(L + 1) patch's box. They differ by `terrain_range × 2^L`, 623 m at level 0 against a box 64 m across and at most 340 m tall.
- **`culling.h`, `culling.cpp`:** each phase gets its `patches`, `patch_command` and `nodes` buffers, the shared `early_patches` gets a word per patch of the repeated terrain, starting at 0, and the selection runs as the phase's seventh step, after the commands. The barriers add the mesh shader stage, which reads the patches, and `draw_terrain` is one `vkCmdDrawMeshTasksIndirectEXT`, reading the command. `CullTotals` gains each phase's patch count.

### Code
`game-engine/shaders/terrain.slangh`:
```slang
// The height field, for every shader that reads it: the cull's patch selection (cull.slang), the patches' mesh shader (terrain.slang) and the shadow rays' march (shading.slangh). Included after shared.slangh.

// A patch is 8 x 8 quads, drawn by one mesh workgroup. A patch at terrain level L has 2^L samples between its vertices, so it covers 8 x 2^L quads of the finest grid: it's the node at level L + 3 of the quadtree of height bounds, whose level 0 is one node per quad.
static const uint patch_quads = 8;
static const uint patch_level_shift = 3;

// Vertices in the outer 30% of a level's range morph toward the next level's grid, so neighbouring patches at different levels meet without cracks.
static const float terrain_morph_start = 0.7;

// A patch in the cull's lists and the patch list, in one word: its level, and its column and row at that level. Up to 4,096 patches a side.
uint pack_patch(uint level, uint2 at) {
    return (level << 24) | (at.y << 12) | at.x;
}

uint patch_level(uint packed) {
    return packed >> 24;
}

uint2 patch_at(uint packed) {
    return uint2(packed & 0xFFF, (packed >> 12) & 0xFFF);
}

// The camera's height in the world, in metres: the terrain's heights are measured in the world, not from the camera.
float terrain_camera_height(FrameData *frame) {
    return float(frame.camera_cell.y) * cell_size + frame.camera_offset.y;
}

// The height of the drawn terrain under sample position `sample`, which needn't be whole: within its quad, the height of the triangle the mesh shader draws there, every quad split from its (0, 0) corner to its (1, 1) corner.
float terrain_exact_height(TerrainInfo *terrain, float2 sample) {
    const float last = float(terrain.virtual_quads);
    const float2 clamped = clamp(sample, float2(0.0), float2(last - 0.001));
    const uint2 quad = uint2(clamped);
    const float2 f = clamped - float2(quad);
    const float h00 = terrain_height(terrain, quad);
    const float h11 = terrain_height(terrain, quad + 1);

    if (f.x >= f.y) {
        const float h10 = terrain_height(terrain, quad + uint2(1, 0));
        return h00 + (h10 - h00) * f.x + (h11 - h10) * f.y;
    }

    const float h01 = terrain_height(terrain, quad + uint2(0, 1));
    return h00 + (h01 - h00) * f.y + (h11 - h01) * f.x;
}

// The field's sample that stands at `sample` of the repeated terrain: the field is flipped in every other tile, so its edges meet their own reflections and nothing shows a seam. The repeated terrain's far edge, at tiles x quads, belongs to the last tile.
uint2 terrain_mirror(TerrainInfo *terrain, uint2 sample) {
    const uint quads = terrain.samples - 1;
    const uint2 tile = min(sample / quads, uint2(terrain.tiles - 1));
    const uint2 within = sample - tile * quads;
    return uint2((tile.x & 1) != 0 ? quads - within.x : within.x, (tile.y & 1) != 0 ? quads - within.y : within.y);
}

// The heights' scale: metres per step of a sample.
float terrain_height_scale(TerrainInfo *terrain) {
    return (terrain.height_max - terrain.height_min) / 65535.0;
}

// The height at `sample` of the repeated terrain, in metres, offset into place.
float terrain_height(TerrainInfo *terrain, uint2 sample) {
    const uint2 at = terrain_mirror(terrain, sample);
    const uint index = at.y * terrain.samples + at.x;
    const uint word = terrain.heights[index >> 1];
    const uint raw = (index & 1) != 0 ? word >> 16 : word & 0xFFFF;
    return terrain.height_min + terrain.height_offset + float(raw) * terrain_height_scale(terrain);
}

// The field's quadtree node at `level` whose column and row are `at`. Level 0 has one node per quad, (samples - 1) a side; each level above has half as many a side. Levels are stored finest first, so a level starts after the sum of the levels below: with n a side at level 0, that's (4 n^2 - 4 side^2) / 3.
uint terrain_node(TerrainInfo *terrain, uint level, uint2 at) {
    const uint n = terrain.samples - 1;
    const uint side = n >> level;
    return (4 * n * n - 4 * side * side) / 3 + at.y * side + at.x;
}

// The patch at `level` and `at` among every patch of the repeated terrain, counted like the nodes, over the patch levels only: the cull keeps a word per patch (early_patches).
uint terrain_patch_index(TerrainInfo *terrain, uint level, uint2 at) {
    const uint n = terrain.virtual_quads / patch_quads;
    const uint side = n >> level;
    return (4 * n * n - 4 * side * side) / 3 + at.y * side + at.x;
}

// The least and greatest height of the repeated terrain's node at `level` and `at`, in metres. A node within one tile is that tile's node of the field, mirrored with the tile; a node spanning tiles has the whole field's bounds, like every tile.
float2 terrain_node_heights(TerrainInfo *terrain, uint level, uint2 at) {
    const uint quads = terrain.samples - 1;
    const uint field_top = firstbithigh(quads);
    uint node;

    if (level > field_top) {
        node = terrain_node(terrain, field_top, uint2(0));
    } else {
        const uint side = quads >> level;
        const uint2 tile = at / side;
        const uint2 within = at - tile * side;
        node = terrain_node(terrain, level, uint2((tile.x & 1) != 0 ? side - 1 - within.x : within.x, (tile.y & 1) != 0 ? side - 1 - within.y : within.y));
    }

    const uint word = terrain.bounds[node];
    return terrain.height_min + terrain.height_offset + float2(word & 0xFFFF, word >> 16) * terrain_height_scale(terrain);
}

// Where sample (x, z) of the repeated terrain is, at height `height`, relative to the camera: its metres from the terrain's corner, split into whole cells and the rest, so the cells subtract exactly (shared.slangh's camera_relative). A sample's metres are a float exactly, up to 2^24 of them.
float3 terrain_relative(FrameData *frame, TerrainInfo *terrain, float2 sample, float height) {
    const float2 metres = sample * terrain.step;
    const float2 cells = floor(metres / cell_size);
    const int3 cell = terrain.origin_cell + int3(int(cells.x), 0, int(cells.y));
    return camera_relative(frame, cell, float3(metres.x - cells.x * cell_size, height, metres.y - cells.y * cell_size));
}

// The camera's position in sample units from the terrain's corner, for rays. The cells subtract exactly; the camera's own offset is under a cell.
float2 terrain_camera(FrameData *frame, TerrainInfo *terrain) {
    const int3 cells = frame.camera_cell - terrain.origin_cell;
    return (float2(cells.x, cells.z) * cell_size + frame.camera_offset.xz) / terrain.step;
}
```

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
// Each phase also picks the terrain's patches, in one more step (selectEarlyPatchesMain, selectLatePatchesMain). Draws are visited in the cull's order (by list, then primitive), so a group's draws, and a list's groups, are runs. Every write goes to a place the prefix sums fix: the result doesn't depend on which thread runs first.

#include "shared.slangh"
#include "scan.slangh"
#include "terrain.slangh"

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
    uint *patches;          // the terrain patches this phase draws, packed (terrain.slangh)
    uint *patch_command;    // how many, then 1, 1: the mesh dispatch's workgroup counts
    uint *nodes;            // the selection's working lists: two runs of max_terrain_patches
    uint *early_patches;    // per patch (terrain_patch_index): the frame the early phase last drew it in
    uint draw_count;
    uint group_count;
};

// The most patches a level of the selection, and a frame, can hold (terrain.h).
static const uint max_terrain_patches = 65536;

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

// 10. The terrain's patches

// Whether the box from `lo` to `hi`, camera-relative, comes within `range` of the camera: whether its nearest point does.
bool within(float3 lo, float3 hi, float range) {
    const float3 nearest = clamp(float3(0.0), lo, hi);
    return dot(nearest, nearest) < range * range;
}

// The box of the patch at `level` and `at`, relative to the camera: its samples' extent across, its node's height bounds up.
void patch_box(FrameData *frame, TerrainInfo *terrain, uint level, uint2 at, out float3 lo, out float3 hi) {
    const float2 heights = terrain_node_heights(terrain, level + patch_level_shift, at);
    const float2 first = float2(at * (patch_quads << level));
    const float2 last = first + float(patch_quads << level);
    lo = terrain_relative(frame, terrain, first, heights.x);
    hi = terrain_relative(frame, terrain, last, heights.y);
}

// Which patches to draw, in one workgroup: the quadtree walked from its root, level by level (Strugar 2009). A patch outside the view or hidden by the pyramid is dropped with everything under it. One within the range of the level below is split into its four children, for the next level; the rest are drawn whole, at their level, and level 0 always. Each level's nodes are kept in order, and prefix sums place their children and their patches, so the list is the same every frame for the same view. The late phase walks the same tree with the new pyramid, drawing only the patches the early phase didn't: the early phase marks each patch it draws with the frame's number.
void select_patches(uint thread, bool late) {
    FrameData *frame = push.frame;
    TerrainInfo *terrain = frame.terrain;
    CullTables *tables = push.tables;
    uint *lists = tables.nodes;
    const uint top = terrain.quad_levels - 1 - patch_level_shift;

    // This level's nodes start at `read`, the next level's at `write`: the two halves of the lists, swapped each level.
    uint read = 0;
    uint write = max_terrain_patches;
    uint count = 1;
    uint patch_count = 0;

    if (thread == 0) {
        lists[0] = pack_patch(top, uint2(0));
    }
    AllMemoryBarrierWithGroupSync();

    for (uint level = top; ; --level) {
        uint next_count = 0;

        for (uint chunk = 0; chunk < count; chunk += scan_size) {
            const uint i = chunk + thread;
            uint2 at = uint2(0);
            uint split = 0;
            uint draw = 0;

            if (i < count) {
                at = patch_at(lists[read + i]);
                float3 lo;
                float3 hi;
                patch_box(frame, terrain, level, at, lo, hi);

                if (in_view(frame.view_projection, lo, hi) && !occluded(frame, lo, hi)) {
                    if (level > 0 && within(lo, hi, frame.terrain_range * float(1u << (level - 1)))) {
                        split = 1;
                    } else if (!late || tables.early_patches[terrain_patch_index(terrain, level, at)] != frame.frame_index) {
                        draw = 1;
                    }
                }
            }

            uint split_prefix;
            const uint splits = group_exclusive_scan(split, thread, split_prefix);
            uint draw_prefix;
            const uint draws = group_exclusive_scan(draw, thread, draw_prefix);

            if (split != 0 && next_count + 4 * split_prefix + 4 <= max_terrain_patches) {
                for (uint k = 0; k < 4; ++k) {
                    lists[write + next_count + 4 * split_prefix + k] = pack_patch(level - 1, at * 2 + uint2(k & 1, k >> 1));
                }
            }

            if (draw != 0) {
                tables.patches[patch_count + draw_prefix] = pack_patch(level, at);

                if (!late) {
                    tables.early_patches[terrain_patch_index(terrain, level, at)] = frame.frame_index;
                }
            }

            next_count = min(next_count + 4 * splits, max_terrain_patches);
            patch_count += draws;
            AllMemoryBarrierWithGroupSync();
        }

        if (level == 0 || next_count == 0) {
            break;
        }

        const uint swap = read;
        read = write;
        write = swap;
        count = next_count;
    }

    // The mesh dispatch: one workgroup per patch.
    if (thread == 0) {
        tables.patch_command[0] = patch_count;
        tables.patch_command[1] = 1;
        tables.patch_command[2] = 1;
    }
}

[shader("compute")]
[numthreads(scan_size, 1, 1)]
void selectEarlyPatchesMain(uint3 id : SV_GroupThreadID) {
    select_patches(id.x, false);
}

[shader("compute")]
[numthreads(scan_size, 1, 1)]
void selectLatePatchesMain(uint3 id : SV_GroupThreadID) {
    select_patches(id.x, true);
}
```

`game-engine/src/includes/culling.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/image.h"
#include "includes/shader_types.h"
#include "includes/terrain.h"
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
//   - The terrain is culled the same way: each phase walks the height field's quadtree, from the whole field down to patches of 8 x 8 quads, dropping what's out of view or hidden and splitting what's near enough to need more detail, and lists the patches to draw. The late phase lists only what the early one didn't.
//   - Every frame, the cull runs in two phases around the depth prepass, each of six compute steps (cull.slang), plus the terrain's. The early phase tests every draw against the pyramid the previous frame built, under this frame's view: a guess, right wherever the view hasn't changed. The prepass draws what it keeps, the pyramid is built from that depth, and the late phase tests what the early phase left out against it. Whatever the guess hid wrongly is drawn late; nothing visible is missed, and nothing is drawn twice. Each phase writes its own commands, instances and counts, and every pass draws both phases' lists.
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
    Buffer patches;      // the terrain patches to draw, packed (terrain.slangh)
    Buffer patch_command;  // one VkDrawMeshTasksIndirectCommandEXT: how many patches, 1, 1
    Buffer nodes;        // the terrain selection's working lists
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
    vk::raii::Pipeline select_early_patches = nullptr;
    vk::raii::Pipeline select_late_patches = nullptr;

    // Written once, shared by the phases.
    Buffer order;   // draw indices, in the cull's order
    Buffer groups;  // one DrawGroup per group, in order
    Buffer lists;   // one DrawListRange per list

    // Written by the early phase, read by the late: per terrain patch, the frame it was last drawn in.
    Buffer early_patches;

    std::array<CullPhaseBuffers, cull_phases.size()> phases;

    std::array<DrawListRange, draw_list_count> list_ranges{};  // the lists' runs, for the draw calls
    std::uint32_t draw_count = 0;
    std::uint32_t group_count = 0;
};

// `draws` has one CullDraw per draw, in draw order; `terrain` is what the phases pick patches of.
DrawCulling create_draw_culling(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::span<const CullDraw> draws,
    const Terrain &terrain
);

// The totals, as the cull copies them out: how many draws each phase kept, in how many commands, and how many terrain patches.
struct CullTotals {
    std::uint32_t early_draws;
    std::uint32_t late_draws;
    std::uint32_t early_commands;
    std::uint32_t late_commands;
    std::uint32_t early_patches;
    std::uint32_t late_patches;
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

// Draws `phase`'s terrain patches for this frame: one mesh dispatch, a workgroup per patch. The terrain pipeline, its push data with the phase's patches, and the cull mode and front face must already be set.
void draw_terrain(const vk::raii::CommandBuffer &commands, const DrawCulling &culling, CullPhase phase);
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
    std::span<const CullDraw> draws,
    const Terrain &terrain
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
    culling.select_early_patches = create_compute_pipeline(device, "cull", "selectEarlyPatchesMain");
    culling.select_late_patches = create_compute_pipeline(device, "cull", "selectLatePatchesMain");
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
        phase.patches = gpu_numbers(device, gpu, max_terrain_patches);
        phase.patch_command = gpu_numbers(device, gpu, 3, vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc);
        phase.nodes = gpu_numbers(device, gpu, 2 * max_terrain_patches);
    }

    // The frame each patch was last drawn in: 0 to begin with, before any frame.
    const std::vector<std::uint32_t> never(terrain.patch_count);
    culling.early_patches = upload_buffer(device, gpu, queue, pool, std::as_bytes(std::span(never)),
        vk::BufferUsageFlagBits::eShaderDeviceAddress);

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
            .patches = phase.patches.address,
            .patch_command = phase.patch_command.address,
            .nodes = phase.nodes.address,
            .early_patches = culling.early_patches.address,
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

    // The previous frame's draws read the phase's commands, counts, instances and patches, and its copy read the totals: wait for them before rewriting any. Rewriting what was read only needs the wait, so no access is made visible.
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eVertexShader
            | vk::PipelineStageFlagBits2::eMeshShaderEXT | vk::PipelineStageFlagBits2::eCopy,
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

    // 7. The terrain's patches, which read nothing of the above: one workgroup walks the quadtree.
    step(early ? culling.select_early_patches : culling.select_late_patches, 1);

    // The draws read the commands and counts as indirect arguments, and the instances in their vertex shaders; the terrain's mesh dispatch reads its command, and its mesh shaders the patches; the copy reads the totals; the late cull reads the early phase's flags and patches.
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eVertexShader
            | vk::PipelineStageFlagBits2::eMeshShaderEXT | vk::PipelineStageFlagBits2::eCopy
            | vk::PipelineStageFlagBits2::eComputeShader,
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
    commands.copyBuffer(*buffers.patch_command.handle, readback, vk::BufferCopy{
        .srcOffset = 0,
        .dstOffset = early ? offsetof(CullTotals, early_patches) : offsetof(CullTotals, late_patches),
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

void draw_terrain(const vk::raii::CommandBuffer &commands, const DrawCulling &culling, CullPhase phase) {
    // The GPU reads how many workgroups to run from the phase's command: one per patch.
    const CullPhaseBuffers &buffers = culling.phases[static_cast<std::size_t>(phase)];
    commands.drawMeshTasksIndirectEXT(*buffers.patch_command.handle, 0, 1, sizeof(vk::DrawMeshTasksIndirectCommandEXT));
}
```

## 18.6 Drawing patches: `terrain.slang`, `pipeline.h`, `pipeline.cpp`

### Why
A mesh shader makes a patch, and a fragment shader shades it; a pipeline with a mesh stage draws it.

### How
- **`meshMain`**, `[shader("mesh")]`, with `[outputtopology("triangle")]` and 64 threads: one workgroup per patch, from the list the push data names, where a mesh's instances would be. `SetMeshOutputCounts` declares 81 vertices and 128 triangles, and the threads fill both arrays, each taking every 64th.
  - **A vertex** (i, j) of the patch at level L and column and row `at` is sample `at × 8 × 2^L + (i, j) × 2^L`, placed by `terrain_relative`.
  - **Morphing:** a vertex whose i or j is odd isn't on the next level's grid; the point halfway between its two even neighbours lies on the coarser grid's edge, along x, along z, or along the quad's diagonal when both are odd. As the vertex's distance from the camera crosses the last 30% of its level's range, it slides from its own place to that midpoint, height and all: at the range's end it lies exactly on the coarser patch's edge, every fine triangle inside a coarse one, since both levels split their quads the same way, so the two meet with no crack. The sample coordinates the fragment shader gets slide with it, and the normal morphs between the two levels' too. Strugar's vertices slide onto an even neighbour instead, collapsing triangles; the midpoint keeps them.
  - **The normal** of the height field at a sample, from the heights the level's spacing either way: the surface y = h(x, z) has the normal (−∂h/∂x, 1, −∂h/∂z).
  - **The triangles:** every quad split from its (i, j) corner to its (i + 1, j + 1) corner, the way the rays and the CPU read it, both winding counter-clockwise seen from above, glTF's front.
- **`prepassMain`** writes the drawn surface's normal, as the meshes' prepass writes the vertex normal.
- **`fragmentMain`:**
  - **Two materials by slope:** the flat one on level ground and the steep one on slopes, blended between 15° and 35°, each tiled every `uv_repeat` metres of the field: base color, roughness and metalness, occlusion, and the normal maps.
  - **The normal maps' frame:** the textures are tiled by the field's x and z, so a map's +X is the world's +X, and its +Y, which glTF has pointing up the image, the world's −Z: the tangent is +X projected onto the surface, and the bitangent the cross product that gives −Z on level ground.
  - **Shadow rays start on the exact height field** under the pixel, not on the drawn surface: far away the drawn surface is a coarser level, a little above or below the heights the rays march, and a ray from under it would be in shadow at once. They start off it along the drawn surface's normal, and the terrain is in no instance, so its `ShadowSurface` has no transform.
  - **Then as any surface:** a `Surface`, ambient occlusion from the pass's image, the debug views, and `shade_surface`.
- **`create_terrain_pipeline`:** a pipeline whose first stage is `eMeshEXT`, from `terrain.spv`, with the same states as the meshes' pipelines, which `create_scene_pipeline` now builds for both. A mesh pipeline has no vertex input or assembly state: Vulkan ignores both.

### Code
`game-engine/shaders/terrain.slang`:
```slang
// Draws the terrain: each patch of 8 x 8 quads by one mesh workgroup, which reads the patch's heights, places its vertices and writes its triangles itself; no vertex buffer, no index buffer. The level of detail is CDLOD's (Strugar 2009): a patch at level L has 2^L samples between its vertices, and the vertices in the outer part of the level's range morph toward the next level's grid, so patches at different levels meet without cracks. Shaded like every other surface (shading.slangh), with two glTF materials, normal maps and all, blended by slope. Two fragment shaders, as mesh.slang's:
//   prepassMain   the depth prepass's vertex normal
//   fragmentMain  the lighting pass, into the HDR image

#include "shared.slangh"
#include "atmosphere.slangh"
#include "shading.slangh"

// The same push data as the meshes': the frame, and the phase's patches in place of its instances.
[[vk::push_constant]]
ConstantBuffer<PushData> push;

// Stage interface

// What the mesh shader hands to the rasterizer, per vertex. `sample` is the vertex's position in the field, in samples: the fragment shader reads the exact height there, and tiles the textures by it.
struct PatchVertex {
    float4 position : SV_Position;
    float3 relative_position : POSITION;  // camera-relative
    float3 normal : NORMAL;               // of the drawn surface, at this level of detail
    float2 sample : TEXCOORD0;
};

static const uint patch_vertices = (patch_quads + 1) * (patch_quads + 1);  // 81
static const uint patch_triangles = patch_quads * patch_quads * 2;          // 128

// Mesh shader

// The normal of the height field at `sample`, from the heights `spacing` samples either way (central differences): the surface y = h(x, z) has the normal (-dh/dx, 1, -dh/dz). At the field's edge, the samples just inside stand in.
float3 terrain_normal(TerrainInfo *terrain, uint2 sample, uint spacing) {
    const uint last = terrain.virtual_quads;
    const uint2 lo = uint2(sample.x >= spacing ? sample.x - spacing : 0, sample.y >= spacing ? sample.y - spacing : 0);
    const uint2 hi = min(sample + spacing, uint2(last));
    const float dx = (terrain_height(terrain, uint2(hi.x, sample.y)) - terrain_height(terrain, uint2(lo.x, sample.y))) / (float(hi.x - lo.x) * terrain.step);
    const float dz = (terrain_height(terrain, uint2(sample.x, hi.y)) - terrain_height(terrain, uint2(sample.x, lo.y))) / (float(hi.y - lo.y) * terrain.step);
    return normalize(float3(-dx, 1.0, -dz));
}

// One workgroup per patch: 64 threads write 81 vertices and 128 triangles. Vertex (i, j) of the patch at `level` and `at` is sample at x 8 x 2^L + (i, j) x 2^L.
//
// Morphing: a vertex whose i or j is odd isn't on the next level's grid; the point halfway between its two even neighbours lies on the coarser grid's edge, along x, along z, or along the quad's diagonal when both are odd. As the vertex's distance from the camera crosses the last 30% of its level's range, it slides from its own place to that midpoint, height and all: at the range's end it lies exactly on the coarser patch's edge, so the two meet. The sample coordinates slide with it. The normal morphs between the two levels' too.
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void meshMain(
    uint thread : SV_GroupThreadID,
    uint3 group : SV_GroupID,
    out vertices PatchVertex verts[patch_vertices],
    out indices uint3 tris[patch_triangles]
) {
    FrameData *frame = push.frame;
    TerrainInfo *terrain = frame.terrain;
    const uint packed = push.instances[group.x];
    const uint level = patch_level(packed);
    const uint spacing = 1u << level;
    const uint2 first = patch_at(packed) * (patch_quads << level);
    const float range = frame.terrain_range * float(spacing);

    SetMeshOutputCounts(patch_vertices, patch_triangles);

    for (uint v = thread; v < patch_vertices; v += 64) {
        const uint2 ij = uint2(v % (patch_quads + 1), v / (patch_quads + 1));
        const uint2 sample = first + ij * spacing;
        float3 position = terrain_relative(frame, terrain, float2(sample), terrain_height(terrain, sample));
        float2 at = float2(sample);
        float3 normal = terrain_normal(terrain, sample, spacing);

        const float morph = saturate((length(position) / range - terrain_morph_start) / (1.0 - terrain_morph_start));

        if (morph > 0.0 && ((ij.x | ij.y) & 1) != 0) {
            const uint2 step = (ij & 1) * spacing;
            const uint2 a = sample - step;
            const uint2 b = sample + step;
            const float3 pa = terrain_relative(frame, terrain, float2(a), terrain_height(terrain, a));
            const float3 pb = terrain_relative(frame, terrain, float2(b), terrain_height(terrain, b));
            position = lerp(position, (pa + pb) * 0.5, morph);
            at = lerp(at, float2(a + b) * 0.5, morph);
            normal = normalize(lerp(normal, terrain_normal(terrain, sample, spacing * 2), morph));
        }

        verts[v].position = mul(frame.view_projection, float4(position, 1.0));
        verts[v].relative_position = position;
        verts[v].normal = normal;
        verts[v].sample = at;
    }

    // Every quad is split from its (i, j) corner to its (i + 1, j + 1) corner, the way the shadow rays and the CPU read it too. Both triangles wind counter-clockwise seen from above, glTF's front.
    for (uint t = thread; t < patch_triangles; t += 64) {
        const uint quad = t / 2;
        const uint corner = (quad / patch_quads) * (patch_quads + 1) + quad % patch_quads;
        const uint across = corner + patch_quads + 1;
        tris[t] = (t & 1) == 0 ? uint3(corner, across, across + 1) : uint3(corner, across + 1, corner + 1);
    }
}

// Material

// A material slot's texture at `uv`, with its sampler.
float4 sample_material(TextureSlot slot, float2 uv) {
    const Texture2D texture = Texture2D.Handle(uint2(slot.texture, 0));
    const SamplerState sampler = SamplerState.Handle(uint2(slot.sampler, 0));
    return texture.Sample(sampler, uv);
}

// Fragment shaders

// The prepass: the drawn surface's normal, octahedrally encoded, like the meshes' vertex normal.
[shader("fragment")]
float2 prepassMain(PatchVertex input) : SV_Target {
    return encode_octahedral(normalize(input.normal));
}

// The ground's normal at this pixel: the drawn surface's, tilted by the two normal maps, blended like the rest of the material. The textures are tiled by the field's x and z, so the map's +X is the world's +X, and its +Y, which glTF has pointing up the image, the world's -Z: the tangent is +X projected onto the surface, and the bitangent the cross product that gives -Z on level ground. `map_spread` is the maps' spread (mips.slang), for specular antialiasing.
float3 ground_normal(float3 normal, Material flat, Material steep, float flat_share, float2 uv, out float map_spread) {
    const float3 tangent = normalize(float3(1.0, 0.0, 0.0) - normal * normal.x);
    const float3 bitangent = cross(normal, tangent);
    const float4 flat_texel = sample_material(flat.normal, uv);
    const float4 steep_texel = sample_material(steep.normal, uv);
    float3 tangent_space = lerp(steep_texel.xyz * 2.0 - 1.0, flat_texel.xyz * 2.0 - 1.0, flat_share);
    tangent_space.xy *= lerp(steep.normal_scale, flat.normal_scale, flat_share);
    map_spread = 1.0 - lerp(steep_texel.w, flat_texel.w, flat_share);
    return normalize(tangent * tangent_space.x + bitangent * tangent_space.y + normal * tangent_space.z);
}

// The lighting pass. The ground's material is the flat one on level ground and the steep one on slopes, blended between 15 and 35 degrees, each tiled every uv_repeat metres of the field. Shadow rays start from the exact height field under the pixel, not from the drawn surface: far away the drawn surface is a coarser level, a little above or below the heights the rays march, and a ray from under it would be in shadow at once.
[shader("fragment")]
float4 fragmentMain(PatchVertex input) : SV_Target {
    FrameData *frame = push.frame;
    TerrainInfo *terrain = frame.terrain;
    const float3 vertex_normal = normalize(input.normal);
    const float2 uv = input.sample * (terrain.step / terrain.uv_repeat);

    const Material flat = frame.materials[terrain.flat_material];
    const Material steep = frame.materials[terrain.steep_material];
    const float flat_share = smoothstep(cos(radians(35.0)), cos(radians(15.0)), vertex_normal.y);

    float map_spread = 0.0;
    const float3 normal = frame.view != view_vertex_normal
        ? ground_normal(vertex_normal, flat, steep, flat_share, uv, map_spread)
        : vertex_normal;

    const float4 base_color = lerp(sample_material(steep.base_color, uv) * steep.base_color_factor,
        sample_material(flat.base_color, uv) * flat.base_color_factor, flat_share);
    const float4 metallic_roughness = lerp(sample_material(steep.metallic_roughness, uv) * float4(1.0, steep.roughness_factor, steep.metallic_factor, 1.0),
        sample_material(flat.metallic_roughness, uv) * float4(1.0, flat.roughness_factor, flat.metallic_factor, 1.0), flat_share);
    const float occlusion = lerp(1.0 + steep.occlusion_strength * (sample_material(steep.occlusion, uv).r - 1.0),
        1.0 + flat.occlusion_strength * (sample_material(flat.occlusion, uv).r - 1.0), flat_share);
    const float metallic = metallic_roughness.b;
    const float roughness = max(metallic_roughness.g, 0.045);
    const float alpha = antialiased_alpha(roughness * roughness, vertex_normal, map_spread);
    const float shading_roughness = sqrt(alpha);

    // Ambient occlusion, as the meshes use it.
    const float4 gtao = Texture2D.Handle(uint2(frame.ambient_occlusion, 0)).Load(int3(int2(input.position.xy), 0));
    float visibility = occlusion;
    float3 irradiance_normal = normal;

    if (frame.ao_enabled != 0) {
        visibility = min(occlusion, gtao.w);
        irradiance_normal = normalize(rotate_from_to(vertex_normal, gtao.xyz, normal));
    }

    switch (frame.view) {
        case view_base_color: return base_color;
        case view_normal:
        case view_vertex_normal: return float4(normal * 0.5 + 0.5, 1.0);
        case view_metallic: return float4(metallic.xxx, 1.0);
        case view_roughness: return float4(shading_roughness.xxx, 1.0);
        case view_occlusion: return float4(occlusion.xxx, 1.0);
        case view_emissive: return float4(0.0, 0.0, 0.0, 1.0);
        case view_ambient_occlusion: return float4(gtao.www, 1.0);
        case view_light_count: return float4(light_count_heat(frame, input.position.xy, input.relative_position), 1.0);
        default: break;
    }

    // Where shadow rays start: the exact height field under this pixel, off it along the drawn surface's normal, which is never far from the exact surface's. The terrain is in no instance (shading.slangh).
    const ShadowSurface from = {
        terrain_relative(frame, terrain, input.sample, terrain_exact_height(terrain, input.sample)),
        vertex_normal,
        false,
        float3(0.0),
        float4x4(0.0),
        float4x4(0.0),
    };

    if (frame.view == view_shadow) {
        const float sun = frame.sun_direction.y > 0.0
            ? light_visibility(frame, shadow_ray_origin(frame, from, frame.sun_direction), frame.sun_direction, infinite_distance)
            : 0.0;
        return float4(sun.xxx, 1.0);
    }

    const float3 view = normalize(-input.relative_position);
    const Surface surface = {
        base_color.rgb,
        metallic,
        alpha,
        normal,
        view,
        energy_compensation(frame, base_color.rgb, metallic, shading_roughness, max(dot(normal, view), 1e-4)),
    };

    return float4(shade_surface(surface, frame, from, input.position.xy, shading_roughness, visibility, irradiance_normal, float3(0.0)), 1.0);
}
```

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
std::vector<std::uint32_t> read_spirv(const std::filesystem::path &path);

// The passes the scene is drawn in, and the images each draws into.
enum class MeshPass {
    depth_normals,  // the prepass, for opaque and masked materials: the normals
    lighting,       // the full shading of opaque and masked materials: the HDR image
    transparency,   // blended materials, into weighted blended transparency's two sums
};

// Draws shaders/mesh.slang into images of `color_formats`, one per attachment, depth-tested against a `depth_format` depth buffer, for `pass` and materials with alpha mode `alpha_mode`. There is no pipeline layout: shaders find their resources in the descriptor heap. Cull mode and front face are set per draw.
vk::raii::Pipeline create_mesh_pipeline(
    const vk::raii::Device &device,
    std::span<const vk::Format> color_formats,
    vk::Format depth_format,
    AlphaMode alpha_mode,
    MeshPass pass
);

// Draws shaders/terrain.slang's patches into the same images as create_mesh_pipeline's `pass`: a mesh shader makes each patch's vertices and triangles, so there's no vertex input at all. Only the prepass and the lighting pass draw the terrain; it has one material setup and no alpha mode.
vk::raii::Pipeline create_terrain_pipeline(
    const vk::raii::Device &device,
    std::span<const vk::Format> color_formats,
    vk::Format depth_format,
    MeshPass pass
);

// How a full-screen pipeline's output meets what's in the image:
//   replace  overwrites it
//   over     mixes with it by the output's alpha
enum class ColorBlend {
    replace,
    over,
};

// Draws shaders/<shader>.spv's vertexMain and fragmentMain as one full-screen triangle into a `color_format` image: no vertex data, nothing culled. With a `depth_format`, the triangle is depth-tested at depth 0, infinitely far, without writing depth, so it only reaches pixels nothing else has been drawn on: that's how the sky goes behind the scene.
vk::raii::Pipeline create_fullscreen_pipeline(
    const vk::raii::Device &device,
    const char *shader,
    vk::Format color_format,
    vk::Format depth_format = vk::Format::eUndefined,
    ColorBlend blend = ColorBlend::replace
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

// Scene pipelines

namespace {

    // The states every scene pipeline shares, given its shader stages: the mesh pipeline's and the terrain's. `mesh_shader` pipelines make their own triangles, so they have no vertex input or assembly state; Vulkan ignores both for them.
    vk::raii::Pipeline create_scene_pipeline(
        const vk::raii::Device &device,
        std::span<const vk::PipelineShaderStageCreateInfo> stages,
        bool mesh_shader,
        std::span<const vk::Format> color_formats,
        vk::Format depth_format,
        MeshPass pass
    ) {
        const bool prepass = pass == MeshPass::depth_normals;
        const bool transparency = pass == MeshPass::transparency;

        // The transparency pass draws into its two sums, every other pass into one image.
        if (color_formats.size() != (transparency ? 2 : 1)) {
            throw std::invalid_argument("create_scene_pipeline: wrong number of color formats for this pass");
        }

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

        // Depth. With reverse-Z (see camera.cpp) nearer means a *greater* depth value, and the buffer is cleared to 0, infinitely far.
        //   - The prepass keeps a fragment only if it's nearer than what's there, and records its depth: the depth buffer ends up holding the nearest solid surface at every pixel.
        //   - The lighting pass writes no depth. Its solid surfaces pass "greater or equal" only where they are that nearest surface, so each pixel is shaded once.
        //   - The transparency pass writes none either: its see-through surfaces pass wherever they're in front of the nearest solid one.
        const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = prepass ? vk::True : vk::False,
            .depthCompareOp = prepass ? vk::CompareOp::eGreater : vk::CompareOp::eGreaterOrEqual,
        };

        // Color output, one blend state per attachment.
        //   - The prepass and the lighting pass replace what's there.
        //   - The transparency pass sums. Its first image adds every fragment's output; its second keeps what each lets through: accum  = accum + source reveal = reveal * (1 - source)
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

        // Dynamic rendering: instead of a VkRenderPass, the pipeline names the formats of the images it will draw into.
        const vk::PipelineRenderingCreateInfo rendering{
            .colorAttachmentCount = static_cast<std::uint32_t>(color_formats.size()),
            .pColorAttachmentFormats = color_formats.data(),
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
            .pVertexInputState = mesh_shader ? nullptr : &vertex_input,
            .pInputAssemblyState = mesh_shader ? nullptr : &input_assembly,
            .pViewportState = &viewport,
            .pRasterizationState = &rasterization,
            .pMultisampleState = &multisample,
            .pDepthStencilState = &depth_stencil,
            .pColorBlendState = &color_blend,
            .pDynamicState = &dynamic,
            .layout = nullptr,
        });
    }

}  // namespace

// The mesh pipeline

vk::raii::Pipeline create_mesh_pipeline(
    const vk::raii::Device &device,
    std::span<const vk::Format> color_formats,
    vk::Format depth_format,
    AlphaMode alpha_mode,
    MeshPass pass
) {
    const bool prepass = pass == MeshPass::depth_normals;
    const bool transparency = pass == MeshPass::transparency;

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
            .pName = prepass ? "prepassMain" : transparency ? "transparentMain" : "fragmentMain",
            .pSpecializationInfo = &specialization,
        },
    };

    return create_scene_pipeline(device, stages, false, color_formats, depth_format, pass);
}

// The terrain pipeline

vk::raii::Pipeline create_terrain_pipeline(
    const vk::raii::Device &device,
    std::span<const vk::Format> color_formats,
    vk::Format depth_format,
    MeshPass pass
) {
    if (pass == MeshPass::transparency) {
        throw std::invalid_argument("create_terrain_pipeline: the terrain is never see-through");
    }

    const std::vector<std::uint32_t> spirv = read_spirv(std::filesystem::path(SHADER_DIR) / "terrain.spv");

    const vk::raii::ShaderModule module(device, vk::ShaderModuleCreateInfo{
        .codeSize = spirv.size() * sizeof(std::uint32_t),
        .pCode = spirv.data(),
    });

    // A mesh shader stage in place of the vertex stage: each workgroup writes a patch's vertices and triangles itself.
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eMeshEXT,
            .module = *module,
            .pName = "meshMain",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *module,
            .pName = pass == MeshPass::depth_normals ? "prepassMain" : "fragmentMain",
        },
    };

    return create_scene_pipeline(device, stages, true, color_formats, depth_format, pass);
}

// Full-screen pipelines

vk::raii::Pipeline create_fullscreen_pipeline(
    const vk::raii::Device &device,
    const char *shader,
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

    // Each pixel the triangle reaches gets one fragment. "Over" mixes it with what's there, weighted by its alpha:
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

## 18.7 Materials and the frame: `scene.h`, `scene.cpp`, `main.cpp`

### Why
The terrain's materials come from a file with no meshes in it, and the frame draws the patches in both halves of the prepass and in the lighting pass.

### How
- **The textures:** two of Poly Haven's ground sets, `forest_ground_05` and `forest_ground_06`, free (CC0). Download each from <https://polyhaven.com/a/forest_ground_05> and <https://polyhaven.com/a/forest_ground_06> as **glTF, 1K**, and unpack them into `lecture-md/game-engine/assets/terrain/forest_ground_05/` and `.../forest_ground_06/`, so that each has its `textures/` directory with the `diff`, `arm` and `nor_gl` images. The glTF download has the normal maps as JPEG; the plain download has them as EXR, which the loader doesn't read.
- **`terrain.gltf`,** in `lecture-md/game-engine/assets/terrain/`, names the two materials from those images. Its `arm` texture holds ambient occlusion in red, roughness in green and metalness in blue: the metallic-roughness texture and the occlusion texture at once, as glTF lays them out:
```json
{
  "asset": {"version": "2.0"},
  "materials": [
    {
      "name": "forest_ground_05",
      "pbrMetallicRoughness": {
        "baseColorTexture": {"index": 0},
        "metallicRoughnessTexture": {"index": 1}
      },
      "occlusionTexture": {"index": 1},
      "normalTexture": {"index": 2}
    },
    {
      "name": "forest_ground_06",
      "pbrMetallicRoughness": {
        "baseColorTexture": {"index": 3},
        "metallicRoughnessTexture": {"index": 4}
      },
      "occlusionTexture": {"index": 4},
      "normalTexture": {"index": 5}
    }
  ],
  "textures": [
    {"source": 0}, {"source": 1}, {"source": 2}, {"source": 3}, {"source": 4}, {"source": 5}
  ],
  "images": [
    {"uri": "forest_ground_05/textures/forest_ground_05_diff_1k.jpg"},
    {"uri": "forest_ground_05/textures/forest_ground_05_arm_1k.jpg"},
    {"uri": "forest_ground_05/textures/forest_ground_05_nor_gl_1k.jpg"},
    {"uri": "forest_ground_06/textures/forest_ground_06_diff_1k.jpg"},
    {"uri": "forest_ground_06/textures/forest_ground_06_arm_1k.jpg"},
    {"uri": "forest_ground_06/textures/forest_ground_06_nor_gl_1k.jpg"}
  ]
}
```
- **`add_materials`** loads a file's materials, textures and samplers into a scene that already has some, moving every index up by what was there, and returns the first material's index. `read_model`, the file reading that `load_gltf` did inline, serves both. `add_ground` goes.
- **`main`:**
  - **Loads the terrain** once the queue exists, under every scene: the plateau is put 1 cm under the scene's lowest point, as the ground was. `terrain_tiles` is 8.
  - **The altitude** for the atmosphere is the camera's height over the terrain, from `terrain_height_at`.
  - **`terrain_range`** each frame: a sample `step` metres across at distance d covers `step × focal / d` pixels, with the focal length in pixels half the screen's height over the tangent of half the field of view; `terrain_edge_pixels` at the range gives `range = step × focal / 6`: 623 m at 1080p.
  - **`draw_terrain_patches`** binds a terrain pipeline, pushes the frame and the phase's patches, sets the cull mode and front face as for any single-sided surface, and dispatches. The prepass draws the early patches after the early draws, the late patches after the late draws; the lighting pass draws both.
  - **The title** ends with the patches drawn, and how many of them late.

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

// One glTF primitive: a run of indices in the scene's index buffer, drawn against the vertices starting at `vertex_offset` in the vertex buffer.
struct Primitive {
    std::uint32_t first_index = 0;
    std::uint32_t index_count = 0;
    std::int32_t vertex_offset = 0;
    std::uint32_t material = 0;  // index into Scene::materials
};

// glTF's texture filters and wrap modes, as the file stores them: OpenGL enum values (9728 GL_NEAREST, 10497 GL_REPEAT, ...). A filter of -1 means the file leaves it to the renderer.
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

// A glTF metallic-roughness material. The defaults are glTF's: a material that sets nothing is white, fully metallic and fully rough.
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

// One thing to draw: a primitive, placed in the world by a node's transform. A mesh used by several nodes is drawn once per node. The draw is placed in a world cell (cells.h): `model` moves the primitive into it, measured from the cell's corner, and the box is measured from there too.
struct MeshDraw {
    glm::mat4 model{1.0f};
    glm::ivec3 cell{0};
    std::uint32_t primitive = 0;

    // A transform that mirrors the primitive (a negative scale) reverses the order its triangles' corners appear in, which decides which side is the front.
    bool mirrored = false;

    // The draw's box, in its cell: the primitive's box, moved by `model`, and boxed again. It holds every triangle of the draw, if a little loosely when the transform rotates.
    glm::vec3 bounds_min{0.0f};
    glm::vec3 bounds_max{0.0f};
};

// Everything from a glTF file that drawing its geometry needs, flattened into arrays ready to upload: every primitive's vertices and indices back to back.
struct Scene {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<Primitive> primitives;
    std::vector<MeshDraw> draws;

    // The file's materials, plus a plain white one at the end for primitives that don't name a material.
    std::vector<SceneMaterial> materials;
    std::vector<SceneImage> images;
    std::vector<SceneSampler> samplers;

    // KHR_lights_punctual lights, placed in the world by their nodes.
    std::vector<Light> lights;

    // World-space box around everything drawn, roughly, in floats: where the ground and the test lights go.
    glm::vec3 bounds_min{std::numeric_limits<float>::max()};
    glm::vec3 bounds_max{std::numeric_limits<float>::lowest()};
};

// Loads the default scene of a .gltf or .glb file, with its materials, samplers, lights and images, still encoded. The scene is placed with its origin at `origin`, in metres from the world's.
Scene load_gltf(const std::filesystem::path &path, const glm::dvec3 &origin = glm::dvec3{0.0});

// Point and spot lights without a range in the file are given one: where their illuminance falls below light_threshold lux, at most max_light_range metres. 0.001 lux is at most a few 8-bit steps even at the exposure for the darkest scenes, and the falloff fades to it smoothly.
constexpr float light_threshold = 0.001f;
constexpr float max_light_range = 4096.0f;

// Adds the materials of a .gltf file to `scene`, with their textures and samplers, and returns the index of its first one: for surfaces that aren't in any file's scene, like the terrain's. The file needs no meshes or scenes.
std::uint32_t add_materials(Scene &scene, const std::filesystem::path &path);

// Adds `count` coloured point and spot lights to `scene`, spread through its box, the same ones every run: something to test many lights with. Every fourth is a spot shining down; every 64th has no range of its own.
void add_test_lights(Scene &scene, std::uint32_t count);
```

In `game-engine/src/scene.cpp`, add `#include <iterator>` after `#include <cstring>`.

In `game-engine/src/scene.cpp`, add this section before `}  // namespace`:
```cpp
    // Reading a file

    // The file, parsed, with its images' bytes still encoded. .glb packs the JSON and binary data in one file; .gltf is JSON that refers to .bin and image files, or embeds them as base64 data URIs.
    tinygltf::Model read_model(const std::filesystem::path &path) {
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        std::string error;
        std::string warning;

        loader.SetImageLoader(keep_encoded_image, nullptr);

        const bool binary = path.extension() == ".glb";
        const bool loaded = binary
            ? loader.LoadBinaryFromFile(&model, &error, &warning, path.string())
            : loader.LoadASCIIFromFile(&model, &error, &warning, path.string());

        if (!warning.empty()) {
            std::println(stderr, "glTF warning: {}", warning);
        }

        if (!loaded) {
            throw std::runtime_error("can't load " + path.string() + ": " + error);
        }

        // A file lists the extensions it can't be read without. Compressed geometry needs a decoder library we don't include, so refuse it clearly instead of reading compressed bytes as vertices. The others change how things look, not where the geometry is, and later chapters handle them.
        for (const std::string &extension : model.extensionsRequired) {
            if (extension == "KHR_draco_mesh_compression" || extension == "KHR_meshopt_compression"
                || extension == "EXT_meshopt_compression") {
                throw std::runtime_error(path.string() + " needs " + extension + ", which this loader doesn't decode");
            }
        }

        return model;
    }
```

In `game-engine/src/scene.cpp`, delete `add_ground`.

In `game-engine/src/scene.cpp`, replace `load_gltf` with:
```cpp
Scene load_gltf(const std::filesystem::path &path, const glm::dvec3 &origin) {
    tinygltf::Model model = read_model(path);

    Scene scene;
    add_materials_and_images(model, scene);
    const auto default_material = static_cast<std::uint32_t>(scene.materials.size() - 1);

    // Every primitive of every mesh, once. mesh_primitives[m] lists where mesh m's drawable primitives landed in scene.primitives.
    std::vector<std::vector<LoadedPrimitive>> mesh_primitives(model.meshes.size());

    for (std::size_t m = 0; m < model.meshes.size(); ++m) {
        for (const tinygltf::Primitive &primitive : model.meshes[m].primitives) {
            if (const auto loaded_primitive = add_primitive(model, primitive, default_material, scene)) {
                mesh_primitives[m].push_back(*loaded_primitive);
            }
        }
    }

    // The file may contain several scenes; draw its default one.
    if (model.scenes.empty()) {
        throw std::runtime_error(path.string() + " has no scenes");
    }

    const tinygltf::Scene &root = model.scenes.at(model.defaultScene >= 0 ? model.defaultScene : 0);

    for (const int node : root.nodes) {
        visit_node(model, node, glm::translate(glm::dmat4{1.0}, origin), mesh_primitives, scene);
    }

    if (scene.draws.empty()) {
        throw std::runtime_error(path.string() + " has nothing to draw");
    }

    return scene;
}
```

In `game-engine/src/scene.cpp`, add this section before `void add_test_lights(`:
```cpp
std::uint32_t add_materials(Scene &scene, const std::filesystem::path &path) {
    tinygltf::Model model = read_model(path);

    // The file's materials, textures and samplers land after the scene's, so every index they hold moves up by what was there.
    const auto first = static_cast<std::uint32_t>(scene.materials.size());
    const auto image_base = static_cast<std::int32_t>(scene.images.size());
    const auto sampler_base = static_cast<std::int32_t>(scene.samplers.size());

    Scene loaded;
    add_materials_and_images(model, loaded);
    loaded.materials.pop_back();  // the plain white default, which only primitives need

    for (SceneMaterial material : loaded.materials) {
        for (TextureRef *ref : {&material.base_color, &material.metallic_roughness, &material.normal, &material.occlusion, &material.emissive}) {
            ref->image += ref->image >= 0 ? image_base : 0;
            ref->sampler += ref->sampler >= 0 ? sampler_base : 0;
        }
        scene.materials.push_back(material);
    }

    scene.images.insert(scene.images.end(), std::make_move_iterator(loaded.images.begin()), std::make_move_iterator(loaded.images.end()));
    scene.samplers.insert(scene.samplers.end(), loaded.samplers.begin(), loaded.samplers.end());
    return first;
}
```

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
#include "includes/terrain.h"
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

    // Every graphics pipeline a frame uses. The prepass and the lighting pass have one per solid alpha mode, in solid_modes' order, and one for the terrain; see-through surfaces have only the transparency pass's.
    struct ScenePipelines {
        std::vector<vk::raii::Pipeline> prepass;
        std::vector<vk::raii::Pipeline> lighting;
        vk::raii::Pipeline terrain_prepass = nullptr;
        vk::raii::Pipeline terrain_lighting = nullptr;
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

    // Draws the terrain patches cull phase `phase` kept, with `pipeline`: one mesh dispatch, after the terrain's cull mode and front face, the same as any single-sided, unmirrored surface's. The push data names the phase's patches where a mesh's would name its instances.
    void draw_terrain_patches(
        const vk::raii::CommandBuffer &commands,
        const DrawCulling &culling,
        const DrawList &draws,
        CullPhase phase,
        const vk::raii::Pipeline &pipeline
    ) {
        commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

        const PushData push{
            .frame = draws.frame,
            .instances = culling.phases[static_cast<std::size_t>(phase)].patches.address,
        };

        commands.pushDataEXT(vk::PushDataInfoEXT{
            .offset = 0,
            .data = {.address = &push, .size = sizeof(push)},
        });

        commands.setCullMode(vk::CullModeFlagBits::eBack);
        commands.setFrontFace(front_face);
        draw_terrain(commands, culling, phase);
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
    //   1. the depth prepass, in two halves: the early cull, then every solid surface and terrain patch it keeps, its depth and normal; the depth pyramid, from that depth; the late cull against the pyramid, then the surfaces and patches it adds,
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

        // Solid surfaces only: opaque, then masked, then the terrain.
        for (std::size_t i = 0; i < solid_modes.size(); ++i) {
            draw_mode(commands, culling, draws, CullPhase::early, solid_modes[i], pipelines.prepass[i]);
        }

        draw_terrain_patches(commands, culling, draws, CullPhase::early, pipelines.terrain_prepass);
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

        draw_terrain_patches(commands, culling, draws, CullPhase::late, pipelines.terrain_prepass);
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

        for (const CullPhase phase : cull_phases) {
            draw_terrain_patches(commands, culling, draws, phase, pipelines.terrain_lighting);
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

        pipelines.terrain_prepass = create_terrain_pipeline(device, std::array{normal_format}, depth_format, MeshPass::depth_normals);
        pipelines.terrain_lighting = create_terrain_pipeline(device, std::array{hdr_format}, depth_format, MeshPass::lighting);

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

        // The terrain under the scene: a height field 8 km across, whose flat middle is put 1 cm below the scene's lowest point, so it never fights the scene's own floor, and whose materials come from a file of their own. Every scene stands on it.
        const std::filesystem::path terrain_dir = std::filesystem::path(ASSET_DIR) / "terrain";
        const std::uint32_t terrain_materials = add_materials(scene, terrain_dir / "terrain.gltf");

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

        // The terrain's samples and bounds, on the GPU; its plateau is at -1 m in the file. The 8 km field repeats 8 times each way, 64 km, to the horizon and past it.
        constexpr std::uint32_t terrain_tiles = 8;
        const std::uint64_t terrain_start = SDL_GetTicksNS();
        const Terrain terrain = load_terrain(device, *gpu, queue, command_pool, terrain_dir / "field.json", scene_origin, terrain_tiles,
            static_cast<float>(scene.bounds_min.y) - 0.01f + 1.0f, terrain_materials, terrain_materials + 1, 4.0f);

        std::println("Terrain: {} x {} samples, {} patch levels, {} patches in {:.0f} ms", terrain.data.samples, terrain.data.samples,
            terrain.patch_levels, terrain.patch_count, static_cast<double>(SDL_GetTicksNS() - terrain_start) * 1e-6);

        // The cull, its groups and its draw lists, worked out for this scene's draws, and the terrain it picks patches of.
        const DrawCulling culling = create_draw_culling(device, *gpu, queue, command_pool, cull_draws, terrain);

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
            // The camera's height above the terrain. The simulated sky is lit for it too, so climbing far enough, 100 m, rebuilds it.
            const auto altitude = static_cast<float>(std::max(camera.position.y - terrain_height_at(terrain, camera.position), 1.0));
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
                || totals.early_commands != shown_totals.early_commands || totals.late_commands != shown_totals.late_commands
                || totals.early_patches != shown_totals.early_patches || totals.late_patches != shown_totals.late_patches) {
                const auto minutes = static_cast<int>(std::lround(settings.hours * 60.0f));
                const std::string sky = settings.sky == SkySource::atmosphere
                    ? std::format("sky {:02}:{:02}", minutes / 60, minutes % 60)
                    : std::string("photographed sky");
                const std::string title = std::format("game-engine: {}, {}, EV {:.1f}, AO {}, drawn {} of {} ({} early, {} late) in {} commands, {} patches ({} late)",
                    view_names[static_cast<std::size_t>(settings.view)], sky, ev100, settings.ambient_occlusion ? "on" : "off",
                    totals.early_draws + totals.late_draws, culling.draw_count, totals.early_draws, totals.late_draws,
                    totals.early_commands + totals.late_commands, totals.early_patches + totals.late_patches, totals.late_patches);

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

            // How far the terrain's finest level reaches: where its samples, `step` apart, project to terrain_edge_pixels. A sample `step` metres across at distance d covers step x focal / d pixels, with the focal length in pixels half the screen's height over the tangent of half the field of view.
            const float focal = static_cast<float>(swapchain.extent.height) * 0.5f / std::tan(camera.vertical_fov * 0.5f);
            const float terrain_range = terrain.data.step * focal / terrain_edge_pixels;

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
                .terrain = terrain.info.address,
                .terrain_range = terrain_range,
                .frame_index = static_cast<std::uint32_t>(frame_count + 1),
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

## 18.8 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The terminal** shows `Terrain: 2049 x 2049 samples, 12 patch levels, 5592405 patches in 250 ms` or so, after the scene: twelve levels of patches, the repeated terrain's, and every patch there could be.
- **The image:** the box field on its plateau, hills all round, a valley running north and south through the middle, the slopes darker and rougher than the level ground.
- **The title** ends with `drawn 1427 of 16379 (1427 early, 0 late) in 1 commands, 1434 patches (0 late)` from the starting spot: Chapter 17 drew 2,312 of its 20,475, the ground tiles among them; the terrain's patches aren't draws. Fly up (hold E) and the count falls: fewer, coarser patches cover more ground. Fly 2 km toward a hill and watch the title as you cross a ridge: the late count ticks up for a frame as the far side comes into view, then returns to 0.
- **Level transitions** don't show: fly low over a hill and the surface keeps its shape as the patches under you change level. Press 3 for the vertex normal view: the shading is the drawn surface's, and it changes smoothly, without seams.
- **Shadows:** press `[` until about 17:30. The boxes cast long shadows onto the ground, and the hills onto one another and across the valley; every lamp at night (set `test_lights` to 256) lights the ground near it and is blocked by the ground behind it.
- **The horizon:** from 2 km up, the field repeats to the horizon, mirrored, so the valley recurs.
- **Sponza** (`scene_file` back to `Sponza/Sponza.gltf`) stands on the plateau as it stood on the ground.
- **No `[validation …]` lines.**

**What it costs.** Release, 1920 × 1080, RTX 5070 Laptop:

| View | Draws | Patches | Frame |
|---|---|---|---|
| Among the boxes, 10:00 | 1,427 | 1,434 | 5.2 ms |
| Among the boxes, 17:30 | 1,427 | 1,434 | 5.5 ms |
| A hillside 1.2 km out, 10:00 | 0 | 1,282 | 2.6 ms |
| The same hillside, 17:30 | 0 | 1,282 | 2.6 ms |
| The valley floor, 1.5 km out, 17:30 | 0 | 1,440 | 2.3 ms |
| 9 km away, 2.5 km up, 10:00 | 9,873 | 1,091 | 20.7 ms |

- **The terrain costs about a millisecond** among the boxes, against Chapter 17's flat ground: the frame goes from 4.3 to 5.2 ms. Most of it is the lighting pass, which now shades a landscape behind the boxes and marches the height field for every shadow ray; the prepass draws the patches' 180,000 triangles in a tenth of a millisecond.
- **Choosing the patches** costs about 0.1 ms per phase, inside the cull's time: one workgroup walking a few thousand nodes.
- **A low sun** costs little more than a high one: 5.5 against 5.2 ms among the boxes, and nothing on the hillside, where every pixel's ray marches the hills. The first version of the march, going up one level after every quad, took 36 ms there.
- **From 9 km away** the boxes are the cost again, 9,873 of them at full detail: the next chapter's work. The terrain to the horizon is 1,091 patches, 140,000 triangles.

Next, in Chapter 19, meshes get the same treatment: clusters of about a hundred triangles, a hierarchy of simplified versions, and the mesh shader choosing each cluster's level by its size on screen, so a box far away costs a few triangles instead of six thousand.

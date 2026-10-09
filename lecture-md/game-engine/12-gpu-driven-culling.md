# Chapter 12: GPU-driven culling

By the end of this chapter, the CPU no longer decides what to draw. Until now, every pass looped over every draw on the CPU, setting state and recording one `vkCmdDrawIndexed` each, whether or not the draw could be seen. Now compute shaders at the start of the frame test every draw's box against the camera's view, and write **indirect draw commands** for the ones that survive. Each pass then draws them with a handful of **indirect count** calls, which read both the commands and how many there are from GPU memory. The CPU never learns which draws were kept, and records the same few commands every frame, however many draws the scene has.

Two more things come with it:
- **Instancing.** Draws of the same primitive with the same state become one command, whose instances are those of its draws that are in view. ABeautifulGame's chess set is 49 draws of 15 meshes, among them eight pawns of each color, each a top and a body: 15 commands.
- **A stable order.** The commands, and each command's instances, come out in the same order every frame, whatever order the GPU's threads run in. Nothing flickers between frames because a different draw happened to win a tie.

What survives is the draws whose boxes touch the view: **frustum culling**. Looking down Sponza's courtyard, about half the draws remain; facing a wall up close, one. Draws hidden behind other draws still survive. Testing for that, **occlusion culling**, comes in a later chapter. Ray-traced shadows don't depend on any of this: rays search the acceleration structures, which hold every draw.

The window title now ends with `drawn N of M in K commands`: how many draws survived the cull, and in how many commands they were drawn.

This chapter builds on [Chapter 11](11-ray-traced-shadows.md).

## 12.1 Indirect draws on the device: `vulkan_setup.cpp`

### Why
Three features let indirect draws do what CPU-recorded ones did.

### How
- **`multiDrawIndirect`** (Vulkan 1.0): one indirect call that draws more than one command. Without it, a device's `maxDrawIndirectCount` is 1, and an indirect count call draws at most one command.
- **`drawIndirectFirstInstance`** (Vulkan 1.0): commands whose `firstInstance` isn't 0. The cull uses it to tell each command where its instances' draws are listed (12.4).
- **`drawIndirectCount`** (Vulkan 1.2): `vkCmdDrawIndexedIndirectCount`, which reads how many commands to draw from a buffer, so the GPU can decide the count.
- **Picking a GPU** now also requires all three. Every desktop GPU has them.

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
        vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
        vk::PhysicalDeviceRayQueryFeaturesKHR
    >();

    const auto& vulkan10 = supported.get<vk::PhysicalDeviceFeatures2>().features;
    const auto& vulkan12 = supported.get<vk::PhysicalDeviceVulkan12Features>();
    const auto& vulkan13 = supported.get<vk::PhysicalDeviceVulkan13Features>();

    return vulkan10.independentBlend
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
        && supported.get<vk::PhysicalDeviceRayQueryFeaturesKHR>().rayQuery;
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
                .independentBlend = vk::True,           // a different blend for each color attachment
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

## 12.2 Each draw's box: `scene.h`, `scene.cpp`

### Why
To cull a draw, the GPU needs to know where it is: a box around all of its triangles, in world space.

### How
- **`MeshDraw`** gains `bounds_min` and `bounds_max`.
- **`visit_node`** already moved each primitive's box corners into world space, to grow the scene's bounds. It now boxes those 8 corners for the draw too, and grows the scene's bounds by the draw's box.
- **A box of a box:** a rotated box's corners, boxed again along the world's axes, give a box a little larger than the triangles need. That only means a draw is kept a little longer at the edge of the view. It never means one is wrongly dropped.

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

    // The draw's box in world space: the primitive's box, moved by `model`,
    // and boxed again. It holds every triangle of the draw, if a little
    // loosely when the transform rotates.
    glm::vec3 bounds_min{0.0f};
    glm::vec3 bounds_max{0.0f};
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
            MeshDraw draw{
                .model = world,
                .primitive = primitive.index,
                .mirrored = mirrored,
                .bounds_min = glm::vec3{std::numeric_limits<float>::max()},
                .bounds_max = glm::vec3{std::numeric_limits<float>::lowest()},
            };

            // The draw's box, and the scene's, take in the 8 corners of the
            // primitive's box, moved into world space.
            for (int corner = 0; corner < 8; ++corner) {
                const glm::vec3 local{
                    corner & 1 ? primitive.local_max.x : primitive.local_min.x,
                    corner & 2 ? primitive.local_max.y : primitive.local_min.y,
                    corner & 4 ? primitive.local_max.z : primitive.local_min.z,
                };
                const glm::vec3 point = glm::vec3(world * glm::vec4(local, 1.0f));

                draw.bounds_min = glm::min(draw.bounds_min, point);
                draw.bounds_max = glm::max(draw.bounds_max, point);
            }

            scene.bounds_min = glm::min(scene.bounds_min, draw.bounds_min);
            scene.bounds_max = glm::max(scene.bounds_max, draw.bounds_max);
            scene.draws.push_back(draw);
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

## 12.3 The cull's data: `shader_types.h`, `shared.slangh`

### Why
The cull tests each draw's box, groups draws into instanced commands, and needs somewhere to put every intermediate result. The vertex shader needs to find which draw each instance is.

### How
- **`DrawData`** gains its box, `bounds_min` and `bounds_max`: 164 bytes, still with no padding.
- **`FrameData`** gains `instances`, after `draws`: the cull's list of visible draws, which the vertex shader reads (12.5). 272 bytes, no padding.
- **`PushData`** loses `draw_index`. It's pushed once per pipeline now, not once per draw, and each instance finds its draw through `instances`. That also ends the one padding exception: `PushData` is a single 8-byte address.
- **`DrawGroup`**, 24 bytes: draws of one primitive in one draw list (12.4), drawn as one command. It holds where its draws start in the cull's order, how many there are, its list, and the primitive's `drawIndexed` arguments.
- **`DrawListRange`**, 8 bytes: a list's run of groups.
- **`CullTables`**, 88 bytes: the addresses of all of the cull's buffers, plus the numbers of draws and groups, in one buffer every step reads. Every step's push data is then just two addresses, the frame's data and the tables: **`CullPushData`**, 16 bytes.

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

// One per draw, in a GPU buffer the shaders index. A shadow ray that hits a
// draw's triangle finds the triangle's vertices through first_index and
// vertex_offset, as drawIndexed does. The cull tests the draw's box.
struct DrawData {
    glm::mat4 model;            // this primitive's space -> world space
    glm::mat4 normal_matrix;    // transposed inverse of model: keeps normals perpendicular under any scale
    std::uint32_t material;     // index into the material buffer
    std::uint32_t first_index;  // where the primitive's indices start in the index buffer
    std::int32_t vertex_offset; // added to each index: where its vertices start in the vertex buffer
    glm::vec3 bounds_min;       // the draw's box in world space, which holds all of its triangles
    glm::vec3 bounds_max;
};

static_assert(sizeof(DrawData) == 164);
static_assert(offsetof(DrawData, normal_matrix) == 64);
static_assert(offsetof(DrawData, material) == 128);
static_assert(offsetof(DrawData, bounds_min) == 140);

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
struct FrameData {
    glm::mat4 view_projection;          // world space -> clip space
    glm::mat4 inverse_view_projection;  // clip space -> world space, for the sky's view directions
    vk::DeviceAddress vertices;         // the scene's vertices
    vk::DeviceAddress indices;          // the scene's indices, for shadow rays' alpha tests
    vk::DeviceAddress draws;            // one DrawData per draw
    vk::DeviceAddress instances;        // the cull's visible draws: what each instance draws
    vk::DeviceAddress materials;        // the scene's materials
    vk::DeviceAddress lights;           // the file's lights
    vk::DeviceAddress environment;      // the EnvironmentInfo
    vk::DeviceAddress scene_tlas;       // the top-level acceleration structure, for ray queries
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

static_assert(sizeof(FrameData) == 272);
static_assert(offsetof(FrameData, vertices) == 128);
static_assert(offsetof(FrameData, instances) == 152);
static_assert(offsetof(FrameData, scene_tlas) == 184);
static_assert(offsetof(FrameData, camera_position) == 192);
static_assert(offsetof(FrameData, sun_direction) == 208);
static_assert(offsetof(FrameData, sun_illuminance) == 224);
static_assert(offsetof(FrameData, sky_cube) == 240);
static_assert(offsetof(FrameData, sun_angular_radius) == 260);
static_assert(offsetof(FrameData, ambient_occlusion) == 264);

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
// and a few helpers, included by mesh.slang, background.slang, ao.slang and
// cull.slang.
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
    uint first_index;        // where the primitive's indices start
    int vertex_offset;       // added to each index
    float3 bounds_min;       // the draw's box in world space
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
    float3 position;
    float range;       // 0: no limit
    float3 direction;  // the way the light shines
    float spot_scale;
    float3 intensity;  // lux (directional) or candela (point, spot), per channel
    float spot_offset;
    uint type;
};

// View: what the fragment shader outputs (keys 1-9, then 0).
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
    uint* indices;                     // the scene's indices
    DrawData* draws;                   // one DrawData per draw
    uint* instances;                   // the cull's visible draws: what each instance draws
    Material* materials;               // the scene's materials
    Light* lights;                     // the file's lights
    EnvironmentInfo* environment;      // the sky's diffuse light and the sun
    uint64_t scene_tlas;               // the top-level acceleration structure's address
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

// Written with vkCmdPushDataEXT before each pipeline's draws.
struct PushData {
    FrameData* frame;  // this frame's data
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

## 12.4 Lists, groups and the cull: `culling.h`, `culling.cpp`, `cull.slang`

### Why
An indirect count call draws many commands with one pipeline and one dynamic state. Our draws need three things that differ between them: the pipeline (by alpha mode), the cull mode (single- or double-sided) and the front face (mirrored or not). So draws are sorted into **lists**, one per combination, and each list is drawn with its own call. Within a list, draws of the same primitive form a **group**, drawn as one instanced command.

The cull then has to turn "which draws are in view" into those commands, in a fixed order. Letting each surviving draw grab the next free command with an atomic add would be simpler, but the order would then depend on which thread got there first, and change from frame to frame. Two surfaces at exactly the same depth, from different draws, would then flicker: the prepass keeps whichever comes first, and the lighting pass shows whichever comes last. Instead, every result goes to a place fixed by **prefix sums**.

### How
- **Twelve lists:** three alpha modes, times single- or double-sided, times mirrored or not. `draw_list_index` numbers them.
- **The cull's order,** worked out once by `create_draw_culling`: draw indices sorted by list, then by primitive. `stable_sort` keeps draws that tie in draw order, so the order is fixed. In this order:
  - a **group** is a run of draws with the same list and primitive,
  - a **list** is a run of groups.
- **One command slot per group:** a list's run of groups is also its run of commands. A list can never overflow: it has a slot for every group it has.
- **Prefix sums:** given a number per item, the exclusive prefix sum at item `i` is the sum of all numbers before it. Given 1 for each item to keep and 0 for each to drop, it's where item `i` goes when only the kept items are packed together, in their order. That's **stream compaction**, and it takes no atomics.
- **The six steps, every frame:**
  1. **`cullMain`**, one thread per draw in the order: 1 if the draw's box is in view, else 0.
  2. **`scanDrawsMain`:** the prefix sums of those, `draw_slots`. A visible draw's slot is where it goes among the **instances**. Since a group's draws are a run of the order, its visible draws are a run of the instances too, from `draw_slots[first]` to `draw_slots[first + count]`.
  3. **`markGroupsMain`**, one thread per group: 1 if any of its draws is visible.
  4. **`scanGroupsMain`:** the prefix sums of those, `group_slots`. A visible group's command goes in its list's run, after the visible groups before it in that list.
  5. **`writeInstancesMain`**, one thread per draw: each visible draw writes its index into its slot among the instances.
  6. **`writeCommandsMain`**, one thread per group: each visible group writes its command, `drawIndexed`'s arguments with its run of instances: `instanceCount` is how many of its draws are visible, and `firstInstance` where they start. Each list's first group also writes the list's count of commands.
- **Is a box in view?** A point is on screen when its clip-space position, `M · p` for the view-projection matrix `M`, satisfies `−w ≤ x ≤ w`, `−w ≤ y ≤ w` and `0 ≤ z ≤ w`. Each of those six inequalities is a plane in world space, read straight off `M`'s rows (Gribb and Hartmann 2001): `w − x ≥ 0` is `(row 3 − row 0) · (p, 1) ≥ 0`. With reverse-Z, `z ≥ 0` is the far plane and `z ≤ w` the near one. The box is outside if even its corner furthest along a plane's normal is behind that plane: the centre's distance, plus the half-size measured along the normal, `dot(abs(n), half_size)`, is negative.
- **Some boxes outside the view survive.** Near the frustum's corners, a box can be in front of all six planes and still miss the view. That costs one wasted instance, whose triangles the clipper drops.
- **The prefix sums, in one workgroup:** `exclusive_scan` sums 256 numbers at a time. Within a chunk, the threads sum in 8 rounds (Hillis and Steele 1986): in each round, every thread adds the number `offset` places before its own, and `offset` doubles. Each chunk's total carries over to the next. One workgroup is plenty for tens of thousands of draws; far more would need a scan spread over many workgroups.
- **Recording the cull (`record_culling`):**
  1. **A barrier:** the previous frame's draws read the commands, counts and instances, and its copy read the totals. They must finish before anything is rewritten. Overwriting what was only read needs just that wait, no memory made visible.
  2. **The six steps,** with a barrier after each of the first four. Steps 5 and 6 don't read each other's results, so they need none between them. Every step pushes the same two addresses.
  3. **A barrier:** the indirect draws may now read the commands and counts, the vertex shaders the instances, and a copy the totals.
  4. **The totals** are the scans' last numbers: how many draws are visible, and how many commands there are. They're copied into a host-visible buffer of the frame's own, for the window title, with a barrier that makes them visible to the CPU once the frame's fence has signalled.
- **Drawing a list (`draw_list`):** `vkCmdDrawIndexedIndirectCount` reads the list's count from `counts`, and draws that many commands from the start of its run, never more than the run is long.

### Code
`game-engine/src/includes/culling.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/shader_types.h"
#include "includes/vulkan_setup.h"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstdint>
#include <span>

// --- Draw lists --------------------------------------------------------------

// One indirect draw call draws a whole list of commands with one pipeline and
// one dynamic state, so draws are sorted into lists by what those fix:
//   - the alpha mode, which picks the pipeline,
//   - whether the material is double-sided, which sets the cull mode,
//   - whether the transform mirrors, which sets the front face.
// Three alpha modes times two times two: twelve lists, numbered
// alpha mode x 4 + double-sided x 2 + mirrored.
constexpr std::uint32_t draw_list_count = 12;

constexpr std::uint32_t draw_list_index(AlphaMode alpha_mode, bool double_sided, bool mirrored) {
    return static_cast<std::uint32_t>(alpha_mode) * 4 + (double_sided ? 2 : 0) + (mirrored ? 1 : 0);
}

// What the cull needs to know about a draw: its list, which primitive it
// draws (draws of one primitive in one list can share an instanced command),
// and that primitive's drawIndexed arguments.
struct CullDraw {
    std::uint32_t list;
    std::uint32_t primitive;
    std::uint32_t index_count;
    std::uint32_t first_index;
    std::int32_t vertex_offset;
};

// --- GPU culling -------------------------------------------------------------

// The cull's compute pipelines and buffers, for a scene whose draws never
// change: the groups and lists are worked out once, here.
//   - Draws are put in an order: by list, then by primitive, then by draw
//     index. A run of draws with the same list and primitive is a group,
//     drawn as one instanced command; a list's groups are a run too.
//   - Every frame, six compute steps (cull.slang) decide which draws are in
//     view, and write each list's commands, each group's visible draws as its
//     instances, and each list's count of commands.
// Everything a step writes goes to a place fixed by prefix sums over the
// previous steps' results, never by which thread got there first: the same
// view gives the same commands, in the same order, every frame.
struct DrawCulling {
    vk::raii::Pipeline cull = nullptr;
    vk::raii::Pipeline scan_draws = nullptr;
    vk::raii::Pipeline mark_groups = nullptr;
    vk::raii::Pipeline scan_groups = nullptr;
    vk::raii::Pipeline write_instances = nullptr;
    vk::raii::Pipeline write_commands = nullptr;

    // Written once.
    Buffer order;        // draw indices, in the cull's order
    Buffer groups;       // one DrawGroup per group, in order
    Buffer lists;        // one DrawListRange per list
    Buffer tables;       // one CullTables: where all of these buffers are

    // Rewritten every frame, in this order.
    Buffer visible;      // per draw in the order: 1 if in view, else 0
    Buffer draw_slots;   // prefix sums of `visible`, then their total
    Buffer group_flags;  // per group: 1 if any of its draws is in view
    Buffer group_slots;  // prefix sums of `group_flags`, then their total
    Buffer instances;    // the visible draws' indices, group after group
    Buffer commands;     // one VkDrawIndexedIndirectCommand per group: each list's run
    Buffer counts;       // per list, how many of its commands to draw

    std::array<DrawListRange, draw_list_count> list_ranges{};  // the lists' runs, for the draw calls
    std::uint32_t draw_count = 0;
    std::uint32_t group_count = 0;
};

// `draws` has one CullDraw per draw, in draw order.
DrawCulling create_draw_culling(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    std::span<const CullDraw> draws
);

// The visible-draw and command totals, as the cull copies them out.
struct CullTotals {
    std::uint32_t visible_draws;
    std::uint32_t commands;
};

// Records the cull for the frame whose FrameData is at `frame`. Afterwards,
// indirect draws may read the commands and counts, and vertex shaders the
// instances. The totals are also copied into `readback`, a host-visible
// buffer holding one CullTotals, for the CPU to read once the frame is done.
void record_culling(
    const vk::raii::CommandBuffer& commands,
    const DrawCulling& culling,
    vk::DeviceAddress frame,
    vk::Buffer readback
);

// Draws list `list`'s commands for this frame, in one indirect call. The
// pipeline, its push data and the list's dynamic state must already be set.
void draw_list(const vk::raii::CommandBuffer& commands, const DrawCulling& culling, std::uint32_t list);
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

// A device-local buffer of `count` 32-bit numbers, written and read only by
// the GPU, at least one long: a buffer can't be empty.
Buffer gpu_numbers(const vk::raii::Device& device, const GpuChoice& gpu, std::size_t count, vk::BufferUsageFlags usage = {}) {
    return create_buffer(device, gpu, std::max<std::size_t>(count, 1) * sizeof(std::uint32_t),
        vk::BufferUsageFlagBits::eShaderDeviceAddress | usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
}

// Makes the `dst` work wait for the `src` work, and the `src` writes visible
// to the `dst` accesses: for buffers, which need no layouts.
void memory_barrier(
    const vk::raii::CommandBuffer& commands,
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
void compute_to_compute(const vk::raii::CommandBuffer& commands) {
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);
}

}  // namespace

// --- Creating ----------------------------------------------------------------

DrawCulling create_draw_culling(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    std::span<const CullDraw> draws
) {
    DrawCulling culling;
    culling.cull = create_compute_pipeline(device, "cull", "cullMain");
    culling.scan_draws = create_compute_pipeline(device, "cull", "scanDrawsMain");
    culling.mark_groups = create_compute_pipeline(device, "cull", "markGroupsMain");
    culling.scan_groups = create_compute_pipeline(device, "cull", "scanGroupsMain");
    culling.write_instances = create_compute_pipeline(device, "cull", "writeInstancesMain");
    culling.write_commands = create_compute_pipeline(device, "cull", "writeCommandsMain");
    culling.draw_count = static_cast<std::uint32_t>(draws.size());

    // The order: by list, then by primitive. stable_sort keeps draws that tie
    // in draw order, so the order, and every frame's commands, are fixed.
    std::vector<std::uint32_t> order(draws.size());
    std::iota(order.begin(), order.end(), 0u);
    std::ranges::stable_sort(order, [&](std::uint32_t a, std::uint32_t b) {
        return draws[a].list != draws[b].list ? draws[a].list < draws[b].list : draws[a].primitive < draws[b].primitive;
    });

    // Groups: runs of the order with one list and one primitive. Lists: runs
    // of groups, which start where the previous list's end.
    std::vector<DrawGroup> groups;

    for (std::uint32_t position = 0; position < order.size(); ++position) {
        const CullDraw& draw = draws[order[position]];

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
    for (DrawListRange& range : culling.list_ranges) {
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

    // What the steps rewrite every frame. The draws read the commands and
    // counts as indirect arguments, and the counts' totals are copied out.
    culling.visible = gpu_numbers(device, gpu, culling.draw_count);
    culling.draw_slots = gpu_numbers(device, gpu, culling.draw_count + 1, vk::BufferUsageFlagBits::eTransferSrc);
    culling.group_flags = gpu_numbers(device, gpu, culling.group_count);
    culling.group_slots = gpu_numbers(device, gpu, culling.group_count + 1, vk::BufferUsageFlagBits::eTransferSrc);
    culling.instances = gpu_numbers(device, gpu, culling.draw_count);
    culling.commands = create_buffer(device, gpu, std::max<vk::DeviceSize>(culling.group_count, 1) * command_size,
        vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eIndirectBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    culling.counts = gpu_numbers(device, gpu, draw_list_count, vk::BufferUsageFlagBits::eIndirectBuffer);

    const CullTables tables{
        .order = culling.order.address,
        .groups = culling.groups.address,
        .lists = culling.lists.address,
        .visible = culling.visible.address,
        .draw_slots = culling.draw_slots.address,
        .group_flags = culling.group_flags.address,
        .group_slots = culling.group_slots.address,
        .instances = culling.instances.address,
        .commands = culling.commands.address,
        .counts = culling.counts.address,
        .draw_count = culling.draw_count,
        .group_count = culling.group_count,
    };
    culling.tables = upload(std::as_bytes(std::span(&tables, 1)));

    return culling;
}

// --- Recording ---------------------------------------------------------------

void record_culling(
    const vk::raii::CommandBuffer& commands,
    const DrawCulling& culling,
    vk::DeviceAddress frame,
    vk::Buffer readback
) {
    // The previous frame's draws read the commands, counts and instances, and
    // its copy read the totals: wait for them before rewriting any. Rewriting
    // what was read only needs the wait, so no access is made visible.
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eVertexShader
            | vk::PipelineStageFlagBits2::eCopy,
        vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);

    // Every step reads its buffers through the one table.
    const CullPushData push{.frame = frame, .tables = culling.tables.address};

    const auto step = [&](const vk::raii::Pipeline& pipeline, std::uint32_t workgroup_count) {
        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        commands.pushDataEXT(vk::PushDataInfoEXT{
            .offset = 0,
            .data = {.address = &push, .size = sizeof(push)},
        });
        commands.dispatch(workgroup_count, 1, 1);
    };

    // 1. Which draws are in view.
    step(culling.cull, workgroups(culling.draw_count));
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

    // The draws read the commands and counts as indirect arguments, and the
    // instances in their vertex shaders; the copy reads the totals.
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eVertexShader
            | vk::PipelineStageFlagBits2::eCopy,
        vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eShaderStorageRead
            | vk::AccessFlagBits2::eTransferRead);

    // The totals are the scans' last numbers.
    commands.copyBuffer(*culling.draw_slots.handle, readback, vk::BufferCopy{
        .srcOffset = culling.draw_count * sizeof(std::uint32_t),
        .dstOffset = offsetof(CullTotals, visible_draws),
        .size = sizeof(std::uint32_t),
    });
    commands.copyBuffer(*culling.group_slots.handle, readback, vk::BufferCopy{
        .srcOffset = culling.group_count * sizeof(std::uint32_t),
        .dstOffset = offsetof(CullTotals, commands),
        .size = sizeof(std::uint32_t),
    });

    // Visible to the CPU once it has waited for this frame's fence.
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
}

void draw_list(const vk::raii::CommandBuffer& commands, const DrawCulling& culling, std::uint32_t list) {
    // The GPU reads the list's count from `counts`, and draws that many of the
    // commands at the start of its run, never more than the run is long.
    const DrawListRange& range = culling.list_ranges[list];

    commands.drawIndexedIndirectCount(
        *culling.commands.handle, range.first_group * command_size,
        *culling.counts.handle, list * sizeof(std::uint32_t),
        range.group_count, static_cast<std::uint32_t>(command_size));
}
```

`game-engine/shaders/cull.slang`:
```slang
// GPU culling, in six steps (culling.h, record_culling), which together turn
// "which draws are in view" into indirect draw commands:
//   cullMain            per draw: 1 if its box is in view, else 0
//   scanDrawsMain       prefix sums of those: each visible draw's slot among
//                       the instances, and each group's first instance
//   markGroupsMain      per group: 1 if any of its draws is in view
//   scanGroupsMain      prefix sums of those: each group's command slot
//   writeInstancesMain  per visible draw: its index, into its slot
//   writeCommandsMain   per visible group: its instanced command; per list:
//                       how many commands it has
// Draws are visited in the cull's order (by list, then primitive), so a
// group's draws, and a list's groups, are runs. Every write goes to a place
// the prefix sums fix: the result doesn't depend on which thread runs first.

#include "shared.slangh"

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

// Vulkan's VkDrawIndexedIndirectCommand: what vkCmdDrawIndexed's arguments
// would be.
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
    uint* order;            // draw indices, in the cull's order
    DrawGroup* groups;
    DrawListRange* lists;
    uint* visible;          // per position in the order
    uint* draw_slots;       // prefix sums of visible, then their total
    uint* group_flags;      // per group
    uint* group_slots;      // prefix sums of group_flags, then their total
    uint* instances;        // the visible draws' indices
    DrawCommand* commands;  // per group
    uint* counts;           // per list
    uint draw_count;
    uint group_count;
};

struct CullPushData {
    FrameData* frame;
    CullTables* tables;
};

[[vk::push_constant]]
ConstantBuffer<CullPushData> push;

// --- 1. The view -----------------------------------------------------------------

// Whether any of the box from `lo` to `hi` can be inside the view.
//
// A point p is on screen when its clip-space position c = M p, with M the
// view-projection matrix, has -w <= x <= w, -w <= y <= w and 0 <= z <= w.
// Each of those six inequalities is a plane in world space, read off M's
// rows (Gribb and Hartmann 2001): w - x >= 0 is (row 3 - row 0) . (p, 1) >= 0.
// With reverse-Z, z >= 0 is the far plane and z <= w the near one. Slang's
// matrix[i] is row i, whatever the layout in memory.
//
// A box is outside if it lies wholly behind one of the planes: if even its
// corner furthest along the plane's normal is behind it. That corner is the
// box's centre plus its half-size, each axis signed like the normal.
//
// This keeps every box that's really visible. It also keeps a few that
// aren't, near the frustum's corners, where a box can be in front of every
// plane but still outside: one wasted instance each, whose triangles the
// clipper drops.
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
        row2,         // far, with reverse-Z: z >= 0
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

[shader("compute")]
[numthreads(64, 1, 1)]
void cullMain(uint3 id : SV_DispatchThreadID) {
    CullTables* tables = push.tables;
    const uint position = id.x;

    if (position >= tables.draw_count) {
        return;
    }

    const DrawData draw = push.frame.draws[tables.order[position]];
    tables.visible[position] = in_view(push.frame.view_projection, draw.bounds_min, draw.bounds_max) ? 1 : 0;
}

// --- 2 and 4. Prefix sums ----------------------------------------------------------

// The exclusive prefix sums of `input`: output[i] is the sum of input[0] to
// input[i - 1], and output[count] the sum of all of them. Given 0 or 1 per
// item, output[i] is how many items before item i were 1: where item i goes
// when only the 1s are kept, in their order.
//
// One workgroup does all of it, scan_size numbers at a time. Within a
// chunk, the threads sum in log2(scan_size) rounds (Hillis and Steele 1986):
// in each round, every thread adds the number `offset` places before its
// own, and `offset` doubles. Each chunk's total carries over to the next.
// That's enough for tens of thousands of draws; far more would call for a
// scan spread over many workgroups.
static const uint scan_size = 256;

groupshared uint scan_numbers[scan_size];
groupshared uint scan_carry;

void exclusive_scan(uint* input, uint* output, uint count, uint thread) {
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

[shader("compute")]
[numthreads(scan_size, 1, 1)]
void scanDrawsMain(uint3 id : SV_GroupThreadID) {
    CullTables* tables = push.tables;
    exclusive_scan(tables.visible, tables.draw_slots, tables.draw_count, id.x);
}

[shader("compute")]
[numthreads(scan_size, 1, 1)]
void scanGroupsMain(uint3 id : SV_GroupThreadID) {
    CullTables* tables = push.tables;
    exclusive_scan(tables.group_flags, tables.group_slots, tables.group_count, id.x);
}

// --- 3. Groups with something to draw --------------------------------------------

// A group's draws are a run of the order, so its visible draws are a run of
// the instances, from draw_slots[first] to draw_slots[first + count].
uint visible_in_group(CullTables* tables, DrawGroup group) {
    return tables.draw_slots[group.first + group.count] - tables.draw_slots[group.first];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void markGroupsMain(uint3 id : SV_DispatchThreadID) {
    CullTables* tables = push.tables;

    if (id.x >= tables.group_count) {
        return;
    }

    tables.group_flags[id.x] = visible_in_group(tables, tables.groups[id.x]) > 0 ? 1 : 0;
}

// --- 5 and 6. Writing the instances and the commands --------------------------------

[shader("compute")]
[numthreads(64, 1, 1)]
void writeInstancesMain(uint3 id : SV_DispatchThreadID) {
    CullTables* tables = push.tables;
    const uint position = id.x;

    if (position >= tables.draw_count || tables.visible[position] == 0) {
        return;
    }

    tables.instances[tables.draw_slots[position]] = tables.order[position];
}

// A visible group's command goes in its list's run, after the list's
// visible groups before it: group_slots counts those, from the list's first
// group. The command draws the group's visible draws as instances: its
// firstInstance is where they start among the instances, and the vertex
// shader looks each draw's index up there.
[shader("compute")]
[numthreads(64, 1, 1)]
void writeCommandsMain(uint3 id : SV_DispatchThreadID) {
    CullTables* tables = push.tables;
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
```

## 12.5 Which draw an instance is: `mesh.slang`

### Why
Push data can no longer say which draw is being drawn: there's one push per pipeline, many commands per call, and many instances per command.

### How
- **`SV_VulkanInstanceID`** is Vulkan's `gl_InstanceIndex`, which counts from the command's `firstInstance`. The cull pointed that at the command's run of visible draws in `instances`, so `frame.instances[instance]` is the draw this instance is.
- **The fragment shaders need it too,** for the material. The vertex shader passes it on in `VertexOutput`, as a `nointerpolation` value. Vulkan requires integer values passed between stages to be **flat**, the same for the whole triangle, which `nointerpolation` makes them.

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
    float3 world_position : POSITION;
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
    FrameData* frame = push.frame;
    const Vertex vertex = frame.vertices[vertex_id];
    const uint draw_index = frame.instances[instance];
    const DrawData draw = frame.draws[draw_index];

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
float candidate_alpha(FrameData* frame, DrawData draw, Material material, uint triangle, float2 barycentrics) {
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
float light_visibility(FrameData* frame, float3 origin, float3 direction, float distance) {
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
float3 shadow_ray_origin(float3 position, float3 face_normal, float3 l) {
    return offset_ray_origin(position, dot(face_normal, l) >= 0.0 ? face_normal : -face_normal);
}

// shade(), times how much of the light gets through. A ray is only traced
// when the light could reach the surface at all.
float3 shade_shadowed(
    Surface surface, FrameData* frame, float3 position, float3 face_normal,
    float3 l, float3 illuminance, float distance
) {
    if (dot(surface.normal, l) <= 0.0 || all(illuminance == 0.0)) {
        return float3(0.0);
    }

    const float reaching = light_visibility(frame, shadow_ray_origin(position, face_normal, l), l, distance);
    return reaching > 0.0 ? shade(surface, l, illuminance) * reaching : float3(0.0);
}

// --- Lights --------------------------------------------------------------------

// The direction toward a light, how far away it is, and the illuminance it
// gives here, following KHR_lights_punctual. Point and spot lights fade with
// the square of the distance, then smoothly to nothing at `range`; spot
// lights also fade from the inner cone to the outer one. A directional
// light is infinitely far away.
float3 punctual_light(Light light, float3 position, out float3 l, out float distance) {
    if (light.type == light_directional) {
        l = -light.direction;
        distance = infinite_distance;
        return light.intensity;
    }

    const float3 to_light = light.position - position;
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
    FrameData* frame = push.frame;
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

    // The triangle's flat normal, from how the position changes across
    // neighbouring pixels: exact up to rounding, since a triangle is flat.
    // Shadow rays start off the surface along it. A triangle seen exactly
    // edge-on has no area on screen, and no such normal: then the shading
    // normal stands in, rather than a division by zero.
    const float3 face_cross = cross(ddy(input.world_position), ddx(input.world_position));
    const float3 face_normal = dot(face_cross, face_cross) > 1e-24 ? normalize(face_cross) : normal;

    // The shadow view: how much of the sun's light gets through to each point.
    if (frame.view == view_shadow) {
        const float sun = frame.sun_direction.y > 0.0
            ? light_visibility(frame, shadow_ray_origin(input.world_position, face_normal, frame.sun_direction),
                frame.sun_direction, infinite_distance)
            : 0.0;
        return float4(sun.xxx, 1.0);
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

    // Direct light, shadowed: the sun, then every light in the file.
    float3 radiance = shade_shadowed(surface, frame, input.world_position, face_normal,
        frame.sun_direction, frame.sun_illuminance, infinite_distance);

    for (uint i = 0; i < frame.light_count; ++i) {
        float3 l;
        float distance;
        const float3 illuminance = punctual_light(frame.lights[i], input.world_position, l, distance);
        radiance += shade_shadowed(surface, frame, input.world_position, face_normal, l, illuminance, distance);
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

## 12.6 Drawing the lists: `main.cpp`

### Why
`main` creates the cull once the draws are known, and every pass draws from the lists.

### How
- **At load:** each draw's `DrawData` gets its box, and a `CullDraw` says which list and primitive it belongs to. `create_draw_culling` works out the order, the groups and the lists, and the terminal prints how many draws and groups there are. `alpha_modes` and the per-mode batches are gone; `solid_modes` names the modes the prepass and the lighting pass draw.
- **`draw_mode`** replaces `draw_batch`. It binds the pass's pipeline for one alpha mode, pushes the frame's address, then draws each of the mode's four lists, skipping lists with no groups, with that list's cull mode and front face.
- **`record_frame`** records the cull right after binding the heaps and the index buffer, before the prepass.
- **Each frame in flight** has a small host-visible buffer for the cull's totals. After waiting for the frame's fence, the CPU reads the totals from the last time that frame was used, two frames ago, for the title. They start at zero, so the first two frames' titles show nothing drawn.
- **The transparency pass** runs if the scene has blended draws at all. Whether any survive the cull is only known on the GPU; if none do, the pass draws nothing, and the composite discards every pixel.

### Code
`game-engine/src/main.cpp`:
```cpp
#include "includes/acceleration.h"
#include "includes/ambient_occlusion.h"
#include "includes/buffer.h"
#include "includes/camera.h"
#include "includes/culling.h"
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
    Buffer cull_totals;                            // the cull's totals, copied out, host-visible
    const CullTotals* totals = nullptr;            // `cull_totals`, mapped for the CPU to read
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

// The alpha modes the prepass and the lighting pass draw, in the order of
// their pipelines. The see-through mode, AlphaMode::blend, has only the
// transparency pass's.
constexpr std::array solid_modes{AlphaMode::opaque, AlphaMode::mask};

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
    std::uint32_t accum = 0;           // sampled, by the transparency composite
    std::uint32_t reveal = 0;          // sampled, by the transparency composite
};

constexpr std::uint32_t screen_slot_count = 8;

// Every graphics pipeline a frame uses. The prepass and the lighting pass
// have one per solid alpha mode, in solid_modes' order; see-through surfaces
// have only the transparency pass's.
struct ScenePipelines {
    std::vector<vk::raii::Pipeline> prepass;
    std::vector<vk::raii::Pipeline> lighting;
    vk::raii::Pipeline transparency = nullptr;
    vk::raii::Pipeline background = nullptr;
    vk::raii::Pipeline composite = nullptr;
    vk::raii::Pipeline tonemap = nullptr;
};

// What a frame draws and where it finishes: the scene's index buffer, the
// frame's data, the screen images' slots, and where the cull's totals go.
struct DrawList {
    vk::Buffer index_buffer;
    vk::DeviceAddress frame = 0;   // this frame's FrameData
    ScreenSlots screen;
    View view = View::lit;
    vk::Buffer readback;           // this frame's copy of the cull's totals
    bool see_through = false;      // whether the scene has blended draws at all
};

// Draws every list of alpha mode `mode` with `pipeline`: one indirect call
// per list, after setting the list's cull mode and front face. Push data
// says where the frame's data is; each instance finds its DrawData through
// the cull's instances.
void draw_mode(
    const vk::raii::CommandBuffer& commands,
    const DrawCulling& culling,
    const DrawList& draws,
    AlphaMode mode,
    const vk::raii::Pipeline& pipeline
) {
    commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

    const PushData push{.frame = draws.frame};

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

            // Single-sided surfaces are invisible from behind, so the GPU can
            // skip their back faces before running the fragment shader.
            commands.setCullMode(double_sided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack);
            commands.setFrontFace(mirrored ? mirrored_front_face : front_face);
            draw_list(commands, culling, list);
        }
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

// Records a frame: the cull, then five passes:
//   1. the depth prepass: every solid surface's depth and vertex normal,
//   2. ambient occlusion, from those, in compute shaders,
//   3. the lighting, into the HDR image: each solid alpha mode's lists with
//      that mode's pipeline, against the prepass's depth, then the sky
//      behind them,
//   4. transparency, if anything is see-through: the blended lists into two
//      sums, in any order, then those laid over the HDR image,
//   5. tone mapping, from the HDR image into the swapchain image, which is
//      then ready to present.
void record_frame(
    const vk::raii::CommandBuffer& commands,
    const Swapchain& swapchain,
    std::uint32_t image_index,
    const ScenePipelines& pipelines,
    const AmbientOcclusion& ambient_occlusion,
    const DrawCulling& culling,
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

    // --- The cull --------------------------------------------------------------

    // Every pass below draws only what the cull keeps, from its lists.
    record_culling(commands, culling, draws.frame, draws.readback);

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
    for (std::size_t i = 0; i < solid_modes.size(); ++i) {
        draw_mode(commands, culling, draws, solid_modes[i], pipelines.prepass[i]);
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

    for (std::size_t i = 0; i < solid_modes.size(); ++i) {
        draw_mode(commands, culling, draws, solid_modes[i], pipelines.lighting[i]);
    }

    // The sky goes in once everything solid is drawn: it only covers pixels
    // still at the far plane.
    commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.background);

    const PushData sky_push{.frame = draws.frame};
    commands.pushDataEXT(vk::PushDataInfoEXT{
        .offset = 0,
        .data = {.address = &sky_push, .size = sizeof(sky_push)},
    });

    commands.draw(3, 1, 0, 0);
    commands.endRendering();

    // --- Pass 4: transparency ------------------------------------------------

    if (draws.see_through) {
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
        draw_mode(commands, culling, draws, AlphaMode::blend, pipelines.transparency);
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
    "Sun shadow",
};

// Handles every pending event and fills in `input` for this frame. False once
// the window was closed or Escape pressed.
//   1-9 0 pick the view: 0 is the tenth, as on the keyboard
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

            // SDLK_1 to SDLK_9 are consecutive key codes; SDLK_0 comes before them.
            if (key >= SDLK_1 && key <= SDLK_9) {
                settings.view = static_cast<View>(key - SDLK_1);
            } else if (key == SDLK_0) {
                settings.view = View::shadow;
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
            std::println(stderr, "No GPU has Vulkan 1.4, the descriptor heap and ray queries, and can present to this window");
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

            // Starts zeroed: the first wait on each frame reads it before
            // the GPU has written it.
            Buffer cull_totals = create_buffer(device, *gpu, sizeof(CullTotals), vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            auto* totals = static_cast<CullTotals*>(cull_totals.memory.mapMemory(0, sizeof(CullTotals)));
            *totals = CullTotals{.visible_draws = 0, .commands = 0};

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

        // --- Scene -----------------------------------------------------------

        // The glTF file to draw, under lecture-md/game-engine/assets.
        const std::filesystem::path scene_file = std::filesystem::path(ASSET_DIR) / "Sponza/Sponza.gltf";

        const std::uint64_t load_start = SDL_GetTicksNS();
        const Scene scene = load_gltf(scene_file);
        std::println("Loaded {}: {} vertices, {} triangles, {} primitives, {} draws, {} materials, {} images",
            scene_file.filename().string(), scene.vertices.size(), scene.indices.size() / 3,
            scene.primitives.size(), scene.draws.size(), scene.materials.size(), scene.images.size());

        // Each draw's matrices, triangles and box, and what the cull needs to
        // group it. The normal matrix is the transposed inverse of the model
        // matrix: under non-uniform scale, transforming a normal by the model
        // matrix itself would tilt it off the surface.
        std::vector<DrawData> draw_data;
        std::vector<CullDraw> cull_draws;
        std::array<std::size_t, 3> mode_draws{};  // how many draws of each alpha mode

        for (const MeshDraw& draw : scene.draws) {
            const Primitive& primitive = scene.primitives[draw.primitive];
            const SceneMaterial& material = scene.materials[primitive.material];
            const std::uint32_t list = draw_list_index(material.alpha_mode, material.double_sided, draw.mirrored);

            draw_data.push_back(DrawData{
                .model = draw.model,
                .normal_matrix = glm::transpose(glm::inverse(draw.model)),
                .material = primitive.material,
                .first_index = primitive.first_index,
                .vertex_offset = primitive.vertex_offset,
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

        // Vertices and draw data are read through pointers; indices go to the
        // GPU's index fetch, so that buffer is an index buffer. Vertices and
        // indices are also what the acceleration structures are built from,
        // and shadow rays read indices through a pointer too.
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

        // Most files have no lights, and a buffer can't be empty: then there's
        // no buffer, and the shader's light count is 0.
        const Buffer light_buffer = scene.lights.empty() ? Buffer{} : upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.lights)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        std::println("Lights: {} from the file, plus the sun", scene.lights.size());

        // --- Acceleration structures -----------------------------------------

        const std::uint64_t acceleration_start = SDL_GetTicksNS();
        const AccelerationStructures acceleration =
            build_acceleration_structures(device, *gpu, queue, command_pool, scene, vertex_buffer, index_buffer);

        std::println("Acceleration structures: {} BLAS, {} instances in {:.0f} ms", acceleration.blases.size(),
            scene.draws.size(), static_cast<double>(SDL_GetTicksNS() - acceleration_start) * 1e-6);

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
            .accum = first_screen_slot + 6,
            .reveal = first_screen_slot + 7,
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

        // Spawns at the origin, looking down -Z.
        FlyCamera camera;
        CameraInput input;
        Settings settings;
        Settings shown_settings{.hours = -1.0f};  // what the title shows; differs at first
        CullTotals shown_totals{};                // the cull's totals the title shows
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

            // --- Render -----------------------------------------------------

            Frame& frame = frames[frame_count % frames_in_flight];

            // 1. Wait until the GPU is done with this frame's command buffer and
            //    data from last time, then write this frame's data.
            (void)device.waitForFences(*frame.done, vk::True, no_timeout);

            // What the cull kept the last time this frame's resources were
            // used: two frames ago.
            const CullTotals totals = *frame.totals;

            // The title shows the view, the sky, the time, the exposure,
            // whether ambient occlusion is on and what the cull kept,
            // whenever one changes.
            if (settings.view != shown_settings.view || settings.sky != shown_settings.sky
                || settings.hours != shown_settings.hours
                || settings.exposure_compensation != shown_settings.exposure_compensation
                || settings.ambient_occlusion != shown_settings.ambient_occlusion
                || totals.visible_draws != shown_totals.visible_draws || totals.commands != shown_totals.commands) {
                const auto minutes = static_cast<int>(std::lround(settings.hours * 60.0f));
                const std::string sky = settings.sky == SkySource::atmosphere
                    ? std::format("sky {:02}:{:02}", minutes / 60, minutes % 60)
                    : std::string("photographed sky");
                const std::string title = std::format("game-engine: {}, {}, EV {:.1f}, AO {}, drawn {} of {} in {} commands",
                    view_names[static_cast<std::size_t>(settings.view)], sky, ev100,
                    settings.ambient_occlusion ? "on" : "off", totals.visible_draws, culling.draw_count, totals.commands);

                SDL_SetWindowTitle(window.get(), title.c_str());
                shown_settings = settings;
                shown_totals = totals;
            }

            const glm::mat4 view_projection = camera.projection(aspect) * camera.view();

            *frame.mapped = FrameData{
                .view_projection = view_projection,
                .inverse_view_projection = glm::inverse(view_projection),
                .vertices = vertex_buffer.address,
                .indices = index_buffer.address,
                .draws = draw_buffer.address,
                .instances = culling.instances.address,
                .materials = material_buffer.address,
                .lights = light_buffer.address,
                .environment = environment.info.address,
                .scene_tlas = acceleration.tlas_address,
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
                .readback = *frame.cull_totals.handle,
                .see_through = mode_draws[static_cast<std::size_t>(AlphaMode::blend)] > 0,
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
            record_frame(frame.commands, swapchain, image_index, pipelines, ambient_occlusion, culling, heaps, draws);

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

## 12.7 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The terminal** shows `Culling: 103 draws in 103 groups`: each of Sponza's primitives is drawn once, so no draws share a command there.
- **The image** is the same as Chapter 11's: the cull only drops draws that couldn't have shown. On Sponza, from the same camera, the two are identical pixel for pixel.
- **The title** ends with `drawn 51 of 103 in 51 commands` at the start, looking down the courtyard. Turning the camera changes it. Looking almost straight up still draws 48: many of Sponza's draws are large pieces of the building, whose boxes surround the camera. Walking up to the end wall until it fills the screen brings it down to 1.
- **Shadows** still fall from draws that are culled: a shadow ray searches the acceleration structures, which hold every draw, so whatever casts a shadow into the view needn't be in it.
- **ABeautifulGame** (change `scene_file` in `main.cpp`): `Culling: 49 draws in 15 groups`, and with the whole board in view, `drawn 49 of 49 in 15 commands`. Moving in until some pieces leave the view lowers both: a group's command stays while any of its pieces is visible, with fewer instances.
- **Other models** draw as before: NegativeScaleTest shows every check mark, its mirrored copies in their own lists, and AlphaBlendModeTest every panel.
- **No `[validation …]` lines.**

Next, in Chapter 13, the view reaches as far as the horizon: an infinite far plane, and positions measured from the camera, so that precision holds kilometres from the world's origin.

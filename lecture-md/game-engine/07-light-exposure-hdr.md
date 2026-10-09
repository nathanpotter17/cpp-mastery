# Chapter 7: Light, exposure and HDR

By the end of this chapter Sponza is lit the way a camera would see it: by a sun that moves across a midsummer sky, with light measured in physical units, exposed like a photograph and tone mapped for the display. The keyboard steps through the day from sunrise to a pitch-black night, and the exposure follows, as a camera's light meter would. The materials of Chapter 6 are shaded with glTF's own physically based BRDF, and lights stored in a glTF file now shine too.

Four ideas carry the chapter:
- **Physical units.** Light is measured as it is in the real world: sunlight at noon is about 100,000 lux, and a candle is about 1 candela. glTF's lights already use these units.
- **HDR rendering.** Those values range over many orders of magnitude, so the scene is drawn into a 16-bit floating-point image, not straight into the 8-bit swapchain image.
- **Exposure and tone mapping.** A camera's exposure scales the light down to a usable range; a tone mapper then compresses what's still too bright, as film does, into colors a display can show.
- **Per-frame data.** The sun, the exposure and the camera change every frame, and every draw needs them. Rather than push them with every draw, each frame gets a small buffer the CPU writes, and push data shrinks to a pointer to it.

There are no shadows yet: sunlight reaches every surface that faces it, inside or out. Ray-traced shadows come in Chapter 11. Chapter 8 replaces this chapter's simple sky with image-based lighting.

This chapter builds on [Chapter 6](06-materials.md).

## 7.1 Lights, frame data and push data: `shader_types.h`

### Why
Every draw now needs the camera's position, the exposure, the sun and the ambient sky, plus the scene's buffers and lights. That's far more than the ~100 bytes push data has carried so far, and none of it changes from one draw to the next.

### How
- **`Light`:** one `KHR_lights_punctual` light, 52 bytes, in world space.
  - **Types:** a directional light (like the sun) shines parallel rays, measured in **lux** (lumens per square meter, the light falling on a surface). A point light shines in every direction, and a spot light within a cone; both are measured in **candela** (lumens per steradian, the light sent in one direction).
  - **Spot cones:** `spot_scale` and `spot_offset` are precomputed from the cone angles, so the shader's falloff is a single multiply-add (7.2).
- **`FrameData`:** everything that's the same for every draw in a frame, 168 bytes.
  - **Buffers:** the four buffer addresses: vertices, draws, materials and lights.
  - **Camera and exposure:** the camera position, needed for specular reflections, and the exposure.
  - **The sun:** its direction, and its illuminance in lux per color channel.
  - **The ambient sky:** the brightness of the sky above and the ground below, in **nits** (candela per square meter, the brightness of a surface).
  - **The view,** moved here from push data.
  - **The layout:** the pointers come right after the matrix, at offset 64, so all of them land on 8-byte boundaries, and the struct has no padding at all.
- **Push data** shrinks to 12 bytes: the address of this frame's `FrameData`, and the draw index. The C++ struct is 16 bytes: a struct with an 8-byte member is padded to a multiple of 8, so `PushData` is the one struct with (4 bytes of) unavoidable padding at its end. The shader ignores it.
- **`TonemapPushData`:** the tone-mapping pass's push data (7.7): which resource heap slot holds the HDR image, and the view.

### Code
`game-engine/src/includes/shader_types.h`:
```cpp
#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <cstddef>
#include <cstdint>

// C++ mirrors of the structs in shaders/mesh.slang and shaders/tonemap.slang. The GPU reads these bytes as they are, so the two sides must agree on every size and offset; the static_asserts catch a mismatch at compile time.

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

// Per-frame data

// Everything the shaders need that's the same for every draw in a frame. Each frame in flight has its own copy in host-visible memory, rewritten by the CPU before the frame is recorded. Push data points at it.
//   - Lighting values are physical: lux for illuminance, nits (candela per square meter) for the brightness of the sky.
//   - Pointers come right after the matrix, so all of them land on 8-byte boundaries with no padding.
struct FrameData {
    glm::mat4 view_projection;     // world space -> clip space
    vk::DeviceAddress vertices;    // the scene's vertices
    vk::DeviceAddress draws;       // one DrawData per draw
    vk::DeviceAddress materials;   // the scene's materials
    vk::DeviceAddress lights;      // the file's lights
    glm::vec3 camera_position;     // world space
    float exposure;                // scales light into the 0..1 range the tone mapper expects
    glm::vec3 sun_direction;       // unit vector pointing toward the sun
    std::uint32_t light_count;     // how many Lights `lights` holds
    glm::vec3 sun_illuminance;     // lux, per color channel, on a surface facing the sun
    View view;                     // what the fragment shader outputs
    glm::vec3 sky_radiance;        // nits: brightness of the sky, the ambient light from above
    glm::vec3 ground_radiance;     // nits: brightness of the ground, the ambient light from below
};

static_assert(sizeof(FrameData) == 168);
static_assert(offsetof(FrameData, vertices) == 64);
static_assert(offsetof(FrameData, lights) == 88);
static_assert(offsetof(FrameData, camera_position) == 96);
static_assert(offsetof(FrameData, sun_direction) == 112);
static_assert(offsetof(FrameData, sun_illuminance) == 128);
static_assert(offsetof(FrameData, sky_radiance) == 144);

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
```

## 7.2 Lights in the scene: `scene.h`, `scene.cpp`

### Why
A glTF file stores lights in the `KHR_lights_punctual` extension: a list of lights, each placed in the scene by a node, like a mesh. tinygltf 2.9.7 already parses the extension into `model.lights` and each node's `light` index; the loader only has to place them.

### How
- **Placement:** a light sits at its node's origin and shines down the node's −Z axis, so its world position is the world matrix's translation column, and its direction is the matrix's 3×3 part applied to (0, 0, −1).
- **Color and intensity** are multiplied into one RGB value. A missing color means white.
- **Spot cones:** the spec defines full brightness inside `innerConeAngle` and none outside `outerConeAngle`. With `scale = 1 / (cos(inner) − cos(outer))` and `offset = −cos(outer) × scale`, the expression `cos(angle) × scale + offset` is 1 at the inner edge and 0 at the outer one.
- **Unknown light types** are skipped, as glTF allows for extensions it doesn't know.

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

// One thing to draw: a primitive, placed in the world by a node's transform. A mesh used by several nodes is drawn once per node.
struct MeshDraw {
    glm::mat4 model{1.0f};
    std::uint32_t primitive = 0;

    // A transform that mirrors the primitive (a negative scale) reverses the order its triangles' corners appear in, which decides which side is the front.
    bool mirrored = false;

    // The middle of the primitive's box in world space, for sorting see-through draws by distance.
    glm::vec3 center{0.0f};
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

    // World-space box around everything drawn, for placing the camera.
    glm::vec3 bounds_min{std::numeric_limits<float>::max()};
    glm::vec3 bounds_max{std::numeric_limits<float>::lowest()};
};

// Loads the default scene of a .gltf or .glb file, with its materials, samplers, lights and images, still encoded.
Scene load_gltf(const std::filesystem::path &path);
```

In `game-engine/src/scene.cpp`, add `#include <cmath>` after `#include <algorithm>`.

In `game-engine/src/scene.cpp`, add this section before `// Nodes`:
```cpp
    // Lights

    // A KHR_lights_punctual light, placed by its node's world transform. The light sits at the node's origin and shines down the node's -Z axis.
    std::optional<Light> world_light(const tinygltf::Light &source, const glm::mat4 &world) {
        const glm::vec3 color = source.color.size() == 3 ? glm::vec3(glm::make_vec3(source.color.data())) : glm::vec3(1.0f);

        Light light{
            .position = glm::vec3(world[3]),
            .range = static_cast<float>(source.range),
            .direction = glm::normalize(glm::mat3(world) * glm::vec3{0.0f, 0.0f, -1.0f}),
            .spot_scale = 0.0f,
            .intensity = color * static_cast<float>(source.intensity),
            .spot_offset = 0.0f,
            .type = LightType::point,
        };

        if (source.type == "directional") {
            light.type = LightType::directional;
        } else if (source.type == "spot") {
            // Full brightness inside the inner cone, nothing outside the outer one. Precomputing a scale and offset turns the shader's falloff into one multiply-add: cos(angle) * scale + offset is 1 at the inner edge and 0 at the outer edge.
            const float cos_inner = std::cos(static_cast<float>(source.spot.innerConeAngle));
            const float cos_outer = std::cos(static_cast<float>(source.spot.outerConeAngle));
            light.type = LightType::spot;
            light.spot_scale = 1.0f / std::max(cos_inner - cos_outer, 0.001f);
            light.spot_offset = -cos_outer * light.spot_scale;
        } else if (source.type != "point") {
            return std::nullopt;
        }

        return light;
    }
```

In `game-engine/src/scene.cpp`, replace `visit_node` with:
```cpp
    // Walks the node tree. Each node's world transform is its parent's times its own; every primitive of a node's mesh becomes one draw, and a node's light is placed by the same transform.
    void visit_node(
        const tinygltf::Model &model,
        int node_index,
        const glm::mat4 &parent,
        const std::vector<std::vector<LoadedPrimitive>> &mesh_primitives,
        Scene &scene
    ) {
        const tinygltf::Node &node = model.nodes.at(node_index);
        const glm::mat4 world = parent * local_transform(node);

        if (node.mesh >= 0) {
            // A negative determinant means the transform mirrors space.
            const bool mirrored = glm::determinant(glm::mat3(world)) < 0.0f;

            for (const LoadedPrimitive &primitive : mesh_primitives.at(node.mesh)) {
                const glm::vec3 local_center = (primitive.local_min + primitive.local_max) * 0.5f;

                scene.draws.push_back(MeshDraw{
                    .model = world,
                    .primitive = primitive.index,
                    .mirrored = mirrored,
                    .center = glm::vec3(world * glm::vec4(local_center, 1.0f)),
                });

                // Grow the scene bounds by the 8 corners of the primitive's box, moved into world space.
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

## 7.3 The sun, the sky and the exposure: `daylight.h`, `daylight.cpp`

### Why
We want to test the renderer in every lighting condition, from full noon sun to black night, with one control: the time of day. That needs a model of where the sun is and how much light it and the sky give. It also needs an **exposure** that follows the light, as a camera's automatic exposure does. The sky here is deliberately simple; Chapter 8 replaces it with a simulated atmosphere.

### How
- **Where the sun is:** standard solar geometry for latitude 35° north at the June solstice, when the sun's declination is 23.4°.
  - **The hour angle** turns 15° per hour from solar noon.
  - **The direction:** three lines of trigonometry turn the latitude, declination and hour angle into a direction in east-north-up coordinates.
  - **In the scene,** +X is east, +Y up and −Z north.
  - **The result:** noon puts the sun 78° up, sunrise is just before 5:00 and sunset just after 19:00.
- **How much sunlight gets through:**
  - **In space,** the sun gives about 128,000 lux.
  - **Air mass:** on the way down, sunlight crosses the atmosphere, and a low sun crosses more of it. The *air mass*, about 1/sin(elevation), measures how much.
  - **Meinel's formula,** `0.7 ^ (air mass ^ 0.678)`, is an empirical fit for the fraction a clear sky lets through: 70% at noon, much less at sunset.
- **The sky:**
  - **Sky light:** a clear sky scatters some sunlight down, roughly a fifth of the direct sun's light on flat ground.
  - **Twilight:** keeps a faint glow until the sun is 6° below the horizon (the end of civil twilight). After that the sky is black: there's no moon or starlight.
  - **From lux to nits:** a uniformly bright sky of brightness L lights flat ground with E = πL, which turns the sky's illuminance into its brightness.
  - **The ground** reflects 20% of all the light reaching it, its *albedo*.
  - **Colors:** the sky gets a clear blue and the ground a warm grey, each scaled to a luminance of exactly 1, so they change the color but not the brightness.
- **Exposure as EV100:**
  - **What EV is:** photographers measure exposure in *exposure values*; each step of 1 halves the light reaching the sensor. EV100 is the EV for a sensor sensitivity of ISO 100.
  - **Metering:** an *incident* light meter measures the light falling on the subject, E in lux, and suggests `EV100 = log2(E × 100 / C)`, with the calibration constant C = 250. We meter the light on flat ground, so noon gives EV 15.4, and dusk about 5.
  - **Turning EV into a factor:** a sensor saturates at a scene brightness of `1.2 × 2^EV100` nits, the standard result for a typical lens. The exposure factor is one over that, so the brightest value the sensor records becomes 1.0.

**No CMake changes:** `CMakeLists.txt` collects every `.cpp` under `src/` and every `.slang` under `shaders/`, so this chapter's new `daylight.cpp` and `tonemap.slang` are built without any edit. The build reconfigures by itself to find them.

### Code
`game-engine/src/includes/daylight.h`:
```cpp
#pragma once

#include <glm/glm.hpp>

// Daylight

// The sun and sky at one time of day, in physical units. A deliberately simple model; Chapter 8 replaces the sky with a simulated atmosphere.
struct Daylight {
    glm::vec3 sun_direction{0.0f, 1.0f, 0.0f};  // unit vector toward the sun, world space
    float sun_elevation = 0.0f;                 // radians above the horizon; negative below it
    glm::vec3 sun_illuminance{0.0f};            // lux, on a surface facing the sun
    glm::vec3 sky_radiance{0.0f};               // nits: the sky's average brightness
    glm::vec3 ground_radiance{0.0f};            // nits: the ground's, lit by sun and sky
    float horizontal_illuminance = 0.0f;        // lux on flat ground: what a light meter reads
};

// The daylight at `hours` (0 to 24, local solar time) on a midsummer day at latitude 35 degrees north. The scene's -Z is north, +X east, +Y up.
Daylight daylight_at(float hours);

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
#include <numbers>

namespace {

    // The place and the date

    constexpr float latitude = glm::radians(35.0f);     // north of the equator
    constexpr float declination = glm::radians(23.4f);  // the sun's, at the June solstice

    // Sunlight before it enters the atmosphere: about 128,000 lux.
    constexpr float sunlight_in_space = 128000.0f;

    // How much of the light reaching the ground the ground reflects (its albedo).
    constexpr float ground_albedo = 0.2f;

    // Clear-sky blue and a warm ground, as unit-luminance colors: Rec. 709 luminance (0.2126 R + 0.7152 G + 0.0722 B) of each is 1.
    constexpr glm::vec3 sky_color{0.73f, 1.034f, 1.46f};
    constexpr glm::vec3 ground_color{1.116f, 0.982f, 0.837f};

    // The sun's direction

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

}  // namespace

// Daylight

Daylight daylight_at(float hours) {
    Daylight daylight;
    daylight.sun_direction = sun_direction_at(hours);
    daylight.sun_elevation = std::asin(daylight.sun_direction.y);

    const float sin_elevation = std::max(daylight.sun_direction.y, 0.0f);

    // Sunlight crosses more air the lower the sun is: the "air mass", 1 with the sun overhead, about 1/sin(elevation) lower down. Meinel's empirical formula gives the fraction that gets through a clear sky: 0.7 ^ (air mass ^ 0.678).
    if (sin_elevation > 0.0f) {
        const float air_mass = 1.0f / std::max(sin_elevation, 0.01f);
        const float transmitted = std::pow(0.7f, std::pow(air_mass, 0.678f));
        daylight.sun_illuminance = glm::vec3(sunlight_in_space * transmitted);
    }

    // The sky scatters some sunlight down: on a clear day it adds roughly a fifth of the direct sun's illuminance on flat ground. Twilight keeps a little of that glow until the sun is 6 degrees below the horizon; after that, the night is black.
    const float twilight = std::clamp((daylight.sun_elevation + glm::radians(6.0f)) / glm::radians(6.0f), 0.0f, 1.0f);
    const float direct_on_ground = daylight.sun_illuminance.x * sin_elevation;
    const float sky_illuminance = 0.2f * direct_on_ground + 400.0f * twilight * twilight;

    // A sky of uniform brightness L lights flat ground with E = pi * L, so L = E / pi. The ground reflects a fraction of what falls on it.
    daylight.horizontal_illuminance = direct_on_ground + sky_illuminance;
    daylight.sky_radiance = sky_color * (sky_illuminance / std::numbers::pi_v<float>);
    daylight.ground_radiance = ground_color * (ground_albedo * daylight.horizontal_illuminance / std::numbers::pi_v<float>);

    return daylight;
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

## 7.4 An HDR image: `swapchain.h`, `swapchain.cpp`

### Why
Even after exposure, sunlit highlights on metal can be many times brighter than white, and an 8-bit swapchain image would clip them to 1.0. The scene is drawn into an image that can hold them, and tone mapping (7.7) decides how they look.

### How
- **`hdr_format`** is `R16G16B16A16_SFLOAT`: four 16-bit floats per pixel, up to 65,504, with about three decimal digits of precision. It's the standard format for HDR rendering, half the size of 32-bit floats.
- **The HDR image belongs to the `Swapchain`,** like the depth buffer. It must always match the window's size, and `build()` creates both, so a resize rebuilds them together.
- **Its usage** is a color attachment, to draw into, and sampled, so the tone-mapping shader can read it.

### Code
`game-engine/src/includes/swapchain.h`:
```cpp
#pragma once

#include "includes/image.h"
#include "includes/vulkan_setup.h"

#include <vector>

// 32-bit float depth: the precision reverse-Z depth needs (see camera.cpp).
constexpr vk::Format depth_format = vk::Format::eD32Sfloat;

// 16-bit floats per channel for the scene before tone mapping: enough range for sunlit highlights many times brighter than white (up to 65504).
constexpr vk::Format hdr_format = vk::Format::eR16G16B16A16Sfloat;

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

    // One depth buffer and one HDR color image, the size of the swapchain images. Every frame clears both before drawing, so frames in flight can share them. The scene is drawn into `hdr`; tone mapping then writes it into the swapchain image.
    Image depth;
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

In `game-engine/src/swapchain.cpp`, replace `build` with:
```cpp
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

        swapchain.depth = create_image(device, gpu, swapchain.extent, depth_format,
            vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth);

        // Drawn into, then read by the tone-mapping shader.
        swapchain.hdr = create_image(device, gpu, swapchain.extent, hdr_format,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor);

        return swapchain;
    }
```

Then replace `recreate_swapchain` with:
```cpp
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

    // Destroy the old views and semaphores while their images still exist, then the old swapchain itself, its depth buffer and its HDR image.
    swapchain.views.clear();
    swapchain.rendered.clear();
    swapchain = std::move(next);
}
```

## 7.5 A writable descriptor heap: `descriptor_heap.h`, `descriptor_heap.cpp`

### Why
The tone-mapping shader reads the HDR image through the descriptor heap, like any texture. But the HDR image is recreated whenever the window is resized, so its descriptor must be rewritten. Chapter 5 uploaded the resource heap once, to device-local memory the CPU can't touch.

### How
- **Host-visible and mapped:** the resource heap is now created in host-visible, host-coherent memory and mapped once for good. A descriptor is just bytes, so writing one is `writeResourceDescriptorsEXT` straight into the mapped heap; host-coherent memory needs no flush.
- **The fastest memory available:**
  - **Device-local and host-visible:** a discrete GPU makes a window of its own memory writable by the CPU: 256 MB classically, and all of it with Resizable BAR. This laptop's RTX 5070 offers such a memory type.
  - **The choice:** `heap_memory` asks for it, and falls back to plain host-visible memory, which works too, but is slower for the GPU to read.
- **The rule for writing:** a descriptor can be written whenever no submitted work still reads that slot. For the HDR image, that's after `recreate_swapchain`, which waits for the GPU to go idle.
- **Extra slots:**
  - **The layout:** `create_descriptor_heaps` takes a number of extra slots to leave after the textures.
  - **This chapter's slot:** one, for the HDR image.
  - **Chapter 8** adds more, for its environment maps.
- **`write_image_descriptor`** writes one slot from a `VkImageViewCreateInfo`. The textures now go through it too, one at a time, instead of in one batched call.
- **The sampler heap** never changes after loading, so it's still uploaded once to device-local memory.

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
};

// Creates both heaps.
//   - Resource heap: texture i at slot i, then `extra_slots` empty slots, which the program fills itself with write_image_descriptor.
//   - Sampler heap: index 0 is a default sampler; scene sampler i is at index i + 1.
DescriptorHeaps create_descriptor_heaps(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::span<const Texture> textures,
    std::span<const SceneSampler> samplers,
    std::uint32_t extra_slots
);

// Writes resource heap slot `slot`: a sampled image described by `view`, which shaders read as a Texture2D (or TextureCube, ...) handle with that index. The GPU must not be reading the slot while it's written.
void write_image_descriptor(
    const vk::raii::Device &device,
    DescriptorHeaps &heaps,
    std::uint32_t slot,
    const vk::ImageViewCreateInfo &view
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
    const vk::ImageViewCreateInfo &view
) {
    if (slot >= heaps.resource_slots) {
        throw std::runtime_error("resource heap slot " + std::to_string(slot) + " is past the last one");
    }

    // A descriptor is written from a description of the image view, so no VkImageView object is needed. Shaders sample it in this layout.
    const vk::ImageDescriptorInfoEXT image{
        .pView = &view,
        .layout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    const vk::ResourceDescriptorInfoEXT descriptor{
        .type = vk::DescriptorType::eSampledImage,
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

    // Sampler heap: the default sampler at index 0, then one per glTF sampler. The default is a SceneSampler with nothing set: trilinear, anisotropic and repeating.
    std::vector<vk::SamplerCreateInfo> sampler_infos{sampler_info(SceneSampler{}, limits.maxSamplerAnisotropy)};

    for (const SceneSampler &sampler : samplers) {
        sampler_infos.push_back(sampler_info(sampler, limits.maxSamplerAnisotropy));
    }

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

## 7.6 Physically based shading: `mesh.slang`

### Why
Chapter 6 loaded metallic and roughness, but lit everything with one fixed light and a fixed ambient term. glTF's specification defines exactly how its materials reflect light: Appendix B, "BRDF implementation". Following it makes our renders match other glTF viewers.

### How
- **A BRDF** (bidirectional reflectance distribution function) says how much of the light arriving from one direction leaves toward another. For one light, the reflected brightness in nits is `BRDF × illuminance × n·l`. The `n·l` factor accounts for light arriving at an angle spreading over more surface.
- **Microfacets:** the specular part treats a surface as countless tiny mirrors, each facing its own way. Only facets facing the *half vector* `h`, between the directions to the light and to the viewer, reflect the light toward the viewer.
  - **D, the GGX distribution:** how many facets face `h`. Smooth surfaces have a tall narrow peak, rough ones a wide flat one. GGX's α is roughness², which makes the roughness slider feel even.
  - **V, Smith's height-correlated visibility:** the share of those facets that are neither shadowed nor hidden by others, with the BRDF's `1 / (4 n·l n·v)` already folded in.
  - **Fresnel (Schlick's approximation):** every surface reflects more at grazing angles: `(1 − v·h)^5` grows from 0 head-on to 1 at grazing.
- **Metals and dielectrics, as glTF combines them:**
  - **Metals:** a metal's reflection is tinted by its base color, rising to white at grazing angles, and metals have no diffuse light.
  - **Dielectrics** (everything else) reflect 4% head-on, rising to 100% at grazing. The rest enters the surface and scatters back out as **Lambertian** diffuse light: `base color / π`, the same from every direction.
  - **Mixing:** `metallic` blends between the two, which is also how texture filtering between metal and non-metal texels behaves.
- **Punctual lights** follow `KHR_lights_punctual`:
  - **Distance:** point and spot lights fall off with distance squared: `intensity / d²` turns candela into lux.
  - **Range:** a `range` fades the light smoothly to nothing with the spec's `1 − (d/range)⁴` window.
  - **Spot cones:** spot lights square their cone falloff, for a softer edge.
- **A roughness floor:** a perfectly smooth surface reflects a punctual light from a single point, which no pixel could catch, so roughness is clamped to at least 0.045 for direct light.
- **Ambient light from a two-tone sky:**
  - **Diffuse:** for a Lambertian surface, a sky above and ground below give exactly `lerp(ground, sky, 0.5 + 0.5 n.y)`, weighted by how much of each the surface faces.
  - **Specular:** a smooth surface mirrors the sky along the reflected ray. A rough one blurs it, so the ray bends toward the normal as roughness grows.
  - **Splitting the two:** a roughness-aware Fresnel term divides the light between specular and diffuse, and occlusion darkens both.

  This is an approximation; Chapter 8 replaces it with the real sky and image-based lighting.
- **Pre-exposure:**
  - **Before writing:** the shader multiplies by the exposure *before* writing to the HDR image. Noon sunlight on white is about 30,000 nits, so after the 16-bit float limit of 65,504 there's little headroom for highlights; multiplying first keeps every value near 1.
  - **Emission:** glTF defines emissive light in nits, but its spec notes that many engines treat an emissive value of 1 as already exposed, and so do we: it shows as near-white whatever the exposure, like a screen.
- **The views** still output raw material values; the tone-mapping pass leaves them untouched.

### Code
`game-engine/shaders/mesh.slang`:
```slang
// Draws one glTF primitive: its vertices come from the scene's vertex buffer, its place in the world from its DrawData, and its surface from its glTF material, whose textures are read from the descriptor heap. Shaded with glTF's physically based BRDF, lit by the sun, the file's lights and an ambient sky, and written to the HDR image already exposed.

// Data shared with C++ (src/includes/shader_types.h)

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

// The same for every draw in a frame. Natural layout, like the C++ struct: the pointers land on 8-byte boundaries, right after the matrix.
struct FrameData {
    float4x4 view_projection;  // world space -> clip space
    Vertex *vertices;          // the scene's vertices
    DrawData *draws;           // one DrawData per draw
    Material *materials;       // the scene's materials
    Light *lights;             // the file's lights
    float3 camera_position;
    float exposure;            // scene nits -> tone mapper input
    float3 sun_direction;      // toward the sun
    uint light_count;
    float3 sun_illuminance;    // lux, facing the sun
    uint view;                 // what to output
    float3 sky_radiance;       // nits, ambient light from above
    float3 ground_radiance;    // nits, ambient light from below
};

// Written with vkCmdPushDataEXT before each draw.
struct PushData {
    FrameData *frame;  // this frame's data
    uint draw_index;   // which DrawData this draw uses
};

// In a descriptor heap pipeline, the push_constant block is where push data lands.
[[vk::push_constant]]
ConstantBuffer<PushData> push;

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

// Ambient light

// The sky's brightness in direction `d`: a bright sky above and a darker ground below, blended across the horizon. Chapter 8 replaces this with an image of the whole sky.
float3 ambient_radiance(FrameData *frame, float3 d) {
    return lerp(frame.ground_radiance, frame.sky_radiance, 0.5 + 0.5 * d.y);
}

// Light reflected from the whole sky at once.
//   - Diffuse: a Lambertian surface under this sky reflects the sky as seen along its normal, blended exactly like ambient_radiance.
//   - Specular: a smooth surface mirrors the sky along the reflected ray; a rough one blurs it, so the ray bends toward the normal as roughness grows. Fresnel, with a roughness-aware version of Schlick's formula, splits the light between the two parts.
float3 shade_ambient(Surface surface, FrameData *frame, float roughness) {
    const float n_dot_v = max(dot(surface.normal, surface.view), 1e-4);
    const float3 f0 = lerp(float3(0.04), surface.base_color, surface.metallic);
    const float3 fresnel = f0 + (max(float3(1.0 - roughness), f0) - f0) * pow(1.0 - n_dot_v, 5.0);

    const float3 reflected = reflect(-surface.view, surface.normal);
    const float3 blurred = normalize(lerp(reflected, surface.normal, roughness * roughness));

    const float3 diffuse_color = surface.base_color * (1.0 - surface.metallic);
    const float3 diffuse = (1.0 - fresnel) * diffuse_color * ambient_radiance(frame, surface.normal);
    const float3 specular = fresnel * ambient_radiance(frame, blurred);

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
    radiance += shade_ambient(surface, frame, roughness) * occlusion;

    // Exposure scales nits into the tone mapper's range here, before the 16-bit HDR image could overflow. glTF defines emission in nits, but, as its spec notes many engines do, we take it as already exposed: an emissive value of 1 shows as near-white, whatever the exposure.
    return float4(radiance * frame.exposure + emissive, base_color.a);
}
```

## 7.7 Tone mapping: `tonemap.slang`, `pipeline.h`, `pipeline.cpp`

### Why
After exposure, most of the scene lies between 0 and 1, but sunlit highlights and specular reflections can still be far brighter. Clipping them at 1 makes hard-edged, flat white patches, and bright colors shift hue as one channel clips before the others. A **tone mapper** compresses the whole range into 0..1 smoothly, the way film does.

### How
- **Khronos PBR Neutral** is the tone mapper Khronos designed for glTF and product rendering.
  - **Below about 0.8,** measured by a color's brightest channel, colors only lose a small offset, so a material's base color looks like itself.
  - **Above that,** values are compressed toward 1, and very bright ones desaturate toward white, as an overexposed photograph does.
  - **The alternative:** filmic tone mappers like ACES look more cinematic, but shift colors everywhere.
- **The second pass:** tone mapping reads the HDR image and writes the swapchain image: one full-screen triangle.
  - **The triangle:** the vertex shader makes its three corners from the vertex index alone, so there's no vertex buffer.
  - **Reading the image:** `SV_Position` in the fragment shader is the pixel's position, so `Load` reads exactly that pixel of the HDR image. There's no sampler and no filtering.
- **`create_tonemap_pipeline`** is the mesh pipeline stripped down: the same descriptor heap mode, but no depth, no culling and no blending. Its push data is `TonemapPushData`.
- **The swapchain's sRGB format** still does the final encoding for the display.

### Code
`game-engine/shaders/tonemap.slang`:
```slang
// Turns the HDR scene into colors a display can show: one full-screen triangle, one tone-mapped pixel per fragment.

// Data shared with C++ (src/includes/shader_types.h)

struct TonemapPushData {
    uint hdr_image;  // resource heap slot of the HDR scene image
    uint view;       // the scene shader's view; 0 is lit
};

[[vk::push_constant]]
ConstantBuffer<TonemapPushData> push;

static const uint view_lit = 0;

// Vertex shader

// Vertices 0, 1, 2 become the corners (-1, -1), (3, -1) and (-1, 3) in clip space: one triangle twice the screen's size, which the rasterizer clips to exactly the screen. A two-triangle quad would cost a little more: GPUs shade pixels in 2x2 blocks, and each block on the quad's diagonal would be shaded once for each triangle.
[shader("vertex")]
float4 vertexMain(uint vertex_id : SV_VulkanVertexID) : SV_Position {
    const float2 corner = float2((vertex_id << 1) & 2, vertex_id & 2);
    return float4(corner * 2.0 - 1.0, 0.0, 1.0);
}

// Tone mapping

// Khronos PBR Neutral, designed for glTF: a color whose brightest channel is below about 0.8 only loses a small offset (0.04, less near black), so base colors come through almost unchanged; brighter ones are compressed smoothly toward 1, and very bright ones also desaturate toward white, as they do on film.
float3 pbr_neutral(float3 color) {
    const float start_compression = 0.8 - 0.04;
    const float desaturation = 0.15;

    const float lowest = min(color.r, min(color.g, color.b));
    const float offset = lowest < 0.08 ? lowest - 6.25 * lowest * lowest : 0.04;
    color -= offset;

    const float peak = max(color.r, max(color.g, color.b));

    if (peak < start_compression) {
        return color;
    }

    const float d = 1.0 - start_compression;
    const float new_peak = 1.0 - d * d / (peak + d - start_compression);
    color *= new_peak / peak;

    const float g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return lerp(color, float3(new_peak), g);
}

// Fragment shader

// SV_Position in a fragment shader is the pixel's position on screen, so Load reads exactly this pixel of the HDR image: no sampler, no filtering. The scene shader has already applied the exposure; the swapchain's sRGB format encodes the result for the display.
[shader("fragment")]
float4 fragmentMain(float4 position : SV_Position) : SV_Target {
    const Texture2D hdr = Texture2D.Handle(uint2(push.hdr_image, 0));
    const float3 color = hdr.Load(int3(int2(position.xy), 0)).rgb;

    // Material views show data, not light: pass them through untouched.
    if (push.view != view_lit) {
        return float4(color, 1.0);
    }

    return float4(pbr_neutral(color), 1.0);
}
```

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

// Draws shaders/tonemap.slang as one full-screen triangle into a `color_format` image: no vertex data, no depth, nothing culled.
vk::raii::Pipeline create_tonemap_pipeline(const vk::raii::Device &device, vk::Format color_format);
```

At the end of `game-engine/src/pipeline.cpp`, add:
```cpp
// The tone-mapping pipeline

vk::raii::Pipeline create_tonemap_pipeline(const vk::raii::Device &device, vk::Format color_format) {
    const std::vector<std::uint32_t> spirv = read_spirv(std::filesystem::path(SHADER_DIR) / "tonemap.spv");

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

    // Every pixel is written exactly once, so there's nothing to depth-test or blend.
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
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic,
        .layout = nullptr,
    });
}
```

## 7.8 Putting it together: `main.cpp`

### Why
`main` creates the per-frame data buffers and the tone-mapping pipeline, records each frame in two passes, and turns the time of day into light and exposure every frame.

### How
- **Frame data:** each frame in flight gets its own `FrameData` buffer, host-visible and mapped.
  - **When it's written:** after waiting for that frame's fence, so the GPU has finished reading it.
  - **Why one per frame:** while the CPU writes one frame's data, the GPU can still be reading the other's.
- **Two passes per frame:**
  1. **The scene,** into the HDR image, cleared to the sky's color, already exposed. Wherever nothing is drawn, like the sky above Sponza's courtyard, the sky shows.
  2. **Tone mapping,** from the HDR image into the swapchain image.
- **The barriers:**
  - **Between the passes:** the HDR image moves from color attachment to shader-read layout, so the second pass waits until the scene is written.
  - **Between frames:** the frames in flight share one HDR image, like the depth buffer. Each frame's first barrier therefore also waits until the previous frame's tone mapping has finished reading it.
- **The HDR descriptor:**
  - **Its slot** sits right after the textures.
  - **Writing it:** `describe_hdr` writes it at startup, and the `resize` lambda rewrites it after every `recreate_swapchain`, which waits for the GPU first.
- **Lights:** a file without lights would need an empty buffer, which Vulkan doesn't allow. So there's no buffer then, its address is 0, and the light count is 0.
- **Settings:**
  - **Keys:** `[` and `]` move the time of day by a quarter of an hour, `-` and `=` make the picture half a stop darker or brighter, and 1 to 8 pick the view, as before. Holding a key repeats it.
  - **Exposure compensation** works like a camera's: +1 is a stop brighter. A brighter picture means expecting less light, so compensation is *subtracted* from the EV.
  - **The window title** shows the view, the time and the EV whenever one changes.
- **Exposure, each frame:**
  - **Metering:** the daylight's ground illuminance, clamped to EV −2…16, minus the compensation.
  - **Why −2:** without a floor, a black night would ask for unlimited exposure.

### Code
`game-engine/src/main.cpp`:
```cpp
#include "includes/buffer.h"
#include "includes/camera.h"
#include "includes/daylight.h"
#include "includes/descriptor_heap.h"
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
        std::array<float, 4> sky{};    // clear color: the sky, already exposed
        std::uint32_t hdr_slot = 0;    // resource heap slot of swapchain.hdr
        View view = View::lit;
        std::span<const Primitive> primitives;
        std::span<const MeshDraw> mesh_draws;
        std::span<const SceneMaterial> scene_materials;
        std::array<std::span<const std::uint32_t>, alpha_modes.size()> batches;
    };

    // Records a frame in two passes:
    //   1. the scene, into the HDR image: clear to the sky color and clear the depth, then draw each alpha mode's batch with that mode's pipeline,
    //   2. tone mapping, from the HDR image into the swapchain image, which is then ready to present.
    void record_frame(
        const vk::raii::CommandBuffer &commands,
        const Swapchain &swapchain,
        std::uint32_t image_index,
        std::span<const vk::raii::Pipeline> pipelines,
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

        // Wherever nothing is drawn, the sky shows.
        const vk::RenderingAttachmentInfo hdr_attachment{
            .imageView = *swapchain.hdr.view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = draws.sky}},
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
        float hours = 10.0f;                    // time of day, 0 to 24
        float exposure_compensation = 0.0f;     // stops brighter (+) or darker (-) than metered
    };

    // The views' names, in View's order, for the window title.
    constexpr std::array view_names{
        "Lit", "Base color", "Normal", "Vertex normal", "Metallic", "Roughness", "Occlusion", "Emissive",
    };

    // Handles every pending event and fills in `input` for this frame. False once the window was closed or Escape pressed.
    //   1-8   pick the view
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

        const vk::raii::Pipeline tonemap_pipeline = create_tonemap_pipeline(device, swapchain.format);

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
        // One slot after the textures, for the HDR image.
        DescriptorHeaps heaps = create_descriptor_heaps(device, *gpu, queue, command_pool, textures, scene.samplers, 1);
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

            // Light and exposure. The meter reads the daylight falling on flat ground. Compensation works like a camera's: +1 is a stop brighter, which means a lower EV (EV measures the light the camera expects).
            const Daylight daylight = daylight_at(settings.hours);
            const float ev100 = std::clamp(metered_ev100(daylight.horizontal_illuminance), -2.0f, 16.0f)
                - settings.exposure_compensation;
            const float exposure = exposure_from_ev100(ev100);

            // The title shows the view, the time and the exposure, whenever one changes.
            if (settings.view != shown_settings.view || settings.hours != shown_settings.hours
                || settings.exposure_compensation != shown_settings.exposure_compensation) {
                const auto minutes = static_cast<int>(std::lround(settings.hours * 60.0f));
                const std::string title = std::format("game-engine: {}, {:02}:{:02}, EV {:.1f}",
                    view_names[static_cast<std::size_t>(settings.view)], minutes / 60, minutes % 60, ev100);

                SDL_SetWindowTitle(window.get(), title.c_str());
                shown_settings = settings;
            }

            // Render

            Frame &frame = frames[frame_count % frames_in_flight];

            // 1. Wait until the GPU is done with this frame's command buffer and
            //    data from last time, then write this frame's data.
            (void)device.waitForFences(*frame.done, vk::True, no_timeout);

            *frame.mapped = FrameData{
                .view_projection = camera.projection(aspect) * camera.view(),
                .vertices = vertex_buffer.address,
                .draws = draw_buffer.address,
                .materials = material_buffer.address,
                .lights = light_buffer.address,
                .camera_position = camera.position,
                .exposure = exposure,
                .sun_direction = daylight.sun_direction,
                .light_count = static_cast<std::uint32_t>(scene.lights.size()),
                .sun_illuminance = daylight.sun_illuminance,
                .view = settings.view,
                .sky_radiance = daylight.sky_radiance,
                .ground_radiance = daylight.ground_radiance,
            };

            const glm::vec3 sky = daylight.sky_radiance * exposure;

            const DrawList draws{
                .index_buffer = *index_buffer.handle,
                .frame = frame.data.address,
                .sky = {sky.r, sky.g, sky.b, 1.0f},
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
            record_frame(frame.commands, swapchain, image_index, pipelines, tonemap_pipeline, heaps, draws);

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

## 7.9 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The terminal** shows a new line after loading Sponza: `Lights: 0 from the file, plus the sun`.
- **The window** opens at 10:00, with the title `game-engine: Lit, 10:00, EV 15.2`. Stone and floor facing the sun are brightly lit, and sides facing away (there are no shadows yet) are lit by the bluish sky, and the sky shows through the courtyard's open roof.
- **`]` and `[`** move through the day:

  | Time | Sun elevation | Light on the ground | EV |
  |---|---|---|---|
  | 06:00 | 13° | 13,600 lux | 12.4 |
  | 10:00 | 62° | 91,900 lux | 15.2 |
  | 12:00 | 78° | 105,200 lux | 15.4 |
  | 19:00 | 2° | 540 lux | 7.8 |
  | 19:30 | −3° | 73 lux (twilight) | 4.9 |
  | 20:00 | −8° | none | −2.0 |

  The exposure follows the meter, so noon and early evening look similarly bright. What changes is the light's direction, and the balance between the white sun and the blue sky light: as the sun sinks, more of the light is blue sky. The sun itself stays white in this simple model; Chapter 8's atmosphere reddens it. After 19:45 the night is black.
- **`-` and `=`** darken or brighten the picture by half a stop, on top of the meter.
- **Highlights:** metal and glossy surfaces show sharp sun highlights that the tone mapper rolls off instead of clipping. MetalRoughSpheres shows the full range of GGX highlights, from pinpoint to broad.
- **No `[validation …]` lines.**

**Other models:**
- **IridescenceSuzanne and DiffuseTransmissionTest** each have a directional light in the file. At night only those lights remain. DiffuseTransmissionTest's light shines onto the *backs* of its panels, because that model tests light passing through them, which needs the `KHR_materials_diffuse_transmission` extension. From the front the panels stay dark at night.
- **Lights in physical units are honest:** IridescenceSuzanne's light gives only 2 lux, so it disappears next to the sun by day, and at night, metered for darkness, it's overexposed. Press `-` a few times.

Next, in Chapter 8, the sky becomes real: a simulated atmosphere for every time of day, or a captured HDR sky, lights the scene through image-based lighting, and both show behind it.

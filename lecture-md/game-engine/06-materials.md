# Chapter 6: Materials

By the end of this chapter every glTF material in a scene reaches the shader with all of its core properties. Sponza's plants become cut-out leaves instead of dark rectangles, the walls and the lion's head get the fine relief of their normal maps, and the back faces of single-sided surfaces are skipped. Keys 1 to 8 switch the window between the shaded scene and each material input on its own, so we can check that every one of them loaded correctly. Chapter 7 then uses those inputs for real lighting.

A glTF material is more than a color. The core spec gives each material:
- a **base color**: a factor, a texture, and the vertex colors, multiplied together,
- **metallic and roughness** values, from factors and a shared texture,
- a **normal map** that tilts the surface's normal per texel,
- an **occlusion** map that darkens creases ambient light can't reach,
- an **emissive** color the surface gives off itself,
- an **alpha mode**: opaque, masked (cut out) or blended (see-through),
- whether it's **double-sided**,
- and, for every texture, a **sampler** (filtering and wrapping) and which set of texture coordinates to use.

This chapter builds on [Chapter 5](05-textures-and-descriptor-heap.md).

## 6.1 Vertices, materials and push data: `shader_types.h`

### Why
Every new material input needs data on the GPU. Normal maps need each vertex's **tangent**. Some textures use a second set of texture coordinates, and glTF multiplies the base color by **vertex colors** when a mesh has them. Each material needs its factors and five texture slots. And the shader needs to know which view to output.

### How
- **`Vertex` grows to 72 bytes:**
  - **`tangent`:** the direction on the surface in which the texture's u coordinate grows. Its `w` is +1 or −1, the sign that turns `cross(normal, tangent)` into the **bitangent**, the direction in which v *decreases*: up the image, since glTF's v runs downward. Together the three form the surface's **tangent frame**, which a normal map is written in (6.4).
  - **`uv1`:** glTF's `TEXCOORD_1`. Occlusion maps often use a second set of texture coordinates, laid out without overlaps.
  - **`color`:** `COLOR_0`, white when the file has none.
  - **The layout:** behind a pointer, Slang aligns each member only to the size of its scalar type. Every member here is made of 4-byte floats, so `tangent` sits at offset 24 even though it's a `float4`. glm does the same, so the `static_assert`s hold on both sides. A member made of 8-byte scalars, like a pointer, would need an 8-byte boundary; ordering members so each lands on its own boundary keeps every struct packed tight, with no padding anywhere.
- **`AlphaMode`:** glTF's three alpha modes, as an `enum class` with a 4-byte underlying type, so it can go into GPU data as-is. Each mode gets its own pipeline (6.6).
- **`TextureSlot`:** one texture reference: which texture in the resource heap, which sampler in the sampler heap, and which set of texture coordinates. Index 0 in each heap is a default: the white texture, and a default sampler.
- **`Material`:** every core glTF material property, 112 bytes. The scalars come first and fill the first 52 bytes, and the five texture slots follow, 12 bytes each. Alpha mode isn't here, because the pipeline already knows it.
- **`View` and push data:** `PushData` gains the view to output, at offset 92, the 4 bytes after `draw_index`. Push data is per draw, but every draw in a frame gets the same view.

### Code
`game-engine/src/includes/shader_types.h`:
```cpp
#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <cstddef>
#include <cstdint>

// C++ mirrors of the structs in shaders/mesh.slang. The GPU reads these
// bytes as they are, so the two sides must agree on every size and offset;
// the static_asserts catch a mismatch at compile time.

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

// --- Views -------------------------------------------------------------------

// What the fragment shader outputs: the shaded scene, or one material input
// on its own, for checking that each one loaded correctly. Keys 1-8 pick one.
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

// --- Push data ---------------------------------------------------------------

// Written with vkCmdPushDataEXT before each draw. The push block uses std430
// rules: the float4x4 comes first, then 8-byte pointers, then the indices.
struct PushData {
    glm::mat4 view_projection;   // world space -> clip space, the same for every draw
    vk::DeviceAddress vertices;  // where the first Vertex is in GPU memory
    vk::DeviceAddress draws;     // where the first DrawData is in GPU memory
    vk::DeviceAddress materials; // where the first Material is in GPU memory
    std::uint32_t draw_index;    // which DrawData this draw uses
    View view;                   // what the fragment shader outputs
};

static_assert(offsetof(PushData, vertices) == 64);
static_assert(offsetof(PushData, draws) == 72);
static_assert(offsetof(PushData, materials) == 80);
static_assert(offsetof(PushData, draw_index) == 88);
static_assert(offsetof(PushData, view) == 92);
```

## 6.2 Loading materials: `scene.h`, `scene.cpp`

### Why
tinygltf has already parsed every material, sampler and vertex attribute; Chapter 5 only kept the base color. Now the loader keeps the rest, in our own types, and fills in glTF's defaults for anything a file leaves out.

### How
- **`SceneSampler`:** a glTF sampler as the file stores it: OpenGL enum values, like `9729` for `GL_LINEAR` and `10497` for `GL_REPEAT`. A filter of −1 means the file leaves it to the renderer. 6.3 turns these into Vulkan samplers.
- **`TextureRef`:** a texture reference as the file names it: an image, a sampler, and a set of texture coordinates. We load two sets, so a reference to `TEXCOORD_2` or later reads set 0; files rarely use more than two.
- **`SceneMaterial`:** every core property, with glTF's defaults written into the struct. A material that sets nothing is white, fully metallic and fully rough, opaque, single-sided, with an alpha cutoff of 0.5.
- **Optional attributes:** `add_primitive` reads `TANGENT`, `TEXCOORD_0`, `TEXCOORD_1` and `COLOR_0` with one helper lambda.
  - **The helper:** it takes the attribute's name, the function that reads it, and the vector to fill. The vector starts filled with the value for "missing": a zero tangent, which the shader replaces (6.4), coordinates of (0, 0), and white.
  - **Colors** may be RGB or RGBA, as floats or as normalized 8- or 16-bit integers. `read_colors` reads either width, and Chapter 4's `read_floats` already handles the integer forms.
- **Mirrored draws:** a node transform with a negative scale mirrors its mesh. A mirror reverses the order in which each triangle's corners appear on screen, and that order is how the GPU decides which side is the front (6.6). The determinant of the transform's 3×3 part is negative exactly when it mirrors, so `visit_node` stores that in `MeshDraw::mirrored`.
- **Draw centers:** each draw also stores the middle of its bounding box in world space. Blended draws are sorted by it (6.7).
- **`texture_ref`** follows material → texture → image and sampler, and checks the indices are in range, as Chapter 5's `texture_image` does for images.

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

    // The middle of the primitive's box in world space, for sorting
    // see-through draws by distance.
    glm::vec3 center{0.0f};
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

    // World-space box around everything drawn, for placing the camera.
    glm::vec3 bounds_min{std::numeric_limits<float>::max()};
    glm::vec3 bounds_max{std::numeric_limits<float>::lowest()};
};

// Loads the default scene of a .gltf or .glb file, with its materials,
// samplers and images, still encoded.
Scene load_gltf(const std::filesystem::path &path);
```

`game-engine/src/scene.cpp`:
```cpp
#include "includes/scene.h"

#include <tiny_gltf.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cstring>
#include <optional>
#include <print>
#include <stdexcept>
#include <string>

namespace {

// --- Reading accessors -------------------------------------------------------

// glTF stores vertex data in buffers, viewed through two layers:
//   buffer      raw bytes, from a .bin file, a data URI, or a .glb's binary chunk
//   bufferView  a slice of a buffer, with an optional stride between elements
//   accessor    how to read that slice: element type, count, offset
// tinygltf loads every buffer into memory; reading elements is up to us.

// Copies `count` elements of `element_size` bytes, `stride` apart, starting at
// `offset` in bufferView `view_index`, after checking they fit in the buffer.
std::vector<unsigned char> copy_elements(
    const tinygltf::Model &model, int view_index, std::size_t offset,
    std::size_t count, std::size_t element_size, std::size_t stride
) {
    const tinygltf::BufferView &view = model.bufferViews.at(view_index);
    const tinygltf::Buffer &buffer = model.buffers.at(view.buffer);

    const std::size_t start = view.byteOffset + offset;
    const std::size_t end = count == 0 ? start : start + (count - 1) * stride + element_size;

    if (end > view.byteOffset + view.byteLength || end > buffer.data.size()) {
        throw std::runtime_error("bufferView " + std::to_string(view_index) + " is read past its end");
    }

    std::vector<unsigned char> elements(count * element_size);

    for (std::size_t i = 0; i < count; ++i) {
        std::memcpy(elements.data() + i * element_size, buffer.data.data() + start + i * stride, element_size);
    }

    return elements;
}

// An accessor's elements packed one after another, whatever the buffer layout:
//   - byteStride 0 means elements are packed already; otherwise they're spread out,
//   - without a bufferView every element starts as zeros,
//   - a sparse accessor then replaces the elements its index list names.
std::vector<unsigned char> accessor_elements(const tinygltf::Model &model, int accessor_index) {
    const tinygltf::Accessor &accessor = model.accessors.at(accessor_index);

    const int component_size = tinygltf::GetComponentSizeInBytes(static_cast<std::uint32_t>(accessor.componentType));
    const int components = tinygltf::GetNumComponentsInType(static_cast<std::uint32_t>(accessor.type));

    if (component_size <= 0 || components <= 0) {
        throw std::runtime_error("accessor " + std::to_string(accessor_index) + " has an invalid type");
    }

    const std::size_t element_size = static_cast<std::size_t>(component_size) * static_cast<std::size_t>(components);
    std::vector<unsigned char> elements(accessor.count * element_size, 0);

    if (accessor.bufferView >= 0) {
        const int byte_stride = accessor.ByteStride(model.bufferViews.at(accessor.bufferView));

        if (byte_stride <= 0) {
            throw std::runtime_error("accessor " + std::to_string(accessor_index) + " has an invalid byte stride");
        }

        elements = copy_elements(model, accessor.bufferView, accessor.byteOffset, accessor.count,
            element_size, static_cast<std::size_t>(byte_stride));
    }

    if (accessor.sparse.isSparse) {
        const auto &sparse = accessor.sparse;
        const int index_size = tinygltf::GetComponentSizeInBytes(static_cast<std::uint32_t>(sparse.indices.componentType));
        const auto count = static_cast<std::size_t>(sparse.count);

        const std::vector<unsigned char> indices = copy_elements(model, sparse.indices.bufferView,
            sparse.indices.byteOffset, count, static_cast<std::size_t>(index_size), static_cast<std::size_t>(index_size));
        const std::vector<unsigned char> values = copy_elements(model, sparse.values.bufferView,
            sparse.values.byteOffset, count, element_size, element_size);

        for (std::size_t i = 0; i < count; ++i) {
            std::uint32_t target = 0;

            switch (index_size) {
                case 1: target = indices[i]; break;
                case 2: { std::uint16_t v; std::memcpy(&v, &indices[i * 2], 2); target = v; break; }
                case 4: std::memcpy(&target, &indices[i * 4], 4); break;
                default: throw std::runtime_error("accessor " + std::to_string(accessor_index) + " has invalid sparse indices");
            }

            if (target >= accessor.count) {
                throw std::runtime_error("accessor " + std::to_string(accessor_index) + " has a sparse index out of range");
            }

            std::memcpy(&elements[target * element_size], &values[i * element_size], element_size);
        }
    }

    return elements;
}

// Reads an accessor as floats, `components` per element. Besides 32-bit
// floats, glTF allows integers here (KHR_mesh_quantization), and "normalized"
// integers mean a fraction of their range: 255 as an unsigned byte is 1.0.
std::vector<float> read_floats(const tinygltf::Model &model, int accessor_index, int components) {
    const tinygltf::Accessor &accessor = model.accessors.at(accessor_index);

    if (tinygltf::GetNumComponentsInType(static_cast<std::uint32_t>(accessor.type)) != components) {
        throw std::runtime_error("accessor " + std::to_string(accessor_index) + " has the wrong number of components");
    }

    const std::vector<unsigned char> bytes = accessor_elements(model, accessor_index);
    std::vector<float> values(accessor.count * static_cast<std::size_t>(components));

    // Component i of type T, converted to float.
    const auto convert = [&]<typename T>(float normalize_by) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            T value;
            std::memcpy(&value, bytes.data() + i * sizeof(T), sizeof(T));

            // The spec's rule for signed values clamps the most negative one to -1.
            values[i] = accessor.normalized ? std::max(static_cast<float>(value) / normalize_by, -1.0f) : static_cast<float>(value);
        }
    };

    switch (accessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_FLOAT: std::memcpy(values.data(), bytes.data(), bytes.size()); break;
        case TINYGLTF_COMPONENT_TYPE_BYTE: convert.operator()<std::int8_t>(127.0f); break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: convert.operator()<std::uint8_t>(255.0f); break;
        case TINYGLTF_COMPONENT_TYPE_SHORT: convert.operator()<std::int16_t>(32767.0f); break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: convert.operator()<std::uint16_t>(65535.0f); break;
        default: throw std::runtime_error("accessor " + std::to_string(accessor_index) + " can't be read as floats");
    }

    return values;
}

std::vector<glm::vec3> read_vec3(const tinygltf::Model &model, int accessor_index) {
    const std::vector<float> floats = read_floats(model, accessor_index, 3);
    std::vector<glm::vec3> values(floats.size() / 3);
    std::memcpy(values.data(), floats.data(), floats.size() * sizeof(float));
    return values;
}

std::vector<glm::vec2> read_vec2(const tinygltf::Model &model, int accessor_index) {
    const std::vector<float> floats = read_floats(model, accessor_index, 2);
    std::vector<glm::vec2> values(floats.size() / 2);
    std::memcpy(values.data(), floats.data(), floats.size() * sizeof(float));
    return values;
}

std::vector<glm::vec4> read_vec4(const tinygltf::Model &model, int accessor_index) {
    const std::vector<float> floats = read_floats(model, accessor_index, 4);
    std::vector<glm::vec4> values(floats.size() / 4);
    std::memcpy(values.data(), floats.data(), floats.size() * sizeof(float));
    return values;
}

// Vertex colors may be RGB or RGBA; RGB ones get an alpha of 1.
std::vector<glm::vec4> read_colors(const tinygltf::Model &model, int accessor_index) {
    if (model.accessors.at(accessor_index).type == TINYGLTF_TYPE_VEC4) {
        return read_vec4(model, accessor_index);
    }

    std::vector<glm::vec4> colors;

    for (const glm::vec3 rgb : read_vec3(model, accessor_index)) {
        colors.emplace_back(rgb, 1.0f);
    }

    return colors;
}

// Indices may be 8, 16 or 32 bits, even within one file; they're widened to
// 32 bits so the whole scene can share one index buffer.
std::vector<std::uint32_t> read_indices(const tinygltf::Model &model, int accessor_index) {
    const tinygltf::Accessor &accessor = model.accessors.at(accessor_index);
    const std::vector<unsigned char> bytes = accessor_elements(model, accessor_index);
    std::vector<std::uint32_t> indices(accessor.count);

    const auto widen = [&]<typename T>() {
        for (std::size_t i = 0; i < indices.size(); ++i) {
            T value;
            std::memcpy(&value, bytes.data() + i * sizeof(T), sizeof(T));
            indices[i] = value;
        }
    };

    switch (accessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: widen.operator()<std::uint8_t>(); break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: widen.operator()<std::uint16_t>(); break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: widen.operator()<std::uint32_t>(); break;
        default: throw std::runtime_error("accessor " + std::to_string(accessor_index) + " has an invalid index type");
    }

    return indices;
}

// --- Meshes ------------------------------------------------------------------

// Where a primitive landed in the scene, plus the box around its vertices in
// its own space, which visit_node() turns into world-space scene bounds.
struct LoadedPrimitive {
    std::uint32_t index;
    glm::vec3 local_min;
    glm::vec3 local_max;
};

// Turns triangle strips and fans into a plain list of triangles, keeping
// every triangle's corners in the same winding order.
std::vector<std::uint32_t> to_triangle_list(int mode, const std::vector<std::uint32_t> &indices) {
    std::vector<std::uint32_t> list;

    for (std::size_t i = 2; i < indices.size(); ++i) {
        if (mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
            // Every other triangle in a strip faces the other way; swap two corners back.
            const bool even = (i % 2) == 0;
            list.insert(list.end(), {indices[even ? i - 2 : i - 1], indices[even ? i - 1 : i - 2], indices[i]});
        } else {
            // A fan: every triangle shares the first vertex.
            list.insert(list.end(), {indices[i - 1], indices[i], indices[0]});
        }
    }

    return list;
}

// Appends one primitive to the scene. Points, lines, and primitives without
// positions are skipped, as glTF allows: this renderer draws triangles.
// `default_material` is used when the primitive doesn't name one.
std::optional<LoadedPrimitive> add_primitive(
    const tinygltf::Model &model, const tinygltf::Primitive &source, std::uint32_t default_material, Scene &scene
) {
    const auto position = source.attributes.find("POSITION");
    const bool triangles = source.mode == TINYGLTF_MODE_TRIANGLES
        || source.mode == TINYGLTF_MODE_TRIANGLE_STRIP
        || source.mode == TINYGLTF_MODE_TRIANGLE_FAN;

    if (!triangles || position == source.attributes.end()) {
        return std::nullopt;
    }

    // The default material sits right after the file's own, so anything at or
    // past it isn't one of the file's materials.
    if (source.material >= static_cast<int>(default_material)) {
        throw std::runtime_error("a primitive names a material that doesn't exist");
    }

    const std::vector<glm::vec3> positions = read_vec3(model, position->second);

    // Normals are optional. Missing ones stay (0, 0, 0), which tells the shader
    // to shade the triangle flat, as glTF asks.
    std::vector<glm::vec3> normals(positions.size(), glm::vec3{0.0f});

    if (const auto normal = source.attributes.find("NORMAL"); normal != source.attributes.end()) {
        normals = read_vec3(model, normal->second);

        if (normals.size() != positions.size()) {
            throw std::runtime_error("a primitive has a different number of normals and positions");
        }
    }

    // The other attributes are optional too. Each one missing from the file
    // keeps the value given here for every vertex:
    //   - tangents (0, 0, 0, 0): the shader works them out itself,
    //   - texture coordinates (0, 0): every vertex samples the same texel,
    //   - colors white: multiplying by white changes nothing.
    std::vector<glm::vec4> tangents(positions.size(), glm::vec4{0.0f});
    std::vector<glm::vec2> uv0s(positions.size(), glm::vec2{0.0f});
    std::vector<glm::vec2> uv1s(positions.size(), glm::vec2{0.0f});
    std::vector<glm::vec4> colors(positions.size(), glm::vec4{1.0f});

    // Reads attribute `name` with `read`, if the primitive has it.
    const auto read_attribute = [&](const char *name, auto read, auto &values) {
        if (const auto attribute = source.attributes.find(name); attribute != source.attributes.end()) {
            values = read(model, attribute->second);

            if (values.size() != positions.size()) {
                throw std::runtime_error(std::string("a primitive has a different number of ") + name + " and POSITION values");
            }
        }
    };

    read_attribute("TANGENT", read_vec4, tangents);
    read_attribute("TEXCOORD_0", read_vec2, uv0s);
    read_attribute("TEXCOORD_1", read_vec2, uv1s);
    read_attribute("COLOR_0", read_colors, colors);

    // Indices are optional too; without them, vertices form triangles in order.
    std::vector<std::uint32_t> indices;

    if (source.indices >= 0) {
        indices = read_indices(model, source.indices);
    } else {
        for (std::uint32_t i = 0; i < positions.size(); ++i) {
            indices.push_back(i);
        }
    }

    if (source.mode != TINYGLTF_MODE_TRIANGLES) {
        indices = to_triangle_list(source.mode, indices);
    }

    // An index past the last vertex would make the GPU read another primitive's vertices.
    for (const std::uint32_t index : indices) {
        if (index >= positions.size()) {
            throw std::runtime_error("a primitive's index points past its vertices");
        }
    }

    LoadedPrimitive loaded{
        .index = static_cast<std::uint32_t>(scene.primitives.size()),
        .local_min = glm::vec3{std::numeric_limits<float>::max()},
        .local_max = glm::vec3{std::numeric_limits<float>::lowest()},
    };

    scene.primitives.push_back(Primitive{
        .first_index = static_cast<std::uint32_t>(scene.indices.size()),
        .index_count = static_cast<std::uint32_t>(indices.size()),
        .vertex_offset = static_cast<std::int32_t>(scene.vertices.size()),
        .material = source.material >= 0 ? static_cast<std::uint32_t>(source.material) : default_material,
    });

    for (std::size_t i = 0; i < positions.size(); ++i) {
        scene.vertices.push_back(Vertex{
            .position = positions[i],
            .normal = normals[i],
            .tangent = tangents[i],
            .uv0 = uv0s[i],
            .uv1 = uv1s[i],
            .color = colors[i],
        });
        loaded.local_min = glm::min(loaded.local_min, positions[i]);
        loaded.local_max = glm::max(loaded.local_max, positions[i]);
    }

    scene.indices.insert(scene.indices.end(), indices.begin(), indices.end());
    return loaded;
}

// --- Nodes -------------------------------------------------------------------

// A node's transform relative to its parent: either a full matrix, or
// translation * rotation * scale. glTF stores doubles; we draw with floats.
glm::mat4 local_transform(const tinygltf::Node &node) {
    if (node.matrix.size() == 16) {
        return glm::mat4(glm::make_mat4(node.matrix.data()));  // column-major, like glm
    }

    glm::mat4 transform{1.0f};

    if (node.translation.size() == 3) {
        transform = glm::translate(transform, glm::vec3(glm::make_vec3(node.translation.data())));
    }

    if (node.rotation.size() == 4) {
        // glTF stores (x, y, z, w); glm's constructor takes w first.
        const auto &r = node.rotation;
        const glm::quat rotation(static_cast<float>(r[3]), static_cast<float>(r[0]),
            static_cast<float>(r[1]), static_cast<float>(r[2]));
        transform *= glm::mat4_cast(rotation);
    }

    if (node.scale.size() == 3) {
        transform = glm::scale(transform, glm::vec3(glm::make_vec3(node.scale.data())));
    }

    return transform;
}

// Walks the node tree. Each node's world transform is its parent's times its
// own, and every primitive of a node's mesh becomes one draw.
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

    for (const int child : node.children) {
        visit_node(model, child, world, mesh_primitives, scene);
    }
}

// --- Materials and images ----------------------------------------------------

// The image a texture reference points at: material -> texture -> image.
// -1 if there's no texture, or the texture's image is in a form we don't
// read (KHR_texture_basisu puts it in an extension instead of `source`).
std::int32_t texture_image(const tinygltf::Model &model, int texture_index) {
    if (texture_index < 0) {
        return -1;
    }

    const int image = model.textures.at(texture_index).source;

    if (image >= static_cast<int>(model.images.size())) {
        throw std::runtime_error("texture " + std::to_string(texture_index) + " points past the last image");
    }

    return image;
}

// A material's texture reference: which image, which sampler, which set of
// texture coordinates. We load two sets, TEXCOORD_0 and TEXCOORD_1; glTF
// allows more, but files rarely use them, so any later set reads set 0.
TextureRef texture_ref(const tinygltf::Model &model, int texture_index, int tex_coord) {
    TextureRef ref{
        .image = texture_image(model, texture_index),
        .uv_set = tex_coord == 1 ? 1u : 0u,
    };

    if (texture_index >= 0) {
        ref.sampler = model.textures.at(texture_index).sampler;

        if (ref.sampler >= static_cast<int>(model.samplers.size())) {
            throw std::runtime_error("texture " + std::to_string(texture_index) + " points past the last sampler");
        }
    }

    return ref;
}

AlphaMode alpha_mode(const std::string &mode) {
    if (mode == "MASK") {
        return AlphaMode::mask;
    }

    if (mode == "BLEND") {
        return AlphaMode::blend;
    }

    return AlphaMode::opaque;
}

void add_materials_and_images(tinygltf::Model &model, Scene &scene) {
    for (const tinygltf::Material &source : model.materials) {
        const auto &pbr = source.pbrMetallicRoughness;

        scene.materials.push_back(SceneMaterial{
            .base_color_factor = glm::vec4(glm::make_vec4(pbr.baseColorFactor.data())),
            .base_color = texture_ref(model, pbr.baseColorTexture.index, pbr.baseColorTexture.texCoord),
            .metallic_factor = static_cast<float>(pbr.metallicFactor),
            .roughness_factor = static_cast<float>(pbr.roughnessFactor),
            .metallic_roughness = texture_ref(model, pbr.metallicRoughnessTexture.index, pbr.metallicRoughnessTexture.texCoord),
            .normal = texture_ref(model, source.normalTexture.index, source.normalTexture.texCoord),
            .normal_scale = static_cast<float>(source.normalTexture.scale),
            .occlusion = texture_ref(model, source.occlusionTexture.index, source.occlusionTexture.texCoord),
            .occlusion_strength = static_cast<float>(source.occlusionTexture.strength),
            .emissive_factor = glm::vec3(glm::make_vec3(source.emissiveFactor.data())),
            .emissive = texture_ref(model, source.emissiveTexture.index, source.emissiveTexture.texCoord),
            .alpha_mode = alpha_mode(source.alphaMode),
            .alpha_cutoff = static_cast<float>(source.alphaCutoff),
            .double_sided = source.doubleSided,
        });
    }

    // For primitives that don't name a material, glTF's default: plain white.
    scene.materials.push_back(SceneMaterial{});

    for (const tinygltf::Sampler &sampler : model.samplers) {
        scene.samplers.push_back(SceneSampler{
            .mag_filter = sampler.magFilter,
            .min_filter = sampler.minFilter,
            .wrap_s = sampler.wrapS,
            .wrap_t = sampler.wrapT,
        });
    }

    // The image bytes move out of tinygltf's model; it's discarded afterwards.
    for (tinygltf::Image &image : model.images) {
        scene.images.push_back(SceneImage{
            .encoded = std::move(image.image),
            .name = image.uri.empty() ? image.name : image.uri,
        });
    }

    // Colors are stored sRGB-encoded; everything else (normals, roughness, ...)
    // is plain data. Base color and emissive textures hold colors.
    for (const tinygltf::Material &source : model.materials) {
        for (const int texture : {source.pbrMetallicRoughness.baseColorTexture.index, source.emissiveTexture.index}) {
            if (const std::int32_t image = texture_image(model, texture); image >= 0) {
                scene.images.at(static_cast<std::size_t>(image)).srgb = true;
            }
        }
    }
}

// --- Images ------------------------------------------------------------------

// tinygltf finds every image's bytes, whether in its own file, in a data URI,
// or inside a .glb, and hands them to this callback. We keep the encoded bytes
// (PNG, JPEG, ...) as they are; Chapter 5 decodes them.
bool keep_encoded_image(
    tinygltf::Image *image, int /*image_index*/, std::string* /*error*/, std::string* /*warning*/,
    int /*required_width*/, int /*required_height*/, const unsigned char *bytes, int size, void* /*user_data*/
) {
    image->image.assign(bytes, bytes + size);
    image->as_is = true;
    return true;
}

}  // namespace

// --- Loading -----------------------------------------------------------------

Scene load_gltf(const std::filesystem::path &path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string error;
    std::string warning;

    loader.SetImageLoader(keep_encoded_image, nullptr);

    // .glb packs the JSON and binary data in one file; .gltf is JSON that
    // refers to .bin and image files, or embeds them as base64 data URIs.
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

    // A file lists the extensions it can't be read without. Compressed
    // geometry needs a decoder library we don't include, so refuse it clearly
    // instead of reading compressed bytes as vertices. The others change how
    // things look, not where the geometry is, and later chapters handle them.
    for (const std::string &extension : model.extensionsRequired) {
        if (extension == "KHR_draco_mesh_compression" || extension == "KHR_meshopt_compression"
            || extension == "EXT_meshopt_compression") {
            throw std::runtime_error(path.string() + " needs " + extension + ", which this loader doesn't decode");
        }
    }

    Scene scene;
    add_materials_and_images(model, scene);
    const auto default_material = static_cast<std::uint32_t>(scene.materials.size() - 1);

    // Every primitive of every mesh, once. mesh_primitives[m] lists where
    // mesh m's drawable primitives landed in scene.primitives.
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
        visit_node(model, node, glm::mat4{1.0f}, mesh_primitives, scene);
    }

    if (scene.draws.empty()) {
        throw std::runtime_error(path.string() + " has nothing to draw");
    }

    return scene;
}
```

## 6.3 Samplers: `descriptor_heap.h`, `descriptor_heap.cpp`

### Why
A sampler decides how a texture is read between and beyond its texels: how texels are blended, how mip levels are chosen, and what happens to coordinates outside 0..1. glTF lets every texture name its own sampler. MetalRoughSpheres clamps its textures to the edge, for example, and TextureTransformTest asks for no mipmaps at all. Chapter 5 used one sampler for everything.

### How
- **The sampler heap** now holds the default sampler at index 0, then one descriptor per glTF sampler: scene sampler `i` is at heap index `i + 1`, the same shift as textures. Every sampler descriptor is written in a single `writeSamplerDescriptorsEXT` call, as Chapter 5 wrote the image descriptors.
- **glTF's minifying filters** are OpenGL's, and each names two things at once: how to filter within a mip level, and how to choose between levels. `LINEAR_MIPMAP_NEAREST`, for instance, blends texels within the nearest mip level. `sampler_info` splits them into Vulkan's `minFilter` and `mipmapMode`.
- **No mipmaps:** plain `NEAREST` and `LINEAR` only ever read level 0, and Vulkan has no direct equivalent. The Vulkan spec's own recipe is nearest mipmapping with `maxLod = 0.25`. Level 0 is then always the one chosen, but the minifying filter still applies, which `maxLod = 0` would prevent.
- **Wrapping:** glTF's `REPEAT`, `CLAMP_TO_EDGE` and `MIRRORED_REPEAT` map directly to Vulkan address modes.
- **Unset filters** get our best quality: trilinear, plus anisotropic filtering. Anisotropy is only switched on for trilinear filtering; a file that asks for nearest or unmipmapped filtering wants exactly that look.
- **The default sampler** is simply `sampler_info(SceneSampler{})`: a sampler with nothing set.

### Code
`game-engine/src/includes/descriptor_heap.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/texture.h"

#include <span>

// The two descriptor heaps shaders read: a resource heap of image
// descriptors and a sampler heap. Each is a plain buffer of descriptor bytes
// that we write ourselves, ending in a range reserved for the driver.
struct DescriptorHeaps {
    Buffer resources;
    vk::DeviceSize resource_reserved_offset = 0;
    vk::DeviceSize resource_reserved_size = 0;

    Buffer samplers;
    vk::DeviceSize sampler_reserved_offset = 0;
    vk::DeviceSize sampler_reserved_size = 0;
};

// Writes one image descriptor per texture (texture i at index i of the
// resource heap) and one sampler descriptor per glTF sampler, and uploads
// both heaps to device-local memory. Sampler heap index 0 is a default
// sampler; scene sampler i is at index i + 1.
DescriptorHeaps create_descriptor_heaps(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::span<const Texture> textures,
    std::span<const SceneSampler> samplers
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

// Uploads a heap's bytes and checks the GPU address lands where heaps must
// start: a multiple of `alignment`.
Buffer upload_heap(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::span<const std::byte> bytes,
    vk::DeviceSize alignment
) {
    Buffer heap = upload_buffer(device, gpu, queue, pool, bytes,
        vk::BufferUsageFlagBits::eDescriptorHeapEXT | vk::BufferUsageFlagBits::eShaderDeviceAddress);

    if (heap.address % alignment != 0) {
        throw std::runtime_error("a descriptor heap's address isn't aligned to " + std::to_string(alignment) + " bytes");
    }

    return heap;
}

// --- Samplers ----------------------------------------------------------------

// glTF's sampler settings as Vulkan sampler state. glTF uses OpenGL's enums,
// whose minifying filters name two things at once: how to filter within a
// mip level, and how to pick between levels. Filters the file leaves out
// get our best quality: trilinear and anisotropic.
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

    // Plain NEAREST and LINEAR don't use mipmaps at all, only level 0.
    // Vulkan has no such filter; the spec's recipe is nearest mipmapping
    // with maxLod 0.25, so level 0 is always picked but the minifying
    // filter is still used.
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

// --- Creating the heaps ------------------------------------------------------

DescriptorHeaps create_descriptor_heaps(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::span<const Texture> textures,
    std::span<const SceneSampler> samplers
) {
    const auto properties = gpu.device.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceDescriptorHeapPropertiesEXT
    >();
    const auto &heap = properties.get<vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();
    const auto &limits = properties.get<vk::PhysicalDeviceProperties2>().properties.limits;

    DescriptorHeaps heaps;

    // Resource heap. Descriptor i sits at i * imageDescriptorSize, which is
    // where a shader's Texture2D.Handle(i) reads it. The driver's reserved
    // range comes after the last descriptor.
    heaps.resource_reserved_offset = align_up(heap.imageDescriptorSize * textures.size(), heap.imageDescriptorAlignment);
    heaps.resource_reserved_size = heap.minResourceHeapReservedRange;

    std::vector<std::byte> resource_bytes(
        align_up(heaps.resource_reserved_offset + heaps.resource_reserved_size, heap.resourceHeapAlignment));

    // A descriptor is written from a description of the image view, so no
    // VkImageView object is needed. The descriptor info structs point at each
    // other, so all of them must stay alive until the write; sizing the
    // vectors up front means no pointer moves.
    std::vector<vk::ImageViewCreateInfo> views(textures.size());
    std::vector<vk::ImageDescriptorInfoEXT> images(textures.size());
    std::vector<vk::ResourceDescriptorInfoEXT> descriptors(textures.size());
    std::vector<vk::HostAddressRangeEXT> destinations(textures.size());

    for (std::size_t i = 0; i < textures.size(); ++i) {
        views[i] = vk::ImageViewCreateInfo{
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
        };

        images[i] = vk::ImageDescriptorInfoEXT{
            .pView = &views[i],
            .layout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        descriptors[i] = vk::ResourceDescriptorInfoEXT{
            .type = vk::DescriptorType::eSampledImage,
            .data = {.pImage = &images[i]},
        };

        destinations[i] = vk::HostAddressRangeEXT{
            .address = resource_bytes.data() + i * heap.imageDescriptorSize,
            .size = heap.imageDescriptorSize,
        };
    }

    // One call writes every descriptor's bytes into resource_bytes.
    device.writeResourceDescriptorsEXT(descriptors, destinations);
    heaps.resources = upload_heap(device, gpu, queue, pool, resource_bytes, heap.resourceHeapAlignment);

    // Sampler heap: the default sampler at index 0, then one per glTF sampler.
    // The default is a SceneSampler with nothing set: trilinear, anisotropic
    // and repeating.
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
    heaps.samplers = upload_heap(device, gpu, queue, pool, sampler_bytes, heap.samplerHeapAlignment);

    return heaps;
}

// --- Binding -----------------------------------------------------------------

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

## 6.4 The material in the shader: `mesh.slang`

### Why
The fragment shader now evaluates the whole material: the base color with its alpha handling, the normal with its normal map, metallic, roughness, occlusion and emissive. The lighting is still Chapter 5's single light, now with the mapped normal, occlusion and emissive added. Physically based lighting that uses metallic and roughness comes in Chapter 7; until then, the views show those two.

### How
- **`sample_slot`** reads one material slot: its texture and sampler from the heaps, at its set of texture coordinates. Every texture read in the shader goes through it.
- **Base color** is factor × texture × vertex color.
- **Alpha modes:**
  - **Opaque** ignores alpha: it's set to 1.
  - **Mask** `discard`s pixels whose alpha is below `alpha_cutoff`. A discarded pixel writes neither color nor depth, so whatever is behind it shows. That turns the plants' rectangles into leaves.
  - **Blend** keeps alpha for the pipeline's blending (6.6).
- **The specialization constant:** `[vk::constant_id(0)] const uint alpha_mode` is a constant whose value is supplied when a pipeline is created. Each of the three pipelines bakes in its own value, and the compiler deletes the branches for the other modes. That matters for opaque surfaces: a shader that might `discard` can keep the GPU from testing depth *before* running the fragment shader (early depth testing), and early testing is what spares hidden pixels from being shaded at all.
- **The tangent frame** (`vertexMain`):
  - **Tangent and bitangent** lie along the surface, so they're transformed by the model matrix, like positions. Only the normal needs the normal matrix.
  - **The bitangent** is built from glTF's rule, `B = cross(N, T) * w`, *before* the transform. Transforming it afterwards also mirrors it correctly under a mirroring node.
- **The normal** (`surface_normal`), step by step:
  1. **The vertex normal:** interpolated and normalized, or the flat triangle normal, `cross(ddy, ddx)`, when the file has none, as in Chapter 4.
  2. **The tangent frame:** from the file, or derived per pixel when the file has none (below).
  3. **Double-sided back faces:** `SV_IsFrontFace` tells the shader which side of the triangle it's drawing. On a double-sided material's back face the surface faces the other way, so the whole frame is flipped: normal, tangent and bitangent. The flip comes after deriving the frame, so derived frames flip too.
  4. **The normal map:** stores a direction in the tangent frame, each component packed into 0..1. Unpacked to −1..1, `x` goes along the tangent, `y` along the bitangent and `z` along the normal. `normal_scale` scales the X and Y tilt, as glTF specifies.
  5. **Normalizing the frame:** all three axes must be unit length, or the map's tilt shrinks or grows with them. The normal already is. The tangent and bitangent come out of the model matrix scaled with it, and Sponza's root node scales the whole model by 0.008: without normalizing, its normal maps would have almost no effect.
- **Tangents the file doesn't provide:** glTF says to generate them with the MikkTSpace algorithm, which needs a precomputation pass over the mesh. Instead, the shader solves for them per pixel.
  - **Two equations:** `ddx` and `ddy` give how the world position *p* and the texture coordinates *(u, v)* change between neighbouring pixels: `dp/dx = P_u·du/dx + P_v·dv/dx` and `dp/dy = P_u·du/dy + P_v·dv/dy`.
  - **The unknowns:** *P_u* and *P_v* are how the position changes per unit of u and of v: the tangent, and the negated bitangent. The bitangent is negated because glTF's v runs down the image, while a normal map's +Y points up.
  - **The solution:** solving the two equations gives both. Only their directions matter, so the determinant's sign stands in for dividing by it. That sign keeps the frame right on mirrored texture coordinates, and on Vulkan's downward screen Y.
  - **How close it gets:** on Sponza, whose files do have tangents, this frame matches them to within 4° on 95% of pixels. The differences are at triangle edges.
- **Occlusion:** `1 + strength × (texture − 1)` blends between no effect at strength 0 and the full map at strength 1. It darkens only the ambient part of the lighting: occlusion models light arriving from all around, not from one light.
- **Emissive** is factor × texture. It's added after lighting, since the surface gives this light off whether or not anything lights it.
- **The views:** keys 1 to 8 select what the fragment shader returns. Directions are shown as colors, with each component mapped from −1..1 to 0..1. The swapchain is sRGB, so these data values are displayed brighter than they are, but every view still shows where a value is high or low.

### Code
`game-engine/shaders/mesh.slang`:
```slang
// Draws one glTF primitive: its vertices come from the scene's vertex buffer,
// its place in the world from its DrawData, and its surface from its glTF
// material, whose textures are read from the descriptor heap. Lit by a
// single light until Chapter 7.

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

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

// View: what the fragment shader outputs (keys 1-8).
static const uint view_lit = 0;
static const uint view_base_color = 1;
static const uint view_normal = 2;
static const uint view_vertex_normal = 3;
static const uint view_metallic = 4;
static const uint view_roughness = 5;
static const uint view_occlusion = 6;
static const uint view_emissive = 7;

// Written with vkCmdPushDataEXT before each draw.
struct PushData {
    float4x4 view_projection;  // world space -> clip space
    Vertex *vertices;          // the scene's vertices
    DrawData *draws;           // one DrawData per draw
    Material *materials;       // the scene's materials
    uint draw_index;           // which DrawData this draw uses
    uint view;                 // what to output
};

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
    const Vertex vertex = push.vertices[vertex_id];
    const DrawData draw = push.draws[push.draw_index];

    const float4 world = mul(draw.model, float4(vertex.position, 1.0));

    // Tangent and bitangent lie along the surface, so they move with the
    // model matrix, like positions; only the normal needs the normal matrix.
    // The bitangent is built before the transform, from glTF's rule
    // B = cross(N, T) * w: a mirroring transform then mirrors it too.
    const float3 bitangent = cross(vertex.normal, vertex.tangent.xyz) * vertex.tangent.w;

    VertexOutput output;
    output.position = mul(push.view_projection, world);
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

// --- Fragment shader ---------------------------------------------------------

// Fixed light from above and to the side, until Chapter 7 brings real lighting.
static const float3 light_direction = normalize(float3(0.4, 1.0, 0.3));

// SV_Target: the value written to color attachment 0.
// SV_IsFrontFace: whether this triangle faces the camera.
[shader("fragment")]
float4 fragmentMain(VertexOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    const Material material = push.materials[push.draws[push.draw_index].material];

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

    const float3 normal = surface_normal(input, material, front_face, push.view != view_vertex_normal);

    // The debug views show one input each. Directions are shown as colors:
    // each component's -1..1 mapped to 0..1.
    switch (push.view) {
        case view_base_color: return base_color;
        case view_normal:
        case view_vertex_normal: return float4(normal * 0.5 + 0.5, 1.0);
        case view_metallic: return float4(metallic.xxx, 1.0);
        case view_roughness: return float4(roughness.xxx, 1.0);
        case view_occlusion: return float4(occlusion.xxx, 1.0);
        case view_emissive: return float4(emissive, 1.0);
        default: break;
    }

    // Lit: ambient light, darkened by occlusion, plus the light's direct
    // diffuse part, plus the light the surface gives off itself.
    const float diffuse = max(dot(normal, light_direction), 0.0);
    const float3 lit = base_color.rgb * (0.15 * occlusion + 0.85 * diffuse) + emissive;

    return float4(lit, base_color.a);
}
```

## 6.5 A device feature for `discard`: `vulkan_setup.cpp`

### Why
With SPIR-V 1.6, Slang compiles `discard` to `OpDemoteToHelperInvocation`. A demoted pixel writes nothing, but its shader keeps running as a *helper* for its 2×2 quad. Its neighbours' `ddx`, `ddy` and texture mip selection still need its values, and the normal code reads derivatives after the alpha test. Using demote requires the `shaderDemoteToHelperInvocation` feature. Every Vulkan 1.3 device supports it, but like any feature it must be enabled. Without it, the validation layer reports an error for every pipeline.

### How
- **`has_features`** checks it alongside the other features we require.
- **`create_device`** enables it in the Vulkan 1.3 feature struct. Designated initializers must follow the struct's member order, and `shaderDemoteToHelperInvocation` comes before `synchronization2`.

### Code
In `game-engine/src/vulkan_setup.cpp`, replace `has_features` with:
```cpp
// Only valid once has_extensions() is true: the extension structs in the
// chain may not be queried on a device that lacks their extension.
bool has_features(const vk::raii::PhysicalDevice &device) {
    const Features supported = device.getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceDescriptorHeapFeaturesEXT,
        vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR
    >();

    const auto &vulkan12 = supported.get<vk::PhysicalDeviceVulkan12Features>();
    const auto &vulkan13 = supported.get<vk::PhysicalDeviceVulkan13Features>();

    return supported.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy
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
vk::raii::Device create_device(const GpuChoice &gpu) {
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
            .features = {.samplerAnisotropy = vk::True},  // sharper textures seen at an angle
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

## 6.6 One pipeline per alpha mode: `pipeline.h`, `pipeline.cpp`

### Why
The three alpha modes need different fixed-function state as well as different shader code. Blended surfaces mix with what's behind them and mustn't hide what's drawn after them. Culling depends on each draw: on its material, and on whether its transform mirrors it.

### How
- **The specialization constant's value** is passed when the fragment stage is created.
  - **`VkSpecializationMapEntry`:** says constant ID 0 reads `sizeof(AlphaMode)` bytes at offset 0 of the data.
  - **`VkSpecializationInfo`:** points at the data, here the `alpha_mode` argument itself.
- **Blending** (blend mode only): `color = source.rgb × source.a + destination.rgb × (1 − source.a)`. This is the standard "over" operation glTF's alpha blending describes.
- **Depth writes** are off for blended surfaces. A see-through surface still hides behind solid ones, because the depth *test* stays on. But it doesn't block anything drawn after it, which might be behind it.
- **Back-face culling:** a single-sided surface is only visible from the front, so the GPU can drop triangles facing away before running the fragment shader. A closed mesh hides roughly half its triangles this way.
- **Which side is the front:** that's decided by the order in which a triangle's corners appear on screen, clockwise or counter-clockwise. glTF's front faces are counter-clockwise seen from the front. Our projection's Y flip only makes up for Vulkan's downward Y, so on screen they still appear counter-clockwise. A mirroring transform reverses that.
- **Dynamic culling state:** cull mode and front face are *dynamic state* (core since Vulkan 1.3). They're set while recording, before each draw, so one pipeline per alpha mode serves every combination of double-sided and mirrored.

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

// Draws shaders/mesh.slang into a `color_format` image, depth-tested against
// a `depth_format` depth buffer, for materials with alpha mode `alpha_mode`.
// There is no pipeline layout: shaders find their resources in the
// descriptor heap. Cull mode and front face are set per draw.
vk::raii::Pipeline create_mesh_pipeline(
    const vk::raii::Device &device,
    vk::Format color_format,
    vk::Format depth_format,
    AlphaMode alpha_mode
);
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

// --- The mesh pipeline -------------------------------------------------------

vk::raii::Pipeline create_mesh_pipeline(
    const vk::raii::Device &device,
    vk::Format color_format,
    vk::Format depth_format,
    AlphaMode alpha_mode
) {
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
            .pName = "fragmentMain",
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

    const bool blend = alpha_mode == AlphaMode::blend;

    // Depth: keep a fragment only if it's nearer than what's already there.
    // With reverse-Z (see camera.cpp) nearer means a *greater* depth value,
    // and the buffer is cleared to 0, the far plane. Solid surfaces then
    // record their depth. See-through ones don't: something drawn behind
    // them later must still show through.
    const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = blend ? vk::False : vk::True,
        .depthCompareOp = vk::CompareOp::eGreater,
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

## 6.7 Drawing by alpha mode: `main.cpp`

### Why
Draws now go in three batches, one per alpha mode, each with its own pipeline. Blending depends on drawing order, so the blended batch is sorted every frame. The materials buffer carries every property, and the keyboard picks the view.

### How
- **Drawing order:**
  1. **opaque**, then
  2. **masked**: masked surfaces write depth like opaque ones, so these two batches could go in any order. Keeping them separate means the masked pipeline's `discard` only costs masked draws.
  3. **blended** last, so each blends over everything solid behind it.
- **Sorting blended draws:** blending onto what's already drawn only gives the right result back to front, the farthest surface first. Each frame, the blended batch is sorted by distance from the camera to each draw's center. Comparing squared distances avoids a square root, and gives the same order. This is the usual approximation: it's right for separate objects, and can be wrong for large or intersecting ones.
- **`record_frame`** binds each batch's pipeline, and sets cull mode and front face before each draw:
  - **cull mode:** none for double-sided materials, back faces otherwise,
  - **front face:** clockwise for mirrored draws, counter-clockwise otherwise.

  The descriptor heaps stay bound across pipeline changes.
- **Materials:** the `slot` lambda turns each `TextureRef` into heap indices, adding 1 to the image and the sampler index, so −1, meaning "none", becomes 0, the white texture or the default sampler.
- **Keys 1 to 8:** `poll_events` maps them to a `View`, and shows its name in the window title. `SDLK_1` to `SDLK_8` are consecutive key codes, so the view is just the key minus `SDLK_1`.
- **The draw count line:** `Draws: 89 opaque, 14 masked, 0 blended` for Sponza, printed after loading.

### Code
`game-engine/src/main.cpp`:
```cpp
#include "includes/buffer.h"
#include "includes/camera.h"
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
#include <cstdlib>
#include <exception>
#include <filesystem>
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

// The three alpha modes, in the order they're drawn: solid surfaces first,
// so see-through ones blend over everything behind them.
constexpr std::array alpha_modes{AlphaMode::opaque, AlphaMode::mask, AlphaMode::blend};

// glTF's front faces wind counter-clockwise, seen from the front. Our
// projection's Y flip (see camera.cpp) only undoes the difference between
// OpenGL's upward Y and Vulkan's downward one, so on screen they still wind
// counter-clockwise. A mirroring transform reverses that.
constexpr vk::FrontFace front_face = vk::FrontFace::eCounterClockwise;
constexpr vk::FrontFace mirrored_front_face = vk::FrontFace::eClockwise;

// What to draw: every primitive draw in a scene, and the buffers they read.
// `batches` lists draw indices per alpha mode, in drawing order.
struct DrawList {
    vk::Buffer index_buffer;
    vk::DeviceAddress vertices = 0;
    vk::DeviceAddress draws = 0;
    vk::DeviceAddress materials = 0;
    glm::mat4 view_projection{1.0f};
    View view = View::lit;
    std::span<const Primitive> primitives;
    std::span<const MeshDraw> mesh_draws;
    std::span<const SceneMaterial> scene_materials;
    std::array<std::span<const std::uint32_t>, alpha_modes.size()> batches;
};

// Records: swapchain image -> clear color and depth -> draw everything in
// `draws`, each alpha mode's batch with that mode's pipeline, textures from
// `heaps` -> ready to present.
void record_frame(
    const vk::raii::CommandBuffer &commands,
    const Swapchain &swapchain,
    std::uint32_t image_index,
    std::array<float, 4> color,
    std::span<const vk::raii::Pipeline> pipelines,
    const DescriptorHeaps &heaps,
    const DrawList &draws
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

    // Every texture and sampler the shaders read comes from these two heaps.
    // They stay bound when the pipeline changes.
    bind_descriptor_heaps(commands, heaps);

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

    // One index buffer for the whole scene. Indices go through the GPU's
    // fixed-function index fetch, which also lets it reuse vertices shared
    // between neighbouring triangles.
    commands.bindIndexBuffer(draws.index_buffer, 0, vk::IndexType::eUint32);

    // One draw per primitive per node, batch by batch. Push data says which
    // DrawData to use; the primitive's index range and vertex offset go to
    // drawIndexed.
    for (std::size_t mode = 0; mode < alpha_modes.size(); ++mode) {
        commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines[mode]);

        for (const std::uint32_t i : draws.batches[mode]) {
            const MeshDraw &mesh_draw = draws.mesh_draws[i];
            const Primitive &primitive = draws.primitives[mesh_draw.primitive];
            const SceneMaterial &material = draws.scene_materials[primitive.material];

            // Single-sided surfaces are invisible from behind, so the GPU can
            // skip their back faces before running the fragment shader.
            commands.setCullMode(material.double_sided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack);
            commands.setFrontFace(mesh_draw.mirrored ? mirrored_front_face : front_face);

            const PushData push{
                .view_projection = draws.view_projection,
                .vertices = draws.vertices,
                .draws = draws.draws,
                .materials = draws.materials,
                .draw_index = i,
                .view = draws.view,
            };

            commands.pushDataEXT(vk::PushDataInfoEXT{
                .offset = 0,
                .data = {.address = &push, .size = sizeof(push)},
            });

            commands.drawIndexed(primitive.index_count, 1, primitive.first_index, primitive.vertex_offset, 0);
        }
    }

    commands.endRendering();

    transition(commands, image,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone
    );

    commands.end();
}

// --- Events ------------------------------------------------------------------

// The views' names, in View's order, for the window title.
constexpr std::array view_names{
    "Lit", "Base color", "Normal", "Vertex normal", "Metallic", "Roughness", "Occlusion", "Emissive",
};

// Handles every pending event and fills in `input` for this frame. Keys 1-8
// pick the view. False once the window was closed or Escape pressed.
bool poll_events(SDL_Window *window, CameraInput &input, View &view) {
    input = CameraInput{};
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        const bool escape = event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE;

        if (event.type == SDL_EVENT_QUIT || escape) {
            return false;
        }

        // SDLK_1 to SDLK_8 are consecutive key codes.
        if (event.type == SDL_EVENT_KEY_DOWN && event.key.key >= SDLK_1 && event.key.key < SDLK_1 + view_names.size()) {
            const std::uint32_t index = event.key.key - SDLK_1;
            view = static_cast<View>(index);
            SDL_SetWindowTitle(window, (std::string("game-engine: ") + view_names[index]).c_str());
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

        // One per alpha mode, in alpha_modes' order. Built for swapchain.format
        // and depth_format; recreate_swapchain() picks the same formats again,
        // so the pipelines stay valid across resizes.
        std::vector<vk::raii::Pipeline> pipelines;

        for (const AlphaMode mode : alpha_modes) {
            pipelines.push_back(create_mesh_pipeline(device, swapchain.format, depth_format, mode));
        }

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
        for (vk::raii::CommandBuffer &commands : command_buffers) {
            frames.push_back(Frame{
                .commands = std::move(commands),
                .image_acquired = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
                // Start signalled, so the first wait on each frame returns straight away.
                .done = vk::raii::Fence(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}),
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

        for (const MeshDraw &draw : scene.draws) {
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

        // --- Textures and materials ------------------------------------------

        // Decode every image, upload them with mipmaps, and describe them in
        // the descriptor heap. Texture 0 is white; scene image i is texture i + 1.
        const std::uint64_t texture_start = SDL_GetTicksNS();
        const std::vector<Texture> textures = create_scene_textures(device, *gpu, queue, command_pool, scene);
        const DescriptorHeaps heaps = create_descriptor_heaps(device, *gpu, queue, command_pool, textures, scene.samplers);

        std::println("Textures: {} in {:.0f} ms (whole load {:.0f} ms)", textures.size(),
            static_cast<double>(SDL_GetTicksNS() - texture_start) * 1e-6,
            static_cast<double>(SDL_GetTicksNS() - load_start) * 1e-6);

        // Heap indices are one past the scene's: image i is texture i + 1 and
        // sampler i is sampler i + 1, so "none" (-1) becomes 0, the white
        // texture or the default sampler.
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
        View view = View::lit;

        std::uint64_t previous_ticks = SDL_GetTicksNS();

        // --- Frame loop ------------------------------------------------------

        const std::array black{0.0f, 0.0f, 0.0f, 1.0f};
        std::uint64_t frame_count = 0;

        while (poll_events(window.get(), input, view)) {
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
            previous_ticks = ticks;

            update_camera(camera, input, seconds);

            const float aspect = static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height);

            // Blending mixes with what's already drawn, so see-through draws go
            // back to front: farthest from the camera first. Sorting by each
            // draw's center is approximate, but right for separate objects.
            std::vector<std::uint32_t> &blended = batches[static_cast<std::size_t>(AlphaMode::blend)];

            std::ranges::sort(blended, std::ranges::greater{}, [&](std::uint32_t i) {
                const glm::vec3 offset = scene.draws[i].center - camera.position;
                return glm::dot(offset, offset);
            });

            const DrawList draws{
                .index_buffer = *index_buffer.handle,
                .vertices = vertex_buffer.address,
                .draws = draw_buffer.address,
                .materials = material_buffer.address,
                .view_projection = camera.projection(aspect) * camera.view(),
                .view = view,
                .primitives = scene.primitives,
                .mesh_draws = scene.draws,
                .scene_materials = scene.materials,
                .batches = {batches[0], batches[1], batches[2]},
            };

            // --- Render -----------------------------------------------------

            Frame &frame = frames[frame_count % frames_in_flight];

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
            record_frame(frame.commands, swapchain, image_index, black, pipelines, heaps, draws);

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
    } catch (const std::exception &e) {
        std::println(stderr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

## 6.8 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The terminal** shows the same `Loaded Sponza.gltf: …` and `Textures: …` lines as Chapter 5, then `Draws: 89 opaque, 14 masked, 0 blended`. In release, the whole load takes about 235 ms on the machine this was written on.
- **The plants** in their pots and on the pillars are cut-out leaves; the dark rectangles of Chapter 5 are gone.
- **Normal maps** give the walls their brick joints, the arches their carved edges and the lion's head its relief. The effect is easiest to see by switching between keys 3 and 4, which show the normal with and without the normal map.
- **The views:**

  | Key | View | Sponza shows |
  |---|---|---|
  | 1 | Lit | the shaded scene |
  | 2 | Base color | the textures without lighting |
  | 3 | Normal | the final normal, normal map included |
  | 4 | Vertex normal | the smooth interpolated normal |
  | 5 | Metallic | black almost everywhere; white on the metal fittings, like the curtain rods and chains, and on the curtains' gold patterns |
  | 6 | Roughness | mostly bright: Sponza's stone and cloth are rough |
  | 7 | Occlusion | white: Sponza has no occlusion maps |
  | 8 | Emissive | black: nothing in Sponza glows |
- **No `[validation …]` lines.**

**Other models** test what Sponza doesn't use. To load one, change `scene_file` in `main.cpp`:
- **AlphaBlendModeTest:** every panel shows its check mark. The blended panel is see-through, and the three masked ones cut off at their cutoffs.
- **NegativeScaleTest:** every check mark, front and back, mirrored or not. Those check marks are the culling and front-face logic at work.
- **DamagedHelmet:** a normal map without tangents in the file, emissive lights (key 8) and an occlusion map (key 7).
- **NormalTangentMirrorTest:** every cell looks alike when the file's tangents are used correctly.
- **SheenChair:** its occlusion map uses `TEXCOORD_1`.
- **MetalRoughSpheres:** key 5 shows the metal spheres at the top and the non-metal ones at the bottom, and key 6 shows roughness rising from left to right.

All 52 models that loaded in Chapter 5 render without validation errors.

Next, in Chapter 7, the material inputs feed physically based lighting: metallic and roughness drive a real reflection model, replacing the single fixed light.

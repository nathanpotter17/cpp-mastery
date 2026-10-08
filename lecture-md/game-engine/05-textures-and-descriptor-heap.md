# Chapter 5: Textures and the descriptor heap

By the end of this chapter Sponza is textured: curtains, stone, floor and urns in their real colors. Getting there takes three steps:
1. **decode** the PNG and JPEG images tinygltf loaded in Chapter 4, on every CPU core at once,
2. **upload** them as GPU images with full mip chains, all in one submission,
3. make them visible to shaders through the **descriptor heap**, `VK_EXT_descriptor_heap`, the binding model Chapter 0 set up the device for.

This chapter builds on [Chapter 4](04-gltf-geometry.md).

## 5.1 Image decoders: `vendor/jpeg-compressor`, `vendor/libspng`

### Why
glTF images are almost always JPEG or PNG; 65 of Sponza's 69 are JPEG. Both are compressed formats a GPU can't sample directly, so we decode them to plain pixels on the CPU first.

### How
- **jpgd** (from `jpeg-compressor`) decodes JPEG. It's a single `.cpp` file, public domain or Apache 2.0, with SSE2 acceleration built in. Its repository also has an encoder and example programs, so instead of `add_subdirectory` we compile just `jpgd.cpp` as a small library. Recent clang warns about a few unused variables in it; `-w` keeps that third-party compile quiet.
- **libspng** decodes PNG. It's a CMake project like SDL, configured by setting variables before `add_subdirectory`. It inflates PNG's compressed data with the system's zlib, which its CMakeLists.txt finds with `find_package(ZLIB)`. On Ubuntu that's the `zlib1g-dev` package.

### Code
From the repo root:
```bash
git clone https://github.com/richgel999/jpeg-compressor game-engine/vendor/jpeg-compressor
```
```bash
git -C game-engine/vendor/jpeg-compressor checkout aeb7d3b
```
```bash
git clone --depth 1 --branch v0.7.4 https://github.com/randy408/libspng game-engine/vendor/libspng
```

In `game-engine/CMakeLists.txt`, add these sections after the `vendor/tinygltf` section:
```cmake
# --- vendor/jpeg-compressor (jpgd) -------------------------------------------
# A JPEG decoder in one .cpp file. Its repository also holds an encoder and
# example programs; we compile just the decoder, quietly, as a small library.

add_library(jpgd STATIC vendor/jpeg-compressor/jpgd.cpp)
target_include_directories(jpgd SYSTEM PUBLIC vendor/jpeg-compressor)
target_compile_options(jpgd PRIVATE -w)

# --- vendor/libspng ----------------------------------------------------------
# A PNG decoder. It inflates the compressed data with the system's zlib
# (find_package(ZLIB) inside its CMakeLists.txt).

set(SPNG_SHARED OFF)
set(SPNG_STATIC ON)
set(BUILD_EXAMPLES OFF)
add_subdirectory(vendor/libspng SYSTEM)
```

Then link both. This replaces the `target_link_libraries` line:
```cmake
target_link_libraries(game-engine PRIVATE SDL3::SDL3 Vulkan::Headers glm::glm tinygltf jpgd spng_static)
```

**Build once and restart clangd.** Nothing uses the decoders yet, so the code from the previous chapter still builds. Run `./game-engine/build.bash` and pick option 1: CMake reconfigures and writes the decoders' include paths into `compile_commands.json`. Then reload the editor window (**Developer: Reload Window**, or **clangd: Restart language server**), so clangd finds `<jpgd.h>` or `<spng.h>` when the code below includes it.

## 5.2 Texture coordinates and materials: `shader_types.h`

### Why
To sample a texture, each vertex needs **texture coordinates**, often called "UVs": where on the image that point of the surface lies. Each draw needs to know its **material**, and the material says which texture to use and a color factor to multiply it by.

### How
- **`Vertex` gains `uv`** and grows to 32 bytes.
- **`DrawData` gains a material index.** Materials are shared, since many draws use the same one, so they get their own buffer, and each draw stores an index into it.
- **`Material` holds the glTF base color factor and a texture index.**
  - The index counts descriptors in the resource heap.
  - Index 0 is a 1×1 white texture, so a material without a texture still samples one: white times the factor is just the factor, and the shader needs no branch for "no texture".
- **Padding:** `DrawData` and `Material` are padded to multiples of 16 bytes, so each `float4` in an array of them starts 16-byte aligned, which GPUs load fastest. Chapter 6 fills `Material`'s padding with more material properties. The padding members get `{}` default values, so leaving them out of a designated initializer doesn't trigger clang's `-Wmissing-designated-field-initializers`.
- **Push data** gains the material buffer's address.

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

// Slang lays out data behind a pointer like C: no padding between members,
// so position is at byte 0, normal at 12, uv at 24, and a vertex is 32.
// A normal of (0, 0, 0) means the file had none (see mesh.slang).
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;  // texture coordinates (TEXCOORD_0)
};

static_assert(sizeof(Vertex) == 32);
static_assert(offsetof(Vertex, normal) == 12);
static_assert(offsetof(Vertex, uv) == 24);

// --- Per-draw data -----------------------------------------------------------

// One per draw, in a GPU buffer the shader indexes. Padded to a multiple of
// 16 bytes, so every DrawData in the array starts 16-byte aligned.
struct DrawData {
    glm::mat4 model;          // this primitive's space -> world space
    glm::mat4 normal_matrix;  // transposed inverse of model: keeps normals perpendicular under any scale
    std::uint32_t material;   // index into the material buffer
    std::uint32_t padding[3]{};
};

static_assert(sizeof(DrawData) == 144);
static_assert(offsetof(DrawData, normal_matrix) == 64);
static_assert(offsetof(DrawData, material) == 128);

// --- Materials ---------------------------------------------------------------

// What the shader needs to know about a glTF material, so far. Texture
// indices count descriptors in the resource heap; 0 is a 1x1 white texture,
// so a material without a texture multiplies by white. Chapter 6 fills the
// padding with more material properties.
struct Material {
    glm::vec4 base_color_factor;      // linear RGBA, multiplies the texture
    std::uint32_t base_color_texture; // resource heap index
    std::uint32_t padding[3]{};
};

static_assert(sizeof(Material) == 32);
static_assert(offsetof(Material, base_color_texture) == 16);

// --- Push data ---------------------------------------------------------------

// Written with vkCmdPushDataEXT before each draw. The push block uses std430
// rules: the float4x4 comes first, then 8-byte pointers, then the index.
struct PushData {
    glm::mat4 view_projection;   // world space -> clip space, the same for every draw
    vk::DeviceAddress vertices;  // where the first Vertex is in GPU memory
    vk::DeviceAddress draws;     // where the first DrawData is in GPU memory
    vk::DeviceAddress materials; // where the first Material is in GPU memory
    std::uint32_t draw_index;    // which DrawData this draw uses
};

static_assert(offsetof(PushData, vertices) == 64);
static_assert(offsetof(PushData, draws) == 72);
static_assert(offsetof(PushData, materials) == 80);
static_assert(offsetof(PushData, draw_index) == 88);
```

## 5.3 Materials and images in the loader: `scene.h` / `scene.cpp`

### Why
The loader has to bring three more things out of the glTF file:
- texture coordinates,
- each primitive's material,
- the images themselves.

It also has to know which images hold **colors** and which hold **data**, because they're stored differently.

### How
- **Texture coordinates:** `TEXCOORD_0` is read with `read_floats`, so the normalized-integer forms glTF allows work too. Primitives without UVs get (0, 0).
- **Materials:** `add_materials_and_images` converts each glTF material to a `SceneMaterial`: base color factor plus the image its base color texture uses. The chain is material → texture → image; a texture can also name a sampler, which Chapter 6 uses. A plain white default material goes at the end, for primitives that don't name one.
- **Images:** each image's encoded bytes move out of tinygltf's model, and its file name is kept for error messages.
- **sRGB vs. linear:** color images are stored *sRGB-encoded*, a nonlinear curve that spends more of the 8 bits on dark shades, where the eye notices steps. Data images (normals, roughness) are plain linear numbers. Base color and emissive textures hold colors, so the loader marks their images `srgb`.
- **Validation:** a primitive's material index and a texture's image index are both checked against their arrays. Out of range, the GPU would read past the material buffer, or the heap would hand out the wrong descriptor.

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

// A glTF material, reduced to what we draw so far.
struct SceneMaterial {
    glm::vec4 base_color_factor{1.0f};
    std::int32_t base_color_image = -1;  // index into Scene::images, or -1 for none
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

    // World-space box around everything drawn, for placing the camera.
    glm::vec3 bounds_min{std::numeric_limits<float>::max()};
    glm::vec3 bounds_max{std::numeric_limits<float>::lowest()};
};

// Loads the default scene of a .gltf or .glb file, with its materials and
// its images, still encoded.
Scene load_gltf(const std::filesystem::path& path);
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
    const tinygltf::Model& model, int view_index, std::size_t offset,
    std::size_t count, std::size_t element_size, std::size_t stride
) {
    const tinygltf::BufferView& view = model.bufferViews.at(view_index);
    const tinygltf::Buffer& buffer = model.buffers.at(view.buffer);

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
std::vector<unsigned char> accessor_elements(const tinygltf::Model& model, int accessor_index) {
    const tinygltf::Accessor& accessor = model.accessors.at(accessor_index);

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
        const auto& sparse = accessor.sparse;
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
std::vector<float> read_floats(const tinygltf::Model& model, int accessor_index, int components) {
    const tinygltf::Accessor& accessor = model.accessors.at(accessor_index);

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

std::vector<glm::vec3> read_vec3(const tinygltf::Model& model, int accessor_index) {
    const std::vector<float> floats = read_floats(model, accessor_index, 3);
    std::vector<glm::vec3> values(floats.size() / 3);
    std::memcpy(values.data(), floats.data(), floats.size() * sizeof(float));
    return values;
}

std::vector<glm::vec2> read_vec2(const tinygltf::Model& model, int accessor_index) {
    const std::vector<float> floats = read_floats(model, accessor_index, 2);
    std::vector<glm::vec2> values(floats.size() / 2);
    std::memcpy(values.data(), floats.data(), floats.size() * sizeof(float));
    return values;
}

// Indices may be 8, 16 or 32 bits, even within one file; they're widened to
// 32 bits so the whole scene can share one index buffer.
std::vector<std::uint32_t> read_indices(const tinygltf::Model& model, int accessor_index) {
    const tinygltf::Accessor& accessor = model.accessors.at(accessor_index);
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
std::vector<std::uint32_t> to_triangle_list(int mode, const std::vector<std::uint32_t>& indices) {
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
    const tinygltf::Model& model, const tinygltf::Primitive& source, std::uint32_t default_material, Scene& scene
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

    // Texture coordinates too; without them every vertex samples the corner (0, 0).
    std::vector<glm::vec2> uvs(positions.size(), glm::vec2{0.0f});

    if (const auto uv = source.attributes.find("TEXCOORD_0"); uv != source.attributes.end()) {
        uvs = read_vec2(model, uv->second);

        if (uvs.size() != positions.size()) {
            throw std::runtime_error("a primitive has a different number of texture coordinates and positions");
        }
    }

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
        scene.vertices.push_back(Vertex{.position = positions[i], .normal = normals[i], .uv = uvs[i]});
        loaded.local_min = glm::min(loaded.local_min, positions[i]);
        loaded.local_max = glm::max(loaded.local_max, positions[i]);
    }

    scene.indices.insert(scene.indices.end(), indices.begin(), indices.end());
    return loaded;
}

// --- Nodes -------------------------------------------------------------------

// A node's transform relative to its parent: either a full matrix, or
// translation * rotation * scale. glTF stores doubles; we draw with floats.
glm::mat4 local_transform(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        return glm::mat4(glm::make_mat4(node.matrix.data()));  // column-major, like glm
    }

    glm::mat4 transform{1.0f};

    if (node.translation.size() == 3) {
        transform = glm::translate(transform, glm::vec3(glm::make_vec3(node.translation.data())));
    }

    if (node.rotation.size() == 4) {
        // glTF stores (x, y, z, w); glm's constructor takes w first.
        const auto& r = node.rotation;
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
    const tinygltf::Model& model,
    int node_index,
    const glm::mat4& parent,
    const std::vector<std::vector<LoadedPrimitive>>& mesh_primitives,
    Scene& scene
) {
    const tinygltf::Node& node = model.nodes.at(node_index);
    const glm::mat4 world = parent * local_transform(node);

    if (node.mesh >= 0) {
        for (const LoadedPrimitive& primitive : mesh_primitives.at(node.mesh)) {
            scene.draws.push_back(MeshDraw{.model = world, .primitive = primitive.index});

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
std::int32_t texture_image(const tinygltf::Model& model, int texture_index) {
    if (texture_index < 0) {
        return -1;
    }

    const int image = model.textures.at(texture_index).source;

    if (image >= static_cast<int>(model.images.size())) {
        throw std::runtime_error("texture " + std::to_string(texture_index) + " points past the last image");
    }

    return image;
}

void add_materials_and_images(tinygltf::Model& model, Scene& scene) {
    for (const tinygltf::Material& source : model.materials) {
        const auto& pbr = source.pbrMetallicRoughness;

        scene.materials.push_back(SceneMaterial{
            .base_color_factor = glm::vec4(glm::make_vec4(pbr.baseColorFactor.data())),
            .base_color_image = texture_image(model, pbr.baseColorTexture.index),
        });
    }

    // For primitives that don't name a material, glTF's default: plain white.
    scene.materials.push_back(SceneMaterial{});

    // The image bytes move out of tinygltf's model; it's discarded afterwards.
    for (tinygltf::Image& image : model.images) {
        scene.images.push_back(SceneImage{
            .encoded = std::move(image.image),
            .name = image.uri.empty() ? image.name : image.uri,
        });
    }

    // Colors are stored sRGB-encoded; everything else (normals, roughness, ...)
    // is plain data. Base color and emissive textures hold colors.
    for (const tinygltf::Material& source : model.materials) {
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
    tinygltf::Image* image, int /*image_index*/, std::string* /*error*/, std::string* /*warning*/,
    int /*required_width*/, int /*required_height*/, const unsigned char* bytes, int size, void* /*user_data*/
) {
    image->image.assign(bytes, bytes + size);
    image->as_is = true;
    return true;
}

}  // namespace

// --- Loading -----------------------------------------------------------------

Scene load_gltf(const std::filesystem::path& path) {
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
    for (const std::string& extension : model.extensionsRequired) {
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
        for (const tinygltf::Primitive& primitive : model.meshes[m].primitives) {
            if (const auto loaded_primitive = add_primitive(model, primitive, default_material, scene)) {
                mesh_primitives[m].push_back(*loaded_primitive);
            }
        }
    }

    // The file may contain several scenes; draw its default one.
    if (model.scenes.empty()) {
        throw std::runtime_error(path.string() + " has no scenes");
    }

    const tinygltf::Scene& root = model.scenes.at(model.defaultScene >= 0 ? model.defaultScene : 0);

    for (const int node : root.nodes) {
        visit_node(model, node, glm::mat4{1.0f}, mesh_primitives, scene);
    }

    if (scene.draws.empty()) {
        throw std::runtime_error(path.string() + " has nothing to draw");
    }

    return scene;
}
```

## 5.4 Decoding and uploading textures: `texture.h` / `texture.cpp`

### Why
Two parts of texture loading take real time, and each has a standard fix:
- **Decoding is CPU work, and images are independent,** so they can all decode at once, one per CPU core. Sponza's 69 take about 100 ms that way.
- **Uploading is GPU work.** Submitting and waiting once per texture would wait for the GPU 70 times. Recording every copy into one command buffer waits once.

Textures also need **mipmaps**: copies at half, quarter, … size down to 1×1. A distant surface samples a smaller copy, which avoids shimmering and is faster to read. glTF files only ship the full-size image, so we generate the rest on the GPU.

### How
- **Choosing a decoder:** `decode_image` looks at the file's first bytes, its "magic number": `89 'P' 'N' 'G'` is PNG and `FF D8 FF` is JPEG. That's more reliable than the file name or the glTF `mimeType`.
  - **jpgd**, asked for 4 components, returns RGBA in a `malloc`ed buffer; a `unique_ptr` with `std::free` releases it.
  - **libspng**'s `SPNG_FMT_RGBA8` converts any PNG (grey, palette, 16-bit) to 8-bit RGBA, and `SPNG_DECODE_TRNS` turns a transparency chunk into real alpha.
- **Failures don't stop the load.** An image that can't be decoded, whether truncated, empty, or another format like WebP, becomes a single magenta pixel, impossible to miss on screen, and a message names the image and the reason.
- **Decoding in parallel:** `decode_images` starts one `std::jthread` per CPU core. Each takes the next image from a shared `std::atomic` counter until none are left. The threads write to different elements of the result, so they need no lock. A `jthread` joins automatically when it goes out of scope, which is the end of the inner block.
- **`create_texture`:**
  - **Mip count:** `std::bit_width` of the larger side; a 1024-wide image has 11 levels.
  - **Usage:** transfer-destination (the upload), transfer-source (making mips from it), and sampled (shaders).
- **Formats:** `R8G8B8A8_SRGB` for color images, `R8G8B8A8_UNORM` for data. With an sRGB format, the GPU converts to linear whenever it samples, and filters mips in linear space, which is correct for blending colors.
- **Format support:** linear-filtered blits aren't allowed for every format, so `create_scene_textures` checks both formats first.
- **`record_upload`, for each texture:**
  1. move every mip level to transfer-destination layout,
  2. copy the pixels from the staging buffer into level 0,
  3. for each smaller level, make the level above a blit source and blit it down with linear filtering, which averages each 2×2 block,
  4. move every level to shader-read layout.

  Barriers order each step after the one before; `transition_mips` wraps them.
- **One staging buffer and one submission:** all images' pixels go into one staging buffer, back to back. Every texture's commands go into one command buffer, submitted once with `submit_and_wait`.
- **Texture 0 is the white 1×1 texture,** and scene image *i* becomes texture *i* + 1.

### Code
`game-engine/src/includes/texture.h`:
```cpp
#pragma once

#include "includes/scene.h"
#include "includes/vulkan_setup.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

// --- Decoding ----------------------------------------------------------------

// Pixels decoded from a PNG or JPEG: 8-bit RGBA, row by row, top to bottom.
struct DecodedImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> rgba;
    std::string error{};  // why decoding failed, if it did
};

// Decodes every image in `images`, several at once, one per CPU core. An
// image that can't be decoded becomes a single magenta pixel, so it's easy
// to spot on screen, and its `error` says why.
std::vector<DecodedImage> decode_images(std::span<const SceneImage> images);

// --- GPU textures ------------------------------------------------------------

// A sampled image with a full mip chain, in device-local memory, ready for
// shaders: its layout is eShaderReadOnlyOptimal. Members are destroyed
// bottom-up, so the image goes before its memory.
struct Texture {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image handle = nullptr;
    vk::Format format = vk::Format::eUndefined;
    vk::Extent2D extent;
    std::uint32_t mip_levels = 1;
};

// The textures for a scene: index 0 is a 1x1 white texture, for materials
// without one, and scene image i is texture i + 1. All of them are uploaded
// and given mipmaps in a single submission.
std::vector<Texture> create_scene_textures(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    const Scene& scene
);
```

`game-engine/src/texture.cpp`:
```cpp
#include "includes/texture.h"

#include "includes/buffer.h"

#include <jpgd.h>
#include <spng.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <print>
#include <stdexcept>
#include <thread>

namespace {

// --- Decoding ----------------------------------------------------------------

// What a failed image becomes: one magenta pixel, impossible to miss.
DecodedImage missing_image(std::string error) {
    return DecodedImage{.width = 1, .height = 1, .rgba = {255, 0, 255, 255}, .error = std::move(error)};
}

DecodedImage decode_jpeg(std::span<const std::uint8_t> encoded) {
    int width = 0;
    int height = 0;
    int components = 0;

    // Asking for 4 components gives RGBA, with alpha 255. jpgd allocates the
    // pixels with malloc, so free() releases them.
    const std::unique_ptr<unsigned char, decltype(&std::free)> pixels(
        jpgd::decompress_jpeg_image_from_memory(encoded.data(), static_cast<int>(encoded.size()), &width, &height, &components, 4),
        &std::free
    );

    if (!pixels) {
        return missing_image("not a JPEG jpgd can decode");
    }

    const std::size_t size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4;

    return DecodedImage{
        .width = static_cast<std::uint32_t>(width),
        .height = static_cast<std::uint32_t>(height),
        .rgba = std::vector<std::uint8_t>(pixels.get(), pixels.get() + size),
    };
}

DecodedImage decode_png(std::span<const std::uint8_t> encoded) {
    const std::unique_ptr<spng_ctx, decltype(&spng_ctx_free)> context(spng_ctx_new(0), &spng_ctx_free);

    spng_ihdr header{};
    std::size_t size = 0;

    // SPNG_FMT_RGBA8 converts any PNG (grey, palette, 16-bit, ...) to 8-bit
    // RGBA; SPNG_DECODE_TRNS turns a transparency chunk into real alpha.
    const bool decoded = context
        && spng_set_png_buffer(context.get(), encoded.data(), encoded.size()) == 0
        && spng_get_ihdr(context.get(), &header) == 0
        && spng_decoded_image_size(context.get(), SPNG_FMT_RGBA8, &size) == 0;

    if (!decoded) {
        return missing_image("not a PNG libspng can read");
    }

    DecodedImage image{.width = header.width, .height = header.height, .rgba = std::vector<std::uint8_t>(size)};

    if (spng_decode_image(context.get(), image.rgba.data(), size, SPNG_FMT_RGBA8, SPNG_DECODE_TRNS) != 0) {
        return missing_image("libspng failed to decode it");
    }

    return image;
}

// Chooses the decoder from the file's first bytes, its "magic number", which
// is more reliable than the file name or the glTF mimeType.
DecodedImage decode_image(std::span<const std::uint8_t> encoded) {
    constexpr std::uint8_t png[] = {0x89, 'P', 'N', 'G'};
    constexpr std::uint8_t jpeg[] = {0xFF, 0xD8, 0xFF};

    if (encoded.size() >= 4 && std::equal(std::begin(png), std::end(png), encoded.begin())) {
        return decode_png(encoded);
    }

    if (encoded.size() >= 3 && std::equal(std::begin(jpeg), std::end(jpeg), encoded.begin())) {
        return decode_jpeg(encoded);
    }

    return missing_image(encoded.empty() ? "no image data" : "not a PNG or JPEG");
}

// --- Uploading ---------------------------------------------------------------

// Moves mip levels [base, base + count) of `image` between layouts.
void transition_mips(
    const vk::raii::CommandBuffer& commands,
    vk::Image image,
    std::uint32_t base,
    std::uint32_t count,
    vk::ImageLayout from,
    vk::ImageLayout to,
    vk::PipelineStageFlags2 src_stage,
    vk::AccessFlags2 src_access,
    vk::PipelineStageFlags2 dst_stage,
    vk::AccessFlags2 dst_access
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
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = base,
            .levelCount = count,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    commands.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}

// Records: staging bytes -> mip 0 -> each smaller mip blitted from the one
// above it -> every level ready for shaders to sample.
void record_upload(const vk::raii::CommandBuffer& commands, const Texture& texture, vk::Buffer staging, vk::DeviceSize offset) {
    const vk::Image image = *texture.handle;

    transition_mips(commands, image, 0, texture.mip_levels,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
        vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);

    commands.copyBufferToImage(staging, image, vk::ImageLayout::eTransferDstOptimal, vk::BufferImageCopy{
        .bufferOffset = offset,
        .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageExtent = {texture.extent.width, texture.extent.height, 1},
    });

    // Each level is half the size of the one above, down to 1x1. A linear
    // blit averages each 2x2 block of the larger level into one texel.
    std::int32_t width = static_cast<std::int32_t>(texture.extent.width);
    std::int32_t height = static_cast<std::int32_t>(texture.extent.height);

    for (std::uint32_t level = 1; level < texture.mip_levels; ++level) {
        // The level above was just written; make it the blit's source.
        transition_mips(commands, image, level - 1, 1,
            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
            vk::PipelineStageFlagBits2::eCopy | vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead);

        const std::int32_t next_width = std::max(width / 2, 1);
        const std::int32_t next_height = std::max(height / 2, 1);

        const vk::ImageBlit2 region{
            .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = level - 1, .baseArrayLayer = 0, .layerCount = 1},
            .srcOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{width, height, 1}},
            .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = level, .baseArrayLayer = 0, .layerCount = 1},
            .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{next_width, next_height, 1}},
        };

        commands.blitImage2(vk::BlitImageInfo2{
            .srcImage = image,
            .srcImageLayout = vk::ImageLayout::eTransferSrcOptimal,
            .dstImage = image,
            .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
            .regionCount = 1,
            .pRegions = &region,
            .filter = vk::Filter::eLinear,
        });

        width = next_width;
        height = next_height;
    }

    // Every level but the last was a blit source; the last was only written.
    if (texture.mip_levels > 1) {
        transition_mips(commands, image, 0, texture.mip_levels - 1,
            vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead);
    }

    transition_mips(commands, image, texture.mip_levels - 1, 1,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::PipelineStageFlagBits2::eCopy | vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead);
}

// An image for `decoded` with room for every mip level, in device-local memory.
Texture create_texture(const vk::raii::Device& device, const GpuChoice& gpu, const DecodedImage& decoded, vk::Format format) {
    Texture texture;
    texture.format = format;
    texture.extent = vk::Extent2D{.width = decoded.width, .height = decoded.height};

    // Halving until 1x1: a 1024x1024 image has 11 levels (1024, 512, ..., 1).
    texture.mip_levels = std::bit_width(std::max(decoded.width, decoded.height));

    texture.handle = vk::raii::Image(device, vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {decoded.width, decoded.height, 1},
        .mipLevels = texture.mip_levels,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        // Copied into (dst), blitted from (src) to make mips, then sampled.
        .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    });

    const vk::MemoryRequirements requirements = texture.handle.getMemoryRequirements();

    texture.memory = vk::raii::DeviceMemory(device, vk::MemoryAllocateInfo{
        .allocationSize = requirements.size,
        .memoryTypeIndex = find_memory_type(gpu, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
    });

    texture.handle.bindMemory(*texture.memory, 0);
    return texture;
}

}  // namespace

// --- Decoding ----------------------------------------------------------------

std::vector<DecodedImage> decode_images(std::span<const SceneImage> images) {
    std::vector<DecodedImage> decoded(images.size());

    // Each worker thread takes the next undecoded image until none are left.
    // Images are independent, so the threads never touch the same one.
    std::atomic<std::size_t> next{0};
    const unsigned workers = std::max(1u, std::thread::hardware_concurrency());

    {
        std::vector<std::jthread> threads;

        for (unsigned t = 0; t < workers; ++t) {
            threads.emplace_back([&] {
                for (std::size_t i = next++; i < images.size(); i = next++) {
                    decoded[i] = decode_image(std::span(images[i].encoded.data(), images[i].encoded.size()));
                }
            });
        }
    }  // jthreads join here, when they go out of scope

    for (std::size_t i = 0; i < images.size(); ++i) {
        if (!decoded[i].error.empty()) {
            std::println(stderr, "Image {} ({}): {}; using magenta", i, images[i].name, decoded[i].error);
        }
    }

    return decoded;
}

// --- GPU textures ------------------------------------------------------------

std::vector<Texture> create_scene_textures(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    const Scene& scene
) {
    // Index 0: white, so "no texture" can sample like any other.
    std::vector<DecodedImage> images{DecodedImage{.width = 1, .height = 1, .rgba = {255, 255, 255, 255}}};
    std::vector<vk::Format> formats{vk::Format::eR8G8B8A8Srgb};

    for (DecodedImage& image : decode_images(scene.images)) {
        images.push_back(std::move(image));
    }

    // sRGB formats make the GPU decode colors to linear when sampling (and
    // filter them correctly when blitting mips); data textures stay as they are.
    for (const SceneImage& image : scene.images) {
        formats.push_back(image.srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm);
    }

    // Mip generation blits with linear filtering, which not every format allows.
    for (const vk::Format format : {vk::Format::eR8G8B8A8Srgb, vk::Format::eR8G8B8A8Unorm}) {
        const vk::FormatFeatureFlags needed = vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst
            | vk::FormatFeatureFlagBits::eSampledImageFilterLinear;

        if ((gpu.device.getFormatProperties(format).optimalTilingFeatures & needed) != needed) {
            throw std::runtime_error(vk::to_string(format) + " can't be used to generate mipmaps on this GPU");
        }
    }

    // One staging buffer holds every image's pixels back to back.
    vk::DeviceSize total = 0;
    for (const DecodedImage& image : images) {
        total += image.rgba.size();
    }

    const Buffer staging = create_buffer(device, gpu, total, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    auto* mapped = static_cast<std::uint8_t*>(staging.memory.mapMemory(0, total));
    std::vector<vk::DeviceSize> offsets;
    vk::DeviceSize offset = 0;

    for (const DecodedImage& image : images) {
        std::memcpy(mapped + offset, image.rgba.data(), image.rgba.size());
        offsets.push_back(offset);
        offset += image.rgba.size();
    }

    staging.memory.unmapMemory();

    std::vector<Texture> textures;
    for (std::size_t i = 0; i < images.size(); ++i) {
        textures.push_back(create_texture(device, gpu, images[i], formats[i]));
    }

    // Every copy and blit in one command buffer and one submission.
    submit_and_wait(device, queue, pool, [&](const vk::raii::CommandBuffer& commands) {
        for (std::size_t i = 0; i < textures.size(); ++i) {
            record_upload(commands, textures[i], *staging.handle, offsets[i]);
        }
    });

    return textures;
}
```

## 5.5 The descriptor heap: `descriptor_heap.h` / `descriptor_heap.cpp`

### Why
A shader needs a **descriptor** to read an image: a small block of bytes that tells the GPU where the image is, its format, and its mip levels. Classic Vulkan wraps descriptors in descriptor set layouts, pools and sets, bound to pipeline layouts. `VK_EXT_descriptor_heap` removes all of that. A **heap** is just a buffer we own, full of descriptor bytes, and shaders index it like an array.

There are two heaps:
- **the resource heap** holds image and buffer descriptors,
- **the sampler heap** holds samplers, which describe *how* to read: filtering, wrapping, anisotropy.

### How
- **Sizes come from the GPU.** `PhysicalDeviceDescriptorHeapPropertiesEXT` gives each descriptor's size and alignment (32 bytes for an image on this GPU, as Chapter 0 printed), and each heap's required alignment and **reserved range**: a block of the heap's bytes the driver needs for itself.
- **Layout:** descriptor *i* sits at *i* × `imageDescriptorSize`, which is where a shader's `Texture2D.Handle(i)` reads. The reserved range follows the last descriptor, and the total is rounded up to the heap alignment.
- **Writing descriptors:** `writeResourceDescriptorsEXT` writes the bytes for every image in one call. Each image is described by a *view description* (`ImageViewCreateInfo`) plus its layout, so no `VkImageView` objects are needed. The info structs point at each other, so the vectors holding them are sized up front, and no element moves before the write.
- **The sampler:** `writeSamplerDescriptorsEXT` writes one sampler into the sampler heap:
  - **trilinear:** linear filtering within a mip level and between levels,
  - **anisotropic** at the GPU's maximum: extra samples along the direction a surface recedes in, which keeps floors sharp at grazing angles. Chapter 0 enabled the `samplerAnisotropy` feature for this,
  - **repeat** addressing, so texture coordinates outside 0..1 tile.
- **Uploading:** both heaps go to device-local memory with `upload_buffer`, using usage `eDescriptorHeapEXT | eShaderDeviceAddress`. A heap must start at an address that's a multiple of the heap alignment, so `upload_heap` checks that rather than assuming it.
- **Binding:** `bind_descriptor_heaps` records `bindResourceHeapEXT` and `bindSamplerHeapEXT`, each with the heap's address range and where its reserved range is. Bound heaps stay in effect for the rest of the command buffer.

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
// resource heap) and one sampler (index 0 of the sampler heap), and uploads
// both heaps to device-local memory.
DescriptorHeaps create_descriptor_heaps(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    std::span<const Texture> textures
);

// Makes `heaps` the ones shaders read for the rest of `commands`.
void bind_descriptor_heaps(const vk::raii::CommandBuffer& commands, const DescriptorHeaps& heaps);
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
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
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

}  // namespace

// --- Creating the heaps ------------------------------------------------------

DescriptorHeaps create_descriptor_heaps(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    std::span<const Texture> textures
) {
    const auto properties = gpu.device.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceDescriptorHeapPropertiesEXT
    >();
    const auto& heap = properties.get<vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();
    const auto& limits = properties.get<vk::PhysicalDeviceProperties2>().properties.limits;

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

    // Sampler heap: one sampler at index 0. Trilinear filtering blends between
    // mip levels; anisotropic filtering keeps surfaces seen at a grazing angle,
    // like floors, sharp; repeat wraps texture coordinates outside 0..1.
    heaps.sampler_reserved_offset = align_up(heap.samplerDescriptorSize, heap.samplerDescriptorAlignment);
    heaps.sampler_reserved_size = heap.minSamplerHeapReservedRange;

    std::vector<std::byte> sampler_bytes(
        align_up(heaps.sampler_reserved_offset + heaps.sampler_reserved_size, heap.samplerHeapAlignment));

    const vk::SamplerCreateInfo sampler{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eRepeat,
        .addressModeW = vk::SamplerAddressMode::eRepeat,
        .anisotropyEnable = vk::True,
        .maxAnisotropy = limits.maxSamplerAnisotropy,
        .maxLod = vk::LodClampNone,
    };

    const vk::HostAddressRangeEXT sampler_destination{.address = sampler_bytes.data(), .size = heap.samplerDescriptorSize};

    device.writeSamplerDescriptorsEXT(sampler, sampler_destination);
    heaps.samplers = upload_heap(device, gpu, queue, pool, sampler_bytes, heap.samplerHeapAlignment);

    return heaps;
}

// --- Binding -----------------------------------------------------------------

void bind_descriptor_heaps(const vk::raii::CommandBuffer& commands, const DescriptorHeaps& heaps) {
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

## 5.6 Sampling in the shader: `mesh.slang` and `CMakeLists.txt`

### Why
The fragment shader now looks up its material, reads the base color texture through the descriptor heap, and multiplies the result by the material's factor.

### How
- **Descriptor handles:** `Texture2D.Handle(uint2(index, 0))` makes a *descriptor handle*: a typed reference to descriptor `index` in the bound resource heap. `SamplerState.Handle(uint2(0, 0))` reads sampler 0 from the sampler heap. The handle converts to a `Texture2D` or `SamplerState`, used as normal.
- **Compiling it:** `-capability spvDescriptorHeapEXT` makes slangc compile handles to direct heap access, the `SPV_EXT_descriptor_heap` SPIR-V extension. That needs a recent Slang; this was tested with 2026.19. It also uses untyped pointers, which is why Chapter 0 enabled `shaderUntypedPointers`.
- **Color:** the material index is the same for a whole draw, so every pixel of the draw reads the same descriptor. The sRGB textures are converted to linear when sampled, so the base color is linear, like the factor, and the swapchain's sRGB format encodes the result on the way out.
- **The vertex shader** passes the texture coordinates through.

### Code
`game-engine/shaders/mesh.slang`:
```slang
// Draws one glTF primitive: its vertices come from the scene's vertex buffer,
// its place in the world from its DrawData, its color from its material's
// base color texture in the descriptor heap, lit by a single light.

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

// Data behind a pointer is laid out like C: no padding between members, so
// this matches the C++ Vertex exactly (32 bytes).
struct Vertex {
    float3 position;
    float3 normal;  // (0, 0, 0) when the file had no normals
    float2 uv;
};

struct DrawData {
    float4x4 model;          // this primitive's space -> world space
    float4x4 normal_matrix;  // transposed inverse of model
    uint material;           // index into the materials
    uint3 padding;
};

struct Material {
    float4 base_color_factor;  // linear RGBA, multiplies the texture
    uint base_color_texture;   // resource heap index; 0 is plain white
    uint3 padding;
};

// Written with vkCmdPushDataEXT before each draw.
struct PushData {
    float4x4 view_projection;  // world space -> clip space
    Vertex* vertices;          // the scene's vertices
    DrawData* draws;           // one DrawData per draw
    Material* materials;       // the scene's materials
    uint draw_index;           // which DrawData this draw uses
};

// In a descriptor heap pipeline, the push_constant block is where push data lands.
[[vk::push_constant]]
ConstantBuffer<PushData> push;

// --- Stage interface ---------------------------------------------------------

// What the vertex shader hands to the rasterizer. SV_Position is the
// clip-space position; every other field is interpolated across the triangle.
struct VertexOutput {
    float4 position : SV_Position;
    float3 world_position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
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

    VertexOutput output;
    output.position = mul(push.view_projection, world);
    output.world_position = world.xyz;
    output.normal = mul((float3x3)draw.normal_matrix, vertex.normal);
    output.uv = vertex.uv;
    return output;
}

// --- Fragment shader ---------------------------------------------------------

// Fixed light from above and to the side, until Chapter 7 brings real lighting.
static const float3 light_direction = normalize(float3(0.4, 1.0, 0.3));

// SV_Target: the value written to color attachment 0.
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    const Material material = push.materials[push.draws[push.draw_index].material];

    // Descriptor heap access: a handle made from an index reads that
    // descriptor from the bound heap. The texture's index comes from the
    // material; there's a single sampler, at index 0 of the sampler heap.
    const Texture2D base_color_map = Texture2D.Handle(uint2(material.base_color_texture, 0));
    const SamplerState linear_repeat = SamplerState.Handle(uint2(0, 0));

    // sRGB textures are decoded to linear by the sampler, so this is linear
    // color, like the factor it's multiplied by.
    const float4 base_color = material.base_color_factor * base_color_map.Sample(linear_repeat, input.uv);

    // Without normals in the file, glTF asks for flat shading. The triangle's
    // own normal is the cross product of how the position changes across
    // neighbouring pixels (ddx, ddy).
    float3 normal = input.normal;

    if (all(normal == 0.0)) {
        normal = cross(ddx(input.world_position), ddy(input.world_position));
    }

    normal = normalize(normal);

    // abs(): there's no back-face culling yet, so light both sides alike.
    const float diffuse = abs(dot(normal, light_direction));

    return float4(base_color.rgb * (0.15 + 0.85 * diffuse), 1.0);
}
```

In `game-engine/CMakeLists.txt`, replace the shader's `add_custom_command` and the comment above it with:
```cmake
    # -fvk-use-entrypoint-name: keep vertexMain/fragmentMain as the SPIR-V
    #   entry point names (a file with a single entry point would get "main").
    # -matrix-layout-column-major: read float4x4 data column by column, the way
    #   glm stores it, so mul(M, v) in a shader means M * v in C++.
    # -capability spvDescriptorHeapEXT: compile descriptor handles to direct
    #   descriptor heap access (SPV_EXT_descriptor_heap).
    # -g: keep names and source in the SPIR-V for debuggers like RenderDoc.
    add_custom_command(
        OUTPUT ${spirv}
        COMMAND ${SLANGC} ${source} -target spirv -profile spirv_1_6 -fvk-use-entrypoint-name
                -matrix-layout-column-major -capability spvDescriptorHeapEXT -g -o ${spirv}
        DEPENDS ${source}
        COMMENT "Compiling ${shader_name}.slang"
        VERBATIM
    )
```

## 5.7 Putting it together: `main.cpp`

### Why
`main` creates the textures, heaps and material buffer once at load time, and binds the heaps in every frame.

### How
- **Load order:**
  1. the scene and its geometry buffers, as before,
  2. `create_scene_textures` decodes and uploads every image,
  3. `create_descriptor_heaps` describes them,
  4. a material buffer is built from the scene's materials, where texture index = image + 1, so −1 (no texture) becomes 0, white.
- **Load timing:** printed, so you can see what decoding and uploading cost.
- **`DrawData`** gets each draw's material from its primitive.
- **`record_frame`** binds the heaps right after the pipeline, and push data includes the material buffer's address.

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

#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
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

// What to draw: every primitive draw in a scene, and the buffers they read.
struct DrawList {
    vk::Buffer index_buffer;
    vk::DeviceAddress vertices = 0;
    vk::DeviceAddress draws = 0;
    vk::DeviceAddress materials = 0;
    glm::mat4 view_projection{1.0f};
    std::span<const Primitive> primitives;
    std::span<const MeshDraw> mesh_draws;
};

// Records: swapchain image -> clear color and depth -> draw everything in
// `draws` with `pipeline`, textures from `heaps` -> ready to present.
void record_frame(
    const vk::raii::CommandBuffer& commands,
    const Swapchain& swapchain,
    std::uint32_t image_index,
    std::array<float, 4> color,
    const vk::raii::Pipeline& pipeline,
    const DescriptorHeaps& heaps,
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

    // Reverse-Z: 0 is infinitely far. Depth is only needed while drawing this
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

    // Every texture and sampler the shaders read comes from these two heaps.
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

    // One draw per primitive per node. Push data says which DrawData to use;
    // the primitive's index range and vertex offset go to drawIndexed.
    for (std::uint32_t i = 0; i < draws.mesh_draws.size(); ++i) {
        const Primitive& primitive = draws.primitives[draws.mesh_draws[i].primitive];

        const PushData push{
            .view_projection = draws.view_projection,
            .vertices = draws.vertices,
            .draws = draws.draws,
            .materials = draws.materials,
            .draw_index = i,
        };

        commands.pushDataEXT(vk::PushDataInfoEXT{
            .offset = 0,
            .data = {.address = &push, .size = sizeof(push)},
        });

        commands.drawIndexed(primitive.index_count, 1, primitive.first_index, primitive.vertex_offset, 0);
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

        // --- Textures and materials ------------------------------------------

        // Decode every image, upload them with mipmaps, and describe them in
        // the descriptor heap. Texture 0 is white; scene image i is texture i + 1.
        const std::uint64_t texture_start = SDL_GetTicksNS();
        const std::vector<Texture> textures = create_scene_textures(device, *gpu, queue, command_pool, scene);
        const DescriptorHeaps heaps = create_descriptor_heaps(device, *gpu, queue, command_pool, textures);

        std::println("Textures: {} in {:.0f} ms (whole load {:.0f} ms)", textures.size(),
            static_cast<double>(SDL_GetTicksNS() - texture_start) * 1e-6,
            static_cast<double>(SDL_GetTicksNS() - load_start) * 1e-6);

        std::vector<Material> materials;

        for (const SceneMaterial& material : scene.materials) {
            materials.push_back(Material{
                .base_color_factor = material.base_color_factor,
                .base_color_texture = static_cast<std::uint32_t>(material.base_color_image + 1),
            });
        }

        const Buffer material_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(materials)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        // Spawns at the origin, looking down -Z.
        FlyCamera camera;
        CameraInput input;

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
            previous_ticks = ticks;

            update_camera(camera, input, seconds);

            const float aspect = static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height);

            const DrawList draws{
                .index_buffer = *index_buffer.handle,
                .vertices = vertex_buffer.address,
                .draws = draw_buffer.address,
                .materials = material_buffer.address,
                .view_projection = camera.projection(aspect) * camera.view(),
                .primitives = scene.primitives,
                .mesh_draws = scene.draws,
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
            record_frame(frame.commands, swapchain, image_index, black, pipeline, heaps, draws);

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

## 5.8 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The terminal** shows `Loaded Sponza.gltf: 192496 vertices, 262267 triangles, 103 primitives, 103 draws, 26 materials, 69 images`, followed by a `Textures: 70 in …` line. On the machine this was written on that's about 260 ms in a debug build and 170 ms in release; the whole load is 215 ms in release.
- **The window**, now 1920×1080, shows textured Sponza: green and red curtains with gold patterns, stone columns and floor, marble urns. The floor stays sharp into the distance thanks to anisotropic filtering.
- **The plants** show their leaf textures in dark rectangles. Those pixels should be transparent; alpha masking is the first thing Chapter 6 adds.
- **No `[validation …]` lines**, with the descriptor heap bound and sampled.

**Other models:** every image in `assets/` decodes: 233 images across 45 models, including images embedded in `.glb` files and base64 data URIs. DamagedHelmet's `.glb` and BoxTextured's embedded variant are good checks.

Next, in Chapter 6, materials get the rest of their glTF properties:
- alpha masking (the plants),
- double-sided surfaces and back-face culling,
- per-texture samplers,
- normal maps,
- metallic-roughness values, ready for lighting in Chapter 7.

# Chapter 4: Loading glTF geometry

By the end of this chapter the window shows Sponza: 262,000 triangles in 103 draws, loaded from a glTF file and lit by a single light. No textures yet; those are Chapters 5 and 6. Every model in `lecture-md/game-engine/assets/` either loads its geometry correctly or is refused with a clear reason. Along the way the per-draw data moves out of push data into a buffer, which scales to big scenes and is the layout ray tracing will need later.

This chapter builds on [Chapter 3](03-camera-and-depth.md).

## 4.1 tinygltf: `vendor/tinygltf` and `CMakeLists.txt`

### Why
glTF is the standard format for 3D scenes. A `.gltf` file is JSON describing the scene, its meshes and materials, plus binary data: vertex buffers and images. These can be in separate files, embedded in the JSON as base64 "data URIs", or packed together in a single `.glb`. **tinygltf** parses all of that and loads every buffer and image into memory, whatever form they come in.

### How
- **Which version:** we use **tinygltf 2.9.7**, the classic C++ API that the SIGGRAPH 2026 tutorial, nvpro's renderer and Sascha Willems' samples all use. tinygltf 3 is a rewrite as a C library. It validates indices more strictly, but its 3.0.1 release doesn't load image bytes at all, from external files, data URIs or `.glb` files alike; tested on DamagedHelmet in all three forms.
- **As a library:** tinygltf ships a CMake project that builds `tiny_gltf.cc` as a small static library, so our code never needs its `TINYGLTF_IMPLEMENTATION` define.
- **A CMake policy fix:** tinygltf declares CMake 3.6, which predates policy CMP0077. Under the old behavior, `option()` ignores variables we set before `add_subdirectory`, so our settings were silently overridden (it built its example program and kept its install rules). `CMAKE_POLICY_DEFAULT_CMP0077 NEW` restores the expected behavior.
- **No bundled image decoder:** `TINYGLTF_NO_STB_IMAGE` and `TINYGLTF_NO_STB_IMAGE_WRITE` leave out tinygltf's copy of stb_image. Instead, our loader registers its own image callback that keeps each image's encoded bytes, and Chapter 5 decodes them. The defines are `PUBLIC`, so our code sees the same configuration as the library.
- **Quiet third-party code:** its bundled `json.hpp` trips a deprecation warning in recent clang. `-w` on tinygltf's own compile silences that; our code keeps its full warning set.
- **Finding the models:** `ASSET_DIR` tells the program where the test models are: `lecture-md/game-engine/assets`, next to the project.

### Code
From the repo root:
```bash
git clone --depth 1 --branch v2.9.7 https://github.com/syoyo/tinygltf game-engine/vendor/tinygltf
```

In `game-engine/CMakeLists.txt`, add this section after the `vendor/glm` section:
```cmake
# --- vendor/tinygltf ---------------------------------------------------------
# Parses glTF files. Built as its own small library from tiny_gltf.cc, so our
# code never needs its TINYGLTF_IMPLEMENTATION define. It finds every buffer
# and image itself (separate files, base64 data URIs, inside .glb files).
# The NO_STB defines leave out its bundled image decoder: scene.cpp keeps
# images encoded, and decodes them its own way in Chapter 5.

# tinygltf declares CMake 3.6, which predates option() honouring variables
# set here (policy CMP0077); without this line the settings below are ignored.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(TINYGLTF_HEADER_ONLY OFF)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF)
set(TINYGLTF_INSTALL OFF)
add_subdirectory(vendor/tinygltf SYSTEM)

target_compile_definitions(tinygltf PUBLIC TINYGLTF_NO_STB_IMAGE TINYGLTF_NO_STB_IMAGE_WRITE)

# Its bundled json.hpp trips a deprecation warning in recent compilers. That's
# third-party code we don't edit, so its own compile stays quiet.
target_compile_options(tinygltf PRIVATE -w)
```

Then replace the `target_link_libraries` line with:
```cmake
target_link_libraries(game-engine PRIVATE SDL3::SDL3 Vulkan::Headers glm::glm tinygltf)

# glTF test models live in lecture-md/game-engine/assets, next to this project.
target_compile_definitions(game-engine PRIVATE ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../lecture-md/game-engine/assets")
```

**Build once and restart clangd.** Nothing uses tinygltf yet, so the code from the previous chapter still builds. Run `./game-engine/build.bash` and pick option 1: CMake reconfigures and writes tinygltf's include path into `compile_commands.json`. Then reload the editor window (**Developer: Reload Window**, or **clangd: Restart language server**), so clangd finds `<tiny_gltf.h>` when the code below includes it.

## 4.2 Vertices, draws and push data: `shader_types.h`

### Why
glTF scenes have many meshes, each made of one or more **primitives**: a set of triangles with one material. A node in the scene's tree places a mesh in the world, and one mesh can be placed by many nodes. For drawing we flatten all of that: every primitive's vertices and indices go into one shared vertex buffer and one index buffer, and every (node, primitive) pair becomes one **draw**.

Each draw needs two matrices, and Chapter 3's habit of pushing them would hit the 256-byte push data limit. Materials and lights will need room too. So each draw's matrices go into a **draw-data buffer**, and push data carries just the addresses of the buffers plus which draw this is.

### How
- **`Vertex`** swaps the color for a **normal**, the direction the surface faces, which lighting needs. A normal of (0, 0, 0) marks a primitive whose file had no normals.
- **`DrawData`** holds two matrices per draw, 128 bytes, read through a pointer like the vertices:
  - **the model matrix** places the primitive in the world,
  - **the normal matrix** turns normals the same way. It's the transposed inverse of the model matrix. For rotation and uniform scale that equals the model matrix, but under non-uniform scale (stretching along one axis), transforming normals by the model matrix would tilt them off the surface.
- **`PushData`** is now the view-projection matrix, two pointers and an index: 84 bytes.

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

// Slang lays out data behind a pointer like C: a float3 is 12 bytes, with no
// padding, so position is at byte 0, normal at byte 12, and a vertex is 24.
// A normal of (0, 0, 0) means the file had none (see mesh.slang).
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
};

static_assert(sizeof(Vertex) == 24);
static_assert(offsetof(Vertex, normal) == 12);

// --- Per-draw data -----------------------------------------------------------

// One per draw, in a GPU buffer the shader indexes. Kept out of push data,
// which is limited to 256 bytes on this GPU and needed for other things later.
struct DrawData {
    glm::mat4 model;          // this primitive's space -> world space
    glm::mat4 normal_matrix;  // transposed inverse of model: keeps normals perpendicular under any scale
};

static_assert(sizeof(DrawData) == 128);
static_assert(offsetof(DrawData, normal_matrix) == 64);

// --- Push data ---------------------------------------------------------------

// Written with vkCmdPushDataEXT before each draw. The push block uses std430
// rules: the float4x4 comes first, then 8-byte pointers, then the index.
struct PushData {
    glm::mat4 view_projection;   // world space -> clip space, the same for every draw
    vk::DeviceAddress vertices;  // where the first Vertex is in GPU memory
    vk::DeviceAddress draws;     // where the first DrawData is in GPU memory
    std::uint32_t draw_index;    // which DrawData this draw uses
};

static_assert(offsetof(PushData, vertices) == 64);
static_assert(offsetof(PushData, draws) == 72);
static_assert(offsetof(PushData, draw_index) == 80);
```

## 4.3 The loader: `scene.h` / `scene.cpp`

### Why
tinygltf parses the file and loads its bytes, but glTF stores vertex data in a very general way, and reading it correctly is up to us. The test models deliberately cover the awkward cases:
- vertices interleaved with other data, or spread out with a stride,
- indices of 8, 16 or 32 bits, even within one file,
- quantized positions stored as integers (`KHR_mesh_quantization`),
- **sparse** accessors, which replace a few elements of their base data,
- primitives with no indices, triangle strips and fans, points and lines,
- missing normals,
- files with several scenes.

### How
**Reading data.** glTF reaches vertex data through three layers:
- a **buffer** is raw bytes,
- a **bufferView** is a slice of it, with an optional stride between elements,
- an **accessor** says how to read the slice: the element type, the count, and the starting offset.

The loader turns any accessor into tightly packed elements in four steps:
1. **`copy_elements`** copies elements out of a buffer view at a given stride, after checking they fit inside it.
2. **`accessor_elements`** handles the rest of an accessor's layout. An accessor with no buffer view starts as zeros, and a sparse accessor then gets the elements its index list names replaced. tinygltf's `ByteStride` and component-size helpers supply the sizes.
3. **`read_floats`** converts any component type to float. "Normalized" integers mean a fraction of their range: an unsigned byte of 255 is 1.0, and the most negative signed value clamps to −1, as the glTF spec says. That's what makes quantized files load.
4. **`read_indices`** widens 8- and 16-bit indices to 32 bits, so the whole scene shares one index buffer.

**Primitives.** `add_primitive` reads positions, normals if present (all zeros otherwise) and indices if present (0, 1, 2, … otherwise). Triangle strips and fans are converted to plain triangle lists, keeping every triangle's winding. Points, lines, and primitives without positions are skipped; glTF allows that, and this renderer draws triangles. It also checks every index against the vertex count, because an index past the end would make the GPU read another primitive's vertices.

**Nodes.** `local_transform` builds a node's matrix from either its full matrix or its translation × rotation × scale. Two details:
- glTF stores quaternions as (x, y, z, w), while glm's constructor takes w first.
- glTF uses doubles, and we draw with floats.

`visit_node` walks the tree from the default scene's roots. Each node's world matrix is its parent's times its own, and every primitive of a node's mesh becomes one `MeshDraw`. The world-space bounds of the scene are collected along the way.

**What the loader refuses.** `extensionsRequired` lists the extensions a file can't be read without. Draco and meshopt compression need decoder libraries we don't include, so those files are refused with a message rather than having compressed bytes read as vertices. The other required extensions (texture transforms, lights, unlit materials) only change how things look, and later chapters handle them.

**Images.** tinygltf finds every image's bytes, in its own file, in a data URI, or inside a `.glb`, and passes them to `keep_encoded_image`. That keeps them encoded (PNG, JPEG) for Chapter 5.

### Code
`game-engine/src/includes/scene.h`:
```cpp
#pragma once

#include "includes/shader_types.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <vector>

// One glTF primitive: a run of indices in the scene's index buffer, drawn
// against the vertices starting at `vertex_offset` in the vertex buffer.
struct Primitive {
    std::uint32_t first_index = 0;
    std::uint32_t index_count = 0;
    std::int32_t vertex_offset = 0;
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

    // World-space box around everything drawn, for placing the camera.
    glm::vec3 bounds_min{std::numeric_limits<float>::max()};
    glm::vec3 bounds_max{std::numeric_limits<float>::lowest()};
};

// Loads the default scene of a .gltf or .glb file. Images aren't loaded;
// their file names stay in the file for Chapter 5.
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
std::optional<LoadedPrimitive> add_primitive(const tinygltf::Model &model, const tinygltf::Primitive &source, Scene &scene) {
    const auto position = source.attributes.find("POSITION");
    const bool triangles = source.mode == TINYGLTF_MODE_TRIANGLES
        || source.mode == TINYGLTF_MODE_TRIANGLE_STRIP
        || source.mode == TINYGLTF_MODE_TRIANGLE_FAN;

    if (!triangles || position == source.attributes.end()) {
        return std::nullopt;
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
    });

    for (std::size_t i = 0; i < positions.size(); ++i) {
        scene.vertices.push_back(Vertex{.position = positions[i], .normal = normals[i]});
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
        for (const LoadedPrimitive &primitive : mesh_primitives.at(node.mesh)) {
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

    // Every primitive of every mesh, once. mesh_primitives[m] lists where
    // mesh m's drawable primitives landed in scene.primitives.
    Scene scene;
    std::vector<std::vector<LoadedPrimitive>> mesh_primitives(model.meshes.size());

    for (std::size_t m = 0; m < model.meshes.size(); ++m) {
        for (const tinygltf::Primitive &primitive : model.meshes[m].primitives) {
            if (const auto loaded_primitive = add_primitive(model, primitive, scene)) {
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

## 4.4 The shader: `mesh.slang`

### Why
The shader now looks up its draw's matrices, transforms the normal, and does simple lighting so the geometry's shape is visible.

### How
- **The vertex shader** reads `push.draws[push.draw_index]` for the matrices. It transforms the normal by the normal matrix. `(float3x3)` takes the upper 3×3, since normals are directions and translation doesn't apply to them. It also passes the world position on.
- **`SV_VulkanVertexID` pays off here.** It includes the draw's `vertexOffset`, so every primitive's indices start at 0 and the draw call says where that primitive's vertices begin in the shared buffer.
- **Missing normals:** glTF says a primitive without normals is flat-shaded. In the fragment shader, `ddx` and `ddy` give how the world position changes between neighboring pixels, to the right and downward. Their cross product is the triangle's own normal, with no extra vertex data needed. The order matters: Vulkan's screen Y points down, so `cross(ddy, ddx)` is the one that points toward the camera.
- **Lighting:** one fixed directional light, with 15% ambient so unlit sides aren't black. There's no back-face culling yet (that needs materials' `doubleSided` flag, in Chapter 6), so `abs()` lights both sides alike.

### Code
`game-engine/shaders/mesh.slang`:
```slang
// Draws one glTF primitive: its vertices come from the scene's vertex buffer,
// its place in the world from its DrawData, and it's lit by a single light.

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

// Data behind a pointer is laid out like C: a float3 is 12 bytes and nothing
// is padded, so this matches the C++ Vertex exactly (24 bytes).
struct Vertex {
    float3 position;
    float3 normal;  // (0, 0, 0) when the file had no normals
};

struct DrawData {
    float4x4 model;          // this primitive's space -> world space
    float4x4 normal_matrix;  // transposed inverse of model
};

// Written with vkCmdPushDataEXT before each draw.
struct PushData {
    float4x4 view_projection;  // world space -> clip space
    Vertex *vertices;          // the scene's vertices
    DrawData *draws;           // one DrawData per draw
    uint draw_index;           // which one this draw uses
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
    return output;
}

// --- Fragment shader ---------------------------------------------------------

// Fixed light from above and to the side, until Chapter 7 brings real lighting.
static const float3 light_direction = normalize(float3(0.4, 1.0, 0.3));

// SV_Target: the value written to color attachment 0.
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    // Without normals in the file, glTF asks for flat shading. The triangle's
    // own normal is the cross product of how the position changes across
    // neighbouring pixels (ddx, ddy). Vulkan's screen Y points down, so
    // cross(ddy, ddx) is the order that points toward the camera.
    float3 normal = input.normal;

    if (all(normal == 0.0)) {
        normal = cross(ddy(input.world_position), ddx(input.world_position));
    }

    normal = normalize(normal);

    // abs(): there's no back-face culling yet, so light both sides alike.
    const float diffuse = abs(dot(normal, light_direction));
    const float3 color = float3(0.8) * (0.15 + 0.85 * diffuse);

    return float4(color, 1.0);
}
```

## 4.5 Drawing the scene: `main.cpp`

### Why
`main` loads the scene, uploads its buffers once, and draws every primitive each frame. The cube and its spinning grid are gone.

### How
- **Loading:** `load_gltf` reads the scene file; the path is `ASSET_DIR` plus `scene_file`. Change `scene_file` to view any other model.
- **Draw data:** for each draw, the normal matrix is computed on the CPU as `transpose(inverse(model))`, once at load time rather than per vertex.
- **Three uploads:**
  - vertices and draw data are read through pointers, so they get `eShaderDeviceAddress`,
  - indices go to the GPU's fixed-function index fetch, so that buffer gets `eIndexBuffer`. Index fetch also lets the GPU reuse vertices shared by neighboring triangles.
- **`record_frame`:** binds the index buffer once. Then, for each draw, it pushes the draw's index and calls `drawIndexed` with the primitive's index count, first index and vertex offset.
- **The camera** spawns at the origin, as in Chapter 3. Sponza's origin is in the middle of its atrium, about a meter above the floor.

### Code
`game-engine/src/main.cpp`:
```cpp
#include "includes/buffer.h"
#include "includes/camera.h"
#include "includes/pipeline.h"
#include "includes/scene.h"
#include "includes/sdl.h"
#include "includes/shader_types.h"
#include "includes/swapchain.h"
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

// What to draw: every primitive draw in a scene, and the buffers they read.
struct DrawList {
    vk::Buffer index_buffer;
    vk::DeviceAddress vertices = 0;
    vk::DeviceAddress draws = 0;
    glm::mat4 view_projection{1.0f};
    std::span<const Primitive> primitives;
    std::span<const MeshDraw> mesh_draws;
};

// Records: swapchain image -> clear color and depth -> draw everything in
// `draws` with `pipeline` -> ready to present.
void record_frame(
    const vk::raii::CommandBuffer &commands,
    const Swapchain &swapchain,
    std::uint32_t image_index,
    std::array<float, 4> color,
    const vk::raii::Pipeline &pipeline,
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

    // One index buffer for the whole scene. Indices go through the GPU's
    // fixed-function index fetch, which also lets it reuse vertices shared
    // between neighbouring triangles.
    commands.bindIndexBuffer(draws.index_buffer, 0, vk::IndexType::eUint32);

    // One draw per primitive per node. Push data says which DrawData to use;
    // the primitive's index range and vertex offset go to drawIndexed.
    for (std::uint32_t i = 0; i < draws.mesh_draws.size(); ++i) {
        const Primitive &primitive = draws.primitives[draws.mesh_draws[i].primitive];

        const PushData push{
            .view_projection = draws.view_projection,
            .vertices = draws.vertices,
            .draws = draws.draws,
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
bool poll_events(SDL_Window *window, CameraInput &input) {
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

        const Scene scene = load_gltf(scene_file);
        std::println("Loaded {}: {} vertices, {} triangles, {} primitives, {} draws",
            scene_file.filename().string(), scene.vertices.size(), scene.indices.size() / 3,
            scene.primitives.size(), scene.draws.size());

        // Each draw's matrices. The normal matrix is the transposed inverse of
        // the model matrix: under non-uniform scale, transforming a normal by
        // the model matrix itself would tilt it off the surface.
        std::vector<DrawData> draw_data;

        for (const MeshDraw &draw : scene.draws) {
            draw_data.push_back(DrawData{
                .model = draw.model,
                .normal_matrix = glm::transpose(glm::inverse(draw.model)),
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
                .view_projection = camera.projection(aspect) * camera.view(),
                .primitives = scene.primitives,
                .mesh_draws = scene.draws,
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
    } catch (const std::exception &e) {
        std::println(stderr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

## 4.6 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The terminal** prints `Loaded Sponza.gltf: 192496 vertices, 262267 triangles, 103 primitives, 103 draws`.
- **The window** starts inside Sponza's atrium: columns, arches, curtains, urns and the lion's head, lit from one side. Fly around with the Chapter 3 controls.
- **Plants** look like solid leaf cards; alpha masking comes with materials.
- **No `[validation …]` lines.**

**Other models:** change `scene_file` in `main.cpp`. Most models sit around the origin, so step back with S or the scroll wheel after they load. Across all 54 models in `assets/`:
- 52 load. Each of those is built to test one feature, mostly material features that later chapters render; this chapter only shows their geometry.
- **`Box-Draco`** is refused by our extension check.
- **`MeshoptCubeTest`** is refused by tinygltf itself, which doesn't accept the file's placeholder buffer for compressed data.

The loader checks also caught real cases: quantized positions (Duck-Quantized), sparse vertex data (SimpleSparseAccessor), strips and fans (MeshPrimitiveModes), and negative scale (NegativeScaleTest) all render with the right shape.

Next, in Chapter 5, we decode the images tinygltf already loaded, JPEG and PNG, generate their mipmaps on the GPU, and upload them as textures.

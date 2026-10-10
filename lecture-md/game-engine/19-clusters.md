# Chapter 19: Clusters

By the end of this chapter, every mesh is drawn in **clusters**: groups of up to 128 triangles, each with a bounding sphere and a cone of normals, cut from the mesh when it loads. The cull decides per cluster, not per draw, what to draw: a box seen edge-on costs its facing clusters, not its back. And each cluster exists at many **levels of detail**, built from one another: clusters are grouped, each group simplified into half as many triangles and cut into new clusters, and those grouped and simplified again, up to a handful of triangles for the whole mesh. The cull picks, for every part of every mesh, the one level whose error is too small to see at the camera's distance, so a dragon of a million triangles costs a million close up and a few hundred a kilometre away, and the levels meet without a seam. This is the scheme Unreal's Nanite brought to real time (Karis, Stubbe and Wihlidal, "Nanite: A Deep Dive", SIGGRAPH 2021), after Cignoni et al.'s batched multi-triangulation (2005), as meshoptimizer's `clusterlod` builds it.

Mesh shaders draw the clusters, as they draw the terrain's patches: one workgroup per cluster, reading its vertices through the cluster's table and writing its triangles. The vertex shader, the instance tables and the indirect indexed draws of Chapter 12 go. The cull's two phases become walks down each draw's hierarchy of clusters, every level of the walk one compute dispatch per item with prefix sums across the workgroups, so a million items would take the same steps as a thousand. Whatever the early walk sets aside on the depth pyramid alone goes to the late walk as a list of candidates, draws, nodes or clusters, with nothing kept per cluster between frames.

Two things this chapter found along the way. Under a mesh shader, a fragment shader's triangle index is undefined unless the mesh shader writes it, which the shadow rays' origins depend on. And a simplification that counts only positions keeps a surface's shape to a millimetre while its shading swings, so normals and texture coordinates count in the error too, with borders and seams still locked.

To test the range of it, a scanned dragon of a million triangles stands twelve times around the plateau, from 100 m to 4 km out: Artec3D's "Dragon with pearl" (Sketchfab, CC-BY-4.0).

This chapter builds on [Chapter 18](18-terrain.md).

## 19.1 meshoptimizer: `CMakeLists.txt`, `clusterlod.cpp`

### Why
Cutting a mesh into clusters, grouping them, simplifying a group with its border locked, and building a tree over the groups are each a few hundred lines of careful geometry. meshoptimizer (Kapoulkine) has them all, and its `clusterlod.h` puts them together into the hierarchy.

### How
- **Vendor it:** version 1.2, next to the other libraries:
```bash
git clone --branch v1.2 --depth 1 https://github.com/zeux/meshoptimizer game-engine/vendor/meshoptimizer
```
- **`CMakeLists.txt`** adds it with `add_subdirectory`, its install step off, and links the library. `clusterlod.h` lives in its `demo` directory, which goes on the include path.
- **`clusterlod.cpp`** compiles the header's implementation, and nothing else: the implementation has a `Cluster` of its own, which must not meet ours.
- **Build once** (option 1) and restart clangd, so the new headers resolve.

### Code
In `game-engine/CMakeLists.txt`, add this section before `# --- game-engine -------------------------------------------------------------`:
```cmake
# --- vendor/meshoptimizer ----------------------------------------------------
# Cuts meshes into clusters and builds their level-of-detail hierarchy
# (clusters.cpp). Its library holds the algorithms; the hierarchy builder,
# clusterlod.h, lives in its demo directory as a header, compiled into
# clusterlod.cpp.

set(MESHOPT_INSTALL OFF)
add_subdirectory(vendor/meshoptimizer SYSTEM)
```

In `game-engine/CMakeLists.txt`, replace the line `target_link_libraries(game-engine PRIVATE SDL3::SDL3 Vulkan::Headers glm::glm tinygltf jpgd spng_static)` with:
```cmake
target_include_directories(game-engine SYSTEM PRIVATE vendor/meshoptimizer/demo)
target_link_libraries(game-engine PRIVATE SDL3::SDL3 Vulkan::Headers glm::glm tinygltf jpgd spng_static meshoptimizer)
```

`game-engine/src/clusterlod.cpp`:
```cpp
// meshoptimizer's cluster LOD builder is a header, clusterlod.h, whose implementation one source file compiles by defining CLUSTERLOD_IMPLEMENTATION first. This is that file, and nothing else: the implementation has a Cluster of its own, which must not meet ours (shader_types.h).
#include <meshoptimizer.h>

#define CLUSTERLOD_IMPLEMENTATION
#include <clusterlod.h>
```

## 19.2 Building the hierarchy: `clusters.h`, `clusters.cpp`

### Why
Every primitive's clusters, groups and tree, built once at load and uploaded for the cull and the mesh shader.

### How
- **The build,** `clodBuild`, per primitive, on the primitive's own triangles and vertices:
  1. **Clusters:** the triangles cut into clusters of at most 128 triangles and 128 vertices, spatially compact.
  2. **Groups:** about 16 clusters at a time, chosen to share borders, so that what's simplified together has as little border as possible.
  3. **Simplification:** each group's triangles, as one mesh, collapsed edge by edge into half as many, with the group's border vertices locked, so that neighbouring groups still meet. The error of that simplification, how far the new surface strays from the old, is the group's `error`, and errors only grow up the hierarchy: a group's error is at least the errors of the groups it was built from.
  4. **Again:** the simplified triangles cut into new clusters, which record which group refined them; those grouped and simplified; until a group can't be simplified further, whose error is set to infinity: the coarsest level.
- **Strict collapse.** meshoptimizer can also collapse across attribute seams (its permissive mode) or ignore topology altogether (sloppy simplification), which reach further on seam-heavy meshes at a cost in fidelity. Both are off: the chain stops where the asset's topology stops.
- **Attributes in the error.** A collapse's error counts the normal and the texture coordinates it moves, not only the position. Weights put them in the position's units as a share of the primitive's size: a normal turned right around, or texture coordinates moved across the whole texture, count like a vertex moved a quarter of the way across the primitive. The attribute error is clamped to the position error's scale, `clusterlod`'s default, so it can't make a level conservative out of all proportion. Without this, the box in front of the camera was drawn simplified at 4 m, its shape within a pixel but its shading visibly flattened.
- **The tree:** `clodBuildHierarchy` builds a tree of spheres over each level's groups, 8 children a node, every node carrying the worst error under it, and one root per level. The cull walks it from the roots and prunes whatever is too fine to matter.
- **What's uploaded:** `Cluster` records with the sphere, the normal cone (`meshopt_computeClusterBounds`), the cluster's place in the vertex and triangle tables, its group and the group that refines it; `ClusterGroup` records with the simplified sphere and error, the DAG depth and the group's clusters; `ClusterNode` records; `PrimitiveClusters` saying where each primitive's start. The vertex tables index the scene's vertices, so the vertex buffer, the index buffer and the acceleration structures are untouched: simplification removes triangles and never adds vertices.
- **The print:** per primitive, how far the chain goes. The box reaches 289 triangles from 6,268 over 6 levels; the dragon's parts reach a few dozen from a hundred thousand over 11 to 13.

### Code
`game-engine/src/includes/clusters.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/scene.h"
#include "includes/shader_types.h"
#include "includes/vulkan_setup.h"

#include <cstdint>
#include <vector>

// Clusters

// Every primitive, cut into clusters of up to 128 triangles and simplified level by level into a hierarchy (meshoptimizer's clusterlod, after Nanite):
//   - Clusters are grouped, about 16 at a time, and each group is simplified as a whole, with its border locked, into half as many triangles, which are cut into new clusters; those are grouped and simplified again, until a group can't be simplified further. Each group records the error of its simplification: how far, at most, the simplified surface strays from the one it replaced. Errors only grow up the hierarchy.
//   - A cluster is drawn when its group's error is too large to accept at the camera's distance, while the error of the group that refines it is small enough: the one level of detail that's coarse enough and no coarser. Neighbouring clusters chosen that way always meet, since a group's border only changes when the whole group is replaced.
//   - Over the groups of each level, a tree of spheres, 8 children a node, each with the worst error under it: the cull walks it from the roots and prunes whatever is too fine to matter.
// All of it is built at load, once per primitive, in the primitive's own space; the vertices are the scene's, untouched. Simplification removes triangles and never adds vertices, so the vertex buffer, the index buffer and the acceleration structures stay as they were.
struct Clusters {
    Buffer clusters;           // one Cluster per cluster, primitive after primitive
    Buffer vertices;           // the clusters' vertex tables: indices into the scene's vertices
    Buffer triangles;          // the clusters' triangles: three bytes each
    Buffer groups;             // one ClusterGroup per group
    Buffer nodes;              // one ClusterNode per node of the hierarchies
    Buffer primitives;         // one PrimitiveClusters per primitive
    std::uint32_t cluster_count = 0;
    std::uint32_t group_count = 0;
    std::uint32_t node_count = 0;
    std::uint32_t levels = 0;  // the deepest hierarchy's level count
    std::vector<std::uint32_t> level_triangles;  // per DAG level, the triangles of its clusters, summed over the primitives
};

// Triangles and vertices per cluster at most, and the mesh shader's threads: one per triangle or vertex.
constexpr std::uint32_t cluster_max_triangles = 128;
constexpr std::uint32_t cluster_max_vertices = 128;

// Nodes of the hierarchy have up to this many children.
constexpr std::uint32_t cluster_node_width = 8;

// Builds every primitive's clusters and hierarchy from the scene's triangles, and uploads them.
Clusters build_clusters(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    const Scene &scene
);
```

`game-engine/src/clusters.cpp`:
```cpp
#include "includes/clusters.h"

#include <clusterlod.h>
#include <meshoptimizer.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstddef>
#include <limits>
#include <print>
#include <span>
#include <stdexcept>

namespace {

    constexpr std::uint32_t none = std::numeric_limits<std::uint32_t>::max();

    // What one primitive's build collects: clusterlod hands over each group with its clusters as it finishes them, finest level first.
    struct PrimitiveBuild {
        std::vector<Cluster> clusters;
        std::vector<ClusterGroup> groups;
        std::vector<std::uint32_t> vertices;
        std::vector<std::uint8_t> triangles;
        std::vector<clodGroup> clod_groups;  // for the hierarchy
    };

    // Cuts one primitive into clusters and its hierarchy, appending to `build`. The primitive's triangles index its own vertices, from 0; the cluster tables index the scene's, so its vertex offset is added.
    void build_primitive(const Scene &scene, const Primitive &primitive, PrimitiveBuild &build) {
        const std::span<const std::uint32_t> indices(scene.indices.data() + primitive.first_index, primitive.index_count);
        const auto vertex_count = static_cast<std::size_t>(scene.vertices.size()) - static_cast<std::size_t>(primitive.vertex_offset);

        // Simplification by edge collapse alone, borders and seams locked: the chain stops where the asset's topology stops. meshoptimizer offers permissive and sloppy fallbacks that reach further at a cost in fidelity; they're off.
        clodConfig config = clodDefaultConfig(cluster_max_triangles);
        config.max_vertices = cluster_max_vertices;
        config.simplify_permissive = false;
        config.simplify_fallback_permissive = false;
        config.simplify_fallback_sloppy = false;

        // The error a collapse costs counts the normal and the texture coordinates it moves, not only the position: a surface can keep its shape to a millimetre while its shading swings. The weights put attribute changes in the position's units, metres or whatever the file uses, as a share of the primitive's size: a normal turned right around, or texture coordinates moved across the whole texture, count like a vertex moved a quarter of the way across the primitive. Position, normal, tangent and the first texture coordinates follow one another in a Vertex, so they're one run of floats; the tangent's weights are 0.
        const Vertex *vertices = scene.vertices.data() + primitive.vertex_offset;
        glm::vec3 lo{std::numeric_limits<float>::max()};
        glm::vec3 hi{std::numeric_limits<float>::lowest()};

        for (std::size_t i = 0; i < vertex_count; ++i) {
            lo = glm::min(lo, vertices[i].position);
            hi = glm::max(hi, vertices[i].position);
        }

        const float size = glm::length(hi - lo);
        const float normal_weight = size * 0.125f;  // a normal's length is 1; turned right around it changes by 2
        const float uv_weight = size * 0.25f;
        const std::array<float, 9> weights{normal_weight, normal_weight, normal_weight, 0.0f, 0.0f, 0.0f, 0.0f, uv_weight, uv_weight};

        const clodMesh mesh{
            .indices = indices.data(),
            .index_count = indices.size(),
            .vertex_count = vertex_count,
            .vertex_positions = &vertices[0].position.x,
            .vertex_positions_stride = sizeof(Vertex),
            .vertex_attributes = &vertices[0].normal.x,
            .vertex_attributes_stride = sizeof(Vertex),
            .vertex_lock = nullptr,
            .attribute_weights = weights.data(),
            .attribute_count = weights.size(),
            .attribute_protect_mask = 0,
        };

        clodBuild(config, mesh, [&](clodGroup group, const clodCluster *clusters, std::size_t cluster_count) -> int {
            const auto group_index = static_cast<std::uint32_t>(build.groups.size());

            build.groups.push_back(ClusterGroup{
                .center = {group.simplified.center[0], group.simplified.center[1], group.simplified.center[2]},
                .radius = group.simplified.radius,
                .error = group.simplified.error,
                .depth = static_cast<std::uint32_t>(group.depth),
                .first_cluster = static_cast<std::uint32_t>(build.clusters.size()),
                .cluster_count = static_cast<std::uint32_t>(cluster_count),
            });
            build.clod_groups.push_back(group);

            for (std::size_t i = 0; i < cluster_count; ++i) {
                const clodCluster &cluster = clusters[i];

                // The cluster's vertex table and local triangles, and its bounds: the sphere and the normal cone.
                std::vector<std::uint32_t> local_vertices(cluster.vertex_count);
                std::vector<std::uint8_t> local_triangles(cluster.index_count);
                clodLocalIndices(local_vertices.data(), local_triangles.data(), cluster.indices, cluster.index_count);

                const meshopt_Bounds bounds = meshopt_computeClusterBounds(cluster.indices, cluster.index_count,
                    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride);

                build.clusters.push_back(Cluster{
                    .center = {bounds.center[0], bounds.center[1], bounds.center[2]},
                    .radius = bounds.radius,
                    .cone_axis = {bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2]},
                    .cone_cutoff = bounds.cone_cutoff,
                    .first_vertex = static_cast<std::uint32_t>(build.vertices.size()),
                    .first_triangle = static_cast<std::uint32_t>(build.triangles.size() / 3),
                    .vertex_count = static_cast<std::uint32_t>(cluster.vertex_count),
                    .triangle_count = static_cast<std::uint32_t>(cluster.index_count / 3),
                    .group = group_index,
                    .refined = cluster.refined < 0 ? none : static_cast<std::uint32_t>(cluster.refined),
                });

                for (const std::uint32_t vertex : local_vertices) {
                    build.vertices.push_back(vertex + static_cast<std::uint32_t>(primitive.vertex_offset));
                }

                build.triangles.insert(build.triangles.end(), local_triangles.begin(), local_triangles.end());
            }

            return static_cast<int>(group_index);
        });
    }

}  // namespace

Clusters build_clusters(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    const Scene &scene
) {
    Clusters result;
    std::vector<Cluster> clusters;
    std::vector<ClusterGroup> groups;
    std::vector<std::uint32_t> vertices;
    std::vector<std::uint8_t> triangles;
    std::vector<ClusterNode> nodes;
    std::vector<PrimitiveClusters> primitives;

    for (const Primitive &primitive : scene.primitives) {
        PrimitiveBuild build;
        build_primitive(scene, primitive, build);

        // Groups come out finest level first, so the last one's depth is the deepest.
        const auto levels = static_cast<std::uint32_t>(build.clod_groups.back().depth + 1);
        result.levels = std::max(result.levels, levels);

        // The hierarchy: a tree per level, whose roots come first.
        std::vector<clodNode> tree(clodBuildHierarchyBound(build.clod_groups.size(), cluster_node_width, levels));
        tree.resize(clodBuildHierarchy(tree.data(), build.clod_groups.data(), build.clod_groups.size(), cluster_node_width, levels));

        // Everything moves into the scene-wide buffers, its indices moved up by what's there.
        const auto first_cluster = static_cast<std::uint32_t>(clusters.size());
        const auto first_group = static_cast<std::uint32_t>(groups.size());
        const auto first_node = static_cast<std::uint32_t>(nodes.size());
        const auto first_vertex = static_cast<std::uint32_t>(vertices.size());
        const auto first_triangle = static_cast<std::uint32_t>(triangles.size() / 3);

        primitives.push_back(PrimitiveClusters{
            .first_cluster = first_cluster,
            .first_group = first_group,
            .first_node = first_node,
            .levels = levels,
        });

        for (Cluster cluster : build.clusters) {
            cluster.first_vertex += first_vertex;
            cluster.first_triangle += first_triangle;
            cluster.group += first_group;
            cluster.refined += cluster.refined == none ? 0 : first_group;
            clusters.push_back(cluster);
        }

        for (ClusterGroup group : build.groups) {
            group.first_cluster += first_cluster;
            groups.push_back(group);
        }

        for (const clodNode &node : tree) {
            nodes.push_back(ClusterNode{
                .center = {node.bounds.center[0], node.bounds.center[1], node.bounds.center[2]},
                .radius = node.bounds.radius,
                .error = node.bounds.error,
                .group = node.group < 0 ? none : first_group + static_cast<std::uint32_t>(node.group),
                .first_child = first_node + node.child_offset,
                .child_count = node.child_count,
            });
        }

        vertices.insert(vertices.end(), build.vertices.begin(), build.vertices.end());
        triangles.insert(triangles.end(), build.triangles.begin(), build.triangles.end());

        if (result.level_triangles.size() < levels) {
            result.level_triangles.resize(levels);
        }

        std::uint32_t finest = 0;
        std::uint32_t coarsest = 0;

        for (const Cluster &cluster : build.clusters) {
            const std::uint32_t depth = build.groups[cluster.group].depth;
            result.level_triangles[depth] += cluster.triangle_count;
            finest += depth == 0 ? cluster.triangle_count : 0;
            coarsest += depth + 1 == levels ? cluster.triangle_count : 0;
        }

        // How far simplification took the primitive: with borders and seams locked, a mesh of many seams stops early.
        std::println("  primitive {}: {} clusters, {} levels, {} triangles at the finest, {} at the coarsest", primitives.size() - 1,
            build.clusters.size(), levels, finest, coarsest);
    }

    // Three bytes a triangle, four to a word: pad to whole words.
    triangles.resize((triangles.size() + 3) & ~std::size_t{3}, 0);

    result.cluster_count = static_cast<std::uint32_t>(clusters.size());
    result.group_count = static_cast<std::uint32_t>(groups.size());
    result.node_count = static_cast<std::uint32_t>(nodes.size());

    const auto upload = [&](std::span<const std::byte> bytes) {
        return upload_buffer(device, gpu, queue, pool, bytes, vk::BufferUsageFlagBits::eShaderDeviceAddress);
    };
    result.clusters = upload(std::as_bytes(std::span(clusters)));
    result.vertices = upload(std::as_bytes(std::span(vertices)));
    result.triangles = upload(std::as_bytes(std::span(triangles)));
    result.groups = upload(std::as_bytes(std::span(groups)));
    result.nodes = upload(std::as_bytes(std::span(nodes)));
    result.primitives = upload(std::as_bytes(std::span(primitives)));

    return result;
}
```

## 19.3 The data: `shader_types.h`, `shared.slangh`

### Why
The shaders read the hierarchy, the cull's walk needs its lists and counters, and the push data names what a dispatch draws.

### How
- **`Cluster`** (56 bytes), **`ClusterGroup`** (32), **`ClusterNode`** (32), **`PrimitiveClusters`** (16): the hierarchy, as 19.2 builds it.
- **`ClusterItem`** (8): what the walk handles: a draw with one of its nodes or clusters, the kind in the top two bits of `what`.
- **`ClusterDispatch`** (32): a draw list's mesh dispatch: the three workgroup counts first, so the GPU can read it as a `VkDrawMeshTasksIndirectCommandEXT`, then the list's cluster count, where its clusters start, and the items. Vulkan guarantees only 65,535 mesh workgroups per dimension, `maxMeshWorkGroupCount`, so `x` is capped there and `y` takes the rest; their product must stay under `maxMeshWorkGroupTotalCount`, at least 4,194,304, which the item capacity keeps it well under.
- **`WalkCounters`** (148): the walk's counts, read by the steps and, at `dispatch_x`, by the GPU as the next step's workgroup count; clusters per draw list and per DAG level, and the triangles, for the title and the terminal.
- **`DrawData`** gains `primitive`, 180 bytes; **`FrameData`** the hierarchy's six addresses, `cluster_error_scale` and `subpixel_scale`, 432 bytes; **`PushData`** names what a dispatch draws, `drawn`, a `ClusterDispatch` or the terrain's patches; **`CullTables`** is rebuilt around the walk, 144 bytes; **`CullPushData`** gains the level and the phase, 48 bytes. The groups, commands and instance tables of Chapter 12 go.

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
    std::uint32_t primitive;    // index into the primitives' cluster hierarchies (PrimitiveClusters)
};

static_assert(sizeof(DrawData) == 180);
static_assert(offsetof(DrawData, normal_matrix) == 64);
static_assert(offsetof(DrawData, material) == 128);
static_assert(offsetof(DrawData, cell) == 140);
static_assert(offsetof(DrawData, bounds_min) == 152);
static_assert(offsetof(DrawData, primitive) == 176);

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
    vk::DeviceAddress clusters;         // every primitive's clusters (clusters.h)
    vk::DeviceAddress cluster_vertices; // the clusters' vertex tables: indices into the vertices
    vk::DeviceAddress cluster_triangles;// the clusters' triangles: three bytes each, local to the cluster
    vk::DeviceAddress cluster_groups;   // the clusters' groups
    vk::DeviceAddress cluster_nodes;    // the hierarchy over the groups
    vk::DeviceAddress primitives;       // per primitive, where its clusters, groups and nodes start
    float cluster_error_scale;          // an error of e metres at distance d is over the pixel threshold when e x this > d
    float subpixel_scale;               // a sphere of radius r at distance d is under a pixel across when r x this < d
};

static_assert(sizeof(FrameData) == 432);
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
static_assert(offsetof(FrameData, clusters) == 376);
static_assert(offsetof(FrameData, cluster_error_scale) == 424);

// Push data

// Written with vkCmdPushDataEXT before each pipeline's dispatches: where this frame's data is, and what the dispatch draws: a draw list's ClusterDispatch, or the terrain's patches.
struct PushData {
    vk::DeviceAddress frame;
    vk::DeviceAddress drawn;
};

static_assert(sizeof(PushData) == 16);

// Clusters (clusters.h, cull.slang, mesh.slang)

// A cluster: up to 128 triangles of one primitive, with a bounding sphere and a normal cone for culling, and its place in the level-of-detail hierarchy: its group, whose simplified error says whether the cluster is coarse enough to draw, and the group that refines it, whose error says whether a finer one would do. Positions are in the primitive's space.
struct Cluster {
    glm::vec3 center;
    float radius;
    glm::vec3 cone_axis;            // the normal cone's axis
    float cone_cutoff;              // meshoptimizer's cutoff: the cluster faces away when the direction from the camera to its centre makes an angle with the axis whose cosine is at least this, plus the sphere's share; 1 when the cone is too wide to say
    std::uint32_t first_vertex;     // into cluster_vertices
    std::uint32_t first_triangle;   // into cluster_triangles, in triangles
    std::uint32_t vertex_count;
    std::uint32_t triangle_count;
    std::uint32_t group;            // index into the groups
    std::uint32_t refined;          // the group that refines this cluster, or none (~0)
};

static_assert(sizeof(Cluster) == 56);
static_assert(offsetof(Cluster, cone_axis) == 16);
static_assert(offsetof(Cluster, first_vertex) == 32);

// A group of clusters simplified together: the sphere and error of what they were simplified into, which every cluster they refine shares. The error is infinite for a group that was never simplified further: the coarsest level.
struct ClusterGroup {
    glm::vec3 center;
    float radius;
    float error;                    // metres, in the primitive's space
    std::uint32_t depth;            // the DAG level: 0 is the finest
    std::uint32_t first_cluster;    // the group's clusters, in the clusters
    std::uint32_t cluster_count;
};

static_assert(sizeof(ClusterGroup) == 32);

// A node of the hierarchy over the groups: a bounding sphere, the worst error under it, and either a group (a leaf) or its children.
struct ClusterNode {
    glm::vec3 center;
    float radius;
    float error;
    std::uint32_t group;            // the leaf's group, or none (~0) for a node with children
    std::uint32_t first_child;      // into the nodes
    std::uint32_t child_count;
};

static_assert(sizeof(ClusterNode) == 32);

// Where a primitive's clusters, groups and nodes are: the hierarchy's roots are its first `levels` nodes, one per DAG level.
struct PrimitiveClusters {
    std::uint32_t first_cluster;
    std::uint32_t first_group;
    std::uint32_t first_node;
    std::uint32_t levels;
};

static_assert(sizeof(PrimitiveClusters) == 16);

// What the cull walks and what it draws: a draw with one of its hierarchy's nodes or clusters. `what` holds the kind in its top two bits and the index in the rest: a draw (the index unused), a node, or a cluster.
struct ClusterItem {
    std::uint32_t draw;
    std::uint32_t what;
};

static_assert(sizeof(ClusterItem) == 8);

constexpr std::uint32_t item_kind_shift = 30;
constexpr std::uint32_t item_kind_draw = 0;
constexpr std::uint32_t item_kind_node = 1;
constexpr std::uint32_t item_kind_cluster = 2;

// One draw list's mesh dispatch: VkDrawMeshTasksIndirectCommandEXT's three workgroup counts first, then where the list's clusters start among the phase's cluster items, and the items themselves. Vulkan guarantees only 65,535 mesh workgroups per dimension (maxMeshWorkGroupCount), so x is capped there and y takes the rest; their product must stay under maxMeshWorkGroupTotalCount, at least 4,194,304, which the item capacity keeps it well under.
struct ClusterDispatch {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    std::uint32_t count;            // the list's clusters: what x times y launch, less the wrap's
    std::uint32_t start;
    std::uint32_t pad;              // keeps `items` on an 8-byte boundary
    vk::DeviceAddress items;
};

static_assert(sizeof(ClusterDispatch) == 32);
static_assert(offsetof(ClusterDispatch, items) == 24);

constexpr std::uint32_t max_dispatch_width = 65535;

// The cull's counters, which the walk's steps read and write, and the dispatch of the next step. Per phase.
struct WalkCounters {
    std::uint32_t items;            // items in the level being walked
    std::uint32_t next_items;       // items the level produced for the next
    std::uint32_t candidates;       // items set aside for the late phase, so far
    std::uint32_t clusters;         // cluster items drawn, so far
    std::uint32_t triangles;        // their triangles
    std::uint32_t subpixel_draws;   // draws whose sphere is under a pixel across
    std::uint32_t dispatch_x;       // VkDispatchIndirectCommand for the next level's steps: workgroups of walk_workgroup_size
    std::uint32_t dispatch_y;
    std::uint32_t dispatch_z;
    std::uint32_t list_counts[12];  // cluster items per draw list
    std::uint32_t level_clusters[16];  // cluster items per DAG level
};

static_assert(sizeof(WalkCounters) == 148);
static_assert(offsetof(WalkCounters, dispatch_x) == 24);
static_assert(offsetof(WalkCounters, list_counts) == 36);

// GPU culling (culling.h, cull.slang)

// Where all of one cull phase's buffers are, in one table its steps read. The two phases share the order and the terrain's flags, and have the rest each.
struct CullTables {
    vk::DeviceAddress order;          // draw indices, by list, then primitive: the early phase's first items
    vk::DeviceAddress draw_items;     // one ClusterItem per draw, in the order
    vk::DeviceAddress items[2];       // the walk's levels, alternating
    vk::DeviceAddress results;        // per item of the level: what it produced (three counts)
    vk::DeviceAddress block_totals;   // per workgroup of the level: the sums of those
    vk::DeviceAddress block_bases;    // prefix sums of block_totals
    vk::DeviceAddress counters;       // the WalkCounters
    vk::DeviceAddress candidates;     // items set aside for the late phase
    vk::DeviceAddress cluster_items;  // the clusters to draw, by draw list
    vk::DeviceAddress dispatches;     // one ClusterDispatch per draw list
    vk::DeviceAddress early_candidates;  // the early phase's candidates: the late phase's first items
    vk::DeviceAddress early_counters;    // the early phase's counters: how many
    vk::DeviceAddress patches;        // the terrain patches this phase draws: quadtree node indices (terrain.h)
    vk::DeviceAddress patch_command;  // one VkDrawMeshTasksIndirectCommandEXT: how many patches, 1, 1
    vk::DeviceAddress nodes;          // the terrain selection's working lists: two runs of max_terrain_patches
    vk::DeviceAddress early_patches;  // per terrain patch: the frame the early phase last drew it in
    std::uint32_t draw_count;
    std::uint32_t item_capacity;      // the most items a level, the candidates or the cluster items can hold
};

static_assert(sizeof(CullTables) == 144);
static_assert(offsetof(CullTables, items) == 16);
static_assert(offsetof(CullTables, draw_count) == 136);

// The cull steps' push data (cull.slang). The walk's steps read the frame, their tables, the level being walked and the phase; the depth pyramid steps read which levels to copy or reduce, and their sizes. Push data follows std430 rules, where a uvec2 starts on an 8-byte boundary: the two pointers and four numbers fill the first 32 bytes, so source_size lands on one.
struct CullPushData {
    vk::DeviceAddress frame = 0;   // this frame's FrameData
    vk::DeviceAddress tables = 0;  // the phase's CullTables
    std::uint32_t source = 0;      // resource heap slot: the depth buffer (sampled), or the level above (storage)
    std::uint32_t target = 0;      // resource heap slot: the pyramid level written (storage)
    std::uint32_t level = 0;       // the walk: which level of items is being walked
    std::uint32_t late = 0;        // the walk: 1 in the late phase
    glm::uvec2 source_size{0};     // in texels
    glm::uvec2 target_size{0};
};

static_assert(sizeof(CullPushData) == 48);
static_assert(offsetof(CullPushData, source_size) == 32);

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
    uint primitive;          // index into the primitives' cluster hierarchies
};

// Clusters (src/includes/clusters.h)

// Up to 128 triangles of one primitive, with a bounding sphere and a normal cone, and its place in the level-of-detail hierarchy. Positions are in the primitive's space.
struct Cluster {
    float3 center;
    float radius;
    float3 cone_axis;       // the normal cone's axis
    float cone_cutoff;      // meshoptimizer's cutoff for facing away (cull.slang); 1 when the cone is too wide to say
    uint first_vertex;      // into cluster_vertices
    uint first_triangle;    // into cluster_triangles, in triangles
    uint vertex_count;
    uint triangle_count;
    uint group;             // its group, whose error says whether it's coarse enough to draw
    uint refined;           // the group that refines it, or none (~0)
};

// A group of clusters simplified together: the sphere and error of what they became. Infinite error for the coarsest level.
struct ClusterGroup {
    float3 center;
    float radius;
    float error;            // metres, in the primitive's space
    uint depth;             // the DAG level: 0 is the finest
    uint first_cluster;
    uint cluster_count;
};

// A node of the hierarchy over the groups: a leaf's group, or children.
struct ClusterNode {
    float3 center;
    float radius;
    float error;            // the worst error under it
    uint group;             // the leaf's group, or none (~0)
    uint first_child;
    uint child_count;
};

// Where a primitive's clusters, groups and nodes are; the hierarchy's roots are its first `levels` nodes.
struct PrimitiveClusters {
    uint first_cluster;
    uint first_group;
    uint first_node;
    uint levels;
};

// What the cull walks and draws: a draw with one of its nodes or clusters. `what` holds the kind in its top two bits.
struct ClusterItem {
    uint draw;
    uint what;
};

static const uint item_kind_shift = 30;
static const uint item_kind_draw = 0;
static const uint item_kind_node = 1;
static const uint item_kind_cluster = 2;

uint item_kind(ClusterItem item) {
    return item.what >> item_kind_shift;
}

uint item_index(ClusterItem item) {
    return item.what & ((1u << item_kind_shift) - 1);
}

ClusterItem make_item(uint draw, uint kind, uint index) {
    return ClusterItem(draw, (kind << item_kind_shift) | index);
}

// One draw list's mesh dispatch: the workgroup counts, then where its clusters start among the items, and the items. Vulkan guarantees 65,535 workgroups per dimension; y takes the rest.
struct ClusterDispatch {
    uint x;
    uint y;
    uint z;
    uint count;             // the list's clusters: the workgroups x times y launch, less the wrap's
    uint start;
    uint pad;
    ClusterItem *items;
};

static const uint max_dispatch_width = 65535;

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
    Cluster *clusters;                 // every primitive's clusters
    uint *cluster_vertices;            // the clusters' vertex tables: indices into the vertices
    uint *cluster_triangles;           // the clusters' triangles, three bytes each, four bytes to a word
    ClusterGroup *cluster_groups;      // the clusters' groups
    ClusterNode *cluster_nodes;        // the hierarchy over the groups
    PrimitiveClusters *primitives;     // per primitive, where its clusters, groups and nodes start
    float cluster_error_scale;         // an error of e metres at distance d is over the threshold when e x this > d
    float subpixel_scale;              // a sphere of radius r at distance d is under a pixel when r x this < d
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

// Written with vkCmdPushDataEXT before each pipeline's dispatches: the frame, and what the dispatch draws: a draw list's ClusterDispatch, or the terrain's patches.
struct PushData {
    FrameData *frame;
    uint64_t drawn;
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

## 19.4 The walk: `cull.slang`, `culling.h`, `culling.cpp`

### Why
Each phase has to turn the draws into the clusters to draw, at the right level of detail, in a fixed order, however many there are.

### How
- **Items, level by level.** The early phase's first level is every draw, in the cull's order; the late phase's, the early phase's candidates. Each level runs four steps:
  1. **`walkStartMain`,** one thread: how many items the level has, and the workgroups the next two steps take, written where the GPU reads them as indirect dispatch arguments.
  2. **`walkMain`,** a thread per item: what the item produces. A **draw** outside the view produces nothing; one hidden by the pyramid is a candidate; otherwise its hierarchy's roots, one per level, go on. A **node** is pruned when even its worst error projects under the threshold, or it's out of view; hidden, it's a candidate; otherwise its children go on, or, for a leaf, its group's clusters. A **cluster** is drawn when its own group's error projects over the threshold, so the coarser level won't do, and the group that refines it projects under, so the finer isn't needed: the DAG cut, which the monotone errors make consistent across every border. It must also be in view, not facing away, and not hidden, which makes it a candidate instead. Facing away is meshoptimizer's normal-cone test, in the primitive's own space, where the camera goes by the inverse model matrix: exact under any transform, mirroring included, and widened by the sphere's radius for a camera near the cluster, so that no cluster with a triangle facing the camera is ever dropped. The step keeps three counts per item, and each workgroup's totals of them.
  3. **`walkScanMain`,** one workgroup: prefix sums of the workgroups' totals, for each of the three outputs, and the level's results: how many items the next level has. Candidates and clusters accumulate across levels.
  4. **`walkWriteMain`,** a thread per item: the item's output, where the prefix sums put it: the next level's items, the candidates, or the cluster items, counted per draw list and per DAG level with their triangles.
  Eight levels are enough for any hierarchy here: a draw, its tree's depth, a group's clusters, the clusters.
- **The error test:** a group's error of `e` metres, scaled by the draw's largest scale, on a sphere whose nearest point is `d` away, projects to `e × focal / d` pixels; `cluster_error_scale` is `focal / cluster_error_pixels`, so the test is `e × scale > d`, with `d` never under the near plane.
- **The order** survives: every level's items are visited in order, and the prefix sums place their outputs in that order, so the cluster items come out in the draws' order, which is the draw lists' order. **`writeDispatchesMain`** then cuts the cluster items into the lists' runs by the per-list counts, and writes each list's `ClusterDispatch`.
- **The late phase** walks the candidates against the new pyramid, with nothing per cluster kept between frames: a candidate is wherever the early walk stopped, a draw, a node or a cluster, and the late walk continues from exactly there.
- **`culling.h`, `culling.cpp`:** each phase's buffers for the walk, a capacity of half a million items each, the counters read as indirect arguments, the dispatches read as indirect draw arguments, and `record_culling` recording the levels' steps with a barrier between each, then the dispatches and the terrain's patches. `draw_list` is one `vkCmdDrawMeshTasksIndirectEXT` reading the list's dispatch; the readback copies each phase's counters whole.

### Code
`game-engine/shaders/cull.slang`:
```slang
// GPU culling: which clusters to draw, from every draw's level-of-detail hierarchy (clusters.h), turned into one mesh dispatch per draw list. Two phases (culling.h, record_culling), each a walk down the hierarchies, level by level, in three steps per level:
//   walkStartMain   one thread: how many items the level has, and the workgroups its steps take
//   walkMain        per item: what it produces, tested against the view, the depth pyramid and the error threshold
//   walkScanMain    one workgroup: where each workgroup's output goes, and the next level's size
//   walkWriteMain   per item: its output, into its place
// An item is a draw, a node of its hierarchy, or a cluster. The early phase starts from every draw, in the cull's order; whatever it sets aside on the depth pyramid alone goes to the late phase, which starts from those. Every write goes to a place the prefix sums fix: the result doesn't depend on which thread runs first, and the clusters come out in the draws' order, which is the draw lists' order. Between the phases, two steps build the depth pyramid (record_depth_pyramid):
//   copyDepthMain       level 0: the depth buffer, copied
//   reduceDepthMain     each level below: the farthest depth of the texels above it
// Each phase also picks the terrain's patches (selectEarlyPatchesMain, selectLatePatchesMain).

#include "shared.slangh"
#include "scan.slangh"
#include "terrain.slangh"

// Data shared with C++ (src/includes/shader_types.h)

// The cull's counters, which the walk's steps read and write, and the dispatch of the next step.
struct WalkCounters {
    uint items;             // items in the level being walked
    uint next_items;        // items the level produced for the next
    uint candidates;        // items set aside for the late phase, so far
    uint clusters;          // cluster items drawn, so far
    uint triangles;         // their triangles
    uint subpixel_draws;    // draws whose sphere is under a pixel across
    uint dispatch_x;        // the next level's steps' workgroups
    uint dispatch_y;
    uint dispatch_z;
    uint list_counts[12];   // cluster items per draw list
    uint level_clusters[16];  // cluster items per DAG level
};

struct CullTables {
    uint *order;                 // draw indices, in the cull's order
    ClusterItem *draw_items;     // one item per draw, in the order: the early phase's first level
    ClusterItem *items[2];       // the walk's levels, alternating
    uint *results;               // per item of the level: three counts (what it produced)
    uint *block_totals;          // per workgroup of the level: the sums of those
    uint *block_bases;           // prefix sums of block_totals
    WalkCounters *counters;
    ClusterItem *candidates;     // items set aside for the late phase
    ClusterItem *cluster_items;  // the clusters to draw, by draw list
    ClusterDispatch *dispatches; // one per draw list
    ClusterItem *early_candidates;  // the early phase's candidates: the late phase's first level
    WalkCounters *early_counters;   // the early phase's counters: how many
    uint *patches;               // the terrain patches this phase draws, packed (terrain.slangh)
    uint *patch_command;         // how many, then 1, 1: the mesh dispatch's workgroup counts
    uint *nodes;                 // the terrain selection's working lists: two runs of max_terrain_patches
    uint *early_patches;         // per patch (terrain_patch_index): the frame the early phase last drew it in
    uint draw_count;
    uint item_capacity;          // the most items a level, the candidates or the cluster items can hold
};

struct CullPushData {
    FrameData *frame;
    CullTables *tables;
    uint source;        // the depth pyramid steps: the resource heap slot read, the depth buffer or the level above
    uint target;        // the level written
    uint level;         // the walk: which level of items is being walked
    uint late;          // the walk: 1 in the late phase
    uint2 source_size;  // in texels
    uint2 target_size;
};

[[vk::push_constant]]
ConstantBuffer<CullPushData> push;

// The most patches a level of the selection, and a frame, can hold (terrain.h).
static const uint max_terrain_patches = 65536;

// Threads per workgroup of the walk's steps, one item each; a level's prefix sums run per workgroup, then across them.
static const uint walk_workgroup_size = scan_size;

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

// 3. Walking the hierarchies

// A draw's sphere or box, relative to the camera. A cluster's or node's sphere is in the primitive's space: its centre goes through the draw's model matrix into the cell, then relative to the camera; its radius, and its error, grow by the most the matrix stretches anything, the longest of its columns.
float model_scale(float4x4 model) {
    return sqrt(max(max(dot(model[0].xyz, model[0].xyz), dot(model[1].xyz, model[1].xyz)), dot(model[2].xyz, model[2].xyz)));
}

float3 relative_center(FrameData *frame, DrawData draw, float3 center) {
    return camera_relative(frame, draw.cell, mul(draw.model, float4(center, 1.0)).xyz);
}

// Whether a sphere is in view and, if asked, not hidden: the frustum test on the sphere, the pyramid test on its box.
bool sphere_in_view(float4x4 view_projection, float3 center, float radius) {
    return in_view(view_projection, center - radius, center + radius);
}

// The draw list a draw belongs to (culling.h's draw_list_index): its material's alpha mode, times whether it's double-sided and whether its transform mirrors.
uint draw_list_of(FrameData *frame, DrawData draw) {
    const Material material = frame.materials[draw.material];
    const bool mirrored = determinant((float3x3)draw.model) < 0.0;
    return material.alpha_mode * 4 + (material.double_sided != 0 ? 2 : 0) + (mirrored ? 1 : 0);
}

// What walkMain found out about an item: how many items it sends to the next level, whether it goes to the candidates, whether it's a cluster to draw. Three words per item.
static const uint result_next = 0;
static const uint result_candidate = 1;
static const uint result_cluster = 2;

// The item's DAG error test: an error of `error` metres (already scaled by the draw) on a sphere at `center` with `radius` is over the threshold when it projects to more than cluster_error_pixels, measured at the sphere's nearest point, never nearer than the near plane (clusterlod's rule: the test must give a parent at least its children's answer, which the hierarchy's errors guarantee).
bool over_threshold(FrameData *frame, float3 center, float radius, float error) {
    const float distance = max(length(center) - radius, 0.05);
    return error * frame.cluster_error_scale > distance;
}

// Before a level: how many items it has, and how many workgroups its steps take. The first level of the early phase is every draw; of the late phase, the early phase's candidates. Every other level is what the level before produced. The first level also clears the counts.
[shader("compute")]
[numthreads(1, 1, 1)]
void walkStartMain() {
    CullTables *tables = push.tables;
    WalkCounters *counters = tables.counters;

    if (push.level == 0) {
        counters.items = push.late != 0 ? tables.early_counters.candidates : tables.draw_count;
        counters.next_items = 0;
        counters.candidates = 0;
        counters.clusters = 0;
        counters.triangles = 0;
        counters.subpixel_draws = 0;

        for (uint list = 0; list < 12; ++list) {
            counters.list_counts[list] = 0;
        }

        for (uint level = 0; level < 16; ++level) {
            counters.level_clusters[level] = 0;
        }
    } else {
        counters.items = counters.next_items;
    }

    counters.dispatch_x = (counters.items + walk_workgroup_size - 1) / walk_workgroup_size;
    counters.dispatch_y = 1;
    counters.dispatch_z = 1;
}

// Per item of the level: what it produces. A draw outside the view produces nothing; one hidden by the pyramid goes to the candidates; otherwise its hierarchy's roots, one per DAG level, go to the next level. A node is pruned when even its worst error is under the threshold, when it's out of view or when it's hidden (a candidate, then); otherwise its children go on, or, for a leaf, its group's clusters. A cluster is drawn when its own group is over the threshold and the group that refines it isn't, and it's in view, not hidden, and not facing away.
[shader("compute")]
[numthreads(walk_workgroup_size, 1, 1)]
void walkMain(uint3 id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint thread : SV_GroupThreadID) {
    FrameData *frame = push.frame;
    CullTables *tables = push.tables;
    WalkCounters *counters = tables.counters;
    const uint count = counters.items;
    const uint i = id.x;

    uint next = 0;
    uint candidate = 0;
    uint cluster = 0;

    if (i < count) {
        const ClusterItem item = push.level == 0 && push.late == 0 ? tables.draw_items[i]
            : push.level == 0 ? tables.early_candidates[i]
            : tables.items[push.level & 1][i];
        const DrawData draw = frame.draws[item.draw];
        const uint kind = item_kind(item);

        if (kind == item_kind_draw) {
            // The draw's box, from its cell to the camera: both corners in the same cell.
            const float3 lo = camera_relative(frame, draw.cell, draw.bounds_min);
            const float3 hi = camera_relative(frame, draw.cell, draw.bounds_max);

            if (in_view(frame.view_projection, lo, hi)) {
                if (occluded(frame, lo, hi)) {
                    candidate = 1;
                } else {
                    next = frame.primitives[draw.primitive].levels;

                    // The box's sphere, for the count of draws smaller than a pixel.
                    const float3 centre = (lo + hi) * 0.5;
                    if (length(hi - lo) * 0.5 * frame.subpixel_scale < length(centre)) {
                        InterlockedAdd(counters.subpixel_draws, 1);
                    }
                }
            }
        } else if (kind == item_kind_node) {
            const ClusterNode node = frame.cluster_nodes[item_index(item)];
            const float scale = model_scale(draw.model);
            const float3 center = relative_center(frame, draw, node.center);
            const float radius = node.radius * scale;

            if (over_threshold(frame, center, radius, node.error * scale) && sphere_in_view(frame.view_projection, center, radius)) {
                if (occluded(frame, center - radius, center + radius)) {
                    candidate = 1;
                } else {
                    next = node.group == ~0u ? node.child_count : frame.cluster_groups[node.group].cluster_count;
                }
            }
        } else {
            const Cluster c = frame.clusters[item_index(item)];
            const ClusterGroup own = frame.cluster_groups[c.group];
            const float scale = model_scale(draw.model);
            const float3 center = relative_center(frame, draw, c.center);
            const float radius = c.radius * scale;

            // The level of detail: coarse enough, and no coarser than it must be.
            const bool coarse_enough = over_threshold(frame, relative_center(frame, draw, own.center), own.radius * scale, own.error * scale);
            const bool finer_unneeded = c.refined == ~0u
                || !over_threshold(frame, relative_center(frame, draw, frame.cluster_groups[c.refined].center),
                    frame.cluster_groups[c.refined].radius * scale, frame.cluster_groups[c.refined].error * scale);

            if (coarse_enough && finer_unneeded && sphere_in_view(frame.view_projection, center, radius)) {
                // Facing away: every triangle's normal is within the cone's half-angle of its axis, so when the camera is behind all of them, none can face it. meshoptimizer's test, in the primitive's own space, where the cone and the sphere are: the camera goes there by the inverse model matrix, the normal matrix's transpose, exact under any transform, mirroring included, since the front faces are the primitive's own. The sphere's share widens the test for a camera near the cluster. A cutoff of 1 means the cone is too wide to say.
                const float3 camera_in_model = mul(transpose(draw.normal_matrix), float4(-camera_relative(frame, draw.cell, float3(0.0)), 1.0)).xyz;
                const float3 from_camera = c.center - camera_in_model;
                const float distance = length(from_camera);
                const bool away = c.cone_cutoff < 1.0 && distance > 0.0
                    && dot(from_camera, c.cone_axis) >= c.cone_cutoff * distance + c.radius;

                if (!away) {
                    if (occluded(frame, center - radius, center + radius)) {
                        candidate = 1;
                    } else {
                        cluster = 1;
                    }
                }
            }
        }
    }

    // The workgroup's prefix sums aren't kept: walkWriteMain works them out again. Only the totals go on, to walkScanMain.
    uint prefix;
    const uint next_total = group_exclusive_scan(next, thread, prefix);
    const uint candidate_total = group_exclusive_scan(candidate, thread, prefix);
    const uint cluster_total = group_exclusive_scan(cluster, thread, prefix);

    if (i < count) {
        tables.results[i * 3 + result_next] = next;
        tables.results[i * 3 + result_candidate] = candidate;
        tables.results[i * 3 + result_cluster] = cluster;
    }

    if (thread == 0) {
        tables.block_totals[group_id.x * 3 + result_next] = next_total;
        tables.block_totals[group_id.x * 3 + result_candidate] = candidate_total;
        tables.block_totals[group_id.x * 3 + result_cluster] = cluster_total;
    }
}

// One workgroup: the prefix sums of the workgroups' totals, for each of the three outputs; then the level's results: how many items the next level has, and how many workgroups its steps take. Candidates and clusters accumulate across levels, so their bases start where the previous levels left off.
[shader("compute")]
[numthreads(scan_size, 1, 1)]
void walkScanMain(uint3 id : SV_GroupThreadID) {
    CullTables *tables = push.tables;
    WalkCounters *counters = tables.counters;
    const uint thread = id.x;
    const uint blocks = (counters.items + walk_workgroup_size - 1) / walk_workgroup_size;

    uint carry[3] = {0, counters.candidates, counters.clusters};

    for (uint chunk = 0; chunk < blocks; chunk += scan_size) {
        const uint block = chunk + thread;

        for (uint output = 0; output < 3; ++output) {
            const uint value = block < blocks ? tables.block_totals[block * 3 + output] : 0;
            uint prefix;
            const uint total = group_exclusive_scan(value, thread, prefix);

            if (block < blocks) {
                tables.block_bases[block * 3 + output] = carry[output] + prefix;
            }

            carry[output] += total;
        }
    }

    if (thread == 0) {
        counters.next_items = min(carry[0], tables.item_capacity);
        counters.candidates = min(carry[1], tables.item_capacity);
        counters.clusters = min(carry[2], tables.item_capacity);
    }
}

// Per item of the level: its output, where the prefix sums put it. Draws send their roots, nodes their children or their group's clusters, both as items for the next level; candidates go to the candidates; clusters to the cluster items, counted per draw list and per DAG level, with their triangles.
[shader("compute")]
[numthreads(walk_workgroup_size, 1, 1)]
void walkWriteMain(uint3 id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint thread : SV_GroupThreadID) {
    FrameData *frame = push.frame;
    CullTables *tables = push.tables;
    WalkCounters *counters = tables.counters;
    const uint count = counters.items;
    const uint i = id.x;
    const bool inside = i < count;

    const uint next = inside ? tables.results[i * 3 + result_next] : 0;
    const uint candidate = inside ? tables.results[i * 3 + result_candidate] : 0;
    const uint cluster = inside ? tables.results[i * 3 + result_cluster] : 0;

    uint next_prefix;
    group_exclusive_scan(next, thread, next_prefix);
    uint candidate_prefix;
    group_exclusive_scan(candidate, thread, candidate_prefix);
    uint cluster_prefix;
    group_exclusive_scan(cluster, thread, cluster_prefix);

    if (!inside) {
        return;
    }

    const ClusterItem item = push.level == 0 && push.late == 0 ? tables.draw_items[i]
        : push.level == 0 ? tables.early_candidates[i]
        : tables.items[push.level & 1][i];
    const uint kind = item_kind(item);
    ClusterItem *out = tables.items[(push.level + 1) & 1];

    if (next != 0) {
        const uint base = tables.block_bases[group_id.x * 3 + result_next] + next_prefix;

        if (kind == item_kind_draw) {
            const PrimitiveClusters primitive = frame.primitives[frame.draws[item.draw].primitive];
            for (uint k = 0; k < next && base + k < tables.item_capacity; ++k) {
                out[base + k] = make_item(item.draw, item_kind_node, primitive.first_node + k);
            }
        } else {
            const ClusterNode node = frame.cluster_nodes[item_index(item)];
            for (uint k = 0; k < next && base + k < tables.item_capacity; ++k) {
                out[base + k] = node.group == ~0u
                    ? make_item(item.draw, item_kind_node, node.first_child + k)
                    : make_item(item.draw, item_kind_cluster, frame.cluster_groups[node.group].first_cluster + k);
            }
        }
    }

    if (candidate != 0) {
        const uint at = tables.block_bases[group_id.x * 3 + result_candidate] + candidate_prefix;
        if (at < tables.item_capacity) {
            tables.candidates[at] = item;
        }
    }

    if (cluster != 0) {
        const uint at = tables.block_bases[group_id.x * 3 + result_cluster] + cluster_prefix;
        if (at < tables.item_capacity) {
            tables.cluster_items[at] = item;
        }

        const Cluster c = frame.clusters[item_index(item)];
        InterlockedAdd(counters.triangles, c.triangle_count);
        InterlockedAdd(counters.level_clusters[min(frame.cluster_groups[c.group].depth, 15)], 1);
        InterlockedAdd(counters.list_counts[draw_list_of(frame, frame.draws[item.draw])], 1);
    }
}

// 4. The dispatches

// After the last level: each draw list's dispatch. The cluster items are in the draws' order, which is the lists' order, so a list's clusters are a run: its start is the lists' before it, summed. The width is capped at the 65,535 workgroups Vulkan guarantees per dimension; y takes the rest, and the mesh shader skips the workgroups past the count.
[shader("compute")]
[numthreads(1, 1, 1)]
void writeDispatchesMain() {
    CullTables *tables = push.tables;
    WalkCounters *counters = tables.counters;
    uint start = 0;

    for (uint list = 0; list < 12; ++list) {
        const uint count = counters.list_counts[list];
        tables.dispatches[list] = ClusterDispatch(min(count, max_dispatch_width), (count + max_dispatch_width - 1) / max_dispatch_width, 1, count, start, 0, tables.cluster_items);
        start += count;
    }
}

// 5. The depth pyramid's levels

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

// 6. The terrain's patches

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
#include "includes/clusters.h"
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

// One mesh dispatch draws a whole list of clusters with one pipeline and one dynamic state, so draws are sorted into lists by what those fix:
//   - the alpha mode, which picks the pipeline,
//   - whether the material is double-sided, which sets the cull mode,
//   - whether the transform mirrors, which sets the front face.
// Three alpha modes times two times two: twelve lists, numbered alpha mode x 4 + double-sided x 2 + mirrored.
constexpr std::uint32_t draw_list_count = 12;

constexpr std::uint32_t draw_list_index(AlphaMode alpha_mode, bool double_sided, bool mirrored) {
    return static_cast<std::uint32_t>(alpha_mode) * 4 + (double_sided ? 2 : 0) + (mirrored ? 1 : 0);
}

// What the cull needs to know about a draw: its list, and which primitive it draws.
struct CullDraw {
    std::uint32_t list;
    std::uint32_t primitive;
};

// GPU culling

// The cull's compute pipelines and buffers, for a scene whose draws never change: the order is worked out once, here.
//   - Draws are put in an order: by list, then by primitive, then by draw index, so that the clusters the cull produces, which keep the draws' order, come out list by list.
//   - A draw is kept when its box is in view and not hidden: not wholly behind what the depth pyramid holds. The pyramid is the depth buffer with a mip chain where each texel holds the farthest depth of the texels above it, so a box can be tested against the depth under its whole screen rectangle in four reads.
//   - A kept draw is walked: its primitive's hierarchy of clusters (clusters.h), from the roots down, each node tested like the draw, and against the error threshold, until the clusters at the one level of detail the distance calls for. The walk goes level by level, every level one item per thread: what each item produces, prefix sums across the workgroups, then the items written where the sums put them.
//   - Every frame, the cull runs in two phases around the depth prepass. The early phase walks every draw against the pyramid the previous frame built, under this frame's view: a guess, right wherever the view hasn't changed. Whatever it sets aside on the pyramid alone, a draw, a node or a cluster, goes on a list of candidates. The prepass draws what it keeps, the pyramid is built from that depth, and the late phase walks the candidates against it. Whatever the guess hid wrongly is drawn late; nothing visible is missed, and nothing is drawn twice.
//   - The terrain is culled the same way: each phase walks the height field's quadtree down to patches, and lists the patches to draw.
// Everything a step writes goes to a place fixed by prefix sums over the previous steps' results, never by which thread got there first: the same view gives the same clusters, in the same order, every frame.
enum class CullPhase : std::size_t {
    early,
    late,
};

constexpr std::array cull_phases{CullPhase::early, CullPhase::late};

// The most items a level of the walk, the candidates or the clusters to draw can hold: 2^19, half a million.
constexpr std::uint32_t max_cull_items = 1u << 19;

// Threads per workgroup of the walk's steps: cull.slang's walk_workgroup_size.
constexpr std::uint32_t walk_workgroup_size = 256;

// What one phase rewrites every frame, and the table that names all of it.
struct CullPhaseBuffers {
    std::array<Buffer, 2> items;  // the walk's levels, alternating
    Buffer results;               // per item of a level: what it produced, three counts
    Buffer block_totals;          // per workgroup of a level: the sums of those
    Buffer block_bases;           // prefix sums of block_totals
    Buffer counters;              // one WalkCounters; read as indirect dispatch arguments too
    Buffer candidates;            // items set aside for the late phase
    Buffer cluster_items;         // the clusters to draw, by draw list
    Buffer dispatches;            // one ClusterDispatch per draw list
    Buffer patches;               // the terrain patches to draw, packed (terrain.slangh)
    Buffer patch_command;         // one VkDrawMeshTasksIndirectCommandEXT: how many patches, 1, 1
    Buffer nodes;                 // the terrain selection's working lists
    Buffer tables;                // one CullTables: where all of these are, and the shared tables
};

struct DrawCulling {
    vk::raii::Pipeline walk_start = nullptr;
    vk::raii::Pipeline walk = nullptr;
    vk::raii::Pipeline walk_scan = nullptr;
    vk::raii::Pipeline walk_write = nullptr;
    vk::raii::Pipeline write_dispatches = nullptr;
    vk::raii::Pipeline copy_depth = nullptr;
    vk::raii::Pipeline reduce_depth = nullptr;
    vk::raii::Pipeline select_early_patches = nullptr;
    vk::raii::Pipeline select_late_patches = nullptr;

    // Written once, shared by the phases.
    Buffer order;          // draw indices, in the cull's order
    Buffer draw_items;     // one ClusterItem per draw, in the order
    Buffer early_patches;  // per terrain patch, the frame the early phase last drew it in

    std::array<CullPhaseBuffers, cull_phases.size()> phases;

    std::array<std::uint32_t, draw_list_count> list_draws{};  // draws per list, for skipping empty lists
    std::uint32_t draw_count = 0;
    std::uint32_t walk_levels = 0;  // how many levels the walk takes: a draw, its hierarchy's depth, a group's clusters
};

// `draws` has one CullDraw per draw, in draw order; `terrain` and `clusters` are what the phases walk.
DrawCulling create_draw_culling(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::span<const CullDraw> draws,
    const Terrain &terrain,
    const Clusters &clusters
);

// The totals, as the cull copies them out: each phase's counters, and its terrain patches.
struct CullTotals {
    WalkCounters early;
    WalkCounters late;
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

// Records one phase of the cull for the frame whose FrameData is at `frame`: the late phase must come after the early one, and after record_depth_pyramid. Afterwards, mesh dispatches may read the phase's dispatches, and mesh shaders its cluster items and patches. Its counters are also copied into their half of `readback`, a host-visible buffer holding one CullTotals, for the CPU to read once the frame is done.
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

// The address of `phase`'s ClusterDispatch for list `list`: what the list's push data names as drawn.
vk::DeviceAddress dispatch_address(const DrawCulling &culling, CullPhase phase, std::uint32_t list);

// Draws list `list`'s clusters from `phase` for this frame: one mesh dispatch, a workgroup per cluster. The pipeline, its push data naming the list's dispatch, and the list's dynamic state must already be set.
void draw_list(const vk::raii::CommandBuffer &commands, const DrawCulling &culling, CullPhase phase, std::uint32_t list);

// Draws `phase`'s terrain patches for this frame: one mesh dispatch, a workgroup per patch. The terrain pipeline, its push data naming the phase's patches, and the cull mode and front face must already be set.
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
    const Terrain &terrain,
    const Clusters &clusters
) {
    DrawCulling culling;
    culling.walk_start = create_compute_pipeline(device, "cull", "walkStartMain");
    culling.walk = create_compute_pipeline(device, "cull", "walkMain");
    culling.walk_scan = create_compute_pipeline(device, "cull", "walkScanMain");
    culling.walk_write = create_compute_pipeline(device, "cull", "walkWriteMain");
    culling.write_dispatches = create_compute_pipeline(device, "cull", "writeDispatchesMain");
    culling.copy_depth = create_compute_pipeline(device, "cull", "copyDepthMain");
    culling.reduce_depth = create_compute_pipeline(device, "cull", "reduceDepthMain");
    culling.select_early_patches = create_compute_pipeline(device, "cull", "selectEarlyPatchesMain");
    culling.select_late_patches = create_compute_pipeline(device, "cull", "selectLatePatchesMain");
    culling.draw_count = static_cast<std::uint32_t>(draws.size());

    // The walk's levels: the draw, then its hierarchy's nodes, down to a group's clusters, then the clusters themselves. A tree of 8-wide nodes over g groups is ceil(log8 g) + 1 deep; counted over every group in the scene rather than per level of one primitive, it's a safe overestimate, and a level with nothing in it costs four empty dispatches.
    std::uint32_t deepest = 1;
    for (std::uint32_t groups = clusters.group_count; groups > 1; groups = (groups + cluster_node_width - 1) / cluster_node_width) {
        ++deepest;
    }
    culling.walk_levels = 1 + deepest + 2;

    // The order: by list, then by primitive. stable_sort keeps draws that tie in draw order, so the order, and every frame's clusters, are fixed.
    std::vector<std::uint32_t> order(draws.size());
    std::iota(order.begin(), order.end(), 0u);
    std::ranges::stable_sort(order, [&](std::uint32_t a, std::uint32_t b) {
        return draws[a].list != draws[b].list ? draws[a].list < draws[b].list : draws[a].primitive < draws[b].primitive;
    });

    std::vector<ClusterItem> draw_items;
    for (const std::uint32_t draw : order) {
        draw_items.push_back(ClusterItem{.draw = draw, .what = item_kind_draw << item_kind_shift});
        ++culling.list_draws[draws[draw].list];
    }

    // The tables, uploaded once; a buffer can't be empty.
    const auto upload = [&](std::span<const std::byte> bytes) {
        const std::vector<std::byte> one(sizeof(std::uint32_t));
        return upload_buffer(device, gpu, queue, pool, bytes.empty() ? std::span(one) : bytes,
            vk::BufferUsageFlagBits::eShaderDeviceAddress);
    };
    culling.order = upload(std::as_bytes(std::span(order)));
    culling.draw_items = upload(std::as_bytes(std::span(draw_items)));

    // The frame each patch was last drawn in: 0 to begin with, before any frame.
    const std::vector<std::uint32_t> never(terrain.patch_count);
    culling.early_patches = upload(std::as_bytes(std::span(never)));

    // What each phase's steps rewrite every frame. The counters are read as indirect dispatch arguments and copied out; the dispatches are read as indirect draw arguments.
    const auto blocks = (max_cull_items + walk_workgroup_size - 1) / walk_workgroup_size;

    for (CullPhaseBuffers &phase : culling.phases) {
        for (Buffer &items : phase.items) {
            items = gpu_numbers(device, gpu, static_cast<std::size_t>(max_cull_items) * 2);
        }
        phase.results = gpu_numbers(device, gpu, static_cast<std::size_t>(max_cull_items) * 3);
        phase.block_totals = gpu_numbers(device, gpu, static_cast<std::size_t>(blocks) * 3);
        phase.block_bases = gpu_numbers(device, gpu, static_cast<std::size_t>(blocks) * 3);
        phase.counters = gpu_numbers(device, gpu, sizeof(WalkCounters) / sizeof(std::uint32_t),
            vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc);
        phase.candidates = gpu_numbers(device, gpu, static_cast<std::size_t>(max_cull_items) * 2);
        phase.cluster_items = gpu_numbers(device, gpu, static_cast<std::size_t>(max_cull_items) * 2);
        phase.dispatches = create_buffer(device, gpu, draw_list_count * sizeof(ClusterDispatch),
            vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eIndirectBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        phase.patches = gpu_numbers(device, gpu, max_terrain_patches);
        phase.patch_command = gpu_numbers(device, gpu, 3, vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc);
        phase.nodes = gpu_numbers(device, gpu, 2 * max_terrain_patches);
    }

    // Each phase's table: its own buffers, the shared ones, and the early phase's candidates and counters, which the late phase starts from.
    const CullPhaseBuffers &early = culling.phases[static_cast<std::size_t>(CullPhase::early)];

    for (CullPhaseBuffers &phase : culling.phases) {
        const CullTables tables{
            .order = culling.order.address,
            .draw_items = culling.draw_items.address,
            .items = {phase.items[0].address, phase.items[1].address},
            .results = phase.results.address,
            .block_totals = phase.block_totals.address,
            .block_bases = phase.block_bases.address,
            .counters = phase.counters.address,
            .candidates = phase.candidates.address,
            .cluster_items = phase.cluster_items.address,
            .dispatches = phase.dispatches.address,
            .early_candidates = early.candidates.address,
            .early_counters = early.counters.address,
            .patches = phase.patches.address,
            .patch_command = phase.patch_command.address,
            .nodes = phase.nodes.address,
            .early_patches = culling.early_patches.address,
            .draw_count = culling.draw_count,
            .item_capacity = max_cull_items,
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

    // The previous frame's mesh dispatches read the phase's dispatches, its mesh shaders the cluster items and patches, and its copy the counters: wait for them before rewriting any. Rewriting what was read only needs the wait, so no access is made visible.
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eMeshShaderEXT
            | vk::PipelineStageFlagBits2::eCopy,
        vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);

    // Every step reads its buffers through the phase's table; the walk's steps also read which level they're on.
    CullPushData push{.frame = frame, .tables = buffers.tables.address, .late = early ? 0u : 1u};

    // A step that reads the counters as its workgroup count.
    const auto indirect = [&](const vk::raii::Pipeline &pipeline) {
        bind(commands, pipeline, push);
        commands.dispatchIndirect(*buffers.counters.handle, offsetof(WalkCounters, dispatch_x));
    };

    // Between a step that writes the counters and one that reads them as its workgroup count, as well as in its shader.
    const auto to_indirect = [&] {
        memory_barrier(commands,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eShaderStorageRead);
    };

    // The walk, level by level: four steps a level, the first one thread, the third one workgroup, the others a thread per item.
    for (std::uint32_t level = 0; level < culling.walk_levels; ++level) {
        push.level = level;

        bind(commands, culling.walk_start, push);
        commands.dispatch(1, 1, 1);
        to_indirect();

        indirect(culling.walk);
        compute_to_compute(commands);

        bind(commands, culling.walk_scan, push);
        commands.dispatch(1, 1, 1);
        compute_to_compute(commands);

        indirect(culling.walk_write);
        compute_to_compute(commands);
    }

    // The draw lists' dispatches, from the counts the walk left.
    bind(commands, culling.write_dispatches, push);
    commands.dispatch(1, 1, 1);

    // The terrain's patches, which read nothing of the above: one workgroup walks the quadtree.
    bind(commands, early ? culling.select_early_patches : culling.select_late_patches, push);
    commands.dispatch(1, 1, 1);

    // The mesh dispatches read the dispatches and the patch command as indirect arguments, and the mesh shaders the cluster items and patches; the copy reads the counters; the late cull reads the early phase's candidates, counters and patches.
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eMeshShaderEXT
            | vk::PipelineStageFlagBits2::eCopy | vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eShaderStorageRead
            | vk::AccessFlagBits2::eTransferRead);

    // The totals: the phase's counters, and its patch count.
    commands.copyBuffer(*buffers.counters.handle, readback, vk::BufferCopy{
        .srcOffset = 0,
        .dstOffset = early ? offsetof(CullTotals, early) : offsetof(CullTotals, late),
        .size = sizeof(WalkCounters),
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

vk::DeviceAddress dispatch_address(const DrawCulling &culling, CullPhase phase, std::uint32_t list) {
    return culling.phases[static_cast<std::size_t>(phase)].dispatches.address + list * sizeof(ClusterDispatch);
}

void draw_list(const vk::raii::CommandBuffer &commands, const DrawCulling &culling, CullPhase phase, std::uint32_t list) {
    // The GPU reads the list's workgroup counts from its dispatch: one workgroup per cluster, wrapped into rows of 65,535.
    const CullPhaseBuffers &buffers = culling.phases[static_cast<std::size_t>(phase)];
    commands.drawMeshTasksIndirectEXT(*buffers.dispatches.handle, list * sizeof(ClusterDispatch), 1, sizeof(ClusterDispatch));
}


void draw_terrain(const vk::raii::CommandBuffer &commands, const DrawCulling &culling, CullPhase phase) {
    // The GPU reads how many workgroups to run from the phase's command: one per patch.
    const CullPhaseBuffers &buffers = culling.phases[static_cast<std::size_t>(phase)];
    commands.drawMeshTasksIndirectEXT(*buffers.patch_command.handle, 0, 1, sizeof(vk::DrawMeshTasksIndirectCommandEXT));
}
```

## 19.5 Drawing clusters: `mesh.slang`, `pipeline.h`, `pipeline.cpp`, `terrain.slang`, `vulkan_setup.cpp`

### Why
A mesh shader draws a cluster as one draws a terrain patch, and the fragment shaders follow unchanged, but for the triangle they ask about.

### How
- **`meshMain`,** 128 threads, one per vertex and one per triangle: the workgroup finds its cluster item at the list's start plus its index, the item's draw and cluster, and writes the cluster's vertices, transformed as the vertex shader used to, and its triangles from the cluster's tables. `VertexOutput` gains the cluster, flat like the draw index.
- **The primitive index, written.** A fragment shader's `SV_PrimitiveID` under a mesh shader is, in the specification's words, the primitive index the mesh shader wrote per triangle, and with nothing written, whatever was in the output, unlike the vertex path's, which counts triangles itself. `PrimitiveOutput` writes each triangle's index within its cluster; the first build didn't, and every triangle lookup was garbage: wrong flat normals, ray origins on the wrong side of the surface, shadows that weren't there.
- **`fetch_triangle`:** a triangle's three vertices, from the cluster's tables, with their positions relative to the camera and their normals: the flat normal comes from them, as before, and so does the shadow terminator's fix.
- **The shadow terminator.** The vertex normals describe a smooth surface the triangles only approximate; where a triangle tilts away from a light that the smooth normal faces, a ray from the triangle starts inside the surface and hits it at once. On a scanned mesh whose triangles are smaller than a pixel, that's half the pixels on the lit side: the dragon came out dark with white speckles. Hanika's fix ("Hacking the Shadow Terminator", Ray Tracing Gems II, 2021) moves the ray's start to where the smooth surface would have it: onto the plane through each vertex perpendicular to its normal, when the point is beneath that plane, the three results blended by the point's barycentric coordinates, which `terminator_origin` works out from the areas the point cuts the triangle into. Chiang, Li and Burley's "Taming the Shadow Terminator" (2019) solves the same problem the other way, by fading the shading near the terminator instead of moving the ray. On a flat-shaded or truly flat surface nothing moves: the box field is pixel-identical to Chapter 18's at full detail.
- **Flat shading** for files without normals uses the triangle's own normal now, signed by its winding and the draw's mirroring, rather than screen-space derivatives.
- **Pipelines:** `create_mesh_pipeline` takes its first stage from `meshMain` as `eMeshEXT`; `create_scene_pipeline` builds both it and the terrain's. The device no longer needs the indirect draw features of Chapter 12: `multiDrawIndirect`, `drawIndirectFirstInstance` and `drawIndirectCount` go.
- **`terrain.slang`'s `meshMain`** reads its patches from `push.drawn`.

### Code
`game-engine/shaders/mesh.slang`:
```slang
// Draws glTF primitives, a cluster at a time: a mesh shader writes each cluster's vertices from the scene's vertex buffer, placed in the world by its draw's DrawData, and its triangles from the cluster's tables; the surface comes from its glTF material, whose textures are read from the descriptor heap. Shaded by shading.slangh: glTF's physically based BRDF, lit by the sun and the file's lights, with ray-traced shadows, and by the sky around the scene, already exposed. Three fragment shaders:
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

// What the mesh shader hands to the rasterizer. SV_Position is the clip-space position; every other field but draw_index and cluster is interpolated across the triangle. Vulkan requires integer fields to be flat, which nointerpolation makes them.
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
    nointerpolation uint cluster : CLUSTER;        // likewise: which cluster the triangle is from
};

// The triangle's normal

// Triangle `triangle` of `cluster`: its three vertices' indices, from the cluster's tables, three bytes a triangle, four to a word.
uint3 cluster_triangle(FrameData *frame, Cluster cluster, uint triangle) {
    const uint first = (cluster.first_triangle + triangle) * 3;
    uint3 local;

    for (uint k = 0; k < 3; ++k) {
        const uint byte = first + k;
        local[k] = (frame.cluster_triangles[byte >> 2] >> ((byte & 3) * 8)) & 0xFF;
    }

    return local;
}

// Triangle `primitive` of `cluster`, as the shading sees it: its three vertices' positions, relative to the camera, and their normals, in the world's axes.
struct Triangle {
    float3 position[3];
    float3 normal[3];
};

Triangle fetch_triangle(FrameData *frame, DrawData draw, uint cluster_index, uint primitive) {
    const Cluster cluster = frame.clusters[cluster_index];
    const uint3 local = cluster_triangle(frame, cluster, primitive);
    Triangle triangle;

    for (uint k = 0; k < 3; ++k) {
        const Vertex vertex = frame.vertices[frame.cluster_vertices[cluster.first_vertex + local[k]]];
        triangle.position[k] = camera_relative(frame, draw.cell, mul(draw.model, float4(vertex.position, 1.0)).xyz);
        triangle.normal[k] = normalize(mul((float3x3)draw.normal_matrix, vertex.normal));
    }

    return triangle;
}

// The triangle's flat normal, from its vertices: where shadow rays start, off the surface along it (shading.slangh's shadow_ray_origin picks the light's side, so its sign doesn't matter). Derivatives of the position across neighbouring pixels would give it more cheaply, but along a silhouette the neighbouring pixels belong to whatever is behind.
float3 triangle_normal(Triangle triangle) {
    const float3 n = cross(triangle.position[1] - triangle.position[0], triangle.position[2] - triangle.position[0]);
    return dot(n, n) > 0.0 ? normalize(n) : float3(0.0, 1.0, 0.0);
}

// The shadow terminator: where shadow rays start on a smooth-shaded surface. The vertex normals describe a smooth surface the triangles only approximate; where a triangle tilts away from a light that the smooth normal faces, a ray from the triangle starts inside the surface and hits it at once, and the lit side of every curve breaks into dark bands and speckles, worst on a scanned mesh whose triangles are smaller than a pixel. Hanika's fix ("Hacking the Shadow Terminator", Ray Tracing Gems II, 2021) moves the point to where the smooth surface would have it: onto the plane through each vertex perpendicular to its normal, when it's beneath that plane, and the three results blended by the point's barycentric coordinates. On a flat-shaded or truly flat surface nothing moves. The point is written in the TLAS's space afterwards, like any ray origin.
float3 terminator_origin(float3 position, Triangle triangle) {
    // The barycentric coordinates of the point, from the areas it cuts the triangle into.
    const float3 e0 = triangle.position[1] - triangle.position[0];
    const float3 e1 = triangle.position[2] - triangle.position[0];
    const float3 n = cross(e0, e1);
    const float area = dot(n, n);

    if (area <= 0.0) {
        return position;
    }

    const float3 d = position - triangle.position[0];
    const float v = dot(cross(e0, d), n) / area;
    const float w = dot(cross(d, e1), n) / area;
    const float3 weight = float3(1.0 - v - w, w, v);

    float3 moved = float3(0.0);

    for (uint k = 0; k < 3; ++k) {
        const float3 to_point = position - triangle.position[k];
        const float below = min(dot(to_point, triangle.normal[k]), 0.0);
        moved += weight[k] * (position - triangle.normal[k] * below);
    }

    return moved;
}

// Mesh shader

// What the mesh shader writes per triangle: its index within the cluster. A fragment shader's SV_PrimitiveID is whatever the mesh shader wrote here, and with nothing written, whatever was in the output, unlike the vertex path's, which counts triangles itself.
struct PrimitiveOutput {
    uint primitive : SV_PrimitiveID;
};

// One workgroup per cluster the cull chose for this draw list (ClusterDispatch): 128 threads, one per vertex and one per triangle. The workgroup finds its cluster item at the list's start plus its index, the item's draw and cluster, and writes the cluster's vertices, transformed as the vertex shader used to, and its triangles, from the cluster's tables, each numbered. A dispatch wider than 65,535 workgroups wraps into y, so workgroups past the count exit with nothing.
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void meshMain(
    uint thread : SV_GroupThreadID,
    uint3 group : SV_GroupID,
    out vertices VertexOutput verts[128],
    out indices uint3 tris[128],
    out primitives PrimitiveOutput prims[128]
) {
    FrameData *frame = push.frame;
    ClusterDispatch *dispatch = (ClusterDispatch*)push.drawn;
    const uint index = group.y * max_dispatch_width + group.x;

    // Past the list's clusters, a workgroup launched only by the wrap: nothing to write.
    const ClusterItem item = dispatch.items[dispatch.start + min(index, max(dispatch.count, 1) - 1)];
    const Cluster cluster = frame.clusters[item_index(item)];
    const DrawData draw = frame.draws[item.draw];
    const bool real = index < dispatch.count;

    SetMeshOutputCounts(real ? cluster.vertex_count : 0, real ? cluster.triangle_count : 0);

    if (!real) {
        return;
    }

    if (thread < cluster.vertex_count) {
        const Vertex vertex = frame.vertices[frame.cluster_vertices[cluster.first_vertex + thread]];

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
        output.draw_index = item.draw;
        output.cluster = item_index(item);
        verts[thread] = output;
    }

    if (thread < cluster.triangle_count) {
        tris[thread] = cluster_triangle(frame, cluster, thread);
        prims[thread].primitive = thread;
    }
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
//   1. The interpolated vertex normal. Without normals in the file, glTF asks for flat shading: the triangle's own normal, `face_normal`, from its vertices.
//   2. A normal map tilts it, per texel, within the surface's tangent frame.
//   3. On a double-sided material's back face, the surface faces the other way.
// `map_spread`: how much the normal map's normals spread here, as the standard deviation of their angle, which the mips keep in its alpha as 1 - spread (mips.slang); 0 without a map.
float3 surface_normal(VertexOutput input, DrawData draw, Material material, bool front_face, bool apply_normal_map, float3 face_normal, out float map_spread) {
    float3 normal = input.normal;
    map_spread = 0.0;

    // Without normals in the file, the triangle's own, facing the way its winding says.
    if (all(normal == 0.0)) {
        normal = determinant((float3x3)draw.model) < 0.0 ? -face_normal : face_normal;
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

// The interpolated vertex normal, facing the viewer on a double-sided material's back face; the triangle's own, `face_normal`, when the file has no normals, which glTF asks be shaded flat. This is the surface at the scale the mesh describes it, which ambient occlusion searches against: a normal map's detail isn't in the depth buffer. The face normal's sign follows the triangle's winding, which faces the camera on a front face; a mirroring transform reverses the winding, and the sign with it.
float3 vertex_normal(VertexOutput input, DrawData draw, Material material, bool front_face, float3 face_normal) {
    float3 normal = input.normal;

    if (all(normal == 0.0)) {
        normal = determinant((float3x3)draw.model) < 0.0 ? -face_normal : face_normal;
    }

    normal = normalize(normal);
    return material.double_sided != 0 && !front_face ? -normal : normal;
}

// The prepass draws every opaque and masked surface first, writing only its depth and its vertex normal, octahedrally encoded. Masked surfaces cut out their transparent texels here too, so the depth buffer holds exactly the surfaces the lighting pass will shade.
[shader("fragment")]
float2 prepassMain(VertexOutput input, bool front_face : SV_IsFrontFace, uint primitive : SV_PrimitiveID) : SV_Target {
    FrameData *frame = push.frame;
    const DrawData draw = frame.draws[input.draw_index];
    const Material material = frame.materials[draw.material];

    if (alpha_mode == alpha_mask) {
        const float alpha = material.base_color_factor.a * sample_slot(material.base_color, input).a * input.color.a;
        if (alpha < material.alpha_cutoff) {
            discard;
        }
    }

    return encode_octahedral(vertex_normal(input, draw, material, front_face, triangle_normal(fetch_triangle(frame, draw, input.cluster, primitive))));
}

// Shading a fragment

// The surface at this fragment: its exposed radiance (or one input, in a debug view), and its alpha. The lighting pass writes it as it is; the transparency pass adds it into its sums. `front_face`: whether this triangle faces the camera; `primitive`: which of its cluster's triangles it is.
float4 shade_fragment(VertexOutput input, bool front_face, uint primitive) {
    FrameData *frame = push.frame;
    const DrawData draw = frame.draws[input.draw_index];
    const Material material = frame.materials[draw.material];

    // The triangle itself: its flat normal, which shadow rays start off the surface along and surfaces without normals are shaded by, and its vertices, for the shadow terminator.
    const Triangle triangle = fetch_triangle(frame, draw, input.cluster, primitive);
    const float3 face_normal = triangle_normal(triangle);

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
    const float3 normal = surface_normal(input, draw, material, front_face, frame.view != view_vertex_normal, face_normal, map_spread);

    // Roughness, widened where the normals spread within the pixel. A perfectly smooth surface would reflect a punctual light from a single point, too small for any pixel to catch; a floor on roughness keeps highlights visible.
    const float floored = max(roughness, 0.045);
    const float alpha = antialiased_alpha(floored * floored, vertex_normal(input, draw, material, front_face, face_normal), map_spread);
    const float shading_roughness = sqrt(alpha);

    // Ambient occlusion, from the AO pass's image at this pixel. It combines with the occlusion map by min, not product: both estimate the same thing, at two scales. The bent normal is a deflection from the vertex normal; turning the shading normal by the same deflection keeps the normal map's detail. See-through surfaces aren't in the prepass, so the image there holds whatever is behind them: they use the map alone.
    const float4 gtao = Texture2D.Handle(uint2(frame.ambient_occlusion, 0)).Load(int3(int2(input.position.xy), 0));
    float visibility = occlusion;
    float3 irradiance_normal = normal;

    if (frame.ao_enabled != 0 && alpha_mode != alpha_blend) {
        visibility = min(occlusion, gtao.w);
        irradiance_normal = normalize(rotate_from_to(vertex_normal(input, draw, material, front_face, face_normal), gtao.xyz, normal));
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
    const ShadowSurface from = {
        terminator_origin(input.relative_position, triangle),
        face_normal,
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

// The lighting pass, for opaque and masked surfaces. SV_Target: the value written to color attachment 0. SV_IsFrontFace: whether this triangle faces the camera; SV_PrimitiveID: which of its cluster's triangles it is, as the mesh shader numbered them.
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

// Draws shaders/mesh.slang's clusters into images of `color_formats`, one per attachment, depth-tested against a `depth_format` depth buffer, for `pass` and materials with alpha mode `alpha_mode`. A mesh shader makes each cluster's triangles. There is no pipeline layout: shaders find their resources in the descriptor heap. Cull mode and front face are set per draw list.
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

    // The states every scene pipeline shares, given its shader stages: the meshes' and the terrain's. `mesh_shader` pipelines make their own triangles, so they have no vertex input or assembly state; Vulkan ignores both for them. No pipeline uses a vertex shader any more; the parameter stays for the day one does.
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

    // A mesh shader stage in place of the vertex stage: each workgroup writes one cluster's vertices and triangles.
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eMeshEXT,
            .module = *module,
            .pName = "meshMain",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *module,
            .pName = prepass ? "prepassMain" : transparency ? "transparentMain" : "fragmentMain",
            .pSpecializationInfo = &specialization,
        },
    };

    return create_scene_pipeline(device, stages, true, color_formats, depth_format, pass);
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

In `game-engine/shaders/terrain.slang`, replace `meshMain` with:
```slang
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
    uint *patches = (uint*)push.drawn;
    const uint packed = patches[group.x];
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
```

In `game-engine/shaders/terrain.slang`, replace the line `// The same push data as the meshes': the frame, and the phase's patches in place of its instances.` with:
```slang
// The same push data as the meshes': the frame, and the phase's patches as what's drawn.
```

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
            && vulkan10.samplerAnisotropy
            && vulkan10.shaderInt64
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
                .samplerAnisotropy = vk::True,          // sharper textures seen at an angle
                .shaderInt64 = vk::True,                // 64-bit integers: the TLAS's address
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

## 19.6 Scenes from several files: `scene.h`, `scene.cpp`, `main.cpp`

### Why
The dragons come from a second file, placed twelve times, and the frame draws dispatches instead of draws.

### How
- **`add_gltf`** adds a file's default scene to a scene that already has one, once for each placement given: its materials, textures, samplers and images land after the scene's, every index moved up by what was there, and its meshes are loaded once and drawn once per placement. `load_gltf` becomes `add_gltf` into an empty scene with one placement; `add_primitive` takes where the file's materials start.
- **`main`:**
  - **The dragons:** after the terrain, so they can stand on it: twelve placements around the plateau, 100 m to 4 km out, spaced by a factor of 1.4, each turned to face the centre, scaled from centimetres, feet on the ground.
  - **Clusters** are built once the scene is complete, before the cull, which walks them.
  - **`cluster_error_pixels`,** 1: how far a level's error may project before a finer level takes over. 2 halves the triangles for a visible softening; 0.5 doubles them.
  - **`draw_mode`** pushes each list's dispatch and dispatches it; `draw_terrain_patches` pushes the patches.
  - **The title** says clusters drawn, how many late, triangles, patches, and draws under a pixel across. Every two seconds, the terminal prints the clusters drawn per level of detail, for setting the threshold against a frame budget at a chosen spot.

### Code
`game-engine/src/includes/scene.h`:
```cpp
#pragma once

#include "includes/shader_types.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
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

// Adds a .gltf or .glb file's default scene to `scene`, its materials, samplers, lights and images too, once for each of `placements`: a transform from the file's space to the world's, in metres. The file's meshes are loaded once; each placement draws them again.
void add_gltf(Scene &scene, const std::filesystem::path &path, std::span<const glm::dmat4> placements);

// Point and spot lights without a range in the file are given one: where their illuminance falls below light_threshold lux, at most max_light_range metres. 0.001 lux is at most a few 8-bit steps even at the exposure for the darkest scenes, and the falloff fades to it smoothly.
constexpr float light_threshold = 0.001f;
constexpr float max_light_range = 4096.0f;

// Adds the materials of a .gltf file to `scene`, with their textures and samplers, and returns the index of its first one: for surfaces that aren't in any file's scene, like the terrain's. The file needs no meshes or scenes.
std::uint32_t add_materials(Scene &scene, const std::filesystem::path &path);

// Adds `count` coloured point and spot lights to `scene`, spread through its box, the same ones every run: something to test many lights with. Every fourth is a spot shining down; every 64th has no range of its own.
void add_test_lights(Scene &scene, std::uint32_t count);
```

In `game-engine/src/scene.cpp`, replace `add_primitive` with:
```cpp
    // Appends one primitive to the scene. Points, lines, and primitives without positions are skipped, as glTF allows: this renderer draws triangles. The file's materials start at `first_material` among the scene's; `default_material` is used when the primitive doesn't name one.
    std::optional<LoadedPrimitive> add_primitive(
        const tinygltf::Model &model, const tinygltf::Primitive &source, std::uint32_t first_material, std::uint32_t default_material, Scene &scene
    ) {
        const auto position = source.attributes.find("POSITION");
        const bool triangles = source.mode == TINYGLTF_MODE_TRIANGLES
            || source.mode == TINYGLTF_MODE_TRIANGLE_STRIP
            || source.mode == TINYGLTF_MODE_TRIANGLE_FAN;

        if (!triangles || position == source.attributes.end()) {
            return std::nullopt;
        }

        // The default material sits right after the file's own, so anything at or past it isn't one of the file's materials.
        if (source.material >= static_cast<int>(default_material - first_material)) {
            throw std::runtime_error("a primitive names a material that doesn't exist");
        }

        const std::vector<glm::vec3> positions = read_vec3(model, position->second);

        // Normals are optional. Missing ones stay (0, 0, 0), which tells the shader to shade the triangle flat, as glTF asks.
        std::vector<glm::vec3> normals(positions.size(), glm::vec3{0.0f});

        if (const auto normal = source.attributes.find("NORMAL"); normal != source.attributes.end()) {
            normals = read_vec3(model, normal->second);

            if (normals.size() != positions.size()) {
                throw std::runtime_error("a primitive has a different number of normals and positions");
            }
        }

        // The other attributes are optional too. Each one missing from the file keeps the value given here for every vertex:
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
            .material = source.material >= 0 ? first_material + static_cast<std::uint32_t>(source.material) : default_material,
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
```

In `game-engine/src/scene.cpp`, replace `load_gltf` with:
```cpp
void add_gltf(Scene &scene, const std::filesystem::path &path, std::span<const glm::dmat4> placements) {
    tinygltf::Model model = read_model(path);

    // The file's materials, textures and samplers land after the scene's; its lights after the scene's too.
    const auto first_material = static_cast<std::uint32_t>(scene.materials.size());
    const auto image_base = static_cast<std::int32_t>(scene.images.size());
    const auto sampler_base = static_cast<std::int32_t>(scene.samplers.size());

    Scene loaded;
    add_materials_and_images(model, loaded);

    for (SceneMaterial material : loaded.materials) {
        for (TextureRef *ref : {&material.base_color, &material.metallic_roughness, &material.normal, &material.occlusion, &material.emissive}) {
            ref->image += ref->image >= 0 ? image_base : 0;
            ref->sampler += ref->sampler >= 0 ? sampler_base : 0;
        }
        scene.materials.push_back(material);
    }

    scene.images.insert(scene.images.end(), std::make_move_iterator(loaded.images.begin()), std::make_move_iterator(loaded.images.end()));
    scene.samplers.insert(scene.samplers.end(), loaded.samplers.begin(), loaded.samplers.end());

    const auto default_material = static_cast<std::uint32_t>(scene.materials.size() - 1);

    // Every primitive of every mesh, once. mesh_primitives[m] lists where mesh m's drawable primitives landed in scene.primitives.
    std::vector<std::vector<LoadedPrimitive>> mesh_primitives(model.meshes.size());

    for (std::size_t m = 0; m < model.meshes.size(); ++m) {
        for (const tinygltf::Primitive &primitive : model.meshes[m].primitives) {
            if (const auto loaded_primitive = add_primitive(model, primitive, first_material, default_material, scene)) {
                mesh_primitives[m].push_back(*loaded_primitive);
            }
        }
    }

    // The file may contain several scenes; draw its default one, once per placement.
    if (model.scenes.empty()) {
        throw std::runtime_error(path.string() + " has no scenes");
    }

    const tinygltf::Scene &root = model.scenes.at(model.defaultScene >= 0 ? model.defaultScene : 0);

    for (const glm::dmat4 &placement : placements) {
        for (const int node : root.nodes) {
            visit_node(model, node, placement, mesh_primitives, scene);
        }
    }

    if (scene.draws.empty()) {
        throw std::runtime_error(path.string() + " has nothing to draw");
    }
}

Scene load_gltf(const std::filesystem::path &path, const glm::dvec3 &origin) {
    Scene scene;
    const glm::dmat4 placement = glm::translate(glm::dmat4{1.0}, origin);
    add_gltf(scene, path, std::span(&placement, 1));
    return scene;
}
```

`game-engine/src/main.cpp`:
```cpp
#include "includes/acceleration.h"
#include "includes/ambient_occlusion.h"
#include "includes/buffer.h"
#include "includes/camera.h"
#include "includes/cells.h"
#include "includes/clusters.h"
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

#include <glm/gtc/matrix_transform.hpp>

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

    // Draws every list of alpha mode `mode` that cull phase `phase` kept, with `pipeline`: one mesh dispatch per list, after setting the list's cull mode and front face. Push data says where the frame's data is and which dispatch is drawn; each workgroup finds its cluster and draw through it.
    void draw_mode(
        const vk::raii::CommandBuffer &commands,
        const DrawCulling &culling,
        const DrawList &draws,
        CullPhase phase,
        AlphaMode mode,
        const vk::raii::Pipeline &pipeline
    ) {
        commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

        for (const bool double_sided : {false, true}) {
            for (const bool mirrored : {false, true}) {
                const std::uint32_t list = draw_list_index(mode, double_sided, mirrored);

                if (culling.list_draws[list] == 0) {
                    continue;
                }

                const PushData push{.frame = draws.frame, .drawn = dispatch_address(culling, phase, list)};
                commands.pushDataEXT(vk::PushDataInfoEXT{
                    .offset = 0,
                    .data = {.address = &push, .size = sizeof(push)},
                });

                // Single-sided surfaces are invisible from behind, so the GPU can skip their back faces before running the fragment shader.
                commands.setCullMode(double_sided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack);
                commands.setFrontFace(mirrored ? mirrored_front_face : front_face);
                draw_list(commands, culling, phase, list);
            }
        }
    }

    // Draws the terrain patches cull phase `phase` kept, with `pipeline`: one mesh dispatch, after the terrain's cull mode and front face, the same as any single-sided, unmirrored surface's. The push data names the phase's patches as what's drawn.
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
            .drawn = culling.phases[static_cast<std::size_t>(phase)].patches.address,
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

        const PushData sky_push{.frame = draws.frame, .drawn = 0};  // one triangle, nothing to look up
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

        // How far a level of detail's error may project, in pixels, before a finer level takes over: 1, which no eye catches. 2 halves the triangles for a visible softening; 0.5 doubles them.
        constexpr float cluster_error_pixels = 1.0f;

        // Shadow rays for point and spot lights: 0 traces one for every light that reaches a pixel, exactly. 1 to 16 trace only that many, for the lights that give the pixel the most light; the rest light it unshadowed. Cheaper where many lights overlap, but light from the fainter ones can leak through walls.
        constexpr std::uint32_t shadow_ray_budget = 0;

        // Directional lights first: they reach everywhere, so they're not clustered, and the shader takes the first directional_count. The rest keep their order.
        const auto directional_end = std::ranges::stable_partition(scene.lights,
            [](const Light &light) { return light.type == LightType::directional; });
        const auto directional_count = static_cast<std::uint32_t>(directional_end.begin() - scene.lights.begin());

        // The terrain's samples and bounds, on the GPU; its plateau is at -1 m in the file. The 8 km field repeats 8 times each way, 64 km, to the horizon and past it.
        constexpr std::uint32_t terrain_tiles = 8;
        const std::uint64_t terrain_start = SDL_GetTicksNS();
        const Terrain terrain = load_terrain(device, *gpu, queue, command_pool, terrain_dir / "field.json", scene_origin, terrain_tiles,
            static_cast<float>(scene.bounds_min.y) - 0.01f + 1.0f, terrain_materials, terrain_materials + 1, 4.0f);

        std::println("Terrain: {} x {} samples, {} patch levels, {} patches in {:.0f} ms", terrain.data.samples, terrain.data.samples,
            terrain.patch_levels, terrain.patch_count, static_cast<double>(SDL_GetTicksNS() - terrain_start) * 1e-6);

        // Twelve dragons in a ring around the plateau, 100 m to 4 km out, standing on the terrain: a million triangles each at full detail, so the level of detail has something to work with. Artec3D's scan is in centimetres, its feet 1.014 m under its origin.
        {
            const std::filesystem::path dragon_file = std::filesystem::path(ASSET_DIR) / "dragon_with_pearl/scene.gltf";
            std::vector<glm::dmat4> placements;

            for (int i = 0; i < 12; ++i) {
                const double angle = glm::radians(30.0 * i);
                const double distance = 100.0 * std::pow(40.0, i / 11.0);
                const glm::dvec3 at = scene_origin + glm::dvec3{distance * std::cos(angle), 0.0, distance * std::sin(angle)};
                const double height = terrain_height_at(terrain, at) + 1.014;
                placements.push_back(glm::scale(glm::rotate(glm::translate(glm::dmat4{1.0}, glm::dvec3{at.x, height, at.z}),
                    -angle, glm::dvec3{0.0, 1.0, 0.0}), glm::dvec3{0.01}));
            }

            add_gltf(scene, dragon_file, placements);
        }

        std::println("With the dragons: {} vertices, {} triangles, {} primitives, {} draws", scene.vertices.size(),
            scene.indices.size() / 3, scene.primitives.size(), scene.draws.size());

        // Each draw's matrices, triangles and box, and what the cull needs to order it. The normal matrix is the transposed inverse of the model matrix: under non-uniform scale, transforming a normal by the model matrix itself would tilt it off the surface.
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
                .primitive = draw.primitive,
            });
            cull_draws.push_back(CullDraw{.list = list, .primitive = draw.primitive});
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

        // Every primitive's clusters and its level-of-detail hierarchy.
        const std::uint64_t cluster_start = SDL_GetTicksNS();
        const Clusters clusters = build_clusters(device, *gpu, queue, command_pool, scene);

        std::println("Clusters: {} in {} groups, {} nodes, {} levels, in {:.0f} ms", clusters.cluster_count, clusters.group_count,
            clusters.node_count, clusters.levels, static_cast<double>(SDL_GetTicksNS() - cluster_start) * 1e-6);

        for (std::size_t level = 0; level < clusters.level_triangles.size(); ++level) {
            std::println("  level {}: {} triangles", level, clusters.level_triangles[level]);
        }

        // The cull, its order and its draw lists, worked out for this scene's draws, the hierarchies it walks and the terrain it picks patches of.
        const DrawCulling culling = create_draw_culling(device, *gpu, queue, command_pool, cull_draws, terrain, clusters);

        std::println("Culling: {} draws, {} walk levels", culling.draw_count, culling.walk_levels);

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
            const std::uint32_t clusters_drawn = totals.early.clusters + totals.late.clusters;
            const std::uint32_t triangles_drawn = totals.early.triangles + totals.late.triangles;

            // The title shows the view, the sky, the time, the exposure, whether ambient occlusion is on and what the cull kept, whenever one changes.
            if (settings.view != shown_settings.view || settings.sky != shown_settings.sky
                || settings.hours != shown_settings.hours
                || settings.exposure_compensation != shown_settings.exposure_compensation
                || settings.ambient_occlusion != shown_settings.ambient_occlusion
                || clusters_drawn != shown_totals.early.clusters + shown_totals.late.clusters || totals.late.clusters != shown_totals.late.clusters
                || triangles_drawn != shown_totals.early.triangles + shown_totals.late.triangles
                || totals.early.subpixel_draws != shown_totals.early.subpixel_draws
                || totals.early_patches != shown_totals.early_patches || totals.late_patches != shown_totals.late_patches) {
                const auto minutes = static_cast<int>(std::lround(settings.hours * 60.0f));
                const std::string sky = settings.sky == SkySource::atmosphere
                    ? std::format("sky {:02}:{:02}", minutes / 60, minutes % 60)
                    : std::string("photographed sky");
                const std::string title = std::format("game-engine: {}, {}, EV {:.1f}, AO {}, {} clusters ({} late), {} triangles, {} patches ({} late), {} sub-pixel draws",
                    view_names[static_cast<std::size_t>(settings.view)], sky, ev100, settings.ambient_occlusion ? "on" : "off",
                    clusters_drawn, totals.late.clusters, triangles_drawn, totals.early_patches + totals.late_patches, totals.late_patches,
                    totals.early.subpixel_draws);

                SDL_SetWindowTitle(window.get(), title.c_str());
                shown_settings = settings;
                shown_totals = totals;
            }

            // Every two seconds or so, the clusters drawn per level of detail, for setting cluster_error_pixels against a frame budget: level 0 is full detail.
            if (frame_count % 120 == 0 && clusters_drawn != 0) {
                std::string levels;
                for (std::uint32_t level = 0; level < std::min<std::uint32_t>(clusters.levels, 16); ++level) {
                    levels += std::format(" {}", totals.early.level_clusters[level] + totals.late.level_clusters[level]);
                }
                std::println("Clusters per level:{}; {} triangles", levels, triangles_drawn);
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

            // The clusters' level of detail: a group's error of e metres at distance d projects to e x focal / d pixels, and a group is coarse enough once that's under cluster_error_pixels. The shader compares e x focal / cluster_error_pixels with d. A draw is under a pixel across when its sphere's diameter, 2r, projects to less than one: 2r x focal < d.

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
                .clusters = clusters.clusters.address,
                .cluster_vertices = clusters.vertices.address,
                .cluster_triangles = clusters.triangles.address,
                .cluster_groups = clusters.groups.address,
                .cluster_nodes = clusters.nodes.address,
                .primitives = clusters.primitives.address,
                .cluster_error_scale = focal / cluster_error_pixels,
                .subpixel_scale = 2.0f * focal,
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

## 19.7 The dragon

### Why
The box field shows clusters on the meshes you'll actually have: a few thousand triangles, every bevel a seam. A scan of a million triangles with no seams at all shows what the hierarchy does with its full range.

### How
- **Artec3D's "Dragon with pearl"**, free under CC-BY-4.0, which asks for this credit wherever it's shared: *This work is based on "Dragon with pearl" (<https://sketchfab.com/3d-models/dragon-with-pearl-93d65f56fdd34311ad55112f90ba4a82>) by Artec3D (<https://sketchfab.com/Artec3D>) licensed under CC-BY-4.0 (<http://creativecommons.org/licenses/by/4.0/>).* Download it from that page as **glTF** and unpack it into `lecture-md/game-engine/assets/dragon_with_pearl/`, so that `scene.gltf` and `scene.bin` are there: 10 primitives, 999,634 triangles, 648,060 vertices (the page says 500,000: the export cuts the mesh into ten pieces and repeats the vertices along the cuts), positions and normals only, one grey material, in centimetres.
- **Twelve of them** stand round the plateau (19.6).

## 19.8 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The terminal** shows the dragons loading, `With the dragons: 653128 vertices, 1005902 triangles, 11 primitives, 16499 draws`, then each primitive's chain: `primitive 0: 116 clusters, 6 levels, 6268 triangles at the finest, 289 at the coarsest` for the box, and 11 to 13 levels for each of the dragon's ten parts, from about a hundred thousand triangles down to a few dozen; then `Clusters: 18307 in 1228 groups, 1435 nodes, 13 levels` and the build's time, a few seconds in debug, and `Culling: 16499 draws, 8 walk levels`.
- **The image** at the start is Chapter 18's, pixel for pixel on the boxes: at this distance they're at full detail.
- **The title** starts at `22490 clusters (0 late), 2670491 triangles, 1434 patches (0 late), 0 sub-pixel draws`.
- **Walk to a dragon:** the nearest is 100 m east, through the field. Close up it's a million triangles of scan; step back and the title's triangles fall as the levels change, and nothing pops or cracks.
- **Fly up** 40 m and look across the field: the boxes beyond a few dozen metres go to their coarser levels; from 9 km away the whole field is a few thousand sub-pixel draws, each its coarsest level.
- **Every two seconds** the terminal prints `Clusters per level:` with a count per DAG level, and the triangles.
- **No `[validation …]` lines.**

**What it costs.** Release, 1920 × 1080, RTX 5070 Laptop, by day:

| View | Clusters | Triangles | Frame | Chapter 18's frame |
|---|---|---|---|---|
| Among the boxes, at eye level | 22,490 | 2.67 M | 3.2 ms | 5.2 ms |
| The overlook, 1.2 km out and 240 m up | 143 | 14,000 | 2.4 ms | 2.6 ms |
| Toward the nearest dragon, 100 m | 22,245 | 2.63 M | 3.3 ms | — |
| 40 m up, across the field | 233,106 | 22.0 M | 14.4 ms | — |
| 9 km away, 2.5 km up | 73,403 | 7.11 M | 6.8 ms | 20.7 ms |
| The inspection scene, 17 dragons at 4 to 44 m | 83,383 | 9.58 M | 8.3 ms | — |

- **The walk** costs 0.2 ms early and 0.1 ms late, in eight levels of four steps each: about a hundred small dispatches a frame. The pyramid and the terrain are as before.
- **Among the boxes,** the triangles fall from Chapter 18's 14 million to 2.7 million, and the frame from 5.2 to 3.2 ms, with the image pixel for pixel the same: the boxes in front stay at full detail, and the LOD takes the rows behind.
- **From 9 km** the frame falls from 20.7 to 6.8 ms. Nearly 10,000 boxes are under a pixel across, each still its coarsest level of 289 triangles: 2.9 million triangles for a few thousand pixels. That's the far form's job, in Chapter 21.
- **From 40 m up** the field is 22 million triangles and 14 ms: more than from 9 km, since the boxes are a few metres away and the shading error keeps them fine. The weights in `clusters.cpp` are the knob: halve them and the triangles fall, and the shading of the nearest rows begins to soften before it should.
- **The dragons** at 4 to 44 m cost 9.6 million triangles together: with a surface that bumpy, a millimetre of simplification turns normals enough to count, and the levels change later than the geometry alone would allow. The chain still takes the dragon from a million triangles to a few hundred by 4 km.

Next, in Chapter 20, the world at scale: two million placements of twenty models, in the terrain's cells, with the cull walking cells before draws, and the scans across workgroups carrying the load.

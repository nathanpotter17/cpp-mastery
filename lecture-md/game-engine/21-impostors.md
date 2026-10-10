# Chapter 21: Impostors

By the end of this chapter, a placement too small on screen for its clusters is drawn as an **impostor**: one quad facing the camera, textured from pictures of its model. The pictures are baked when the engine starts, by the engine's own mesh pipeline, from 64 directions over the upper hemisphere laid out on a hemi-octahedron, so that any direction the camera sees a placement from falls between three of them, and the quad blends those three. A picture is not a shading but a G-buffer: albedo and coverage, the normal with roughness and metallic, and depth, so an impostor is lit by the frame's sun, sky, ambient occlusion and shadows like every other surface, its trunk meets the ground where the mesh's did, and nothing in the atlases depends on the lighting. The idea is old, textured stand-ins for distant geometry (Maciel and Shirley 1995), and this form of it, pictures over an octahedron relit from stored normals and depth, is what current engines bake for distant trees (Brucks, "Octahedral Impostors", 2018, which Unreal's impostor baker grew from).

The walk gains one decision: a placement whose sphere projects under `impostor_pixels` across, 32 by default, isn't walked into its hierarchy at all. It goes to the clusters as itself, into a thirteenth draw list, and a mesh shader draws 32 of them per workgroup. On the hilltop of Chapter 20, 131,000 placements that were 15.7 million triangles become 131,000 quads at 262,000, and a hillside two kilometres off keeps its canopies green where the clusters' leaf cards had collapsed to specks.

The bake is a second of GPU time at every start and nothing on disk. There is no second renderer: the bake draws with `meshMain`, the real materials, textures and descriptor heap, and only the fragment stage differs, because it stores instead of shades. The directions and image axes of the 64 pictures are a table the bake writes and the shaders read, so the two can't disagree. And the bake is checked, not eyeballed: a capture with the threshold forced to zero and one with it forced to infinity draw the same views as clusters and as impostors, and the chapter compares them.

This chapter builds on [Chapter 20](20-placements.md).

## 21.1 What a picture holds: `shader_types.h`, `shared.slangh`, `mips.slang`

### Why
The impostor's data is the atlases' heap slots per model, the table of directions, and two numbers in the frame: the threshold, as a scale like the sub-pixel test's, and the grid's size. The counters gain a list, and the mips a kind.

### How
- **`ModelAtlas`** (12): a model's three atlases as resource heap slots: albedo with coverage in alpha; the surface, the normal octahedrally encoded in RG, roughness in B, coverage in A; the rest, depth in R, metallic in G, coverage in A. Three RGBA8 images rather than one wide one, so each goes through the mip steps as what it is.
- **`ImpostorFrame`** (36): a direction the model was baked from, toward the viewer in the model's space, and the picture's right and up axes there.
- **`FrameData`** points at the atlases and the frames, and carries `impostor_scale`, twice the focal length over the threshold, like `subpixel_scale` with its pixel, and `impostor_grid`, 8.
- **`WalkCounters::list_counts`** has 13 entries: the twelve lists of Chapter 19 and the impostors'.
- **`MipKind::data_see_through`**: plain numbers weighted by coverage, for the surface and the rest: outside the silhouette a texel is zero and must not bleed into the average.
- **In Slang,** the same, with `impostor_frames` and `atlases` as pointers.

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

// Placements

// Everything drawn is a placement of a model (placements.h): where in the world, how it's turned, how big, and which model. A glTF node with a mesh is one placement; a node with EXT_mesh_gpu_instancing is one per instance. The shaders build the model matrix from these (shared.slangh's instance_of): the rotation times the scale, then the offset in the cell.
struct Placement {
    glm::ivec3 cell;            // the world cell the placement's origin is in
    glm::vec3 offset;           // where in the cell, in metres from its corner
    glm::vec4 rotation;         // a unit quaternion, (x, y, z, w) as glTF stores it
    glm::vec3 scale;            // along the model's axes; a negative one mirrors
    std::uint32_t model;        // index into the models
};

static_assert(sizeof(Placement) == 56);
static_assert(offsetof(Placement, rotation) == 24);
static_assert(offsetof(Placement, model) == 52);

// A model: a glTF mesh, its primitives one after another among the primitives, and the sphere around all of them in its own space. Shadow rays meet it as its BLAS (acceleration.h).
struct Model {
    glm::vec3 center;
    float radius;
    std::uint32_t first_primitive;
    std::uint32_t primitive_count;
    vk::DeviceAddress blas;     // its bottom-level acceleration structure
};

static_assert(sizeof(Model) == 32);
static_assert(offsetof(Model, blas) == 24);

// What the shaders need of a primitive: its material, and where its triangles are, for a shadow ray's alpha test (shading.slangh).
struct PrimitiveData {
    std::uint32_t material;     // index into the material buffer
    std::uint32_t first_index;  // where the primitive's indices start in the index buffer
    std::int32_t vertex_offset; // added to each index: where its vertices start in the vertex buffer
};

static_assert(sizeof(PrimitiveData) == 12);

// A column of world cells that holds placements (placements.h). The placements are sorted by column, so a column's are a run. Its box, measured from `cell`'s corner, holds every one of them whole; `max_radius` is the largest among them, for dropping a whole column that's under a pixel.
struct PlacementCell {
    glm::ivec3 cell;            // the column's x and z; the y of its first placement
    glm::vec3 bounds_min;
    glm::vec3 bounds_max;
    float max_radius;
    std::uint32_t first;        // into the placements
    std::uint32_t count;
};

static_assert(sizeof(PlacementCell) == 48);
static_assert(offsetof(PlacementCell, first) == 40);

// Impostors (impostors.h)

// A model's impostor atlases, as resource heap slots: its albedo with coverage in alpha; its surface, the normal octahedrally encoded in RG, roughness in B, coverage in A; and the rest, depth in R, metallic in G, coverage in A.
struct ModelAtlas {
    std::uint32_t albedo;
    std::uint32_t surface;
    std::uint32_t extra;
};

static_assert(sizeof(ModelAtlas) == 12);

// One of the directions a model was baked from: toward the viewer, in the model's space, and the baked image's right and up axes there. The bake made the table; the shaders read it, so both agree by construction.
struct ImpostorFrame {
    glm::vec3 direction;
    glm::vec3 right;
    glm::vec3 up;
};

static_assert(sizeof(ImpostorFrame) == 36);

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
    vk::DeviceAddress placements;       // one Placement per placement (placements.h)
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
    vk::DeviceAddress models;           // one Model per model
    vk::DeviceAddress primitive_data;   // one PrimitiveData per primitive
    vk::DeviceAddress cells;            // the placement cells (placements.h)
    float mesh_draw_distance;           // metres: placements farther than this aren't drawn
    float shadow_radius;                // metres: placements within this cast ray-traced shadows
    std::uint32_t shadow_capacity;      // the most of them a frame's TLAS holds
    std::uint32_t placement_count;
    vk::DeviceAddress atlases;          // one ModelAtlas per model (impostors.h)
    vk::DeviceAddress impostor_frames;  // impostor_grid x impostor_grid ImpostorFrames
    float impostor_scale;               // a sphere of radius r at distance d is drawn as an impostor when r x this < d
    std::uint32_t impostor_grid;        // frames per side of the atlas
};

static_assert(sizeof(FrameData) == 496);
static_assert(offsetof(FrameData, atlases) == 472);
static_assert(offsetof(FrameData, models) == 432);
static_assert(offsetof(FrameData, mesh_draw_distance) == 456);
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

// What the cull walks and what it draws: a cell of placements, a placement, or a placement with a node or a cluster of one of its model's primitives. `what` holds the primitive in its top 8 bits, the kind in the 2 below, and the index in the 22 that are left: a node's or a cluster's, unused for a cell (whose index is in `placement`) or a placement.
struct ClusterItem {
    std::uint32_t placement;
    std::uint32_t what;
};

static_assert(sizeof(ClusterItem) == 8);

constexpr std::uint32_t item_index_bits = 22;
constexpr std::uint32_t item_kind_shift = 22;
constexpr std::uint32_t item_primitive_shift = 24;
constexpr std::uint32_t item_kind_cell = 0;
constexpr std::uint32_t item_kind_placement = 1;
constexpr std::uint32_t item_kind_node = 2;
constexpr std::uint32_t item_kind_cluster = 3;

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
    std::uint32_t list_counts[13];  // cluster items per draw list; the thirteenth is the impostors
    std::uint32_t level_clusters[16];  // cluster items per DAG level
    std::uint32_t far_placements;   // placements past mesh_draw_distance, whole cells of them included
    std::uint32_t level_items[16];  // items walked per level of the walk
    std::uint32_t dropped_items;    // items a level, the candidates or the clusters had no room for
};

static_assert(sizeof(WalkCounters) == 224);
static_assert(offsetof(WalkCounters, far_placements) == 152);
static_assert(offsetof(WalkCounters, dispatch_x) == 24);
static_assert(offsetof(WalkCounters, list_counts) == 36);

// GPU culling (culling.h, cull.slang)

// Where all of one cull phase's buffers are, in one table its steps read. The two phases share the order and the terrain's flags, and have the rest each.
struct CullTables {
    vk::DeviceAddress cells;          // the placement cells: the early phase's first items
    vk::DeviceAddress items[2];       // the walk's levels, alternating
    vk::DeviceAddress results;        // per item of the level: what it produced (three counts)
    vk::DeviceAddress block_totals;   // per workgroup of the level: the sums of those
    vk::DeviceAddress block_bases;    // prefix sums of block_totals
    vk::DeviceAddress counters;       // the WalkCounters
    vk::DeviceAddress candidates;     // items set aside for the late phase
    vk::DeviceAddress cluster_items;  // the clusters to draw, in the walk's order
    vk::DeviceAddress sorted_items;   // the same, sorted by draw list: what the dispatches draw
    vk::DeviceAddress dispatches;     // one ClusterDispatch per draw list
    vk::DeviceAddress early_candidates;  // the early phase's candidates: the late phase's first items
    vk::DeviceAddress early_counters;    // the early phase's counters: how many
    vk::DeviceAddress patches;        // the terrain patches this phase draws: quadtree node indices (terrain.h)
    vk::DeviceAddress patch_command;  // one VkDrawMeshTasksIndirectCommandEXT: how many patches, 1, 1
    vk::DeviceAddress nodes;          // the terrain selection's working lists: two runs of max_terrain_patches
    vk::DeviceAddress early_patches;  // per terrain patch: the frame the early phase last drew it in
    std::uint32_t cell_count;
    std::uint32_t item_capacity;      // the most items a level, the candidates or the cluster items can hold
};

static_assert(sizeof(CullTables) == 144);
static_assert(offsetof(CullTables, items) == 8);
static_assert(offsetof(CullTables, cell_count) == 136);

// Push data of shaders/shadows.slang: the frame, and this frame's ray-tracing instances to write and count.
struct ShadowPushData {
    vk::DeviceAddress frame;
    vk::DeviceAddress instances;  // shadow_capacity VkAccelerationStructureInstanceKHR
    vk::DeviceAddress count;      // one uint32_t
};

static_assert(sizeof(ShadowPushData) == 24);

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
    data_see_through = 4,  // plain numbers with coverage in alpha: weighted by it
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

// Placements (src/includes/placements.h)

// Everything drawn is a placement of a model: where, how turned, how big, which. The shaders build its model matrix from these (instance_of, below).
struct Placement {
    int3 cell;          // the world cell the placement's origin is in
    float3 offset;      // where in the cell, from its corner
    float4 rotation;    // a unit quaternion (x, y, z, w)
    float3 scale;       // along the model's axes; a negative one mirrors
    uint model;
};

// A model: a glTF mesh, its primitives a run among the primitives, the sphere around them in its own space, and its BLAS.
struct Model {
    float3 center;
    float radius;
    uint first_primitive;
    uint primitive_count;
    uint64_t blas;
};

// What the shaders need of a primitive.
struct PrimitiveData {
    uint material;
    uint first_index;    // where its indices start
    int vertex_offset;   // added to each index
};

// A column of world cells and the run of placements in it, with the box around all of them from `cell`'s corner.
struct PlacementCell {
    int3 cell;
    float3 bounds_min;
    float3 bounds_max;
    float max_radius;    // the largest placement's sphere
    uint first;
    uint count;
};

// A model's impostor atlases (impostors.h), as resource heap slots: albedo with coverage in alpha; the surface, the normal octahedrally encoded in RG, roughness in B, coverage in A; the rest, depth in R, metallic in G, coverage in A.
struct ModelAtlas {
    uint albedo;
    uint surface;
    uint extra;
};

// A direction a model was baked from, toward the viewer in the model's space, and the baked image's right and up axes there.
struct ImpostorFrame {
    float3 direction;
    float3 right;
    float3 up;
};

// A VkAccelerationStructureInstanceKHR (shadows.slang): the instance's transform as three rows, its custom index in 24 bits under an 8-bit mask, its hit group offset under 8 bits of flags, and its BLAS. All zero is an inactive instance.
struct RayInstance {
    float4 row0;
    float4 row1;
    float4 row2;
    uint custom_mask;
    uint sbt_flags;
    uint64_t blas;
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

// What the cull walks and draws: a cell of placements (its index in `placement`), a placement, or a placement with a node or a cluster of one of its model's primitives. `what` holds the primitive in its top 8 bits, the kind in the 2 below, the node's or cluster's index in the 22 left.
struct ClusterItem {
    uint placement;
    uint what;
};

static const uint item_index_bits = 22;
static const uint item_kind_shift = 22;
static const uint item_primitive_shift = 24;
static const uint item_kind_cell = 0;
static const uint item_kind_placement = 1;
static const uint item_kind_node = 2;
static const uint item_kind_cluster = 3;

uint item_kind(ClusterItem item) {
    return (item.what >> item_kind_shift) & 3;
}

uint item_primitive(ClusterItem item) {
    return item.what >> item_primitive_shift;
}

uint item_index(ClusterItem item) {
    return item.what & ((1u << item_index_bits) - 1);
}

ClusterItem make_item(uint placement, uint kind, uint primitive, uint index) {
    return ClusterItem(placement, (primitive << item_primitive_shift) | (kind << item_kind_shift) | index);
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
    Placement *placements;             // one Placement per placement
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
    Model *models;                     // one Model per model
    PrimitiveData *primitive_data;     // one PrimitiveData per primitive
    PlacementCell *cells;              // the placement cells
    float mesh_draw_distance;          // metres: placements farther than this aren't drawn
    float shadow_radius;               // metres: placements within this cast ray-traced shadows
    uint shadow_capacity;              // the most of them a frame's TLAS holds
    uint placement_count;
    ModelAtlas *atlases;               // one per model
    ImpostorFrame *impostor_frames;    // impostor_grid x impostor_grid of them
    float impostor_scale;              // a sphere of radius r at distance d is an impostor when r x this < d
    uint impostor_grid;                // frames per side of the atlas
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

// Instances

// A placement as the shaders use it: its model matrix in two parts, the linear part, the rotation times the scale, and its origin relative to the camera; and the linear part's inverse, whose transpose is the normal matrix.
struct Instance {
    Placement placement;
    float3x3 linear;
    float3x3 inverse;
    float3 origin;
};

// A unit quaternion as a matrix: mul(m, v) turns v as the quaternion does.
float3x3 rotation_matrix(float4 q) {
    const float3 v = q.xyz;
    const float w = q.w;
    return float3x3(
        1.0 - 2.0 * (v.y * v.y + v.z * v.z), 2.0 * (v.x * v.y - w * v.z), 2.0 * (v.x * v.z + w * v.y),
        2.0 * (v.x * v.y + w * v.z), 1.0 - 2.0 * (v.x * v.x + v.z * v.z), 2.0 * (v.y * v.z - w * v.x),
        2.0 * (v.x * v.z - w * v.y), 2.0 * (v.y * v.z + w * v.x), 1.0 - 2.0 * (v.x * v.x + v.y * v.y));
}

// Placement `index`, ready to transform with. R x S scales each column of R by the scale along that axis; its inverse, S^-1 x R^T, divides each row of R^T by it.
Instance instance_of(FrameData *frame, uint index) {
    Instance instance;
    instance.placement = frame.placements[index];

    const float3x3 r = rotation_matrix(instance.placement.rotation);
    const float3x3 rt = transpose(r);
    const float3 s = instance.placement.scale;

    instance.linear = float3x3(r[0] * s, r[1] * s, r[2] * s);
    instance.inverse = float3x3(rt[0] / s.x, rt[1] / s.y, rt[2] / s.z);
    instance.origin = camera_relative(frame, instance.placement.cell, instance.placement.offset);
    return instance;
}

// A point of the model, relative to the camera.
float3 instance_point(Instance instance, float3 p) {
    return instance.origin + mul(instance.linear, p);
}

// A normal of the model in the world's axes, by the normal matrix; not yet normalized.
float3 instance_normal(Instance instance, float3 n) {
    return mul(transpose(instance.inverse), n);
}

// The most the placement stretches anything: a sphere's radius grows by this.
float instance_scale(Instance instance) {
    const float3 s = abs(instance.placement.scale);
    return max(s.x, max(s.y, s.z));
}

// Whether it mirrors: an odd number of negative scales reverses the order a triangle's corners appear in.
bool instance_mirrored(Instance instance) {
    const float3 s = instance.placement.scale;
    return s.x * s.y * s.z < 0.0;
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

In `game-engine/shaders/mips.slang`, replace the line `static const uint mip_normal = 3;       // normals: averaged as vectors` with:
```slang
static const uint mip_normal = 3;       // normals: averaged as vectors
static const uint mip_data_see_through = 4;  // plain numbers with coverage in alpha: weighted by it
```

In `game-engine/shaders/mips.slang`, replace `downsampleMain` with:
```slang
// One texel of the smaller level: the average of the source texels its area covers, a box filter. With an even size, that's a 2 x 2 block; with an odd one, the footprint is between 2 and 3 texels wide, and the texels it cuts get their share. The averages are of what the texels stand for:
//   colors       in linear light
//   see-through  in linear light, weighted by alpha: a texel that's transparent has no color to contribute, so the cut-out parts of a leaf texture don't darken its edges
//   normals      as vectors, with the average's spread in alpha
[shader("compute")]
[numthreads(8, 8, 1)]
void downsampleMain(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= push.target_size)) {
        return;
    }

    const float2 ratio = float2(push.source_size) / float2(push.target_size);
    const float2 start = float2(id.xy) * ratio;
    const float2 end = start + ratio;

    // Rounding can put the last footprint's end a hair past the level's edge.
    const uint2 first = uint2(start);
    const uint2 last = min(uint2(ceil(end)), push.source_size);

    float4 sum = 0.0;          // see-through: rgb weighted by alpha, and alpha
    float3 normal_sum = 0.0;
    float weight_sum = 0.0;

    for (uint y = first.y; y < last.y; ++y) {
        for (uint x = first.x; x < last.x; ++x) {
            const float weight = overlap(float(x), start.x, end.x) * overlap(float(y), start.y, end.y);
            const float4 texel = unpack(push.source[y * push.source_size.x + x]);

            if (push.kind == mip_color) {
                sum += weight * float4(srgb_to_linear(texel.rgb), texel.a);
            } else if (push.kind == mip_see_through) {
                sum += weight * float4(srgb_to_linear(texel.rgb) * texel.a, texel.a);
            } else if (push.kind == mip_normal) {
                normal_sum += weight * unpack_normal(texel);
            } else if (push.kind == mip_data_see_through) {
                sum += weight * float4(texel.rgb * texel.a, texel.a);
            } else {
                sum += weight * texel;
            }

            weight_sum += weight;
        }
    }

    float4 result;

    if (push.kind == mip_color) {
        const float4 average = sum / weight_sum;
        result = float4(linear_to_srgb(average.rgb), average.a);
    } else if (push.kind == mip_see_through) {
        // Fully transparent everywhere: no color to weight, so keep black.
        const float3 color = sum.a > 0.0 ? sum.rgb / sum.a : 0.0;
        result = float4(linear_to_srgb(color), sum.a / weight_sum);
    } else if (push.kind == mip_normal) {
        result = pack_normal(normal_sum / weight_sum);
    } else if (push.kind == mip_data_see_through) {
        result = float4(sum.a > 0.0 ? sum.rgb / sum.a : 0.0, sum.a / weight_sum);
    } else {
        result = sum / weight_sum;
    }

    push.target[id.y * push.target_size.x + id.x] = pack(result);
}
```

## 21.2 The bake: `pipeline.h`, `pipeline.cpp`, `mesh.slang`, `clusters.h`, `clusters.cpp`, `texture.h`, `texture.cpp`, `impostors.h`, `impostors.cpp`

### Why
A baker is where engines accumulate a second renderer, a file format and a cache that goes stale. This one is none of those. It's a fourth pass of the mesh pipeline, `MeshPass::bake`, with three colour attachments and a fragment shader that writes what it would have shaded with; it runs at load, in one submission, after the textures and before the first frame; and it keeps its atlases in GPU memory for the run. What the pictures show is whatever the mesh looks like, because they are drawn by exactly what draws the mesh.

The pictures' directions come from a hemi-octahedron (Cigolle et al. 2014): the upper hemisphere folded onto a square, so an 8 × 8 grid over the square is 64 directions spread evenly over the sky, and any direction lands in a triangle of three of them whose barycentric coordinates are the blend. The bake decodes grid points to directions in C++; the shader encodes a direction to a grid point, six lines that are the exact inverse. Everything else about a picture, its direction and its two axes, is a table uploaded from the C++ that baked with it.

### How
- **`MeshPass::bake`** in `create_scene_pipeline`: three attachments, all replacing, depth written and tested like the prepass's. `create_mesh_pipeline` picks `bakeMain` for it.
- **`bakeMain`** (`mesh.slang`): the mesh pipeline's fragment stage for the bake. The base colour with the material's alpha test, the normal through `surface_normal` with the normal map, roughness and metallic from the material; the bake draws the model at the origin unturned, so the normal is already in the model's space. Depth is 1 minus the reverse-Z depth: the bake's projection makes 1 the camera's side of the sphere and 0 the far side, so what's stored is how far into the sphere the surface is, 0 to 1 across its diameter. Every output's alpha is 1: coverage, with the cleared 0 everywhere nothing was drawn.
- **The finest clusters.** `Clusters` keeps, per primitive, where its clusters start and how many of the first are at full detail, which is what the bake draws: clodBuild hands groups over finest level first, so they're the first ones.
- **`generate_image_mips`** (`texture.h`): Chapter 6's mip steps for images that already exist on the GPU. `texture.cpp` is rearranged so the planning of a chain, the pipelines and buffers, the recording of the levels, and the copy back are functions the scene's textures and the atlases share. An atlas's top level is copied out of the image into the mip buffer, the levels below made as the scene textures' are, and all of them copied back. The albedo atlas goes through as a masked texture with a cutoff of a half, so a tree's canopy keeps its coverage at every level, where a plain average would thin it to nothing by the fourth; the other two as data weighted by coverage.
- **`bake_impostors`** (`impostors.cpp`), in order:
  1. **The frames:** row j, column i of the grid is frame j × 8 + i, its direction from the hemi-octahedron and its axes from `make_frame`: right across the view, up along the world's up, or along z for the view straight down, which has no up.
  2. **What to draw:** one placement per model at the origin, unturned and unscaled, so the model's space is the camera's; and per model and alpha mode, its finest clusters as items with a dispatch over them, like a draw list's. Masked and blended bake alike, cut out at the cutoff.
  3. **A `FrameData` per model and frame:** the given pointers, the bake's placements, the camera at its cell's origin, and `frame_view_projection`: the camera on the model's sphere along the frame's direction looking at the centre, the frame's axes as the image's, orthographic over the sphere's width, reverse-Z from 1 at the camera to 0 at the far side, and Vulkan's downward clip y so the picture is the right way up.
  4. **The atlases,** 1024² RGBA8, sRGB for the albedo, with a view of their top level alone, which an attachment must be, and a sampled descriptor over all their levels; one depth buffer the tiles share; and the `ModelAtlas` table.
  5. **The bake pipelines,** one per solid alpha mode.
  6. **The bake:** per model, the atlases and the depth cleared to zero, then each of the 64 tiles drawn with its viewport and scissor, both modes, no culling, since every side of every triangle is a surface from somewhere; then the top levels to the transfer layout.
  7. **The mips.**
- **Memory:** three atlases with mips are 16.8 MB per model, 134 MB for eight. `impostor_atlas_size` is the knob: 2048 is four times that, for tiles of 256 pixels, if the threshold is ever raised far past 32.

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
std::vector<std::uint32_t> read_spirv(const std::filesystem::path &path);

// The passes the scene is drawn in, and the images each draws into.
enum class MeshPass {
    depth_normals,  // the prepass, for opaque and masked materials: the normals
    lighting,       // the full shading of opaque and masked materials: the HDR image
    transparency,   // blended materials, into weighted blended transparency's two sums
    bake,           // the impostor bake (impostors.h): albedo, surface and the rest, into a model's three atlases
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

// Draws shaders/impostor.slang's quads into the same images as create_mesh_pipeline's `pass`, the prepass or the lighting pass: a mesh shader makes 32 placements' quads per workgroup. Impostors are never see-through and have no alpha mode: they cut out their coverage.
vk::raii::Pipeline create_impostor_pipeline(
    const vk::raii::Device &device,
    std::span<const vk::Format> color_formats,
    vk::Format depth_format,
    MeshPass pass
);

// A compute pipeline running `entry_point` from shaders/<shader>.spv.
vk::raii::Pipeline create_compute_pipeline(const vk::raii::Device &device, const char *shader, const char *entry_point);
```

In `game-engine/src/pipeline.cpp`, replace `create_scene_pipeline` with:
```cpp
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
        const bool bake = pass == MeshPass::bake;

        // The transparency pass draws into its two sums, the bake into a model's three atlases, every other pass into one image.
        if (color_formats.size() != (transparency ? 2 : bake ? 3 : 1)) {
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
        //   - The prepass keeps a fragment only if it's nearer than what's there, and records its depth: the depth buffer ends up holding the nearest solid surface at every pixel. The bake does the same into its own depth buffer.
        //   - The lighting pass writes no depth. Its solid surfaces pass "greater or equal" only where they are that nearest surface, so each pixel is shaded once.
        //   - The transparency pass writes none either: its see-through surfaces pass wherever they're in front of the nearest solid one.
        const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = prepass || bake ? vk::True : vk::False,
            .depthCompareOp = prepass || bake ? vk::CompareOp::eGreater : vk::CompareOp::eGreaterOrEqual,
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
        const std::array replace_three{replace, replace, replace};

        const vk::PipelineColorBlendStateCreateInfo color_blend{
            .attachmentCount = static_cast<std::uint32_t>(color_formats.size()),
            .pAttachments = transparency ? sums.data() : replace_three.data(),
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
```

Then replace `create_mesh_pipeline` with:
```cpp
vk::raii::Pipeline create_mesh_pipeline(
    const vk::raii::Device &device,
    std::span<const vk::Format> color_formats,
    vk::Format depth_format,
    AlphaMode alpha_mode,
    MeshPass pass
) {
    const bool prepass = pass == MeshPass::depth_normals;
    const bool transparency = pass == MeshPass::transparency;
    const bool bake = pass == MeshPass::bake;

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
            .pName = prepass ? "prepassMain" : transparency ? "transparentMain" : bake ? "bakeMain" : "fragmentMain",
            .pSpecializationInfo = &specialization,
        },
    };

    return create_scene_pipeline(device, stages, true, color_formats, depth_format, pass);
}
```

In `game-engine/src/pipeline.cpp`, add this section before `// The terrain pipeline`:
```cpp
// The impostor pipeline

vk::raii::Pipeline create_impostor_pipeline(
    const vk::raii::Device &device,
    std::span<const vk::Format> color_formats,
    vk::Format depth_format,
    MeshPass pass
) {
    if (pass != MeshPass::depth_normals && pass != MeshPass::lighting) {
        throw std::invalid_argument("create_impostor_pipeline: impostors are drawn by the prepass and the lighting pass only");
    }

    const std::vector<std::uint32_t> spirv = read_spirv(std::filesystem::path(SHADER_DIR) / "impostor.spv");

    const vk::raii::ShaderModule module(device, vk::ShaderModuleCreateInfo{
        .codeSize = spirv.size() * sizeof(std::uint32_t),
        .pCode = spirv.data(),
    });

    // A mesh shader stage in place of the vertex stage: each workgroup writes 32 placements' quads.
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
```

In `game-engine/shaders/mesh.slang`, add this section before `// Weighted blended transparency`:
```slang
// The impostor bake

// What a tile of the impostor atlases holds per texel (impostors.h): the base colour and coverage; the normal, octahedrally encoded, with roughness; and depth, with metallic. The bake draws a model at the origin, so the normal is in the model's space, and depth is how far into the sphere the surface is: 1 minus the reverse-Z depth, which the bake's projection makes 1 at the camera and 0 at the sphere's far side.
struct BakeOutput {
    float4 albedo : SV_Target0;
    float4 surface : SV_Target1;
    float4 extra : SV_Target2;
};

[shader("fragment")]
BakeOutput bakeMain(VertexOutput input, bool front_face : SV_IsFrontFace, uint primitive : SV_PrimitiveID) {
    FrameData *frame = push.frame;
    const Instance instance = instance_of(frame, input.placement);
    const Material material = frame.materials[frame.primitive_data[input.primitive].material];
    const Triangle triangle = fetch_triangle(frame, instance, input.cluster, primitive);

    const float4 base_color = material.base_color_factor * sample_slot(material.base_color, input) * input.color;

    if (alpha_mode != alpha_opaque && base_color.a < material.alpha_cutoff) {
        discard;
    }

    const float4 metallic_roughness = sample_slot(material.metallic_roughness, input);
    float map_spread;
    const float3 normal = surface_normal(input, instance, material, front_face, true, triangle_normal(triangle), map_spread);

    BakeOutput output;
    output.albedo = float4(base_color.rgb, 1.0);
    output.surface = float4(encode_octahedral(normal) * 0.5 + 0.5, material.roughness_factor * metallic_roughness.g, 1.0);
    output.extra = float4(1.0 - input.position.z, material.metallic_factor * metallic_roughness.b, 0.0, 1.0);
    return output;
}
```

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
    std::vector<std::uint32_t> first_clusters;   // per primitive, where its clusters start
    std::vector<std::uint32_t> finest_counts;    // per primitive, how many of its clusters, the first ones, are at full detail: what the impostor bake draws
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

In `game-engine/src/clusters.cpp`, replace the line `std::uint32_t coarsest = 0;` with:
```cpp
        std::uint32_t coarsest = 0;
        std::uint32_t finest_count = 0;
```

In `game-engine/src/clusters.cpp`, replace the line `finest += depth == 0 ? cluster.triangle_count : 0;` with:
```cpp
            finest += depth == 0 ? cluster.triangle_count : 0;
            finest_count += depth == 0 ? 1 : 0;
```

In `game-engine/src/clusters.cpp`, add this section before `// How far simplification took the primitive`:
```cpp
        // The finest clusters come first: clodBuild hands groups over finest level first.
        result.first_clusters.push_back(first_cluster);
        result.finest_counts.push_back(finest_count);
```

`game-engine/src/includes/texture.h`:
```cpp
#pragma once

#include "includes/scene.h"
#include "includes/shader_types.h"
#include "includes/vulkan_setup.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

// Decoding

// Pixels decoded from a PNG or JPEG: 8-bit RGBA, row by row, top to bottom.
struct DecodedImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> rgba;
    std::string error{};  // why decoding failed, if it did
};

// Decodes every image in `images`, several at once, one per CPU core. An image that can't be decoded becomes a single magenta pixel, so it's easy to spot on screen, and its `error` says why.
std::vector<DecodedImage> decode_images(std::span<const SceneImage> images);

// GPU textures

// A sampled image with a full mip chain, in device-local memory, ready for shaders: its layout is eShaderReadOnlyOptimal. Members are destroyed bottom-up, so the image goes before its memory.
struct Texture {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image handle = nullptr;
    vk::Format format = vk::Format::eUndefined;
    vk::Extent2D extent;
    std::uint32_t mip_levels = 1;
};

// An image already on the GPU whose levels below the top are to be made: the impostor atlases (impostors.h). Its top level must be in eTransferSrcOptimal, the others in any layout; on return every level is in eShaderReadOnlyOptimal. The levels are made as the scene textures' are: `kind` says how they average, and a `cutoff` above 0 keeps the coverage of the texels above it at every level.
struct MipRequest {
    vk::Image image;
    vk::Extent2D extent;
    std::uint32_t mip_levels = 1;
    MipKind kind = MipKind::data;
    float cutoff = 0.0f;
};

void generate_image_mips(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::span<const MipRequest> requests
);

// The textures for a scene: index 0 is a 1x1 white texture, for materials without one, and scene image i is texture i + 1. All of them are uploaded and given mipmaps in a single submission.
std::vector<Texture> create_scene_textures(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    const Scene &scene
);
```

`game-engine/src/texture.cpp`:
```cpp
#include "includes/texture.h"

#include "includes/buffer.h"
#include "includes/pipeline.h"
#include "includes/shader_types.h"

#include <jpgd.h>
#include <spng.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <print>
#include <thread>

namespace {

    // Decoding

    // What a failed image becomes: one magenta pixel, impossible to miss.
    DecodedImage missing_image(std::string error) {
        return DecodedImage{.width = 1, .height = 1, .rgba = {255, 0, 255, 255}, .error = std::move(error)};
    }

    DecodedImage decode_jpeg(std::span<const std::uint8_t> encoded) {
        int width = 0;
        int height = 0;
        int components = 0;

        // Asking for 4 components gives RGBA, with alpha 255. jpgd allocates the pixels with malloc, so free() releases them.
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

        // SPNG_FMT_RGBA8 converts any PNG (grey, palette, 16-bit, ...) to 8-bit RGBA; SPNG_DECODE_TRNS turns a transparency chunk into real alpha.
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

    // Chooses the decoder from the file's first bytes, its "magic number", which is more reliable than the file name or the glTF mimeType.
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

    // Uploading

    // Moves mip levels [base, base + count) of `image` between layouts.
    void transition_mips(
        const vk::raii::CommandBuffer &commands,
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

    // Makes writes by `src_stage` visible to `dst_stage`, for every buffer.
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

    // An image for `decoded` with room for every mip level, in device-local memory.
    Texture create_texture(const vk::raii::Device &device, const GpuChoice &gpu, const DecodedImage &decoded, vk::Format format) {
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
            // Every level is copied in from the mip buffer, then sampled.
            .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
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

    // Mip chains

    // One image's levels in the mip buffer, and how they're made.
    struct MipChain {
        std::vector<glm::uvec2> sizes;          // each level's, in texels
        std::vector<vk::DeviceSize> offsets;    // each level's, in bytes into the mip buffer
        MipKind kind = MipKind::data;
        float cutoff = 0.0f;                    // a masked base color's: the texture alpha that's kept; 0 for none
        vk::DeviceSize coverage = 0;            // masked: where its histograms, then its scales, start in the coverage buffer
    };

    // The chain of an image `size` across: its levels halve down to one texel, laid out from `offset` on, which moves past them.
    MipChain plan_chain(glm::uvec2 size, MipKind kind, float cutoff, vk::DeviceSize &offset) {
        MipChain chain;
        chain.kind = kind;
        chain.cutoff = cutoff;
        const std::uint32_t levels = std::bit_width(std::max(size.x, size.y));

        for (std::uint32_t level = 0; level < levels; ++level) {
            chain.sizes.push_back(size);
            chain.offsets.push_back(offset);
            offset += vk::DeviceSize{size.x} * size.y * 4;
            size = glm::max(size / 2u, glm::uvec2{1});
        }

        return chain;
    }

    // What each image is used for, from the materials: how its mips average (MipKind), and, for a masked material's base color, the alpha its texels must reach to be kept. Texture i + 1 is image i; texture 0 is white.
    std::vector<MipChain> plan_mips(const Scene &scene, std::span<const DecodedImage> images) {
        std::vector<bool> as_normal(images.size(), false);
        std::vector<bool> as_other(images.size(), false);        // anything but a normal map
        std::vector<bool> as_see_through(images.size(), false);  // a masked or blended base color
        std::vector<bool> as_opaque_color(images.size(), false); // any other color: its alpha means nothing
        std::vector<float> cutoffs(images.size(), 0.0f);

        for (const SceneMaterial &material : scene.materials) {
            for (const TextureRef *ref : {&material.base_color, &material.metallic_roughness, &material.occlusion, &material.emissive}) {
                if (ref->image >= 0) {
                    as_other[static_cast<std::size_t>(ref->image) + 1] = true;
                }
            }

            if (material.normal.image >= 0) {
                as_normal[static_cast<std::size_t>(material.normal.image) + 1] = true;
            }

            if (material.base_color.image >= 0) {
                const auto image = static_cast<std::size_t>(material.base_color.image) + 1;
                (material.alpha_mode == AlphaMode::opaque ? as_opaque_color : as_see_through)[image] = true;
            }

            if (material.emissive.image >= 0) {
                as_opaque_color[static_cast<std::size_t>(material.emissive.image) + 1] = true;
            }

            // The material keeps a pixel where factor x texture x vertex color reaches the cutoff; with vertex color 1, the texture's alpha must reach cutoff / factor. Above 1, nothing is kept at any level.
            const float factor = material.base_color_factor.a;
            if (material.alpha_mode == AlphaMode::mask && material.base_color.image >= 0 && factor > 0.0f) {
                float &cutoff = cutoffs[static_cast<std::size_t>(material.base_color.image) + 1];
                const float wanted = material.alpha_cutoff / factor;

                // A texture shared by masked materials with different cutoffs keeps the first one's coverage.
                if (cutoff == 0.0f && wanted > 0.0f && wanted <= 1.0f) {
                    cutoff = wanted;
                }
            }
        }

        std::vector<MipChain> chains;
        vk::DeviceSize offset = 0;

        for (std::size_t i = 0; i < images.size(); ++i) {
            const bool srgb = i == 0 || scene.images[i - 1].srgb;

            // Colors weight by alpha only where every use takes alpha as coverage or opacity: an opaque material shows its texels whatever their alpha. An image that's a normal map and something else too is averaged as plain data: its alpha may mean something.
            const MipKind kind = srgb
                ? (as_see_through[i] && !as_opaque_color[i] ? MipKind::see_through : MipKind::color)
                : (as_normal[i] && !as_other[i] ? MipKind::normal : MipKind::data);

            chains.push_back(plan_chain({images[i].width, images[i].height}, kind, cutoffs[i], offset));
        }

        return chains;
    }

    // mips.slang's pipelines, and the buffers the levels and the coverage counts live in while they're made.
    struct MipBuild {
        vk::raii::Pipeline prepare_normals = nullptr;
        vk::raii::Pipeline downsample = nullptr;
        vk::raii::Pipeline histogram = nullptr;
        vk::raii::Pipeline solve_coverage = nullptr;
        vk::raii::Pipeline scale_alpha = nullptr;
        Buffer mips;      // every level of every chain, where the compute shaders make the levels below the top through its address
        Buffer coverage;  // masked chains' per-level 256-bin alpha histograms, then their scales
    };

    // Room for `chains`, whose masked ones are given their place in the coverage buffer.
    MipBuild create_mip_build(const vk::raii::Device &device, const GpuChoice &gpu, std::span<MipChain> chains) {
        MipBuild build;
        build.prepare_normals = create_compute_pipeline(device, "mips", "prepareNormalsMain");
        build.downsample = create_compute_pipeline(device, "mips", "downsampleMain");
        build.histogram = create_compute_pipeline(device, "mips", "histogramMain");
        build.solve_coverage = create_compute_pipeline(device, "mips", "coverageMain");
        build.scale_alpha = create_compute_pipeline(device, "mips", "scaleAlphaMain");

        const MipChain &last = chains.back();
        const vk::DeviceSize mip_bytes = last.offsets.back() + vk::DeviceSize{last.sizes.back().x} * last.sizes.back().y * 4;
        build.mips = create_buffer(device, gpu, mip_bytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress
                | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        vk::DeviceSize coverage_bytes = 4;  // never empty
        for (MipChain &chain : chains) {
            if (chain.cutoff > 0.0f) {
                chain.coverage = coverage_bytes;
                coverage_bytes += chain.sizes.size() * (256 + 1) * 4;
            }
        }

        build.coverage = create_buffer(device, gpu, coverage_bytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress
                | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        return build;
    }

    // Runs one of mips.slang's steps, over `x` x `y` threads in workgroups of `group_x` x `group_y`.
    void dispatch_mips(
        const vk::raii::CommandBuffer &commands,
        const MipPushData &push,
        std::uint32_t x,
        std::uint32_t y,
        std::uint32_t group_x,
        std::uint32_t group_y
    ) {
        commands.pushDataEXT(vk::PushDataInfoEXT{
            .offset = 0,
            .data = {.address = &push, .size = sizeof(push)},
        });
        commands.dispatch((x + group_x - 1) / group_x, (y + group_y - 1) / group_y, 1);
    }

    // Every level below each chain's top, from the tops already copied into the mip buffer: normal maps' tops prepared, then the levels, then the masked chains' coverage kept. On return the levels are written and visible to a copy.
    void record_mip_levels(const vk::raii::CommandBuffer &commands, const MipBuild &build, std::span<const MipChain> chains) {
        const auto level_address = [&](const MipChain &chain, std::size_t level) { return build.mips.address + chain.offsets[level]; };
        const auto compute_to_compute = [&] {
            memory_barrier(commands,
                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
        };

        // 1. The coverage buffer zeroed, and the tops, which copies put there, visible.
        commands.fillBuffer(*build.coverage.handle, 0, vk::WholeSize, 0);

        memory_barrier(commands,
            vk::PipelineStageFlagBits2::eCopy | vk::PipelineStageFlagBits2::eClear, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

        // 2. Normal maps' top levels: no spread, in alpha.
        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *build.prepare_normals);
        for (const MipChain &chain : chains) {
            if (chain.kind == MipKind::normal) {
                const MipPushData push{.source = level_address(chain, 0), .source_size = chain.sizes[0]};
                dispatch_mips(commands, push, chain.sizes[0].x, chain.sizes[0].y, 8, 8);
            }
        }

        compute_to_compute();

        // 3. Level by level, every chain's next level from the one above. Chains are independent, so one barrier per level covers them all.
        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *build.downsample);
        std::size_t most_levels = 0;
        for (const MipChain &chain : chains) {
            most_levels = std::max(most_levels, chain.sizes.size());
        }

        for (std::size_t level = 1; level < most_levels; ++level) {
            for (const MipChain &chain : chains) {
                if (level < chain.sizes.size()) {
                    const MipPushData push{
                        .source = level_address(chain, level - 1),
                        .target = level_address(chain, level),
                        .source_size = chain.sizes[level - 1],
                        .target_size = chain.sizes[level],
                        .kind = static_cast<std::uint32_t>(chain.kind),
                    };
                    dispatch_mips(commands, push, chain.sizes[level].x, chain.sizes[level].y, 8, 8);
                }
            }

            compute_to_compute();
        }

        // 4. Masked chains: count every level's alpha, work out each level's scale, then apply it below the top.
        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *build.histogram);
        for (const MipChain &chain : chains) {
            for (std::size_t level = 0; chain.cutoff > 0.0f && level < chain.sizes.size(); ++level) {
                const MipPushData push{
                    .source = level_address(chain, level),
                    .target = build.coverage.address + chain.coverage + level * 256 * 4,
                    .source_size = chain.sizes[level],
                };
                dispatch_mips(commands, push, chain.sizes[level].x, chain.sizes[level].y, 8, 8);
            }
        }

        compute_to_compute();

        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *build.solve_coverage);
        for (const MipChain &chain : chains) {
            if (chain.cutoff > 0.0f) {
                const MipPushData push{
                    .source = build.coverage.address + chain.coverage,
                    .target = build.coverage.address + chain.coverage + chain.sizes.size() * 256 * 4,
                    .kind = static_cast<std::uint32_t>(chain.sizes.size()),
                    .cutoff = chain.cutoff,
                };
                dispatch_mips(commands, push, static_cast<std::uint32_t>(chain.sizes.size()), 1, 64, 1);
            }
        }

        compute_to_compute();

        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *build.scale_alpha);
        for (const MipChain &chain : chains) {
            for (std::size_t level = 1; chain.cutoff > 0.0f && level < chain.sizes.size(); ++level) {
                const MipPushData push{
                    .source = level_address(chain, level),
                    .target = build.coverage.address + chain.coverage + chain.sizes.size() * 256 * 4 + level * 4,
                    .source_size = chain.sizes[level],
                };
                dispatch_mips(commands, push, chain.sizes[level].x, chain.sizes[level].y, 8, 8);
            }
        }

        memory_barrier(commands,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
    }

    // Every level of a chain from the mip buffer into its image, which is then ready for fragment shaders. The image's levels must be in eTransferDstOptimal.
    void copy_levels_in(const vk::raii::CommandBuffer &commands, const MipBuild &build, const MipChain &chain, vk::Image image) {
        std::vector<vk::BufferImageCopy> regions;
        for (std::uint32_t level = 0; level < chain.sizes.size(); ++level) {
            regions.push_back(vk::BufferImageCopy{
                .bufferOffset = chain.offsets[level],
                .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = level, .baseArrayLayer = 0, .layerCount = 1},
                .imageExtent = {chain.sizes[level].x, chain.sizes[level].y, 1},
            });
        }

        commands.copyBufferToImage(*build.mips.handle, image, vk::ImageLayout::eTransferDstOptimal, regions);

        transition_mips(commands, image, 0, static_cast<std::uint32_t>(chain.sizes.size()),
            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead);
    }

}  // namespace

// Decoding

std::vector<DecodedImage> decode_images(std::span<const SceneImage> images) {
    std::vector<DecodedImage> decoded(images.size());

    // Each worker thread takes the next undecoded image until none are left. Images are independent, so the threads never touch the same one.
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

// GPU textures

std::vector<Texture> create_scene_textures(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    const Scene &scene
) {
    // Index 0: white, so "no texture" can sample like any other.
    std::vector<DecodedImage> images{DecodedImage{.width = 1, .height = 1, .rgba = {255, 255, 255, 255}}};
    std::vector<vk::Format> formats{vk::Format::eR8G8B8A8Srgb};

    for (DecodedImage &image : decode_images(scene.images)) {
        images.push_back(std::move(image));
    }

    // sRGB formats make the GPU decode colors to linear when sampling; data textures stay as they are. Vulkan requires both formats to support linear filtering, so every GPU can sample them smoothly.
    for (const SceneImage &image : scene.images) {
        formats.push_back(image.srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm);
    }

    std::vector<MipChain> chains = plan_mips(scene, images);

    // One staging buffer holds every image's pixels back to back.
    vk::DeviceSize total = 0;
    for (const DecodedImage &image : images) {
        total += image.rgba.size();
    }

    const Buffer staging = create_buffer(device, gpu, total, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    auto *mapped = static_cast<std::uint8_t*>(staging.memory.mapMemory(0, total));
    std::vector<vk::DeviceSize> offsets;
    vk::DeviceSize offset = 0;

    for (const DecodedImage &image : images) {
        std::memcpy(mapped + offset, image.rgba.data(), image.rgba.size());
        offsets.push_back(offset);
        offset += image.rgba.size();
    }

    staging.memory.unmapMemory();

    const MipBuild build = create_mip_build(device, gpu, chains);

    std::vector<Texture> textures;
    for (std::size_t i = 0; i < images.size(); ++i) {
        textures.push_back(create_texture(device, gpu, images[i], formats[i]));
    }

    // Everything in one command buffer and one submission: each image's pixels into its top level, the levels below made, every level into its image.
    submit_and_wait(device, queue, pool, [&](const vk::raii::CommandBuffer &commands) {
        for (std::size_t i = 0; i < images.size(); ++i) {
            commands.copyBuffer(*staging.handle, *build.mips.handle, vk::BufferCopy{
                .srcOffset = offsets[i],
                .dstOffset = chains[i].offsets[0],
                .size = images[i].rgba.size(),
            });
        }

        record_mip_levels(commands, build, chains);

        for (std::size_t i = 0; i < textures.size(); ++i) {
            transition_mips(commands, *textures[i].handle, 0, textures[i].mip_levels,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
            copy_levels_in(commands, build, chains[i], *textures[i].handle);
        }
    });

    return textures;
}

void generate_image_mips(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::span<const MipRequest> requests
) {
    std::vector<MipChain> chains;
    vk::DeviceSize offset = 0;

    for (const MipRequest &request : requests) {
        chains.push_back(plan_chain({request.extent.width, request.extent.height}, request.kind, request.cutoff, offset));
    }

    const MipBuild build = create_mip_build(device, gpu, chains);

    // Each image's top level out into the mip buffer, the levels below made, and every level back in: the top level over itself, which a copy may do.
    submit_and_wait(device, queue, pool, [&](const vk::raii::CommandBuffer &commands) {
        for (std::size_t i = 0; i < requests.size(); ++i) {
            commands.copyImageToBuffer(requests[i].image, vk::ImageLayout::eTransferSrcOptimal, *build.mips.handle, vk::BufferImageCopy{
                .bufferOffset = chains[i].offsets[0],
                .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                .imageExtent = {requests[i].extent.width, requests[i].extent.height, 1},
            });
        }

        record_mip_levels(commands, build, chains);

        for (std::size_t i = 0; i < requests.size(); ++i) {
            transition_mips(commands, requests[i].image, 0, 1,
                vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eTransferDstOptimal,
                vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead,
                vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
            transition_mips(commands, requests[i].image, 1, requests[i].mip_levels - 1,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
            copy_levels_in(commands, build, chains[i], requests[i].image);
        }
    });
}
```

`game-engine/src/includes/impostors.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/clusters.h"
#include "includes/descriptor_heap.h"
#include "includes/image.h"
#include "includes/scene.h"
#include "includes/shader_types.h"
#include "includes/vulkan_setup.h"

#include <cstdint>
#include <vector>

// Impostors

// A placement too small on screen for its clusters is drawn as an impostor: one quad facing the camera, textured from pictures of its model baked at load. The pictures are taken from impostor_grid x impostor_grid directions over the upper hemisphere, laid out over a hemi-octahedron (Cigolle et al. 2014) so that any direction falls in a triangle of three of them, and the quad blends those three. Each picture is a G-buffer, not a shading: albedo and coverage, the normal, roughness and metallic, and depth, so an impostor is lit by the frame's sun, sky and shadows like every other surface, and nothing in the atlases depends on the lighting. The bake draws the model's finest clusters with the engine's own mesh pipeline, through its materials and textures, into a tile per direction of three atlases per model, then makes their mips with the same coverage-preserving steps the scene's masked textures get (texture.h).
//
// The directions and the image axes of each tile are a table the bake writes and the shaders read, so the two never disagree. Nothing is kept on disk: the bake is a second of GPU time at every start, and the atlases die with the run.
struct ModelAtlases {
    Image albedo;   // RGBA8 sRGB: base color, coverage
    Image surface;  // RGBA8: the normal, octahedrally encoded, roughness, coverage
    Image extra;    // RGBA8: depth, metallic, 0, coverage
};

struct Impostors {
    std::vector<ModelAtlases> atlases;  // one per model
    Buffer atlas_table;                 // one ModelAtlas per model: the atlases' heap slots
    Buffer frames;                      // impostor_grid x impostor_grid ImpostorFrames
    std::uint32_t first_slot = 0;       // resource heap slots: model m's albedo at first_slot + 3m, its surface and extra after it
};

// Frames per side of the atlas, and the atlas's size: tiles of atlas_size / grid pixels. At 1024, a tile is 128 pixels, enough for an impostor drawn at up to 32 pixels across with its mips; 2048 doubles that at four times the memory.
constexpr std::uint32_t impostor_grid = 8;
constexpr std::uint32_t impostor_atlas_size = 1024;
constexpr std::uint32_t impostor_slots_per_model = 3;

// Bakes every model's atlases and writes their descriptors from `first_slot` on, impostor_slots_per_model per model. `frame` is a FrameData with every pointer the mesh shader reads filled in (vertices, materials, clusters, primitive data, models); the bake supplies its own placements and view. Waits for the GPU.
Impostors bake_impostors(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    DescriptorHeaps &heaps,
    std::uint32_t first_slot,
    const Scene &scene,
    const Clusters &clusters,
    FrameData frame
);
```

`game-engine/src/impostors.cpp`:
```cpp
#include "includes/impostors.h"

#include "includes/pipeline.h"
#include "includes/swapchain.h"
#include "includes/texture.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace {

    // The frames

    // The direction frame (i, j) of the grid looks from. The grid is the hemi-octahedron's square: (i, j) over grid - 1 mapped to -1..1 each way, then folded onto the upper hemisphere (Cigolle et al. 2014, "A Survey of Efficient Representations for Independent Unit Vectors"): the square's diagonals become the x and z axes, and what's left of 1 after both is the height. impostor.slang's hemi_octahedral is the reverse.
    glm::vec3 frame_direction(std::uint32_t i, std::uint32_t j) {
        const glm::vec2 e = glm::vec2(i, j) / static_cast<float>(impostor_grid - 1) * 2.0f - 1.0f;
        const glm::vec2 t{(e.x + e.y) * 0.5f, (e.x - e.y) * 0.5f};
        return glm::normalize(glm::vec3{t.x, 1.0f - std::abs(t.x) - std::abs(t.y), t.y});
    }

    // The image axes of a picture taken from `direction`: right across, and up along the world's up, or along z when looking straight down, where up has no meaning.
    ImpostorFrame make_frame(glm::vec3 direction) {
        const glm::vec3 helper = std::abs(direction.y) > 0.99f ? glm::vec3{0.0f, 0.0f, 1.0f} : glm::vec3{0.0f, 1.0f, 0.0f};
        const glm::vec3 right = glm::normalize(glm::cross(helper, direction));
        return ImpostorFrame{.direction = direction, .right = right, .up = glm::cross(direction, right)};
    }

    // The view-projection a picture is taken with: the camera on the model's sphere along the frame's direction, looking at the centre, with the frame's axes as the image's; orthographic over the sphere's width; depth with reverse-Z, 1 at the camera and 0 at the sphere's far side; and Vulkan's downward clip-space y, so the picture is the right way up.
    glm::mat4 frame_view_projection(const ImpostorFrame &frame, glm::vec3 center, float radius) {
        // The view's rows are the frame's axes: view space's x is right, y is up, z is the direction, toward the camera. glm indexes [column][row].
        glm::mat4 view{1.0f};

        for (int c = 0; c < 3; ++c) {
            view[c][0] = frame.right[c];
            view[c][1] = frame.up[c];
            view[c][2] = frame.direction[c];
        }

        const glm::vec3 eye = center + frame.direction * radius;
        view[3] = glm::vec4(-glm::vec3(view * glm::vec4(eye, 0.0f)), 1.0f);

        glm::mat4 projection{0.0f};
        projection[0][0] = 1.0f / radius;
        projection[1][1] = -1.0f / radius;
        projection[2][2] = 1.0f / (2.0f * radius);
        projection[3][2] = 1.0f;
        projection[3][3] = 1.0f;
        return projection * view;
    }

    // Images

    // Moves mip levels [base, base + count) of `image` between layouts.
    void transition(
        const vk::raii::CommandBuffer &commands,
        vk::Image image,
        vk::ImageAspectFlags aspect,
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
            .subresourceRange = {.aspectMask = aspect, .baseMipLevel = base, .levelCount = count, .baseArrayLayer = 0, .layerCount = 1},
        };

        commands.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    }

    // A view of an atlas's top level alone: what a color attachment must be.
    vk::raii::ImageView top_level_view(const vk::raii::Device &device, const Image &image) {
        return vk::raii::ImageView(device, vk::ImageViewCreateInfo{
            .image = *image.handle,
            .viewType = vk::ImageViewType::e2D,
            .format = image.format,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1},
        });
    }

}  // namespace

Impostors bake_impostors(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    DescriptorHeaps &heaps,
    std::uint32_t first_slot,
    const Scene &scene,
    const Clusters &clusters,
    FrameData frame
) {
    Impostors impostors;
    impostors.first_slot = first_slot;

    const std::uint32_t tile = impostor_atlas_size / impostor_grid;
    const std::uint32_t mip_levels = std::bit_width(impostor_atlas_size);
    const std::size_t frame_count = static_cast<std::size_t>(impostor_grid) * impostor_grid;
    const std::size_t model_count = scene.models.size();

    const auto upload = [&](std::span<const std::byte> bytes) {
        return upload_buffer(device, gpu, queue, pool, bytes, vk::BufferUsageFlagBits::eShaderDeviceAddress);
    };

    // 1. The frames: row j, column i of the grid is frame j x grid + i.
    std::vector<ImpostorFrame> frames;

    for (std::uint32_t j = 0; j < impostor_grid; ++j) {
        for (std::uint32_t i = 0; i < impostor_grid; ++i) {
            frames.push_back(make_frame(frame_direction(i, j)));
        }
    }

    impostors.frames = upload(std::as_bytes(std::span(frames)));

    // 2. What the bake draws: one placement per model, at the origin, unturned and unscaled, so the model's space is the camera's; and per model and alpha mode, its finest clusters as items, with a dispatch over them. Masked and blended materials bake alike, cut out at their cutoff.
    std::vector<Placement> placements;
    std::vector<ClusterItem> items;
    std::vector<ClusterDispatch> dispatches;

    for (std::uint32_t m = 0; m < model_count; ++m) {
        placements.push_back(Placement{.cell = {0, 0, 0}, .offset = {0.0f, 0.0f, 0.0f}, .rotation = {0.0f, 0.0f, 0.0f, 1.0f}, .scale = {1.0f, 1.0f, 1.0f}, .model = m});
        const SceneModel &model = scene.models[m];

        for (const AlphaMode mode : {AlphaMode::opaque, AlphaMode::mask}) {
            const auto start = static_cast<std::uint32_t>(items.size());

            for (std::uint32_t p = model.first_primitive; p < model.first_primitive + model.primitive_count; ++p) {
                const AlphaMode primitive_mode = scene.materials[scene.primitives[p].material].alpha_mode;

                if ((primitive_mode == AlphaMode::opaque) != (mode == AlphaMode::opaque)) {
                    continue;
                }

                for (std::uint32_t k = 0; k < clusters.finest_counts[p]; ++k) {
                    items.push_back(ClusterItem{
                        .placement = m,
                        .what = (p << item_primitive_shift) | (item_kind_cluster << item_kind_shift) | (clusters.first_clusters[p] + k),
                    });
                }
            }

            const auto count = static_cast<std::uint32_t>(items.size()) - start;
            dispatches.push_back(ClusterDispatch{.x = count, .y = 1, .z = 1, .count = count, .start = start, .pad = 0, .items = 0});
        }
    }

    // A buffer can't be empty, and a scene with nothing to draw has no items.
    if (items.empty()) {
        items.push_back(ClusterItem{});
    }

    const Buffer bake_placements = upload(std::as_bytes(std::span(placements)));
    const Buffer bake_items = upload(std::as_bytes(std::span(items)));

    for (ClusterDispatch &dispatch : dispatches) {
        dispatch.items = bake_items.address;
    }

    const Buffer bake_dispatches = upload(std::as_bytes(std::span(dispatches)));

    // 3. A FrameData per model and frame: the given pointers, the bake's placements, the camera at the origin of its cell, and the frame's view of the model's sphere.
    frame.placements = bake_placements.address;
    frame.camera_cell = {0, 0, 0};
    frame.camera_offset = {0.0f, 0.0f, 0.0f};
    frame.view = View::lit;

    std::vector<FrameData> frame_data;

    for (std::uint32_t m = 0; m < model_count; ++m) {
        const SceneModel &model = scene.models[m];
        const glm::vec3 center = (model.bounds_min + model.bounds_max) * 0.5f;
        const float radius = glm::length(model.bounds_max - model.bounds_min) * 0.5f;

        for (const ImpostorFrame &view : frames) {
            frame.view_projection = frame_view_projection(view, center, radius);
            frame.inverse_view_projection = glm::inverse(frame.view_projection);
            frame_data.push_back(frame);
        }
    }

    const Buffer bake_frames = upload(std::as_bytes(std::span(frame_data)));

    // 4. The atlases, their top-level views for drawing into, their descriptors for sampling, and one depth buffer the tiles share. The atlases are also copied out of and into for their mips.
    const vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
        | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
    const vk::Extent2D extent{impostor_atlas_size, impostor_atlas_size};
    std::vector<vk::raii::ImageView> views;
    std::vector<ModelAtlas> table;

    for (std::uint32_t m = 0; m < model_count; ++m) {
        ModelAtlases atlases{
            .albedo = create_image(device, gpu, extent, vk::Format::eR8G8B8A8Srgb, usage, vk::ImageAspectFlagBits::eColor, mip_levels),
            .surface = create_image(device, gpu, extent, vk::Format::eR8G8B8A8Unorm, usage, vk::ImageAspectFlagBits::eColor, mip_levels),
            .extra = create_image(device, gpu, extent, vk::Format::eR8G8B8A8Unorm, usage, vk::ImageAspectFlagBits::eColor, mip_levels),
        };

        const std::uint32_t slot = first_slot + m * impostor_slots_per_model;
        table.push_back(ModelAtlas{.albedo = slot, .surface = slot + 1, .extra = slot + 2});

        for (const Image *image : {&atlases.albedo, &atlases.surface, &atlases.extra}) {
            views.push_back(top_level_view(device, *image));
            write_image_descriptor(device, heaps, slot + static_cast<std::uint32_t>(views.size() - 1) % impostor_slots_per_model, vk::ImageViewCreateInfo{
                .image = *image->handle,
                .viewType = vk::ImageViewType::e2D,
                .format = image->format,
                .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = mip_levels, .layerCount = 1},
            });
        }

        impostors.atlases.push_back(std::move(atlases));
    }

    impostors.atlas_table = upload(std::as_bytes(std::span(table)));

    const Image depth = create_image(device, gpu, extent, depth_format, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth);

    // 5. The bake pipelines: the mesh pipeline with the bake's fragment shader, per solid alpha mode.
    const std::array formats{vk::Format::eR8G8B8A8Srgb, vk::Format::eR8G8B8A8Unorm, vk::Format::eR8G8B8A8Unorm};
    const std::array pipelines{
        create_mesh_pipeline(device, formats, depth_format, AlphaMode::opaque, MeshPass::bake),
        create_mesh_pipeline(device, formats, depth_format, AlphaMode::mask, MeshPass::bake),
    };

    // 6. The bake: per model, its three atlases and the depth cleared, then each frame's tile drawn, both alpha modes, with no culling: every side of every triangle is a surface from somewhere.
    submit_and_wait(device, queue, pool, [&](const vk::raii::CommandBuffer &commands) {
        bind_descriptor_heaps(commands, heaps);

        transition(commands, *depth.handle, vk::ImageAspectFlagBits::eDepth, 0, 1,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal,
            vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite);

        for (std::uint32_t m = 0; m < model_count; ++m) {
            const ModelAtlases &atlases = impostors.atlases[m];

            for (const Image *image : {&atlases.albedo, &atlases.surface, &atlases.extra}) {
                transition(commands, *image->handle, vk::ImageAspectFlagBits::eColor, 0, 1,
                    vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                    vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                    vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite);
            }

            // Cleared to zero: coverage 0 wherever nothing is drawn.
            std::array<vk::RenderingAttachmentInfo, 3> color_attachments;

            for (std::size_t k = 0; k < 3; ++k) {
                color_attachments[k] = vk::RenderingAttachmentInfo{
                    .imageView = *views[m * 3 + k],
                    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .loadOp = vk::AttachmentLoadOp::eClear,
                    .storeOp = vk::AttachmentStoreOp::eStore,
                    .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = std::array{0.0f, 0.0f, 0.0f, 0.0f}}},
                };
            }

            const vk::RenderingAttachmentInfo depth_attachment{
                .imageView = *depth.view,
                .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eDontCare,
                .clearValue = vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = 0.0f}},
            };

            commands.beginRendering(vk::RenderingInfo{
                .renderArea = {.offset = {0, 0}, .extent = extent},
                .layerCount = 1,
                .colorAttachmentCount = 3,
                .pColorAttachments = color_attachments.data(),
                .pDepthAttachment = &depth_attachment,
            });

            commands.setCullMode(vk::CullModeFlagBits::eNone);
            commands.setFrontFace(vk::FrontFace::eCounterClockwise);

            for (std::size_t f = 0; f < frame_count; ++f) {
                const auto x = static_cast<float>((f % impostor_grid) * tile);
                const auto y = static_cast<float>((f / impostor_grid) * tile);
                commands.setViewport(0, vk::Viewport{.x = x, .y = y, .width = static_cast<float>(tile), .height = static_cast<float>(tile), .minDepth = 0.0f, .maxDepth = 1.0f});
                commands.setScissor(0, vk::Rect2D{.offset = {static_cast<std::int32_t>(x), static_cast<std::int32_t>(y)}, .extent = {tile, tile}});

                for (std::size_t mode = 0; mode < 2; ++mode) {
                    const ClusterDispatch &dispatch = dispatches[m * 2 + mode];

                    if (dispatch.count == 0) {
                        continue;
                    }

                    commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines[mode]);

                    const PushData push{
                        .frame = bake_frames.address + (m * frame_count + f) * sizeof(FrameData),
                        .drawn = bake_dispatches.address + (m * 2 + mode) * sizeof(ClusterDispatch),
                    };
                    commands.pushDataEXT(vk::PushDataInfoEXT{.offset = 0, .data = {.address = &push, .size = sizeof(push)}});
                    commands.drawMeshTasksEXT(dispatch.count, 1, 1);
                }
            }

            commands.endRendering();

            // Ready to be copied out for the mips.
            for (const Image *image : {&atlases.albedo, &atlases.surface, &atlases.extra}) {
                transition(commands, *image->handle, vk::ImageAspectFlagBits::eColor, 0, 1,
                    vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
                    vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
            }
        }
    });

    // 7. The mips: albedo as a masked texture's, its coverage kept at every level; the surface and the rest as data weighted by their coverage, so nothing outside the silhouette bleeds in.
    std::vector<MipRequest> requests;

    for (const ModelAtlases &atlases : impostors.atlases) {
        requests.push_back(MipRequest{.image = *atlases.albedo.handle, .extent = extent, .mip_levels = mip_levels, .kind = MipKind::see_through, .cutoff = 0.5f});
        requests.push_back(MipRequest{.image = *atlases.surface.handle, .extent = extent, .mip_levels = mip_levels, .kind = MipKind::data_see_through, .cutoff = 0.0f});
        requests.push_back(MipRequest{.image = *atlases.extra.handle, .extent = extent, .mip_levels = mip_levels, .kind = MipKind::data_see_through, .cutoff = 0.0f});
    }

    generate_image_mips(device, gpu, queue, pool, requests);
    return impostors;
}
```

## 21.3 Drawing impostors: `impostor.slang`, `shading.slangh`, `terrain.slang`, `cull.slang`, `culling.h`

### Why
An impostor is a quad at the model's sphere's centre, facing the camera, as wide as the sphere. Its fragment shader finds the direction the camera sees the placement from, in the model's space; the three pictures around it and their weights; and, for each picture, where the fragment's point falls in it: the point projected onto the picture's plane, mapped into the picture's tile. The three samples blend by their weights, each weighted by its coverage too, so a picture that shows nothing there says nothing. Under half coverage the fragment is cut out, like a masked material's. What's left is a surface: albedo, a normal in the world through the placement's normal matrix, roughness, metallic, and a depth into the sphere that moves the fragment's point off the quad to where the pictures put the surface, along the view. That point's projected depth is written, so the depth buffer holds the model's shape rather than the quad's: an impostor's trunk meets a slope where the mesh's did, and the prepass's normals and the ambient occlusion see a tree, not a card.

Shadows need one more thought. An impostor's placement is still in the TLAS within the shadow radius, and the surface the pictures reconstruct is within a fraction of its radius of the real one: a shadow ray from it would hit the model it stands for at once, everywhere, and the impostor would go black. So the ray passes through its own placement's triangles for a quarter of the radius, and is shadowed by them beyond that: a canopy still shadows itself as the mesh's does, and the brightness doesn't jump at the switch. That takes forcing the ray's hits non-opaque so the shader sees each one's instance and distance; only the impostors' rays pay for it.

### How
- **`meshMain`** (`impostor.slang`): a workgroup per 32 impostors of the list's dispatch, thread t making corner t % 4 of impostor t / 4, and triangle t of the 64. The quad's axes come from the view of the centre: right across it, up from there, z as up for the view straight down.
- **`hemi_octahedral`** is `frame_direction` in reverse; `nearest_frames` cuts the square into the grid's cells and each cell into two triangles, and gives the direction's three corners and barycentric weights.
- **`sample_picture`**: the fragment's point in the model's space, relative to the sphere's centre and in its radius, dotted with the picture's right and up, is where it falls in the picture, −1..1; mapped to the tile and sampled through the clamp sampler, which stays within the atlas, while the tile's edges, outside the sphere, are empty anyway.
- **`impostor_surface`** blends the three and moves the point by the depth: from the quad's plane, which cuts the sphere through its centre, toward the camera by the radius, then back into the sphere by the depth times the diameter.
- **`prepassMain`** writes the normal and the depth, which costs the impostors' pixels the early depth test, since a shader that writes depth can't be tested before it runs; **`fragmentMain`** the depth and the shading: ambient occlusion from the prepass's image, the debug views, the sun shadow view, and `shade_surface` with a `ShadowSurface` that isn't instanced, since the point is already in the world, and names the placement as its own, with a quarter of the radius to clear.
- **`light_visibility`** takes the placement to pass through and how far: with one, the ray is forced non-opaque, and a candidate from that instance nearer than the clearance is skipped; an opaque triangle beyond it commits as opaque triangles do. `ShadowSurface` carries both; meshes and the terrain pass none.
- **The walk** (`cull.slang`): a placement in view, not dropped and not hidden, whose sphere is under the threshold, counts as a cluster to draw: the item is itself. `draw_list_of` takes the item and answers the impostors' list for a placement; the write step counts two triangles for one; the sort runs over thirteen lists, and the dispatches put 32 impostors to a workgroup.
- **`culling.h`:** `draw_list_count` is 13, `impostor_list` 12, `impostors_per_workgroup` 32.

### Code
`game-engine/shaders/impostor.slang`:
```slang
// Draws impostors (impostors.h): a placement whose model is too small on screen for its clusters, as one quad facing the camera, textured from the pictures of its model baked at load. A mesh shader makes 32 placements' quads per workgroup; the fragment shaders find the three baked directions nearest the one the camera sees the placement from, blend their pictures, and shade the result like any surface, with the frame's sun, sky and shadows. Two fragment shaders, as for meshes:
//   prepassMain   the depth prepass: the normal, and the baked depth
//   fragmentMain  the lighting pass, into the HDR image

#include "shared.slangh"
#include "atmosphere.slangh"
#include "shading.slangh"

[[vk::push_constant]]
ConstantBuffer<PushData> push;

static const uint impostors_per_workgroup = 32;

// Stage interface

// The quad's corner, and what every corner shares: the placement, the model's centre and the quad's axes, all relative to the camera.
struct QuadVertex {
    float4 position : SV_Position;
    float3 relative_position : POSITION;    // camera-relative
    nointerpolation uint placement : PLACEMENT;
    nointerpolation float3 center : CENTER;  // the model's sphere, placed
    nointerpolation float3 right : RIGHT;    // the quad's axes, unit length, in the world
    nointerpolation float3 up : UP;
};

struct PrimitiveOutput {
    uint primitive : SV_PrimitiveID;
};

// Mesh shader

// A workgroup per 32 impostors of the list (ClusterDispatch): thread t makes corner t % 4 of impostor t / 4, and triangle t of the 64. The quad stands at the model's sphere's centre, facing the camera, as wide as the sphere.
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void meshMain(
    uint thread : SV_GroupThreadID,
    uint3 group : SV_GroupID,
    out vertices QuadVertex verts[128],
    out indices uint3 tris[64],
    out primitives PrimitiveOutput prims[64]
) {
    FrameData *frame = push.frame;
    ClusterDispatch *dispatch = (ClusterDispatch*)push.drawn;
    const uint first = (group.y * max_dispatch_width + group.x) * impostors_per_workgroup;
    const uint count = min(dispatch.count - min(first, dispatch.count), impostors_per_workgroup);

    SetMeshOutputCounts(count * 4, count * 2);

    if (thread < count * 4) {
        const uint k = thread / 4;
        const uint corner = thread % 4;
        const ClusterItem item = dispatch.items[dispatch.start + first + k];
        const Instance instance = instance_of(frame, item.placement);
        const Model model = frame.models[instance.placement.model];
        const float3 center = instance_point(instance, model.center);
        const float radius = model.radius * instance_scale(instance);

        // The quad's axes: across the view of the centre, and up from there; the view straight down takes z as up.
        const float3 toward = normalize(-center);
        const float3 helper = abs(toward.y) > 0.99 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
        const float3 right = normalize(cross(helper, toward));
        const float3 up = cross(toward, right);
        const float2 sign = float2(corner & 1 ? 1.0 : -1.0, corner & 2 ? 1.0 : -1.0);
        const float3 relative_position = center + (right * sign.x + up * sign.y) * radius;

        QuadVertex output;
        output.position = mul(frame.view_projection, float4(relative_position, 1.0));
        output.relative_position = relative_position;
        output.placement = item.placement;
        output.center = center;
        output.right = right;
        output.up = up;
        verts[thread] = output;
    }

    if (thread < count * 2) {
        const uint k = thread / 2;
        const uint v = k * 4;
        tris[thread] = thread % 2 == 0 ? uint3(v, v + 1, v + 2) : uint3(v + 2, v + 1, v + 3);
        prims[thread].primitive = thread;
    }
}

// The baked pictures

// A unit direction with y >= 0 to the hemi-octahedron's square, 0..1 each way: the bake's frame_direction in reverse (impostors.cpp).
float2 hemi_octahedral(float3 d) {
    const float3 n = d / (abs(d.x) + abs(d.y) + abs(d.z));
    return float2(n.x + n.z, n.x - n.z) * 0.5 + 0.5;
}

// The three frames around a direction, and how much of each: the square cut into cells of the grid, each cell into two triangles, the direction's barycentric coordinates in its triangle.
void nearest_frames(FrameData *frame, float3 direction, out uint3 frames, out float3 weights) {
    const uint grid = frame.impostor_grid;
    const float2 scaled = hemi_octahedral(direction) * float(grid - 1);
    const uint2 cell = min(uint2(scaled), grid - 2);
    const float2 f = scaled - float2(cell);
    const uint corner = cell.y * grid + cell.x;

    if (f.x + f.y <= 1.0) {
        frames = uint3(corner, corner + 1, corner + grid);
        weights = float3(1.0 - f.x - f.y, f.x, f.y);
    } else {
        frames = uint3(corner + grid + 1, corner + 1, corner + grid);
        weights = float3(f.x + f.y - 1.0, 1.0 - f.y, 1.0 - f.x);
    }
}

// What the three pictures say about the point under a fragment, blended.
struct Picture {
    float3 albedo;
    float coverage;
    float3 normal;     // in the model's space
    float roughness;
    float metallic;
    float depth;       // into the sphere, 0 at the camera's side, 1 at the far, as the bake stored it
};

// Picture `index` of model `model`, at the point `q` of the model's space, relative to its sphere's centre and in its radius: the point projected onto the picture's plane, and that mapped into the picture's tile of the atlases.
Picture sample_picture(FrameData *frame, uint model, uint index, float3 q) {
    const ImpostorFrame picture = frame.impostor_frames[index];
    const ModelAtlas atlas = frame.atlases[model];
    const SamplerState sampler = SamplerState.Handle(uint2(frame.clamp_sampler, 0));
    const float grid = float(frame.impostor_grid);

    // Right and up in the picture, -1..1 across the sphere; the picture's v runs down, so up is subtracted.
    const float2 in_picture = float2(dot(q, picture.right), -dot(q, picture.up));
    const float2 uv = (saturate(in_picture * 0.5 + 0.5) + float2(index % frame.impostor_grid, index / frame.impostor_grid)) / grid;

    const float4 albedo = Texture2D.Handle(uint2(atlas.albedo, 0)).Sample(sampler, uv);
    const float4 surface = Texture2D.Handle(uint2(atlas.surface, 0)).Sample(sampler, uv);
    const float4 extra = Texture2D.Handle(uint2(atlas.extra, 0)).Sample(sampler, uv);

    Picture result;
    result.albedo = albedo.rgb;
    result.coverage = albedo.a;
    result.normal = decode_octahedral(surface.xy * 2.0 - 1.0);
    result.roughness = surface.z;
    result.metallic = extra.y;
    result.depth = extra.x;
    return result;
}

// The surface under a fragment: the three nearest pictures, sampled where the fragment's point falls in each, blended by their weights. The normal and the point come out in the world, relative to the camera; `coverage` under a half is nothing. The point is moved from the quad to where the pictures' depth puts the surface, along the view, so the fragment's depth is the model's and not the quad's.
struct ImpostorSurface {
    float3 albedo;
    float coverage;
    float3 normal;
    float roughness;
    float metallic;
    float3 position;
    float radius;      // the placed model's sphere
};

ImpostorSurface impostor_surface(FrameData *frame, QuadVertex input) {
    const Instance instance = instance_of(frame, input.placement);
    const Model model = frame.models[instance.placement.model];
    const float radius = model.radius * instance_scale(instance);

    // The direction the camera sees the placement from, in the model's space, and clamped to the hemisphere that was baked: a camera below the centre gets the horizon's pictures.
    float3 direction = normalize(mul(transpose(instance.linear), normalize(-input.center)));
    direction.y = max(direction.y, 0.0);
    direction = normalize(direction);

    uint3 frames;
    float3 weights;
    nearest_frames(frame, direction, frames, weights);

    // The fragment's point, from the quad into the model's space, relative to the sphere's centre and in its radius.
    const float3 q = mul(instance.inverse, input.relative_position - input.center) / model.radius;

    Picture sum;
    sum.albedo = 0.0;
    sum.coverage = 0.0;
    sum.normal = 0.0;
    sum.roughness = 0.0;
    sum.metallic = 0.0;
    sum.depth = 0.0;

    for (uint k = 0; k < 3; ++k) {
        const Picture picture = sample_picture(frame, instance.placement.model, frames[k], q);
        const float weight = weights[k] * picture.coverage;
        sum.albedo += weight * picture.albedo;
        sum.coverage += weights[k] * picture.coverage;
        sum.normal += weight * picture.normal;
        sum.roughness += weight * picture.roughness;
        sum.metallic += weight * picture.metallic;
        sum.depth += weight * picture.depth;
    }

    ImpostorSurface surface;
    surface.coverage = sum.coverage;
    surface.radius = radius;

    if (sum.coverage > 0.0) {
        surface.albedo = sum.albedo / sum.coverage;
        surface.normal = normalize(instance_normal(instance, sum.normal));
        surface.roughness = sum.roughness / sum.coverage;
        surface.metallic = sum.metallic / sum.coverage;
        const float depth = sum.depth / sum.coverage;
        surface.position = input.relative_position + normalize(-input.center) * (radius - 2.0 * radius * depth);
    } else {
        surface.albedo = 0.0;
        surface.normal = float3(0.0, 1.0, 0.0);
        surface.roughness = 1.0;
        surface.metallic = 0.0;
        surface.position = input.relative_position;
    }

    return surface;
}

// The depth the pictures put the surface at, for the depth buffer: the moved point, projected.
float surface_depth(FrameData *frame, float3 position) {
    const float4 clip = mul(frame.view_projection, float4(position, 1.0));
    return clip.z / clip.w;
}

// Fragment shaders

// The depth prepass: the normal, as the meshes write it, and the baked depth. Under half coverage, nothing: an impostor cuts out like a masked material.
struct PrepassOutput {
    float2 normal : SV_Target;
    float depth : SV_Depth;
};

[shader("fragment")]
PrepassOutput prepassMain(QuadVertex input) {
    FrameData *frame = push.frame;
    const ImpostorSurface surface = impostor_surface(frame, input);

    if (surface.coverage < 0.5) {
        discard;
    }

    PrepassOutput output;
    output.normal = encode_octahedral(surface.normal);
    output.depth = surface_depth(frame, surface.position);
    return output;
}

// The lighting pass: the baked surface, shaded as a mesh's is. Its shadow rays pass through the placement's own triangles for a quarter of its radius, the most the pictures' surface strays from the real one, and are shadowed by them beyond that, as the mesh's are.
struct LightingOutput {
    float4 color : SV_Target;
    float depth : SV_Depth;
};

[shader("fragment")]
LightingOutput fragmentMain(QuadVertex input) {
    FrameData *frame = push.frame;
    const ImpostorSurface surface = impostor_surface(frame, input);

    if (surface.coverage < 0.5) {
        discard;
    }

    LightingOutput output;
    output.depth = surface_depth(frame, surface.position);

    const float roughness = max(surface.roughness, 0.045);
    const float alpha = roughness * roughness;
    const float4 gtao = Texture2D.Handle(uint2(frame.ambient_occlusion, 0)).Load(int3(int2(input.position.xy), 0));
    const float visibility = frame.ao_enabled != 0 ? gtao.w : 1.0;
    const float3 irradiance_normal = frame.ao_enabled != 0 ? normalize(rotate_from_to(surface.normal, gtao.xyz, surface.normal)) : surface.normal;

    switch (frame.view) {
        case view_base_color: output.color = float4(surface.albedo, 1.0); return output;
        case view_normal:
        case view_vertex_normal: output.color = float4(surface.normal * 0.5 + 0.5, 1.0); return output;
        case view_metallic: output.color = float4(surface.metallic.xxx, 1.0); return output;
        case view_roughness: output.color = float4(roughness.xxx, 1.0); return output;
        case view_occlusion: output.color = float4(1.0, 1.0, 1.0, 1.0); return output;
        case view_emissive: output.color = float4(0.0, 0.0, 0.0, 1.0); return output;
        case view_ambient_occlusion: output.color = float4(gtao.www, 1.0); return output;
        case view_light_count: output.color = float4(light_count_heat(frame, input.position.xy, surface.position), 1.0); return output;
        default: break;
    }

    const ShadowSurface from = {
        surface.position,
        surface.normal,
        false,
        float3(0.0),
        float3x3(0.0),
        float3x3(0.0),
        input.placement,
        surface.radius * 0.25,
    };

    if (frame.view == view_shadow) {
        const float sun = frame.sun_direction.y > 0.0
            ? light_visibility(frame, shadow_ray_origin(frame, from, frame.sun_direction), frame.sun_direction, infinite_distance, from.self, from.clearance)
            : 0.0;
        output.color = float4(sun.xxx, 1.0);
        return output;
    }

    const float3 view = normalize(-surface.position);
    const Surface lit = {
        surface.albedo,
        surface.metallic,
        alpha,
        surface.normal,
        view,
        energy_compensation(frame, surface.albedo, surface.metallic, roughness, max(dot(surface.normal, view), 1e-4)),
    };

    output.color = float4(shade_surface(lit, frame, from, input.position.xy, roughness, visibility, irradiance_normal, float3(0.0)), 1.0);
    return output;
}
```

In `game-engine/shaders/shading.slangh`, replace `light_visibility` with:
```slang
// How much of a light gets from `origin` to `distance` along `direction`: 0 when something solid is in the way, otherwise the share every see-through layer on the way lets through. A ray query walks the TLAS and BLASes:
//   - an opaque triangle ends it at once: any blocking hit will do,
//   - a masked one comes back as a candidate, which blocks where its alpha reaches the cutoff, and lets the light through its cut-out texels,
//   - a blended one comes back as a candidate that lets 1 - alpha of the light through, as the transparency pass's reveal sum does. It never ends the ray: the light goes on, dimmed, to whatever is behind.
// Then the terrain is marched, if nothing solid was met.
// `self` is a placement whose own triangles the ray passes through for its first `clearance` metres, or none (~0): an impostor's surface is only a picture of its model, within a fraction of its radius of the real one, and a ray from it would hit the model it stands for at once. Past the clearance the model's own triangles count again: a canopy shadows itself as the mesh's does. Then every hit is a candidate, so the ray can look at each one's instance and distance.
float light_visibility(FrameData *frame, float3 origin, float3 direction, float distance, uint self, float clearance) {
    const RaytracingAccelerationStructure scene = RaytracingAccelerationStructure(frame.scene_tlas);

    RayDesc ray;
    ray.Origin = origin;
    ray.TMin = 0.0;
    ray.Direction = direction;
    ray.TMax = distance;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(scene, self == ~0u ? RAY_FLAG_NONE : RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);

    float transmittance = 1.0;

    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE
            || (query.CandidateInstanceID() == self && query.CandidateTriangleRayT() < clearance)) {
            continue;
        }

        // The instance's custom index is the placement; its geometry index is the primitive within the model (acceleration.h).
        const Model model = frame.models[frame.placements[query.CandidateInstanceID()].model];
        const PrimitiveData primitive = frame.primitive_data[model.first_primitive + query.CandidateGeometryIndex()];
        const Material material = frame.materials[primitive.material];
        const float alpha = candidate_alpha(frame, primitive, material, query.CandidatePrimitiveIndex(),
            query.CandidateTriangleBarycentrics());

        // A triangle forced non-opaque for the self test keeps its own mode: an opaque one blocks the ray.
        if (material.alpha_mode == alpha_blend) {
            transmittance *= 1.0 - saturate(alpha);
        } else if (material.alpha_mode == alpha_opaque || alpha >= material.alpha_cutoff) {
            query.CommitNonOpaqueTriangleHit();
        }
    }

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        return 0.0;
    }

    // Then the terrain, which is in no acceleration structure. The ray goes back to camera-relative space: the TLAS's is offset from it (shadow_ray_origin).
    return terrain_blocks(frame, origin - frame.tlas_offset, direction, distance) ? 0.0 : transmittance;
}
```

In `game-engine/shaders/shading.slangh`, remove these lines:
```slang
    float3x3 inverse;    // its inverse
};
```

In `game-engine/shaders/shading.slangh`, add this section before `// Where a ray toward the light`:
```slang
    float3x3 inverse;    // its inverse
    uint self;           // a placement whose triangles the rays pass through at first, or none (~0): an impostor's own
    float clearance;     // how far they pass through them, in metres
};
```

In `game-engine/shaders/shading.slangh`, replace the line `const float reaching = light_visibility(frame, shadow_ray_origin(frame, from, l), l, distance);` with:
```slang
    const float reaching = light_visibility(frame, shadow_ray_origin(frame, from, l), l, distance, from.self, from.clearance);
```

In `game-engine/shaders/shading.slangh`, replace the line `? lit * light_visibility(frame, shadow_ray_origin(frame, from, l), l, distance)` with:
```slang
                ? lit * light_visibility(frame, shadow_ray_origin(frame, from, l), l, distance, from.self, from.clearance)
```

In `game-engine/shaders/mesh.slang`, replace the line `instance.inverse,` with:
```slang
        instance.inverse,
        ~0u,
        0.0,
```

In `game-engine/shaders/mesh.slang`, replace the line `? light_visibility(frame, shadow_ray_origin(frame, from, frame.sun_direction), frame.sun_direction, infinite_distance)` with:
```slang
            ? light_visibility(frame, shadow_ray_origin(frame, from, frame.sun_direction), frame.sun_direction, infinite_distance, from.self, from.clearance)
```

In `game-engine/shaders/terrain.slang`, replace `fragmentMain` with:
```slang
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
        float3x3(0.0),
        float3x3(0.0),
        ~0u,
        0.0,
    };

    if (frame.view == view_shadow) {
        const float sun = frame.sun_direction.y > 0.0
            ? light_visibility(frame, shadow_ray_origin(frame, from, frame.sun_direction), frame.sun_direction, infinite_distance, from.self, from.clearance)
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

`game-engine/shaders/cull.slang`:
```slang
// GPU culling: which clusters to draw, from the world's placements (placements.h) and their models' level-of-detail hierarchies (clusters.h), turned into one mesh dispatch per draw list. Two phases (culling.h, record_culling), each a walk down from the placement cells, level by level, in four steps per level:
//   walkStartMain   one thread: how many items the level has, and the workgroups its steps take
//   walkMain        per item: what it produces, tested against the view, the distance, the depth pyramid and the error threshold
//   walkScanMain    one workgroup: where each workgroup's output goes, and the next level's size
//   walkWriteMain   per item: its output, into its place
// An item is a cell of placements, a placement, a node of one of its primitives' hierarchies, or a cluster. The early phase starts from every cell; whatever it sets aside on the depth pyramid alone goes to the late phase, which starts from those. Every write goes to a place the prefix sums fix: the result doesn't depend on which thread runs first. The clusters come out cell by cell, every draw list mixed, so a counting sort by list follows, in the same four steps (sortStartMain, sortCountMain, sortScanMain, sortWriteMain), and each list's clusters are a run. Between the phases, two steps build the depth pyramid (record_depth_pyramid):
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
    uint list_counts[13];   // cluster items per draw list; the thirteenth is the impostors
    uint level_clusters[16];  // cluster items per DAG level
    uint far_placements;    // placements past the draw distance, whole cells of them included
    uint level_items[16];   // items walked per level
    uint dropped_items;     // items a level, the candidates or the clusters had no room for
};

struct CullTables {
    PlacementCell *cells;        // the placement cells: the early phase's first level
    ClusterItem *items[2];       // the walk's levels, alternating
    uint *results;               // per item of the level: three counts (what it produced)
    uint *block_totals;          // per workgroup of the level: the sums of those
    uint *block_bases;           // prefix sums of block_totals
    WalkCounters *counters;
    ClusterItem *candidates;     // items set aside for the late phase
    ClusterItem *cluster_items;  // the clusters to draw, in the walk's order
    ClusterItem *sorted_items;   // the same, sorted by draw list: what the dispatches draw
    ClusterDispatch *dispatches; // one per draw list
    ClusterItem *early_candidates;  // the early phase's candidates: the late phase's first level
    WalkCounters *early_counters;   // the early phase's counters: how many
    uint *patches;               // the terrain patches this phase draws, packed (terrain.slangh)
    uint *patch_command;         // how many, then 1, 1: the mesh dispatch's workgroup counts
    uint *nodes;                 // the terrain selection's working lists: two runs of max_terrain_patches
    uint *early_patches;         // per patch (terrain_patch_index): the frame the early phase last drew it in
    uint cell_count;
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

// 3. Walking the world

// Whether a sphere is in view and, if asked, not hidden: the frustum test on the sphere, the pyramid test on its box.
bool sphere_in_view(float4x4 view_projection, float3 center, float radius) {
    return in_view(view_projection, center - radius, center + radius);
}

// The draw list an item to draw belongs to (culling.h's draw_list_index): for a cluster, its material's alpha mode, times whether it's double-sided and whether the placement mirrors; for a placement, the impostors'.
static const uint impostor_list = 12;

uint draw_list_of(FrameData *frame, ClusterItem item) {
    if (item_kind(item) == item_kind_placement) {
        return impostor_list;
    }

    const Instance instance = instance_of(frame, item.placement);
    const Material material = frame.materials[frame.primitive_data[item_primitive(item)].material];
    return material.alpha_mode * 4 + (material.double_sided != 0 ? 2 : 0) + (instance_mirrored(instance) ? 1 : 0);
}

// What walkMain found out about an item: how many items it sends to the next level, whether it goes to the candidates, whether it's a cluster to draw. Three words per item.
static const uint result_next = 0;
static const uint result_candidate = 1;
static const uint result_cluster = 2;

// The item's DAG error test: an error of `error` metres (already scaled by the placement) on a sphere at `center` with `radius` is over the threshold when it projects to more than cluster_error_pixels, measured at the sphere's nearest point, never nearer than the near plane (clusterlod's rule: the test must give a parent at least its children's answer, which the hierarchy's errors guarantee).
bool over_threshold(FrameData *frame, float3 center, float radius, float error) {
    const float distance = max(length(center) - radius, 0.05);
    return error * frame.cluster_error_scale > distance;
}

// A sphere's nearest point to the camera, in metres: how near what it holds can be.
float nearest(float3 center, float radius) {
    return max(length(center) - radius, 0.0);
}

// A box's: the camera's distance to the box, 0 inside it.
float nearest(float3 lo, float3 hi) {
    return length(max(max(lo, -hi), 0.0));
}

// Whether a sphere is too far to draw as a mesh (mesh_draw_distance) or too small to see, under a pixel across: what both the cells and the placements test. Both counts are kept for the title.
bool dropped(FrameData *frame, WalkCounters *counters, float distance, float radius, uint placements) {
    if (distance > frame.mesh_draw_distance) {
        InterlockedAdd(counters.far_placements, placements);
        return true;
    }

    if (radius * frame.subpixel_scale < distance) {
        InterlockedAdd(counters.subpixel_draws, placements);
        return true;
    }

    return false;
}

// A root of a primitive's hierarchy is worth walking when its error, the worst under it, is over the threshold at its sphere: the same test the node level would apply.
bool root_wanted(FrameData *frame, Instance instance, float scale, uint node_index) {
    const ClusterNode node = frame.cluster_nodes[node_index];
    return over_threshold(frame, instance_point(instance, node.center), node.radius * scale, node.error * scale);
}

// How many of a placement's roots are worth walking, over all its model's primitives.
uint root_count(FrameData *frame, Instance instance, Model model) {
    const float scale = instance_scale(instance);
    uint count = 0;

    for (uint p = 0; p < model.primitive_count; ++p) {
        const PrimitiveClusters primitive = frame.primitives[model.first_primitive + p];
        for (uint level = 0; level < primitive.levels; ++level) {
            count += root_wanted(frame, instance, scale, primitive.first_node + level) ? 1 : 0;
        }
    }

    return count;
}

// The item thread `i` of the level walks: the early phase's first level is the cells themselves, the late phase's the early phase's candidates; every other level is what the level before wrote.
ClusterItem item_at(CullTables *tables, uint i) {
    return push.level == 0 && push.late == 0 ? make_item(i, item_kind_cell, 0, 0)
        : push.level == 0 ? tables.early_candidates[i]
        : tables.items[push.level & 1][i];
}

// Before a level: how many items it has, and how many workgroups its steps take. The first level of the early phase is every cell; of the late phase, the early phase's candidates. Every other level is what the level before produced. The first level also clears the counts.
[shader("compute")]
[numthreads(1, 1, 1)]
void walkStartMain() {
    CullTables *tables = push.tables;
    WalkCounters *counters = tables.counters;

    if (push.level == 0) {
        counters.items = push.late != 0 ? tables.early_counters.candidates : tables.cell_count;
        counters.next_items = 0;
        counters.candidates = 0;
        counters.clusters = 0;
        counters.triangles = 0;
        counters.subpixel_draws = 0;
        counters.far_placements = 0;
        counters.dropped_items = 0;

        for (uint list = 0; list < 13; ++list) {
            counters.list_counts[list] = 0;
        }

        for (uint level = 0; level < 16; ++level) {
            counters.level_clusters[level] = 0;
            counters.level_items[level] = 0;
        }
    } else {
        counters.items = counters.next_items;
    }

    counters.level_items[min(push.level, 15)] = counters.items;
    counters.dispatch_x = (counters.items + walk_workgroup_size - 1) / walk_workgroup_size;
    counters.dispatch_y = 1;
    counters.dispatch_z = 1;
}

// Per item of the level: what it produces.
//   - A cell outside the view, past the draw distance or under a pixel produces nothing; one hidden by the pyramid goes to the candidates; otherwise its placements go to the next level.
//   - A placement is tested the same way, by its model's sphere. Under impostor_pixels across, it's drawn as an impostor: the item itself goes to the clusters. Otherwise its primitives' hierarchies' roots go on: one per DAG level, but only those whose error is over the threshold, since a root under it would be pruned at the next level anyway, and a tree of three primitives has 39 of them.
//   - A node is pruned when even its worst error is under the threshold, when it's out of view or when it's hidden (a candidate, then); otherwise its children go on, or, for a leaf, its group's clusters.
//   - A cluster is drawn when its own group is over the threshold and the group that refines it isn't, and it's in view, not hidden, and not facing away.
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
        const ClusterItem item = item_at(tables, i);
        const uint kind = item_kind(item);

        if (kind == item_kind_cell) {
            // The cell's box, from its cell to the camera: both corners in the same cell.
            const PlacementCell cell = tables.cells[item.placement];
            const float3 lo = camera_relative(frame, cell.cell, cell.bounds_min);
            const float3 hi = camera_relative(frame, cell.cell, cell.bounds_max);

            if (in_view(frame.view_projection, lo, hi) && !dropped(frame, counters, nearest(lo, hi), cell.max_radius, cell.count)) {
                if (occluded(frame, lo, hi)) {
                    candidate = 1;
                } else {
                    next = cell.count;
                }
            }
        } else {
            // Everything else is a placement, or belongs to one: its transform, and for a node or a cluster, a sphere in the model's space, whose centre goes through the transform and whose radius, like its error, grows by the most the transform stretches anything.
            const Instance instance = instance_of(frame, item.placement);
            const float scale = instance_scale(instance);

            if (kind == item_kind_placement) {
                const Model model = frame.models[instance.placement.model];
                const float3 center = instance_point(instance, model.center);
                const float radius = model.radius * scale;

                if (sphere_in_view(frame.view_projection, center, radius) && !dropped(frame, counters, nearest(center, radius), radius, 1)) {
                    if (occluded(frame, center - radius, center + radius)) {
                        candidate = 1;
                    } else if (radius * frame.impostor_scale < nearest(center, radius)) {
                        cluster = 1;
                    } else {
                        next = root_count(frame, instance, model);
                    }
                }
            } else if (kind == item_kind_node) {
                const ClusterNode node = frame.cluster_nodes[item_index(item)];
                const float3 center = instance_point(instance, node.center);
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
                const float3 center = instance_point(instance, c.center);
                const float radius = c.radius * scale;

                // The level of detail: coarse enough, and no coarser than it must be.
                const bool coarse_enough = over_threshold(frame, instance_point(instance, own.center), own.radius * scale, own.error * scale);
                const bool finer_unneeded = c.refined == ~0u
                    || !over_threshold(frame, instance_point(instance, frame.cluster_groups[c.refined].center),
                        frame.cluster_groups[c.refined].radius * scale, frame.cluster_groups[c.refined].error * scale);

                if (coarse_enough && finer_unneeded && sphere_in_view(frame.view_projection, center, radius)) {
                    // Facing away: every triangle's normal is within the cone's half-angle of its axis, so when the camera is behind all of them, none can face it. meshoptimizer's test, in the model's own space, where the cone and the sphere are: the camera goes there by the inverse transform, exact under any transform, mirroring included, since the front faces are the model's own. The sphere's share widens the test for a camera near the cluster. A cutoff of 1 means the cone is too wide to say.
                    const float3 camera_in_model = mul(instance.inverse, -instance.origin);
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

// Per item of the level: its output, into the places the prefix sums gave it. A cell writes its placements; a placement its primitives' roots that are worth walking; a node its children, or its group's clusters. A candidate goes to the candidates, a cluster to the clusters, counted by triangles, level and list.
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

    const ClusterItem item = item_at(tables, i);
    const uint kind = item_kind(item);
    ClusterItem *out = tables.items[(push.level + 1) & 1];

    if (next != 0) {
        const uint base = tables.block_bases[group_id.x * 3 + result_next] + next_prefix;

        // Past the capacity, nothing is written, and the count of what wasn't is kept.
        if (base + next > tables.item_capacity) {
            InterlockedAdd(counters.dropped_items, base + next - max(base, tables.item_capacity));
        }

        if (kind == item_kind_cell) {
            const PlacementCell cell = tables.cells[item.placement];
            for (uint k = 0; k < next && base + k < tables.item_capacity; ++k) {
                out[base + k] = make_item(cell.first + k, item_kind_placement, 0, 0);
            }
        } else if (kind == item_kind_placement) {
            // The roots worth walking, found again as walkMain found them.
            const Instance instance = instance_of(frame, item.placement);
            const Model model = frame.models[instance.placement.model];
            const float scale = instance_scale(instance);
            uint k = 0;
            for (uint p = 0; p < model.primitive_count; ++p) {
                const PrimitiveClusters primitive = frame.primitives[model.first_primitive + p];
                for (uint level = 0; level < primitive.levels && base + k < tables.item_capacity; ++level) {
                    if (root_wanted(frame, instance, scale, primitive.first_node + level)) {
                        out[base + k] = make_item(item.placement, item_kind_node, model.first_primitive + p, primitive.first_node + level);
                        ++k;
                    }
                }
            }
        } else {
            const ClusterNode node = frame.cluster_nodes[item_index(item)];
            for (uint k = 0; k < next && base + k < tables.item_capacity; ++k) {
                out[base + k] = node.group == ~0u
                    ? make_item(item.placement, item_kind_node, item_primitive(item), node.first_child + k)
                    : make_item(item.placement, item_kind_cluster, item_primitive(item), frame.cluster_groups[node.group].first_cluster + k);
            }
        }
    }

    if (candidate != 0) {
        const uint at = tables.block_bases[group_id.x * 3 + result_candidate] + candidate_prefix;
        if (at < tables.item_capacity) {
            tables.candidates[at] = item;
        } else {
            InterlockedAdd(counters.dropped_items, 1);
        }
    }

    if (cluster != 0) {
        const uint at = tables.block_bases[group_id.x * 3 + result_cluster] + cluster_prefix;
        if (at < tables.item_capacity) {
            tables.cluster_items[at] = item;
        } else {
            InterlockedAdd(counters.dropped_items, 1);
        }

        // An impostor is two triangles; a cluster its own.
        if (kind == item_kind_placement) {
            InterlockedAdd(counters.triangles, 2);
        } else {
            const Cluster c = frame.clusters[item_index(item)];
            InterlockedAdd(counters.triangles, c.triangle_count);
            InterlockedAdd(counters.level_clusters[min(frame.cluster_groups[c.group].depth, 15)], 1);
        }

        InterlockedAdd(counters.list_counts[draw_list_of(frame, item)], 1);
    }
}

// 4. The sort by list, and the dispatches

// The clusters came out cell by cell, every list mixed, impostors among them. A counting sort puts each list's together: per workgroup of clusters, how many of each list (sortCountMain); the prefix sums of those, per list, starting where the lists before it end (sortScanMain); then each cluster into its list's run (sortWriteMain). Thirteen prefix sums per workgroup instead of three, and the same rule: every cluster's place is fixed by the sums, so the lists come out in the walk's order, every frame the same.

// Before the sort: a workgroup per walk_workgroup_size clusters.
[shader("compute")]
[numthreads(1, 1, 1)]
void sortStartMain() {
    WalkCounters *counters = push.tables.counters;
    counters.dispatch_x = (min(counters.clusters, push.tables.item_capacity) + walk_workgroup_size - 1) / walk_workgroup_size;
    counters.dispatch_y = 1;
    counters.dispatch_z = 1;
}

// The cluster's list, or 13 for a thread past the clusters: in no list.
uint sort_list(FrameData *frame, CullTables *tables, uint i) {
    if (i >= min(tables.counters.clusters, tables.item_capacity)) {
        return 13;
    }

    return draw_list_of(frame, tables.cluster_items[i]);
}

// Per cluster: its list; per workgroup: how many of each.
[shader("compute")]
[numthreads(walk_workgroup_size, 1, 1)]
void sortCountMain(uint3 id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint thread : SV_GroupThreadID) {
    const uint list = sort_list(push.frame, push.tables, id.x);
    uint prefix;

    for (uint l = 0; l < 13; ++l) {
        const uint total = group_exclusive_scan(list == l ? 1 : 0, thread, prefix);
        if (thread == 0) {
            push.tables.block_totals[group_id.x * 13 + l] = total;
        }
    }
}

// One workgroup: for each list, the exclusive prefix sums of the workgroups' counts, starting at the list's start: the sum of the lists before it, which the walk counted.
[shader("compute")]
[numthreads(scan_size, 1, 1)]
void sortScanMain(uint3 id : SV_GroupThreadID) {
    CullTables *tables = push.tables;
    WalkCounters *counters = tables.counters;
    const uint thread = id.x;
    const uint blocks = counters.dispatch_x;
    uint start = 0;

    for (uint l = 0; l < 13; ++l) {
        // The blocks, scan_size at a time, carrying the running total across.
        uint carry = start;

        for (uint first = 0; first < blocks; first += scan_size) {
            const uint b = first + thread;
            const uint value = b < blocks ? tables.block_totals[b * 13 + l] : 0;
            uint prefix;
            const uint total = group_exclusive_scan(value, thread, prefix);

            if (b < blocks) {
                tables.block_bases[b * 13 + l] = carry + prefix;
            }

            carry += total;
        }

        start += counters.list_counts[l];
    }
}

// Per cluster: into its list's run, at its workgroup's base for the list plus its place among the workgroup's clusters of that list.
[shader("compute")]
[numthreads(walk_workgroup_size, 1, 1)]
void sortWriteMain(uint3 id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint thread : SV_GroupThreadID) {
    CullTables *tables = push.tables;
    const uint list = sort_list(push.frame, tables, id.x);
    uint place = 0;

    for (uint l = 0; l < 13; ++l) {
        uint prefix;
        group_exclusive_scan(list == l ? 1 : 0, thread, prefix);
        place = list == l ? tables.block_bases[group_id.x * 13 + l] + prefix : place;
    }

    if (list < 13) {
        tables.sorted_items[place] = tables.cluster_items[id.x];
    }
}

// After the sort: each draw list's dispatch. A list's clusters are a run of the sorted items: its start is the lists' before it, summed. A cluster is a workgroup; impostors go 32 to a workgroup (impostor.slang). The width is capped at the 65,535 workgroups Vulkan guarantees per dimension; y takes the rest, and the mesh shaders skip the workgroups past the count.
static const uint impostors_per_workgroup = 32;

[shader("compute")]
[numthreads(1, 1, 1)]
void writeDispatchesMain() {
    CullTables *tables = push.tables;
    WalkCounters *counters = tables.counters;
    uint start = 0;

    for (uint list = 0; list < 13; ++list) {
        const uint count = counters.list_counts[list];
        const uint workgroups = list == impostor_list ? (count + impostors_per_workgroup - 1) / impostors_per_workgroup : count;
        tables.dispatches[list] = ClusterDispatch(min(workgroups, max_dispatch_width), (workgroups + max_dispatch_width - 1) / max_dispatch_width, 1, count, start, 0, tables.sorted_items);
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
#include "includes/placements.h"
#include "includes/shader_types.h"
#include "includes/terrain.h"
#include "includes/vulkan_setup.h"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

// Draw lists

// One mesh dispatch draws a whole list of clusters with one pipeline and one dynamic state, so clusters are sorted into lists by what those fix:
//   - the alpha mode, which picks the pipeline,
//   - whether the material is double-sided, which sets the cull mode,
//   - whether the placement mirrors, which sets the front face.
// Three alpha modes times two times two: twelve lists, numbered alpha mode x 4 + double-sided x 2 + mirrored. A thirteenth holds the impostors (impostors.h): placements drawn as one quad each, 32 to a workgroup.
constexpr std::uint32_t draw_list_count = 13;
constexpr std::uint32_t impostor_list = 12;
constexpr std::uint32_t impostors_per_workgroup = 32;

constexpr std::uint32_t draw_list_index(AlphaMode alpha_mode, bool double_sided, bool mirrored) {
    return static_cast<std::uint32_t>(alpha_mode) * 4 + (double_sided ? 2 : 0) + (mirrored ? 1 : 0);
}

// GPU culling

// The cull's compute pipelines and buffers, for a world whose placements never change.
//   - The walk starts from the placement cells (placements.h): a thread per column of cells, kept when its box is in view, within the draw distance, not under a pixel, and not hidden: not wholly behind what the depth pyramid holds. The pyramid is the depth buffer with a mip chain where each texel holds the farthest depth of the texels above it, so a box can be tested against the depth under its whole screen rectangle in four reads. A kept cell's placements are the next level's items, tested the same way by their models' spheres.
//   - A kept placement is walked: each of its model's primitives' hierarchy of clusters (clusters.h), from the roots down, each node tested like the placement, and against the error threshold, until the clusters at the one level of detail the distance calls for. The walk goes level by level, every level one item per thread: what each item produces, prefix sums across the workgroups, then the items written where the sums put them.
//   - Every frame, the cull runs in two phases around the depth prepass. The early phase walks every cell against the pyramid the previous frame built, under this frame's view: a guess, right wherever the view hasn't changed. Whatever it sets aside on the pyramid alone, a cell, a placement, a node or a cluster, goes on a list of candidates. The prepass draws what it keeps, the pyramid is built from that depth, and the late phase walks the candidates against it. Whatever the guess hid wrongly is drawn late; nothing visible is missed, and nothing is drawn twice.
//   - A placement whose sphere projects under impostor_pixels across isn't walked at all: it goes straight to the clusters as itself, an impostor to draw (impostors.h).
//   - The clusters come out in the walk's order, cell by cell, with every list mixed together, so a counting sort by list follows, with the same prefix sums: then each list's clusters are a run, and one dispatch draws it.
//   - The terrain is culled the same way: each phase walks the height field's quadtree down to patches, and lists the patches to draw.
// Everything a step writes goes to a place fixed by prefix sums over the previous steps' results, never by which thread got there first: the same view gives the same clusters, in the same order, every frame.
enum class CullPhase : std::size_t {
    early,
    late,
};

constexpr std::array cull_phases{CullPhase::early, CullPhase::late};

// The most items a level of the walk, the candidates or the clusters to draw can hold: 2^21, room for every placement in this world at once. A view from high up walks half a million placements; whatever a level can't hold is dropped and counted (WalkCounters::dropped_items), and the title says so.
constexpr std::uint32_t max_cull_items = 1u << 21;

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
    Buffer cluster_items;         // the clusters to draw, in the walk's order
    Buffer sorted_items;          // the same, sorted by draw list
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
    vk::raii::Pipeline sort_start = nullptr;
    vk::raii::Pipeline sort_count = nullptr;
    vk::raii::Pipeline sort_scan = nullptr;
    vk::raii::Pipeline sort_write = nullptr;
    vk::raii::Pipeline write_dispatches = nullptr;
    vk::raii::Pipeline copy_depth = nullptr;
    vk::raii::Pipeline reduce_depth = nullptr;
    vk::raii::Pipeline select_early_patches = nullptr;
    vk::raii::Pipeline select_late_patches = nullptr;

    // Written once, shared by the phases.
    Buffer early_patches;  // per terrain patch, the frame the early phase last drew it in

    std::array<CullPhaseBuffers, cull_phases.size()> phases;

    std::uint32_t cell_count = 0;
    std::uint32_t walk_levels = 0;  // how many levels the walk takes: the cells, their placements, the hierarchies' depth, a group's clusters
};

// `placements`, `terrain` and `clusters` are what the phases walk.
DrawCulling create_draw_culling(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    const PlacementBuffers &placements,
    const Terrain &terrain,
    const Clusters &clusters
);

// The totals, as the cull copies them out: each phase's counters, its terrain patches, and the frame's shadow instances (acceleration.h).
struct CullTotals {
    WalkCounters early;
    WalkCounters late;
    std::uint32_t early_patches;
    std::uint32_t late_patches;
    std::uint32_t shadow_instances;
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

## 21.4 The frame: `main.cpp`

### Why
The bake after the materials, since it draws with them; the impostor pipelines beside the terrain's; the impostors drawn in both halves of the prepass and in the lighting pass, after the solid lists and before the terrain; the threshold; and the title's count.

### How
- **`impostor_pixels`**, 32, next to the other knobs; 0 means none, the Chapter 20 engine.
- **The heap** reserves three slots per model after the environment's.
- **`bake_impostors`** gets a `FrameData` with every pointer the mesh shader reads, built by assignment, and reports its time: under a second.
- **`draw_impostors`** binds the impostor pipeline, pushes the thirteenth list's dispatch, and draws it with no culling.
- **The title** shows clusters and impostors apart, and the triangles, now counting an impostor as two.

### Code
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
#include "includes/impostors.h"
#include "includes/light_clusters.h"
#include "includes/pipeline.h"
#include "includes/placements.h"
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

    // Memory

    // Device-local memory in use and the budget the driver allows, in megabytes, from VK_EXT_memory_budget: the driver's own figures, an estimate by the spec's word, and on this driver one that includes what it allocates behind our backs, for the acceleration structures and the pipelines among others.
    std::pair<std::uint64_t, std::uint64_t> device_memory(const GpuChoice &gpu) {
        const auto properties = gpu.device.getMemoryProperties2<vk::PhysicalDeviceMemoryProperties2, vk::PhysicalDeviceMemoryBudgetPropertiesEXT>();
        const vk::PhysicalDeviceMemoryProperties &heaps = properties.get<vk::PhysicalDeviceMemoryProperties2>().memoryProperties;
        const auto &budget = properties.get<vk::PhysicalDeviceMemoryBudgetPropertiesEXT>();
        std::uint64_t used = 0;
        std::uint64_t available = 0;

        for (std::uint32_t i = 0; i < heaps.memoryHeapCount; ++i) {
            if (heaps.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                used += budget.heapUsage[i];
                available += budget.heapBudget[i];
            }
        }

        return {used >> 20, available >> 20};
    }

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
        vk::raii::Pipeline impostor_prepass = nullptr;
        vk::raii::Pipeline impostor_lighting = nullptr;
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
        bool see_through = false;      // whether the scene has blended primitives at all
        glm::vec3 sun_direction{0.0f}; // toward the sun
        float altitude = 0.0f;         // the camera's height above the ground, in metres
        std::size_t frame_index = 0;   // which frame in flight this is: whose TLAS
        std::uint32_t placement_count = 0;
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

    // Draws the impostors cull phase `phase` kept, with `pipeline`: the thirteenth list's dispatch, 32 to a workgroup, every quad seen from both sides.
    void draw_impostors(
        const vk::raii::CommandBuffer &commands,
        const DrawCulling &culling,
        const DrawList &draws,
        CullPhase phase,
        const vk::raii::Pipeline &pipeline
    ) {
        commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

        const PushData push{.frame = draws.frame, .drawn = dispatch_address(culling, phase, impostor_list)};
        commands.pushDataEXT(vk::PushDataInfoEXT{
            .offset = 0,
            .data = {.address = &push, .size = sizeof(push)},
        });

        commands.setCullMode(vk::CullModeFlagBits::eNone);
        commands.setFrontFace(front_face);
        draw_list(commands, culling, phase, impostor_list);
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

    // Records a frame: the atmosphere's tables, the light clusters and this frame's shadow TLAS in compute shaders, then five passes, with the cull's two phases around the first:
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
        const AccelerationStructures &acceleration,
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

        // The shadow TLAS

        // This frame's ray-tracing instances, the placements within shadow_radius of the camera, and the TLAS over them, which the lighting pass traces against.
        record_shadow_tlas(commands, acceleration, draws.frame_index, draws.frame, draws.placement_count,
            draws.readback, offsetof(CullTotals, shadow_instances));

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

        // Solid surfaces only: opaque, then masked, then the impostors, then the terrain.
        for (std::size_t i = 0; i < solid_modes.size(); ++i) {
            draw_mode(commands, culling, draws, CullPhase::early, solid_modes[i], pipelines.prepass[i]);
        }

        draw_impostors(commands, culling, draws, CullPhase::early, pipelines.impostor_prepass);
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

        draw_impostors(commands, culling, draws, CullPhase::late, pipelines.impostor_prepass);
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
            draw_impostors(commands, culling, draws, phase, pipelines.impostor_lighting);
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
        pipelines.impostor_prepass = create_impostor_pipeline(device, std::array{normal_format}, depth_format, MeshPass::depth_normals);
        pipelines.impostor_lighting = create_impostor_pipeline(device, std::array{hdr_format}, depth_format, MeshPass::lighting);

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

        // The glTF file to draw, under lecture-md/game-engine/assets: the world, two million placements of eight models over the terrain, through EXT_mesh_gpu_instancing (tools/world_scene.py). utility_box_02/BoxField.gltf and Sponza/Sponza.gltf draw as before: every node is a placement now.
        const std::filesystem::path scene_file = std::filesystem::path(ASSET_DIR) / "world/world.gltf";

        // Where in the world the scene is placed, in metres: its origin. Move it far away, to {100000.0, 0.0, 100000.0} say, 141 km out, and the image stays the same: everything is drawn relative to the camera.
        const glm::dvec3 scene_origin{0.0, 0.0, 0.0};

        const std::uint64_t load_start = SDL_GetTicksNS();
        Scene scene = load_gltf(scene_file, scene_origin);
        std::println("Loaded {}: {} vertices, {} triangles, {} primitives in {} models, {} placements, {} materials, {} images, in {:.0f} ms",
            scene_file.filename().string(), scene.vertices.size(), scene.indices.size() / 3, scene.primitives.size(),
            scene.models.size(), scene.placements.size(), scene.materials.size(), scene.images.size(),
            static_cast<double>(SDL_GetTicksNS() - load_start) * 1e-6);

        // The terrain under the scene: a height field 8 km across, as its file has it, with its flat middle at -1 m, which the world's placements were scattered onto; its materials come from a file of their own.
        const std::filesystem::path terrain_dir = std::filesystem::path(ASSET_DIR) / "terrain";
        const std::uint32_t terrain_materials = add_materials(scene, terrain_dir / "terrain.gltf");

        // Lights to test the clusters with, through the scene's box: 0 for just the file's. Try 16, 256 and 1024.
        constexpr std::uint32_t test_lights = 0;
        add_test_lights(scene, test_lights);

        // How far a level of detail's error may project, in pixels, before a finer level takes over: 1, which no eye catches. 2 halves the triangles for a visible softening; 0.5 doubles them.
        constexpr float cluster_error_pixels = 1.0f;

        // How far placements are drawn as meshes, in metres: no limit, the sub-pixel test alone decides, now that every model's level of detail reaches a few dozen triangles. 512 m makes a ring, the chapter's table has what each costs. And how far they cast ray-traced shadows: the TLAS holds the placements within this radius, up to shadow_instance_capacity.
        constexpr float mesh_draw_distance = std::numeric_limits<float>::infinity();
        constexpr float shadow_radius = 512.0f;

        // How small a placement is on screen, in pixels across, before it's drawn as an impostor instead of its clusters: 32. The chapter's table measures 16 and 64; 0 draws every placement as its clusters, as Chapter 20 did.
        constexpr float impostor_pixels = 32.0f;

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
            0.0f, terrain_materials, terrain_materials + 1, 4.0f);

        std::println("Terrain: {} x {} samples, {} patch levels, {} patches in {:.0f} ms", terrain.data.samples, terrain.data.samples,
            terrain.patch_levels, terrain.patch_count, static_cast<double>(SDL_GetTicksNS() - terrain_start) * 1e-6);

        // Which alpha modes the models use: the transparency pass runs only when something is see-through.
        std::array<std::size_t, 3> mode_primitives{};

        for (const Primitive &primitive : scene.primitives) {
            ++mode_primitives[static_cast<std::size_t>(scene.materials[primitive.material].alpha_mode)];
        }

        std::println("Primitives: {} opaque, {} masked, {} blended", mode_primitives[0], mode_primitives[1], mode_primitives[2]);

        // Vertices are read through pointers; indices go to the GPU's index fetch, so that buffer is an index buffer. Vertices and indices are also what the acceleration structures are built from, and shadow rays read indices through a pointer too.
        const vk::BufferUsageFlags build_input = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
            | vk::BufferUsageFlagBits::eShaderDeviceAddress;
        const Buffer vertex_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.vertices)), build_input);
        const Buffer index_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.indices)), vk::BufferUsageFlagBits::eIndexBuffer | build_input);

        // Every primitive's clusters and its level-of-detail hierarchy.
        const std::uint64_t cluster_start = SDL_GetTicksNS();
        const Clusters clusters = build_clusters(device, *gpu, queue, command_pool, scene);

        std::println("Clusters: {} in {} groups, {} nodes, {} levels, in {:.0f} ms", clusters.cluster_count, clusters.group_count,
            clusters.node_count, clusters.levels, static_cast<double>(SDL_GetTicksNS() - cluster_start) * 1e-6);

        for (std::size_t level = 0; level < clusters.level_triangles.size(); ++level) {
            std::println("  level {}: {} triangles", level, clusters.level_triangles[level]);
        }

        // Most files have no lights, and a buffer can't be empty: then there's no buffer, and the shader's light count is 0.
        const Buffer light_buffer = scene.lights.empty() ? Buffer{} : upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.lights)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        std::println("Lights: {} ({} directional), plus the sun", scene.lights.size(), directional_count);

        // Acceleration structures

        // The models' BLASes, once, and each frame's TLAS, ready to be built every frame.
        const std::uint64_t acceleration_start = SDL_GetTicksNS();
        const AccelerationStructures acceleration = build_acceleration_structures(device, *gpu, queue, command_pool,
            scene, vertex_buffer, index_buffer, frames_in_flight);

        std::println("Acceleration structures: {} BLAS in {:.0f} ms; {} TLAS of up to {} instances", acceleration.blases.size(),
            static_cast<double>(SDL_GetTicksNS() - acceleration_start) * 1e-6, acceleration.frames.size(), shadow_instance_capacity);

        // Placements

        // The placements, sorted into cells and uploaded with their models, whose BLASes they now know.
        const std::uint64_t placement_start = SDL_GetTicksNS();
        const PlacementBuffers placements = upload_placements(device, *gpu, queue, command_pool, scene, acceleration.blas_addresses);

        std::println("Placements: {} in {} cells, in {:.0f} ms", placements.placement_count, placements.cell_count,
            static_cast<double>(SDL_GetTicksNS() - placement_start) * 1e-6);

        // The cull, for these cells, the hierarchies it walks and the terrain it picks patches of.
        const DrawCulling culling = create_draw_culling(device, *gpu, queue, command_pool, placements, terrain, clusters);

        std::println("Culling: {} cells, {} walk levels", culling.cell_count, culling.walk_levels);

        // Textures and materials

        // Decode every image, upload them with mipmaps, and describe them in the descriptor heap. Texture 0 is white; scene image i is texture i + 1.
        const std::uint64_t texture_start = SDL_GetTicksNS();
        const std::vector<Texture> textures = create_scene_textures(device, *gpu, queue, command_pool, scene);
        // After the textures: the swapchain images' slots, then the environment's, then the impostor atlases', three per model.
        const auto impostor_slot_count = static_cast<std::uint32_t>(scene.models.size()) * impostor_slots_per_model;
        DescriptorHeaps heaps = create_descriptor_heaps(device, *gpu, queue, command_pool, textures, scene.samplers,
            screen_slot_count + environment_slot_count + impostor_slot_count);

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

        // The impostors

        // Every model's pictures, baked by the mesh pipeline itself: it needs what the mesh shader reads.
        const std::uint64_t impostor_start = SDL_GetTicksNS();
        FrameData bake_frame{};
        bake_frame.vertices = vertex_buffer.address;
        bake_frame.indices = index_buffer.address;
        bake_frame.materials = material_buffer.address;
        bake_frame.clamp_sampler = environment.clamp_sampler;
        bake_frame.clusters = clusters.clusters.address;
        bake_frame.cluster_vertices = clusters.vertices.address;
        bake_frame.cluster_triangles = clusters.triangles.address;
        bake_frame.cluster_groups = clusters.groups.address;
        bake_frame.cluster_nodes = clusters.nodes.address;
        bake_frame.primitives = clusters.primitives.address;
        bake_frame.models = placements.models.address;
        bake_frame.primitive_data = placements.primitives.address;

        const Impostors impostors = bake_impostors(device, *gpu, queue, command_pool, heaps,
            first_screen_slot + screen_slot_count + environment_slot_count, scene, clusters, bake_frame);

        std::println("Impostors: {} models, {} x {} frames in {} x {} atlases, in {:.0f} ms", scene.models.size(), impostor_grid, impostor_grid,
            impostor_atlas_size, impostor_atlas_size, static_cast<double>(SDL_GetTicksNS() - impostor_start) * 1e-6);

        // Spawns at the scene's origin, at eye height above the terrain, looking down -Z.
        FlyCamera camera{.position = scene_origin + glm::dvec3{0.0, terrain_height_at(terrain, scene_origin) + 1.7, 0.0}};
        CameraInput input;
        Settings settings;
        Settings shown_settings{.hours = -1.0f};  // what the title shows; differs at first
        CullTotals shown_totals{};                // the cull's totals the title shows
        std::uint64_t shown_memory = 0;           // the memory in use it shows, in megabytes
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
            const std::uint32_t impostors_drawn = totals.early.list_counts[impostor_list] + totals.late.list_counts[impostor_list];
            const std::uint32_t triangles_drawn = totals.early.triangles + totals.late.triangles;

            // The title shows the view, the sky, the time, the exposure, whether ambient occlusion is on, what the cull kept and dropped, the shadow instances and the GPU's memory, whenever one changes.
            const auto [memory_used, memory_budget] = device_memory(*gpu);

            if (settings.view != shown_settings.view || settings.sky != shown_settings.sky
                || settings.hours != shown_settings.hours
                || settings.exposure_compensation != shown_settings.exposure_compensation
                || settings.ambient_occlusion != shown_settings.ambient_occlusion
                || clusters_drawn != shown_totals.early.clusters + shown_totals.late.clusters || totals.late.clusters != shown_totals.late.clusters
                || triangles_drawn != shown_totals.early.triangles + shown_totals.late.triangles
                || totals.early.subpixel_draws != shown_totals.early.subpixel_draws
                || totals.early.far_placements != shown_totals.early.far_placements
                || impostors_drawn != shown_totals.early.list_counts[impostor_list] + shown_totals.late.list_counts[impostor_list]
                || totals.early_patches != shown_totals.early_patches || totals.late_patches != shown_totals.late_patches
                || totals.shadow_instances != shown_totals.shadow_instances || memory_used != shown_memory
                || totals.early.dropped_items + totals.late.dropped_items != shown_totals.early.dropped_items + shown_totals.late.dropped_items) {
                const auto minutes = static_cast<int>(std::lround(settings.hours * 60.0f));
                const std::string sky = settings.sky == SkySource::atmosphere
                    ? std::format("sky {:02}:{:02}", minutes / 60, minutes % 60)
                    : std::string("photographed sky");
                // Dropped items mean a level of the walk overflowed max_cull_items: whatever didn't fit wasn't drawn.
                const std::uint32_t dropped = totals.early.dropped_items + totals.late.dropped_items;
                const std::string title = std::format("game-engine: {}, {}, EV {:.1f}, AO {}, {} clusters ({} late), {} impostors, {} triangles, {} patches ({} late), {} sub-pixel, {} far, {} shadow instances, {} / {} MB{}",
                    view_names[static_cast<std::size_t>(settings.view)], sky, ev100, settings.ambient_occlusion ? "on" : "off",
                    clusters_drawn - impostors_drawn, totals.late.clusters, impostors_drawn, triangles_drawn, totals.early_patches + totals.late_patches, totals.late_patches,
                    totals.early.subpixel_draws, totals.early.far_placements, totals.shadow_instances, memory_used, memory_budget,
                    dropped != 0 ? std::format(", {} DROPPED", dropped) : "");

                SDL_SetWindowTitle(window.get(), title.c_str());
                shown_settings = settings;
                shown_totals = totals;
                shown_memory = memory_used;
            }

            // Every two seconds or so, the clusters drawn per level of detail, for setting cluster_error_pixels against a frame budget, level 0 being full detail; and the items each level of the walk looked at, cells first, then placements, then nodes and clusters.
            if (frame_count % 120 == 0 && clusters_drawn != 0) {
                std::string levels;
                for (std::uint32_t level = 0; level < std::min<std::uint32_t>(clusters.levels, 16); ++level) {
                    levels += std::format(" {}", totals.early.level_clusters[level] + totals.late.level_clusters[level]);
                }
                std::string items;
                for (std::uint32_t level = 0; level < std::min<std::uint32_t>(culling.walk_levels, 16); ++level) {
                    items += std::format(" {}", totals.early.level_items[level] + totals.late.level_items[level]);
                }
                std::println("Clusters per level:{}; {} triangles. Items per walk level:{}", levels, triangles_drawn, items);
            }

            // Where the camera is: a cell and an offset, like everything the GPU places. This frame's TLAS is built around its cell (acceleration.h).
            const CellPosition camera_at = to_cell(camera.position);

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
                .placements = placements.placements.address,
                .materials = material_buffer.address,
                .lights = light_buffer.address,
                .environment = environment.info.address,
                .scene_tlas = acceleration.frames[frame_count % frames_in_flight].address,
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
                .tlas_offset = camera_at.offset,
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
                .models = placements.models.address,
                .primitive_data = placements.primitives.address,
                .cells = placements.cells.address,
                .mesh_draw_distance = mesh_draw_distance,
                .shadow_radius = shadow_radius,
                .shadow_capacity = shadow_instance_capacity,
                .placement_count = placements.placement_count,
                .atlases = impostors.atlas_table.address,
                .impostor_frames = impostors.frames.address,
                .impostor_scale = impostor_pixels > 0.0f ? 2.0f * focal / impostor_pixels : std::numeric_limits<float>::infinity(),
                .impostor_grid = impostor_grid,
            };

            const DrawList draws{
                .index_buffer = *index_buffer.handle,
                .frame = frame.data.address,
                .screen = screen,
                .view = settings.view,
                .readback = *frame.cull_totals.handle,
                .see_through = mode_primitives[static_cast<std::size_t>(AlphaMode::blend)] > 0,
                .sun_direction = sun_direction,
                .altitude = altitude,
                .frame_index = frame_count % frames_in_flight,
                .placement_count = placements.placement_count,
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
            record_frame(frame.commands, swapchain, image_index, pipelines, ambient_occlusion, culling, acceleration, light_clusters, environment, heaps, draws);

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

## 21.5 Build and run

At the end of `game-engine/lsan.supp`, add:
```text
leak:libwayland-client
leak:libdecor
```

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The terminal** shows the loads of Chapter 20, then `Impostors: 8 models, 8 x 8 frames in 1024 x 1024 atlases, in 611 ms`, half a second in release; under the sanitizers the mips' readbacks make it thirty.
- **The image** at the start is Chapter 20's, every placement within a few dozen metres still its clusters. The title starts at `4389 clusters (0 late), 41516 impostors, 529604 triangles, 1475 patches (0 late), 149461 sub-pixel, 0 far, 24771 shadow instances, 943 / 7656 MB`: 41,000 impostors in view from the plateau, and the triangles down from Chapter 20's 5.6 million to 530,000.
- **Walk away from a tree** and watch the title: at about 240 m its clusters become an impostor, with a small jump in its silhouette and none in its brightness. The test is on the model's sphere, the one around its box, which for the tree is 8 m across, wider than its 5 m of crown: that's what is 32 pixels at 240 m; a boulder's switch comes nearer.
- **The hillsides** keep their trees to the horizon, green: what the sub-pixel test drops is still only what's under a pixel.
- **No `[validation …]` lines.** (One run in ten of this chapter's captures printed a reserved-range report from the descriptor heap validation, with a range that was plainly garbage; it never reproduced, and the binding it names is the one every chapter has made every frame.) The sanitizer build, option 11, once reported a leak inside libdecor's GTK plugin, SDL's window decorations on Wayland, from a stack that never enters our code: two more lines in `lsan.supp` keep it quiet.

**The check.** Two capture runs of the same views, one with the threshold forced to 0, every placement its clusters, one with it forced to infinity, every placement its impostor, compared pixel for pixel. At 80 m from a tree, 4% of the pixels differ by more than 32 of 255, and the band through the trees is within 1% in brightness: the pictures are the right way up, from the right directions, and lit the same. At 20 m the impostor is plainly a cut-out, as a 128-pixel picture must be at 300 pixels on screen; that's what the threshold is for. From the hilltop the impostor hillside is greener than the cluster one, since a tree's coarsest clusters are a few leaf cards that mostly fail the alpha test by then: the pictures' coverage-preserving mips keep the canopy.

**What it costs.** Release, 1920 × 1080, RTX 5070 Laptop, by day, the three views of Chapter 20, at four thresholds:

| Threshold | View | Clusters | Impostors | Triangles | Frame |
|---|---|---|---|---|---|
| 32 px | Eye level | 4,389 | 41,516 | 530,000 | 6.0 ms |
| | 40 m up | 2,379 | 65,899 | 357,000 | 7.9 ms |
| | Hilltop | 0 | 130,959 | 262,000 | 8.9 ms |
| 16 px | Eye level | 8,895 | 40,926 | 931,000 | 6.0 ms |
| | 40 m up | 7,098 | 65,010 | 783,000 | 7.7 ms |
| | Hilltop | 1,174 | 130,570 | 333,000 | 8.7 ms |
| 64 px | Eye level | 2,364 | 41,751 | 335,000 | 6.1 ms |
| | 40 m up | 386 | 66,111 | 164,000 | 7.4 ms |
| | Hilltop | 0 | 130,959 | 262,000 | 7.9 ms |
| None | Eye level | 68,626 | 0 | 5.59 M | 7.8 ms |
| | 40 m up | 106,193 | 0 | 8.29 M | 8.7 ms |
| | Hilltop | 216,131 | 0 | 15.7 M | 9.7 ms |

- **The hilltop** at 32 pixels: 131,000 impostors, 262,000 triangles, 8.9 ms, against the same view without impostors, 216,000 clusters, 15.7 million triangles and 9.7 ms in this run. The triangles fall sixty-fold and the frame by less than a millisecond, because the lighting pass is where the time goes, as it was: a pixel costs its shadow rays whatever it belongs to, and the impostors' rays pay a little more for looking at their own hits. The prepass is what the triangles bought: 2.2 ms down to 0.7.
- **Eye level** goes from 7.8 to 6.0 ms, with 41,000 impostors beyond their switch distances, 100 to 250 m by model, and the near field still clusters.
- **16 against 64 pixels** moves 600,000 triangles at eye level and a tenth of a millisecond. The threshold is a quality knob with a small price either way; the atlas's resolution is what sets where it can go.
- **Memory:** the atlases add 130 MB: the title reads 943 MB against Chapter 20's 813.

Next, in Chapter 22, what's left beyond a pixel: 374,000 placements the hilltop still drops, as something per cell rather than per placement.

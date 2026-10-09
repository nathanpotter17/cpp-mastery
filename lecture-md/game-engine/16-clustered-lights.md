# Chapter 16: Clustered lights

By the end of this chapter, a scene can hold thousands of point and spot lights, and each pixel only pays for the ones that reach it. Until now, every pixel looped over every light in the file. That was fine for the handful of lights glTF sample files carry, and hopeless for a street of lamps: every pixel visited every lamp, even the ones whose range stopped kilometres short of it. A lamp without a range was worse: it reached every pixel facing it, and traced a shadow ray at each.

The fix is **clustered shading** (Olsson, Billeter and Assarsson, "Clustered Deferred and Forward Shading", HPG 2012). The view is cut into a grid of boxes, **clusters**: tiles of 64 × 64 pixels across the screen, times 24 slices in depth. Each frame, compute shaders work out which lights reach which clusters, and the fragment shader lights each pixel with only its cluster's lights. Our grid is the dense, fixed kind, with every light tested against every cluster, as Persson and Olsson's "Practical Clustered Shading" (SIGGRAPH 2013) describes it; the original paper clusters only the depths the screen actually holds. Slices are spaced two per doubling of the distance: each is about 41% of its distance deep, so clusters keep the same shape at every distance. With these tiles, that shape is about six times deeper than wide; near-cubic clusters, which the original paper aims for, would take about ten slices per doubling, and a much larger grid.

Each cluster stores its lights as bits: one per light in view, up to 4,096 of them. A pixel walks its cluster's bits in order, so the same view always lights a pixel with the same lights in the same order, and no thread races another to write a list.

Clusters need every light to stop somewhere. KHR_lights_punctual lets a file leave a light's range out, and then the light falls off with the square of the distance forever. Such lights now get a range: where their illuminance drops below 0.001 lux, at most 4,096 m. Every point and spot light now fades to zero at its range through Karis's window, `(1 − (d/r)⁴)²` ("Real Shading in Unreal Engine 4", 2013). It reaches zero with zero slope, so the edge of a light's reach never shows.

Shadows stay exact: every light that reaches a pixel traces its ray (Chapter 11). That's the cost that grows: where many lights overlap, a pixel traces many rays. A knob, `shadow_ray_budget`, traces rays only for the strongest few lights at each pixel instead, for scenes dense enough to need it. Scaling shadows to many lights properly needs more than rays per light; that waits for a later chapter, when the scene gets a representation that global illumination can trace cheaply.

To test it all, `main` scatters 256 coloured point and spot lights through Sponza. A new debug view, on key L, shows how many lights each pixel's cluster holds.

This chapter builds on [Chapter 15](15-distance-anti-aliasing.md).

## 16.1 Lights and clusters in the data: `shader_types.h`, `shared.slangh`

### Why
The lights' passes need push data and tables of their own, and the fragment shader needs to find its cluster.

### How
- **`FrameData`:**
  - **`light_count` becomes `directional_light_count`.** Lights are now ordered with the directional ones first. They reach everywhere, so they're not clustered, and the shader loops over just those; nothing reads the total any more.
  - **`camera_forward`:** the direction the camera looks. A pixel's slice comes from its view depth: its distance along this axis.
  - **`shadow_ray_budget`:** 0 for exact shadows, or how many lights trace rays at each pixel (16.6).
  - **`cluster_tiles`:** the tiles across and down.
  - **`light_clusters`** and **`visible_lights`:** the clusters' bits, and the lights in view, as a count, then their indices.

  352 bytes, no padding.
- **`LightTables`** (32 bytes) and **`LightPushData`** (24 bytes), for the lights' passes, like the cull's (Chapter 12).
- **`Light::range`** is now always set for point and spot lights.
- **`View::light_count`**, the heat map, on key L.
- **`shared.slangh`** has the cluster grid's numbers, which `light_clusters.h` repeats, and:
  - **`cluster_slice`:** the slice for a view depth. Slice `s` starts at `0.05 × 2^(s/2)` metres, the camera's near plane times a factor of √2 per slice. The last of the 24 starts at 145 m and holds everything beyond.
  - **`FrameData::lights`** now holds the test lights too, after the file's, with the directional ones first.
  - **`cluster_bits`:** where a pixel's cluster's bits start, from its tile and slice.

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
    vk::DeviceAddress instances;        // the cull's visible draws: what each instance draws
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
    vk::DeviceAddress light_clusters;   // per cluster, one bit per visible light
    vk::DeviceAddress visible_lights;   // how many local lights are in view, then their indices
};

static_assert(sizeof(FrameData) == 352);
static_assert(offsetof(FrameData, vertices) == 128);
static_assert(offsetof(FrameData, instances) == 152);
static_assert(offsetof(FrameData, scene_tlas) == 184);
static_assert(offsetof(FrameData, camera_cell) == 192);
static_assert(offsetof(FrameData, sun_direction) == 208);
static_assert(offsetof(FrameData, sun_illuminance) == 224);
static_assert(offsetof(FrameData, sky_cube) == 240);
static_assert(offsetof(FrameData, sun_angular_radius) == 260);
static_assert(offsetof(FrameData, ambient_occlusion) == 264);
static_assert(offsetof(FrameData, camera_offset) == 272);
static_assert(offsetof(FrameData, tlas_offset) == 284);
static_assert(offsetof(FrameData, sky_view) == 296);
static_assert(offsetof(FrameData, camera_forward) == 312);
static_assert(offsetof(FrameData, light_clusters) == 336);

// Push data

// Written with vkCmdPushDataEXT before each pipeline's draws: where this frame's data is. Which DrawData an instance draws, the vertex shader looks up among the cull's instances.
struct PushData {
    vk::DeviceAddress frame;
};

static_assert(sizeof(PushData) == 8);

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

// The same for every draw in a frame. Natural layout, like the C++ struct: the pointers land on 8-byte boundaries, after the matrices and at the end.
struct FrameData {
    float4x4 view_projection;          // camera-relative space -> clip space
    float4x4 inverse_view_projection;  // clip space -> camera-relative space
    Vertex *vertices;                  // the scene's vertices
    uint *indices;                     // the scene's indices
    DrawData *draws;                   // one DrawData per draw
    uint *instances;                   // the cull's visible draws: what each instance draws
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
    uint *light_clusters;              // per cluster, one bit per visible light
    uint *visible_lights;              // how many local lights are in view, then their indices
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

## 16.2 A shared scan: `scan.slangh`, `cull.slang`

### Why
Listing the lights in view, in order, is the same problem as listing the draws in view: prefix sums over 0s and 1s. Chapter 12's scan moves into a header both use.

### How
- **`scan.slangh`:** `exclusive_scan` and its shared memory, unchanged.
- **`cull.slang`** includes it, and loses its own copy.

### Code
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
```

`game-engine/shaders/cull.slang`:
```slang
// GPU culling, in six steps (culling.h, record_culling), which together turn "which draws are in view" into indirect draw commands:
//   cullMain            per draw: 1 if its box is in view, else 0
//   scanDrawsMain       prefix sums of those: each visible draw's slot among the instances, and each group's first instance
//   markGroupsMain      per group: 1 if any of its draws is in view
//   scanGroupsMain      prefix sums of those: each group's command slot
//   writeInstancesMain  per visible draw: its index, into its slot
//   writeCommandsMain   per visible group: its instanced command; per list: how many commands it has
// Draws are visited in the cull's order (by list, then primitive), so a group's draws, and a list's groups, are runs. Every write goes to a place the prefix sums fix: the result doesn't depend on which thread runs first.

#include "shared.slangh"
#include "scan.slangh"

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
    uint *visible;          // per position in the order
    uint *draw_slots;       // prefix sums of visible, then their total
    uint *group_flags;      // per group
    uint *group_slots;      // prefix sums of group_flags, then their total
    uint *instances;        // the visible draws' indices
    DrawCommand *commands;  // per group
    uint *counts;           // per list
    uint draw_count;
    uint group_count;
};

struct CullPushData {
    FrameData *frame;
    CullTables *tables;
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

[shader("compute")]
[numthreads(64, 1, 1)]
void cullMain(uint3 id : SV_DispatchThreadID) {
    CullTables *tables = push.tables;
    const uint position = id.x;

    if (position >= tables.draw_count) {
        return;
    }

    // The draw's box, from its cell to the camera: both corners in the same cell.
    FrameData *frame = push.frame;
    const DrawData draw = frame.draws[tables.order[position]];
    const float3 lo = camera_relative(frame, draw.cell, draw.bounds_min);
    const float3 hi = camera_relative(frame, draw.cell, draw.bounds_max);

    tables.visible[position] = in_view(frame.view_projection, lo, hi) ? 1 : 0;
}

// 2 and 4. Prefix sums (scan.slangh)

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

// 3. Groups with something to draw

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

// 5 and 6. Writing the instances and the commands

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
```

## 16.3 The cluster passes: `lights.slang`

### Why
Four compute steps turn the lights and the view into each cluster's bits.

### How
1. **`markLightsMain`:** per point or spot light, 1 if its range's sphere reaches into the view: behind none of the frustum's five planes by more than the range. There's no far plane to test. Like the cull's test, it keeps a few spheres near the frustum's corners that miss it.
2. **`scanLightsMain`:** prefix sums of those: each visible light's slot.
3. **`listLightsMain`:** each visible light's index goes into its slot, up to 4,096, and its sphere, in view space, into a table for the next step. The first thread writes how many there are.
4. **`clusterMain`:** one thread per cluster and 32 visible lights, writing one word of bits:
   - **The cluster's box,** in view space: its tile's edges, at its slice's near and far depths. At depth `d`, a point at screen position `s` (−1 to 1) is `s × slope × d` across and down, where `slope` is the tangent of half the field of view. The box spans the corners' extremes.
   - **The test:** a light touches the cluster if the point of the box nearest its centre is within its range. That's conservative: near the box's corners, a sphere can touch the box but miss the frustum cell inside it. A light in too many clusters costs a little shading; one missing from a cluster would be a visible error.
   - **Words past the visible lights** are cleared, so a pixel never reads last frame's bits.
- **View space** comes from the inverse view-projection: points on the near plane at the screen's centre, its right edge and its bottom edge give the camera's axes, and their distances from the centre, over the near plane's distance, give the slopes.
- **Spot lights** are tested as their range's sphere, the whole of it, not their cone. Simple, and conservative again.
- **The workgroups:** 32 threads, one per word of a cluster; 4 of them across its 128 words, times the tiles, times the slices. At 1080p, that's 4 × 510 × 24.

### Code
`game-engine/shaders/lights.slang`:
```slang
// Light clusters, in four steps (light_clusters.h, record_light_clusters), which together turn "which lights reach which part of the view" into one bit per light per cluster:
//   markLightsMain     per local light: 1 if its range reaches into the view
//   scanLightsMain     prefix sums of those: each visible light's slot
//   listLightsMain     per visible light: its index, into its slot, and its sphere in view space, for the clusters to test
//   clusterMain        per cluster and 32 visible lights: a word of bits, one per light whose sphere touches the cluster
// The visible lights keep the lights' order, and each word is written by one thread: the same view gives the same bits every frame, and shading visits a cluster's lights in that order.

#include "shared.slangh"
#include "scan.slangh"

// Data shared with C++ (src/includes/shader_types.h)

struct LightTables {
    uint *flags;       // per local light: 1 if in view
    uint *slots;       // prefix sums of flags, then their total
    float4 *spheres;   // per visible light: its centre in view space, and its range
    uint first_local;  // the first point or spot light: after the directional ones
    uint local_count;  // how many point and spot lights there are
};

struct LightPushData {
    FrameData *frame;
    LightTables *tables;
    uint2 extent;      // the screen, in pixels
};

[[vk::push_constant]]
ConstantBuffer<LightPushData> push;

// View space

// The camera's axes, from the inverse view-projection: points on the near plane at the screen's centre, its right edge and its bottom edge (Vulkan's y points down), as camera-relative positions. Their distances from the centre, over the near plane's distance, are tan(half the field of view), across and down: `slope`.
struct ViewAxes {
    float3 right;
    float3 down;
    float3 forward;
    float2 slope;  // per metre of depth, how far the screen's edge is, across and down
};

float3 near_point(FrameData *frame, float2 ndc) {
    const float4 point = mul(frame.inverse_view_projection, float4(ndc, 1.0, 1.0));
    return point.xyz / point.w;
}

ViewAxes view_axes(FrameData *frame) {
    const float3 centre = near_point(frame, float2(0.0, 0.0));
    const float3 right = near_point(frame, float2(1.0, 0.0)) - centre;
    const float3 down = near_point(frame, float2(0.0, 1.0)) - centre;

    ViewAxes axes;
    axes.right = normalize(right);
    axes.down = normalize(down);
    axes.forward = frame.camera_forward;
    axes.slope = float2(length(right), length(down)) / length(centre);
    return axes;
}

// 1. Lights in view

// Whether a sphere reaches into the view: in front of every frustum plane, or behind one by less than its radius. The planes come from the view-projection's rows, as in cull.slang's in_view.
bool sphere_in_view(float4x4 view_projection, float3 centre, float radius) {
    const float4 row0 = view_projection[0];
    const float4 row1 = view_projection[1];
    const float4 row2 = view_projection[2];
    const float4 row3 = view_projection[3];

    const float4 planes[5] = {row3 + row0, row3 - row0, row3 + row1, row3 - row1, row3 - row2};

    for (int i = 0; i < 5; ++i) {
        const float4 plane = planes[i];

        if (dot(plane.xyz, centre) + plane.w < -radius * length(plane.xyz)) {
            return false;
        }
    }

    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void markLightsMain(uint3 id : SV_DispatchThreadID) {
    LightTables *tables = push.tables;

    if (id.x >= tables.local_count) {
        return;
    }

    FrameData *frame = push.frame;
    const Light light = frame.lights[tables.first_local + id.x];
    const float3 centre = camera_relative(frame, light.cell, light.offset);

    // A light with no range gives no light at all.
    tables.flags[id.x] = light.range > 0.0 && sphere_in_view(frame.view_projection, centre, light.range) ? 1 : 0;
}

// 2. Prefix sums (scan.slangh)

[shader("compute")]
[numthreads(scan_size, 1, 1)]
void scanLightsMain(uint3 id : SV_GroupThreadID) {
    LightTables *tables = push.tables;
    exclusive_scan(tables.flags, tables.slots, tables.local_count, id.x);
}

// 3. The visible lights

// Each visible light goes to its slot, up to max_visible_lights: its index among all the lights, for shading, and its sphere in view space, for the clusters. The first thread writes how many there are.
[shader("compute")]
[numthreads(64, 1, 1)]
void listLightsMain(uint3 id : SV_DispatchThreadID) {
    LightTables *tables = push.tables;
    FrameData *frame = push.frame;

    if (id.x == 0) {
        frame.visible_lights[0] = min(tables.slots[tables.local_count], max_visible_lights);
    }

    if (id.x >= tables.local_count || tables.flags[id.x] == 0) {
        return;
    }

    const uint slot = tables.slots[id.x];

    if (slot >= max_visible_lights) {
        return;
    }

    const uint index = tables.first_local + id.x;
    const Light light = frame.lights[index];
    const float3 centre = camera_relative(frame, light.cell, light.offset);
    const ViewAxes axes = view_axes(frame);

    frame.visible_lights[1 + slot] = index;
    tables.spheres[slot] = float4(dot(centre, axes.right), dot(centre, axes.down), dot(centre, axes.forward), light.range);
}

// 4. The clusters

// One thread per cluster and word: 32 visible lights, tested against the cluster's box in view space, one bit each. The box bounds the part of the frustum the cluster covers: its tile's edges, from its slice's near depth to its far one. The last slice reaches on to infinity; a large number stands in for it. Words past the visible lights are cleared, so shading never reads last frame's bits.
//
// A light touches the cluster if its sphere reaches the box: if the point of the box nearest the light's centre is within its range. That's conservative: it keeps a few lights whose sphere only reaches the box's corners, outside the frustum itself.
[shader("compute")]
[numthreads(32, 1, 1)]
void clusterMain(uint3 id : SV_DispatchThreadID) {
    FrameData *frame = push.frame;
    const uint word = id.x;
    const uint2 tile = uint2(id.y % frame.cluster_tiles.x, id.y / frame.cluster_tiles.x);
    const uint slice = id.z;
    const uint cluster = (slice * frame.cluster_tiles.y + tile.y) * frame.cluster_tiles.x + tile.x;

    const uint visible_count = frame.visible_lights[0];
    uint bits = 0;

    if (word * 32 < visible_count) {
        // The tile's edges, as -1..1 across and down the screen.
        const float2 lo = float2(tile * cluster_tile_size) / float2(push.extent) * 2.0 - 1.0;
        const float2 hi = float2(min((tile + 1) * cluster_tile_size, push.extent)) / float2(push.extent) * 2.0 - 1.0;

        // The slice's depths.
        const float near = cluster_near * exp2(float(slice) * 0.5);
        const float far = slice == cluster_slices - 1 ? 1e9 : cluster_near * exp2(float(slice + 1) * 0.5);

        // In view space, a point at screen position s and depth d is at s x slope x d across and down: the box's extremes are at its corners, at the near depth or the far one.
        const ViewAxes axes = view_axes(frame);
        const float2 a = lo * axes.slope;
        const float2 b = hi * axes.slope;
        const float3 box_lo = float3(min(min(a * near, a * far), min(b * near, b * far)), near);
        const float3 box_hi = float3(max(max(a * near, a * far), max(b * near, b * far)), far);

        const uint last = min(32, visible_count - word * 32);

        for (uint i = 0; i < last; ++i) {
            const float4 sphere = push.tables.spheres[word * 32 + i];
            const float3 nearest = clamp(sphere.xyz, box_lo, box_hi);
            const float3 gap = sphere.xyz - nearest;

            if (dot(gap, gap) <= sphere.w * sphere.w) {
                bits |= 1u << i;
            }
        }
    }

    frame.light_clusters[cluster * cluster_words + word] = bits;
}
```

## 16.4 The clusters on the CPU: `light_clusters.h`, `light_clusters.cpp`

### Why
The steps' pipelines and buffers, created once, and the clusters, sized for the screen.

### How
- **`create_light_clusters`:** the four pipelines, and buffers for the flags, the slots, the spheres, the visible list, and the table of their addresses.
- **`resize_light_clusters`:** the clusters' buffer, for the screen's tiles. 1920 × 1080 is 30 × 17 tiles, times 24 slices, times 128 words: 6.3 MB. `main` calls it again after recreating the swapchain.
- **`record_light_clusters`:** the four steps, with a barrier between each, like the cull's.
  - **First, a wait** for the previous frame's fragment shaders, which read the clusters and the visible list this rewrites.
  - **Last, a barrier** making them visible to this frame's fragment shaders.

### Code
`game-engine/src/includes/light_clusters.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/shader_types.h"
#include "includes/vulkan_setup.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>

// Light clusters

// The view is cut into clusters: tiles of cluster_tile_size pixels across the screen, times cluster_slices slices in depth, two per doubling of the distance from the camera's near plane. Each frame, compute steps (lights.slang) find the point and spot lights in view, list them in the lights' order, and give every cluster one bit per listed light whose range reaches it. Shading reads its cluster's bits and lights only those. The same numbers are in shared.slangh.
constexpr std::uint32_t cluster_tile_size = 64;
constexpr std::uint32_t cluster_slices = 24;
constexpr std::uint32_t max_visible_lights = 4096;
constexpr std::uint32_t cluster_words = max_visible_lights / 32;

struct LightClusters {
    vk::raii::Pipeline mark = nullptr;
    vk::raii::Pipeline scan = nullptr;
    vk::raii::Pipeline list = nullptr;
    vk::raii::Pipeline cluster = nullptr;

    Buffer flags;     // per local light: 1 if in view
    Buffer slots;     // prefix sums of flags, then their total
    Buffer spheres;   // per visible light: view-space centre and range
    Buffer visible;   // how many lights are listed, then their indices
    Buffer tables;    // one LightTables: where the buffers above are

    // Sized for the screen: resize_light_clusters makes them again.
    Buffer clusters;  // per cluster, cluster_words words of bits
    vk::Extent2D extent;
    glm::uvec2 tiles{0};  // tiles across and down

    std::uint32_t first_local = 0;
    std::uint32_t local_count = 0;
};

// For a scene whose lights never change, `light_count` of them, the first `directional_count` directional; the rest point and spot lights.
LightClusters create_light_clusters(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::uint32_t light_count,
    std::uint32_t directional_count,
    vk::Extent2D extent
);

// Makes the clusters for a screen of `extent` pixels. The GPU must be idle.
void resize_light_clusters(LightClusters &clusters, const vk::raii::Device &device, const GpuChoice &gpu, vk::Extent2D extent);

// Records the four steps for the frame whose FrameData is at `frame`, which must point at these clusters and visible lights. Afterwards, fragment shaders may read them.
void record_light_clusters(const vk::raii::CommandBuffer &commands, const LightClusters &clusters, vk::DeviceAddress frame);
```

`game-engine/src/light_clusters.cpp`:
```cpp
#include "includes/light_clusters.h"

#include "includes/pipeline.h"

#include <algorithm>
#include <cstddef>

namespace {

    // Threads per workgroup for the steps that take one light each.
    constexpr std::uint32_t workgroup_size = 64;

    std::uint32_t workgroups(std::uint32_t threads) {
        return (threads + workgroup_size - 1) / workgroup_size;
    }

    // A device-local buffer of `count` 32-bit numbers, written and read only by the GPU, at least one long: a buffer can't be empty.
    Buffer gpu_numbers(const vk::raii::Device &device, const GpuChoice &gpu, std::size_t count) {
        return create_buffer(device, gpu, std::max<std::size_t>(count, 1) * sizeof(std::uint32_t),
            vk::BufferUsageFlagBits::eShaderDeviceAddress, vk::MemoryPropertyFlagBits::eDeviceLocal);
    }

    // Makes the `dst` work wait for the `src` work, and the `src` writes visible to the `dst` accesses.
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

    void compute_to_compute(const vk::raii::CommandBuffer &commands) {
        memory_barrier(commands,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);
    }

}  // namespace

// Creating

LightClusters create_light_clusters(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    std::uint32_t light_count,
    std::uint32_t directional_count,
    vk::Extent2D extent
) {
    LightClusters clusters;
    clusters.mark = create_compute_pipeline(device, "lights", "markLightsMain");
    clusters.scan = create_compute_pipeline(device, "lights", "scanLightsMain");
    clusters.list = create_compute_pipeline(device, "lights", "listLightsMain");
    clusters.cluster = create_compute_pipeline(device, "lights", "clusterMain");

    clusters.first_local = directional_count;
    clusters.local_count = light_count - directional_count;

    clusters.flags = gpu_numbers(device, gpu, clusters.local_count);
    clusters.slots = gpu_numbers(device, gpu, clusters.local_count + 1);
    clusters.spheres = create_buffer(device, gpu, std::size_t{max_visible_lights} * sizeof(glm::vec4),
        vk::BufferUsageFlagBits::eShaderDeviceAddress, vk::MemoryPropertyFlagBits::eDeviceLocal);
    clusters.visible = gpu_numbers(device, gpu, 1 + max_visible_lights);

    const LightTables tables{
        .flags = clusters.flags.address,
        .slots = clusters.slots.address,
        .spheres = clusters.spheres.address,
        .first_local = clusters.first_local,
        .local_count = clusters.local_count,
    };

    clusters.tables = upload_buffer(device, gpu, queue, pool, std::as_bytes(std::span(&tables, 1)),
        vk::BufferUsageFlagBits::eShaderDeviceAddress);

    resize_light_clusters(clusters, device, gpu, extent);
    return clusters;
}

void resize_light_clusters(LightClusters &clusters, const vk::raii::Device &device, const GpuChoice &gpu, vk::Extent2D extent) {
    clusters.extent = extent;
    clusters.tiles = glm::uvec2{
        (extent.width + cluster_tile_size - 1) / cluster_tile_size,
        (extent.height + cluster_tile_size - 1) / cluster_tile_size,
    };

    // 1920 x 1080: 30 x 17 tiles, 24 slices, 128 words of 4 bytes, 6.3 MB.
    const std::size_t cluster_count = std::size_t{clusters.tiles.x} * clusters.tiles.y * cluster_slices;
    clusters.clusters = Buffer{};  // the old one goes first
    clusters.clusters = gpu_numbers(device, gpu, cluster_count * cluster_words);
}

// Recording

void record_light_clusters(const vk::raii::CommandBuffer &commands, const LightClusters &clusters, vk::DeviceAddress frame) {
    // The previous frame's fragment shaders read the clusters and the visible lights: wait for them before rewriting either.
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);

    const LightPushData push{
        .frame = frame,
        .tables = clusters.tables.address,
        .extent = glm::uvec2{clusters.extent.width, clusters.extent.height},
    };

    const auto step = [&](const vk::raii::Pipeline &pipeline, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        commands.pushDataEXT(vk::PushDataInfoEXT{
            .offset = 0,
            .data = {.address = &push, .size = sizeof(push)},
        });
        commands.dispatch(x, y, z);
    };

    // 1. Which lights reach into the view.
    step(clusters.mark, workgroups(clusters.local_count), 1, 1);
    compute_to_compute(commands);

    // 2. Each visible light's slot. One workgroup.
    step(clusters.scan, 1, 1, 1);
    compute_to_compute(commands);

    // 3. The visible lights, in order. At least one workgroup, whose first thread writes the count, even with no lights.
    step(clusters.list, std::max(workgroups(clusters.local_count), 1u), 1, 1);
    compute_to_compute(commands);

    // 4. Every cluster's bits: 32 threads per 32 words, one workgroup per tile and slice.
    step(clusters.cluster, cluster_words / 32, clusters.tiles.x * clusters.tiles.y, cluster_slices);

    // The lighting and transparency passes' fragment shaders read them.
    memory_barrier(commands,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderStorageRead);
}
```

## 16.5 Ranges and test lights: `scene.h`, `scene.cpp`

### Why
Every point and spot light needs a range, and there need to be lights to test with.

### How
- **`light_range`:** a light's own range if the file gives one. Otherwise, the distance where its illuminance falls to `light_threshold`, 0.001 lux: `√(intensity / threshold)`, with the intensity of its brightest color in candela, at most `max_light_range`, 4,096 m. A 20 cd lamp reaches 141 m.
  - **Why 0.001 lux:** at the exposure the meter picks for the darkest scenes, EV −2 (before any exposure compensation), light that faint on a white surface is at most about 3.5 of 255 8-bit steps: that much where a strongly coloured light is all there is, about half a step on a surface lit by anything else, and almost nothing for white light in the dark, where the tone mapper's toe darkens it further. What hides the cut-off is the window, which fades a light to zero smoothly, with zero slope at its range; the threshold keeps the step it fades from small.
  - **In scene-referred units:** the range doesn't change with exposure, so a light looks the same however the camera is set.
- **`world_light`** gives every point and spot light its range.
- **`add_test_lights`:** `count` lights through the scene's box, a tenth of its size in from each side, in its lower half. Each light's numbers come from a hash of its index, so the lights are the same every run:
  - **Colors:** strongly coloured, at least 30% of the brightest channel in the others, their hues spread by the golden ratio, at 2 to 12 candela.
  - **Ranges:** 2 to 5 m; every 64th light has none of its own, and gets one from `light_range`: 45 to 110 m. With 256 lights, those are four, the 64th, 128th, 192nd and 256th, and all four are spots, like every fourth light.
  - **Every fourth** is a spot shining down, 20° inside, 35° outside.

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

// Adds a flat, plain ground to `scene`: a square `size` metres across, centred under `centre` at height `height`, made of `tile`-metre tiles. The tiles are one primitive drawn once per tile, each placed in its own cell, so the ground holds a float's precision wherever the camera is on it.
void add_ground(Scene &scene, const glm::dvec3 &centre, double height, double size, double tile);

// Adds `count` coloured point and spot lights to `scene`, spread through its box, the same ones every run: something to test many lights with. Every fourth is a spot shining down; every 64th has no range of its own.
void add_test_lights(Scene &scene, std::uint32_t count);
```

In `game-engine/src/scene.cpp`, replace `world_light` with:
```cpp
    // How far a point or spot light reaches. A file may give a range; without one, KHR_lights_punctual's light falls off with the square of the distance forever, so the light gets one where its illuminance, at its brightest color, drops below light_threshold: sqrt(intensity / threshold), at most max_light_range.
    float light_range(glm::vec3 intensity, double file_range) {
        if (file_range > 0.0) {
            return static_cast<float>(file_range);
        }

        const float brightest = std::max({intensity.r, intensity.g, intensity.b});
        return std::min(std::sqrt(brightest / light_threshold), max_light_range);
    }

    // A KHR_lights_punctual light, placed by its node's world transform. The light sits at the node's origin and shines down the node's -Z axis.
    std::optional<Light> world_light(const tinygltf::Light &source, const glm::dmat4 &world) {
        const glm::vec3 color = source.color.size() == 3 ? glm::vec3(glm::make_vec3(source.color.data())) : glm::vec3(1.0f);
        const CellPosition position = to_cell(glm::dvec3(world[3]));

        Light light{
            .offset = position.offset,
            .range = 0.0f,
            .direction = glm::normalize(glm::vec3(glm::dmat3(world) * glm::dvec3{0.0, 0.0, -1.0})),
            .spot_scale = 0.0f,
            .intensity = color * static_cast<float>(source.intensity),
            .spot_offset = 0.0f,
            .cell = position.cell,
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

        if (light.type != LightType::directional) {
            light.range = light_range(light.intensity, source.range);
        }

        return light;
    }
```

At the end of `game-engine/src/scene.cpp`, add:
```cpp
void add_test_lights(Scene &scene, std::uint32_t count) {
    // A hash of the light's number and which value is wanted, as 0..1: the same lights every run, without a random number generator's state.
    const auto random = [](std::uint32_t light, std::uint32_t value) {
        std::uint32_t x = light * 0x9E3779B9u + value * 0x85EBCA6Bu;
        x ^= x >> 16;
        x *= 0x7FEB352Du;
        x ^= x >> 15;
        x *= 0x846CA68Bu;
        x ^= x >> 16;
        return static_cast<float>(x >> 8) / static_cast<float>(1u << 24);
    };

    // Inside the box, a tenth of its size in from each side, and in its lower half.
    const glm::vec3 size = scene.bounds_max - scene.bounds_min;
    const glm::vec3 lo = scene.bounds_min + size * 0.1f;
    const glm::vec3 hi = glm::vec3{scene.bounds_max.x - size.x * 0.1f, scene.bounds_min.y + size.y * 0.5f, scene.bounds_max.z - size.z * 0.1f};

    for (std::uint32_t i = 0; i < count; ++i) {
        const glm::dvec3 position = glm::mix(lo, hi, glm::vec3{random(i, 0), random(i, 1), random(i, 2)});
        const CellPosition placed = to_cell(position);

        // A saturated colour, its hue spread evenly by the golden ratio, at 2 to 12 candela.
        const float hue = std::fmod(static_cast<float>(i) * 0.618034f, 1.0f) * 6.0f;
        const glm::vec3 color = glm::clamp(glm::vec3{
            std::abs(hue - 3.0f) - 1.0f,
            2.0f - std::abs(hue - 2.0f),
            2.0f - std::abs(hue - 4.0f),
        }, 0.0f, 1.0f) * 0.7f + 0.3f;
        const glm::vec3 intensity = color * glm::mix(2.0f, 12.0f, random(i, 3));

        Light light{
            .offset = placed.offset,
            .range = light_range(intensity, i % 64 == 63 ? 0.0 : glm::mix(2.0, 5.0, static_cast<double>(random(i, 4)))),
            .direction = glm::vec3{0.0f, -1.0f, 0.0f},
            .spot_scale = 0.0f,
            .intensity = intensity,
            .spot_offset = 0.0f,
            .cell = placed.cell,
            .type = LightType::point,
        };

        // Spots: 20 degrees inside, 35 outside, as world_light works it out.
        if (i % 4 == 3) {
            const float cos_inner = std::cos(glm::radians(20.0f));
            const float cos_outer = std::cos(glm::radians(35.0f));
            light.type = LightType::spot;
            light.spot_scale = 1.0f / (cos_inner - cos_outer);
            light.spot_offset = -cos_outer * light.spot_scale;
        }

        scene.lights.push_back(light);
    }
}
```

## 16.6 Shading with clusters: `mesh.slang`

### Why
The fragment shader lights each pixel with its cluster's lights, and the light count view shows the clusters at work.

### How
- **Karis's window:** `punctual_light` fades point and spot lights by `(1 − (d/r)⁴)²` over `d²`. Karis divides by `d² + 1`, to keep a light finite at its centre; we keep KHR_lights_punctual's inverse square. KHR_lights_punctual recommends the same window without the square. Both are zero at the range; squared, the falloff also flattens to zero there, so the edge of a light's reach doesn't show. Squaring also dims a light within its range a little more: at 80% of the range, to 0.35 of the inverse square rather than 0.59.
- **Directional lights** from the file, the first `directional_light_count`, light every pixel, as before.
- **`shade_local_lights`:**
  - **The cluster:** from the pixel's tile and its view depth, `dot(position, camera_forward)`.
  - **Walking the bits:** for each word, `firstbitlow` finds the lowest set bit, and `remaining &= remaining − 1` clears it: one light at a time, in the lights' order. The bit's position in the visible list gives the light's index.
  - **Exact shadows:** every light that brings the surface any light, `luminance(illuminance) × n·l` above 0, traces its shadow ray.
- **The ray budget:** with `shadow_ray_budget` N, only the N lights that bring the surface the most light, before shadows, trace their rays.
  - **Finding them:** a first pass over the cluster's lights keeps the 16 largest amounts in a sorted array. Each amount goes in through a fixed chain of compare-and-swaps, every index known when the shader is compiled, so the compiler can keep the array in registers. Indexing an array with a variable can send it to slow local memory, as it typically does on NVIDIA: an earlier version that did was slower than tracing every ray.
  - **Then** the Nth largest is the cutoff: lights at or above it trace rays; those below light the pixel unshadowed. Ties at the cutoff all trace. The two passes work the amounts out separately, and the compiler may round them a hair differently, so the cutoff is lowered by a millionth of itself: the Nth light always makes it.
  - **Skipped** when the cluster holds no more lights than the budget: counting its bits is cheaper than the first pass.
  - **The price:** unshadowed light leaks through walls. Where few lights overlap, it changes nothing and costs a little; where many do, it saves time and brightens the image (16.8).
- **`light_count_heat`:** the cluster's bits counted, as colors: black for none, dark blue for 1 or 2, then blue, green, yellow and red at 4, 8, 16 and 32 or more.

### Code
`game-engine/shaders/mesh.slang`:
```slang
// Draws one glTF primitive: its vertices come from the scene's vertex buffer, its place in the world from its DrawData, and its surface from its glTF material, whose textures are read from the descriptor heap. Shaded with glTF's physically based BRDF, lit by the sun and the file's lights, with ray-traced shadows, and by the sky around the scene, already exposed. Three fragment shaders:
//   prepassMain      the depth prepass's vertex normal
//   fragmentMain     opaque and masked surfaces, into the HDR image
//   transparentMain  blended surfaces, into weighted blended transparency's sums

// Data shared with C++ (src/includes/shader_types.h)

#include "shared.slangh"
#include "atmosphere.slangh"

// In a descriptor heap pipeline, the push_constant block is where push data lands.
[[vk::push_constant]]
ConstantBuffer<PushData> push;

// The alpha mode this pipeline was built for (AlphaMode in C++): 0 opaque, 1 mask, 2 blend. A specialization constant: its value is fixed when the pipeline is created, so each pipeline's fragment shader keeps only the code its mode needs.
[vk::constant_id(0)]
const uint alpha_mode = 0;

static const uint alpha_opaque = 0;
static const uint alpha_mask = 1;
static const uint alpha_blend = 2;

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

// SV_VulkanVertexID is Vulkan's own gl_VertexIndex, which includes the draw's vertexOffset: each primitive's indices start at 0, and the draw adds where that primitive's vertices begin in the shared buffer. SV_VulkanInstanceID is gl_InstanceIndex, which likewise counts from the command's firstInstance: the cull points that at the command's run of visible draws in `instances`, so each instance finds its draw there.
[shader("vertex")]
VertexOutput vertexMain(uint vertex_id : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    FrameData *frame = push.frame;
    const Vertex vertex = frame.vertices[vertex_id];
    const uint draw_index = frame.instances[instance];
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

// How much of a light gets from `origin` to `distance` along `direction`: 0 when something solid is in the way, otherwise the share every see-through layer on the way lets through. A ray query walks the TLAS and BLASes:
//   - an opaque triangle ends it at once: any blocking hit will do,
//   - a masked one comes back as a candidate, which blocks where its alpha reaches the cutoff, and lets the light through its cut-out texels,
//   - a blended one comes back as a candidate that lets 1 - alpha of the light through, as the transparency pass's reveal sum does. It never ends the ray: the light goes on, dimmed, to whatever is behind.
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

    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0 : transmittance;
}

// Where a ray toward the light `l` starts: off the surface on the light's side, along `face_normal`, the triangle's own flat normal. An interpolated or normal-mapped normal can disagree about which side the light is on.
//
// The TLAS's space is measured from its origin cell, near the camera, not from the camera itself (acceleration.h): tlas_offset, the camera's position in it, moves the camera-relative position there first, so the offset grows with the coordinates the ray is really traced at.
float3 shadow_ray_origin(FrameData *frame, float3 position, float3 face_normal, float3 l) {
    return offset_ray_origin(position + frame.tlas_offset, dot(face_normal, l) >= 0.0 ? face_normal : -face_normal);
}

// shade(), times how much of the light gets through. A ray is only traced when the light could reach the surface at all.
float3 shade_shadowed(
    Surface surface, FrameData *frame, float3 position, float3 face_normal,
    float3 l, float3 illuminance, float distance
) {
    if (dot(surface.normal, l) <= 0.0 || all(illuminance == 0.0)) {
        return float3(0.0);
    }

    const float reaching = light_visibility(frame, shadow_ray_origin(frame, position, face_normal, l), l, distance);
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

// One light's shadowed contribution at `position`.
float3 shade_light(Surface surface, FrameData *frame, Light light, float3 position, float3 face_normal) {
    float3 l;
    float distance;
    const float3 illuminance = punctual_light(frame, light, position, l, distance);
    return shade_shadowed(surface, frame, position, face_normal, l, illuminance, distance);
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

// The point and spot lights whose range reaches this pixel's cluster, in the lights' order, each with its shadow ray: a light that brings the surface no light at all traces none. With a budget of N rays, only the N lights that bring it the most light, without shadows, trace theirs; the rest light the pixel unshadowed.
float3 shade_local_lights(Surface surface, FrameData *frame, VertexOutput input, float3 face_normal) {
    const float3 position = input.relative_position;
    uint *bits = cluster_bits(frame, input.position.xy, dot(position, frame.camera_forward));
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
                ? lit * light_visibility(frame, shadow_ray_origin(frame, position, face_normal, l), l, distance)
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

// The light count view

// How many lights this pixel's cluster holds, as a heat map: black for none, dark blue for 1 or 2, then blue, green, yellow and red at 4, 8, 16 and 32 or more.
float3 light_count_heat(FrameData *frame, VertexOutput input) {
    uint *bits = cluster_bits(frame, input.position.xy, dot(input.relative_position, frame.camera_forward));
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

// Shading a fragment

// The surface at this fragment: its exposed radiance (or one input, in a debug view), and its alpha. The lighting pass writes it as it is; the transparency pass adds it into its sums. `front_face`: whether this triangle faces the camera.
float4 shade_fragment(VertexOutput input, bool front_face) {
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
        case view_light_count: return float4(light_count_heat(frame, input), 1.0);
        default: break;
    }

    // The triangle's flat normal, from how the position changes across neighbouring pixels: exact up to rounding, since a triangle is flat. Shadow rays start off the surface along it. A triangle seen exactly edge-on has no area on screen, and no such normal: then the shading normal stands in, rather than a division by zero.
    const float3 face_cross = cross(ddy(input.relative_position), ddx(input.relative_position));
    const float3 face_normal = dot(face_cross, face_cross) > 1e-24 ? normalize(face_cross) : normal;

    // The shadow view: how much of the sun's light gets through to each point.
    if (frame.view == view_shadow) {
        const float sun = frame.sun_direction.y > 0.0
            ? light_visibility(frame, shadow_ray_origin(frame, input.relative_position, face_normal, frame.sun_direction),
                frame.sun_direction, infinite_distance)
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

    // Direct light, shadowed: the sun, the file's directional lights, then the point and spot lights that reach this pixel's cluster.
    float3 radiance = shade_shadowed(surface, frame, input.relative_position, face_normal,
        frame.sun_direction, frame.sun_illuminance, infinite_distance);

    for (uint i = 0; i < frame.directional_light_count; ++i) {
        radiance += shade_light(surface, frame, frame.lights[i], input.relative_position, face_normal);
    }

    radiance += shade_local_lights(surface, frame, input, face_normal);

    // Indirect light from the sky, darkened by occlusion. Ambient occlusion only ever reaches this indirect light: the sun and the lights are direct, and only a shadow can block them.
    radiance += shade_environment(surface, frame, shading_roughness, visibility, irradiance_normal);

    // The air between the camera and the surface, with the simulated sky: it dims the surface's light, emission included, and adds its own. The photograph has no air to go with it.
    float3 transmittance = 1.0;
    float3 inscatter = 0.0;

    if (frame.atmosphere != 0) {
        inscatter = aerial_perspective(frame, input.relative_position, transmittance);
    }

    // Exposure scales nits into the tone mapper's range here, before the 16-bit HDR image could overflow. glTF defines emission in nits, but, as its spec notes many engines do, we take it as already exposed: an emissive value of 1 shows as near-white, whatever the exposure.
    return float4((radiance * transmittance + inscatter) * frame.exposure + emissive * transmittance, base_color.a);
}

// Fragment shaders

// The lighting pass, for opaque and masked surfaces. SV_Target: the value written to color attachment 0. SV_IsFrontFace: whether this triangle faces the camera.
[shader("fragment")]
float4 fragmentMain(VertexOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    return shade_fragment(input, front_face);
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
TransparentOutput transparentMain(VertexOutput input, bool front_face : SV_IsFrontFace) {
    const float4 color = shade_fragment(input, front_face);
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

## 16.7 Wiring it in: `main.cpp`

### Why
`main` adds the test lights, orders the lights, creates the clusters, and records them every frame.

### How
- **`test_lights`:** how many lights `add_test_lights` scatters, 256. 0 keeps just the file's.
- **`shadow_ray_budget`:** 0, exact.
- **Directional lights first:** `stable_partition` moves them to the front, keeping each group's order.
- **The clusters** are created once the swapchain exists, and resized with it.
- **`record_frame`** records them after the atmosphere, before the depth prepass. Each frame's `FrameData` gets the cluster grid and the buffers' addresses.
- **Key L** picks the light count view.

### Code
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
    };

    constexpr std::uint32_t screen_slot_count = 11;

    // Every graphics pipeline a frame uses. The prepass and the lighting pass have one per solid alpha mode, in solid_modes' order; see-through surfaces have only the transparency pass's.
    struct ScenePipelines {
        std::vector<vk::raii::Pipeline> prepass;
        std::vector<vk::raii::Pipeline> lighting;
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

    // Draws every list of alpha mode `mode` with `pipeline`: one indirect call per list, after setting the list's cull mode and front face. Push data says where the frame's data is; each instance finds its DrawData through the cull's instances.
    void draw_mode(
        const vk::raii::CommandBuffer &commands,
        const DrawCulling &culling,
        const DrawList &draws,
        AlphaMode mode,
        const vk::raii::Pipeline &pipeline
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

                // Single-sided surfaces are invisible from behind, so the GPU can skip their back faces before running the fragment shader.
                commands.setCullMode(double_sided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack);
                commands.setFrontFace(mirrored ? mirrored_front_face : front_face);
                draw_list(commands, culling, list);
            }
        }
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

    // Records a frame: the cull, the atmosphere's tables and the light clusters, all in compute shaders, then five passes:
    //   1. the depth prepass: every solid surface's depth and vertex normal,
    //   2. ambient occlusion, from those, in compute shaders,
    //   3. the lighting, into the HDR image: each solid alpha mode's lists with that mode's pipeline, against the prepass's depth, then the sky behind them,
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

        // The cull

        // Every pass below draws only what the cull keeps, from its lists.
        record_culling(commands, culling, draws.frame, draws.readback);

        // The atmosphere

        // The sky around the camera and the air in front of it, for this frame's sun, height and view: the background and the lighting read them.
        record_atmosphere(commands, environment, draws.frame, draws.sun_direction, draws.altitude);

        // The light clusters

        // Which point and spot lights reach which part of the view: the lighting and transparency passes light each pixel with its cluster's lights.
        record_light_clusters(commands, light_clusters, draws.frame);

        // Pass 1: the depth prepass

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

        // Reverse-Z: 0 is infinitely far. The depth is stored this time: the AO pass and the lighting pass both read it.
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
            draw_mode(commands, culling, draws, solid_modes[i], pipelines.lighting[i]);
        }

        // The sky goes in once everything solid is drawn: it only covers pixels still at depth 0, infinitely far.
        commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.background);

        const PushData sky_push{.frame = draws.frame};
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
            draw_mode(commands, culling, draws, AlphaMode::blend, pipelines.transparency);
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

        // Scene

        // The glTF file to draw, under lecture-md/game-engine/assets.
        const std::filesystem::path scene_file = std::filesystem::path(ASSET_DIR) / "Sponza/Sponza.gltf";

        // Where in the world the scene is placed, in metres: its origin. Move it far away, to {100000.0, 0.0, 100000.0} say, 141 km out, and the image stays the same: everything is drawn relative to the camera.
        const glm::dvec3 scene_origin{0.0, 0.0, 0.0};

        const std::uint64_t load_start = SDL_GetTicksNS();
        Scene scene = load_gltf(scene_file, scene_origin);
        std::println("Loaded {}: {} vertices, {} triangles, {} primitives, {} draws, {} materials, {} images",
            scene_file.filename().string(), scene.vertices.size(), scene.indices.size() / 3,
            scene.primitives.size(), scene.draws.size(), scene.materials.size(), scene.images.size());

        // Ground under the scene, out to 32 km each way: something to see the air over, as far as the aerial perspective reaches. 1 cm below the scene's lowest point, so it never fights the scene's own floor.
        add_ground(scene, scene_origin, static_cast<double>(scene.bounds_min.y) - 0.01, 65536.0, 1024.0);

        // Lights to test the clusters with, through the scene's box: 0 for just the file's. Try 16, 256 and 1024.
        constexpr std::uint32_t test_lights = 256;
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

        // The cull, its groups and its draw lists, worked out for this scene's draws.
        const DrawCulling culling = create_draw_culling(device, *gpu, queue, command_pool, cull_draws);

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
            // The camera's height above the ground, which is at y = 0. The simulated sky is lit for it too, so climbing far enough, 100 m, rebuilds it.
            const auto altitude = static_cast<float>(std::max(camera.position.y, 1.0));
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

            // What the cull kept the last time this frame's resources were used: two frames ago.
            const CullTotals totals = *frame.totals;

            // The title shows the view, the sky, the time, the exposure, whether ambient occlusion is on and what the cull kept, whenever one changes.
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

            // Where the camera is: a cell and an offset, like everything the GPU places. Once it's far from the TLAS's origin cell, the TLAS is rebuilt around the camera's cell. That waits for the GPU to stop using the old one: a brief pause, once per kilometre or so.
            const CellPosition camera_at = to_cell(camera.position);

            if (glm::any(glm::greaterThan(glm::abs(camera_at.cell - acceleration.origin_cell), glm::ivec3(tlas_reach_cells)))) {
                device.waitIdle();
                build_tlas(device, *gpu, queue, command_pool, acceleration, scene, camera_at.cell);
                std::println("TLAS rebuilt around cell ({}, {}, {})", camera_at.cell.x, camera_at.cell.y, camera_at.cell.z);
            }

            // The view-projection matrix works in camera-relative space: the view only turns the world, the camera being at its origin.
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
                .light_clusters = light_clusters.clusters.address,
                .visible_lights = light_clusters.visible.address,
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

## 16.8 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **By day,** Sponza looks almost as it did in Chapter 15: the sun outshines the test lights.
- **At night** (press `[` until about 22:00), the courtyard is lit by the 256 coloured lights, each casting its own shadows.
- **The light count view** (key L): blocks of 64 × 64 pixels, coloured by how many lights their cluster holds. They change from slice to slice with depth: near and far surfaces in the same tile can hold different lights. In the test scene, every cluster holds 16 or more. A cluster is deep, so far fewer of its lights reach any one pixel in it: the rest bring the surface no light and trace no ray.
- **Change `test_lights`** to 16 or 1024, and `shadow_ray_budget` to 8, to see how the costs grow.
- **No `[validation …]` lines.**

**What it costs.** Release, 1920 × 1080, RTX 5070 Laptop, at night from the courtyard's end:

| Test lights | Light clusters | Lighting, exact shadows | Frame | Lighting, budget of 8 | Frame |
|---|---|---|---|---|---|
| 16 | 0.09 ms | 1.21 ms | 2.01 ms | — | — |
| 256 | 0.38 ms | 7.69 ms | 8.90 ms | 8.31 ms | 9.61 ms |
| 1,024 | 0.85 ms | 37.1 ms | 39.0 ms | 21.1 ms | 23.1 ms |

- **The clusters** cost a fraction of a millisecond, growing with the lights in view.
- **The lighting** grows with how many lights overlap each pixel, and only a little with how many are in view: each pixel walks one word of bits per 32 visible lights, up to 128. Most of it is the shadow rays.
- **The lighting without shadow rays** took a quarter to a third as long in a test build: the rays are most of it.
- **The ray budget** pays off only where many lights overlap. At 256 lights it changes 1% of the pixels slightly and costs a little more; at 1,024 it saves over two fifths of the lighting, and the light it leaks through walls visibly brightens about a sixth of the pixels.

**The scaling limit:** with exact shadows, a pixel traces one ray per light that reaches it. The number of lights is bounded only by memory and the 4,096 in view; the number overlapping a pixel is bounded by rays per frame. With 1,024 lights in the courtyard, where every cluster holds 16 or more, the lighting takes 37 ms. Shadows for that many lights need another approach, which comes with global illumination, when the scene gets a representation cheap enough to trace many rays against.

Next, in Chapter 17, occlusion culling: a hierarchical depth buffer from the previous frame lets the cull skip draws hidden behind others, not just those outside the view.

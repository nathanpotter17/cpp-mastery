# Chapter 13: Range

By the end of this chapter, the view has no far limit, and the scene can be anywhere in the world. Until now, two things limited how far the engine could reach:
- **The far plane, at 500 m.** Nothing beyond it was drawn. A city skyline or a mountain range would simply stop.
- **Float positions.** Shaders work in 32-bit floats, which hold 24 bits of precision. One metre from the world's origin, the step between neighbouring floats is about 0.12 µm; 10 km out, about 1 mm; 1,000 km out, about 6 cm. Every position the GPU handled was measured from the world's origin, so a scene far from it lost its detail: vertices snapped to a coarse grid, and shadow rays started off their surfaces by the wrong amount.

To see the second problem, move Chapter 12's Sponza 1,448 km from the origin: its surfaces break up into blocks, and its ray-traced shadows dissolve into speckles. After this chapter, the same scene at the same distance, seen from the same starting view, is identical, pixel for pixel, to the scene at the origin.

The two fixes:
1. **Reverse-Z with an infinite far plane.** The projection no longer has a far plane at all. Reverse-Z already put float depth's precision where perspective needs it, and it stays just as precise when the far distance goes to infinity.
2. **Camera-relative rendering, with world cells.** On the CPU, positions are doubles. The GPU gets each one as an integer **cell** of the world, 64 m on a side, plus a float **offset** within it. Shaders subtract the camera's cell as an integer, exactly, and only then turn the difference into metres. Everything the GPU computes is relative to the camera, so the camera's surroundings always keep a float's full precision, wherever in the world they are. Open-world engines use this kind of scheme to keep precision across worlds many kilometres wide.

The ray-traced shadows need one more piece: the TLAS holds floats too, so its space starts at a cell near the camera, and it's rebuilt around the camera when the camera strays more than about a kilometre away.

This chapter builds on [Chapter 12](12-gpu-driven-culling.md).

## 13.1 Cells: `cells.h`

### Why
Every position the GPU sees, of a draw, a light or the camera, needs to be split the same way.

### How
- **`cell_size`** is 64 m. Any size works as long as an offset within a cell stays small: below 64 m, floats are under 4 µm apart.
- **`to_cell`** splits a double position: its cell is `floor(position / 64)` on each axis, and its offset what's left, from 0 to 64 m.
- **`cell_corner`** is where one cell's corner is, measured from another's: the cells' difference, an exact integer, times 64. A float holds any multiple of 64 m exactly, up to 2^24 cells, about a billion metres.
- **Why integers:** two doubles 1,000 km out, subtracted, give the exact difference. Two floats 1,000 km out don't: each has already lost its centimetres. The cells keep the large part of a position in integers, which subtract exactly on the GPU, and the floats only ever hold the small part.

### Code
`game-engine/src/includes/cells.h`:
```cpp
#pragma once

#include <glm/glm.hpp>

#include <cmath>

// --- Positions in cells ------------------------------------------------------

// A float has 24 bits of precision: 1 m from the origin, neighbouring floats
// are about 0.12 micrometres apart, 10 km from it about 1 mm. Shaders work
// in floats, so world positions kilometres from the origin would lose their
// detail.
//
// So positions are kept in doubles on the CPU, and the GPU gets each as an
// integer cell of the world, cell_size metres on a side, plus a float offset
// within it, under cell_size. Shaders subtract cells as integers, exactly,
// and only then turn the difference into metres: the camera's surroundings
// come out with a float's full precision wherever in the world they are.
// shared.slangh has the same size.
constexpr double cell_size = 64.0;

struct CellPosition {
    glm::ivec3 cell{0};
    glm::vec3 offset{0.0f};  // metres from the cell's corner, each 0 to cell_size
};

// The cell `position` is in, and where in it.
inline CellPosition to_cell(const glm::dvec3& position) {
    const glm::dvec3 cell = glm::floor(position / cell_size);
    return {glm::ivec3(cell), glm::vec3(position - cell * cell_size)};
}

// Where cell `cell`'s corner is, in metres from `origin`'s: exact, since the
// cells' difference is an integer, and a multiple of cell_size is a float
// for any difference under 2^24 cells.
inline glm::vec3 cell_corner(const glm::ivec3& cell, const glm::ivec3& origin) {
    return glm::vec3(cell - origin) * static_cast<float>(cell_size);
}
```

## 13.2 Cells in the data: `shader_types.h`, `shared.slangh`

### Why
The GPU's structs carry positions: each draw's placement and box, each light's position, the camera's.

### How
- **`DrawData`** gains `cell`. Its model matrix now moves the primitive into that cell, measured from the cell's corner, and its box is measured from there too. 176 bytes, no padding.
- **`Light`** gains `cell`, and its `position` becomes `offset`, within the cell. 64 bytes, no padding.
- **`FrameData`:**
  - `camera_position` is gone. Shaders work in **camera-relative space**: the world's axes, with the camera at the origin. The camera's position there is always 0.
  - `camera_cell` and `camera_offset` say where the camera is, in the same form as everything else.
  - `tlas_offset` is the camera's position in the TLAS's space (13.5).
  - `view_projection` now takes camera-relative positions to clip space.

  296 bytes, no padding.
- **`camera_relative`,** in `shared.slangh`, moves a cell and offset into camera-relative space: `(cell − camera_cell) × 64 + (offset − camera_offset)`. The cells' difference is an integer, exact. Near the camera, both differences are small, so the result keeps a float's full precision.

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

// One per draw, in a GPU buffer the shaders index. A draw is placed in a
// world cell (cells.h): its model matrix moves the primitive into the cell,
// measured from the cell's corner. A shadow ray that hits a draw's triangle
// finds the triangle's vertices through first_index and vertex_offset, as
// drawIndexed does. The cull tests the draw's box.
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

// One light from the file, placed in a world cell (cells.h); its direction is
// along the world's axes.
struct Light {
    glm::vec3 offset;     // point and spot lights: where in `cell` the light is (cells.h)
    float range;          // distance where the light fades to nothing; 0 for no limit
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
// Shaders work in camera-relative space: the world's axes, with the camera
// at the origin. A position given by a cell and an offset (cells.h) is moved
// into it by subtracting the camera's cell and offset.
struct FrameData {
    glm::mat4 view_projection;          // camera-relative space -> clip space
    glm::mat4 inverse_view_projection;  // clip space -> camera-relative space
    vk::DeviceAddress vertices;         // the scene's vertices
    vk::DeviceAddress indices;          // the scene's indices, for shadow rays' alpha tests
    vk::DeviceAddress draws;            // one DrawData per draw
    vk::DeviceAddress instances;        // the cull's visible draws: what each instance draws
    vk::DeviceAddress materials;        // the scene's materials
    vk::DeviceAddress lights;           // the file's lights
    vk::DeviceAddress environment;      // the EnvironmentInfo
    vk::DeviceAddress scene_tlas;       // the top-level acceleration structure, for ray queries
    glm::ivec3 camera_cell;             // the world cell the camera is in
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
    std::uint32_t ambient_occlusion;    // resource heap slot: the GTAO image, full resolution
    std::uint32_t ao_enabled;           // 0: ignore it, to compare
    glm::vec3 camera_offset;            // where in its cell the camera is
    glm::vec3 tlas_offset;              // the camera, measured from the TLAS's origin (acceleration.h)
};

static_assert(sizeof(FrameData) == 296);
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

// The ambient occlusion compute shaders' push data (ao.slang), the same for
// all four steps. The half-resolution images and `source` and `target` are
// storage images; each step reads and writes the ones it needs.
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
    float range;       // 0: no limit
    float3 direction;  // the way the light shines
    float spot_scale;
    float3 intensity;  // lux (directional) or candela (point, spot), per channel
    float spot_offset;
    int3 cell;         // the world cell the light is in
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
    float4x4 view_projection;          // camera-relative space -> clip space
    float4x4 inverse_view_projection;  // clip space -> camera-relative space
    Vertex* vertices;                  // the scene's vertices
    uint* indices;                     // the scene's indices
    DrawData* draws;                   // one DrawData per draw
    uint* instances;                   // the cull's visible draws: what each instance draws
    Material* materials;               // the scene's materials
    Light* lights;                     // the file's lights
    EnvironmentInfo* environment;      // the sky's diffuse light and the sun
    uint64_t scene_tlas;               // the top-level acceleration structure's address
    int3 camera_cell;                  // the world cell the camera is in
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
    float3 camera_offset;              // where in its cell the camera is
    float3 tlas_offset;                // the camera, from the TLAS's origin
};

// --- Camera-relative positions ---------------------------------------------------

// The side of a world cell, in metres: cells.h's cell_size.
static const float cell_size = 64.0;

// A position given as a world cell and an offset in it (cells.h), relative
// to the camera. The cells are subtracted as integers, exactly; only their
// difference, and the offsets' difference, become floats. Near the camera,
// both are small, and keep a float's full precision anywhere in the world.
float3 camera_relative(FrameData* frame, int3 cell, float3 offset) {
    return float3(cell - frame.camera_cell) * cell_size + (offset - frame.camera_offset);
}

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

## 13.3 Placing the scene in doubles: `scene.h`, `scene.cpp`

### Why
glTF's node transforms are decimal numbers, which tinygltf reads into doubles. Until now we turned them into floats right away; now they stay doubles until each draw and light is placed in its cell.

### How
- **`load_gltf`** takes an `origin`: where in the world the scene's own origin goes. It becomes the root nodes' parent transform.
- **`local_transform`** builds each node's transform in doubles, `glm::dmat4`.
- **`visit_node`:**
  - **The world transform** is in doubles, parent times child.
  - **The node's cell** is the one its origin is in. Its draws' model matrix is the world transform with that cell's corner taken off the translation, in doubles, and only then turned into floats.
  - **The draw's box** is measured from the same corner. The scene's box stays in world floats: nothing uses it yet, and when something does, it will only need it roughly.
- **`world_light`** places a light by its cell and offset the same way.
- **`MeshDraw`** gains `cell`. Its `model` and box are now within the cell.

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
// A mesh used by several nodes is drawn once per node. The draw is placed in
// a world cell (cells.h): `model` moves the primitive into it, measured from
// the cell's corner, and the box is measured from there too.
struct MeshDraw {
    glm::mat4 model{1.0f};
    glm::ivec3 cell{0};
    std::uint32_t primitive = 0;

    // A transform that mirrors the primitive (a negative scale) reverses the
    // order its triangles' corners appear in, which decides which side is
    // the front.
    bool mirrored = false;

    // The draw's box, in its cell: the primitive's box, moved by `model`, and
    // boxed again. It holds every triangle of the draw, if a little loosely
    // when the transform rotates.
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

    // World-space box around everything drawn, roughly, in floats. Nothing
    // uses it yet: it's there for a camera that frames the scene.
    glm::vec3 bounds_min{std::numeric_limits<float>::max()};
    glm::vec3 bounds_max{std::numeric_limits<float>::lowest()};
};

// Loads the default scene of a .gltf or .glb file, with its materials,
// samplers, lights and images, still encoded.
// The scene is placed with its origin at `origin`, in metres from the
// world's.
Scene load_gltf(const std::filesystem::path& path, const glm::dvec3& origin = glm::dvec3{0.0});
```

In `game-engine/src/scene.cpp`, add `#include "includes/cells.h"` after `#include "includes/scene.h"`.

In `game-engine/src/scene.cpp`, replace the line `// its own space, which visit_node() turns into world-space scene bounds.` with:
```cpp
// its own space, which visit_node() turns into each draw's box in its cell,
// and the scene's box.
```

Then replace `world_light` with:
```cpp
// A KHR_lights_punctual light, placed by its node's world transform. The
// light sits at the node's origin and shines down the node's -Z axis.
std::optional<Light> world_light(const tinygltf::Light& source, const glm::dmat4& world) {
    const glm::vec3 color = source.color.size() == 3 ? glm::vec3(glm::make_vec3(source.color.data())) : glm::vec3(1.0f);
    const CellPosition position = to_cell(glm::dvec3(world[3]));

    Light light{
        .offset = position.offset,
        .range = static_cast<float>(source.range),
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
        // Full brightness inside the inner cone, nothing outside the outer
        // one. Precomputing a scale and offset turns the shader's falloff
        // into one multiply-add: cos(angle) * scale + offset is 1 at the
        // inner edge and 0 at the outer edge.
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

Then replace `local_transform` with:
```cpp
// A node's transform relative to its parent: either a full matrix, or
// translation * rotation * scale. glTF stores doubles, and so do we, until
// each draw is placed in its cell: a scene far from the origin keeps its
// precision.
glm::dmat4 local_transform(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        return glm::make_mat4(node.matrix.data());  // column-major, like glm
    }

    glm::dmat4 transform{1.0};

    if (node.translation.size() == 3) {
        transform = glm::translate(transform, glm::make_vec3(node.translation.data()));
    }

    if (node.rotation.size() == 4) {
        // glTF stores (x, y, z, w); glm's constructor takes w first.
        const auto& r = node.rotation;
        transform *= glm::mat4_cast(glm::dquat(r[3], r[0], r[1], r[2]));
    }

    if (node.scale.size() == 3) {
        transform = glm::scale(transform, glm::make_vec3(node.scale.data()));
    }

    return transform;
}
```

Then replace `visit_node` with:
```cpp
// Walks the node tree. Each node's world transform is its parent's times its
// own; every primitive of a node's mesh becomes one draw, and a node's light
// is placed by the same transform.
void visit_node(
    const tinygltf::Model& model,
    int node_index,
    const glm::dmat4& parent,
    const std::vector<std::vector<LoadedPrimitive>>& mesh_primitives,
    Scene& scene
) {
    const tinygltf::Node& node = model.nodes.at(node_index);
    const glm::dmat4 world = parent * local_transform(node);

    if (node.mesh >= 0) {
        // A negative determinant means the transform mirrors space.
        const bool mirrored = glm::determinant(glm::dmat3(world)) < 0.0;

        // The cell the node's origin is in. The draw's model matrix moves the
        // primitive into that cell, from its corner: the same transform, with
        // the cell's corner taken off its translation, in doubles first.
        const CellPosition placed = to_cell(glm::dvec3(world[3]));
        const glm::dvec3 corner_position = glm::dvec3(placed.cell) * cell_size;
        glm::dmat4 in_cell = world;
        in_cell[3] = glm::dvec4(glm::dvec3(world[3]) - corner_position, 1.0);

        for (const LoadedPrimitive& primitive : mesh_primitives.at(node.mesh)) {
            MeshDraw draw{
                .model = glm::mat4(in_cell),
                .cell = placed.cell,
                .primitive = primitive.index,
                .mirrored = mirrored,
                .bounds_min = glm::vec3{std::numeric_limits<float>::max()},
                .bounds_max = glm::vec3{std::numeric_limits<float>::lowest()},
            };

            // The draw's box, in its cell, and the scene's, in the world, take
            // in the 8 corners of the primitive's box, moved into the world.
            for (int corner = 0; corner < 8; ++corner) {
                const glm::dvec3 local{
                    corner & 1 ? primitive.local_max.x : primitive.local_min.x,
                    corner & 2 ? primitive.local_max.y : primitive.local_min.y,
                    corner & 4 ? primitive.local_max.z : primitive.local_min.z,
                };
                const glm::dvec3 point = glm::dvec3(world * glm::dvec4(local, 1.0));

                draw.bounds_min = glm::min(draw.bounds_min, glm::vec3(point - corner_position));
                draw.bounds_max = glm::max(draw.bounds_max, glm::vec3(point - corner_position));
                scene.bounds_min = glm::min(scene.bounds_min, glm::vec3(point));
                scene.bounds_max = glm::max(scene.bounds_max, glm::vec3(point));
            }

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

Then replace `load_gltf` with:
```cpp
Scene load_gltf(const std::filesystem::path& path, const glm::dvec3& origin) {
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
        visit_node(model, node, glm::translate(glm::dmat4{1.0}, origin), mesh_primitives, scene);
    }

    if (scene.draws.empty()) {
        throw std::runtime_error(path.string() + " has nothing to draw");
    }

    return scene;
}
```

## 13.4 A camera in doubles, without a far plane: `camera.h`, `camera.cpp`

### Why
The camera is the one position every frame depends on, and it must keep its precision too: moving a millimetre 1,000 km from the origin is a change a float can't hold. And the projection's far plane has to go.

### How
- **`position`** is a `glm::dvec3`. Movement is computed in floats, as before, and added to it in doubles.
- **`view`** only turns the world: in camera-relative space the camera is at the origin, so there's nothing to move. `glm::lookAt` from the origin along `forward()`.
- **`projection`** is written out by hand: reverse-Z with no far plane. A point at distance `d` in front of the camera, at view-space `z = −d`, gets clip-space `z = near` and `w = d`, so its depth is `near / d`:
  - 1 at the near plane, falling toward 0 with distance, and 0 only at infinity.
  - A 32-bit float depth resolves about `d × 2^−23` at distance `d`: 0.12 mm at 1 km, 2.4 mm at 20 km. It's as precise, relative to the distance, as a float position. Reverse-Z made that true; dropping the far plane costs nothing.
  - `far_plane` is gone.
- **`GLM_FORCE_DEPTH_ZERO_TO_ONE`** made `glm::perspective` produce Vulkan's 0-to-1 depth range. Nothing calls `glm::perspective` any more, so the definition goes.
- **What depends on the far plane:**
  - **The sky** sits at depth 0, now infinitely far. It still draws only where nothing else did.
  - **The cull's far plane,** `z ≥ 0`, is now always true. That row of the matrix is `(0, 0, 0, near)`, and the plane culls nothing.
  - **The sky's view directions** came from points on the far plane, which is now at infinity, where `w` is 0. They come from the near plane instead (13.6).

### Code
`game-engine/src/includes/camera.h`:
```cpp
#pragma once

#include <glm/glm.hpp>

// What the mouse did since the last frame. The keyboard is read directly
// from SDL in update_camera().
struct CameraInput {
    glm::vec2 mouse_delta{0.0f};  // pixels moved since the last frame
    float wheel = 0.0f;           // scroll steps; positive is away from you
    bool right_button = false;    // held: look around and fly
    bool left_button = false;     // held: turn and move along the ground
    bool middle_button = false;   // held: pan
};

// A fly camera with the Unreal Editor viewport's controls:
//   right button held   mouse looks around, WASD moves, Q/E go down/up,
//                       scroll changes the flying speed
//   left button drag    left/right turns, up/down moves forward/back
//   middle button drag  pans sideways and up/down
//   scroll              moves forward/back
//
// The position is in doubles, in metres from the world's origin, so it keeps
// its precision anywhere: moving 1 mm at 100 km is still a change a double
// can tell. Shaders never see it: they work relative to the camera.
struct FlyCamera {
    glm::dvec3 position{0.0};
    float yaw = 0.0f;    // radians around +Y; 0 looks down -Z, glm's "forward"
    float pitch = 0.0f;  // radians up (+) or down (-)
    float vertical_fov = glm::radians(60.0f);
    float near_plane = 0.05f;
    float speed = 3.0f;  // world units per second while flying

    // The unit vector the camera looks along.
    glm::vec3 forward() const;

    // Camera-relative space -> view space: the world as seen from the
    // camera, which is at the origin of camera-relative space. Only a
    // rotation: there's nothing to move.
    glm::mat4 view() const;

    // View space -> clip space, for an image `aspect` (width / height) wide,
    // with reverse-Z and no far plane.
    glm::mat4 projection(float aspect) const;
};

// Moves and turns `camera` from `input` and the keyboard, `seconds` after
// the last update.
void update_camera(FlyCamera& camera, const CameraInput& input, float seconds);
```

`game-engine/src/camera.cpp`:
```cpp
#include "includes/camera.h"

#include <SDL3/SDL.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace {

constexpr glm::vec3 world_up{0.0f, 1.0f, 0.0f};

constexpr float look_sensitivity = 0.003f;  // radians per pixel of mouse movement
constexpr float drag_sensitivity = 0.01f;   // fraction of `speed` moved per pixel of drag
constexpr float scroll_step = 0.2f;         // fraction of `speed` moved per scroll step
constexpr float speed_step = 1.25f;         // flying speed multiplier per scroll step
constexpr float min_speed = 0.05f;
constexpr float max_speed = 500.0f;

}  // namespace

// --- Matrices ----------------------------------------------------------------

glm::vec3 FlyCamera::forward() const {
    // Yaw turns around +Y, pitch tilts up and down. At yaw 0 and pitch 0 this
    // is (0, 0, -1): glm's view space looks down -Z.
    return {
        -std::sin(yaw) * std::cos(pitch),
        std::sin(pitch),
        -std::cos(yaw) * std::cos(pitch),
    };
}

glm::mat4 FlyCamera::view() const {
    return glm::lookAt(glm::vec3{0.0f}, forward(), world_up);
}

glm::mat4 FlyCamera::projection(float aspect) const {
    // Reverse-Z with an infinite far plane. A point at distance d in front of
    // the camera (view-space z = -d) gets clip-space
    //     z = near,  w = d,  so depth = z / w = near / d:
    // 1 at the near plane, falling toward 0 as d grows, and 0 only at
    // infinity. There's no far plane: nothing is too far to draw. Floats are
    // most precise near 0, where the distant depths crowd together, so a
    // 32-bit float depth resolves about d x 2^-23 at any distance, like a
    // float position.
    const float focal = 1.0f / std::tan(vertical_fov * 0.5f);

    glm::mat4 projection{0.0f};  // glm indexes [column][row]
    projection[0][0] = focal / aspect;
    projection[1][1] = -focal;   // Vulkan's clip-space +Y points down
    projection[2][3] = -1.0f;    // w = -z = d
    projection[3][2] = near_plane;
    return projection;
}

// --- Controls ----------------------------------------------------------------

void update_camera(FlyCamera& camera, const CameraInput& input, float seconds) {
    const glm::vec3 forward = camera.forward();
    const glm::vec3 right = glm::normalize(glm::cross(forward, world_up));

    // Forward with the pitch removed, for moving along the ground.
    const glm::vec3 ground_forward{-std::sin(camera.yaw), 0.0f, -std::cos(camera.yaw)};

    const float drag_distance = camera.speed * drag_sensitivity;

    if (input.right_button) {
        // Look: each pixel of mouse movement is a small angle. Pitch stops just
        // short of straight up or down, where "forward" and "up" would coincide.
        camera.yaw -= input.mouse_delta.x * look_sensitivity;
        camera.pitch -= input.mouse_delta.y * look_sensitivity;
        camera.pitch = std::clamp(camera.pitch, glm::radians(-89.0f), glm::radians(89.0f));

        // Scrolling while flying changes the speed, 25% per step.
        camera.speed = std::clamp(camera.speed * std::pow(speed_step, input.wheel), min_speed, max_speed);

        // Fly: SDL keeps the current up/down state of every key.
        const bool* keys = SDL_GetKeyboardState(nullptr);

        glm::vec3 direction{0.0f};
        if (keys[SDL_SCANCODE_W]) direction += forward;
        if (keys[SDL_SCANCODE_S]) direction -= forward;
        if (keys[SDL_SCANCODE_D]) direction += right;
        if (keys[SDL_SCANCODE_A]) direction -= right;
        if (keys[SDL_SCANCODE_E]) direction += world_up;
        if (keys[SDL_SCANCODE_Q]) direction -= world_up;

        // Speed times seconds since the last frame: the same speed at any frame rate.
        if (direction != glm::vec3{0.0f}) {
            camera.position += glm::dvec3(glm::normalize(direction) * camera.speed * seconds);
        }

        return;
    }

    if (input.left_button) {
        // Left drag: sideways turns, up/down moves along the ground.
        camera.yaw -= input.mouse_delta.x * look_sensitivity;
        camera.position -= glm::dvec3(ground_forward * input.mouse_delta.y * drag_distance);
    } else if (input.middle_button) {
        // Middle drag: pan in the plane facing the camera.
        const glm::vec3 up = glm::cross(right, forward);
        camera.position += glm::dvec3((right * input.mouse_delta.x - up * input.mouse_delta.y) * drag_distance);
    }

    // Scrolling without flying moves forward and back.
    camera.position += glm::dvec3(forward * input.wheel * camera.speed * scroll_step);
}
```

In `game-engine/CMakeLists.txt`, remove these lines:
```cmake
# glm follows OpenGL, whose clip-space depth runs from -1 to 1. Vulkan's runs
# from 0 to 1; this makes glm::perspective produce that range instead.
target_compile_definitions(game-engine PRIVATE GLM_FORCE_DEPTH_ZERO_TO_ONE)
```

## 13.5 A TLAS near the camera: `acceleration.h`, `acceleration.cpp`

### Why
The TLAS is the one structure the GPU traces against directly, without the shaders moving anything first. Its instances' transforms are floats, so the TLAS needs an origin of its own near the camera, or rays far from the world's origin would be traced at coordinates that have lost their detail.

### How
- **The TLAS's origin cell:** its space is measured from one cell's corner. Each instance's transform is its draw's model matrix, moved from its own cell's corner to the origin cell's: `cell_corner(draw.cell, origin_cell)`, exact.
- **Rays in TLAS space:** a shadow ray starts at a camera-relative position, plus `tlas_offset`, the camera's position in the TLAS's space. Near the camera, that's at most about 1.1 km, where floats are 0.06 to 0.12 mm apart.
- **Rebuilding:** `build_tlas` builds just the TLAS, around a given origin cell, for the BLASes already built. `build_acceleration_structures` builds the BLASes, then calls it. Main calls it again when the camera has strayed more than 16 cells, about 1 km, from the origin cell (13.7).
- **Replacing the old TLAS:** the new one is built in buffers of its own first. Then the old structure is destroyed, then the old buffers it lived in and was built from.
- **Barriers across submissions:** the BLAS build ends with a barrier that makes the BLASes visible to later acceleration structure builds, including TLAS builds in later submissions. A pipeline barrier's second scope covers every command submitted after it on the queue, not just the rest of its own command buffer.
- **Uploads need the same:** `upload_buffer` (Chapter 2) copies into a device-local buffer and waits for the copy to finish. Waiting tells the CPU the copy is done, but it doesn't make the copy's writes visible to the GPU's later reads: only a barrier does that. Drivers have tolerated the gap, but the TLAS is now rebuilt at run time from a freshly uploaded instance buffer, so `upload_buffer` now ends with a barrier from the copy's writes to every later command's reads.

### Code
`game-engine/src/includes/acceleration.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/scene.h"
#include "includes/vulkan_setup.h"

#include <vector>

// --- Acceleration structures -------------------------------------------------

// The scene organized for tracing rays, in two levels:
//   - one bottom-level acceleration structure (BLAS) per glTF primitive: a
//     tree of boxes around its triangles, in the primitive's own space,
//   - one top-level acceleration structure (TLAS) over every draw: each an
//     instance of its primitive's BLAS, placed by the draw's model matrix.
// A ray first finds which instances' boxes it crosses, then which
// triangles. Members are destroyed bottom-up, so the acceleration structures
// go before the buffers that hold them.
//
// The TLAS holds floats, like everything the GPU sees, so its space is
// measured from the corner of one world cell near the camera, its origin
// cell (cells.h): a ray near the camera is traced at small coordinates, with
// a float's full precision. Once the camera has moved far from that cell,
// main rebuilds the TLAS around the camera's cell.
struct AccelerationStructures {
    Buffer blas_storage;   // every BLAS, one after another
    Buffer instances;      // one VkAccelerationStructureInstanceKHR per draw
    Buffer tlas_storage;

    std::vector<vk::raii::AccelerationStructureKHR> blases;  // one per primitive
    vk::raii::AccelerationStructureKHR tlas = nullptr;
    vk::DeviceAddress tlas_address = 0;                      // what shaders trace against
    glm::ivec3 origin_cell{0};                               // the cell the TLAS's space starts at
};

// Builds a BLAS for every primitive and a TLAS for every draw around
// `origin_cell`, on the GPU, and waits for them. The triangles are read
// straight from the scene's vertex and index buffers, which must have been
// created with eAccelerationStructureBuildInputReadOnlyKHR.
AccelerationStructures build_acceleration_structures(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    const Scene& scene,
    const Buffer& vertex_buffer,
    const Buffer& index_buffer,
    const glm::ivec3& origin_cell
);

// Builds the TLAS again, around `origin_cell`, and waits for it. The old one
// is destroyed, so the GPU must not be using it.
void build_tlas(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    AccelerationStructures& structures,
    const Scene& scene,
    const glm::ivec3& origin_cell
);
```

`game-engine/src/acceleration.cpp`:
```cpp
#include "includes/acceleration.h"

#include "includes/cells.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace {

vk::DeviceSize align_up(vk::DeviceSize value, vk::DeviceSize alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Acceleration structures live inside buffers, at offsets that are multiples
// of 256 bytes: the Vulkan spec's rule.
constexpr vk::DeviceSize storage_alignment = 256;

// What the GPU requires a build's scratch address to be a multiple of.
vk::DeviceSize scratch_alignment(const GpuChoice& gpu) {
    const auto properties = gpu.device.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceAccelerationStructurePropertiesKHR
    >();
    return properties.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>().minAccelerationStructureScratchOffsetAlignment;
}

// A buffer acceleration structures are stored in.
Buffer create_storage(const vk::raii::Device& device, const GpuChoice& gpu, vk::DeviceSize size) {
    return create_buffer(device, gpu, size,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
}

// Scratch memory: working space for a build, needed only while it runs. Its
// address must be a multiple of `alignment`, so the buffer is a little
// larger and the returned address rounded up within it.
Buffer create_scratch(const vk::raii::Device& device, const GpuChoice& gpu, vk::DeviceSize size, vk::DeviceSize alignment) {
    return create_buffer(device, gpu, size + alignment,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
}

// Vulkan wants an instance's transform as a 3x4 matrix of rows; glm stores
// 4x4 matrices by columns, so element (row, column) is model[column][row].
// The bottom row of a model matrix is always (0, 0, 0, 1) and is left out.
vk::TransformMatrixKHR to_transform(const glm::mat4& model) {
    vk::TransformMatrixKHR transform{};

    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 4; ++column) {
            transform.matrix[row][column] = model[column][row];
        }
    }

    return transform;
}

}  // namespace

// --- Building ----------------------------------------------------------------

AccelerationStructures build_acceleration_structures(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    const Scene& scene,
    const Buffer& vertex_buffer,
    const Buffer& index_buffer,
    const glm::ivec3& origin_cell
) {
    const vk::DeviceSize alignment = scratch_alignment(gpu);

    AccelerationStructures structures;

    // --- One BLAS per primitive ----------------------------------------------

    // Each primitive's triangles, described where they already are: its run of
    // indices in the index buffer, and its vertices in the vertex buffer.
    // Only the position, the first 12 bytes of each Vertex, is read.
    const std::size_t count = scene.primitives.size();
    std::vector<vk::AccelerationStructureGeometryKHR> geometries(count);
    std::vector<vk::AccelerationStructureBuildGeometryInfoKHR> builds(count);
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges(count);
    std::vector<vk::AccelerationStructureBuildSizesInfoKHR> sizes(count);
    std::vector<vk::DeviceSize> storage_offsets(count);
    std::vector<vk::DeviceSize> scratch_offsets(count);

    vk::DeviceSize storage_size = 0;
    vk::DeviceSize scratch_size = 0;

    for (std::size_t i = 0; i < count; ++i) {
        const Primitive& primitive = scene.primitives[i];

        // The highest vertex an index refers to, which the build must know.
        const auto first = scene.indices.begin() + primitive.first_index;
        const std::uint32_t max_vertex = *std::max_element(first, first + primitive.index_count);

        // Opaque triangles let a ray stop at the first one it hits. The others
        // are reported to the shader, which checks their alpha: a ray passes
        // through a leaf's empty corners (masked), or is dimmed by a
        // see-through layer (blended). A blended triangle must be reported
        // only once per ray, or a ray would be dimmed by it twice; without the
        // flag, the GPU may report one more than once.
        const AlphaMode alpha_mode = scene.materials[primitive.material].alpha_mode;
        const vk::GeometryFlagsKHR geometry_flags = alpha_mode == AlphaMode::opaque ? vk::GeometryFlagBitsKHR::eOpaque
            : alpha_mode == AlphaMode::blend ? vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation
            : vk::GeometryFlagsKHR{};

        geometries[i] = vk::AccelerationStructureGeometryKHR{
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry = {.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR{
                .vertexFormat = vk::Format::eR32G32B32Sfloat,
                .vertexData = {.deviceAddress = vertex_buffer.address + static_cast<vk::DeviceSize>(primitive.vertex_offset) * sizeof(Vertex)},
                .vertexStride = sizeof(Vertex),
                .maxVertex = max_vertex,
                .indexType = vk::IndexType::eUint32,
                .indexData = {.deviceAddress = index_buffer.address + primitive.first_index * sizeof(std::uint32_t)},
            }},
            .flags = geometry_flags,
        };

        // Built once, traced every frame: optimize for tracing.
        builds[i] = vk::AccelerationStructureBuildGeometryInfoKHR{
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = 1,
            .pGeometries = &geometries[i],
        };

        ranges[i] = vk::AccelerationStructureBuildRangeInfoKHR{.primitiveCount = primitive.index_count / 3};

        // The driver says how much memory the structure and its build need.
        sizes[i] = device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, builds[i], ranges[i].primitiveCount);

        storage_offsets[i] = storage_size;
        storage_size = align_up(storage_size + sizes[i].accelerationStructureSize, storage_alignment);
        scratch_offsets[i] = scratch_size;
        scratch_size = align_up(scratch_size + sizes[i].buildScratchSize, alignment);
    }

    // One buffer holds every BLAS, and one scratch buffer every build's
    // working space, so all of them can be built at once.
    structures.blas_storage = create_storage(device, gpu, storage_size);
    const Buffer blas_scratch = create_scratch(device, gpu, scratch_size, alignment);
    const vk::DeviceAddress blas_scratch_address = align_up(blas_scratch.address, alignment);

    for (std::size_t i = 0; i < count; ++i) {
        structures.blases.emplace_back(device, vk::AccelerationStructureCreateInfoKHR{
            .buffer = *structures.blas_storage.handle,
            .offset = storage_offsets[i],
            .size = sizes[i].accelerationStructureSize,
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        });

        builds[i].dstAccelerationStructure = *structures.blases[i];
        builds[i].scratchData.deviceAddress = blas_scratch_address + scratch_offsets[i];
    }

    // --- Building, on the GPU --------------------------------------------------

    // Every BLAS in one call. The barrier makes them visible to the TLAS
    // build, which reads them: here, and whenever the TLAS is rebuilt.
    std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> range_pointers;
    for (const auto& range : ranges) {
        range_pointers.push_back(&range);
    }

    submit_and_wait(device, queue, pool, [&](const vk::raii::CommandBuffer& commands) {
        commands.buildAccelerationStructuresKHR(builds, range_pointers);

        const vk::MemoryBarrier2 blases_built{
            .srcStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            .srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            .dstStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR,
        };
        commands.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &blases_built});
    });

    // The BLAS builds are done: their scratch buffer can go, on return.
    build_tlas(device, gpu, queue, pool, structures, scene, origin_cell);
    return structures;
}

// --- The TLAS ------------------------------------------------------------------

void build_tlas(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    AccelerationStructures& structures,
    const Scene& scene,
    const glm::ivec3& origin_cell
) {
    // An instance places a BLAS in the TLAS's space: the draw's model matrix,
    // moved from its cell's corner to the origin cell's. Its custom index is
    // the draw's index, which a ray query reports back, so the shader can
    // find the draw's material and triangles.
    std::vector<vk::AccelerationStructureInstanceKHR> instances;

    for (std::uint32_t i = 0; i < scene.draws.size(); ++i) {
        const MeshDraw& draw = scene.draws[i];

        glm::mat4 model = draw.model;
        model[3] += glm::vec4(cell_corner(draw.cell, origin_cell), 0.0f);

        instances.push_back(vk::AccelerationStructureInstanceKHR{
            .transform = to_transform(model),
            .instanceCustomIndex = i,
            .mask = 0xFF,
            .instanceShaderBindingTableRecordOffset = 0,
            // Rays hit both sides of a triangle anyway, unless a ray asks to
            // cull one side; this keeps a shadow-casting surface from ever
            // being culled, even by such a ray.
            .flags = static_cast<VkGeometryInstanceFlagsKHR>(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable),
            .accelerationStructureReference = device.getAccelerationStructureAddressKHR(
                vk::AccelerationStructureDeviceAddressInfoKHR{.accelerationStructure = *structures.blases[draw.primitive]}),
        });
    }

    Buffer instance_buffer = upload_buffer(device, gpu, queue, pool, std::as_bytes(std::span(instances)),
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress);

    const vk::AccelerationStructureGeometryKHR instance_geometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry = {.instances = vk::AccelerationStructureGeometryInstancesDataKHR{
            .data = {.deviceAddress = instance_buffer.address},
        }},
    };

    vk::AccelerationStructureBuildGeometryInfoKHR tlas_build{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &instance_geometry,
    };

    const vk::AccelerationStructureBuildRangeInfoKHR tlas_range{.primitiveCount = static_cast<std::uint32_t>(instances.size())};
    const vk::AccelerationStructureBuildSizesInfoKHR tlas_sizes = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, tlas_build, tlas_range.primitiveCount);

    const vk::DeviceSize alignment = scratch_alignment(gpu);
    Buffer tlas_storage = create_storage(device, gpu, tlas_sizes.accelerationStructureSize);
    const Buffer tlas_scratch = create_scratch(device, gpu, tlas_sizes.buildScratchSize, alignment);

    vk::raii::AccelerationStructureKHR tlas(device, vk::AccelerationStructureCreateInfoKHR{
        .buffer = *tlas_storage.handle,
        .size = tlas_sizes.accelerationStructureSize,
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
    });

    tlas_build.dstAccelerationStructure = *tlas;
    tlas_build.scratchData.deviceAddress = align_up(tlas_scratch.address, alignment);

    // The barrier makes the TLAS visible to fragment shaders' ray queries.
    const vk::AccelerationStructureBuildRangeInfoKHR* tlas_range_pointer = &tlas_range;

    submit_and_wait(device, queue, pool, [&](const vk::raii::CommandBuffer& commands) {
        commands.buildAccelerationStructuresKHR(tlas_build, tlas_range_pointer);

        const vk::MemoryBarrier2 tlas_built{
            .srcStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            .srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR,
        };
        commands.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &tlas_built});
    });

    // The new TLAS replaces the old one: first the old structure goes, then
    // the buffers it lived in and was built from.
    structures.tlas = std::move(tlas);
    structures.tlas_storage = std::move(tlas_storage);
    structures.instances = std::move(instance_buffer);
    structures.tlas_address = device.getAccelerationStructureAddressKHR(
        vk::AccelerationStructureDeviceAddressInfoKHR{.accelerationStructure = *structures.tlas});
    structures.origin_cell = origin_cell;
}
```

In `game-engine/src/buffer.cpp`, replace `upload_buffer` with:
```cpp
Buffer upload_buffer(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    std::span<const std::byte> bytes,
    vk::BufferUsageFlags usage
) {
    // The CPU can't write device-local memory directly, so the bytes go into a
    // host-visible "staging" buffer first. Host-coherent means our writes are
    // visible to the GPU without an explicit flush.
    const Buffer staging = create_buffer(device, gpu, bytes.size(),
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void* mapped = staging.memory.mapMemory(0, bytes.size());
    std::memcpy(mapped, bytes.data(), bytes.size());
    staging.memory.unmapMemory();

    // The real buffer lives in device-local memory and is filled by a GPU copy.
    Buffer buffer = create_buffer(device, gpu, bytes.size(),
        usage | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    submit_and_wait(device, queue, pool, [&](const vk::raii::CommandBuffer& commands) {
        commands.copyBuffer(*staging.handle, *buffer.handle, vk::BufferCopy{.size = bytes.size()});

        // Whatever reads the buffer later, in this submission or any after
        // it, sees the copy's writes. Waiting for the submission to finish
        // only tells the CPU the copy is done; it doesn't make the copy's
        // writes visible to the GPU's later reads. A barrier does.
        const vk::MemoryBarrier2 copied{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
        };
        commands.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &copied});
    });

    // Waiting above means the copy is done, so `staging` can be destroyed on return.
    return buffer;
}
```

## 13.6 Shaders in camera-relative space: `mesh.slang`, `ao.slang`, `cull.slang`, `background.slang`

### Why
Every position a shader computes is now relative to the camera. Where they compared against the camera's position before, the camera is now at the origin.

### How
- **`mesh.slang`:**
  - **The vertex shader** moves each vertex into its draw's cell with the model matrix, then to the camera with `camera_relative`.
  - **`relative_position`** replaces `world_position` in `VertexOutput`: the world's axes, the camera at the origin.
  - **The view direction** is `normalize(−relative_position)`.
  - **Lights:** `punctual_light` moves each light's cell and offset into camera-relative space.
  - **Shadow rays:** `shadow_ray_origin` adds `tlas_offset` first, then pushes the origin off the surface. The push grows with the coordinates the ray is really traced at, in the TLAS's space.
- **`ao.slang`:** it already measured everything from the camera, subtracting the camera's position after unprojecting and adding it back before projecting. `unproject` and `to_pixels` now drop both: the inverse view-projection gives camera-relative positions, and the view-projection takes them. Its rays now come from the near plane instead of the far one: depth 0 is infinitely far now, and with nothing taken off, the near plane loses no digits.
- **`cull.slang`:** each draw's box corners go through `camera_relative` before the frustum test. The far plane culls nothing now.
- **`background.slang`:** a pixel's view direction comes from its point on the near plane, at depth 1, divided by `w`. The far plane is at infinity: there, `w` is 0, and the division fails.

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
    float3 relative_position : POSITION;  // camera-relative: the world's axes, the camera at the origin
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

    // The vertex in its draw's cell, then relative to the camera.
    const float3 in_cell = mul(draw.model, float4(vertex.position, 1.0)).xyz;
    const float3 relative_position = camera_relative(frame, draw.cell, in_cell);

    // Tangent and bitangent lie along the surface, so they move with the
    // model matrix, like positions; only the normal needs the normal matrix.
    // The bitangent is built before the transform, from glTF's rule
    // B = cross(N, T) * w: a mirroring transform then mirrors it too.
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
        normal = cross(ddy(input.relative_position), ddx(input.relative_position));
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
//
// The TLAS's space is measured from its origin cell, near the camera, not
// from the camera itself (acceleration.h): tlas_offset, the camera's
// position in it, moves the camera-relative position there first, so the
// offset grows with the coordinates the ray is really traced at.
float3 shadow_ray_origin(FrameData* frame, float3 position, float3 face_normal, float3 l) {
    return offset_ray_origin(position + frame.tlas_offset, dot(face_normal, l) >= 0.0 ? face_normal : -face_normal);
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

    const float reaching = light_visibility(frame, shadow_ray_origin(frame, position, face_normal, l), l, distance);
    return reaching > 0.0 ? shade(surface, l, illuminance) * reaching : float3(0.0);
}

// --- Lights --------------------------------------------------------------------

// The direction toward a light, how far away it is, and the illuminance it
// gives here, following KHR_lights_punctual. Point and spot lights fade with
// the square of the distance, then smoothly to nothing at `range`; spot
// lights also fade from the inner cone to the outer one. A directional
// light is infinitely far away.
float3 punctual_light(FrameData* frame, Light light, float3 position, out float3 l, out float distance) {
    if (light.type == light_directional) {
        l = -light.direction;
        distance = infinite_distance;
        return light.intensity;
    }

    const float3 to_light = camera_relative(frame, light.cell, light.offset) - position;
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
        normal = cross(ddy(input.relative_position), ddx(input.relative_position));
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

    // A perfectly smooth surface would reflect a punctual light from a single
    // point, too small for any pixel to catch; a floor on roughness keeps
    // highlights visible.
    const Surface surface = {
        base_color.rgb,
        metallic,
        max(roughness, 0.045) * max(roughness, 0.045),
        normal,
        normalize(-input.relative_position),  // toward the camera, at the origin
    };

    // Direct light, shadowed: the sun, then every light in the file.
    float3 radiance = shade_shadowed(surface, frame, input.relative_position, face_normal,
        frame.sun_direction, frame.sun_illuminance, infinite_distance);

    for (uint i = 0; i < frame.light_count; ++i) {
        float3 l;
        float distance;
        const float3 illuminance = punctual_light(frame, frame.lights[i], input.relative_position, l, distance);
        radiance += shade_shadowed(surface, frame, input.relative_position, face_normal, l, illuminance, distance);
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
// and distances from 0.1 m to 500 m. Past a few hundred metres it bottoms
// out at its floor, so distant layers that overlap count equally. It falls
// steeply with the distance in front of the camera, so where layers overlap,
// the nearest dominates; the
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
    const float view_depth = mul(push.frame.view_projection, float4(input.relative_position, 1.0)).w;
    const float weight = transparent_weight(coverage, view_depth);

    TransparentOutput output;
    output.accum = float4(min(color.rgb, transparent_max) * coverage, coverage) * weight;
    output.reveal = coverage;
    return output;
}
```

`game-engine/shaders/ao.slang`:
```slang
// Ambient occlusion: how much of the sky each visible point can see, from the
// depth buffer and the prepass's normals. Ground-truth ambient occlusion
// (GTAO: Jimenez et al. 2016, "Practical Real-Time Strategies for Accurate
// Indirect Occlusion"), in the form Intel's XeGTAO gives it, at half
// resolution, in four steps:
//   prefilterMain  per 2 x 2 block of pixels: the nearest surface's distance
//                  in front of the camera and its normal, at half resolution
//   gtaoMain       for a few slices of the hemisphere around the view
//                  direction, walk outward on both sides and keep the highest
//                  horizon found; the visible arc between the two horizons,
//                  integrated against the cosine, is the visibility, and its
//                  centroid the bent normal
//   blurMain       a separable, depth-aware blur of the visibility
//   upsampleMain   back to full resolution: each pixel blends its four
//                  nearest half-resolution results, weighted by distance and
//                  by how close their depth is to its own
// It's deterministic: a fixed per-pixel hash, no frame index, no accumulation.

#include "shared.slangh"

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

struct AoPushData {
    FrameData* frame;
    uint depth;       // the depth buffer, full resolution
    uint normals;     // the prepass's normals, full resolution
    uint ao_depth;    // half resolution: distance in front of the camera
    uint ao_normals;  // half resolution: the normal, octahedral
    uint source;
    uint target;
    uint width;       // full resolution
    uint height;
    float radius;
    uint slices;
    uint steps;
    uint blur_axis;
};

[[vk::push_constant]]
ConstantBuffer<AoPushData> push;

static const float pi = 3.14159265;
static const float half_pi = 1.57079633;

// --- Positions ---------------------------------------------------------------------

// Everything here is measured from the camera, as in every shader: the
// view direction toward it is just the negated position.

// The camera-relative position of the surface at full-resolution pixel
// position `pixel` with depth `depth`: its clip-space coordinates, back
// through the inverse view-projection, divided by w.
float3 unproject(FrameData* frame, float2 pixel, float depth) {
    const float2 ndc = pixel / float2(push.width, push.height) * 2.0 - 1.0;
    const float4 position = mul(frame.inverse_view_projection, float4(ndc, depth, 1.0));
    return position.xyz / position.w;
}

// Where a camera-relative position lands on screen, in full-resolution
// pixels.
float2 to_pixels(FrameData* frame, float3 position) {
    const float4 clip = mul(frame.view_projection, float4(position, 1.0));
    return (clip.xy / clip.w * 0.5 + 0.5) * float2(push.width, push.height);
}

// The direction the camera looks along: through the centre of the screen.
// Both this and view_ray unproject at the near plane, depth 1: with no far
// plane, depth 0 is infinitely far, and positions are already measured from
// the camera, so nothing is taken off them to lose digits.
float3 camera_forward(FrameData* frame) {
    return normalize(unproject(frame, float2(push.width, push.height) * 0.5, 1.0));
}

// The ray through full-resolution pixel position `pixel`, scaled so that one
// step along it is one metre further in front of the camera: a point at
// distance d in front of the camera, along the ray, is at ray x d. Rays
// through the pixels of one row or column change by the same amount from
// pixel to pixel, so a few rays give all the others by adding.
float3 view_ray(FrameData* frame, float2 pixel, float3 forward) {
    const float3 near_point = unproject(frame, pixel, 1.0);
    return near_point / dot(near_point, forward);
}

uint2 half_size() {
    return (uint2(push.width, push.height) + 1) / 2;
}

// A half-resolution pixel position, in full-resolution pixels.
float2 full_pixel(float2 half_pixel) {
    return half_pixel * 2.0;
}

// --- 1. Prefilter -------------------------------------------------------------------

// The search runs on a quarter of the pixels, and reads small images it can
// keep in its caches: per 2 x 2 block, the nearest surface's distance in
// front of the camera (0 for the sky) and its normal. The nearest, so that
// a thin pole in front of a wall keeps its own occlusion. Both images are
// storage images, which the later steps read back as storage images too:
// one descriptor each, and they stay in eGeneral throughout.
[shader("compute")]
[numthreads(8, 8, 1)]
void prefilterMain(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= half_size())) {
        return;
    }

    FrameData* frame = push.frame;
    const Texture2D depth_buffer = Texture2D.Handle(uint2(push.depth, 0));
    const Texture2D normals = Texture2D.Handle(uint2(push.normals, 0));
    RWTexture2D<float> ao_depth = RWTexture2D<float>.Handle(uint2(push.ao_depth, 0));
    RWTexture2D<float2> ao_normals = RWTexture2D<float2>.Handle(uint2(push.ao_normals, 0));

    // With reverse-Z, nearer means a greater depth.
    int2 nearest = int2(id.xy) * 2;
    float nearest_depth = 0.0;

    for (int i = 0; i < 4; ++i) {
        const int2 pixel = min(int2(id.xy) * 2 + int2(i & 1, i >> 1), int2(push.width, push.height) - 1);
        const float depth = depth_buffer.Load(int3(pixel, 0)).r;

        if (depth > nearest_depth) {
            nearest_depth = depth;
            nearest = pixel;
        }
    }

    if (nearest_depth == 0.0) {
        ao_depth[id.xy] = 0.0;
        ao_normals[id.xy] = float2(0.0);
        return;
    }

    const float3 position = unproject(frame, float2(nearest) + 0.5, nearest_depth);
    ao_depth[id.xy] = dot(position, camera_forward(frame));
    ao_normals[id.xy] = normals.Load(int3(nearest, 0)).xy;
}

// --- Small helpers ---------------------------------------------------------------

// acos to within about 0.01 radians, for a fraction of the cost: it runs
// three times per slice. The polynomial is XeGTAO's.
float fast_acos(float x) {
    const float a = abs(x);
    const float r = (-0.156583 * a + half_pi) * sqrt(1.0 - a);
    return x >= 0.0 ? r : pi - r;
}

// A number in 0..1 per pixel, different for each `salt`. A hash rather than
// a regular pattern, so the blur that follows can smooth it away without
// leaving bands or diagonals.
float spatial_hash(uint2 pixel, uint salt) {
    uint h = pixel.x * 73856093u ^ pixel.y * 19349663u ^ salt * 83492791u;
    h = (h ^ (h >> 16)) * 0x45D9F3Bu;
    h = (h ^ (h >> 16)) * 0x45D9F3Bu;
    return float(h & 0xFFFFu) / 65535.0;
}

// --- The visible arc ---------------------------------------------------------------

// Within one slice, angles h are measured from the view direction, and n is
// the angle of the surface normal projected into the slice. The share of the
// slice's light arriving between the view direction and angle h, weighted by
// the cosine to the normal and by |sin h| (each angle's share of the solid
// angle), has a closed form; so do its components along the view direction
// and along the slice, which give the bent normal:
//   A(h)  = (cos n + 2 h sin n - cos(2h - n)) / 4
//   Cv(h) = cos n (-cos^3 h / 3) + sin n (sin^3 h / 3)
//   Co(h) = cos n (sin^3 h / 3) + sin n (cos^3 h / 3 - cos h)
// Cv and Co leave out their constants: over both sides of the slice, those
// add (2/3) cos n and (4/3) sin n.
void arc_terms(float h, float cos_n, float sin_n, out float a, out float along_view, out float along_slice) {
    const float c = cos(h);
    const float s = sin(h);
    const float c3 = c * c * c;
    const float s3 = s * s * s;

    // cos(2h - n), expanded so there's one cos and one sin to compute.
    const float cos_2h_n = (1.0 - 2.0 * s * s) * cos_n + 2.0 * s * c * sin_n;

    a = 0.25 * (cos_n + 2.0 * h * sin_n - cos_2h_n);
    along_view = cos_n * (-c3 / 3.0) + sin_n * (s3 / 3.0);
    along_slice = cos_n * (s3 / 3.0) + sin_n * (c3 / 3.0 - c);
}

// --- 2. The horizon search -----------------------------------------------------------

// Runs per half-resolution pixel, on the prefiltered images. A sample's
// position is its pixel's ray times its distance in front of the camera:
// the rays come from three computed here, by adding, with no matrix per
// sample.
[shader("compute")]
[numthreads(8, 8, 1)]
void gtaoMain(uint3 id : SV_DispatchThreadID) {
    const uint2 size = half_size();

    if (any(id.xy >= size)) {
        return;
    }

    FrameData* frame = push.frame;
    const int2 pixel = int2(id.xy);
    RWTexture2D<float> ao_depth = RWTexture2D<float>.Handle(uint2(push.ao_depth, 0));
    RWTexture2D<float2> ao_normals = RWTexture2D<float2>.Handle(uint2(push.ao_normals, 0));
    RWTexture2D<float4> target = RWTexture2D<float4>.Handle(uint2(push.target, 0));

    // A distance of 0 is the sky, which nothing occludes.
    const float distance_here = ao_depth[pixel];
    if (distance_here == 0.0) {
        target[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // This pixel's ray, and how the ray changes from one half-resolution
    // pixel to the next, across and down. The block's centre stands for
    // whichever of its pixels the prefilter kept.
    const float3 forward = camera_forward(frame);
    const float2 here = float2(pixel) + 0.5;
    const float3 ray = view_ray(frame, full_pixel(here), forward);
    const float3 ray_across = view_ray(frame, full_pixel(here + float2(1.0, 0.0)), forward) - ray;
    const float3 ray_down = view_ray(frame, full_pixel(here + float2(0.0, 1.0)), forward) - ray;

    const float3 centre = ray * distance_here;
    const float3 view_dir = normalize(-centre);  // toward the camera
    const float2 centre_pixels = full_pixel(here);

    // The interpolated vertex normal: the surface at the scale the mesh
    // describes it. A normal facing away from the viewer has no arc the
    // integral can describe, so it's tilted just far enough to face it.
    float3 normal = decode_octahedral(ao_normals[pixel]);
    normal = normalize(normal + max(0.0, 1e-3 - dot(normal, view_dir)) * view_dir);

    // Two axes perpendicular to the view direction. A slice's direction is
    // picked uniformly around the view direction, in 3D: the integral weighs
    // each slice for exactly that. Picking angles on the screen instead would
    // bias it toward the edges of the screen, where the view direction meets
    // the image at a slant.
    const float3 helper = abs(view_dir.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 axis_a = normalize(cross(helper, view_dir));
    const float3 axis_b = cross(view_dir, axis_a);

    const float rotation_noise = spatial_hash(id.xy, 0);
    const float step_noise = spatial_hash(id.xy, 1);

    float visibility = 0.0;
    float3 bent = 0.0;

    for (uint slice = 0; slice < push.slices; ++slice) {
        const float phi = pi * (float(slice) + rotation_noise) / float(push.slices);
        const float3 ortho = axis_a * cos(phi) + axis_b * sin(phi);
        const float3 slice_normal = cross(view_dir, ortho);

        // Where the slice runs on screen, and how many half-resolution
        // pixels the radius covers along it: project a point one radius
        // away. Capped at 8 of them, 16 full-resolution pixels, per step on
        // average, so the steps never skip far over the surface.
        const float2 reach = (to_pixels(frame, centre + ortho * push.radius) - centre_pixels) * 0.5;
        const float reach_length = length(reach);
        if (reach_length < 1e-3) {
            continue;
        }
        const float2 omega = reach / reach_length;
        const float radius_pixels = min(reach_length, 8.0 * float(push.steps));

        // The normal projected into the slice's plane, and its angle n from
        // the view direction, signed toward `ortho`.
        float3 projected = normal - slice_normal * dot(normal, slice_normal);
        const float projected_length = length(projected);
        if (projected_length < 1e-4) {
            continue;
        }
        projected /= projected_length;

        const float cos_n = clamp(dot(projected, view_dir), -1.0, 1.0);
        const float n = sign(dot(projected, ortho)) * fast_acos(cos_n);
        const float sin_n = sin(n);

        // Both horizons start fully open, at the edge of the hemisphere
        // around the normal, kept as cosines from the view direction:
        // cos(n + pi/2) = -sin n on the positive side, cos(n - pi/2) = sin n
        // on the negative one.
        const float open_positive = -sin_n;
        const float open_negative = sin_n;
        float horizon_positive = open_positive;
        float horizon_negative = open_negative;

        for (uint step = 0; step < push.steps; ++step) {
            // Steps bunch up near the centre (t squared), where contact
            // shadows are, and each is at least one pixel further out.
            const float t = (float(step) + 0.5 + step_noise) / float(push.steps);
            const float distance_pixels = max(t * t * radius_pixels, 1.0 + float(step));
            const int2 offset = int2(round(omega * distance_pixels));

            for (int side = 0; side < 2; ++side) {
                const int2 sample_offset = side == 0 ? offset : -offset;
                const int2 sample_pixel = pixel + sample_offset;
                if (any(sample_pixel < 0) || any(sample_pixel >= int2(size))) {
                    continue;
                }

                const float sample_distance = ao_depth[sample_pixel];
                if (sample_distance == 0.0) {
                    continue;
                }

                const float3 sample_ray = ray + ray_across * float(sample_offset.x) + ray_down * float(sample_offset.y);
                const float3 delta = sample_ray * sample_distance - centre;
                const float distance = length(delta);
                if (distance <= 1e-4 || distance > push.radius) {
                    continue;
                }

                // A sample that doesn't rise above the surface's own tangent
                // plane is this same surface, so it can't occlude it.
                if (dot(delta, normal) <= 0.0) {
                    continue;
                }

                // The sample's horizon, faded back toward fully open as it
                // nears the radius, so occluders don't pop in and out.
                const float retract = 1.0 - smoothstep(0.6 * push.radius, push.radius, distance);
                const float horizon = dot(delta / distance, view_dir);

                if (side == 0) {
                    horizon_positive = max(horizon_positive, lerp(open_positive, horizon, retract));
                } else {
                    horizon_negative = max(horizon_negative, lerp(open_negative, horizon, retract));
                }
            }
        }

        // The visible arc runs from angle h0 (negative side) to h1 (positive
        // side), clamped to the hemisphere around the normal. Its value is the
        // slice's visibility; its centroid adds to the bent normal. Each slice
        // counts in proportion to the projected normal's length: the cosine
        // to the real normal is that length times the cosine within the slice.
        float h0 = -fast_acos(clamp(horizon_negative, -1.0, 1.0));
        float h1 = fast_acos(clamp(horizon_positive, -1.0, 1.0));
        h0 = n + max(h0 - n, -half_pi);
        h1 = n + min(h1 - n, half_pi);

        float a0, view0, slice0;
        float a1, view1, slice1;
        arc_terms(h0, cos_n, sin_n, a0, view0, slice0);
        arc_terms(h1, cos_n, sin_n, a1, view1, slice1);

        visibility += (a0 + a1) * projected_length;
        bent += (view_dir * (view0 + view1 + (2.0 / 3.0) * cos_n)
            + ortho * (slice0 + slice1 + (4.0 / 3.0) * sin_n)) * projected_length;
    }

    visibility = saturate(visibility / float(push.slices));
    const float3 bent_normal = dot(bent, bent) > 1e-8 ? normalize(bent) : normal;
    target[pixel] = float4(bent_normal, visibility);
}

// --- 3. The blur ------------------------------------------------------------------------

// How much a neighbour at distance `other` in front of the camera counts
// next to one at `here`: a Gaussian in their relative difference, with a 5%
// difference one standard deviation. A foreground edge never mixes with
// what's behind it.
float depth_weight(float here, float other) {
    const float difference = (other - here) / here;
    return exp(-difference * difference * 200.0);
}

// Nine taps along one axis, at half resolution, Gaussian-weighted and
// depth-weighted. Only the visibility is blurred: a bent normal averaged
// across an edge would point into whatever it was averaged with.
[shader("compute")]
[numthreads(8, 8, 1)]
void blurMain(uint3 id : SV_DispatchThreadID) {
    const uint2 size = half_size();

    if (any(id.xy >= size)) {
        return;
    }

    const int2 pixel = int2(id.xy);
    RWTexture2D<float> ao_depth = RWTexture2D<float>.Handle(uint2(push.ao_depth, 0));
    RWTexture2D<float4> source = RWTexture2D<float4>.Handle(uint2(push.source, 0));
    RWTexture2D<float4> target = RWTexture2D<float4>.Handle(uint2(push.target, 0));

    const float4 centre = source[pixel];
    const float centre_distance = ao_depth[pixel];

    if (centre_distance == 0.0) {
        target[pixel] = centre;
        return;
    }

    const int2 axis = push.blur_axis == 0 ? int2(1, 0) : int2(0, 1);
    const float weights[5] = {0.20416, 0.18017, 0.12383, 0.06628, 0.02763};  // a Gaussian, sigma 2

    float sum = centre.w * weights[0];
    float total = weights[0];

    for (int i = 1; i <= 4; ++i) {
        for (int side = 0; side < 2; ++side) {
            const int2 tap = pixel + axis * (side == 0 ? i : -i);
            if (any(tap < 0) || any(tap >= int2(size))) {
                continue;
            }

            const float tap_distance = ao_depth[tap];
            if (tap_distance == 0.0) {
                continue;
            }

            const float weight = weights[i] * depth_weight(centre_distance, tap_distance);
            sum += source[tap].w * weight;
            total += weight;
        }
    }

    target[pixel] = float4(centre.xyz, sum / max(total, 1e-4));
}

// --- 4. Back to full resolution ------------------------------------------------------

// Each full-resolution pixel sits among four half-resolution ones. Blending
// them by distance alone (bilinearly) would smear occlusion across edges, so
// each is also weighted by how close its depth is to this pixel's. Where
// none is close, as on a thin edge whose surface no half-resolution pixel
// kept, the closest in depth stands in alone.
[shader("compute")]
[numthreads(8, 8, 1)]
void upsampleMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= push.width || id.y >= push.height) {
        return;
    }

    FrameData* frame = push.frame;
    const int2 pixel = int2(id.xy);
    const int2 size = int2(half_size());
    const Texture2D depth_buffer = Texture2D.Handle(uint2(push.depth, 0));
    RWTexture2D<float> ao_depth = RWTexture2D<float>.Handle(uint2(push.ao_depth, 0));
    RWTexture2D<float4> source = RWTexture2D<float4>.Handle(uint2(push.source, 0));
    RWTexture2D<float4> target = RWTexture2D<float4>.Handle(uint2(push.target, 0));

    const float depth = depth_buffer.Load(int3(pixel, 0)).r;
    if (depth == 0.0) {
        target[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float here = dot(unproject(frame, float2(pixel) + 0.5, depth), camera_forward(frame));

    // The half-resolution pixel centres around this one, and how far along
    // between them it is.
    const float2 position = (float2(pixel) + 0.5) * 0.5 - 0.5;
    const int2 base = int2(floor(position));
    const float2 along = position - float2(base);

    float4 sum = 0.0;
    float total = 0.0;
    float4 closest = float4(0.0, 0.0, 0.0, 1.0);
    float closest_difference = 1e30;

    for (int i = 0; i < 4; ++i) {
        const int2 corner = int2(i & 1, i >> 1);
        const int2 neighbour = clamp(base + corner, int2(0), size - 1);
        const float other = ao_depth[neighbour];
        if (other == 0.0) {
            continue;
        }

        const float4 value = source[neighbour];
        const float2 bilinear = lerp(1.0 - along, along, float2(corner));
        const float weight = bilinear.x * bilinear.y * depth_weight(here, other);
        sum += float4(value.xyz * weight, value.w * weight);
        total += weight;

        if (abs(other - here) < closest_difference) {
            closest_difference = abs(other - here);
            closest = value;
        }
    }

    if (total < 1e-3) {
        target[pixel] = closest;
        return;
    }

    const float3 bent = dot(sum.xyz, sum.xyz) > 1e-8 ? normalize(sum.xyz) : closest.xyz;
    target[pixel] = float4(bent, sum.w / total);
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
// Each of those six inequalities is a plane in camera-relative space, read
// off M's rows (Gribb and Hartmann 2001): w - x >= 0 is
// (row 3 - row 0) . (p, 1) >= 0.
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
    CullTables* tables = push.tables;
    const uint position = id.x;

    if (position >= tables.draw_count) {
        return;
    }

    // The draw's box, from its cell to the camera: both corners in the same cell.
    FrameData* frame = push.frame;
    const DrawData draw = frame.draws[tables.order[position]];
    const float3 lo = camera_relative(frame, draw.cell, draw.bounds_min);
    const float3 hi = camera_relative(frame, draw.cell, draw.bounds_max);

    tables.visible[position] = in_view(frame.view_projection, lo, hi) ? 1 : 0;
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

`game-engine/shaders/background.slang`:
```slang
// Draws the sky behind the scene: one full-screen triangle at depth 0,
// infinitely far away, depth-tested so it only covers pixels nothing else
// has drawn on.

#include "shared.slangh"

// In a descriptor heap pipeline, the push_constant block is where push data lands.
[[vk::push_constant]]
ConstantBuffer<PushData> push;

// --- Vertex shader -----------------------------------------------------------

struct VertexOutput {
    float4 position : SV_Position;
    float2 clip : TEXCOORD0;  // this point's clip-space x and y
};

// The full-screen triangle of tonemap.slang, at depth 0: with reverse-Z and
// no far plane, that's infinitely far, so the depth test lets it through
// only where the depth buffer still holds its cleared 0. Its clip-space corners are also
// passed on: interpolated across the triangle, they arrive at each pixel as
// that pixel's own clip-space position.
[shader("vertex")]
VertexOutput vertexMain(uint vertex_id : SV_VulkanVertexID) {
    const float2 corner = float2((vertex_id << 1) & 2, vertex_id & 2) * 2.0 - 1.0;

    VertexOutput output;
    output.position = float4(corner, 0.0, 1.0);
    output.clip = corner;
    return output;
}

// --- Fragment shader ---------------------------------------------------------

// Each pixel looks along the ray from the camera through it. Turning the
// pixel's clip-space position at the near plane (depth 1) back into
// camera-relative space gives a point on that ray; the direction is from the
// camera, at the origin, to it. The far plane won't do: with none, depth 0
// is infinitely far, and w there is 0.
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    FrameData* frame = push.frame;

    const float4 near_point = mul(frame.inverse_view_projection, float4(input.clip, 1.0, 1.0));
    const float3 direction = normalize(near_point.xyz / near_point.w);

    const TextureCube sky = TextureCube.Handle(uint2(frame.sky_cube, 0));
    const SamplerState clamped = SamplerState.Handle(uint2(frame.clamp_sampler, 0));
    float3 radiance = sky.SampleLevel(clamped, direction, 0.0).rgb;

    // The sun's disk, which the sky cube leaves out: its illuminance spread
    // over the tiny solid angle it covers, pi r^2 for an angular radius r.
    if (dot(direction, frame.sun_direction) > cos(frame.sun_angular_radius)) {
        const float solid_angle = 3.14159265 * frame.sun_angular_radius * frame.sun_angular_radius;
        radiance += frame.sun_illuminance / solid_angle;
    }

    // Exposed like the scene, and kept below the 16-bit float limit: the
    // sun's disk is over a billion nits.
    return float4(min(radiance * frame.exposure, 60000.0), 1.0);
}
```

## 13.7 Placing everything: `pipeline.h`, `pipeline.cpp`, `main.cpp`

### Why
`main` places the scene, keeps the camera's cell, and moves the TLAS when the camera strays.

### How
- **`scene_origin`:** where in the world the scene goes, `{0, 0, 0}` by default. Set it to `{100000.0, 0.0, 100000.0}`, 141 km out, and nothing changes on screen.
- **The camera** starts at `scene_origin`.
- **The TLAS** is first built around the scene origin's cell.
- **Every frame:**
  - **The camera's cell and offset** come from `to_cell(camera.position)`, into `FrameData`.
  - **`tlas_offset`** is the camera's position, in doubles, minus the TLAS's origin cell's corner, then a float.
  - **Re-anchoring:** if the camera's cell is more than `tlas_reach_cells`, 16, from the TLAS's origin cell on any axis, `main` waits for the GPU to go idle and rebuilds the TLAS around the camera's cell. That's a brief pause about once per kilometre travelled; the terminal says `TLAS rebuilt around cell …`.
  - **`DrawData`** gets each draw's `cell`.
- **`pipeline.h`, `pipeline.cpp`:** comments only. Depth 0 is now infinitely far, not the far plane.

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
// With a `depth_format`, the triangle is depth-tested at depth 0, infinitely
// far, without writing depth, so it only reaches pixels nothing else has
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
    // value, and the buffer is cleared to 0, infinitely far.
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

// How far, in cells each way, the camera may stray from the TLAS's origin
// cell before the TLAS is rebuilt around it: 16 cells of 64 m, about 1 km.
// Within that, a ray near the camera is traced at coordinates under about
// 1.1 km, where floats are 0.06 to 0.12 mm apart.
constexpr int tlas_reach_cells = 16;

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
    AoTargets ao_targets;              // storage, for the AO pass
    std::uint32_t accum = 0;           // sampled, by the transparency composite
    std::uint32_t reveal = 0;          // sampled, by the transparency composite
};

constexpr std::uint32_t screen_slot_count = 11;

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

    // Reverse-Z: 0 is infinitely far. The depth is stored this time: the AO
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

    for (std::size_t i = 0; i < solid_modes.size(); ++i) {
        draw_mode(commands, culling, draws, solid_modes[i], pipelines.lighting[i]);
    }

    // The sky goes in once everything solid is drawn: it only covers pixels
    // still at depth 0, infinitely far.
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

        // Where in the world the scene is placed, in metres: its origin. Move
        // it far away, to {100000.0, 0.0, 100000.0} say, 141 km out, and the
        // image stays the same: everything is drawn relative to the camera.
        const glm::dvec3 scene_origin{0.0, 0.0, 0.0};

        const std::uint64_t load_start = SDL_GetTicksNS();
        const Scene scene = load_gltf(scene_file, scene_origin);
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
        // Around the cell the camera starts in: the scene's origin.
        AccelerationStructures acceleration = build_acceleration_structures(device, *gpu, queue, command_pool,
            scene, vertex_buffer, index_buffer, to_cell(scene_origin).cell);

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

            // Where the camera is: a cell and an offset, like everything the
            // GPU places. Once it's far from the TLAS's origin cell, the TLAS
            // is rebuilt around the camera's cell. That waits for the GPU to
            // stop using the old one: a brief pause, once per kilometre or so.
            const CellPosition camera_at = to_cell(camera.position);

            if (glm::any(glm::greaterThan(glm::abs(camera_at.cell - acceleration.origin_cell), glm::ivec3(tlas_reach_cells)))) {
                device.waitIdle();
                build_tlas(device, *gpu, queue, command_pool, acceleration, scene, camera_at.cell);
                std::println("TLAS rebuilt around cell ({}, {}, {})", camera_at.cell.x, camera_at.cell.y, camera_at.cell.z);
            }

            // The view-projection matrix works in camera-relative space: the
            // view only turns the world, the camera being at its origin.
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
                .camera_offset = camera_at.offset,
                .tlas_offset = glm::vec3(camera.position - glm::dvec3(acceleration.origin_cell) * cell_size),
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

## 13.8 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **Sponza** looks as it did in Chapter 12. The new projection and the camera-relative math round a few values differently: a handful of pixels along shadow edges change, nothing you'd see.
- **Fly away** from the courtyard, several kilometres. Sponza stays drawn, getting smaller, as far as you go: there's no far plane. Past about a kilometre, the terminal prints `TLAS rebuilt around cell …`, once per kilometre or so.
- **Move the scene far away:** set `scene_origin` to `{1024000.0, 0.0, 1024000.0}`, 1,448 km out. A multiple of 64 m puts every draw in the same place within its cell as at the origin, so the starting view is identical to the one at the origin, pixel for pixel: shadows, ambient occlusion and all. Fly around and it stays the same to the eye; the camera's doubles round in their last bits differently out there, which can move a value by a float's last bit. Any other distance looks the same too, but rounds a few values differently.
- **No `[validation …]` lines.**

Next, in Chapter 14, the air between the camera and what's far away: Hillaire's atmosphere, with lookup tables for the sky and aerial perspective, the haze that fades distant things toward the sky's color. Distant surfaces get their light through the atmosphere, too.

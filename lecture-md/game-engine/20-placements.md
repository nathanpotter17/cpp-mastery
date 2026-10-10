# Chapter 20: Two Million Placements

By the end of this chapter, the engine draws a world: two million **placements** of eight models, a game tree, a log, two boulders and four street props, scattered over the 8 km field of Chapter 18, each standing on the terrain at its own size and turn. A placement is 56 bytes: a cell, an offset, a rotation, a scale and a model. The draws of Chapter 12, each with two matrices and its own box, are gone; the shaders build a placement's matrices themselves, once per workgroup. A glTF node with a mesh is one placement, and a node with `EXT_mesh_gpu_instancing`, the extension that puts a mesh's instances in accessors like vertex attributes, is one per instance: the box field, Sponza and the dragon ring all load through the same path, and so does a file with two million instances.

The cull grows two levels at the top. Placements are sorted at load into the 64 m columns of the world's cells, and the walk starts from the cells: one thread per column, tested by its box against the view, the distance and the depth pyramid, and dropped as a whole when even its largest object is under a pixel. A kept cell's placements are the next level, tested by their models' spheres, and a kept placement's hierarchy roots the one after, as Chapter 19 left them. The clusters then come out cell by cell, every draw list mixed, so a counting sort by list follows with the same prefix sums, and each list's clusters are a run again.

Ray-traced shadows can't have two million instances. Each frame, a compute shader writes the placements within 512 m of the camera into a fixed-capacity instance buffer, zeroed first so that what it doesn't reach is inactive, and the frame's TLAS is built from it in the command buffer: a top-level structure per frame in flight, rebuilt around the camera every frame, 25,000 instances for under a millisecond. The window title gains the GPU's memory in use and the driver's budget, through `VK_EXT_memory_budget`, next to the counts of what the cull kept and dropped.

And one thing the world found that twelve dragons hadn't, with a scanned tree of a million triangles in the game tree's slot. Chapter 19's strict simplification, edge collapse with every border locked, has a floor: a group it can't reduce is marked with an infinite error and drawn at full detail from any distance. A tree modelled leaf by leaf, each leaf a little mesh of nothing but borders, keeps most of its 700,000 triangles from the far side of the field, whatever the error threshold says: 22 million triangles at eye level, and the same at 1, 2 or 4 pixels of error. meshoptimizer's sloppy fallback, vertex clustering for the groups edge collapse falls short on, takes it to 2.2 million, and lets the draw distance go. And the scan then shows a second thing, which no simplifier fixes: its leaves vanish past 20 m, because a geometric error can't see coverage. The game tree, 368 triangles of alpha-masked leaf cards, keeps its canopy to the horizon. That is why the world is built from the one and only tested against the other.

This chapter builds on [Chapter 19](19-clusters.md).

## 20.1 Models and placements: `scene.h`, `scene.cpp`, `world_scene.py`, `split_variant.py`

### Why
A `MeshDraw` held a model matrix, a normal matrix, a cell and a box: 180 bytes once uploaded, 360 MB for two million, and a walk that starts from every one of them. Everything a draw needs can be rebuilt from far less. A rotation is four numbers, a scale three, and a position a cell and an offset, as the lights have been since Chapter 13; the box is the model's, which every placement shares. So the scene now holds **models**, one per glTF mesh, and **placements** of them, and nothing per placement that isn't its own.

The file format already has this shape. [`EXT_mesh_gpu_instancing`](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/EXT_mesh_gpu_instancing) puts, on a node with a mesh, accessors of `TRANSLATION`, `ROTATION` and `SCALE`, one element per instance: the same buffers and views as any vertex attribute, so a file of two million instances is one node per model and 80 MB of floats. The spec fixes the transform's order: an instance's world matrix is the node's world matrix times its own translation, rotation and scale, in that order, and the extension applies only to nodes with a mesh.

### How
- **A primitive keeps its box** in its mesh's space, which `add_primitive` finds as it reads the positions: `Primitive` gains `bounds_min` and `bounds_max`, and the `LoadedPrimitive` that carried them to the draws goes.
- **A model** (`SceneModel`) is a glTF mesh with anything to draw: its primitives are a run in the scene's, and its box is around all of them. `mesh_models[m]` says which model mesh `m` became, or −1.
- **A placement** (`ScenePlacement`) is a position, a rotation, a scale and a model, in doubles like every world position. `placement_from` makes one from any node transform: the translation is the last column; the other three columns are the model's axes in the world, whose lengths are the scale, and which, divided by their lengths, are the rotation, if they're perpendicular to one another. A transform with a negative determinant mirrors; the mirror goes onto the x axis, so what's left is a proper rotation. A transform that shears can't be a placement, and the loader says so rather than draw it wrong: no exporter writes one.
- **`visit_node`** makes one placement of a node's mesh, or, with the extension, one per instance: the attributes come through `read_floats`, which already handles the normalized bytes and shorts the spec allows for `ROTATION`; an instance's matrix is built in the spec's order, and the node's world matrix multiplied on the left.
- **`placement_sphere`** gives a placement's sphere in the world: its model's box's sphere, moved and scaled by it. The scene's box grows by each one; the cells and the shadow instances (20.3, 20.4) use it too.
- **The world**, `world_scene.py`, is a script, since two million placements are nothing to type. It merges eight models into one `world/world.gltf`: each model's buffer views, accessors, images, textures, samplers, materials and mesh appended with their indices moved up, its `.bin` and textures referenced where they are by relative paths, so nothing is copied. Each model gets one node with the extension, whose three accessors point into `world/placements.bin`: a model chosen uniformly, a spot uniform over the field 50 m in from its edge and clear of the origin, a yaw, a scale between 0.8 and 1.25, and a height from the terrain by the engine's own rule (`terrain_height_at`: every quad split from its (0, 0) corner), less the model's base, so it stands on the ground. Two things it puts right in any model it's given: a material exported blended or masked whose base colour is a JPEG has no alpha to blend or mask with, so it's opaque, whatever the file says (the scanned tree's leaves are such); and a material that tiles its textures through `KHR_texture_transform`, which the loader doesn't read, has the transform baked into a copy of its primitives' texture coordinates in `world/baked.bin`, in the spec's order: scaled, turned, then offset. Every texture of a material shares it, so the result is exact. The script takes a count, a seed, a folder name and a model list, so the same placements can be built with different models in them: that's how the scan was measured against the game tree.
- **Two of the props** ship as two variants side by side, clean and rusted, in several part nodes each, and the tree as a trunk node with a leaves node under it. `split_variant.py` runs in Blender without a window, keeps the parts not named for the dropped variant, joins them into one mesh, stands it on the ground at the origin and exports it again: one mesh, so one model and one placement.
- **The models:** a game tree of the usual kind, `trees/trees.gltf`: a Blender-made trunk and two materials of leaf cards with alpha-masked PNG textures, 368 triangles in all, which the script joins into `game_tree/`; and from Poly Haven, CC0, `dead_tree_trunk_02`, `namaqualand_boulder_02`, `boulder_01`, `concrete_road_barrier`, `metal_trash_can`, `fire_hydrant` and Chapter 17's `utility_box_02`, each downloaded as **glTF, 1k** from [polyhaven.com/models](https://polyhaven.com/models) into `lecture-md/game-engine/assets/<name>/`: the `.gltf`, its `.bin`, and `textures/`. Together they're 364,000 triangles. For the experiment in 20.7, Poly Haven's `island_tree_02`, a scan of 1.07 million triangles, takes the tree's slot.

### Code
`game-engine/src/includes/scene.h`:
```cpp
#pragma once

#include "includes/shader_types.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <vector>

// One glTF primitive: a run of indices in the scene's index buffer, drawn against the vertices starting at `vertex_offset` in the vertex buffer, and the box around its vertices in its mesh's space.
struct Primitive {
    std::uint32_t first_index = 0;
    std::uint32_t index_count = 0;
    std::int32_t vertex_offset = 0;
    std::uint32_t material = 0;  // index into Scene::materials
    glm::vec3 bounds_min{0.0f};
    glm::vec3 bounds_max{0.0f};
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

// A model: one glTF mesh, whatever node draws it. Its primitives are a run in the scene's; the box is around all of them, in the mesh's own space.
struct SceneModel {
    std::uint32_t first_primitive = 0;
    std::uint32_t primitive_count = 0;
    glm::vec3 bounds_min{0.0f};
    glm::vec3 bounds_max{0.0f};
    std::string name;
};

// One thing to draw: a model placed in the world, in doubles. A node with a mesh is one placement; a node with EXT_mesh_gpu_instancing is one per instance. Each is a translation, a rotation and a scale along the model's axes: what the extension gives, and what any node transform that doesn't shear comes apart into. A negative scale mirrors the model, which reverses the order its triangles' corners appear in, and so which side is the front.
struct ScenePlacement {
    glm::dvec3 position{0.0};
    glm::dquat rotation{1.0, 0.0, 0.0, 0.0};
    glm::dvec3 scale{1.0};
    std::uint32_t model = 0;  // index into Scene::models
};

// Everything from a glTF file that drawing its geometry needs, flattened into arrays ready to upload: every primitive's vertices and indices back to back, the models they make up, and where the models are placed.
struct Scene {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<Primitive> primitives;
    std::vector<SceneModel> models;
    std::vector<ScenePlacement> placements;

    // The file's materials, plus a plain white one at the end for primitives that don't name a material.
    std::vector<SceneMaterial> materials;
    std::vector<SceneImage> images;
    std::vector<SceneSampler> samplers;

    // KHR_lights_punctual lights, placed in the world by their nodes.
    std::vector<Light> lights;

    // World-space box around everything drawn, roughly, in floats: around every placement's sphere. Where the test lights go.
    glm::vec3 bounds_min{std::numeric_limits<float>::max()};
    glm::vec3 bounds_max{std::numeric_limits<float>::lowest()};
};

// The sphere around a placement's model, in the world: the sphere around the model's box, moved and scaled by the placement.
struct PlacementSphere {
    glm::dvec3 center;
    double radius;
};

PlacementSphere placement_sphere(const Scene &scene, const ScenePlacement &placement);

// Loads the default scene of a .gltf or .glb file, with its materials, samplers, lights and images, still encoded. The scene is placed with its origin at `origin`, in metres from the world's.
Scene load_gltf(const std::filesystem::path &path, const glm::dvec3 &origin = glm::dvec3{0.0});

// Adds a .gltf or .glb file's default scene to `scene`, its materials, samplers, lights and images too, once for each of `placements`: a transform from the file's space to the world's, in metres. The file's meshes are loaded once, as models; each node that draws one, and each instance of a node with EXT_mesh_gpu_instancing, becomes a placement, for every one of `placements`.
void add_gltf(Scene &scene, const std::filesystem::path &path, std::span<const glm::dmat4> placements);

// Point and spot lights without a range in the file are given one: where their illuminance falls below light_threshold lux, at most max_light_range metres. 0.001 lux is at most a few 8-bit steps even at the exposure for the darkest scenes, and the falloff fades to it smoothly.
constexpr float light_threshold = 0.001f;
constexpr float max_light_range = 4096.0f;

// Adds the materials of a .gltf file to `scene`, with their textures and samplers, and returns the index of its first one: for surfaces that aren't in any file's scene, like the terrain's. The file needs no meshes or scenes.
std::uint32_t add_materials(Scene &scene, const std::filesystem::path &path);

// Adds `count` coloured point and spot lights to `scene`, spread through its box, the same ones every run: something to test many lights with. Every fourth is a spot shining down; every 64th has no range of its own.
void add_test_lights(Scene &scene, std::uint32_t count);
```

In `game-engine/src/scene.cpp`, remove these lines:
```cpp
    // Where a primitive landed in the scene, plus the box around its vertices in its own space, which visit_node() turns into each draw's box in its cell, and the scene's box.
    struct LoadedPrimitive {
        std::uint32_t index;
        glm::vec3 local_min;
        glm::vec3 local_max;
    };
```

In `game-engine/src/scene.cpp`, replace `add_primitive` with:
```cpp
    // Appends one primitive to the scene, and says whether it did. Points, lines, and primitives without positions are skipped, as glTF allows: this renderer draws triangles. The file's materials start at `first_material` among the scene's; `default_material` is used when the primitive doesn't name one.
    bool add_primitive(
        const tinygltf::Model &model, const tinygltf::Primitive &source, std::uint32_t first_material, std::uint32_t default_material, Scene &scene
    ) {
        const auto position = source.attributes.find("POSITION");
        const bool triangles = source.mode == TINYGLTF_MODE_TRIANGLES
            || source.mode == TINYGLTF_MODE_TRIANGLE_STRIP
            || source.mode == TINYGLTF_MODE_TRIANGLE_FAN;

        if (!triangles || position == source.attributes.end()) {
            return false;
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

        Primitive primitive{
            .first_index = static_cast<std::uint32_t>(scene.indices.size()),
            .index_count = static_cast<std::uint32_t>(indices.size()),
            .vertex_offset = static_cast<std::int32_t>(scene.vertices.size()),
            .material = source.material >= 0 ? first_material + static_cast<std::uint32_t>(source.material) : default_material,
            .bounds_min = glm::vec3{std::numeric_limits<float>::max()},
            .bounds_max = glm::vec3{std::numeric_limits<float>::lowest()},
        };

        for (std::size_t i = 0; i < positions.size(); ++i) {
            scene.vertices.push_back(Vertex{
                .position = positions[i],
                .normal = normals[i],
                .tangent = tangents[i],
                .uv0 = uv0s[i],
                .uv1 = uv1s[i],
                .color = colors[i],
            });
            primitive.bounds_min = glm::min(primitive.bounds_min, positions[i]);
            primitive.bounds_max = glm::max(primitive.bounds_max, positions[i]);
        }

        scene.indices.insert(scene.indices.end(), indices.begin(), indices.end());
        scene.primitives.push_back(primitive);
        return true;
    }
```

In `game-engine/src/scene.cpp`, add this section before `// Walks the node tree.`:
```cpp
    // A placement from a transform: its translation, and the rest taken apart into a rotation and a scale along the model's axes. The columns of the linear part are the model's axes in the world: their lengths are the scale, and divided by them they're the rotation, if they're perpendicular to one another. When they aren't, the transform shears, which a placement can't hold, so the loader refuses it.
    ScenePlacement placement_from(const glm::dmat4 &world, std::uint32_t model) {
        const glm::dmat3 linear(world);
        glm::dvec3 scale{glm::length(linear[0]), glm::length(linear[1]), glm::length(linear[2])};

        if (glm::any(glm::equal(scale, glm::dvec3{0.0}))) {
            throw std::runtime_error("a node is scaled to nothing");
        }

        // A negative determinant means the transform mirrors space. The mirror goes on the x axis, so what's left is a proper rotation.
        if (glm::determinant(linear) < 0.0) {
            scale.x = -scale.x;
        }

        const glm::dmat3 rotation(linear[0] / scale.x, linear[1] / scale.y, linear[2] / scale.z);

        for (int axis = 0; axis < 3; ++axis) {
            if (std::abs(glm::dot(rotation[axis], rotation[(axis + 1) % 3])) > 1e-4) {
                throw std::runtime_error("a node's transform shears, which a placement can't");
            }
        }

        return ScenePlacement{
            .position = glm::dvec3(world[3]),
            .rotation = glm::normalize(glm::quat_cast(rotation)),
            .scale = scale,
            .model = model,
        };
    }

    // The accessor an EXT_mesh_gpu_instancing attribute names, or -1 without it.
    int instance_accessor(const tinygltf::Value &attributes, const char *name) {
        return attributes.Has(name) ? attributes.Get(name).GetNumberAsInt() : -1;
    }
```

Then replace `visit_node` with:
```cpp
    // Walks the node tree. Each node's world transform is its parent's times its own; a node's mesh becomes one placement of its model, or, with EXT_mesh_gpu_instancing, one per instance, and a node's light is placed by the same transform. `mesh_models` says which model each mesh became, or -1 for a mesh with nothing to draw.
    void visit_node(
        const tinygltf::Model &model,
        int node_index,
        const glm::dmat4 &parent,
        const std::vector<std::int32_t> &mesh_models,
        Scene &scene
    ) {
        const tinygltf::Node &node = model.nodes.at(node_index);
        const glm::dmat4 world = parent * local_transform(node);

        if (node.mesh >= 0 && mesh_models.at(node.mesh) >= 0) {
            const auto model_index = static_cast<std::uint32_t>(mesh_models[node.mesh]);
            const auto instancing = node.extensions.find("EXT_mesh_gpu_instancing");

            if (instancing == node.extensions.end()) {
                scene.placements.push_back(placement_from(world, model_index));
            } else {
                // The node's mesh is drawn once per instance, each placed by its own translation, rotation and scale, which the extension stores in accessors like any vertex attribute. An instance's transform is the node's world transform times its own: the instances are in the node's space.
                const tinygltf::Value &attributes = instancing->second.Get("attributes");
                const int translation = instance_accessor(attributes, "TRANSLATION");
                const int rotation = instance_accessor(attributes, "ROTATION");
                const int scaling = instance_accessor(attributes, "SCALE");
                const int any = translation >= 0 ? translation : rotation >= 0 ? rotation : scaling;

                if (any < 0) {
                    throw std::runtime_error("an instanced node has no instance attributes");
                }

                const std::size_t count = model.accessors.at(any).count;
                const std::vector<float> translations = translation >= 0 ? read_floats(model, translation, 3) : std::vector<float>(count * 3, 0.0f);
                const std::vector<float> rotations = rotation >= 0 ? read_floats(model, rotation, 4) : std::vector<float>{};
                const std::vector<float> scales = scaling >= 0 ? read_floats(model, scaling, 3) : std::vector<float>(count * 3, 1.0f);

                if (translations.size() != count * 3 || scales.size() != count * 3 || (rotation >= 0 && rotations.size() != count * 4)) {
                    throw std::runtime_error("an instanced node's attributes have different counts");
                }

                for (std::size_t i = 0; i < count; ++i) {
                    // glTF stores (x, y, z, w); glm's constructor takes w first.
                    const glm::dquat q = rotation >= 0
                        ? glm::dquat(rotations[i * 4 + 3], rotations[i * 4], rotations[i * 4 + 1], rotations[i * 4 + 2])
                        : glm::dquat(1.0, 0.0, 0.0, 0.0);
                    const glm::dmat4 instance = glm::scale(
                        glm::translate(glm::dmat4{1.0}, glm::dvec3(glm::make_vec3(&translations[i * 3]))) * glm::mat4_cast(q),
                        glm::dvec3(glm::make_vec3(&scales[i * 3])));
                    scene.placements.push_back(placement_from(world * instance, model_index));
                }
            }
        }

        if (node.light >= 0) {
            if (const auto light = world_light(model.lights.at(node.light), world)) {
                scene.lights.push_back(*light);
            }
        }

        for (const int child : node.children) {
            visit_node(model, child, world, mesh_models, scene);
        }
    }
```

Then replace `add_gltf` with:
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

    // Every primitive of every mesh, once, and every mesh with any to draw as a model. mesh_models[m] is mesh m's model, or -1.
    std::vector<std::int32_t> mesh_models(model.meshes.size(), -1);

    for (std::size_t m = 0; m < model.meshes.size(); ++m) {
        SceneModel scene_model{
            .first_primitive = static_cast<std::uint32_t>(scene.primitives.size()),
            .bounds_min = glm::vec3{std::numeric_limits<float>::max()},
            .bounds_max = glm::vec3{std::numeric_limits<float>::lowest()},
            .name = model.meshes[m].name.empty() ? path.stem().string() + " mesh " + std::to_string(m) : model.meshes[m].name,
        };

        for (const tinygltf::Primitive &primitive : model.meshes[m].primitives) {
            if (add_primitive(model, primitive, first_material, default_material, scene)) {
                const Primitive &added = scene.primitives.back();
                scene_model.bounds_min = glm::min(scene_model.bounds_min, added.bounds_min);
                scene_model.bounds_max = glm::max(scene_model.bounds_max, added.bounds_max);
                ++scene_model.primitive_count;
            }
        }

        if (scene_model.primitive_count != 0) {
            mesh_models[m] = static_cast<std::int32_t>(scene.models.size());
            scene.models.push_back(scene_model);
        }
    }

    // The file may contain several scenes; draw its default one, once per placement.
    if (model.scenes.empty()) {
        throw std::runtime_error(path.string() + " has no scenes");
    }

    const tinygltf::Scene &root = model.scenes.at(model.defaultScene >= 0 ? model.defaultScene : 0);
    const std::size_t first_placement = scene.placements.size();

    for (const glm::dmat4 &placement : placements) {
        for (const int node : root.nodes) {
            visit_node(model, node, placement, mesh_models, scene);
        }
    }

    if (scene.placements.size() == first_placement) {
        throw std::runtime_error(path.string() + " has nothing to draw");
    }

    // The scene's box grows by each new placement's sphere.
    for (std::size_t i = first_placement; i < scene.placements.size(); ++i) {
        const PlacementSphere sphere = placement_sphere(scene, scene.placements[i]);
        scene.bounds_min = glm::min(scene.bounds_min, glm::vec3(sphere.center - sphere.radius));
        scene.bounds_max = glm::max(scene.bounds_max, glm::vec3(sphere.center + sphere.radius));
    }
}
```

In `game-engine/src/scene.cpp`, add this section before `Scene load_gltf(`:
```cpp
PlacementSphere placement_sphere(const Scene &scene, const ScenePlacement &placement) {
    const SceneModel &model = scene.models.at(placement.model);
    const glm::dvec3 center = (glm::dvec3(model.bounds_min) + glm::dvec3(model.bounds_max)) * 0.5;
    const double radius = glm::length(glm::dvec3(model.bounds_max) - glm::dvec3(model.bounds_min)) * 0.5;
    const glm::dvec3 stretch = glm::abs(placement.scale);

    return PlacementSphere{
        .center = placement.position + placement.rotation * (center * placement.scale),
        .radius = radius * std::max({stretch.x, stretch.y, stretch.z}),
    };
}
```

Save this as `lecture-md/game-engine/assets/split_variant.py`, and run it from that directory, once per prop:
```bash
blender -b --python split_variant.py -- metal_trash_can/metal_trash_can_1k.gltf metal_trash_can_clean rust
blender -b --python split_variant.py -- fire_hydrant/fire_hydrant_1k.gltf fire_hydrant_clean aged
blender -b --python split_variant.py -- trees/trees.gltf game_tree
```
```python
# Blender, headless: keeps one variant of a Poly Haven prop that ships two side by side, joins its parts into one mesh standing on the ground at the origin, and exports it as a separate .gltf/.bin with its textures.
# blender -b --python split_variant.py -- <in.gltf> <out_dir> <exclude-substring> [<exclude-substring>...]
import bpy, sys, os
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:]
source, out_dir, excludes = args[0], args[1], args[2:]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=source)

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
keep = [o for o in meshes if not any(x in o.name.lower() for x in excludes)]
drop = [o for o in meshes if o not in keep]
print('keep', [o.name for o in keep], 'drop', [o.name for o in drop])

bpy.ops.object.select_all(action='DESELECT')
for o in drop:
    o.select_set(True)
bpy.ops.object.delete()

for o in keep:
    o.select_set(True)
bpy.context.view_layer.objects.active = keep[0]
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
if len(keep) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

# Blender is Z-up: the base goes to z = 0, the footprint's centre to x = y = 0.
corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
obj.location -= Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
obj.name = os.path.basename(out_dir)
obj.data.name = obj.name
print('size', (hi - lo)[:], 'materials', [m.name for m in obj.data.materials])

os.makedirs(out_dir, exist_ok=True)
bpy.ops.export_scene.gltf(filepath=os.path.join(out_dir, obj.name + '.gltf'), export_format='GLTF_SEPARATE',
    export_texture_dir='textures', export_image_format='AUTO', export_yup=True, export_apply=True,
    export_normals=True, export_texcoords=True, export_tangents=False, export_materials='EXPORT', export_cameras=False, export_lights=False)
```

Save this as `lecture-md/game-engine/assets/world_scene.py`, and run it from that directory with `python3 world_scene.py`; for 20.7's experiment, once more with `python3 world_scene.py 2000000 20 world_scan island_tree_02,dead_tree_trunk_02,namaqualand_boulder_02,boulder_01,concrete_road_barrier,metal_trash_can_clean,fire_hydrant_clean,utility_box_02`:
```python
#!/usr/bin/env python3
"""Builds world/world.gltf next to this script: the models merged into one glTF, each drawn through EXT_mesh_gpu_instancing by a node whose
instances are written to world/placements.bin, uniformly at random over the terrain's 8 km field, standing on it.
    python3 world_scene.py [count] [seed] [out_dir] [model,model,...]
The defaults make the chapter's world; another folder name and model list make a variant with the same placements, for comparing models.
The models' own .bin files and textures are referenced where they are, by relative paths; nothing is copied."""
import json, sys, math, pathlib
import numpy as np

assets = pathlib.Path(__file__).resolve().parent
count = int(sys.argv[1]) if len(sys.argv) > 1 else 2000000
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20
out_dir = sys.argv[3] if len(sys.argv) > 3 else 'world'
models = sys.argv[4].split(',') if len(sys.argv) > 4 else ['game_tree', 'dead_tree_trunk_02', 'namaqualand_boulder_02', 'boulder_01',
          'concrete_road_barrier', 'metal_trash_can_clean', 'fire_hydrant_clean', 'utility_box_02']

# The terrain, sampled as the engine samples it (terrain.cpp's terrain_height_at): every quad split from its (0, 0) corner to its (1, 1) corner.
field = json.load(open(assets / 'terrain' / 'field.json'))
samples = field['samples']
heights = np.fromfile(assets / 'terrain' / field['heights'], dtype=np.uint16).reshape(samples, samples)  # [z, x]
heights = field['height_min'] + heights.astype(np.float64) * ((field['height_max'] - field['height_min']) / 65535.0)
origin = np.array(field['origin'], dtype=np.float64)
step = field['step']

def terrain_height(x, z):
    fx = (x - origin[0]) / step
    fz = (z - origin[1]) / step
    qx = np.floor(fx).astype(np.int64)
    qz = np.floor(fz).astype(np.int64)
    fx -= qx
    fz -= qz
    h00 = heights[qz, qx]
    h10 = heights[qz, qx + 1]
    h01 = heights[qz + 1, qx]
    h11 = heights[qz + 1, qx + 1]
    lower = h00 + (h10 - h00) * fx + (h11 - h10) * fz
    upper = h00 + (h01 - h00) * fz + (h11 - h01) * fx
    return np.where(fx >= fz, lower, upper)

# The merged file: each model's arrays appended, its indices moved up by what's there.
out = {'asset': {'version': '2.0', 'generator': 'world_scene.py'}, 'buffers': [], 'bufferViews': [], 'accessors': [],
       'images': [], 'textures': [], 'samplers': [], 'materials': [], 'meshes': [], 'nodes': [], 'scenes': [{'nodes': []}], 'scene': 0,
       'extensionsUsed': ['EXT_mesh_gpu_instancing'], 'extensionsRequired': ['EXT_mesh_gpu_instancing']}
base_y = []
baked = bytearray()  # texture coordinates rewritten for KHR_texture_transform: buffer len(models) + 1

for name in models:
    folder = assets / name
    source = next(p for p in folder.glob('*.gltf') if 'BoxField' not in p.name)
    g = json.load(open(source))
    assert len(g['meshes']) == 1 and len(g['nodes']) == 1, name
    node = g['nodes'][0]
    assert not any(k in node for k in ('translation', 'rotation', 'scale', 'matrix')), name + ' has a node transform'

    base = {k: len(out[k]) for k in ('buffers', 'bufferViews', 'accessors', 'images', 'textures', 'samplers', 'materials')}

    for b in g['buffers']:
        out['buffers'].append({'uri': f'../{name}/{b["uri"]}', 'byteLength': b['byteLength']})
    for v in g['bufferViews']:
        v = dict(v); v['buffer'] += base['buffers']; out['bufferViews'].append(v)
    for a in g['accessors']:
        a = dict(a)
        if 'bufferView' in a: a['bufferView'] += base['bufferViews']
        out['accessors'].append(a)
    for im in g.get('images', []):
        out['images'].append({'uri': f'../{name}/{im["uri"]}', 'name': im.get('name', '')})
    for s in g.get('samplers', []):
        out['samplers'].append(dict(s))
    for t in g.get('textures', []):
        t = dict(t)
        if 'source' in t: t['source'] += base['images']
        if 'sampler' in t: t['sampler'] += base['samplers']
        out['textures'].append(t)
    transforms = []  # per material of this model: its texture transform, or None
    for m in g.get('materials', []):
        m = json.loads(json.dumps(m))
        pbr = m.get('pbrMetallicRoughness', {})
        # A material exported blended or masked whose base colour is a JPEG has no alpha to blend or mask with: it's opaque, whatever the file says.
        colour = pbr.get('baseColorTexture')
        if m.get('alphaMode', 'OPAQUE') != 'OPAQUE' and colour is not None:
            image = g['images'][g['textures'][colour['index']]['source']]
            if image.get('uri', '').lower().endswith(('.jpg', '.jpeg')):
                m['alphaMode'] = 'OPAQUE'
                m.pop('alphaCutoff', None)
        for key in ('normalTexture', 'occlusionTexture', 'emissiveTexture'):
            if key in m: m[key]['index'] += base['textures']
        for key in ('baseColorTexture', 'metallicRoughnessTexture'):
            if key in pbr: pbr[key]['index'] += base['textures']
        # KHR_texture_transform: a material's textures tiled and offset over the mesh. Every texture of a material here shares one transform, so it's baked into the texture coordinates below, and the extension dropped.
        transform = None
        for slot in [m.get('normalTexture'), m.get('occlusionTexture'), m.get('emissiveTexture'), pbr.get('baseColorTexture'), pbr.get('metallicRoughnessTexture')]:
            if slot and 'extensions' in slot and 'KHR_texture_transform' in slot['extensions']:
                t = slot['extensions'].pop('KHR_texture_transform')
                assert transform is None or transform == t, 'textures of one material with different transforms'
                transform = t
                if not slot['extensions']: del slot['extensions']
        transforms.append(transform)
        out['materials'].append(m)

    mesh = json.loads(json.dumps(g['meshes'][0]))
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    for p in mesh['primitives']:
        # A primitive whose material had a texture transform gets its texture coordinates transformed, into the baked buffer, in the extension's order: scaled, then turned, then offset. The turn's sign is the one the exporters and viewers agree on (the spec's example, not its matrix).
        t = transforms[p['material']] if 'material' in p else None
        if t is not None:
            acc = g['accessors'][p['attributes']['TEXCOORD_0']]
            view = g['bufferViews'][acc['bufferView']]
            assert acc['componentType'] == 5126 and acc['type'] == 'VEC2' and view.get('byteStride', 8) == 8
            data = (folder / g['buffers'][view['buffer']]['uri']).read_bytes()
            start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
            uv = np.frombuffer(data[start:start + acc['count'] * 8], dtype=np.float32).reshape(-1, 2).astype(np.float64)
            r = t.get('rotation', 0.0); c, s_ = math.cos(r), math.sin(r)
            uv = uv * np.array(t.get('scale', [1, 1]))
            uv = np.stack([c * uv[:, 0] + s_ * uv[:, 1], -s_ * uv[:, 0] + c * uv[:, 1]], axis=1) + np.array(t.get('offset', [0, 0]))
            uv = uv.astype(np.float32)
            out['bufferViews'].append({'buffer': len(models) + 1, 'byteOffset': len(baked), 'byteLength': uv.nbytes})
            baked += uv.tobytes()
            out['accessors'].append({'bufferView': len(out['bufferViews']) - 1, 'componentType': 5126, 'count': acc['count'], 'type': 'VEC2'})
            p['attributes']['TEXCOORD_0'] = len(out['accessors']) - 1
            print(f'  {name}: texture transform scale {t.get("scale")} offset {t.get("offset")} baked into {acc["count"]} texture coordinates')
        for k, v in p['attributes'].items():
            if not (t is not None and k == 'TEXCOORD_0'):
                p['attributes'][k] = v + base['accessors']
        if 'indices' in p: p['indices'] += base['accessors']
        if 'material' in p: p['material'] += base['materials']
        pos = out['accessors'][p['attributes']['POSITION']]
        lo = np.minimum(lo, pos['min']); hi = np.maximum(hi, pos['max'])
    mesh['name'] = name
    out['meshes'].append(mesh)
    base_y.append(lo[1])
    for ext in g.get('extensionsUsed', []):
        if ext not in out['extensionsUsed']: out['extensionsUsed'].append(ext)
    print(f'{name}: {len(mesh["primitives"])} primitives, size {np.round(hi - lo, 2).tolist()}, base y {lo[1]:.3f}')

# The placements: a model, a spot on the field 50 m in from its edge and 12 m clear of the origin, a yaw, a size; the model's base on the terrain.
rng = np.random.default_rng(seed)
model = rng.integers(len(models), size=count)
half = (samples - 1) * step / 2 - 50.0
x = rng.uniform(-half, half, count)
z = rng.uniform(-half, half, count)
near = np.hypot(x, z) < 12.0
x[near] += np.sign(x[near] + 1e-9) * 12.0
yaw = rng.uniform(0.0, 2.0 * math.pi, count)
scale = np.exp(rng.uniform(math.log(0.8), math.log(1.25), count))
y = terrain_height(x, z) - np.array(base_y)[model] * scale

placements = pathlib.Path(assets / out_dir / 'placements.bin')
placements.parent.mkdir(exist_ok=True)
blob = bytearray()
for m, name in enumerate(models):
    mask = model == m
    n = int(mask.sum())
    translation = np.stack([x[mask], y[mask], z[mask]], axis=1).astype(np.float32)
    rotation = np.stack([np.zeros(n), np.sin(yaw[mask] / 2), np.zeros(n), np.cos(yaw[mask] / 2)], axis=1).astype(np.float32)
    scaling = np.repeat(scale[mask].astype(np.float32)[:, None], 3, axis=1)
    attributes = {}
    for key, array, kind in (('TRANSLATION', translation, 'VEC3'), ('ROTATION', rotation, 'VEC4'), ('SCALE', scaling, 'VEC3')):
        out['bufferViews'].append({'buffer': len(models), 'byteOffset': len(blob), 'byteLength': array.nbytes})
        accessor = {'bufferView': len(out['bufferViews']) - 1, 'componentType': 5126, 'count': n, 'type': kind}
        if key == 'TRANSLATION':
            accessor['min'] = array.min(axis=0).tolist(); accessor['max'] = array.max(axis=0).tolist()
        out['accessors'].append(accessor)
        attributes[key] = len(out['accessors']) - 1
        blob += array.tobytes()
    out['nodes'].append({'mesh': m, 'name': name, 'extensions': {'EXT_mesh_gpu_instancing': {'attributes': attributes}}})
    out['scenes'][0]['nodes'].append(m)

out['buffers'].append({'uri': 'placements.bin', 'byteLength': len(blob)})
placements.write_bytes(blob)
# A glTF buffer can't be empty: the baked one exists only when some material had a transform.
if baked:
    out['buffers'].append({'uri': 'baked.bin', 'byteLength': len(baked)})
    (assets / out_dir / 'baked.bin').write_bytes(baked)
out['extensionsUsed'] = [e for e in out['extensionsUsed'] if e != 'KHR_texture_transform']
json.dump(out, open(assets / out_dir / 'world.gltf', 'w'), indent=1)
print(f'{count} placements of {len(models)} models: {len(blob) / 1e6:.1f} MB; heights {y.min():.1f} to {y.max():.1f} m')
```

## 20.2 The data: `shader_types.h`, `shared.slangh`

### Why
What the GPU sees of a placement, a model, a primitive and a column of cells; the items the walk carries, which now name a primitive as well; and the instances the shadow TLAS is built from.

### How
- **`Placement`** (56): the cell, the offset in it, the rotation as a unit quaternion in glTF's (x, y, z, w), the scale along the model's axes, and the model. A negative scale mirrors.
- **`Model`** (32): the sphere around the model's primitives in its own space, where its primitives are, and the address of its BLAS (20.4): it's 8-byte aligned, so it goes last.
- **`PrimitiveData`** (12): the material, and where the primitive's indices and vertices start, which a shadow ray's alpha test needs.
- **`PlacementCell`** (48): a column of the world's cells: the cell it's measured from, the box around every placement in it, the largest placement's sphere, and the run of placements, which are sorted by column.
- **`ClusterItem`** stays 8 bytes, repacked: the placement, then a word with the primitive in its top 8 bits, the kind in the 2 below, and 22 bits of index, a node's or a cluster's, up to four million. Four kinds now: a cell (its index in the first word), a placement, a node, a cluster.
- **`WalkCounters`** gains the placements dropped as too far, the items walked per level, and the items a level had no room for.
- **`CullTables`** names the cells instead of the draws and their order, and the sorted clusters next to the unsorted.
- **`FrameData`** points at the placements, the models, the primitives and the cells, and carries the draw distance, the shadow radius and capacity, and the placement count.
- **`ShadowPushData`**, for `shadows.slang`: the frame, the instances, and their count.
- **In Slang,** the same structs, a `RayInstance` laid out like `VkAccelerationStructureInstanceKHR`, and an **`Instance`**: a placement ready to transform with. Its linear part is the rotation times the scale, R × S, which scales each column of R by the scale along that axis; its inverse is S⁻¹ × Rᵀ, each row of Rᵀ divided by the same, and the inverse's transpose is the normal matrix. `rotation_matrix` turns the quaternion into R, `instance_of` builds all of it from the placement and the camera, and `instance_point`, `instance_normal`, `instance_scale` and `instance_mirrored` are what the shaders ask of it.

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
};

static_assert(sizeof(FrameData) == 472);
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
    std::uint32_t list_counts[12];  // cluster items per draw list
    std::uint32_t level_clusters[16];  // cluster items per DAG level
    std::uint32_t far_placements;   // placements past mesh_draw_distance, whole cells of them included
    std::uint32_t level_items[16];  // items walked per level of the walk
    std::uint32_t dropped_items;    // items a level, the candidates or the clusters had no room for
};

static_assert(sizeof(WalkCounters) == 220);
static_assert(offsetof(WalkCounters, far_placements) == 148);
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

## 20.3 Placements on the GPU: `placements.h`, `placements.cpp`

### Why
Two million placements can't each be a thread of every frame's cull, or rather can, at a cost that grows with the world and not with the view. Sorted into the columns of the world's cells, 64 m on a side, they're 16,384 columns on the 8 km field, and the cull walks those: a column behind the camera, hidden, too far, or whose largest object is under a pixel drops every placement in it at once. A frame then touches only the placements of the columns it keeps.

### How
- **The sort** is by the column a placement's origin falls in, x before z, and stable, so placements that share a column keep the file's order: the GPU's tables are the same every run.
- **A cell record** starts at the first placement of each column, at that placement's cell, and grows by each placement's sphere, boxed, measured from that cell's corner: subtracted in doubles, so the floats hold only the small remainder. Its `max_radius` is the largest sphere, for the sub-pixel test.
- **The models and the primitives** are uploaded here too, the models with their BLAS addresses, so this comes after the acceleration structures.
- **A limit:** the shadow TLAS names a placement by an instance's custom index, 24 bits, so sixteen million placements is the most this world can have.

### Code
`game-engine/src/includes/placements.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/scene.h"
#include "includes/shader_types.h"
#include "includes/vulkan_setup.h"

#include <cstdint>
#include <span>

// Placements on the GPU

// The scene's placements, sorted into columns of world cells (cells.h) and uploaded with their cells, their models and the models' primitives. The cull walks the cells first: a column out of view, hidden, too far or under a pixel drops every placement in it at once, so a frame never looks at most of the world's placements one by one.
struct PlacementBuffers {
    Buffer placements;  // one Placement per placement, in cell order
    Buffer cells;       // one PlacementCell per column that has any
    Buffer models;      // one Model per model
    Buffer primitives;  // one PrimitiveData per primitive
    std::uint32_t placement_count = 0;
    std::uint32_t cell_count = 0;
};

// Sorts `scene.placements` by column, in place, and uploads everything. `blas_addresses` has one per model (acceleration.h).
PlacementBuffers upload_placements(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    Scene &scene,
    std::span<const vk::DeviceAddress> blas_addresses
);
```

`game-engine/src/placements.cpp`:
```cpp
#include "includes/placements.h"

#include "includes/cells.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

    // The column a placement's origin is in: its cell's x and z. The sort key, x before z.
    std::pair<int, int> column_of(const ScenePlacement &placement) {
        const glm::ivec3 cell = to_cell(placement.position).cell;
        return {cell.x, cell.z};
    }

}  // namespace

PlacementBuffers upload_placements(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    Scene &scene,
    std::span<const vk::DeviceAddress> blas_addresses
) {
    // A ray-tracing instance's custom index, which names the placement, has 24 bits.
    if (scene.placements.size() >= (1u << 24)) {
        throw std::runtime_error("more than 16 million placements");
    }

    // Stable: placements that share a column keep the file's order, so the GPU's placements are the same every run.
    std::ranges::stable_sort(scene.placements, {}, column_of);

    std::vector<Placement> placements;
    std::vector<PlacementCell> cells;
    placements.reserve(scene.placements.size());

    for (std::size_t i = 0; i < scene.placements.size(); ++i) {
        const ScenePlacement &placement = scene.placements[i];
        const CellPosition at = to_cell(placement.position);

        placements.push_back(Placement{
            .cell = at.cell,
            .offset = at.offset,
            .rotation = glm::vec4(placement.rotation.x, placement.rotation.y, placement.rotation.z, placement.rotation.w),
            .scale = glm::vec3(placement.scale),
            .model = placement.model,
        });

        // A new column starts its cell record at its first placement's cell; the rest of the run goes into the same record.
        if (cells.empty() || column_of(scene.placements[cells.back().first]) != column_of(placement)) {
            cells.push_back(PlacementCell{
                .cell = at.cell,
                .bounds_min = glm::vec3{std::numeric_limits<float>::max()},
                .bounds_max = glm::vec3{std::numeric_limits<float>::lowest()},
                .max_radius = 0.0f,
                .first = static_cast<std::uint32_t>(i),
                .count = 0,
            });
        }

        // The placement's sphere, boxed, measured from the cell's corner: subtracted in doubles, so the floats only hold the small remainder.
        PlacementCell &cell = cells.back();
        const PlacementSphere sphere = placement_sphere(scene, placement);
        const glm::dvec3 corner = glm::dvec3(cell.cell) * cell_size;

        cell.bounds_min = glm::min(cell.bounds_min, glm::vec3(sphere.center - sphere.radius - corner));
        cell.bounds_max = glm::max(cell.bounds_max, glm::vec3(sphere.center + sphere.radius - corner));
        cell.max_radius = std::max(cell.max_radius, static_cast<float>(sphere.radius));
        ++cell.count;
    }

    // The models: the sphere around each one's box, and its BLAS.
    std::vector<Model> models;

    for (std::size_t m = 0; m < scene.models.size(); ++m) {
        const SceneModel &model = scene.models[m];
        models.push_back(Model{
            .center = (model.bounds_min + model.bounds_max) * 0.5f,
            .radius = glm::length(model.bounds_max - model.bounds_min) * 0.5f,
            .first_primitive = model.first_primitive,
            .primitive_count = model.primitive_count,
            .blas = blas_addresses[m],
        });
    }

    std::vector<PrimitiveData> primitives;

    for (const Primitive &primitive : scene.primitives) {
        primitives.push_back(PrimitiveData{
            .material = primitive.material,
            .first_index = primitive.first_index,
            .vertex_offset = primitive.vertex_offset,
        });
    }

    const auto upload = [&](std::span<const std::byte> bytes) {
        return upload_buffer(device, gpu, queue, pool, bytes, vk::BufferUsageFlagBits::eShaderDeviceAddress);
    };

    return PlacementBuffers{
        .placements = upload(std::as_bytes(std::span(placements))),
        .cells = upload(std::as_bytes(std::span(cells))),
        .models = upload(std::as_bytes(std::span(models))),
        .primitives = upload(std::as_bytes(std::span(primitives))),
        .placement_count = static_cast<std::uint32_t>(placements.size()),
        .cell_count = static_cast<std::uint32_t>(cells.size()),
    };
}
```

## 20.4 Shadows near the camera: `acceleration.h`, `acceleration.cpp`, `shadows.slang`, `vulkan_setup.cpp`

### Why
A TLAS over two million instances is 128 MB of instance data and a build of many milliseconds, every time the camera crosses a kilometre; and rays from the camera's surroundings never reach most of it. Objects beyond a few hundred metres cast shadows a pixel across. So the TLAS holds what's near: each frame, the placements within `shadow_radius` of the camera, and nothing else. The terrain's shadow still comes from the height-field march of Chapter 18, which reaches 4 km; a distant object just doesn't cast one.

The instances have to be chosen on the GPU, where the placements are, and a build needs an instance count. The GPU-side count would be `vkCmdBuildAccelerationStructuresIndirectKHR`, which reads it from a buffer; the driver this chapter was written on doesn't offer that feature. The other way is the spec's inactive instance: an instance whose acceleration structure reference is zero is skipped by the build. The buffer is filled with zeros, the shader writes the instances it finds, and the build runs over the whole capacity: the slots it didn't reach cost the build only their read, 64 bytes each.

### How
- **One BLAS per model,** with a geometry per primitive, instead of one per primitive: a hit then reports the instance's custom index, which is the placement, and its geometry index, which is the primitive within the model. The build sizes are asked for with one triangle count per geometry; the ranges are the model's primitives' run.
- **One `ShadowTlas` per frame in flight:** the instance buffer, `shadow_instance_capacity` × 64 bytes, a count, the structure's storage, its scratch, and its address. The sizes come from the capacity, whatever the instances turn out to be, and the structure is rebuilt from scratch each frame: an update would need the same instances in the same slots.
- **`writeInstancesMain`** (`shadows.slang`), a thread per placement: the model's sphere under the placement; if its nearest point is within the radius, a slot from the atomic count, and, under the capacity, the instance: the linear part's three rows, each ending in the placement's origin in the TLAS's space; the placement as the custom index under a mask of 0xFF; the facing-cull-disable flag, as before; and the model's BLAS. The TLAS's space is the camera's cell's, from its corner: the camera-relative origin plus the camera's offset. The order the slots fill in changes from frame to frame, which a TLAS doesn't mind.
- **`record_shadow_tlas`,** every frame: both buffers filled with zeros, the writer, a barrier to the build, which the spec puts under `VK_ACCESS_2_SHADER_READ_BIT` at the build stage for its inputs, the build over the whole capacity, a barrier to the fragment shaders, and the count copied out for the title. The frame's last use of these buffers, two frames ago, is over: the CPU waited for its fence.
- **The capacity,** 65,536: at one placement per 32 m², 512 m needs about 25,000. A placement past the capacity goes without a shadow; the title's count says how close it is.
- **`VK_EXT_memory_budget`** is enabled with the device extensions: no feature, just the properties `main.cpp` reads for the title.

### Code
`game-engine/src/includes/acceleration.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/scene.h"
#include "includes/vulkan_setup.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Acceleration structures

// The scene organized for tracing rays, in two levels:
//   - one bottom-level acceleration structure (BLAS) per model: a tree of boxes around its triangles, a geometry per primitive, in the model's own space,
//   - one top-level acceleration structure (TLAS) per frame in flight, rebuilt every frame over the placements within shadow_radius of the camera: each an instance of its model's BLAS, placed by the placement's transform.
// A ray first finds which instances' boxes it crosses, then which triangles, and reports the instance's custom index, the placement, and the geometry's index, the primitive within the model. Members are destroyed bottom-up, so the acceleration structures go before the buffers that hold them.
//
// The TLAS holds floats, like everything the GPU sees, so its space is measured from the corner of the camera's world cell (cells.h): a ray near the camera is traced at small coordinates, with a float's full precision. The GPU writes the instances itself (shaders/shadows.slang) into a buffer of fixed capacity; slots it doesn't reach stay zero, which the build takes as inactive instances. Placements beyond the radius cast no ray-traced shadow; the terrain's shadow comes from its own march (shading.slangh), which reaches further.
struct ShadowTlas {
    Buffer instances;  // shadow_instance_capacity VkAccelerationStructureInstanceKHR, written by the GPU each frame
    Buffer count;      // one uint32_t: how many it wrote
    Buffer storage;
    Buffer scratch;
    vk::raii::AccelerationStructureKHR tlas = nullptr;
    vk::DeviceAddress address = 0;  // what shaders trace against
};

struct AccelerationStructures {
    Buffer blas_storage;  // every BLAS, one after another
    std::vector<vk::raii::AccelerationStructureKHR> blases;  // one per model
    std::vector<vk::DeviceAddress> blas_addresses;
    vk::raii::Pipeline write_instances = nullptr;  // shaders/shadows.slang: the placements within reach, as instances
    std::vector<ShadowTlas> frames;  // one per frame in flight
    vk::DeviceSize scratch_alignment = 0;  // what the GPU requires a build's scratch address to be a multiple of
};

// Threads per workgroup of shaders/shadows.slang's writeInstancesMain, one placement each.
constexpr std::uint32_t shadow_workgroup_size = 256;

// The most instances a frame's TLAS holds: 65,536. At this world's density, one placement per 32 m², a radius of 512 m needs about 25,000; beyond the capacity, placements go without shadows.
constexpr std::uint32_t shadow_instance_capacity = 65536;

// Builds a BLAS for every model on the GPU, and waits for them; makes each frame's TLAS, its instance buffers and scratch, ready to be built every frame. The triangles are read straight from the scene's vertex and index buffers, which must have been created with eAccelerationStructureBuildInputReadOnlyKHR.
AccelerationStructures build_acceleration_structures(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    const Scene &scene,
    const Buffer &vertex_buffer,
    const Buffer &index_buffer,
    std::size_t frames_in_flight
);

// Records frame `frame_index`'s TLAS: its instances written from the placements within shadow_radius of the camera, for the frame whose FrameData is at `frame`, then the build. The frame's previous use of these buffers must be over. Afterwards fragment shaders may trace against it, and the instance count is in `readback` at `readback_offset`, for the CPU once the frame is done.
void record_shadow_tlas(
    const vk::raii::CommandBuffer &commands,
    const AccelerationStructures &structures,
    std::size_t frame_index,
    vk::DeviceAddress frame,
    std::uint32_t placement_count,
    vk::Buffer readback,
    vk::DeviceSize readback_offset
);
```

`game-engine/src/acceleration.cpp`:
```cpp
#include "includes/acceleration.h"

#include "includes/cells.h"
#include "includes/pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace {

    vk::DeviceSize align_up(vk::DeviceSize value, vk::DeviceSize alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    // Acceleration structures live inside buffers, at offsets that are multiples of 256 bytes: the Vulkan spec's rule.
    constexpr vk::DeviceSize storage_alignment = 256;

    // What the GPU requires a build's scratch address to be a multiple of.
    vk::DeviceSize scratch_alignment(const GpuChoice &gpu) {
        const auto properties = gpu.device.getProperties2<
            vk::PhysicalDeviceProperties2,
            vk::PhysicalDeviceAccelerationStructurePropertiesKHR
        >();
        return properties.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>().minAccelerationStructureScratchOffsetAlignment;
    }

    // A buffer acceleration structures are stored in.
    Buffer create_storage(const vk::raii::Device &device, const GpuChoice &gpu, vk::DeviceSize size) {
        return create_buffer(device, gpu, size,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
    }

    // Scratch memory: working space for a build, needed only while it runs. Its address must be a multiple of `alignment`, so the buffer is a little larger and the returned address rounded up within it.
    Buffer create_scratch(const vk::raii::Device &device, const GpuChoice &gpu, vk::DeviceSize size, vk::DeviceSize alignment) {
        return create_buffer(device, gpu, size + alignment,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
    }

}  // namespace

// Building

AccelerationStructures build_acceleration_structures(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    const Scene &scene,
    const Buffer &vertex_buffer,
    const Buffer &index_buffer,
    std::size_t frames_in_flight
) {
    const vk::DeviceSize alignment = scratch_alignment(gpu);

    AccelerationStructures structures;
    structures.scratch_alignment = alignment;

    // One BLAS per model

    // Each primitive's triangles, described where they already are: its run of indices in the index buffer, and its vertices in the vertex buffer. Only the position, the first 12 bytes of each Vertex, is read. A model's primitives are its BLAS's geometries, in order, so a hit's geometry index is the primitive within the model.
    const std::size_t count = scene.models.size();
    std::vector<vk::AccelerationStructureGeometryKHR> geometries(scene.primitives.size());
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges(scene.primitives.size());
    std::vector<vk::AccelerationStructureBuildGeometryInfoKHR> builds(count);
    std::vector<vk::AccelerationStructureBuildSizesInfoKHR> sizes(count);
    std::vector<vk::DeviceSize> storage_offsets(count);
    std::vector<vk::DeviceSize> scratch_offsets(count);

    vk::DeviceSize storage_size = 0;
    vk::DeviceSize scratch_size = 0;

    for (std::size_t p = 0; p < scene.primitives.size(); ++p) {
        const Primitive &primitive = scene.primitives[p];

        // The highest vertex an index refers to, which the build must know.
        const auto first = scene.indices.begin() + primitive.first_index;
        const std::uint32_t max_vertex = *std::max_element(first, first + primitive.index_count);

        // Opaque triangles let a ray stop at the first one it hits. The others are reported to the shader, which checks their alpha: a ray passes through a leaf's empty corners (masked), or is dimmed by a see-through layer (blended). A blended triangle must be reported only once per ray, or a ray would be dimmed by it twice; without the flag, the GPU may report one more than once.
        const AlphaMode alpha_mode = scene.materials[primitive.material].alpha_mode;
        const vk::GeometryFlagsKHR geometry_flags = alpha_mode == AlphaMode::opaque ? vk::GeometryFlagBitsKHR::eOpaque
            : alpha_mode == AlphaMode::blend ? vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation
            : vk::GeometryFlagsKHR{};

        geometries[p] = vk::AccelerationStructureGeometryKHR{
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

        ranges[p] = vk::AccelerationStructureBuildRangeInfoKHR{.primitiveCount = primitive.index_count / 3};
    }

    for (std::size_t m = 0; m < count; ++m) {
        const SceneModel &model = scene.models[m];

        // Built once, traced every frame: optimize for tracing.
        builds[m] = vk::AccelerationStructureBuildGeometryInfoKHR{
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = model.primitive_count,
            .pGeometries = &geometries[model.first_primitive],
        };

        // The driver says how much memory the structure and its build need, given each geometry's triangle count.
        std::vector<std::uint32_t> triangle_counts;
        for (std::uint32_t p = 0; p < model.primitive_count; ++p) {
            triangle_counts.push_back(ranges[model.first_primitive + p].primitiveCount);
        }

        sizes[m] = device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, builds[m], triangle_counts);

        storage_offsets[m] = storage_size;
        storage_size = align_up(storage_size + sizes[m].accelerationStructureSize, storage_alignment);
        scratch_offsets[m] = scratch_size;
        scratch_size = align_up(scratch_size + sizes[m].buildScratchSize, alignment);
    }

    // One buffer holds every BLAS, and one scratch buffer every build's working space, so all of them can be built at once.
    structures.blas_storage = create_storage(device, gpu, storage_size);
    const Buffer blas_scratch = create_scratch(device, gpu, scratch_size, alignment);
    const vk::DeviceAddress blas_scratch_address = align_up(blas_scratch.address, alignment);

    for (std::size_t m = 0; m < count; ++m) {
        structures.blases.emplace_back(device, vk::AccelerationStructureCreateInfoKHR{
            .buffer = *structures.blas_storage.handle,
            .offset = storage_offsets[m],
            .size = sizes[m].accelerationStructureSize,
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        });

        builds[m].dstAccelerationStructure = *structures.blases[m];
        builds[m].scratchData.deviceAddress = blas_scratch_address + scratch_offsets[m];
        structures.blas_addresses.push_back(device.getAccelerationStructureAddressKHR(
            vk::AccelerationStructureDeviceAddressInfoKHR{.accelerationStructure = *structures.blases[m]}));
    }

    // Building, on the GPU

    // Every BLAS in one call: each build's ranges are its model's primitives'. The barrier makes them visible to the TLAS builds, which read them every frame.
    std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> range_pointers;
    for (std::size_t m = 0; m < count; ++m) {
        range_pointers.push_back(&ranges[scene.models[m].first_primitive]);
    }

    submit_and_wait(device, queue, pool, [&](const vk::raii::CommandBuffer &commands) {
        commands.buildAccelerationStructuresKHR(builds, range_pointers);

        const vk::MemoryBarrier2 blases_built{
            .srcStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            .srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            .dstStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR,
        };
        commands.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &blases_built});
    });

    // The frames' TLASes

    // Each frame in flight gets its own: the GPU may still be tracing against one frame's while the next frame's is written and built. Their size comes from the capacity, whatever the instances turn out to be, and they're rebuilt from scratch every frame: an update would need the same instances in the same slots, which a culled set never has.
    structures.write_instances = create_compute_pipeline(device, "shadows", "writeInstancesMain");

    for (std::size_t f = 0; f < frames_in_flight; ++f) {
        ShadowTlas frame;
        frame.instances = create_buffer(device, gpu, shadow_instance_capacity * sizeof(vk::AccelerationStructureInstanceKHR),
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress
                | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        frame.count = create_buffer(device, gpu, sizeof(std::uint32_t),
            vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        const vk::AccelerationStructureGeometryKHR instance_geometry{
            .geometryType = vk::GeometryTypeKHR::eInstances,
            .geometry = {.instances = vk::AccelerationStructureGeometryInstancesDataKHR{
                .data = {.deviceAddress = frame.instances.address},
            }},
        };

        const vk::AccelerationStructureBuildGeometryInfoKHR tlas_build{
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = 1,
            .pGeometries = &instance_geometry,
        };

        const vk::AccelerationStructureBuildSizesInfoKHR tlas_sizes = device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, tlas_build, shadow_instance_capacity);

        frame.storage = create_storage(device, gpu, tlas_sizes.accelerationStructureSize);
        frame.scratch = create_scratch(device, gpu, tlas_sizes.buildScratchSize, alignment);
        frame.tlas = vk::raii::AccelerationStructureKHR(device, vk::AccelerationStructureCreateInfoKHR{
            .buffer = *frame.storage.handle,
            .size = tlas_sizes.accelerationStructureSize,
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        });
        frame.address = device.getAccelerationStructureAddressKHR(
            vk::AccelerationStructureDeviceAddressInfoKHR{.accelerationStructure = *frame.tlas});

        structures.frames.push_back(std::move(frame));
    }

    return structures;
}

// Every frame

void record_shadow_tlas(
    const vk::raii::CommandBuffer &commands,
    const AccelerationStructures &structures,
    std::size_t frame_index,
    vk::DeviceAddress frame,
    std::uint32_t placement_count,
    vk::Buffer readback,
    vk::DeviceSize readback_offset
) {
    const ShadowTlas &tlas = structures.frames[frame_index];

    const auto barrier = [&](vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
        const vk::MemoryBarrier2 memory{
            .srcStageMask = src_stage,
            .srcAccessMask = src_access,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
        };
        commands.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &memory});
    };

    // 1. Every slot inactive and the count zero. The frame's last use of them, two frames ago, is over: the CPU waited for its fence.
    commands.fillBuffer(*tlas.instances.handle, 0, vk::WholeSize, 0);
    commands.fillBuffer(*tlas.count.handle, 0, vk::WholeSize, 0);
    barrier(vk::PipelineStageFlagBits2::eClear, vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

    // 2. The instances: a thread per placement.
    const ShadowPushData push{.frame = frame, .instances = tlas.instances.address, .count = tlas.count.address};
    commands.bindPipeline(vk::PipelineBindPoint::eCompute, *structures.write_instances);
    commands.pushDataEXT(vk::PushDataInfoEXT{
        .offset = 0,
        .data = {.address = &push, .size = sizeof(push)},
    });
    commands.dispatch((placement_count + shadow_workgroup_size - 1) / shadow_workgroup_size, 1, 1);

    // The build reads the instances as its input, which the spec puts under shader reads at the build stage; the copy reads the count.
    barrier(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits2::eCopy,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead);

    // 3. The build, over the whole buffer: the inactive slots cost it nothing.
    const vk::AccelerationStructureGeometryKHR instance_geometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry = {.instances = vk::AccelerationStructureGeometryInstancesDataKHR{
            .data = {.deviceAddress = tlas.instances.address},
        }},
    };

    const vk::AccelerationStructureBuildGeometryInfoKHR tlas_build{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .dstAccelerationStructure = *tlas.tlas,
        .geometryCount = 1,
        .pGeometries = &instance_geometry,
        .scratchData = {.deviceAddress = align_up(tlas.scratch.address, structures.scratch_alignment)},
    };

    const vk::AccelerationStructureBuildRangeInfoKHR tlas_range{.primitiveCount = shadow_instance_capacity};
    const vk::AccelerationStructureBuildRangeInfoKHR *tlas_range_pointer = &tlas_range;
    commands.buildAccelerationStructuresKHR(tlas_build, tlas_range_pointer);

    // Visible to the fragment shaders' ray queries.
    barrier(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR, vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eAccelerationStructureReadKHR);

    // 4. The count, for the title.
    commands.copyBuffer(*tlas.count.handle, readback, vk::BufferCopy{.srcOffset = 0, .dstOffset = readback_offset, .size = sizeof(std::uint32_t)});
}
```

`game-engine/shaders/shadows.slang`:
```slang
// The ray-tracing instances of this frame's TLAS (acceleration.h): one thread per placement, writing a VkAccelerationStructureInstanceKHR for each within shadow_radius of the camera, into the slot an atomic counter hands out, up to the capacity. The TLAS is built from the whole buffer afterwards; slots never written stay zero, which the build takes as inactive instances. The order the slots fill in changes from frame to frame, which a TLAS doesn't mind: a ray finds the same triangles whatever the order.

#include "shared.slangh"

// Data shared with C++ (src/includes/shader_types.h)

struct ShadowPushData {
    FrameData *frame;
    RayInstance *instances;
    uint *count;
};

[[vk::push_constant]]
ConstantBuffer<ShadowPushData> push;

static const uint shadow_workgroup_size = 256;

// VkGeometryInstanceFlagBitsKHR: rays hit both sides of a triangle, even when a ray asks to cull one side, so a shadow-casting surface is never culled.
static const uint instance_triangle_facing_cull_disable = 0x2;

[shader("compute")]
[numthreads(shadow_workgroup_size, 1, 1)]
void writeInstancesMain(uint3 id : SV_DispatchThreadID) {
    FrameData *frame = push.frame;
    const uint i = id.x;

    if (i >= frame.placement_count) {
        return;
    }

    const Instance instance = instance_of(frame, i);
    const Model model = frame.models[instance.placement.model];
    const float3 center = instance_point(instance, model.center);
    const float radius = model.radius * instance_scale(instance);

    if (length(center) - radius > frame.shadow_radius) {
        return;
    }

    uint slot;
    InterlockedAdd(push.count[0], 1, slot);

    if (slot >= frame.shadow_capacity) {
        return;
    }

    // The TLAS's space is the camera's cell's, from its corner: the placement's origin there is its camera-relative origin plus the camera's offset in the cell. The transform's rows are the linear part's, each ending in that origin's coordinate.
    const float3 origin = instance.origin + frame.camera_offset;

    RayInstance out;
    out.row0 = float4(instance.linear[0], origin.x);
    out.row1 = float4(instance.linear[1], origin.y);
    out.row2 = float4(instance.linear[2], origin.z);
    out.custom_mask = i | (0xFFu << 24);
    out.sbt_flags = instance_triangle_facing_cull_disable << 24;
    out.blas = model.blas;
    push.instances[slot] = out;
}
```

In `game-engine/src/vulkan_setup.cpp`, replace the line `// Device creation doesn't need VK_KHR_shader_untyped_pointers, but shaders that index the descriptor heap compile to SPIR-V untyped pointers. Acceleration structures (the scene, organized for tracing rays) need deferred host operations, an extension they're built on, even though we build them on the GPU. Ray queries trace rays from any shader. Mesh shaders draw the terrain (terrain.slang).` with:
```cpp
    // Device creation doesn't need VK_KHR_shader_untyped_pointers, but shaders that index the descriptor heap compile to SPIR-V untyped pointers. Acceleration structures (the scene, organized for tracing rays) need deferred host operations, an extension they're built on, even though we build them on the GPU. Ray queries trace rays from any shader. Mesh shaders draw the terrain (terrain.slang) and the clusters. Memory budget reports how much of the GPU's memory is in use and how much the driver will let us have (main.cpp's device_memory).
```

In `game-engine/src/vulkan_setup.cpp`, replace the line `vk::EXTMeshShaderExtensionName,` with:
```cpp
        vk::EXTMeshShaderExtensionName,
        vk::EXTMemoryBudgetExtensionName,
```

## 20.5 The walk, from the cells: `cull.slang`, `culling.h`, `culling.cpp`

### Why
Chapter 19's walk started from every draw and ended at clusters in draw order, which was list order. Now it starts from the cells, two levels above, and the clusters come out in the walk's order, cell by cell, with every list mixed. Both changes are the same machinery: a level is a thread per item that says what it produces, prefix sums over the workgroups, and the items written where the sums put them. The sort by list is that too, once more, with twelve counts per workgroup instead of three.

### How
- **Four kinds of item.** `item_at` gives a thread its item: in the early phase's first level, the thread's index as a cell; in the late phase's, the early phase's candidates; after that, what the level before wrote.
  1. **A cell:** its box, both corners from its cell; in view; not `dropped`: nearer than `mesh_draw_distance` and not under a pixel, each counted; hidden by the pyramid makes it a candidate; otherwise it produces its placements.
  2. **A placement:** the model's sphere through `instance_point`, its radius by `instance_scale`; the same tests; kept, it produces the roots of its primitives' hierarchies that are worth walking: one per DAG level of each primitive, but only those whose error is over the threshold at their sphere, which `root_wanted` tests as the node level would. A root under it would be pruned there anyway, and a tree of three primitives has 39 roots: without this, a hilltop view wrote 782,000 root items and overflowed.
  3. **A node and a cluster** as in Chapter 19, with the placement's `Instance` where the draw's matrices were: a sphere's centre through `instance_point`, its radius and error by `instance_scale`, and the cone test's camera through `instance.inverse`.
- **`walkWriteMain`** writes what each kind produces: a cell its run of placements, a placement its wanted roots, found again the same way, a node its children or its group's clusters; `make_item` now carries the primitive. Where a level, the candidates or the clusters have no room, nothing is written and the shortfall is counted in `dropped_items`.
- **The sort** (`sortStartMain`, `sortCountMain`, `sortScanMain`, `sortWriteMain`): a workgroup per 256 clusters; per workgroup, twelve group scans, one per list, of a 1 where the cluster is in that list, and their totals written; one workgroup scans each list's totals over the blocks, starting where the lists before it end, which the walk counted, carrying across 256 blocks at a time; then each cluster goes to its list's base for its workgroup plus its place among the workgroup's clusters of that list, found by the same twelve scans. Every cluster's place is fixed by the sums: the lists come out in the walk's order, every frame the same. The block totals and bases hold twelve counts per workgroup now, and the dispatches read the sorted clusters.
- **`culling.h`:** no draws, no order, no `list_draws`: the twelve lists are always dispatched, an empty one as zero workgroups. `max_cull_items` is 2²¹, room for every placement at once, since a hilltop view walks half a million of them; the walk's levels are the cells, the placements, the hierarchies' depth and the clusters. `CullTotals` gains the shadow instance count.
- **`record_culling`** runs the sort's four steps after the walk, before the dispatches are written.

### Code
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
    uint list_counts[12];   // cluster items per draw list
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

// The draw list a placement's primitive belongs to (culling.h's draw_list_index): the material's alpha mode, times whether it's double-sided and whether the placement mirrors.
uint draw_list_of(FrameData *frame, Instance instance, uint primitive) {
    const Material material = frame.materials[frame.primitive_data[primitive].material];
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

        for (uint list = 0; list < 12; ++list) {
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
//   - A placement is tested the same way, by its model's sphere; kept, its primitives' hierarchies' roots go on: one per DAG level, but only those whose error is over the threshold, since a root under it would be pruned at the next level anyway, and a tree of three primitives has 39 of them.
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

        const Cluster c = frame.clusters[item_index(item)];
        InterlockedAdd(counters.triangles, c.triangle_count);
        InterlockedAdd(counters.level_clusters[min(frame.cluster_groups[c.group].depth, 15)], 1);
        InterlockedAdd(counters.list_counts[draw_list_of(frame, instance_of(frame, item.placement), item_primitive(item))], 1);
    }
}

// 4. The sort by list, and the dispatches

// The clusters came out cell by cell, every list mixed. A counting sort puts each list's together: per workgroup of clusters, how many of each list (sortCountMain); the prefix sums of those, per list, starting where the lists before it end (sortScanMain); then each cluster into its list's run (sortWriteMain). Twelve prefix sums per workgroup instead of three, and the same rule: every cluster's place is fixed by the sums, so the lists come out in the walk's order, every frame the same.

// Before the sort: a workgroup per walk_workgroup_size clusters.
[shader("compute")]
[numthreads(1, 1, 1)]
void sortStartMain() {
    WalkCounters *counters = push.tables.counters;
    counters.dispatch_x = (min(counters.clusters, push.tables.item_capacity) + walk_workgroup_size - 1) / walk_workgroup_size;
    counters.dispatch_y = 1;
    counters.dispatch_z = 1;
}

// The cluster's list, or 12 for a thread past the clusters: in no list.
uint sort_list(FrameData *frame, CullTables *tables, uint i) {
    if (i >= min(tables.counters.clusters, tables.item_capacity)) {
        return 12;
    }

    const ClusterItem item = tables.cluster_items[i];
    return draw_list_of(frame, instance_of(frame, item.placement), item_primitive(item));
}

// Per cluster: its list; per workgroup: how many of each.
[shader("compute")]
[numthreads(walk_workgroup_size, 1, 1)]
void sortCountMain(uint3 id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint thread : SV_GroupThreadID) {
    const uint list = sort_list(push.frame, push.tables, id.x);
    uint prefix;

    for (uint l = 0; l < 12; ++l) {
        const uint total = group_exclusive_scan(list == l ? 1 : 0, thread, prefix);
        if (thread == 0) {
            push.tables.block_totals[group_id.x * 12 + l] = total;
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

    for (uint l = 0; l < 12; ++l) {
        // The blocks, scan_size at a time, carrying the running total across.
        uint carry = start;

        for (uint first = 0; first < blocks; first += scan_size) {
            const uint b = first + thread;
            const uint value = b < blocks ? tables.block_totals[b * 12 + l] : 0;
            uint prefix;
            const uint total = group_exclusive_scan(value, thread, prefix);

            if (b < blocks) {
                tables.block_bases[b * 12 + l] = carry + prefix;
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

    for (uint l = 0; l < 12; ++l) {
        uint prefix;
        group_exclusive_scan(list == l ? 1 : 0, thread, prefix);
        place = list == l ? tables.block_bases[group_id.x * 12 + l] + prefix : place;
    }

    if (list < 12) {
        tables.sorted_items[place] = tables.cluster_items[id.x];
    }
}

// After the sort: each draw list's dispatch. A list's clusters are a run of the sorted items: its start is the lists' before it, summed. The width is capped at the 65,535 workgroups Vulkan guarantees per dimension; y takes the rest, and the mesh shader skips the workgroups past the count.
[shader("compute")]
[numthreads(1, 1, 1)]
void writeDispatchesMain() {
    CullTables *tables = push.tables;
    WalkCounters *counters = tables.counters;
    uint start = 0;

    for (uint list = 0; list < 12; ++list) {
        const uint count = counters.list_counts[list];
        tables.dispatches[list] = ClusterDispatch(min(count, max_dispatch_width), (count + max_dispatch_width - 1) / max_dispatch_width, 1, count, start, 0, tables.sorted_items);
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
// Three alpha modes times two times two: twelve lists, numbered alpha mode x 4 + double-sided x 2 + mirrored.
constexpr std::uint32_t draw_list_count = 12;

constexpr std::uint32_t draw_list_index(AlphaMode alpha_mode, bool double_sided, bool mirrored) {
    return static_cast<std::uint32_t>(alpha_mode) * 4 + (double_sided ? 2 : 0) + (mirrored ? 1 : 0);
}

// GPU culling

// The cull's compute pipelines and buffers, for a world whose placements never change.
//   - The walk starts from the placement cells (placements.h): a thread per column of cells, kept when its box is in view, within the draw distance, not under a pixel, and not hidden: not wholly behind what the depth pyramid holds. The pyramid is the depth buffer with a mip chain where each texel holds the farthest depth of the texels above it, so a box can be tested against the depth under its whole screen rectangle in four reads. A kept cell's placements are the next level's items, tested the same way by their models' spheres.
//   - A kept placement is walked: each of its model's primitives' hierarchy of clusters (clusters.h), from the roots down, each node tested like the placement, and against the error threshold, until the clusters at the one level of detail the distance calls for. The walk goes level by level, every level one item per thread: what each item produces, prefix sums across the workgroups, then the items written where the sums put them.
//   - Every frame, the cull runs in two phases around the depth prepass. The early phase walks every cell against the pyramid the previous frame built, under this frame's view: a guess, right wherever the view hasn't changed. Whatever it sets aside on the pyramid alone, a cell, a placement, a node or a cluster, goes on a list of candidates. The prepass draws what it keeps, the pyramid is built from that depth, and the late phase walks the candidates against it. Whatever the guess hid wrongly is drawn late; nothing visible is missed, and nothing is drawn twice.
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

In `game-engine/src/culling.cpp`, replace the line `#include <numeric>` with:
```cpp
#include <stdexcept>
```

In `game-engine/src/culling.cpp`, replace `create_draw_culling` with:
```cpp
DrawCulling create_draw_culling(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::Queue &queue,
    const vk::raii::CommandPool &pool,
    const PlacementBuffers &placements,
    const Terrain &terrain,
    const Clusters &clusters
) {
    DrawCulling culling;
    culling.walk_start = create_compute_pipeline(device, "cull", "walkStartMain");
    culling.walk = create_compute_pipeline(device, "cull", "walkMain");
    culling.walk_scan = create_compute_pipeline(device, "cull", "walkScanMain");
    culling.walk_write = create_compute_pipeline(device, "cull", "walkWriteMain");
    culling.sort_start = create_compute_pipeline(device, "cull", "sortStartMain");
    culling.sort_count = create_compute_pipeline(device, "cull", "sortCountMain");
    culling.sort_scan = create_compute_pipeline(device, "cull", "sortScanMain");
    culling.sort_write = create_compute_pipeline(device, "cull", "sortWriteMain");
    culling.write_dispatches = create_compute_pipeline(device, "cull", "writeDispatchesMain");
    culling.copy_depth = create_compute_pipeline(device, "cull", "copyDepthMain");
    culling.reduce_depth = create_compute_pipeline(device, "cull", "reduceDepthMain");
    culling.select_early_patches = create_compute_pipeline(device, "cull", "selectEarlyPatchesMain");
    culling.select_late_patches = create_compute_pipeline(device, "cull", "selectLatePatchesMain");
    culling.cell_count = placements.cell_count;

    // An item names a node or a cluster in 22 bits and a primitive in 8 (shader_types.h).
    if (clusters.cluster_count >= (1u << item_index_bits) || clusters.node_count >= (1u << item_index_bits)) {
        throw std::runtime_error("more than 4 million clusters or nodes");
    }

    // The walk's levels: the cells, their placements, then each primitive's hierarchy's nodes, down to a group's clusters, then the clusters themselves. A tree of 8-wide nodes over g groups is ceil(log8 g) + 1 deep; counted over every group in the scene rather than per level of one primitive, it's a safe overestimate, and a level with nothing in it costs four empty dispatches.
    std::uint32_t deepest = 1;
    for (std::uint32_t groups = clusters.group_count; groups > 1; groups = (groups + cluster_node_width - 1) / cluster_node_width) {
        ++deepest;
    }
    culling.walk_levels = 2 + deepest + 2;

    // The tables, uploaded once; a buffer can't be empty.
    const auto upload = [&](std::span<const std::byte> bytes) {
        const std::vector<std::byte> one(sizeof(std::uint32_t));
        return upload_buffer(device, gpu, queue, pool, bytes.empty() ? std::span(one) : bytes,
            vk::BufferUsageFlagBits::eShaderDeviceAddress);
    };

    // The frame each patch was last drawn in: 0 to begin with, before any frame.
    const std::vector<std::uint32_t> never(terrain.patch_count);
    culling.early_patches = upload(std::as_bytes(std::span(never)));

    // What each phase's steps rewrite every frame. The counters are read as indirect dispatch arguments and copied out; the dispatches are read as indirect draw arguments. The block totals and bases hold three counts per workgroup for the walk and twelve, one per list, for the sort.
    const auto blocks = (max_cull_items + walk_workgroup_size - 1) / walk_workgroup_size;

    for (CullPhaseBuffers &phase : culling.phases) {
        for (Buffer &items : phase.items) {
            items = gpu_numbers(device, gpu, static_cast<std::size_t>(max_cull_items) * 2);
        }
        phase.results = gpu_numbers(device, gpu, static_cast<std::size_t>(max_cull_items) * 3);
        phase.block_totals = gpu_numbers(device, gpu, static_cast<std::size_t>(blocks) * draw_list_count);
        phase.block_bases = gpu_numbers(device, gpu, static_cast<std::size_t>(blocks) * draw_list_count);
        phase.counters = gpu_numbers(device, gpu, sizeof(WalkCounters) / sizeof(std::uint32_t),
            vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc);
        phase.candidates = gpu_numbers(device, gpu, static_cast<std::size_t>(max_cull_items) * 2);
        phase.cluster_items = gpu_numbers(device, gpu, static_cast<std::size_t>(max_cull_items) * 2);
        phase.sorted_items = gpu_numbers(device, gpu, static_cast<std::size_t>(max_cull_items) * 2);
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
            .cells = placements.cells.address,
            .items = {phase.items[0].address, phase.items[1].address},
            .results = phase.results.address,
            .block_totals = phase.block_totals.address,
            .block_bases = phase.block_bases.address,
            .counters = phase.counters.address,
            .candidates = phase.candidates.address,
            .cluster_items = phase.cluster_items.address,
            .sorted_items = phase.sorted_items.address,
            .dispatches = phase.dispatches.address,
            .early_candidates = early.candidates.address,
            .early_counters = early.counters.address,
            .patches = phase.patches.address,
            .patch_command = phase.patch_command.address,
            .nodes = phase.nodes.address,
            .early_patches = culling.early_patches.address,
            .cell_count = culling.cell_count,
            .item_capacity = max_cull_items,
        };
        phase.tables = upload(std::as_bytes(std::span(&tables, 1)));
    }

    return culling;
}
```

Then replace `record_culling` with:
```cpp
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

    // The sort by list: the clusters' workgroups, each list's count per workgroup, the prefix sums, and the clusters written into their lists.
    bind(commands, culling.sort_start, push);
    commands.dispatch(1, 1, 1);
    to_indirect();

    indirect(culling.sort_count);
    compute_to_compute(commands);

    bind(commands, culling.sort_scan, push);
    commands.dispatch(1, 1, 1);
    compute_to_compute(commands);

    indirect(culling.sort_write);
    compute_to_compute(commands);

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
```

## 20.6 Drawing placements: `mesh.slang`, `shading.slangh`, `terrain.slang`

### Why
The mesh shader placed a cluster's vertices by its draw's matrices; now it builds them. `instance_of` costs a quaternion's nine products and a handful of divisions, once per workgroup of 128 vertices, and once per fragment, where the shading needs the transform for the shadow terminator and the ray origin's instance space.

### How
- **`VertexOutput`** carries the placement, the primitive and the cluster, all flat.
- **`meshMain`** builds the `Instance` once; positions through `instance_point`, normals through `instance_normal`, tangents through the linear part, as they moved with the model matrix before.
- **The fragment shaders** rebuild it from the placement, and find the material through the primitive. Flat-shaded surfaces without normals use `instance_mirrored` where they used the determinant.
- **`ShadowSurface`** holds the placement's origin, linear part and inverse instead of a cell corner and two 4 × 4 matrices; `shadow_ray_origin` moves the point into the instance's space with the inverse and back with the linear part. The terrain, which is in no instance, passes zeros as before.
- **A shadow ray's alpha test** finds the primitive through the hit: the instance's custom index is the placement, whose model's first primitive plus the hit's geometry index is the primitive; `candidate_alpha` takes its `PrimitiveData`.

### Code
In `game-engine/shaders/mesh.slang`, replace the line `// Draws glTF primitives, a cluster at a time: a mesh shader writes each cluster's vertices from the scene's vertex buffer, placed in the world by its draw's DrawData, and its triangles from the cluster's tables; the surface comes from its glTF material, whose textures are read from the descriptor heap. Shaded by shading.slangh: glTF's physically based BRDF, lit by the sun and the file's lights, with ray-traced shadows, and by the sky around the scene, already exposed. Three fragment shaders:` with:
```slang
// Draws glTF primitives, a cluster at a time: a mesh shader writes each cluster's vertices from the scene's vertex buffer, placed in the world by its placement (shared.slangh's Instance), and its triangles from the cluster's tables; the surface comes from its glTF material, whose textures are read from the descriptor heap. Shaded by shading.slangh: glTF's physically based BRDF, lit by the sun and the file's lights, with ray-traced shadows, and by the sky around the scene, already exposed. Three fragment shaders:
```

In `game-engine/shaders/mesh.slang`, replace the line `// What the mesh shader hands to the rasterizer. SV_Position is the clip-space position; every other field but draw_index and cluster is interpolated across the triangle. Vulkan requires integer fields to be flat, which nointerpolation makes them.` with:
```slang
// What the mesh shader hands to the rasterizer. SV_Position is the clip-space position; every other field but the three indices is interpolated across the triangle. Vulkan requires integer fields to be flat, which nointerpolation makes them.
```

In `game-engine/shaders/mesh.slang`, replace the line `nointerpolation uint draw_index : DRAW_INDEX;  // the same for a whole triangle, so never interpolated` with:
```slang
    nointerpolation uint placement : PLACEMENT;  // the same for a whole triangle, so never interpolated
    nointerpolation uint primitive : PRIMITIVE;  // likewise: which primitive, for its material
```

In `game-engine/shaders/mesh.slang`, replace the line `nointerpolation uint cluster : CLUSTER;        // likewise: which cluster the triangle is from` with:
```slang
    nointerpolation uint cluster : CLUSTER;      // and which cluster the triangle is from
```

In `game-engine/shaders/mesh.slang`, replace `fetch_triangle` with:
```slang
Triangle fetch_triangle(FrameData *frame, Instance instance, uint cluster_index, uint primitive) {
    const Cluster cluster = frame.clusters[cluster_index];
    const uint3 local = cluster_triangle(frame, cluster, primitive);
    Triangle triangle;

    for (uint k = 0; k < 3; ++k) {
        const Vertex vertex = frame.vertices[frame.cluster_vertices[cluster.first_vertex + local[k]]];
        triangle.position[k] = instance_point(instance, vertex.position);
        triangle.normal[k] = normalize(instance_normal(instance, vertex.normal));
    }

    return triangle;
}
```

Then replace `meshMain` with:
```slang
// One workgroup per cluster the cull chose for this draw list (ClusterDispatch): 128 threads, one per vertex and one per triangle. The workgroup finds its cluster item at the list's start plus its index, the item's placement and cluster, builds the placement's transform once, and writes the cluster's vertices, transformed as the vertex shader used to, and its triangles, from the cluster's tables, each numbered. A dispatch wider than 65,535 workgroups wraps into y, so workgroups past the count exit with nothing.
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
    const Instance instance = instance_of(frame, item.placement);
    const bool real = index < dispatch.count;

    SetMeshOutputCounts(real ? cluster.vertex_count : 0, real ? cluster.triangle_count : 0);

    if (!real) {
        return;
    }

    if (thread < cluster.vertex_count) {
        const Vertex vertex = frame.vertices[frame.cluster_vertices[cluster.first_vertex + thread]];

        // The vertex, placed, relative to the camera.
        const float3 relative_position = instance_point(instance, vertex.position);

        // Tangent and bitangent lie along the surface, so they move with the model matrix's linear part, like positions; only the normal needs the normal matrix. The bitangent is built before the transform, from glTF's rule B = cross(N, T) * w: a mirroring transform then mirrors it too.
        const float3 bitangent = cross(vertex.normal, vertex.tangent.xyz) * vertex.tangent.w;

        VertexOutput output;
        output.position = mul(frame.view_projection, float4(relative_position, 1.0));
        output.relative_position = relative_position;
        output.normal = instance_normal(instance, vertex.normal);
        output.tangent = mul(instance.linear, vertex.tangent.xyz);
        output.bitangent = mul(instance.linear, bitangent);
        output.uv0 = vertex.uv0;
        output.uv1 = vertex.uv1;
        output.color = vertex.color;
        output.placement = item.placement;
        output.primitive = item_primitive(item);
        output.cluster = item_index(item);
        verts[thread] = output;
    }

    if (thread < cluster.triangle_count) {
        tris[thread] = cluster_triangle(frame, cluster, thread);
        prims[thread].primitive = thread;
    }
}
```

Then replace `surface_normal` with:
```slang
// The direction the surface faces at this pixel, for lighting.
//   1. The interpolated vertex normal. Without normals in the file, glTF asks for flat shading: the triangle's own normal, `face_normal`, from its vertices.
//   2. A normal map tilts it, per texel, within the surface's tangent frame.
//   3. On a double-sided material's back face, the surface faces the other way.
// `map_spread`: how much the normal map's normals spread here, as the standard deviation of their angle, which the mips keep in its alpha as 1 - spread (mips.slang); 0 without a map.
float3 surface_normal(VertexOutput input, Instance instance, Material material, bool front_face, bool apply_normal_map, float3 face_normal, out float map_spread) {
    float3 normal = input.normal;
    map_spread = 0.0;

    // Without normals in the file, the triangle's own, facing the way its winding says.
    if (all(normal == 0.0)) {
        normal = instance_mirrored(instance) ? -face_normal : face_normal;
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
```

Then replace `vertex_normal` with:
```slang
// The interpolated vertex normal, facing the viewer on a double-sided material's back face; the triangle's own, `face_normal`, when the file has no normals, which glTF asks be shaded flat. This is the surface at the scale the mesh describes it, which ambient occlusion searches against: a normal map's detail isn't in the depth buffer. The face normal's sign follows the triangle's winding, which faces the camera on a front face; a mirroring transform reverses the winding, and the sign with it.
float3 vertex_normal(VertexOutput input, Instance instance, Material material, bool front_face, float3 face_normal) {
    float3 normal = input.normal;

    if (all(normal == 0.0)) {
        normal = instance_mirrored(instance) ? -face_normal : face_normal;
    }

    normal = normalize(normal);
    return material.double_sided != 0 && !front_face ? -normal : normal;
}
```

Then replace `prepassMain` with:
```slang
// The prepass draws every opaque and masked surface first, writing only its depth and its vertex normal, octahedrally encoded. Masked surfaces cut out their transparent texels here too, so the depth buffer holds exactly the surfaces the lighting pass will shade.
[shader("fragment")]
float2 prepassMain(VertexOutput input, bool front_face : SV_IsFrontFace, uint primitive : SV_PrimitiveID) : SV_Target {
    FrameData *frame = push.frame;
    const Instance instance = instance_of(frame, input.placement);
    const Material material = frame.materials[frame.primitive_data[input.primitive].material];

    if (alpha_mode == alpha_mask) {
        const float alpha = material.base_color_factor.a * sample_slot(material.base_color, input).a * input.color.a;
        if (alpha < material.alpha_cutoff) {
            discard;
        }
    }

    return encode_octahedral(vertex_normal(input, instance, material, front_face, triangle_normal(fetch_triangle(frame, instance, input.cluster, primitive))));
}
```

Then replace `shade_fragment` with:
```slang
// The surface at this fragment: its exposed radiance (or one input, in a debug view), and its alpha. The lighting pass writes it as it is; the transparency pass adds it into its sums. `front_face`: whether this triangle faces the camera; `primitive`: which of its cluster's triangles it is.
float4 shade_fragment(VertexOutput input, bool front_face, uint primitive) {
    FrameData *frame = push.frame;
    const Instance instance = instance_of(frame, input.placement);
    const Material material = frame.materials[frame.primitive_data[input.primitive].material];

    // The triangle itself: its flat normal, which shadow rays start off the surface along and surfaces without normals are shaded by, and its vertices, for the shadow terminator.
    const Triangle triangle = fetch_triangle(frame, instance, input.cluster, primitive);
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
    const float3 normal = surface_normal(input, instance, material, front_face, frame.view != view_vertex_normal, face_normal, map_spread);

    // Roughness, widened where the normals spread within the pixel. A perfectly smooth surface would reflect a punctual light from a single point, too small for any pixel to catch; a floor on roughness keeps highlights visible.
    const float floored = max(roughness, 0.045);
    const float alpha = antialiased_alpha(floored * floored, vertex_normal(input, instance, material, front_face, face_normal), map_spread);
    const float shading_roughness = sqrt(alpha);

    // Ambient occlusion, from the AO pass's image at this pixel. It combines with the occlusion map by min, not product: both estimate the same thing, at two scales. The bent normal is a deflection from the vertex normal; turning the shading normal by the same deflection keeps the normal map's detail. See-through surfaces aren't in the prepass, so the image there holds whatever is behind them: they use the map alone.
    const float4 gtao = Texture2D.Handle(uint2(frame.ambient_occlusion, 0)).Load(int3(int2(input.position.xy), 0));
    float visibility = occlusion;
    float3 irradiance_normal = normal;

    if (frame.ao_enabled != 0 && alpha_mode != alpha_blend) {
        visibility = min(occlusion, gtao.w);
        irradiance_normal = normalize(rotate_from_to(vertex_normal(input, instance, material, front_face, face_normal), gtao.xyz, normal));
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

    // Where shadow rays start: off the triangle, along its own flat normal, in the placement's space and the TLAS's (shading.slangh).
    const ShadowSurface from = {
        terminator_origin(input.relative_position, triangle),
        face_normal,
        true,
        instance.origin,
        instance.linear,
        instance.inverse,
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
```

In `game-engine/shaders/shading.slangh`, replace `candidate_alpha` with:
```slang
// The alpha of a masked or blended triangle a ray met, at the hit point. The hit's barycentric coordinates weight the triangle's three vertices; the texture is read at full resolution, since there are no neighbouring pixels to pick a mip level from. Rays from neighbouring pixels can hit different materials, so the texture's heap index differs between them: that's fine, since descriptor heap access is non-uniform unless the SPIR-V marks it uniform, and Slang doesn't.
float candidate_alpha(FrameData *frame, PrimitiveData primitive, Material material, uint triangle, float2 barycentrics) {
    const uint first = primitive.first_index + triangle * 3;
    const Vertex v0 = frame.vertices[int(frame.indices[first]) + primitive.vertex_offset];
    const Vertex v1 = frame.vertices[int(frame.indices[first + 1]) + primitive.vertex_offset];
    const Vertex v2 = frame.vertices[int(frame.indices[first + 2]) + primitive.vertex_offset];
    const float3 weight = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);

    const float2 uv = material.base_color.uv_set == 0
        ? v0.uv0 * weight.x + v1.uv0 * weight.y + v2.uv0 * weight.z
        : v0.uv1 * weight.x + v1.uv1 * weight.y + v2.uv1 * weight.z;
    const float vertex_alpha = v0.color.a * weight.x + v1.color.a * weight.y + v2.color.a * weight.z;

    const Texture2D texture = Texture2D.Handle(uint2(material.base_color.texture, 0));
    const SamplerState sampler = SamplerState.Handle(uint2(material.base_color.sampler, 0));
    return material.base_color_factor.a * vertex_alpha * texture.SampleLevel(sampler, uv, 0.0).a;
}
```

Then replace `light_visibility` with:
```slang
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

        // The instance's custom index is the placement; its geometry index is the primitive within the model (acceleration.h).
        const Model model = frame.models[frame.placements[query.CandidateInstanceID()].model];
        const PrimitiveData primitive = frame.primitive_data[model.first_primitive + query.CandidateGeometryIndex()];
        const Material material = frame.materials[primitive.material];
        const float alpha = candidate_alpha(frame, primitive, material, query.CandidatePrimitiveIndex(),
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
```

In `game-engine/shaders/shading.slangh`, remove these lines:
```slang
    bool instanced;      // in the TLAS: the rays meet its triangles in the instance's space, so the origin is moved off the surface there too
    float3 cell_corner;  // camera-relative: the corner of the draw's cell, where its model matrix measures from
    float4x4 model;      // the draw's model matrix
    float4x4 to_model;   // its inverse
};
```

In `game-engine/shaders/shading.slangh`, add this section before `// Where a ray toward the light`:
```slang
    bool instanced;      // in the TLAS: the rays meet its triangles in the instance's space, so the origin is moved off the surface there too
    float3 origin;       // camera-relative: the placement's origin, where its transform measures from (shared.slangh's Instance)
    float3x3 linear;     // the transform's linear part
    float3x3 inverse;    // its inverse
};
```

In `game-engine/shaders/shading.slangh`, replace `shadow_ray_origin` with:
```slang
// Where a ray toward the light `l` starts: off the surface on the light's side, along the flat normal, by offset_ray_origin.
//
// The offset is a few hundred units in the last place of the coordinates, so it must be measured where the ray is really intersected. The TLAS's space is measured from its origin cell, near the camera, not from the camera itself (acceleration.h): tlas_offset, the camera's position in it, moves the camera-relative position there first. But a ray meets an instance's triangles in that instance's own space, after the inverse of its transform: there a roof is a metre up, while in the TLAS's space it may be a few centimetres up, and an offset measured there is sixteen times too small to clear the triangle it starts on. So an instanced surface is first moved off in its own space, through the inverse model matrix, and then in the TLAS's, where the instance transform's rounding needs an offset of its own. The terrain is in no instance: its rays start from the exact height field, and meet only the TLAS.
float3 shadow_ray_origin(FrameData *frame, ShadowSurface surface, float3 l) {
    const float3 n = dot(surface.face_normal, l) >= 0.0 ? surface.face_normal : -surface.face_normal;
    float3 position = surface.position;

    if (surface.instanced) {
        // Normals go to the instance's space by the linear part's transpose: the inverse of how they leave it.
        const float3 in_model = mul(surface.inverse, position - surface.origin);
        const float3 n_model = normalize(mul(transpose(surface.linear), n));
        position = surface.origin + mul(surface.linear, offset_ray_origin(in_model, n_model));
    }

    return offset_ray_origin(position + frame.tlas_offset, n);
}
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

## 20.7 Where simplification stops: `clusters.cpp`

### Why
The world was first built with a scanned tree in the tree's slot, Poly Haven's `island_tree_02`: 1.07 million triangles, its leaves modelled one by one. It found two things.

`clodBuild` simplifies each group by edge collapse with its border locked, aiming at half its triangles, and when a group loses less than 15% that way (`simplify_threshold`, 0.85), it stops: the group is terminal, its error `FLT_MAX`, and every cluster of it is drawn at full detail at any distance, since nothing coarser exists. Chapter 19 chose that, strictly: the chain stops where the asset's topology stops, and on the box and the dragon it stopped late enough. The scan stops at once. Its leaves, 715,000 triangles of them, are each a few triangles with nothing but border edges, and a leaf can't collapse into its neighbour across a gap. The scan world's first run drew 22 million triangles at eye level, and exactly the same at 1, 2 and 4 pixels of error: the threshold has no say over a terminal group.

meshoptimizer has a second simplifier for this, `meshopt_simplifySloppy`: vertex clustering, which snaps vertices to a grid and merges whatever lands together, topology or not. It's crude up close and right at a distance, which is where terminal groups are drawn. `clusterlod` offers it as a fallback, for any group edge collapse brings short of its half: strict collapse first, as before, and the group remeshed sloppily where that falls short, with the sloppy result's error counted double (`simplify_error_factor_sloppy`). The box and the dragon reached their halves, so their builds are unchanged. The permissive mode, which collapses across seams everywhere, stays off.

The second thing no simplifier fixes. With the fallback the scan costs what it should, but walk away from one and its leaves are gone by 20 m, flakes floating off bare twigs, while within 5 m it's all there. The error that drives the levels is geometric, how far a vertex moved: a leaf merged into its neighbour or collapsed to a point moves its vertices a centimetre, a pixel at 12 m, and loses all of its area. The metric is right for a surface, whose silhouette stays within a pixel, and blind to coverage: foliage, grass, hair. The game tree has none of this. Its leaves are cards, their coverage in the texture's alpha, and a card simplified is a card moved, which the error sees; at the coarsest level a tree is 180 triangles, and its crown is still a crown at 160 m. Same placements, same engine: the one keeps its canopy to the horizon, the other is specks past 20 m. Unreal's Nanite, which uses the same kind of error, has a separate "preserve area" setting for foliage for the same reason.

### How
- **`simplify_fallback_sloppy`** on; the rest as it was. The game tree's primitives, 112 to 136 triangles each, reach their halves in one step: 2 levels, 3 clusters.
- **The scan** stays in the assets as what it is: a test of triangle throughput, and of where geometric error stops working. Chapter 21's far field, built from the clusters at full detail, gives even the scan its canopy back beyond the impostor distance; the band between 5 m and there would need a per-primitive error scale or an asset made for it.

### Code
In `game-engine/src/clusters.cpp`, replace the line `// Simplification by edge collapse alone, borders and seams locked: the chain stops where the asset's topology stops. meshoptimizer offers permissive and sloppy fallbacks that reach further at a cost in fidelity; they're off.` with:
```cpp
        // Simplification by edge collapse, borders and seams locked, as far as that goes: a group it can't bring to half its triangles is simplified sloppily instead, meshoptimizer's vertex clustering, which snaps vertices to a grid and merges whatever lands together, topology or not, its error counted double. Without the fallback a group that loses less than 15% that way is a dead end, its clusters drawn at full detail at any distance: a tree modelled leaf by leaf, each leaf its own little mesh with nothing but borders, keeps most of its 700,000 triangles from the far side of the field. The permissive mode, which collapses across seams everywhere, stays off: the chain still stops where the asset's topology stops, and only the groups that fall short are remeshed.
```

In `game-engine/src/clusters.cpp`, replace the line `config.simplify_fallback_sloppy = false;` with:
```cpp
        config.simplify_fallback_sloppy = true;
```

## 20.8 The world: `main.cpp`

### Why
The scene is the world file; the dragons and the draw data go; the placements are uploaded after the acceleration structures, whose BLAS addresses they carry; the TLAS is this frame's; and the title says what the frame costs in memory.

### How
- **The scene** is `world/world.gltf`. `BoxField.gltf` and Sponza draw as before, every node a placement; the terrain keeps its file's heights now, its plateau at −1 m, which the world was scattered onto, so the box field stands a metre above it.
- **The knobs:** `mesh_draw_distance`, no limit by default, the sub-pixel test alone; 512 m makes a ring, and the table below has what each costs. `shadow_radius`, 512 m.
- **The order:** vertices and indices, the clusters, the acceleration structures, the placements with the BLAS addresses, the cull with the cells.
- **The camera** starts at the origin at eye height above the terrain.
- **`device_memory`:** `VkPhysicalDeviceMemoryBudgetPropertiesEXT` on the memory properties: per heap, what's in use and what the driver will allow, summed over the device-local heaps. The spec calls them estimates; on this driver they include what it allocates on its own.
- **The title:** the clusters, triangles and patches as before, then the placements dropped as sub-pixel and as too far, the shadow instances, the memory in use over the budget in megabytes, and, only when it's not zero, `N DROPPED`: a level of the walk overflowed `max_cull_items`, and whatever didn't fit wasn't drawn.
- **Every two seconds,** the clusters per level as before, and the items each level of the walk looked at: cells first, then placements, then roots, nodes and clusters.
- **`record_frame`** records the shadow TLAS after the light clusters; `DrawList` carries the frame's index and the placement count for it, and every list is dispatched.

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
                || totals.early_patches != shown_totals.early_patches || totals.late_patches != shown_totals.late_patches
                || totals.shadow_instances != shown_totals.shadow_instances || memory_used != shown_memory
                || totals.early.dropped_items + totals.late.dropped_items != shown_totals.early.dropped_items + shown_totals.late.dropped_items) {
                const auto minutes = static_cast<int>(std::lround(settings.hours * 60.0f));
                const std::string sky = settings.sky == SkySource::atmosphere
                    ? std::format("sky {:02}:{:02}", minutes / 60, minutes % 60)
                    : std::string("photographed sky");
                // Dropped items mean a level of the walk overflowed max_cull_items: whatever didn't fit wasn't drawn.
                const std::uint32_t dropped = totals.early.dropped_items + totals.late.dropped_items;
                const std::string title = std::format("game-engine: {}, {}, EV {:.1f}, AO {}, {} clusters ({} late), {} triangles, {} patches ({} late), {} sub-pixel, {} far, {} shadow instances, {} / {} MB{}",
                    view_names[static_cast<std::size_t>(settings.view)], sky, ev100, settings.ambient_occlusion ? "on" : "off",
                    clusters_drawn, totals.late.clusters, triangles_drawn, totals.early_patches + totals.late_patches, totals.late_patches,
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

## 20.9 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **The terminal** shows `Loaded world.gltf: 234669 vertices, 364364 triangles, 10 primitives in 8 models, 2000000 placements, 11 materials, 24 images` in about a second, `Primitives: 8 opaque, 2 masked, 0 blended`, then each primitive's chain, the tree's three at `3 clusters, 2 levels`, then `Clusters: 6152 in 406 groups, 480 nodes, 11 levels`, the BLAS build, `2 TLAS of up to 65536 instances`, `Placements: 2000000 in 16384 cells` in under a second in release and a few in debug, and `Culling: 16384 cells, 8 walk levels`.
- **The image** is the plateau, with trees, logs, boulders and props to the hills, every one on the ground, with hard shadows under the near ones, the trees' through their leaves.
- **The title** starts at `68626 clusters (0 late), 5587908 triangles, 1475 patches (0 late), 146044 sub-pixel, 0 far, 24771 shadow instances, 813 / 7658 MB`.
- **Walk through it:** 60 fps everywhere on this GPU, and the trees keep their crowns to the horizon. Climb a hill, and the triangles rise with the view while the title's sub-pixel count climbs into the hundreds of thousands: the far field thinning out as placements drop under a pixel, which is Chapter 21's cue.
- **The shadow instances** stay near 25,000; the memory reads about 0.8 GB of a 7.7 GB budget.
- **Every two seconds** the terminal prints the clusters per level and the items per walk level.
- **No `[validation …]` lines,** and no `DROPPED` in the title.

**What it costs.** Release, 1920 × 1080, RTX 5070 Laptop, by day, three views: eye level on the plateau, 40 m up looking down at 20°, and a hilltop 600 m out and 300 m up. The chapter's world is the first block; the ring is the knob; the scan world of 20.7, the same placements with the scanned tree in the tree's slot, is the experiment.

| World, build, ring | View | Clusters | Triangles | Frame |
|---|---|---|---|---|
| Game tree, no limit | Eye level | 68,626 | 5.59 M | 7.7 ms |
| | 40 m up | 106,193 | 8.29 M | 9.6 ms |
| | Hilltop | 216,131 | 15.7 M | 11.4 ms |
| Game tree, 512 m | Eye level | 15,778 | 1.55 M | 4.6 ms |
| | 40 m up | 16,031 | 1.55 M | 6.4 ms |
| | Hilltop | 4,748 | 0.40 M | 7.4 ms |
| Game tree, 256 m | Eye level | 9,398 | 0.99 M | 4.6 ms |
| Scan, strict, 512 m | Eye level | 221,816 | 22.2 M | 15.7 ms |
| | 40 m up | 413,030 | 41.6 M | 18.3 ms |
| | Hilltop | 249,124 | 25.0 M | 19.8 ms |
| Scan, sloppy fallback, 512 m | Eye level | 22,579 | 2.19 M | 4.7 ms |
| Scan, sloppy fallback, no limit | Eye level | 74,133 | 5.96 M | 7.6 ms |
| | 40 m up | 101,263 | 7.59 M | 9.1 ms |
| | Hilltop | 204,371 | 13.5 M | 10.8 ms |

- **The walk** in the early phase is 0.6 ms from the plateau and 0.9 ms from the hilltop, for the whole world. Its items, from the plateau: the 16,384 cells and the late phase's 88,000 candidates; 271,737 placements; 71,157 roots; then 80,331, 3,480 and 1,319 nodes and clusters down the hierarchies. With a 512 m ring the first level keeps 9,730 placements and the walk takes 0.2 ms.
- **The shadow TLAS,** written and built, is 0.8 to 1.0 ms a frame, for 25,000 instances.
- **The scan** is the chapter's finding: 22 million triangles at eye level inside 512 m, 15.7 ms, and the same at any error threshold, because its terminal groups are drawn whole. The sloppy fallback takes the same view to 2.2 million and 4.7 ms, the game tree's cost, and still loses the leaves past 20 m, which the game tree never does.
- **The ring** then matters little: the sub-pixel test does most of the dropping, and the whole field costs 3.1 ms more than 512 m from eye level, 4.0 ms from the hilltop, for 150,000 to 370,000 placements dropped under a pixel. The far field is already thin from the hilltop, and that thinness is the next chapter.

Next, in Chapter 21, the far field: what the sub-pixel test drops, drawn as something else, built from the clusters and placed by the same cells.

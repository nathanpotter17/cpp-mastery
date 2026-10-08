# Chapter 0: Project, window and device

By the end of this chapter we'll have a window cleared to black by a Vulkan 1.4 device with the descriptor heap enabled. Nothing is drawn yet. What we get is the foundation every later chapter builds on: the build, the window, the GPU choice and the frame loop.

## 0.1 Project layout

### Why
The repo's root CMake treats every `lecture_*` directory as a small single-target lecture. The engine will grow into many files, shaders and assets, so it gets its own CMake project in a directory that doesn't match `lecture_*`. The root build and `build.bash` never see it, and we can configure and build it on its own.

### How
- **Sources:** `src/` holds `.cpp` files, and `src/includes/` holds headers, included as `"includes/foo.h"`.
- **Vendored libraries:** third-party code lives in `vendor/`, copied rather than downloaded. lecture_7 already has the two libraries we need, SDL3 and Vulkan-Headers.
- **Build output and editor setup:** builds go in `build/<config>/`. `vendor/` and `build/` are kept out of git. A `.clangd` file points the editor at the clang debug build's list of compile commands.

### Code
From the repo root:
```bash
mkdir -p game-engine/src/includes
```
```bash
cp -r lecture_7/vendor game-engine/vendor
```

`game-engine/.gitignore`:
```text
build/
vendor/
```

`game-engine/.clangd`:
```yaml
# clangd reads compile flags from the Debug Clang build (build.bash option 1 or 7).
CompileFlags:
  CompilationDatabase: build/debug-clang
```

## 0.2 The build: `CMakeLists.txt`

### Why
CMake has to do three things:
- build SDL with only the parts we use,
- make the Vulkan headers available,
- build our executable with the same strict settings as the lectures: C++23, warnings, sanitizers, and link-time optimization in release builds.

### How
- **Configuring SDL:** SDL is configured through variables. Any normal variable set *before* `add_subdirectory(vendor/SDL3)` becomes one of SDL's build options. We turn off every subsystem except video, and every video backend except Wayland and Vulkan. Each one is listed explicitly, so turning on audio later is a one-word change.
- **No Vulkan library to link:** `vk::raii::Context` loads the Vulkan loader at runtime, so Vulkan-Headers only puts `vulkan.h` and `vulkan.hpp` on the include path.
- **`SYSTEM`** keeps third-party headers from triggering our warnings.
- **`compile_commands.json`** is the list clangd reads. It's turned on *after* the vendored libraries are added, so it only lists our files. clangd gives a header the flags of the most similarly named file in that list, and it would otherwise parse our `sdl.h` as C because of SDL's `SDL.c`.
- **Two vulkan.hpp settings:**
  - `NO_CONSTRUCTORS` makes every Vulkan struct a plain aggregate, so we can write `.field = value`.
  - `HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS` lets us handle a resized window as a return value instead of an exception.

### Code
`game-engine/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.25)

# A standalone project, separate from the lecture_* build at the repo root.
# Configure and build it with ./build.bash.
project(GameEngine LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# --- vendor/SDL3 -------------------------------------------------------------
# Normal variables set before add_subdirectory() act as SDL's options. We build
# a static library with only what a Vulkan window needs, and list every
# subsystem so turning one on later (SDL_AUDIO) is a one-word edit.

set(SDL_SHARED OFF)
set(SDL_STATIC ON)
set(SDL_TEST_LIBRARY OFF)
set(SDL_TESTS OFF)
set(SDL_EXAMPLES OFF)
set(SDL_INSTALL OFF)

set(SDL_VIDEO ON)       # windows, input events
set(SDL_AUDIO OFF)
set(SDL_GPU OFF)        # SDL's own GPU API; we use Vulkan directly
set(SDL_RENDER OFF)     # SDL's 2D renderer
set(SDL_CAMERA OFF)
set(SDL_JOYSTICK OFF)
set(SDL_HAPTIC OFF)
set(SDL_HIDAPI OFF)
set(SDL_POWER OFF)
set(SDL_SENSOR OFF)
set(SDL_DIALOG OFF)
set(SDL_TRAY OFF)

# Video backends: Vulkan surfaces on Wayland (with libdecor for title bars on
# GNOME), or Windows' own backend there. No OpenGL, X11, KMS/DRM or headless.
set(SDL_VULKAN ON)
set(SDL_OPENGL OFF)
set(SDL_OPENGLES OFF)
set(SDL_WAYLAND ON)
set(SDL_WAYLAND_LIBDECOR ON)
set(SDL_X11 OFF)
set(SDL_KMSDRM OFF)
set(SDL_OFFSCREEN OFF)
set(SDL_DUMMYVIDEO OFF)

# Linux desktop integration we don't use: D-Bus (screensaver, portals), IBus
# (input methods), udev (device hotplug), io_uring, PipeWire (audio/camera).
set(SDL_DBUS OFF)
set(SDL_IBUS OFF)
set(SDL_LIBUDEV OFF)
set(SDL_LIBURING OFF)
set(SDL_PIPEWIRE OFF)

# SYSTEM: -Wall and friends don't warn about third-party headers.
add_subdirectory(vendor/SDL3 SYSTEM)

# --- vendor/Vulkan-Headers ---------------------------------------------------
# vulkan.h and vulkan.hpp only. There is nothing to link: vk::raii::Context
# loads the Vulkan loader (libvulkan) at runtime.

add_subdirectory(vendor/Vulkan-Headers SYSTEM)

# --- game-engine -------------------------------------------------------------

# compile_commands.json for clangd. Turned on only now, so it lists our sources
# and not SDL's: clangd gives a header the flags of the closest-named file in
# it, and would parse includes/sdl.h as C, borrowing SDL's SDL.c.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(game-engine)

# src/ is the include root: headers are included as "includes/foo.h".
file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.cpp)
target_sources(game-engine PRIVATE ${sources})
target_include_directories(game-engine PRIVATE src)

target_compile_features(game-engine PRIVATE cxx_std_23)
set_target_properties(game-engine PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(game-engine PRIVATE -Wall -Wextra -Wpedantic)

# vulkan.hpp:
# - NO_CONSTRUCTORS: plain aggregate structs, so designated initializers work.
# - HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS: acquireNextImage/presentKHR return
#   eErrorOutOfDateKHR (e.g. after a resize) instead of throwing it.
target_compile_definitions(game-engine PRIVATE
    VULKAN_HPP_NO_CONSTRUCTORS
    VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
)

target_link_libraries(game-engine PRIVATE SDL3::SDL3 Vulkan::Headers)

# --- build options -----------------------------------------------------------

if(ENABLE_SANITIZERS)
    target_compile_options(game-engine PRIVATE
        -g
        -O1
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls
        -fsanitize=address,undefined
    )
    target_link_options(game-engine PRIVATE -fsanitize=address,undefined)
endif()

# Link-time optimization for Release builds, when the toolchain supports it.
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported LANGUAGES CXX)

if(ipo_supported)
    set_property(TARGET game-engine PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()
```

## 0.3 Building: `build.bash` and `lsan.supp`

### Why
Six build configurations (clang or g++; debug, release, or debug with sanitizers) each need several CMake flags. A small script keeps them consistent and puts each configuration in its own directory, so switching between them never forces a full rebuild. Most of the time we want to see the result straight away, so the script can also run the program after building it.

### How
- **One menu:** run `./build.bash` and pick a number.
  - **1–6** configure and build one configuration, each in its own `build/<config>/` directory.
  - **7–12** do the same, then run the program. Pressing Enter picks 7: build in debug with clang, then run.
- **Running:** the script sets the sanitizer options before starting the program; they only take effect in the sanitizer builds. Close the window or press Escape to quit.
- **`lsan.supp`:** on Wayland, SDL loads libdecor to draw the title bar. Its GTK plugin holds some memory until the program exits, and LeakSanitizer would report that. `lsan.supp` lists those system libraries so only our own leaks are reported.

### Code
`game-engine/build.bash`:
```bash
#!/usr/bin/env bash
set -euo pipefail

# Configures and builds game-engine into build/<config>/, and optionally runs it.

# CMake paths below are relative to this script's directory.
cd "$(dirname "$0")"

# --- Menu --------------------------------------------------------------------

echo "
Build                            Build & run
 1) Debug Clang                   7) Debug Clang
 2) Debug G++                     8) Debug G++
 3) Release Clang                 9) Release Clang
 4) Release G++                  10) Release G++
 5) Debug Clang + Asan & UbSan   11) Debug Clang + Asan & UbSan
 6) Debug G++ + Asan & UbSan     12) Debug G++ + Asan & UbSan
"
read -rp "Choose [1-12, Enter for 7]: " choice
choice="${choice:-7}"

if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > 12 )); then
  echo "Invalid choice: $choice" >&2
  exit 1
fi

# 7-12 are 1-6 plus a run afterwards.
run=false
if (( choice > 6 )); then
  run=true
  choice=$(( choice - 6 ))
fi

# --- Configurations ----------------------------------------------------------

clang=(-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld)
gcc=(-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++)

# Release is already -O3 -DNDEBUG, plus LTO from CMakeLists.txt.
case "$choice" in
  1) config=debug-clang;      flags=(-DCMAKE_BUILD_TYPE=Debug "${clang[@]}") ;;
  2) config=debug-gcc;        flags=(-DCMAKE_BUILD_TYPE=Debug "${gcc[@]}") ;;
  3) config=release-clang;    flags=(-DCMAKE_BUILD_TYPE=Release "${clang[@]}") ;;
  4) config=release-gcc;      flags=(-DCMAKE_BUILD_TYPE=Release "${gcc[@]}") ;;
  5) config=debug-clang-asan; flags=(-DCMAKE_BUILD_TYPE=Debug "${clang[@]}" -DENABLE_SANITIZERS=ON) ;;
  6) config=debug-gcc-asan;   flags=(-DCMAKE_BUILD_TYPE=Debug "${gcc[@]}" -DENABLE_SANITIZERS=ON) ;;
esac

# --- Build -------------------------------------------------------------------

build_dir="build/$config"

cmake -S . -B "$build_dir" -G Ninja "${flags[@]}"
cmake --build "$build_dir"

echo
echo "Build succeeded: game-engine/$build_dir/game-engine"

if [[ "$run" == false ]]; then
  exit 0
fi

# --- Run ---------------------------------------------------------------------

# The sanitizer options only affect sanitizer builds. lsan.supp lists leaks
# inside system libraries (libdecor's GTK plugin on Wayland) that aren't ours.
echo
LSAN_OPTIONS="suppressions=$PWD/lsan.supp" \
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
"./$build_dir/game-engine"
```

Make it executable:
```bash
chmod +x game-engine/build.bash
```

`game-engine/lsan.supp`:
```text
# LeakSanitizer suppressions, used when build.bash runs a sanitizer build (11 or 12).
# On Wayland, SDL loads libdecor for window decorations, and its GTK plugin
# brings in fontconfig/pango and dbus, which keep allocations until exit.
# None of these stacks pass through our code or SDL.
leak:libfontconfig.so
leak:libdbus-1.so
```

## 0.4 First build and the editor: a temporary `src/main.cpp`

### Why
The editor's C++ support, clangd, has to know how each file is compiled before it can follow `#include <SDL3/SDL.h>` into `vendor/SDL3/include`. It reads that from `build/debug-clang/compile_commands.json`, which the `.clangd` file points to. CMake writes that file while it configures a build, so until the first build, clangd doesn't know the file exists.

Without it, clangd falls back to guessing: `'SDL3/SDL.h' file not found`, then a wave of errors for every SDL and Vulkan name, in code that compiles fine. So before writing the real code, we build once with a tiny `main.cpp` that only includes SDL and starts it.

### How
- **The temporary `main.cpp`** includes `SDL3/SDL.h` and initializes and shuts down SDL's video subsystem. That's enough for CMake to have one source file to list, and it shows that SDL builds and links. Section 0.8 replaces this file with the real frame loop.
- **Build Debug Clang** (option 1 or 7). The first build compiles SDL itself, so it takes a while; later builds only recompile what changed.
- **Restart clangd** once `compile_commands.json` exists. clangd doesn't notice the file when it appears in a project that's already open. In VS Code or VSCodium, open the command palette (Ctrl+Shift+P) and run **Developer: Reload Window**, or **clangd: Restart language server**. After that, the include resolves and the errors go away.
- **This happens again whenever we vendor a new library** (glm in Chapter 3, tinygltf in Chapter 4, the image decoders in Chapter 5). Each of those chapters adds its library first, then builds once and restarts clangd, before any code includes the new headers.

### Code
`game-engine/src/main.cpp` (temporary):
```cpp
// Temporary: just enough for a first build, so CMake writes the
// compile_commands.json clangd needs. Section 0.8 replaces this file.
#include <SDL3/SDL.h>

#include <print>

int main() {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::println(stderr, "SDL failed to initialize: {}", SDL_GetError());
        return 1;
    }

    std::println("SDL {}.{}.{} initialized", SDL_MAJOR_VERSION, SDL_MINOR_VERSION, SDL_MICRO_VERSION);
    SDL_Quit();
    return 0;
}
```

Build and run it:
```bash
./game-engine/build.bash
```

Press Enter (option 7). After SDL compiles, the program prints `SDL 3.4.18 initialized` and exits. Then reload the editor window, open `src/main.cpp`, and check that `SDL_Init` has no red underline.

## 0.5 The window: `src/includes/sdl.h`

### Why
Vulkan can't open a window by itself, because windows belong to the operating system. SDL does that for us, gives us keyboard and window events, and later will provide audio. It also tells Vulkan two things: which *instance extensions* are needed to present to this platform's windows (Wayland here), and how to create a *surface*, the Vulkan object that represents the window.

### How
- **`SdlContext` follows the RAII pattern from lecture 7:** `SDL_Init` runs in the constructor and `SDL_Quit` in the destructor, so SDL is shut down no matter how `main` exits.
- **Its two static helpers** report which video driver SDL chose and list the Vulkan instance extensions SDL needs. SDL can only answer the second question once it has loaded Vulkan, which happens when we create a window with `SDL_WINDOW_VULKAN`. That fixes an ordering rule in `main`: the window comes before the instance.
- **`Window` is a `std::unique_ptr` with `SDL_DestroyWindow` as its deleter.** SDL windows aren't resizable by default, so we ask for that.
- **Include order:** `vulkan.h` is included *before* the SDL headers. `SDL_vulkan.h` otherwise declares its own copies of `VkInstance` and `VkSurfaceKHR`.

### Code
`game-engine/src/includes/sdl.h`:
```cpp
#pragma once

// SDL_vulkan.h declares its own VkInstance/VkSurfaceKHR unless vulkan.h has
// been included first; include it first so there's only one definition.
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

// --- SDL itself --------------------------------------------------------------

// Owns SDL_Init()/SDL_Quit() for the lifetime of the program.
class SdlContext {
public:
    SdlContext() {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            throw std::runtime_error(std::string("SDL_Init failed (") + SDL_GetError() + ")");
        }
    }

    ~SdlContext() { SDL_Quit(); }

    SdlContext(const SdlContext&) = delete;
    SdlContext& operator=(const SdlContext&) = delete;

    // "wayland", "x11", "windows", ...
    static std::string_view video_driver() { return SDL_GetCurrentVideoDriver(); }

    // Instance extensions the platform needs to present to a window. SDL only
    // knows them once Vulkan is loaded, which creating a Vulkan window does.
    static std::span<const char* const> required_vulkan_extensions() {
        std::uint32_t count = 0;
        const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);

        if (!names) {
            throw std::runtime_error(std::string("SDL found no Vulkan support (") + SDL_GetError() + ")");
        }

        return {names, count};
    }
};

// --- Windows -----------------------------------------------------------------

using Window = std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)>;

// A resizable window that Vulkan can draw into.
inline Window make_vulkan_window(int width, int height, const char* title, bool visible) {
    SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;

    if (!visible) {
        flags |= SDL_WINDOW_HIDDEN;
    }

    Window window(SDL_CreateWindow(title, width, height, flags), &SDL_DestroyWindow);

    if (!window) {
        throw std::runtime_error(std::string("SDL_CreateWindow failed (") + SDL_GetError() + ")");
    }

    return window;
}
```

## 0.6 Instance and device: `vulkan_setup.h` / `vulkan_setup.cpp`

This file holds everything between "we have a window" and "we have a device to send work to". It covers four concepts.

### Why: the instance
The instance is our connection to the Vulkan loader. We create it with:
- the instance extensions SDL listed,
- in debug builds, the **validation layer**, which checks every API call against the spec,
- in debug builds, a **debug messenger** that prints the layer's findings.

We ask for Vulkan 1.4 as the newest version the app will use. Then we create the **surface** from the window. We need it before choosing a GPU, because "can this GPU present to this window?" is one of the questions we ask.

### Why: choosing a GPU, extensions vs. features
A machine can have several Vulkan devices. The one this was written on has an Intel iGPU, an RTX 5070, and llvmpipe (a CPU renderer). The engine is built around **`VK_EXT_descriptor_heap`**, which turns descriptors into plain bytes in buffers we own (Chapter 6). Only the RTX 5070's driver supports it, so the GPU check needs to be precise. Vulkan exposes optional functionality at two levels:
- an **extension** makes new functions and structs exist,
- a **feature** is a switch you turn on, and some features belong to an extension.

You may only query an extension's feature struct after confirming the device has that extension. So `pick_gpu` checks in a fixed order (can present → Vulkan 1.4 → extensions → features) and prints the first check each GPU fails.

### Why: the logical device and its features
The logical device is our handle to the chosen GPU, with the extensions and features turned on. We enable everything the rasterizer (Chapters 1–7) relies on now, so the device stays the same until Chapter 8 adds the ray tracing extensions:

| Feature | Used for |
|---|---|
| `synchronization2`, `dynamicRendering` | simpler barriers and submits, and rendering without `VkRenderPass` objects (both from lecture 7) |
| `samplerAnisotropy` | sharper textures on surfaces seen at an angle, like floors (Chapter 5) |
| `bufferDeviceAddress` | buffers as 64-bit GPU pointers: vertex data in Chapter 2, heap binding in Chapter 6 |
| `scalarBlockLayout` | shader structs laid out exactly like C++ structs (Chapter 2) |
| `descriptorHeap` | descriptors stored in buffers we own (Chapter 6) |
| `shaderUntypedPointers` | shaders that index the heap compile to SPIR-V *untyped pointers*. Creating the device doesn't need it, but those shaders do |

### How
- **`vk::StructureChain`:** feature structs are passed as a linked list through `pNext`. `StructureChain` builds that list at compile time, in the order the types are listed. We use one alias, `Features`, both to query what a GPU supports and to enable it, so the two can't drift apart.
- **`PhysicalDeviceFeatures2` heads the chain** and replaces `pEnabledFeatures`. Its `features` member holds the original Vulkan 1.0 features, such as `samplerAnisotropy`.
- **Designated initializers must follow each struct's declaration order.** That's why `scalarBlockLayout` comes before `bufferDeviceAddress`.
- **`print_descriptor_heap_properties`:** shows how big this GPU's descriptors are. The numbers explain what the heap is before we use it.

### Code
`game-engine/src/includes/vulkan_setup.h`:
```cpp
#pragma once

#include <vulkan/vulkan_raii.hpp>

#include "includes/sdl.h"

#include <cstdint>
#include <optional>
#include <span>

// A GPU plus the queue family we'll use for both graphics and presenting.
struct GpuChoice {
    vk::raii::PhysicalDevice device;
    std::uint32_t queue_family;
};

// --- Instance ----------------------------------------------------------------

// True if VK_LAYER_KHRONOS_validation is installed (it ships with the SDK,
// or `sudo apt install vulkan-validationlayers`).
bool validation_layer_available(const vk::raii::Context& context);

// `extensions` are the instance extensions to enable (e.g. from SDL).
// With `validation`, also enables the validation layer and VK_EXT_debug_utils.
vk::raii::Instance create_instance(
    const vk::raii::Context& context,
    std::span<const char* const> extensions,
    bool validation
);

// Prints validation warnings and errors to stderr. Needs an instance
// created with `validation`.
vk::raii::DebugUtilsMessengerEXT create_debug_messenger(const vk::raii::Instance& instance);

vk::raii::SurfaceKHR create_surface(const vk::raii::Instance& instance, SDL_Window* window);

// --- Device ------------------------------------------------------------------

// Prints every GPU, then picks a Vulkan 1.4 one that can draw and present to
// `surface` and has everything create_device() turns on, preferring discrete
// over integrated over everything else.
std::optional<GpuChoice> pick_gpu(const vk::raii::Instance& instance, const vk::raii::SurfaceKHR& surface);

// A logical device with one queue from `gpu.queue_family`, the swapchain and
// descriptor heap extensions, and every feature the renderer relies on.
vk::raii::Device create_device(const GpuChoice& gpu);

// Prints how big `gpu`'s descriptors are and how big its heaps may get.
void print_descriptor_heap_properties(const GpuChoice& gpu);
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

constexpr const char* validation_layer = "VK_LAYER_KHRONOS_validation";

// --- What the renderer needs from a GPU --------------------------------------

// Device creation doesn't need VK_KHR_shader_untyped_pointers, but shaders
// that index the descriptor heap compile to SPIR-V untyped pointers.
constexpr std::array device_extensions{
    vk::KHRSwapchainExtensionName,
    vk::EXTDescriptorHeapExtensionName,
    vk::KHRShaderUntypedPointersExtensionName,
};

// Every feature struct we read in pick_gpu() and write in create_device(),
// linked through pNext by StructureChain.
using Features = vk::StructureChain<
    vk::PhysicalDeviceFeatures2,
    vk::PhysicalDeviceVulkan12Features,
    vk::PhysicalDeviceVulkan13Features,
    vk::PhysicalDeviceDescriptorHeapFeaturesEXT,
    vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR
>;

// --- Helpers -----------------------------------------------------------------

VKAPI_ATTR vk::Bool32 VKAPI_CALL on_validation_message(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT /*types*/,
    const vk::DebugUtilsMessengerCallbackDataEXT* data,
    void* /*user_data*/
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

bool has_extensions(const vk::raii::PhysicalDevice& device) {
    const std::vector<vk::ExtensionProperties> available = device.enumerateDeviceExtensionProperties();

    return std::ranges::all_of(device_extensions, [&](std::string_view name) {
        return std::ranges::any_of(available, [&](const vk::ExtensionProperties& extension) {
            return name == extension.extensionName.data();
        });
    });
}

// Only valid once has_extensions() is true: the extension structs in the
// chain may not be queried on a device that lacks their extension.
bool has_features(const vk::raii::PhysicalDevice& device) {
    const Features supported = device.getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceDescriptorHeapFeaturesEXT,
        vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR
    >();

    const auto& vulkan12 = supported.get<vk::PhysicalDeviceVulkan12Features>();
    const auto& vulkan13 = supported.get<vk::PhysicalDeviceVulkan13Features>();

    return supported.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy
        && vulkan12.bufferDeviceAddress
        && vulkan12.scalarBlockLayout
        && vulkan13.synchronization2
        && vulkan13.dynamicRendering
        && supported.get<vk::PhysicalDeviceDescriptorHeapFeaturesEXT>().descriptorHeap
        && supported.get<vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR>().shaderUntypedPointers;
}

}  // namespace

// --- Instance ----------------------------------------------------------------

bool validation_layer_available(const vk::raii::Context& context) {
    return std::ranges::any_of(context.enumerateInstanceLayerProperties(), [](const vk::LayerProperties& layer) {
        return std::string_view(layer.layerName.data()) == validation_layer;
    });
}

vk::raii::Instance create_instance(
    const vk::raii::Context& context,
    std::span<const char* const> extensions,
    bool validation
) {
    std::vector<const char*> enabled_extensions(extensions.begin(), extensions.end());
    std::vector<const char*> enabled_layers;

    if (validation) {
        enabled_layers.push_back(validation_layer);
        enabled_extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    // apiVersion is the newest Vulkan the app will use. Each GPU reports its
    // own version, which pick_gpu() checks.
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

vk::raii::DebugUtilsMessengerEXT create_debug_messenger(const vk::raii::Instance& instance) {
    using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;

    const vk::DebugUtilsMessengerCreateInfoEXT create_info{
        .messageSeverity = Severity::eWarning | Severity::eError,
        .messageType = Type::eGeneral | Type::eValidation | Type::ePerformance,
        .pfnUserCallback = &on_validation_message,
    };

    return vk::raii::DebugUtilsMessengerEXT(instance, create_info);
}

vk::raii::SurfaceKHR create_surface(const vk::raii::Instance& instance, SDL_Window* window) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    // SDL speaks the C API, so hand it the raw handle and wrap the result.
    if (!SDL_Vulkan_CreateSurface(window, static_cast<VkInstance>(*instance), nullptr, &surface)) {
        throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface failed (") + SDL_GetError() + ")");
    }

    return vk::raii::SurfaceKHR(instance, surface);
}

// --- Picking a GPU -----------------------------------------------------------

std::optional<GpuChoice> pick_gpu(const vk::raii::Instance& instance, const vk::raii::SurfaceKHR& surface) {
    std::optional<GpuChoice> best;
    int best_rank = -1;

    for (const vk::raii::PhysicalDevice& device : instance.enumeratePhysicalDevices()) {
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
            verdict = "no descriptor heap";
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

// --- Logical device ----------------------------------------------------------

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
            .features = {.samplerAnisotropy = vk::True},  // sharper textures seen at an angle
        },
        vk::PhysicalDeviceVulkan12Features{
            .scalarBlockLayout = vk::True,    // shader structs laid out like C++ structs
            .bufferDeviceAddress = vk::True,  // buffers as 64-bit GPU pointers
        },
        vk::PhysicalDeviceVulkan13Features{
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

// --- Descriptor heap limits --------------------------------------------------

void print_descriptor_heap_properties(const GpuChoice& gpu) {
    const auto properties = gpu.device.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceDescriptorHeapPropertiesEXT
    >();
    const auto& heap = properties.get<vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();

    std::println("Descriptor heap:");
    std::println("  image descriptor    {:>4} bytes, {:>3}-byte aligned", heap.imageDescriptorSize, heap.imageDescriptorAlignment);
    std::println("  buffer descriptor   {:>4} bytes, {:>3}-byte aligned", heap.bufferDescriptorSize, heap.bufferDescriptorAlignment);
    std::println("  sampler descriptor  {:>4} bytes, {:>3}-byte aligned", heap.samplerDescriptorSize, heap.samplerDescriptorAlignment);
    std::println("  resource heap       up to {} bytes, {} reserved for the driver", heap.maxResourceHeapSize, heap.minResourceHeapReservedRange);
    std::println("  sampler heap        up to {} bytes, {} reserved for the driver", heap.maxSamplerHeapSize, heap.minSamplerHeapReservedRange);
    std::println("  push data           {} bytes", heap.maxPushDataSize);
}
```

## 0.7 The swapchain: `swapchain.h` / `swapchain.cpp`

### Why
We never draw into the window directly. The **swapchain** is a small set of images the window system lends us. We acquire one, draw into it and give it back to be shown. Each image also needs an *image view*, which says how to read or write it, and a semaphore that signals "drawing is finished, safe to present". When the window is resized, the images have the wrong size, so the whole swapchain is rebuilt.

### How
- **Choosing the settings:**
  - Format: 8-bit sRGB, so the GPU converts our linear colors to sRGB when it writes them.
  - Size: the window's size in pixels. Wayland leaves the choice to us.
  - Image count: one more than the minimum, so we rarely wait for an image.
  - Present mode: FIFO (vsync), the only mode every driver must support.
- **Resizing:** `recreate_swapchain` waits for the GPU to go idle, then builds a new swapchain. It passes the old one as `oldSwapchain` so the driver can reuse resources, and destroys the old views and semaphores before the old swapchain.
- **Destruction order in `Swapchain`:** members are destroyed bottom-up, so the views and semaphores go before the swapchain that owns the images.

### Code
`game-engine/src/includes/swapchain.h`:
```cpp
#pragma once

#include "includes/vulkan_setup.h"

#include <vector>

// The window's images, plus what we need per image to draw into them.
// Members are destroyed bottom-up, so the views and semaphores go before
// the swapchain that owns the images.
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
};

Swapchain create_swapchain(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::SurfaceKHR& surface,
    SDL_Window* window
);

// Rebuilds `swapchain` for the window's current size (after a resize).
// Waits for the GPU to go idle first.
void recreate_swapchain(
    Swapchain& swapchain,
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::SurfaceKHR& surface,
    SDL_Window* window
);
```

`game-engine/src/swapchain.cpp`:
```cpp
#include "includes/swapchain.h"

#include <algorithm>
#include <limits>

namespace {

// --- Choosing the swapchain's settings ---------------------------------------

vk::SurfaceFormatKHR choose_format(const std::vector<vk::SurfaceFormatKHR>& formats) {
    // 8-bit BGRA with sRGB encoding: shaders write linear colors and the GPU
    // encodes them to sRGB on the way out. Otherwise take what the surface offers.
    for (const vk::SurfaceFormatKHR& format : formats) {
        if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return format;
        }
    }

    return formats.front();
}

vk::Extent2D choose_extent(const vk::SurfaceCapabilitiesKHR& capabilities, int width, int height) {
    // Most platforms dictate the size. Wayland reports 0xFFFFFFFF and lets us pick.
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    return {
        std::clamp(static_cast<std::uint32_t>(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp(static_cast<std::uint32_t>(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
    };
}

vk::CompositeAlphaFlagBitsKHR choose_composite_alpha(vk::CompositeAlphaFlagsKHR supported) {
    for (auto mode : {vk::CompositeAlphaFlagBitsKHR::eOpaque, vk::CompositeAlphaFlagBitsKHR::eInherit,
                      vk::CompositeAlphaFlagBitsKHR::ePreMultiplied, vk::CompositeAlphaFlagBitsKHR::ePostMultiplied}) {
        if (supported & mode) {
            return mode;
        }
    }

    return vk::CompositeAlphaFlagBitsKHR::eOpaque;
}

// --- Building one ------------------------------------------------------------

Swapchain build(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::SurfaceKHR& surface,
    SDL_Window* window,
    vk::SwapchainKHR old_swapchain
) {
    Swapchain swapchain;
    SDL_GetWindowSizeInPixels(window, &swapchain.window_width, &swapchain.window_height);

    const auto capabilities = gpu.device.getSurfaceCapabilitiesKHR(*surface);
    const auto format = choose_format(gpu.device.getSurfaceFormatsKHR(*surface));

    swapchain.format = format.format;
    swapchain.extent = choose_extent(capabilities, swapchain.window_width, swapchain.window_height);

    // One more than the minimum, so we rarely wait on the driver for an image.
    // A maxImageCount of 0 means no limit.
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

    return swapchain;
}

}  // namespace

// --- Create and recreate -----------------------------------------------------

Swapchain create_swapchain(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::SurfaceKHR& surface,
    SDL_Window* window
) {
    return build(device, gpu, surface, window, nullptr);
}

void recreate_swapchain(
    Swapchain& swapchain,
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::SurfaceKHR& surface,
    SDL_Window* window
) {
    // Nothing may still be using the old images, views or semaphores.
    device.waitIdle();

    Swapchain next = build(device, gpu, surface, window, *swapchain.handle);

    // Destroy the old views and semaphores while their images still exist,
    // then the old swapchain itself.
    swapchain.views.clear();
    swapchain.rendered.clear();
    swapchain = std::move(next);
}
```

## 0.8 The frame loop: `main.cpp`

### Why
The GPU runs on its own timeline. We *record* commands into a command buffer, *submit* them, and the GPU executes them later. To keep both processors busy, the CPU records frame N+1 while the GPU is still drawing frame N. That requires synchronization:
- **a fence per frame,** so the CPU doesn't reuse a command buffer the GPU is still reading,
- **an "image acquired" semaphore,** so drawing waits until the window system has handed us the image,
- **the swapchain's "rendered" semaphore,** so presenting waits until drawing is finished.

### How
- **Two frames in flight,** each with its own command buffer, semaphore and fence. Fences start signalled, so the first wait returns immediately.
- **Each loop iteration:**
  1. handle events,
  2. sleep while minimized,
  3. rebuild the swapchain if the size changed,
  4. wait for this frame's fence,
  5. acquire an image,
  6. record the frame,
  7. submit,
  8. present.
- **`record_frame`:** moves the image into "color attachment" layout, begins rendering with a clear, ends rendering, and moves the image into "present" layout. `transition` wraps the image barrier that does each move. The "Draw calls go here" comment marks where Chapter 1 adds its draw.
- **Destruction order:** objects in `main` are destroyed in reverse declaration order. The window is declared before the instance, because SDL needs the window to have loaded Vulkan first.

### Code
`game-engine/src/main.cpp`, replacing the temporary one from section 0.4:
```cpp
#include "includes/sdl.h"
#include "includes/swapchain.h"
#include "includes/vulkan_setup.h"

#include <array>
#include <cstdlib>
#include <exception>
#include <limits>
#include <print>
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
void transition(
    const vk::raii::CommandBuffer& commands,
    vk::Image image,
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

// Records: swapchain image -> clear to `color` -> ready to present.
void record_frame(
    const vk::raii::CommandBuffer& commands,
    const Swapchain& swapchain,
    std::uint32_t image_index,
    std::array<float, 4> color
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

    // loadOp eClear does the clearing when rendering begins.
    const vk::RenderingAttachmentInfo color_attachment{
        .imageView = *swapchain.views[image_index],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = color}},
    };

    commands.beginRendering(vk::RenderingInfo{
        .renderArea = {.offset = {0, 0}, .extent = swapchain.extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
    });

    // Draw calls go here.

    commands.endRendering();

    transition(commands, image,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone
    );

    commands.end();
}

// --- Events ------------------------------------------------------------------

// Handles every pending event. False once the window was closed or Escape pressed.
bool poll_events() {
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        const bool escape = event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE;

        if (event.type == SDL_EVENT_QUIT || escape) {
            return false;
        }
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

        // --- Frame loop ------------------------------------------------------

        const std::array black{0.0f, 0.0f, 0.0f, 1.0f};
        std::uint64_t frame_count = 0;

        while (poll_events()) {
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
            record_frame(frame.commands, swapchain, image_index, black);

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

## 0.9 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run. On the machine this was written on, the program prints:
```
SDL 3.4.18 on wayland
Validation layer on
GPUs:
  Intel(R) Graphics (ARL)                       IntegratedGpu  Vulkan 1.4.335  no descriptor heap
  NVIDIA GeForce RTX 5070 Laptop GPU            DiscreteGpu    Vulkan 1.4.329  usable
  llvmpipe (LLVM 21.1.8, 256 bits)              Cpu            Vulkan 1.4.335  no descriptor heap
Using NVIDIA GeForce RTX 5070 Laptop GPU
Descriptor heap:
  image descriptor      32 bytes,  32-byte aligned
  buffer descriptor     16 bytes,   8-byte aligned
  sampler descriptor    32 bytes,  32-byte aligned
  resource heap       up to 33554432 bytes, 96768 reserved for the driver
  sampler heap        up to 131072 bytes, 512 reserved for the driver
  push data           256 bytes
Presented 353 frames
```

**What the heap numbers mean:**
- On this GPU, a texture descriptor is 32 bytes we will write into our own buffer. The 32 MiB resource heap holds about a million of them.
- The "reserved" bytes are a slice of *our* buffer that we hand over to the driver.
- The 256 bytes of push data replace push constants.

The last line appears when you close the window; the count depends on how long it was open. There should be no `[validation …]` lines.

We have a black window and a device that's ready for the descriptor heap. Next, in [Chapter 1](01-slang-first-pipeline.md), let's draw something.

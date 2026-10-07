# Lecture 7: A Vulkan window

By the end of this lecture an SDL window is cleared to black by Vulkan every frame, at the display's refresh rate, and survives being resized and minimized. Nothing is drawn yet. This lecture is the whole skeleton every Vulkan program needs before its first triangle: instance, GPU, device, swapchain, and a synchronized frame loop.

## 7.1 Project layout and dependencies

### Why
Two libraries do the platform work for us:
- **SDL3** opens the window, delivers keyboard and window events, and knows how to connect Vulkan to the platform's windows. Later it will also play audio.
- **Vulkan-Headers** provides `vulkan.h` and the C++ bindings `vulkan.hpp`/`vulkan_raii.hpp`. We use the RAII bindings throughout, so every Vulkan object destroys itself.

There is no Vulkan library to link: `vk::raii::Context` loads the system's Vulkan loader at runtime.

### How
- **Vendored libraries:** both live in `lecture_7/vendor/`. The root CMake adds every `vendor/<lib>` that has a `CMakeLists.txt` and links its main target, so `SDL3::SDL3` and `Vulkan-Headers` are linked without extra code.
- **The source layout follows the earlier lectures:** headers in `src/includes/`, sources in `src/`.
- **A `.clangd` file** points the editor at the clang debug build's compile commands.

### Code
From the repo root, create the directories, then copy in SDL 3.4.18's source as `lecture_7/vendor/SDL3` and Vulkan-Headers 1.4.341 as `lecture_7/vendor/Vulkan-Headers`:
```bash
mkdir -p lecture_7/src/includes lecture_7/vendor
```
```bash
git clone --depth 1 --branch release-3.4.18 https://github.com/libsdl-org/SDL lecture_7/vendor/SDL3
```
```bash
git clone --depth 1 --branch v1.4.341 https://github.com/KhronosGroup/Vulkan-Headers lecture_7/vendor/Vulkan-Headers
```

`lecture_7/.clangd`:
```yaml
# clangd reads compile flags from the clang debug build (build-debug-clang.bash).
CompileFlags:
  CompilationDatabase: build/debug-clang
```

## 7.2 Build options: `lecture.cmake`

### Why
SDL is a large library: audio, gamepads, a 2D renderer, its own GPU API, camera input and more. A Vulkan window needs almost none of that. We also need two vulkan.hpp settings for this lecture's code style.

### How
- **Configuring SDL:** the root CMake's `add_lecture()` includes `lecture.cmake` *before* adding `vendor/`, so normal variables set here become the vendored projects' options. They are not written to the cache, so they don't leak into other lectures. We turn every SDL subsystem off except video, and every video backend off except Wayland, with libdecor for title bars on GNOME, and Vulkan.
- **vulkan.hpp settings:**
  - `VULKAN_HPP_NO_CONSTRUCTORS` makes every Vulkan struct a plain aggregate, so we can use designated initializers (`.field = value`).
  - `VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS` makes a resized window a return value we check, not an exception.

### Code
`lecture_7/lecture.cmake`:
```cmake
# Included by add_lecture() before vendor/ is added, so normal variables set
# here act as options for the vendored projects (and stay out of the cache,
# so they don't leak into other lectures). ${name} is this lecture's target.
#
# vendor/SDL3            -> target SDL3::SDL3      (linked automatically)
# vendor/Vulkan-Headers  -> target Vulkan-Headers  (linked automatically)

# SDL3: a static library with only what a Vulkan window needs. Every
# subsystem is listed, so turning one on later (SDL_AUDIO) is a one-word edit.
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

# vulkan.hpp:
# - NO_CONSTRUCTORS: plain aggregate structs, so designated initializers work.
# - HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS: acquireNextImage/presentKHR return
#   eErrorOutOfDateKHR (e.g. after a resize) instead of throwing it.
target_compile_definitions(${name} PRIVATE
    VULKAN_HPP_NO_CONSTRUCTORS
    VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
)
```

`lecture_7/lsan.supp`, read by the sanitizer build scripts:
```text
# LeakSanitizer suppressions, picked up by the *-ub build scripts.
# On Wayland, SDL loads libdecor for window decorations, and its GTK plugin
# brings in fontconfig/pango and dbus, which keep allocations until exit.
# None of these stacks pass through our code or SDL.
leak:libfontconfig.so
leak:libdbus-1.so
```

## 7.3 The window: `src/includes/sdl.h`

### Why
Vulkan can't open a window, because windows belong to the operating system. SDL does that, and also answers two Vulkan questions: which *instance extensions* are needed to present to this platform's windows, and how to create a *surface*, the Vulkan object that stands for the window.

### How
- **`SdlContext`** calls `SDL_Init` in its constructor and `SDL_Quit` in its destructor (RAII), so SDL shuts down however `main` exits.
- **`required_vulkan_extensions()`** only works once SDL has loaded Vulkan, which creating a window with `SDL_WINDOW_VULKAN` does. So in `main` the window is created before the instance.
- **`Window`** is a `std::unique_ptr` whose deleter is `SDL_DestroyWindow`. We ask for a resizable window, because SDL windows aren't resizable by default.
- **Include order:** `vulkan.h` comes before the SDL headers. Otherwise `SDL_vulkan.h` declares its own copies of `VkInstance` and `VkSurfaceKHR`.

### Code
`lecture_7/src/includes/sdl.h`:
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

## 7.4 Instance, GPU and device: `vulkan_setup.h` / `vulkan_setup.cpp`

### Why
Three objects stand between "we have a window" and "we can send the GPU work":
- **The instance** connects us to the Vulkan loader. In debug builds it also enables the **validation layer**, which checks every call against the spec, and a **debug messenger** that prints what the layer finds.
- **The physical device** is a GPU we choose. We need one with a queue family that can both draw and present to our window, and that supports Vulkan 1.3.
- **The logical device** is our handle to that GPU, with the extensions and features we use switched on.

### How
- **The instance and surface:** the instance enables the extensions SDL listed. With validation, it also enables `VK_EXT_debug_utils` for the messenger. The surface is created by SDL from the window and wrapped in `vk::raii::SurfaceKHR`.
- **Choosing a GPU:** `pick_gpu` prints every GPU with a verdict, and prefers discrete over integrated over the rest.
- **Two Vulkan 1.3 features** do most of the simplifying:
  - `dynamicRendering` lets us draw without `VkRenderPass`/`VkFramebuffer` objects,
  - `synchronization2` gives the simpler `pipelineBarrier2` and `submit2`.
- **Enabling features:** they're switched on by chaining `PhysicalDeviceVulkan13Features` into the device create info through `pNext`. The only device extension is `VK_KHR_swapchain`.

### Code
`lecture_7/src/includes/vulkan_setup.h`:
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

// Prints every GPU, then picks a Vulkan 1.3 one that can draw and present to
// `surface`, preferring discrete over integrated over everything else.
std::optional<GpuChoice> pick_gpu(const vk::raii::Instance& instance, const vk::raii::SurfaceKHR& surface);

// A logical device with one queue from `gpu.queue_family`, VK_KHR_swapchain,
// and the Vulkan 1.3 dynamicRendering and synchronization2 features on.
vk::raii::Device create_device(const GpuChoice& gpu);
```

`lecture_7/src/vulkan_setup.cpp`:
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

}  // namespace

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

    const vk::ApplicationInfo app_info{
        .pApplicationName = "lecture_7",
        .applicationVersion = vk::makeApiVersion(0, 1, 0, 0),
        .pEngineName = "none",
        .engineVersion = 0,
        .apiVersion = vk::ApiVersion13,
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

        // Dynamic rendering and synchronization2 are core in Vulkan 1.3.
        const bool usable = family && properties.apiVersion >= vk::ApiVersion13;

        std::println(
            "  {:<45} {:<14} Vulkan {}.{}.{}  {}",
            properties.deviceName.data(),
            vk::to_string(properties.deviceType),
            vk::apiVersionMajor(properties.apiVersion),
            vk::apiVersionMinor(properties.apiVersion),
            vk::apiVersionPatch(properties.apiVersion),
            usable ? "usable" : family ? "needs Vulkan 1.3" : "can't present"
        );

        if (usable && rank(properties.deviceType) > best_rank) {
            best = GpuChoice{device, *family};
            best_rank = rank(properties.deviceType);
        }
    }

    return best;
}

vk::raii::Device create_device(const GpuChoice& gpu) {
    const float priority = 1.0f;

    const vk::DeviceQueueCreateInfo queue_info{
        .queueFamilyIndex = gpu.queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };

    const std::array extensions{vk::KHRSwapchainExtensionName};

    // Vulkan 1.3 features: draw without VkRenderPass/VkFramebuffer objects,
    // and the simpler vkCmdPipelineBarrier2/vkQueueSubmit2.
    vk::PhysicalDeviceVulkan13Features features13{
        .synchronization2 = vk::True,
        .dynamicRendering = vk::True,
    };

    const vk::DeviceCreateInfo create_info{
        .pNext = &features13,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    return vk::raii::Device(gpu.device, create_info);
}
```

## 7.5 The swapchain: `swapchain.h` / `swapchain.cpp`

### Why
We never draw into the window itself. The **swapchain** is a small set of images the window system lends us. We acquire one, draw into it, and hand it back to be shown. Each image needs an *image view* (how to access it) and a semaphore that signals "drawing finished, safe to present". After a resize the images are the wrong size, so the whole swapchain is rebuilt.

### How
- **Choosing the settings:**
  - Format: 8-bit BGRA sRGB when available.
  - Size: the window's size in pixels. Wayland leaves the choice to us.
  - Image count: one more than the minimum.
  - Present mode: FIFO (vsync), the only mode every driver must support.
- **Resizing:** `recreate_swapchain` waits for the GPU to go idle, then builds a new swapchain, passing the old one as `oldSwapchain` so the driver can reuse resources. It destroys the old views and semaphores before the old swapchain.
- **Destruction order:** members of `Swapchain` are destroyed bottom-up, so the views and semaphores go before the swapchain that owns the images.

### Code
`lecture_7/src/includes/swapchain.h`:
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

`lecture_7/src/swapchain.cpp`:
```cpp
#include "includes/swapchain.h"

#include <algorithm>
#include <limits>

namespace {

vk::SurfaceFormatKHR choose_format(const std::vector<vk::SurfaceFormatKHR>& formats) {
    // 8-bit BGRA with sRGB encoding is the common choice; any format will do for black.
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

## 7.6 The frame loop: `src/main.cpp`

### Why
The GPU runs on its own timeline. We record commands into a command buffer, submit them, and the GPU runs them later. To keep both processors busy, the CPU records the next frame while the GPU is still drawing the current one, and that needs synchronization:
- **a fence per frame,** so we don't reuse a command buffer the GPU is still reading,
- **an "image acquired" semaphore,** so drawing waits until the window system hands us the image,
- **a "rendered" semaphore per swapchain image,** so presenting waits until drawing is done.

### How
- **Frames in flight:** two, each with a command buffer, a semaphore and a fence. The fences start signalled, so the first wait returns at once.
- **Each loop iteration:**
  1. handle events,
  2. sleep while minimized,
  3. rebuild the swapchain if the size changed,
  4. wait for this frame's fence,
  5. acquire an image,
  6. record,
  7. submit,
  8. present.
- **`record_clear`:**
  - moves the image from "undefined" to "color attachment" layout with a barrier,
  - begins rendering with `loadOp = eClear`, which does the clearing,
  - ends rendering,
  - moves the image to "present" layout.

  `transition` wraps the barrier.
- **Destruction:** objects in `main` are destroyed in reverse declaration order, and `device.waitIdle()` at the end makes sure the GPU is done with all of them.

### Code
`lecture_7/src/main.cpp`:
```cpp
#include "includes/sdl.h"
#include "includes/swapchain.h"
#include "includes/vulkan_setup.h"

#include <array>
#include <charconv>
#include <cstdlib>
#include <exception>
#include <limits>
#include <print>
#include <string_view>
#include <vector>

namespace {

// How many frames the CPU may record ahead of the GPU.
constexpr std::size_t frames_in_flight = 2;

constexpr std::uint64_t no_timeout = std::numeric_limits<std::uint64_t>::max();

// What each in-flight frame needs for itself.
struct Frame {
    vk::raii::CommandBuffer commands = nullptr;
    vk::raii::Semaphore image_acquired = nullptr;  // swapchain image is ready to draw into
    vk::raii::Fence done = nullptr;                // GPU finished this frame's commands
};

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
void record_clear(
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

// `--frames N` closes the window after N frames (for scripted runs); 0 means never.
std::uint64_t frame_limit(int argc, char** argv) {
    std::uint64_t limit = 0;

    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string_view(argv[i]) == "--frames") {
            const std::string_view value = argv[i + 1];
            std::from_chars(value.data(), value.data() + value.size(), limit);
        }
    }

    return limit;
}

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

int main(int argc, char** argv) {
    try {
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
        Window window = make_vulkan_window(800, 600, "lecture_7", true);

        vk::raii::Instance instance = create_instance(context, SdlContext::required_vulkan_extensions(), validation);
        vk::raii::DebugUtilsMessengerEXT messenger = validation
            ? create_debug_messenger(instance)
            : vk::raii::DebugUtilsMessengerEXT(nullptr);

        vk::raii::SurfaceKHR surface = create_surface(instance, window.get());

        std::println("GPUs:");
        std::optional<GpuChoice> gpu = pick_gpu(instance, surface);

        if (!gpu) {
            std::println(stderr, "No Vulkan 1.3 GPU can present to this window");
            return EXIT_FAILURE;
        }

        std::println("Using {}", gpu->device.getProperties().deviceName.data());

        vk::raii::Device device = create_device(*gpu);
        vk::raii::Queue queue = device.getQueue(gpu->queue_family, 0);
        Swapchain swapchain = create_swapchain(device, *gpu, surface, window.get());

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

        const std::array black{0.0f, 0.0f, 0.0f, 1.0f};
        const std::uint64_t limit = frame_limit(argc, argv);
        std::uint64_t frame_count = 0;

        while (poll_events() && (limit == 0 || frame_count < limit)) {
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

            // 3. Record and submit: wait for the image, clear it, signal `rendered` and `done`.
            record_clear(frame.commands, swapchain, image_index, black);

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

## 7.7 Build and run

```bash
./build-scripts-linux/build-debug-clang.bash lecture_7 && ./lecture_7/build/debug-clang/lecture_7 --frames 60
```

On the machine this was written on:
```text
SDL 3.4.18 on wayland
Validation layer on
GPUs:
  Intel(R) Graphics (ARL)                       IntegratedGpu  Vulkan 1.4.335  usable
  NVIDIA GeForce RTX 5070 Laptop GPU            DiscreteGpu    Vulkan 1.4.329  usable
  llvmpipe (LLVM 21.1.8, 256 bits)              Cpu            Vulkan 1.4.335  usable
Using NVIDIA GeForce RTX 5070 Laptop GPU
Presented 60 frames
```

All three GPUs pass this lecture's checks, and the discrete one wins. Without `--frames`, the window stays open until you close it or press Escape. There should be no `[validation …]` lines, including after resizing and minimizing the window.

The window is black because nothing is drawn yet: the `// Draw calls go here.` comment marks the spot. Drawing needs shaders and a pipeline, which is where the game-engine project picks up.

#include "includes/glfw.h"
#include "includes/vulkan_setup.h"

#include <cstdlib>
#include <exception>
#include <print>
#include <ranges>

int main() {
    try {
        GlfwContext glfw;
        std::println("GLFW {} on {}", glfwGetVersionString(), GlfwContext::platform());

        // Loads libvulkan at runtime, so nothing has to link against it.
        vk::raii::Context context;
        const std::uint32_t loader_version = context.enumerateInstanceVersion();
        std::println("Vulkan loader {}.{}.{}",
            vk::apiVersionMajor(loader_version),
            vk::apiVersionMinor(loader_version),
            vk::apiVersionPatch(loader_version)
        );

#ifdef NDEBUG
        const bool validation = false;
#else
        const bool validation = validation_layer_available(context);
#endif
        std::println("Validation layer {}", validation ? "on" : "off");

        const auto extensions = GlfwContext::required_vulkan_extensions();
        std::println("GLFW needs {}", extensions);

        // Declaration order matters: each object is destroyed before the ones above it.
        vk::raii::Instance instance = create_instance(context, extensions, validation);
        vk::raii::DebugUtilsMessengerEXT messenger = validation
            ? create_debug_messenger(instance)
            : vk::raii::DebugUtilsMessengerEXT(nullptr);

        Window window = make_vulkan_window(800, 600, "lecture_7", false);
        vk::raii::SurfaceKHR surface = create_surface(instance, window.get());

        std::println("GPUs:");
        std::optional<GpuChoice> gpu = pick_gpu(instance, surface);

        if (!gpu) {
            std::println(stderr, "No GPU can present to this window");
            return EXIT_FAILURE;
        }

        vk::raii::Device device = create_device(*gpu);
        vk::raii::Queue queue = device.getQueue(gpu->queue_family, 0);

        const auto formats = gpu->device.getSurfaceFormatsKHR(*surface);
        const auto present_modes = gpu->device.getSurfacePresentModesKHR(*surface);
        const auto to_name = [](auto value) { return vk::to_string(value); };

        std::println("Using {} (queue family {})", gpu->device.getProperties().deviceName.data(), gpu->queue_family);
        std::println("  {} surface formats, first is {} / {}",
            formats.size(),
            vk::to_string(formats.front().format),
            vk::to_string(formats.front().colorSpace)
        );
        std::println("  present modes {}", present_modes | std::views::transform(to_name));

        queue.waitIdle();
        std::println("Device and queue ready");
    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

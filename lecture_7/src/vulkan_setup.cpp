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

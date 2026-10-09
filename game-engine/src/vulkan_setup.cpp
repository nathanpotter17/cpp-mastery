#include "includes/vulkan_setup.h"
#include "vulkan/vulkan.hpp"

#include <SDL3/SDL_vulkan.h>
#include <algorithm>
#include <array>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

namespace {
    constexpr const char* validation_layer = "VK_LAYER_KHRONOS_validation";

    constexpr std::array device_extensions{
        vk::KHRSwapchainExtensionName,
        vk::EXTDescriptorHeapExtensionName,
        vk::KHRShaderUntypedPointersExtensionName,
    };

    using Features = vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceDescriptorHeapFeaturesEXT,
        vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR
    >;

    VKAPI_ATTR vk::Bool32 VKAPI_CALL on_validation_message(
        vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
        vk::DebugUtilsMessageTypeFlagsEXT,
        const vk::DebugUtilsMessengerCallbackDataEXT* data,
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
}

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

    if (!SDL_Vulkan_CreateSurface(window, static_cast<VkInstance>(*instance), nullptr, &surface)) {
        throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface failed (") + SDL_GetError() + ")");
    }

    return vk::raii::SurfaceKHR(instance, surface);
}

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

vk::raii::Device create_device(const GpuChoice &gpu) {
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
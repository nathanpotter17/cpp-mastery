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

// Instance

// True if VK_LAYER_KHRONOS_validation is installed
bool validation_layer_available(const vk::raii::Context &context);

// Create the instance with extensions to enable (e.g. from SDL), and the validation layer and VK_EXT_debug_utils.
vk::raii::Instance create_instance(
    const vk::raii::Context &context,
    std::span<const char *const> extensions,
    bool validation
);

// Prints validation warnings and errors to stderr. Needs an instance created with `validation`.
vk::raii::DebugUtilsMessengerEXT create_debug_messenger(const vk::raii::Instance &instance);

vk::raii::SurfaceKHR create_surface(const vk::raii::Instance &instance, SDL_Window *window);

// Device

// Prints every GPU, then picks a Vulkan 1.4 one.
std::optional<GpuChoice> pick_gpu(const vk::raii::Instance &instance, const vk::raii::SurfaceKHR &surface);

vk::raii::Device create_device(const GpuChoice &gpu);

void print_descriptor_heap_properties(const GpuChoice &gpu);
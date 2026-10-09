#pragma once

#include "includes/vulkan_setup.h"

#include <vector>
#include <vulkan/vulkan_raii.hpp>

struct Swapchain {
    vk::raii::SwapchainKHR handle = nullptr;
    vk::Format format = vk::Format::eUndefined;
    vk::Extent2D extent;

    // Framebuffer size
    int window_width = 0;
    int window_height = 0;

    std::vector<vk::Image> images;               // owned by `handle`
    std::vector<vk::raii::ImageView> views;      // one per image
    std::vector<vk::raii::Semaphore> rendered;   // one per image, signalled when drawing is done
};

Swapchain create_swapchain(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::SurfaceKHR &surface,
    SDL_Window *window
);

// Rebuilds `swapchain` for the window's current size (GPU to go idle first).
void recreate_swapchain(
    Swapchain &swapchain,
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::SurfaceKHR &surface,
    SDL_Window *window
);
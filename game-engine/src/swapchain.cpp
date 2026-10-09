#include "includes/swapchain.h"

#include <algorithm>
#include <limits>

namespace {

    // Choosing the swapchain's settings

    vk::SurfaceFormatKHR choose_format(const std::vector<vk::SurfaceFormatKHR> &formats) {
        // 8-bit BGRA with sRGB encoding: shaders write linear colors and the GPU
        // encodes them to sRGB on the way out. Otherwise take what the surface offers.
        for (const vk::SurfaceFormatKHR &format : formats) {
            if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return format;
            }
        }

        return formats.front();
    }

    vk::Extent2D choose_extent(const vk::SurfaceCapabilitiesKHR &capabilities, int width, int height) {
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

    // Building swapchain

    Swapchain build(
        const vk::raii::Device &device,
        const GpuChoice &gpu,
        const vk::raii::SurfaceKHR &surface,
        SDL_Window *window,
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

// Create and recreate

Swapchain create_swapchain(
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::SurfaceKHR &surface,
    SDL_Window *window
) {
    return build(device, gpu, surface, window, nullptr);
}

void recreate_swapchain(
    Swapchain &swapchain,
    const vk::raii::Device &device,
    const GpuChoice &gpu,
    const vk::raii::SurfaceKHR &surface,
    SDL_Window *window
) {
    // Nothing may still be actively using the old images, views or semaphores.
    device.waitIdle();

    Swapchain next = build(device, gpu, surface, window, *swapchain.handle);

    // Destroy the old views and semaphores while their images still exist, then the old swapchain itself.
    swapchain.views.clear();
    swapchain.rendered.clear();
    swapchain = std::move(next);
}

#pragma once

#include <SDL3/SDL_oldnames.h>
#include <SDL3/SDL_video.h>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

class SdlContext {
    public:
        SdlContext() {
            if (!SDL_Init(SDL_INIT_VIDEO)) {
                throw std::runtime_error(std::string("SDL failed to initialize: (") + SDL_GetError() + ")");
            }
        }

        ~SdlContext() { SDL_Quit(); }

        SdlContext(const SdlContext&) = delete;
        SdlContext& operator=(const SdlContext&) = delete;

        static std::string_view video_driver() { return SDL_GetCurrentVideoDriver(); }

        static std::span<const char* const> required_vulkan_extensions() {
            std::uint32_t count = 0;
            const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);

            if (!names) {
                throw std::runtime_error(std::string("SDL found no Vulkan support: (") + SDL_GetError() + ")");
            }

            return {names,count};
        }
};

using Window = std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)>;

inline Window make_vulkan_window(int width, int height, const char* title, bool visible) {
    SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;

    if (!visible) {
        flags |= SDL_WINDOW_HIDDEN;
    }

    Window window(SDL_CreateWindow(title, width, height, flags), &SDL_DestroyWindow);

    if (!window) {
        throw std::runtime_error(std::string("SDL Create Window failed (") + SDL_GetError() + ")");
    }

    return window;
}
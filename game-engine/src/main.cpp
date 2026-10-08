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

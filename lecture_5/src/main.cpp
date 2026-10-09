#include "includes/image.h"

#include <cstdlib>
#include <filesystem>
#include <print>
#include <utility>

int main() {
    const Image original = make_gradient(64, 48);
    const auto path = std::filesystem::temp_directory_path() / "lecture_5_gradient.png";

    if (auto written = write_png(original, path); !written) {
        std::println(stderr, "{}", written.error());
        return EXIT_FAILURE;
    }

    std::println("Wrote {} ({} bytes)", path.string(), std::filesystem::file_size(path));

    auto loaded = load_png(path);

    if (!loaded) {
        std::println(stderr, "{}", loaded.error());
        return EXIT_FAILURE;
    }

    std::println("Loaded {}x{}", loaded->width, loaded->height);

    // PNG is lossless, so every byte should survive the round trip.
    if (loaded->pixels != original.pixels) {
        std::println(stderr, "Round trip changed the pixels");
        return EXIT_FAILURE;
    }

    for (auto [x, y] : {std::pair{0, 0}, {63, 0}, {0, 47}, {63, 47}}) {
        const auto *pixel = loaded->at(x, y);
        std::println("  pixel ({:2}, {:2}) = rgb({:3}, {:3}, {:3})", x, y, pixel[0], pixel[1], pixel[2]);
    }

    std::println("Round trip OK");
    return EXIT_SUCCESS;
}

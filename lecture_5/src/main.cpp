// stb's implementations must be compiled in exactly one .cpp file.
// Headers in src/includes get our full warning set, so quiet the one
// warning stb's implementation trips (GCC and Clang both read these pragmas).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "includes/stb_image.h"
#include "includes/stb_image_write.h"
#undef STB_IMAGE_IMPLEMENTATION
#undef STB_IMAGE_WRITE_IMPLEMENTATION
#pragma GCC diagnostic pop

#include "includes/image.h"

#include <cstdlib>
#include <filesystem>
#include <print>
#include <ranges>

int main() {
    const Image original = make_gradient(64, 48);
    const auto path = std::filesystem::temp_directory_path() / "lecture_5_gradient.png";

    if (auto written = write_png(original, path); !written) {
        std::println(stderr, "{}", written.error());
        return EXIT_FAILURE;
    }

    std::println("Wrote {} ({} bytes)", path.string(), std::filesystem::file_size(path));

    auto loaded = load_image(path);

    if (!loaded) {
        std::println(stderr, "{}", loaded.error());
        return EXIT_FAILURE;
    }

    std::println("Loaded {}x{} with {} channels", loaded->width, loaded->height, loaded->channels);

    // PNG is lossless, so every byte should survive the round trip.
    const auto mismatch = std::ranges::mismatch(original.pixels, loaded->pixels);

    if (mismatch.in1 != original.pixels.end() || loaded->pixels.size() != original.pixels.size()) {
        std::println(stderr, "Round trip changed the pixels");
        return EXIT_FAILURE;
    }

    for (auto [x, y] : {std::pair{0, 0}, {63, 0}, {0, 47}, {63, 47}}) {
        const auto* pixel = loaded->at(x, y);
        std::println("  pixel ({:2}, {:2}) = rgb({:3}, {:3}, {:3})", x, y, pixel[0], pixel[1], pixel[2]);
    }

    std::println("Round trip OK");
    return EXIT_SUCCESS;
}

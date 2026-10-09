# Lecture 5: Reading and writing PNG images

By the end of this lecture we'll write a generated gradient to a PNG file, load it back, and prove the round trip was lossless. Along the way we pull in our first third-party code, and handle errors with `std::expected` instead of exceptions.

## 5.1 Project layout and LodePNG

### Why
Reading and writing image formats correctly is a lot of code: PNG alone means zlib compression, filters and checksums. We don't want to write that ourselves, and we don't want a build-system dependency for it either. **LodePNG** is a PNG encoder and decoder in just two files, `lodepng.h` and `lodepng.cpp`, with a C++ API built on `std::vector`.

### How
- **Where the files go:** both go in `src/includes/lodepng/`.
  - The repo's root CMake compiles every `.cpp` under `src/`, so `lodepng.cpp` is built automatically.
  - It puts `src/` on the include path, so we include the header as `"includes/lodepng/lodepng.h"`.
  - The lecture needs no `lecture.cmake`.
- **Kept out of git:** `src/includes/` is in `.gitignore`, so library code stays out of the repository.
- **The version:** this lecture uses LodePNG 20261006, from <https://github.com/lvandeve/lodepng>. The commands below fetch the latest version; it's on the first lines of `lodepng.h`.

### Code
From the repo root:
```bash
mkdir -p lecture_5/src/includes/lodepng
```
```bash
curl -L -o lecture_5/src/includes/lodepng/lodepng.h https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.h
```
```bash
curl -L -o lecture_5/src/includes/lodepng/lodepng.cpp https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.cpp
```

## 5.2 The image type: `src/includes/image.h`

### Why
We want a value type that owns its pixels, and functions whose return type says they can fail. LodePNG reports failure as a number: `0` means success, anything else is an error code that `lodepng_error_text()` turns into a message.

### How
- **`Image`** stores width, height and a `std::vector<std::uint8_t>` of RGB pixels, 3 bytes each, row by row. `at(x, y)` returns the address of one pixel. It has a mutable and a `const` overload, so it works on both kinds of `Image`.
- **`make_gradient`:** builds a test image where red grows left to right, green grows top to bottom, and blue stays at 128. The corner pixels are therefore known values we can check.
- **`std::expected<T, std::string>`** returns either a value or an error message. Callers test it like a pointer (`if (!result)`) and read `result.error()` on failure. Neither function throws.
- **`write_png` and `load_png`** call `lodepng::encode` and `lodepng::decode` with `LCT_RGB`, meaning "3 bytes per pixel". The `if (const unsigned error = ...)` form declares the error code and tests it in one line; a nonzero code is turned into an error message.
- **Converting sizes:** LodePNG measures sizes as `unsigned`, and we keep them as `int` like the rest of our code, so they're converted at the call.
- **Decoding** goes straight into a `std::vector`, which we move into the returned `Image`.

### Code
`lecture_5/src/includes/image.h`:
```cpp
#pragma once

#include "includes/lodepng/lodepng.h"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

// 8-bit RGB pixels, row-major: 3 bytes per pixel.
struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;

    std::uint8_t *at(int x, int y) {
        return pixels.data() + (static_cast<std::size_t>(y) * width + x) * 3;
    }

    const std::uint8_t *at(int x, int y) const {
        return pixels.data() + (static_cast<std::size_t>(y) * width + x) * 3;
    }
};

// Red increases left to right, green top to bottom, blue is constant.
inline Image make_gradient(int width, int height) {
    Image image{width, height, std::vector<std::uint8_t>(static_cast<std::size_t>(width) * height * 3)};

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::uint8_t *pixel = image.at(x, y);
            pixel[0] = static_cast<std::uint8_t>(x * 255 / (width - 1));
            pixel[1] = static_cast<std::uint8_t>(y * 255 / (height - 1));
            pixel[2] = 128;
        }
    }

    return image;
}

// LodePNG returns 0 on success, or an error code that lodepng_error_text() describes.
// LCT_RGB: our pixels are 3 bytes each.
inline std::expected<void, std::string> write_png(const Image &image, const std::filesystem::path &path) {
    const auto width = static_cast<unsigned>(image.width);
    const auto height = static_cast<unsigned>(image.height);

    if (const unsigned error = lodepng::encode(path.string(), image.pixels, width, height, LCT_RGB)) {
        return std::unexpected(path.string() + ": " + lodepng_error_text(error));
    }

    return {};
}

inline std::expected<Image, std::string> load_png(const std::filesystem::path &path) {
    std::vector<std::uint8_t> pixels;
    unsigned width = 0;
    unsigned height = 0;

    // LCT_RGB: decode to 3 bytes per pixel, whatever the file stores.
    if (const unsigned error = lodepng::decode(pixels, width, height, path.string(), LCT_RGB)) {
        return std::unexpected(path.string() + ": " + lodepng_error_text(error));
    }

    return Image{static_cast<int>(width), static_cast<int>(height), std::move(pixels)};
}
```

## 5.3 The round trip: `src/main.cpp`

### Why
`main` runs the test and reports the result. All the library code lives behind `image.h`, so `main` reads as the plain steps of the test.

### How
1. **Write:** save the gradient to the system temp directory.
2. **Load:** read the file back.
3. **Compare:** `!=` on two vectors compares their sizes and every element. PNG is lossless, so they must be equal.
4. **Spot-check:** print the four corner pixels, which we know from how the gradient was built.

The first two steps return early on failure, printing the error message from `std::expected`.

### Code
`lecture_5/src/main.cpp`:
```cpp
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
```

## 5.4 Build and run

```bash
./build-scripts-linux/build-debug-clang.bash lecture_5 && ./lecture_5/build/debug-clang/lecture_5
```

```text
Wrote /tmp/lecture_5_gradient.png (169 bytes)
Loaded 64x48
  pixel ( 0,  0) = rgb(  0,   0, 128)
  pixel (63,  0) = rgb(255,   0, 128)
  pixel ( 0, 47) = rgb(  0, 255, 128)
  pixel (63, 47) = rgb(255, 255, 128)
Round trip OK
```

The corners show the gradient. Red is 0 on the left and 255 on the right, green is 0 at the top and 255 at the bottom, and blue is 128 everywhere. Every byte came back unchanged.

In lecture 6 we keep the images in memory instead of on disk, and describe them with JSON instead of hardcoding them.

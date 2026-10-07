# Lecture 6: JSON-driven images

By the end of this lecture a JSON config describes a set of gradient "swatches". For each one we generate the image, encode it to PNG **in memory**, decode it back, and print a JSON report. Two new ideas: a vendored library that brings its own CMake project, and mapping C++ structs to JSON with almost no code.

## 6.1 Project layout and nlohmann_json

### Why
Parsing JSON by hand is error-prone, and we'd still need to turn the parsed values into our own types. **nlohmann_json** is the de facto standard C++ JSON library. It parses and prints JSON, and it maps JSON to and from our own structs.

### How
- **Vendoring with CMake:** nlohmann_json 3.12.0 comes with its own `CMakeLists.txt`. We copy the repository into `lecture_6/vendor/nlohmann_json/`. The root CMake's `add_vendored_lib()` adds any `vendor/<lib>` that has a `CMakeLists.txt` as a subproject and links its `<lib>::<lib>` target, here `nlohmann_json::nlohmann_json`. That puts `<nlohmann/json.hpp>` on our include path with no `lecture.cmake`.
- **LodePNG:** comes over from lecture 5 into `src/includes/lodepng/`. As before, the root build compiles its `.cpp` automatically.

### Code
From the repo root:
```bash
mkdir -p lecture_6/src/includes lecture_6/vendor
```
```bash
cp -r lecture_5/src/includes/lodepng lecture_6/src/includes/
```
```bash
git clone --depth 1 --branch v3.12.0 https://github.com/nlohmann/json lecture_6/vendor/nlohmann_json
```

## 6.2 The image type: `src/includes/image.h`

### Why
Every image in this lecture is RGB, as in lecture 5. We need two new operations: averaging an image's color, and encoding and decoding without touching the disk.

### How
- **`Rgb`** is a `std::array<std::uint8_t, 3>`. It gets value semantics and comparison for free, and nlohmann_json maps it to a JSON array `[r, g, b]` automatically.
- **`make_gradient`** blends `from` on the left edge into `to` on the right edge, one channel at a time.
- **`average`** sums each channel and divides by the pixel count.
- **Encoding and decoding in memory:** `encode_png` and `decode_png` use the in-memory overloads of the LodePNG functions from lecture 5. Instead of a file name:
  - `lodepng::encode` writes the PNG into a `std::vector`,
  - `lodepng::decode` reads it from a pointer and size, which a `std::span` provides with `data()` and `size()`.

  Error handling is the same as in lecture 5: a nonzero code becomes the message in `std::unexpected`.

### Code
`lecture_6/src/includes/image.h`:
```cpp
#pragma once

#include "includes/lodepng/lodepng.h"

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

using Rgb = std::array<std::uint8_t, 3>;

// 8-bit RGB pixels, row-major.
struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;

    Rgb at(int x, int y) const {
        const auto i = (static_cast<std::size_t>(y) * width + x) * 3;
        return {pixels[i], pixels[i + 1], pixels[i + 2]};
    }
};

// Blends `from` on the left edge into `to` on the right edge.
inline Image make_gradient(int width, int height, Rgb from, Rgb to) {
    Image image{width, height, {}};
    image.pixels.reserve(static_cast<std::size_t>(width) * height * 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (std::size_t c = 0; c < 3; ++c) {
                const int blended = from[c] + (to[c] - from[c]) * x / (width - 1);
                image.pixels.push_back(static_cast<std::uint8_t>(blended));
            }
        }
    }

    return image;
}

inline Rgb average(const Image& image) {
    std::array<std::size_t, 3> sums{};

    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        sums[i % 3] += image.pixels[i];
    }

    const std::size_t count = image.pixels.size() / 3;
    return {
        static_cast<std::uint8_t>(sums[0] / count),
        static_cast<std::uint8_t>(sums[1] / count),
        static_cast<std::uint8_t>(sums[2] / count),
    };
}

// Encodes to PNG in memory rather than to a file. LodePNG returns 0 on
// success, or an error code that lodepng_error_text() describes.
inline std::expected<std::vector<std::uint8_t>, std::string> encode_png(const Image& image) {
    const auto width = static_cast<unsigned>(image.width);
    const auto height = static_cast<unsigned>(image.height);
    std::vector<std::uint8_t> png;

    if (const unsigned error = lodepng::encode(png, image.pixels, width, height, LCT_RGB)) {
        return std::unexpected(lodepng_error_text(error));
    }

    return png;
}

inline std::expected<Image, std::string> decode_png(std::span<const std::uint8_t> png) {
    std::vector<std::uint8_t> pixels;
    unsigned width = 0;
    unsigned height = 0;

    // LCT_RGB: decode to 3 bytes per pixel, whatever the PNG stores.
    if (const unsigned error = lodepng::decode(pixels, width, height, png.data(), png.size(), LCT_RGB)) {
        return std::unexpected(lodepng_error_text(error));
    }

    return Image{static_cast<int>(width), static_cast<int>(height), std::move(pixels)};
}
```

## 6.3 Swatches as JSON: `src/includes/swatch.h`

### Why
A swatch is just a name, a size and two colors. We want `json.get<Swatch>()` to turn JSON into a `Swatch`, and `json(swatch)` to do the reverse, without writing either conversion by hand.

### How
`NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Swatch, name, width, height, from, to)` generates `to_json` and `from_json` functions for the listed members, using each member's name as its JSON key. "Non-intrusive" means `Swatch` itself stays a plain struct. Each member type must be convertible itself: `std::string`, `int` and `Rgb` (a `std::array`) already are.

### Code
`lecture_6/src/includes/swatch.h`:
```cpp
#pragma once

#include "includes/image.h"

#include <nlohmann/json.hpp>

#include <string>

// One gradient image described in the JSON config.
struct Swatch {
    std::string name;
    int width = 0;
    int height = 0;
    Rgb from{};
    Rgb to{};
};

// Generates to_json/from_json, so json.get<Swatch>() and json(swatch) just work.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Swatch, name, width, height, from, to)
```

## 6.4 The program: `src/main.cpp`

### Why
`main` ties it together: read the config, process each swatch, and report everything as JSON. The exit code says whether every round trip succeeded.

### How
- **The config** is a raw string literal (`R"json(...)json"`), so the JSON needs no escaping.
- **Parsing:** `json::parse(config).at("swatches").get<std::vector<Swatch>>()` parses the text, looks up the array, and converts every element with the generated `from_json`. nlohmann_json reports bad input by throwing `nlohmann::json::exception`, and we catch it at that one spot.
- **For each swatch:**
  1. generate the image,
  2. encode it,
  3. decode it,
  4. append a report entry.

  The two `std::expected` results are checked with an early return.
- **The report** is built from initializer lists. A `Swatch` nests inside the entry automatically through the generated `to_json`.
- **`template get<bool>()`:** in the final `all_of`, `entry` comes from a lambda with an `auto` parameter, so its type is dependent. Calling a member template on a dependent type needs the `template` keyword so the compiler parses the `<` correctly.

### Code
`lecture_6/src/main.cpp`:
```cpp
#include "includes/image.h"
#include "includes/swatch.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <print>
#include <ranges>
#include <vector>

constexpr auto config = R"json(
{
    "swatches": [
        { "name": "sunset", "width": 64, "height": 16, "from": [255, 94, 77],  "to": [255, 195, 113] },
        { "name": "ocean",  "width": 32, "height": 32, "from": [0, 63, 92],    "to": [88, 80, 141] },
        { "name": "mono",   "width": 16, "height": 8,  "from": [0, 0, 0],      "to": [255, 255, 255] }
    ]
}
)json";

int main() {
    std::vector<Swatch> swatches;

    try {
        swatches = nlohmann::json::parse(config).at("swatches").get<std::vector<Swatch>>();
    } catch (const nlohmann::json::exception& e) {
        std::println(stderr, "Bad config: {}", e.what());
        return EXIT_FAILURE;
    }

    auto report = nlohmann::json::array();

    for (const Swatch& swatch : swatches) {
        const Image image = make_gradient(swatch.width, swatch.height, swatch.from, swatch.to);

        auto png = encode_png(image);
        if (!png) {
            std::println(stderr, "{}: {}", swatch.name, png.error());
            return EXIT_FAILURE;
        }

        auto decoded = decode_png(*png);
        if (!decoded) {
            std::println(stderr, "{}: {}", swatch.name, decoded.error());
            return EXIT_FAILURE;
        }

        report.push_back({
            {"swatch", swatch},
            {"png_bytes", png->size()},
            {"average", average(*decoded)},
            {"round_trip", std::ranges::equal(image.pixels, decoded->pixels)},
        });
    }

    std::println("{}", report.dump(2));

    const bool all_ok = std::ranges::all_of(report, [](const auto& entry) { return entry["round_trip"].template get<bool>(); });
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

## 6.5 Build and run

```bash
./build-scripts-linux/build-debug-clang.bash lecture_6 && ./lecture_6/build/debug-clang/lecture_6
```

```text
[
  {
    "average": [
      255,
      144,
      94
    ],
    "png_bytes": 346,
    "round_trip": true,
    "swatch": {
      "from": [
        255,
        94,
        77
      ],
      "height": 16,
      "name": "sunset",
      "to": [
        255,
        195,
        113
      ],
      "width": 64
    }
  },
  {
    "average": [
      43,
      71,
      116
    ],
    "png_bytes": 218,
    "round_trip": true,
    "swatch": {
      "from": [
        0,
        63,
        92
      ],
      "height": 32,
      "name": "ocean",
      "to": [
        88,
        80,
        141
      ],
      "width": 32
    }
  },
  {
    "average": [
      127,
      127,
      127
    ],
    "png_bytes": 92,
    "round_trip": true,
    "swatch": {
      "from": [
        0,
        0,
        0
      ],
      "height": 8,
      "name": "mono",
      "to": [
        255,
        255,
        255
      ],
      "width": 16
    }
  }
]
```

Each entry repeats its swatch, the encoded PNG size, the average color and the round-trip result. nlohmann_json prints object keys in alphabetical order by default. The `mono` swatch averages to 127 on every channel, as a black-to-white blend should.

In lecture 7 we leave files behind and put pixels on screen with Vulkan.

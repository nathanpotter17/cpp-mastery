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

#include "includes/grid.h"
#include "includes/patterns.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// --- C arrays ----------------------------------------------------------------

void demo_c_arrays() {
    std::println("C arrays:");
    std::println("    sizeof(\"abc\") = {}: the literal is a const char[4] ending in '\\0'", sizeof("abc"));
    std::println("    glider: {} rows, {} bytes", std::size(patterns::glider), sizeof(patterns::glider));

    // An array converts ("decays") to a pointer to its first element, and
    // the pointer has no idea how many elements follow.
    const std::string_view* first = patterns::glider;
    std::println("    as a pointer: {} bytes, first row {:?}, row count lost", sizeof(first), *first);

    // A span keeps the pointer and the count together.
    const std::span<const std::string_view> rows = patterns::glider;
    std::println("    as a span: {} rows, last row {:?}", rows.size(), rows.back());
}

// --- std::array --------------------------------------------------------------

void demo_std_array() {
    std::println("std::array:");
    Grid original;
    place(original, patterns::blinker, 2, 3);

    // Unlike a C array, a std::array copies with =, and the copy is independent.
    Grid copy = original;
    copy.set(0, 0, true);
    std::println("    copy == original: {}, alive {} vs {}", copy == original, count_alive(copy.cells), count_alive(original.cells));

    // [] trusts the index. at() checks it and throws an exception if it's
    // out of range; catch stops the exception and handles it.
    try {
        std::println("    cell 64: {}", original.cells.at(64));
    } catch (const std::out_of_range&) {
        std::println("    at(64) threw std::out_of_range");
    }
}

// --- std::span ---------------------------------------------------------------

void demo_spans(const Grid& grid) {
    std::println("std::span:");

    // A fixed-extent span knows its size at compile time, so it stores only
    // a pointer; a dynamic one stores a pointer and a size.
    std::println("    sizeof: span<const bool> {}, span<const bool, 64> {}",
        sizeof(std::span<const bool>), sizeof(std::span<const bool, 64>));

    // One function, three kinds of contiguous storage.
    const bool flags[] = {true, false, true};
    std::println("    alive in a C array: {}", count_alive(flags));
    std::println("    alive in the std::array: {}", count_alive(grid.cells));
    std::println("    alive in row 2: {}", count_alive(grid.row(2)));
    std::println("    alive in the first two rows: {}", count_alive(std::span{grid.cells}.first(2 * Grid::width)));
}

// --- std::string and std::string_view ----------------------------------------

void demo_strings(const Grid& grid) {
    std::println("std::string and std::string_view:");

    // A std::string owns its characters, and frees them when it's destroyed.
    std::string text = render(grid);
    std::println("    rendered {} characters", text.size());

    // A string_view only looks at characters someone else owns. substr on a
    // view makes another view, with no copy.
    const std::string_view view = text;
    const std::string_view row = view.substr(Grid::width + 1, Grid::width);
    std::println("    row 1 is {:?}, first '#' at {}", row, row.find('#'));

    // find returns npos ("no position") when there's no match.
    std::println("    has an '@': {}", row.find('@') != std::string_view::npos);

    // The view has no characters of its own, so it sees changes to text.
    text[Grid::width + 1] = '@';
    std::println("    after changing text: {:?}, has an '@': {}", row, row.find('@') != std::string_view::npos);
}

// --- Running until a generation repeats --------------------------------------

struct Cycle {
    std::size_t start = 0;  // the first generation of the cycle
    std::size_t period = 0; // generations per cycle; 0 if none was found
};

Cycle find_cycle(Grid grid, std::vector<Grid>& history, std::size_t limit) {
    for (std::size_t generation = 0; generation < limit; ++generation) {
        // Use the iterator before push_back: a push_back that grows the
        // vector moves every element, and old iterators would dangle.
        const auto seen = std::ranges::find(history, grid);
        if (seen != history.end()) {
            const auto start = static_cast<std::size_t>(seen - history.begin());
            return {start, generation - start};
        }

        const std::size_t capacity = history.capacity();
        history.push_back(grid);
        if (history.capacity() != capacity) {
            std::println("    size {:>2}: capacity {:>2} -> {:>2}", history.size(), capacity, history.capacity());
        }

        grid = step(grid);
    }

    return {};
}

void print_side_by_side(std::span<const Grid> frames) {
    std::vector<std::string> texts;
    texts.reserve(frames.size());
    for (const Grid& frame : frames) {
        // render returns a temporary string, so push_back moves it in.
        texts.push_back(render(frame));
    }

    for (int y = 0; y < Grid::height; ++y) {
        std::string line = "   ";
        for (const std::string& text : texts) {
            const auto offset = static_cast<std::size_t>(y * (Grid::width + 1));
            line += ' ';
            line += std::string_view{text}.substr(offset, Grid::width);
        }
        std::println("{}", line);
    }
}

// --- The program -------------------------------------------------------------

struct Start {
    std::string_view name;
    std::span<const std::string_view> pattern;
};

int main() {
    demo_c_arrays();
    std::println();
    demo_std_array();
    std::println();

    Grid glider;
    place(glider, patterns::glider, 0, 0);
    demo_spans(glider);
    std::println();
    demo_strings(glider);
    std::println();

    // A C array of structs; each span views one of the pattern arrays.
    const Start starts[] = {
        {"block", patterns::block},
        {"blinker", patterns::blinker},
        {"glider", patterns::glider},
    };

    for (const Start& start : starts) {
        std::println("{}:", start.name);

        Grid grid;
        place(grid, start.pattern, 2, 2);

        std::vector<Grid> history;
        const Cycle cycle = find_cycle(grid, history, 100);
        std::println("    repeats from generation {} every {} generations", cycle.start, cycle.period);

        if (start.name == "glider") {
            // The first five generations: the glider moves one cell diagonally.
            print_side_by_side(std::span{history}.first(5));
        }
    }

    return EXIT_SUCCESS;
}

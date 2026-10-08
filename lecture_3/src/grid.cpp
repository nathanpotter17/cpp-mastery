#include "includes/grid.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

// --- Cells -------------------------------------------------------------------

namespace {

// % keeps the sign of its left side (-1 % 8 is -1), so we add the size
// once more to land in 0 .. size - 1.
int wrap(int value, int size) {
    return (value % size + size) % size;
}

std::size_t index_of(int x, int y) {
    return static_cast<std::size_t>(wrap(y, Grid::height) * Grid::width + wrap(x, Grid::width));
}

} // namespace

bool Grid::alive(int x, int y) const {
    return cells[index_of(x, y)];
}

void Grid::set(int x, int y, bool value) {
    cells[index_of(x, y)] = value;
}

std::span<const bool> Grid::row(int y) const {
    // subspan(offset, count): a smaller view inside the view of all cells.
    return std::span{cells}.subspan(index_of(0, y), width);
}

// --- Counting and placing ----------------------------------------------------

int count_alive(std::span<const bool> cells) {
    // An algorithm instead of a loop: count the elements equal to true.
    return static_cast<int>(std::ranges::count(cells, true));
}

void place(Grid& grid, std::span<const std::string_view> pattern, int x, int y) {
    // Spelled out with iterators, this is what a range-for loop does:
    // begin() points at the first element, end() one past the last.
    for (auto row = pattern.begin(); row != pattern.end(); ++row) {
        const int dy = static_cast<int>(row - pattern.begin());

        for (std::size_t dx = 0; dx < row->size(); ++dx) {
            grid.set(x + static_cast<int>(dx), y + dy, (*row)[dx] == '#');
        }
    }
}

// --- One generation ----------------------------------------------------------

Grid step(const Grid& grid) {
    Grid next;

    for (int y = 0; y < Grid::height; ++y) {
        for (int x = 0; x < Grid::width; ++x) {
            int neighbors = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if ((dx != 0 || dy != 0) && grid.alive(x + dx, y + dy)) {
                        ++neighbors;
                    }
                }
            }

            const bool alive = grid.alive(x, y);
            next.set(x, y, neighbors == 3 || (alive && neighbors == 2));
        }
    }

    return next;
}

// --- Text --------------------------------------------------------------------

std::string render(const Grid& grid) {
    std::string text;
    // One allocation up front: each row is width characters plus a '\n'.
    text.reserve(static_cast<std::size_t>((Grid::width + 1) * Grid::height));

    for (int y = 0; y < Grid::height; ++y) {
        for (const bool cell : grid.row(y)) {
            text += cell ? '#' : '.';
        }
        text += '\n';
    }

    return text;
}

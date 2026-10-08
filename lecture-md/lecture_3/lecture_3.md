# Lecture 3: Contiguous data

By the end of this lecture we'll run Conway's Game of Life on a small board that wraps around at the edges. Each run steps a starting pattern forward until a generation repeats, then reports the cycle it found. The glider prints its first five generations side by side as it crawls diagonally.

The Game of Life is just a board of cells, stored one after another in memory. That makes it the right vehicle for this lecture's subject, *contiguous* data: elements side by side in memory, and the types that own them, view them and walk them:
- C arrays, how they decay to pointers, and `sizeof` traps;
- `std::array`: a fixed-size array with value semantics;
- `std::vector`: a growable array, its capacity, and iterator invalidation;
- `std::span`: a non-owning view of any contiguous elements;
- `std::string` and `std::string_view`, the same owner/view pair for text;
- iterators, and the first standard algorithms;
- exceptions, through `std::array::at`.

Lecture 2's ideas are everywhere here. `std::vector` and `std::string` are RAII owners of heap blocks, and `std::span` and `std::string_view` are the "borrow, don't own" pointers, packaged with a size.

## 3.1 Project layout

### Why
The board and its rules are one concern, the starting patterns another, and the demos and simulation a third, so each gets a file. As with lecture 2's arena, the board's functions are declared in a header and defined in a `.cpp` file.

### How
- **Layout:** the root CMake compiles `src/main.cpp` and `src/grid.cpp`, with `src/` on the include path.
- **The rules of Life:** each cell is alive or dead, and has 8 neighbors. In each generation:
  - a live cell with 2 or 3 live neighbors survives;
  - a dead cell with exactly 3 comes alive;
  - every other cell is dead in the next generation.

### Code
From the repo root:
```bash
mkdir -p lecture_3/src/includes
```

`lecture_3/.clangd`:
```yaml
# clangd reads compile flags from the clang debug build (build-debug-clang.bash).
CompileFlags:
  CompilationDatabase: build/debug-clang
```

## 3.2 The board: `src/includes/grid.h`

### Why
A board is a fixed number of cells known at compile time: 8 × 8. We want to store them in one contiguous block, copy whole boards, compare them, and look at one row without copying it.

### How
- **`std::array<bool, 64>`** is a C array wrapped in a struct. It's exactly 64 `bool`s, side by side, with no heap allocation. Unlike a C array, it copies with `=`, compares with `==`, and knows its own `size()`. `cells{}` value-initializes all 64 to `false`.
- **Two dimensions in one:** the grid is stored *row-major*, row 0's cells first, then row 1's. Cell `(x, y)` is at index `y * width + x`. Nearly every image format, including lecture 5's PNGs, stores pixels this way.
- **`static constexpr` members** belong to the type rather than to each object: `Grid::width` is one compile-time constant, not a member stored in every grid. Being `constexpr`, it can size the `std::array`.
- **`std::span<const bool>`:**
  - **What it is:** a view of contiguous `bool`s, just a pointer to the first element and a count. It owns nothing, so copying one is cheap and never copies the elements.
  - **`row(y)`:** returns a span of one row, a window into `cells`.
  - **`const bool`:** the viewer can read the cells, but not change them.
- **`std::span<const std::string_view>`:** `place` takes its pattern as a span of `string_view`s. Lecture 1 used `const char*` for text; a `std::string_view` is the modern version, a view of characters that also knows its length. Here each row of the pattern is one `string_view`. 3.5 covers strings and views fully.
- **Defaulted comparison:** `bool operator==(const Grid&) const = default;` (C++20) asks the compiler to write `==` for us by comparing every member in order. Here that's the `std::array`, which compares all 64 cells. The simulation uses it to spot a repeated generation.
- **Passing by reference:** `step(const Grid& grid)` reads a grid without copying it, and `place(Grid& grid, ...)` changes the caller's grid. That's lecture 2's guideline: a `const` reference to read, a non-`const` reference to modify.

### Code
`lecture_3/src/includes/grid.h`:
```cpp
#pragma once

#include <array>
#include <span>
#include <string>
#include <string_view>

// --- The board ---------------------------------------------------------------

// A Game of Life board that wraps around: the cells on the right edge are
// neighbors of those on the left edge, and the bottom row of the top row.
struct Grid {
    static constexpr int width = 8;
    static constexpr int height = 8;

    // Row-major order: all of row 0, then all of row 1, and so on. Cell (x, y)
    // is at index y * width + x. The {} sets every cell to false.
    std::array<bool, width * height> cells{};

    // x and y wrap around, so (-1, 0) is the last cell of row 0.
    bool alive(int x, int y) const;
    void set(int x, int y, bool value);

    // A view of one row: no copy, just a pointer to its first cell and a size.
    std::span<const bool> row(int y) const;

    // = default asks the compiler to compare every member: here, all cells.
    bool operator==(const Grid&) const = default;
};

// --- Functions on boards -----------------------------------------------------

// Counts the true values in any contiguous run of bools.
int count_alive(std::span<const bool> cells);

// Copies a pattern onto the grid with its top-left corner at (x, y).
// Each row of the pattern is text: '#' is alive, anything else is dead.
void place(Grid& grid, std::span<const std::string_view> pattern, int x, int y);

// The next generation: a live cell with 2 or 3 live neighbors survives,
// a dead cell with exactly 3 comes alive, and every other cell is dead.
Grid step(const Grid& grid);

// The grid as text, one line per row.
std::string render(const Grid& grid);
```

## 3.3 Starting patterns: `src/includes/patterns.h`

### Why
Patterns are small, fixed tables of text, known at compile time. This is the natural place for C arrays, which are still everywhere in C++ code: in C libraries, in older code, and as the simplest way to write a table.

### How
- **C arrays:** `std::string_view glider[] = {...}` is an array of three `string_view`s. Leaving the size out of `[]` makes the compiler count the initializers.
- **`inline constexpr`:** the arrays are built at compile time, and `inline` makes every file that includes the header share one copy, instead of one per translation unit.
- **A namespace** keeps the pattern names (`patterns::glider`) apart from everything else.
- **The pattern format:** each row is text, with `#` for a live cell and `.` for a dead one, so the patterns look like what they draw.

### Code
`lecture_3/src/includes/patterns.h`:
```cpp
#pragma once

#include <string_view>

// --- Starting patterns -------------------------------------------------------

// C arrays: the element count comes from the initializer. inline lets every
// file that includes this header share one array.
namespace patterns {

inline constexpr std::string_view block[] = {
    "##",
    "##",
};

inline constexpr std::string_view blinker[] = {
    "###",
};

inline constexpr std::string_view glider[] = {
    ".#.",
    "..#",
    "###",
};

} // namespace patterns
```

## 3.4 The rules: `src/grid.cpp`

### Why
This file holds the board's behavior: reading and writing cells, counting, placing patterns, stepping a generation and rendering text. It uses each contiguous type in the role it's best at.

### How
- **The unnamed namespace:** `namespace { ... }` makes `wrap` and `index_of` private to this file. Another `.cpp` file can have its own `wrap` without the linker confusing the two. Helpers that no other file needs belong here.
- **Wrapping around:** the board is a torus, so cell `(-1, 0)` is the last cell of row 0. In C++, `%` keeps the sign of its left operand (`-1 % 8` is `-1`), so `wrap` adds the size once more and takes the remainder again.
- **`std::array`'s `[]`** doesn't check the index; an out-of-range index is undefined behavior. Our indexes come from `index_of`, which can only produce valid ones.
- **`std::span{cells}`** makes a span of the whole array. From a `std::array<bool, 64>` it deduces a *fixed-extent* `std::span<const bool, 64>`, with the count in its type. `subspan(offset, count)` with run-time arguments returns a smaller, *dynamic-extent* view into the same elements, with the count stored at run time.
- **`std::ranges::count(cells, true)`** from `<algorithm>` counts the elements equal to `true`. It works on any range, so `count_alive` takes a span, and accepts anything contiguous. Prefer a named algorithm to a hand-written loop: the name says what the loop would only show.
- **Iterators** are the general way to walk a sequence:
  - **The ends:** `begin()` points at the first element, and `end()` *one past* the last. A loop runs `while (it != end())`.
  - **The operations:** `++` moves to the next element, `*it` is the element, and `it->` reaches its members.
  - **Positions:** subtracting two iterators gives the distance between them, which `place` uses as the row number.
  - **Range-for:** `for (x : range)` is shorthand for essentially this loop; it calls `end()` only once, before the first pass. `place` spells it out once, so we can see it.
- **`step`** builds a new grid from the old one, because every cell must see the *previous* generation's neighbors. It returns the `Grid` by value. Returning a local object doesn't copy it: the compiler builds it directly in the caller's variable (*copy elision*), and if it can't, it moves it.
- **`render`** builds a `std::string`:
  - **The owner:** a `std::string` owns its characters on the heap, and frees them in its destructor: lecture 2's RAII again.
  - **`reserve()`:** allocates room for all 72 characters up front.
  - **`+=`:** appends a character, and the string grows as needed.

### Code
`lecture_3/src/grid.cpp`:
```cpp
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
```

## 3.5 The program: `src/main.cpp`

### Why
`main` first demonstrates each contiguous type on its own, then runs three patterns to their cycles. Each demo is a small function, so its output is labeled and its variables don't leak into the next one.

### How
- **C arrays and decay:**
  - **String literals are arrays:** `"abc"` is a `const char[4]`: its three characters, then a `'\0'` that marks the end of the text for C functions. So `sizeof("abc")` is 4.
  - **`sizeof` vs `std::size`:** `sizeof(patterns::glider)` is the array's size in *bytes*, 48: three `string_view`s of 16 bytes each. `std::size` gives the element count, 3.
  - **Decay:** an array converts implicitly to a pointer to its first element, and the pointer has no idea how many elements follow. `sizeof` on it gives the pointer's size, 8. This is why C functions take a pointer *and* a length.
  - **The fix:** a span is that pointer and length, packaged together, so converting the array to a span keeps the count.
- **`std::array`:**
  - **Value semantics:** `Grid copy = original;` copies all 64 cells, and changing the copy leaves the original alone. A C array can't even be assigned with `=`.
  - **`at()`:** unlike `[]`, `at()` checks the index, and *throws an exception* if it's out of range.
  - **Exceptions:** throwing abandons the rest of the `try` block, destroying its locals as it goes (RAII again), and jumps to a `catch` that matches the exception's type. Here that's `std::out_of_range`, from `<stdexcept>`.
  - **When to throw:** exceptions are for failures the code at hand can't deal with. Lecture 5 introduces `std::expected` for failures the caller is expected to handle.
- **`std::span`:**
  - **Fixed vs dynamic extent:** `std::span<const bool, 64>` has its size in its type, so it stores only a pointer: 8 bytes against 16 for a dynamic span.
  - **One function, any storage:** `count_alive` accepts a C array, a `std::array`, a row view, or a `first(n)` sub-view, all as a `std::span<const bool>`. It would accept a `std::vector<bool>` too, except that `std::vector<bool>` is a special case: it packs its elements into bits, so it isn't really an array of `bool`s, and can't be viewed by a span.
- **`std::string` and `std::string_view`:**
  - **Owner and view:** `text` owns its characters, and `view` and `row` only look at them. `substr` on a view makes another view, with no copy.
  - **Not found:** `find` returns the position of a match, or `std::string_view::npos` ("no position") if there isn't one.
  - **A view sees changes:** after `text` changes, `row` shows the change, because it has no characters of its own.
  - **Dangling views:** a view must not outlive the string it views. `std::string_view v = render(grid);` compiles (clang warns about it), but the temporary string is destroyed at the end of that statement, and `v` dangles: lecture 2's dangling pointer, in a new form.
- **`std::vector<Grid>`** holds every generation of a run:
  - **Size and capacity:** `size()` is how many elements it holds, and `capacity()` how many fit in its current heap block. When `push_back` finds the block full, it allocates a bigger one (our standard library doubles it), moves every element across, and frees the old block. The output shows each of these reallocations.
  - **`reserve(n)`** allocates room for `n` elements up front, so pushing up to `n` never reallocates. `print_side_by_side` uses it.
  - **Iterator invalidation:** a reallocation moves every element, so iterators, pointers and references into the old block all dangle. `find_cycle` finishes using the iterator `seen`, converting it to an index, before it calls `push_back`. That order is deliberate.
  - **`push_back(render(frame))`** receives a temporary string, so it *moves* it into the vector instead of copying. `emplace_back(args...)` goes one step further, and constructs the element in place from constructor arguments.
- **Finding a cycle:**
  - **The search:** `std::ranges::find(history, grid)` compares the new generation with every earlier one, using `Grid`'s defaulted `==`, and returns an iterator to the first match, or `history.end()`.
  - **The index:** subtracting `history.begin()` turns the iterator into an index.
  - **The period:** the cycle's length is the current generation minus that index.
- **The table of starts** is a C array of `Start` structs. Each holds a `std::span` of one pattern array, so patterns of different sizes fit in one table. `std::span{history}.first(5)` views the glider's first five generations without copying them.

### Code
`lecture_3/src/main.cpp`:
```cpp
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
```

## 3.6 Build and run

```bash
./build-scripts-linux/build-debug-clang.bash lecture_3 && ./lecture_3/build/debug-clang/lecture_3
```

```text
C arrays:
    sizeof("abc") = 4: the literal is a const char[4] ending in '\0'
    glider: 3 rows, 48 bytes
    as a pointer: 8 bytes, first row ".#.", row count lost
    as a span: 3 rows, last row "###"

std::array:
    copy == original: false, alive 4 vs 3
    at(64) threw std::out_of_range

std::span:
    sizeof: span<const bool> 16, span<const bool, 64> 8
    alive in a C array: 2
    alive in the std::array: 5
    alive in row 2: 3
    alive in the first two rows: 2

std::string and std::string_view:
    rendered 72 characters
    row 1 is "..#.....", first '#' at 2
    has an '@': false
    after changing text: "@.#.....", has an '@': true

block:
    size  1: capacity  0 ->  1
    repeats from generation 0 every 1 generations
blinker:
    size  1: capacity  0 ->  1
    size  2: capacity  1 ->  2
    repeats from generation 0 every 2 generations
glider:
    size  1: capacity  0 ->  1
    size  2: capacity  1 ->  2
    size  3: capacity  2 ->  4
    size  5: capacity  4 ->  8
    size  9: capacity  8 -> 16
    size 17: capacity 16 -> 32
    repeats from generation 0 every 32 generations
    ........ ........ ........ ........ ........
    ........ ........ ........ ........ ........
    ...#.... ........ ........ ........ ........
    ....#... ..#.#... ....#... ...#.... ....#...
    ..###... ...##... ..#.#... ....##.. .....#..
    ........ ...#.... ...##... ...##... ...###..
    ........ ........ ........ ........ ........
    ........ ........ ........ ........ ........
```

Reading the output:
- **The decayed pointer** knows its first row, but not that there are three.
- **The copy** has 4 live cells and the original 3: copies of a `std::array` are independent.
- **The spans:** a fixed-extent span is half the size of a dynamic one. `count_alive` counted a C array, the whole board, one row, and the first two rows, all through the same parameter type.
- **The vector's capacity** doubles each time it fills: 1, 2, 4, 8, 16, 32. Each of those lines was a reallocation that moved every grid stored so far. Had we known the count, `reserve` would have made it one allocation.
- **The block** never changes, so it repeats with a period of 1; the blinker flips between two shapes, a period of 2. The glider returns to its starting shape every 4 generations, one cell further down and to the right. On an 8 × 8 board that wraps around, it's back in exactly the same place after 8 × 4 = 32 generations.

In lecture 4 we name our types precisely with aliases, and lean much harder on the standard library: optional values, parsing, maps and ranges.

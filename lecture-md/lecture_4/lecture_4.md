# Lecture 4: Type aliases and the standard library

By the end of this lecture we'll have a small batch calculator. It reads expressions like `"12 + 12"` from text, evaluates them with 32-bit arithmetic that can't silently overflow, prints every result in decimal, hex and binary, and summarizes the batch. The calculator is only the excuse. The lecture has two threads:

- **Type aliases.** We give types names with `typedef` and `using`, then go further: function pointer aliases, alias templates, member aliases, and the aliases the standard library itself is built on, like `std::int32_t`, `std::size_t` and `std::make_unsigned_t`. We'll also see what an alias *can't* do, and use an `enum class` with a chosen underlying type where we need a real new type.
- **The standard library.** Instead of writing helpers ourselves, we use the library's vocabulary types and algorithms. New in this lecture: `std::optional`, `std::from_chars`, `std::map`, `std::function`, `<bit>`, `std::numeric_limits`, `std::in_range`, and `std::ranges` projections, views and folds. Lecture 3's arrays, vectors, spans and string views are used throughout.

## 4.1 Project layout

### Why
The program is split by concern, one header per idea, so each section of this lecture introduces exactly one file.

### How
- **The layout is lecture 2's:** `src/main.cpp`, with headers in `src/includes/`.
- **Header-only:** this time every function except `main.cpp`'s own lives in a header. Each one is `inline`, `constexpr` or a template, and any of those may be defined in a header that several `.cpp` files include, without breaking the one-definition rule. Generic code has to live in headers anyway, and most of this lecture is generic.
- **Nothing to download:** everything we use ships with the compiler.

### Code
From the repo root:
```bash
mkdir -p lecture_4/src/includes
```

`lecture_4/.clangd`:
```yaml
# clangd reads compile flags from the clang debug build (build-debug-clang.bash).
CompileFlags:
  CompilationDatabase: build/debug-clang
```

## 4.2 Naming types: `src/includes/types.h`

### Why
A program that does careful integer arithmetic has to say exactly which integer it means, and says it a lot. `std::int32_t` is precise but long; `int` is short but its size isn't fixed by the language. Type aliases give us short, precise names, and let a whole program change a type by editing one line.

### How
- **`typedef` versus `using`.** Both declare a new *name* for an existing type:
  - `typedef std::int32_t i32;` is the C spelling. The name sits where a variable's name would, which gets hard to read for complicated types.
  - `using i32 = std::int32_t;` is the C++11 spelling: name on the left, type on the right, like an assignment. Prefer it in new code; it also works for templates, which `typedef` can't do.

  We declare `i32` both ways. Redeclaring an alias is legal when it names the same type, so the second line compiling proves the two spellings mean the same thing.
- **The standard integer names are aliases too.** `<cstdint>` declares `std::int32_t`, `std::int64_t`, `std::uint32_t` and friends as aliases of whichever built-in type has that exact size on the platform. `std::int32_t` is `int` almost everywhere, but `std::int64_t` is `long` on 64-bit Linux and `long long` on Windows, where `long` is only 4 bytes. Code that says `std::int64_t` doesn't have to care. `std::size_t` (the type of `sizeof`) and `std::ptrdiff_t` (the type of a pointer difference) are aliases as well. Our `i32`, `i64` and `u32` are aliases of aliases, the short names game engines commonly use.
- **Checking at compile time.** `static_assert` stops the build if its condition is false. `std::is_same_v<A, B>` is `true` only if `A` and `B` are the same type, and for an alias it always is.
- **An alias is not a new type.** That makes aliases free, but it also means they can't catch mistakes. With `using Meters = double;` and `using Seconds = double;`, the compiler happily adds meters to seconds. When the difference matters, we need a real type, such as a small `struct`, an `enum class` (4.3), or a library type like `std::chrono::seconds`. (That one is itself an alias, for a `std::chrono::duration` of a 64-bit integer, but `duration` is a class template, so seconds and meters can't mix.) We'll see a real consequence of "an alias is the same type" in 4.7.
- **Aliasing library types.** `using Result = std::optional<i32>;` names the outcome of one calculation. A `std::optional<i32>` holds either an `i32` or nothing (`std::nullopt`), which is how a calculation says it failed. Writing `Result` everywhere keeps signatures short, and says what the value *means*.
- **Function pointer aliases.** A function's address can be stored in a pointer, and the pointer called like the function. The raw type is awkward: a function taking two `i32`s and returning a `Result` has the pointer type `Result (*)(i32, i32)`, and the `typedef` for it hides the new name in the middle. With `using`, `BinaryOp` reads left to right.
- **Alias templates.** An alias can have template parameters, which `typedef` can't. `Unsigned<T>` is the unsigned type with the same size as `T`. It's a thin wrapper around `std::make_unsigned_t<T>`, which is itself an alias template: the type trait `std::make_unsigned<T>` stores its answer in a member named `type`, and `_t` saves us writing `typename std::make_unsigned<T>::type`. The `_v` helpers like `std::is_same_v` are the same idea for values.
- **`std::integral`** is a concept from `<concepts>`. `template <std::integral T>` accepts only integer types, so `Unsigned<double>` is an error at the point of use instead of deep inside the library.
- **`bits()`** reinterprets a value as its unsigned counterpart. A negative number keeps its bit pattern, so hex and binary output shows the two's complement bits (`-1` is `0xffffffff`) instead of a minus sign.

### Code
`lecture_4/src/includes/types.h`:
```cpp
#pragma once

#include <concepts>
#include <cstdint>
#include <optional>
#include <type_traits>

// --- Two spellings, one meaning ----------------------------------------------

// C's typedef: the new name sits where a variable's name would.
typedef std::int32_t i32;

// C++11's alias declaration: the name on the left, the type on the right.
// Redeclaring an alias is allowed when it names the same type, so this line
// compiling proves both spellings mean exactly the same thing.
using i32 = std::int32_t;

using i64 = std::int64_t;
using u32 = std::uint32_t;

// An alias is another name, not a new type: the compiler can't tell them apart.
static_assert(std::is_same_v<i32, std::int32_t>);
static_assert(sizeof(i32) == 4 && sizeof(i64) == 8);

// --- Aliases for standard library types --------------------------------------

// A calculation either produces an i32 or fails (overflow, division by zero).
using Result = std::optional<i32>;

// --- Function pointer aliases ------------------------------------------------

// The typedef spelling hides the name inside the type:
//     typedef Result (*BinaryOp)(i32, i32);
using BinaryOp = Result (*)(i32, i32);

// --- Alias templates ---------------------------------------------------------

// The standard library's _t helpers are alias templates too:
// std::make_unsigned_t<T> is short for typename std::make_unsigned<T>::type.
template <std::integral T>
using Unsigned = std::make_unsigned_t<T>;

static_assert(std::is_same_v<Unsigned<i32>, u32>);

// The same bits, read as unsigned. -1 becomes 0xffffffff, so hex and binary
// output shows the two's complement pattern instead of a minus sign.
template <std::integral T>
constexpr Unsigned<T> bits(T value) {
    return static_cast<Unsigned<T>>(value);
}
```

## 4.3 Checked arithmetic: `src/includes/operations.h`

### Why
Signed integer overflow is *undefined behavior* in C++: `2147483647 + 1` on an `i32` isn't guaranteed to wrap around, or to do anything in particular. The optimizer is allowed to assume it never happens. A calculator has to detect it instead, and say "no result". We also want the operators in a table, so finding and adding them is data, not a chain of `if`s.

### How
- **Widening:** both operands are converted to `i64` before the operation. The product of two `i32`s needs at most 63 bits, so no `i32` operation can overflow an `i64`. `i64{a}` is a braced conversion: it would refuse to compile if it could lose information, which `i32` to `i64` never does.
- **`narrow()`** keeps the wide result only if it fits back into an `i32`. `std::in_range<i32>(wide)` from `<utility>` checks exactly that, and is correct even when mixing signed and unsigned types, where hand-written comparisons often aren't.
- **Two surprising overflows:**
  - `-2147483648 / -1` is `2147483648`, one more than the largest `i32`. `narrow()` catches it.
  - Division by zero is undefined even in `i64`, so `/` and `%` check for it first. `Result{}` is an empty optional, the same as `std::nullopt`.
- **`enum class Symbol : char`:** lecture 1's `enum class` used the default underlying type, `int`. Here we choose it ourselves, after the `:`, and give each value the character that writes it.
  - **Every `char` is a `Symbol`:** with a fixed underlying type, every value of that type is a valid `Symbol`, named or not. So `static_cast<Symbol>(c)` is safe for any `char` `c`, even one like `'^'` that names no symbol.
  - **A real new type:** unlike an alias, `Symbol` won't mix with `char` by accident. `Symbol s = '+';` doesn't compile; only the explicit cast converts.
  - **Its underlying type:** `std::underlying_type_t<Symbol>` is the alias template that names it, here `char`.
- **The table:**
  - **The element type:** `Operation` holds a `Symbol`, a name and a `BinaryOp`.
  - **The functions:** a lambda with no captures converts implicitly to a plain function pointer, so each lambda fits a `BinaryOp` field.
  - **The array type:** with *class template argument deduction* (CTAD), `std::array operations{...}` works out `std::array<Operation, 5>` from the initializers.
  - **`inline constexpr`:** the table is built at compile time, and every file that includes the header shares one copy.
- **`find_operation()`:**
  - **The search:** `std::ranges::find(operations, static_cast<Symbol>(symbol), &Operation::symbol)` uses a *projection*. Before comparing, find applies `&Operation::symbol` to each element, so it compares symbols, not whole `Operation`s, without a hand-written lambda.
  - **The result:** `find` returns an iterator, equal to `operations.end()` when nothing matched. We turn it into a pointer that is `nullptr` on a miss.

### Code
`lecture_4/src/includes/operations.h`:
```cpp
#pragma once

#include "includes/types.h"

#include <algorithm>
#include <array>
#include <string_view>
#include <type_traits>
#include <utility>

// --- Checked arithmetic ------------------------------------------------------

// Every i32 operation is done in i64, where it can't overflow, and the result
// is kept only if it fits back into an i32.
constexpr Result narrow(i64 wide) {
    if (!std::in_range<i32>(wide)) {
        return std::nullopt;
    }

    return static_cast<i32>(wide);
}

// --- Operator symbols --------------------------------------------------------

// An enum class with a fixed underlying type: every Symbol is stored as a char.
// Every char value is then a valid Symbol, even one without a name, so any
// char may be static_cast to a Symbol. Nothing converts back implicitly.
enum class Symbol : char {
    add = '+',
    subtract = '-',
    multiply = '*',
    divide = '/',
    remainder = '%',
};

// underlying_type_t is an alias template, like make_unsigned_t.
static_assert(std::is_same_v<std::underlying_type_t<Symbol>, char>);

// --- The operation table -----------------------------------------------------

struct Operation {
    Symbol symbol;
    std::string_view name;
    BinaryOp apply;
};

// Captureless lambdas convert to plain function pointers, so each fits BinaryOp.
// CTAD deduces std::array<Operation, 5> from the initializers.
inline constexpr std::array operations{
    Operation{Symbol::add, "add", [](i32 a, i32 b) { return narrow(i64{a} + b); }},
    Operation{Symbol::subtract, "subtract", [](i32 a, i32 b) { return narrow(i64{a} - b); }},
    Operation{Symbol::multiply, "multiply", [](i32 a, i32 b) { return narrow(i64{a} * b); }},
    Operation{Symbol::divide, "divide", [](i32 a, i32 b) { return b == 0 ? Result{} : narrow(i64{a} / b); }},
    Operation{Symbol::remainder, "remainder", [](i32 a, i32 b) { return b == 0 ? Result{} : narrow(i64{a} % b); }},
};

// The projection &Operation::symbol makes find compare each element's symbol.
constexpr const Operation* find_operation(char symbol) {
    const auto found = std::ranges::find(operations, static_cast<Symbol>(symbol), &Operation::symbol);
    return found == operations.end() ? nullptr : &*found;
}
```

## 4.4 Parsing text: `src/includes/expression.h`

### Why
Expressions arrive as text, so we need to turn `"-7 * 6"` into two numbers and an `Operation`. Parsing is where C++ programs traditionally reach for `std::stoi` (throws on bad input), `atoi` (silently returns 0), or a `std::istringstream` (slow, locale-dependent). The standard library has a better tool for exactly this job.

### How
- **`std::string_view`,** from lecture 3, is the parser's natural input. Taking one by value is as cheap as passing two integers, and shrinking it with `remove_prefix()` or `substr()` never copies or allocates.
- **Member aliases.** Standard library classes publish their related types as members. `std::string_view::size_type` is the type of a position (an alias of `std::size_t`), and lecture 3's `std::string_view::npos` is a member constant of that type, the value `find_first_not_of` returns when it finds nothing.
- **`std::from_chars`** from `<charconv>` reads a number from a range of characters:
  - **The rules:** it doesn't skip whitespace, never throws, never allocates, and ignores the locale.
  - **The result:** it returns a `std::from_chars_result` struct, whose `ptr` and `ec` members a *structured binding* unpacks into `end` (one past the last character it used) and `error`.
  - **Errors:** `error` is a `std::errc`, and `std::errc{}` means success. `"abc"` gives `invalid_argument`, and `"99999999999"` gives `result_out_of_range`, because it doesn't fit in an `i32`.
- **Two size aliases meet.** Subtracting two pointers gives a `std::ptrdiff_t`, which is signed because the difference could be negative. `remove_prefix` wants a `std::size_t`, which is unsigned. Naming both types in the code makes the change of type visible, and the explicit `static_cast` marks the signed-to-unsigned conversion as intended.
- **`take_number`** takes the view by reference, so it can consume the number it read from the caller's view.
- **`parse`** reads number, operator, number, and accepts spaces anywhere between them. It returns `std::nullopt` for anything else: a missing number, an operator that isn't in the table, or text left over at the end.
- **`Expression`** keeps a pointer to its `Operation`, so `evaluate()` is one call through the `BinaryOp`.

### Code
`lecture_4/src/includes/expression.h`:
```cpp
#pragma once

#include "includes/operations.h"
#include "includes/types.h"

#include <charconv>
#include <cstddef>
#include <optional>
#include <string_view>
#include <system_error>

// --- Expressions -------------------------------------------------------------

// "lhs op rhs", with the operator already looked up in the table.
struct Expression {
    i32 lhs = 0;
    const Operation* operation = nullptr;
    i32 rhs = 0;

    Result evaluate() const {
        return operation->apply(lhs, rhs);
    }
};

// --- Parsing helpers ---------------------------------------------------------

// Drops leading spaces. npos ("no position") is what find_* returns on a miss.
inline std::string_view skip_spaces(std::string_view text) {
    const std::string_view::size_type start = text.find_first_not_of(' ');
    return start == std::string_view::npos ? std::string_view{} : text.substr(start);
}

// Reads an i32 from the front of text, and removes it from text.
// from_chars doesn't throw or allocate; it reports where it stopped and why.
inline std::optional<i32> take_number(std::string_view& text) {
    text = skip_spaces(text);

    i32 value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);

    // No digits at all, or a number too big for an i32.
    if (error != std::errc{}) {
        return std::nullopt;
    }

    // Subtracting pointers gives a std::ptrdiff_t, a signed alias;
    // string_view sizes are std::size_t, an unsigned one.
    const std::ptrdiff_t consumed = end - text.data();
    text.remove_prefix(static_cast<std::size_t>(consumed));
    return value;
}

// --- The parser --------------------------------------------------------------

// Accepts "12 + 12", "-7*6" and so on: number, operator, number.
inline std::optional<Expression> parse(std::string_view text) {
    const std::optional<i32> lhs = take_number(text);
    if (!lhs) {
        return std::nullopt;
    }

    text = skip_spaces(text);
    if (text.empty()) {
        return std::nullopt;
    }

    const Operation* operation = find_operation(text.front());
    if (!operation) {
        return std::nullopt;
    }
    text.remove_prefix(1);

    const std::optional<i32> rhs = take_number(text);
    if (!rhs || !skip_spaces(text).empty()) {
        return std::nullopt;
    }

    return Expression{*lhs, operation, *rhs};
}
```

## 4.5 Summaries: `src/includes/stats.h`

### Why
After the batch, we want the minimum, median, maximum and total of the results. We'd like to write that once, for any integer type and any kind of range, rather than for a `std::vector<i32>` only. Generic code is where aliases earn their keep: inside a template we don't know the types by name, so we *compute* them.

### How
- **`std::conditional_t<Condition, A, B>`** is an alias template that picks `A` if `Condition` is true, else `B`, all at compile time. `Wider<T>` uses it to pick `std::int64_t` for signed values and `std::uint64_t` for unsigned ones. A total of many `i32`s can exceed the `i32` range even when every value fits, so the total gets its own, wider type.
- **Member aliases of our own.** `Summary<T>` declares `value_type` and `total_type`, following the naming the standard containers use (`std::vector<int>::value_type` is `int`). Code holding a `Summary` can ask `decltype(summary)::total_type` without knowing how it was chosen.
- **`summarize` takes any range:**
  - **The parameter:** `std::ranges::input_range` accepts anything iterable: a `std::vector`, a `std::array`, or a view.
  - **The element type:** `std::ranges::range_value_t<R>` is another alias template, and the `requires` clause demands it be an integer.
  - **A local alias:** inside the function, `using T = ...;` names the element type, so the rest of the body reads naturally.
  - **`R&&` and `std::forward`:** `R&&` on a template parameter is a *forwarding reference*. It binds to anything, and `std::forward` passes the argument on exactly as it came in.
- **The algorithms:**
  - `std::ranges::to<std::vector>()` (C++23) collects any range into a new `std::vector`, deducing the element type. Sorting our own copy leaves the caller's range alone.
  - `std::ranges::sort` sorts it in place, so the minimum is `front()`, the maximum is `back()`, and the median is in the middle. With an even count we take the upper of the two middle values.
  - `std::ranges::fold_left` (C++23) combines the elements left to right, starting from `Wider<T>{0}`, with `std::plus{}`, the standard library's function object for `+`. The type of the start value is the type the sum is computed in.

### Code
`lecture_4/src/includes/stats.h`:
```cpp
#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

// --- Choosing a type at compile time -----------------------------------------

// std::conditional_t picks the first type if the condition holds, else the
// second. Totals use the widest type of the same signedness as the values.
template <std::integral T>
using Wider = std::conditional_t<std::is_signed_v<T>, std::int64_t, std::uint64_t>;

// --- The summary -------------------------------------------------------------

template <std::integral T>
struct Summary {
    // Member aliases, named the way the standard containers name theirs.
    using value_type = T;
    using total_type = Wider<T>;

    std::size_t count = 0;
    value_type min{};
    value_type median{};
    value_type max{};
    total_type total{};
};

// Accepts any range of integers: a vector, an array, or a view over either.
template <std::ranges::input_range R>
    requires std::integral<std::ranges::range_value_t<R>>
auto summarize(R&& values) {
    // range_value_t is an alias template: the element type of R.
    using T = std::ranges::range_value_t<R>;

    // Our own sorted copy, whatever kind of range we were given.
    auto sorted = std::forward<R>(values) | std::ranges::to<std::vector>();
    std::ranges::sort(sorted);

    Summary<T> summary;
    if (sorted.empty()) {
        return summary;
    }

    summary.count = sorted.size();
    summary.min = sorted.front();
    summary.median = sorted[sorted.size() / 2];
    summary.max = sorted.back();
    summary.total = std::ranges::fold_left(sorted, Wider<T>{0}, std::plus{});
    return summary;
}
```

## 4.6 Callbacks: `src/includes/batch.h`

### Why
Running a batch is always the same loop: parse each input, then evaluate it. What to *do* with each result varies: print it, count it, collect it. We want to write the loop once, and let the caller supply the varying part as a function. The caller's function will need to update the caller's own variables, and that's more than a function pointer can do.

### How
- **Why not a function pointer:** a lambda that *captures* variables carries them with it. A function pointer is just an address with nowhere to keep them, so only captureless lambdas convert to one (4.3).
- **`std::function<R(Args...)>`** from `<functional>` holds *any* callable with that signature: a function, a function object, or a lambda with captures.
  - **The cost:** it may allocate to store a large callable, and every call goes through an extra indirection.
  - **When to use it:** when a callable has to be *stored*, or passed across a non-template interface like this one. A template parameter would avoid the cost, but then `run_batch` would need to be a template.
- **`RowHandler`** names the `std::function` type. The parameter names in its signature (`input`, `expression`, `value`) are only documentation, but they tell a reader what each argument is.
- **`run_batch`** takes the inputs as a `std::span`, so any contiguous sequence of `string_view`s works, and the handler by `const` reference. The caller's lambda is converted into a temporary `std::function` once, and the reference stops that object being copied again.

### Code
`lecture_4/src/includes/batch.h`:
```cpp
#pragma once

#include "includes/expression.h"
#include "includes/types.h"

#include <functional>
#include <optional>
#include <span>
#include <string_view>

// --- Callbacks ---------------------------------------------------------------

// Called once per input with what parse and evaluate made of it. A
// std::function holds any callable with this signature: a function, or a
// lambda that captures variables, which a plain function pointer can't hold.
using RowHandler = std::function<void(std::string_view input, const std::optional<Expression>& expression, Result value)>;

// --- Running a batch ---------------------------------------------------------

// Parses and evaluates every input, and leaves what to do with the results to
// the caller's handler.
inline void run_batch(std::span<const std::string_view> inputs, const RowHandler& handle) {
    for (const std::string_view input : inputs) {
        const std::optional<Expression> expression = parse(input);
        const Result value = expression ? expression->evaluate() : Result{};
        handle(input, expression, value);
    }
}
```

## 4.7 The program: `src/main.cpp`

### Why
`main` puts the aliases on display, then runs the batch and reports on it. Almost every line is a call into the standard library, and the format strings do all of the layout.

### How
- **Measuring the aliases.** `print_limits<T>` prints a type's size and range, with `std::numeric_limits<T>::min()` and `max()` from `<limits>`. The template parameter can be any integer type, alias or not.
- **An alias's real type shows through.** `std::int8_t` is an alias for `signed char`. iostream's `<<` has an overload for `signed char` that prints it as a *character*, so `std::int8_t{65}` comes out as `A`. `std::print` formats `signed char` as a number. The alias didn't create a new type, so overload resolution only ever saw `signed char`. It's lecture 1's `type_name` overloads at work, in the standard library.
- **The price of `std::function`:** `sizeof(BinaryOp)` is one pointer. `sizeof(RowHandler)` is several, because a `std::function` stores small callables inside itself.
- **Printing an `enum class`:** `std::print` has no formatter for our `Symbol`, so `std::to_underlying` (C++23, `<utility>`) converts it back to its `char`. It's shorthand for `static_cast<std::underlying_type_t<Symbol>>(symbol)`.
- **The inputs:**
  - **The array:** `std::to_array<std::string_view>({...})` builds a `std::array` and deduces its size, so adding an input is one line.
  - **No copies:** string literals live in the program for its whole run, so views of them never dangle.
  - **The cases:** the list covers normal results, negative results, the edges of the `i32` range, all the overflow cases from 4.3, and two inputs `parse` rejects.
- **The batch:** `run_batch` calls our lambda once per input, and the lambda prints a row:
  - **`[&]`** captures every local variable the lambda uses *by reference*, so it updates `main`'s own `results` and `uses`. With a capture by value (`[=]`), the lambda would hold `const` copies, so `push_back` wouldn't even compile. The capture is why the lambda can't be a `BinaryOp`-style function pointer, and needs `std::function`.
  - **`return`** inside the lambda ends this call of the lambda, so it moves on to the next input, much like `continue` in a loop.
  - **`++uses[name]`:** a `std::map`'s `operator[]` inserts a zero the first time it sees a key, so counting needs no setup. A map keeps its keys sorted.
  - **`results`** keeps every parsed expression's `Result`, including the empty ones.
- **Counting bits:** `std::popcount` from `<bit>` counts the 1 bits in a value. Like most of `<bit>`'s bit-counting functions, it only accepts unsigned types, because a sign makes "the bits" ambiguous. That's what `bits()` and `Unsigned<T>` are for. `<bit>` also has the library version of lecture 2's power-of-two test: `std::has_single_bit(alignment)`.
- **Format specifications** go after the `:` in a `{}`:
  - `{:<20}` left-aligns in 20 columns, and `{:>12}` right-aligns.
  - `{:?}` (C++23) prints a string quoted and escaped, so the spaces in `"  8-3 "` show.
  - `{:#010x}` is hex with a `0x` prefix (`#`), padded with zeros (`0`) to 10 characters including the prefix.
  - `{:032b}` is binary padded to 32 digits.
  - `{:.2f}` is a floating point number with two decimals.
  - `{{` prints a literal `{`.
- **The summary:**
  - **Counting failures:** `std::ranges::count_if` counts the empty results.
  - **A lazy view:** `results | std::views::filter(...) | std::views::transform(...)` keeps only the results that have a value, and unwraps each one. The view computes nothing when it is created; each element is filtered and unwrapped as the reader reaches it. Here the readers are `summarize`, which copies the elements into its own vector, and `std::println`.
  - **Printing ranges:** `std::println("{}", values)` (C++23) formats any range as `[a, b, c]`, and a map as `{key: value, ...}`.
  - **The total's type:** `static_assert` confirms that the total of `i32`s was computed as an `i64`.

### Code
`lecture_4/src/main.cpp`:
```cpp
#include "includes/batch.h"
#include "includes/expression.h"
#include "includes/operations.h"
#include "includes/stats.h"
#include "includes/types.h"

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <print>
#include <ranges>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// --- The aliases, measured ---------------------------------------------------

// numeric_limits is a class template: each integer type gets its own min/max.
template <std::integral T>
void print_limits(std::string_view name) {
    std::println("{:<12} {:>5}  {:>20}  {}", name, sizeof(T), std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
}

void print_types() {
    std::println("{:<12} {:>5}  {:>20}  {}", "type", "bytes", "min", "max");
    print_limits<std::int8_t>("std::int8_t");
    print_limits<i32>("i32");
    print_limits<u32>("u32");
    print_limits<i64>("i64");
    print_limits<std::size_t>("std::size_t");

    // std::int8_t is an alias for signed char, and iostream prints chars as
    // characters. std::format treats signed char as a number.
    std::cout << "\nstd::int8_t{65} through iostream:   " << std::int8_t{65} << '\n';
    std::println("std::int8_t{{65}} through std::print: {}", std::int8_t{65});

    // A function pointer is one address; a std::function also has room to
    // store a lambda's captured variables.
    std::println("\nsizeof(BinaryOp) = {}, sizeof(RowHandler) = {}", sizeof(BinaryOp), sizeof(RowHandler));

    // An enum class has no formatter, so we print its underlying char.
    std::print("operations:");
    for (const Operation& operation : operations) {
        std::print(" {} {}", std::to_underlying(operation.symbol), operation.name);
    }
    std::println();
}

// --- The program -------------------------------------------------------------

int main() {
    print_types();

    // std::to_array deduces the size; the element type we give explicitly.
    constexpr auto inputs = std::to_array<std::string_view>({
        "12 + 12",
        "7 * 6",
        "100 / 7",
        "-100 % 7",
        "  8-3 ",
        "-1 + 0",
        "2147483647 + 0",
        "46340 * 46340",
        "46341 * 46341",
        "2147483647 + 1",
        "-2147483648 / -1",
        "10 / 0",
        "3 ^ 4",
        "99999999999 + 1",
    });

    std::vector<Result> results;
    results.reserve(inputs.size());

    // Operation name -> times used. A std::map keeps its keys sorted.
    std::map<std::string_view, int> uses;

    // {:?} prints a string quoted and escaped, so the spaces in "  8-3 " show.
    std::println("\n{:<20} {:>12}  {:<10}  {:<32}  {}", "expression", "decimal", "hex", "binary", "ones");

    // The lambda captures results and uses by reference ([&]), so it can
    // fill them in. That capture is why it needs a std::function.
    run_batch(inputs, [&](std::string_view input, const std::optional<Expression>& expression, Result value) {
        if (!expression) {
            std::println("{:<20?} {:>12}", input, "bad input");
            return;
        }

        ++uses[expression->operation->name];
        results.push_back(value);

        if (!value) {
            std::println("{:<20?} {:>12}", input, "no result");
            return;
        }

        // std::popcount counts the 1 bits. It only accepts unsigned types,
        // so bits() is what makes it usable on an i32.
        const u32 pattern = bits(*value);
        std::println("{:<20?} {:>12}  {:#010x}  {:032b}  {:>4}", input, *value, pattern, pattern, std::popcount(pattern));
    });

    // --- The summary ---------------------------------------------------------

    const auto failed = std::ranges::count_if(results, [](const Result& result) { return !result; });

    // A lazy view: nothing is filtered or copied until something reads it.
    auto values = results
        | std::views::filter([](const Result& result) { return result.has_value(); })
        | std::views::transform([](const Result& result) { return *result; });

    const auto summary = summarize(values);
    static_assert(std::is_same_v<decltype(summary)::total_type, i64>);

    std::println("\n{} parsed, {} without a result", results.size(), failed);
    std::println("values: {}", values);
    std::println("min {}, median {}, max {}", summary.min, summary.median, summary.max);
    std::println("total {}, mean {:.2f}", summary.total, static_cast<double>(summary.total) / static_cast<double>(summary.count));
    std::println("uses:   {}", uses);

    return EXIT_SUCCESS;
}
```

## 4.8 Build and run

```bash
./build-scripts-linux/build-debug-clang.bash lecture_4 && ./lecture_4/build/debug-clang/lecture_4
```

```text
type         bytes                   min  max
std::int8_t      1                  -128  127
i32              4           -2147483648  2147483647
u32              4                     0  4294967295
i64              8  -9223372036854775808  9223372036854775807
std::size_t      8                     0  18446744073709551615

std::int8_t{65} through iostream:   A
std::int8_t{65} through std::print: 65

sizeof(BinaryOp) = 8, sizeof(RowHandler) = 32
operations: + add - subtract * multiply / divide % remainder

expression                decimal  hex         binary                            ones
"12 + 12"                      24  0x00000018  00000000000000000000000000011000     2
"7 * 6"                        42  0x0000002a  00000000000000000000000000101010     3
"100 / 7"                      14  0x0000000e  00000000000000000000000000001110     3
"-100 % 7"                     -2  0xfffffffe  11111111111111111111111111111110    31
"  8-3 "                        5  0x00000005  00000000000000000000000000000101     2
"-1 + 0"                       -1  0xffffffff  11111111111111111111111111111111    32
"2147483647 + 0"       2147483647  0x7fffffff  01111111111111111111111111111111    31
"46340 * 46340"        2147395600  0x7ffea810  01111111111111101010100000010000    18
"46341 * 46341"         no result
"2147483647 + 1"        no result
"-2147483648 / -1"      no result
"10 / 0"                no result
"3 ^ 4"                 bad input
"99999999999 + 1"       bad input

12 parsed, 4 without a result
values: [24, 42, 14, -2, 5, -1, 2147483647, 2147395600]
min -2, median 24, max 2147483647
total 4294879329, mean 536859916.12
uses:   {"add": 4, "divide": 3, "multiply": 3, "remainder": 1, "subtract": 1}
```

Reading the output:
- **The type table** shows the aliases as the compiler sees them. `std::size_t` is 8 bytes on a 64-bit platform and 4 on a 32-bit one. On 64-bit Windows, `unsigned long` is only 4 bytes while `std::size_t` is 8, which is exactly why code that means "a size" says `std::size_t` instead of `unsigned long`.
- **`std::int8_t{65}`** prints as `A` through iostream and as `65` through `std::print`, because the alias is just `signed char`.
- **`sizeof(RowHandler)`** is 32 bytes against `BinaryOp`'s 8: room for a small lambda's captures inside the `std::function` itself.
- **Negative numbers** show their two's complement bits in the hex and binary columns: `-1` is all ones, and `-2` is all ones but the last. The `ones` column agrees: 32 and 31.
- **`-100 % 7`** is `-2`, not `5`. C++ integer division truncates toward zero, and `%` takes the sign of the left operand.
- **`46340 * 46340`** is the largest square that fits in an `i32`; `46341 * 46341` overflows, as do `2147483647 + 1` and `-2147483648 / -1`, and `10 / 0` is refused.
- **Bad input:** `"3 ^ 4"` has no `^` in the operation table. `99999999999` doesn't fit in an `i32`, so `from_chars` reports `result_out_of_range`. Neither expression is counted.
- **The total, `4294879329`,** is about twice the largest `i32`. It's correct because `Wider<i32>` is `i64`.
- **`uses`** lists the operations in alphabetical order, because a `std::map` keeps its keys sorted.

In lecture 5 we start using third-party code, writing and reading PNG images. There, a failure carries a reason: `std::optional` becomes `std::expected`, which holds either a value or an error message.

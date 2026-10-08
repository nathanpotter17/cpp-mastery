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

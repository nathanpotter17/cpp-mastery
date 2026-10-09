# Lecture 1: Fundamentals

By the end of this lecture we'll have a loan calculator. It works out the payment on a loan, prints how the balance falls year by year, and shows how much interest an extra payment saves. Before the loans, the program takes a tour of C++'s fundamental types and how arithmetic converts between them.

Along the way we cover the ground every later lecture stands on:
- fundamental types, literals, and conversions;
- `const` and `constexpr`;
- functions: overloads, default arguments, and declaring before use;
- control flow: `if`, `switch`, `for`, `while`, `continue`, and the conditional operator;
- `struct`, `enum class` and namespaces;
- formatted output with `std::print`.

The whole program is one file, `lecture_1/main.cpp`. Each section below explains the next part of it; typed in order, the parts make up the complete file.

## 1.1 Project layout

### Why
One file is all a program this size needs, and it keeps the focus on the language itself. Lecture 2 splits a program into several files, once there's a reason to.

### How
- **The build:** the repo's root CMake builds each `lecture_N/` as an executable named `lecture_N`, from `lecture_N/main.cpp`.
  - It compiles as C++23 with `-Wall -Wextra -Wpedantic`, so the compiler warns about anything suspicious.
  - The build scripts in `build-scripts-linux/` configure and build one lecture each, in debug or release, with clang or g++.
- **`#include`** pastes a header's declarations into our file, so we can use what it declares. Standard library headers are written in angle brackets: `<print>` declares `std::println`, `<cmath>` the math functions, and `<cstdlib>` `EXIT_SUCCESS`.
- **A `.clangd` file** points the editor at the clang debug build's compile commands, so code completion and errors in the editor match the real build.

### Code
`lecture_1/.clangd`:
```yaml
# clangd reads compile flags from the clang debug build (build-debug-clang.bash).
CompileFlags:
  CompilationDatabase: build/debug-clang
```

## 1.2 Asking the compiler for a type

### Why
Every value in C++ has a type, and arithmetic quietly converts between types: `7 / 2` and `7 / 2.0` give different answers *and* different types. We want to see those types, not guess them. The compiler already knows the type of every expression, and *overloading* lets us ask it.

### How
- **Overloads:** several functions may share a name if their parameter types differ. For a call, the compiler picks the version whose parameter type best matches the argument. `type_name(7 / 2)` calls the `int` version, because `7 / 2` is an `int`.
- **Unnamed parameters:** each version only needs the argument's *type*, never its value, so the parameter has no name. Leaving it unnamed also tells the compiler (and `-Wextra`) it's unused on purpose.
- **Returning text:** a string literal like `"bool"` lives in the program for its whole run, so returning it is safe. Its type is an array of `const char`, which is returned as a `const char*`, a *pointer* to its first character. Lecture 2 covers pointers, and lecture 3 arrays and strings; for now, `std::print` knows how to print one.
- **`constexpr` functions** can run at compile time, and also at run time.
- **The fundamental types:**
  - `bool`: `true` or `false`.
  - `char`: one character, which is also a small integer.
  - `int`, `unsigned int` and `long long`: integers. An `unsigned` type holds no negative numbers, and twice as many positive ones.
  - `float` and `double`: floating point numbers. `double` has about 16 significant digits; `float` about 7.

### Code
`lecture_1/main.cpp`, part 1:
```cpp
#include <cmath>
#include <cstdlib>
#include <print>

// --- Asking the compiler for a type -----------------------------------------

// One function name, one version per parameter type. The compiler picks the
// version whose parameter matches the argument's type, so calling type_name
// on any expression tells us what type that expression has.
constexpr const char *type_name(bool) {
    return "bool";
}

constexpr const char *type_name(char) {
    return "char";
}

constexpr const char *type_name(int) {
    return "int";
}

constexpr const char *type_name(unsigned int) {
    return "unsigned int";
}

constexpr const char *type_name(long long) {
    return "long long";
}

constexpr const char *type_name(float) {
    return "float";
}

constexpr const char *type_name(double) {
    return "double";
}
```

## 1.3 A tour of types and arithmetic

### Why
Most surprises in everyday C++ arithmetic come from a few rules: how big each type is, what type a literal has, and how mixed expressions convert. Two functions print each rule in action, using `type_name` to show the type of every result.

### How
- **`sizeof`** gives the size of a type in bytes. The language fixes only `sizeof(char)` at 1; the other sizes are chosen by the platform. They're the same on every platform this course targets.
- **Literals have types:**
  - `'A'` is a `char`, `3.0` a `double`, and `3.0f` a `float`.
  - `0xff` is written in hex and `0b1010` in binary; both are ordinary `int`s.
  - `'` separates digits for readability: `1'000'000`. It doesn't change the value.
  - Suffixes choose the type: `LL` makes a `long long`, and `u` an `unsigned int`.
- **Arithmetic conversions:**
  - `int / int` is integer division: `7 / 2` is `3`, and `7 % 2` is the remainder, `1`.
  - If either operand is a `double`, the other is converted to `double` first, so `7 / 2.0` is `3.5`.
  - Arithmetic on a `char` promotes it to `int` first, so `'A' + 1` is the `int` `66`. `static_cast<char>(...)` converts it back explicitly, to `'B'`.
  - A comparison like `apples > people` is a `bool`.
  - `static_cast<int>(3.99)` truncates toward zero, to `3`. Brace initialization refuses *narrowing* conversions that can lose information: `int bad{3.99};` won't compile, but `int bad = 3.99;` compiles and stores `3`. Clang warns about it; the language doesn't require a warning, and g++ gives none by default.
  - Unsigned arithmetic wraps around: `0u - 1` is the largest `unsigned int`. Signed overflow, by contrast, is undefined behavior; lecture 4 deals with it.
- **`auto`** takes a variable's type from its initializer. `const auto big = 3'000'000'000LL;` is a `long long`.
- **Floating point is binary.** Most decimal fractions, like `0.1`, have no exact binary form, so they're rounded. `0.1 + 0.2` is not exactly `0.3`, and comparing doubles with `==` is rarely what we want. `{:.17f}` prints enough digits to show the rounding.
- **`const` vs `constexpr` variables:** a `const` variable can't change after it's initialized, but its value may only be known at run time. A `constexpr` variable must be computed at compile time, so it can be used where the language needs a constant.
- **`std::println`** prints a line, replacing each `{}` with the next argument. After the `:`, a format specification controls the layout: `{:<10}` left-aligns in 10 columns, `{:>5}` right-aligns in 5, and `{:.17f}` prints 17 decimals.

### Code
`lecture_1/main.cpp`, part 2:
```cpp
// --- Fundamental types -------------------------------------------------------

void print_types() {
    // sizeof gives a type's size in bytes. Only char is fixed (always 1);
    // the others are what this platform chose.
    std::println("{:<10} {:>5}", "type", "bytes");
    std::println("{:<10} {:>5}", "bool", sizeof(bool));
    std::println("{:<10} {:>5}", "char", sizeof(char));
    std::println("{:<10} {:>5}", "int", sizeof(int));
    std::println("{:<10} {:>5}", "long long", sizeof(long long));
    std::println("{:<10} {:>5}", "float", sizeof(float));
    std::println("{:<10} {:>5}", "double", sizeof(double));
}

// --- Literals, arithmetic and conversions ------------------------------------

void print_arithmetic() {
    // Literals have types too. ' separates digits; 0x is hex, 0b binary.
    std::println("1'000'000 = {}, 0xff = {}, 0b1010 = {}", 1'000'000, 0xff, 0b1010);
    std::println("'A' is a {}, 3.0f a {}, 3.0 a {}", type_name('A'), type_name(3.0f), type_name(3.0));

    // int / int is integer division: the fraction is thrown away.
    constexpr int apples = 7;
    constexpr int people = 2;
    std::println("{} / {} = {} ({}), remainder {}", apples, people, apples / people, type_name(apples / people), apples % people);

    // If either side is a double, the int is converted and the result is a double.
    std::println("{} / 2.0 = {} ({})", apples, apples / 2.0, type_name(apples / 2.0));

    // Arithmetic on a char promotes it to int first. static_cast converts back.
    std::println("'A' + 1 = {} ({}), as a char: {}", 'A' + 1, type_name('A' + 1), static_cast<char>('A' + 1));

    // A comparison is a bool.
    std::println("apples > people is {} ({})", apples > people, type_name(apples > people));

    // Converting a double to an int truncates toward zero. Brace
    // initialization refuses to do it silently: int bad{3.99}; won't compile.
    const int truncated = static_cast<int>(3.99);
    std::println("static_cast<int>(3.99) = {}", truncated);

    // Unsigned arithmetic wraps around: below 0 is the largest value.
    constexpr unsigned int zero = 0;
    std::println("0u - 1 = {} ({})", zero - 1, type_name(zero - 1));

    // auto takes its type from the initializer.
    const auto big = 3'000'000'000LL;
    std::println("3'000'000'000LL is a {}", type_name(big));

    // Most decimal fractions have no exact binary form, so doubles round.
    std::println("0.1 + 0.2 = {:.17f}, equal to 0.3: {}", 0.1 + 0.2, 0.1 + 0.2 == 0.3);
}
```

## 1.4 Describing a loan

### Why
A loan is a handful of values that belong together: the amount, the interest rate, the length, and how often payments are made. We want them in one type, and we want the payment frequency to be one of a fixed set of choices rather than any number.

### How
- **`namespace loan`:** every name declared inside is really `loan::Loan`, `loan::Frequency` and so on. Namespaces stop names from different parts of a program, or different libraries, clashing. `std::` is the standard library's namespace.
- **`enum class Frequency`:** a type whose values are exactly `Frequency::monthly`, `Frequency::biweekly` and `Frequency::weekly`. Unlike plain integers, they don't convert to `int` by accident. Lecture 4 shows how to choose the integer an `enum class` is stored as.
- **`switch`:** jumps to the `case` matching the value. Each of our cases `return`s, so none falls through into the next. With `-Wall`, the compiler warns if a `switch` over an `enum class` misses a value. Even so, the function needs a `return` after the `switch`: an enum variable can hold values that have no name.
- **`constexpr` and `static_assert`:** `periods_per_year` is `constexpr`, so the compiler can run it during compilation. `static_assert(periods_per_year(Frequency::monthly) == 12)` is checked while compiling. If it's false, the build fails, and the program never runs.
- **`struct Loan`:**
  - **Members:** `principal`, `annual_rate`, `years` and `frequency`. Each has a *default member initializer*, so any member a `Loan` is created without gets a sensible value.
  - **Member functions:** `periods()` and `period_rate()` compute values from the members. `const` after the parameter list promises they don't change the loan, so they can be called on a `const Loan`.
- **`struct Payoff`** holds the two results of paying a loan off early. A struct is also how a function returns more than one value.

### Code
`lecture_1/main.cpp`, part 3:
```cpp
// --- Payment frequency -------------------------------------------------------

// Everything about loans lives in namespace loan, so its names (Loan,
// payment, ...) can't clash with names from other code.
namespace loan {

// An enum class is a type with a fixed set of named values. Its values are
// written Frequency::monthly, and don't convert to int by accident.
enum class Frequency {
    monthly,
    biweekly,
    weekly,
};

// constexpr: can run at compile time, so its result can be checked with
// static_assert.
constexpr int periods_per_year(Frequency frequency) {
    switch (frequency) {
        case Frequency::monthly:
            return 12;
        case Frequency::biweekly:
            return 26;
        case Frequency::weekly:
            return 52;
    }
    return 0; // not reached for the three values above
}

static_assert(periods_per_year(Frequency::monthly) == 12);

constexpr const char *name(Frequency frequency) {
    switch (frequency) {
        case Frequency::monthly:
            return "monthly";
        case Frequency::biweekly:
            return "biweekly";
        case Frequency::weekly:
            return "weekly";
    }
    return "?";
}

// --- The loan ----------------------------------------------------------------

// A struct groups related values. Each member has a default value, so
// Loan{} is a loan of nothing at 0% for 0 years.
struct Loan {
    double principal = 0.0;   // the amount borrowed
    double annual_rate = 0.0; // 0.06 means 6% per year
    int years = 0;
    Frequency frequency = Frequency::monthly;

    // Member functions. const: they read the loan, but can't change it.
    int periods() const {
        return years * periods_per_year(frequency);
    }

    double period_rate() const {
        return annual_rate / periods_per_year(frequency);
    }
};

// The result of paying a loan off early.
struct Payoff {
    int periods = 0;
    double interest = 0.0;
};

} // namespace loan
```

## 1.5 The arithmetic

### Why
These three functions do the loan calculator's actual work: the payment, the year-by-year schedule, and the effect of paying extra.

### How
- **Reopening a namespace:** a second `namespace loan { ... }` block adds to the first, so these functions are `loan::payment` and so on. Inside the namespace, they use `Loan` and `periods_per_year` without the `loan::` prefix.
- **Pass by value:** `payment(Loan loan)` receives its own copy of the caller's loan. For a few numbers that's cheap. Lecture 2 shows how to pass large objects without copying them.
- **The payment:** the standard annuity formula. With rate `r` per period and `n` periods, each payment is `P * r / (1 - (1 + r)^-n)`.
  - `std::pow` from `<cmath>` raises a number to a power.
  - `const` locals: `rate` and `count` are set once, and the compiler stops us changing them by mistake.
  - At a 0% rate the formula would divide by zero, so an `if` handles that case first and returns early.
- **The schedule** steps through every payment with a `for` loop:
  - `for (int period = 1; period <= loan.periods(); ++period)` has three parts: the start, the condition checked before every pass, and the step taken after every pass. `++period` adds one.
  - Each payment first pays the period's interest, and the rest reduces the balance. `+=` and `-=` update a variable in place.
  - `continue` skips the rest of this pass when the period isn't the end of a printed year. `%` is the remainder of an integer division, so `period % 12 == 0` at the end of every year of monthly payments.
  - `std::abs` and the conditional operator `a ? b : c` turn the last balance, a rounding error of a fraction of a cent either side of zero, into a clean `0.00`.
  - `{:>12.2f}` right-aligns a number in 12 columns with 2 decimals.
- **Paying off early** uses a `while` loop, because we don't know in advance how many periods it takes. The loop body runs as long as `balance > 0.0` holds. The last payment overshoots slightly, which doesn't change the count.

### Code
`lecture_1/main.cpp`, part 4:
```cpp
// --- The payment -------------------------------------------------------------

// A namespace can be opened again: this block adds to the one above.
namespace loan {

// The fixed payment that repays the loan in exactly loan.periods() payments.
// The standard annuity formula: with rate r per period and n periods, the
// payment is P * r / (1 - (1 + r)^-n).
double payment(Loan loan) {
    const double rate = loan.period_rate();
    const int count = loan.periods();

    // At 0% the formula divides by zero; the payment is just an equal share.
    if (rate == 0.0) {
        return loan.principal / count;
    }

    return loan.principal * rate / (1.0 - std::pow(1.0 + rate, -count));
}

// --- The schedule ------------------------------------------------------------

// Prints the balance and interest paid so far, every `every_years` years.
void print_schedule(Loan loan, int every_years) {
    const double amount = payment(loan);
    const int per_year = periods_per_year(loan.frequency);

    double balance = loan.principal;
    double interest_paid = 0.0;

    std::println("  {:>4}  {:>12}  {:>12}", "year", "balance", "interest");

    // Periods are counted from 1, so period 12 is the end of year 1.
    for (int period = 1; period <= loan.periods(); ++period) {
        const double interest = balance * loan.period_rate();
        interest_paid += interest;
        balance -= amount - interest;

        // Skip every period that isn't the end of a printed year.
        if (period % (per_year * every_years) != 0) {
            continue;
        }

        // The last balance is a tiny rounding error, positive or negative.
        // The conditional operator a ? b : c picks b if a is true, else c.
        const double shown = std::abs(balance) < 0.005 ? 0.0 : balance;
        std::println("  {:>4}  {:>12.2f}  {:>12.2f}", period / per_year, shown, interest_paid);
    }
}

// --- Paying off early --------------------------------------------------------

// Pays `extra` on top of every payment, until nothing is owed.
Payoff pay_off_early(Loan loan, double extra) {
    const double amount = payment(loan) + extra;

    Payoff payoff;
    double balance = loan.principal;

    // A while loop runs as long as its condition holds; we don't know in
    // advance how many periods it takes.
    while (balance > 0.0) {
        const double interest = balance * loan.period_rate();
        payoff.interest += interest;
        balance -= amount - interest;
        ++payoff.periods;
    }

    return payoff;
}

} // namespace loan
```

## 1.6 The program

### Why
`main` is where the program starts. Ours runs the tour, then works through two loans. Each step is its own function, so `main` itself reads like a table of contents.

### How
- **Calling across namespaces:** `print_loan` and `print_payoff` aren't in `namespace loan`, so they write `loan::payment`, `loan::Loan` and so on in full.
- **Default arguments:** `int every_years = 1` lets a caller leave the last argument out. `print_loan("Car", car)` means `print_loan("Car", car, 1)`.
- **Declare before use:** C++ reads a file from top to bottom, and a function must be declared before it's called. Every function here is defined above the code that calls it, and a definition is also a declaration. That's why `main` comes last.
- **Designated initializers** (C++20) name each member they set: `{.principal = 250'000.0, .annual_rate = 0.06, .years = 30}`. They must follow the members' declaration order, and any member left out keeps its default. The house loan leaves `frequency` out, so it's monthly.
- **The exit code:** `main` returns `EXIT_SUCCESS` (from `<cstdlib>`), which tells the shell the program succeeded.

### Code
`lecture_1/main.cpp`, part 5:
```cpp
// --- Printing loans ----------------------------------------------------------

// A default argument: print_loan(title, loan) means print_loan(title, loan, 1).
void print_loan(const char *title, loan::Loan loan, int every_years = 1) {
    std::println("{}: {:.2f} at {:.2f}% for {} years, {}", title, loan.principal, loan.annual_rate * 100.0, loan.years, loan::name(loan.frequency));
    std::println("  {} payments of {:.2f}", loan.periods(), loan::payment(loan));
    loan::print_schedule(loan, every_years);
}

void print_payoff(loan::Loan loan, double extra) {
    const double total = loan::payment(loan) * loan.periods();
    const double interest = total - loan.principal;
    const loan::Payoff payoff = loan::pay_off_early(loan, extra);

    const int per_year = loan::periods_per_year(loan.frequency);
    std::println("  paying {:.2f} extra: done after {} years and {} payments, saving {:.2f} interest",
        extra, payoff.periods / per_year, payoff.periods % per_year, interest - payoff.interest);
}

// --- The program -------------------------------------------------------------

// A function must be declared before it's called. Every function above is
// defined before main, and a definition is also a declaration.
int main() {
    print_types();
    std::println();
    print_arithmetic();
    std::println();

    // Designated initializers name the members they set, in declaration
    // order. Members left out keep their defaults.
    constexpr loan::Loan house{.principal = 250'000.0, .annual_rate = 0.06, .years = 30};
    constexpr loan::Loan car{
        .principal = 20'000.0,
        .annual_rate = 0.045,
        .years = 5,
        .frequency = loan::Frequency::biweekly,
    };

    print_loan("House", house, 5);
    print_payoff(house, 200.0);
    std::println();

    print_loan("Car", car);
    print_payoff(car, 50.0);

    return EXIT_SUCCESS;
}
```

## 1.7 Build and run

```bash
./build-scripts-linux/build-debug-clang.bash lecture_1 && ./lecture_1/build/debug-clang/lecture_1
```

```text
type       bytes
bool           1
char           1
int            4
long long      8
float          4
double         8

1'000'000 = 1000000, 0xff = 255, 0b1010 = 10
'A' is a char, 3.0f a float, 3.0 a double
7 / 2 = 3 (int), remainder 1
7 / 2.0 = 3.5 (double)
'A' + 1 = 66 (int), as a char: B
apples > people is true (bool)
static_cast<int>(3.99) = 3
0u - 1 = 4294967295 (unsigned int)
3'000'000'000LL is a long long
0.1 + 0.2 = 0.30000000000000004, equal to 0.3: false

House: 250000.00 at 6.00% for 30 years, monthly
  360 payments of 1498.88
  year       balance      interest
     5     232635.89      72568.47
    10     209214.31     139079.47
    15     177622.11     197419.85
    20     135008.97     244739.28
    25      77530.22     277193.11
    30          0.00     289595.47
  paying 200.00 extra: done after 22 years and 3 payments, saving 86232.96 interest

Car: 20000.00 at 4.50% for 5 years, biweekly
  130 payments of 171.93
  year       balance      interest
     1      16351.37        821.68
     2      12534.95       1475.56
     3       8543.03       1953.95
     4       4367.52       2248.75
     5          0.00       2351.53
  paying 50.00 extra: done after 3 years and 21 payments, saving 589.13 interest
```

Reading the output:
- **The types:** `int` is 4 bytes, so it holds about ±2.1 billion; `long long` is 8 bytes. `3'000'000'000` is too big for an `int`. Without a suffix, the compiler would quietly make it the next type that fits, a `long`; `LL` makes it a `long long` explicitly.
- **The conversions:** every line shows a value together with the type the compiler gave it. `0u - 1` wrapped around to 4294967295, the largest `unsigned int`.
- **`0.30000000000000004`** is the double nearest to `0.1 + 0.2`, and it isn't the double nearest to `0.3`.
- **The house loan** pays 1498.88 a month for 30 years. The interest column shows how front-loaded a loan is: after 5 years, 72568.47 has gone to interest, and the balance has only dropped by about 17000. Paying 200 extra each month finishes almost 8 years early.
- **The car loan** is biweekly, so it has 26 payments a year and 130 in total. Its schedule prints every year, thanks to the default argument.

In lecture 2 we look at where these values actually live: memory, addresses, and who is responsible for cleaning up.

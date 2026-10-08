#include <cmath>
#include <cstdlib>
#include <print>

// --- Asking the compiler for a type -----------------------------------------

// One function name, one version per parameter type. The compiler picks the
// version whose parameter matches the argument's type, so calling type_name
// on any expression tells us what type that expression has.
constexpr const char* type_name(bool) {
    return "bool";
}

constexpr const char* type_name(char) {
    return "char";
}

constexpr const char* type_name(int) {
    return "int";
}

constexpr const char* type_name(unsigned int) {
    return "unsigned int";
}

constexpr const char* type_name(long long) {
    return "long long";
}

constexpr const char* type_name(float) {
    return "float";
}

constexpr const char* type_name(double) {
    return "double";
}

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

constexpr const char* name(Frequency frequency) {
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

// --- Printing loans ----------------------------------------------------------

// A default argument: print_loan(title, loan) means print_loan(title, loan, 1).
void print_loan(const char* title, loan::Loan loan, int every_years = 1) {
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

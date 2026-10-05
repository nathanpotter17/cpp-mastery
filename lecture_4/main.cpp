#include <print>

int addNumbers(int f1, int f2) {
    return f1 + f2;
}

int main (int argc, char **argv) {
    int firstNum = 12;
    int secondNum = 12;
    int sum = addNumbers(firstNum, secondNum);
    std::print("The sum is {} (hex: {:x})", sum, sum);
    std::println();
    return 0;
}
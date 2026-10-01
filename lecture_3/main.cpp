#include <print>
#include <vector>

int main() {
    std::vector<int> numbers{1, 2, 3, 4};

    for (int n : numbers) {
        if (n % 2 == 0) {
            std::println("{}", n);
        }
    }
}


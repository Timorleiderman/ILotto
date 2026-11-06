// Host-only driver for sha_outcome.h: the same hash code the CUDA kernel runs,
// compiled as plain C++ so it can be checked against Python without a GPU.
// stdin: "<seed> <YYYY-MM-DD>" per line. stdout: one outcome per line.
// Exits 3 if the rejection-boundary checks fail.
#include <iostream>
#include "sha_outcome.h"

int main() {
    U32 h[8];
    for (U32& x : h) x = 0xffffffffU;
    U32 value;
    if (reduce(h, value)) return 3;
    h[7] = U32(0x100000000ULL - TAIL);
    if (reduce(h, value)) return 3;
    --h[7];
    if (!reduce(h, value) || value != OUTCOMES - 1) return 3;

    U64 seed;
    std::string date;
    try {
        while (std::cin >> seed >> date) std::cout << outcome(seed, prepare(date, 0)) << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
    return 0;
}

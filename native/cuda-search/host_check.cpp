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

    std::string seed_tok, date;
    try {
        while (std::cin >> seed_tok >> date) {
            U64 seed = parse_u64_strict(seed_tok);
            Prepared p = prepare(date, 0);
            U32 out = outcome(seed, p);
            p.expected = out;
            if (filter_match(seed, p) != 1) return 3;
            p.expected = (out + 1) % U32(OUTCOMES);
            if (filter_match(seed, p) != 0) return 3;
            std::cout << out << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
    return 0;
}

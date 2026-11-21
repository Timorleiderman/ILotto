// calendar-sha256-v1 outcome, shared by the CUDA worker and a host-only check.
// Compiles as CUDA (nvcc) or as plain C++17, so the exact code the GPU runs can
// be cross-checked against Python on machines without a GPU.
#pragma once
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __noinline__
#endif

using U64 = unsigned long long;
using U32 = unsigned int;
constexpr U64 OUTCOMES = 16273488;
constexpr U64 ODD_MOD = OUTCOMES >> 4; // 1,017,093 (16 | OUTCOMES)
constexpr char SEED_DOMAIN[] = "calendar-sha256-v1:";
static_assert(sizeof(SEED_DOMAIN) - 1 == 19, "domain length is part of the generator");
static_assert((OUTCOMES & 15ULL) == 0 && (ODD_MOD & 1ULL) == 1, "OUTCOMES = 16 * odd");

constexpr U64 odd_inverse(U64 m) {
    U64 x = 1;
    for (int i = 0; i < 6; ++i) x *= 2ULL - m * x;
    return x;
}
constexpr U64 ODD_INV = odd_inverse(ODD_MOD);
constexpr U64 ODD_MAX_Q = ~U64(0) / ODD_MOD;
static_assert(ODD_MOD * ODD_INV == 1ULL, "ODD_INV * ODD_MOD == 1 mod 2^64");

// 2^(32k) mod OUTCOMES for k = 0..8; entry 8 is 2^256 mod OUTCOMES (the rejected tail).
struct Powers { U64 v[9]; };
constexpr Powers make_powers() {
    Powers p{};
    p.v[0] = 1;
    for (int k = 1; k < 9; ++k) p.v[k] = (p.v[k - 1] << 32) % OUTCOMES;
    return p;
}
constexpr Powers POWERS = make_powers();
constexpr U32 TAIL = U32(POWERS.v[8]);
static_assert(TAIL == 9050512, "2^256 mod OUTCOMES");

// Words 0-3 hold the first 16 domain bytes, identical for every seed and date,
// so rounds 0-3 are computed once per date (`mid`) instead of once per seed.
constexpr int MID_ROUNDS = 4;
struct Prepared { U32 words[16]; U32 mid[8]; U32 expected; };

__host__ __device__ inline U32 rotate(U32 x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

__host__ __device__ inline bool in_rejected_tail(const U32* h) {
    if (h[0] != 0xffffffffU) return false;
    bool top = true;
    #pragma unroll
    for (int i = 1; i < 7; ++i) top &= h[i] == 0xffffffffU;
    return top && h[7] >= U32(0x100000000ULL - TAIL);
}

__host__ __device__ inline bool reduce(const U32* h, U32& result) {
    if (in_rejected_tail(h)) return false;
    // Each term < 2^32 * 2^24 = 2^56; eight sum below 2^59. One 64-bit modulo
    // replaces eight: on a GPU each is a long software routine, not one instruction.
    // Local constexpr array: device code may not read global constexpr arrays.
    constexpr U64 pw[8] = {POWERS.v[7], POWERS.v[6], POWERS.v[5], POWERS.v[4],
                           POWERS.v[3], POWERS.v[2], POWERS.v[1], POWERS.v[0]};
    U64 sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) sum += U64(h[i]) * pw[i];
    result = U32(sum % OUTCOMES);
    return true;
}

// Fast exact check: returns 1 if non-rejected hash equals expected, 0 if not, -1 if rejected.
__host__ __device__ inline int matches_expected_hash(const U32* h, U32 expected) {
    if (in_rejected_tail(h)) return -1;
    // Since 16 | OUTCOMES and 16 | 2^32, H mod 16 == h[7] & 15; rejects 15/16 immediately.
    if ((h[7] & 15U) != (expected & 15U)) return 0;
    constexpr U32 pw16[7] = {U32(POWERS.v[7] >> 4), U32(POWERS.v[6] >> 4), U32(POWERS.v[5] >> 4),
                             U32(POWERS.v[4] >> 4), U32(POWERS.v[3] >> 4), U32(POWERS.v[2] >> 4),
                             U32(POWERS.v[1] >> 4)};
    U64 y = (U64(h[7]) + (OUTCOMES - U64(expected))) >> 4;
    #pragma unroll
    for (int i = 0; i < 7; ++i) y += U64(h[i]) * U64(pw16[i]);
    return (y * ODD_INV <= ODD_MAX_Q) ? 1 : 0;
}

#define ILOTTO_SHA256_K { \
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5, \
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174, \
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da, \
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967, \
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85, \
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070, \
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3, \
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2}

#define ILOTTO_SHA256_INIT {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a, \
                            0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}

__host__ __device__ inline void sha256_from_mid(U32* w, const U32* mid, U32* hash) {
    constexpr U32 k[64] = ILOTTO_SHA256_K;
    constexpr U32 init[8] = ILOTTO_SHA256_INIT;
    U32 a=mid[0], b=mid[1], c=mid[2], d=mid[3];
    U32 e=mid[4], f=mid[5], g=mid[6], h=mid[7];
    #pragma unroll
    for (int i = MID_ROUNDS; i < 64; ++i) {
        if (i >= 16) {
            U32 x=w[(i-15)&15], y=w[(i-2)&15];
            w[i&15] += (rotate(x,7)^rotate(x,18)^(x>>3)) + w[(i-7)&15]
                      + (rotate(y,17)^rotate(y,19)^(y>>10));
        }
        U32 t1=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+k[i]+w[i&15];
        U32 t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    hash[0]=a+init[0]; hash[1]=b+init[1]; hash[2]=c+init[2]; hash[3]=d+init[3];
    hash[4]=e+init[4]; hash[5]=f+init[5]; hash[6]=g+init[6]; hash[7]=h+init[7];
}

// Slow path for the ~7.8e-71 rejection tail, kept out of the main kernel register budget.
__host__ __device__ __noinline__ U32 outcome_slow(U64 seed, const Prepared& p) {
    for (U32 counter = 1; counter != 0; ++counter) {
        U32 w[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = p.words[i];
        w[4] |= U32(seed >> 56);
        w[5] = U32(seed >> 24);
        w[6] |= U32(seed) << 8;
        w[9] |= counter >> 8;
        w[10] |= counter << 24;
        U32 hash[8];
        sha256_from_mid(w, p.mid, hash);
        U32 result;
        if (reduce(hash, result)) return result;
    }
    return 0xffffffffU; // Exhaustion fails the batch instead of silently wrapping.
}

__host__ __device__ U32 outcome(U64 seed, const Prepared& p) {
    U32 w[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = p.words[i];
    w[4] |= U32(seed >> 56);
    w[5] = U32(seed >> 24);
    w[6] |= U32(seed) << 8;
    U32 hash[8];
    sha256_from_mid(w, p.mid, hash);
    U32 result;
    if (reduce(hash, result)) return result;
    return outcome_slow(seed, p);
}

// Fast exact first-draw match test for GPU filter kernel (returns 1 match, 0 miss, -1 error).
__host__ __device__ inline int filter_match(U64 seed, const Prepared& p) {
    U32 w[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = p.words[i];
    w[4] |= U32(seed >> 56);
    w[5] = U32(seed >> 24);
    w[6] |= U32(seed) << 8;
    U32 hash[8];
    sha256_from_mid(w, p.mid, hash);
    int m = matches_expected_hash(hash, p.expected);
    if (m >= 0) return m;
    U32 value = outcome_slow(seed, p);
    if (value == 0xffffffffU) return -1;
    return value == p.expected ? 1 : 0;
}

// Strict unsigned decimal parser: rejects leading signs, spaces, or overflow.
inline U64 parse_u64_strict(const std::string& token) {
    if (token.empty() || token.size() > 20) throw std::runtime_error("Invalid unsigned integer");
    U64 value = 0;
    for (char ch : token) {
        if (ch < '0' || ch > '9') throw std::runtime_error("Invalid unsigned integer");
        U64 digit = U64(ch - '0');
        if (value > (std::numeric_limits<U64>::max() - digit) / 10ULL)
            throw std::runtime_error("Unsigned integer overflow");
        value = value * 10ULL + digit;
    }
    return value;
}

inline unsigned parse_u32_strict(const std::string& token) {
    U64 value = parse_u64_strict(token);
    if (value > std::numeric_limits<unsigned>::max()) throw std::runtime_error("Unsigned integer overflow");
    return static_cast<unsigned>(value);
}

// Host only: build the fixed message block and the rounds 0-3 midstate for one date.
inline Prepared prepare(const std::string& date, U64 expected) {
    if (date.size()!=10 || date[4]!='-' || date[7]!='-' || expected>=OUTCOMES)
        throw std::runtime_error("Invalid date or outcome");
    for (int i=0;i<10;++i)
        if (i!=4 && i!=7 && (date[i]<'0' || date[i]>'9'))
            throw std::runtime_error("Invalid date digits");
    int year=std::stoi(date.substr(0,4)), month=std::stoi(date.substr(5,2)), day=std::stoi(date.substr(8,2));
    constexpr int days[12]={31,28,31,30,31,30,31,31,30,31,30,31};
    if (year<1 || month<1 || month>12 || day<1 ||
        day>days[month-1]+(month==2 && year%4==0 && (year%100!=0 || year%400==0)))
        throw std::runtime_error("Invalid calendar date");
    unsigned char block[64]={};
    std::memcpy(block,SEED_DOMAIN,19);
    std::memcpy(block+27,date.data(),10);
    block[41]=0x80; block[62]=1; block[63]=72; // 41*8 bits
    Prepared p{}; p.expected=U32(expected);
    for (int i=0;i<16;++i)
        p.words[i]=(U32(block[4*i])<<24)|(U32(block[4*i+1])<<16)|(U32(block[4*i+2])<<8)|block[4*i+3];
    constexpr U32 k[64] = ILOTTO_SHA256_K;
    constexpr U32 init[8] = ILOTTO_SHA256_INIT;
    U32 s[8];
    for (int i=0;i<8;++i) s[i]=init[i];
    for (int i=0;i<MID_ROUNDS;++i) {  // words 0-3 never contain seed or counter bytes
        U32 t1=s[7]+(rotate(s[4],6)^rotate(s[4],11)^rotate(s[4],25))+((s[4]&s[5])^(~s[4]&s[6]))+k[i]+p.words[i];
        U32 t2=(rotate(s[0],2)^rotate(s[0],13)^rotate(s[0],22))+((s[0]&s[1])^(s[0]&s[2])^(s[1]&s[2]));
        s[7]=s[6]; s[6]=s[5]; s[5]=s[4]; s[4]=s[3]+t1; s[3]=s[2]; s[2]=s[1]; s[1]=s[0]; s[0]=t1+t2;
    }
    for (int i=0;i<8;++i) p.mid[i]=s[i];
    return p;
}

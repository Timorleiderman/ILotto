// Exact calendar-sha256-v1 batches. Persistent stdin keeps the CUDA context warm.
// GPU filters the first draw; the host verifies the rare hits against every draw.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using U64 = unsigned long long;
using U32 = unsigned int;
constexpr U64 OUTCOMES = 16273488;
constexpr U32 TAIL = 9050512; // 2^256 mod OUTCOMES
constexpr unsigned CAP = 10000;
constexpr char DOMAIN[] = "calendar-sha256-v1:";
static_assert(sizeof(DOMAIN) - 1 == 19);
struct Prepared { U32 words[16]; U32 expected; };

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__host__ __device__ inline U32 rotate(U32 x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

__host__ __device__ inline bool reduce(const U32* h, U32& result) {
    bool top = true;
    for (int i = 0; i < 7; ++i) top &= h[i] == 0xffffffffU;
    if (top && h[7] >= U32(0x100000000ULL - TAIL)) return false;
    U64 r = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) r = ((r << 32) | h[i]) % OUTCOMES;
    result = U32(r);
    return true;
}

__host__ __device__ U32 outcome(U64 seed, const Prepared& p) {
    constexpr U32 k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    U32 counter = 0;
    do {
        U32 w[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = p.words[i];
        w[4] |= U32(seed >> 56);
        w[5] = U32(seed >> 24);
        w[6] |= U32(seed) << 8;
        w[9] |= counter >> 8;
        w[10] |= counter << 24;
        U32 a=0x6a09e667, b=0xbb67ae85, c=0x3c6ef372, d=0xa54ff53a;
        U32 e=0x510e527f, f=0x9b05688c, g=0x1f83d9ab, h=0x5be0cd19;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            if (i >= 16) {
                U32 x=w[(i-15)&15], y=w[(i-2)&15];
                w[i&15] += (rotate(x,7)^rotate(x,18)^(x>>3)) + w[(i-7)&15]
                          + (rotate(y,17)^rotate(y,19)^(y>>10));
            }
            U32 t1=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+k[i]+w[i&15];
            U32 t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        U32 hash[8]={a+0x6a09e667,b+0xbb67ae85,c+0x3c6ef372,d+0xa54ff53a,
                     e+0x510e527f,f+0x9b05688c,g+0x1f83d9ab,h+0x5be0cd19};
        U32 result;
        if (reduce(hash, result)) return result;
    } while (++counter != 0);
    return 0xffffffffU; // Exhaustion fails the batch instead of silently wrapping.
}

Prepared prepare(const std::string& date, U64 expected) {
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
    std::memcpy(block,DOMAIN,19);
    std::memcpy(block+27,date.data(),10);
    block[41]=0x80; block[62]=1; block[63]=72; // 41*8 bits
    Prepared p{}; p.expected=U32(expected);
    for (int i=0;i<16;++i)
        p.words[i]=(U32(block[4*i])<<24)|(U32(block[4*i+1])<<16)|(U32(block[4*i+2])<<8)|block[4*i+3];
    return p;
}

__constant__ Prepared first;
__global__ void filter(U64 start, U64 count, U64* hits, unsigned* found, unsigned* error) {
    U64 stride=U64(blockDim.x)*gridDim.x;
    for (U64 offset=U64(blockIdx.x)*blockDim.x+threadIdx.x; offset<count; offset+=stride) {
        U32 value=outcome(start+offset,first);
        if (value==0xffffffffU) atomicExch(error,1);
        if (value==first.expected) {
            unsigned slot=atomicAdd(found,1);
            if (slot<CAP) hits[slot]=start+offset;
        }
    }
}

__global__ void vectors(const U64* seeds, const Prepared* dates, U32* results, unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) results[i]=outcome(seeds[i],dates[i]);
}

template<class T> struct Device {
    T* p=nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p,n*sizeof(T))); }
    ~Device() { cudaFree(p); }
    Device(const Device&)=delete;
    Device& operator=(const Device&)=delete;
};

std::string decimal(unsigned __int128 n) {
    std::string s;
    do { s.push_back(char('0'+n%10)); n/=10; } while(n);
    std::reverse(s.begin(),s.end()); return s;
}

void self_test() {
    U32 h[8]; std::fill(h,h+8,0xffffffffU); U32 value;
    if(reduce(h,value)) throw std::runtime_error("Rejection maximum failed");
    h[7]=U32(0x100000000ULL-TAIL);
    if(reduce(h,value)) throw std::runtime_error("Rejection boundary failed");
    --h[7];
    if(!reduce(h,value)||value!=OUTCOMES-1) throw std::runtime_error("Mapping boundary failed");
    if(outcome(0,prepare("2020-01-01",0))!=15611497 ||
       outcome(45123,prepare("2020-01-01",0))!=6083258 ||
       outcome(~U64(0),prepare("2026-11-06",0))!=9078729)
        throw std::runtime_error("Host SHA-256 known vectors failed");
}

int main() {
    try {
        self_test();
        Device<U64> hits(CAP); Device<unsigned> found(1), error(1);
        std::string command;
        while(std::cin>>command) {
            if(command=="V") {
                unsigned n;
                if(!(std::cin>>n)||!n||n>10000) throw std::runtime_error("Invalid vector count");
                std::vector<U64> seeds(n); std::vector<Prepared> dates(n); std::vector<U32> result(n);
                for(unsigned i=0;i<n;++i) {
                    std::string date;
                    if(!(std::cin>>seeds[i]>>date)) throw std::runtime_error("Truncated vectors");
                    dates[i]=prepare(date,0);
                }
                Device<U64> ds(n); Device<Prepared> dp(n); Device<U32> dr(n);
                check(cudaMemcpy(ds.p,seeds.data(),n*sizeof(U64),cudaMemcpyHostToDevice));
                check(cudaMemcpy(dp.p,dates.data(),n*sizeof(Prepared),cudaMemcpyHostToDevice));
                vectors<<<(n+127)/128,128>>>(ds.p,dp.p,dr.p,n);
                check(cudaGetLastError());
                check(cudaMemcpy(result.data(),dr.p,n*sizeof(U32),cudaMemcpyDeviceToHost));
                std::cout<<"{\"outcomes\":[";
                for(unsigned i=0;i<n;++i) std::cout<<(i?",":"")<<result[i];
                std::cout<<"]}"<<std::endl;
                continue;
            }
            if(command!="B") throw std::runtime_error("Unknown command");
            U64 start,count; unsigned threads,blocks,n;
            if(!(std::cin>>start>>count>>threads>>blocks>>n) || !count || count>1000000000ULL ||
               count-1>std::numeric_limits<U64>::max()-start ||
               (threads!=64&&threads!=128&&threads!=256&&threads!=512) || !blocks || blocks>65535 || !n || n>10000)
                throw std::runtime_error("Invalid batch bounds or launch configuration");
            std::vector<Prepared> prefix; std::string previous;
            for(unsigned i=0;i<n;++i) {
                std::string date; U64 expected;
                if(!(std::cin>>date>>expected)||date<=previous) throw std::runtime_error("Invalid ordered prefix");
                prefix.push_back(prepare(date,expected)); previous=date;
            }
            auto began=std::chrono::steady_clock::now();
            check(cudaMemcpyToSymbol(first,prefix.data(),sizeof(Prepared)));
            check(cudaMemset(found.p,0,sizeof(unsigned))); check(cudaMemset(error.p,0,sizeof(unsigned)));
            filter<<<blocks,threads>>>(start,count,hits.p,found.p,error.p);
            check(cudaGetLastError());
            unsigned total,failed;
            check(cudaMemcpy(&total,found.p,sizeof(unsigned),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(&failed,error.p,sizeof(unsigned),cudaMemcpyDeviceToHost));
            if(total>CAP||failed) throw std::runtime_error("Filter overflow or hash counter exhaustion; batch not counted");
            std::vector<U64> candidates(total),survivors,counts(n+1,0);
            check(cudaMemcpy(candidates.data(),hits.p,total*sizeof(U64),cudaMemcpyDeviceToHost));
            std::sort(candidates.begin(),candidates.end()); counts[0]=count-total;
            for(U64 seed:candidates) {
                unsigned matched=1;
                for(;matched<n;++matched) {
                    U32 value=outcome(seed,prefix[matched]);
                    if(value==0xffffffffU) throw std::runtime_error("Host hash counter exhaustion");
                    if(value!=prefix[matched].expected) break;
                }
                ++counts[matched]; if(matched==n) survivors.push_back(seed);
            }
            double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
            std::cout<<"{\"generator\":\"calendar-sha256-v1\",\"start\":\""<<start
                     <<"\",\"stop\":\""<<decimal((unsigned __int128)start+count)<<"\",\"tested\":"<<count
                     <<",\"matched_prefix_counts\":[";
            for(unsigned i=0;i<=n;++i) std::cout<<(i?",":"")<<counts[i];
            std::cout<<"],\"survivors\":[";
            for(size_t i=0;i<survivors.size();++i) std::cout<<(i?",":"")<<survivors[i];
            std::cout<<"],\"survivor_overflow\":false,\"batch_seconds\":"<<elapsed<<"}"<<std::endl;
        }
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; }
}

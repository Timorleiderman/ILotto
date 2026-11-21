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
#include "sha_outcome.h"

// First-draw matches per batch: ~count/16.3M expected (about 1,230 at the
// 2e10 batch cap). The buffer is sized for that with a wide margin.
constexpr unsigned CAP = 1u << 16;
constexpr size_t SURVIVOR_CAP = 10000;
constexpr U64 MAX_BATCH = 20000000000ULL;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__constant__ Prepared first;
__global__ void filter(U64 start, U64 count, U64* hits, unsigned* found, unsigned* error) {
    U64 stride=U64(blockDim.x)*gridDim.x;
    for (U64 offset=U64(blockIdx.x)*blockDim.x+threadIdx.x; offset<count; offset+=stride) {
        int m=filter_match(start+offset,first);
        if (m<0) atomicExch(error,1);
        else if (m>0) {
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
            if(command=="C") {  // capabilities: lets the driver size batches for this build
                std::cout<<"{\"generator\":\"calendar-sha256-v1\",\"protocol\":2,\"max_batch\":"
                         <<MAX_BATCH<<"}"<<std::endl;
                continue;
            }
            if(command=="V") {
                std::string n_tok;
                if(!(std::cin>>n_tok)) throw std::runtime_error("Invalid vector count");
                unsigned n=parse_u32_strict(n_tok);
                if(!n||n>10000) throw std::runtime_error("Invalid vector count");
                std::vector<U64> seeds(n); std::vector<Prepared> dates(n); std::vector<U32> result(n);
                for(unsigned i=0;i<n;++i) {
                    std::string seed_tok, date;
                    if(!(std::cin>>seed_tok>>date)) throw std::runtime_error("Truncated vectors");
                    seeds[i]=parse_u64_strict(seed_tok);
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
            std::string start_tok,count_tok,threads_tok,blocks_tok,n_tok;
            if(!(std::cin>>start_tok>>count_tok>>threads_tok>>blocks_tok>>n_tok))
                throw std::runtime_error("Invalid batch bounds or launch configuration");
            U64 start=parse_u64_strict(start_tok), count=parse_u64_strict(count_tok);
            unsigned threads=parse_u32_strict(threads_tok), blocks=parse_u32_strict(blocks_tok), n=parse_u32_strict(n_tok);
            if(!count || count>MAX_BATCH ||
               count-1>std::numeric_limits<U64>::max()-start ||
               (threads!=64&&threads!=128&&threads!=256&&threads!=512) || !blocks || blocks>65535 || !n || n>10000)
                throw std::runtime_error("Invalid batch bounds or launch configuration");
            std::vector<Prepared> prefix; std::string previous;
            for(unsigned i=0;i<n;++i) {
                std::string date, expected_tok;
                if(!(std::cin>>date>>expected_tok)||date<=previous) throw std::runtime_error("Invalid ordered prefix");
                U64 expected=parse_u64_strict(expected_tok);
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
            if(total>0) check(cudaMemcpy(candidates.data(),hits.p,total*sizeof(U64),cudaMemcpyDeviceToHost));
            std::sort(candidates.begin(),candidates.end()); counts[0]=count-total;
            bool survivor_overflow=false;
            for(U64 seed:candidates) {
                unsigned matched=1;
                for(;matched<n;++matched) {
                    U32 value=outcome(seed,prefix[matched]);
                    if(value==0xffffffffU) throw std::runtime_error("Host hash counter exhaustion");
                    if(value!=prefix[matched].expected) break;
                }
                ++counts[matched];
                if(matched==n) {
                    if(survivors.size()<SURVIVOR_CAP) survivors.push_back(seed);
                    else survivor_overflow=true;
                }
            }
            double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
            std::cout<<"{\"generator\":\"calendar-sha256-v1\",\"start\":\""<<start
                     <<"\",\"stop\":\""<<decimal((unsigned __int128)start+count)<<"\",\"tested\":"<<count
                     <<",\"matched_prefix_counts\":[";
            for(unsigned i=0;i<=n;++i) std::cout<<(i?",":"")<<counts[i];
            std::cout<<"],\"survivors\":[";
            for(size_t i=0;i<survivors.size();++i) std::cout<<(i?",":"")<<survivors[i];
            std::cout<<"],\"survivor_overflow\":"<<(survivor_overflow?"true":"false")
                     <<",\"batch_seconds\":"<<elapsed<<"}"<<std::endl;
        }
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; }
}

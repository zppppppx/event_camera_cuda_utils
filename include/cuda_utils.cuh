#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(err__)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Simple random number generator (header-defined to allow inlining in device code)
__device__ __forceinline__ unsigned xorshift32(unsigned &s)
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

__device__ __forceinline__ int rand_int(unsigned &s, int lo, int hi)
{
    unsigned r = xorshift32(s);
    int span = hi - lo + 1;
    return lo + (int)(r % span);
}
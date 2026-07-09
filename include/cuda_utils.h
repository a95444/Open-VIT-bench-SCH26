#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Wrap every CUDA runtime call: if it fails, print where and why, then abort.
// Usage:  CUDA_CHECK(cudaMalloc(&p, n));
#define CUDA_CHECK(call)                                                     \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_err));         \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#endif // __CUDA_UTILS_H__

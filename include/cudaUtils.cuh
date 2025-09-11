#pragma once
#include <cuda_runtime.h>
#include <builtin_types.h>  

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <stdexcept>

#include <cstdio>   
#include <cstdlib> 
#include <cstdint>
#include <cmath>    

inline constexpr unsigned BLOCK_SIZE_X = 256u;
inline constexpr unsigned BLOCK_SIZE_Y = 1u;
inline constexpr unsigned BLOCK_SIZE_Z = 1u;

inline constexpr int HALO   = 1;
inline constexpr int TILE_X = static_cast<int>(BLOCK_SIZE_X) + 2*HALO;
inline constexpr int TILE_Y = static_cast<int>(BLOCK_SIZE_Y) + 2*HALO;
inline constexpr int TILE_Z = static_cast<int>(BLOCK_SIZE_Z) + 2*HALO;

#define ENABLE_FP16

#if defined(ENABLE_FP16)

    #include <cuda_fp16.h>
    using pop_t = __half;

    __host__ __device__ __forceinline__
    pop_t to_pop(
        float x
    ) { 
        return __float2half(x); 
    }

    __host__ __device__ __forceinline__
    float from_pop(
        pop_t x
    ) { 
        return __half2float(x); 
    }

#else

    using pop_t = float;

    __host__ __device__ __forceinline__
    pop_t to_pop(
        float x
    ) { 
        return x; 
    }

    __host__ __device__ __forceinline__
    float from_pop(
        pop_t x
    ) { 
        return x; 
    }

#endif

using ci_t  = int; // probably best type for -1,0,1
using idx_t = uint32_t;

#define checkCudaErrors(err) __checkCudaErrors((err), #err, __FILE__, __LINE__)
#define checkCudaErrorsOutline(err) __checkCudaErrorsOutline((err), #err, __FILE__, __LINE__)
#define getLastCudaError(msg) __getLastCudaError((msg), __FILE__, __LINE__)
#define getLastCudaErrorOutline(msg) __getLastCudaErrorOutline((msg), __FILE__, __LINE__)

void __checkCudaErrorsOutline(
    cudaError_t err,
    const char* const func,
    const char* const file,
    const int line
) noexcept {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s(%d) \"%s\": [%d] %s.\n",
                file, line, func, (int)err, cudaGetErrorString(err));
        fflush(stderr);
        std::abort();
    }
}

inline void __checkCudaErrors(
    cudaError_t err,
    const char* const func,
    const char* const file,
    const int line
) noexcept {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s(%d) \"%s\": [%d] %s.\n",
                file, line, func, (int)err, cudaGetErrorString(err));
        fflush(stderr);
        std::abort();
    }
}

void __getLastCudaErrorOutline(
    const char* const errorMessage,
    const char* const file,
    const int line
) noexcept {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s(%d): [%d] %s. Context: %s\n",
                file, line, (int)err, cudaGetErrorString(err), errorMessage);
        fflush(stderr);
        std::abort();
    }
}

inline void __getLastCudaError(
    const char* const errorMessage,
    const char* const file,
    const int line
) noexcept {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s(%d): [%d] %s. Context: %s\n",
                file, line, (int)err, cudaGetErrorString(err), errorMessage);
        fflush(stderr);
        std::abort();
    }
}




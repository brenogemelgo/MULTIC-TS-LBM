#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <builtin_types.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

static inline constexpr unsigned blockSizeX = 32u;
static inline constexpr unsigned blockSizeY = 2u;
static inline constexpr unsigned blockSizeZ = 2u;

static inline constexpr int haloPad = 1;
static inline constexpr int tileX = static_cast<int>(blockSizeX) + 2 * haloPad;
static inline constexpr int tileY = static_cast<int>(blockSizeY) + 2 * haloPad;
static inline constexpr int tileZ = static_cast<int>(blockSizeZ) + 2 * haloPad;

#if defined(ENABLE_FP16)

    #include <cuda_fp16.h>
    using pop_t = __half;

    [[nodiscard]] __device__ __forceinline__ 
    pop_t to_pop(
        const float x
    ) noexcept {
        return __float2half(x); 
    }

    [[nodiscard]] __device__ __forceinline__ 
    float from_pop(
        const pop_t x
    ) noexcept {
        return __half2float(x);
    }

#else

    using pop_t = float;

    [[nodiscard]] __device__ __forceinline__ constexpr 
    pop_t to_pop(
        const float x
    ) noexcept {
        return x;
    }

    [[nodiscard]] __device__ __forceinline__ constexpr
    float from_pop(
        const pop_t x
    ) noexcept {
        return x;
    }

#endif

using ci_t = int;
using idx_t = uint32_t;
using seed_t = uint32_t;

#define checkCudaErrors(err) __checkCudaErrors((err), #err, __FILE__, __LINE__)
#define checkCudaErrorsOutline(err) __checkCudaErrorsOutline((err), #err, __FILE__, __LINE__)
#define getLastCudaError(msg) __getLastCudaError((msg), __FILE__, __LINE__)
#define getLastCudaErrorOutline(msg) __getLastCudaErrorOutline((msg), __FILE__, __LINE__)

static __host__ 
void __checkCudaErrorsOutline(
    cudaError_t err,
    const char *const func,
    const char *const file,
    const int line
) noexcept {
    if (err != cudaSuccess) {
        fprintf(
            stderr, "CUDA error at %s(%d) \"%s\": [%d] %s.\n", file, line, func, (int)err, cudaGetErrorString(err));
        fflush(stderr);
        std::abort();
    }
}

static __host__ 
void __getLastCudaErrorOutline(
    const char *const errorMessage,
    const char *const file,
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

static __host__ __forceinline__
void __checkCudaErrors(
    cudaError_t err,
    const char *const func,
    const char *const file,
    const int line
) noexcept {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s(%d) \"%s\": [%d] %s.\n",
                file, line, func, (int)err, cudaGetErrorString(err));
        fflush(stderr);
        std::abort();
    }
}

static __host__ __forceinline__
void __getLastCudaError(
    const char *const errorMessage,
    const char *const file,
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

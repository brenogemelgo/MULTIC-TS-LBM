#pragma once
#include "cudaUtils.cuh"

#if defined(D3Q19) //             0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18

__constant__ ci_t CIX[19] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0};
__constant__ ci_t CIY[19] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1};
__constant__ ci_t CIZ[19] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1};
__constant__ scalar_t W[19] = {1.0f / 3.0f,
                               1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
                               1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};
static constexpr scalar_t W_0 = 1.0f / 3.0f;  // 0
static constexpr scalar_t W_1 = 1.0f / 18.0f; // 1 to 6
static constexpr scalar_t W_2 = 1.0f / 36.0f; // 7 to 18
static constexpr label_t NLINKS = 19;

#elif defined(D3Q27) //           0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26

__constant__ ci_t CIX[27] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};
__constant__ ci_t CIY[27] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1};
__constant__ ci_t CIZ[27] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1};
__constant__ scalar_t W[27] = {8.0f / 27.0f,
                               2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f,
                               1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f,
                               1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f};
static constexpr scalar_t W_0 = 8.0f / 27.0f;  // 0
static constexpr scalar_t W_1 = 2.0f / 27.0f;  // 1 to 6
static constexpr scalar_t W_2 = 1.0f / 54.0f;  // 7 to 18
static constexpr scalar_t W_3 = 1.0f / 216.0f; // 19 to 26
static constexpr label_t NLINKS = 27;

#endif

namespace VelocitySet
{

    template <label_t Q>
    struct Dir;

    template <>
    struct Dir<0>
    {
        static constexpr int cx = 0, cy = 0, cz = 0;
        static constexpr scalar_t w = W_0;
    };
    template <>
    struct Dir<1>
    {
        static constexpr int cx = 1, cy = 0, cz = 0;
        static constexpr scalar_t w = W_1;
    };
    template <>
    struct Dir<2>
    {
        static constexpr int cx = -1, cy = 0, cz = 0;
        static constexpr scalar_t w = W_1;
    };
    template <>
    struct Dir<3>
    {
        static constexpr int cx = 0, cy = 1, cz = 0;
        static constexpr scalar_t w = W_1;
    };
    template <>
    struct Dir<4>
    {
        static constexpr int cx = 0, cy = -1, cz = 0;
        static constexpr scalar_t w = W_1;
    };
    template <>
    struct Dir<5>
    {
        static constexpr int cx = 0, cy = 0, cz = 1;
        static constexpr scalar_t w = W_1;
    };
    template <>
    struct Dir<6>
    {
        static constexpr int cx = 0, cy = 0, cz = -1;
        static constexpr scalar_t w = W_1;
    };
    template <>
    struct Dir<7>
    {
        static constexpr int cx = 1, cy = 1, cz = 0;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct Dir<8>
    {
        static constexpr int cx = -1, cy = -1, cz = 0;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct Dir<9>
    {
        static constexpr int cx = 1, cy = 0, cz = 1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct Dir<10>
    {
        static constexpr int cx = -1, cy = 0, cz = -1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct Dir<11>
    {
        static constexpr int cx = 0, cy = 1, cz = 1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct Dir<12>
    {
        static constexpr int cx = 0, cy = -1, cz = -1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct Dir<13>
    {
        static constexpr int cx = 1, cy = -1, cz = 0;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct Dir<14>
    {
        static constexpr int cx = -1, cy = 1, cz = 0;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct Dir<15>
    {
        static constexpr int cx = 1, cy = 0, cz = -1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct Dir<16>
    {
        static constexpr int cx = -1, cy = 0, cz = 1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct Dir<17>
    {
        static constexpr int cx = 0, cy = 1, cz = -1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct Dir<18>
    {
        static constexpr int cx = 0, cy = -1, cz = 1;
        static constexpr scalar_t w = W_2;
    };

#if defined(D3Q27)

    template <>
    struct Dir<19>
    {
        static constexpr int cx = 1, cy = 1, cz = 1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct Dir<20>
    {
        static constexpr int cx = -1, cy = -1, cz = -1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct Dir<21>
    {
        static constexpr int cx = 1, cy = 1, cz = -1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct Dir<22>
    {
        static constexpr int cx = -1, cy = -1, cz = 1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct Dir<23>
    {
        static constexpr int cx = 1, cy = -1, cz = 1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct Dir<24>
    {
        static constexpr int cx = -1, cy = 1, cz = -1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct Dir<25>
    {
        static constexpr int cx = -1, cy = 1, cz = 1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct Dir<26>
    {
        static constexpr int cx = 1, cy = -1, cz = -1;
        static constexpr scalar_t w = W_3;
    };

#endif

}
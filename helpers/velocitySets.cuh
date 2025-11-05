#pragma once
#include "cudaUtils.cuh"

#define G_LOW_ORDER
// #define G_HIGH_ORDER

#if defined(D3Q19)

static constexpr label_t FLINKS = 19;

__constant__ ci_t CIX[FLINKS] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0};
__constant__ ci_t CIY[FLINKS] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1};
__constant__ ci_t CIZ[FLINKS] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1};

static constexpr scalar_t W_0 = 1.0f / 3.0f;
static constexpr scalar_t W_1 = 1.0f / 18.0f;
static constexpr scalar_t W_2 = 1.0f / 36.0f;

__constant__ scalar_t W[FLINKS] = {W_0,
                                   W_1, W_1, W_1, W_1, W_1, W_1,
                                   W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2};

#elif defined(D3Q27)

static constexpr label_t FLINKS = 27;

__constant__ ci_t CIX[FLINKS] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};
__constant__ ci_t CIY[FLINKS] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1};
__constant__ ci_t CIZ[FLINKS] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1};

static constexpr scalar_t W_0 = 8.0f / 27.0f;  // 0
static constexpr scalar_t W_1 = 2.0f / 27.0f;  // 1 to 6
static constexpr scalar_t W_2 = 1.0f / 54.0f;  // 7 to 18
static constexpr scalar_t W_3 = 1.0f / 216.0f; // 19 to 26

__constant__ scalar_t W[FLINKS] = {W_0,
                                   W_1, W_1, W_1, W_1, W_1, W_1,
                                   W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2, W_2,
                                   W_3, W_3, W_3, W_3, W_3, W_3, W_3, W_3};

#endif

#if defined(G_LOW_ORDER)

static constexpr label_t GLINKS = 7;

static constexpr scalar_t WG_0 = 1.0f / 4.0f; // 0
static constexpr scalar_t WG_1 = 1.0f / 8.0f; // 1 to 6

__constant__ scalar_t W_G[GLINKS] = {1.0f / 4.0f,
                                     1.0f / 8.0f, 1.0f / 8.0f, 1.0f / 8.0f, 1.0f / 8.0f, 1.0f / 8.0f, 1.0f / 8.0f};

static constexpr scalar_t AS2_H = 3.0f;
static constexpr scalar_t AS2_P = 4.0f;

#elif defined(G_HIGH_ORDER)

static constexpr label_t GLINKS = FLINKS;

static constexpr scalar_t WG_0 = W_0;
static constexpr scalar_t WG_1 = W_1;
static constexpr scalar_t WG_2 = W_2;

#define W_G W

#if defined(D3Q27)

static constexpr scalar_t WG_3 = W_3;

#endif

static constexpr scalar_t AS2_H = 3.0f;
static constexpr scalar_t AS2_P = 3.0f;

#endif

namespace VelocitySet
{

    template <label_t Q>
    struct F;

    template <>
    struct F<0>
    {
        static constexpr int cx = 0, cy = 0, cz = 0;
        static constexpr scalar_t w = W_0;
    };
    template <>
    struct F<1>
    {
        static constexpr int cx = 1, cy = 0, cz = 0;
        static constexpr scalar_t w = W_1;
    };
    template <>
    struct F<2>
    {
        static constexpr int cx = -1, cy = 0, cz = 0;
        static constexpr scalar_t w = W_1;
    };
    template <>
    struct F<3>
    {
        static constexpr int cx = 0, cy = 1, cz = 0;
        static constexpr scalar_t w = W_1;
    };
    template <>
    struct F<4>
    {
        static constexpr int cx = 0, cy = -1, cz = 0;
        static constexpr scalar_t w = W_1;
    };
    template <>
    struct F<5>
    {
        static constexpr int cx = 0, cy = 0, cz = 1;
        static constexpr scalar_t w = W_1;
    };
    template <>
    struct F<6>
    {
        static constexpr int cx = 0, cy = 0, cz = -1;
        static constexpr scalar_t w = W_1;
    };
    template <>
    struct F<7>
    {
        static constexpr int cx = 1, cy = 1, cz = 0;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct F<8>
    {
        static constexpr int cx = -1, cy = -1, cz = 0;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct F<9>
    {
        static constexpr int cx = 1, cy = 0, cz = 1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct F<10>
    {
        static constexpr int cx = -1, cy = 0, cz = -1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct F<11>
    {
        static constexpr int cx = 0, cy = 1, cz = 1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct F<12>
    {
        static constexpr int cx = 0, cy = -1, cz = -1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct F<13>
    {
        static constexpr int cx = 1, cy = -1, cz = 0;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct F<14>
    {
        static constexpr int cx = -1, cy = 1, cz = 0;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct F<15>
    {
        static constexpr int cx = 1, cy = 0, cz = -1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct F<16>
    {
        static constexpr int cx = -1, cy = 0, cz = 1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct F<17>
    {
        static constexpr int cx = 0, cy = 1, cz = -1;
        static constexpr scalar_t w = W_2;
    };
    template <>
    struct F<18>
    {
        static constexpr int cx = 0, cy = -1, cz = 1;
        static constexpr scalar_t w = W_2;
    };

#if defined(D3Q27)

    template <>
    struct F<19>
    {
        static constexpr int cx = 1, cy = 1, cz = 1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct F<20>
    {
        static constexpr int cx = -1, cy = -1, cz = -1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct F<21>
    {
        static constexpr int cx = 1, cy = 1, cz = -1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct F<22>
    {
        static constexpr int cx = -1, cy = -1, cz = 1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct F<23>
    {
        static constexpr int cx = 1, cy = -1, cz = 1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct F<24>
    {
        static constexpr int cx = -1, cy = 1, cz = -1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct F<25>
    {
        static constexpr int cx = -1, cy = 1, cz = 1;
        static constexpr scalar_t w = W_3;
    };
    template <>
    struct F<26>
    {
        static constexpr int cx = 1, cy = -1, cz = -1;
        static constexpr scalar_t w = W_3;
    };

#endif

    template <label_t Q>
    struct G;

    template <>
    struct G<0>
    {
        static constexpr int cx = 0, cy = 0, cz = 0;
        static constexpr scalar_t wg = WG_0;
    };
    template <>
    struct G<1>
    {
        static constexpr int cx = 1, cy = 0, cz = 0;
        static constexpr scalar_t wg = WG_1;
    };
    template <>
    struct G<2>
    {
        static constexpr int cx = -1, cy = 0, cz = 0;
        static constexpr scalar_t wg = WG_1;
    };
    template <>
    struct G<3>
    {
        static constexpr int cx = 0, cy = 1, cz = 0;
        static constexpr scalar_t wg = WG_1;
    };
    template <>
    struct G<4>
    {
        static constexpr int cx = 0, cy = -1, cz = 0;
        static constexpr scalar_t wg = WG_1;
    };
    template <>
    struct G<5>
    {
        static constexpr int cx = 0, cy = 0, cz = 1;
        static constexpr scalar_t wg = WG_1;
    };
    template <>
    struct G<6>
    {
        static constexpr int cx = 0, cy = 0, cz = -1;
        static constexpr scalar_t wg = WG_1;
    };

#if defined(G_HIGH_ORDER)

    template <>
    struct G<7>
    {
        static constexpr int cx = 1, cy = 1, cz = 0;
        static constexpr scalar_t wg = WG_2;
    };
    template <>
    struct G<8>
    {
        static constexpr int cx = -1, cy = -1, cz = 0;
        static constexpr scalar_t wg = WG_2;
    };
    template <>
    struct G<9>
    {
        static constexpr int cx = 1, cy = 0, cz = 1;
        static constexpr scalar_t wg = WG_2;
    };
    template <>
    struct G<10>
    {
        static constexpr int cx = -1, cy = 0, cz = -1;
        static constexpr scalar_t wg = WG_2;
    };
    template <>
    struct G<11>
    {
        static constexpr int cx = 0, cy = 1, cz = 1;
        static constexpr scalar_t wg = WG_2;
    };
    template <>
    struct G<12>
    {
        static constexpr int cx = 0, cy = -1, cz = -1;
        static constexpr scalar_t wg = WG_2;
    };
    template <>
    struct G<13>
    {
        static constexpr int cx = 1, cy = -1, cz = 0;
        static constexpr scalar_t wg = WG_2;
    };
    template <>
    struct G<14>
    {
        static constexpr int cx = -1, cy = 1, cz = 0;
        static constexpr scalar_t wg = WG_2;
    };
    template <>
    struct G<15>
    {
        static constexpr int cx = 1, cy = 0, cz = -1;
        static constexpr scalar_t wg = WG_2;
    };
    template <>
    struct G<16>
    {
        static constexpr int cx = -1, cy = 0, cz = 1;
        static constexpr scalar_t wg = WG_2;
    };
    template <>
    struct G<17>
    {
        static constexpr int cx = 0, cy = 1, cz = -1;
        static constexpr scalar_t wg = WG_2;
    };
    template <>
    struct G<18>
    {
        static constexpr int cx = 0, cy = -1, cz = 1;
        static constexpr scalar_t wg = WG_2;
    };

#if defined(D3Q27)

    template <>
    struct G<19>
    {
        static constexpr int cx = 1, cy = 1, cz = 1;
        static constexpr scalar_t wg = WG_3;
    };
    template <>
    struct G<20>
    {
        static constexpr int cx = -1, cy = -1, cz = -1;
        static constexpr scalar_t wg = WG_3;
    };
    template <>
    struct G<21>
    {
        static constexpr int cx = 1, cy = 1, cz = -1;
        static constexpr scalar_t wg = WG_3;
    };
    template <>
    struct G<22>
    {
        static constexpr int cx = -1, cy = -1, cz = 1;
        static constexpr scalar_t wg = WG_3;
    };
    template <>
    struct G<23>
    {
        static constexpr int cx = 1, cy = -1, cz = 1;
        static constexpr scalar_t wg = WG_3;
    };
    template <>
    struct G<24>
    {
        static constexpr int cx = -1, cy = 1, cz = -1;
        static constexpr scalar_t wg = WG_3;
    };
    template <>
    struct G<25>
    {
        static constexpr int cx = -1, cy = 1, cz = 1;
        static constexpr scalar_t wg = WG_3;
    };
    template <>
    struct G<26>
    {
        static constexpr int cx = 1, cy = -1, cz = -1;
        static constexpr scalar_t wg = WG_3;
    };

#endif
#endif
}
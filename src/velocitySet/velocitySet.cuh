/*---------------------------------------------------------------------------*\
|                                                                             |
| MULTIC-TS-LBM: CUDA-based multicomponent Lattice Boltzmann Method           |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/brenogemelgo/MULTIC-TS-LBM                       |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Breno Gemelgo (Geoenergia Lab, UDESC)

License
    This file is part of MULTIC-TS-LBM.

    MULTIC-TS-LBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    Definition of velocity sets and their associated weights for the
    Lattice Boltzmann Method (LBM) simulations. It includes definitions
    for both D3Q19 and D3Q27 velocity models, as well as D3Q7 for phase
    field and passive scalars if enabled

SourceFiles
    velocitySet.cuh

\*---------------------------------------------------------------------------*/

#ifndef VELOCITYSET_CUH
#define VELOCITYSET_CUH

#include "../cuda/utils.cuh"

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

static constexpr label_t GLINKS = 7;

static constexpr scalar_t WG_0 = 1.0f / 4.0f; // 0
static constexpr scalar_t WG_1 = 1.0f / 8.0f; // 1 to 6

__constant__ scalar_t W_G[GLINKS] = {1.0f / 4.0f,
                                     1.0f / 8.0f, 1.0f / 8.0f, 1.0f / 8.0f, 1.0f / 8.0f, 1.0f / 8.0f, 1.0f / 8.0f};

#if PASSIVE_SCALAR

static constexpr label_t HLINKS = 7;

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

#if PASSIVE_SCALAR

    template <label_t Q>
    struct H;

    template <>
    struct H<0>
    {
        static constexpr int cx = 0, cy = 0, cz = 0;
        static constexpr scalar_t wh = WG_0;
    };

    template <>
    struct H<1>
    {
        static constexpr int cx = 1, cy = 0, cz = 0;
        static constexpr scalar_t wh = WG_1;
    };

    template <>
    struct H<2>
    {
        static constexpr int cx = -1, cy = 0, cz = 0;
        static constexpr scalar_t wh = WG_1;
    };

    template <>
    struct H<3>
    {
        static constexpr int cx = 0, cy = 1, cz = 0;
        static constexpr scalar_t wh = WG_1;
    };

    template <>
    struct H<4>
    {
        static constexpr int cx = 0, cy = -1, cz = 0;
        static constexpr scalar_t wh = WG_1;
    };

    template <>
    struct H<5>
    {
        static constexpr int cx = 0, cy = 0, cz = 1;
        static constexpr scalar_t wh = WG_1;
    };

    template <>
    struct H<6>
    {
        static constexpr int cx = 0, cy = 0, cz = -1;
        static constexpr scalar_t wh = WG_1;
    };

#endif
}

#endif
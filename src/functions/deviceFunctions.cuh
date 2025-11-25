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
    Device functions for various calculations

Namespace
    device

SourceFiles
    deviceFunctions.cuh

\*---------------------------------------------------------------------------*/

#ifndef DEVICEFUNCTIONS_CUH
#define DEVICEFUNCTIONS_CUH

#include "constants.cuh"
#include "globalFunctions.cuh"

namespace device
{
    __device__ [[nodiscard]] inline label_t global3(
        const label_t x,
        const label_t y,
        const label_t z) noexcept
    {
        return x + y * mesh::nx + z * size::stride();
    }

    __device__ [[nodiscard]] inline label_t global4(
        const label_t x,
        const label_t y,
        const label_t z,
        const label_t Q) noexcept
    {
        return Q * size::plane() + global3(x, y, z);
    }

    __device__ [[nodiscard]] inline label_t globalThreadIdx(
        const label_t tx,
        const label_t ty,
        const label_t tz,
        const label_t bx,
        const label_t by,
        const label_t bz) noexcept
    {
        return tx + block::nx * (ty + block::ny * (tz + block::nz * (bx + block::num_block_x() * (by + block::num_block_y() * bz))));
    }

    template <const int Dir>
    __device__ [[nodiscard]] static inline constexpr label_t safeStream(const label_t x) noexcept
    {
        if constexpr (Dir == -1)
        {
            return x - 1;
        }
        else if constexpr (Dir == 1)
        {
            return x + 1;
        }
        else
        {
            return x;
        }
    }

#if defined(JET)

    __device__ [[nodiscard]] inline scalar_t cubic_sponge(const label_t z) noexcept
    {
        const scalar_t zn = static_cast<scalar_t>(z) * sponge::inv_nz_m1();
        const scalar_t s = fminf(fmaxf((zn - sponge::z_start()) * sponge::inv_sponge(), 0.0f), 1.0f);
        const scalar_t s2 = s * s;
        const scalar_t ramp = s2 * s;
        return fmaf(ramp, relaxation::omega_delta(), relaxation::omega_ref());
    }

#endif

    __device__ [[nodiscard]] inline scalar_t smoothstep(
        const scalar_t edge0,
        const scalar_t edge1,
        scalar_t x) noexcept
    {
        x = __saturatef((x - edge0) / (edge1 - edge0));
        return x * x * (3.0f - 2.0f * x);
    }

    __device__ [[nodiscard]] inline scalar_t interpolate_rho(scalar_t phi) noexcept
    {
        return fmaf(phi, (physics::rho_oil() - physics::rho_water()), physics::rho_water());
    }

    template <const label_t Start, const label_t End, typename F>
    __device__ inline constexpr void constexpr_for(F &&f) noexcept
    {
        if constexpr (Start < End)
        {
            f(integralConstant<label_t, Start>());
            if constexpr (Start + 1 < End)
            {
                constexpr_for<Start + 1, End>(std::forward<F>(f));
            }
        }
    }
}

#endif
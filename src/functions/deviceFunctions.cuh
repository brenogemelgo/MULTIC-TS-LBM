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

    __device__ [[nodiscard]] inline label_t wrapX(const label_t xx) noexcept
    {
        if (xx == 0)
        {
            return mesh::nx - 2;
        }
        if (xx == mesh::nx - 1)
        {
            return 1;
        }
        return xx;
    }

    __device__ [[nodiscard]] inline label_t wrapY(const label_t yy) noexcept
    {
        if (yy == 0)
        {
            return mesh::ny - 2;
        }
        if (yy == mesh::ny - 1)
        {
            return 1;
        }
        return yy;
    }

    __device__ [[nodiscard]] inline label_t wrapZ(const label_t zz) noexcept
    {
        if (zz == 0)
        {
            return mesh::nz - 2;
        }
        if (zz == mesh::nz - 1)
        {
            return 1;
        }
        return zz;
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

    __device__ static inline void prefetch_L2(const void *ptr) noexcept
    {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
    }

    __device__ [[nodiscard]] inline scalar_t cubic_sponge(const label_t z) noexcept
    {
        if constexpr (LBM::FlowCase::jet_case())
        {
            const scalar_t zn = static_cast<scalar_t>(z) * sponge::inv_nz_m1();
            const scalar_t s = math::min(math::max((zn - sponge::z_start()) * sponge::inv_sponge(), static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
            const scalar_t s2 = s * s;
            const scalar_t ramp = s2 * s;
            return fmaf(ramp, relaxation::omega_delta(), relaxation::omega_ref());
        }
        else if constexpr (LBM::FlowCase::droplet_case())
        {
            return 0;
        }
    }
}

#endif
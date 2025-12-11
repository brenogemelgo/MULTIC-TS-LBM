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
    Reynolds moments kernels

Namespace
    LBM

SourceFiles
    reynoldsMoments.cuh

\*---------------------------------------------------------------------------*/

#ifndef REYNOLDSMOMENTS_CUH
#define REYNOLDSMOMENTS_CUH

#include "functions/ioFields.cuh"

#if D_REYNOLDS_MOMENTS

namespace LBM
{
    __global__ void reynoldsMomentsAverage(
        LBMFields d,
        const label_t t)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        const scalar_t ux = d.ux[idx3];
        const scalar_t uy = d.uy[idx3];
        const scalar_t uz = d.uz[idx3];

        const scalar_t uxux = ux * ux;
        const scalar_t uyuy = uy * uy;
        const scalar_t uzuz = uz * uz;

        const scalar_t uxuy = ux * uy;
        const scalar_t uxuz = ux * uz;
        const scalar_t uyuz = uy * uz;

        auto update = [t] __device__(scalar_t oldv, scalar_t instv)
        {
            return oldv + (instv - oldv) / static_cast<scalar_t>(t);
        };

        d.avg_uxux[idx3] = update(d.avg_uxux[idx3], uxux);
        d.avg_uyuy[idx3] = update(d.avg_uyuy[idx3], uyuy);
        d.avg_uzuz[idx3] = update(d.avg_uzuz[idx3], uzuz);
        d.avg_uxuy[idx3] = update(d.avg_uxuy[idx3], uxuy);
        d.avg_uxuz[idx3] = update(d.avg_uxuz[idx3], uxuz);
        d.avg_uyuz[idx3] = update(d.avg_uyuz[idx3], uyuz);
    }
}

namespace Derived
{
    namespace Reynolds
    {
        constexpr bool enabled =
#if D_REYNOLDS_MOMENTS
            true;
#else
            false;
#endif

        constexpr std::array<host::FieldConfig, 6> fields{{
            {host::FieldID::Avg_uxux, "avg_uxux", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_uyuy, "avg_uyuy", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_uzuz, "avg_uzuz", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_uxuy, "avg_uxuy", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_uxuz, "avg_uxuz", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_uyuz, "avg_uyuz", host::FieldDumpShape::Grid3D, true},
        }};

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void launch(
            cudaStream_t queue,
            LBMFields d,
            const label_t t) noexcept
        {
#if D_REYNOLDS_MOMENTS
            LBM::reynoldsMomentsAverage<<<grid, block, dynamic, queue>>>(d, t + 1);
#endif
        }

        __host__ static inline void free(LBMFields &d)
        {
#if D_REYNOLDS_MOMENTS
            cudaFree(d.avg_uxux);
            cudaFree(d.avg_uyuy);
            cudaFree(d.avg_uzuz);
            cudaFree(d.avg_uxuy);
            cudaFree(d.avg_uxuz);
            cudaFree(d.avg_uyuz);
#endif
        }
    }
}

#endif

#endif

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
    Time averaging kernels

Namespace
    LBM

SourceFiles
    timeAverage.cuh

\*---------------------------------------------------------------------------*/

#ifndef TIMEAVERAGE_CUH
#define TIMEAVERAGE_CUH

#include "functions/ioFields.cuh"

#if D_TIMEAVG

namespace LBM
{
    __global__ void timeAverage(
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

        const scalar_t phi = d.phi[idx3];
        const scalar_t ux = d.ux[idx3];
        const scalar_t uy = d.uy[idx3];
        const scalar_t uz = d.uz[idx3];

        const scalar_t umag = sqrt(ux * ux + uy * uy + uz * uz);

        auto update = [t] __device__(scalar_t old_val, scalar_t new_val)
        {
            return old_val + (new_val - old_val) / static_cast<scalar_t>(t);
        };

        d.avg_phi[idx3] = update(d.avg_phi[idx3], phi);
        d.avg_uz[idx3] = update(d.avg_uz[idx3], uz);
        d.avg_umag[idx3] = update(d.avg_umag[idx3], umag);
    }
}

namespace Derived
{
    namespace TimeAvg
    {
        constexpr bool enabled =
#if D_TIMEAVG
            true;
#else
            false;
#endif

        constexpr std::array<host::FieldConfig, 3> fields{{
            {host::FieldID::Avg_phi, "avg_phi", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_uz, "avg_uz", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_umag, "avg_umag", host::FieldDumpShape::Grid3D, true},
        }};

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void launch(
            cudaStream_t queue,
            LBMFields d,
            const label_t t) noexcept
        {
#if D_TIMEAVG
            LBM::timeAverage<<<grid, block, dynamic, queue>>>(d, t + 1);
#endif
        }

    }
}

#endif

#endif

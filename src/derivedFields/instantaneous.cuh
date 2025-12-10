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
    Instantaneous derived fields (no neighbors)

Namespace
    LBM

SourceFiles
    instantaneous.cuh

\*---------------------------------------------------------------------------*/

#ifndef INSTANTANEOUS_CUH
#define INSTANTANEOUS_CUH

#include "../cuda/utils.cuh"
#include "functions/ioFields.cuh"

#if D_INSTANTANEOUS

namespace LBM
{
    __global__ void computeKinematics(LBMFields d)
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

        const scalar_t umag2 = ux * ux + uy * uy + uz * uz;
        const scalar_t umag = sqrt(umag2);

        d.umag[idx3] = umag;
    }

    __global__ void computeEnergyFields(LBMFields d)
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
        const scalar_t rho = d.rho[idx3];

        const scalar_t umag2 = ux * ux + uy * uy + uz * uz;

        const scalar_t k = static_cast<scalar_t>(0.5) * umag2;

        const scalar_t q = static_cast<scalar_t>(0.5) * rho * umag2;

        d.k[idx3] = k;
        d.q_dyn[idx3] = q;
    }
}

namespace Derived
{
    namespace Instant
    {

        constexpr bool enabled =

#if D_INSTANTANEOUS

            true;

#else

            false;

#endif

        constexpr std::array<host::FieldConfig, 4> fields{{
            {host::FieldID::Umag, "umag", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::K, "k", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Q_dyn, "q_dyn", host::FieldDumpShape::Grid3D, true},
        }};

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void launch(
            cudaStream_t queue,
            LBMFields d) noexcept
        {

#if D_INSTANTANEOUS

            LBM::computeKinematics<<<grid, block, dynamic, queue>>>(d);
            LBM::computeEnergyFields<<<grid, block, dynamic, queue>>>(d);

#endif
        }

    }
}

#endif

#endif

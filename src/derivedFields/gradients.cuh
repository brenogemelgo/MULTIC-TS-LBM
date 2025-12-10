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
    Gradient-based derived fields kernels

Namespace
    LBM

SourceFiles
    gradients.cuh

\*---------------------------------------------------------------------------*/

#ifndef GRADIENTS_CUH
#define GRADIENTS_CUH

#include "functions/ioFields.cuh"

#if D_GRADIENTS

namespace LBM
{
    __device__ [[nodiscard]] static inline scalar_t fetch(
        const scalar_t *f,
        label_t x,
        label_t y,
        label_t z) noexcept
    {
        if (x >= mesh::nx)
        {
            x = mesh::nx - 1;
        }
        if (y >= mesh::ny)
        {
            y = mesh::ny - 1;
        }
        if (z >= mesh::nz)
        {
            z = mesh::nz - 1;
        }

        return f[device::global3(x, y, z)];
    }

    __global__ void computeVorticity(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        const scalar_t ux_xp = fetch(d.ux, x + 1, y, z);
        const scalar_t ux_xm = fetch(d.ux, x - 1, y, z);
        const scalar_t ux_yp = fetch(d.ux, x, y + 1, z);
        const scalar_t ux_ym = fetch(d.ux, x, y - 1, z);
        const scalar_t ux_zp = fetch(d.ux, x, y, z + 1);
        const scalar_t ux_zm = fetch(d.ux, x, y, z - 1);

        const scalar_t uy_xp = fetch(d.uy, x + 1, y, z);
        const scalar_t uy_xm = fetch(d.uy, x - 1, y, z);
        const scalar_t uy_yp = fetch(d.uy, x, y + 1, z);
        const scalar_t uy_ym = fetch(d.uy, x, y - 1, z);
        const scalar_t uy_zp = fetch(d.uy, x, y, z + 1);
        const scalar_t uy_zm = fetch(d.uy, x, y, z - 1);

        const scalar_t uz_xp = fetch(d.uz, x + 1, y, z);
        const scalar_t uz_xm = fetch(d.uz, x - 1, y, z);
        const scalar_t uz_yp = fetch(d.uz, x, y + 1, z);
        const scalar_t uz_ym = fetch(d.uz, x, y - 1, z);
        const scalar_t uz_zp = fetch(d.uz, x, y, z + 1);
        const scalar_t uz_zm = fetch(d.uz, x, y, z - 1);

        const scalar_t dux_dy = (ux_yp - ux_ym) * 0.5;
        const scalar_t dux_dz = (ux_zp - ux_zm) * 0.5;

        const scalar_t duy_dx = (uy_xp - uy_xm) * 0.5;
        const scalar_t duy_dz = (uy_zp - uy_zm) * 0.5;

        const scalar_t duz_dx = (uz_xp - uz_xm) * 0.5;
        const scalar_t duz_dy = (uz_yp - uz_ym) * 0.5;

        const scalar_t wx = duz_dy - duy_dz;
        const scalar_t wy = dux_dz - duz_dx;
        const scalar_t wz = duy_dx - dux_dy;

        const scalar_t vort = sqrt(wx * wx + wy * wy + wz * wz);

        d.vort[idx3] = vort;
    }

    __global__ void computeQCriterion(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        const scalar_t ux_xp = fetch(d.ux, x + 1, y, z);
        const scalar_t ux_xm = fetch(d.ux, x - 1, y, z);
        const scalar_t ux_yp = fetch(d.ux, x, y + 1, z);
        const scalar_t ux_ym = fetch(d.ux, x, y - 1, z);
        const scalar_t ux_zp = fetch(d.ux, x, y, z + 1);
        const scalar_t ux_zm = fetch(d.ux, x, y, z - 1);

        const scalar_t uy_xp = fetch(d.uy, x + 1, y, z);
        const scalar_t uy_xm = fetch(d.uy, x - 1, y, z);
        const scalar_t uy_yp = fetch(d.uy, x, y + 1, z);
        const scalar_t uy_ym = fetch(d.uy, x, y - 1, z);
        const scalar_t uy_zp = fetch(d.uy, x, y, z + 1);
        const scalar_t uy_zm = fetch(d.uy, x, y, z - 1);

        const scalar_t uz_xp = fetch(d.uz, x + 1, y, z);
        const scalar_t uz_xm = fetch(d.uz, x - 1, y, z);
        const scalar_t uz_yp = fetch(d.uz, x, y + 1, z);
        const scalar_t uz_ym = fetch(d.uz, x, y - 1, z);
        const scalar_t uz_zp = fetch(d.uz, x, y, z + 1);
        const scalar_t uz_zm = fetch(d.uz, x, y, z - 1);

        const scalar_t dux_dx = (ux_xp - ux_xm) * 0.5;
        const scalar_t dux_dy = (ux_yp - ux_ym) * 0.5;
        const scalar_t dux_dz = (ux_zp - ux_zm) * 0.5;

        const scalar_t duy_dx = (uy_xp - uy_xm) * 0.5;
        const scalar_t duy_dy = (uy_yp - uy_ym) * 0.5;
        const scalar_t duy_dz = (uy_zp - uy_zm) * 0.5;

        const scalar_t duz_dx = (uz_xp - uz_xm) * 0.5;
        const scalar_t duz_dy = (uz_yp - uz_ym) * 0.5;
        const scalar_t duz_dz = (uz_zp - uz_zm) * 0.5;

        const scalar_t Sxx = dux_dx;
        const scalar_t Syy = duy_dy;
        const scalar_t Szz = duz_dz;

        const scalar_t Sxy = static_cast<scalar_t>(0.5) * (dux_dy + duy_dx);
        const scalar_t Sxz = static_cast<scalar_t>(0.5) * (dux_dz + duz_dx);
        const scalar_t Syz = static_cast<scalar_t>(0.5) * (duy_dz + duz_dy);

        const scalar_t Oxy = static_cast<scalar_t>(0.5) * (dux_dy - duy_dx);
        const scalar_t Oxz = static_cast<scalar_t>(0.5) * (dux_dz - duz_dx);
        const scalar_t Oyz = static_cast<scalar_t>(0.5) * (duy_dz - duz_dy);

        const scalar_t S2 = Sxx * Sxx + Syy * Syy + Szz * Szz + static_cast<scalar_t>(2.0) * (Sxy * Sxy + Sxz * Sxz + Syz * Syz);

        const scalar_t O2 = static_cast<scalar_t>(2.0) * (Oxy * Oxy + Oxz * Oxz + Oyz * Oyz);

        const scalar_t Q = static_cast<scalar_t>(0.5) * (O2 - S2);

        d.q_crit[idx3] = Q;
    }
}

#endif

namespace Derived
{
    namespace Gradients
    {
        constexpr bool enabled =
#if D_GRADIENTS
            true;
#else
            false;
#endif

        constexpr std::array<host::FieldConfig, 2> fields{{
            {host::FieldID::Vort, "vort", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Q_crit, "q_crit", host::FieldDumpShape::Grid3D, true},
        }};

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void launch(
            cudaStream_t queue,
            LBMFields d) noexcept
        {
#if D_GRADIENTS
            LBM::computeVorticity<<<grid, block, dynamic, queue>>>(d);
            LBM::computeQCriterion<<<grid, block, dynamic, queue>>>(d);
#endif
        }

    }
}

#endif

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
    Initial conditions kernels

Namespace
    LBM

SourceFiles
    initialConditions.cuh

\*---------------------------------------------------------------------------*/

#ifndef INITIALCONDITIONS_CUH
#define INITIALCONDITIONS_CUH

namespace LBM
{
    __global__ void setFields(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        d.ux[idx3] = 0.0f;
        d.uy[idx3] = 0.0f;
        d.uz[idx3] = 0.0f;
        d.phi[idx3] = 0.0f;
        d.ffx[idx3] = 0.0f;
        d.ffy[idx3] = 0.0f;
        d.ffz[idx3] = 0.0f;
        d.ind[idx3] = 0.0f;
        d.normx[idx3] = 0.0f;
        d.normy[idx3] = 0.0f;
        d.normz[idx3] = 0.0f;
        d.rho[idx3] = 1.0f;
        d.pxx[idx3] = 0.0f;
        d.pyy[idx3] = 0.0f;
        d.pzz[idx3] = 0.0f;
        d.pxy[idx3] = 0.0f;
        d.pxz[idx3] = 0.0f;
        d.pyz[idx3] = 0.0f;

#if AVERAGE_UZ

        d.avg[idx3] = 0.0f;

#endif
    }

#if defined(JET)

    __global__ void setJet(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= mesh::nx || y >= mesh::ny)
        {
            return;
        }

        const scalar_t dx = static_cast<scalar_t>(x) - geometry::center_x();
        const scalar_t dy = static_cast<scalar_t>(y) - geometry::center_y();
        const scalar_t r2 = dx * dx + dy * dy;

        if (r2 > geometry::R2())
        {
            return;
        }

        const label_t idx3_in = device::global3(x, y, 0);

        d.phi[idx3_in] = 1.0f;
        d.uz[idx3_in] = physics::u_ref;
    }

#elif defined(DROPLET)

    __global__ void setDroplet(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz ||
            x == 0 || x == mesh::nx - 1 ||
            y == 0 || y == mesh::ny - 1 ||
            z == 0 || z == mesh::nz - 1)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        const scalar_t dx = (static_cast<scalar_t>(x) - geometry::center_x()) / 2.0f;
        const scalar_t dy = static_cast<scalar_t>(y) - geometry::center_y();
        const scalar_t dz = static_cast<scalar_t>(z) - geometry::center_z();
        const scalar_t radialDist = sqrtf(dx * dx + dy * dy + dz * dz);

        const scalar_t phi = 0.5f + 0.5f * tanhf(2.0f * (static_cast<scalar_t>(mesh::radius) - radialDist) / 3.0f);
        d.phi[idx3] = phi;
    }

#endif

    __global__ void setDistros(LBMFields d)
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

        const scalar_t uu = 1.5f * (ux * ux + uy * uy + uz * uz);
        device::constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q)
            {
                constexpr scalar_t w = VelocitySet::w<Q>();
                constexpr scalar_t cx = static_cast<scalar_t>(VelocitySet::cx<Q>());
                constexpr scalar_t cy = static_cast<scalar_t>(VelocitySet::cy<Q>());
                constexpr scalar_t cz = static_cast<scalar_t>(VelocitySet::cz<Q>());

                const scalar_t cu = 3.0f * (cx * ux + cy * uy + cz * uz);

                scalar_t feq = 0.0f;
                if constexpr (VelocitySet::Q() == 19)
                {
                    feq = w * d.rho[idx3] * (1.0f - uu + cu + 0.5f * cu * cu) - w;
                }
                else if constexpr (VelocitySet::Q() == 27)
                {
                    feq = w * d.rho[idx3] * (1.0f - uu + cu + 0.5f * cu * cu + math::oos() * cu * cu * cu) - w;
                }

                d.f[device::global4(x, y, z, Q)] = to_pop(feq);
            });

        device::constexpr_for<0, PhaseVelocitySet::Q()>(
            [&](const auto Q)
            {
                constexpr scalar_t cx = static_cast<scalar_t>(PhaseVelocitySet::cx<Q>());
                constexpr scalar_t cy = static_cast<scalar_t>(PhaseVelocitySet::cy<Q>());
                constexpr scalar_t cz = static_cast<scalar_t>(PhaseVelocitySet::cz<Q>());

                d.g[device::global4(x, y, z, Q)] = PhaseVelocitySet::w<Q>() * d.phi[idx3] * (1.0f + 4.0f * (cx * ux + cy * uy + cz * uz));
            });
    }
}

#endif

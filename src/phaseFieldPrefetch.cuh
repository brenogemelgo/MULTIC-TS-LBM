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
    CUDA kernels for phase field calculations

Namespace
    phase

SourceFiles
    phaseField.cuh

\*---------------------------------------------------------------------------*/

#ifndef PHASEFIELD_CUH
#define PHASEFIELD_CUH

namespace Phase
{
    namespace detail
    {
        __device__ static inline void prefetch_L2(const void *ptr) noexcept
        {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
        }
    }

    __global__ void computePhase(LBMFields d)
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

        scalar_t phi = static_cast<scalar_t>(0);
        device::constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q)
            {
                phi += d.g[Q * size::plane() + idx3];
            });

        d.phi[idx3] = phi;
    }

    __global__ void computeNormals(LBMFields d)
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

        const scalar_t w1 = LBM::VelocitySet::w_1();
        const scalar_t w2 = LBM::VelocitySet::w_2();

        scalar_t sgx;
        scalar_t sgy;
        scalar_t sgz;

        // Corner indices (only meaningful for D3Q27)
        label_t idx_ppp, idx_mmm, idx_ppm, idx_mmp;
        label_t idx_pmp, idx_mpm, idx_pmm, idx_mpp;

        // Prefetch corner nodes EARLY, then spend cycles on w1/w2 faces/edges
        if constexpr (LBM::VelocitySet::Q() == 27)
        {
            idx_ppp = device::global3(x + 1, y + 1, z + 1);
            idx_mmm = device::global3(x - 1, y - 1, z - 1);
            idx_ppm = device::global3(x + 1, y + 1, z - 1);
            idx_mmp = device::global3(x - 1, y - 1, z + 1);
            idx_pmp = device::global3(x + 1, y - 1, z + 1);
            idx_mpm = device::global3(x - 1, y + 1, z - 1);
            idx_pmm = device::global3(x + 1, y - 1, z - 1);
            idx_mpp = device::global3(x - 1, y + 1, z + 1);

            detail::prefetch_L2(&d.phi[idx_ppp]);
            detail::prefetch_L2(&d.phi[idx_mmm]);
            detail::prefetch_L2(&d.phi[idx_ppm]);
            detail::prefetch_L2(&d.phi[idx_mmp]);
            detail::prefetch_L2(&d.phi[idx_pmp]);
            detail::prefetch_L2(&d.phi[idx_mpm]);
            detail::prefetch_L2(&d.phi[idx_pmm]);
            detail::prefetch_L2(&d.phi[idx_mpp]);
        }

        // --- D3Q19-like stencil: faces + edges (w1, w2) ---

        sgx = w1 * (d.phi[device::global3(x + 1, y, z)] -
                    d.phi[device::global3(x - 1, y, z)]) +
              w2 * (d.phi[device::global3(x + 1, y + 1, z)] -
                    d.phi[device::global3(x - 1, y - 1, z)] +
                    d.phi[device::global3(x + 1, y, z + 1)] -
                    d.phi[device::global3(x - 1, y, z - 1)] +
                    d.phi[device::global3(x + 1, y - 1, z)] -
                    d.phi[device::global3(x - 1, y + 1, z)] +
                    d.phi[device::global3(x + 1, y, z - 1)] -
                    d.phi[device::global3(x - 1, y, z + 1)]);

        sgy = w1 * (d.phi[device::global3(x, y + 1, z)] -
                    d.phi[device::global3(x, y - 1, z)]) +
              w2 * (d.phi[device::global3(x + 1, y + 1, z)] -
                    d.phi[device::global3(x - 1, y - 1, z)] +
                    d.phi[device::global3(x, y + 1, z + 1)] -
                    d.phi[device::global3(x, y - 1, z - 1)] +
                    d.phi[device::global3(x - 1, y + 1, z)] -
                    d.phi[device::global3(x + 1, y - 1, z)] +
                    d.phi[device::global3(x, y + 1, z - 1)] -
                    d.phi[device::global3(x, y - 1, z + 1)]);

        sgz = w1 * (d.phi[device::global3(x, y, z + 1)] -
                    d.phi[device::global3(x, y, z - 1)]) +
              w2 * (d.phi[device::global3(x + 1, y, z + 1)] -
                    d.phi[device::global3(x - 1, y, z - 1)] +
                    d.phi[device::global3(x, y + 1, z + 1)] -
                    d.phi[device::global3(x, y - 1, z - 1)] +
                    d.phi[device::global3(x - 1, y, z + 1)] -
                    d.phi[device::global3(x + 1, y, z - 1)] +
                    d.phi[device::global3(x, y - 1, z + 1)] -
                    d.phi[device::global3(x, y + 1, z - 1)]);

        // --- D3Q27 corners (w3), using values we prefetched earlier ---

        if constexpr (LBM::VelocitySet::Q() == 27)
        {
            const scalar_t phi_ppp = d.phi[idx_ppp];
            const scalar_t phi_mmm = d.phi[idx_mmm];
            const scalar_t phi_ppm = d.phi[idx_ppm];
            const scalar_t phi_mmp = d.phi[idx_mmp];
            const scalar_t phi_pmp = d.phi[idx_pmp];
            const scalar_t phi_mpm = d.phi[idx_mpm];
            const scalar_t phi_pmm = d.phi[idx_pmm];
            const scalar_t phi_mpp = d.phi[idx_mpp];

            const scalar_t w3 = LBM::d3q27::w_3();

            // Patterns preserved exactly as in your original code

            sgx += w3 * ((phi_ppp - phi_mmm) +
                         (phi_ppm - phi_mmp) +
                         (phi_pmp - phi_mpm) +
                         (phi_pmm - phi_mpp));

            sgy += w3 * ((phi_ppp - phi_mmm) +
                         (phi_ppm - phi_mmp) +
                         (phi_mpm - phi_pmp) +
                         (phi_mpp - phi_pmm));

            sgz += w3 * ((phi_ppp - phi_mmm) +
                         (phi_mmp - phi_ppm) +
                         (phi_pmp - phi_mpm) +
                         (phi_mpp - phi_pmm));
        }

        const scalar_t gx = LBM::VelocitySet::as2() * sgx;
        const scalar_t gy = LBM::VelocitySet::as2() * sgy;
        const scalar_t gz = LBM::VelocitySet::as2() * sgz;

        const scalar_t ind = sqrtf(gx * gx + gy * gy + gz * gz);
        const scalar_t invInd = static_cast<scalar_t>(1) /
                                (ind + static_cast<scalar_t>(1e-9));

        d.ind[idx3] = ind;
        d.normx[idx3] = gx * invInd;
        d.normy[idx3] = gy * invInd;
        d.normz[idx3] = gz * invInd;
    }

    __global__ void computeForces(LBMFields d)
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

        const scalar_t w1 = LBM::VelocitySet::w_1();
        const scalar_t w2 = LBM::VelocitySet::w_2();

        scalar_t scx;
        scalar_t scy;
        scalar_t scz;

        // Corner indices (only used for D3Q27 contribution)
        label_t idx_ppp, idx_mmm, idx_ppm, idx_mmp;
        label_t idx_pmp, idx_mpm, idx_pmm, idx_mpp;

        if constexpr (LBM::VelocitySet::Q() == 27)
        {
            idx_ppp = device::global3(x + 1, y + 1, z + 1);
            idx_mmm = device::global3(x - 1, y - 1, z - 1);
            idx_ppm = device::global3(x + 1, y + 1, z - 1);
            idx_mmp = device::global3(x - 1, y - 1, z + 1);
            idx_pmp = device::global3(x + 1, y - 1, z + 1);
            idx_mpm = device::global3(x - 1, y + 1, z - 1);
            idx_pmm = device::global3(x + 1, y - 1, z - 1);
            idx_mpp = device::global3(x - 1, y + 1, z + 1);

            // Prefetch normals at corners for x,y,z components
            detail::prefetch_L2(&d.normx[idx_ppp]);
            detail::prefetch_L2(&d.normx[idx_mmm]);
            detail::prefetch_L2(&d.normx[idx_ppm]);
            detail::prefetch_L2(&d.normx[idx_mmp]);
            detail::prefetch_L2(&d.normx[idx_pmp]);
            detail::prefetch_L2(&d.normx[idx_mpm]);
            detail::prefetch_L2(&d.normx[idx_pmm]);
            detail::prefetch_L2(&d.normx[idx_mpp]);

            detail::prefetch_L2(&d.normy[idx_ppp]);
            detail::prefetch_L2(&d.normy[idx_mmm]);
            detail::prefetch_L2(&d.normy[idx_ppm]);
            detail::prefetch_L2(&d.normy[idx_mmp]);
            detail::prefetch_L2(&d.normy[idx_pmp]);
            detail::prefetch_L2(&d.normy[idx_mpm]);
            detail::prefetch_L2(&d.normy[idx_pmm]);
            detail::prefetch_L2(&d.normy[idx_mpp]);

            detail::prefetch_L2(&d.normz[idx_ppp]);
            detail::prefetch_L2(&d.normz[idx_mmm]);
            detail::prefetch_L2(&d.normz[idx_ppm]);
            detail::prefetch_L2(&d.normz[idx_mmp]);
            detail::prefetch_L2(&d.normz[idx_pmp]);
            detail::prefetch_L2(&d.normz[idx_mpm]);
            detail::prefetch_L2(&d.normz[idx_pmm]);
            detail::prefetch_L2(&d.normz[idx_mpp]);
        }

        // --- Faces + edges (w1, w2) ---

        scx = w1 * (d.normx[device::global3(x + 1, y, z)] -
                    d.normx[device::global3(x - 1, y, z)]) +
              w2 * (d.normx[device::global3(x + 1, y + 1, z)] -
                    d.normx[device::global3(x - 1, y - 1, z)] +
                    d.normx[device::global3(x + 1, y, z + 1)] -
                    d.normx[device::global3(x - 1, y, z - 1)] +
                    d.normx[device::global3(x + 1, y - 1, z)] -
                    d.normx[device::global3(x - 1, y + 1, z)] +
                    d.normx[device::global3(x + 1, y, z - 1)] -
                    d.normx[device::global3(x - 1, y, z + 1)]);

        scy = w1 * (d.normy[device::global3(x, y + 1, z)] -
                    d.normy[device::global3(x, y - 1, z)]) +
              w2 * (d.normy[device::global3(x + 1, y + 1, z)] -
                    d.normy[device::global3(x - 1, y - 1, z)] +
                    d.normy[device::global3(x, y + 1, z + 1)] -
                    d.normy[device::global3(x, y - 1, z - 1)] +
                    d.normy[device::global3(x - 1, y + 1, z)] -
                    d.normy[device::global3(x + 1, y - 1, z)] +
                    d.normy[device::global3(x, y + 1, z - 1)] -
                    d.normy[device::global3(x, y - 1, z + 1)]);

        scz = w1 * (d.normz[device::global3(x, y, z + 1)] -
                    d.normz[device::global3(x, y, z - 1)]) +
              w2 * (d.normz[device::global3(x + 1, y, z + 1)] -
                    d.normz[device::global3(x - 1, y, z - 1)] +
                    d.normz[device::global3(x, y + 1, z + 1)] -
                    d.normz[device::global3(x, y - 1, z - 1)] +
                    d.normz[device::global3(x - 1, y, z + 1)] -
                    d.normz[device::global3(x + 1, y, z - 1)] +
                    d.normz[device::global3(x, y - 1, z + 1)] -
                    d.normz[device::global3(x, y + 1, z - 1)]);

        // --- Corner (w3) part for D3Q27, now using re-used, prefetched normals ---

        if constexpr (LBM::VelocitySet::Q() == 27)
        {
            const scalar_t nx_ppp = d.normx[idx_ppp];
            const scalar_t nx_mmm = d.normx[idx_mmm];
            const scalar_t nx_ppm = d.normx[idx_ppm];
            const scalar_t nx_mmp = d.normx[idx_mmp];
            const scalar_t nx_pmp = d.normx[idx_pmp];
            const scalar_t nx_mpm = d.normx[idx_mpm];
            const scalar_t nx_pmm = d.normx[idx_pmm];
            const scalar_t nx_mpp = d.normx[idx_mpp];

            const scalar_t ny_ppp = d.normy[idx_ppp];
            const scalar_t ny_mmm = d.normy[idx_mmm];
            const scalar_t ny_ppm = d.normy[idx_ppm];
            const scalar_t ny_mmp = d.normy[idx_mmp];
            const scalar_t ny_pmp = d.normy[idx_pmp];
            const scalar_t ny_mpm = d.normy[idx_mpm];
            const scalar_t ny_pmm = d.normy[idx_pmm];
            const scalar_t ny_mpp = d.normy[idx_mpp];

            const scalar_t nz_ppp = d.normz[idx_ppp];
            const scalar_t nz_mmm = d.normz[idx_mmm];
            const scalar_t nz_ppm = d.normz[idx_ppm];
            const scalar_t nz_mmp = d.normz[idx_mmp];
            const scalar_t nz_pmp = d.normz[idx_pmp];
            const scalar_t nz_mpm = d.normz[idx_mpm];
            const scalar_t nz_pmm = d.normz[idx_pmm];
            const scalar_t nz_mpp = d.normz[idx_mpp];

            const scalar_t w3 = LBM::d3q27::w_3();

            // Same patterns as your original code

            scx += w3 * ((nx_ppp - nx_mmm) +
                         (nx_ppm - nx_mmp) +
                         (nx_pmp - nx_mpm) +
                         (nx_pmm - nx_mpp));

            scy += w3 * ((ny_ppp - ny_mmm) +
                         (ny_ppm - ny_mmp) +
                         (ny_mpm - ny_pmp) +
                         (ny_mpp - ny_pmm));

            scz += w3 * ((nz_ppp - nz_mmm) +
                         (nz_mmp - nz_ppm) +
                         (nz_pmp - nz_mpm) +
                         (nz_mpp - nz_pmm));
        }

        const scalar_t curvature = LBM::VelocitySet::as2() * (scx + scy + scz);

        const scalar_t stCurv = -physics::sigma * curvature * d.ind[idx3];
        d.ffx[idx3] = stCurv * d.normx[idx3];
        d.ffy[idx3] = stCurv * d.normy[idx3];
        d.ffz[idx3] = stCurv * d.normz[idx3];
    }
}

#endif
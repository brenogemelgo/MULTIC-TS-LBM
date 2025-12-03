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
    A class applying boundary conditions

Namespace
    LBM

SourceFiles
    boundaryConditions.cuh

\*---------------------------------------------------------------------------*/

#ifndef BOUNDARYCONDITIONS_CUH
#define BOUNDARYCONDITIONS_CUH

namespace LBM
{
    class BoundaryConditions
    {
    public:
        __host__ __device__ [[nodiscard]] inline consteval BoundaryConditions(){};

#if defined(JET)

        __device__ static inline constexpr void applyInflow(
            LBMFields d,
            const label_t t) noexcept
        {
            const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= mesh::nx || y >= mesh::ny)
            {
                return;
            }

            // Calculate distance from the jet center
            const scalar_t dx = static_cast<scalar_t>(x) - geometry::center_x();
            const scalar_t dy = static_cast<scalar_t>(y) - geometry::center_y();
            const scalar_t r2 = dx * dx + dy * dy;

            // Check if inside the jet radius
            if (r2 > geometry::R2())
            {
                return;
            }

            // Indices at the boundary and one node into the domain
            const label_t idx3_bnd = device::global3(x, y, 0);
            const label_t idx3_zp1 = device::global3(x, y, 1);

            // Inlet velocity with white noise (Box-Muller)
            const scalar_t z = gaussian_noise<10>(x, y, t);
            const scalar_t uz = physics::u_ref + 0.004f * z;

            // Macroscopic variables at the boundary
            d.rho[idx3_bnd] = 1.0f;
            d.phi[idx3_bnd] = 1.0f;
            d.ux[idx3_bnd] = 0.0f;
            d.uy[idx3_bnd] = 0.0f;
            d.uz[idx3_bnd] = uz;

            // Equilibrium polynomial (simplified for ux = uy = 0)
            const scalar_t P = 1.0f + 3.0f * uz + 3.0f * uz * uz;

            // Set incoming hydrodynamic populations
            device::constexpr_for<0, VelocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (VelocitySet::cz<Q>() == 1)
                    {
                        const label_t xx = x + static_cast<label_t>(VelocitySet::cx<Q>());
                        const label_t yy = y + static_cast<label_t>(VelocitySet::cy<Q>());

                        const label_t fluidNode = device::global3(xx, yy, 1);

                        constexpr scalar_t w = VelocitySet::w<Q>();

                        // Assuming rho = 1
                        const scalar_t feq = w * P - w;

                        const scalar_t fneq = computeFneq<Q>(d, fluidNode);

                        d.f[Q * size::plane() + fluidNode] = to_pop(feq + relaxation::omco_zmin() * fneq);
                    }
                });

            // Set incoming phase field populations (assuming phi = 1)
            d.g[5 * size::plane() + idx3_zp1] = PhaseVelocitySet::w<5>() * (1.0f + 4.0f * uz);
        }

        __device__ static inline constexpr void applyOutflow(LBMFields d) noexcept
        {
            const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= mesh::nx || y >= mesh::ny)
            {
                return;
            }

            // Indices at the boundary and one node into the domain
            const label_t idx3_bnd = device::global3(x, y, mesh::nz - 1);
            const label_t idx3_zm1 = device::global3(x, y, mesh::nz - 2);

            d.rho[idx3_bnd] = d.rho[idx3_zm1];
            d.phi[idx3_bnd] = d.phi[idx3_zm1];
            d.ux[idx3_bnd] = d.ux[idx3_zm1];
            d.uy[idx3_bnd] = d.uy[idx3_zm1];
            d.uz[idx3_bnd] = d.uz[idx3_zm1];

            // Registers for macroscopic velocities
            const scalar_t rho = d.rho[idx3_bnd];
            const scalar_t phi = d.phi[idx3_bnd];
            const scalar_t ux = d.ux[idx3_bnd];
            const scalar_t uy = d.uy[idx3_bnd];
            const scalar_t uz = d.uz[idx3_bnd];

            const scalar_t uu = 1.5f * (ux * ux + uy * uy + uz * uz);

            // Set incoming hydrodynamic populations
            device::constexpr_for<0, VelocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (VelocitySet::cz<Q>() == -1)
                    {
                        const label_t xx = x + static_cast<label_t>(VelocitySet::cx<Q>());
                        const label_t yy = y + static_cast<label_t>(VelocitySet::cy<Q>());

                        const label_t fluidNode = device::global3(xx, yy, mesh::nz - 2);

                        constexpr scalar_t w = VelocitySet::w<Q>();
                        constexpr scalar_t cx = static_cast<scalar_t>(VelocitySet::cx<Q>());
                        constexpr scalar_t cy = static_cast<scalar_t>(VelocitySet::cy<Q>());
                        constexpr scalar_t cz = static_cast<scalar_t>(VelocitySet::cz<Q>());

                        const scalar_t cu = 3.0f * (cx * ux + cy * uy + cz * uz);

                        scalar_t feq = 0.0f;
                        if constexpr (VelocitySet::Q() == 19)
                        {
                            feq = w * rho * (1.0f - uu + cu + 0.5f * cu * cu) - w;
                        }
                        else if constexpr (VelocitySet::Q() == 27)
                        {
                            feq = w * rho * (1.0f - uu + cu + 0.5f * cu * cu + math::oos() * cu * cu * cu - uu * cu) - w;
                        }

                        const scalar_t fneq = computeFneq<Q>(d, fluidNode);

                        d.f[Q * size::plane() + fluidNode] = to_pop(feq + relaxation::omco_zmax() * fneq);
                    }
                });

            d.g[6 * size::plane() + idx3_zm1] = PhaseVelocitySet::w<6>() * phi * (1.0f - 4.0f * physics::u_ref);
        }

        __device__ static inline constexpr void periodicX(LBMFields d) noexcept
        {
            const label_t y = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t z = threadIdx.y + blockIdx.y * blockDim.y;

            if (y <= 0 || y >= mesh::ny - 1 || z <= 0 || z >= mesh::nz - 1)
            {
                return;
            }

            const label_t bL = device::global3(1, y, z);
            const label_t bR = device::global3(mesh::nx - 2, y, z);

            device::constexpr_for<0, VelocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (VelocitySet::cx<Q>() > 0)
                    {
                        d.f[Q * size::plane() + bL] = d.f[Q * size::plane() + bR];
                    }
                    if constexpr (VelocitySet::cx<Q>() < 0)
                    {
                        d.f[Q * size::plane() + bR] = d.f[Q * size::plane() + bL];
                    }
                });

            device::constexpr_for<0, PhaseVelocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (PhaseVelocitySet::cx<Q>() > 0)
                    {
                        d.g[Q * size::plane() + bL] = d.g[Q * size::plane() + bR];
                    }
                    if constexpr (PhaseVelocitySet::cx<Q>() < 0)
                    {
                        d.g[Q * size::plane() + bR] = d.g[Q * size::plane() + bL];
                    }
                });

            const label_t gL = device::global3(0, y, z);
            const label_t gR = device::global3(mesh::nx - 1, y, z);

            d.rho[gL] = d.rho[bR];
            d.rho[gR] = d.rho[bL];

            d.phi[gL] = d.phi[bR];
            d.phi[gR] = d.phi[bL];

            d.ux[gL] = d.ux[bR];
            d.ux[gR] = d.ux[bL];

            d.uy[gL] = d.uy[bR];
            d.uy[gR] = d.uy[bL];

            d.uz[gL] = d.uz[bR];
            d.uz[gR] = d.uz[bL];
        }

        __device__ static inline constexpr void periodicY(LBMFields d) noexcept
        {
            const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t z = threadIdx.y + blockIdx.y * blockDim.y;

            if (x <= 0 || x >= mesh::nx - 1 || z <= 0 || z >= mesh::nz - 1)
            {
                return;
            }

            const label_t bB = device::global3(x, 1, z);
            const label_t bT = device::global3(x, mesh::ny - 2, z);

            device::constexpr_for<0, VelocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (VelocitySet::cy<Q>() > 0)
                    {
                        d.f[Q * size::plane() + bB] = d.f[Q * size::plane() + bT];
                    }
                    if constexpr (VelocitySet::cy<Q>() < 0)
                    {
                        d.f[Q * size::plane() + bT] = d.f[Q * size::plane() + bB];
                    }
                });

            device::constexpr_for<0, PhaseVelocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (PhaseVelocitySet::cy<Q>() > 0)
                    {
                        d.g[Q * size::plane() + bB] = d.g[Q * size::plane() + bT];
                    }
                    if constexpr (PhaseVelocitySet::cy<Q>() < 0)
                    {
                        d.g[Q * size::plane() + bT] = d.g[Q * size::plane() + bB];
                    }
                });

            const label_t gB = device::global3(x, 0, z);
            const label_t gT = device::global3(x, mesh::ny - 1, z);

            d.rho[gB] = d.rho[bT];
            d.rho[gT] = d.rho[bB];

            d.phi[gB] = d.phi[bT];
            d.phi[gT] = d.phi[bB];

            d.ux[gB] = d.ux[bT];
            d.ux[gT] = d.ux[bB];

            d.uy[gB] = d.uy[bT];
            d.uy[gT] = d.uy[bB];

            d.uz[gB] = d.uz[bT];
            d.uz[gT] = d.uz[bB];
        }

#elif defined(DROPLET)

#endif

    private:
        template <label_t Q>
        __device__ static inline scalar_t computeFneq(
            const LBMFields &d,
            const label_t fluidNode) noexcept
        {
            constexpr scalar_t w = VelocitySet::w<Q>();
            constexpr scalar_t cx = static_cast<scalar_t>(VelocitySet::cx<Q>());
            constexpr scalar_t cy = static_cast<scalar_t>(VelocitySet::cy<Q>());
            constexpr scalar_t cz = static_cast<scalar_t>(VelocitySet::cz<Q>());

            if constexpr (VelocitySet::Q() == 19)
            {
                return (w * 4.5f) *
                       ((cx * cx - cs2()) * d.pxx[fluidNode] +
                        (cy * cy - cs2()) * d.pyy[fluidNode] +
                        (cz * cz - cs2()) * d.pzz[fluidNode] +
                        2.0f * (cx * cy * d.pxy[fluidNode] +
                                cx * cz * d.pxz[fluidNode] +
                                cy * cz * d.pyz[fluidNode]));
            }
            else if constexpr (VelocitySet::Q() == 27)
            {
                return (w * 4.5f) *
                       ((cx * cx - cs2()) * d.pxx[fluidNode] +
                        (cy * cy - cs2()) * d.pyy[fluidNode] +
                        (cz * cz - cs2()) * d.pzz[fluidNode] +
                        2.0f * (cx * cy * d.pxy[fluidNode] +
                                cx * cz * d.pxz[fluidNode] +
                                cy * cz * d.pyz[fluidNode]) +
                        (cx * cx * cx - cx) * (3.0f * d.ux[fluidNode] * d.pxx[fluidNode]) +
                        (cy * cy * cy - cy) * (3.0f * d.uy[fluidNode] * d.pyy[fluidNode]) +
                        (cz * cz * cz - cz) * (3.0f * d.uz[fluidNode] * d.pzz[fluidNode]) +
                        3.0f * ((cx * cx * cy - cs2() * cy) * (d.pxx[fluidNode] * d.uy[fluidNode] + 2.0f * d.ux[fluidNode] * d.pxy[fluidNode]) +
                                (cx * cx * cz - cs2() * cz) * (d.pxx[fluidNode] * d.uz[fluidNode] + 2.0f * d.ux[fluidNode] * d.pxz[fluidNode]) +
                                (cx * cy * cy - cs2() * cx) * (d.pxy[fluidNode] * d.uy[fluidNode] + 2.0f * d.ux[fluidNode] * d.pyy[fluidNode]) +
                                (cy * cy * cz - cs2() * cz) * (d.pyy[fluidNode] * d.uz[fluidNode] + 2.0f * d.uy[fluidNode] * d.pyz[fluidNode]) +
                                (cx * cz * cz - cs2() * cx) * (d.pxz[fluidNode] * d.uz[fluidNode] + 2.0f * d.ux[fluidNode] * d.pzz[fluidNode]) +
                                (cy * cz * cz - cs2() * cy) * (d.pyz[fluidNode] * d.uz[fluidNode] + 2.0f * d.uy[fluidNode] * d.pzz[fluidNode])) +
                        6.0f * (cx * cy * cz) * (d.ux[fluidNode] * d.pyz[fluidNode] + d.uy[fluidNode] * d.pxz[fluidNode] + d.uz[fluidNode] * d.pxy[fluidNode]));
            }
        }

        __device__ [[nodiscard]] static inline uint32_t hash32(label_t x) noexcept
        {
            x ^= x >> 16;
            x *= 0x7FEB352Du;
            x ^= x >> 15;
            x *= 0x846CA68Bu;
            x ^= x >> 16;
            return x;
        }

        __device__ [[nodiscard]] static inline scalar_t uniform01(const uint32_t seed) noexcept
        {
            const scalar_t inv2_32 = static_cast<scalar_t>(2.3283064365386963e-10f);
            return (static_cast<scalar_t>(seed) + static_cast<scalar_t>(0.5f)) * inv2_32;
        }

        __device__ static inline void box_muller(
            scalar_t rrx,
            scalar_t rry,
            scalar_t &z1,
            scalar_t &z2) noexcept
        {
            rrx = fmaxf(rrx, 1e-12f);
            const scalar_t r = sqrtf(-2.0f * logf(rrx));
            const scalar_t theta = math::two_pi() * rry;
            scalar_t s, c;
            sincosf(theta, &s, &c);
            z1 = r * c;
            z2 = r * s;
        }

        template <label_t NOISE_PERIOD = 10>
        __device__ [[nodiscard]] static inline scalar_t gaussian_noise(
            const label_t x,
            const label_t y,
            const label_t STEP) noexcept
        {
            const label_t call_idx = STEP / NOISE_PERIOD;
            const label_t pair_idx = call_idx >> 1;
            const bool use_second = (call_idx & 1) != 0;
            const label_t base = 0x9E3779B9u ^ x ^ (y * 0x85EBCA6Bu) ^ (pair_idx * 0xC2B2AE35u);
            const scalar_t rrx = uniform01(hash32(base));
            const scalar_t rry = uniform01(hash32(base ^ 0x68BC21EBu));
            scalar_t z1, z2;
            box_muller(rrx, rry, z1, z2);
            return use_second ? z2 : z1;
        }
    };
}

#endif

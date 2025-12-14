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

            const scalar_t dx = static_cast<scalar_t>(x) - geometry::center_x();
            const scalar_t dy = static_cast<scalar_t>(y) - geometry::center_y();
            const scalar_t r2 = dx * dx + dy * dy;

            if (r2 > geometry::R2())
            {
                return;
            }

            const label_t idx3_bnd = device::global3(x, y, 0);
            const label_t idx3_zp1 = device::global3(x, y, 1);

            const scalar_t sigma_u = static_cast<scalar_t>(0.08) * physics::u_ref;

            const scalar_t zx = white_noise<10, 0xA341316Cu>(x, y, t);
            const scalar_t zy = white_noise<10, 0xC8013EA4u>(x, y, t);

            const scalar_t rho = static_cast<scalar_t>(1);
            const scalar_t phi = static_cast<scalar_t>(1);
            const scalar_t ux = sigma_u * zx;
            const scalar_t uy = sigma_u * zy;
            const scalar_t uz = physics::u_ref;

            d.rho[idx3_bnd] = rho;
            d.phi[idx3_bnd] = phi;
            d.ux[idx3_bnd] = ux;
            d.uy[idx3_bnd] = uy;
            d.uz[idx3_bnd] = uz;

            const scalar_t uu = static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz);

            device::constexpr_for<0, VelocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (VelocitySet::cz<Q>() == 1)
                    {
                        const label_t xx = x + static_cast<label_t>(VelocitySet::cx<Q>());
                        const label_t yy = y + static_cast<label_t>(VelocitySet::cy<Q>());

                        const label_t fluidNode = device::global3(xx, yy, 1);

                        constexpr scalar_t w = VelocitySet::w<Q>();
                        constexpr scalar_t cx = static_cast<scalar_t>(VelocitySet::cx<Q>());
                        constexpr scalar_t cy = static_cast<scalar_t>(VelocitySet::cy<Q>());
                        constexpr scalar_t cz = static_cast<scalar_t>(VelocitySet::cz<Q>());

                        const scalar_t cu = VelocitySet::as2() * (cx * ux + cy * uy + cz * uz);

                        const scalar_t feq = VelocitySet::f_eq<Q>(rho, uu, cu);
                        const scalar_t fneq = VelocitySet::f_neq<Q>(d.pxx[fluidNode], d.pyy[fluidNode], d.pzz[fluidNode],
                                                                    d.pxy[fluidNode], d.pxz[fluidNode], d.pyz[fluidNode],
                                                                    d.ux[fluidNode], d.uy[fluidNode], d.uz[fluidNode]);

                        d.f[Q * size::cells() + fluidNode] = to_pop(feq + relaxation::omco_zmin() * fneq);
                    }
                });

            d.g[5 * size::cells() + idx3_zp1] = Phase::VelocitySet::w<5>() * phi * (static_cast<scalar_t>(1) + Phase::VelocitySet::as2() * uz);
        }

        __device__ static inline constexpr void applyOutflow(LBMFields d) noexcept
        {
            const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= mesh::nx || y >= mesh::ny)
            {
                return;
            }

            const label_t idx3_bnd = device::global3(x, y, mesh::nz - 1);
            const label_t idx3_zm1 = device::global3(x, y, mesh::nz - 2);

            d.rho[idx3_bnd] = d.rho[idx3_zm1];
            d.phi[idx3_bnd] = d.phi[idx3_zm1];
            d.ux[idx3_bnd] = d.ux[idx3_zm1];
            d.uy[idx3_bnd] = d.uy[idx3_zm1];
            d.uz[idx3_bnd] = d.uz[idx3_zm1];

            const scalar_t rho = d.rho[idx3_bnd];
            const scalar_t phi = d.phi[idx3_bnd];
            const scalar_t ux = d.ux[idx3_bnd];
            const scalar_t uy = d.uy[idx3_bnd];
            const scalar_t uz = d.uz[idx3_bnd];

            const scalar_t uu = static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz);

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

                        const scalar_t cu = VelocitySet::as2() * (cx * ux + cy * uy + cz * uz);

                        const scalar_t feq = VelocitySet::f_eq<Q>(rho, uu, cu);
                        const scalar_t fneq = VelocitySet::f_neq<Q>(d.pxx[fluidNode], d.pyy[fluidNode], d.pzz[fluidNode],
                                                                    d.pxy[fluidNode], d.pxz[fluidNode], d.pyz[fluidNode],
                                                                    d.ux[fluidNode], d.uy[fluidNode], d.uz[fluidNode]);

                        d.f[Q * size::cells() + fluidNode] = to_pop(feq + relaxation::omco_zmax() * fneq);
                    }
                });

            d.g[6 * size::cells() + idx3_zm1] = Phase::VelocitySet::w<6>() * phi * (static_cast<scalar_t>(1) - Phase::VelocitySet::as2() * physics::u_ref);
        }

    private:
        __device__ [[nodiscard]] static inline constexpr uint32_t hash32(uint32_t x) noexcept
        {
            x ^= x >> 16;
            x *= 0x7FEB352Du;
            x ^= x >> 15;
            x *= 0x846CA68Bu;
            x ^= x >> 16;

            return x;
        }

        __device__ [[nodiscard]] static inline constexpr scalar_t uniform01(const uint32_t seed) noexcept
        {
            const scalar_t inv2_32 = static_cast<scalar_t>(2.3283064365386963e-10);

            return (static_cast<scalar_t>(seed) + static_cast<scalar_t>(0.5)) * inv2_32;
        }

        __device__ [[nodiscard]] static inline scalar_t box_muller(
            scalar_t rrx,
            const scalar_t rry) noexcept
        {
            rrx = math::max(rrx, static_cast<scalar_t>(1e-12));
            const scalar_t r = math::sqrt(-static_cast<scalar_t>(2) * math::log(rrx));
            const scalar_t theta = math::two_pi() * rry;

            return r * math::cos(theta);
        }

        template <label_t NOISE_PERIOD = 10, uint32_t SALT = 0u>
        __device__ [[nodiscard]] static inline constexpr scalar_t white_noise(
            const label_t x,
            const label_t y,
            const label_t STEP) noexcept
        {
            const label_t t = STEP / NOISE_PERIOD;
            const uint32_t base = (0x9E3779B9u ^ SALT) ^ static_cast<uint32_t>(x) ^ (static_cast<uint32_t>(y) * 0x85EBCA6Bu) ^ (static_cast<uint32_t>(t) * 0xC2B2AE35u);

            const scalar_t rrx = uniform01(hash32(base));
            const scalar_t rry = uniform01(hash32(base ^ 0x68BC21EBu));

            return box_muller(rrx, rry);
        }
    };
}

#endif

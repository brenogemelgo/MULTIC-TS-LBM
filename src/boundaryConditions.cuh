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

            constexpr scalar_t sigma_u = static_cast<scalar_t>(0.08) * physics::u_inf;

            const scalar_t zx = white_noise<0xA341316Cu>(x, y, t);
            const scalar_t zy = white_noise<0xC8013EA4u>(x, y, t);

            const scalar_t p = static_cast<scalar_t>(0);
            const scalar_t phi = static_cast<scalar_t>(1);
            const scalar_t ux = sigma_u * zx;
            const scalar_t uy = sigma_u * zy;
            const scalar_t uz = physics::u_inf;

            d.p[idx3_bnd] = p;
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

                        const scalar_t feq = VelocitySet::f_eq<Q>(p, uu, cu);
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

            d.p[idx3_bnd] = d.p[idx3_zm1];
            d.phi[idx3_bnd] = d.phi[idx3_zm1];
            d.ux[idx3_bnd] = d.ux[idx3_zm1];
            d.uy[idx3_bnd] = d.uy[idx3_zm1];
            d.uz[idx3_bnd] = d.uz[idx3_zm1];

            const scalar_t p = d.p[idx3_bnd];
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

                        const scalar_t feq = VelocitySet::f_eq<Q>(p, uu, cu);
                        const scalar_t fneq = VelocitySet::f_neq<Q>(d.pxx[fluidNode], d.pyy[fluidNode], d.pzz[fluidNode],
                                                                    d.pxy[fluidNode], d.pxz[fluidNode], d.pyz[fluidNode],
                                                                    d.ux[fluidNode], d.uy[fluidNode], d.uz[fluidNode]);

                        d.f[Q * size::cells() + fluidNode] = to_pop(feq + relaxation::omco_zmax(phi) * fneq);
                    }
                });

            d.g[6 * size::cells() + idx3_zm1] = Phase::VelocitySet::w<6>() * phi * (static_cast<scalar_t>(1) - Phase::VelocitySet::as2() * physics::u_inf);
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
            constexpr scalar_t inv2_32 = static_cast<scalar_t>(2.3283064365386963e-10);

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

        template <uint32_t SALT = 0u>
        __device__ [[nodiscard]] static inline constexpr scalar_t white_noise(
            const label_t x,
            const label_t y,
            const label_t STEP) noexcept
        {
            // const label_t t = STEP / static_cast<label_t>(10);
            const label_t t = STEP; // white-in-time. uncomment above to call each denominator steps
            const uint32_t base = (0x9E3779B9u ^ SALT) ^ static_cast<uint32_t>(x) ^ (static_cast<uint32_t>(y) * 0x85EBCA6Bu) ^ (static_cast<uint32_t>(t) * 0xC2B2AE35u);

            const scalar_t rrx = uniform01(hash32(base));
            const scalar_t rry = uniform01(hash32(base ^ 0x68BC21EBu));

            return box_muller(rrx, rry);
        }

        __device__ static inline int clamp_int(
            const int v,
            const int lo,
            const int hi) noexcept
        {
            return (v < lo) ? lo : ((v > hi) ? hi : v);
        }

        __device__ static inline scalar_t window_r2(const scalar_t r2) noexcept
        {
            const scalar_t invR2 = static_cast<scalar_t>(1) / geometry::R2();
            const scalar_t s2 = r2 * invR2;

            const scalar_t one_minus = (s2 < static_cast<scalar_t>(1)) ? (static_cast<scalar_t>(1) - s2) : static_cast<scalar_t>(0);

            return one_minus * one_minus;
        }

        template <uint32_t Seed>
        __device__ static inline scalar_t ou_noise_2d(
            const int x,
            const int y,
            const label_t t) noexcept
        {
            constexpr scalar_t alpha = static_cast<scalar_t>(0.80);
            constexpr scalar_t beta = static_cast<scalar_t>(0.60);

            const label_t tm1 = (t > 0) ? (t - 1) : t;

            const scalar_t n0 = white_noise<Seed>(static_cast<label_t>(x), static_cast<label_t>(y), t);
            const scalar_t n1 = white_noise<Seed ^ 0x9E3779B9u>(static_cast<label_t>(x), static_cast<label_t>(y), tm1);

            return alpha * n1 + beta * n0;
        }

        __device__ static inline void coeffs_9tap(
            scalar_t &a0,
            scalar_t &a1, scalar_t &a2,
            scalar_t &a3,
            scalar_t &a4) noexcept
        {
            a0 = static_cast<scalar_t>(0.1531393568);
            a1 = static_cast<scalar_t>(0.1449280831);
            a2 = static_cast<scalar_t>(0.1226769711);
            a3 = static_cast<scalar_t>(0.0929521065);
            a4 = static_cast<scalar_t>(0.0628731615);
        }

        __device__ static inline scalar_t a_of_abs_9(
            const int kabs,
            const scalar_t a0,
            const scalar_t a1,
            const scalar_t a2,
            const scalar_t a3,
            const scalar_t a4) noexcept
        {
            return (kabs == 0) ? a0 : (kabs == 1) ? a1
                                  : (kabs == 2)   ? a2
                                  : (kabs == 3)   ? a3
                                                  : a4;
        }

        __device__ static inline scalar_t b_of_j_9(
            const int j,
            const scalar_t a0,
            const scalar_t a1,
            const scalar_t a2,
            const scalar_t a3,
            const scalar_t a4) noexcept
        {
            if (j == 0)
                return static_cast<scalar_t>(0);

            const int aj = (j < 0) ? -j : j;

            scalar_t bj = static_cast<scalar_t>(0);

            if (aj == 1)
                bj = static_cast<scalar_t>(0.5) * (a0 - a2);
            else if (aj == 2)
                bj = static_cast<scalar_t>(0.5) * (a1 - a3);
            else if (aj == 3)
                bj = static_cast<scalar_t>(0.5) * (a2 - a4);
            else /* aj == 4 */
                bj = static_cast<scalar_t>(0.5) * (a3 - static_cast<scalar_t>(0));

            return (j > 0) ? bj : -bj;
        }

        template <uint32_t Seed>
        __device__ static inline void synthetic_uv_hat_9x9(const int x,
                                                           const int y,
                                                           const label_t t,
                                                           scalar_t &ux_hat,
                                                           scalar_t &uy_hat) noexcept
        {
            scalar_t a0, a1, a2, a3, a4;
            coeffs_9tap(a0, a1, a2, a3, a4);

            ux_hat = static_cast<scalar_t>(0);
            uy_hat = static_cast<scalar_t>(0);

            const int x0 = 0;
            const int y0 = 0;
            const int x1 = static_cast<int>(mesh::nx) - 1;
            const int y1 = static_cast<int>(mesh::ny) - 1;

#pragma unroll
            for (int i = -4; i <= 4; ++i)
            {
                const int xi = clamp_int(x + i, x0, x1);
                const int iabs = (i < 0) ? -i : i;

                const scalar_t ai = a_of_abs_9(iabs, a0, a1, a2, a3, a4);
                const scalar_t bi = b_of_j_9(i, a0, a1, a2, a3, a4);

#pragma unroll
                for (int j = -4; j <= 4; ++j)
                {
                    const int yj = clamp_int(y + j, y0, y1);
                    const int jabs = (j < 0) ? -j : j;

                    const scalar_t aj = a_of_abs_9(jabs, a0, a1, a2, a3, a4);
                    const scalar_t bj = b_of_j_9(j, a0, a1, a2, a3, a4);

                    const scalar_t xi_noise = ou_noise_2d<Seed>(xi, yj, t);

                    // ux_hat = dpsi/dy
                    ux_hat += ai * bj * xi_noise;

                    // uy_hat = -dpsi/dx
                    uy_hat -= bi * aj * xi_noise;
                }
            }
        }
    };
}

#endif

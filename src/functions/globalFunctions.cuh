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
    Global functions used across the entire codebase

Namespace
    block
    physics
    geometry
    relaxation
    LBM
    math
    size
    sponge (only if JET is defined)

SourceFiles
    globalFunctions.cuh

\*---------------------------------------------------------------------------*/

#ifndef GLOBALFUNCTIONS_CUH
#define GLOBALFUNCTIONS_CUH

#include "constants.cuh"

namespace block
{
    __host__ __device__ [[nodiscard]] static inline consteval unsigned num_block_x() noexcept
    {
        return (mesh::nx + block::nx - 1) / block::nx;
    }

    __host__ __device__ [[nodiscard]] static inline consteval unsigned num_block_y() noexcept
    {
        return (mesh::ny + block::ny - 1) / block::ny;
    }

    __host__ __device__ [[nodiscard]] static inline consteval unsigned size() noexcept
    {
        return block::nx * block::ny * block::nz;
    }
}

namespace size
{
    __host__ __device__ [[nodiscard]] static inline consteval label_t stride() noexcept
    {
        return mesh::nx * mesh::ny;
    }

    __host__ __device__ [[nodiscard]] static inline consteval label_t cells() noexcept
    {
        return mesh::nx * mesh::ny * mesh::nz;
    }
}

namespace geometry
{
    __host__ __device__ [[nodiscard]] static inline consteval scalar_t R2() noexcept
    {
        return static_cast<scalar_t>(mesh::radius * mesh::radius);
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t center_x() noexcept
    {
        return static_cast<scalar_t>(mesh::nx - 1) * static_cast<scalar_t>(0.5);
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t center_y() noexcept
    {
        return static_cast<scalar_t>(mesh::ny - 1) * static_cast<scalar_t>(0.5);
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t center_z() noexcept
    {
        return static_cast<scalar_t>(mesh::nz - 1) * static_cast<scalar_t>(0.5);
    }
}

namespace physics
{
    __host__ __device__ [[nodiscard]] static inline consteval scalar_t rho_water() noexcept
    {
        return static_cast<scalar_t>(1);
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t rho_oil() noexcept
    {
        return static_cast<scalar_t>(0.8);
    }
}

namespace math
{
    __host__ __device__ [[nodiscard]] static inline consteval scalar_t two_pi() noexcept
    {
        return static_cast<scalar_t>(2) * static_cast<scalar_t>(CUDART_PI_F);
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t sqrt(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::sqrtf(x);
        }
        else
        {
            return ::sqrt(x);
        }
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t log(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::logf(x);
        }
        else
        {
            return ::log(x);
        }
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t min(
        const scalar_t a,
        const scalar_t b) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fminf(a, b);
        }
        else
        {
            return ::fmin(a, b);
        }
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t max(
        const scalar_t a,
        const scalar_t b) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fmaxf(a, b);
        }
        else
        {
            return ::fmax(a, b);
        }
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t cos(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::cosf(x);
        }
        else
        {
            return ::cos(x);
        }
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t tanh(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::tanhf(x);
        }
        else
        {
            return ::tanh(x);
        }
    }

    __host__ __device__ static inline void sincos(
        const scalar_t x,
        scalar_t *s,
        scalar_t *c) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            ::sincosf(x, s, c);
        }
        else
        {
            ::sincos(x, s, c);
        }
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t fma(
        const scalar_t a,
        const scalar_t b,
        const scalar_t c) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fmaf(a, b, c);
        }
        else
        {
            return ::fma(a, b, c);
        }
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t abs(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fabsf(x);
        }
        else
        {
            return ::fabs(x);
        }
    }
}

namespace sponge
{
    __host__ __device__ [[nodiscard]] static inline consteval scalar_t K_gain() noexcept
    {
        return static_cast<scalar_t>(100);
    }

    __host__ __device__ [[nodiscard]] static inline consteval int sponge_cells() noexcept
    {
        return static_cast<int>(mesh::nz / 12);
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t sponge() noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(sponge_cells()) / static_cast<double>(mesh::nz - 1));
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t z_start() noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(mesh::nz - 1 - sponge_cells()) / static_cast<double>(mesh::nz - 1));
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t inv_nz_m1() noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(mesh::nz - 1));
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t inv_sponge() noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(sponge()));
    }
}

namespace relaxation
{

#if defined(JET)

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t visc_ref() noexcept
    {
        return static_cast<scalar_t>((static_cast<double>(physics::u_ref) * static_cast<double>(mesh::diam)) / static_cast<double>(physics::reynolds));
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omega_from_nu(const scalar_t nu) noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(1) / (static_cast<double>(0.5) + static_cast<double>(LBM::VelocitySet::as2()) * static_cast<double>(nu)));
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omega_ref() noexcept
    {
        return omega_from_nu(visc_ref());
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omega_zmin() noexcept
    {
        return omega_ref();
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omega_zmax() noexcept
    {
        return omega_from_nu(visc_ref() * (static_cast<scalar_t>(1) + sponge::K_gain()));
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omega_delta() noexcept
    {
        return omega_zmax() - omega_zmin();
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omco_ref() noexcept
    {
        return static_cast<scalar_t>(1) - omega_ref();
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omco_zmin() noexcept
    {
        return static_cast<scalar_t>(1) - omega_zmin();
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omco_zmax() noexcept
    {
        return static_cast<scalar_t>(1) - omega_zmax();
    }

#elif defined(DROPLET)

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omega_from_nu(const scalar_t nu) noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(1) / (static_cast<double>(0.5) + static_cast<double>(LBM::VelocitySet::as2()) * static_cast<double>(nu)));
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omega_ref() noexcept
    {
        return omega_from_nu(physics::visc_ref);
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omega_zmin() noexcept
    {
        return omega_ref();
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omega_zmax() noexcept
    {
        return omega_ref();
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omega_delta() noexcept
    {
        return omega_zmax() - omega_zmin();
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omco_ref() noexcept
    {
        return static_cast<scalar_t>(1) - omega_ref();
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omco_zmin() noexcept
    {
        return static_cast<scalar_t>(1) - omega_zmin();
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omco_zmax() noexcept
    {
        return static_cast<scalar_t>(1) - omega_zmax();
    }

#endif

}

#endif
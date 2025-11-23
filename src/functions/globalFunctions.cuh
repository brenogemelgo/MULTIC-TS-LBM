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

namespace block
{
    __host__ __device__ [[nodiscard]] static inline consteval unsigned num_block_x() noexcept
    {
        return (mesh::nx + block::nx - 1u) / block::nx;
    }

    __host__ __device__ [[nodiscard]] static inline consteval unsigned num_block_y() noexcept
    {
        return (mesh::ny + block::ny - 1u) / block::ny;
    }

    __host__ __device__ [[nodiscard]] static inline consteval unsigned size() noexcept
    {
        return block::nx * block::ny * block::nx;
    }
}

namespace physics
{
    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T rho_water() noexcept
    {
        return static_cast<T>(1.0f);
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T rho_oil() noexcept
    {
        return static_cast<T>(0.8f);
    }
}

namespace geometry
{
    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T R2() noexcept
    {
        return static_cast<T>(mesh::radius) * static_cast<T>(mesh::radius);
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T center_x() noexcept
    {
        return (static_cast<T>(mesh::nx) - static_cast<T>(1)) * static_cast<T>(0.5f);
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T center_y() noexcept
    {
        return (static_cast<T>(mesh::ny) - static_cast<T>(1)) * static_cast<T>(0.5f);
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T center_z() noexcept
    {
        return (static_cast<T>(mesh::nz) - static_cast<T>(1)) * static_cast<T>(0.5f);
    }
}

namespace relaxation
{

#if defined(JET)

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T visc_ref() noexcept
    {
        return (static_cast<T>(physics::u_ref) * static_cast<T>(mesh::diam)) / static_cast<T>(physics::reynolds);
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T omega_ref() noexcept
    {
        return static_cast<T>(1.0f) / (static_cast<T>(0.5f) + static_cast<T>(3.0f) * visc_ref<T>());
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T omco_zmin() noexcept
    {
        return static_cast<T>(1.0f) - omega_ref<T>();
    }

#elif defined(DROPLET)

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T omega_ref() noexcept
    {
        return static_cast<T>(1.0f) / (static_cast<T>(0.5f) + static_cast<T>(3.0f) * static_cast<T>(physics::visc_ref));
    }

#endif

}

namespace LBM
{
    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T cssq() noexcept
    {
        return static_cast<T>(1.0f) / static_cast<T>(3.0f);
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T cssq_d3q7() noexcept
    {
        return static_cast<T>(1.0f) / static_cast<T>(4.0f);
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T omco() noexcept
    {
        return static_cast<T>(1.0f) - relaxation::omega_ref<T>();
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T oos() noexcept
    {
        return static_cast<T>(1.0f) / static_cast<T>(6.0f);
    }
}

namespace math
{
    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T two_pi() noexcept
    {
        return static_cast<T>(2.0f) * static_cast<T>(CUDART_PI_F);
    }
}

namespace size
{
    __host__ __device__ [[nodiscard]] static inline consteval label_t stride() noexcept
    {
        return mesh::nx * mesh::ny;
    }

    __host__ __device__ [[nodiscard]] static inline consteval label_t plane() noexcept
    {
        return mesh::nx * mesh::ny * mesh::nz;
    }
}

#if defined(JET)
namespace sponge
{
    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T K_gain() noexcept
    {
        return static_cast<T>(100.0f);
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T P_gain() noexcept
    {
        return static_cast<T>(3.0f);
    }

    template <typename T = int>
    __host__ __device__ [[nodiscard]] static inline consteval T sponge_cells() noexcept
    {
        return static_cast<T>(mesh::nz / 12u);
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T sponge() noexcept
    {
        return sponge_cells<T>() / (static_cast<T>(mesh::nz) - static_cast<T>(1));
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T z_start() noexcept
    {
        return (static_cast<T>(mesh::nz) - static_cast<T>(1) - sponge_cells<T>()) / (static_cast<T>(mesh::nz) - static_cast<T>(1));
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T inv_nz_m1() noexcept
    {
        return static_cast<T>(1.0f) / (static_cast<T>(mesh::nz) - static_cast<T>(1));
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T inv_sponge() noexcept
    {
        return static_cast<T>(1.0f) / sponge<T>();
    }
}

namespace relaxation
{

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T omega_zmax() noexcept
    {
        return static_cast<T>(1.0f) / (static_cast<T>(0.5f) + static_cast<T>(3.0f) * visc_ref<T>() * (sponge::K_gain<T>() + static_cast<T>(1.0f)));
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T omco_zmax() noexcept
    {
        return static_cast<T>(1.0f) - omega_zmax<T>();
    }

    template <typename T = scalar_t>
    __host__ __device__ [[nodiscard]] static inline consteval T omega_delta() noexcept
    {
        return omega_zmax<T>() - omega_ref<T>();
    }
}
#endif

#endif
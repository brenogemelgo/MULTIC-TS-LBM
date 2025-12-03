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
    D3Q7 velocity set class declaration

SourceFiles
    D3Q7.cuh

\*---------------------------------------------------------------------------*/

#ifndef D3Q7_CUH
#define D3Q7_CUH

#include "../cuda/utils.cuh"
#include "velocitySet.cuh"

namespace LBM
{
    class d3q7 : private velocitySet
    {
    public:
        __host__ __device__ [[nodiscard]] inline consteval d3q7(){};

        __host__ __device__ [[nodiscard]] static inline consteval label_t Q() noexcept
        {
            return static_cast<label_t>(Q_);
        }

        __host__ __device__ [[nodiscard]] static inline consteval scalar_t as2() noexcept
        {
            return static_cast<scalar_t>(4);
        }

        __host__ __device__ [[nodiscard]] static inline consteval scalar_t cs2() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(4));
        }

        __host__ __device__ [[nodiscard]] static inline consteval scalar_t w_0() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(4));
        }

        __host__ __device__ [[nodiscard]] static inline consteval scalar_t w_1() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(8));
        }

        template <label_t Dir>
        __host__ __device__ [[nodiscard]] static inline consteval scalar_t w() noexcept
        {
            if constexpr (Dir == 0)
            {
                return w_0();
            }
            else
            {
                return w_1();
            }
        }

        template <label_t Dir>
        __host__ __device__ [[nodiscard]] static inline consteval int cx() noexcept
        {
            if constexpr (Dir == 1)
            {
                return 1;
            }
            else if constexpr (Dir == 2)
            {
                return -1;
            }
            else
            {
                return 0;
            }
        }

        template <label_t Dir>
        __host__ __device__ [[nodiscard]] static inline consteval int cy() noexcept
        {
            if constexpr (Dir == 3)
            {
                return 1;
            }
            else if constexpr (Dir == 4)
            {
                return -1;
            }
            else
            {
                return 0;
            }
        }

        template <label_t Dir>
        __host__ __device__ [[nodiscard]] static inline consteval int cz() noexcept
        {
            if constexpr (Dir == 5)
            {
                return 1;
            }
            else if constexpr (Dir == 6)
            {
                return -1;
            }
            else
            {
                return 0;
            }
        }

        template <label_t Dir>
        __host__ __device__ [[nodiscard]] static inline consteval scalar_t f_eq() noexcept
        {
            foo;
        }

        template <label_t Dir>
        __host__ __device__ [[nodiscard]] static inline consteval scalar_t f_neq() noexcept
        {
            foo;
        }

    private:
        static constexpr label_t Q_ = 7;
    };
}

#endif
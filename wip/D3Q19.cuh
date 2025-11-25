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
    D3Q19 velocity set class declaration

SourceFiles
    D3Q19.cuh

\*---------------------------------------------------------------------------*/

#ifndef D3Q19_CUH
#define D3Q19_CUH

#include "../cuda/utils.cuh"
#include "velocitySet.cuh"

namespace LBM::Hydro
{
    class D3Q19 : private VelocitySet
    {
    public:
        __host__ __device__ [[nodiscard]] inline consteval D3Q19(){};

        __device__ [[nodiscard]] static inline consteval label_t Q() noexcept
        {
            return static_cast<label_t>(Q_);
        }

        __device__ [[nodiscard]] static inline consteval scalar_t as2() noexcept
        {
            return static_cast<scalar_t>(3);
        }

        __device__ [[nodiscard]] static inline consteval scalar_t cs2() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(3));
        }

        __device__ [[nodiscard]] static inline consteval scalar_t w_0() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(3));
        }

        __device__ [[nodiscard]] static inline consteval scalar_t w_1() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(18));
        }

        __device__ [[nodiscard]] static inline consteval scalar_t w_2() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(36));
        }

        template <label_t Q>
        __device__ [[nodiscard]] static inline consteval scalar_t w() noexcept
        {
            if constexpr (Q == 0)
            {
                return w_0();
            }
            else if constexpr (Q >= 1 && Q <= 6)
            {
                return w_1();
            }
            else
            {
                return w_2();
            }
        }

        template <label_t Q>
        __device__ [[nodiscard]] static inline consteval scalar_t cx() noexcept
        {
            if constexpr (Q == 1 || Q == 7 || Q == 9 || Q == 13 || Q == 15)
            {
                return 1;
            }
            else if constexpr (Q == 2 || Q == 8 || Q == 10 || Q == 14 || Q == 16)
            {
                return -1;
            }
            else
            {
                return 0;
            }
        }

        template <label_t Q>
        __device__ [[nodiscard]] static inline consteval scalar_t cy() noexcept
        {
            if constexpr (Q == 3 || Q == 7 || Q == 11 || Q == 14 || Q == 17)
            {
                return 1;
            }
            else if constexpr (Q == 4 || Q == 8 || Q == 12 || Q == 13 || Q == 18)
            {
                return -1;
            }
            else
            {
                return 0;
            }
        }

        template <label_t Q>
        __device__ [[nodiscard]] static inline consteval scalar_t cz() noexcept
        {
            if constexpr (Q == 5 || Q == 9 || Q == 11 || Q == 16 || Q == 18)
            {
                return 1;
            }
            else if constexpr (Q == 6 || Q == 10 || Q == 12 || Q == 15 || Q == 17)
            {
                return -1;
            }
            else
            {
                return 0;
            }
        }

    private:
        static constexpr label_t Q_ = 19;
    };
}

#endif
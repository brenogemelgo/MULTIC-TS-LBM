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
    Compile-time device loop unrolling implementation

SourceFiles
    constexprFor.cuh

\*---------------------------------------------------------------------------*/

#ifndef CONSTEXPRFOR_CUH
#define CONSTEXPRFOR_CUH

template <typename T, T v>
struct integralConstant
{
    static constexpr const T value = v;
    using value_type = T;
    using type = integralConstant;

    __device__ [[nodiscard]] inline consteval operator value_type() const noexcept
    {
        return value;
    }

    __device__ [[nodiscard]] inline consteval value_type operator()() const noexcept
    {
        return value;
    }
};

namespace device
{
    template <const label_t Start, const label_t End, typename F>
    __device__ inline constexpr void constexpr_for(F &&f) noexcept
    {
        if constexpr (Start < End)
        {
            f(integralConstant<label_t, Start>());
            if constexpr (Start + 1 < End)
            {
                constexpr_for<Start + 1, End>(std::forward<F>(f));
            }
        }
    }
}

#endif
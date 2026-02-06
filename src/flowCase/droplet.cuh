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
    Droplet flow case class declaration

Namespace
    LBM

SourceFiles
    droplet.cuh

\*---------------------------------------------------------------------------*/

#ifndef DROPLET_CUH
#define DROPLET_CUH

#include "flowCase.cuh"

namespace LBM
{
    class droplet : private flowCase
    {
    public:
        __host__ __device__ [[nodiscard]] inline consteval droplet(){};

        __host__ __device__ [[nodiscard]] static inline consteval bool droplet_case() noexcept
        {
            return true;
        }

        __host__ __device__ [[nodiscard]] static inline consteval bool jet_case() noexcept
        {
            return false;
        }

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void initialConditions(
            const LBMFields &fields,
            const cudaStream_t queue)
        {
            setDroplet<<<grid, block, dynamic, queue>>>(fields);
        }

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void boundaryConditions(
            const LBMFields &fields,
            const cudaStream_t queue,
            const label_t STEP)
        {
            // Full periodicity with wrap in streaming
        }

    private:
        // No private methods
    };
}

#endif
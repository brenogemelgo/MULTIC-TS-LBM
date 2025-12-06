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
    Droplet velocity set class declaration

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

        __host__ static inline void initialConditions(
            const LBMFields &fields,
            const dim3 grid3D,
            const dim3 block3D,
            const size_t dynamic,
            const cudaStream_t queue)
        {
            LBM::setDroplet<<<grid3D, block3D, dynamic, queue>>>(fields);
        }

        __host__ static inline void boundaryConditions(
            const LBMFields &fields,
            const dim3 gridZ,
            const dim3 blockZ,
            const size_t dynamic,
            const cudaStream_t queue,
            const label_t STEP)
        {
            // Placeholder
        }

    private:
        // No private methods
    };
}

#endif
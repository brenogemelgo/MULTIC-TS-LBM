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
    A header defining the constants used in the simulation

    Namespace

SourceFiles
    constants.cuh

\*---------------------------------------------------------------------------*/

#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include "cuda/utils.cuh"
#include "structs/LBMFields.cuh"
#include "functions/constexprFor.cuh"
#include "velocitySet/velocitySet.cuh"

namespace LBM
{
#if defined(VS_D3Q19)
    using VelocitySet = D3Q19;
#elif defined(VS_D3Q27)
    using VelocitySet = D3Q27;
#endif
}

namespace Phase
{
    using VelocitySet = LBM::D3Q7;
}

#define RUN_MODE
// #define SAMPLE_MODE
// #define PROFILE_MODE

#if defined(RUN_MODE)

static constexpr int MACRO_SAVE = 1000;
static constexpr int NSTEPS = 200000;

#elif defined(SAMPLE_MODE)

static constexpr int MACRO_SAVE = 100;
static constexpr int NSTEPS = 1000;

#elif defined(PROFILE_MODE)

static constexpr int MACRO_SAVE = 1;
static constexpr int NSTEPS = 0;

#endif

namespace mesh
{
    static constexpr label_t res = 200;
    static constexpr label_t nx = res;
    static constexpr label_t ny = res;
    static constexpr label_t nz = res * 4;
    static constexpr int diam = 16;
    static constexpr int radius = diam / 2;
}

namespace physics
{
    static constexpr scalar_t u_inf = static_cast<scalar_t>(0.05);

    // static constexpr int reynolds_water = 1400; // deprecated: set at globalFunctions by viscosity
    static constexpr scalar_t reynolds_oil = static_cast<scalar_t>(9.7e2);

    static constexpr scalar_t weber = static_cast<scalar_t>(3.5e5);

    static constexpr scalar_t rho_water = 1;     // phi = 0
    static constexpr scalar_t rho_oil = 0.832;   // phi = 1
    static constexpr scalar_t rho_ref = rho_oil; // reference density (oil due to diameter)
    static constexpr scalar_t sigma = rho_ref * (u_inf * u_inf * mesh::diam) / static_cast<scalar_t>(weber);

    static constexpr scalar_t width = static_cast<scalar_t>(3);                                // prefer odd values due to grid symmetry
    static constexpr scalar_t gamma = static_cast<scalar_t>(3) / static_cast<scalar_t>(width); // gamma = 1 gives approximately ~2 lu of interface width
}

#endif
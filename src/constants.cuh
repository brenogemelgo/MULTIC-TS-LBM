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
#include "flowCase/flowCase.cuh"

namespace LBM
{
#if defined(VS_D3Q19)
    using VelocitySet = D3Q19;
#elif defined(VS_D3Q27)
    using VelocitySet = D3Q27;
#endif

#if defined(DROPLET)
    using FlowCase = droplet;
#elif defined(JET)
    using FlowCase = jet;
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
static constexpr int NSTEPS = 100000;

#elif defined(SAMPLE_MODE)

static constexpr int MACRO_SAVE = 100;
static constexpr int NSTEPS = 10000;

#elif defined(PROFILE_MODE)

static constexpr int MACRO_SAVE = 1;
static constexpr int NSTEPS = 0;

#endif

#if defined(JET)

namespace mesh
{
    static constexpr label_t res = 200;
    static constexpr label_t nx = res;
    static constexpr label_t ny = res;
    static constexpr label_t nz = res * 2;
    static constexpr int diam = 20;
    static constexpr int radius = diam / 2;
}

namespace physics
{
    static constexpr scalar_t u_inf = static_cast<scalar_t>(0.05);

    static constexpr int reynolds_zero = 1400;
    static constexpr int reynolds_one = 450;

    static constexpr int weber = 500;

    static constexpr scalar_t rho_zero = 1;       // phi = 0
    static constexpr scalar_t rho_one = 0.852;    // phi = 1
    static constexpr scalar_t rho_ref = rho_zero; // ambient phase

    static constexpr scalar_t sigma = rho_ref * (u_inf * u_inf * mesh::diam) / weber;

    static constexpr scalar_t width = static_cast<scalar_t>(1);
    static constexpr scalar_t gamma = static_cast<scalar_t>(1);
}

#elif defined(DROPLET)

namespace mesh
{
    static constexpr label_t res = 64;
    static constexpr label_t nx = res;
    static constexpr label_t ny = res;
    static constexpr label_t nz = res;
    static constexpr int radius = 10;
    static constexpr int diam = 2 * radius;
}

namespace physics
{

    static constexpr scalar_t u_inf = static_cast<scalar_t>(0);

    static constexpr int reynolds = 0;
    static constexpr int weber = 0;

    static constexpr scalar_t rho_zero = 1;       // phi = 0
    static constexpr scalar_t rho_one = 0.852;    // phi = 1
    static constexpr scalar_t rho_ref = rho_zero; // ambient phase

    static constexpr scalar_t sigma = static_cast<scalar_t>(0.1);

    static constexpr scalar_t width = static_cast<scalar_t>(2);
    static constexpr scalar_t gamma = static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(width));

    static constexpr scalar_t tau_zero = static_cast<scalar_t>(0.55); // phi = 0
    static constexpr scalar_t tau_one = static_cast<scalar_t>(0.55);  // phi = 1
    static constexpr scalar_t tau_ref = tau_zero;                     // ambient phase

    static constexpr scalar_t visc_zero = (tau_zero - static_cast<scalar_t>(0.5)) / LBM::VelocitySet::as2();
    static constexpr scalar_t visc_one = (tau_one - static_cast<scalar_t>(0.5)) / LBM::VelocitySet::as2();
    static constexpr scalar_t visc_ref = (tau_ref - static_cast<scalar_t>(0.5)) / LBM::VelocitySet::as2();
}

#endif

#endif
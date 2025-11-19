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
    Caller CUDA kernels for initial and boundary conditions

Namespace
    LBM

SourceFiles
    caller.cuh

\*---------------------------------------------------------------------------*/

#ifndef MAIN_CUH
#define MAIN_CUH

namespace LBM
{
    // Initial conditions
    __global__ void callSetFields(
        LBMFields d)
    {
        InitialConditions::setFields(d);
    }

#if defined(JET)

    __global__ void callSetJet(
        LBMFields d)
    {
        InitialConditions::setJet(d);
    }

#elif defined(DROPLET)

    __global__ void callSetDroplet(
        LBMFields d)
    {
        InitialConditions::setDroplet(d);
    }

#endif

    __global__ void callSetDistros(
        LBMFields d)
    {
        InitialConditions::setDistros(d);
    }

    // Boundary conditions
    __global__ void callInflow(
        LBMFields d)
    {
        BoundaryConditions::applyInflow(d);
    }

    __global__ void callOutflow(
        LBMFields d)
    {
        BoundaryConditions::applyOutflow(d);
    }

    __global__ void callPeriodicX(
        LBMFields d)
    {
        BoundaryConditions::periodicX(d);
    }

    __global__ void callPeriodicY(
        LBMFields d)
    {
        BoundaryConditions::periodicY(d);
    }
}

#endif
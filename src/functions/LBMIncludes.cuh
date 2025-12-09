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
    General includes

Namespace
    LBM

SourceFiles
    LBMIncludes.cuh

\*---------------------------------------------------------------------------*/

#ifndef LBMINCLUDES_CUH
#define LBMINCLUDES_CUH

namespace LBM
{
    // Initial conditions
    __global__ void setFields(LBMFields d);
    __global__ void setJet(LBMFields d);
    __global__ void setDroplet(LBMFields d);
    __global__ void setDistros(LBMFields d);

    // Moments and core routines
    __global__ void computeMoments(LBMFields d);
    __global__ void streamCollide(LBMFields d);

    // Boundary conditions
    __global__ void callInflow(LBMFields d, const label_t t);
    __global__ void callOutflow(LBMFields d);
}

#endif

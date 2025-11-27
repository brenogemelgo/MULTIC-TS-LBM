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
    Allocate device fields

SourceFiles
    allocateFields.cuh

\*---------------------------------------------------------------------------*/

#ifndef ALLOCATEFIELDS_CUH
#define ALLOCATEFIELDS_CUH

struct LBMFields
{
    scalar_t *rho;
    scalar_t *ux;
    scalar_t *uy;
    scalar_t *uz;
    scalar_t *pxx;
    scalar_t *pyy;
    scalar_t *pzz;
    scalar_t *pxy;
    scalar_t *pxz;
    scalar_t *pyz;

    scalar_t *phi;
    scalar_t *normx;
    scalar_t *normy;
    scalar_t *normz;
    scalar_t *ind;
    scalar_t *ffx;
    scalar_t *ffy;
    scalar_t *ffz;

    pop_t *f;
    scalar_t *g;

#if AVERAGE_UZ

    scalar_t *avg;

#endif
};

LBMFields fields{};

__host__ [[gnu::cold]] static inline void setDeviceFields()
{
    constexpr size_t NCELLS = static_cast<size_t>(mesh::nx) * static_cast<size_t>(mesh::ny) * static_cast<size_t>(mesh::nz);
    constexpr size_t SIZE = NCELLS * sizeof(scalar_t);
    constexpr size_t F_DIST_SIZE = NCELLS * static_cast<size_t>(LBM::VelocitySet::Q()) * sizeof(pop_t);
    constexpr size_t G_DIST_SIZE = NCELLS * static_cast<size_t>(LBM::PhaseVelocitySet::Q()) * sizeof(scalar_t);

    static_assert(NCELLS > 0, "Empty grid?");
    static_assert(SIZE / sizeof(scalar_t) == NCELLS, "SIZE overflow");
    static_assert(F_DIST_SIZE / sizeof(pop_t) == NCELLS * size_t(LBM::VelocitySet::Q()), "F_DIST_SIZE overflow");
    static_assert(G_DIST_SIZE / sizeof(scalar_t) == NCELLS * size_t(LBM::PhaseVelocitySet::Q()), "G_DIST_SIZE overflow");

    checkCudaErrors(cudaMalloc(&fields.rho, SIZE));
    checkCudaErrors(cudaMalloc(&fields.ux, SIZE));
    checkCudaErrors(cudaMalloc(&fields.uy, SIZE));
    checkCudaErrors(cudaMalloc(&fields.uz, SIZE));
    checkCudaErrors(cudaMalloc(&fields.pxx, SIZE));
    checkCudaErrors(cudaMalloc(&fields.pyy, SIZE));
    checkCudaErrors(cudaMalloc(&fields.pzz, SIZE));
    checkCudaErrors(cudaMalloc(&fields.pxy, SIZE));
    checkCudaErrors(cudaMalloc(&fields.pxz, SIZE));
    checkCudaErrors(cudaMalloc(&fields.pyz, SIZE));

    checkCudaErrors(cudaMalloc(&fields.phi, SIZE));
    checkCudaErrors(cudaMalloc(&fields.normx, SIZE));
    checkCudaErrors(cudaMalloc(&fields.normy, SIZE));
    checkCudaErrors(cudaMalloc(&fields.normz, SIZE));
    checkCudaErrors(cudaMalloc(&fields.ind, SIZE));
    checkCudaErrors(cudaMalloc(&fields.ffx, SIZE));
    checkCudaErrors(cudaMalloc(&fields.ffy, SIZE));
    checkCudaErrors(cudaMalloc(&fields.ffz, SIZE));

    checkCudaErrors(cudaMalloc(&fields.f, F_DIST_SIZE));
    checkCudaErrors(cudaMalloc(&fields.g, G_DIST_SIZE));

#if AVERAGE_UZ

    checkCudaErrors(cudaMalloc(&fields.avg, SIZE));

#endif

    getLastCudaErrorOutline("setDeviceFields: post-initialization");
}

#endif
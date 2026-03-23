/*---------------------------------------------------------------------------*\
|                                                                             |
| phaseFieldLBM: CUDA-based multicomponent Lattice Boltzmann Method           |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/brenogemelgo/phaseFieldLBM                       |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Breno Gemelgo (Geoenergia Lab, UDESC)

Description
    Compile-time phase velocity-set selection and stencil compatibility checks

SourceFiles
    PhaseVelocitySet.cuh

\*---------------------------------------------------------------------------*/

#ifndef PHASEVELOCITYSET_CUH
#define PHASEVELOCITYSET_CUH

#include "velocitySet/VelocitySet.cuh"

namespace phase
{
    using velocitySet = lbm::D3Q7;
}

static_assert(lbm::D3Q19::max_abs_c() == phase::velocitySet::max_abs_c(), "Hydrodynamic and phase velocity sets must have identical stencil radius.");
static_assert(lbm::D3Q27::max_abs_c() == phase::velocitySet::max_abs_c(), "Hydrodynamic and phase velocity sets must have identical stencil radius.");

#endif

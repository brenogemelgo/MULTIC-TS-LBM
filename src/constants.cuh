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
    Deprecated compatibility shim. Use focused headers directly:
      - runtime/RuntimeState.cuh
      - functions/constexprFor.cuh
      - config/PhaseVelocitySet.cuh

SourceFiles
    constants.cuh

\*---------------------------------------------------------------------------*/

#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#if !defined(PHASEFIELDLBM_SUPPRESS_DEPRECATED_CONSTANTS_HEADER)
#if defined(__GNUC__) || defined(__clang__)
#pragma message("constants.cuh is deprecated; include runtime/RuntimeState.cuh, functions/constexprFor.cuh, and config/PhaseVelocitySet.cuh directly.")
#endif
#endif

#include "runtime/RuntimeState.cuh"
#include "functions/constexprFor.cuh"
#include "config/PhaseVelocitySet.cuh"

#endif

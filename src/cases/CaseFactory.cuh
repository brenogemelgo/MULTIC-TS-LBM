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
    Runtime case object factory

SourceFiles
    CaseFactory.cuh

\*---------------------------------------------------------------------------*/

#ifndef CASEFACTORY_CUH
#define CASEFACTORY_CUH

#include "CaseBase.cuh"
#include "JetCase.cuh"
#include "DropletCase.cuh"

namespace cases
{
    __host__ [[nodiscard]] static inline std::unique_ptr<Case> createCase(const std::string &caseName)
    {
        const std::string lowered = runtime::toLower(caseName);

        if (lowered == "jet")
        {
            return std::make_unique<JetCase>();
        }
        if (lowered == "droplet")
        {
            return std::make_unique<DropletCase>();
        }

        throw std::runtime_error(
            "Unsupported caseName in programControl: '" + caseName +
            "'. Supported values: jet, droplet.");
    }
}

#endif

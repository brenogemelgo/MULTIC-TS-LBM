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
    Runtime case interface for case-specific validation and constant setup

SourceFiles
    CaseBase.cuh

\*---------------------------------------------------------------------------*/

#ifndef CASEBASE_CUH
#define CASEBASE_CUH

#include "runtime/RuntimeConfig.cuh"
#include "runtime/RuntimeState.cuh"
#include "velocitySet/D3Q7.cuh"

namespace runtime
{
    enum class CaseKind
    {
        Jet,
        Droplet
    };
}

namespace cases
{
    class Case
    {
    public:
        virtual ~Case() = default;

        [[nodiscard]] virtual std::string_view name() const noexcept = 0;
        [[nodiscard]] virtual runtime::CaseKind kind() const noexcept = 0;

        virtual void validate(const runtime::RuntimeConfig &cfg) const = 0;

        virtual void applyCaseConstants(
            runtime::DeviceConstants &constants,
            const runtime::RuntimeConfig &cfg) const = 0;

    protected:
        [[nodiscard]] static inline label_t diameterFromCharLength(const scalar_t L_char)
        {
            const scalar_t rounded = std::round(L_char);
            if (rounded < static_cast<scalar_t>(2))
            {
                throw std::runtime_error("programControl: L_char is too small to derive a valid diameter (>= 2 required).");
            }

            return static_cast<label_t>(rounded);
        }

        [[nodiscard]] static inline scalar_t computeGammaFromInterfaceWidth(const scalar_t interfaceWidth)
        {
            const scalar_t tau_g = static_cast<scalar_t>(1);
            const scalar_t diff_int = lbm::D3Q7::cs2() * (tau_g - static_cast<scalar_t>(0.5));
            const scalar_t kappa = static_cast<scalar_t>(4) * diff_int / interfaceWidth;
            return kappa / lbm::D3Q7::cs2();
        }
    };
}

#endif

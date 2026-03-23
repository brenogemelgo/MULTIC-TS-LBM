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
    Runtime-selected jet case metadata and validation

SourceFiles
    JetCase.cuh

\*---------------------------------------------------------------------------*/

#ifndef CASES_JETCASE_CUH
#define CASES_JETCASE_CUH

#include "CaseBase.cuh"

namespace cases
{
    class JetCase final : public Case
    {
    public:
        [[nodiscard]] std::string_view name() const noexcept override { return "jetFlow"; }

        [[nodiscard]] runtime::CaseKind kind() const noexcept override
        {
            return runtime::CaseKind::Jet;
        }

        void validate(const runtime::RuntimeConfig &cfg) const override
        {
            const auto &p = cfg.control;
            const auto &m = cfg.mesh;

            if (p.ReA <= static_cast<scalar_t>(0) || p.ReB <= static_cast<scalar_t>(0))
            {
                throw std::runtime_error("Jet case requires ReA > 0 and ReB > 0.");
            }
            if (p.We <= static_cast<scalar_t>(0))
            {
                throw std::runtime_error("Jet case requires We > 0.");
            }
            if (p.u_inf <= static_cast<scalar_t>(0))
            {
                throw std::runtime_error("Jet case requires u_inf > 0.");
            }
            if (m.nz < 12)
            {
                throw std::runtime_error("Jet case requires nz >= 12 for sponge layer setup.");
            }
        }

        void applyCaseConstants(
            runtime::DeviceConstants &constants,
            const runtime::RuntimeConfig &cfg) const override
        {
            const auto &p = cfg.control;

            const label_t diameter = diameterFromCharLength(p.L_char);
            const label_t radius = diameter / 2;

            const scalar_t nuWater = (p.u_inf * static_cast<scalar_t>(diameter)) / p.ReA;
            const scalar_t nuOil = (p.u_inf * static_cast<scalar_t>(diameter)) / p.ReB;

            // D3Q19 and D3Q27 both use as2 = 3 in this solver.
            const scalar_t tauWater = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3) * nuWater;
            const scalar_t tauOil = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3) * nuOil;

            constants.diameter = diameter;
            constants.radius = radius;

            constants.sigma = (p.u_inf * p.u_inf * static_cast<scalar_t>(diameter)) / p.We;
            constants.interface_width = static_cast<scalar_t>(4);
            constants.gamma = computeGammaFromInterfaceWidth(constants.interface_width);

            constants.tau_water = tauWater;
            constants.tau_oil = tauOil;
            constants.omega_ref = static_cast<scalar_t>(1) / tauWater;
        }
    };
}

#endif

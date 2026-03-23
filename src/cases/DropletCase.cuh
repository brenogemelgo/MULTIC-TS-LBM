/*---------------------------------------------------------------------------*\
|                                                                             |
| phaseFieldLBM: CUDA-based multicomponent Lattice Boltzmann Method           |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/brenogemelgo/phaseFieldLBM                       |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Description
    Runtime-selected droplet case metadata and validation

SourceFiles
    DropletCase.cuh

\*---------------------------------------------------------------------------*/

#ifndef CASES_DROPLETCASE_CUH
#define CASES_DROPLETCASE_CUH

#include "CaseBase.cuh"

namespace cases
{
    class DropletCase final : public Case
    {
    public:
        [[nodiscard]] std::string_view name() const noexcept override { return "droplet"; }

        [[nodiscard]] runtime::CaseKind kind() const noexcept override
        {
            return runtime::CaseKind::Droplet;
        }

        void validate(const runtime::RuntimeConfig &cfg) const override
        {
            (void)cfg;
        }

        void applyCaseConstants(
            runtime::DeviceConstants &constants,
            const runtime::RuntimeConfig &cfg) const override
        {
            const auto &p = cfg.control;

            const label_t diameter = diameterFromCharLength(p.L_char);
            const label_t radius = diameter / 2;

            constants.diameter = diameter;
            constants.radius = radius;

            constants.interface_width = static_cast<scalar_t>(4);
            constants.gamma = computeGammaFromInterfaceWidth(constants.interface_width);
            constants.sigma = static_cast<scalar_t>(0.03);

            constants.tau_water = static_cast<scalar_t>(0);
            constants.tau_oil = static_cast<scalar_t>(0);
            constants.omega_ref = static_cast<scalar_t>(1) / static_cast<scalar_t>(0.55);
        }
    };
}

#endif

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
    Compile-time case policies selected once at startup from runtime case object

SourceFiles
    CasePolicies.cuh

\*---------------------------------------------------------------------------*/

#ifndef CASEPOLICIES_CUH
#define CASEPOLICIES_CUH

#include "BoundaryConditions.cuh"

namespace case_policy
{
    struct Jet
    {
        template <typename HydroVS>
        __device__ [[nodiscard]] static inline scalar_t collisionOmega(
            const scalar_t phi,
            const label_t z) noexcept
        {
            const scalar_t tau_phi = relaxation::tau_local(phi);
            const scalar_t r = device::sponge_ramp(z);
            const scalar_t tau_eff = tau_phi + r * (relaxation::tau_zmax(phi) - tau_phi);
            return static_cast<scalar_t>(1) / tau_eff;
        }

        __device__ static inline void applyPeriodic(label_t &xx, label_t &yy, label_t &zz) noexcept
        {
            xx = device::periodic_wrap(xx, mesh::nx());
            yy = device::periodic_wrap(yy, mesh::ny());
            (void)zz;
        }

        template <typename HydroVS>
        __host__ static inline void launchInitial(
            const dim3 &grid,
            const dim3 &block,
            const size_t dynamic,
            const LBMFields &fields,
            const cudaStream_t queue)
        {
            (void)sizeof(HydroVS);
            lbm::setJet<<<grid, block, dynamic, queue>>>(fields);
        }

        template <typename HydroVS>
        __host__ static inline void launchBoundary(
            const LBMFields &fields,
            const cudaStream_t queue,
            const label_t t,
            const dim3 &gridX,
            const dim3 &blockX,
            const dim3 &gridY,
            const dim3 &blockY,
            const dim3 &gridZ,
            const dim3 &blockZ,
            const size_t dynamic)
        {
            (void)gridX;
            (void)blockX;
            (void)gridY;
            (void)blockY;
            lbm::callInflow<HydroVS><<<gridZ, blockZ, dynamic, queue>>>(fields, t);
            lbm::callOutflow<HydroVS><<<gridZ, blockZ, dynamic, queue>>>(fields);
        }
    };

    struct Droplet
    {
        template <typename HydroVS>
        __device__ [[nodiscard]] static inline scalar_t collisionOmega(
            const scalar_t phi,
            const label_t z) noexcept
        {
            (void)sizeof(HydroVS);
            (void)phi;
            (void)z;
            return relaxation::omega_ref();
        }

        __device__ static inline void applyPeriodic(label_t &xx, label_t &yy, label_t &zz) noexcept
        {
            xx = device::periodic_wrap(xx, mesh::nx());
            yy = device::periodic_wrap(yy, mesh::ny());
            zz = device::periodic_wrap(zz, mesh::nz());
        }

        template <typename HydroVS>
        __host__ static inline void launchInitial(
            const dim3 &grid,
            const dim3 &block,
            const size_t dynamic,
            const LBMFields &fields,
            const cudaStream_t queue)
        {
            (void)sizeof(HydroVS);
            lbm::setDroplet<<<grid, block, dynamic, queue>>>(fields);
        }

        template <typename HydroVS>
        __host__ static inline void launchBoundary(
            const LBMFields &fields,
            const cudaStream_t queue,
            const label_t t,
            const dim3 &gridX,
            const dim3 &blockX,
            const dim3 &gridY,
            const dim3 &blockY,
            const dim3 &gridZ,
            const dim3 &blockZ,
            const size_t dynamic)
        {
            (void)sizeof(HydroVS);
            (void)fields;
            (void)queue;
            (void)t;
            (void)gridX;
            (void)blockX;
            (void)gridY;
            (void)blockY;
            (void)gridZ;
            (void)blockZ;
            (void)dynamic;
        }
    };
}

#endif

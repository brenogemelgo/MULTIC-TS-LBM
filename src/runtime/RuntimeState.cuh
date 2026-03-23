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
    Host/device runtime constants mirrored to CUDA constant memory

SourceFiles
    RuntimeState.cuh

\*---------------------------------------------------------------------------*/

#ifndef RUNTIMESTATE_CUH
#define RUNTIMESTATE_CUH

#include "LBMIncludes.cuh"

namespace runtime
{
    struct DeviceConstants
    {
        label_t nx = 0;
        label_t ny = 0;
        label_t nz = 0;

        label_t stride = 0;
        label_t cells = 0;

        label_t diameter = 0;
        label_t radius = 0;

        scalar_t u_inf = static_cast<scalar_t>(0);
        scalar_t reynolds_water = static_cast<scalar_t>(0);
        scalar_t reynolds_oil = static_cast<scalar_t>(0);
        scalar_t weber = static_cast<scalar_t>(0);

        scalar_t sigma = static_cast<scalar_t>(0);
        scalar_t interface_width = static_cast<scalar_t>(0);
        scalar_t gamma = static_cast<scalar_t>(0);

        scalar_t tau_water = static_cast<scalar_t>(0);
        scalar_t tau_oil = static_cast<scalar_t>(0);
        scalar_t omega_ref = static_cast<scalar_t>(1);

        label_t nTimeSteps = 0;
        label_t saveInterval = 0;
    };

    extern DeviceConstants HOST_CONSTANTS;
    extern __device__ __constant__ DeviceConstants DEVICE_CONSTANTS;

    __host__ static inline void uploadDeviceConstants(const DeviceConstants &constants)
    {
        HOST_CONSTANTS = constants;
        checkCudaErrorsOutline(cudaMemcpyToSymbol(DEVICE_CONSTANTS, &constants, sizeof(DeviceConstants)));
    }

    __host__ __device__ [[nodiscard]] static inline const DeviceConstants &constants() noexcept
    {
#ifdef __CUDA_ARCH__
        return DEVICE_CONSTANTS;
#else
        return HOST_CONSTANTS;
#endif
    }
}

#endif

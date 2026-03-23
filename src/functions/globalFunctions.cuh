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
    Runtime geometry, math, relaxation, and utility functions shared across solver modules

Namespace
    block
    mesh
    physics
    control
    geometry
    relaxation
    math
    size
    sponge

SourceFiles
    globalFunctions.cuh

\*---------------------------------------------------------------------------*/

#ifndef GLOBALFUNCTIONS_CUH
#define GLOBALFUNCTIONS_CUH

#include "runtime/RuntimeState.cuh"
#include <type_traits>

namespace mesh
{
    __host__ __device__ [[nodiscard]] static inline label_t nx() noexcept
    {
        return runtime::constants().nx;
    }

    __host__ __device__ [[nodiscard]] static inline label_t ny() noexcept
    {
        return runtime::constants().ny;
    }

    __host__ __device__ [[nodiscard]] static inline label_t nz() noexcept
    {
        return runtime::constants().nz;
    }

    __host__ __device__ [[nodiscard]] static inline label_t diam() noexcept
    {
        return runtime::constants().diameter;
    }

    __host__ __device__ [[nodiscard]] static inline label_t radius() noexcept
    {
        return runtime::constants().radius;
    }
}

namespace physics
{
    __host__ __device__ [[nodiscard]] static inline scalar_t u_inf() noexcept
    {
        return runtime::constants().u_inf;
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t reynolds_water() noexcept
    {
        return runtime::constants().reynolds_water;
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t reynolds_oil() noexcept
    {
        return runtime::constants().reynolds_oil;
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t weber() noexcept
    {
        return runtime::constants().weber;
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t sigma() noexcept
    {
        return runtime::constants().sigma;
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t interface_width() noexcept
    {
        return runtime::constants().interface_width;
    }

    __host__ __device__ [[nodiscard]] static inline scalar_t gamma() noexcept
    {
        return runtime::constants().gamma;
    }
}

namespace control
{
    __host__ __device__ [[nodiscard]] static inline label_t nTimeSteps() noexcept
    {
        return runtime::constants().nTimeSteps;
    }

    __host__ __device__ [[nodiscard]] static inline label_t saveInterval() noexcept
    {
        return runtime::constants().saveInterval;
    }
}

namespace block
{
    __device__ __host__ [[nodiscard]] static inline consteval label_t nx() noexcept
    {
        return 32;
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t ny() noexcept
    {
        return 4;
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t nz() noexcept
    {
        return 4;
    }

    __device__ __host__ [[nodiscard]] static inline label_t num_x() noexcept
    {
        return (mesh::nx() + block::nx() - 1) / block::nx();
    }

    __device__ __host__ [[nodiscard]] static inline label_t num_y() noexcept
    {
        return (mesh::ny() + block::ny() - 1) / block::ny();
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t size() noexcept
    {
        return block::nx() * block::ny() * block::nz();
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t pad() noexcept
    {
        return 1;
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t tile_nx() noexcept
    {
        return block::nx() + 2 * block::pad();
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t tile_ny() noexcept
    {
        return block::ny() + 2 * block::pad();
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t tile_nz() noexcept
    {
        return block::nz() + 2 * block::pad();
    }
}

namespace size
{
    __device__ __host__ [[nodiscard]] static inline label_t stride() noexcept
    {
        return runtime::constants().stride;
    }

    __device__ __host__ [[nodiscard]] static inline label_t cells() noexcept
    {
        return runtime::constants().cells;
    }
}

namespace geometry
{
    __device__ __host__ [[nodiscard]] static inline scalar_t R2() noexcept
    {
        const scalar_t r = static_cast<scalar_t>(mesh::radius());
        return r * r;
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t center_x() noexcept
    {
        return static_cast<scalar_t>(mesh::nx() - 1) * static_cast<scalar_t>(0.5);
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t center_y() noexcept
    {
        return static_cast<scalar_t>(mesh::ny() - 1) * static_cast<scalar_t>(0.5);
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t center_z() noexcept
    {
        return static_cast<scalar_t>(mesh::nz() - 1) * static_cast<scalar_t>(0.5);
    }
}

namespace math
{
    __device__ __host__ [[nodiscard]] static inline consteval scalar_t two_pi() noexcept
    {
        return static_cast<scalar_t>(2) * static_cast<scalar_t>(CUDART_PI_F);
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t sqrt(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::sqrtf(x);
        }
        else
        {
            return ::sqrt(x);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t log(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::logf(x);
        }
        else
        {
            return ::log(x);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t min(
        const scalar_t a,
        const scalar_t b) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fminf(a, b);
        }
        else
        {
            return ::fmin(a, b);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t max(
        const scalar_t a,
        const scalar_t b) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fmaxf(a, b);
        }
        else
        {
            return ::fmax(a, b);
        }
    }

    template <scalar_t Lo, scalar_t Hi>
    __device__ __host__ [[nodiscard]] static inline scalar_t clamp(const scalar_t x) noexcept
    {
        static_assert(Lo <= Hi, "Invalid clamp bounds");
        return max(Lo, min(x, Hi));
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t cos(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::cosf(x);
        }
        else
        {
            return ::cos(x);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t tanh(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::tanhf(x);
        }
        else
        {
            return ::tanh(x);
        }
    }

    __device__ __host__ static inline void sincos(
        const scalar_t x,
        scalar_t *s,
        scalar_t *c) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            ::sincosf(x, s, c);
        }
        else
        {
            ::sincos(x, s, c);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t fma(
        const scalar_t a,
        const scalar_t b,
        const scalar_t c) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fmaf(a, b, c);
        }
        else
        {
            return ::fma(a, b, c);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t abs(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fabsf(x);
        }
        else
        {
            return ::fabs(x);
        }
    }
}

namespace sponge
{
    __device__ __host__ [[nodiscard]] static inline consteval scalar_t K_gain() noexcept
    {
        return static_cast<scalar_t>(100);
    }

    __device__ __host__ [[nodiscard]] static inline label_t sponge_cells() noexcept
    {
        return (mesh::nz() / 12u) > 0u ? (mesh::nz() / 12u) : 1u;
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t sponge() noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(sponge_cells()) / static_cast<double>(mesh::nz() - 1));
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t z_start() noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(mesh::nz() - 1 - sponge_cells()) / static_cast<double>(mesh::nz() - 1));
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t inv_nz_m1() noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(mesh::nz() - 1));
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t inv_sponge() noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(sponge()));
    }
}

namespace relaxation
{
    template <typename VelocitySet>
    __device__ __host__ [[nodiscard]] static inline constexpr scalar_t omega_from_nu(const scalar_t nu) noexcept
    {
        return static_cast<scalar_t>(1) /
               (static_cast<scalar_t>(0.5) + static_cast<scalar_t>(VelocitySet::as2()) * nu);
    }

    template <typename VelocitySet>
    __device__ __host__ [[nodiscard]] static inline constexpr scalar_t tau_from_nu(const scalar_t nu) noexcept
    {
        return static_cast<scalar_t>(0.5) + static_cast<scalar_t>(VelocitySet::as2()) * nu;
    }

    __device__ __host__ [[nodiscard]] static inline constexpr scalar_t omega_from_tau(const scalar_t tau) noexcept
    {
        return static_cast<scalar_t>(1) / tau;
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t tau_water() noexcept
    {
        return runtime::constants().tau_water;
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t tau_oil() noexcept
    {
        return runtime::constants().tau_oil;
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t omega_ref() noexcept
    {
        return runtime::constants().omega_ref;
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t tau_local(const scalar_t phi) noexcept
    {
        return (static_cast<scalar_t>(1) - phi) * tau_water() + phi * tau_oil();
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t tau_zmax(const scalar_t phi) noexcept
    {
        const scalar_t tw = tau_water();
        const scalar_t to = tau_oil();
        if (tw <= static_cast<scalar_t>(0) || to <= static_cast<scalar_t>(0))
        {
            return static_cast<scalar_t>(1) / omega_ref();
        }

        const scalar_t tau = tau_local(phi);
        return static_cast<scalar_t>(0.5) + (tau - static_cast<scalar_t>(0.5)) * (static_cast<scalar_t>(1) + sponge::K_gain());
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t omega_zmin() noexcept
    {
        const scalar_t tau = tau_water();
        if (tau <= static_cast<scalar_t>(0))
        {
            return omega_ref();
        }
        return omega_from_tau(tau);
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t omega_zmax(const scalar_t phi) noexcept
    {
        return omega_from_tau(tau_zmax(phi));
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t omco_ref() noexcept
    {
        return static_cast<scalar_t>(1) - omega_ref();
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t omco_zmin() noexcept
    {
        return static_cast<scalar_t>(1) - omega_zmin();
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t omco_zmax(const scalar_t phi) noexcept
    {
        return static_cast<scalar_t>(1) - omega_zmax(phi);
    }
}

#endif

#pragma once

#include "constants.cuh"
#include "auxFunctions.cuh"

namespace device
{
    __device__ [[nodiscard]] inline label_t global3(
        const label_t x, const label_t y, const label_t z) noexcept
    {
        return x + y * mesh::nx + z * size::stride();
    }

    __device__ [[nodiscard]] inline label_t global4(
        const label_t x, const label_t y, const label_t z,
        const label_t Q) noexcept
    {
        return Q * size::plane() + global3(x, y, z);
    }

    __device__ [[nodiscard]] inline label_t globalThreadIdx(
        const label_t tx, const label_t ty, const label_t tz,
        const label_t bx, const label_t by, const label_t bz) noexcept
    {
        return (tx + block::nx * (ty + block::ny * (tz + block::nz * (bx + block::num_block_x() * (by + block::num_block_y() * bz)))));
    }

#if defined(JET)

    __device__ [[nodiscard]] inline scalar_t cubic_sponge(const label_t z) noexcept
    {
        const scalar_t zn = static_cast<scalar_t>(z) * sponge::inv_nz_m1<scalar_t>();
        const scalar_t s = fminf(fmaxf((zn - sponge::z_start<scalar_t>()) * sponge::inv_sponge<scalar_t>(), 0.0f), 1.0f);
        const scalar_t s2 = s * s;
        const scalar_t ramp = s2 * s;
        return fmaf(ramp, relaxation::omega_delta<scalar_t>(), relaxation::omega_ref<scalar_t>());
    }

#endif

    __device__ [[nodiscard]] inline scalar_t smoothstep(
        const scalar_t edge0, const scalar_t edge1,
        scalar_t x) noexcept
    {
        x = __saturatef((x - edge0) / (edge1 - edge0));
        return x * x * (3.0f - 2.0f * x);
    }

    __device__ [[nodiscard]] inline scalar_t interpolate_rho(scalar_t phi) noexcept
    {
        return fmaf(phi, (physics::rho_oil<scalar_t>() - physics::rho_water<scalar_t>()), physics::rho_water<scalar_t>());
    }

    template <const label_t Start, const label_t End, typename F>
    __device__ inline constexpr void constexpr_for(F &&f) noexcept
    {
        if constexpr (Start < End)
        {
            f(integralConstant<label_t, Start>());
            if constexpr (Start + 1 < End)
            {
                constexpr_for<Start + 1, End>(std::forward<F>(f));
            }
        }
    }
}
#pragma once
#include "constants.cuh"

namespace device
{
    __device__ [[nodiscard]] inline label_t global3(
        const label_t x, const label_t y, const label_t z) noexcept
    {
        return x + y * mesh::nx + z * STRIDE;
    }

    __device__ [[nodiscard]] inline label_t global4(
        const label_t x, const label_t y, const label_t z,
        const label_t Q) noexcept
    {
        return Q * PLANE + global3(x, y, z);
    }

    __device__ [[nodiscard]] inline label_t globalThreadIdx(
        const label_t tx, const label_t ty, const label_t tz,
        const label_t bx, const label_t by, const label_t bz) noexcept
    {
        return (tx + block::nx * (ty + block::ny * (tz + block::nz * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz)))));
    }

    __device__ [[nodiscard]] inline label_t hash32(label_t x) noexcept
    {
        x ^= x >> 16;
        x *= 0x7FEB352Du;
        x ^= x >> 15;
        x *= 0x846CA68Bu;
        x ^= x >> 16;
        return x;
    }

    __device__ [[nodiscard]] inline scalar_t uniform01(const label_t seed) noexcept
    {
        return (static_cast<scalar_t>(seed) + 0.5f) * 2.3283064365386963e-10f;
    }

    __device__ [[nodiscard]] inline scalar_t pseudo_box_muller(scalar_t rrx) noexcept
    {
        const scalar_t r = sqrtf(-2.0f * logf(fmaxf(rrx, 1e-12f)));
        const scalar_t theta = TWO_PI * rrx;
        return r * cosf(theta);
    }

    __device__ inline void box_muller(
        scalar_t rrx, scalar_t rry,
        scalar_t &z1, scalar_t &z2) noexcept
    {
        rrx = fmaxf(rrx, 1e-12f);
        const scalar_t r = sqrtf(-2.0f * logf(rrx));
        const scalar_t theta = TWO_PI * rry;
        scalar_t s, c;
        sincosf(theta, &s, &c);
        z1 = r * c;
        z2 = r * s;
    }

    template <label_t NOISE_PERIOD = 10>
    __device__ [[nodiscard]] inline scalar_t pseudo_gaussian_noise(
        const label_t x, const label_t y,
        const label_t STEP) noexcept
    {
        const label_t call_idx = STEP / NOISE_PERIOD;
        const label_t seed = 0x9E3779B9u ^ x ^ (y * 0x85EBCA6Bu) ^ (call_idx * 0xC2B2AE35u);
        const scalar_t u = uniform01(hash32(seed));
        return pseudo_box_muller(u);
    }

    template <label_t NOISE_PERIOD = 10>
    __device__ [[nodiscard]] inline scalar_t gaussian_noise(
        const label_t x, const label_t y,
        const label_t STEP) noexcept
    {
        const label_t call_idx = STEP / NOISE_PERIOD;
        const label_t pair_idx = call_idx >> 1;
        const bool use_second = (call_idx & 1) != 0;
        const label_t base = 0x9E3779B9u ^ x ^ (y * 0x85EBCA6Bu) ^ (pair_idx * 0xC2B2AE35u);
        const scalar_t rrx = uniform01(hash32(base));
        const scalar_t rry = uniform01(hash32(base ^ 0x68BC21EBu));
        scalar_t z1, z2;
        box_muller(rrx, rry, z1, z2);
        return use_second ? z2 : z1;
    }

    // const scalar_t z = gaussian_noise<10>(x, y, STEP);
    // const scalar_t uz = d.uz[idx3_in] + 0.004f * z;

    // const scalar_t z = pseudo_gaussian_noise<10>(x, y, STEP);
    // const scalar_t uz = d.uz[idx3_in] + 0.004f * z;

#if defined(JET)

    __device__ [[nodiscard]] inline scalar_t cubic_sponge(const label_t z) noexcept
    {
        const scalar_t zn = static_cast<scalar_t>(z) * INV_NZ_M1;
        const scalar_t s = fminf(fmaxf((zn - Z_START) * INV_SPONGE, 0.0f), 1.0f);
        const scalar_t s2 = s * s;
        const scalar_t ramp = s2 * s;
        return fmaf(ramp, OMEGA_DELTA, OMEGA_REF);
    }

#endif

    __device__ [[nodiscard]] inline scalar_t smoothstep(
        scalar_t edge0, scalar_t edge1,
        scalar_t x) noexcept
    {
        x = __saturatef((x - edge0) / (edge1 - edge0));
        return x * x * (3.0f - 2.0f * x);
    }

    __device__ [[nodiscard]] inline scalar_t interpolate_rho(scalar_t phi) noexcept
    {
        return fmaf(phi, (RHO_OIL - RHO_WATER), RHO_WATER);
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
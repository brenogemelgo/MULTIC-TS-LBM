#pragma once
#include "constants.cuh"

[[nodiscard]] __device__ __forceinline__ 
idx_t global3(
    const idx_t x,
    const idx_t y,
    const idx_t z
) noexcept {
    return x + y * NX + z * STRIDE;
}

[[nodiscard]] __device__ __forceinline__ 
idx_t global4(
    const idx_t x, 
    const idx_t y,
    const idx_t z,
    const idx_t Q
) noexcept {
    return Q * PLANE + global3(x, y, z);
}

[[nodiscard]] __device__ __forceinline__ 
seed_t hash32(
    seed_t x
) noexcept {
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}


[[nodiscard]] __device__ __forceinline__
float u01(
    const seed_t x
) noexcept {
    return (__uint2float_rn(x) + 0.5f) * 2.3283064365386963e-10f; 
}

__device__ __forceinline__ 
void box_muller(
    float u1, 
    const float u2, 
    float &z1, 
    float &z2
) noexcept {
    u1 = fmaxf(u1, 1e-12f);
    const float r = __fsqrt_rn(-2.0f * __logf(u1));
    const float ang = 6.2831853071795864769f * u2; 
    float s, c; __sincosf(ang, &s, &c);
    z1 = r * c;
    z2 = r * s;
}

template<idx_t NOISE_PERIOD = 10>
[[nodiscard]] __device__ __forceinline__
float normal_from_xy_everyN(
    const idx_t x, 
    const idx_t y, 
    const idx_t STEP
) noexcept {
    const idx_t call_idx = STEP / NOISE_PERIOD;     
    const idx_t pair_idx = call_idx >> 1;           
    const bool use_second = (call_idx & 1) != 0;  

    const seed_t base =
        0x9E3779B9u
        ^ x
        ^ (y * 0x85EBCA6Bu)
        ^ (pair_idx * 0xC2B2AE35u);

    const float u1 = u01(hash32(base));
    const float u2 = u01(hash32(base ^ 0x68BC21EBu));

    float z1, z2; box_muller(u1, u2, z1, z2);
    return use_second ? z2 : z1;
}

#if defined(JET)

[[nodiscard]] __device__ __forceinline__ 
float cubic_sponge(
    const idx_t z
) noexcept {
    const float zn = static_cast<float>(z) * INV_NZ_M1;
    const float s = fminf(fmaxf((zn - Z_START) * INV_SPONGE, 0.0f), 1.0f);
    const float s2 = s * s;
    const float ramp = s2 * s;
    return fmaf(ramp, OMEGA_DELTA, OMEGA_REF);
}

#endif

[[nodiscard]] __device__ __forceinline__ 
float smoothstep(
    float edge0,
    float edge1,
    float x
) noexcept {
    x = __saturatef((x - edge0) / (edge1 - edge0));
    return x * x * (3.0f - 2.0f * x);
}

[[nodiscard]] __device__ __forceinline__ 
float interpolate_rho(
    float phi
) noexcept {
    return fmaf(phi, (RHO_OIL - RHO_WATER), RHO_WATER);
}

template<typename T, T v>
struct integralConstant {
    static constexpr const T value = v;
    using value_type = T;
    using type = integralConstant;

    [[nodiscard]] __device__ __forceinline__ consteval 
    operator value_type() const noexcept {
        return value;
    }

    [[nodiscard]] __device__ __forceinline__ consteval 
    value_type operator()() const noexcept {
        return value;
    }
};

template<const idx_t Start, const idx_t End, typename F>
__device__ __forceinline__ constexpr 
void constexpr_for(
    F&& f
) noexcept {
    if constexpr (Start < End) {
        f(integralConstant<idx_t, Start>());
        if constexpr (Start + 1 < End) {
            constexpr_for<Start + 1, End>(std::forward<F>(f));
        }
    }
}

struct LBMFields {
    float *rho;
    float *phi;
    float *ux;
    float *uy;
    float *uz;
    float *pxx;
    float *pyy;
    float *pzz;
    float *pxy;
    float *pxz;
    float *pyz;
    float *normx;
    float *normy;
    float *normz;
    float *ind;
    float *ffx;
    float *ffy;
    float *ffz;
    pop_t *f;
    float *g;
};

LBMFields fields{};
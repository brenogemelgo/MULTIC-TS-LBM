#pragma once
#include "constants.cuh"

__device__ __forceinline__ 
idx_t global3(
    const idx_t x,
    const idx_t y,
    const idx_t z
) {
    return x + y * NX + z * STRIDE;
}

__device__ __forceinline__ 
idx_t global4(
    const idx_t x,
    const idx_t y,
    const idx_t z,
    const idx_t Q
) {
    return Q * PLANE + global3(x, y, z);
}

__device__ __forceinline__ 
float interpolateRho(
    float phi
) {
    return fmaf(phi, (RHO_OIL - RHO_WATER), RHO_WATER);
}

/*
template<const label_t Start, const label_t End, typename F>
__device__ __forceinline__
constexpr void constexpr_for(
    F&& f
) {
    if constexpr (Start < End) {
        f(integralConstant<label_t, Start>());
        if constexpr (Start + 1 < End) {
            constexpr_for<Start + 1, End>(std::forward<F>(f));
        }
    }
}
*/

#include "../helpers/lbmFunctions.cuh"

#if defined(JET)

__device__ __forceinline__ 
float cubicSponge(
    const idx_t z
) {
    const float zn = static_cast<float>(z) * INV_NZ_M1;
    const float s = fminf(fmaxf((zn - Z_START) * INV_SPONGE, 0.0f), 1.0f);
    const float s2 = s * s;
    const float ramp = s2 * s;
    return fmaf(ramp, OMEGA_DELTA, OMEGA_REF);
}

__device__ __forceinline__ 
float smoothstep(
    float edge0,
    float edge1,
    float x
) {
    x = __saturatef((x - edge0) / (edge1 - edge0));
    return x * x * (3.0f - 2.0f * x);
}

#endif

/*
template<typename T, T v>
struct integralConstant {
    static constexpr const T value = v;
    using value_type = T;
    using type = integralConstant;

    __device__ __forceinline__
    consteval operator value_type()
    const noexcept {
        return value;
    }

    __device__ __forceinline__
    consteval value_type operator()()
    const noexcept {
        return value;
    }
};
*/

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
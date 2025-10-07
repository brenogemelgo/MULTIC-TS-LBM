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
    return Q * PLANE + global3(x,y,z);
}

[[nodiscard]] __device__ __forceinline__ 
float computeFeq(
    const float rho,
    const float ux,
    const float uy,
    const float uz,
    const float uu,
    const idx_t Q
) noexcept{
    const float cu = ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q];
    #if defined(D3Q19)
        return W[Q] * rho * (1.0f - 1.5f*uu + 3.0f*cu + 4.5f*cu*cu) - W[Q];
    #elif defined(D3Q27)
        return W[Q] * rho * (1.0f - 1.5f*uu + 3.0f*cu + 4.5f*cu*cu + 4.5f*cu*cu*cu - 4.5f*uu*cu) - W[Q];
    #endif
}

[[nodiscard]] __device__ __forceinline__ 
float computeGeq(
    const float phi,
    const float ux,
    const float uy,
    const float uz,
    const idx_t Q
) noexcept {
    const float cu = ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q];
    return W_G[Q] * phi * (1.0f + 4.0f * cu);
}

[[nodiscard]] __device__ __forceinline__ 
float computeNeq(
    const float pxx,
    const float pyy,
    const float pzz,
    const float pxy,
    const float pxz,
    const float pyz,
    const float ux,
    const float uy,
    const float uz,
    const idx_t Q
) noexcept {
    #if defined(D3Q19)
        return (W[Q] * 4.5f) * ((CIX[Q] * CIX[Q] - CSSQ) * pxx +
                                (CIY[Q] * CIY[Q] - CSSQ) * pyy +
                                (CIZ[Q] * CIZ[Q] - CSSQ) * pzz +
                                2.0f * (CIX[Q] * CIY[Q] * pxy + CIX[Q] * CIZ[Q] * pxz + CIY[Q] * CIZ[Q] * pyz));
    #elif defined(D3Q27)
        return (W[Q] * 4.5f) *
               ((CIX[Q] * CIX[Q] - CSSQ) * pxx +
                (CIY[Q] * CIY[Q] - CSSQ) * pyy +
                (CIZ[Q] * CIZ[Q] - CSSQ) * pzz +
                2.0f * (CIX[Q] * CIY[Q] * pxy + CIX[Q] * CIZ[Q] * pxz + CIY[Q] * CIZ[Q] * pyz) +
                (CIX[Q] * CIX[Q] * CIX[Q] - 3.0f * CSSQ * CIX[Q]) * (3.0f * ux * pxx) +
                (CIY[Q] * CIY[Q] * CIY[Q] - 3.0f * CSSQ * CIY[Q]) * (3.0f * uy * pyy) +
                (CIZ[Q] * CIZ[Q] * CIZ[Q] - 3.0f * CSSQ * CIZ[Q]) * (3.0f * uz * pzz) +
                3.0f * ((CIX[Q] * CIX[Q] * CIY[Q] - CSSQ * CIY[Q]) * (pxx * uy + 2.0f * ux * pxy) +
                        (CIX[Q] * CIX[Q] * CIZ[Q] - CSSQ * CIZ[Q]) * (pxx * uz + 2.0f * ux * pxz) +
                        (CIX[Q] * CIY[Q] * CIY[Q] - CSSQ * CIX[Q]) * (pxy * uy + 2.0f * ux * pyy) +
                        (CIY[Q] * CIY[Q] * CIZ[Q] - CSSQ * CIZ[Q]) * (pyy * uz + 2.0f * uy * pyz) +
                        (CIX[Q] * CIZ[Q] * CIZ[Q] - CSSQ * CIX[Q]) * (pxz * uz + 2.0f * ux * pzz) +
                        (CIY[Q] * CIZ[Q] * CIZ[Q] - CSSQ * CIY[Q]) * (pyz * uz + 2.0f * uy * pzz)) +
                6.0f * (CIX[Q] * CIY[Q] * CIZ[Q]) * (pxy * uz + ux * pyz + uy * pxz));
    #endif
}

[[nodiscard]] __device__ __forceinline__ 
float computeForce(
    const float feq,
    const float ux,
    const float uy,
    const float uz,
    const float ffx,
    const float ffy,
    const float ffz,
    const float aux,
    const idx_t Q
) noexcept {
    #if defined(D3Q19)
        return feq * ((CIX[Q] - ux) * ffx + 
                      (CIY[Q] - uy) * ffy + 
                      (CIZ[Q] - uz) * ffz) * aux;
    #elif defined(D3Q27)
        const float cu = 3.0f * (ux * CIX[Q] + uy * CIY[Q] + uz * CIZ[Q]);
        return W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * cu * CIX[Q]) * ffx +
                       (3.0f * (CIY[Q] - uy) + 3.0f * cu * CIY[Q]) * ffy +
                       (3.0f * (CIZ[Q] - uz) + 3.0f * cu * CIZ[Q]) * ffz);
    #endif
}

[[nodiscard]] __device__ __forceinline__ 
float cubicSponge(
    const idx_t z
) noexcept {
    const float zn = static_cast<float>(z) * INV_NZ_M1;
    const float s = fminf(fmaxf((zn - Z_START) * INV_SPONGE, 0.0f), 1.0f);
    const float s2 = s * s;
    const float ramp = s2 * s;
    return fmaf(ramp, OMEGA_DELTA, OMEGA_REF);
}

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
float interpolateRho(
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
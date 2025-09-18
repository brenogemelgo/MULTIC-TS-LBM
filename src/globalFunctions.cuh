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
    return Q * PLANE + global3(x,y,z);
}

template<idx_t Q>
__device__ __forceinline__
float computeFeq(
    const float rho, 
    const float ux, 
    const float uy, 
    const float uz
) {
    const float uu = ux*ux + uy*uy + uz*uz;
    const float cu = ux*FDir<Q>::cx + uy*FDir<Q>::cy + uz*FDir<Q>::cz;
    #if defined(D3Q19)
        return FDir<Q>::w * rho * (1.0f - 1.5f*uu + 3.0f*cu + 4.5f*cu*cu) - FDir<Q>::w;
    #elif defined(D3Q27)
        return FDir<Q>::w * rho * (1.0f - 1.5f*uu + 3.0f*cu + 4.5f*cu*cu + 4.5f*cu*cu*cu - 4.5f*uu*cu) - FDir<Q>::w;
    #endif
}

template<idx_t Q>
__device__ __forceinline__ 
float computeGeq(
    const float phi, 
    const float ux, 
    const float uy, 
    const float uz
) {
    const float cu = ux*GDir<Q>::cx + uy*GDir<Q>::cy + uz*GDir<Q>::cz;
    return GDir<Q>::wg * phi * (1.0f + 3.0f * cu);
}

template<idx_t Q>
__device__ __forceinline__
float computeNeq(
    const float PXX, 
    const float PYY, 
    const float PZZ,
    const float PXY, 
    const float PXZ, 
    const float PYZ,
    const float ux,
    const float uy,
    const float uz
) {
    constexpr int cx = FDir<Q>::cx;
    constexpr int cy = FDir<Q>::cy;
    constexpr int cz = FDir<Q>::cz;
    constexpr float w = FDir<Q>::w;

    #if defined(D3Q19)
        return (w * 4.5f) * ((cx*cx - CSSQ) * PXX +
                             (cy*cy - CSSQ) * PYY +
                             (cz*cz - CSSQ) * PZZ +
                            2.0f * (cx*cy*PXY + cx*cz*PXZ + cy*cz*PYZ));
    #elif defined(D3Q27)
        return (w * 4.5f) * 
        ((cx*cx - CSSQ) * PXX +
         (cy*cy - CSSQ) * PYY +
         (cz*cz - CSSQ) * PZZ +
        2.0f * (cx*cy*PXY + cx*cz*PXZ + cy*cz*PYZ) +
        (cx*cx*cx - 3.0f*CSSQ*cx) * (3.0f * ux * PXX) +
        (cy*cy*cy - 3.0f*CSSQ*cy) * (3.0f * uy * PYY) +
        (cz*cz*cz - 3.0f*CSSQ*cz) * (3.0f * uz * PZZ) +
        3.0f * ((cx*cx*cy - CSSQ*cy) * (PXX*uy + 2.0f*ux*PXY) +
                (cx*cx*cz - CSSQ*cz) * (PXX*uz + 2.0f*ux*PXZ) +
                (cx*cy*cy - CSSQ*cx) * (PXY*uy + 2.0f*ux*PYY) +
                (cy*cy*cz - CSSQ*cx) * (PYY*uz + 2.0f*uy*PYZ) +
                (cx*cz*cz - CSSQ*cx) * (PXZ*uz + 2.0f*ux*PZZ) +
                (cy*cz*cz - CSSQ*cy) * (PYZ*uz + 2.0f*uy*PZZ)) +
                6.0f * (cx*cy*cz) * (PXY*uz + ux*PYZ + uy*PXZ));
    #endif
}

template<idx_t Q>
__device__ __forceinline__ 
float computeForce(
    const float coeff, 
    const float feq, 
    const float ux, 
    const float uy, 
    const float uz, 
    const float ffx, 
    const float ffy, 
    const float ffz, 
    const float aux
) {
    constexpr int cx = FDir<Q>::cx;
    constexpr int cy = FDir<Q>::cy;
    constexpr int cz = FDir<Q>::cz;

    #if defined(D3Q19)
        const float val = (cx*ffx + cy*ffy + cz*ffz) - (ux*ffx + uy*ffy + uz*ffz);
        return coeff * feq * val * aux;
    #elif defined(D3Q27)
        constexpr float w = FDir<Q>::w;
        const float cu = ux*cx + uy*cy + uz*cz;             
        const float tfx = (3.0f*(cx - ux) + 9.0f*cu*cx) * ffx;
        const float tfy = (3.0f*(cy - uy) + 9.0f*cu*cy) * ffy;
        const float tfz = (3.0f*(cz - uz) + 9.0f*cu*cz) * ffz;
        return coeff * w * (tfx + tfy + tfz);
    #endif
}

__device__ __forceinline__ 
float computeEquilibria(
    const float density, 
    const float ux, 
    const float uy, 
    const float uz, 
    const idx_t Q
) {
    const float uu = 1.5f * (ux*ux + uy*uy + uz*uz);
    #if defined(D3Q19)
        const float eqbase = density * (-uu + (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * (3.0f + 4.5f*(ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q])));
    #elif defined(D3Q27)
        const float eqbase = density * (-uu + (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * (3.0f + (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * (4.5f + 4.5f*(ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q])) - 3.0f*uu));
    #endif
    return W[Q] * (density + eqbase) - W[Q];
}

__device__ __forceinline__ 
float computeTruncatedEquilibria(
    const float density, 
    const float ux, 
    const float uy, 
    const float uz, 
    const idx_t Q
) {
    const float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
    return W_G[Q] * density * (1.0f + cu);
}

#if defined(JET)

    __device__ __forceinline__ 
    float cubicSponge(
        const idx_t z
    ) {
        const float zn = static_cast<float>(z) * INV_NZ_M1;
        const float s  = fminf(fmaxf((zn - Z_START) * INV_SPONGE, 0.0f), 1.0f);
        const float s2 = s * s;
        const float ramp = s2 * s;
        return fmaf(ramp, OMEGA_DELTA, OMEGA_REF);
    }

#endif

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
LBMFields lbm;

#if defined(D_FIELDS)
struct DerivedFields {
    float *vorticity_mag;
    float *velocity_mag;
};
DerivedFields dfields;
#endif 
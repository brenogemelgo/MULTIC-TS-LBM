#pragma once
#include "constants.cuh"
 
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

__device__ __forceinline__ 
float computeEquilibria(
    const float rho, 
    const float ux, 
    const float uy, 
    const float uz, 
    const idx_t Q
) {
    #if defined(D3Q19)
        return W[Q] * rho * (
              1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q])
            + 4.5f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q])
        ) - W[Q];
    #elif defined(D3Q27)
        return W[Q] * rho * (
              1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q])
            + 4.5f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q])
            + 4.5f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q])
        ) - W[Q];
    #endif
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

__device__ __forceinline__ 
float computeNonEquilibria(
    const float PXX, 
    const float PYY, 
    const float PZZ, 
    const float PXY, 
    const float PXZ, 
    const float PYZ,  
    const float ux, 
    const float uy, 
    const float uz, 
    const idx_t Q
) {
    #if defined(D3Q19)
        return (W[Q] * 4.5f) * ((CIX[Q]*CIX[Q] - CSSQ) * PXX + 
                                (CIY[Q]*CIY[Q] - CSSQ) * PYY + 
                                (CIZ[Q]*CIZ[Q] - CSSQ) * PZZ + 
                                2.0f * CIX[Q] * CIY[Q] * PXY + 
                                2.0f * CIX[Q] * CIZ[Q] * PXZ +
                                2.0f * CIY[Q] * CIZ[Q] * PYZ);
    #elif defined(D3Q27)
        return (W[Q] * 4.5f) * (
            // 2nd order
            (CIX[Q]*CIX[Q] - CSSQ) * PXX +
            (CIY[Q]*CIY[Q] - CSSQ) * PYY +
            (CIZ[Q]*CIZ[Q] - CSSQ) * PZZ +
            2.0f * CIX[Q] * CIY[Q] * PXY +
            2.0f * CIX[Q] * CIZ[Q] * PXZ +
            2.0f * CIY[Q] * CIZ[Q] * PYZ +

            // 3rd order
            (CIX[Q]*CIX[Q]*CIX[Q] - 3.0f*CSSQ*CIX[Q]) * (3.0f * ux * PXX) +
            (CIY[Q]*CIY[Q]*CIY[Q] - 3.0f*CSSQ*CIY[Q]) * (3.0f * uy * PYY) +
            (CIZ[Q]*CIZ[Q]*CIZ[Q] - 3.0f*CSSQ*CIZ[Q]) * (3.0f * uz * PZZ) +
            3.0f * (
                (CIX[Q]*CIX[Q]*CIY[Q] - CSSQ*CIY[Q]) * (PXX*uy + 2.0f*ux*PXY) +
                (CIX[Q]*CIX[Q]*CIZ[Q] - CSSQ*CIZ[Q]) * (PXX*uz + 2.0f*ux*PXZ) +
                (CIX[Q]*CIY[Q]*CIY[Q] - CSSQ*CIX[Q]) * (PXY*uy + 2.0f*ux*PYY) +
                (CIY[Q]*CIY[Q]*CIZ[Q] - CSSQ*CIZ[Q]) * (PYY*uz + 2.0f*uy*PYZ) +
                (CIX[Q]*CIZ[Q]*CIZ[Q] - CSSQ*CIX[Q]) * (PXZ*uz + 2.0f*ux*PZZ) +
                (CIY[Q]*CIZ[Q]*CIZ[Q] - CSSQ*CIY[Q]) * (PYZ*uz + 2.0f*uy*PZZ)
            ) +
            6.0f * (CIX[Q]*CIY[Q]*CIZ[Q]) * (PXY*uz + ux*PYZ + uy*PXZ)
        );
    #endif 
}

__device__ __forceinline__ 
float computeForceTerm(
    const float coeff, 
    const float feq, 
    const float ux, 
    const float uy, 
    const float uz, 
    const float ffx, 
    const float ffy, 
    const float ffz, 
    const float aux, 
    const idx_t Q
) {
    #if defined(D3Q19)
        return coeff * feq * ((CIX[Q] - ux) * ffx +
                              (CIY[Q] - uy) * ffy +
                              (CIZ[Q] - uz) * ffz) * aux;
    #elif defined(D3Q27)
        return coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                               (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                               (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
    #endif 
}

template<typename T, idx_t... Qs>
__device__ __forceinline__
void copyDirs(
    T* __restrict__ arr, 
    const idx_t dst, 
    const idx_t src
) {
    ((arr[Qs * PLANE + dst] = arr[Qs * PLANE + src]), ...);
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
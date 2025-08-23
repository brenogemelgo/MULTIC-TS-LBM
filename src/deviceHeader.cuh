#pragma once
#include "constants.cuh"

extern __constant__ float W[FLINKS];
extern __constant__ float W_G[GLINKS];

extern __constant__ ci_t CIX[FLINKS], CIY[FLINKS], CIZ[FLINKS];

#ifdef PERTURBATION
extern __constant__ float PERTURBATION_DATA[200];
#endif
 
struct LBMFields {
    float *rho, *phi;
    float *ux, *uy, *uz;
    float *pxx, *pyy, *pzz, *pxy, *pxz, *pyz;
    float *ind, *normx, *normy, *normz;
    float *ffx, *ffy, *ffz;
    pop_t *f; 
    float *g; 
};
extern LBMFields lbm;

#ifdef D_FIELDS
struct DerivedFields {
    float *vorticity_mag;
    float *velocity_mag;
    float *pressure;
};
extern DerivedFields dfields;
#endif // D_FIELDS

__device__ __forceinline__ idx_t global3(const int x, const int y, const int z) {
    return x + y * NX + z * STRIDE;
}

__device__ __forceinline__ idx_t global4(const int x, const int y, const int z, const int Q) {
    return Q * PLANE + global3(x,y,z);
}

__device__ __forceinline__ idx_t shared3(const int tx, const int ty, const int tz) {
    return tx + ty * TILE_X + tz * TILE_X * TILE_Y;
}

__device__ __forceinline__ float smoothstep(float edge0, float edge1, float x) {
    x = __saturatef((x - edge0) / (edge1 - edge0));
    return x * x * (3.0f - 2.0f * x);
}

__device__ __forceinline__ float omegaSponge(int z) {
    float zn = __int2float_rn(z) * INV_NZ_M1;
    float s  = fminf(fmaxf((zn - Z_START) * INV_SPONGE, 0.0f), 1.0f);
    float s2 = s * s;
    float ramp = s2 * s;
    return fmaf(ramp, OMEGA_DELTA, OMEGA);
}

__device__ __forceinline__ float computeEquilibria(const float density, const float ux, const float uy, const float uz, const int Q) {
    const float uu = 1.5f * (ux*ux + uy*uy + uz*uz);
    #ifdef D3Q19
        const float eqbase = density * (-uu + (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * (3.0f + 4.5f*(ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q])));
    #elif defined(D3Q27)
        const float eqbase = density * (-uu + (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * (3.0f + (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * (4.5f + 4.5f*(ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q])) - 3.0f*uu));
    #endif // VELOCITY_SET 
    return W[Q] * (density + eqbase) - W[Q];
}

__device__ __forceinline__ float computeTruncatedEquilibria(const float density, const float ux, const float uy, const float uz, const int Q) {
    const float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
    return W_G[Q] * density * (1.0f + cu);
}

__device__ __forceinline__ float computeNonEquilibria(const float PXX, const float PYY, const float PZZ, const float PXY, const float PXZ, const float PYZ,  const float ux, const float uy, const float uz, const int Q) {
    #ifdef D3Q19
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
    #endif // VELOCITY_SET 
}

__device__ __forceinline__ float computeForceTerm(const float coeff, const float feq, const float ux, const float uy, const float uz, const float ffx, const float ffy, const float ffz, const float aux, const int Q) {
    #ifdef D3Q19
        return coeff * feq * ((CIX[Q] - ux) * ffx +
                              (CIY[Q] - uy) * ffy +
                              (CIZ[Q] - uz) * ffz) * aux;
    #elif defined(D3Q27)
        const float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
        return coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * cu * CIX[Q] ) * ffx +
                               (3.0f * (CIY[Q] - uy) + 3.0f * cu * CIY[Q] ) * ffy +
                               (3.0f * (CIZ[Q] - uz) + 3.0f * cu * CIZ[Q] ) * ffz);
    #endif // VELOCITY_SET 
}

template<typename T, int... Qs>
__device__ __forceinline__
void copyDirs(T* __restrict__ arr, idx_t dst, idx_t src) {
    ((arr[Qs*PLANE+dst] = arr[Qs*PLANE+src]), ...);
}

// usage: idx_t nidx = idx3 + nbrOff(q);
__device__ __forceinline__ idx_t nbrOff(int Q) {
    return idx_t(CIX[Q]) + idx_t(CIY[Q]) * idx_t(NX) + idx_t(CIZ[Q]) * idx_t(STRIDE);
}

__device__ __forceinline__ 
float computeFeq(
    const float rho, 
    const float ux, 
    const float uy, 
    const float uz, 
    const float uu,
    const idx_t Q
) {
    const float cu = ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q];
    #if defined(D3Q19)
        return W[Q] * rho * (1.0f - 1.5f*uu + 3.0f*cu + 4.5f*cu*cu) - W[Q];
    #elif defined(D3Q27)
        return W[Q] * rho * (1.0f - 1.5f*uu + 3.0f*cu + 4.5f*cu*cu + 4.5f*cu*cu*cu - 4.5f*uu*cu) - W[Q];
    #endif
}

__device__ __forceinline__ 
float computeGeq(
    const float phi, 
    const float ux, 
    const float uy, 
    const float uz, 
    const idx_t Q
) {
    const float cu = ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q];
    return W_G[Q] * phi * (1.0f + 3.0f * cu);
}

__device__ __forceinline__ 
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
) {
    #if defined(D3Q19)
        return (W[Q] * 4.5f) * ((CIX[Q]*CIX[Q] - CSSQ) * pxx + 
                                (CIY[Q]*CIY[Q] - CSSQ) * pyy + 
                                (CIZ[Q]*CIZ[Q] - CSSQ) * pzz + 
                                2.0f * (CIX[Q]*CIY[Q]*pxy + CIX[Q]*CIZ[Q]*pxz + CIY[Q]*CIZ[Q]*pyz));
    #elif defined(D3Q27)
        return (W[Q] * 4.5f) * 
        ((CIX[Q]*CIX[Q] - CSSQ) * pxx +
         (CIY[Q]*CIY[Q] - CSSQ) * pyy +
         (CIZ[Q]*CIZ[Q] - CSSQ) * pzz +
        2.0f * (CIX[Q]*CIY[Q]*pxy + CIX[Q]*CIZ[Q]*pxz + CIY[Q]*CIZ[Q]*pyz) +
        (CIX[Q]*CIX[Q]*CIX[Q] - 3.0f*CSSQ*CIX[Q]) * (3.0f * ux * pxx) +
        (CIY[Q]*CIY[Q]*CIY[Q] - 3.0f*CSSQ*CIY[Q]) * (3.0f * uy * pyy) +
        (CIZ[Q]*CIZ[Q]*CIZ[Q] - 3.0f*CSSQ*CIZ[Q]) * (3.0f * uz * pzz) +
        3.0f * ((CIX[Q]*CIX[Q]*CIY[Q] - CSSQ*CIY[Q]) * (pxx*uy + 2.0f*ux*pxy) +
                (CIX[Q]*CIX[Q]*CIZ[Q] - CSSQ*CIZ[Q]) * (pxx*uz + 2.0f*ux*pxz) +
                (CIX[Q]*CIY[Q]*CIY[Q] - CSSQ*CIX[Q]) * (pxy*uy + 2.0f*ux*pyy) +
                (CIY[Q]*CIY[Q]*CIZ[Q] - CSSQ*CIZ[Q]) * (pyy*uz + 2.0f*uy*pyz) +
                (CIX[Q]*CIZ[Q]*CIZ[Q] - CSSQ*CIX[Q]) * (pxz*uz + 2.0f*ux*pzz) +
                (CIY[Q]*CIZ[Q]*CIZ[Q] - CSSQ*CIY[Q]) * (pyz*uz + 2.0f*uy*pzz)) +
                6.0f * (CIX[Q]*CIY[Q]*CIZ[Q]) * (pxy*uz + ux*pyz + uy*pxz));
    #endif 
}

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
    const float aux, 
    const idx_t Q
) {
    #if defined(D3Q19)
        return coeff * feq * ((CIX[Q] - ux) * ffx +
                              (CIY[Q] - uy) * ffy +
                              (CIZ[Q] - uz) * ffz) * aux;
    #elif defined(D3Q27)
        const float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
        return coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * cu * CIX[Q] ) * ffx +
                               (3.0f * (CIY[Q] - uy) + 3.0f * cu * CIY[Q] ) * ffy +
                               (3.0f * (CIZ[Q] - uz) + 3.0f * cu * CIZ[Q] ) * ffz);
    #endif 
}
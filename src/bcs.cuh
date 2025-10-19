#pragma once

#if defined(JET)

__global__ 
void applyInflow(
    LBMFields d,
    const idx_t STEP
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= NX || y >= NY) return;

    const float dx = static_cast<float>(x) - CENTER_X;
    const float dy = static_cast<float>(y) - CENTER_Y;
    const float r2 = dx * dx + dy * dy;
    if (r2 > R2) return;

    const idx_t idx3_bnd = global3(x, y, 0);
    //const idx_t idx3_zp1 = global3(x, y, 1);

    //d.rho[idx3_bnd] = d.rho[idx3_zp1];

    const float uz = d.uz[idx3_bnd];

    const float P = 1.0f + 3.0f * uz + 3.0f * uz * uz;
    constexpr_for<0, FLINKS>([&] __device__ (auto I) {
        constexpr idx_t Q = decltype(I)::value;

        if constexpr (FDir<Q>::cz == 1) { 
            const idx_t xx = x + static_cast<idx_t>(FDir<Q>::cx);
            const idx_t yy = y + static_cast<idx_t>(FDir<Q>::cy);

            idx_t fluidNode = global3(xx, yy, 1);

            constexpr float w  = FDir<Q>::w;
            constexpr float cx = static_cast<float>(FDir<Q>::cx);
            constexpr float cy = static_cast<float>(FDir<Q>::cy);
            constexpr float cz = static_cast<float>(FDir<Q>::cz);

            const float feq = w * d.rho[fluidNode] * P - w; 

            #if defined(D3Q19)
                const float fneq = (w * 4.5f) *
                    ((cx * cx - CSSQ) * d.pxx[fluidNode] +
                     (cy * cy - CSSQ) * d.pyy[fluidNode] +
                     (cz * cz - CSSQ) * d.pzz[fluidNode] +
                      2.0f * (cx * cy * d.pxy[fluidNode] + 
                              cx * cz * d.pxz[fluidNode] + 
                              cy * cz * d.pyz[fluidNode]));
            #elif defined(D3Q27)
                const float fneq = (w * 4.5f) *
                    ((cx * cx - CSSQ) * d.pxx[fluidNode] +
                     (cy * cy - CSSQ) * d.pyy[fluidNode] +
                     (cz * cz - CSSQ) * d.pzz[fluidNode] +
                      2.0f * (cx * cy * d.pxy[fluidNode] + 
                              cx * cz * d.pxz[fluidNode] + 
                              cy * cz * d.pyz[fluidNode]) +
                     (cx * cx * cx - 3.0f * CSSQ * cx) * (3.0f * d.ux[fluidNode] * d.pxx[fluidNode]) +
                     (cy * cy * cy - 3.0f * CSSQ * cy) * (3.0f * d.uy[fluidNode] * d.pyy[fluidNode]) +
                     (cz * cz * cz - 3.0f * CSSQ * cz) * (3.0f * d.uz[fluidNode] * d.pzz[fluidNode]) +
                      3.0f * ((cx * cx * cy - CSSQ * cy) * (d.pxx[fluidNode] * d.uy[fluidNode] + 2.0f * d.ux[fluidNode] * d.pxy[fluidNode]) +
                              (cx * cx * cz - CSSQ * cz) * (d.pxx[fluidNode] * d.uz[fluidNode] + 2.0f * d.ux[fluidNode] * d.pxz[fluidNode]) +
                              (cx * cy * cy - CSSQ * cx) * (d.pxy[fluidNode] * d.uy[fluidNode] + 2.0f * d.ux[fluidNode] * d.pyy[fluidNode]) +
                              (cy * cy * cz - CSSQ * cz) * (d.pyy[fluidNode] * d.uz[fluidNode] + 2.0f * d.uy[fluidNode] * d.pyz[fluidNode]) +
                              (cx * cz * cz - CSSQ * cx) * (d.pxz[fluidNode] * d.uz[fluidNode] + 2.0f * d.ux[fluidNode] * d.pzz[fluidNode]) +
                              (cy * cz * cz - CSSQ * cy) * (d.pyz[fluidNode] * d.uz[fluidNode] + 2.0f * d.uy[fluidNode] * d.pzz[fluidNode])) +
                            6.0f * (cx * cy * cz) * (d.ux[fluidNode] * d.pyz[fluidNode] + 
                                                     d.uy[fluidNode] * d.pxz[fluidNode] +
                                                     d.uz[fluidNode] * d.pxy[fluidNode]));
            #endif

            d.f[Q * PLANE + fluidNode] = to_pop(feq + OMCO_ZMIN * fneq);
        }
    });

    d.g[5 * PLANE + global3(x, y, 1)] = GDir<5>::wg * d.phi[idx3_bnd] * (1.0f + 4.0f * uz);
}

__global__ 
void applyOutflow(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= NX || y >= NY) return;

    const idx_t idx3_bnd = global3(x, y, NZ - 1);
    const idx_t idx3_zm1 = global3(x, y, NZ - 2);

    d.rho[idx3_bnd] = d.rho[idx3_zm1];
    d.phi[idx3_bnd] = d.phi[idx3_zm1];
    d.ux[idx3_bnd] = d.ux[idx3_zm1];
    d.uy[idx3_bnd] = d.uy[idx3_zm1];
    d.uz[idx3_bnd] = d.uz[idx3_zm1];
 
    const float ux = d.ux[idx3_zm1];
    const float uy = d.uy[idx3_zm1];
    const float uz = d.uz[idx3_zm1];
    const float uu = 1.5f * (ux*ux + uy*uy + uz*uz);

    constexpr_for<0, FLINKS>([&] __device__ (auto I) {
        constexpr idx_t Q = decltype(I)::value;

        if constexpr (FDir<Q>::cz == -1) { 
            const idx_t xx = x + static_cast<idx_t>(FDir<Q>::cx);
            const idx_t yy = y + static_cast<idx_t>(FDir<Q>::cy);

            idx_t fluidNode = global3(xx, yy, NZ-2);

            constexpr float w  = FDir<Q>::w;
            constexpr float cx = static_cast<float>(FDir<Q>::cx);
            constexpr float cy = static_cast<float>(FDir<Q>::cy);
            constexpr float cz = static_cast<float>(FDir<Q>::cz);

            const float cu = 3.0f * (ux*cx + uy*cy + uz*cz);    

            #if defined(D3Q19)
                const float feq = w * d.rho[fluidNode] * (1.0f - uu + cu + 0.5f*cu*cu) - w; 
            #elif defined(D3Q27)
                const float feq = w * d.rho[fluidNode] * (1.0f - uu + cu + 0.5f*cu*cu + OOS*cu*cu*cu - uu*cu) - w; 
            #endif

            #if defined(D3Q19)
                const float fneq = (w * 4.5f) *
                    ((cx * cx - CSSQ) * d.pxx[fluidNode] +
                     (cy * cy - CSSQ) * d.pyy[fluidNode] +
                     (cz * cz - CSSQ) * d.pzz[fluidNode] +
                      2.0f * (cx * cy * d.pxy[fluidNode] + 
                              cx * cz * d.pxz[fluidNode] + 
                              cy * cz * d.pyz[fluidNode]));
            #elif defined(D3Q27)
                const float fneq = (w * 4.5f) *
                    ((cx * cx - CSSQ) * d.pxx[fluidNode] +
                     (cy * cy - CSSQ) * d.pyy[fluidNode] +
                     (cz * cz - CSSQ) * d.pzz[fluidNode] +
                      2.0f * (cx * cy * d.pxy[fluidNode] + 
                              cx * cz * d.pxz[fluidNode] + 
                              cy * cz * d.pyz[fluidNode]) +
                     (cx * cx * cx - 3.0f * CSSQ * cx) * (3.0f * d.ux[fluidNode] * d.pxx[fluidNode]) +
                     (cy * cy * cy - 3.0f * CSSQ * cy) * (3.0f * d.uy[fluidNode] * d.pyy[fluidNode]) +
                     (cz * cz * cz - 3.0f * CSSQ * cz) * (3.0f * d.uz[fluidNode] * d.pzz[fluidNode]) +
                      3.0f * ((cx * cx * cy - CSSQ * cy) * (d.pxx[fluidNode] * d.uy[fluidNode] + 2.0f * d.ux[fluidNode] * d.pxy[fluidNode]) +
                              (cx * cx * cz - CSSQ * cz) * (d.pxx[fluidNode] * d.uz[fluidNode] + 2.0f * d.ux[fluidNode] * d.pxz[fluidNode]) +
                              (cx * cy * cy - CSSQ * cx) * (d.pxy[fluidNode] * d.uy[fluidNode] + 2.0f * d.ux[fluidNode] * d.pyy[fluidNode]) +
                              (cy * cy * cz - CSSQ * cz) * (d.pyy[fluidNode] * d.uz[fluidNode] + 2.0f * d.uy[fluidNode] * d.pyz[fluidNode]) +
                              (cx * cz * cz - CSSQ * cx) * (d.pxz[fluidNode] * d.uz[fluidNode] + 2.0f * d.ux[fluidNode] * d.pzz[fluidNode]) +
                              (cy * cz * cz - CSSQ * cy) * (d.pyz[fluidNode] * d.uz[fluidNode] + 2.0f * d.uy[fluidNode] * d.pzz[fluidNode])) +
                            6.0f * (cx * cy * cz) * (d.ux[fluidNode] * d.pyz[fluidNode] + 
                                                     d.uy[fluidNode] * d.pxz[fluidNode] +
                                                     d.uz[fluidNode] * d.pxy[fluidNode]));
            #endif

            d.f[Q * PLANE + fluidNode] = to_pop(feq + OMCO_ZMAX * fneq);
        }
    });

    d.g[6 * PLANE + idx3_zm1] = GDir<6>::wg * d.phi[idx3_zm1] * (1.0f - 4.0f * U_REF);
}

__global__ 
void periodicX(
    LBMFields d
) {
    const idx_t y = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t z = threadIdx.y + blockIdx.y * blockDim.y;

    if (y <= 0 || y >= NY-1 || z <= 0 || z >= NZ-1) return;

    const idx_t bL = global3(1, y, z);
    const idx_t bR = global3(NX - 2, y, z);

    // positive x contributions
    d.f[     PLANE + bL] = d.f[     PLANE + bR];
    d.f[7  * PLANE + bL] = d.f[7  * PLANE + bR];
    d.f[9  * PLANE + bL] = d.f[9  * PLANE + bR];
    d.f[13 * PLANE + bL] = d.f[13 * PLANE + bR];
    d.f[15 * PLANE + bL] = d.f[15 * PLANE + bR];
    #if defined(D3Q27)
    d.f[19 * PLANE + bL] = d.f[19 * PLANE + bR];
    d.f[21 * PLANE + bL] = d.f[21 * PLANE + bR];
    d.f[23 * PLANE + bL] = d.f[23 * PLANE + bR];
    d.f[26 * PLANE + bL] = d.f[26 * PLANE + bR];
    #endif
    d.g[     PLANE + bL] = d.g[     PLANE + bR];

    // negative x contributions
    d.f[2  * PLANE + bR] = d.f[2  * PLANE + bL];
    d.f[8  * PLANE + bR] = d.f[8  * PLANE + bL];
    d.f[10 * PLANE + bR] = d.f[10 * PLANE + bL];
    d.f[14 * PLANE + bR] = d.f[14 * PLANE + bL];
    d.f[16 * PLANE + bR] = d.f[16 * PLANE + bL];
    #if defined(D3Q27)
    d.f[20 * PLANE + bR] = d.f[20 * PLANE + bL];
    d.f[22 * PLANE + bR] = d.f[22 * PLANE + bL];
    d.f[24 * PLANE + bR] = d.f[24 * PLANE + bL];
    d.f[25 * PLANE + bR] = d.f[25 * PLANE + bL];
    #endif
    d.g[2  * PLANE + bR] = d.g[2  * PLANE + bL];

    const idx_t gL = global3(0, y, z);
    const idx_t gR = global3(NX - 1, y, z);

    // ghost cells
    d.phi[gL] = d.phi[bR];
    d.phi[gR] = d.phi[bL];

    d.rho[gL] = d.rho[bR];
    d.rho[gR] = d.rho[bL];

    d.ux[gL] = d.ux[bR];
    d.ux[gR] = d.ux[bL];

    d.uy[gL] = d.uy[bR];
    d.uy[gR] = d.uy[bL];

    d.uz[gL] = d.uz[bR];
    d.uz[gR] = d.uz[bL];
}

__global__ 
void periodicY(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t z = threadIdx.y + blockIdx.y * blockDim.y;

    if (x <= 0 || x >= NX-1 || z <= 0 || z >= NZ-1) return;

    const idx_t bB = global3(x, 1, z);
    const idx_t bT = global3(x, NY - 2, z);

    // positive y contributions
    d.f[3  * PLANE + bB] = d.f[3  * PLANE + bT];
    d.f[7  * PLANE + bB] = d.f[7  * PLANE + bT];
    d.f[11 * PLANE + bB] = d.f[11 * PLANE + bT];
    d.f[14 * PLANE + bB] = d.f[14 * PLANE + bT];
    d.f[17 * PLANE + bB] = d.f[17 * PLANE + bT];
    #if defined(D3Q27)
    d.f[19 * PLANE + bB] = d.f[19 * PLANE + bT];
    d.f[21 * PLANE + bB] = d.f[21 * PLANE + bT];
    d.f[24 * PLANE + bB] = d.f[24 * PLANE + bT];
    d.f[25 * PLANE + bB] = d.f[25 * PLANE + bT];
    #endif
    d.g[3  * PLANE + bB] = d.g[3  * PLANE + bT];

    // negative y contributions
    d.f[4  * PLANE + bT] = d.f[4  * PLANE + bB];
    d.f[8  * PLANE + bT] = d.f[8  * PLANE + bB];
    d.f[12 * PLANE + bT] = d.f[12 * PLANE + bB];
    d.f[13 * PLANE + bT] = d.f[13 * PLANE + bB];
    d.f[18 * PLANE + bT] = d.f[18 * PLANE + bB];
    #if defined(D3Q27)
    d.f[20 * PLANE + bT] = d.f[20 * PLANE + bB];
    d.f[22 * PLANE + bT] = d.f[22 * PLANE + bB];
    d.f[23 * PLANE + bT] = d.f[23 * PLANE + bB];
    d.f[26 * PLANE + bT] = d.f[26 * PLANE + bB];
    #endif
    d.g[4   * PLANE + bT] = d.g[4   * PLANE + bB];

    const idx_t gB = global3(x, 0, z);
    const idx_t gT = global3(x, NY - 1, z);

    // ghost cells
    d.phi[gB] = d.phi[bT];
    d.phi[gT] = d.phi[bB];

    d.rho[gB] = d.rho[bT];
    d.rho[gT] = d.rho[bB];

    d.ux[gB] = d.ux[bT];
    d.ux[gT] = d.ux[bB];

    d.uy[gB] = d.uy[bT];
    d.uy[gT] = d.uy[bB];

    d.uz[gB] = d.uz[bT];
    d.uz[gT] = d.uz[bB];
}

#elif defined(DROPLET)

#endif

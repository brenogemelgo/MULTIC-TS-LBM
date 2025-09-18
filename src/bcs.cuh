#pragma once

#if defined(JET)

__global__ 
void applyInflow(
    LBMFields d, 
    const int STEP
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    // const idx_t z = 0;

    if (x >= NX || y >= NY) return;

    const float dx = static_cast<float>(x) - CENTER_X;
    const float dy = static_cast<float>(y) - CENTER_Y;
    const float r2 = dx*dx + dy*dy;
    if (r2 > RR) return;

    const idx_t idx3_in = global3(x,y,0);
    const float uzIn = 
    #if defined(PERTURBATION)
        U_REF * (1.0f + PERTURBATION_DATA[(STEP / MACRO_SAVE) % 200] * 10.0f);
    #else
        U_REF;
    #endif 

    idx_t fluidNode = global3(x,y,1);
    float feq  = computeFeq<5>(d.rho[fluidNode],0.0f,0.0f,uzIn);
    float fneq = computeNeq<5>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                            d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                            d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[5 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMIN * fneq);

    feq = computeGeq<5>(d.phi[idx3_in],0.0f,0.0f,uzIn);
    d.g[5 * PLANE + fluidNode] = feq;

    fluidNode = global3(x+1,y,1);
    feq  = computeFeq<9>(d.rho[fluidNode],0.0f,0.0f,uzIn);
    fneq = computeNeq<9>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[9 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMIN * fneq);

    fluidNode = global3(x,y+1,1);
    feq  = computeFeq<11>(d.rho[fluidNode],0.0f,0.0f,uzIn);
    fneq = computeNeq<11>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[11 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMIN * fneq);

    fluidNode = global3(x-1,y,1);
    feq  = computeFeq<16>(d.rho[fluidNode],0.0f,0.0f,uzIn);
    fneq = computeNeq<16>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[16 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMIN * fneq);

    fluidNode = global3(x,y-1,1);
    feq  = computeFeq<18>(d.rho[fluidNode],0.0f,0.0f,uzIn);
    fneq = computeNeq<18>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[18 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMIN * fneq);

    #if defined(D3Q27)
    fluidNode = global3(x+1,y+1,1);
    feq  = computeFeq<19>(d.rho[fluidNode],0.0f,0.0f,uzIn);
    fneq = computeNeq<19>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[19 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMIN * fneq);

    fluidNode = global3(x-1,y-1,1);
    feq  = computeFeq<22>(d.rho[fluidNode],0.0f,0.0f,uzIn);
    fneq = computeNeq<22>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[22 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMIN * fneq);

    fluidNode = global3(x+1,y-1,1);
    feq  = computeFeq<23>(d.rho[fluidNode],0.0f,0.0f,uzIn);
    fneq = computeNeq<23>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[23 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMIN * fneq);

    fluidNode = global3(x-1,y+1,1);
    feq  = computeFeq<25>(d.rho[fluidNode],0.0f,0.0f,uzIn);
    fneq = computeNeq<25>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[25 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMIN * fneq);
    #endif 
}

__global__ 
void applyOutflow(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    //const idx_t z = NZ-1;

    if (x >= NX || y >= NY) return;

    const idx_t idx3_zm1 = global3(x,y,NZ-2);
    d.phi[global3(x,y,NZ-1)] = d.phi[idx3_zm1];

    const float uxOut = d.ux[idx3_zm1];
    const float uyOut = d.uy[idx3_zm1];
    const float uzOut = d.uz[idx3_zm1];

    float feq  = computeFeq<6>(d.rho[idx3_zm1],uxOut,uyOut,uzOut);
    float fneq = computeNeq<6>(d.pxx[idx3_zm1],d.pyy[idx3_zm1],d.pzz[idx3_zm1],
                            d.pxy[idx3_zm1],d.pxz[idx3_zm1],d.pyz[idx3_zm1],
                            d.ux[idx3_zm1],d.uy[idx3_zm1],d.uz[idx3_zm1]);
    d.f[6 * PLANE + idx3_zm1] = to_pop(feq + OMCO_ZMAX * fneq);

    feq = computeGeq<6>(d.phi[idx3_zm1],uxOut,uyOut,uzOut);
    d.g[6 * PLANE + idx3_zm1] = feq;

    idx_t fluidNode = global3(x-1,y,NZ-2);
    feq  = computeFeq<10>(d.rho[fluidNode],uxOut,uyOut,uzOut);
    fneq = computeNeq<10>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[10 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMAX * fneq);

    fluidNode = global3(x,y-1,NZ-2);
    feq  = computeFeq<12>(d.rho[fluidNode],uxOut,uyOut,uzOut);
    fneq = computeNeq<12>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[12 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMAX * fneq);

    fluidNode = global3(x+1,y,NZ-2);
    feq  = computeFeq<15>(d.rho[fluidNode],uxOut,uyOut,uzOut);
    fneq = computeNeq<15>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[15 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMAX * fneq);

    fluidNode = global3(x,y+1,NZ-2);
    feq  = computeFeq<17>(d.rho[fluidNode],uxOut,uyOut,uzOut);
    fneq = computeNeq<17>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[17 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMAX * fneq);

    #if defined(D3Q27)
    fluidNode = global3(x-1,y-1,NZ-2);
    feq  = computeFeq<20>(d.rho[fluidNode],uxOut,uyOut,uzOut);
    fneq = computeNeq<20>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[20 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMAX * fneq);
    
    fluidNode = global3(x+1,y+1,NZ-2);
    feq  = computeFeq<21>(d.rho[fluidNode],uxOut,uyOut,uzOut);
    fneq = computeNeq<21>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[21 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMAX * fneq);

    fluidNode = global3(x-1,y+1,NZ-2);
    feq  = computeFeq<24>(d.rho[fluidNode],uxOut,uyOut,uzOut);
    fneq = computeNeq<24>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[24 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMAX * fneq);

    fluidNode = global3(x+1,y-1,NZ-2);
    feq  = computeFeq<26>(d.rho[fluidNode],uxOut,uyOut,uzOut);
    fneq = computeNeq<26>(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode]);
    d.f[26 * PLANE + fluidNode] = to_pop(feq + OMCO_ZMAX * fneq);
    #endif 
}

__global__ 
void periodicX(
    LBMFields d
) {
    const idx_t y = threadIdx.x + blockIdx.x * blockDim.x;              
    const idx_t z = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (y <= 0 || y >= NY-1 || z <= 0 || z >= NZ-1) return;

    const idx_t bL = global3(1,y,z);
    const idx_t bR = global3(NX-2,y,z);

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
    d.g[PLANE + bL] = d.g[PLANE + bR];

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
    d.g[2 * PLANE + bR] = d.g[2 * PLANE + bL];

    // ghost cells
    d.phi[global3(0,y,z)]    = d.phi[bR];
    d.phi[global3(NX-1,y,z)] = d.phi[bL];
}

__global__ 
void periodicY(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t z = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x <= 0 || x >= NX-1 || z <= 0 || z >= NZ-1) return;

    const idx_t bB = global3(x,1,z);
    const idx_t bT = global3(x,NY-2,z);

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
    d.g[4  * PLANE + bT] = d.g[4  * PLANE + bB];

    // ghost cells
    d.phi[global3(x,0,z)]    = d.phi[bT];
    d.phi[global3(x,NY-1,z)] = d.phi[bB];
}

#elif defined(DROPLET)

// still undefined

#endif 


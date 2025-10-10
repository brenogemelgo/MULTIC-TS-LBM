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
    const float r2 = dx * dx + dy * dy;
    if (r2 > R2) return;

    const idx_t idx3_in = global3(x,y,0);
    
    const float uzIn =
    #if defined(PERTURBATION)
    d.uz[idx3_in] * (1.0f + PERTURBATION_DATA[(STEP / MACRO_SAVE) % 200] * 10.0f);
    #else
    d.uz[idx3_in];
    #endif

    const float uu = uzIn*uzIn;

    idx_t fluidNode = global3(x,y,1);
    float feq  = computeFeq(d.rho[fluidNode],0.0f,0.0f,uzIn,uu,5);
    float fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                            d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                            d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],5);
    d.f[5 * PLANE + fluidNode] = toPop(feq + OMCO_ZMIN * fneq);

    fluidNode = global3(x+1,y,1);
    feq  = computeFeq(d.rho[fluidNode],0.0f,0.0f,uzIn,uu,9);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                      d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                      d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],9);
    d.f[9 * PLANE + fluidNode] = toPop(feq + OMCO_ZMIN * fneq);

    fluidNode = global3(x,y+1,1);
    feq  = computeFeq(d.rho[fluidNode],0.0f,0.0f,uzIn,uu,11);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                      d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                      d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],11);
    d.f[11 * PLANE + fluidNode] = toPop(feq + OMCO_ZMIN * fneq);

    fluidNode = global3(x-1,y,1);
    feq  = computeFeq(d.rho[fluidNode],0.0f,0.0f,uzIn,uu,16);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                      d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                      d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],16);
    d.f[16 * PLANE + fluidNode] = toPop(feq + OMCO_ZMIN * fneq);

    fluidNode = global3(x,y-1,1);
    feq  = computeFeq(d.rho[fluidNode],0.0f,0.0f,uzIn,uu,18);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                      d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                      d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],18);
    d.f[18 * PLANE + fluidNode] = toPop(feq + OMCO_ZMIN * fneq);

    #if defined(D3Q27)
    fluidNode = global3(x+1,y+1,1);
    feq  = computeFeq(d.rho[fluidNode],0.0f,0.0f,uzIn,uu,19);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],19);
    d.f[19 * PLANE + fluidNode] = toPop(feq + OMCO_ZMIN * fneq);

    fluidNode = global3(x-1,y-1,1);
    feq  = computeFeq(d.rho[fluidNode],0.0f,0.0f,uzIn,uu,22);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],22);
    d.f[22 * PLANE + fluidNode] = toPop(feq + OMCO_ZMIN * fneq);

    fluidNode = global3(x+1,y-1,1);
    feq  = computeFeq(d.rho[fluidNode],0.0f,0.0f,uzIn,uu,23);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],23);
    d.f[23 * PLANE + fluidNode] = toPop(feq + OMCO_ZMIN * fneq);

    fluidNode = global3(x-1,y+1,1);
    feq  = computeFeq(d.rho[fluidNode],0.0f,0.0f,uzIn,uu,25);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                        d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                        d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],25);
    d.f[25 * PLANE + fluidNode] = toPop(feq + OMCO_ZMIN * fneq);
    #endif

    feq = computeGeq(d.phi[idx3_in],0.0f,0.0f,uzIn,5);
    d.g[5 * PLANE + global3(x,y,1)] = feq;
}

__global__ 
void applyOutflow(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    // const idx_t z = NZ-1;

    if (x >= NX || y >= NY) return;

    const idx_t idx3_zm1 = global3(x,y,NZ-2);
    d.phi[global3(x,y,NZ-1)] = d.phi[idx3_zm1];
 
    const float uxOut = d.ux[idx3_zm1];
    const float uyOut = d.uy[idx3_zm1];
    const float uzOut = d.uz[idx3_zm1];
    const float uu = uxOut*uxOut + uyOut*uyOut + uzOut*uzOut;

    float feq  = computeFeq(d.rho[idx3_zm1],uxOut,uyOut,uzOut,uu,6);
    float fneq = computeNeq(d.pxx[idx3_zm1],d.pyy[idx3_zm1],d.pzz[idx3_zm1],
                            d.pxy[idx3_zm1],d.pxz[idx3_zm1],d.pyz[idx3_zm1],
                            d.ux[idx3_zm1],d.uy[idx3_zm1],d.uz[idx3_zm1],6);
    d.f[6 * PLANE + idx3_zm1] = toPop(feq + OMCO_ZMAX * fneq);

    idx_t fluidNode = global3(x-1,y,NZ-2);
    feq  = computeFeq(d.rho[fluidNode],uxOut,uyOut,uzOut,uu,10);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                      d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                      d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],10);
    d.f[10 * PLANE + fluidNode] = toPop(feq + OMCO_ZMAX * fneq);

    fluidNode = global3(x,y-1,NZ-2);
    feq  = computeFeq(d.rho[fluidNode],uxOut,uyOut,uzOut,uu,12);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                      d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                      d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],12);
    d.f[12 * PLANE + fluidNode] = toPop(feq + OMCO_ZMAX * fneq);

    fluidNode = global3(x+1,y,NZ-2);
    feq  = computeFeq(d.rho[fluidNode],uxOut,uyOut,uzOut,uu,15);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                      d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                      d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],15);
    d.f[15 * PLANE + fluidNode] = toPop(feq + OMCO_ZMAX * fneq);

    fluidNode = global3(x,y+1,NZ-2);
    feq  = computeFeq(d.rho[fluidNode],uxOut,uyOut,uzOut,uu,17);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                      d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                      d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],17);
    d.f[17 * PLANE + fluidNode] = toPop(feq + OMCO_ZMAX * fneq);

    #if defined(D3Q27)
    fluidNode = global3(x-1,y-1,NZ-2);
    feq  = computeFeq(d.rho[fluidNode],uxOut,uyOut,uzOut,uu,20);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                      d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                      d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],20);
    d.f[20 * PLANE + fluidNode] = toPop(feq + OMCO_ZMAX * fneq);

    fluidNode = global3(x+1,y+1,NZ-2);
    feq  = computeFeq(d.rho[fluidNode],uxOut,uyOut,uzOut,uu,21);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                      d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                      d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],21);
    d.f[21 * PLANE + fluidNode] = toPop(feq + OMCO_ZMAX * fneq);

    fluidNode = global3(x-1,y+1,NZ-2);
    feq  = computeFeq(d.rho[fluidNode],uxOut,uyOut,uzOut,uu,24);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                      d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                      d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],24);
    d.f[24 * PLANE + fluidNode] = toPop(feq + OMCO_ZMAX * fneq);

    fluidNode = global3(x+1,y-1,NZ-2);
    feq  = computeFeq(d.rho[fluidNode],uxOut,uyOut,uzOut,uu,26);
    fneq = computeNeq(d.pxx[fluidNode],d.pyy[fluidNode],d.pzz[fluidNode],
                      d.pxy[fluidNode],d.pxz[fluidNode],d.pyz[fluidNode],
                      d.ux[fluidNode],d.uy[fluidNode],d.uz[fluidNode],26);
    d.f[26 * PLANE + fluidNode] = toPop(feq + OMCO_ZMAX * fneq);
    #endif

    feq = computeGeq(d.phi[idx3_zm1],0.0f,0.0f,U_REF,6);
    d.g[6 * PLANE + idx3_zm1] = feq;
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
    d.g[4   * PLANE + bT] = d.g[4   * PLANE + bB];

    // ghost cells
    d.phi[global3(x,0,z)]    = d.phi[bT];
    d.phi[global3(x,NY-1,z)] = d.phi[bB];
}

#elif defined(DROPLET)

#endif

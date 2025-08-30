#pragma once

#if defined(JET)

__global__ 
void applyInflow(
    LBMFields d, 
    const int STEP
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = 0;

    if (x >= NX || y >= NY) return;

    const float centerX = (NX-1) * 0.5f;
    const float centerY = (NY-1) * 0.5f;

    const float dx = x-centerX, dy = y-centerY;
    const float radialDist = sqrtf(dx*dx + dy*dy);
    const float radius = 0.5f * DIAM;
    if (radialDist > radius) return;

    const idx_t idx3_in = global3(x,y,z);
    const float uzIn = 
    #if defined(PERTURBATION)
        /* apply perturbation */ U_REF * (1.0f + PERTURBATION_DATA[(STEP/MACRO_SAVE)%200] * 10.0f);
    #else
        /* straightforward */ U_REF;
    #endif 

    d.uz[idx3_in] = uzIn;

    idx_t nbrIdx = global3(x,y,z+1);
    float feq = computeEquilibria(d.rho[nbrIdx],0.0f,0.0f,uzIn,5);
    float fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                         d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                         d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],5);
    d.f[PLANE5+nbrIdx] = to_pop(feq + OMCO_ZMIN * fneqReg);

    feq = computeTruncatedEquilibria(1.0f,0.0f,0.0f,uzIn,5);
    d.g[PLANE5+nbrIdx] = feq;

    nbrIdx = global3(x+1,y,z+1);
    feq = computeEquilibria(d.rho[nbrIdx],0.0f,0.0f,uzIn,9);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],9);
    d.f[PLANE9+nbrIdx] = to_pop(feq + OMCO_ZMIN * fneqReg);

    nbrIdx = global3(x,y+1,z+1);
    feq = computeEquilibria(d.rho[nbrIdx],0.0f,0.0f,uzIn,11);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],11);
    d.f[PLANE11+nbrIdx] = to_pop(feq + OMCO_ZMIN * fneqReg);

    nbrIdx = global3(x-1,y,z+1);
    feq = computeEquilibria(d.rho[nbrIdx],0.0f,0.0f,uzIn,16);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],16);
    d.f[PLANE16+nbrIdx] = to_pop(feq + OMCO_ZMIN * fneqReg);

    nbrIdx = global3(x,y-1,z+1);
    feq = computeEquilibria(d.rho[nbrIdx],0.0f,0.0f,uzIn,18);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],18);
    d.f[PLANE18+nbrIdx] = to_pop(feq + OMCO_ZMIN * fneqReg);

    #if defined(D3Q27)
    nbrIdx = global3(x+1,y+1,z+1);
    feq = computeEquilibria(d.rho[nbrIdx],0.0f,0.0f,uzIn,19);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],19);
    d.f[PLANE19+nbrIdx] = to_pop(feq + OMCO_ZMIN * fneqReg);

    nbrIdx = global3(x-1,y-1,z+1);
    feq = computeEquilibria(d.rho[nbrIdx],0.0f,0.0f,uzIn,22);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],22);
    d.f[PLANE22+nbrIdx] = to_pop(feq + OMCO_ZMIN * fneqReg);

    nbrIdx = global3(x+1,y-1,z+1);
    feq = computeEquilibria(d.rho[nbrIdx],0.0f,0.0f,uzIn,23);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],23);
    d.f[PLANE23+nbrIdx] = to_pop(feq + OMCO_ZMIN * fneqReg);

    nbrIdx = global3(x-1,y+1,z+1);
    feq = computeEquilibria(d.rho[nbrIdx],0.0f,0.0f,uzIn,25);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],25);
    d.f[PLANE25+nbrIdx] = to_pop(feq + OMCO_ZMIN * fneqReg);
    #endif 
}

__global__ 
void applyOutflow(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = NZ-1;

    if (x >= NX || y >= NY) return;

    const idx_t idx3 = global3(x,y,z);
    const idx_t idx3_zm1 = global3(x,y,z-1);
    
    //d.rho[idx3] = d.rho[idx3_zm1];
    d.phi[idx3] = d.phi[idx3_zm1];
    d.ux[idx3] = d.ux[idx3_zm1];
    d.uy[idx3] = d.uy[idx3_zm1];
    d.uz[idx3] = d.uz[idx3_zm1];

    const float uxOut = d.ux[idx3];
    const float uyOut = d.uy[idx3];
    const float uzOut = d.uz[idx3];

    // CI[6] = idx3_zm1
    float feq = computeEquilibria(d.rho[idx3_zm1],uxOut,uyOut,uzOut,6);
    float fneqReg = computeNonEquilibria(d.pxx[idx3_zm1],d.pyy[idx3_zm1],d.pzz[idx3_zm1],
                                         d.pxy[idx3_zm1],d.pxz[idx3_zm1],d.pyz[idx3_zm1],
                                         d.ux[idx3_zm1],d.uy[idx3_zm1],d.uz[idx3_zm1],6);
    d.f[PLANE6+idx3_zm1] = to_pop(feq + OMCO_ZMAX * fneqReg);

    feq = computeTruncatedEquilibria(d.phi[idx3_zm1],uxOut,uyOut,uzOut,6);
    d.g[PLANE6+idx3_zm1] = feq;

    idx_t nbrIdx = global3(x-1,y,z-1);
    feq = computeEquilibria(d.rho[nbrIdx],uxOut,uyOut,uzOut,10);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],10);
    d.f[PLANE10+nbrIdx] = to_pop(feq + OMCO_ZMAX * fneqReg);

    nbrIdx = global3(x,y-1,z-1);
    feq = computeEquilibria(d.rho[nbrIdx],uxOut,uyOut,uzOut,12);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],12);
    d.f[PLANE12+nbrIdx] = to_pop(feq + OMCO_ZMAX * fneqReg);

    nbrIdx = global3(x+1,y,z-1);
    feq = computeEquilibria(d.rho[nbrIdx],uxOut,uyOut,uzOut,15);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],15);
    d.f[PLANE15+nbrIdx] = to_pop(feq + OMCO_ZMAX * fneqReg);

    nbrIdx = global3(x,y+1,z-1);
    feq = computeEquilibria(d.rho[nbrIdx],uxOut,uyOut,uzOut,17);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],17);
    d.f[PLANE17+nbrIdx] = to_pop(feq + OMCO_ZMAX * fneqReg);

    #if defined(D3Q27)
    nbrIdx = global3(x-1,y-1,z-1);
    feq = computeEquilibria(d.rho[nbrIdx],uxOut,uyOut,uzOut,20);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],20);
    d.f[PLANE20+nbrIdx] = to_pop(feq + OMCO_ZMAX * fneqReg);

    nbrIdx = global3(x+1,y+1,z-1);
    feq = computeEquilibria(d.rho[nbrIdx],uxOut,uyOut,uzOut,21);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],21);
    d.f[PLANE21+nbrIdx] = to_pop(feq + OMCO_ZMAX * fneqReg);

    nbrIdx = global3(x-1,y+1,z-1);
    feq = computeEquilibria(d.rho[nbrIdx],uxOut,uyOut,uzOut,24);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],24);
    d.f[PLANE24+nbrIdx] = to_pop(feq + OMCO_ZMAX * fneqReg);

    nbrIdx = global3(x+1,y-1,z-1);
    feq = computeEquilibria(d.rho[nbrIdx],uxOut,uyOut,uzOut,26);
    fneqReg = computeNonEquilibria(d.pxx[nbrIdx],d.pyy[nbrIdx],d.pzz[nbrIdx],
                                   d.pxy[nbrIdx],d.pxz[nbrIdx],d.pyz[nbrIdx],
                                   d.ux[nbrIdx],d.uy[nbrIdx],d.uz[nbrIdx],26);
    d.f[PLANE26+nbrIdx] = to_pop(feq + OMCO_ZMAX * fneqReg);
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
    copyDirs<pop_t,1,7,9,13,15>(d.f,bL,bR);   
    #if defined(D3Q27)
    copyDirs<pop_t,19,21,23,26>(d.f,bL,bR);
    #endif 

    // negative x contributions
    copyDirs<pop_t,2,8,10,14,16>(d.f,bR,bL); 
    #if defined(D3Q27)
    copyDirs<pop_t,20,22,24,25>(d.f,bR,bL);
    #endif 

    d.g[PLANE+bL] = d.g[PLANE+bR];
    d.g[2*PLANE+bR] = d.g[2*PLANE+bL];
    d.phi[global3(0,y,z)] = d.phi[bR];
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

    d.g[3*PLANE+bB] = d.g[3*PLANE+bT];
    d.g[4*PLANE+bT] = d.g[4*PLANE+bB];
    d.phi[global3(x,0,z)] = d.phi[bT];
    d.phi[global3(x,NY-1,z)] = d.phi[bB];

    // positive y contributions
    copyDirs<pop_t,3,7,11,14,17>(d.f,bB,bT);
    #if defined(D3Q27)
    copyDirs<pop_t,19,21,24,25>(d.f,bB,bT);
    #endif 

    // negative y contributions
    copyDirs<pop_t,4,8,12,13,18>(d.f,bT,bB);
    #if defined(D3Q27)
    copyDirs<pop_t,20,22,23,26>(d.f,bT,bB);
    #endif 

    //d.g[3*PLANE+bB] = d.g[3*PLANE+bT];
    //d.g[4*PLANE+bT] = d.g[4*PLANE+bB];
    //d.phi[global3(x,0,z)] = d.phi[bT];
    //d.phi[global3(x,NY-1,z)] = d.phi[bB];
}

#elif defined(DROPLET)

// still undefined

#endif 


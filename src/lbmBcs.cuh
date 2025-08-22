#pragma once

#ifdef JET_CASE

__global__ void gpuApplyInflow(LBMFields d, const int STEP) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = 0;

    if (x >= NX || y >= NY) return;

    const float centerX = (NX-1) * 0.5f;
    const float centerY = (NY-1) * 0.5f;

    const float dx = x-centerX, dy = y-centerY;
    const float radialDist = sqrtf(dx*dx + dy*dy);
    const float radius = 0.5f * DIAM;
    if (radialDist > radius) return;

    const idx_t idx3_in = global3(x,y,z);
    const float uzIn = 
    #ifdef PERTURBATION
        /* apply perturbation */ U_JET * (1.0f + PERTURBATION_DATA[(STEP/MACRO_SAVE)%200] * 10.0f);
    #else
        /* straightforward */ U_JET;
    #endif 

    d.uz[idx3_in] = uzIn;

    int neighborIdx = global3(x,y,z+1);
    float feq = computeEquilibria(d.rho[neighborIdx],0.0f,0.0f,uzIn,5);
    float fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                                d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                                d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],5);
    d.f[global4(x,y,z+1,5)] = to_pop(feq + OMCO * fneqReg);

    feq = computeTruncatedEquilibria(1.0f,0.0f,0.0f,uzIn,5);
    d.g[global4(x,y,z+1,5)] = feq;

    neighborIdx = global3(x+1,y,z+1);
    feq = computeEquilibria(d.rho[neighborIdx],0.0f,0.0f,uzIn,9);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],9);
    d.f[global4(x+1,y,z+1,9)] = to_pop(feq + OMCO * fneqReg);

    neighborIdx = global3(x,y+1,z+1);
    feq = computeEquilibria(d.rho[neighborIdx],0.0f,0.0f,uzIn,11);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],11);
    d.f[global4(x,y+1,z+1,11)] = to_pop(feq + OMCO * fneqReg);

    neighborIdx = global3(x-1,y,z+1);
    feq = computeEquilibria(d.rho[neighborIdx],0.0f,0.0f,uzIn,16);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],16);
    d.f[global4(x-1,y,z+1,16)] = to_pop(feq + OMCO * fneqReg);

    neighborIdx = global3(x,y-1,z+1);
    feq = computeEquilibria(d.rho[neighborIdx],0.0f,0.0f,uzIn,18);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],18);
    d.f[global4(x,y-1,z+1,18)] = to_pop(feq + OMCO * fneqReg);

    #ifdef D3Q27
    neighborIdx = global3(x+1,y+1,z+1);
    feq = computeEquilibria(d.rho[neighborIdx],0.0f,0.0f,uzIn,19);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],19);
    d.f[global4(x+1,y+1,z+1,19)] = to_pop(feq + OMCO * fneqReg);

    neighborIdx = global3(x-1,y-1,z+1);
    feq = computeEquilibria(d.rho[neighborIdx],0.0f,0.0f,uzIn,22);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],22);
    d.f[global4(x-1,y-1,z+1,22)] = to_pop(feq + OMCO * fneqReg);

    neighborIdx = global3(x+1,y-1,z+1);
    feq = computeEquilibria(d.rho[neighborIdx],0.0f,0.0f,uzIn,23);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],23);
    d.f[global4(x+1,y-1,z+1,23)] = to_pop(feq + OMCO * fneqReg);

    neighborIdx = global3(x-1,y+1,z+1);
    feq = computeEquilibria(d.rho[neighborIdx],0.0f,0.0f,uzIn,25);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],25);
    d.f[global4(x-1,y+1,z+1,25)] = to_pop(feq + OMCO * fneqReg);
    #endif // D3Q27
}

__global__ void gpuApplyOutflow(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = NZ-1;

    if (x >= NX || y >= NY) return;

    const int idx_outer = global3(x,y,z);
    const int idx_inner = global3(x,y,z-1);
    
    //d.rho[idx_outer] = d.rho[idx_inner];
    d.phi[idx_outer] = d.phi[idx_inner];
    d.ux[idx_outer] = d.ux[idx_inner];
    d.uy[idx_outer] = d.uy[idx_inner];
    d.uz[idx_outer] = d.uz[idx_inner];

    const float uxOut = d.ux[idx_outer];
    const float uyOut = d.uy[idx_outer];
    const float uzOut = d.uz[idx_outer];

    int neighborIdx = global3(x,y,z-1);
    float feq = computeEquilibria(d.rho[neighborIdx],uxOut,uyOut,uzOut,6);
    float fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                                d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                                d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],6);
    d.f[global4(x,y,z-1,6)] = to_pop(feq + OMCO_MAX * fneqReg);

    feq = computeTruncatedEquilibria(d.phi[neighborIdx],uxOut,uyOut,uzOut,6);
    d.g[global4(x,y,z-1,6)] = feq;

    neighborIdx = global3(x-1,y,z-1);
    feq = computeEquilibria(d.rho[neighborIdx],uxOut,uyOut,uzOut,10);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],10);
    d.f[global4(x-1,y,z-1,10)] = to_pop(feq + OMCO_MAX * fneqReg);

    neighborIdx = global3(x,y-1,z-1);
    feq = computeEquilibria(d.rho[neighborIdx],uxOut,uyOut,uzOut,12);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],12);
    d.f[global4(x,y-1,z-1,12)] = to_pop(feq + OMCO_MAX * fneqReg);

    neighborIdx = global3(x+1,y,z-1);
    feq = computeEquilibria(d.rho[neighborIdx],uxOut,uyOut,uzOut,15);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],15);
    d.f[global4(x+1,y,z-1,15)] = to_pop(feq + OMCO_MAX * fneqReg);

    neighborIdx = global3(x,y+1,z-1);
    feq = computeEquilibria(d.rho[neighborIdx],uxOut,uyOut,uzOut,17);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],17);
    d.f[global4(x,y+1,z-1,17)] = to_pop(feq + OMCO_MAX * fneqReg);

    #ifdef D3Q27
    neighborIdx = global3(x-1,y-1,z-1);
    feq = computeEquilibria(d.rho[neighborIdx],uxOut,uyOut,uzOut,20);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],20);
    d.f[global4(x-1,y-1,z-1,20)] = to_pop(feq + OMCO_MAX * fneqReg);

    neighborIdx = global3(x+1,y+1,z-1);
    feq = computeEquilibria(d.rho[neighborIdx],uxOut,uyOut,uzOut,21);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],21);
    d.f[global4(x+1,y+1,z-1,21)] = to_pop(feq + OMCO_MAX * fneqReg);

    neighborIdx = global3(x-1,y+1,z-1);
    feq = computeEquilibria(d.rho[neighborIdx],uxOut,uyOut,uzOut,24);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],24);
    d.f[global4(x-1,y+1,z-1,24)] = to_pop(feq + OMCO_MAX * fneqReg);

    neighborIdx = global3(x+1,y-1,z-1);
    feq = computeEquilibria(d.rho[neighborIdx],uxOut,uyOut,uzOut,26);
    fneqReg = computeNonEquilibria(d.pxx[neighborIdx],d.pyy[neighborIdx],d.pzz[neighborIdx],
                                          d.pxy[neighborIdx],d.pxz[neighborIdx],d.pyz[neighborIdx],
                                          d.ux[neighborIdx],d.uy[neighborIdx],d.uz[neighborIdx],26);
    d.f[global4(x+1,y-1,z-1,26)] = to_pop(feq + OMCO_MAX * fneqReg);
    #endif // D3Q27
}

__global__ void gpuApplyPeriodicX(LBMFields d) {
    const int y = threadIdx.x + blockIdx.x * blockDim.x;
    const int z = threadIdx.y + blockIdx.y * blockDim.y;
    
    //if (y <= 0 || y >= NY-1 || z <= 0 || z >= NZ-1) return;
    if (y >= NY || z >= NZ || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t bL = global3(1,y,z);
    const idx_t bR = global3(NX-2,y,z);

    // positive x contributions
    copyDirs<pop_t,1,7,9,13,15>(d.f,bL,bR);   
    #ifdef D3Q27
    copyDirs<pop_t,19,21,23,26>(d.f,bL,bR);
    #endif // D3Q27

    // negative x contributions
    copyDirs<pop_t,2,8,10,14,16>(d.f,bR,bL); 
    #ifdef D3Q27
    copyDirs<pop_t,20,22,24,25>(d.f,bR,bL);
    #endif // D3Q27

    d.g[PLANE+bL] = d.g[PLANE+bR];
    d.g[2*PLANE+bR] = d.g[2*PLANE+bL];
    d.phi[global3(0,y,z)] = d.phi[bR];
    d.phi[global3(NX-1,y,z)] = d.phi[bL];
}

__global__ void gpuApplyPeriodicY(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int z = threadIdx.y + blockIdx.y * blockDim.y;
    
    //if (x <= 0 || x >= NX-1 || z <= 0 || z >= NZ-1) return;
    if (x >= NX || z >= NZ || 
        x == 0 || x == NX-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t bB = global3(x,1,z);
    const idx_t bT = global3(x,NY-2,z);

    // positive y contributions
    copyDirs<pop_t,3,7,11,14,17>(d.f,bB,bT);
    #ifdef D3Q27
    copyDirs<pop_t,19,21,24,25>(d.f,bB,bT);
    #endif // D3Q27

    // negative y contributions
    copyDirs<pop_t,4,8,12,13,18>(d.f,bT,bB);
    #ifdef D3Q27
    copyDirs<pop_t,20,22,23,26>(d.f,bT,bB);
    #endif // D3Q27

    d.g[3*PLANE+bB] = d.g[3*PLANE+bT];
    d.g[4*PLANE+bT] = d.g[4*PLANE+bB];
    d.phi[global3(x,0,z)] = d.phi[bT];
    d.phi[global3(x,NY-1,z)] = d.phi[bB];
}

#elif defined(DROPLET_CASE)

// still undefined

#endif // FLOW_CASE


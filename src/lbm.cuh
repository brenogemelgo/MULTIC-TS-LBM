#pragma once

__global__ 
void computePhase(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = global3(x,y,z);

    const float phi = 
        d.g[idx3] + 
        d.g[PLANE + idx3] + 
        d.g[2 * PLANE + idx3] + 
        d.g[3 * PLANE + idx3] + 
        d.g[4 * PLANE + idx3] + 
        d.g[5 * PLANE + idx3] + 
        d.g[6 * PLANE + idx3];
                      
    d.phi[idx3] = phi;
}

__global__ 
void computeNormals(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = global3(x,y,z);

    float sumGradX = W_1 * (d.phi[global3(x+1,y,z)] - d.phi[global3(x-1,y,z)]) +
                     W_2 * (d.phi[global3(x+1,y+1,z)] - d.phi[global3(x-1,y-1,z)] +
                            d.phi[global3(x+1,y,z+1)] - d.phi[global3(x-1,y,z-1)] +
                            d.phi[global3(x+1,y-1,z)] - d.phi[global3(x-1,y+1,z)] +
                            d.phi[global3(x+1,y,z-1)] - d.phi[global3(x-1,y,z+1)]);

    float sumGradY = W_1 * (d.phi[global3(x,y+1,z)] - d.phi[global3(x,y-1,z)]) +
                     W_2 * (d.phi[global3(x+1,y+1,z)] - d.phi[global3(x-1,y-1,z)] +
                            d.phi[global3(x,y+1,z+1)] - d.phi[global3(x,y-1,z-1)] +
                            d.phi[global3(x-1,y+1,z)] - d.phi[global3(x+1,y-1,z)] +
                            d.phi[global3(x,y+1,z-1)] - d.phi[global3(x,y-1,z+1)]);

    float sumGradZ = W_1 * (d.phi[global3(x,y,z+1)] - d.phi[global3(x,y,z-1)]) +
                     W_2 * (d.phi[global3(x+1,y,z+1)] - d.phi[global3(x-1,y,z-1)] +
                            d.phi[global3(x,y+1,z+1)] - d.phi[global3(x,y-1,z-1)] +
                            d.phi[global3(x-1,y,z+1)] - d.phi[global3(x+1,y,z-1)] +
                            d.phi[global3(x,y-1,z+1)] - d.phi[global3(x,y+1,z-1)]);
    #if defined(D3Q27)
    sumGradX += W_3 * (d.phi[global3(x+1,y+1,z+1)] - d.phi[global3(x-1,y-1,z-1)] +
                       d.phi[global3(x+1,y+1,z-1)] - d.phi[global3(x-1,y-1,z+1)] +
                       d.phi[global3(x+1,y-1,z+1)] - d.phi[global3(x-1,y+1,z-1)] +
                       d.phi[global3(x+1,y-1,z-1)] - d.phi[global3(x-1,y+1,z+1)]);

    sumGradY += W_3 * (d.phi[global3(x+1,y+1,z+1)] - d.phi[global3(x-1,y-1,z-1)] +
                       d.phi[global3(x+1,y+1,z-1)] - d.phi[global3(x-1,y-1,z+1)] +
                       d.phi[global3(x-1,y+1,z-1)] - d.phi[global3(x+1,y-1,z+1)] +
                       d.phi[global3(x-1,y+1,z+1)] - d.phi[global3(x+1,y-1,z-1)]);

    sumGradZ += W_3 * (d.phi[global3(x+1,y+1,z+1)] - d.phi[global3(x-1,y-1,z-1)] +
                       d.phi[global3(x-1,y-1,z+1)] - d.phi[global3(x+1,y+1,z-1)] +
                       d.phi[global3(x+1,y-1,z+1)] - d.phi[global3(x-1,y+1,z-1)] +
                       d.phi[global3(x-1,y+1,z+1)] - d.phi[global3(x+1,y-1,z-1)]);
    #endif 
        
    const float gradX = 3.0f * sumGradX;
    const float gradY = 3.0f * sumGradY;
    const float gradZ = 3.0f * sumGradZ;
    
    const float ind = sqrtf(gradX*gradX + gradY*gradY + gradZ*gradZ);
    const float invInd = 1.0f / (ind + 1e-9f);

    const float normX = gradX * invInd;
    const float normY = gradY * invInd;
    const float normZ = gradZ * invInd;

    d.ind[idx3] = ind;
    d.normx[idx3] = normX;
    d.normy[idx3] = normY;
    d.normz[idx3] = normZ;
}

__global__ 
void computeForces(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = global3(x,y,z);

    float sumCurvX = W_1 * (d.normx[global3(x+1,y,z)] - d.normx[global3(x-1,y,z)]) +
                     W_2 * (d.normx[global3(x+1,y+1,z)] - d.normx[global3(x-1,y-1,z)] +
                            d.normx[global3(x+1,y,z+1)] - d.normx[global3(x-1,y,z-1)] +
                            d.normx[global3(x+1,y-1,z)] - d.normx[global3(x-1,y+1,z)] +
                            d.normx[global3(x+1,y,z-1)] - d.normx[global3(x-1,y,z+1)]);

    float sumCurvY = W_1 * (d.normy[global3(x,y+1,z)] - d.normy[global3(x,y-1,z)]) +
                     W_2 * (d.normy[global3(x+1,y+1,z)] - d.normy[global3(x-1,y-1,z)] +
                            d.normy[global3(x,y+1,z+1)] - d.normy[global3(x,y-1,z-1)] +
                            d.normy[global3(x-1,y+1,z)] - d.normy[global3(x+1,y-1,z)] +
                            d.normy[global3(x,y+1,z-1)] - d.normy[global3(x,y-1,z+1)]);

    float sumCurvZ = W_1 * (d.normz[global3(x,y,z+1)] - d.normz[global3(x,y,z-1)]) +
                     W_2 * (d.normz[global3(x+1,y,z+1)] - d.normz[global3(x-1,y,z-1)] +
                            d.normz[global3(x,y+1,z+1)] - d.normz[global3(x,y-1,z-1)] +
                            d.normz[global3(x-1,y,z+1)] - d.normz[global3(x+1,y,z-1)] +
                            d.normz[global3(x,y-1,z+1)] - d.normz[global3(x,y+1,z-1)]);
    #if defined(D3Q27)
    sumCurvX += W_3 * (d.normx[global3(x+1,y+1,z+1)] - d.normx[global3(x-1,y-1,z-1)] +
                       d.normx[global3(x+1,y+1,z-1)] - d.normx[global3(x-1,y-1,z+1)] +
                       d.normx[global3(x+1,y-1,z+1)] - d.normx[global3(x-1,y+1,z-1)] +
                       d.normx[global3(x+1,y-1,z-1)] - d.normx[global3(x-1,y+1,z+1)]);

    sumCurvY += W_3 * (d.normy[global3(x+1,y+1,z+1)] - d.normy[global3(x-1,y-1,z-1)] +
                       d.normy[global3(x+1,y+1,z-1)] - d.normy[global3(x-1,y-1,z+1)] +
                       d.normy[global3(x-1,y+1,z-1)] - d.normy[global3(x+1,y-1,z+1)] +
                       d.normy[global3(x-1,y+1,z+1)] - d.normy[global3(x+1,y-1,z-1)]);

    sumCurvZ += W_3 * (d.normz[global3(x+1,y+1,z+1)] - d.normz[global3(x-1,y-1,z-1)] +
                       d.normz[global3(x-1,y-1,z+1)] - d.normz[global3(x+1,y+1,z-1)] +
                       d.normz[global3(x+1,y-1,z+1)] - d.normz[global3(x-1,y+1,z-1)] +
                       d.normz[global3(x-1,y+1,z+1)] - d.normz[global3(x+1,y-1,z-1)]);
    #endif 
    const float curvature = -3.0f * (sumCurvX + sumCurvY + sumCurvZ);    

    const float stCurv = SIGMA * curvature * d.ind[idx3];
    d.ffx[idx3] = stCurv * d.normx[idx3];
    d.ffy[idx3] = stCurv * d.normy[idx3];
    d.ffz[idx3] = stCurv * d.normz[idx3];
}

__global__ 
void streamCollide(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = global3(x,y,z);

    const float pop0 = from_pop(d.f[idx3]);         
    const float pop1 = from_pop(d.f[PLANE + idx3]);   
    const float pop2 = from_pop(d.f[2 * PLANE + idx3]);  
    const float pop3 = from_pop(d.f[3 * PLANE + idx3]);  
    const float pop4 = from_pop(d.f[4 * PLANE + idx3]);  
    const float pop5 = from_pop(d.f[5 * PLANE + idx3]);  
    const float pop6 = from_pop(d.f[6 * PLANE + idx3]);  
    const float pop7 = from_pop(d.f[7 * PLANE + idx3]);  
    const float pop8 = from_pop(d.f[8 * PLANE + idx3]);  
    const float pop9 = from_pop(d.f[9 * PLANE + idx3]);  
    const float pop10 = from_pop(d.f[10 * PLANE + idx3]);
    const float pop11 = from_pop(d.f[11 * PLANE + idx3]);
    const float pop12 = from_pop(d.f[12 * PLANE + idx3]); 
    const float pop13 = from_pop(d.f[13 * PLANE + idx3]); 
    const float pop14 = from_pop(d.f[14 * PLANE + idx3]);
    const float pop15 = from_pop(d.f[15 * PLANE + idx3]); 
    const float pop16 = from_pop(d.f[16 * PLANE + idx3]); 
    const float pop17 = from_pop(d.f[17 * PLANE + idx3]);
    const float pop18 = from_pop(d.f[18 * PLANE + idx3]); 
    #if defined(D3Q27)
    const float pop19 = from_pop(d.f[19 * PLANE + idx3]); 
    const float pop20 = from_pop(d.f[20 * PLANE + idx3]); 
    const float pop21 = from_pop(d.f[21 * PLANE + idx3]); 
    const float pop22 = from_pop(d.f[22 * PLANE + idx3]); 
    const float pop23 = from_pop(d.f[23 * PLANE + idx3]); 
    const float pop24 = from_pop(d.f[24 * PLANE + idx3]);
    const float pop25 = from_pop(d.f[25 * PLANE + idx3]);
    const float pop26 = from_pop(d.f[26 * PLANE + idx3]); 
    #endif 

    float rho = pop0 + pop1 + pop2 + pop3 + pop4 + pop5 + pop6 + pop7 + pop8 + pop9 + pop10 + pop11 + pop12 + pop13 + pop14 + pop15 + pop16 + pop17 + pop18;
    #if defined(D3Q27)
    rho += pop19 + pop20 + pop21 + pop22 + pop23 + pop24 + pop25 + pop26;
    #endif
    rho += 1.0f; 
    d.rho[idx3] = rho;

    const float ffx = d.ffx[idx3];
    const float ffy = d.ffy[idx3];
    const float ffz = d.ffz[idx3];

    float ux, uy, uz;
    const float invRho = 1.0f / rho;
    
    #if defined(D3Q19)
    ux = invRho * (pop1 - pop2 + pop7 - pop8 + pop9 - pop10 + pop13 - pop14 + pop15 - pop16);
    uy = invRho * (pop3 - pop4 + pop7 - pop8 + pop11 - pop12 + pop14 - pop13 + pop17 - pop18);
    uz = invRho * (pop5 - pop6 + pop9 - pop10 + pop11 - pop12 + pop16 - pop15 + pop18 - pop17);
    #elif defined(D3Q27)
    ux = invRho * (pop1 - pop2 + pop7 - pop8 + pop9 - pop10 + pop13 - pop14 + pop15 - pop16 + pop19 - pop20 + pop21 - pop22 + pop23 - pop24 + pop26 - pop25);
    uy = invRho * (pop3 - pop4 + pop7 - pop8  + pop11 - pop12 + pop14 - pop13 + pop17 - pop18 + pop19 - pop20 + pop21 - pop22 + pop24 - pop23 + pop25 - pop26);
    uz = invRho * (pop5 - pop6 + pop9 - pop10 + pop11 - pop12 + pop16 - pop15 + pop18 - pop17 + pop19 - pop20 + pop22 - pop21 + pop23 - pop24 + pop25 - pop26);
    #endif
    
    ux += ffx * 0.5f * invRho;
    uy += ffy * 0.5f * invRho;
    uz += ffz * 0.5f * invRho;

    d.ux[idx3] = ux; 
    d.uy[idx3] = uy; 
    d.uz[idx3] = uz;

    float pxx, pyy, pzz, pxy, pxz, pyz;
    { // ====================================== MOMENTUM FLUX TENSOR ====================================== //
        float feq, fneq;

        feq  = computeFeq<1>(rho,ux,uy,uz);
        fneq = pop1 - feq;
        pxx = fneq;

        feq  = computeFeq<2>(rho,ux,uy,uz);
        fneq = pop2 - feq;
        pxx += fneq;

        feq  = computeFeq<3>(rho,ux,uy,uz);
        fneq = pop3 - feq;
        pyy = fneq;

        feq  = computeFeq<4>(rho,ux,uy,uz);
        fneq = pop4 - feq;
        pyy += fneq;

        feq  = computeFeq<5>(rho,ux,uy,uz);
        fneq = pop5 - feq;
        pzz = fneq;

        feq  = computeFeq<6>(rho,ux,uy,uz);
        fneq = pop6 - feq;
        pzz += fneq;

        feq  = computeFeq<7>(rho,ux,uy,uz);
        fneq = pop7 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pxy = fneq;

        feq  = computeFeq<8>(rho,ux,uy,uz);
        fneq = pop8 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pxy += fneq;

        feq  = computeFeq<9>(rho,ux,uy,uz);
        fneq = pop9 - feq;
        pxx += fneq; 
        pzz += fneq; 
        pxz = fneq;

        feq  = computeFeq<10>(rho,ux,uy,uz);
        fneq = pop10 - feq;
        pxx += fneq; 
        pzz += fneq; 
        pxz += fneq;

        feq  = computeFeq<11>(rho,ux,uy,uz);
        fneq = pop11 - feq;
        pyy += fneq;
        pzz += fneq; 
        pyz = fneq;

        feq  = computeFeq<12>(rho,ux,uy,uz);
        fneq = pop12 - feq;
        pyy += fneq; 
        pzz += fneq; 
        pyz += fneq;

        feq  = computeFeq<13>(rho,ux,uy,uz);
        fneq = pop13 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pxy -= fneq;

        feq  = computeFeq<14>(rho,ux,uy,uz);
        fneq = pop14 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pxy -= fneq;

        feq  = computeFeq<15>(rho,ux,uy,uz);
        fneq = pop15 - feq;
        pxx += fneq; 
        pzz += fneq; 
        pxz -= fneq;

        feq  = computeFeq<16>(rho,ux,uy,uz);
        fneq = pop16 - feq;
        pxx += fneq; 
        pzz += fneq; 
        pxz -= fneq;

        feq  = computeFeq<17>(rho,ux,uy,uz);
        fneq = pop17 - feq;
        pyy += fneq; 
        pzz += fneq; 
        pyz -= fneq;

        feq  = computeFeq<18>(rho,ux,uy,uz);
        fneq = pop18 - feq;
        pyy += fneq; 
        pzz += fneq; 
        pyz -= fneq;
        
        #if defined(D3Q27)
        feq  = computeFeq<19>(rho,ux,uy,uz);
        fneq = pop19 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy += fneq; 
        pxz += fneq; 
        pyz += fneq;

        feq  = computeFeq<20>(rho,ux,uy,uz);
        fneq = pop20 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy += fneq; 
        pxz += fneq; 
        pyz += fneq;

        feq  = computeFeq<21>(rho,ux,uy,uz);
        fneq = pop21 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy += fneq; 
        pxz -= fneq; 
        pyz -= fneq;

        feq  = computeFeq<22>(rho,ux,uy,uz);
        fneq = pop22 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy += fneq; 
        pxz -= fneq; 
        pyz -= fneq; 

        feq  = computeFeq<23>(rho,ux,uy,uz);
        fneq = pop23 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy -= fneq; 
        pxz += fneq; 
        pyz -= fneq;

        feq  = computeFeq<24>(rho,ux,uy,uz);
        fneq = pop24 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy -= fneq; 
        pxz += fneq; 
        pyz -= fneq;

        feq  = computeFeq<25>(rho,ux,uy,uz);
        fneq = pop25 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy -= fneq; 
        pxz -= fneq; 
        pyz += fneq;

        feq  = computeFeq<26>(rho,ux,uy,uz);
        fneq = pop26 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy -= fneq; 
        pxz -= fneq; 
        pyz += fneq;
        #endif 

        d.pxx[idx3] = pxx;
        d.pyy[idx3] = pyy;
        d.pzz[idx3] = pzz;
        d.pxy[idx3] = pxy;
        d.pxz[idx3] = pxz;   
        d.pyz[idx3] = pyz;
    } // ============================================== END ============================================== //

    float omcoLocal;
    #if defined(VISC_CONTRAST)
    {
        const float phi = d.phi[idx3]; 
        const float nuLocal = fmaf(phi, (VISC_OIL - VISC_WATER), VISC_WATER);
        const float omegaPhys = 1.0f / (0.5f + 3.0f * nuLocal);

        #if defined(JET) 
        omcoLocal = 1.0f - fminf(omegaPhys, cubicSponge(z));
        #elif defined(DROPLET) 
        omcoLocal = 1.0f - omegaPhys;
        #endif 
    }
    #else
    {
        #if defined(JET)
        omcoLocal = 1.0f - cubicSponge(z);
        #elif defined(DROPLET)
        omcoLocal = 1.0f - OMEGA_REF;
        #endif
    }
    #endif

    { // ====================================== COLLISION-STREAMING ====================================== //
        const float coeff = 0.5f + 0.5f * omcoLocal;
        const float aux = 3.0f * invRho;

        float feq, fneq, force;

        // ========================== Q0 ========================== //
        feq   = computeFeq<0>(rho,ux,uy,uz);
        force = computeForce<0>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<0>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[idx3] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q1 ========================== //
        feq   = computeFeq<1>(rho,ux,uy,uz);
        force = computeForce<1>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<1>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x+1,y,z,1)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q2 ========================== //
        feq   = computeFeq<2>(rho,ux,uy,uz);
        force = computeForce<2>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<2>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x-1,y,z,2)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q3 ========================== //
        feq   = computeFeq<3>(rho,ux,uy,uz);
        force = computeForce<3>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<3>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x,y+1,z,3)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q4 ========================== //
        feq   = computeFeq<4>(rho,ux,uy,uz);
        force = computeForce<4>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<4>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x,y-1,z,4)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q5 ========================== //
        feq   = computeFeq<5>(rho,ux,uy,uz);
        force = computeForce<5>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<5>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x,y,z+1,5)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q6 ========================== //
        feq   = computeFeq<6>(rho,ux,uy,uz);
        force = computeForce<6>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<6>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x,y,z-1,6)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q7 ========================== //
        feq   = computeFeq<7>(rho,ux,uy,uz);
        force = computeForce<7>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<7>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x+1,y+1,z,7)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q8 ========================== //
        feq   = computeFeq<8>(rho,ux,uy,uz);
        force = computeForce<8>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<8>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x-1,y-1,z,8)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q9 ========================== //
        feq   = computeFeq<9>(rho,ux,uy,uz);
        force = computeForce<9>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<9>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x+1,y,z+1,9)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q10 ========================== //
        feq   = computeFeq<10>(rho,ux,uy,uz);
        force = computeForce<10>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<10>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x-1,y,z-1,10)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q11 ========================== //
        feq   = computeFeq<11>(rho,ux,uy,uz);
        force = computeForce<11>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<11>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x,y+1,z+1,11)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q12 ========================== //
        feq   = computeFeq<12>(rho,ux,uy,uz);
        force = computeForce<12>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<12>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x,y-1,z-1,12)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q13 ========================== //
        feq   = computeFeq<13>(rho,ux,uy,uz);
        force = computeForce<13>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<13>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x+1,y-1,z,13)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q14 ========================== //
        feq   = computeFeq<14>(rho,ux,uy,uz);
        force = computeForce<14>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<14>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x-1,y+1,z,14)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q15 ========================== //
        feq   = computeFeq<15>(rho,ux,uy,uz);
        force = computeForce<15>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<15>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x+1,y,z-1,15)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q16 ========================== //
        feq   = computeFeq<16>(rho,ux,uy,uz);
        force = computeForce<16>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<16>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x-1,y,z+1,16)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q17 ========================== //
        feq   = computeFeq<17>(rho,ux,uy,uz);
        force = computeForce<17>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<17>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x,y+1,z-1,17)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q18 ========================== //
        feq   = computeFeq<18>(rho,ux,uy,uz);
        force = computeForce<18>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<18>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x,y-1,z+1,18)] = to_pop(feq + omcoLocal * fneq + force);

        #if defined(D3Q27)
        // ========================== Q19 ========================== //
        feq   = computeFeq<19>(rho,ux,uy,uz);
        force = computeForce<19>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<19>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x+,y+1,z+1,19)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q20 ========================== //
        feq   = computeFeq<20>(rho,ux,uy,uz);
        force = computeForce<20>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<20>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x-1,y-1,z-1,20)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q21 ========================== //
        feq   = computeFeq<21>(rho,ux,uy,uz);
        force = computeForce<21>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<21>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x+1,y+1,z-1,21)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q22 ========================== //
        feq   = computeFeq<22>(rho,ux,uy,uz);
        force = computeForce<22>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<22>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x-1,y-1,z+1,22)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q23 ========================== //
        feq   = computeFeq<23>(rho,ux,uy,uz);
        force = computeForce<23>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<23>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x+1,y-1,z+1,23)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q24 ========================== //
        feq   = computeFeq<24>(rho,ux,uy,uz);
        force = computeForce<24>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<24>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x-1,y+1,z-1,24)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q25 ========================== //
        feq   = computeFeq<25>(rho,ux,uy,uz);
        force = computeForce<25>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<25>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x-1,y+1,z+1,25)] = to_pop(feq + omcoLocal * fneq + force);

        // ========================== Q26 ========================== //
        feq   = computeFeq<26>(rho,ux,uy,uz);
        force = computeForce<26>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
        fneq  = computeNeq<26>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
        d.f[global4(x+1,y-1,z-1,26)] = to_pop(feq + omcoLocal * fneq + force);
        #endif
    } // ============================================== END ============================================== //

    { // ====================================== ADVECTION-DIFFUSION ====================================== //
        const float phi = d.phi[idx3];
        d.g[idx3] = W_G_1 * phi;

        const float phiNorm = W_G_2 * GAMMA * phi * (1.0f - phi);
        const float multPhi = W_G_2 * phi;
        const float a3 = 3.0f * multPhi;

        float feq = multPhi + a3 * ux;
        float force = phiNorm * d.normx[idx3];
        d.g[global4(x+1,y,z,1)] = feq + force;
        
        feq = multPhi - a3 * ux;
        d.g[global4(x-1,y,z,2)] = feq - force;

        feq = multPhi + a3 * uy;
        force = phiNorm * d.normy[idx3];
        d.g[global4(x,y+1,z,3)] = feq + force;

        feq = multPhi - a3 * uy;
        d.g[global4(x,y-1,z,4)] = feq - force;

        feq = multPhi + a3 * uz;
        force = phiNorm * d.normz[idx3];
        d.g[global4(x,y,z+1,5)] = feq + force;

        feq = multPhi - a3 * uz;
        d.g[global4(x,y,z-1,6)] = feq - force;
    } // ============================================= END ============================================= //
}

#pragma once

#if !defined(LESS_POINTERS)

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

    const float invRho = 1.0f / rho;
    
    #if defined(D3Q19)
    float ux = invRho * (pop1 - pop2 + pop7 - pop8 + pop9 - pop10 + pop13 - pop14 + pop15 - pop16);
    float uy = invRho * (pop3 - pop4 + pop7 - pop8 + pop11 - pop12 + pop14 - pop13 + pop17 - pop18);
    float uz = invRho * (pop5 - pop6 + pop9 - pop10 + pop11 - pop12 + pop16 - pop15 + pop18 - pop17);
    #elif defined(D3Q27)
    float ux = invRho * (pop1 - pop2 + pop7 - pop8 + pop9 - pop10 + pop13 - pop14 + pop15 - pop16 + pop19 - pop20 + pop21 - pop22 + pop23 - pop24 + pop26 - pop25);
    float uy = invRho * (pop3 - pop4 + pop7 - pop8  + pop11 - pop12 + pop14 - pop13 + pop17 - pop18 + pop19 - pop20 + pop21 - pop22 + pop24 - pop23 + pop25 - pop26);
    float uz = invRho * (pop5 - pop6 + pop9 - pop10 + pop11 - pop12 + pop16 - pop15 + pop18 - pop17 + pop19 - pop20 + pop22 - pop21 + pop23 - pop24 + pop25 - pop26);
    #endif
    
    ux += ffx * 0.5f * invRho;
    uy += ffy * 0.5f * invRho;
    uz += ffz * 0.5f * invRho;

    d.ux[idx3] = ux; 
    d.uy[idx3] = uy; 
    d.uz[idx3] = uz;

    const float invRhoCssq = 3.0f * invRho;

    // =============================================== SECOND-ORDER MOMENTUM FLUX TENSOR =============================================== //

        //             0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
        // CIX[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
        // CIY[27] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1 };
        // CIZ[27] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1 };

        float feq, fneq, pxx, pyy, pzz, pxy, pxz, pyz;

        // pop1
        #if defined(D3Q19)
        feq = W[1] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1])
            + 4.5f * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1]) * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1])
        );
        #elif defined(D3Q27)
        feq = W[1] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1])
            + 4.5f * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1]) * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1])
            + 4.5f * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1]) * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1]) * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1])
        );
        #endif
        fneq = pop1 - feq;
        pxx = fneq;

        // pop2
        #if defined(D3Q19)
        feq = W[2] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2])
            + 4.5f * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2]) * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2])
        );
        #elif defined(D3Q27)
        feq = W[2] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2])
            + 4.5f * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2]) * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2])
            + 4.5f * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2]) * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2]) * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2])
        );
        #endif
        fneq = pop2 - feq;
        pxx += fneq;

        // pop3
        #if defined(D3Q19)
        feq = W[3] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3])
            + 4.5f * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3]) * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3])
        );
        #elif defined(D3Q27)
        feq = W[3] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3])
            + 4.5f * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3]) * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3])
            + 4.5f * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3]) * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3]) * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3])
        );
        #endif
        fneq = pop3 - feq;
        pyy = fneq;

        // pop4
        #if defined(D3Q19)
        feq = W[4] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4])
            + 4.5f * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4]) * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4])
        );
        #elif defined(D3Q27)
        feq = W[4] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4])
            + 4.5f * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4]) * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4])
            + 4.5f * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4]) * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4]) * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4])
        );
        #endif
        fneq = pop4 - feq;
        pyy += fneq;

        // pop5
        #if defined(D3Q19)
        feq = W[5] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5])
            + 4.5f * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5]) * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5])
        );
        #elif defined(D3Q27)
        feq = W[5] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5])
            + 4.5f * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5]) * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5])
            + 4.5f * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5]) * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5]) * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5])
        );
        #endif
        fneq = pop5 - feq;
        pzz = fneq;

        // pop6
        #if defined(D3Q19)
        feq = W[6] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6])
            + 4.5f * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6]) * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6])
        );
        #elif defined(D3Q27)
        feq = W[6] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6])
            + 4.5f * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6]) * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6])
            + 4.5f * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6]) * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6]) * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6])
        );
        #endif
        fneq = pop6 - feq;
        pzz += fneq;

        // pop7
        #if defined(D3Q19)
        feq = W[7] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7])
            + 4.5f * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7]) * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7])
        );
        #elif defined(D3Q27)
        feq = W[7] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7])
            + 4.5f * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7]) * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7])
            + 4.5f * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7]) * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7]) * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7])
        );
        #endif
        fneq = pop7 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pxy = fneq;

        // pop8
        #if defined(D3Q19)
        feq = W[8] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8])
            + 4.5f * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8]) * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8])
        );
        #elif defined(D3Q27)
        feq = W[8] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8])
            + 4.5f * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8]) * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8])
            + 4.5f * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8]) * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8]) * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8])
        );
        #endif
        fneq = pop8 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pxy += fneq;

        // pop9
        #if defined(D3Q19)
        feq = W[9] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9])
            + 4.5f * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9]) * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9])
        );
        #elif defined(D3Q27)
        feq = W[9] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9])
            + 4.5f * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9]) * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9])
            + 4.5f * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9]) * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9]) * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9])
        );
        #endif
        fneq = pop9 - feq;
        pxx += fneq; 
        pzz += fneq; 
        pxz = fneq;

        // pop10
        #if defined(D3Q19)
        feq = W[10] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10])
            + 4.5f * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10]) * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10])
        );
        #elif defined(D3Q27)
        feq = W[10] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10])
            + 4.5f * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10]) * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10])
            + 4.5f * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10]) * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10]) * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10])
        );
        #endif
        fneq = pop10 - feq;
        pxx += fneq; 
        pzz += fneq; 
        pxz += fneq;

        // pop11
        #if defined(D3Q19)
        feq = W[11] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11])
            + 4.5f * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11]) * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11])
        );
        #elif defined(D3Q27)
        feq = W[11] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11])
            + 4.5f * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11]) * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11])
            + 4.5f * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11]) * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11]) * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11])
        );
        #endif
        fneq = pop11 - feq;
        pyy += fneq;
        pzz += fneq; 
        pyz = fneq;

        // pop12
        #if defined(D3Q19)
        feq = W[12] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12])
            + 4.5f * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12]) * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12])
        );
        #elif defined(D3Q27)
        feq = W[12] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12])
            + 4.5f * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12]) * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12])
            + 4.5f * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12]) * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12]) * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12])
        );
        #endif
        fneq = pop12 - feq;
        pyy += fneq; 
        pzz += fneq; 
        pyz += fneq;

        // pop13
        #if defined(D3Q19)
        feq = W[13] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13])
            + 4.5f * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13]) * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13])
        );
        #elif defined(D3Q27)
        feq = W[13] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13])
            + 4.5f * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13]) * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13])
            + 4.5f * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13]) * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13]) * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13])
        );
        #endif
        fneq = pop13 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pxy -= fneq;

        // pop14
        #if defined(D3Q19)
        feq = W[14] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14])
            + 4.5f * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14]) * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14])
        );
        #elif defined(D3Q27)
        feq = W[14] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14])
            + 4.5f * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14]) * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14])
            + 4.5f * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14]) * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14]) * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14])
        );
        #endif
        fneq = pop14 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pxy -= fneq;

        // pop15
        #if defined(D3Q19)
        feq = W[15] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15])
            + 4.5f * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15]) * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15])
        );
        #elif defined(D3Q27)
        feq = W[15] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15])
            + 4.5f * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15]) * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15])
            + 4.5f * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15]) * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15]) * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15])
        );
        #endif
        fneq = pop15 - feq;
        pxx += fneq; 
        pzz += fneq; 
        pxz -= fneq;

        // pop16
        #if defined(D3Q19)
        feq = W[16] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16])
            + 4.5f * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16]) * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16])
        );
        #elif defined(D3Q27)
        feq = W[16] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16])
            + 4.5f * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16]) * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16])
            + 4.5f * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16]) * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16]) * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16])
        );
        #endif
        fneq = pop16 - feq;
        pxx += fneq; 
        pzz += fneq; 
        pxz -= fneq;

        // pop17
        #if defined(D3Q19)
        feq = W[17] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17])
            + 4.5f * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17]) * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17])
        );
        #elif defined(D3Q27)
        feq = W[17] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17])
            + 4.5f * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17]) * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17])
            + 4.5f * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17]) * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17]) * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17])
        );
        #endif
        fneq = pop17 - feq;
        pyy += fneq; 
        pzz += fneq; 
        pyz -= fneq;

        // pop18
        #if defined(D3Q19)
        feq = W[18] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18])
            + 4.5f * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18]) * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18])
        );
        #elif defined(D3Q27)
        feq = W[18] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18])
            + 4.5f * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18]) * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18])
            + 4.5f * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18]) * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18]) * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18])
        );
        #endif
        fneq = pop18 - feq;
        pyy += fneq; 
        pzz += fneq; 
        pyz -= fneq;
        
        // pop19
        #if defined(D3Q27)
        feq = W[19] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19])
            + 4.5f * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19]) * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19])
            + 4.5f * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19]) * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19]) * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19])
        );
        fneq = pop19 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy += fneq; 
        pxz += fneq; 
        pyz += fneq;

        // pop20
        feq = W[20] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20])
            + 4.5f * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20]) * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20])
            + 4.5f * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20]) * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20]) * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20])
        );
        fneq = pop20 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy += fneq; 
        pxz += fneq; 
        pyz += fneq;

        // pop21
        feq = W[21] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21])
            + 4.5f * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21]) * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21])
            + 4.5f * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21]) * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21]) * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21])
        );
        fneq = pop21 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy += fneq; 
        pxz -= fneq; 
        pyz -= fneq;

        // pop22
        feq = W[22] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22])
            + 4.5f * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22]) * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22])
            + 4.5f * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22]) * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22]) * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22])
        );
        fneq = pop22 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy += fneq; 
        pxz -= fneq; 
        pyz -= fneq; 

        // pop23
        feq = W[23] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23])
            + 4.5f * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23]) * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23])
            + 4.5f * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23]) * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23]) * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23])
        );
        fneq = pop23 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy -= fneq; 
        pxz += fneq; 
        pyz -= fneq;

        // pop24
        feq = W[24] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24])
            + 4.5f * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24]) * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24])
            + 4.5f * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24]) * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24]) * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24])
        );
        fneq = pop24 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy -= fneq; 
        pxz += fneq; 
        pyz -= fneq;

        // pop25
        feq = W[25] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25])
            + 4.5f * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25]) * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25])
            + 4.5f * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25]) * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25]) * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25])
        );
        fneq = pop25 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy -= fneq; 
        pxz -= fneq; 
        pyz += fneq;

        // pop26
        feq = W[26] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26])
            + 4.5f * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26]) * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26])
            + 4.5f * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26]) * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26]) * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26])
        );
        fneq = pop26 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy -= fneq; 
        pxz -= fneq; 
        pyz += fneq;
        #endif 
    // =============================================== END =============================================== //

    pxx += CSSQ;
    pyy += CSSQ;
    pzz += CSSQ;

    d.pxx[idx3] = pxx;
    d.pyy[idx3] = pyy;
    d.pzz[idx3] = pzz;
    d.pxy[idx3] = pxy;
    d.pxz[idx3] = pxz;   
    d.pyz[idx3] = pyz;

    #if defined(VISC_CONTRAST)

        const float phi = d.phi[idx3]; 
        nuLocal = fmaf(phi, (VISC_OIL - VISC_WATER), VISC_WATER);

        const float omegaPhys = 1.0f / (0.5f + 3.0f * nuLocal);

        #if defined(JET) 
            const float omegaLocal = fminf(omegaPhys, cubicSponge(z));
        #elif defined(DROPLET) 
            const float omegaLocal = omegaPhys;
        #endif

    #else

        #if defined(JET)
            const float omegaLocal = cubicSponge(z);
        #elif defined(DROPLET)
            const float omegaLocal = OMEGA_REF;
        #endif

    #endif

    const float omcoLocal = 1.0f - omegaLocal;
    const float coeffForce = 1.0f - 0.5f * omegaLocal;

    // =============================================== FUSED COLLISION-STREAMING =============================================== //

        //             0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
        // CIX[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
        // CIY[27] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1 };
        // CIZ[27] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1 };

        float forceCorr;

        // ========================== ZERO ========================== //
        #if defined(D3Q19)
        feq = W[0] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[0] + uy*CIY[0] + uz*CIZ[0])
            + 4.5f * (ux*CIX[0] + uy*CIY[0] + uz*CIZ[0]) * (ux*CIX[0] + uy*CIY[0] + uz*CIZ[0])
        ) - W[0];
        #elif defined(D3Q27)
        feq = W[0] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[0] + uy*CIY[0] + uz*CIZ[0])
            + 4.5f * (ux*CIX[0] + uy*CIY[0] + uz*CIZ[0]) * (ux*CIX[0] + uy*CIY[0] + uz*CIZ[0])
            + 4.5f * (ux*CIX[0] + uy*CIY[0] + uz*CIZ[0]) * (ux*CIX[0] + uy*CIY[0] + uz*CIZ[0]) * (ux*CIX[0] + uy*CIY[0] + uz*CIZ[0])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[0] + uy*CIY[0] + uz*CIZ[0])
        ) - W[0];
        #endif
        #if defined(D3Q19)
        forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
        forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                    (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                    (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,0);
        d.f[idx3] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== ONE ========================== //
        #if defined(D3Q19)
        feq = W[1] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1])
            + 4.5f * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1]) * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1])
        ) - W[1];
        #elif defined(D3Q27)
        feq = W[1] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1])
            + 4.5f * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1]) * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1])
            + 4.5f * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1]) * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1]) * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[1] + uy*CIY[1] + uz*CIZ[1])
        ) - W[1];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,1);
        d.f[global4(x+1,y,z,1)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWO ========================== //
        #if defined(D3Q19)
        feq = W[2] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2])
            + 4.5f * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2]) * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2])
        ) - W[2];
        #elif defined(D3Q27)
        feq = W[2] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2])
            + 4.5f * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2]) * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2])
            + 4.5f * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2]) * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2]) * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[2] + uy*CIY[2] + uz*CIZ[2])
        ) - W[2];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,2);
        d.f[global4(x-1,y,z,2)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== THREE ========================== //
        #if defined(D3Q19)
        feq = W[3] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3])
            + 4.5f * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3]) * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3])
        ) - W[3];
        #elif defined(D3Q27)
        feq = W[3] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3])
            + 4.5f * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3]) * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3])
            + 4.5f * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3]) * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3]) * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[3] + uy*CIY[3] + uz*CIZ[3])
        ) - W[3];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,3);
        d.f[global4(x,y+1,z,3)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== FOUR ========================== //
        #if defined(D3Q19)
        feq = W[4] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4])
            + 4.5f * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4]) * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4])
        ) - W[4];
        #elif defined(D3Q27)
        feq = W[4] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4])
            + 4.5f * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4]) * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4])
            + 4.5f * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4]) * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4]) * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[4] + uy*CIY[4] + uz*CIZ[4])
        ) - W[4];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,4);
        d.f[global4(x,y-1,z,4)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== FIVE ========================== //
        #if defined(D3Q19)
        feq = W[5] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5])
            + 4.5f * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5]) * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5])
        ) - W[5];
        #elif defined(D3Q27)
        feq = W[5] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5])
            + 4.5f * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5]) * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5])
            + 4.5f * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5]) * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5]) * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[5] + uy*CIY[5] + uz*CIZ[5])
        ) - W[5];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,5);
        d.f[global4(x,y,z+1,5)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== SIX ========================== //
        #if defined(D3Q19)
        feq = W[6] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6])
            + 4.5f * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6]) * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6])
        ) - W[6];
        #elif defined(D3Q27)
        feq = W[6] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6])
            + 4.5f * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6]) * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6])
            + 4.5f * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6]) * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6]) * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[6] + uy*CIY[6] + uz*CIZ[6])
        ) - W[6];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,6);
        d.f[global4(x,y,z-1,6)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== SEVEN ========================== //
        #if defined(D3Q19)
        feq = W[7] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7])
            + 4.5f * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7]) * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7])
        ) - W[7];
        #elif defined(D3Q27)
        feq = W[7] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7])
            + 4.5f * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7]) * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7])
            + 4.5f * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7]) * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7]) * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[7] + uy*CIY[7] + uz*CIZ[7])
        ) - W[7];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,7);
        d.f[global4(x+1,y+1,z,7)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== EIGHT ========================== //
        #if defined(D3Q19)
        feq = W[8] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8])
            + 4.5f * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8]) * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8])
        );
        #elif defined(D3Q27)
        feq = W[8] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8])
            + 4.5f * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8]) * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8])
            + 4.5f * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8]) * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8]) * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[8] + uy*CIY[8] + uz*CIZ[8])
        ) - W[8];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,8);
        d.f[global4(x-1,y-1,z,8)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== NINE ========================== //
        #if defined(D3Q19)
        feq = W[9] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9])
            + 4.5f * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9]) * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9])
        ) - W[9];
        #elif defined(D3Q27)
        feq = W[9] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9])
            + 4.5f * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9]) * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9])
            + 4.5f * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9]) * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9]) * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[9] + uy*CIY[9] + uz*CIZ[9])
        ) - W[9];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,9);
        d.f[global4(x+1,y,z+1,9)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TEN ========================== //
        #if defined(D3Q19)
        feq = W[10] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10])
            + 4.5f * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10]) * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10])
        ) - W[10];
        #elif defined(D3Q27)
        feq = W[10] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10])
            + 4.5f * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10]) * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10])
            + 4.5f * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10]) * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10]) * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[10] + uy*CIY[10] + uz*CIZ[10])
        ) - W[10];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,10);
        d.f[global4(x-1,y,z-1,10)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== ELEVEN ========================== //
        #if defined(D3Q19)
        feq = W[11] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11])
            + 4.5f * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11]) * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11])
        ) - W[11];
        #elif defined(D3Q27)
        feq = W[11] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11])
            + 4.5f * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11]) * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11])
            + 4.5f * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11]) * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11]) * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[11] + uy*CIY[11] + uz*CIZ[11])
        ) - W[11];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,11);
        d.f[global4(x,y+1,z+1,11)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWELVE ========================== //
        #if defined(D3Q19)
        feq = W[12] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12])
            + 4.5f * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12]) * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12])
        ) - W[12];
        #elif defined(D3Q27)
        feq = W[12] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12])
            + 4.5f * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12]) * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12])
            + 4.5f * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12]) * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12]) * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[12] + uy*CIY[12] + uz*CIZ[12])
        ) - W[12];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,12);
        d.f[global4(x,y-1,z-1,12)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== THIRTEEN ========================== //
        #if defined(D3Q19)
        feq = W[13] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13])
            + 4.5f * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13]) * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13])
        ) - W[13];
        #elif defined(D3Q27)
        feq = W[13] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13])
            + 4.5f * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13]) * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13])
            + 4.5f * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13]) * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13]) * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[13] + uy*CIY[13] + uz*CIZ[13])
        ) - W[13];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,13);
        d.f[global4(x+1,y-1,z,13)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== FOURTEEN ========================== //
        #if defined(D3Q19)
        feq = W[14] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14])
            + 4.5f * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14]) * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14])
        ) - W[14];
        #elif defined(D3Q27)
        feq = W[14] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14])
            + 4.5f * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14]) * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14])
            + 4.5f * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14]) * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14]) * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[14] + uy*CIY[14] + uz*CIZ[14])
        ) - W[14];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,14);
        d.f[global4(x-1,y+1,z,14)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== FIFTEEN ========================== //
        #if defined(D3Q19)
        feq = W[15] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15])
            + 4.5f * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15]) * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15])
        ) - W[15];
        #elif defined(D3Q27)
        feq = W[15] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15])
            + 4.5f * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15]) * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15])
            + 4.5f * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15]) * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15]) * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[15] + uy*CIY[15] + uz*CIZ[15])
        ) - W[15];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,15);
        d.f[global4(x+1,y,z-1,15)] = to_pop(feq + omcoLocal * fneq + forceCorr); 

        // ========================== SIXTEEN ========================== //
        #if defined(D3Q19)
        feq = W[16] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16])
            + 4.5f * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16]) * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16])
        ) - W[16];
        #elif defined(D3Q27)
        feq = W[16] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16])
            + 4.5f * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16]) * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16])
            + 4.5f * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16]) * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16]) * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[16] + uy*CIY[16] + uz*CIZ[16])
        ) - W[16];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,16);
        d.f[global4(x-1,y,z+1,16)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== SEVENTEEN ========================== //
        #if defined(D3Q19)
        feq = W[17] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17])
            + 4.5f * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17]) * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17])
        ) - W[17];
        #elif defined(D3Q27)
        feq = W[17] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17])
            + 4.5f * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17]) * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17])
            + 4.5f * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17]) * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17]) * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[17] + uy*CIY[17] + uz*CIZ[17])
        ) - W[17];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,17);
        d.f[global4(x,y+1,z-1,17)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== EIGHTEEN ========================== //
        #if defined(D3Q19)
        feq = W[18] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18])
            + 4.5f * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18]) * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18])
        ) - W[18];
        #elif defined(D3Q27)
        feq = W[18] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18])
            + 4.5f * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18]) * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18])
            + 4.5f * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18]) * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18]) * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[18] + uy*CIY[18] + uz*CIZ[18])
        ) - W[18];
        #endif
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,18);
        d.f[global4(x,y-1,z+1,18)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== NINETEEN ========================== //
        #if defined(D3Q27)
        feq = W[19] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19])
            + 4.5f * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19]) * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19])
            + 4.5f * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19]) * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19]) * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[19] + uy*CIY[19] + uz*CIZ[19])
        ) - W[19];
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,19);
        d.f[global4(x+1,y+1,z+1,19)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWENTY ========================== //
        feq = W[20] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20])
            + 4.5f * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20]) * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20])
            + 4.5f * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20]) * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20]) * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[20] + uy*CIY[20] + uz*CIZ[20])
        ) - W[20];
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,20);
        d.f[global4(x-1,y-1,z-1,20)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWENTY ONE ========================== //
        feq = W[21] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21])
            + 4.5f * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21]) * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21])
            + 4.5f * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21]) * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21]) * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[21] + uy*CIY[21] + uz*CIZ[21])
        ) - W[21];
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,21);
        d.f[global4(x+1,y+1,z-1,21)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWENTY TWO ========================== //
        feq = W[22] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22])
            + 4.5f * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22]) * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22])
            + 4.5f * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22]) * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22]) * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[22] + uy*CIY[22] + uz*CIZ[22])
        ) - W[22];
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,22);
        d.f[global4(x-1,y-1,z+1,22)] = to_pop(feq + omcoLocal * fneq + forceCorr);    

        // ========================== TWENTY THREE ========================== //
        feq = W[23] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23])
            + 4.5f * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23]) * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23])
            + 4.5f * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23]) * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23]) * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[23] + uy*CIY[23] + uz*CIZ[23])
        ) - W[23];
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,23);
        d.f[global4(x+1,y-1,z+1,23)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWENTY FOUR ========================== //
        feq = W[24] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24])
            + 4.5f * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24]) * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24])
            + 4.5f * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24]) * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24]) * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[24] + uy*CIY[24] + uz*CIZ[24])
        ) - W[24];
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,24);
        d.f[global4(x-1,y+1,z-1,24)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWENTY FIVE ========================== //
        feq = W[25] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25])
            + 4.5f * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25]) * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25])
            + 4.5f * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25]) * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25]) * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[25] + uy*CIY[25] + uz*CIZ[25])
        ) - W[25];
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,25);
        d.f[global4(x-1,y+1,z+1,25)] = to_pop(feq + omcoLocal * fneq + forceCorr);    
        
        // ========================== TWENTY SIX ========================== //
        feq = W[26] * rho * (
            1.0f
            - 1.5f * (ux*ux + uy*uy + uz*uz)
            + 3.0f * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26])
            + 4.5f * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26]) * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26])
            + 4.5f * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26]) * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26]) * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26])
            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux*CIX[26] + uy*CIY[26] + uz*CIZ[26])
        ) - W[26];
        #if defined(D3Q19)
            forceCorr = coeff * feq * ((CIX[Q] - ux) * ffx +
                                (CIY[Q] - uy) * ffy +
                                (CIZ[Q] - uz) * ffz) * aux;
        #elif defined(D3Q27)
            forceCorr = coeff * W[Q] * ((3.0f * (CIX[Q] - ux) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIX[Q] ) * ffx +
                                (3.0f * (CIY[Q] - uy) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIY[Q] ) * ffy +
                                (3.0f * (CIZ[Q] - uz) + 3.0f * 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]) * CIZ[Q] ) * ffz);
        #endif 
        fneq = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,26);
        d.f[global4(x+1,y-1,z-1,26)] = to_pop(feq + omcoLocal * fneq + forceCorr);
        #endif 
    // =

    #if !defined(VISC_CONTRAST)
    const float phi = d.phi[idx3];
    #endif

    d.g[idx3] = W_G_1 * phi;

    const float phiNorm = W_G_2 * GAMMA * phi * (1.0f - phi);
    const float multPhi = W_G_2 * phi;
    const float a3 = 3.0f * multPhi;

    feq = multPhi + a3 * ux;
    forceCorr = phiNorm * d.normx[idx3];
    d.g[global4(x+1,y,z,1)] = feq + forceCorr;
    
    feq = multPhi - a3 * ux;
    d.g[global4(x-1,y,z,2)] = feq - forceCorr;

    feq = multPhi + a3 * uy;
    forceCorr = phiNorm * d.normy[idx3];
    d.g[global4(x,y+1,z,3)] = feq + forceCorr;

    feq = multPhi - a3 * uy;
    d.g[global4(x,y-1,z,4)] = feq - forceCorr;

    feq = multPhi + a3 * uz;
    forceCorr = phiNorm * d.normz[idx3];
    d.g[global4(x,y,z+1,5)] = feq + forceCorr;

    feq = multPhi - a3 * uz;
    d.g[global4(x,y,z-1,6)] = feq - forceCorr;
}

#endif




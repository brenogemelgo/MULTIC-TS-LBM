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



    // =============================================== SECOND-ORDER MOMENTUM FLUX TENSOR =============================================== //

        //             0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
        // CIX[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
        // CIY[27] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1 };
        // CIZ[27] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1 };

        float feq, fneq, pxx, pyy, pzz, pxy, pxz, pyz;

        // pop1 (c = +x̂)
        #if defined(D3Q19)
        feq = W_1 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * ux
                        + 4.5f * ux * ux);
        #elif defined(D3Q27)
        feq = W_1 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * ux
                        + 4.5f * ux * ux
                        + 4.5f * ux * ux * ux
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * ux);
        #endif
        fneq = pop1 - feq;
        pxx = fneq;

        // pop2 (c = -x̂)
        #if defined(D3Q19)
        feq = W_1 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        - 3.0f * ux
                        + 4.5f * ux * ux);
        #elif defined(D3Q27)
        feq = W_1 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        - 3.0f * ux
                        + 4.5f * ux * ux
                        - 4.5f * ux * ux * ux
                        + 4.5f * (ux*ux + uy*uy + uz*uz) * ux);
        #endif
        fneq = pop2 - feq;
        pxx += fneq;

        // pop3 (c = +ŷ)
        #if defined(D3Q19)
        feq = W_1 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * uy
                        + 4.5f * uy * uy);
        #elif defined(D3Q27)
        feq = W_1 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * uy
                        + 4.5f * uy * uy
                        + 4.5f * uy * uy * uy
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * uy);
        #endif
        fneq = pop3 - feq;
        pyy = fneq;

        // pop4 (c = -ŷ)
        #if defined(D3Q19)
        feq = W_1 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        - 3.0f * uy
                        + 4.5f * uy * uy);
        #elif defined(D3Q27)
        feq = W_1 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        - 3.0f * uy
                        + 4.5f * uy * uy
                        - 4.5f * uy * uy * uy
                        + 4.5f * (ux*ux + uy*uy + uz*uz) * uy);
        #endif
        fneq = pop4 - feq;
        pyy += fneq;

        // pop5 (c = +ẑ)
        #if defined(D3Q19)
        feq = W_1 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * uz
                        + 4.5f * uz * uz);
        #elif defined(D3Q27)
        feq = W_1 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * uz
                        + 4.5f * uz * uz
                        + 4.5f * uz * uz * uz
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * uz);
        #endif
        fneq = pop5 - feq;
        pzz = fneq;

        // pop6 (c = -ẑ)
        #if defined(D3Q19)
        feq = W_1 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        - 3.0f * uz
                        + 4.5f * uz * uz);
        #elif defined(D3Q27)
        feq = W_1 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        - 3.0f * uz
                        + 4.5f * uz * uz
                        - 4.5f * uz * uz * uz
                        + 4.5f * (ux*ux + uy*uy + uz*uz) * uz);
        #endif
        fneq = pop6 - feq;
        pzz += fneq;

        // pop7 (c = +x̂ +ŷ)
        #if defined(D3Q19)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (ux + uy)
                        + 4.5f * (ux + uy) * (ux + uy));
        #elif defined(D3Q27)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (ux + uy)
                        + 4.5f * (ux + uy) * (ux + uy)
                        + 4.5f * (ux + uy) * (ux + uy) * (ux + uy)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uy));
        #endif
        fneq = pop7 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pxy = fneq;

        // pop8 (c = -x̂ -ŷ)
        #if defined(D3Q19)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        - 3.0f * (ux + uy)
                        + 4.5f * (ux + uy) * (ux + uy));
        #elif defined(D3Q27)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        - 3.0f * (ux + uy)
                        + 4.5f * (ux + uy) * (ux + uy)
                        - 4.5f * (ux + uy) * (ux + uy) * (ux + uy)
                        + 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uy));
        #endif
        fneq = pop8 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pxy += fneq;

        // pop9 (c = +x̂ +ẑ)
        #if defined(D3Q19)
        feq = W_2 * rho * ( 1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (ux + uz)
                        + 4.5f * (ux + uz) * (ux + uz));
        #elif defined(D3Q27)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (ux + uz)
                        + 4.5f * (ux + uz) * (ux + uz)
                        + 4.5f * (ux + uz) * (ux + uz) * (ux + uz)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uz));
        #endif
        fneq = pop9 - feq;
        pxx += fneq; 
        pzz += fneq; 
        pxz = fneq;

        // pop10 (c = -x̂ -ẑ)
        #if defined(D3Q19)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        - 3.0f * (ux + uz)
                        + 4.5f * (ux + uz) * (ux + uz));
        #elif defined(D3Q27)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        - 3.0f * (ux + uz)
                        + 4.5f * (ux + uz) * (ux + uz)
                        - 4.5f * (ux + uz) * (ux + uz) * (ux + uz)
                        + 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uz));
        #endif
        fneq = pop10 - feq;
        pxx += fneq; 
        pzz += fneq; 
        pxz += fneq;

        // pop11 (c = +ŷ +ẑ)
        #if defined(D3Q19)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (uy + uz)
                        + 4.5f * (uy + uz) * (uy + uz));
        #elif defined(D3Q27)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (uy + uz)
                        + 4.5f * (uy + uz) * (uy + uz)
                        + 4.5f * (uy + uz) * (uy + uz) * (uy + uz)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (uy + uz));
        #endif
        fneq = pop11 - feq;
        pyy += fneq;
        pzz += fneq; 
        pyz = fneq;

        // pop12 (c = -ŷ -ẑ)
        #if defined(D3Q19)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        - 3.0f * (uy + uz)
                        + 4.5f * (uy + uz) * (uy + uz));
        #elif defined(D3Q27)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        - 3.0f * (uy + uz)
                        + 4.5f * (uy + uz) * (uy + uz)
                        - 4.5f * (uy + uz) * (uy + uz) * (uy + uz)
                        + 4.5f * (ux*ux + uy*uy + uz*uz) * (uy + uz));
        #endif
        fneq = pop12 - feq;
        pyy += fneq; 
        pzz += fneq; 
        pyz += fneq;

        // pop13 (c = +x̂ -ŷ)
        #if defined(D3Q19)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (ux - uy)
                        + 4.5f * (ux - uy) * (ux - uy));
        #elif defined(D3Q27)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (ux - uy)
                        + 4.5f * (ux - uy) * (ux - uy)
                        + 4.5f * (ux - uy) * (ux - uy) * (ux - uy)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux - uy));
        #endif
        fneq = pop13 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pxy -= fneq;

        // pop14 (c = -x̂ +ŷ)
        #if defined(D3Q19)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (uy - ux)
                        + 4.5f * (uy - ux) * (uy - ux));
        #elif defined(D3Q27)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (uy - ux)
                        + 4.5f * (uy - ux) * (uy - ux)
                        + 4.5f * (uy - ux) * (uy - ux) * (uy - ux)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (uy - ux));
        #endif
        fneq = pop14 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pxy -= fneq;

        // pop15 (c = +x̂ -ẑ)
        #if defined(D3Q19)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (ux - uz)
                        + 4.5f * (ux - uz) * (ux - uz));
        #elif defined(D3Q27)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (ux - uz)
                        + 4.5f * (ux - uz) * (ux - uz)
                        + 4.5f * (ux - uz) * (ux - uz) * (ux - uz)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux - uz));
        #endif
        fneq = pop15 - feq;
        pxx += fneq; 
        pzz += fneq; 
        pxz -= fneq;

        // pop16 (c = -x̂ +ẑ)
        #if defined(D3Q19)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (uz - ux)
                        + 4.5f * (uz - ux) * (uz - ux));
        #elif defined(D3Q27)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (uz - ux)
                        + 4.5f * (uz - ux) * (uz - ux)
                        + 4.5f * (uz - ux) * (uz - ux) * (uz - ux)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (uz - ux));
        #endif
        fneq = pop16 - feq;
        pxx += fneq; 
        pzz += fneq; 
        pxz -= fneq;

        // pop17 (c = +ŷ -ẑ)
        #if defined(D3Q19)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (uy - uz)
                        + 4.5f * (uy - uz) * (uy - uz));
        #elif defined(D3Q27)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (uy - uz)
                        + 4.5f * (uy - uz) * (uy - uz)
                        + 4.5f * (uy - uz) * (uy - uz) * (uy - uz)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (uy - uz));
        #endif
        fneq = pop17 - feq;
        pyy += fneq; 
        pzz += fneq; 
        pyz -= fneq;

        // pop18 (c = -ŷ +ẑ)
        #if defined(D3Q19)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (uz - uy)
                        + 4.5f * (uz - uy) * (uz - uy));
        #elif defined(D3Q27)
        feq = W_2 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (uz - uy)
                        + 4.5f * (uz - uy) * (uz - uy)
                        + 4.5f * (uz - uy) * (uz - uy) * (uz - uy)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (uz - uy));
        #endif
        fneq = pop18 - feq;
        pyy += fneq; 
        pzz += fneq; 
        pyz -= fneq;
        
        #if defined(D3Q27)
        // pop19 (c = +x̂ +ŷ +ẑ)
        feq = W_3 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (ux + uy + uz)
                        + 4.5f * (ux + uy + uz) * (ux + uy + uz)
                        + 4.5f * (ux + uy + uz) * (ux + uy + uz) * (ux + uy + uz)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uy + uz));
        fneq = pop19 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy += fneq; 
        pxz += fneq; 
        pyz += fneq;

        // pop20 (c = -x̂ -ŷ -ẑ)
        feq = W_3 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        - 3.0f * (ux + uy + uz)
                        + 4.5f * (ux + uy + uz) * (ux + uy + uz)
                        - 4.5f * (ux + uy + uz) * (ux + uy + uz) * (ux + uy + uz)
                        + 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uy + uz));
        fneq = pop20 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy += fneq; 
        pxz += fneq; 
        pyz += fneq;

        // pop21 (c = +x̂ +ŷ -ẑ)
        feq = W_3 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (ux + uy - uz)
                        + 4.5f * (ux + uy - uz) * (ux + uy - uz)
                        + 4.5f * (ux + uy - uz) * (ux + uy - uz) * (ux + uy - uz)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uy - uz));
        fneq = pop21 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy += fneq; 
        pxz -= fneq; 
        pyz -= fneq;

        // pop22 (c = -x̂ -ŷ +ẑ)
        feq = W_3 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (uz - ux - uy)
                        + 4.5f * (uz - ux - uy) * (uz - ux - uy)
                        + 4.5f * (uz - ux - uy) * (uz - ux - uy) * (uz - ux - uy)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (uz - ux - uy));
        fneq = pop22 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy += fneq; 
        pxz -= fneq; 
        pyz -= fneq; 

        // pop23 (c = +x̂ -ŷ +ẑ)
        feq = W_3 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (ux - uy + uz)
                        + 4.5f * (ux - uy + uz) * (ux - uy + uz)
                        + 4.5f * (ux - uy + uz) * (ux - uy + uz) * (ux - uy + uz)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux - uy + uz));
        fneq = pop23 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy -= fneq; 
        pxz += fneq; 
        pyz -= fneq;

        // pop24 (c = -x̂ +ŷ -ẑ)
        feq = W_3 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (uy - ux - uz)
                        + 4.5f * (uy - ux - uz) * (uy - ux - uz)
                        + 4.5f * (uy - ux - uz) * (uy - ux - uz) * (uy - ux - uz)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (uy - ux - uz));
        fneq = pop24 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy -= fneq; 
        pxz += fneq; 
        pyz -= fneq;

        // pop25 (c = -x̂ +ŷ +ẑ)
        feq = W_3 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (uy - ux + uz)
                        + 4.5f * (uy - ux + uz) * (uy - ux + uz)
                        + 4.5f * (uy - ux + uz) * (uy - ux + uz) * (uy - ux + uz)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (uy - ux + uz));
        fneq = pop25 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy -= fneq; 
        pxz -= fneq; 
        pyz += fneq;

        // pop26 (c = +x̂ -ŷ -ẑ)
        feq = W_3 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * (ux - uy - uz)
                        + 4.5f * (ux - uy - uz) * (ux - uy - uz)
                        + 4.5f * (ux - uy - uz) * (ux - uy - uz) * (ux - uy - uz)
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux - uy - uz));
        fneq = pop26 - feq;
        pxx += fneq; 
        pyy += fneq; 
        pzz += fneq;
        pxy -= fneq; 
        pxz -= fneq; 
        pyz += fneq;
        #endif 

        pxx += CSSQ;
        pyy += CSSQ;
        pzz += CSSQ;

        d.pxx[idx3] = pxx;
        d.pyy[idx3] = pyy;
        d.pzz[idx3] = pzz;
        d.pxy[idx3] = pxy;
        d.pxz[idx3] = pxz;   
        d.pyz[idx3] = pyz;

    // ============================================================= END ============================================================= //



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

        #if defined(D3Q19)
            
            // check if right. brute force it again
            const float invRhoCssq = 3.0f * invRho;

        // ========================== ZERO ========================== //

            feq = W_0 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz)) - W_0;
            forceCorr = -coeffForce * feq * (ux * ffx + uy * ffy + uz * ffz) * invRhoCssq;
            fneq = (W_0 * 4.5f) * (-CSSQ * pxx - CSSQ * pyy - CSSQ * pzz);
            d.f[idx3] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== ONE ========================== //

            feq = W_1 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) + 3.0f * ux + 4.5f * ux * ux) - W_1;
            forceCorr = coeffForce * feq * ((1.0f - ux) * ffx - uy * ffy - uz * ffz) * invRhoCssq;
            fneq = (W_1 * 4.5f) * (CSCO * pxx - CSSQ * pyy - CSSQ * pzz);
            d.f[global4(x+1,y,z,1)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWO ========================== //

            feq = W_1 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) - 3.0f * ux + 4.5f * ux * ux) - W_1;
            forceCorr = coeffForce * feq * ((-1.0f - ux) * ffx - uy * ffy - uz * ffz) * invRhoCssq;
            fneq = (W_1 * 4.5f) * (CSCO * pxx - CSSQ * pyy - CSSQ * pzz);
            d.f[global4(x-1,y,z,2)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== THREE ========================== //

            feq = W_1 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) + 3.0f * uy + 4.5f * uy * uy) - W_1;
            forceCorr = coeffForce * feq * (-ux * ffx + (1.0f - uy) * ffy - uz * ffz ) * invRhoCssq;
            fneq = (W_1 * 4.5f) * (-CSSQ * pxx + CSCO * pyy - CSSQ * pzz );
            d.f[global4(x,y+1,z,3)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== FOUR ========================== //

            feq = W_1 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) - 3.0f * uy + 4.5f * uy * uy) - W_1;
            forceCorr = coeffForce * feq * (-ux * ffx + (-1.0f - uy) * ffy - uz * ffz) * invRhoCssq;
            fneq = (W_1 * 4.5f) * (-CSSQ * pxx + CSCO * pyy - CSSQ * pzz);
            d.f[global4(x,y-1,z,4)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== FIVE ========================== //

            feq = W_1 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) + 3.0f * uz + 4.5f * uz * uz) - W_1;
            forceCorr = coeffForce * feq * (-ux * ffx - uy * ffy + (1.0f - uz) * ffz) * invRhoCssq;
            fneq = (W_1 * 4.5f) * (-CSSQ * pxx - CSSQ * pyy + CSCO * pzz);
            d.f[global4(x,y,z+1,5)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== SIX ========================== //

            feq = W_1 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) - 3.0f * uz + 4.5f * uz * uz ) - W_1;
            forceCorr = coeffForce * feq * (-ux * ffx - uy * ffy + (-1.0f - uz) * ffz) * invRhoCssq;
            fneq = (W_1 * 4.5f) * (-CSSQ * pxx - CSSQ * pyy + CSCO * pzz);
            d.f[global4(x,y,z-1,6)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== SEVEN ========================== //

            feq = W_2 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) + 3.0f * (ux + uy) + 4.5f * (ux + uy) * (ux + uy)) - W_2;
            forceCorr = coeffForce * feq * ((1.0f - ux) * ffx + (1.0f - uy) * ffy - uz * ffz) * invRhoCssq;
            fneq = (W_2 * 4.5f) * (CSCO * pxx + CSCO * pyy - CSSQ * pzz + 2.0f * pxy);
            d.f[global4(x+1,y+1,z,7)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== EIGHT ========================== //

            feq = W_2 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) - 3.0f * (ux + uy) + 4.5f * (ux + uy) * (ux + uy)) - W_2;
            forceCorr = coeffForce * feq * ((-1.0f - ux) * ffx + (-1.0f - uy) * ffy - uz * ffz) * invRhoCssq;
            fneq = (W_2 * 4.5f) * (CSCO * pxx + CSCO * pyy - CSSQ * pzz + 2.0f * pxy);
            d.f[global4(x-1,y-1,z,8)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== NINE ========================== //

            feq = W_2 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) + 3.0f * (ux + uz) + 4.5f * (ux + uz) * (ux + uz)) - W_2;
            forceCorr = coeffForce * feq * ((1.0f - ux) * ffx - uy * ffy + (1.0f - uz) * ffz) * invRhoCssq;
            fneq = (W_2 * 4.5f) * (CSCO * pxx - CSSQ * pyy + CSCO * pzz + 2.0f * pxz);
            d.f[global4(x+1,y,z+1,9)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TEN ========================== //

            feq = W_2 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) - 3.0f * (ux + uz) + 4.5f * (ux + uz) * (ux + uz)) - W_2;
            forceCorr = coeffForce * feq * ((-1.0f - ux) * ffx - uy * ffy + (-1.0f - uz) * ffz) * invRhoCssq;
            fneq = (W_2 * 4.5f) * (CSCO * pxx - CSSQ * pyy + CSCO * pzz + 2.0f * pxz);
            d.f[global4(x-1,y,z-1,10)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== ELEVEN ========================== //

            feq = W_2 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) + 3.0f * (uy + uz) + 4.5f * (uy + uz) * (uy + uz)) - W_2;
            forceCorr = coeffForce * feq * (-ux * ffx + (1.0f - uy) * ffy + (1.0f - uz) * ffz) * invRhoCssq;
            fneq = (W_2 * 4.5f) * (-CSSQ * pxx + CSCO * pyy + CSCO * pzz + 2.0f * pyz);
            d.f[global4(x,y+1,z+1,11)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWELVE ========================== //

            feq = W_2 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) - 3.0f * (uy + uz) + 4.5f * (uy + uz) * (uy + uz)) - W_2;
            forceCorr = coeffForce * feq * (-ux * ffx + (-1.0f - uy) * ffy + (-1.0f - uz) * ffz) * invRhoCssq;
            fneq = (W_2 * 4.5f) * (-CSSQ * pxx + CSCO * pyy + CSCO * pzz + 2.0f * pyz);
            d.f[global4(x,y-1,z-1,12)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== THIRTEEN ========================== //

            feq = W_2 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) + 3.0f * (ux - uy) + 4.5f * (ux - uy) * (ux - uy)) - W_2;
            forceCorr = coeffForce * feq * ((1.0f - ux) * ffx + (-1.0f - uy) * ffy - uz * ffz) * invRhoCssq;
            fneq = (W_2 * 4.5f) * (CSCO * pxx + CSCO * pyy - CSSQ * pzz - 2.0f * pxy);
            d.f[global4(x+1,y-1,z,13)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== FOURTEEN ========================== //

            feq = W_2 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) + 3.0f * (uy - ux) + 4.5f * (uy - ux) * (uy - ux)) - W_2;
            forceCorr = coeffForce * feq * ((-1.0f - ux) * ffx + (1.0f - uy) * ffy - uz * ffz) * invRhoCssq;
            fneq = (W_2 * 4.5f) * (CSCO * pxx + CSCO * pyy - CSSQ * pzz - 2.0f * pxy);
            d.f[global4(x-1,y+1,z,14)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== FIFTEEN ========================== //

            feq = W_2 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) + 3.0f * (ux - uz) + 4.5f * (ux - uz) * (ux - uz)) - W_2;
            forceCorr = coeffForce * feq * ((1.0f - ux) * ffx - uy * ffy + (-1.0f - uz) * ffz) * invRhoCssq;
            fneq = (W_2 * 4.5f) * (CSCO * pxx - CSSQ * pyy + CSCO * pzz - 2.0f * pxz);
            d.f[global4(x+1,y,z-1,15)] = to_pop(feq + omcoLocal * fneq + forceCorr); 

        // ========================== SIXTEEN ========================== //

            feq = W_2 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) + 3.0f * (uz - ux) + 4.5f * (uz - ux) * (uz - ux)) - W_2;
            forceCorr = coeffForce * feq * ((-1.0f - ux) * ffx - uy * ffy + (1.0f - uz) * ffz) * invRhoCssq;
            fneq = (W_2 * 4.5f) * (CSCO * pxx - CSSQ * pyy + CSCO * pzz - 2.0f * pxz);
            d.f[global4(x-1,y,z+1,16)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== SEVENTEEN ========================== //

            feq = W_2 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) + 3.0f * (uy - uz) + 4.5f * (uy - uz) * (uy - uz)) - W_2;
            forceCorr = coeffForce * feq * (-ux * ffx + (1.0f - uy) * ffy + (-1.0f - uz) * ffz) * invRhoCssq;
            fneq = (W_2 * 4.5f) * (-CSSQ * pxx + CSCO * pyy + CSCO * pzz - 2.0f * pyz);
            d.f[global4(x,y+1,z-1,17)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== EIGHTEEN ========================== //

            feq = W_2 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz) + 3.0f * (uz - uy) + 4.5f * (uz - uy) * (uz - uy)) - W_2;
            forceCorr = coeffForce * feq * (-ux * ffx + (-1.0f - uy) * ffy + (1.0f - uz) * ffz) * invRhoCssq;
            fneq = (W_2 * 4.5f) * (-CSSQ * pxx + CSCO * pyy + CSCO * pzz - 2.0f * pyz);
            d.f[global4(x,y-1,z+1,18)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        #elif defined(D3Q27)

        // ========================== ZERO ========================== //

            feq =  W_0 * rho * (1.0f - 1.5f * (ux*ux + uy*uy + uz*uz)) - W_0;

            forceCorr = coeffForce * W_0 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*0) * 0 ) * ffx +
                               (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*0) * 0 ) * ffy +
                               (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*0) * 0 ) * ffz);

            fneq = (W_0 * 4.5f) * ((0*0 - CSSQ) * pxx +
                                (0*0 - CSSQ) * pyy +
                                (0*0 - CSSQ) * pzz +
                                2.0f * 0 * 0 * pxy +
                                2.0f * 0 * 0 * pxz +
                                2.0f * 0 * 0 * pyz +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                                3.0f * ((0*0*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                        (0*0*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                        (0*0*0 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                        (0*0*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                        (0*0*0 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                        (0*0*0 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (0*0*0) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[idx3] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== ONE ========================== //

            feq = W_1 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * ux
                            + 4.5f * ux * ux
                            + 4.5f * ux * ux * ux
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * ux);

            forceCorr = coeffForce * W_1 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*0) * 1 ) * ffx +
                               (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*0) * 0 ) * ffy +
                               (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*0) * 0 ) * ffz);

            fneq = (W_1 * 4.5f) * ((1*1 - CSSQ) * pxx +
                                (0*0 - CSSQ) * pyy +
                                (0*0 - CSSQ) * pzz +
                                2.0f * 1 * 0 * pxy +
                                2.0f * 1 * 0 * pxz +
                                2.0f * 0 * 0 * pyz +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                                3.0f * ((1*1*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                        (1*1*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                        (1*0*0 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (0*0*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                        (1*0*0 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (0*0*0 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (1*0*0) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x+1,y,z,1)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWO ========================== //

            feq = W_1 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            - 3.0f * ux
                            + 4.5f * ux * ux
                            - 4.5f * ux * ux * ux
                            + 4.5f * (ux*ux + uy*uy + uz*uz) * ux);

            forceCorr = coeffForce * W_1 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*0) * -1 ) * ffx +
                               (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*0) * 0 ) * ffy +
                               (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*0) * 0 ) * ffz);

            fneq = (W_1 * 4.5f) * ((-1*-1 - CSSQ) * pxx +
                                (0*0 - CSSQ) * pyy +
                                (0*0 - CSSQ) * pzz +
                                2.0f * -1 * 0 * pxy +
                                2.0f * -1 * 0 * pxz +
                                2.0f * 0 * 0 * pyz +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                                3.0f * ((-1*-1*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                        (-1*-1*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                        (-1*0*0 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (0*0*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                        (-1*0*0 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (0*0*0 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (-1*0*0) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x-1,y,z,2)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== THREE ========================== //

            feq = W_1 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * uy
                            + 4.5f * uy * uy
                            + 4.5f * uy * uy * uy
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * uy);

            forceCorr = coeffForce * W_1 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*0) * 0 ) * ffx +
                               (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*0) * 1 ) * ffy +
                               (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*0) * 0 ) * ffz);

            fneq = (W_1 * 4.5f) * ((0*0 - CSSQ) * pxx +
                                (1*1 - CSSQ) * pyy +
                                (0*0 - CSSQ) * pzz +
                                2.0f * 0 * 1 * pxy +
                                2.0f * 0 * 0 * pxz +
                                2.0f * 1 * 0 * pyz +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                                3.0f * ((0*0*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (0*0*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                        (0*1*1 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                        (1*1*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                        (0*0*0 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                        (1*0*0 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (0*1*0) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x,y+1,z,3)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== FOUR ========================== //

            feq = W_1 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            - 3.0f * uy
                            + 4.5f * uy * uy
                            - 4.5f * uy * uy * uy
                            + 4.5f * (ux*ux + uy*uy + uz*uz) * uy);

            forceCorr = coeffForce * W_1 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*0) * 0 ) * ffx +
                               (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*0) * -1 ) * ffy +
                               (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*0) * 0 ) * ffz);

            fneq = (W_1 * 4.5f) * ((0*0 - CSSQ) * pxx +
                                (-1*-1 - CSSQ) * pyy +
                                (0*0 - CSSQ) * pzz +
                                2.0f * 0 * -1 * pxy +
                                2.0f * 0 * 0 * pxz +
                                2.0f * -1 * 0 * pyz +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                                3.0f * ((0*0*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (0*0*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                        (0*-1*-1 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                        (-1*-1*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                        (0*0*0 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                        (-1*0*0 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (0*-1*0) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x,y-1,z,4)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== FIVE ========================== //

            feq = W_1 * rho * (1.0f
                        - 1.5f * (ux*ux + uy*uy + uz*uz)
                        + 3.0f * uz
                        + 4.5f * uz * uz
                        + 4.5f * uz * uz * uz
                        - 4.5f * (ux*ux + uy*uy + uz*uz) * uz);

            forceCorr = coeffForce * W_1 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*1) * 0 ) * ffx +
                               (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*1) * 0 ) * ffy +
                               (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*1) * 1 ) * ffz);

            fneq = (W_1 * 4.5f) * ((0*0 - CSSQ) * pxx +
                                (0*0 - CSSQ) * pyy +
                                (1*1 - CSSQ) * pzz +
                                2.0f * 0 * 0 * pxy +
                                2.0f * 0 * 1 * pxz +
                                2.0f * 0 * 1 * pyz +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                                3.0f * ((0*0*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                        (0*0*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (0*0*0 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                        (0*0*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (0*1*1 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                        (0*1*1 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (0*0*1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x,y,z+1,5)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== SIX ========================== //

            feq = W_1 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            - 3.0f * uz
                            + 4.5f * uz * uz
                            - 4.5f * uz * uz * uz
                            + 4.5f * (ux*ux + uy*uy + uz*uz) * uz);

            forceCorr = coeffForce * W_1 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*-1) * 0 ) * ffx +
                               (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*-1) * 0 ) * ffy +
                               (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*-1) * -1 ) * ffz);

            fneq = (W_1 * 4.5f) * ((0*0 - CSSQ) * pxx +
                                (0*0 - CSSQ) * pyy +
                                (-1*-1 - CSSQ) * pzz +
                                2.0f * 0 * 0 * pxy +
                                2.0f * 0 * -1 * pxz +
                                2.0f * 0 * -1 * pyz +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                                3.0f * ((0*0*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                        (0*0*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (0*0*0 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                        (0*0*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (0*-1*-1 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                        (0*-1*-1 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (0*0*-1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x,y,z-1,6)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== SEVEN ========================== //

            feq = W_2 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (ux + uy)
                            + 4.5f * (ux + uy) * (ux + uy)
                            + 4.5f * (ux + uy) * (ux + uy) * (ux + uy)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uy));

            forceCorr = coeffForce * W_2 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*0) * 1 ) * ffx +
                               (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*0) * 1 ) * ffy +
                               (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*0) * 0 ) * ffz);

            fneq = (W_2 * 4.5f) * ((1*1 - CSSQ) * pxx +
                                (1*1 - CSSQ) * pyy +
                                (0*0 - CSSQ) * pzz +
                                2.0f * 1 * 1 * pxy +
                                2.0f * 1 * 0 * pxz +
                                2.0f * 1 * 0 * pyz +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                                3.0f * ((1*1*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (1*1*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                        (1*1*1 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (1*1*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                        (1*0*0 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (1*0*0 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (1*1*0) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x+1,y+1,z,7)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== EIGHT ========================== //

            feq = W_2 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            - 3.0f * (ux + uy)
                            + 4.5f * (ux + uy) * (ux + uy)
                            - 4.5f * (ux + uy) * (ux + uy) * (ux + uy)
                            + 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uy));

            forceCorr = coeffForce * W_2 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*0) * -1 ) * ffx +
                               (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*0) * -1 ) * ffy +
                               (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*0) * 0 ) * ffz);

            fneq = (W_2 * 4.5f) * ((-1*-1 - CSSQ) * pxx +
                                (-1*-1 - CSSQ) * pyy +
                                (0*0 - CSSQ) * pzz +
                                2.0f * -1 * -1 * pxy +
                                2.0f * -1 * 0 * pxz +
                                2.0f * -1 * 0 * pyz +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                                3.0f * ((-1*-1*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (-1*-1*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                        (-1*-1*-1 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (-1*-1*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                        (-1*0*0 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (-1*0*0 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (-1*-1*0) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x-1,y-1,z,8)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== NINE ========================== //

            feq = W_2 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (ux + uz)
                            + 4.5f * (ux + uz) * (ux + uz)
                            + 4.5f * (ux + uz) * (ux + uz) * (ux + uz)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uz));

            forceCorr = coeffForce * W_2 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*1) * 1 ) * ffx +
                               (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*1) * 0 ) * ffy +
                               (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*1) * 1 ) * ffz);

            fneq = (W_2 * 4.5f) * ((1*1 - CSSQ) * pxx +
                                (0*0 - CSSQ) * pyy +
                                (1*1 - CSSQ) * pzz +
                                2.0f * 1 * 0 * pxy +
                                2.0f * 1 * 1 * pxz +
                                2.0f * 0 * 1 * pyz +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                                3.0f * ((1*1*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                        (1*1*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (1*0*0 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (0*0*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (1*1*1 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (0*1*1 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (1*0*1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x+1,y,z+1,9)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TEN ========================== //

            feq = W_2 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            - 3.0f * (ux + uz)
                            + 4.5f * (ux + uz) * (ux + uz)
                            - 4.5f * (ux + uz) * (ux + uz) * (ux + uz)
                            + 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uz));

            forceCorr = coeffForce * W_2 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*-1) * -1 ) * ffx +
                               (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*-1) * 0 ) * ffy +
                               (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*-1) * -1 ) * ffz);

            fneq = (W_2 * 4.5f) * ((-1*-1 - CSSQ) * pxx +
                                (0*0 - CSSQ) * pyy +
                                (-1*-1 - CSSQ) * pzz +
                                2.0f * -1 * 0 * pxy +
                                2.0f * -1 * -1 * pxz +
                                2.0f * 0 * -1 * pyz +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                                3.0f * ((-1*-1*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                        (-1*-1*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (-1*0*0 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (0*0*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (-1*-1*-1 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (0*-1*-1 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (-1*0*-1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x-1,y,z-1,10)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== ELEVEN ========================== //

            feq = W_2 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (uy + uz)
                            + 4.5f * (uy + uz) * (uy + uz)
                            + 4.5f * (uy + uz) * (uy + uz) * (uy + uz)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (uy + uz));

            forceCorr = coeffForce * W_2 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*1) * 0 ) * ffx +
                               (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*1) * 1 ) * ffy +
                               (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*1) * 1 ) * ffz);

            fneq = (W_2 * 4.5f) * ((0*0 - CSSQ) * pxx +
                                (1*1 - CSSQ) * pyy +
                                (1*1 - CSSQ) * pzz +
                                2.0f * 0 * 1 * pxy +
                                2.0f * 0 * 1 * pxz +
                                2.0f * 1 * 1 * pyz +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                                3.0f * ((0*0*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (0*0*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (0*1*1 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                        (1*1*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (0*1*1 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                        (1*1*1 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (0*1*1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x,y+1,z+1,11)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWELVE ========================== //

            feq = W_2 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            - 3.0f * (uy + uz)
                            + 4.5f * (uy + uz) * (uy + uz)
                            - 4.5f * (uy + uz) * (uy + uz) * (uy + uz)
                            + 4.5f * (ux*ux + uy*uy + uz*uz) * (uy + uz));

            forceCorr = coeffForce * W_2 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*-1) * 0 ) * ffx +
                               (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*-1) * -1 ) * ffy +
                               (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*-1) * -1 ) * ffz);

            fneq = (W_2 * 4.5f) * ((0*0 - CSSQ) * pxx +
                                (-1*-1 - CSSQ) * pyy +
                                (-1*-1 - CSSQ) * pzz +
                                2.0f * 0 * -1 * pxy +
                                2.0f * 0 * -1 * pxz +
                                2.0f * -1 * -1 * pyz +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                                3.0f * ((0*0*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (0*0*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (0*-1*-1 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                        (-1*-1*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (0*-1*-1 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                        (-1*-1*-1 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (0*-1*-1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x,y-1,z-1,12)] = to_pop(feq + omcoLocal * fneq + forceCorr);
        
        // ========================== THIRTEEN ========================== //

            feq = W_2 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (ux - uy)
                            + 4.5f * (ux - uy) * (ux - uy)
                            + 4.5f * (ux - uy) * (ux - uy) * (ux - uy)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux - uy));

            forceCorr = coeffForce * W_2 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*0) * 1 ) * ffx +
                               (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*0) * -1 ) * ffy +
                               (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*0) * 0 ) * ffz);

            fneq = (W_2 * 4.5f) * ((1*1 - CSSQ) * pxx +
                                (-1*-1 - CSSQ) * pyy +
                                (0*0 - CSSQ) * pzz +
                                2.0f * 1 * -1 * pxy +
                                2.0f * 1 * 0 * pxz +
                                2.0f * -1 * 0 * pyz +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                                3.0f * ((1*1*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (1*1*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                        (1*-1*-1 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (-1*-1*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                        (1*0*0 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (-1*0*0 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (1*-1*0) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x+1,y-1,z,13)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== FOURTEEN ========================== //

            feq = W_2 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (uy - ux)
                            + 4.5f * (uy - ux) * (uy - ux)
                            + 4.5f * (uy - ux) * (uy - ux) * (uy - ux)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (uy - ux));

            forceCorr = coeffForce * W_2 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*0) * -1 ) * ffx +
                               (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*0) * 1 ) * ffy +
                               (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*0) * 0 ) * ffz);

            fneq = (W_2 * 4.5f) * ((-1*-1 - CSSQ) * pxx +
                                (1*1 - CSSQ) * pyy +
                                (0*0 - CSSQ) * pzz +
                                2.0f * -1 * 1 * pxy +
                                2.0f * -1 * 0 * pxz +
                                2.0f * 1 * 0 * pyz +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                                3.0f * ((-1*-1*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (-1*-1*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                        (-1*1*1 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (1*1*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                        (-1*0*0 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (1*0*0 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (-1*1*0) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x-1,y+1,z,14)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== FIFTEEN ========================== //

            feq = W_2 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (ux - uz)
                            + 4.5f * (ux - uz) * (ux - uz)
                            + 4.5f * (ux - uz) * (ux - uz) * (ux - uz)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux - uz));

            forceCorr = coeffForce * W_2 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*-1) * 1 ) * ffx +
                               (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*-1) * 0 ) * ffy +
                               (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*-1) * -1 ) * ffz);

            fneq = (W_2 * 4.5f) * ((1*1 - CSSQ) * pxx +
                                (0*0 - CSSQ) * pyy +
                                (-1*-1 - CSSQ) * pzz +
                                2.0f * 1 * 0 * pxy +
                                2.0f * 1 * -1 * pxz +
                                2.0f * 0 * -1 * pyz +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                                3.0f * ((1*1*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                        (1*1*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (1*0*0 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (0*0*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (1*-1*-1 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (0*-1*-1 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (1*0*-1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x+1,y,z-1,15)] = to_pop(feq + omcoLocal * fneq + forceCorr); 

        // ========================== SIXTEEN ========================== //

            feq = W_2 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (uz - ux)
                            + 4.5f * (uz - ux) * (uz - ux)
                            + 4.5f * (uz - ux) * (uz - ux) * (uz - ux)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (uz - ux));

            forceCorr = coeffForce * W_2 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*1) * -1 ) * ffx +
                               (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*1) * 0 ) * ffy +
                               (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*1) * 1 ) * ffz);

            fneq = (W_2 * 4.5f) * ((-1*-1 - CSSQ) * pxx +
                                (0*0 - CSSQ) * pyy +
                                (1*1 - CSSQ) * pzz +
                                2.0f * -1 * 0 * pxy +
                                2.0f * -1 * 1 * pxz +
                                2.0f * 0 * 1 * pyz +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                                3.0f * ((-1*-1*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                        (-1*-1*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (-1*0*0 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (0*0*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (-1*1*1 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (0*1*1 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (-1*0*1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x-1,y,z+1,16)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== SEVENTEEN ========================== //

            feq = W_2 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (uy - uz)
                            + 4.5f * (uy - uz) * (uy - uz)
                            + 4.5f * (uy - uz) * (uy - uz) * (uy - uz)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (uy - uz));

            forceCorr = coeffForce * W_2 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*-1) * 0 ) * ffx +
                               (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*-1) * 1 ) * ffy +
                               (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*-1) * -1 ) * ffz);

            fneq = (W_2 * 4.5f) * ((0*0 - CSSQ) * pxx +
                                (1*1 - CSSQ) * pyy +
                                (-1*-1 - CSSQ) * pzz +
                                2.0f * 0 * 1 * pxy +
                                2.0f * 0 * -1 * pxz +
                                2.0f * 1 * -1 * pyz +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                                3.0f * ((0*0*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (0*0*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (0*1*1 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                        (1*1*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (0*-1*-1 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                        (1*-1*-1 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (0*1*-1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x,y+1,z-1,17)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== EIGHTEEN ========================== //

            feq = W_2 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (uz - uy)
                            + 4.5f * (uz - uy) * (uz - uy)
                            + 4.5f * (uz - uy) * (uz - uy) * (uz - uy)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (uz - uy));

            forceCorr = coeffForce * W_2 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*1) * 0 ) * ffx +
                               (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*1) * -1 ) * ffy +
                               (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*1) * 1 ) * ffz);

            fneq = (W_2 * 4.5f) * ((0*0 - CSSQ) * pxx +
                                (-1*-1 - CSSQ) * pyy +
                                (1*1 - CSSQ) * pzz +
                                2.0f * 0 * -1 * pxy +
                                2.0f * 0 * 1 * pxz +
                                2.0f * -1 * 1 * pyz +
                                (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                                3.0f * ((0*0*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (0*0*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (0*-1*-1 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                        (-1*-1*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (0*1*1 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                        (-1*1*1 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (0*-1*1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x,y-1,z+1,18)] = to_pop(feq + omcoLocal * fneq + forceCorr);
        
        // ========================== NINETEEN ========================== //

            feq = W_3 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (ux + uy + uz)
                            + 4.5f * (ux + uy + uz) * (ux + uy + uz)
                            + 4.5f * (ux + uy + uz) * (ux + uy + uz) * (ux + uy + uz)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uy + uz));

            forceCorr = coeffForce * W_3 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*1) * 1 ) * ffx +
                               (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*1) * 1 ) * ffy +
                               (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*1) * 1 ) * ffz);

            fneq = (W_3 * 4.5f) * ((1*1 - CSSQ) * pxx +
                                (1*1 - CSSQ) * pyy +
                                (1*1 - CSSQ) * pzz +
                                2.0f * 1 * 1 * pxy +
                                2.0f * 1 * 1 * pxz +
                                2.0f * 1 * 1 * pyz +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                                3.0f * ((1*1*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (1*1*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (1*1*1 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (1*1*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (1*1*1 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (1*1*1 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (1*1*1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x+1,y+1,z+1,19)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWENTY ========================== //

            feq = W_3 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            - 3.0f * (ux + uy + uz)
                            + 4.5f * (ux + uy + uz) * (ux + uy + uz)
                            - 4.5f * (ux + uy + uz) * (ux + uy + uz) * (ux + uy + uz)
                            + 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uy + uz));

            forceCorr = coeffForce * W_3 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*-1) * -1 ) * ffx +
                               (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*-1) * -1 ) * ffy +
                               (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*-1) * -1 ) * ffz);

            fneq = (W_3 * 4.5f) * ((-1*-1 - CSSQ) * pxx +
                                (-1*-1 - CSSQ) * pyy +
                                (-1*-1 - CSSQ) * pzz +
                                2.0f * -1 * -1 * pxy +
                                2.0f * -1 * -1 * pxz +
                                2.0f * -1 * -1 * pyz +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                                3.0f * ((-1*-1*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (-1*-1*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (-1*-1*-1 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (-1*-1*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (-1*-1*-1 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (-1*-1*-1 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (-1*-1*-1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x-1,y-1,z-1,20)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWENTY ONE ========================== //

            feq = W_3 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (ux + uy - uz)
                            + 4.5f * (ux + uy - uz) * (ux + uy - uz)
                            + 4.5f * (ux + uy - uz) * (ux + uy - uz) * (ux + uy - uz)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux + uy - uz));        

            forceCorr = coeffForce * W_3 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*-1) * 1 ) * ffx +
                               (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*-1) * 1 ) * ffy +
                               (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*-1) * -1 ) * ffz);

            fneq = (W_3 * 4.5f) * ((1*1 - CSSQ) * pxx +
                                (1*1 - CSSQ) * pyy +
                                (-1*-1 - CSSQ) * pzz +
                                2.0f * 1 * 1 * pxy +
                                2.0f * 1 * -1 * pxz +
                                2.0f * 1 * -1 * pyz +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                                3.0f * ((1*1*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (1*1*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (1*1*1 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (1*1*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (1*-1*-1 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (1*-1*-1 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (1*1*-1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x+1,y+1,z-1,21)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWENTY TWO ========================== //

            feq = W_3 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (uz - ux - uy)
                            + 4.5f * (uz - ux - uy) * (uz - ux - uy)
                            + 4.5f * (uz - ux - uy) * (uz - ux - uy) * (uz - ux - uy)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (uz - ux - uy));

            forceCorr = coeffForce * W_3 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*1) * -1 ) * ffx +
                               (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*1) * -1 ) * ffy +
                               (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*1) * 1 ) * ffz);

            fneq = (W_3 * 4.5f) * ((-1*-1 - CSSQ) * pxx +
                                (-1*-1 - CSSQ) * pyy +
                                (1*1 - CSSQ) * pzz +
                                2.0f * -1 * -1 * pxy +
                                2.0f * -1 * 1 * pxz +
                                2.0f * -1 * 1 * pyz +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                                3.0f * ((-1*-1*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (-1*-1*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (-1*-1*-1 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (-1*-1*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (-1*1*1 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (-1*1*1 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (-1*-1*1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x-1,y-1,z+1,22)] = to_pop(feq + omcoLocal * fneq + forceCorr);    

        // ========================== TWENTY THREE ========================== //
        
            feq = W_3 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (ux - uy + uz)
                            + 4.5f * (ux - uy + uz) * (ux - uy + uz)
                            + 4.5f * (ux - uy + uz) * (ux - uy + uz) * (ux - uy + uz)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux - uy + uz));

            forceCorr = coeffForce * W_3 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*1) * 1 ) * ffx +
                               (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*1) * -1 ) * ffy +
                               (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*1) * 1 ) * ffz);

            fneq = (W_3 * 4.5f) * ((1*1 - CSSQ) * pxx +
                                (-1*-1 - CSSQ) * pyy +
                                (1*1 - CSSQ) * pzz +
                                2.0f * 1 * -1 * pxy +
                                2.0f * 1 * 1 * pxz +
                                2.0f * -1 * 1 * pyz +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                                3.0f * ((1*1*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (1*1*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (1*-1*-1 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (-1*-1*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (1*1*1 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (-1*1*1 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (1*-1*1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x+1,y-1,z+1,23)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWENTY FOUR ========================== //

            feq = W_3 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (uy - ux - uz)
                            + 4.5f * (uy - ux - uz) * (uy - ux - uz)
                            + 4.5f * (uy - ux - uz) * (uy - ux - uz) * (uy - ux - uz)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (uy - ux - uz));

            forceCorr = coeffForce * W_3 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*-1) * -1 ) * ffx +
                               (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*-1) * 1 ) * ffy +
                               (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*-1) * -1 ) * ffz);

            fneq = (W_3 * 4.5f) * ((-1*-1 - CSSQ) * pxx +
                                (1*1 - CSSQ) * pyy +
                                (-1*-1 - CSSQ) * pzz +
                                2.0f * -1 * 1 * pxy +
                                2.0f * -1 * -1 * pxz +
                                2.0f * 1 * -1 * pyz +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                                3.0f * ((-1*-1*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (-1*-1*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (-1*1*1 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (1*1*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (-1*-1*-1 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (1*-1*-1 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (-1*1*-1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x-1,y+1,z-1,24)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        // ========================== TWENTY FIVE ========================== //

            feq = W_3 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (uy - ux + uz)
                            + 4.5f * (uy - ux + uz) * (uy - ux + uz)
                            + 4.5f * (uy - ux + uz) * (uy - ux + uz) * (uy - ux + uz)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (uy - ux + uz));

            forceCorr = coeffForce * W_3 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*1) * -1 ) * ffx +
                               (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*1) * 1 ) * ffy +
                               (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*1) * 1 ) * ffz);

            fneq = (W_3 * 4.5f) * ((-1*-1 - CSSQ) * pxx +
                                (1*1 - CSSQ) * pyy +
                                (1*1 - CSSQ) * pzz +
                                2.0f * -1 * 1 * pxy +
                                2.0f * -1 * 1 * pxz +
                                2.0f * 1 * 1 * pyz +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                                3.0f * ((-1*-1*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (-1*-1*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (-1*1*1 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (1*1*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (-1*1*1 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (1*1*1 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (-1*1*1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x-1,y+1,z+1,25)] = to_pop(feq + omcoLocal * fneq + forceCorr);    
        
        // ========================== TWENTY SIX ========================== //

            feq = W_3 * rho * (1.0f
                            - 1.5f * (ux*ux + uy*uy + uz*uz)
                            + 3.0f * (ux - uy - uz)
                            + 4.5f * (ux - uy - uz) * (ux - uy - uz)
                            + 4.5f * (ux - uy - uz) * (ux - uy - uz) * (ux - uy - uz)
                            - 4.5f * (ux*ux + uy*uy + uz*uz) * (ux - uy - uz));

            forceCorr = coeffForce * W_3 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*-1) * 1 ) * ffx +
                               (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*-1) * -1 ) * ffy +
                               (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*-1) * -1 ) * ffz);

            fneq = (W_3 * 4.5f) * ((1*1 - CSSQ) * pxx +
                                (-1*-1 - CSSQ) * pyy +
                                (-1*-1 - CSSQ) * pzz +
                                2.0f * 1 * -1 * pxy +
                                2.0f * 1 * -1 * pxz +
                                2.0f * -1 * -1 * pyz +
                                (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                                (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                                3.0f * ((1*1*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                        (1*1*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                        (1*-1*-1 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                        (-1*-1*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                        (1*-1*-1 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                        (-1*-1*-1 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                                6.0f * (1*-1*-1) * (pxy*uz + ux*pyz + uy*pxz));

            d.f[global4(x+1,y-1,z-1,26)] = to_pop(feq + omcoLocal * fneq + forceCorr);

        #endif 

    // ========================================================== END ========================================================== //



    // ================================================== ADVECTION-DIFFUSION ================================================== //

        //             0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
        // CIX[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
        // CIY[27] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1 };
        // CIZ[27] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1 };

        #if !defined(VISC_CONTRAST)
        const float phi = d.phi[idx3];
        #endif

        // =======
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
    
    // ========================================================== END ========================================================== //
    
}

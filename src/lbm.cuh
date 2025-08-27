#pragma once

__global__ void gpuPhi(LBMFields d) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = global3(x,y,z);

    const float phi = d.g[idx3] + d.g[PLANE+idx3] + d.g[PLANE2+idx3] + d.g[PLANE3+idx3] + d.g[PLANE4+idx3] + d.g[PLANE5+idx3] + d.g[PLANE6+idx3];
    d.phi[idx3] = phi;
}

__global__ void gpuNormals(LBMFields d) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = global3(x,y,z);

    float sumGradX = W_1 * (d.phi[global3(x+1,y,z)]   - d.phi[global3(x-1,y,z)])  +
                     W_2 * (d.phi[global3(x+1,y+1,z)] - d.phi[global3(x-1,y-1,z)] +
                            d.phi[global3(x+1,y,z+1)] - d.phi[global3(x-1,y,z-1)] +
                            d.phi[global3(x+1,y-1,z)] - d.phi[global3(x-1,y+1,z)] +
                            d.phi[global3(x+1,y,z-1)] - d.phi[global3(x-1,y,z+1)]);

    float sumGradY = W_1 * (d.phi[global3(x,y+1,z)]   - d.phi[global3(x,y-1,z)])  +
                     W_2 * (d.phi[global3(x+1,y+1,z)] - d.phi[global3(x-1,y-1,z)] +
                            d.phi[global3(x,y+1,z+1)] - d.phi[global3(x,y-1,z-1)] +
                            d.phi[global3(x-1,y+1,z)] - d.phi[global3(x+1,y-1,z)] +
                            d.phi[global3(x,y+1,z-1)] - d.phi[global3(x,y-1,z+1)]);

    float sumGradZ = W_1 * (d.phi[global3(x,y,z+1)]   - d.phi[global3(x,y,z-1)])  +
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

__global__ void gpuForces(LBMFields d) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = global3(x,y,z);

    const float normX = d.normx[idx3];
    const float normY = d.normy[idx3];
    const float normZ = d.normz[idx3];
    const float ind = d.ind[idx3];

    float sumCurvX = W_1 * (d.normx[global3(x+1,y,z)]   - d.normx[global3(x-1,y,z)])  +
                     W_2 * (d.normx[global3(x+1,y+1,z)] - d.normx[global3(x-1,y-1,z)] +
                            d.normx[global3(x+1,y,z+1)] - d.normx[global3(x-1,y,z-1)] +
                            d.normx[global3(x+1,y-1,z)] - d.normx[global3(x-1,y+1,z)] +
                            d.normx[global3(x+1,y,z-1)] - d.normx[global3(x-1,y,z+1)]);

    float sumCurvY = W_1 * (d.normy[global3(x,y+1,z)]   - d.normy[global3(x,y-1,z)])  +
                     W_2 * (d.normy[global3(x+1,y+1,z)] - d.normy[global3(x-1,y-1,z)] +
                            d.normy[global3(x,y+1,z+1)] - d.normy[global3(x,y-1,z-1)] +
                            d.normy[global3(x-1,y+1,z)] - d.normy[global3(x+1,y-1,z)] +
                            d.normy[global3(x,y+1,z-1)] - d.normy[global3(x,y-1,z+1)]);

    float sumCurvZ = W_1 * (d.normz[global3(x,y,z+1)]   - d.normz[global3(x,y,z-1)])  +
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
    float curvature = -3.0f * (sumCurvX + sumCurvY + sumCurvZ);   

    const float stCurv = SIGMA * curvature;
    d.ffx[idx3] = stCurv * normX * ind;
    d.ffy[idx3] = stCurv * normY * ind;
    d.ffz[idx3] = stCurv * normZ * ind;
}

__global__ void gpuCollisionStream(LBMFields d) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = global3(x,y,z);
        
    const float pop0  = from_pop(d.f[idx3]);         // 0
    const float pop1  = from_pop(d.f[PLANE+idx3]);   // 1
    const float pop2  = from_pop(d.f[PLANE2+idx3]);  // 2
    const float pop3  = from_pop(d.f[PLANE3+idx3]);  // 3
    const float pop4  = from_pop(d.f[PLANE4+idx3]);  // 4
    const float pop5  = from_pop(d.f[PLANE5+idx3]);  // 5 
    const float pop6  = from_pop(d.f[PLANE6+idx3]);  // 6
    const float pop7  = from_pop(d.f[PLANE7+idx3]);  // 7
    const float pop8  = from_pop(d.f[PLANE8+idx3]);  // 8
    const float pop9  = from_pop(d.f[PLANE9+idx3]);  // 9
    const float pop10 = from_pop(d.f[PLANE10+idx3]); // 10
    const float pop11 = from_pop(d.f[PLANE11+idx3]); // 11
    const float pop12 = from_pop(d.f[PLANE12+idx3]); // 12
    const float pop13 = from_pop(d.f[PLANE13+idx3]); // 13
    const float pop14 = from_pop(d.f[PLANE14+idx3]); // 14
    const float pop15 = from_pop(d.f[PLANE15+idx3]); // 15
    const float pop16 = from_pop(d.f[PLANE16+idx3]); // 16
    const float pop17 = from_pop(d.f[PLANE17+idx3]); // 17
    const float pop18 = from_pop(d.f[PLANE18+idx3]); // 18
    #if defined(D3Q27)
    const float pop19 = from_pop(d.f[PLANE19+idx3]); // 19
    const float pop20 = from_pop(d.f[PLANE20+idx3]); // 20
    const float pop21 = from_pop(d.f[PLANE21+idx3]); // 21
    const float pop22 = from_pop(d.f[PLANE22+idx3]); // 22
    const float pop23 = from_pop(d.f[PLANE23+idx3]); // 23
    const float pop24 = from_pop(d.f[PLANE24+idx3]); // 24
    const float pop25 = from_pop(d.f[PLANE25+idx3]); // 25 
    const float pop26 = from_pop(d.f[PLANE26+idx3]); // 26
    #endif 

    #if defined(D3Q19)
    float rho = pop0 + pop1 + pop2 + pop3 + pop4 + pop5 + pop6 + pop7 + pop8 + pop9 + pop10 + pop11 + pop12 + pop13 + pop14 + pop15 + pop16 + pop17 + pop18;
    #elif defined(D3Q27)
    float rho = pop0 + pop1 + pop2 + pop3 + pop4 + pop5 + pop6 + pop7 + pop8 + pop9 + pop10 + pop11 + pop12 + pop13 + pop14 + pop15 + pop16 + pop17 + pop18 + pop19 + pop20 + pop21 + pop22 + pop23 + pop24 + pop25 + pop26;
    #endif
    rho += 1.0f; 
    d.rho[idx3] = rho;

    const float invRho = 1.0f / rho;
    const float ffx = d.ffx[idx3];
    const float ffy = d.ffy[idx3];
    const float ffz = d.ffz[idx3];

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

    // NOTE: weight compensation piggybacks on the LAST contrib of each diagonal.
    // if order changes, move/revisit this!

    #if defined(D3Q19)
    float feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*ux + 4.5f*ux*ux);
    #elif defined(D3Q27)
    float feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*ux + 4.5f*ux*ux + 4.5f*ux*ux*ux - 3.0f*ux * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    float fneq = pop1 - feq;
    float pxx = fneq;

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*ux + 4.5f*ux*ux);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*ux + 4.5f*ux*ux - 4.5f*ux*ux*ux + 3.0f*ux * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop2 - feq;
    pxx += fneq;

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*uy + 4.5f*uy*uy);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*uy + 4.5f*uy*uy + 4.5f*uy*uy*uy - 3.0f*uy * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop3 - feq;
    float pyy = fneq;

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*uy + 4.5f*uy*uy);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*uy + 4.5f*uy*uy - 4.5f*uy*uy*uy + 3.0f*uy * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop4 - feq;
    pyy += fneq;

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*uz + 4.5f*uz*uz);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*uz + 4.5f*uz*uz + 4.5f*uz*uz*uz - 3.0f*uz * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop5 - feq;
    float pzz = fneq;

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*uz + 4.5f*uz*uz);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*uz + 4.5f*uz*uz - 4.5f*uz*uz*uz + 3.0f*uz * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop6 - feq;
    pzz += fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uy) + 4.5f*(ux + uy)*(ux + uy));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uy) + 4.5f*(ux + uy)*(ux + uy) + 4.5f*(ux + uy)*(ux + uy)*(ux + uy) - 3.0f*(ux + uy) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop7 - feq;
    pxx += fneq; 
    pyy += fneq; 
    float pxy = fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uy) + 4.5f*(ux + uy)*(ux + uy));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uy) + 4.5f*(ux + uy)*(ux + uy) - 4.5f*(ux + uy)*(ux + uy)*(ux + uy) + 3.0f*(ux + uy) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop8 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pxy += fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uz) + 4.5f*(ux + uz)*(ux + uz));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uz) + 4.5f*(ux + uz)*(ux + uz) + 4.5f*(ux + uz)*(ux + uz)*(ux + uz) - 3.0f*(ux + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop9 - feq;
    pxx += fneq; 
    pzz += fneq; 
    float pxz = fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uz) + 4.5f*(ux + uz)*(ux + uz));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uz) + 4.5f*(ux + uz)*(ux + uz) - 4.5f*(ux + uz)*(ux + uz)*(ux + uz) + 3.0f*(ux + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop10 - feq;
    pxx += fneq; 
    pzz += fneq; 
    pxz += fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy + uz) + 4.5f*(uy + uz)*(uy + uz));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy + uz) + 4.5f*(uy + uz)*(uy + uz) + 4.5f*(uy + uz)*(uy + uz)*(uy + uz) - 3.0f*(uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop11 - feq;
    pyy += fneq;
    pzz += fneq; 
    float pyz = fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(uy + uz) + 4.5f*(uy + uz)*(uy + uz));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(uy + uz) + 4.5f*(uy + uz)*(uy + uz) - 4.5f*(uy + uz)*(uy + uz)*(uy + uz) + 3.0f*(uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop12 - feq;
    pyy += fneq; 
    pzz += fneq; 
    pyz += fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uy) + 4.5f*(ux - uy)*(ux - uy));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uy) + 4.5f*(ux - uy)*(ux - uy) + 4.5f*(ux - uy)*(ux - uy)*(ux - uy) - 3.0f*(ux - uy) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop13 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pxy -= fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - ux) + 4.5f*(uy - ux)*(uy - ux));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - ux) + 4.5f*(uy - ux)*(uy - ux) + 4.5f*(uy - ux)*(uy - ux)*(uy - ux) - 3.0f*(uy - ux) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop14 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pxy -= fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uz) + 4.5f*(ux - uz)*(ux - uz));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uz) + 4.5f*(ux - uz)*(ux - uz) + 4.5f*(ux - uz)*(ux - uz)*(ux - uz) - 3.0f*(ux - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop15 - feq;
    pxx += fneq; 
    pzz += fneq; 
    pxz -= fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - ux) + 4.5f*(uz - ux)*(uz - ux));
    pxx += fneq + CSSQ; 
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - ux) + 4.5f*(uz - ux)*(uz - ux) + 4.5f*(uz - ux)*(uz - ux)*(uz - ux) - 3.0f*(uz - ux) * 1.5f*(ux*ux + uy*uy + uz*uz));
    pxx += fneq; 
    #endif
    fneq = pop16 - feq;
    pzz += fneq; 
    pxz -= fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - uz) + 4.5f*(uy - uz)*(uy - uz));
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - uz) + 4.5f*(uy - uz)*(uy - uz) + 4.5f*(uy - uz)*(uy - uz)*(uy - uz) - 3.0f*(uy - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    #endif
    fneq = pop17 - feq;
    pyy += fneq; 
    pzz += fneq; 
    pyz -= fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - uy) + 4.5f*(uz - uy)*(uz - uy));
    fneq = pop18 - feq;
    pyy += fneq + CSSQ; 
    pzz += fneq + CSSQ;
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - uy) + 4.5f*(uz - uy)*(uz - uy) + 4.5f*(uz - uy)*(uz - uy)*(uz - uy) - 3.0f*(uz - uy) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop18 - feq;
    pyy += fneq; 
    pzz += fneq; 
    #endif
    pyz -= fneq;
    
    // THIRD ORDER TERMS
    #if defined(D3Q27)
    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uy + uz) + 4.5f*(ux + uy + uz)*(ux + uy + uz) + 4.5f*(ux + uy + uz)*(ux + uy + uz)*(ux + uy + uz) - 3.0f*(ux + uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop19 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy += fneq; 
    pxz += fneq; 
    pyz += fneq;

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uy + uz) + 4.5f*(ux + uy + uz)*(ux + uy + uz) - 4.5f*(ux + uy + uz)*(ux + uy + uz)*(ux + uy + uz) + 3.0f*(ux + uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop20 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy += fneq; 
    pxz += fneq; 
    pyz += fneq;

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uy - uz) + 4.5f*(ux + uy - uz)*(ux + uy - uz) + 4.5f*(ux + uy - uz)*(ux + uy - uz)*(ux + uy - uz) - 3.0f*(ux + uy - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));    
    fneq = pop21 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy += fneq; 
    pxz -= fneq; 
    pyz -= fneq;

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - uy - ux) + 4.5f*(uz - uy - ux)*(uz - uy - ux) + 4.5f*(uz - uy - ux)*(uz - uy - ux)*(uz - uy - ux) - 3.0f*(uz - uy - ux) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop22 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy += fneq; 
    pxz -= fneq; 
    pyz -= fneq; 

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uy + uz) + 4.5f*(ux - uy + uz)*(ux - uy + uz) + 4.5f*(ux - uy + uz)*(ux - uy + uz)*(ux - uy + uz) - 3.0f*(ux - uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop23 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy -= fneq; 
    pxz += fneq; 
    pyz -= fneq;

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - ux - uz) + 4.5f*(uy - ux - uz)*(uy - ux - uz) + 4.5f*(uy - ux - uz)*(uy - ux - uz)*(uy - ux - uz) - 3.0f*(uy - ux - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop24 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy -= fneq; 
    pxz += fneq; 
    pyz -= fneq;

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - ux + uz) + 4.5f*(uy - ux + uz)*(uy - ux + uz) + 4.5f*(uy - ux + uz)*(uy - ux + uz)*(uy - ux + uz) - 3.0f*(uy - ux + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop25 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy -= fneq; 
    pxz -= fneq; 
    pyz += fneq;

    feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uy - uz) + 4.5f*(ux - uy - uz)*(ux - uy - uz) + 4.5f*(ux - uy - uz)*(ux - uy - uz)*(ux - uy - uz) - 3.0f*(ux - uy - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
    fneq = pop26 - feq;
    pxx += fneq + CSSQ; 
    pyy += fneq + CSSQ; 
    pzz += fneq + CSSQ;
    pxy -= fneq; 
    pxz -= fneq; 
    pyz += fneq;
    #endif 

    //pxx += CSSQ;
    //pyy += CSSQ;
    //pzz += CSSQ;

    d.pxx[idx3] = pxx;
    d.pyy[idx3] = pyy;
    d.pzz[idx3] = pzz;
    d.pxy[idx3] = pxy;
    d.pxz[idx3] = pxz;   
    d.pyz[idx3] = pyz;

    const float omegaLocal = omegaSponge(z);
    const float omcoLocal = 1.0f - omegaLocal;
    const float coeffForce = 1.0f - 0.5f * omegaLocal;

    // #if defined(D3Q19)
    // feq = W_0 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz));
    // #elif defined(D3Q27)
    // feq = W_0 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz));
    // #endif

    feq = computeEquilibria(rho,ux,uy,uz,0);
    float forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,0);
    float fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,0);
    d.f[global4(x,y,z,0)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,1);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,1);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,1);
    d.f[global4(x+1,y,z,1)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,2);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,2);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,2);
    d.f[global4(x-1,y,z,2)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,3);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,3);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,3);
    d.f[global4(x,y+1,z,3)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,4);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,4);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,4);
    d.f[global4(x,y-1,z,4)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,5);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,5);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,5);
    d.f[global4(x,y,z+1,5)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,6);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,6);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,6);
    d.f[global4(x,y,z-1,6)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,7);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,7);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,7);
    d.f[global4(x+1,y+1,z,7)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,8);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,8);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,8);
    d.f[global4(x-1,y-1,z,8)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,9);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,9);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,9);
    d.f[global4(x+1,y,z+1,9)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,10);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,10);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,10);
    d.f[global4(x-1,y,z-1,10)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,11);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,11);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,11);
    d.f[global4(x,y+1,z+1,11)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,12);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,12);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,12);
    d.f[global4(x,y-1,z-1,12)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,13);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,13);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,13);
    d.f[global4(x+1,y-1,z,13)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,14);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,14);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,14);
    d.f[global4(x-1,y+1,z,14)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,15);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,15);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,15);
    d.f[global4(x+1,y,z-1,15)] = to_pop(feq + omcoLocal * fneqReg + forceCorr); 

    feq = computeEquilibria(rho,ux,uy,uz,16);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,16);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,16);
    d.f[global4(x-1,y,z+1,16)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,17);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,17);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,17);
    d.f[global4(x,y+1,z-1,17)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,18);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,18);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,18);
    d.f[global4(x,y-1,z+1,18)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    #if defined(D3Q27)
    feq = computeEquilibria(rho,ux,uy,uz,19);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,19);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,19);
    d.f[global4(x+1,y+1,z+1,19)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,20);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,20);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,20);
    d.f[global4(x-1,y-1,z-1,20)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,21);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,21);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,21);
    d.f[global4(x+1,y+1,z-1,21)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,22);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,22);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,22);
    d.f[global4(x-1,y-1,z+1,22)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);    

    feq = computeEquilibria(rho,ux,uy,uz,23);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,23);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,23);
    d.f[global4(x+1,y-1,z+1,23)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,24);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,24);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,24);
    d.f[global4(x-1,y+1,z-1,24)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

    feq = computeEquilibria(rho,ux,uy,uz,25);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,25);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,25);
    d.f[global4(x-1,y+1,z+1,25)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);    
    
    feq = computeEquilibria(rho,ux,uy,uz,26);
    forceCorr = computeForceTerm(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,26);
    fneqReg = computeNonEquilibria(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,26);
    d.f[global4(x+1,y-1,z-1,26)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);
    #endif 
}

__global__ void gpuEvolvePhaseField(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;
        
    const idx_t idx3 = global3(x,y,z);

    const float phi = d.phi[idx3];
    const float ux = d.ux[idx3];
    const float uy = d.uy[idx3];
    const float uz = d.uz[idx3];

    d.g[global4(x,y,z,0)] = W_G_1 * phi;

    const float phiNorm = W_G_2 * GAMMA * phi * (1.0f - phi);
    const float multPhi = W_G_2 * phi;
    const float a3 = 3.0f * multPhi;

    float geq = multPhi + a3 * ux;
    float antiDiff = phiNorm * d.normx[idx3];
    d.g[global4(x+1,y,z,1)] = geq + antiDiff;
    
    geq = multPhi - a3 * ux;
    d.g[global4(x-1,y,z,2)] = geq - antiDiff;

    geq = multPhi + a3 * uy;
    antiDiff = phiNorm * d.normy[idx3];
    d.g[global4(x,y+1,z,3)] = geq + antiDiff;

    geq = multPhi - a3 * uy;
    d.g[global4(x,y-1,z,4)] = geq - antiDiff;

    geq = multPhi + a3 * uz;
    antiDiff = phiNorm * d.normz[idx3];
    d.g[global4(x,y,z+1,5)] = geq + antiDiff;

    geq = multPhi - a3 * uz;
    d.g[global4(x,y,z-1,6)] = geq - antiDiff;
} 




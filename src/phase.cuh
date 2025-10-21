
#define HO_GRAD_STENCIL

__global__ 
void computePhase(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ ||
        x == 0 || x == NX - 1 ||
        y == 0 || y == NY - 1 ||
        z == 0 || z == NZ - 1) return;

    const idx_t idx3 = global3(x, y, z);

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
        x == 0 || x == NX - 1 ||
        y == 0 || y == NY - 1 ||
        z == 0 || z == NZ - 1) return;

    const idx_t idx3 = global3(x, y, z);

    #if defined(HO_GRAD_STENCIL)

        float sgx = W_1 * (d.phi[global3(x+1,y,z)]   - d.phi[global3(x-1,y,z)]) +
                    W_2 * (d.phi[global3(x+1,y+1,z)] - d.phi[global3(x-1,y-1,z)] +
                           d.phi[global3(x+1,y,z+1)] - d.phi[global3(x-1,y,z-1)] +
                           d.phi[global3(x+1,y-1,z)] - d.phi[global3(x-1,y+1,z)] +
                           d.phi[global3(x+1,y,z-1)] - d.phi[global3(x-1,y,z+1)]);

        float sgy = W_1 * (d.phi[global3(x,y+1,z)]   - d.phi[global3(x,y-1,z)]) +
                    W_2 * (d.phi[global3(x+1,y+1,z)] - d.phi[global3(x-1,y-1,z)] +
                           d.phi[global3(x,y+1,z+1)] - d.phi[global3(x,y-1,z-1)] +
                           d.phi[global3(x-1,y+1,z)] - d.phi[global3(x+1,y-1,z)] +
                           d.phi[global3(x,y+1,z-1)] - d.phi[global3(x,y-1,z+1)]);

        float sgz = W_1 * (d.phi[global3(x,y,z+1)]   - d.phi[global3(x,y,z-1)]) +
                    W_2 * (d.phi[global3(x+1,y,z+1)] - d.phi[global3(x-1,y,z-1)] +
                           d.phi[global3(x,y+1,z+1)] - d.phi[global3(x,y-1,z-1)] +
                           d.phi[global3(x-1,y,z+1)] - d.phi[global3(x+1,y,z-1)] +
                           d.phi[global3(x,y-1,z+1)] - d.phi[global3(x,y+1,z-1)]);
        #if defined(D3Q27)
            sgx += W_3 * (d.phi[global3(x+1,y+1,z+1)] - d.phi[global3(x-1,y-1,z-1)] +
                          d.phi[global3(x+1,y+1,z-1)] - d.phi[global3(x-1,y-1,z+1)] +
                          d.phi[global3(x+1,y-1,z+1)] - d.phi[global3(x-1,y+1,z-1)] +
                          d.phi[global3(x+1,y-1,z-1)] - d.phi[global3(x-1,y+1,z+1)]);

            sgy += W_3 * (d.phi[global3(x+1,y+1,z+1)] - d.phi[global3(x-1,y-1,z-1)] +
                          d.phi[global3(x+1,y+1,z-1)] - d.phi[global3(x-1,y-1,z+1)] +
                          d.phi[global3(x-1,y+1,z-1)] - d.phi[global3(x+1,y-1,z+1)] +
                          d.phi[global3(x-1,y+1,z+1)] - d.phi[global3(x+1,y-1,z-1)]);

            sgz += W_3 * (d.phi[global3(x+1,y+1,z+1)] - d.phi[global3(x-1,y-1,z-1)] +
                          d.phi[global3(x-1,y-1,z+1)] - d.phi[global3(x+1,y+1,z-1)] +
                          d.phi[global3(x+1,y-1,z+1)] - d.phi[global3(x-1,y+1,z-1)] +
                          d.phi[global3(x-1,y+1,z+1)] - d.phi[global3(x+1,y-1,z-1)]);
        #endif

        const float gradX = 3.0f * sgx; 
        const float gradY = 3.0f * sgy; 
        const float gradZ = 3.0f * sgz; 
    
    #else

        const float sgx = WG_1 * (d.phi[global3(x+1,y,z)] - d.phi[global3(x-1,y,z)]);
        const float sgy = WG_1 * (d.phi[global3(x,y+1,z)] - d.phi[global3(x,y-1,z)]);
        const float sgz = WG_1 * (d.phi[global3(x,y,z+1)] - d.phi[global3(x,y,z-1)]);

        const float gradX = 4.0f * sgx;
        const float gradY = 4.0f * sgy;
        const float gradZ = 4.0f * sgz;

    #endif
    
    const float ind = sqrtf(gradX * gradX + gradY * gradY + gradZ * gradZ); 
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
        x == 0 || x == NX - 1 ||
        y == 0 || y == NY - 1 ||
        z == 0 || z == NZ - 1) return;

    const idx_t idx3 = global3(x, y, z);

    #if defined(HO_GRAD_STENCIL)

        float sumCurvX = W_1 * (d.normx[global3(x+1,y,z)]   - d.normx[global3(x-1,y,z)]) +
                         W_2 * (d.normx[global3(x+1,y+1,z)] - d.normx[global3(x-1,y-1,z)] +
                                d.normx[global3(x+1,y,z+1)] - d.normx[global3(x-1,y,z-1)] +
                                d.normx[global3(x+1,y-1,z)] - d.normx[global3(x-1,y+1,z)] +
                                d.normx[global3(x+1,y,z-1)] - d.normx[global3(x-1,y,z+1)]);

        float sumCurvY = W_1 * (d.normy[global3(x,y+1,z)]   - d.normy[global3(x,y-1,z)]) +
                         W_2 * (d.normy[global3(x+1,y+1,z)] - d.normy[global3(x-1,y-1,z)] +
                                d.normy[global3(x,y+1,z+1)] - d.normy[global3(x,y-1,z-1)] +
                                d.normy[global3(x-1,y+1,z)] - d.normy[global3(x+1,y-1,z)] +
                                d.normy[global3(x,y+1,z-1)] - d.normy[global3(x,y-1,z+1)]);

        float sumCurvZ = W_1 * (d.normz[global3(x,y,z+1)]   - d.normz[global3(x,y,z-1)]) +
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

    #else

        const float scx = WG_1 * (d.normx[global3(x+1,y,z)] - d.normx[global3(x-1,y,z)]);
        const float scy = WG_1 * (d.normy[global3(x,y+1,z)] - d.normy[global3(x,y-1,z)]);
        const float scz = WG_1 * (d.normz[global3(x,y,z+1)] - d.normz[global3(x,y,z-1)]);

        const float curvature = -4.0f * (scx + scy + scz);

    #endif

    const float stCurv = SIGMA * curvature * d.ind[idx3];
    d.ffx[idx3] = stCurv * d.normx[idx3];
    d.ffy[idx3] = stCurv * d.normy[idx3];
    d.ffz[idx3] = stCurv * d.normz[idx3];
}
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

    /*
    const float sg2 = fmaf(sgx, sgx, fmaf(sgy, sgy, sgz * sgz));

    const float sg2eps = fmaxf(sg2, 1e-18f);
    const float invInd = rsqrtf(sg2eps);
    const float ind = 3.0f * sg2eps * invInd;

    const float normX = sgx * invInd;
    const float normY = sgy * invInd;
    const float normZ = sgz * invInd;

    d.ind[idx3] = ind;
    d.normx[idx3] = normX;
    d.normy[idx3] = normY;
    d.normz[idx3] = normZ;
    */

    const float gradX = 3.0f * sgx; 
    const float gradY = 3.0f * sgy; 
    const float gradZ = 3.0f * sgz; 
    
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
void computeNormalsShared(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    const idx_t idx3 = global3(x, y, z);

    __shared__ float sharedPhi[blockSizeZ + 2][blockSizeY + 2][blockSizeX + 2];

    const idx_t lx = threadIdx.x + 1;
    const idx_t ly = threadIdx.y + 1;
    const idx_t lz = threadIdx.z + 1;

    sharedPhi[lz][ly][lx] = d.phi[global3(x, y, z)];

    if (threadIdx.x == 0 && x > 0)
        sharedPhi[lz][ly][lx-1] = d.phi[global3(x-1, y, z)];
    if (threadIdx.x == blockSizeX - 1 && x < NX-1)
        sharedPhi[lz][ly][lx+1] = d.phi[global3(x+1, y, z)];

    if (threadIdx.y == 0 && y > 0)
        sharedPhi[lz][ly-1][lx] = d.phi[global3(x, y-1, z)];
    if (threadIdx.y == blockSizeY - 1 && y < NY-1)
        sharedPhi[lz][ly+1][lx] = d.phi[global3(x, y+1, z)];

    if (threadIdx.z == 0 && z > 0)
        sharedPhi[lz-1][ly][lx] = d.phi[global3(x, y, z-1)];
    if (threadIdx.z == blockSizeZ - 1 && z < NZ-1)
        sharedPhi[lz+1][ly][lx] = d.phi[global3(x, y, z+1)];


    if (threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0)
        sharedPhi[lz][ly-1][lx-1] = d.phi[global3(x-1, y-1, z)];
    if (threadIdx.x == 0 && threadIdx.y == blockSizeY - 1 && x > 0 && y < NY-1)
        sharedPhi[lz][ly+1][lx-1] = d.phi[global3(x-1, y+1, z)];
    if (threadIdx.x == blockSizeX - 1 && threadIdx.y == 0 && x < NX-1 && y > 0)
        sharedPhi[lz][ly-1][lx+1] = d.phi[global3(x+1, y-1, z)];
    if (threadIdx.x == blockSizeX - 1 && threadIdx.y == blockSizeY - 1 && x < NX-1 && y < NY-1)
        sharedPhi[lz][ly+1][lx+1] = d.phi[global3(x+1, y+1, z)];


    if (threadIdx.x == 0 && threadIdx.z == 0 && x > 0 && z > 0)
        sharedPhi[lz-1][ly][lx-1] = d.phi[global3(x-1, y, z-1)];
    if (threadIdx.x == 0 && threadIdx.z == blockSizeZ - 1 && x > 0 && z < NZ-1)
        sharedPhi[lz+1][ly][lx-1] = d.phi[global3(x-1, y, z+1)];
    if (threadIdx.x == blockSizeX - 1 && threadIdx.z == 0 && x < NX-1 && z > 0)
        sharedPhi[lz-1][ly][lx+1] = d.phi[global3(x+1, y, z-1)];
    if (threadIdx.x == blockSizeX - 1 && threadIdx.z == blockSizeZ - 1 && x < NX-1 && z < NZ-1)
        sharedPhi[lz+1][ly][lx+1] = d.phi[global3(x+1, y, z+1)];


    if (threadIdx.y == 0 && threadIdx.z == 0 && y > 0 && z > 0)
        sharedPhi[lz-1][ly-1][lx] = d.phi[global3(x, y-1, z-1)];
    if (threadIdx.y == 0 && threadIdx.z == blockSizeZ - 1 && y > 0 && z < NZ-1)
        sharedPhi[lz+1][ly-1][lx] = d.phi[global3(x, y-1, z+1)];
    if (threadIdx.y == blockSizeY - 1 && threadIdx.z == 0 && y < NY-1 && z > 0)
        sharedPhi[lz-1][ly+1][lx] = d.phi[global3(x, y+1, z-1)];
    if (threadIdx.y == blockSizeY - 1 && threadIdx.z == blockSizeZ - 1 && y < NY-1 && z < NZ-1)
        sharedPhi[lz+1][ly+1][lx] = d.phi[global3(x, y+1, z+1)];

    __syncthreads();

    if (x == 0 || x == NX-1 ||
        y == 0 || y == NY-1 ||
        z == 0 || z == NZ-1) return;

    float sgx = W_1 * (sharedPhi[lz][ly][lx+1]   - sharedPhi[lz][ly][lx-1]) +
                W_2 * (sharedPhi[lz][ly+1][lx+1] - sharedPhi[lz][ly-1][lx-1] +
                       sharedPhi[lz+1][ly][lx+1] - sharedPhi[lz-1][ly][lx-1] +
                       sharedPhi[lz][ly-1][lx+1] - sharedPhi[lz][ly+1][lx-1] +
                       sharedPhi[lz-1][ly][lx+1] - sharedPhi[lz+1][ly][lx-1]); 

    float sgy = W_1 * (sharedPhi[lz][ly+1][lx]   - sharedPhi[lz][ly-1][lx]) +
                W_2 * (sharedPhi[lz][ly+1][lx+1] - sharedPhi[lz][ly-1][lx-1] +
                       sharedPhi[lz+1][ly+1][lx] - sharedPhi[lz-1][ly-1][lx] +
                       sharedPhi[lz][ly+1][lx-1] - sharedPhi[lz][ly-1][lx+1] +
                       sharedPhi[lz-1][ly+1][lx] - sharedPhi[lz+1][ly-1][lx]);

    float sgz = W_1 * (sharedPhi[lz+1][ly][lx]   - sharedPhi[lz-1][ly][lx]) +
                W_2 * (sharedPhi[lz+1][ly][lx+1] - sharedPhi[lz-1][ly][lx-1] +
                       sharedPhi[lz+1][ly+1][lx] - sharedPhi[lz-1][ly-1][lx] +
                       sharedPhi[lz+1][ly][lx-1] - sharedPhi[lz-1][ly][lx+1] +
                       sharedPhi[lz+1][ly-1][lx] - sharedPhi[lz-1][ly+1][lx]);

    const float gradX = 3.0f * sgx;
    const float gradY = 3.0f * sgy;
    const float gradZ = 3.0f * sgz;

    const float ind = sqrtf(gradX * gradX + gradY * gradY + gradZ * gradZ);
    const float invInd = 1.0f / (ind + 1e-9f);

    d.ind[idx3] = ind;
    d.normx[idx3] = gradX * invInd;
    d.normy[idx3] = gradY * invInd;
    d.normz[idx3] = gradZ * invInd;
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

    const float stCurv = SIGMA * curvature * d.ind[idx3];
    d.ffx[idx3] = stCurv * d.normx[idx3];
    d.ffy[idx3] = stCurv * d.normy[idx3];
    d.ffz[idx3] = stCurv * d.normz[idx3];
}
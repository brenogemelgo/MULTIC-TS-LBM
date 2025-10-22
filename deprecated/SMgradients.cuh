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

    __shared__ float sharedPhi[blockSizeZ + 2][blockSizeY + 2][blockSizeX + 2];

    const idx_t lx = threadIdx.x + 1;
    const idx_t ly = threadIdx.y + 1;
    const idx_t lz = threadIdx.z + 1;

    sharedPhi[lz][ly][lx] = d.phi[global3(x, y, z)];

    if (threadIdx.x == 0)              sharedPhi[lz][ly][lx-1] = d.phi[global3(x-1, y, z)];
    if (threadIdx.x == blockSizeX - 1) sharedPhi[lz][ly][lx+1] = d.phi[global3(x+1, y, z)];

    if (threadIdx.y == 0 )             sharedPhi[lz][ly-1][lx] = d.phi[global3(x, y-1, z)];
    if (threadIdx.y == blockSizeY - 1) sharedPhi[lz][ly+1][lx] = d.phi[global3(x, y+1, z)];

    if (threadIdx.z == 0)              sharedPhi[lz-1][ly][lx] = d.phi[global3(x, y, z-1)];
    if (threadIdx.z == blockSizeZ - 1) sharedPhi[lz+1][ly][lx] = d.phi[global3(x, y, z+1)];

    __syncthreads();

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
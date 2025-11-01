__global__ 
void computePhase(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz ||
        x == 0 || x == mesh::nx - 1 ||
        y == 0 || y == mesh::ny - 1 ||
        z == 0 || z == mesh::nz - 1) return;

    const idx_t idx3 = global3(x, y, z);

    float phi = 0.0f;
    constexpr_for<0, FLINKS>([&] __device__ (auto I) {
        constexpr idx_t Q = decltype(I)::value; 
        phi += d.g[Q * PLANE + idx3];
    });

    d.phi[idx3] = phi;
}

__global__ 
void computeNormals(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz ||
        x == 0 || x == mesh::nx - 1 ||
        y == 0 || y == mesh::ny - 1 ||
        z == 0 || z == mesh::nz - 1) return;

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

    const float gx = 3.0f * sgx; 
    const float gy = 3.0f * sgy; 
    const float gz = 3.0f * sgz; 
    
    const float ind = sqrtf(gx * gx + gy * gy + gz * gz); 
    const float invInd = 1.0f / (ind + 1e-9f); 
    
    const float normX = gx * invInd; 
    const float normY = gy * invInd; 
    const float normZ = gz * invInd; 
    
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

    if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz ||
        x == 0 || x == mesh::nx - 1 ||
        y == 0 || y == mesh::ny - 1 ||
        z == 0 || z == mesh::nz - 1) return;

    const idx_t idx3 = global3(x, y, z);

    float scx = W_1 * (d.normx[global3(x+1,y,z)]   - d.normx[global3(x-1,y,z)]) +
                W_2 * (d.normx[global3(x+1,y+1,z)] - d.normx[global3(x-1,y-1,z)] +
                       d.normx[global3(x+1,y,z+1)] - d.normx[global3(x-1,y,z-1)] +
                       d.normx[global3(x+1,y-1,z)] - d.normx[global3(x-1,y+1,z)] +
                       d.normx[global3(x+1,y,z-1)] - d.normx[global3(x-1,y,z+1)]);

    float scy = W_1 * (d.normy[global3(x,y+1,z)]   - d.normy[global3(x,y-1,z)]) +
                W_2 * (d.normy[global3(x+1,y+1,z)] - d.normy[global3(x-1,y-1,z)] +
                       d.normy[global3(x,y+1,z+1)] - d.normy[global3(x,y-1,z-1)] +
                       d.normy[global3(x-1,y+1,z)] - d.normy[global3(x+1,y-1,z)] +
                       d.normy[global3(x,y+1,z-1)] - d.normy[global3(x,y-1,z+1)]);

    float scz = W_1 * (d.normz[global3(x,y,z+1)]   - d.normz[global3(x,y,z-1)]) +
                W_2 * (d.normz[global3(x+1,y,z+1)] - d.normz[global3(x-1,y,z-1)] +
                       d.normz[global3(x,y+1,z+1)] - d.normz[global3(x,y-1,z-1)] +
                       d.normz[global3(x-1,y,z+1)] - d.normz[global3(x+1,y,z-1)] +
                       d.normz[global3(x,y-1,z+1)] - d.normz[global3(x,y+1,z-1)]);
                            
    #if defined(D3Q27)

        scx += W_3 * (d.normx[global3(x+1,y+1,z+1)] - d.normx[global3(x-1,y-1,z-1)] +
                      d.normx[global3(x+1,y+1,z-1)] - d.normx[global3(x-1,y-1,z+1)] +
                      d.normx[global3(x+1,y-1,z+1)] - d.normx[global3(x-1,y+1,z-1)] +
                      d.normx[global3(x+1,y-1,z-1)] - d.normx[global3(x-1,y+1,z+1)]);

        scy += W_3 * (d.normy[global3(x+1,y+1,z+1)] - d.normy[global3(x-1,y-1,z-1)] +
                      d.normy[global3(x+1,y+1,z-1)] - d.normy[global3(x-1,y-1,z+1)] +
                      d.normy[global3(x-1,y+1,z-1)] - d.normy[global3(x+1,y-1,z+1)] +
                      d.normy[global3(x-1,y+1,z+1)] - d.normy[global3(x+1,y-1,z-1)]);

        scz += W_3 * (d.normz[global3(x+1,y+1,z+1)] - d.normz[global3(x-1,y-1,z-1)] +
                      d.normz[global3(x-1,y-1,z+1)] - d.normz[global3(x+1,y+1,z-1)] +
                      d.normz[global3(x+1,y-1,z+1)] - d.normz[global3(x-1,y+1,z-1)] +
                      d.normz[global3(x-1,y+1,z+1)] - d.normz[global3(x+1,y-1,z-1)]);
                        
    #endif

    const float curvature = -3.0f * (scx + scy + scz);

    const float stCurv = physics::sigma * curvature * d.ind[idx3];
    d.ffx[idx3] = stCurv * d.normx[idx3];
    d.ffy[idx3] = stCurv * d.normy[idx3];
    d.ffz[idx3] = stCurv * d.normz[idx3];
}
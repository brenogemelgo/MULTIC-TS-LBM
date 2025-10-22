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

    if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz ||
        x == 0 || x == mesh::nx - 1 ||
        y == 0 || y == mesh::ny - 1 ||
        z == 0 || z == mesh::nz - 1) return;

    const idx_t idx3 = global3(x, y, z);

    const float sgx = WG_1 * (d.phi[global3(x + 1, y, z)] - d.phi[global3(x - 1, y, z)]);
    const float sgy = WG_1 * (d.phi[global3(x, y + 1, z)] - d.phi[global3(x, y - 1, z)]);
    const float sgz = WG_1 * (d.phi[global3(x, y, z + 1)] - d.phi[global3(x, y, z - 1)]);

    const float gx = 4.0f * sgx;
    const float gy = 4.0f * sgy;
    const float gz = 4.0f * sgz;
    
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

    const float scx = WG_1 * (d.normx[global3(x + 1, y, z)] - d.normx[global3(x - 1, y, z)]);
    const float scy = WG_1 * (d.normy[global3(x, y + 1, z)] - d.normy[global3(x, y - 1, z)]);
    const float scz = WG_1 * (d.normz[global3(x, y, z + 1)] - d.normz[global3(x, y, z - 1)]);

    const float curvature = -4.0f * (scx + scy + scz);

    const float stCurv = physics::sigma * curvature * d.ind[idx3];
    d.ffx[idx3] = stCurv * d.normx[idx3];
    d.ffy[idx3] = stCurv * d.normy[idx3];
    d.ffz[idx3] = stCurv * d.normz[idx3];
}
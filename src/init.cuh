#pragma once

__global__ 
void setFields(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    const idx_t idx3 = global3(x,y,z);

    d.ux[idx3] = 0.0f;
    d.uy[idx3] = 0.0f;
    d.uz[idx3] = 0.0f;
    d.phi[idx3] = 0.0f;
    d.rho[idx3] = 1.0f;
    d.ffx[idx3] = 0.0f * 1e-7f;
    d.ffy[idx3] = 0.0f * 1e-5f;
    d.ffz[idx3] = 0.0f * 1e-5f;
    d.normx[idx3] = 0.0f;
    d.normy[idx3] = 0.0f;
    d.normz[idx3] = 0.0f;
    d.pxx[idx3] = 0.0f;
    d.pyy[idx3] = 0.0f;
    d.pzz[idx3] = 0.0f;
    d.pxy[idx3] = 0.0f;
    d.pxz[idx3] = 0.0f;
    d.pyz[idx3] = 0.0f;
}

__global__ 
void setDistros(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    const idx_t idx3 = global3(x,y,z);

    #pragma unroll FLINKS
    for (idx_t Q = 0; Q < FLINKS; ++Q) {
        d.f[global4(x,y,z,Q)] = computeEquilibria(d.rho[idx3],d.ux[idx3],d.uy[idx3],d.uz[idx3],Q);
    }
    #pragma unroll GLINKS
    for (idx_t Q = 0; Q < GLINKS; ++Q) {
        d.g[global4(x+CIX[Q],y+CIY[Q],z+CIZ[Q],Q)] = computeTruncatedEquilibria(d.phi[idx3],d.ux[idx3],d.uy[idx3],d.uz[idx3],Q);
    }
} 

#if defined(JET)

__global__ 
void setJet(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = 0;

    if (x >= NX || y >= NY) return;

    const float dx = static_cast<float>(x) - CENTER_X;
    const float dy = static_cast<float>(y) - CENTER_Y;
    const float radialDist = sqrtf(dx*dx + dy*dy);
    const float radius = 0.5f * static_cast<float>(DIAM);
    if (radialDist > radius) return;

    const idx_t idx3_in = global3(x,y,z);
    d.uz[idx3_in] = U_REF;
    d.phi[idx3_in] = 1.0f;
}

#elif defined(DROPLET)

__global__ 
void setDroplet(
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

    const float dx = (static_cast<float>(x) - CENTER_X) / 2.0f;
    const float dy = static_cast<float>(y) - CENTER_Y;
    const float dz = static_cast<float>(z) - CENTER_Z;
    const float radialDist = sqrtf(dx*dx + dy*dy + dz*dz);

    const float phi = 0.5f + 0.5f * tanhf(2.0f * (static_cast<float>(RADIUS)-radialDist) / 3.0f);
    d.phi[idx3] = phi;
}

#endif


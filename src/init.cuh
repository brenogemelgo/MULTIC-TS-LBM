#pragma once

__global__ 
void setFields(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz) return;

    const idx_t idx3 = global3(x, y, z);

    d.ux[idx3] = 0.0f;
    d.uy[idx3] = 0.0f;
    d.uz[idx3] = 0.0f;
    d.phi[idx3] = 0.0f;
    d.ffx[idx3] = 0.0f;
    d.ffy[idx3] = 0.0f;
    d.ffz[idx3] = 0.0f;
    d.normx[idx3] = 0.0f;
    d.normy[idx3] = 0.0f;
    d.normz[idx3] = 0.0f;
    d.rho[idx3] = 1.0f;
    d.pxx[idx3] = 0.0f;
    d.pyy[idx3] = 0.0f;
    d.pzz[idx3] = 0.0f;
    d.pxy[idx3] = 0.0f;
    d.pxz[idx3] = 0.0f;
    d.pyz[idx3] = 0.0f;
}

#if defined(JET)

__global__ void setJet(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= mesh::nx || y >= mesh::ny) return;

    const float dx = static_cast<float>(x) - CENTER_X;
    const float dy = static_cast<float>(y) - CENTER_Y;
    const float r2 = dx * dx + dy * dy;
    if (r2 > R2) return;

    const idx_t idx3_in = global3(x, y, 0);

    d.phi[idx3_in] = 1.0f;
    d.uz[idx3_in] = physics::u_ref;
}

#elif defined(DROPLET)

__global__ void 
setDroplet(
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

    const float dx = (static_cast<float>(x) - CENTER_X) / 2.0f;
    const float dy = static_cast<float>(y) - CENTER_Y;
    const float dz = static_cast<float>(z) - CENTER_Z;
    const float radialDist = sqrtf(dx * dx + dy * dy + dz * dz);

    const float phi = 0.5f + 0.5f * tanhf(2.0f * (static_cast<float>(mesh::radius) - radialDist) / 3.0f);
    d.phi[idx3] = phi;
}

#endif

__global__ 
void setDistros(
    LBMFields d
) {
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz)  return;

    const idx_t idx3 = global3(x, y, z);

    const float ux = d.ux[idx3];
    const float uy = d.uy[idx3];
    const float uz = d.uz[idx3];

    const float uu = 1.5f * (ux*ux + uy*uy + uz*uz);
    constexpr_for<0, FLINKS>([&] __device__ (auto I) {
        constexpr idx_t Q = decltype(I)::value;

        constexpr float w  = VelocitySet::F<Q>::w;
        constexpr float cx = static_cast<float>(VelocitySet::F<Q>::cx); 
        constexpr float cy = static_cast<float>(VelocitySet::F<Q>::cy); 
        constexpr float cz = static_cast<float>(VelocitySet::F<Q>::cz); 

        const float cu = 3.0f * (cx * ux + cy * uy + cz * uz);

        #if defined(D3Q19)

            const float feq = w * d.rho[idx3] * (1.0f - uu + cu + 0.5f * cu * cu) - w;

        #elif defined(D3Q27)

            const float feq = w * d.rho[idx3] * (1.0f - uu + cu + 0.5f * cu * cu + OOS * cu * cu * cu - uu * cu) - w;
            
        #endif

        d.f[global4(x, y, z, Q)] = to_pop(feq);
    });

    constexpr_for<0, GLINKS>([&] __device__ (auto I) {
        constexpr idx_t Q = decltype(I)::value;

        const idx_t xx = x + static_cast<idx_t>(VelocitySet::G<Q>::cx);
        const idx_t yy = y + static_cast<idx_t>(VelocitySet::G<Q>::cy);
        const idx_t zz = z + static_cast<idx_t>(VelocitySet::G<Q>::cz);

        constexpr float wg = VelocitySet::G<Q>::wg;
        constexpr float cx = static_cast<float>(VelocitySet::G<Q>::cx); 
        constexpr float cy = static_cast<float>(VelocitySet::G<Q>::cy); 
        constexpr float cz = static_cast<float>(VelocitySet::G<Q>::cz); 

        const float cu = 4.0f * (cx * ux + cy * uy + cz * uz);

        d.g[global4(xx, yy, zz, Q)] = wg * d.phi[idx3] * (1.0f + cu);
    });
}

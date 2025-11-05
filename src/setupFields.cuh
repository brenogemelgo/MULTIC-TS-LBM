#pragma once

__global__ void setFields(
    LBMFields d)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz)
        return;

    const label_t idx3 = device::global3(x, y, z);

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
    LBMFields d)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= mesh::nx || y >= mesh::ny)
        return;

    const scalar_t dx = static_cast<scalar_t>(x) - CENTER_X;
    const scalar_t dy = static_cast<scalar_t>(y) - CENTER_Y;
    const scalar_t r2 = dx * dx + dy * dy;
    if (r2 > R2)
        return;

    const label_t idx3_in = device::global3(x, y, 0);

    d.phi[idx3_in] = 1.0f;
    d.uz[idx3_in] = physics::u_ref;
}

#elif defined(DROPLET)

__global__ void
setDroplet(
    LBMFields d)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz ||
        x == 0 || x == mesh::nx - 1 ||
        y == 0 || y == mesh::ny - 1 ||
        z == 0 || z == mesh::nz - 1)
        return;

    const label_t idx3 = device::global3(x, y, z);

    const scalar_t dx = (static_cast<scalar_t>(x) - CENTER_X) / 2.0f;
    const scalar_t dy = static_cast<scalar_t>(y) - CENTER_Y;
    const scalar_t dz = static_cast<scalar_t>(z) - CENTER_Z;
    const scalar_t radialDist = sqrtf(dx * dx + dy * dy + dz * dz);

    const scalar_t phi = 0.5f + 0.5f * tanhf(2.0f * (static_cast<scalar_t>(mesh::radius) - radialDist) / 3.0f);
    d.phi[idx3] = phi;
}

#endif

__global__ void setDistros(
    LBMFields d)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz)
        return;

    const label_t idx3 = device::global3(x, y, z);

    const scalar_t ux = d.ux[idx3];
    const scalar_t uy = d.uy[idx3];
    const scalar_t uz = d.uz[idx3];

    const scalar_t uu = 1.5f * (ux * ux + uy * uy + uz * uz);
    device::constexpr_for<0, FLINKS>(
        [&](const auto Q)
        {
            constexpr scalar_t w = VelocitySet::F<Q>::w;
            constexpr scalar_t cx = static_cast<scalar_t>(VelocitySet::F<Q>::cx);
            constexpr scalar_t cy = static_cast<scalar_t>(VelocitySet::F<Q>::cy);
            constexpr scalar_t cz = static_cast<scalar_t>(VelocitySet::F<Q>::cz);

            const scalar_t cu = 3.0f * (cx * ux + cy * uy + cz * uz);

#if defined(D3Q19)

            const scalar_t feq = w * d.rho[idx3] * (1.0f - uu + cu + 0.5f * cu * cu) - w;

#elif defined(D3Q27)

            const scalar_t feq = w * d.rho[idx3] * (1.0f - uu + cu + 0.5f * cu * cu + OOS * cu * cu * cu - uu * cu) - w;

#endif

            d.f[device::global4(x, y, z, Q)] = to_pop(feq);
        });

    device::constexpr_for<0, GLINKS>(
        [&](const auto Q)
        {
            const label_t xx = x + static_cast<label_t>(VelocitySet::G<Q>::cx);
            const label_t yy = y + static_cast<label_t>(VelocitySet::G<Q>::cy);
            const label_t zz = z + static_cast<label_t>(VelocitySet::G<Q>::cz);

            constexpr scalar_t cx = static_cast<scalar_t>(VelocitySet::G<Q>::cx);
            constexpr scalar_t cy = static_cast<scalar_t>(VelocitySet::G<Q>::cy);
            constexpr scalar_t cz = static_cast<scalar_t>(VelocitySet::G<Q>::cz);

            d.g[device::global4(xx, yy, zz, Q)] = VelocitySet::G<Q>::wg * d.phi[idx3] * (1.0f + AS2_P * (cx * ux + cy * uy + cz * uz));
        });
}

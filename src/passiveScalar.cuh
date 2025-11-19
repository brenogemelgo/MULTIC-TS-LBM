#pragma once

#if PASSIVE_SCALAR

namespace LBM
{
    __global__ void passiveScalar(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz ||
            x == 0 || x == mesh::nx - 1 ||
            y == 0 || y == mesh::ny - 1 ||
            z == 0 || z == mesh::nz - 1)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        scalar_t chi = 0.0f;
        device::constexpr_for<0, HLINKS>(
            [&](const auto Q)
            {
                chi += d.h[Q * size::plane() + idx3];
            });

        d.chi[idx3] = chi;

        const scalar_t ux = d.ux[idx3];
        const scalar_t uy = d.uy[idx3];
        const scalar_t uz = d.uz[idx3];

        device::constexpr_for<0, HLINKS>(
            [&](const auto Q)
            {
                constexpr scalar_t wh = VelocitySet::H<Q>::wh;
                constexpr scalar_t cx = static_cast<scalar_t>(VelocitySet::H<Q>::cx);
                constexpr scalar_t cy = static_cast<scalar_t>(VelocitySet::H<Q>::cy);
                constexpr scalar_t cz = static_cast<scalar_t>(VelocitySet::H<Q>::cz);

                const scalar_t cu = 4.0f * (cx * ux + cy * uy + cz * uz);

                const scalar_t heq = wh * chi * (1.0f + cu);

                d.h_post[device::global4(x, y, z, Q)] = heq + (1.0f - (1.0f / 0.55f)) * (d.h[device::global4(x, y, z, Q)] - heq);
            });

        device::constexpr_for<0, HLINKS>(
            [&](const auto Q)
            {
                const label_t xx = x - static_cast<label_t>(VelocitySet::H<Q>::cx);
                const label_t yy = y - static_cast<label_t>(VelocitySet::H<Q>::cy);
                const label_t zz = z - static_cast<label_t>(VelocitySet::H<Q>::cz);

                d.h[device::global4(x, y, z, Q)] = d.h_post[device::global4(xx, yy, zz, Q)];
            });
    }
}

#endif
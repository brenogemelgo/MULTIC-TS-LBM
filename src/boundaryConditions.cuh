#pragma once

#if defined(JET)

__global__ void applyInflow(LBMFields d)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= mesh::nx || y >= mesh::ny)
        return;

    const float dx = static_cast<float>(x) - CENTER_X;
    const float dy = static_cast<float>(y) - CENTER_Y;
    const float r2 = dx * dx + dy * dy;
    if (r2 > R2)
        return;

    const label_t idx3_bnd = device::global3(x, y, 0);
    const label_t idx3_zp1 = device::global3(x, y, 1);

    // d.rho[idx3_bnd] = d.rho[idx3_zp1];

    device::constexpr_for<0, FLINKS>(
        [&](const auto Q)
        {
            if constexpr (VelocitySet::F<Q>::cz == 1)
            {
                const label_t xx = x + static_cast<label_t>(VelocitySet::F<Q>::cx);
                const label_t yy = y + static_cast<label_t>(VelocitySet::F<Q>::cy);

                label_t fluidNode = device::global3(xx, yy, 1);

                constexpr float w = VelocitySet::F<Q>::w;
                constexpr float cx = static_cast<float>(VelocitySet::F<Q>::cx);
                constexpr float cy = static_cast<float>(VelocitySet::F<Q>::cy);
                constexpr float cz = static_cast<float>(VelocitySet::F<Q>::cz);

                float feq = w * d.rho[fluidNode] * P_IN - w;

#if defined(D3Q19)

                const float fneq = (w * 4.5f) *
                                   ((cx * cx - CSSQ) * d.pxx[fluidNode] +
                                    (cy * cy - CSSQ) * d.pyy[fluidNode] +
                                    (cz * cz - CSSQ) * d.pzz[fluidNode] +
                                    2.0f * (cx * cy * d.pxy[fluidNode] +
                                            cx * cz * d.pxz[fluidNode] +
                                            cy * cz * d.pyz[fluidNode]));

#elif defined(D3Q27)

                const float fneq = (w * 4.5f) *
                                   ((cx * cx - CSSQ) * d.pxx[fluidNode] +
                                    (cy * cy - CSSQ) * d.pyy[fluidNode] +
                                    (cz * cz - CSSQ) * d.pzz[fluidNode] +
                                    2.0f * (cx * cy * d.pxy[fluidNode] +
                                            cx * cz * d.pxz[fluidNode] +
                                            cy * cz * d.pyz[fluidNode]) +
                                    (cx * cx * cx - 3.0f * CSSQ * cx) * (3.0f * d.ux[fluidNode] * d.pxx[fluidNode]) +
                                    (cy * cy * cy - 3.0f * CSSQ * cy) * (3.0f * d.uy[fluidNode] * d.pyy[fluidNode]) +
                                    (cz * cz * cz - 3.0f * CSSQ * cz) * (3.0f * d.uz[fluidNode] * d.pzz[fluidNode]) +
                                    3.0f * ((cx * cx * cy - CSSQ * cy) * (d.pxx[fluidNode] * d.uy[fluidNode] + 2.0f * d.ux[fluidNode] * d.pxy[fluidNode]) +
                                            (cx * cx * cz - CSSQ * cz) * (d.pxx[fluidNode] * d.uz[fluidNode] + 2.0f * d.ux[fluidNode] * d.pxz[fluidNode]) +
                                            (cx * cy * cy - CSSQ * cx) * (d.pxy[fluidNode] * d.uy[fluidNode] + 2.0f * d.ux[fluidNode] * d.pyy[fluidNode]) +
                                            (cy * cy * cz - CSSQ * cz) * (d.pyy[fluidNode] * d.uz[fluidNode] + 2.0f * d.uy[fluidNode] * d.pyz[fluidNode]) +
                                            (cx * cz * cz - CSSQ * cx) * (d.pxz[fluidNode] * d.uz[fluidNode] + 2.0f * d.ux[fluidNode] * d.pzz[fluidNode]) +
                                            (cy * cz * cz - CSSQ * cy) * (d.pyz[fluidNode] * d.uz[fluidNode] + 2.0f * d.uy[fluidNode] * d.pzz[fluidNode])) +
                                    6.0f * (cx * cy * cz) * (d.ux[fluidNode] * d.pyz[fluidNode] + d.uy[fluidNode] * d.pxz[fluidNode] + d.uz[fluidNode] * d.pxy[fluidNode]));

#endif

                d.f[Q * PLANE + fluidNode] = to_pop(feq + OMCO_ZMIN * fneq);

                feq = w * d.phi[idx3_bnd] * CONST_IN;

                d.g[Q * PLANE + fluidNode] = feq;
            }
        });
}

__global__ void applyOutflow(LBMFields d)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= mesh::nx || y >= mesh::ny)
        return;

    const label_t idx3_bnd = device::global3(x, y, mesh::nz - 1);
    const label_t idx3_zm1 = device::global3(x, y, mesh::nz - 2);

    d.phi[idx3_bnd] = d.phi[idx3_zm1];
    // d.rho[idx3_bnd] = d.rho[idx3_zm1];
    // d.ux[idx3_bnd] = d.ux[idx3_zm1];
    // d.uy[idx3_bnd] = d.uy[idx3_zm1];
    // d.uz[idx3_bnd] = d.uz[idx3_zm1];

    const float ux = d.ux[idx3_zm1];
    const float uy = d.uy[idx3_zm1];
    const float uz = d.uz[idx3_zm1];

    const float uu = 1.5f * (ux * ux + uy * uy + uz * uz);

    device::constexpr_for<0, FLINKS>(
        [&](const auto Q)
        {
            if constexpr (VelocitySet::F<Q>::cz == -1)
            {
                const label_t xx = x + static_cast<label_t>(VelocitySet::F<Q>::cx);
                const label_t yy = y + static_cast<label_t>(VelocitySet::F<Q>::cy);

                label_t fluidNode = device::global3(xx, yy, mesh::nz - 2);

                constexpr float w = VelocitySet::F<Q>::w;
                constexpr float cx = static_cast<float>(VelocitySet::F<Q>::cx);
                constexpr float cy = static_cast<float>(VelocitySet::F<Q>::cy);
                constexpr float cz = static_cast<float>(VelocitySet::F<Q>::cz);

                const float cu = 3.0f * (cx * ux + cy * uy + cz * uz);

#if defined(D3Q19)

                float feq = w * d.rho[fluidNode] * (1.0f - uu + cu + 0.5f * cu * cu) - w;

#elif defined(D3Q27)

                float feq = w * d.rho[fluidNode] * (1.0f - uu + cu + 0.5f * cu * cu + OOS * cu * cu * cu - uu * cu) - w;

#endif

#if defined(D3Q19)

                const float fneq = (w * 4.5f) *
                                   ((cx * cx - CSSQ) * d.pxx[fluidNode] +
                                    (cy * cy - CSSQ) * d.pyy[fluidNode] +
                                    (cz * cz - CSSQ) * d.pzz[fluidNode] +
                                    2.0f * (cx * cy * d.pxy[fluidNode] +
                                            cx * cz * d.pxz[fluidNode] +
                                            cy * cz * d.pyz[fluidNode]));

#elif defined(D3Q27)

                const float fneq = (w * 4.5f) *
                                   ((cx * cx - CSSQ) * d.pxx[fluidNode] +
                                    (cy * cy - CSSQ) * d.pyy[fluidNode] +
                                    (cz * cz - CSSQ) * d.pzz[fluidNode] +
                                    2.0f * (cx * cy * d.pxy[fluidNode] +
                                            cx * cz * d.pxz[fluidNode] +
                                            cy * cz * d.pyz[fluidNode]) +
                                    (cx * cx * cx - 3.0f * CSSQ * cx) * (3.0f * d.ux[fluidNode] * d.pxx[fluidNode]) +
                                    (cy * cy * cy - 3.0f * CSSQ * cy) * (3.0f * d.uy[fluidNode] * d.pyy[fluidNode]) +
                                    (cz * cz * cz - 3.0f * CSSQ * cz) * (3.0f * d.uz[fluidNode] * d.pzz[fluidNode]) +
                                    3.0f * ((cx * cx * cy - CSSQ * cy) * (d.pxx[fluidNode] * d.uy[fluidNode] + 2.0f * d.ux[fluidNode] * d.pxy[fluidNode]) +
                                            (cx * cx * cz - CSSQ * cz) * (d.pxx[fluidNode] * d.uz[fluidNode] + 2.0f * d.ux[fluidNode] * d.pxz[fluidNode]) +
                                            (cx * cy * cy - CSSQ * cx) * (d.pxy[fluidNode] * d.uy[fluidNode] + 2.0f * d.ux[fluidNode] * d.pyy[fluidNode]) +
                                            (cy * cy * cz - CSSQ * cz) * (d.pyy[fluidNode] * d.uz[fluidNode] + 2.0f * d.uy[fluidNode] * d.pyz[fluidNode]) +
                                            (cx * cz * cz - CSSQ * cx) * (d.pxz[fluidNode] * d.uz[fluidNode] + 2.0f * d.ux[fluidNode] * d.pzz[fluidNode]) +
                                            (cy * cz * cz - CSSQ * cy) * (d.pyz[fluidNode] * d.uz[fluidNode] + 2.0f * d.uy[fluidNode] * d.pzz[fluidNode])) +
                                    6.0f * (cx * cy * cz) * (d.ux[fluidNode] * d.pyz[fluidNode] + d.uy[fluidNode] * d.pxz[fluidNode] + d.uz[fluidNode] * d.pxy[fluidNode]));

#endif

                d.f[Q * PLANE + fluidNode] = to_pop(feq + OMCO_ZMAX * fneq);

                feq = w * d.phi[fluidNode] * CONST_OUT;

                d.g[Q * PLANE + fluidNode] = feq;
            }
        });
}

__global__ void periodicX(LBMFields d)
{
    const label_t y = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t z = threadIdx.y + blockIdx.y * blockDim.y;

    if (y <= 0 || y >= mesh::ny - 1 || z <= 0 || z >= mesh::nz - 1)
        return;

    const label_t bL = device::global3(1, y, z);
    const label_t bR = device::global3(mesh::nx - 2, y, z);

    device::constexpr_for<0, FLINKS>(
        [&](const auto Q)
        {
            constexpr int cx = VelocitySet::F<Q>::cx;

            if constexpr (cx > 0)
            {
                d.f[Q * PLANE + bL] = d.f[Q * PLANE + bR];
                d.g[Q * PLANE + bL] = d.g[Q * PLANE + bR];
            }
            if constexpr (cx < 0)
            {
                d.f[Q * PLANE + bR] = d.f[Q * PLANE + bL];
                d.g[Q * PLANE + bR] = d.g[Q * PLANE + bL];
            }
        });

    const label_t gL = device::global3(0, y, z);
    const label_t gR = device::global3(mesh::nx - 1, y, z);

    d.phi[gL] = d.phi[bR];
    d.phi[gR] = d.phi[bL];

    // d.rho[gL] = d.rho[bR];
    // d.rho[gR] = d.rho[bL];

    // d.ux[gL] = d.ux[bR];
    // d.ux[gR] = d.ux[bL];

    // d.uy[gL] = d.uy[bR];
    // d.uy[gR] = d.uy[bL];

    // d.uz[gL] = d.uz[bR];
    // d.uz[gR] = d.uz[bL];
}

__global__ void periodicY(LBMFields d)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t z = threadIdx.y + blockIdx.y * blockDim.y;

    if (x <= 0 || x >= mesh::nx - 1 || z <= 0 || z >= mesh::nz - 1)
        return;

    const label_t bB = device::global3(x, 1, z);
    const label_t bT = device::global3(x, mesh::ny - 2, z);

    device::constexpr_for<0, FLINKS>(
        [&](const auto Q)
        {
            constexpr int cy = VelocitySet::F<Q>::cy;

            if constexpr (cy > 0)
            {
                d.f[Q * PLANE + bB] = d.f[Q * PLANE + bT];
                d.g[Q * PLANE + bB] = d.g[Q * PLANE + bT];
            }
            if constexpr (cy < 0)
            {
                d.f[Q * PLANE + bT] = d.f[Q * PLANE + bB];
                d.g[Q * PLANE + bT] = d.g[Q * PLANE + bB];
            }
        });

    const label_t gB = device::global3(x, 0, z);
    const label_t gT = device::global3(x, mesh::ny - 1, z);

    // ghost cells
    d.phi[gB] = d.phi[bT];
    d.phi[gT] = d.phi[bB];

    // d.rho[gB] = d.rho[bT];
    // d.rho[gT] = d.rho[bB];

    // d.ux[gB] = d.ux[bT];
    // d.ux[gT] = d.ux[bB];

    // d.uy[gB] = d.uy[bT];
    // d.uy[gT] = d.uy[bB];

    // d.uz[gB] = d.uz[bT];
    // d.uz[gT] = d.uz[bB];
}

#elif defined(DROPLET)

#endif

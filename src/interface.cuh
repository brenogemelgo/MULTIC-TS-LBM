#pragma once

namespace phase
{
    __global__ void computePhase(LBMFields d)
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

        scalar_t phi = 0.0f;
        device::constexpr_for<0, GLINKS>(
            [&](const auto Q)
            {
                phi += d.g[Q * size::plane() + idx3];
            });

        d.phi[idx3] = phi;
    }

    __global__ void computeNormals(LBMFields d)
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

        scalar_t sgx = W_1 * (d.phi[device::global3(x + 1, y, z)] - d.phi[device::global3(x - 1, y, z)]) +
                       W_2 * (d.phi[device::global3(x + 1, y + 1, z)] - d.phi[device::global3(x - 1, y - 1, z)] +
                              d.phi[device::global3(x + 1, y, z + 1)] - d.phi[device::global3(x - 1, y, z - 1)] +
                              d.phi[device::global3(x + 1, y - 1, z)] - d.phi[device::global3(x - 1, y + 1, z)] +
                              d.phi[device::global3(x + 1, y, z - 1)] - d.phi[device::global3(x - 1, y, z + 1)]);

        scalar_t sgy = W_1 * (d.phi[device::global3(x, y + 1, z)] - d.phi[device::global3(x, y - 1, z)]) +
                       W_2 * (d.phi[device::global3(x + 1, y + 1, z)] - d.phi[device::global3(x - 1, y - 1, z)] +
                              d.phi[device::global3(x, y + 1, z + 1)] - d.phi[device::global3(x, y - 1, z - 1)] +
                              d.phi[device::global3(x - 1, y + 1, z)] - d.phi[device::global3(x + 1, y - 1, z)] +
                              d.phi[device::global3(x, y + 1, z - 1)] - d.phi[device::global3(x, y - 1, z + 1)]);

        scalar_t sgz = W_1 * (d.phi[device::global3(x, y, z + 1)] - d.phi[device::global3(x, y, z - 1)]) +
                       W_2 * (d.phi[device::global3(x + 1, y, z + 1)] - d.phi[device::global3(x - 1, y, z - 1)] +
                              d.phi[device::global3(x, y + 1, z + 1)] - d.phi[device::global3(x, y - 1, z - 1)] +
                              d.phi[device::global3(x - 1, y, z + 1)] - d.phi[device::global3(x + 1, y, z - 1)] +
                              d.phi[device::global3(x, y - 1, z + 1)] - d.phi[device::global3(x, y + 1, z - 1)]);

#if defined(D3Q27)

        sgx += W_3 * (d.phi[device::global3(x + 1, y + 1, z + 1)] - d.phi[device::global3(x - 1, y - 1, z - 1)] +
                      d.phi[device::global3(x + 1, y + 1, z - 1)] - d.phi[device::global3(x - 1, y - 1, z + 1)] +
                      d.phi[device::global3(x + 1, y - 1, z + 1)] - d.phi[device::global3(x - 1, y + 1, z - 1)] +
                      d.phi[device::global3(x + 1, y - 1, z - 1)] - d.phi[device::global3(x - 1, y + 1, z + 1)]);

        sgy += W_3 * (d.phi[device::global3(x + 1, y + 1, z + 1)] - d.phi[device::global3(x - 1, y - 1, z - 1)] +
                      d.phi[device::global3(x + 1, y + 1, z - 1)] - d.phi[device::global3(x - 1, y - 1, z + 1)] +
                      d.phi[device::global3(x - 1, y + 1, z - 1)] - d.phi[device::global3(x + 1, y - 1, z + 1)] +
                      d.phi[device::global3(x - 1, y + 1, z + 1)] - d.phi[device::global3(x + 1, y - 1, z - 1)]);

        sgz += W_3 * (d.phi[device::global3(x + 1, y + 1, z + 1)] - d.phi[device::global3(x - 1, y - 1, z - 1)] +
                      d.phi[device::global3(x - 1, y - 1, z + 1)] - d.phi[device::global3(x + 1, y + 1, z - 1)] +
                      d.phi[device::global3(x + 1, y - 1, z + 1)] - d.phi[device::global3(x - 1, y + 1, z - 1)] +
                      d.phi[device::global3(x - 1, y + 1, z + 1)] - d.phi[device::global3(x + 1, y - 1, z - 1)]);

#endif

        // Use the hydrodynamic scale factor
        const scalar_t gx = AS2_H * sgx;
        const scalar_t gy = AS2_H * sgy;
        const scalar_t gz = AS2_H * sgz;

        const scalar_t ind = sqrtf(gx * gx + gy * gy + gz * gz);
        const scalar_t invInd = 1.0f / (ind + 1e-9f);

        const scalar_t normX = gx * invInd;
        const scalar_t normY = gy * invInd;
        const scalar_t normZ = gz * invInd;

        d.ind[idx3] = ind;
        d.normx[idx3] = normX;
        d.normy[idx3] = normY;
        d.normz[idx3] = normZ;
    }

    __global__ void computeForces(LBMFields d)
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

        scalar_t scx = W_1 * (d.normx[device::global3(x + 1, y, z)] - d.normx[device::global3(x - 1, y, z)]) +
                       W_2 * (d.normx[device::global3(x + 1, y + 1, z)] - d.normx[device::global3(x - 1, y - 1, z)] +
                              d.normx[device::global3(x + 1, y, z + 1)] - d.normx[device::global3(x - 1, y, z - 1)] +
                              d.normx[device::global3(x + 1, y - 1, z)] - d.normx[device::global3(x - 1, y + 1, z)] +
                              d.normx[device::global3(x + 1, y, z - 1)] - d.normx[device::global3(x - 1, y, z + 1)]);

        scalar_t scy = W_1 * (d.normy[device::global3(x, y + 1, z)] - d.normy[device::global3(x, y - 1, z)]) +
                       W_2 * (d.normy[device::global3(x + 1, y + 1, z)] - d.normy[device::global3(x - 1, y - 1, z)] +
                              d.normy[device::global3(x, y + 1, z + 1)] - d.normy[device::global3(x, y - 1, z - 1)] +
                              d.normy[device::global3(x - 1, y + 1, z)] - d.normy[device::global3(x + 1, y - 1, z)] +
                              d.normy[device::global3(x, y + 1, z - 1)] - d.normy[device::global3(x, y - 1, z + 1)]);

        scalar_t scz = W_1 * (d.normz[device::global3(x, y, z + 1)] - d.normz[device::global3(x, y, z - 1)]) +
                       W_2 * (d.normz[device::global3(x + 1, y, z + 1)] - d.normz[device::global3(x - 1, y, z - 1)] +
                              d.normz[device::global3(x, y + 1, z + 1)] - d.normz[device::global3(x, y - 1, z - 1)] +
                              d.normz[device::global3(x - 1, y, z + 1)] - d.normz[device::global3(x + 1, y, z - 1)] +
                              d.normz[device::global3(x, y - 1, z + 1)] - d.normz[device::global3(x, y + 1, z - 1)]);

#if defined(D3Q27)

        scx += W_3 * (d.normx[device::global3(x + 1, y + 1, z + 1)] - d.normx[device::global3(x - 1, y - 1, z - 1)] +
                      d.normx[device::global3(x + 1, y + 1, z - 1)] - d.normx[device::global3(x - 1, y - 1, z + 1)] +
                      d.normx[device::global3(x + 1, y - 1, z + 1)] - d.normx[device::global3(x - 1, y + 1, z - 1)] +
                      d.normx[device::global3(x + 1, y - 1, z - 1)] - d.normx[device::global3(x - 1, y + 1, z + 1)]);

        scy += W_3 * (d.normy[device::global3(x + 1, y + 1, z + 1)] - d.normy[device::global3(x - 1, y - 1, z - 1)] +
                      d.normy[device::global3(x + 1, y + 1, z - 1)] - d.normy[device::global3(x - 1, y - 1, z + 1)] +
                      d.normy[device::global3(x - 1, y + 1, z - 1)] - d.normy[device::global3(x + 1, y - 1, z + 1)] +
                      d.normy[device::global3(x - 1, y + 1, z + 1)] - d.normy[device::global3(x + 1, y - 1, z - 1)]);

        scz += W_3 * (d.normz[device::global3(x + 1, y + 1, z + 1)] - d.normz[device::global3(x - 1, y - 1, z - 1)] +
                      d.normz[device::global3(x - 1, y - 1, z + 1)] - d.normz[device::global3(x + 1, y + 1, z - 1)] +
                      d.normz[device::global3(x + 1, y - 1, z + 1)] - d.normz[device::global3(x - 1, y + 1, z - 1)] +
                      d.normz[device::global3(x - 1, y + 1, z + 1)] - d.normz[device::global3(x + 1, y - 1, z - 1)]);

#endif

        // Use the hydrodynamic scale factor
        const scalar_t curvature = -AS2_H * (scx + scy + scz);

        const scalar_t stCurv = physics::sigma * curvature * d.ind[idx3];
        d.ffx[idx3] = stCurv * d.normx[idx3];
        d.ffy[idx3] = stCurv * d.normy[idx3];
        d.ffz[idx3] = stCurv * d.normz[idx3];
    }
}
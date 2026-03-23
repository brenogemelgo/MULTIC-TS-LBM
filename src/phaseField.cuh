/*---------------------------------------------------------------------------*\
|                                                                             |
| phaseFieldLBM: CUDA-based multicomponent Lattice Boltzmann Method           |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/brenogemelgo/phaseFieldLBM                       |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Description
    High-order phase-field kernels computing order parameter, interface normals, curvature, and surface-tension forcing using D3Q19/D3Q27 stencils

Namespace
    phase

SourceFiles
    phaseField.cuh

\*---------------------------------------------------------------------------*/

#ifndef PHASEFIELD_CUH
#define PHASEFIELD_CUH
#include "config/PhaseVelocitySet.cuh"

namespace phase
{
    template <typename VelocitySet>
    __global__ void computePhase(LBMFields d)
    {
        const label_t x = threadIdx.x + block::nx() * blockIdx.x;
        const label_t y = threadIdx.y + block::ny() * blockIdx.y;
        const label_t z = threadIdx.z + block::nz() * blockIdx.z;

        if (device::guard<VelocitySet>(x, y, z))
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        scalar_t phi = static_cast<scalar_t>(0);
        device::constexpr_for<0, velocitySet::Q()>(
            [&](const auto Q)
            {
                phi += d.g[Q * size::cells() + idx3];
            });

        d.phi[idx3] = phi;
    }

    template <typename VelocitySet>
    __global__ void computeNormals(LBMFields d)
    {
        const label_t x = threadIdx.x + block::nx() * blockIdx.x;
        const label_t y = threadIdx.y + block::ny() * blockIdx.y;
        const label_t z = threadIdx.z + block::nz() * blockIdx.z;

        if (device::guard<VelocitySet>(x, y, z))
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        const scalar_t phi_xp1_yp1_z = d.phi[device::global3(x + 1, y + 1, z)];
        const scalar_t phi_xp1_y_zp1 = d.phi[device::global3(x + 1, y, z + 1)];
        const scalar_t phi_xp1_ym1_z = d.phi[device::global3(x + 1, y - 1, z)];
        const scalar_t phi_xp1_y_zm1 = d.phi[device::global3(x + 1, y, z - 1)];
        const scalar_t phi_xm1_ym1_z = d.phi[device::global3(x - 1, y - 1, z)];
        const scalar_t phi_xm1_y_zm1 = d.phi[device::global3(x - 1, y, z - 1)];
        const scalar_t phi_xm1_yp1_z = d.phi[device::global3(x - 1, y + 1, z)];
        const scalar_t phi_xm1_y_zp1 = d.phi[device::global3(x - 1, y, z + 1)];
        const scalar_t phi_x_yp1_zp1 = d.phi[device::global3(x, y + 1, z + 1)];
        const scalar_t phi_x_yp1_zm1 = d.phi[device::global3(x, y + 1, z - 1)];
        const scalar_t phi_x_ym1_zm1 = d.phi[device::global3(x, y - 1, z - 1)];
        const scalar_t phi_x_ym1_zp1 = d.phi[device::global3(x, y - 1, z + 1)];

        scalar_t sgx = VelocitySet::w_1() * (d.phi[device::global3(x + 1, y, z)] - d.phi[device::global3(x - 1, y, z)]) +
                       VelocitySet::w_2() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                             phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                             phi_xp1_ym1_z - phi_xm1_yp1_z +
                                             phi_xp1_y_zm1 - phi_xm1_y_zp1);

        scalar_t sgy = VelocitySet::w_1() * (d.phi[device::global3(x, y + 1, z)] - d.phi[device::global3(x, y - 1, z)]) +
                       VelocitySet::w_2() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                             phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                             phi_xm1_yp1_z - phi_xp1_ym1_z +
                                             phi_x_yp1_zm1 - phi_x_ym1_zp1);

        scalar_t sgz = VelocitySet::w_1() * (d.phi[device::global3(x, y, z + 1)] - d.phi[device::global3(x, y, z - 1)]) +
                       VelocitySet::w_2() * (phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                             phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                             phi_xm1_y_zp1 - phi_xp1_y_zm1 +
                                             phi_x_ym1_zp1 - phi_x_yp1_zm1);

        if constexpr (VelocitySet::Q() == 27)
        {
            const scalar_t phi_xp1_yp1_zp1 = d.phi[device::global3(x + 1, y + 1, z + 1)];
            const scalar_t phi_xp1_yp1_zm1 = d.phi[device::global3(x + 1, y + 1, z - 1)];
            const scalar_t phi_xp1_ym1_zp1 = d.phi[device::global3(x + 1, y - 1, z + 1)];
            const scalar_t phi_xp1_ym1_zm1 = d.phi[device::global3(x + 1, y - 1, z - 1)];
            const scalar_t phi_xm1_ym1_zm1 = d.phi[device::global3(x - 1, y - 1, z - 1)];
            const scalar_t phi_xm1_ym1_zp1 = d.phi[device::global3(x - 1, y - 1, z + 1)];
            const scalar_t phi_xm1_yp1_zm1 = d.phi[device::global3(x - 1, y + 1, z - 1)];
            const scalar_t phi_xm1_yp1_zp1 = d.phi[device::global3(x - 1, y + 1, z + 1)];

            sgx += VelocitySet::w_3() * (phi_xp1_yp1_zp1 - phi_xm1_ym1_zm1 +
                                         phi_xp1_yp1_zm1 - phi_xm1_ym1_zp1 +
                                         phi_xp1_ym1_zp1 - phi_xm1_yp1_zm1 +
                                         phi_xp1_ym1_zm1 - phi_xm1_yp1_zp1);

            sgy += VelocitySet::w_3() * (phi_xp1_yp1_zp1 - phi_xm1_ym1_zm1 +
                                         phi_xp1_yp1_zm1 - phi_xm1_ym1_zp1 +
                                         phi_xm1_yp1_zm1 - phi_xp1_ym1_zp1 +
                                         phi_xm1_yp1_zp1 - phi_xp1_ym1_zm1);

            sgz += VelocitySet::w_3() * (phi_xp1_yp1_zp1 - phi_xm1_ym1_zm1 +
                                         phi_xm1_ym1_zp1 - phi_xp1_yp1_zm1 +
                                         phi_xp1_ym1_zp1 - phi_xm1_yp1_zm1 +
                                         phi_xm1_yp1_zp1 - phi_xp1_ym1_zm1);
        }

        const scalar_t gx = VelocitySet::as2() * sgx;
        const scalar_t gy = VelocitySet::as2() * sgy;
        const scalar_t gz = VelocitySet::as2() * sgz;

        const scalar_t ind = math::sqrt(gx * gx + gy * gy + gz * gz);
        const scalar_t invInd = static_cast<scalar_t>(1) / (ind + static_cast<scalar_t>(1e-9));

        const scalar_t normX = gx * invInd;
        const scalar_t normY = gy * invInd;
        const scalar_t normZ = gz * invInd;

        d.ind[idx3] = ind;
        d.normx[idx3] = normX;
        d.normy[idx3] = normY;
        d.normz[idx3] = normZ;
    }

    template <typename VelocitySet>
    __global__ void computeForces(LBMFields d)
    {
        const label_t x = threadIdx.x + block::nx() * blockIdx.x;
        const label_t y = threadIdx.y + block::ny() * blockIdx.y;
        const label_t z = threadIdx.z + block::nz() * blockIdx.z;

        if (device::guard<VelocitySet>(x, y, z))
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        const label_t xp1_yp1_z = device::global3(x + 1, y + 1, z);
        const label_t xp1_y_zp1 = device::global3(x + 1, y, z + 1);
        const label_t xp1_ym1_z = device::global3(x + 1, y - 1, z);
        const label_t xp1_y_zm1 = device::global3(x + 1, y, z - 1);
        const label_t xm1_ym1_z = device::global3(x - 1, y - 1, z);
        const label_t xm1_y_zm1 = device::global3(x - 1, y, z - 1);
        const label_t xm1_yp1_z = device::global3(x - 1, y + 1, z);
        const label_t xm1_y_zp1 = device::global3(x - 1, y, z + 1);
        const label_t x_yp1_zp1 = device::global3(x, y + 1, z + 1);
        const label_t x_yp1_zm1 = device::global3(x, y + 1, z - 1);
        const label_t x_ym1_zm1 = device::global3(x, y - 1, z - 1);
        const label_t x_ym1_zp1 = device::global3(x, y - 1, z + 1);

        scalar_t scx = VelocitySet::w_1() * (d.normx[device::global3(x + 1, y, z)] - d.normx[device::global3(x - 1, y, z)]) +
                       VelocitySet::w_2() * (d.normx[xp1_yp1_z] - d.normx[xm1_ym1_z] +
                                             d.normx[xp1_y_zp1] - d.normx[xm1_y_zm1] +
                                             d.normx[xp1_ym1_z] - d.normx[xm1_yp1_z] +
                                             d.normx[xp1_y_zm1] - d.normx[xm1_y_zp1]);

        scalar_t scy = VelocitySet::w_1() * (d.normy[device::global3(x, y + 1, z)] - d.normy[device::global3(x, y - 1, z)]) +
                       VelocitySet::w_2() * (d.normy[xp1_yp1_z] - d.normy[xm1_ym1_z] +
                                             d.normy[x_yp1_zp1] - d.normy[x_ym1_zm1] +
                                             d.normy[xm1_yp1_z] - d.normy[xp1_ym1_z] +
                                             d.normy[x_yp1_zm1] - d.normy[x_ym1_zp1]);

        scalar_t scz = VelocitySet::w_1() * (d.normz[device::global3(x, y, z + 1)] - d.normz[device::global3(x, y, z - 1)]) +
                       VelocitySet::w_2() * (d.normz[xp1_y_zp1] - d.normz[xm1_y_zm1] +
                                             d.normz[x_yp1_zp1] - d.normz[x_ym1_zm1] +
                                             d.normz[xm1_y_zp1] - d.normz[xp1_y_zm1] +
                                             d.normz[x_ym1_zp1] - d.normz[x_yp1_zm1]);

        if constexpr (VelocitySet::Q() == 27)
        {
            const label_t xp1_yp1_zp1 = device::global3(x + 1, y + 1, z + 1);
            const label_t xp1_yp1_zm1 = device::global3(x + 1, y + 1, z - 1);
            const label_t xp1_ym1_zp1 = device::global3(x + 1, y - 1, z + 1);
            const label_t xp1_ym1_zm1 = device::global3(x + 1, y - 1, z - 1);
            const label_t xm1_ym1_zm1 = device::global3(x - 1, y - 1, z - 1);
            const label_t xm1_ym1_zp1 = device::global3(x - 1, y - 1, z + 1);
            const label_t xm1_yp1_zm1 = device::global3(x - 1, y + 1, z - 1);
            const label_t xm1_yp1_zp1 = device::global3(x - 1, y + 1, z + 1);

            scx += VelocitySet::w_3() * (d.normx[xp1_yp1_zp1] - d.normx[xm1_ym1_zm1] +
                                         d.normx[xp1_yp1_zm1] - d.normx[xm1_ym1_zp1] +
                                         d.normx[xp1_ym1_zp1] - d.normx[xm1_yp1_zm1] +
                                         d.normx[xp1_ym1_zm1] - d.normx[xm1_yp1_zp1]);

            scy += VelocitySet::w_3() * (d.normy[xp1_yp1_zp1] - d.normy[xm1_ym1_zm1] +
                                         d.normy[xp1_yp1_zm1] - d.normy[xm1_ym1_zp1] +
                                         d.normy[xm1_yp1_zm1] - d.normy[xp1_ym1_zp1] +
                                         d.normy[xm1_yp1_zp1] - d.normy[xp1_ym1_zm1]);

            scz += VelocitySet::w_3() * (d.normz[xp1_yp1_zp1] - d.normz[xm1_ym1_zm1] +
                                         d.normz[xm1_ym1_zp1] - d.normz[xp1_yp1_zm1] +
                                         d.normz[xp1_ym1_zp1] - d.normz[xm1_yp1_zm1] +
                                         d.normz[xm1_yp1_zp1] - d.normz[xp1_ym1_zm1]);
        }

        const scalar_t curvature = VelocitySet::as2() * (scx + scy + scz);

        const scalar_t stCurv = -physics::sigma() * curvature * d.ind[idx3];
        d.fsx[idx3] = stCurv * d.normx[idx3];
        d.fsy[idx3] = stCurv * d.normy[idx3];
        d.fsz[idx3] = stCurv * d.normz[idx3];
    }
}

#endif

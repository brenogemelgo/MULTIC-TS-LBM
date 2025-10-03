#pragma once

// #define D_FIELDS

#if defined(D_FIELDS)

__global__ void computeDerived(
    LBMFields lbm,
    DerivedFields dfields)
{
    const idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const idx_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const idx_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ ||
        x == 0 || x == NX - 1 ||
        y == 0 || y == NY - 1 ||
        z == 0 || z == NZ - 1)
        return;

    const idx_t idx3 = global3(x, y, z);

    const float dudx = (lbm.ux[global3(x + 1, y, z)] - lbm.ux[global3(x - 1, y, z)]) * 0.5f;
    const float dudy = (lbm.ux[global3(x, y + 1, z)] - lbm.ux[global3(x, y - 1, z)]) * 0.5f;
    const float dudz = (lbm.ux[global3(x, y, z + 1)] - lbm.ux[global3(x, y, z - 1)]) * 0.5f;

    const float dvdx = (lbm.uy[global3(x + 1, y, z)] - lbm.uy[global3(x - 1, y, z)]) * 0.5f;
    const float dvdy = (lbm.uy[global3(x, y + 1, z)] - lbm.uy[global3(x, y - 1, z)]) * 0.5f;
    const float dvdz = (lbm.uy[global3(x, y, z + 1)] - lbm.uy[global3(x, y, z - 1)]) * 0.5f;

    const float dwdx = (lbm.uz[global3(x + 1, y, z)] - lbm.uz[global3(x - 1, y, z)]) * 0.5f;
    const float dwdy = (lbm.uz[global3(x, y + 1, z)] - lbm.uz[global3(x, y - 1, z)]) * 0.5f;
    const float dwdz = (lbm.uz[global3(x, y, z + 1)] - lbm.uz[global3(x, y, z - 1)]) * 0.5f;

    const float vort_x = dwdy - dvdz;
    const float vort_y = dudz - dwdx;
    const float vort_z = dvdx - dudy;

    const float vorticity_mag = sqrtf(vort_x * vort_x + vort_y * vort_y + vort_z * vort_z);
    dfields.vorticity_mag[idx3] = vorticity_mag;

    const float velocity_mag = sqrtf(lbm.ux[idx3] * lbm.ux[idx3] + lbm.uy[idx3] * lbm.uy[idx3] + lbm.uz[idx3] * lbm.uz[idx3]);
    dfields.velocity_mag[idx3] = velocity_mag;
}

#endif

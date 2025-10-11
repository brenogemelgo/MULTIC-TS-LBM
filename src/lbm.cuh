#pragma once

[[nodiscard]] __global__ __launch_bounds__(128)
void streamCollide(
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

    float pop[FLINKS];

    const float pop[0] = fromPop(d.f[idx3]);
    const float pop[1] = fromPop(d.f[PLANE + idx3]);
    const float pop[2] = fromPop(d.f[2 * PLANE + idx3]);
    const float pop[3] = fromPop(d.f[3 * PLANE + idx3]);
    const float pop[4] = fromPop(d.f[4 * PLANE + idx3]);
    const float pop[5] = fromPop(d.f[5 * PLANE + idx3]);
    const float pop[6] = fromPop(d.f[6 * PLANE + idx3]);
    const float pop[7] = fromPop(d.f[7 * PLANE + idx3]);
    const float pop[8] = fromPop(d.f[8 * PLANE + idx3]);
    const float pop[9] = fromPop(d.f[9 * PLANE + idx3]);
    const float pop[10] = fromPop(d.f[10 * PLANE + idx3]);
    const float pop[11] = fromPop(d.f[11 * PLANE + idx3]);
    const float pop[12] = fromPop(d.f[12 * PLANE + idx3]);
    const float pop[13] = fromPop(d.f[13 * PLANE + idx3]);
    const float pop[14] = fromPop(d.f[14 * PLANE + idx3]);
    const float pop[15] = fromPop(d.f[15 * PLANE + idx3]);
    const float pop[16] = fromPop(d.f[16 * PLANE + idx3]);
    const float pop[17] = fromPop(d.f[17 * PLANE + idx3]);
    const float pop[18] = fromPop(d.f[18 * PLANE + idx3]);
    #if defined(D3Q27)
    const float pop[19] = fromPop(d.f[19 * PLANE + idx3]);
    const float pop[20] = fromPop(d.f[20 * PLANE + idx3]);
    const float pop[21] = fromPop(d.f[21 * PLANE + idx3]);
    const float pop[22] = fromPop(d.f[22 * PLANE + idx3]);
    const float pop[23] = fromPop(d.f[23 * PLANE + idx3]);
    const float pop[24] = fromPop(d.f[24 * PLANE + idx3]);
    const float pop[25] = fromPop(d.f[25 * PLANE + idx3]);
    const float pop[26] = fromPop(d.f[26 * PLANE + idx3]);
    #endif

    float rho = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    #if defined(D3Q27)
    rho += pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
    #endif
    rho += 1.0f;
    d.rho[idx3] = rho;

    const float ffx = d.ffx[idx3];
    const float ffy = d.ffy[idx3];
    const float ffz = d.ffz[idx3];

    const float invRho = 1.0f / rho;

    #if defined(D3Q19)
    float ux = invRho * (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]);
    float uy = invRho * (pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]);
    float uz = invRho * (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]);
    #elif defined(D3Q27)
    float ux = invRho * (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16] + pop[19] - pop[20] + pop[21] - pop[22] + pop[23] - pop[24] + pop[26] - pop[25]);
    float uy = invRho * (pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18] + pop[19] - pop[20] + pop[21] - pop[22] + pop[24] - pop[23] + pop[25] - pop[26]);
    float uz = invRho * (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17] + pop[19] - pop[20] + pop[22] - pop[21] + pop[23] - pop[24] + pop[25] - pop[26]);
    #endif

    ux += ffx * 0.5f * invRho;
    uy += ffy * 0.5f * invRho;
    uz += ffz * 0.5f * invRho;

    d.ux[idx3] = ux;
    d.uy[idx3] = uy;
    d.uz[idx3] = uz;

    float pxx = 0.0f, pyy = 0.0f, pzz = 0.0f;
    float pxy = 0.0f, pxz = 0.0f, pyz = 0.0f;
    
    #if defined(D3Q19)
    #include "../include/momentumFlux19.cuh"
    #elif defined(D3Q27)
    #include "../include/momentumFlux27.cuh"
    #endif

    d.pxx[idx3] = pxx;
    d.pyy[idx3] = pyy;
    d.pzz[idx3] = pzz;
    d.pxy[idx3] = pxy;
    d.pxz[idx3] = pxz;
    d.pyz[idx3] = pyz;

    float omcoLocal;
    #if defined(VISC_CONTRAST)
    {
        const float phi = d.phi[idx3];
        const float nuLocal = fmaf(phi, (VISC_OIL - VISC_WATER), VISC_WATER);
        const float omegaPhys = 1.0f / (0.5f + 3.0f * nuLocal);

        #if defined(JET)
        omcoLocal = 1.0f - fminf(omegaPhys, cubicSponge(z));
        #elif defined(DROPLET)
        omcoLocal = 1.0f - omegaPhys;
        #endif
    }
    #else
    {
        #if defined(JET)
        omcoLocal = 1.0f - cubicSponge(z);
        #elif defined(DROPLET)
        omcoLocal = 1.0f - OMEGA_REF;
        #endif
    }
    #endif

    #if defined(D3Q19)
    #include "../include/streamCollide19.cuh"
    #elif defined(D3Q27)
    #include "../include/streamCollide27.cuh"
    #endif

    { // ====================================== ADVECTION-DIFFUSION ====================================== //
        const float phi = d.phi[idx3];

        // Q0
        d.g[idx3] = WG_0 * phi;

        const float multPhi = WG_1 * phi;
        const float phiNorm = GAMMA * multPhi * (1.0f - phi);
        const float a4 = 4.0f * multPhi;

        // -------------------------------- X+ (Q1)
        float geq = multPhi + a4 * ux;
        float hi = phiNorm * d.normx[idx3];
        d.g[global4(x+1,y,z,1)] = geq + hi;

        // -------------------------------- X- (Q2)
        geq = multPhi - a4 * ux;
        d.g[global4(x-1,y,z,2)] = geq - hi;

        // -------------------------------- Y+ (Q3)
        geq = multPhi + a4 * uy;
        hi = phiNorm * d.normy[idx3];
        d.g[global4(x,y+1,z,3)] = geq + hi;

        // -------------------------------- Y- (Q4)
        geq = multPhi - a4 * uy;
        d.g[global4(x,y-1,z,4)] = geq - hi;

        // -------------------------------- Z+ (Q5)
        geq = multPhi + a4 * uz;
        hi = phiNorm * d.normz[idx3];
        d.g[global4(x,y,z+1,5)] = geq + hi;

        // -------------------------------- Z- (Q6)
        geq = multPhi - a4 * uz;
        d.g[global4(x,y,z-1,6)] = geq - hi;
    } // ============================================= END ============================================= //
}

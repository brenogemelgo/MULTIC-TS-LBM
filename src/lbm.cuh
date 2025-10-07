#pragma once

__global__ __launch_bounds__(128)
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

    const float pop0 = fromPop(d.f[idx3]);
    const float pop1 = fromPop(d.f[PLANE + idx3]);
    const float pop2 = fromPop(d.f[2 * PLANE + idx3]);
    const float pop3 = fromPop(d.f[3 * PLANE + idx3]);
    const float pop4 = fromPop(d.f[4 * PLANE + idx3]);
    const float pop5 = fromPop(d.f[5 * PLANE + idx3]);
    const float pop6 = fromPop(d.f[6 * PLANE + idx3]);
    const float pop7 = fromPop(d.f[7 * PLANE + idx3]);
    const float pop8 = fromPop(d.f[8 * PLANE + idx3]);
    const float pop9 = fromPop(d.f[9 * PLANE + idx3]);
    const float pop10 = fromPop(d.f[10 * PLANE + idx3]);
    const float pop11 = fromPop(d.f[11 * PLANE + idx3]);
    const float pop12 = fromPop(d.f[12 * PLANE + idx3]);
    const float pop13 = fromPop(d.f[13 * PLANE + idx3]);
    const float pop14 = fromPop(d.f[14 * PLANE + idx3]);
    const float pop15 = fromPop(d.f[15 * PLANE + idx3]);
    const float pop16 = fromPop(d.f[16 * PLANE + idx3]);
    const float pop17 = fromPop(d.f[17 * PLANE + idx3]);
    const float pop18 = fromPop(d.f[18 * PLANE + idx3]);
    #if defined(D3Q27)
    const float pop19 = fromPop(d.f[19 * PLANE + idx3]);
    const float pop20 = fromPop(d.f[20 * PLANE + idx3]);
    const float pop21 = fromPop(d.f[21 * PLANE + idx3]);
    const float pop22 = fromPop(d.f[22 * PLANE + idx3]);
    const float pop23 = fromPop(d.f[23 * PLANE + idx3]);
    const float pop24 = fromPop(d.f[24 * PLANE + idx3]);
    const float pop25 = fromPop(d.f[25 * PLANE + idx3]);
    const float pop26 = fromPop(d.f[26 * PLANE + idx3]);
    #endif

    float rho = pop0 + pop1 + pop2 + pop3 + pop4 + pop5 + pop6 + pop7 + pop8 + pop9 + pop10 + pop11 + pop12 + pop13 + pop14 + pop15 + pop16 + pop17 + pop18;
    #if defined(D3Q27)
    rho += pop19 + pop20 + pop21 + pop22 + pop23 + pop24 + pop25 + pop26;
    #endif
    rho += 1.0f;
    d.rho[idx3] = rho;

    const float ffx = d.ffx[idx3];
    const float ffy = d.ffy[idx3];
    const float ffz = d.ffz[idx3];

    const float invRho = 1.0f / rho;

    #if defined(D3Q19)
    float ux = invRho * (pop1 - pop2 + pop7 - pop8 + pop9 - pop10 + pop13 - pop14 + pop15 - pop16);
    float uy = invRho * (pop3 - pop4 + pop7 - pop8 + pop11 - pop12 + pop14 - pop13 + pop17 - pop18);
    float uz = invRho * (pop5 - pop6 + pop9 - pop10 + pop11 - pop12 + pop16 - pop15 + pop18 - pop17);
    #elif defined(D3Q27)
    float ux = invRho * (pop1 - pop2 + pop7 - pop8 + pop9 - pop10 + pop13 - pop14 + pop15 - pop16 + pop19 - pop20 + pop21 - pop22 + pop23 - pop24 + pop26 - pop25);
    float uy = invRho * (pop3 - pop4 + pop7 - pop8 + pop11 - pop12 + pop14 - pop13 + pop17 - pop18 + pop19 - pop20 + pop21 - pop22 + pop24 - pop23 + pop25 - pop26);
    float uz = invRho * (pop5 - pop6 + pop9 - pop10 + pop11 - pop12 + pop16 - pop15 + pop18 - pop17 + pop19 - pop20 + pop22 - pop21 + pop23 - pop24 + pop25 - pop26);
    #endif

    ux += ffx * 0.5f * invRho;
    uy += ffy * 0.5f * invRho;
    uz += ffz * 0.5f * invRho;

    d.ux[idx3] = ux;
    d.uy[idx3] = uy;
    d.uz[idx3] = uz;

    float pxx = 0.0f, pyy = 0.0f, pzz = 0.0f, pxy = 0.0f, pxz = 0.0f, pyz = 0.0f;
    #if defined(D3Q19)
    #include "../include/momentumFlux19.cuh"
    #elif defined(D3Q27)
    #include "../include/momentumFlux27.cuh"
    #endif

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
    // #include "../include/functionalStreamCollide.cuh"
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

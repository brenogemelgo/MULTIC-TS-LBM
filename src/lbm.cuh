#pragma once

__global__ __launch_bounds__(block::nx * block::ny * block::nz)
void streamCollide(
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

    float pop[FLINKS];
    pop[0] = from_pop(d.f[idx3]);
    pop[1] = from_pop(d.f[PLANE + idx3]);
    pop[2] = from_pop(d.f[2 * PLANE + idx3]);
    pop[3] = from_pop(d.f[3 * PLANE + idx3]);
    pop[4] = from_pop(d.f[4 * PLANE + idx3]);
    pop[5] = from_pop(d.f[5 * PLANE + idx3]);
    pop[6] = from_pop(d.f[6 * PLANE + idx3]);
    pop[7] = from_pop(d.f[7 * PLANE + idx3]);
    pop[8] = from_pop(d.f[8 * PLANE + idx3]);
    pop[9] = from_pop(d.f[9 * PLANE + idx3]);
    pop[10] = from_pop(d.f[10 * PLANE + idx3]);
    pop[11] = from_pop(d.f[11 * PLANE + idx3]);
    pop[12] = from_pop(d.f[12 * PLANE + idx3]);
    pop[13] = from_pop(d.f[13 * PLANE + idx3]);
    pop[14] = from_pop(d.f[14 * PLANE + idx3]);
    pop[15] = from_pop(d.f[15 * PLANE + idx3]);
    pop[16] = from_pop(d.f[16 * PLANE + idx3]);
    pop[17] = from_pop(d.f[17 * PLANE + idx3]);
    pop[18] = from_pop(d.f[18 * PLANE + idx3]);

    #if defined(D3Q27)

        pop[19] = from_pop(d.f[19 * PLANE + idx3]);
        pop[20] = from_pop(d.f[20 * PLANE + idx3]);
        pop[21] = from_pop(d.f[21 * PLANE + idx3]);
        pop[22] = from_pop(d.f[22 * PLANE + idx3]);
        pop[23] = from_pop(d.f[23 * PLANE + idx3]);
        pop[24] = from_pop(d.f[24 * PLANE + idx3]);
        pop[25] = from_pop(d.f[25 * PLANE + idx3]);
        pop[26] = from_pop(d.f[26 * PLANE + idx3]);

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

    #include "../include/CTmomentumFlux.cuh"
    
    d.pxx[idx3] = pxx;
    d.pyy[idx3] = pyy;
    d.pzz[idx3] = pzz;
    d.pxy[idx3] = pxy;
    d.pxz[idx3] = pxz;
    d.pyz[idx3] = pyz;

    #include "../include/relaxationFrequency.cuh"

    #include "../include/CTstreamCollide.cuh"

    { // ====================================== ADVECTION-DIFFUSION ====================================== //
        const float phi = d.phi[idx3];

        // Q0
        d.g[idx3] = WG_0 * phi;

        // -------------------------------- X+ (Q1)
        float geq = WG_1 * phi * (1.0f + 4.0f * ux);
        float hi = WG_1 * physics::gamma * phi * (1.0f - phi) * d.normx[idx3];
        d.g[global4(x + 1, y, z, 1)] = geq + hi;

        // -------------------------------- X- (Q2)
        geq = WG_1 * phi * (1.0f - 4.0f * ux);
        d.g[global4(x - 1, y, z, 2)] = geq - hi;

        // -------------------------------- Y+ (Q3)
        geq = WG_1 * phi * (1.0f + 4.0f * uy);
        hi = WG_1 * physics::gamma * phi * (1.0f - phi) * d.normy[idx3];
        d.g[global4(x, y + 1, z, 3)] = geq + hi;

        // -------------------------------- Y- (Q4)
        geq = WG_1 * phi * (1.0f - 4.0f * uy);
        d.g[global4(x, y - 1, z, 4)] = geq - hi;

        // -------------------------------- Z+ (Q5)
        geq = WG_1 * phi * (1.0f + 4.0f * uz);
        hi = WG_1 * physics::gamma * phi * (1.0f - phi) * d.normz[idx3];
        d.g[global4(x, y, z + 1, 5)] = geq + hi;

        // -------------------------------- Z- (Q6)
        geq = WG_1 * phi * (1.0f - 4.0f * uz);
        d.g[global4(x, y, z - 1, 6)] = geq - hi;
    } // ============================================= END ============================================= //
}


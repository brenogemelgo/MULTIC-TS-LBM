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

    const float pop0 = from_pop(d.f[idx3]);         
    const float pop1 = from_pop(d.f[PLANE + idx3]);   
    const float pop2 = from_pop(d.f[2 * PLANE + idx3]);  
    const float pop3 = from_pop(d.f[3 * PLANE + idx3]);  
    const float pop4 = from_pop(d.f[4 * PLANE + idx3]);  
    const float pop5 = from_pop(d.f[5 * PLANE + idx3]);  
    const float pop6 = from_pop(d.f[6 * PLANE + idx3]);  
    const float pop7 = from_pop(d.f[7 * PLANE + idx3]);  
    const float pop8 = from_pop(d.f[8 * PLANE + idx3]);  
    const float pop9 = from_pop(d.f[9 * PLANE + idx3]);  
    const float pop10 = from_pop(d.f[10 * PLANE + idx3]);
    const float pop11 = from_pop(d.f[11 * PLANE + idx3]);
    const float pop12 = from_pop(d.f[12 * PLANE + idx3]); 
    const float pop13 = from_pop(d.f[13 * PLANE + idx3]); 
    const float pop14 = from_pop(d.f[14 * PLANE + idx3]);
    const float pop15 = from_pop(d.f[15 * PLANE + idx3]); 
    const float pop16 = from_pop(d.f[16 * PLANE + idx3]); 
    const float pop17 = from_pop(d.f[17 * PLANE + idx3]);
    const float pop18 = from_pop(d.f[18 * PLANE + idx3]); 
    #if defined(D3Q27)
    const float pop19 = from_pop(d.f[19 * PLANE + idx3]); 
    const float pop20 = from_pop(d.f[20 * PLANE + idx3]); 
    const float pop21 = from_pop(d.f[21 * PLANE + idx3]); 
    const float pop22 = from_pop(d.f[22 * PLANE + idx3]); 
    const float pop23 = from_pop(d.f[23 * PLANE + idx3]); 
    const float pop24 = from_pop(d.f[24 * PLANE + idx3]);
    const float pop25 = from_pop(d.f[25 * PLANE + idx3]);
    const float pop26 = from_pop(d.f[26 * PLANE + idx3]); 
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
    float uy = invRho * (pop3 - pop4 + pop7 - pop8  + pop11 - pop12 + pop14 - pop13 + pop17 - pop18 + pop19 - pop20 + pop21 - pop22 + pop24 - pop23 + pop25 - pop26);
    float uz = invRho * (pop5 - pop6 + pop9 - pop10 + pop11 - pop12 + pop16 - pop15 + pop18 - pop17 + pop19 - pop20 + pop22 - pop21 + pop23 - pop24 + pop25 - pop26);
    #endif
    
    ux += ffx * 0.5f * invRho;
    uy += ffy * 0.5f * invRho;
    uz += ffz * 0.5f * invRho;

    d.ux[idx3] = ux; 
    d.uy[idx3] = uy; 
    d.uz[idx3] = uz;

    #include "../include/momentumFluxAU.cuh" // Agressive Unrolling
    //#include "../include/momentumFluxNT.cuh" // No Temps

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

    #include "../include/streamCollide.cuh" 

    { // ====================================== ADVECTION-DIFFUSION ====================================== //
        #if !defined(VISC_CONTRAST)
        const float phi = d.phi[idx3];
        #endif
        d.g[idx3] = W_G_1 * phi;

        const float phiNorm = W_G_2 * GAMMA * phi * (1.0f - phi);
        const float multPhi = W_G_2 * phi;
        const float a3 = 3.0f * multPhi;

        float feq = multPhi + a3 * ux;
        float force = phiNorm * d.normx[idx3];
        d.g[global4(x+1,y,z,1)] = feq + force;
        
        feq = multPhi - a3 * ux;
        d.g[global4(x-1,y,z,2)] = feq - force;

        feq = multPhi + a3 * uy;
        force = phiNorm * d.normy[idx3];
        d.g[global4(x,y+1,z,3)] = feq + force;

        feq = multPhi - a3 * uy;
        d.g[global4(x,y-1,z,4)] = feq - force;

        feq = multPhi + a3 * uz;
        force = phiNorm * d.normz[idx3];
        d.g[global4(x,y,z+1,5)] = feq + force;

        feq = multPhi - a3 * uz;
        d.g[global4(x,y,z-1,6)] = feq - force;
    } // ============================================= END ============================================= //
}

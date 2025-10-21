const float uu = 1.5f * (ux*ux + uy*uy + uz*uz);
constexpr_for<0, FLINKS>([&] __device__ (auto I) {
    constexpr idx_t Q = decltype(I)::value; 

    constexpr float w  = FDir<Q>::w;
    constexpr float cx = static_cast<float>(FDir<Q>::cx); 
    constexpr float cy = static_cast<float>(FDir<Q>::cy); 
    constexpr float cz = static_cast<float>(FDir<Q>::cz); 

    const float cu = 3.0f * (ux*cx + uy*cy + uz*cz);

    #if defined(D3Q19)
        const float feq = w * rho * (1.0f - uu + cu + 0.5f*cu*cu) - w;
    #elif defined(D3Q27)
        const float feq = w * rho * (1.0f - uu + cu + 0.5f*cu*cu + OOS*cu*cu*cu - uu*cu) - w;
    #endif

    #if defined(D3Q19)
        const float force = 0.5f * feq * 
            ((cx - ux) * ffx + 
             (cy - uy) * ffy + 
             (cz - uz) * ffz) * 3.0f * invRho;
    #elif defined(D3Q27)
        const float force = 0.5f * w * 
            ((3.0f * (cx - ux) + 3.0f * cu * cx) * ffx +
             (3.0f * (cy - uy) + 3.0f * cu * cy) * ffy +
             (3.0f * (cz - uz) + 3.0f * cu * cz) * ffz);
    #endif

    const float fneq = pop[Q] - (feq - force);  

    pxx += fneq * cx * cx;
    pyy += fneq * cy * cy;
    pzz += fneq * cz * cz;
    pxy += fneq * cx * cy;
    pxz += fneq * cx * cz;
    pyz += fneq * cy * cz;
});


const float uu = 1.5f * (ux * ux + uy * uy + uz * uz);
constexpr_for<0, FLINKS>([&] __device__ (auto I) {
    constexpr idx_t Q = decltype(I)::value; 

    const float cu = 3.0f * 
       (ux * static_cast<float>(FDir<Q>::cx) + 
        uy * static_cast<float>(FDir<Q>::cy) + 
        uz * static_cast<float>(FDir<Q>::cz));

    #if defined(D3Q19)
        const float feq = FDir<Q>::w * rho * (1.0f - uu + cu + 0.5f * cu * cu) - FDir<Q>::w;
    #elif defined(D3Q27)
        const float feq = FDir<Q>::w * rho * (1.0f - uu + cu + 0.5f * cu * cu + OOS * cu * cu * cu - uu * cu) - FDir<Q>::w;
    #endif

    #if defined(D3Q19)
        const float force = 1.5f * invRho * feq * 
            ((static_cast<float>(FDir<Q>::cx) - ux) * ffx + 
             (static_cast<float>(FDir<Q>::cy) - uy) * ffy + 
             (static_cast<float>(FDir<Q>::cz) - uz) * ffz);
    #elif defined(D3Q27)
        const float force = 0.5f * FDir<Q>::w * 
            ((3.0f * (static_cast<float>(FDir<Q>::cx) - ux) + 3.0f * cu * static_cast<float>(FDir<Q>::cx)) * ffx +
             (3.0f * (static_cast<float>(FDir<Q>::cy) - uy) + 3.0f * cu * static_cast<float>(FDir<Q>::cy)) * ffy +
             (3.0f * (static_cast<float>(FDir<Q>::cz) - uz) + 3.0f * cu * static_cast<float>(FDir<Q>::cz)) * ffz);
    #endif

    const float fneq = pop[Q] - (feq - force);  

    pxx += fneq * static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cx);
    pyy += fneq * static_cast<float>(FDir<Q>::cy) * static_cast<float>(FDir<Q>::cy);
    pzz += fneq * static_cast<float>(FDir<Q>::cz) * static_cast<float>(FDir<Q>::cz);
    pxy += fneq * static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cy);
    pxz += fneq * static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cz);
    pyz += fneq * static_cast<float>(FDir<Q>::cy) * static_cast<float>(FDir<Q>::cz);
});


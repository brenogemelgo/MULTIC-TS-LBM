constexpr_for<0, FLINKS>([&] __device__ (auto I) {
    constexpr idx_t Q = decltype(I)::value;

    const idx_t xx = x + static_cast<idx_t>(FDir<Q>::cx);
    const idx_t yy = y + static_cast<idx_t>(FDir<Q>::cy);
    const idx_t zz = z + static_cast<idx_t>(FDir<Q>::cz);

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

    #if defined(D3Q19)
        const float fneq = (w * 4.5f) *
            ((cx * cx - CSSQ) * pxx +
             (cy * cy - CSSQ) * pyy +
             (cz * cz - CSSQ) * pzz +
              2.0f * (cx * cy * pxy + 
                      cx * cz * pxz + 
                      cy * cz * pyz));
    #elif defined(D3Q27)
        const float fneq = (w * 4.5f) *
            ((cx * cx - CSSQ) * pxx +
             (cy * cy - CSSQ) * pyy +
             (cz * cz - CSSQ) * pzz +
              2.0f * (cx * cy * pxy + 
                      cx * cz * pxz + 
                      cy * cz * pyz) +
             (cx * cx * cx - 3.0f * CSSQ * cx) * (3.0f * ux * pxx) +
             (cy * cy * cy - 3.0f * CSSQ * cy) * (3.0f * uy * pyy) +
             (cz * cz * cz - 3.0f * CSSQ * cz) * (3.0f * uz * pzz) +
              3.0f * ((cx * cx * cy - CSSQ * cy) * (pxx * uy + 2.0f * ux * pxy) +
                      (cx * cx * cz - CSSQ * cz) * (pxx * uz + 2.0f * ux * pxz) +
                      (cx * cy * cy - CSSQ * cx) * (pxy * uy + 2.0f * ux * pyy) +
                      (cy * cy * cz - CSSQ * cz) * (pyy * uz + 2.0f * uy * pyz) +
                      (cx * cz * cz - CSSQ * cx) * (pxz * uz + 2.0f * ux * pzz) +
                      (cy * cz * cz - CSSQ * cy) * (pyz * uz + 2.0f * uy * pzz)) +
                    6.0f * (cx * cy * cz) * (ux * pyz + uy * pxz + uz * pxy));
    #endif

    d.f[global4(xx, yy, zz, Q)] = to_pop(feq + omcoLocal * fneq + force);
});

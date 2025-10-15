constexpr_for<0, FLINKS>([&] __device__ (auto I) {
    constexpr idx_t Q = decltype(I)::value;

    const idx_t xx = x + static_cast<idx_t>(FDir<Q>::cx);
    const idx_t yy = y + static_cast<idx_t>(FDir<Q>::cy);
    const idx_t zz = z + static_cast<idx_t>(FDir<Q>::cz);

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

    #if defined(D3Q19)
        const float fneq = (FDir<Q>::w * 4.5f) *
            ((static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cx) - CSSQ) * pxx +
             (static_cast<float>(FDir<Q>::cy) * static_cast<float>(FDir<Q>::cy) - CSSQ) * pyy +
             (static_cast<float>(FDir<Q>::cz) * static_cast<float>(FDir<Q>::cz) - CSSQ) * pzz +
              2.0f * (static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cy) * pxy + 
                      static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cz) * pxz + 
                      static_cast<float>(FDir<Q>::cy) * static_cast<float>(FDir<Q>::cz) * pyz));
    #elif defined(D3Q27)
        const float fneq = (FDir<Q>::w * 4.5f) *
            ((static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cx) - CSSQ) * pxx +
             (static_cast<float>(FDir<Q>::cy) * static_cast<float>(FDir<Q>::cy) - CSSQ) * pyy +
             (static_cast<float>(FDir<Q>::cz) * static_cast<float>(FDir<Q>::cz) - CSSQ) * pzz +
              2.0f * (static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cy) * pxy + 
                      static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cz) * pxz + 
                      static_cast<float>(FDir<Q>::cy) * static_cast<float>(FDir<Q>::cz) * pyz) +
             (static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cx) - 3.0f * CSSQ * static_cast<float>(FDir<Q>::cx)) * (3.0f * ux * pxx) +
             (static_cast<float>(FDir<Q>::cy) * static_cast<float>(FDir<Q>::cy) * static_cast<float>(FDir<Q>::cy) - 3.0f * CSSQ * static_cast<float>(FDir<Q>::cy)) * (3.0f * uy * pyy) +
             (static_cast<float>(FDir<Q>::cz) * static_cast<float>(FDir<Q>::cz) * static_cast<float>(FDir<Q>::cz) - 3.0f * CSSQ * static_cast<float>(FDir<Q>::cz)) * (3.0f * uz * pzz) +
              3.0f * ((static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cy) - CSSQ * static_cast<float>(FDir<Q>::cy)) * (pxx * uy + 2.0f * ux * pxy) +
                      (static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cz) - CSSQ * static_cast<float>(FDir<Q>::cz)) * (pxx * uz + 2.0f * ux * pxz) +
                      (static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cy) * static_cast<float>(FDir<Q>::cy) - CSSQ * static_cast<float>(FDir<Q>::cx)) * (pxy * uy + 2.0f * ux * pyy) +
                      (static_cast<float>(FDir<Q>::cy) * static_cast<float>(FDir<Q>::cy) * static_cast<float>(FDir<Q>::cz) - CSSQ * static_cast<float>(FDir<Q>::cz)) * (pyy * uz + 2.0f * uy * pyz) +
                      (static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cz) * static_cast<float>(FDir<Q>::cz) - CSSQ * static_cast<float>(FDir<Q>::cx)) * (pxz * uz + 2.0f * ux * pzz) +
                      (static_cast<float>(FDir<Q>::cy) * static_cast<float>(FDir<Q>::cz) * static_cast<float>(FDir<Q>::cz) - CSSQ * static_cast<float>(FDir<Q>::cy)) * (pyz * uz + 2.0f * uy * pzz)) +
                    6.0f * (static_cast<float>(FDir<Q>::cx) * static_cast<float>(FDir<Q>::cy) * static_cast<float>(FDir<Q>::cz)) * (pxy * uz + ux * pyz + uy * pxz));
    #endif

    const idx_t streamIdx = global4(xx, yy, zz, Q);
    d.f[streamIdx] = toPop(feq + omcoLocal * fneq + force);
});

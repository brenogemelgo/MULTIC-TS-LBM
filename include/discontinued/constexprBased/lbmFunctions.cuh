template<idx_t Q> struct FDir;

template<> struct FDir<0>  { static constexpr int cx= 0, cy= 0, cz= 0; static constexpr float w=W_0; };
template<> struct FDir<1>  { static constexpr int cx= 1, cy= 0, cz= 0; static constexpr float w=W_1; };
template<> struct FDir<2>  { static constexpr int cx=-1, cy= 0, cz= 0; static constexpr float w=W_1; };
template<> struct FDir<3>  { static constexpr int cx= 0, cy= 1, cz= 0; static constexpr float w=W_1; };
template<> struct FDir<4>  { static constexpr int cx= 0, cy=-1, cz= 0; static constexpr float w=W_1; };
template<> struct FDir<5>  { static constexpr int cx= 0, cy= 0, cz= 1; static constexpr float w=W_1; };
template<> struct FDir<6>  { static constexpr int cx= 0, cy= 0, cz=-1; static constexpr float w=W_1; };
template<> struct FDir<7>  { static constexpr int cx= 1, cy= 1, cz= 0; static constexpr float w=W_2; };
template<> struct FDir<8>  { static constexpr int cx=-1, cy=-1, cz= 0; static constexpr float w=W_2; };
template<> struct FDir<9>  { static constexpr int cx= 1, cy= 0, cz= 1; static constexpr float w=W_2; };
template<> struct FDir<10> { static constexpr int cx=-1, cy= 0, cz=-1; static constexpr float w=W_2; };
template<> struct FDir<11> { static constexpr int cx= 0, cy= 1, cz= 1; static constexpr float w=W_2; };
template<> struct FDir<12> { static constexpr int cx= 0, cy=-1, cz=-1; static constexpr float w=W_2; };
template<> struct FDir<13> { static constexpr int cx= 1, cy=-1, cz= 0; static constexpr float w=W_2; };
template<> struct FDir<14> { static constexpr int cx=-1, cy= 1, cz= 0; static constexpr float w=W_2; };
template<> struct FDir<15> { static constexpr int cx= 1, cy= 0, cz=-1; static constexpr float w=W_2; };
template<> struct FDir<16> { static constexpr int cx=-1, cy= 0, cz= 1; static constexpr float w=W_2; };
template<> struct FDir<17> { static constexpr int cx= 0, cy= 1, cz=-1; static constexpr float w=W_2; };
template<> struct FDir<18> { static constexpr int cx= 0, cy=-1, cz= 1; static constexpr float w=W_2; };
#if defined(D3Q27)
template<> struct FDir<19> { static constexpr int cx= 1, cy= 1, cz= 1; static constexpr float w=W_3; };
template<> struct FDir<20> { static constexpr int cx=-1, cy=-1, cz=-1; static constexpr float w=W_3; };
template<> struct FDir<21> { static constexpr int cx= 1, cy= 1, cz=-1; static constexpr float w=W_3; };
template<> struct FDir<22> { static constexpr int cx=-1, cy=-1, cz= 1; static constexpr float w=W_3; };
template<> struct FDir<23> { static constexpr int cx= 1, cy=-1, cz= 1; static constexpr float w=W_3; };
template<> struct FDir<24> { static constexpr int cx=-1, cy= 1, cz=-1; static constexpr float w=W_3; };
template<> struct FDir<25> { static constexpr int cx=-1, cy= 1, cz= 1; static constexpr float w=W_3; };
template<> struct FDir<26> { static constexpr int cx= 1, cy=-1, cz=-1; static constexpr float w=W_3; };
#endif

template<idx_t Q> struct GDir;

template<> struct GDir<0> { static constexpr int cx= 0, cy= 0, cz= 0; static constexpr float wg=W_G_1; };
template<> struct GDir<1> { static constexpr int cx= 1, cy= 0, cz= 0; static constexpr float wg=W_G_2; };
template<> struct GDir<2> { static constexpr int cx=-1, cy= 0, cz= 0; static constexpr float wg=W_G_2; };
template<> struct GDir<3> { static constexpr int cx= 0, cy= 1, cz= 0; static constexpr float wg=W_G_2; };
template<> struct GDir<4> { static constexpr int cx= 0, cy=-1, cz= 0; static constexpr float wg=W_G_2; };
template<> struct GDir<5> { static constexpr int cx= 0, cy= 0, cz= 1; static constexpr float wg=W_G_2; };
template<> struct GDir<6> { static constexpr int cx= 0, cy= 0, cz=-1; static constexpr float wg=W_G_2; };

template<idx_t Q>
__device__ __forceinline__
float computeFeq(
    const float rho, 
    const float ux, 
    const float uy, 
    const float uz,
    const float uu
) {
    const float cu = ux*FDir<Q>::cx + uy*FDir<Q>::cy + uz*FDir<Q>::cz;
    #if defined(D3Q19)
        return FDir<Q>::w * rho * (1.0f - 1.5f*uu + 3.0f*cu + 4.5f*cu*cu) - FDir<Q>::w;
    #elif defined(D3Q27)
        return FDir<Q>::w * rho * (1.0f - 1.5f*uu + 3.0f*cu + 4.5f*cu*cu + 4.5f*cu*cu*cu - 4.5f*uu*cu) - FDir<Q>::w;
    #endif
}

template<idx_t Q>
__device__ __forceinline__ 
float computeGeq(
    const float phi, 
    const float ux, 
    const float uy, 
    const float uz
) {
    const float cu = ux*GDir<Q>::cx + uy*GDir<Q>::cy + uz*GDir<Q>::cz;
    return GDir<Q>::wg * phi * (1.0f + 3.0f * cu);
}

template<idx_t Q>
__device__ __forceinline__
float computeNeq(
    const float pxx, 
    const float pyy, 
    const float pzz,
    const float pxy, 
    const float pxz, 
    const float pyz,
    const float ux,
    const float uy,
    const float uz
) {
    constexpr int cx = FDir<Q>::cx;
    constexpr int cy = FDir<Q>::cy;
    constexpr int cz = FDir<Q>::cz;
    constexpr float w = FDir<Q>::w;

    #if defined(D3Q19)
        return (w * 4.5f) * ((cx*cx - CSSQ) * pxx +
                             (cy*cy - CSSQ) * pyy +
                             (cz*cz - CSSQ) * pzz +
                            2.0f * (cx*cy*pxy + cx*cz*pxz + cy*cz*pyz));
    #elif defined(D3Q27)
        return (w * 4.5f) * 
        ((cx*cx - CSSQ) * pxx +
         (cy*cy - CSSQ) * pyy +
         (cz*cz - CSSQ) * pzz +
        2.0f * (cx*cy*pxy + cx*cz*pxz + cy*cz*pyz) +
        (cx*cx*cx - 3.0f*CSSQ*cx) * (3.0f * ux * pxx) +
        (cy*cy*cy - 3.0f*CSSQ*cy) * (3.0f * uy * pyy) +
        (cz*cz*cz - 3.0f*CSSQ*cz) * (3.0f * uz * pzz) +
        3.0f * ((cx*cx*cy - CSSQ*cy) * (pxx*uy + 2.0f*ux*pxy) +
                (cx*cx*cz - CSSQ*cz) * (pxx*uz + 2.0f*ux*pxz) +
                (cx*cy*cy - CSSQ*cx) * (pxy*uy + 2.0f*ux*pyy) +
                (cy*cy*cz - CSSQ*cx) * (pyy*uz + 2.0f*uy*pyz) +
                (cx*cz*cz - CSSQ*cx) * (pxz*uz + 2.0f*ux*pzz) +
                (cy*cz*cz - CSSQ*cy) * (pyz*uz + 2.0f*uy*pzz)) +
                6.0f * (cx*cy*cz) * (pxy*uz + ux*pyz + uy*pxz));
    #endif
}

template<idx_t Q>
__device__ __forceinline__ 
float computeForce(
    const float coeff, 
    const float feq, 
    const float ux, 
    const float uy, 
    const float uz, 
    const float ffx, 
    const float ffy, 
    const float ffz, 
    const float aux
) {
    constexpr int cx = FDir<Q>::cx;
    constexpr int cy = FDir<Q>::cy;
    constexpr int cz = FDir<Q>::cz;

    #if defined(D3Q19)
        return coeff * feq * ((cx - ux) * ffx +
                              (cy - uy) * ffy +
                              (cz - uz) * ffz) * aux;
    #elif defined(D3Q27)
        constexpr float w = FDir<Q>::w;
        const float cu = 3.0f * (ux*cx + uy*cy + uz*cz);             
        return coeff * w * ((3.0f * (cx - ux) + 3.0f * cu * cx ) * ffx +
                            (3.0f * (cy - uy) + 3.0f * cu * cy ) * ffy +
                            (3.0f * (cz - uz) + 3.0f * cu * cz ) * ffz);
    #endif
}
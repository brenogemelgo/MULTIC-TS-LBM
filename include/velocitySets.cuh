#pragma once
#include "cudaUtils.cuh"

#if defined(D3Q19) //             0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 

    __constant__ ci_t CIX[19] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0 };
    __constant__ ci_t CIY[19] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1 };
    __constant__ ci_t CIZ[19] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1 };
    __constant__ float W[19] = { 1.0f / 3.0f, 
                                 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
                                 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };
    static constexpr float W_0 = 1.0f / 3.0f;  // 0
    static constexpr float W_1 = 1.0f / 18.0f; // 1 to 6
    static constexpr float W_2 = 1.0f / 36.0f; // 7 to 18
    static constexpr idx_t FLINKS = 19;

#elif defined(D3Q27) //           0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26

    __constant__ ci_t CIX[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
    __constant__ ci_t CIY[27] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1 };
    __constant__ ci_t CIZ[27] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1 };
    __constant__ float W[27] = { 8.0f / 27.0f,
                                 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 
                                 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 
                                 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f };
    static constexpr float W_0 = 8.0f / 27.0f;  // 0
    static constexpr float W_1 = 2.0f / 27.0f;  // 1 to 6
    static constexpr float W_2 = 1.0f / 54.0f;  // 7 to 18
    static constexpr float W_3 = 1.0f / 216.0f; // 19 to 26
    static constexpr idx_t FLINKS = 27;
    
#endif

__constant__ float W_G[7] = { 1.0f / 4.0f, 
                              1.0f / 8.0f, 1.0f / 8.0f, 
                              1.0f / 8.0f, 1.0f / 8.0f, 
                              1.0f / 8.0f, 1.0f / 8.0f };
constexpr float W_G_1 = 1.0f / 4.0f; // 0
constexpr float W_G_2 = 1.0f / 8.0f; // 1 to 6
constexpr idx_t GLINKS = 7;   


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
template<> struct FDir<19> { static constexpr int cx= 1, cy= 1, cz= 1; static constexpr float w=W_3 };
template<> struct FDir<20> { static constexpr int cx=-1, cy=-1, cz=-1; static constexpr float w=W_3 };
template<> struct FDir<21> { static constexpr int cx= 1, cy= 1, cz=-1; static constexpr float w=W_3 };
template<> struct FDir<22> { static constexpr int cx=-1, cy=-1, cz= 1; static constexpr float w=W_3 };
template<> struct FDir<23> { static constexpr int cx= 1, cy=-1, cz= 1; static constexpr float w=W_3 };
template<> struct FDir<24> { static constexpr int cx=-1, cy= 1, cz=-1; static constexpr float w=W_3 };
template<> struct FDir<25> { static constexpr int cx=-1, cy= 1, cz= 1; static constexpr float w=W_3 };
template<> struct FDir<26> { static constexpr int cx= 1, cy=-1, cz=-1; static constexpr float w=W_3 };
#endif

template<idx_t Q> struct GDir;

template<> struct GDir<0> { static constexpr int cx= 0, cy= 0, cz= 0; static constexpr float wg=W_G_1; };
template<> struct GDir<1> { static constexpr int cx= 1, cy= 0, cz= 0; static constexpr float wg=W_G_2; };
template<> struct GDir<2> { static constexpr int cx=-1, cy= 0, cz= 0; static constexpr float wg=W_G_2; };
template<> struct GDir<3> { static constexpr int cx= 0, cy= 1, cz= 0; static constexpr float wg=W_G_2; };
template<> struct GDir<4> { static constexpr int cx= 0, cy=-1, cz= 0; static constexpr float wg=W_G_2; };
template<> struct GDir<5> { static constexpr int cx= 0, cy= 0, cz= 1; static constexpr float wg=W_G_2; };
template<> struct GDir<6> { static constexpr int cx= 0, cy= 0, cz=-1; static constexpr float wg=W_G_2; };

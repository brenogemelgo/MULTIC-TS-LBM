#pragma once
#include "../include/cudaUtils.cuh"
#include "../include/velocitySets.cuh"
#if defined(PERTURBATION)
#include "../include/perturbationData.cuh"
#endif

#define RUN_MODE
//#define SAMPLE_MODE
//#define DEBUG_MODE

#if defined(RUN_MODE)
constexpr int MACRO_SAVE = 100;
constexpr int NSTEPS = 10000;
#elif defined(SAMPLE_MODE)
constexpr int MACRO_SAVE = 100;
constexpr int NSTEPS = 1000;
#elif defined(DEBUG_MODE)
constexpr int MACRO_SAVE = 1;
constexpr int NSTEPS = 0;
#endif

#if defined(JET)

constexpr int MESH = 128;
constexpr int DIAM = 20; 
constexpr int NX   = MESH;
constexpr int NY   = MESH;
constexpr int NZ   = MESH*2;

constexpr float U_REF    = 0.05f; 
constexpr int   REYNOLDS = 5000;     
constexpr int   WEBER    = 500;  

#elif defined(DROPLET)

constexpr int MESH   = 64;
constexpr int RADIUS = 10; 
constexpr int DIAM   = 2 * RADIUS;
constexpr int NX     = MESH;
constexpr int NY     = MESH;
constexpr int NZ     = MESH;

constexpr float U_REF    = 0.05f; 
constexpr int   REYNOLDS = 200;        
constexpr int   WEBER    = 4; 
    
#endif 

constexpr float VISC_WATER = (U_REF * DIAM) / REYNOLDS; 
constexpr float VISC_OIL   = 5.0f * VISC_WATER;     

constexpr float VISC_REF = VISC_WATER;
constexpr float OMEGA    = 1.0f / (0.5f + 3.0f * VISC_REF);

constexpr float TAU_REF  = 0.5f + 3.0f * VISC_REF;        
constexpr float SIGMA    = (U_REF * U_REF * DIAM) / WEBER; 

constexpr float GAMMA = 0.3f * 3.0f; 
constexpr float CSSQ  = 1.0f / 3.0f;  

#if defined(JET)

constexpr float K        = 50.0f;
constexpr float P        = 3.0f;            
constexpr int   CELLS    = int(NZ/12);      
static_assert(CELLS > 0, "CELLS must be > 0");

constexpr float SPONGE     = float(CELLS) / float(NZ-1);
constexpr float Z_START    = float(NZ-1-CELLS) / float(NZ-1);
constexpr float INV_NZ_M1  = 1.0f / float(NZ-1);
constexpr float INV_SPONGE = 1.0f / SPONGE;

constexpr float OMEGA_MAX = 1.0f / ((VISC_REF * (K + 1.0f)) / CSSQ + 0.5f);
constexpr float OMCO_MAX  = 1.0f - OMEGA_MAX; 
constexpr float OMEGA_DELTA = OMEGA_MAX - OMEGA; 

#endif

constexpr idx_t STRIDE = NX * NY;
constexpr idx_t PLANE  = NX * NY * NZ;

constexpr idx_t PLANE2  = 2 * PLANE;
constexpr idx_t PLANE3  = 3 * PLANE;
constexpr idx_t PLANE4  = 4 * PLANE;
constexpr idx_t PLANE5  = 5 * PLANE;
constexpr idx_t PLANE6  = 6 * PLANE;
constexpr idx_t PLANE7  = 7 * PLANE;
constexpr idx_t PLANE8  = 8 * PLANE;
constexpr idx_t PLANE9  = 9 * PLANE;
constexpr idx_t PLANE10 = 10 * PLANE;
constexpr idx_t PLANE11 = 11 * PLANE;
constexpr idx_t PLANE12 = 12 * PLANE;   
constexpr idx_t PLANE13 = 13 * PLANE;
constexpr idx_t PLANE14 = 14 * PLANE;
constexpr idx_t PLANE15 = 15 * PLANE;
constexpr idx_t PLANE16 = 16 * PLANE;
constexpr idx_t PLANE17 = 17 * PLANE;
constexpr idx_t PLANE18 = 18 * PLANE;
#if defined(D3Q27)
constexpr idx_t PLANE19 = 19 * PLANE;
constexpr idx_t PLANE20 = 20 * PLANE;
constexpr idx_t PLANE21 = 21 * PLANE;
constexpr idx_t PLANE22 = 22 * PLANE;
constexpr idx_t PLANE23 = 23 * PLANE;
constexpr idx_t PLANE24 = 24 * PLANE;
constexpr idx_t PLANE25 = 25 * PLANE;
constexpr idx_t PLANE26 = 26 * PLANE;
#endif                     

 
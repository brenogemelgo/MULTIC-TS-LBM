#pragma once
#include "../aux/cudaUtils.cuh"
#include "../aux/velocitySets.cuh"
#if defined(PERTURBATION)
#include "../aux/perturbationData.cuh"
#endif

#define RUN_MODE
//#define SAMPLE_MODE
//#define DEBUG_MODE

#if defined(RUN_MODE)

    static constexpr int MACRO_SAVE = 1000;
    static constexpr int NSTEPS = 100000;
    
#elif defined(SAMPLE_MODE)

    static constexpr int MACRO_SAVE = 100;
    static constexpr int NSTEPS = 1000;

#elif defined(DEBUG_MODE)

    static constexpr int MACRO_SAVE = 1;
    static constexpr int NSTEPS = 0;

#endif

#if defined(JET)

    static constexpr idx_t MESH = 128;
    static constexpr idx_t NX   = MESH;
    static constexpr idx_t NY   = MESH;
    static constexpr idx_t NZ   = MESH*2;

    static constexpr int   DIAM   = 20; 
    static constexpr float RADIUS = 0.5f * static_cast<float>(DIAM);
    static constexpr float R2     = RADIUS * RADIUS; 

    static constexpr float U_REF    = 0.05f; 
    static constexpr int   REYNOLDS = 5000;     
    static constexpr int   WEBER    = 2000;  

#elif defined(DROPLET)

    static constexpr idx_t MESH = 128;
    static constexpr idx_t NX   = MESH;
    static constexpr idx_t NY   = MESH;
    static constexpr idx_t NZ   = MESH;

    static constexpr int RADIUS = 20; 
    static constexpr int DIAM   = 2 * RADIUS;

    static constexpr float U_REF    = 0.05f; 
    static constexpr int   REYNOLDS = 200;        
    static constexpr int   WEBER    = 4; 
    
#endif 

static constexpr float GAMMA = 0.9f;
static constexpr float INT_W = 4.0f / GAMMA; 
//static constexpr float GAMMA = 4.0f / INT_W; 

static constexpr float CENTER_X = (NX-1) * 0.5f;
static constexpr float CENTER_Y = (NY-1) * 0.5f;
static constexpr float CENTER_Z = (NZ-1) * 0.5f;

//#define VISC_CONTRAST
#if defined(VISC_CONTRAST)

    static constexpr float VISC_WATER = (U_REF * DIAM) / REYNOLDS; 
    static constexpr float VISC_OIL   = 10.0f * VISC_WATER;     

    static constexpr float VISC_REF = VISC_WATER; 

    static constexpr float OMEGA_WATER = 1.0f / (0.5f + 3.0f * VISC_WATER);
    static constexpr float OMEGA_OIL   = 1.0f / (0.5f + 3.0f * VISC_OIL);

    static constexpr float OMEGA_REF = OMEGA_WATER;

    static constexpr float OMCO_ZMIN = 1.0f - OMEGA_OIL;

#else

    static constexpr float VISC_REF = (U_REF * DIAM) / REYNOLDS;
    static constexpr float OMEGA_REF = 1.0f / (0.5f + 3.0f * VISC_REF);
 
    static constexpr float OMCO_ZMIN = 1.0f - OMEGA_REF;

#endif

static constexpr float SIGMA = (U_REF * U_REF * DIAM) / WEBER; 
static constexpr float CSSQ  = 1.0f / 3.0f;  
static constexpr float CSCO  = 1.0f - CSSQ;

#if defined(JET)

    static constexpr float K          = 100.0f; 
    static constexpr float P          = 3.0f;            
    static constexpr int SPONGE_CELLS = static_cast<int>(NZ/12);      
    static_assert(SPONGE_CELLS > 0, "SPONGE_CELLS must be > 0");

    static constexpr float SPONGE     = static_cast<float>(SPONGE_CELLS) / static_cast<float>(NZ-1);
    static constexpr float Z_START    = static_cast<float>(NZ-1-SPONGE_CELLS) / static_cast<float>(NZ-1);
    static constexpr float INV_NZ_M1  = 1.0f / static_cast<float>(NZ-1);
    static constexpr float INV_SPONGE = 1.0f / SPONGE;

    static constexpr float OMEGA_ZMAX  = 1.0f / (0.5f + 3.0f * VISC_REF * (K + 1.0f));
    static constexpr float OMCO_ZMAX   = 1.0f - OMEGA_ZMAX; 
    static constexpr float OMEGA_DELTA = OMEGA_ZMAX - OMEGA_REF; 

#endif

static constexpr idx_t STRIDE = NX * NY;
static constexpr idx_t PLANE  = NX * NY * NZ;
                   

 
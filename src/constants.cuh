#pragma once
#include "../include/cudaUtils.cuh"
#include "../include/velocitySets.cuh"
#include "../include/perturbationData.cuh"

#define JET_CASE
//#define DROPLET_CASE

#define RUN_MODE
//#define SAMPLE_MODE
//#define DEBUG_MODE

#ifdef RUN_MODE
    constexpr int MACRO_SAVE = 100;
    constexpr int NSTEPS = 50000;
#elif defined(SAMPLE_MODE)
    constexpr int MACRO_SAVE = 100;
    constexpr int NSTEPS = 1000;
#elif defined(DEBUG_MODE)
    constexpr int MACRO_SAVE = 1;
    constexpr int NSTEPS = 0;
#endif

#ifdef JET_CASE
    // domain size
    constexpr int MESH = 64;
    constexpr int DIAM = 10;
    constexpr int NX   = MESH;
    constexpr int NY   = MESH;
    constexpr int NZ   = MESH*2;
    // jet velocity
    constexpr float U_JET = 0.05f; 
    // adimensional parameters
    constexpr int REYNOLDS = 5000; 
    constexpr int WEBER    = 500; 
    // general model parameters
    constexpr float VISC  = (U_JET * DIAM) / REYNOLDS;      // kinematic viscosity
    constexpr float TAU   = 0.5f + 3.0f * VISC;             // relaxation time
    constexpr float GAMMA = 0.3f * 3.0f;                    // sharpening of the interface
    constexpr float SIGMA = (U_JET * U_JET * DIAM) / WEBER; // surface tension coefficient
#elif defined(DROPLET_CASE)
    // domain size
    constexpr int MESH = 64;
    constexpr int RADIUS = 9; 
    constexpr int NX   = MESH;
    constexpr int NY   = MESH;
    constexpr int NZ   = MESH;
    // general model parameters
    constexpr float TAU      = 0.55f;        // relaxation time
    constexpr float GAMMA    = 0.15f * 5.0f; // sharpening of the interface
    constexpr float SIGMA    = 0.1f;         // surface tension coefficient
#endif // FLOW_CASE

// sponge parameters
constexpr float K   = 50.0f;      // gain factor 
constexpr float P   = 3.0f;       // transition degree (polynomial)
constexpr int CELLS = int(NZ/12); // width    

// general model parameters and auxiliary constants
constexpr float CSSQ   = 1.0f / 3.0f;  // square of speed of sound
constexpr float OMEGA  = 1.0f / TAU;   // relaxation frequency
constexpr float OOS    = 1.0f / 6.0f;  // one over six
constexpr float OMCO   = 1.0f - OMEGA; // complementary of omega
constexpr float CSCO   = 1.0f - CSSQ;  // complementary of cssq

// indexing auxiliary constants
constexpr idx_t PLANE  = (idx_t)NX * NY * NZ;
constexpr idx_t STRIDE = (idx_t)NX * NY;

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
#ifdef D3Q27
constexpr idx_t PLANE19 = 19 * PLANE;
constexpr idx_t PLANE20 = 20 * PLANE;
constexpr idx_t PLANE21 = 21 * PLANE;
constexpr idx_t PLANE22 = 22 * PLANE;
constexpr idx_t PLANE23 = 23 * PLANE;
constexpr idx_t PLANE24 = 24 * PLANE;
constexpr idx_t PLANE25 = 25 * PLANE;
constexpr idx_t PLANE26 = 26 * PLANE;
#endif // D3Q27

// sponge related auxiliary constants
constexpr float SPONGE    = float(CELLS) / float(NZ-1);                 // sponge width in normalized coordinates
constexpr float Z_START   = float(NZ-1-CELLS) / float(NZ-1);            // z coordinate where the sponge starts
constexpr float OMEGA_MAX = 1.0f / ((VISC * (K + 1.0f)) / CSSQ + 0.5f); // omega at z=max
constexpr float OMCO_MAX  = 1.0f - OMEGA_MAX;                           // complementary of omega at z=max

 
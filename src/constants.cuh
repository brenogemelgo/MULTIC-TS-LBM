#pragma once
#include "../helpers/cudaUtils.cuh"
#include "../helpers/globalStructs.cuh"
#include "../helpers/velocitySets.cuh"

#define RUN_MODE
//#define SAMPLE_MODE
//#define DEBUG_MODE

#if defined(RUN_MODE)

    static inline constexpr int MACRO_SAVE = 1000;
    static inline constexpr int NSTEPS = 100000;

#elif defined(SAMPLE_MODE)

    static inline constexpr int MACRO_SAVE = 100;
    static inline constexpr int NSTEPS = 1000;

#elif defined(DEBUG_MODE)

    static inline constexpr int MACRO_SAVE = 1;
    static inline constexpr int NSTEPS = 0;

#endif

#if defined(JET)

    static inline constexpr idx_t MESH = 128;
    static inline constexpr idx_t NX   = MESH;
    static inline constexpr idx_t NY   = MESH;
    static inline constexpr idx_t NZ   = MESH*2;

    static inline constexpr int DIAM   = 13;
    static inline constexpr int RADIUS = DIAM / 2;

    static inline constexpr float U_REF    = 0.05f;
    static inline constexpr int   REYNOLDS = 5000;
    static inline constexpr int   WEBER    = 500;

    static inline constexpr float GAMMA = 1.0f;

#elif defined(DROPLET)

    static inline constexpr idx_t MESH = 150;
    static inline constexpr idx_t NX   = MESH;
    static inline constexpr idx_t NY   = MESH;
    static inline constexpr idx_t NZ   = MESH;

    static inline constexpr int RADIUS = 20;
    static inline constexpr int DIAM   = 2 * RADIUS;

    static inline constexpr float TAU   = 0.55f;
    static inline constexpr float SIGMA = 0.1f;

    static inline constexpr float GAMMA = 1.0f;

#endif

#include "../include/auxConstants.cuh"
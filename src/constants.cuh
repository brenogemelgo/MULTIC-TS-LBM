#pragma once
#include "../helpers/cudaUtils.cuh"
#include "../helpers/commonStructs.cuh"
#include "../helpers/velocitySets.cuh"

//#define RUN_MODE
//#define SAMPLE_MODE
#define DEBUG_MODE

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

    struct mesh {

        static constexpr idx_t res    = 128;
        static constexpr idx_t nx     = res;
        static constexpr idx_t ny     = res;
        static constexpr idx_t nz     = res * 2;
        static constexpr int   diam   = 13;
        static constexpr int   radius = diam / 2;

    };

    struct physics {

        static constexpr float u_ref    = 0.05f;
        static constexpr int   reynolds = 5000;
        static constexpr int   weber    = 500;
        static constexpr float sigma    = (u_ref * u_ref * mesh::diam) / weber;
        static constexpr float gamma    = 1.0f;

    };

#elif defined(DROPLET)

    struct mesh {

        static constexpr idx_t res    = 75;
        static constexpr idx_t nx     = res;
        static constexpr idx_t ny     = res;
        static constexpr idx_t nz     = res;
        static constexpr int   radius = 10;
        static constexpr int   diam   = 2 * radius;

    };

    struct physics {

        static constexpr float u_ref    = 0.0f; 
        static constexpr int   reynolds = 0;
        static constexpr int   weber    = 0;    
        static constexpr float sigma    = 0.1f;
        static constexpr float gamma    = 0.15f * 5.0f;

        static constexpr float tau      = 0.55f;
        static constexpr float visc_ref = (tau - 0.5f) / 3.0f;
    };

#endif

#include "../helpers/auxConstants.cuh"
#pragma once
#include "../helpers/cudaUtils.cuh"
#include "../helpers/velocitySets.cuh"
#if defined(PERTURBATION)
#include "../helpers/perturbationData.cuh"
#endif

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

static inline constexpr idx_t MESH = 64;
static inline constexpr idx_t NX   = MESH;
static inline constexpr idx_t NY   = MESH;
static inline constexpr idx_t NZ   = MESH * 2;

static inline constexpr int DIAM   = 10;
static inline constexpr int RADIUS = DIAM / 2;

static inline constexpr float U_REF    = 0.05f;
static inline constexpr int   REYNOLDS = 5000;
static inline constexpr int   WEBER    = 100;

static inline constexpr float SIGMA = (U_REF * U_REF * DIAM) / WEBER;

static inline constexpr float GAMMA = 0.9f;

#elif defined(DROPLET)

static inline constexpr idx_t MESH = 150;
static inline constexpr idx_t NX   = MESH;
static inline constexpr idx_t NY   = MESH;
static inline constexpr idx_t NZ   = MESH;

static inline constexpr int RADIUS = 20;
static inline constexpr int DIAM   = 2 * RADIUS;

static inline constexpr float TAU   = 0.55f;
static inline constexpr float SIGMA = 0.1f;

static inline constexpr float VISC_REF = (TAU - 0.5f) / 3.0f;

static inline constexpr int   REYNOLDS = 0;
static inline constexpr int   WEBER    = 0;       
static inline constexpr float U_REF    = 0.0f; 

static inline constexpr float GAMMA = 0.45f;

#endif

#include "../include/auxConstants.cuh"

#if defined(JET)

static inline constexpr float K = 100.0f;
static inline constexpr float P = 3.0f;

static inline constexpr int SPONGE_CELLS = static_cast<int>(NZ / 12);
static_assert(SPONGE_CELLS > 0, "SPONGE_CELLS must be > 0");

static inline constexpr float SPONGE     = static_cast<float>(SPONGE_CELLS) / static_cast<float>(NZ - 1);
static inline constexpr float Z_START    = static_cast<float>(NZ - 1 - SPONGE_CELLS) / static_cast<float>(NZ - 1);
static inline constexpr float INV_NZ_M1  = 1.0f / static_cast<float>(NZ - 1);
static inline constexpr float INV_SPONGE = 1.0f / SPONGE;

static inline constexpr float OMEGA_ZMAX  = 1.0f / (0.5f + 3.0f * VISC_REF * (K + 1.0f));
static inline constexpr float OMCO_ZMAX   = 1.0f - OMEGA_ZMAX;
static inline constexpr float OMEGA_DELTA = OMEGA_ZMAX - OMEGA_REF;

#endif
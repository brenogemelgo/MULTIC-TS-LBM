#if defined(JET)

    static inline constexpr float SIGMA = (U_REF * U_REF * DIAM) / WEBER;

#endif

#if defined(DROPLET)

    static inline constexpr float VISC_REF = (TAU - 0.5f) / 3.0f;
    static inline constexpr int   REYNOLDS = 0;
    static inline constexpr int   WEBER    = 0;       
    static inline constexpr float U_REF    = 0.0f; 

#endif

static inline constexpr float INT_W = 4.0f / GAMMA;
// static inline constexpr float GAMMA = 4.0f / INT_W;

static inline constexpr float RHO_WATER = 1.0f;
static inline constexpr float RHO_OIL   = 0.8f;

static inline constexpr float R2 = RADIUS * RADIUS;

#if defined(JET)

    // #define VISC_CONTRAST
    #if defined(VISC_CONTRAST)

        static inline constexpr float VISC_WATER  = (U_REF * DIAM) / REYNOLDS;
        static inline constexpr float VISC_OIL    = 10.0f * VISC_WATER;
        static inline constexpr float VISC_REF    = VISC_WATER;
        static inline constexpr float OMEGA_WATER = 1.0f / (0.5f + 3.0f * VISC_WATER);
        static inline constexpr float OMEGA_OIL   = 1.0f / (0.5f + 3.0f * VISC_OIL);
        static inline constexpr float OMEGA_REF   = OMEGA_WATER;
        static inline constexpr float OMCO_ZMIN   = 1.0f - OMEGA_OIL;

    #else

        static inline constexpr float VISC_REF  = (U_REF * DIAM) / REYNOLDS;
        static inline constexpr float OMEGA_REF = 1.0f / (0.5f + 3.0f * VISC_REF);
        static inline constexpr float OMCO_ZMIN = 1.0f - OMEGA_REF;

    #endif

#else 

    static inline constexpr float OMEGA_REF = 1.0f / (0.5f + 3.0f * VISC_REF);

#endif

static inline constexpr float CSSQ = 1.0f / 3.0f;
static inline constexpr float CSCO = 1.0f - CSSQ;
static inline constexpr float OMCO = 1.0f - OMEGA_REF;
static inline constexpr float OOS  = 1.0f / 6.0f;

static inline constexpr float CENTER_X = (NX - 1) * 0.5f;
static inline constexpr float CENTER_Y = (NY - 1) * 0.5f;
static inline constexpr float CENTER_Z = (NZ - 1) * 0.5f;

static inline constexpr idx_t STRIDE = NX * NY;
static inline constexpr idx_t PLANE  = NX * NY * NZ;

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
static constexpr unsigned NUM_BLOCK_X = (mesh::nx + block::nx - 1u) / block::nx;
static constexpr unsigned NUM_BLOCK_Y = (mesh::ny + block::ny - 1u) / block::ny;

static constexpr float RHO_WATER = 1.0f;
static constexpr float RHO_OIL = 0.8f;

static constexpr float R2 = mesh::radius * mesh::radius;

static constexpr float P_IN = 1.0f + 3.0f * physics::u_ref + 3.0f * physics::u_ref * physics::u_ref;
static constexpr float CONST_IN = 1.0f + 3.0f * physics::u_ref;
static constexpr float CONST_OUT = 1.0f - 3.0f * physics::u_ref;

#if defined(JET)

// #define VISC_CONTRAST
#if defined(VISC_CONTRAST)

static constexpr float VISC_WATER = (physics::u_ref * mesh::diam) / physics::reynolds;
static constexpr float VISC_OIL = 10.0f * VISC_WATER;
static constexpr float VISC_REF = VISC_WATER;
static constexpr float OMEGA_WATER = 1.0f / (0.5f + 3.0f * VISC_WATER);
static constexpr float OMEGA_OIL = 1.0f / (0.5f + 3.0f * VISC_OIL);
static constexpr float OMEGA_REF = OMEGA_WATER;
static constexpr float OMCO_ZMIN = 1.0f - OMEGA_OIL;

#else

static constexpr float VISC_REF = (physics::u_ref * mesh::diam) / physics::reynolds;
static constexpr float OMEGA_REF = 1.0f / (0.5f + 3.0f * VISC_REF);
static constexpr float OMCO_ZMIN = 1.0f - OMEGA_REF;

#endif

#elif defined(DROPLET)

static constexpr float OMEGA_REF = 1.0f / (0.5f + 3.0f * physics::visc_ref);

#endif

static constexpr float CSSQ = 1.0f / 3.0f;
static constexpr float CSCO = 1.0f - CSSQ;
static constexpr float OMCO = 1.0f - OMEGA_REF;
static constexpr float OOS = 1.0f / 6.0f;

static constexpr float TWO_PI = 2.0f * CUDART_PI_F;

static constexpr float CENTER_X = (mesh::nx - 1) * 0.5f;
static constexpr float CENTER_Y = (mesh::ny - 1) * 0.5f;
static constexpr float CENTER_Z = (mesh::nz - 1) * 0.5f;

static constexpr label_t STRIDE = mesh::nx * mesh::ny;
static constexpr label_t PLANE = mesh::nx * mesh::ny * mesh::nz;

#if defined(JET)

static constexpr float K = 100.0f;
static constexpr float P = 3.0f;

static constexpr int SPONGE_CELLS = static_cast<int>(mesh::nz / 12);
static_assert(SPONGE_CELLS > 0, "SPONGE_CELLS must be > 0");

static constexpr float SPONGE = static_cast<float>(SPONGE_CELLS) / static_cast<float>(mesh::nz - 1);
static constexpr float Z_START = static_cast<float>(mesh::nz - 1 - SPONGE_CELLS) / static_cast<float>(mesh::nz - 1);
static constexpr float INV_NZ_M1 = 1.0f / static_cast<float>(mesh::nz - 1);
static constexpr float INV_SPONGE = 1.0f / SPONGE;

static constexpr float OMEGA_ZMAX = 1.0f / (0.5f + 3.0f * VISC_REF * (K + 1.0f));
static constexpr float OMCO_ZMAX = 1.0f - OMEGA_ZMAX;
static constexpr float OMEGA_DELTA = OMEGA_ZMAX - OMEGA_REF;

#endif
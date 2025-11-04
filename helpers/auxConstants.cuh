static constexpr unsigned NUM_BLOCK_X = (mesh::nx + block::nx - 1u) / block::nx;
static constexpr unsigned NUM_BLOCK_Y = (mesh::ny + block::ny - 1u) / block::ny;

static constexpr scalar_t RHO_WATER = 1.0f;
static constexpr scalar_t RHO_OIL = 0.8f;

static constexpr scalar_t R2 = mesh::radius * mesh::radius;

static constexpr scalar_t P_IN = 1.0f + 3.0f * physics::u_ref + 3.0f * physics::u_ref * physics::u_ref;
static constexpr scalar_t CONST_IN = 1.0f + 3.0f * physics::u_ref;
static constexpr scalar_t CONST_OUT = 1.0f - 3.0f * physics::u_ref;

#if defined(JET)

// #define VISC_CONTRAST
#if defined(VISC_CONTRAST)

static constexpr scalar_t VISC_WATER = (physics::u_ref * mesh::diam) / physics::reynolds;
static constexpr scalar_t VISC_OIL = 10.0f * VISC_WATER;
static constexpr scalar_t VISC_REF = VISC_WATER;
static constexpr scalar_t OMEGA_WATER = 1.0f / (0.5f + 3.0f * VISC_WATER);
static constexpr scalar_t OMEGA_OIL = 1.0f / (0.5f + 3.0f * VISC_OIL);
static constexpr scalar_t OMEGA_REF = OMEGA_WATER;
static constexpr scalar_t OMCO_ZMIN = 1.0f - OMEGA_OIL;

#else

static constexpr scalar_t VISC_REF = (physics::u_ref * mesh::diam) / physics::reynolds;
static constexpr scalar_t OMEGA_REF = 1.0f / (0.5f + 3.0f * VISC_REF);
static constexpr scalar_t OMCO_ZMIN = 1.0f - OMEGA_REF;

#endif

#elif defined(DROPLET)

static constexpr scalar_t OMEGA_REF = 1.0f / (0.5f + 3.0f * physics::visc_ref);

#endif

static constexpr scalar_t CSSQ = 1.0f / 3.0f;
static constexpr scalar_t CSCO = 1.0f - CSSQ;
static constexpr scalar_t OMCO = 1.0f - OMEGA_REF;
static constexpr scalar_t OOS = 1.0f / 6.0f;

static constexpr scalar_t TWO_PI = 2.0f * CUDART_PI_F;

static constexpr scalar_t CENTER_X = (mesh::nx - 1) * 0.5f;
static constexpr scalar_t CENTER_Y = (mesh::ny - 1) * 0.5f;
static constexpr scalar_t CENTER_Z = (mesh::nz - 1) * 0.5f;

static constexpr label_t STRIDE = mesh::nx * mesh::ny;
static constexpr label_t PLANE = mesh::nx * mesh::ny * mesh::nz;

#if defined(JET)

static constexpr scalar_t K = 100.0f;
static constexpr scalar_t P = 3.0f;

static constexpr int SPONGE_CELLS = static_cast<int>(mesh::nz / 12);
static_assert(SPONGE_CELLS > 0, "SPONGE_CELLS must be > 0");

static constexpr scalar_t SPONGE = static_cast<scalar_t>(SPONGE_CELLS) / static_cast<scalar_t>(mesh::nz - 1);
static constexpr scalar_t Z_START = static_cast<scalar_t>(mesh::nz - 1 - SPONGE_CELLS) / static_cast<scalar_t>(mesh::nz - 1);
static constexpr scalar_t INV_NZ_M1 = 1.0f / static_cast<scalar_t>(mesh::nz - 1);
static constexpr scalar_t INV_SPONGE = 1.0f / SPONGE;

static constexpr scalar_t OMEGA_ZMAX = 1.0f / (0.5f + 3.0f * VISC_REF * (K + 1.0f));
static constexpr scalar_t OMCO_ZMAX = 1.0f - OMEGA_ZMAX;
static constexpr scalar_t OMEGA_DELTA = OMEGA_ZMAX - OMEGA_REF;

#endif
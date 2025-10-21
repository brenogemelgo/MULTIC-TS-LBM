#pragma once
#include "constants.cuh"

[[nodiscard]] static __host__ __forceinline__ 
std::string createSimulationDirectory(
    const std::string &FLOW_CASE,
    const std::string &VELOCITY_SET,
    const std::string &SIM_ID
) noexcept(false) {
    std::filesystem::path BASE_DIR = std::filesystem::current_path();

    std::filesystem::path SIM_DIR = BASE_DIR / "bin" / FLOW_CASE / VELOCITY_SET / SIM_ID;

    std::error_code EC;
    std::filesystem::create_directories(SIM_DIR, EC);

    return SIM_DIR.string() + std::string(1, std::filesystem::path::preferred_separator);
}

[[gnu::cold]] static __host__ __forceinline__
void generateSimulationInfoFile(
    const std::string &SIM_DIR,
    const std::string &SIM_ID,
    const std::string &VELOCITY_SET,
    const double MLUPS
) noexcept(false) {
    std::filesystem::path INFO_PATH = std::filesystem::path(SIM_DIR) / (SIM_ID + "_info.txt");

    try {
        std::ofstream file(INFO_PATH, std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << INFO_PATH.string() << std::endl;
            return;
        }

        file << "---------------------------- SIMULATION METADATA ----------------------------\n"
             << "ID:                 " << SIM_ID << '\n'
             << "Velocity set:       " << VELOCITY_SET << '\n'
             << "Reference velocity: " << U_REF << '\n'
             << "Reynolds number:    " << REYNOLDS << '\n'
             << "Weber number:       " << WEBER << "\n\n"
             << "Domain size:        NX=" << NX << ", NY=" << NY << ", NZ=" << NZ << '\n'
             << "Diameter:           D=" << DIAM << '\n'
             << "Timesteps:          " << NSTEPS << '\n'
             << "Output interval:    " << MACRO_SAVE << "\n\n"
             << "Performance:        " << MLUPS << " MLUPS\n"
             << "-----------------------------------------------------------------------------\n";

        file.close();
        std::cout << "Simulation information file created in: " << INFO_PATH.string() << std::endl;
    }
    catch (const std::exception &e) {
        std::cerr << "Error generating information file: " << e.what() << std::endl;
    }
}

[[gnu::cold]] static __host__ __forceinline__ 
void copyAndSaveToBinary(
    const float *d_data,
    const size_t SIZE,
    const std::string &SIM_DIR,
    [[maybe_unused]] const std::string &ID,
    const int STEP,
    const std::string &VAR_NAME
) noexcept(false) {
    std::error_code EC;
    std::filesystem::create_directories(std::filesystem::path(SIM_DIR), EC);

    std::vector<float> host_data(SIZE);
    checkCudaErrors(cudaMemcpy(host_data.data(), d_data, SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    std::ostringstream STEP_SAVE;
    STEP_SAVE << std::setw(6) << std::setfill('0') << STEP;
    const std::string filename = VAR_NAME + STEP_SAVE.str() + ".bin";

    const std::filesystem::path OUT_PATH = std::filesystem::path(SIM_DIR) / filename;

    std::ofstream file(OUT_PATH, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file " << OUT_PATH.string() << " for writing." << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char *>(host_data.data()), static_cast<std::streamsize>(host_data.size() * sizeof(float)));
    file.close();
}

[[nodiscard]] [[gnu::cold]] static __host__ __forceinline__ 
int setDeviceFromEnv(
    /* empty */
) noexcept {
    int dev = 0;
    if (const char *env = std::getenv("GPU_INDEX")) {
        char *end = nullptr;
        long v = std::strtol(env, &end, 10);
        if (end != env && v >= 0)
            dev = static_cast<int>(v);
    }

    cudaError_t err = cudaSetDevice(dev);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        std::printf("Using GPU %d: %s (SM %d.%d)\n", dev, prop.name, prop.major, prop.minor);
    }
    else {
        std::printf("Using GPU %d (unknown properties)\n", dev);
    }

    return dev;
}

[[nodiscard]] static __host__ __forceinline__ constexpr
unsigned divUp(
    const unsigned a,
    const unsigned b
) noexcept {
    return (a + b - 1u) / b;
}

[[gnu::cold]] static __host__ __forceinline__ 
void setDeviceFields(
    /* empty */
) noexcept(false) {
    constexpr size_t NCELLS = static_cast<size_t>(NX) * static_cast<size_t>(NY) * static_cast<size_t>(NZ);
    constexpr size_t SIZE = NCELLS * sizeof(float);
    constexpr size_t F_DIST_SIZE = NCELLS * static_cast<size_t>(FLINKS) * sizeof(pop_t);
    constexpr size_t G_DIST_SIZE = NCELLS * static_cast<size_t>(GLINKS) * sizeof(float);

    static_assert(NCELLS > 0, "Empty grid?");
    static_assert(SIZE / sizeof(float) == NCELLS, "SIZE overflow");
    static_assert(F_DIST_SIZE / sizeof(pop_t) == NCELLS * size_t(FLINKS), "F_DIST_SIZE overflow");
    static_assert(G_DIST_SIZE / sizeof(float) == NCELLS * size_t(GLINKS), "G_DIST_SIZE overflow");

    checkCudaErrors(cudaMalloc(&fields.rho, SIZE));
    checkCudaErrors(cudaMalloc(&fields.ux,  SIZE));
    checkCudaErrors(cudaMalloc(&fields.uy,  SIZE));
    checkCudaErrors(cudaMalloc(&fields.uz,  SIZE));
    checkCudaErrors(cudaMalloc(&fields.pxx, SIZE));
    checkCudaErrors(cudaMalloc(&fields.pyy, SIZE));
    checkCudaErrors(cudaMalloc(&fields.pzz, SIZE));
    checkCudaErrors(cudaMalloc(&fields.pxy, SIZE));
    checkCudaErrors(cudaMalloc(&fields.pxz, SIZE));
    checkCudaErrors(cudaMalloc(&fields.pyz, SIZE));

    checkCudaErrors(cudaMalloc(&fields.phi,   SIZE));
    checkCudaErrors(cudaMalloc(&fields.normx, SIZE));
    checkCudaErrors(cudaMalloc(&fields.normy, SIZE));
    checkCudaErrors(cudaMalloc(&fields.normz, SIZE));
    checkCudaErrors(cudaMalloc(&fields.ind,   SIZE));
    checkCudaErrors(cudaMalloc(&fields.ffx,   SIZE));
    checkCudaErrors(cudaMalloc(&fields.ffy,   SIZE));
    checkCudaErrors(cudaMalloc(&fields.ffz,   SIZE));

    checkCudaErrors(cudaMalloc(&fields.f, F_DIST_SIZE));
    checkCudaErrors(cudaMalloc(&fields.g, G_DIST_SIZE));

    getLastCudaErrorOutline("setDeviceFields: post-initialization");
}

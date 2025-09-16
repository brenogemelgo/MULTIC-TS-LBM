#pragma once
#include "constants.cuh"
#include <filesystem>

__host__ __forceinline__
std::string createSimulationDirectory(
    const std::string& FLOW_CASE,
    const std::string& VELOCITY_SET,
    const std::string& SIM_ID
) {
    std::filesystem::path BASE_DIR = std::filesystem::current_path();

    std::filesystem::path SIM_DIR = BASE_DIR / "bin" / FLOW_CASE / VELOCITY_SET / SIM_ID;

    std::error_code EC;
    std::filesystem::create_directories(SIM_DIR, EC); 

    return SIM_DIR.string() + std::filesystem::path::preferred_separator;
}

#include <filesystem>
__host__ __forceinline__
void generateSimulationInfoFile(
    const std::string& SIM_DIR,           
    const std::string& SIM_ID,
    const std::string& VELOCITY_SET,
    const double MLUPS
) {
    std::filesystem::path INFO_PATH = std::filesystem::path(SIM_DIR) / (SIM_ID + "_info.txt");

    try {
        std::ofstream file(INFO_PATH, std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << INFO_PATH.string() << std::endl;
            return;
        }

        file << "---------------------------- SIMULATION METADATA ----------------------------\n"
             << "ID:                " << SIM_ID << '\n'
             << "Velocity set:      " << VELOCITY_SET << '\n'
             << "Reference velocity:" << " " << U_REF << '\n'
             << "Reynolds number:   " << REYNOLDS << '\n'
             << "Weber number:      " << WEBER << "\n\n"
             << "Domain size:       NX=" << NX << ", NY=" << NY << ", NZ=" << NZ << '\n'
             << "Timesteps:         " << NSTEPS << '\n'
             << "Output interval:   " << MACRO_SAVE << "\n\n"
             << "Performance:       " << MLUPS << " MLUPS\n"
             << "-----------------------------------------------------------------------------\n";

        file.close();
        std::cout << "Simulation information file created in: " << INFO_PATH.string() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error generating information file: " << e.what() << std::endl;
    }
}

__host__ __forceinline__
void copyAndSaveToBinary(
    const float* d_data,
    const size_t SIZE,
    const std::string& SIM_DIR,   
    [[maybe_unused]] const std::string& ID,        
    const int STEP,
    const std::string& VAR_NAME
) {
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
    file.write(reinterpret_cast<const char*>(host_data.data()), static_cast<std::streamsize>(host_data.size() * sizeof(float)));
    file.close();
}

__host__ __forceinline__
void printCudaMemUsage(const char* tag = nullptr, bool reset_baseline = false) {
    int dev = -1;
    cudaGetDevice(&dev);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    static size_t baseline_free  = 0;
    static size_t baseline_total = 0;
    static bool   baseline_set   = false;

    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);

    if (reset_baseline || !baseline_set) {
        baseline_free  = freeB;
        baseline_total = totalB; 
        baseline_set   = true;
    }

    const size_t used_now   = totalB - freeB;
    const size_t used_base  = baseline_total - baseline_free;
    const size_t delta_used = (used_now >= used_base) ? (used_now - used_base) : 0;

    auto toMiB = [](size_t b) -> double { return double(b) / (1024.0 * 1024.0); };
    auto toGiB = [](size_t b) -> double { return double(b) / (1024.0 * 1024.0 * 1024.0); };

    std::printf(
        "\n[CUDA MEM] dev=%d (%s)%s\n"
        "  total:  %12zu B  (%.2f MiB, %.2f GiB)\n"
        "  free:   %12zu B  (%.2f MiB, %.2f GiB)\n"
        "  used:   %12zu B  (%.2f MiB, %.2f GiB)\n"
        "  delta:  %12zu B  (%.2f MiB, %.2f GiB)  %s\n",
        dev, prop.name, tag ? tag : "",
        totalB, toMiB(totalB), toGiB(totalB),
        freeB,  toMiB(freeB),  toGiB(freeB),
        used_now, toMiB(used_now), toGiB(used_now),
        delta_used, toMiB(delta_used), toGiB(delta_used),
        reset_baseline ? "[baseline reset]" : ""
    );
}

__host__ __forceinline__
void printCudaMemUsageLight(const char* tag = nullptr, bool reset_baseline = false) {
    int dev = -1;
    cudaGetDevice(&dev);

    static size_t baseline_free  = 0;
    static size_t baseline_total = 0;
    static bool   baseline_set   = false;

    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);

    if (reset_baseline || !baseline_set) {
        baseline_free  = freeB;
        baseline_total = totalB;
        baseline_set   = true;
    }

    const size_t used_now   = totalB - freeB;
    const size_t used_base  = baseline_total - baseline_free;
    const size_t delta_used = (used_now >= used_base) ? (used_now - used_base) : 0;

    auto toMiB = [](size_t b) -> double { return double(b) / (1024.0 * 1024.0); };

    std::printf("[CUDA MEM] %s  used=%.2f MiB  delta=%.2f MiB%s\n",
        tag ? tag : "",
        toMiB(used_now),
        toMiB(delta_used),
        reset_baseline ? " [baseline reset]" : ""
    );
}

__host__ static __forceinline__ constexpr 
unsigned divUp(
    const unsigned a, 
    const unsigned b
) {
    return (a + b - 1u) / b;
}

__host__ __forceinline__ 
void setDevice() {
    constexpr size_t NCELLS = static_cast<size_t>(NX) * static_cast<size_t>(NY) * static_cast<size_t>(NZ);
    constexpr size_t SIZE        = NCELLS * sizeof(float);            
    constexpr size_t F_DIST_SIZE = NCELLS * static_cast<size_t>(FLINKS) * sizeof(pop_t);
    constexpr size_t G_DIST_SIZE = NCELLS * static_cast<size_t>(GLINKS) * sizeof(float); 

    static_assert(NCELLS > 0, "Empty grid?");
    static_assert(SIZE / sizeof(float) == NCELLS, "SIZE overflow");
    static_assert(F_DIST_SIZE / sizeof(pop_t) == NCELLS * size_t(FLINKS), "F_DIST_SIZE overflow");
    static_assert(G_DIST_SIZE / sizeof(float) == NCELLS * size_t(GLINKS), "G_DIST_SIZE overflow");

    checkCudaErrors(cudaMalloc(&lbm.rho, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ux,  SIZE));
    checkCudaErrors(cudaMalloc(&lbm.uy,  SIZE));
    checkCudaErrors(cudaMalloc(&lbm.uz,  SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pxx, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pyy, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pzz, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pxy, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pxz, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pyz, SIZE));

    checkCudaErrors(cudaMalloc(&lbm.phi,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.normx, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.normy, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.normz, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ind,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ffx,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ffy,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ffz,   SIZE));

    checkCudaErrors(cudaMalloc(&lbm.f, F_DIST_SIZE));
    checkCudaErrors(cudaMalloc(&lbm.g, G_DIST_SIZE));

    #if defined(D_FIELDS)
    checkCudaErrors(cudaMalloc(&dfields.vorticity_mag, SIZE));
    checkCudaErrors(cudaMalloc(&dfields.velocity_mag,  SIZE));
    checkCudaErrors(cudaMalloc(&dfields.pressure,      SIZE));
    #endif 

    getLastCudaErrorOutline("initDeviceVars: post-initialization");
}




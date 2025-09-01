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
    const std::string& ID,        
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
    file.write(reinterpret_cast<const char*>(host_data.data()), host_data.size() * sizeof(float));
    file.close();
}

__host__ static __forceinline__ constexpr 
unsigned divUp(
    unsigned a, 
    unsigned b
) {
    return (a + b - 1u) / b;
}

__host__ __forceinline__ 
void setDevice() {
    constexpr size_t SIZE =        NX * NY * NZ          * sizeof(float);            
    constexpr size_t F_DIST_SIZE = NX * NY * NZ * FLINKS * sizeof(pop_t);
    constexpr size_t G_DIST_SIZE = NX * NY * NZ * GLINKS * sizeof(float); 

    checkCudaErrors(cudaMalloc(&lbm.rho,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ux,    SIZE));
    checkCudaErrors(cudaMalloc(&lbm.uy,    SIZE));
    checkCudaErrors(cudaMalloc(&lbm.uz,    SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pxx,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pyy,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pzz,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pxy,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pxz,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.pyz,   SIZE));

    checkCudaErrors(cudaMalloc(&lbm.phi,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ind,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.normx, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.normy, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.normz, SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ffx,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ffy,   SIZE));
    checkCudaErrors(cudaMalloc(&lbm.ffz,   SIZE));

    checkCudaErrors(cudaMalloc(&lbm.f,     F_DIST_SIZE));
    checkCudaErrors(cudaMalloc(&lbm.g,     G_DIST_SIZE));

    #if defined(D_FIELDS)
    checkCudaErrors(cudaMalloc(&dfields.vorticity_mag, SIZE));
    checkCudaErrors(cudaMalloc(&dfields.velocity_mag,  SIZE));
    checkCudaErrors(cudaMalloc(&dfields.pressure,      SIZE));
    #endif 

    getLastCudaError("initDeviceVars: post-initialization");
}




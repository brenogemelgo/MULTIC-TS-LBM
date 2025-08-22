#pragma once
#include "constants.cuh"

__host__ __forceinline__ std::string createSimulationDirectory(
    const std::string& VELOCITY_SET, const std::string& SIM_ID
) {
    std::string BASE_DIR = 
    #ifdef _WIN32
        ".\\";
    #else
        "./";
    #endif

    std::string SIM_DIR = BASE_DIR + "bin/" + VELOCITY_SET + "/" + SIM_ID + "/";
    
    #ifdef _WIN32
        std::string MKDIR_COMMAND = "mkdir \"" + SIM_DIR + "\"";
    #else
        std::string MKDIR_COMMAND = "mkdir -p \"" + SIM_DIR + "\"";
    #endif

    int ret = std::system(MKDIR_COMMAND.c_str());
    (void)ret;

    return SIM_DIR;
}

__host__ __forceinline__ void computeAndPrintOccupancy() {
    int minGridSize = 0, blockSize = 0;
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, gpuCollisionStream, 0, 0);
    if (err != cudaSuccess) {
        std::cerr << "Error in calculating occupancy: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM, gpuCollisionStream, blockSize, 0);

    std::cout << "\n// =============================================== //\n";
    std::cout << "     Optimal block size       : " << blockSize << "\n";
    std::cout << "     Minimum grid size        : " << minGridSize << "\n";
    std::cout << "     Active blocks per SM     : " << maxBlocksPerSM << "\n";
    std::cout << "// =============================================== //\n" << std::endl;
}

__host__ __forceinline__ void generateSimulationInfoFile(
    const std::string& SIM_DIR, const std::string& SIM_ID, const std::string& VELOCITY_SET, 
    const int NSTEPS, const int MACRO_SAVE, 
    const float TAU, const double MLUPS
) {
    std::string INFO_FILE = SIM_DIR + SIM_ID + "_info.txt";
    try {
        std::ofstream file(INFO_FILE);

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << INFO_FILE << std::endl;
            return;
        }

        file << "---------------------------- SIMULATION INFORMATION ----------------------------\n"
             << "                           Simulation ID: " << SIM_ID << '\n'
             << "                           Velocity set: " << VELOCITY_SET << '\n'
             << "                           Precision: float\n"
             << "                           NX: " << NX << '\n'
             << "                           NY: " << NY << '\n'
             << "                           NZ: " << NZ << '\n'
             << "                           NZ_TOTAL: " << NZ << '\n'
             << "                           Tau: " << TAU << '\n'
             << "                           Umax: " << U_JET << '\n'
             << "                           Save steps: " << MACRO_SAVE << '\n'
             << "                           Nsteps: " << NSTEPS << '\n'
             << "                           MLUPS: " << MLUPS << '\n'
             << "--------------------------------------------------------------------------------\n";

        file.close();
        std::cout << "Simulation information file created in: " << INFO_FILE << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error generating information file: " << e.what() << std::endl;
    }
}

__host__ __forceinline__ void copyAndSaveToBinary(
    const float* d_data, size_t SIZE, const std::string& SIM_DIR, 
    const std::string& ID, int STEP, const std::string& VAR_NAME
) {
    std::vector<float> host_data(SIZE);

    checkCudaErrors(cudaMemcpy(host_data.data(), d_data, SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    std::ostringstream FILENAME;
    FILENAME << SIM_DIR << ID << "_" << VAR_NAME << std::setw(6) << std::setfill('0') << STEP << ".bin";

    std::ofstream file(FILENAME.str(), std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file " << FILENAME.str() << " for writing." << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(host_data.data()), host_data.size() * sizeof(float));
    file.close();
}

static inline constexpr unsigned div_up(unsigned n, unsigned d) {
    return (n + d - 1u) / d;
}

__host__ __forceinline__ void initDeviceVars() {
    size_t SIZE =        NX * NY * NZ          * sizeof(float);            
    size_t F_DIST_SIZE = NX * NY * NZ * FLINKS * sizeof(pop_t);
    size_t G_DIST_SIZE = NX * NY * NZ * GLINKS * sizeof(float); 

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

    #ifdef D_FIELDS
    checkCudaErrors(cudaMalloc(&dfields.vorticity_mag, SIZE));
    checkCudaErrors(cudaMalloc(&dfields.velocity_mag,  SIZE));
    checkCudaErrors(cudaMalloc(&dfields.pressure,      SIZE));
    #endif // D_FIELDS

    // initialization with cudamemset, currently using a kernel
    /*
    checkCudaErrors(cudaMemset(lbm.pxx,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.pyy,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.pzz,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.pxy,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.pxz,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.pyz,   0, SIZE));

    checkCudaErrors(cudaMemset(lbm.ux,    0, SIZE));
    checkCudaErrors(cudaMemset(lbm.uy,    0, SIZE));
    checkCudaErrors(cudaMemset(lbm.uz,    0, SIZE));

    checkCudaErrors(cudaMemset(lbm.phi,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.ffx,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.ffy,   0, SIZE));
    checkCudaErrors(cudaMemset(lbm.ffz,   0, SIZE));
    */

    checkCudaErrors(cudaMemcpyToSymbol(W,   &H_W,   FLINKS * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(W_G, &H_W_G, GLINKS * sizeof(float)));

    checkCudaErrors(cudaMemcpyToSymbol(CIX,   &H_CIX,   FLINKS * sizeof(ci_t)));
    checkCudaErrors(cudaMemcpyToSymbol(CIY,   &H_CIY,   FLINKS * sizeof(ci_t)));
    checkCudaErrors(cudaMemcpyToSymbol(CIZ,   &H_CIZ,   FLINKS * sizeof(ci_t)));

    #ifdef PERTURBATION
        checkCudaErrors(cudaMemcpyToSymbol(PERTURBATION_DATA, &H_PERTURBATION, 200 * sizeof(float)));
    #endif

    getLastCudaError("initDeviceVars: post-initialization");
}




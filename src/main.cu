#include "deviceFunctions.cuh"
#include "init.cuh"
#include "phase.cuh"
#include "lbm.cuh"
#include "bcs.cuh"
#include "../helpers/hostFunctions.cuh"

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Error: Usage: " << argv[0] << " <flow case> <velocity set> <ID>\n";
        return 1;
    }
    const std::string FLOW_CASE    = argv[1];
    const std::string VELOCITY_SET = argv[2];
    const std::string SIM_ID       = argv[3];
    const std::string SIM_DIR = createSimulationDirectory(FLOW_CASE, VELOCITY_SET, SIM_ID);

    if (setDeviceFromEnv() < 0) return 1;
    // #define BENCHMARK

    setDeviceFields();

    constexpr dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    constexpr dim3 blockX(BLOCK_SIZE_X / 2, BLOCK_SIZE_Y / 2, 1u);
    constexpr dim3 blockY(BLOCK_SIZE_X / 2, BLOCK_SIZE_Y / 2, 1u);
    constexpr dim3 blockZ(BLOCK_SIZE_X / 2, BLOCK_SIZE_Y / 2, 1u);

    constexpr dim3 grid(divUp(NX, block.x), divUp(NY, block.y), divUp(NZ, block.z));
    constexpr dim3 gridX(divUp(NY, blockX.x), divUp(NZ, blockX.y), 1u);
    constexpr dim3 gridY(divUp(NX, blockY.x), divUp(NZ, blockY.y), 1u);
    constexpr dim3 gridZ(divUp(NX, blockZ.x), divUp(NY, blockZ.y), 1u);

    constexpr size_t dynamic = 0;

    cudaStream_t queue{};
    checkCudaErrors(cudaStreamCreate(&queue));

    // =========================== INITIALIZATION =========================== //

        setFields<<<grid, block, dynamic, queue>>>(fields);

        #if defined(JET)
        setJet<<<grid, block, dynamic, queue>>>(fields);
        #elif defined(DROPLET)
        setDroplet<<<grid, block, dynamic, queue>>>(fields);
        #endif

        setDistros<<<grid, block, dynamic, queue>>>(fields);

    // ===================================================================== //

    const auto START_TIME = std::chrono::high_resolution_clock::now();
    for (int STEP = 0; STEP <= NSTEPS; ++STEP) {
        #if !defined(BENCHMARK)
        // std::cout << "Step " << STEP << " of " << NSTEPS << " started...\n";
        #endif

        // ======================== LATTICE BOLTZMANN RELATED ======================== //

            if (STEP == 0) {
                streamCollide <<<grid, block, dynamic, queue>>>(fields);
                computePhase  <<<grid, block, dynamic, queue>>>(fields);
                computeNormals<<<grid, block, dynamic, queue>>>(fields);
                computeForces <<<grid, block, dynamic, queue>>>(fields);
            } else {
                computePhase  <<<grid, block, dynamic, queue>>>(fields);
                computeNormals<<<grid, block, dynamic, queue>>>(fields);
                computeForces <<<grid, block, dynamic, queue>>>(fields);
                streamCollide <<<grid, block, dynamic, queue>>>(fields);
            }

        // ========================================================================== //

        // ============================== BOUNDARY CONDITIONS ============================== //

            #if defined(JET)
            applyInflow <<<gridZ, blockZ, dynamic, queue>>>(fields, STEP);
            applyOutflow<<<gridZ, blockZ, dynamic, queue>>>(fields);
            periodicX   <<<gridX, blockX, dynamic, queue>>>(fields);
            periodicY   <<<gridY, blockY, dynamic, queue>>>(fields);
            #elif defined(DROPLET)
            // periodicX<<<gridX, blockX, dynamic, queue>>>(fields);
            // periodicY<<<gridY, blockY, dynamic, queue>>>(fields);
            // periodicZ<<<gridZ, blockZ, dynamic, queue>>>(fields);
            #endif

        // ================================================================================= //

        #if !defined(BENCHMARK)

        checkCudaErrors(cudaDeviceSynchronize());

        if (STEP % MACRO_SAVE == 0) {

            // copyAndSaveToBinary(fields.rho, PLANE, SIM_DIR, SIM_ID, STEP, "rho");
            copyAndSaveToBinary(fields.phi, PLANE, SIM_DIR, SIM_ID, STEP, "phi");
            #if defined(JET)
            copyAndSaveToBinary(fields.uz, PLANE, SIM_DIR, SIM_ID, STEP, "uz");
            #elif defined(DROPLET)
            copyAndSaveToBinary(fields.ux, PLANE, SIM_DIR, SIM_ID, STEP, "ux");
            copyAndSaveToBinary(fields.uy, PLANE, SIM_DIR, SIM_ID, STEP, "uy");
            copyAndSaveToBinary(fields.uz, PLANE, SIM_DIR, SIM_ID, STEP, "uz");
            #endif
            std::cout << "Step " << STEP << ": bins in " << SIM_DIR << "\n";

        }

        #endif
    }

    const auto END_TIME = std::chrono::high_resolution_clock::now();
    checkCudaErrorsOutline(cudaStreamDestroy(queue));

    cudaFree(fields.f);
    cudaFree(fields.g);
    cudaFree(fields.phi);
    cudaFree(fields.rho);
    cudaFree(fields.normx);
    cudaFree(fields.normy);
    cudaFree(fields.normz);
    cudaFree(fields.ux);
    cudaFree(fields.uy);
    cudaFree(fields.uz);
    cudaFree(fields.pxx);
    cudaFree(fields.pyy);
    cudaFree(fields.pzz);
    cudaFree(fields.pxy);
    cudaFree(fields.pxz);
    cudaFree(fields.pyz);
    cudaFree(fields.ind);
    cudaFree(fields.ffx);
    cudaFree(fields.ffy);
    cudaFree(fields.ffz);

    const std::chrono::duration<double> ELAPSED_TIME = END_TIME - START_TIME;
    const uint64_t TOTAL_CELLS = static_cast<uint64_t>(NX) * NY * NZ * static_cast<uint64_t>(NSTEPS ? NSTEPS : 1);
    const double MLUPS = static_cast<double>(TOTAL_CELLS) / (ELAPSED_TIME.count() * 1e6);

    std::cout << "\n// =============================================== //\n";
    std::cout << "     Total execution time    : " << ELAPSED_TIME.count() << " s\n";
    std::cout << "     Performance             : " << MLUPS << " MLUPS\n";
    std::cout << "// =============================================== //\n\n";

    generateSimulationInfoFile(SIM_DIR, SIM_ID, VELOCITY_SET, MLUPS);
    getLastCudaErrorOutline("Final sync");

    return 0;
}

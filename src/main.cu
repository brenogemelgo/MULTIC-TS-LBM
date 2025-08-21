#include "device_header.cuh"
#include "device_functions.cuh"
#include "initial_conditions.cuh"
#include "lbm.cuh"
#include "derived_fields.cuh"
#include "boundary_conditions.cuh"
#include "host_functions.cuh"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Error: Usage: " << argv[0] << " <velocity set> <ID>\n";
        return 1;
    }
    const std::string VELOCITY_SET = argv[1];
    const std::string SIM_ID       = argv[2];
    const std::string SIM_DIR = createSimulationDirectory(VELOCITY_SET, SIM_ID);

    //computeAndPrintOccupancy(); 
    initDeviceVars();
    
    const dim3 block(32u, 2u, 2u);
    const dim3 grid (div_up(static_cast<unsigned>(NX), block.x),
                     div_up(static_cast<unsigned>(NY), block.y),
                     div_up(static_cast<unsigned>(NZ), block.z));

    const dim3 blockX(32u, 32u, 1u);
    const dim3 gridX (div_up(static_cast<unsigned>(NY), blockX.x),
                      div_up(static_cast<unsigned>(NZ), blockX.y),
                      1u);

    const dim3 blockY(32u, 32u, 1u);
    const dim3 gridY (div_up(static_cast<unsigned>(NX), blockY.x),
                      div_up(static_cast<unsigned>(NZ), blockY.y),
                      1u);

    const dim3 blockZ(32u, 32u, 1u);
    const dim3 gridZ (div_up(static_cast<unsigned>(NX), blockZ.x),
                      div_up(static_cast<unsigned>(NY), blockZ.y),
                      1u);

    const size_t shmem_bytes = DYNAMIC_SHARED_SIZE;

    cudaStream_t mainStream{};
    checkCudaErrors(cudaStreamCreate(&mainStream));

    // =========================== INITIALIZATION =========================== //

        #if defined(DROPLET_CASE)
        gpuInitDropletShape<<<grid, block, shmem_bytes, mainStream>>>(lbm);
        getLastCudaError("gpuInitDropletShape");
        #endif

        gpuInitFields<<<grid, block, shmem_bytes, mainStream>>>(lbm);
        getLastCudaError("gpuInitFields");

        #if defined(JET_CASE)
        gpuInitJetShape<<<grid, block, shmem_bytes, mainStream>>>(lbm);
        getLastCudaError("gpuInitJetShape");
        #endif

        gpuInitDistributions<<<grid, block, shmem_bytes, mainStream>>>(lbm);
        getLastCudaError("gpuInitDistributions");
    
    // ===================================================================== //

    const auto START_TIME = std::chrono::high_resolution_clock::now();
    for (int STEP = 0; STEP <= NSTEPS; ++STEP) {
        //std::cout << "Step " << STEP << " of " << NSTEPS << " started...\n";

        // ======================== NORMALS AND FORCES ======================== //

            gpuPhi<<<grid, block, shmem_bytes, mainStream>>>(lbm);
            getLastCudaError("gpuPhi");

            gpuNormals<<<grid, block, shmem_bytes, mainStream>>>(lbm);
            getLastCudaError("gpuNormals");

            gpuForces<<<grid, block, shmem_bytes, mainStream>>>(lbm);
            getLastCudaError("gpuForces");

        // =================================================================== //

        // ======================== FLUID FIELD EVOLUTION ======================== //

            gpuCollisionStream<<<grid, block, shmem_bytes, mainStream>>>(lbm);
            getLastCudaError("gpuCollisionStream");

            gpuEvolvePhaseField<<<grid, block, shmem_bytes, mainStream>>>(lbm);
            getLastCudaError("gpuEvolvePhaseField");

        // ====================================================================== //


        // ============================== BOUNDARY CONDITIONS ============================== //

            gpuApplyInflow<<<gridZ, blockZ, shmem_bytes, mainStream>>>(lbm, STEP);
            getLastCudaError("gpuApplyInflow");

            gpuApplyOutflow<<<gridZ, blockZ, shmem_bytes, mainStream>>>(lbm);
            getLastCudaError("gpuApplyOutflow");

            gpuApplyPeriodicX<<<gridX, blockX, shmem_bytes, mainStream>>>(lbm);
            getLastCudaError("gpuApplyPeriodicX");

            gpuApplyPeriodicY<<<gridY, blockY, shmem_bytes, mainStream>>>(lbm);
            getLastCudaError("gpuApplyPeriodicY");

            #ifdef D_FIELDS
            gpuDerivedFields<<<grid, block, shmem_bytes, mainStream>>>(lbm, dfields);
            getLastCudaError("gpuDerivedFields");
            #endif // D_FIELDS

        // ================================================================================= //

        //checkCudaErrors(cudaDeviceSynchronize());

        /*
        if (STEP % MACRO_SAVE == 0) {

            copyAndSaveToBinary(lbm.rho, PLANE, SIM_DIR, SIM_ID, STEP, "rho");
            copyAndSaveToBinary(lbm.phi, PLANE, SIM_DIR, SIM_ID, STEP, "phi");
            copyAndSaveToBinary(lbm.uz,  PLANE, SIM_DIR, SIM_ID, STEP, "uz");
            #ifdef D_FIELDS
            copyAndSaveToBinary(dfields.vorticity_mag, PLANE, SIM_DIR, SIM_ID, STEP, "vorticity_mag");
            copyAndSaveToBinary(dfields.velocity_mag,  PLANE, SIM_DIR, SIM_ID, STEP, "velocity_mag");
            #endif // D_FIELDS
            std::cout << "Step " << STEP << ": Binaries saved in " << SIM_DIR << "\n";

        }
            */
    }

    const auto END_TIME = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaStreamDestroy(mainStream));

    cudaFree(lbm.f);
    cudaFree(lbm.g);
    cudaFree(lbm.phi);
    cudaFree(lbm.rho);
    cudaFree(lbm.ind);
    cudaFree(lbm.normx);
    cudaFree(lbm.normy);
    cudaFree(lbm.normz);
    cudaFree(lbm.ux);
    cudaFree(lbm.uy);
    cudaFree(lbm.uz);
    cudaFree(lbm.pxx);
    cudaFree(lbm.pyy);
    cudaFree(lbm.pzz);
    cudaFree(lbm.pxy);
    cudaFree(lbm.pxz);
    cudaFree(lbm.pyz);
    cudaFree(lbm.ffx);
    cudaFree(lbm.ffy);
    cudaFree(lbm.ffz);

    #ifdef D_FIELDS
    cudaFree(dfields.vorticity_mag);
    cudaFree(dfields.velocity_mag);
    cudaFree(dfields.pressure);
    #endif // D_FIELDS

    const std::chrono::duration<double> ELAPSED_TIME = END_TIME - START_TIME;
    const uint64_t TOTAL_CELLS = static_cast<uint64_t>(NX) * NY * NZ * static_cast<uint64_t>(NSTEPS ? NSTEPS : 1);
    const double   MLUPS       = static_cast<double>(TOTAL_CELLS) / (ELAPSED_TIME.count() * 1e6);

    std::cout << "\n// =============================================== //\n";
    std::cout << "     Total execution time    : " << ELAPSED_TIME.count() << " s\n";
    std::cout << "     Performance             : " << MLUPS << " MLUPS\n";
    std::cout << "// =============================================== //\n\n";

    generateSimulationInfoFile(SIM_DIR, SIM_ID, VELOCITY_SET, NSTEPS, MACRO_SAVE, TAU, MLUPS);
    getLastCudaError("Final sync");

    return 0;
}

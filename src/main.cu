#include "deviceUtils.cuh"
#include "init.cuh"
#include "lbm.cuh"
#include "bcs.cuh"
#include "../include/hostFunctions.cuh"
#if defined(D_FIELDS)
#include "../include/derivedFields.cuh"
#endif 

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Error: Usage: " << argv[0] << " <flow case> <velocity set> <ID>\n";
        return 1;
    }
    const std::string FLOW_CASE    = argv[1];
    const std::string VELOCITY_SET = argv[2];
    const std::string SIM_ID       = argv[3];
    const std::string SIM_DIR = createSimulationDirectory(FLOW_CASE, VELOCITY_SET, SIM_ID);

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

    const size_t dynamic = 0;

    //#define BENCHMARK

    cudaStream_t queue{};
    checkCudaErrors(cudaStreamCreate(&queue));

    // =========================== INITIALIZATION =========================== //

        setFields <<<grid, block, dynamic, queue>>>(lbm);

        #if defined(JET)
        setJet<<<grid, block, dynamic, queue>>>(lbm);
        #elif defined(DROPLET)
        setDroplet<<<grid, block, dynamic, queue>>>(lbm);
        #endif

        setDistros<<<grid, block, dynamic, queue>>>(lbm);
    
    // ===================================================================== //

    const auto START_TIME = std::chrono::high_resolution_clock::now();
    for (int STEP = 0; STEP <= NSTEPS; ++STEP) {
        #if !defined(BENCHMARK)
        std::cout << "Step " << STEP << " of " << NSTEPS << " started...\n";
        #endif

        // ======================== NORMALS AND FORCES ======================== //

            computePhase  <<<grid, block, dynamic, queue>>>(lbm);
            computeNormals<<<grid, block, dynamic, queue>>>(lbm);
            computeForces <<<grid, block, dynamic, queue>>>(lbm);

        // =================================================================== //

        // ======================== FLUID FIELD EVOLUTION ======================== //

            streamCollide<<<grid, block, dynamic, queue>>>(lbm);
            advectDiffuse<<<grid, block, dynamic, queue>>>(lbm);

        // ====================================================================== //


        // ============================== BOUNDARY CONDITIONS ============================== //
        
            #if defined(JET)
            applyInflow <<<gridZ, blockZ, dynamic, queue>>>(lbm, STEP);
            applyOutflow<<<gridZ, blockZ, dynamic, queue>>>(lbm);
            periodicX   <<<gridX, blockX, dynamic, queue>>>(lbm);
            periodicY   <<<gridY, blockY, dynamic, queue>>>(lbm);
            #elif defined(DROPLET)
            // undefined
            #endif

        // ================================================================================= //

        #if defined(D_FIELDS)
        gpuDerivedFields<<<grid, block, dynamic, queue>>>(lbm, dfields);
        #endif 

        #if !defined(BENCHMARK)

        checkCudaErrors(cudaDeviceSynchronize());

        if (STEP % MACRO_SAVE == 0) {

            copyAndSaveToBinary(lbm.rho, PLANE, SIM_DIR, SIM_ID, STEP, "rho");
            copyAndSaveToBinary(lbm.phi, PLANE, SIM_DIR, SIM_ID, STEP, "phi");
            #if defined(JET)
            copyAndSaveToBinary(lbm.uz,  PLANE, SIM_DIR, SIM_ID, STEP, "uz");
            #elif defined(DROPLET)
            copyAndSaveToBinary(lbm.ux,  PLANE, SIM_DIR, SIM_ID, STEP, "ux");
            copyAndSaveToBinary(lbm.uy,  PLANE, SIM_DIR, SIM_ID, STEP, "uy");
            copyAndSaveToBinary(lbm.uz,  PLANE, SIM_DIR, SIM_ID, STEP, "uz");
            #endif
            #if defined(D_FIELDS)
            copyAndSaveToBinary(dfields.vorticity_mag, PLANE, SIM_DIR, SIM_ID, STEP, "vorticity_mag");
            copyAndSaveToBinary(dfields.velocity_mag,  PLANE, SIM_DIR, SIM_ID, STEP, "velocity_mag");
            #endif 
            std::cout << "Step " << STEP << ": Binaries saved in " << SIM_DIR << "\n";

        }

        #endif
    }

    const auto END_TIME = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaStreamDestroy(queue));

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

    #if defined(D_FIELDS)
    cudaFree(dfields.vorticity_mag);
    cudaFree(dfields.velocity_mag);
    #endif 

    const std::chrono::duration<double> ELAPSED_TIME = END_TIME - START_TIME;
    const uint64_t TOTAL_CELLS = static_cast<uint64_t>(NX) * NY * NZ * static_cast<uint64_t>(NSTEPS ? NSTEPS : 1);
    const double   MLUPS       = static_cast<double>(TOTAL_CELLS) / (ELAPSED_TIME.count() * 1e6);

    std::cout << "\n// =============================================== //\n";
    std::cout << "     Total execution time    : " << ELAPSED_TIME.count() << " s\n";
    std::cout << "     Performance             : " << MLUPS << " MLUPS\n";
    std::cout << "// =============================================== //\n\n";

    generateSimulationInfoFile(SIM_DIR, SIM_ID, VELOCITY_SET, MLUPS);
    getLastCudaError("Final sync");

    return 0;
}

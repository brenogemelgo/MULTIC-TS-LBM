#include "deviceFunctions.cuh"
#include "init.cuh"
#include "phase.cuh"
#include "lbm.cuh"
#include "bcs.cuh"
#include "../helpers/hostFunctions.cuh"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Error: Usage: " << argv[0] << " <flow case> <velocity set> <ID>\n";
        return 1;
    }
    const std::string FLOW_CASE    = argv[1];
    const std::string VELOCITY_SET = argv[2];
    const std::string SIM_ID       = argv[3];
    const std::string SIM_DIR = createSimulationDirectory(FLOW_CASE, VELOCITY_SET, SIM_ID);

    if (setDeviceFromEnv() < 0) return 1;
    //#define BENCHMARK

    setDeviceFields();

    const dim3 block3D(block::nx, block::ny, block::nz);

    const dim3 grid3D(divUp(mesh::nx, block3D.x), 
                      divUp(mesh::ny, block3D.y), 
                      divUp(mesh::nz, block3D.z));

    const dim3 blockX(block::nx, block::ny, 1u);
    const dim3 blockY(block::nx, block::ny, 1u);
    const dim3 blockZ(block::nx, block::ny, 1u);
    
    const dim3 gridX(divUp(mesh::ny, blockX.x), divUp(mesh::nz, blockX.y), 1u);
    const dim3 gridY(divUp(mesh::nx, blockY.x), divUp(mesh::nz, blockY.y), 1u);
    const dim3 gridZ(divUp(mesh::nx, blockZ.x), divUp(mesh::ny, blockZ.y), 1u);

    constexpr size_t dynamic = 0;

    cudaStream_t queue{};
    checkCudaErrors(cudaStreamCreate(&queue));

    // =========================== INITIALIZATION =========================== //

        setFields<<<grid3D, block3D, dynamic, queue>>>(fields);

        #if defined(JET)

            setJet<<<grid3D, block3D, dynamic, queue>>>(fields);

        #elif defined(DROPLET)

            setDroplet<<<grid3D, block3D, dynamic, queue>>>(fields);

        #endif

        setDistros<<<grid3D, block3D, dynamic, queue>>>(fields);

    // ===================================================================== //

    checkCudaErrors(cudaDeviceSynchronize());
    const auto START_TIME = std::chrono::high_resolution_clock::now();
    for (idx_t STEP = 0; STEP <= NSTEPS; ++STEP) {

        #if !defined(BENCHMARK)

            // std::cout << "Step " << STEP << " of " << NSTEPS << " started...\n";

        #endif

        // ======================== LATTICE BOLTZMANN RELATED ======================== //

            computePhase  <<<grid3D, block3D, dynamic, queue>>>(fields);
            computeNormals<<<grid3D, block3D, dynamic, queue>>>(fields);
            computeForces <<<grid3D, block3D, dynamic, queue>>>(fields);
            streamCollide <<<grid3D, block3D, dynamic, queue>>>(fields);
            
        // ========================================================================== //

        // ============================== BOUNDARY CONDITIONS ============================== //

            #if defined(JET)

                applyInflow <<<gridZ, blockZ, dynamic, queue>>>(fields);
                applyOutflow<<<gridZ, blockZ, dynamic, queue>>>(fields);
                periodicX   <<<gridX, blockX, dynamic, queue>>>(fields);
                periodicY   <<<gridY, blockY, dynamic, queue>>>(fields);

            #elif defined(DROPLET)

                // undefined

            #endif

        // ================================================================================= //

        #if !defined(BENCHMARK)

            checkCudaErrors(cudaDeviceSynchronize());

            if (STEP % MACRO_SAVE == 0) {

                copyAndSaveToBinary(fields.rho, PLANE, SIM_DIR, SIM_ID, STEP, "rho");
                copyAndSaveToBinary(fields.phi, PLANE, SIM_DIR, SIM_ID, STEP, "phi");

                #if defined(JET)

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
    const uint64_t TOTAL_CELLS = static_cast<uint64_t>(mesh::nx) * mesh::ny * mesh::nz * static_cast<uint64_t>(NSTEPS ? NSTEPS : 1);
    const double MLUPS = static_cast<double>(TOTAL_CELLS) / (ELAPSED_TIME.count() * 1e6);

    std::cout << "\n// =============================================== //\n";
    std::cout << "     Total execution time    : " << ELAPSED_TIME.count() << " s\n";
    std::cout << "     Performance             : " << MLUPS << " MLUPS\n";
    std::cout << "// =============================================== //\n\n";

    generateSimulationInfoFile(SIM_DIR, SIM_ID, VELOCITY_SET, MLUPS);
    getLastCudaErrorOutline("Final sync");

    return 0;
}

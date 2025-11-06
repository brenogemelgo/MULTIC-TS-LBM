#include "../helpers/deviceFunctions.cuh"
#include "setupFields.cuh"
#include "interface.cuh"
#include "lbm.cuh"
#include "boundaryConditions.cuh"
#include "../helpers/hostFunctions.cuh"

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Error: Usage: " << argv[0] << " <flow case> <velocity set> <ID>\n";
        return 1;
    }
    const std::string FLOW_CASE = argv[1];
    const std::string VELOCITY_SET = argv[2];
    const std::string SIM_ID = argv[3];
    const std::string SIM_DIR = host::createSimulationDirectory(FLOW_CASE, VELOCITY_SET, SIM_ID);

// Benchmark define (suppresses saves and step outputs)
#define BENCHMARK

    // Set GPU based on pipeline argument
    if (host::setDeviceFromEnv() < 0)
        return 1;

    // Initialize device arrays
    host::setDeviceFields();

    // Block-wise configuration
    const dim3 block3D(block::nx, block::ny, block::nz);

    const dim3 grid3D(host::divUp(mesh::nx, block3D.x),
                      host::divUp(mesh::ny, block3D.y),
                      host::divUp(mesh::nz, block3D.z));

    const dim3 blockX(block::nx, block::ny, 1u);
    const dim3 blockY(block::nx, block::ny, 1u);
    const dim3 blockZ(block::nx, block::ny, 1u);

    const dim3 gridX(host::divUp(mesh::ny, blockX.x), host::divUp(mesh::nz, blockX.y), 1u);
    const dim3 gridY(host::divUp(mesh::nx, blockY.x), host::divUp(mesh::nz, blockY.y), 1u);
    const dim3 gridZ(host::divUp(mesh::nx, blockZ.x), host::divUp(mesh::ny, blockZ.y), 1u);

    // Dynamic shared memory size
    constexpr size_t dynamic = 0;

    // Create stream
    cudaStream_t queue{};
    checkCudaErrorsOutline(cudaStreamCreate(&queue));

    // Call initialization kernels
    setFields<<<grid3D, block3D, dynamic, queue>>>(fields);

#if defined(JET)

    setJet<<<grid3D, block3D, dynamic, queue>>>(fields);

#elif defined(DROPLET)

    setDroplet<<<grid3D, block3D, dynamic, queue>>>(fields);

#endif

    setDistros<<<grid3D, block3D, dynamic, queue>>>(fields);

    // Make sure all initialization kernels have finished
    checkCudaErrorsOutline(cudaDeviceSynchronize());

    // Time loop
    const auto START_TIME = std::chrono::high_resolution_clock::now();
    for (label_t STEP = 0; STEP <= NSTEPS; ++STEP)
    {
        // Calculate the phase field
        phase::computePhase<<<grid3D, block3D, dynamic, queue>>>(fields);

        // Calculate interface normals
        phase::computeNormals<<<grid3D, block3D, dynamic, queue>>>(fields);

        // Calculate surface tension forces
        phase::computeForces<<<grid3D, block3D, dynamic, queue>>>(fields);

        // Main kernel
        lbm::streamCollide<<<grid3D, block3D, dynamic, queue>>>(fields);

        // Call boundary conditions
#if defined(JET)

        lbm::applyInflow<<<gridZ, blockZ, dynamic, queue>>>(fields);
        lbm::applyOutflow<<<gridZ, blockZ, dynamic, queue>>>(fields);
        lbm::periodicX<<<gridX, blockX, dynamic, queue>>>(fields);
        lbm::periodicY<<<gridY, blockY, dynamic, queue>>>(fields);

#elif defined(DROPLET)

#endif

#if !defined(BENCHMARK)

        // Sync kernels
        checkCudaErrors(cudaDeviceSynchronize());

        // Copy arrays to host
        if (STEP % MACRO_SAVE == 0)
        {

            host::copyAndSaveToBinary(fields.rho, SIM_DIR, STEP, "rho");
            host::copyAndSaveToBinary(fields.ux, SIM_DIR, STEP, "ux");
            host::copyAndSaveToBinary(fields.uy, SIM_DIR, STEP, "uy");
            host::copyAndSaveToBinary(fields.uz, SIM_DIR, STEP, "uz");
            host::copyAndSaveToBinary(fields.phi, SIM_DIR, STEP, "phi");

            // Print step
            std::cout << "Step " << STEP << ": bins in " << SIM_DIR << "\n";
        }

#endif
    }
    const auto END_TIME = std::chrono::high_resolution_clock::now();

    // Destructors
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

    // Performance log
    const std::chrono::duration<double> ELAPSED_TIME = END_TIME - START_TIME;
    const label_t TOTAL_CELLS = static_cast<label_t>(mesh::nx) * mesh::ny * mesh::nz * static_cast<label_t>(NSTEPS ? NSTEPS : 1);
    const double MLUPS = static_cast<double>(TOTAL_CELLS) / (ELAPSED_TIME.count() * 1e6);

    std::cout << "\n// =============================================== //\n";
    std::cout << "     Total execution time    : " << ELAPSED_TIME.count() << " s\n";
    std::cout << "     Performance             : " << MLUPS << " MLUPS\n";
    std::cout << "// =============================================== //\n\n";

    // Generate info
    host::generateSimulationInfoFile(SIM_DIR, SIM_ID, VELOCITY_SET, MLUPS);
    getLastCudaErrorOutline("Final sync");

    return 0;
}

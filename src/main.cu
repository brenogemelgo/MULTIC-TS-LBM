/*---------------------------------------------------------------------------*\
|                                                                             |
| MULTIC-TS-LBM: CUDA-based multicomponent Lattice Boltzmann Method           |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/brenogemelgo/MULTIC-TS-LBM                       |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Breno Gemelgo (Geoenergia Lab, UDESC)

License
    This file is part of MULTIC-TS-LBM.

    MULTIC-TS-LBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    Main program file

SourceFiles
    main.cu

\*---------------------------------------------------------------------------*/

#include "functions/deviceFunctions.cuh"
#include "functions/hostFunctions.cuh"
#include "class/initialConditions.cuh"
#include "class/boundaryConditions.cuh"
#include "phaseField/phaseField.cuh"
#include "caller/caller.cuh"
#include "derivedFields/timeAverage.cuh"

#include "lbm.cuh"

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

    // Set GPU based on pipeline argument
    if (host::setDeviceFromEnv() < 0)
    {
        return 1;
    }

    // Initialize device arrays
    host::setDeviceFields();

    // Block-wise configuration
    constexpr dim3 block3D(block::nx, block::ny, block::nz);

    constexpr dim3 grid3D(host::divUp(mesh::nx, block3D.x),
                          host::divUp(mesh::ny, block3D.y),
                          host::divUp(mesh::nz, block3D.z));

    constexpr dim3 blockX(block::nx, block::ny, 1u);
    constexpr dim3 blockY(block::nx, block::ny, 1u);
    constexpr dim3 blockZ(block::nx, block::ny, 1u);

    constexpr dim3 gridX(host::divUp(mesh::ny, blockX.x), host::divUp(mesh::nz, blockX.y), 1u);
    constexpr dim3 gridY(host::divUp(mesh::nx, blockY.x), host::divUp(mesh::nz, blockY.y), 1u);
    constexpr dim3 gridZ(host::divUp(mesh::nx, blockZ.x), host::divUp(mesh::ny, blockZ.y), 1u);

    // Dynamic shared memory size
    constexpr size_t dynamic = 0;

    // Create stream
    cudaStream_t queue{};
    checkCudaErrorsOutline(cudaStreamCreate(&queue));

    // Call initialization kernels
    LBM::callSetFields<<<grid3D, block3D, dynamic, queue>>>(fields);

#if defined(JET)
    LBM::callSetJet<<<grid3D, block3D, dynamic, queue>>>(fields);
#elif defined(DROPLET)
    LBM::callSetDroplet<<<grid3D, block3D, dynamic, queue>>>(fields);
#endif

    LBM::callSetDistros<<<grid3D, block3D, dynamic, queue>>>(fields);

    // Make sure all initialization kernels have finished
    checkCudaErrorsOutline(cudaDeviceSynchronize());

    // Time loop
    const auto START_TIME = std::chrono::high_resolution_clock::now();
    for (label_t STEP = 0; STEP <= NSTEPS; ++STEP)
    {
        phase::computePhase<<<grid3D, block3D, dynamic, queue>>>(fields);
        phase::computeNormals<<<grid3D, block3D, dynamic, queue>>>(fields);
        phase::computeForces<<<grid3D, block3D, dynamic, queue>>>(fields);

        LBM::computeMoments<<<grid3D, block3D, dynamic, queue>>>(fields);
        LBM::streamCollide<<<grid3D, block3D, dynamic, queue>>>(fields);

#if defined(JET)

        LBM::callInflow<<<gridZ, blockZ, dynamic, queue>>>(fields);
        LBM::callOutflow<<<gridZ, blockZ, dynamic, queue>>>(fields);
        LBM::callPeriodicX<<<gridX, blockX, dynamic, queue>>>(fields);
        LBM::callPeriodicY<<<gridY, blockY, dynamic, queue>>>(fields);

#elif defined(DROPLET)

#endif

#if AVERAGE_UZ

        LBM::timeAverage<<<grid3D, block3D, dynamic, queue>>>(fields, STEP + 1);

#endif

#if !BENCHMARK

        // Sync kernels
        checkCudaErrors(cudaDeviceSynchronize());

        // Copy arrays to host
        if (STEP % MACRO_SAVE == 0)
        {

            host::copyAndSaveToBinary(fields.rho, SIM_DIR, STEP, "rho");
            host::copyAndSaveToBinary(fields.phi, SIM_DIR, STEP, "phi");
            host::copyAndSaveToBinary(fields.uz, SIM_DIR, STEP, "uz");

#if AVERAGE_UZ

            host::copyAndSaveToBinary(fields.avg, SIM_DIR, STEP, "avg");

#endif

            // Print step
            std::cout << "Step " << STEP << ": bins in " << SIM_DIR << "\n";
        }
#endif
    }

    checkCudaErrorsOutline(cudaStreamSynchronize(queue));
    const auto END_TIME = std::chrono::high_resolution_clock::now();

    // Destructors
    checkCudaErrorsOutline(cudaStreamDestroy(queue));

    // Distributions
    cudaFree(fields.f);
    cudaFree(fields.g);

    // Hydrodynamic fields
    cudaFree(fields.rho);
    cudaFree(fields.ux);
    cudaFree(fields.uy);
    cudaFree(fields.uz);
    cudaFree(fields.pxx);
    cudaFree(fields.pyy);
    cudaFree(fields.pzz);
    cudaFree(fields.pxy);
    cudaFree(fields.pxz);
    cudaFree(fields.pyz);

    // Interface fields
    cudaFree(fields.phi);
    cudaFree(fields.normx);
    cudaFree(fields.normy);
    cudaFree(fields.normz);
    cudaFree(fields.ind);
    cudaFree(fields.ffx);
    cudaFree(fields.ffy);
    cudaFree(fields.ffz);

    // Derived fields
#if AVERAGE_UZ

    cudaFree(fields.avg);

#endif

    // Performance log
    const std::chrono::duration<double> ELAPSED_TIME = END_TIME - START_TIME;

    const double steps = static_cast<double>(NSTEPS + 1);
    const double total_lattice_updates = static_cast<double>(mesh::nx) * mesh::ny * mesh::nz * steps;
    const double MLUPS = total_lattice_updates / (ELAPSED_TIME.count() * 1e6);

    std::cout << "\n// =============================================== //\n";
    std::cout << "     Total execution time    : " << ELAPSED_TIME.count() << " s\n";
    std::cout << "     Performance             : " << MLUPS << " MLUPS\n";
    std::cout << "// =============================================== //\n\n";

    // Generate info
    host::generateSimulationInfoFile(SIM_DIR, SIM_ID, VELOCITY_SET, MLUPS);
    getLastCudaErrorOutline("Final sync");

    return 0;
}

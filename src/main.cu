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

#include "functions/deviceUtils.cuh"
#include "functions/hostUtils.cuh"
#include "functions/ioFields.cuh"
#include "functions/vtkWriter.cuh"
#include "initialConditions.cuh"
#include "boundaryConditions.cuh"
#include "phaseField.cuh"
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

    if (host::setDeviceFromEnv() < 0)
    {
        return 1;
    }

    host::setDeviceFields();

    constexpr dim3 block3D(block::nx, block::ny, block::nz);

    constexpr dim3 grid3D(host::divUp(mesh::nx, block3D.x),
                          host::divUp(mesh::ny, block3D.y),
                          host::divUp(mesh::nz, block3D.z));

    constexpr dim3 blockZ(block::nx, block::ny, 1u);

    constexpr dim3 gridZ(host::divUp(mesh::nx, blockZ.x), host::divUp(mesh::ny, blockZ.y), 1u);

    constexpr size_t dynamic = 0;

    cudaStream_t queue{};
    checkCudaErrorsOutline(cudaStreamCreate(&queue));

    LBM::setFields<<<grid3D, block3D, dynamic, queue>>>(fields);

#if defined(JET)
    LBM::setJet<<<grid3D, block3D, dynamic, queue>>>(fields);
#elif defined(DROPLET)
    LBM::setDroplet<<<grid3D, block3D, dynamic, queue>>>(fields);
#endif

    LBM::setDistros<<<grid3D, block3D, dynamic, queue>>>(fields);

    checkCudaErrorsOutline(cudaDeviceSynchronize());

    host::generateSimulationInfoFile(SIM_DIR, SIM_ID, VELOCITY_SET);

#if !BENCHMARK

    std::vector<std::thread> vtk_threads;
    vtk_threads.reserve(NSTEPS / MACRO_SAVE + 2);

    constexpr std::array<host::FieldConfig, 3> OUTPUT_FIELDS{
        {{host::FieldID::Rho, "rho", host::FieldDumpShape::Grid3D, true},
         {host::FieldID::Phi, "phi", host::FieldDumpShape::Grid3D, true},
         {host::FieldID::Uz, "uz", host::FieldDumpShape::Grid3D, true}}};

#endif

    cudaDeviceSynchronize();

    const auto START_TIME = std::chrono::high_resolution_clock::now();
    for (label_t STEP = 0; STEP <= NSTEPS; ++STEP)
    {
        phase::computeNormals<<<grid3D, block3D, dynamic, queue>>>(fields);
        phase::computeForces<<<grid3D, block3D, dynamic, queue>>>(fields);

        LBM::computeMoments<<<grid3D, block3D, dynamic, queue>>>(fields);
        LBM::streamCollide<<<grid3D, block3D, dynamic, queue>>>(fields);

#if defined(JET)

        LBM::callInflow<<<gridZ, blockZ, dynamic, queue>>>(fields, STEP);
        LBM::callOutflow<<<gridZ, blockZ, dynamic, queue>>>(fields);

#elif defined(DROPLET)

#endif

#if AVERAGE_UZ

        LBM::timeAverage<<<grid3D, block3D, dynamic, queue>>>(fields, STEP + 1);

#endif

#if !BENCHMARK

        const bool isOutputStep = (STEP % MACRO_SAVE == 0) || (STEP == NSTEPS);

        if (isOutputStep)
        {
            checkCudaErrors(cudaStreamSynchronize(queue));

            const auto step_copy = STEP;

            host::saveConfiguredFields(OUTPUT_FIELDS, SIM_DIR, step_copy);

            vtk_threads.emplace_back(
                [step_copy,
                 fieldsCfg = OUTPUT_FIELDS,
                 sim_dir = SIM_DIR,
                 sim_id = SIM_ID]
                {
                    host::writeStructuredGrid(fieldsCfg, sim_dir, sim_id, step_copy);
                });

            std::cout << "Step " << STEP << ": bins in " << SIM_DIR << "\n";
        }

#endif
    }

#if !BENCHMARK

    for (auto &t : vtk_threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

#endif

    cudaStreamSynchronize(queue);
    const auto END_TIME = std::chrono::high_resolution_clock::now();
    checkCudaErrorsOutline(cudaStreamDestroy(queue));

    cudaFree(fields.f);
    cudaFree(fields.g);
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
    cudaFree(fields.phi);
    cudaFree(fields.normx);
    cudaFree(fields.normy);
    cudaFree(fields.normz);
    cudaFree(fields.ind);
    cudaFree(fields.ffx);
    cudaFree(fields.ffy);
    cudaFree(fields.ffz);

#if AVERAGE_UZ

    cudaFree(fields.avg);

#endif

    const std::chrono::duration<double> ELAPSED_TIME = END_TIME - START_TIME;

    const double steps = static_cast<double>(NSTEPS + 1);
    const double total_lattice_updates = static_cast<double>(mesh::nx) * mesh::ny * mesh::nz * steps;
    const double MLUPS = total_lattice_updates / (ELAPSED_TIME.count() * 1e6);

    std::cout << "\n// =============================================== //\n";
    std::cout << "     Total execution time    : " << ELAPSED_TIME.count() << " s\n";
    std::cout << "     Performance             : " << MLUPS << " MLUPS\n";
    std::cout << "// =============================================== //\n\n";

    getLastCudaErrorOutline("Final sync");

    return 0;
}

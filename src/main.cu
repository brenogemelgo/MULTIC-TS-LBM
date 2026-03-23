/*---------------------------------------------------------------------------*\
|                                                                             |
| phaseFieldLBM: CUDA-based multicomponent Lattice Boltzmann Method           |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/brenogemelgo/phaseFieldLBM                       |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Breno Gemelgo (Geoenergia Lab, UDESC)

Description
    Main driver orchestrating runtime case loading, CUDA Graph execution, time stepping, boundary handling, and post-processing output

SourceFiles
    main.cu

\*---------------------------------------------------------------------------*/

#include "runtime/RuntimeConfig.cuh"
#include "runtime/RuntimeState.cuh"
#include "velocitySet/VelocitySet.cuh"
#include "cases/CaseFactory.cuh"

#include "functions/deviceFunctions.cuh"
#include "fieldAllocate/FieldAllocate.cuh"
#include "functions/hostFunctions.cuh"
#include "fileIO/fields.cuh"
#include "postProcess/PostProcess.cuh"
#include "initialConditions.cu"
#include "BoundaryConditions.cuh"
#include "phaseField.cuh"
#include "lbm.cu"
#include "cases/CasePolicies.cuh"
#include "cuda/CUDAGraph.cuh"
#include "derivedFields/DerivedFields.cuh"

runtime::DeviceConstants runtime::HOST_CONSTANTS{};
__device__ __constant__ runtime::DeviceConstants runtime::DEVICE_CONSTANTS;

namespace
{
    [[nodiscard]] runtime::DeviceConstants makeDeviceConstants(
        const runtime::RuntimeConfig &cfg,
        const cases::Case &caseObject)
    {
        runtime::DeviceConstants constants{};

        constants.nx = cfg.mesh.nx;
        constants.ny = cfg.mesh.ny;
        constants.nz = cfg.mesh.nz;

        constants.stride = cfg.mesh.nx * cfg.mesh.ny;
        constants.cells = constants.stride * cfg.mesh.nz;

        constants.u_inf = cfg.control.u_inf;
        constants.reynolds_water = cfg.control.ReA;
        constants.reynolds_oil = cfg.control.ReB;
        constants.weber = cfg.control.We;

        constants.nTimeSteps = cfg.control.nTimeSteps;
        constants.saveInterval = cfg.control.saveInterval;

        caseObject.applyCaseConstants(constants, cfg);

        if (constants.cells == 0)
        {
            throw std::runtime_error("Invalid runtime mesh dimensions: number of cells is zero.");
        }

        return constants;
    }

    template <typename VelocitySet, typename CasePolicy>
    int runSimulation(
        const runtime::RuntimeConfig &cfg,
        const runtime::CliOptions &cli,
        const std::string &simDir)
    {
        // Device 3D fields
        const auto scalar = std::to_array<host::FieldDescription<scalar_t>>({
            {"rho", &LBMFields::rho, host::bytesScalar(), true},
            {"ux", &LBMFields::ux, host::bytesScalar(), true},
            {"uy", &LBMFields::uy, host::bytesScalar(), true},
            {"uz", &LBMFields::uz, host::bytesScalar(), true},
            {"pxx", &LBMFields::pxx, host::bytesScalar(), true},
            {"pyy", &LBMFields::pyy, host::bytesScalar(), true},
            {"pzz", &LBMFields::pzz, host::bytesScalar(), true},
            {"pxy", &LBMFields::pxy, host::bytesScalar(), true},
            {"pxz", &LBMFields::pxz, host::bytesScalar(), true},
            {"pyz", &LBMFields::pyz, host::bytesScalar(), true},
            {"phi", &LBMFields::phi, host::bytesScalar(), true},
            {"normx", &LBMFields::normx, host::bytesScalar(), true},
            {"normy", &LBMFields::normy, host::bytesScalar(), true},
            {"normz", &LBMFields::normz, host::bytesScalar(), true},
            {"ind", &LBMFields::ind, host::bytesScalar(), true},
            {"fsx", &LBMFields::fsx, host::bytesScalar(), true},
            {"fsy", &LBMFields::fsy, host::bytesScalar(), true},
            {"fsz", &LBMFields::fsz, host::bytesScalar(), true},
        });

        // Device distribution functions
        const host::FieldDescription<pop_t> f = {"f", &LBMFields::f, host::bytesF<VelocitySet>(), true};
        const host::FieldDescription<scalar_t> g = {"g", &LBMFields::g, host::bytesG(), true};

        // Allocate all device fields
        host::FieldAllocate baseOwner(fields, scalar, f, g);

        // Construct derived fields
        derived::DerivedFields dfields;
        dfields.allocate(fields);

        // Block-wise configuration
        const dim3 block3D(block::nx(), block::ny(), block::nz());
        const dim3 grid3D(host::divUp(mesh::nx(), block3D.x),
                          host::divUp(mesh::ny(), block3D.y),
                          host::divUp(mesh::nz(), block3D.z));

        // Periodic x-direction
        const dim3 blockX(block::ny(), block::nz(), 1u);
        const dim3 gridX(host::divUp(mesh::ny(), blockX.x), host::divUp(mesh::nz(), blockX.y), 1u);

        // Periodic y-direction
        const dim3 blockY(block::nx(), block::nz(), 1u);
        const dim3 gridY(host::divUp(mesh::nx(), blockY.x), host::divUp(mesh::nz(), blockY.y), 1u);

        // Inlet and outlet
        const dim3 blockZ(block::nx(), block::ny(), 1u);
        const dim3 gridZ(host::divUp(mesh::nx(), blockZ.x), host::divUp(mesh::ny(), blockZ.y), 1u);

        // Dynamic shared memory size
        constexpr size_t dynamic = 0;

        // Stream setup
        cudaStream_t queue{};
        checkCudaErrorsOutline(cudaStreamCreate(&queue));

        // Initial conditions
        lbm::setInitialDensity<<<grid3D, block3D, dynamic, queue>>>(fields);
        CasePolicy::template launchInitial<VelocitySet>(grid3D, block3D, dynamic, fields, queue);
        lbm::setDistros<VelocitySet><<<grid3D, block3D, dynamic, queue>>>(fields);

        // Make sure everything is initialized
        checkCudaErrorsOutline(cudaDeviceSynchronize());

        // Generate info file and print diagnostics
        host::generateSimulationInfoFile(simDir, cli.simulationId, runtime::stencilToString(cli.stencil), cfg);
        host::printDiagnostics(runtime::stencilToString(cli.stencil));

#if !BENCHMARK
        // Post-processing instance
        host::PostProcess write;

        // Base fields (always saved)
        static constexpr auto BASE_FIELDS = std::to_array<host::FieldConfig>({
            {host::FieldID::Rho, "rho", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Uz, "uz", host::FieldDumpShape::Grid3D, true},
        });

        // Derived fields from modules (possibly empty)
        const auto DERIVED_FIELDS = dfields.makeOutputFields();

        // Compose final list in a vector
        std::vector<host::FieldConfig> OUTPUT_FIELDS;
        OUTPUT_FIELDS.reserve(BASE_FIELDS.size() + DERIVED_FIELDS.size());
        OUTPUT_FIELDS.insert(OUTPUT_FIELDS.end(), BASE_FIELDS.begin(), BASE_FIELDS.end());
        OUTPUT_FIELDS.insert(OUTPUT_FIELDS.end(), DERIVED_FIELDS.begin(), DERIVED_FIELDS.end());

        // Ensure post-processing only targets full 3D fields
        for (auto &cfgField : OUTPUT_FIELDS)
        {
            cfgField.includeInPost = (cfgField.shape == host::FieldDumpShape::Grid3D);
        }
#endif

        // Warmup (optional)
        checkCudaErrorsOutline(cudaDeviceSynchronize());

        // Build CUDA Graph
        cudaGraph_t graph{};
        cudaGraphExec_t graphExec{};
        graph::captureGraph<VelocitySet, CasePolicy>(graph, graphExec, fields, queue, grid3D, block3D, dynamic);

        // Start clock
        const auto START_TIME = std::chrono::high_resolution_clock::now();

        // Time loop
        for (label_t STEP = 0; STEP <= control::nTimeSteps(); ++STEP)
        {
            // Launch captured sequence
            cudaGraphLaunch(graphExec, queue);

            // Flow case specific boundary conditions
            CasePolicy::template launchBoundary<VelocitySet>(fields, queue, STEP, gridX, blockX, gridY, blockY, gridZ, blockZ, dynamic);

            // Ensure debug output is complete before host logic
            cudaStreamSynchronize(queue);

            // Derived fields
            dfields.launch(queue, grid3D, block3D, dynamic, fields, STEP);

#if !BENCHMARK
            const bool isOutputStep = (STEP % control::saveInterval() == 0) || (STEP == control::nTimeSteps());

            if (isOutputStep)
            {
                checkCudaErrors(cudaStreamSynchronize(queue));
                std::cout << "Step " << STEP << ": bins in " << simDir << "\n";
                write.bin(OUTPUT_FIELDS, simDir, STEP, fields);
                write.vti(OUTPUT_FIELDS, simDir, STEP);
            }
#endif
        }

        // Make sure everything is done on the GPU
        cudaStreamSynchronize(queue);
        const auto END_TIME = std::chrono::high_resolution_clock::now();

        // Destroy CUDA Graph resources
        checkCudaErrorsOutline(cudaGraphExecDestroy(graphExec));
        checkCudaErrorsOutline(cudaGraphDestroy(graph));

        // Destroy stream
        checkCudaErrorsOutline(cudaStreamDestroy(queue));

        const std::chrono::duration<double> ELAPSED_TIME = END_TIME - START_TIME;

        const double steps = static_cast<double>(control::nTimeSteps() + 1);
        const double total_lattice_updates = static_cast<double>(mesh::nx()) * static_cast<double>(mesh::ny()) * static_cast<double>(mesh::nz()) * steps;
        const double MLUPS = total_lattice_updates / (ELAPSED_TIME.count() * 1e6);

        std::cout << "\n// =============================================== //\n";
        std::cout << "     Total execution time    : " << ELAPSED_TIME.count() << " s\n";
        std::cout << "     Performance             : " << MLUPS << " MLUPS\n";
        std::cout << "// =============================================== //\n\n";

        getLastCudaErrorOutline("Final sync");

        return 0;
    }

    template <typename VelocitySet>
    int dispatchCase(
        const runtime::CaseKind caseKind,
        const runtime::RuntimeConfig &cfg,
        const runtime::CliOptions &cli,
        const std::string &simDir)
    {
        switch (caseKind)
        {
        case runtime::CaseKind::Jet:
            return runSimulation<VelocitySet, case_policy::Jet>(cfg, cli, simDir);

        case runtime::CaseKind::Droplet:
            return runSimulation<VelocitySet, case_policy::Droplet>(cfg, cli, simDir);

        default:
            throw std::runtime_error("Unsupported case kind in startup dispatch.");
        }
    }
}

int main(int argc, char *argv[])
{
    try
    {
        const runtime::CliOptions cli = runtime::parseCli(argc, argv);
        const runtime::RuntimeConfig cfg = runtime::loadFromCurrentDirectory();

        std::unique_ptr<cases::Case> caseObject = cases::createCase(cfg.control.caseName);
        caseObject->validate(cfg);

        if (host::setDevice(cli.gpuIndex) < 0)
        {
            return 1;
        }

        const runtime::DeviceConstants deviceConstants = makeDeviceConstants(cfg, *caseObject);
        runtime::uploadDeviceConstants(deviceConstants);

        const std::string simDir = host::ensureOutputDirectory(cfg.outputDirectory);

        std::cout << "Case directory:         " << cfg.caseDirectory.string() << "\n";
        std::cout << "Output directory:       " << simDir << "\n";
        std::cout << "Case name:              " << cfg.control.caseName << "\n";
        std::cout << "Stencil:                " << runtime::stencilToString(cli.stencil) << "\n";
        std::cout << "Simulation ID:          " << cli.simulationId << "\n";

        if (cli.dryRun)
        {
            std::cout << "Dry-run successful: parser, case factory, constants upload, and startup dispatch are valid.\n";
            return 0;
        }

        switch (cli.stencil)
        {
        case runtime::Stencil::D3Q19:
            return dispatchCase<lbm::D3Q19>(caseObject->kind(), cfg, cli, simDir);

        case runtime::Stencil::D3Q27:
            return dispatchCase<lbm::D3Q27>(caseObject->kind(), cfg, cli, simDir);

        default:
            throw std::runtime_error("Unsupported stencil in startup dispatch.");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}

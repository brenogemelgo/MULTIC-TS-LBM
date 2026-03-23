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
   Host-side I/O, diagnostics, and utility routines for simulation setup and data output

Namespace
    host

SourceFiles
    hostFunctions.cuh

\*---------------------------------------------------------------------------*/

#ifndef HOSTFUNCTIONS_CUH
#define HOSTFUNCTIONS_CUH

#include "globalFunctions.cuh"
#include "config/PhaseVelocitySet.cuh"
#include "runtime/RuntimeConfig.cuh"

namespace host
{
    __host__ [[nodiscard]] static inline std::string ensureOutputDirectory(const std::filesystem::path &outputDir)
    {
        std::error_code ec;
        std::filesystem::create_directories(outputDir, ec);
        if (ec)
        {
            throw std::runtime_error("Could not create output directory: " + outputDir.string() + " (" + ec.message() + ")");
        }

        return outputDir.string() + std::string(1, std::filesystem::path::preferred_separator);
    }

    __host__ [[gnu::cold]] static inline void generateSimulationInfoFile(
        const std::string &simDir,
        const std::string &simId,
        const std::string &velocitySet,
        const runtime::RuntimeConfig &cfg)
    {
        std::filesystem::path infoPath = std::filesystem::path(simDir) / (simId + "_info.txt");

        try
        {
            std::ofstream file(infoPath, std::ios::out | std::ios::trunc);
            if (!file.is_open())
            {
                std::cerr << "Error opening file: " << infoPath.string() << std::endl;
                return;
            }

            file << "---------------------------- SIMULATION METADATA ----------------------------\n"
                 << "ID:                    " << simId << '\n'
                 << "Case name:             " << cfg.control.caseName << '\n'
                 << "Velocity set:          " << velocitySet << '\n'
                 << "Reference velocity:    " << physics::u_inf() << '\n'
                 << "Dispersed phase Re:    " << physics::reynolds_oil() << '\n'
                 << "Continuous phase Re:   " << physics::reynolds_water() << '\n'
                 << "Weber number:          " << physics::weber() << '\n'
                 << "Surface tension:       " << physics::sigma() << '\n'
                 << "NX:                    " << mesh::nx() << '\n'
                 << "NY:                    " << mesh::ny() << '\n'
                 << "NZ:                    " << mesh::nz() << '\n'
                 << "Diameter:              " << mesh::diam() << '\n'
                 << "Timesteps:             " << control::nTimeSteps() << '\n'
                 << "Output interval:       " << control::saveInterval() << '\n'
                 << "-----------------------------------------------------------------------------\n";

            file.close();

            std::cout << "Simulation information file created in: " << infoPath.string() << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error generating information file: " << e.what() << std::endl;
        }
    }

    __host__ [[gnu::cold]] static inline void printDiagnostics(const std::string &velocitySet) noexcept
    {
        const scalar_t tauRef = static_cast<scalar_t>(1) / relaxation::omega_ref();

        const scalar_t tau_d = (runtime::constants().tau_oil > static_cast<scalar_t>(0)) ? runtime::constants().tau_oil : tauRef;
        const scalar_t tau_c = (runtime::constants().tau_water > static_cast<scalar_t>(0)) ? runtime::constants().tau_water : tauRef;

        const double Ma = static_cast<double>(physics::u_inf()) * static_cast<double>(3);

        std::cout << "\n---------------------------- SIMULATION METADATA ----------------------------\n"
                  << "Velocity set:          " << velocitySet << '\n'
                  << "Reference velocity:    " << physics::u_inf() << '\n'
                  << "Dispersed phase Re:    " << physics::reynolds_oil() << '\n'
                  << "Continuous phase Re:   " << physics::reynolds_water() << '\n'
                  << "Weber number:          " << physics::weber() << '\n'
                  << "Surface tension:       " << physics::sigma() << '\n'
                  << "NX:                    " << mesh::nx() << '\n'
                  << "NY:                    " << mesh::ny() << '\n'
                  << "NZ:                    " << mesh::nz() << '\n'
                  << "Diameter:              " << mesh::diam() << '\n'
                  << "Total domain size:     " << size::cells() << '\n'
                  << "Mach:                  " << Ma << '\n'
                  << "Dispersed phase tau:   " << tau_d << '\n'
                  << "Continuous phase tau:  " << tau_c << '\n'
                  << "-----------------------------------------------------------------------------\n\n";
    }

    __host__ [[nodiscard]] [[gnu::cold]] static inline int setDevice(const int dev) noexcept
    {
        cudaError_t err = cudaSetDevice(dev);
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(err));

            return -1;
        }

        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        {
            std::printf("Using GPU %d: %s (SM %d.%d)\n", dev, prop.name, prop.major, prop.minor);
        }
        else
        {
            std::printf("Using GPU %d (unknown properties)\n", dev);
        }

        return dev;
    }

    __host__ [[nodiscard]] static inline constexpr unsigned divUp(
        const unsigned a,
        const unsigned b) noexcept
    {
        return (a + b - 1u) / b;
    }

    __host__ [[nodiscard]] static inline size_t bytesScalar() noexcept
    {
        return static_cast<size_t>(size::cells()) * sizeof(scalar_t);
    }

    template <typename VelocitySet>
    __host__ [[nodiscard]] static inline size_t bytesF() noexcept
    {
        return static_cast<size_t>(size::cells()) * static_cast<size_t>(VelocitySet::Q()) * sizeof(pop_t);
    }

    __host__ [[nodiscard]] static inline size_t bytesG() noexcept
    {
        return static_cast<size_t>(size::cells()) * static_cast<size_t>(phase::velocitySet::Q()) * sizeof(scalar_t);
    }
}

#endif

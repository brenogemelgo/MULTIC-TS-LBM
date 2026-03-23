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
    Runtime parsing and validation for case directory inputs and CLI options

SourceFiles
    RuntimeConfig.cuh

\*---------------------------------------------------------------------------*/

#ifndef RUNTIMECONFIG_CUH
#define RUNTIMECONFIG_CUH

#include "LBMIncludes.cuh"

namespace runtime
{
    enum class Stencil
    {
        D3Q19,
        D3Q27
    };

    struct MeshConfig
    {
        label_t nx = 0;
        label_t ny = 0;
        label_t nz = 0;
    };

    struct ProgramControl
    {
        std::string caseName;
        scalar_t ReA = static_cast<scalar_t>(0);
        scalar_t ReB = static_cast<scalar_t>(0);
        scalar_t We = static_cast<scalar_t>(0);
        scalar_t u_inf = static_cast<scalar_t>(0);
        scalar_t L_char = static_cast<scalar_t>(0);
        label_t nTimeSteps = 0;
        label_t saveInterval = 0;
    };

    struct RuntimeConfig
    {
        MeshConfig mesh;
        ProgramControl control;
        std::filesystem::path caseDirectory;
        std::filesystem::path outputDirectory;
    };

    struct CliOptions
    {
        Stencil stencil = Stencil::D3Q27;
        std::string simulationId;
        int gpuIndex = 0;
        bool dryRun = false;
    };

    __host__ [[nodiscard]] static inline std::string trim(const std::string &s)
    {
        const auto begin = s.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos)
        {
            return "";
        }

        const auto end = s.find_last_not_of(" \t\r\n");
        return s.substr(begin, end - begin + 1);
    }

    __host__ [[nodiscard]] static inline std::string toLower(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    __host__ [[nodiscard]] static inline std::string toUpper(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                       { return static_cast<char>(std::toupper(c)); });
        return s;
    }

    __host__ [[nodiscard]] static inline std::unordered_map<std::string, std::string> parseKeyValueFile(
        const std::filesystem::path &filePath,
        const char *fileTag)
    {
        std::ifstream in(filePath);
        if (!in)
        {
            throw std::runtime_error(std::string("Could not open ") + fileTag + " at: " + filePath.string());
        }

        std::unordered_map<std::string, std::string> kv;
        std::string line;
        label_t lineNumber = 0;

        while (std::getline(in, line))
        {
            ++lineNumber;
            const auto commentPos = line.find('#');
            if (commentPos != std::string::npos)
            {
                line.erase(commentPos);
            }

            line = trim(line);
            if (line.empty())
            {
                continue;
            }

            const auto eq = line.find('=');
            if (eq == std::string::npos)
            {
                throw std::runtime_error(std::string(fileTag) + ": invalid line " + std::to_string(lineNumber) +
                                         " (expected key = value): " + line);
            }

            std::string key = trim(line.substr(0, eq));
            std::string value = trim(line.substr(eq + 1));

            if (key.empty() || value.empty())
            {
                throw std::runtime_error(std::string(fileTag) + ": invalid line " + std::to_string(lineNumber) +
                                         " (empty key/value): " + line);
            }

            kv[key] = value;
        }

        return kv;
    }

    template <typename T>
    __host__ [[nodiscard]] static inline T parseNumber(
        const std::unordered_map<std::string, std::string> &kv,
        const char *fileTag,
        const char *key)
    {
        const auto it = kv.find(key);
        if (it == kv.end())
        {
            throw std::runtime_error(std::string(fileTag) + ": missing required key '" + key + "'.");
        }

        try
        {
            if constexpr (std::is_same_v<T, label_t>)
            {
                const auto v = std::stoull(it->second);
                return static_cast<label_t>(v);
            }
            else if constexpr (std::is_same_v<T, int>)
            {
                return std::stoi(it->second);
            }
            else
            {
                return static_cast<T>(std::stod(it->second));
            }
        }
        catch (const std::exception &)
        {
            throw std::runtime_error(std::string(fileTag) + ": invalid numeric value for key '" + key + "': " + it->second);
        }
    }

    __host__ [[nodiscard]] static inline std::string parseString(
        const std::unordered_map<std::string, std::string> &kv,
        const char *fileTag,
        const char *key)
    {
        const auto it = kv.find(key);
        if (it == kv.end())
        {
            throw std::runtime_error(std::string(fileTag) + ": missing required key '" + key + "'.");
        }

        const std::string value = trim(it->second);
        if (value.empty())
        {
            throw std::runtime_error(std::string(fileTag) + ": empty value for key '" + key + "'.");
        }

        return value;
    }

    __host__ [[nodiscard]] static inline MeshConfig parseMeshConfig(const std::filesystem::path &latticeMeshPath)
    {
        const auto kv = parseKeyValueFile(latticeMeshPath, "latticeMesh");

        MeshConfig m{};
        m.nx = parseNumber<label_t>(kv, "latticeMesh", "nx");
        m.ny = parseNumber<label_t>(kv, "latticeMesh", "ny");
        m.nz = parseNumber<label_t>(kv, "latticeMesh", "nz");

        if (m.nx < 3 || m.ny < 3 || m.nz < 3)
        {
            throw std::runtime_error("latticeMesh: nx, ny, nz must all be >= 3.");
        }

        return m;
    }

    __host__ [[nodiscard]] static inline ProgramControl parseProgramControl(const std::filesystem::path &programControlPath)
    {
        const auto kv = parseKeyValueFile(programControlPath, "programControl");

        ProgramControl p{};
        p.caseName = toLower(parseString(kv, "programControl", "caseName"));
        p.ReA = parseNumber<scalar_t>(kv, "programControl", "ReA");
        p.ReB = parseNumber<scalar_t>(kv, "programControl", "ReB");
        p.We = parseNumber<scalar_t>(kv, "programControl", "We");
        p.u_inf = parseNumber<scalar_t>(kv, "programControl", "u_inf");
        p.L_char = parseNumber<scalar_t>(kv, "programControl", "L_char");
        p.nTimeSteps = parseNumber<label_t>(kv, "programControl", "nTimeSteps");
        p.saveInterval = parseNumber<label_t>(kv, "programControl", "saveInterval");

        if (p.nTimeSteps == 0)
        {
            throw std::runtime_error("programControl: nTimeSteps must be > 0.");
        }
        if (p.saveInterval == 0)
        {
            throw std::runtime_error("programControl: saveInterval must be > 0.");
        }
        if (p.u_inf < static_cast<scalar_t>(0))
        {
            throw std::runtime_error("programControl: u_inf must be >= 0.");
        }
        if (p.L_char <= static_cast<scalar_t>(0))
        {
            throw std::runtime_error("programControl: L_char must be > 0.");
        }

        return p;
    }

    __host__ [[nodiscard]] static inline RuntimeConfig loadFromCurrentDirectory()
    {
        RuntimeConfig cfg{};

        cfg.caseDirectory = std::filesystem::current_path();
        const auto latticeMeshPath = cfg.caseDirectory / "latticeMesh";
        const auto programControlPath = cfg.caseDirectory / "programControl";

        cfg.mesh = parseMeshConfig(latticeMeshPath);
        cfg.control = parseProgramControl(programControlPath);

        cfg.outputDirectory = cfg.caseDirectory / "output";
        std::error_code ec{};
        std::filesystem::create_directories(cfg.outputDirectory, ec);
        if (ec)
        {
            throw std::runtime_error("Could not create output directory at: " + cfg.outputDirectory.string() +
                                     " (" + ec.message() + ")");
        }

        return cfg;
    }

    __host__ [[nodiscard]] static inline const char *stencilToString(const Stencil s) noexcept
    {
        return (s == Stencil::D3Q19) ? "D3Q19" : "D3Q27";
    }

    __host__ [[nodiscard]] static inline CliOptions parseCli(int argc, char *argv[])
    {
        CliOptions options{};

        bool hasStencil = false;
        bool hasId = false;

        auto requireValue = [argc, argv](int i, const char *flag) -> std::string
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(std::string("Missing value for ") + flag);
            }
            return std::string(argv[i + 1]);
        };

        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];

            if (arg == "-STENCIL")
            {
                const std::string stencil = toUpper(requireValue(i, "-STENCIL"));
                if (stencil == "D3Q19")
                {
                    options.stencil = Stencil::D3Q19;
                }
                else if (stencil == "D3Q27")
                {
                    options.stencil = Stencil::D3Q27;
                }
                else
                {
                    throw std::runtime_error("Invalid -STENCIL value. Expected D3Q19 or D3Q27.");
                }
                hasStencil = true;
                ++i;
            }
            else if (arg == "-ID")
            {
                options.simulationId = requireValue(i, "-ID");
                if (trim(options.simulationId).empty())
                {
                    throw std::runtime_error("-ID value cannot be empty.");
                }
                hasId = true;
                ++i;
            }
            else if (arg == "-GPU")
            {
                const auto gpu = requireValue(i, "-GPU");
                try
                {
                    options.gpuIndex = std::stoi(gpu);
                }
                catch (const std::exception &)
                {
                    throw std::runtime_error("Invalid -GPU value: " + gpu);
                }

                if (options.gpuIndex < 0)
                {
                    throw std::runtime_error("-GPU must be >= 0.");
                }
                ++i;
            }
            else if (arg == "--dry-run")
            {
                options.dryRun = true;
            }
            else if (arg == "-h" || arg == "--help")
            {
                throw std::runtime_error(
                    "Usage: phaseFieldLBM -STENCIL <D3Q19|D3Q27> -ID <SIMULATION_ID> -GPU <INDEX> [--dry-run]");
            }
            else
            {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }

        if (!hasStencil || !hasId)
        {
            throw std::runtime_error(
                "Usage: phaseFieldLBM -STENCIL <D3Q19|D3Q27> -ID <SIMULATION_ID> -GPU <INDEX> [--dry-run]");
        }

        return options;
    }
}

#endif

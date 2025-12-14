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
    Post-processing common structs

Namespace
    host

SourceFiles
    postCommon.cuh

\*---------------------------------------------------------------------------*/

#ifndef POSTCOMMON_CUH
#define POSTCOMMON_CUH

#include "functions/globalFunctions.cuh"

namespace host
{
    namespace detail
    {
        template <typename T>
        struct vtkScalarTypeName;

        template <>
        struct vtkScalarTypeName<float>
        {
            static constexpr const char *value() noexcept { return "Float32"; }
        };

        template <>
        struct vtkScalarTypeName<double>
        {
            static constexpr const char *value() noexcept { return "Float64"; }
        };

        struct AppendedArray
        {
            std::string name;
            std::filesystem::path path;
            std::uint64_t nbytes;
            std::uint64_t offset;
            bool isPoints;
        };
    }
}

#endif

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
    CUDA Graph implementation

Namespace
    graph

SourceFiles
    CUDAGraph.cuh

\*---------------------------------------------------------------------------*/

#ifndef CUDAGRAPH_CUH
#define CUDAGRAPH_CUH

#include "../phaseField.cuh"

namespace graph
{
    template <dim3 grid, dim3 block, size_t dynamic>
    __host__ inline void captureGraph(
        cudaGraph_t &graph,
        cudaGraphExec_t &graphExec,
        const LBMFields &fields,
        const cudaStream_t queue)
    {
        checkCudaErrorsOutline(cudaStreamBeginCapture(queue, cudaStreamCaptureModeGlobal));

        // Phase field
        Phase::computePhase<<<grid, block, dynamic, queue>>>(fields);
        Phase::computeNormals<<<grid, block, dynamic, queue>>>(fields);
        Phase::computeForces<<<grid, block, dynamic, queue>>>(fields);

        // Hydrodynamics
        LBM::computeMoments<<<grid, block, dynamic, queue>>>(fields);
        LBM::streamCollide<<<grid, block, dynamic, queue>>>(fields);

        // NOTE: We intentionally DO NOT include FlowCase::boundaryConditions
        // or timeAverage here, because they depend on STEP and/or other
        // time-varying parameters. We launch them after the graph each step.

        checkCudaErrorsOutline(cudaStreamEndCapture(queue, &graph));
        checkCudaErrorsOutline(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    }
}

#endif

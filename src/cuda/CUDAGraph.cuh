/*---------------------------------------------------------------------------*\
|                                                                             |
| phaseFieldLBM: CUDA-based multicomponent Lattice Boltzmann Method           |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/brenogemelgo/phaseFieldLBM                       |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Description
    CUDA Graph capture of the core phase-field and hydrodynamic LBM kernel sequence

Namespace
    graph

SourceFiles
    CUDAGraph.cuh

\*---------------------------------------------------------------------------*/

#ifndef CUDAGRAPH_CUH
#define CUDAGRAPH_CUH

#include "phaseField.cuh"
#include "LBMIncludes.cuh"

namespace graph
{
    template <typename HydroVS, typename CasePolicy>
    __host__ inline void captureGraph(
        cudaGraph_t &graph,
        cudaGraphExec_t &graphExec,
        const LBMFields &fields,
        const cudaStream_t queue,
        const dim3 &grid,
        const dim3 &block,
        const size_t dynamic)
    {
        checkCudaErrorsOutline(cudaStreamBeginCapture(queue, cudaStreamCaptureModeGlobal));

        // Phase field
        phase::computePhase<HydroVS><<<grid, block, dynamic, queue>>>(fields);
        phase::computeNormals<HydroVS><<<grid, block, dynamic, queue>>>(fields);
        phase::computeForces<HydroVS><<<grid, block, dynamic, queue>>>(fields);

        // Hydrodynamics
        lbm::computeMoments<HydroVS><<<grid, block, dynamic, queue>>>(fields);
        lbm::streamCollide<HydroVS, CasePolicy><<<grid, block, dynamic, queue>>>(fields);

        // NOTE: We intentionally DO NOT include boundary conditions or
        // derived fields here, because they depend on STEP and/or other
        // time-varying parameters. We launch them after the graph each step.

        checkCudaErrorsOutline(cudaStreamEndCapture(queue, &graph));
        checkCudaErrorsOutline(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    }
}

#endif

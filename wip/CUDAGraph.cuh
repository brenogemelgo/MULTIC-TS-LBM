/*===========================================================================*\
    MULTIC-TS-LBM — CUDA Graph implementation
\*===========================================================================*/

#ifndef CUDAGRAPH_CUH
#define CUDAGRAPH_CUH

#include "../cuda/utils.cuh"
#include "../constants.cuh"
#include "../LBMIncludes.cuh"
#include "../phaseField.cuh"

extern __device__ label_t d_timestep;

// Kernel accessor (safer than extern)
namespace device
{
    __host__ __device__ inline label_t *timestepPtr()
    {
        return &d_timestep;
    }
}

// Graph container ------------------------------------------------------------
struct SimulationGraph
{
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t instance{nullptr};
    bool instantiated{false};

    dim3 grid3D, block3D;
    dim3 gridZ, blockZ;
    size_t dynamic;
    cudaStream_t stream;
};

// ---------------------------------------------------------------------------
// 1) Capture reusable sequence of kernels
// ---------------------------------------------------------------------------
__host__ inline void captureGraph(
    SimulationGraph &G,
    const LBMFields &fields)
{
    // Begin capture
    checkCudaErrors(cudaStreamBeginCapture(G.stream, cudaStreamCaptureModeGlobal));

    // ---------------- PHASE FIELD ----------------
    Phase::computeNormals<<<G.grid3D, G.block3D, 0, G.stream>>>(fields);
    Phase::computeForces<<<G.grid3D, G.block3D, 0, G.stream>>>(fields);

    // ---------------- HYDRODYNAMICS --------------
    LBM::computeMoments<<<G.grid3D, G.block3D, G.dynamic, G.stream>>>(fields);
    LBM::streamCollide<<<G.grid3D, G.block3D, 0, G.stream>>>(fields);

    // ---------------- BOUNDARIES -----------------
    // IMPORTANT: no timestep parameter here — it is read from d_timestep.
    LBM::callInflow<<<G.gridZ, G.blockZ, 0, G.stream>>>(fields);
    LBM::callOutflow<<<G.gridZ, G.blockZ, 0, G.stream>>>(fields);

    // End capture
    checkCudaErrors(cudaStreamEndCapture(G.stream, &G.graph));
    checkCudaErrors(cudaGraphInstantiate(&G.instance, G.graph, nullptr, nullptr, 0));

    G.instantiated = true;
}

// ---------------------------------------------------------------------------
// 2) Execute the graph for timestep t
// ---------------------------------------------------------------------------
__host__ inline void runTimestep(
    SimulationGraph &G,
    const LBMFields &fields,
    const label_t t)
{
    if (!G.instantiated)
    {
        captureGraph(G, fields);
    }

    // Update timestep inside GPU
    checkCudaErrors(cudaMemcpyAsync(
        device::timestepPtr(),
        &t,
        sizeof(label_t),
        cudaMemcpyHostToDevice,
        G.stream));

    // Launch full time-step graph
    checkCudaErrors(cudaGraphLaunch(G.instance, G.stream));
}

#endif

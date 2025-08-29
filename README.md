# MULTIC-TS-LBM

**MULTIC-TS-LBM** is a **GPU-accelerated**, thread-safe Lattice Boltzmann simulator for multicomponent flows. Implemented in CUDA, it supports **D3Q19/D3Q27** for hydrodynamics and **D3Q7** for phase-field evolution, capturing interface dynamics, surface tension, and perturbations. Available cases: **droplet** and **jet**.

---

## 🖥️ Requirements

- **GPU**: NVIDIA (CC ≥ 6.0, ≥ 2 GB, 4+ GB recommended)  
- **CUDA**: Toolkit ≥ 11.0  
- **Compiler**: C++ (`g++`, `nvcc`)  
- **Python 3.x**: `numpy`, `pyevtk`  
- **ParaView**: for `.vtr` visualization  

---

## 🗂️ Structure

- `src/` – C/C++ and CUDA sources  
- `include/` – auxiliary CUDA headers/scripts  
- `post/` – Python post-processing to VTK  
- `bin/` – compiled binaries & results  
- `compile.sh` – build script  
- `pipeline.sh` – compile → run → post-process  

---

## 🚀 Run

```bash
./pipeline.sh <velocity_set> <id>
```

* `velocity_set`: `D3Q19` | `D3Q27`
* `id`: simulation ID (e.g., `000`)

Pipeline: compile → simulate → post-process  

---

## 🧠 File Responsibilities

### `include/` – headers

- `cudaUtils.cuh` – CUDA utilities (types, constants, FP16 helpers, error checks)    
- `derivedFields.cuh` – optional kernel for derived fields (velocity/vorticity magnitudes)    
- `hostFunctions.cuh` – host utilities (dirs, occupancy, info/logs, memory alloc/copy)    
- `perturbationData.cuh` – predefined perturbation array for simulations   
- `velocitySets.cuh` – lattice velocity sets & weights (D3Q19, D3Q27, D3Q7)    

### `post/` – post-processing (Python)

- `getSimInfo.py` – file discovery & metadata  
- `gridToVTK.py` – VTK conversion (`pyevtk`)  
- `processSteps.py` – batch `.vtr` generation  
- `runPost.sh` – wrapper for `processSteps.py`  

### `src/` – simulation (CUDA)

- `bcs.cu` – boundary condition kernels (inflow, outflow, periodic)
- `constants.cuh` – global simulation parameters (mesh, case setup, relaxation, strides)    
- `deviceUtils.cuh` – core GPU data structures & device helpers (LBM fields, equilibria, forcing)   
- `init.cuh` – initialization kernels (fields, jet/droplet shapes, distributions)
- `lbm.cuh` – main CUDA kernels (moments, collision-stream, phase-field, normals, forces)  
- `main.cu` – simulation entry point (initialization, time loop, output, performance stats)   

---

## ⚡ Benchmark

Performance is reported in **MLUPS** (Million Lattice Updates Per Second).  
Each GPU entry shows the average across multiple runs.

| GPU            | D3Q19 (MLUPS) | D3Q27 (MLUPS) |
|----------------|---------------|---------------|
| RTX 3050 (4GB) | **710**       | **565**       |
| RTX 4090 (24GB)| –             | –             |
| A100 (40GB)    | –             | –             |

*Notes:*  
- **D3Q19** uses **He forcing (1st order)** and 2nd-order equilibrium/non-equilibrium expansion.  
- **D3Q27** uses **Guo forcing (2nd order)** and 3rd-order equilibrium/non-equilibrium expansion.  
- These methodological differences contribute to the observed performance gap, beyond the natural cost of upgrading from **19** to **27** velocity directions.

---

## 🧠 Project Context

This code was developed as part of an undergraduate research fellowship at the Geoenergia Lab (UDESC – Balneário Camboriú Campus), under the project:

**"Experiment-based physical and numerical modeling of subsea oil jet dispersion (SUBJET)"**, in partnership with **Petrobras, ANP, FITEJ and SINTEF Ocean**.

---

## 📊 Credits

The post-processing workflow is shared with the project [MR-LBM](https://github.com/CERNN/MR-LBM).

---

## 📬 Contact

For feature requests or contributions, feel free to open an issue or fork the project. 
You may also contact the maintainer via email at:

* breno.gemelgo@edu.udesc.br
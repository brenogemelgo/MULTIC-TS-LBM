# MULTIC-TS-LBM

**MULTIC-TS-LBM** is a **thread-safe**, GPU-accelerated simulator for multicomponent flows using the Lattice Boltzmann Method (LBM). Designed with CUDA for high performance, it supports **D3Q19** and **D3Q27** velocity sets for hydrodynamics, and **D3Q7** for phase-field evolution. The solver simulates flows with sharp interface dynamics, surface tension effects, and configurable perturbations. Currently, two cases are available: **droplet** and **jet**.

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

Example:

```bash
./pipeline.sh D3Q27 000
```

Pipeline: compile → simulate → post-process  

---

## 📁 Output

Results → `bin/<velocity_set>/<id>/`

- `.bin` field data (e.g., `phi`, `uz`)  
- `*_info.txt` metadata  
- `.vtr` from `exampleVTK.py`  

---

## 📊 Post-Processing

The post-processing workflow is shared with https://github.com/CERNN/MR-LBM. It uses Python scripts to parse binary outputs and convert them to `.vtr` files compatible with **ParaView**.

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
- `gridToVtk.py` – VTK conversion (`pyevtk`)  
- `processSteps.py` – batch `.vtr` generation  
- `runPost.sh` – wrapper for `processSteps.py`  

### `src/` – simulation (CUDA)

- `constants.cuh` – global simulation parameters (mesh, case setup, relaxation, strides)    
- `deviceHeader.cuh` – core GPU data structures & device helpers (LBM fields, equilibria, forcing)   
- `deviceSetup.cu` – defines GPU constants & global field instances    
- `lbm.cuh` – main CUDA kernels (collision-stream, phase-field, normals, forces)  
- `lbmBcs.cu` – boundary condition kernels (inflow, outflow, periodic)   
- `main.cu` – simulation entry point (initialization, time loop, BCs, output, performance stats)   

---

## 🧠 Project Context

This code was developed as part of an undergraduate research fellowship at the Geoenergia Lab (UDESC - Balneário Camboriú Campus), under the project:

**"Experiment-based physical and numerical modeling of subsea oil jet dispersion (SUBJET)"**, in partnership with **Petrobras, ANP, FITEJ and SINTEF Ocean**.

## 📬 Contact

For feature requests or contributions, feel free to open an issue or fork the project. You may also contact the maintainer via email at:

* breno.gemelgo@edu.udesc.br

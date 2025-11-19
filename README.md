# MULTIC-TS-LBM

**MULTIC-TS-LBM** is a **GPU-accelerated**, thread-safe Lattice Boltzmann simulator for multicomponent flows. Implemented in CUDA, it supports **D3Q19/D3Q27** for hydrodynamics and **D3Q7** for phase field evolution, capturing interface dynamics and surface tension. Available cases: **droplet** and **jet**.

---

## 🖥️ Requirements

- **GPU**: NVIDIA (Compute Capability ≥ 6.0, 4+ GB VRAM recommended)  
- **CUDA**: Toolkit ≥ 12.0  
- **Compiler**: C++20-capable (GCC ≥ 11) + `nvcc` (partial C++20 support)
- **Python 3.x**: `numpy`, `pyevtk`  (for post-processing)
- **ParaView**: for `.vtr` visualization  

---

## 🚀 Run

```bash
./pipeline.sh <velocity_set> <id>
```

* `velocity_set`: `D3Q19` | `D3Q27`
* `id`: simulation ID (e.g., `000`)

Pipeline: compile → simulate → post-process  

---

## ⚡ Benchmark

Performance is reported in **MLUPS** (Million Lattice Updates Per Second).  
Each GPU entry shows the average across multiple runs.

| GPU            | D3Q19 (MLUPS) | D3Q27 (MLUPS) |
|----------------|---------------|---------------|
| RTX 3050 (4GB) | **760**       | –             |
| RTX 4090 (24GB)| –             | –             |
| A100 (40GB)    | –             | –             |

*Important considerations:*  
- **D3Q19** uses **He forcing (1st order)** and 2nd-order equilibrium/non-equilibrium expansion.  
- **D3Q27** uses **Guo forcing (2nd order)** and 3rd-order equilibrium/non-equilibrium expansion.  
- These methodological differences contribute to the observed performance gap, beyond the natural cost of upgrading from **19** to **27** velocity directions.

---

## 🧠 Project Context

This code was developed as part of an undergraduate research fellowship at the Geoenergia Lab (UDESC – Balneário Camboriú Campus), under the project:

**"Experiment-based physical and numerical modeling of subsea oil jet dispersion (SUBJET)"**, in partnership with **Petrobras, ANP, FITEJ and SINTEF Ocean**.

---

## 📄 License

This project is licensed under the terms of the LICENSE file.

---

## 📊 Credits

The post-processing workflow is mostly shared with the project [MR-LBM](https://github.com/CERNN/MR-LBM).
The implementation is strongly based on the article *[A high-performance lattice Boltzmann model for multicomponent turbulent jet simulations](https://arxiv.org/abs/2403.15773)*.

---

## 📬 Contact

For feature requests or contributions, feel free to open an issue or fork the project. 
You may also contact the maintainer via email at:

* breno.gemelgo@edu.udesc.br
# phaseFieldLBM

`phaseFieldLBM` is a CUDA multicomponent lattice Boltzmann solver using a conservative Allen-Cahn phase-field model.

## Requirements
- NVIDIA GPU (Compute Capability >= 6.0)
- CUDA toolkit with `nvcc`
- C++20-capable host compiler

## Build
```bash
make
```

## Install
```bash
make install
```

By default, `make install` installs to `$HOME/.local/bin`.
For system-wide install, use:
```bash
sudo make install PREFIX=/usr/local
```

## Runtime Usage
Run from inside a case directory containing `latticeMesh` and `programControl`:
```bash
phaseFieldLBM -STENCIL <D3Q19|D3Q27> -ID <SIMULATION_ID> -GPU <INDEX>
```

Optional startup-only sanity mode:
```bash
phaseFieldLBM -STENCIL D3Q27 -ID test01 -GPU 0 --dry-run
```

## Case Directory Layout
```text
cases/
  jet/
    latticeMesh
    programControl
    output/
```

## Input Files
`latticeMesh`:
```text
nx = 64
ny = 64
nz = 128
```

`programControl`:
```text
caseName = jet
ReA = 5000
ReB = 5000
We = 500
u_inf = 0.05
L_char = 10
nTimeSteps = 100000
saveInterval = 1000
```

## Notes
- `FLOW_CASE` is no longer a build-time argument.
- Stencil specialization remains compile-time (runtime startup dispatch to templated solver).
- BC/IC remain hardcoded per case and are selected via runtime `caseName`.
- `constants.cuh` is deprecated as a compatibility shim; new code should include `config/PhaseVelocitySet.cuh`, `runtime/RuntimeState.cuh`, and `functions/constexprFor.cuh` directly.
- `pipeline.sh` is deprecated and exits with guidance.
- `compile.sh` is a legacy wrapper to `make`.

## Refactor Design Note
See `docs/runtime_case_refactor.md` for architecture, migration, and verification notes.

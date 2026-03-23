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
    Centralized kernel declarations for LBM initialization, core routines, and boundary conditions

Namespace
    lbm

SourceFiles
    LBMIncludes.cuh

\*---------------------------------------------------------------------------*/

#ifndef LBMINCLUDES_CUH
#define LBMINCLUDES_CUH

#include <cuda_runtime.h>
#include <math_constants.h>
#include <builtin_types.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <unordered_map>
#include <array>
#include <thread>
#include <future>
#include <queue>
#include <condition_variable>
#include <math.h>
#include <stdlib.h>

#include "cuda/errorCheck.cuh"
#include "cuda/precision.cuh"
#include "functions/constexprFor.cuh"

struct LBMFields
{
    scalar_t *rho;
    scalar_t *ux;
    scalar_t *uy;
    scalar_t *uz;
    scalar_t *pxx;
    scalar_t *pyy;
    scalar_t *pzz;
    scalar_t *pxy;
    scalar_t *pxz;
    scalar_t *pyz;

    scalar_t *phi;
    scalar_t *normx;
    scalar_t *normy;
    scalar_t *normz;
    scalar_t *ind;
    scalar_t *fsx;
    scalar_t *fsy;
    scalar_t *fsz;

    pop_t *f;
    scalar_t *g;

#if TIME_AVERAGE

    scalar_t *avg_phi;
    scalar_t *avg_ux;
    scalar_t *avg_uy;
    scalar_t *avg_uz;

#endif

#if REYNOLDS_MOMENTS

    scalar_t *avg_uxux;
    scalar_t *avg_uyuy;
    scalar_t *avg_uzuz;
    scalar_t *avg_uxuy;
    scalar_t *avg_uxuz;
    scalar_t *avg_uyuz;

#endif

#if VORTICITY_FIELDS

    scalar_t *vort_x;
    scalar_t *vort_y;
    scalar_t *vort_z;
    scalar_t *vort_mag;

#endif

#if PASSIVE_SCALAR

    scalar_t *c;

#endif
};

LBMFields fields{};

namespace lbm
{
    // Initial conditions
    __global__ void setInitialDensity(LBMFields d);
    __global__ void setDroplet(LBMFields d);
    __global__ void setJet(LBMFields d);

    template <typename HydroVS>
    __global__ void setDistros(LBMFields d);

    // Moments and core routines
    template <typename HydroVS>
    __global__ void computeMoments(LBMFields d);

    template <typename HydroVS, typename CasePolicy>
    __global__ void streamCollide(LBMFields d);

    // Boundary conditions
    template <typename HydroVS>
    __global__ void callInflow(LBMFields d, const label_t t);

    template <typename HydroVS>
    __global__ void callOutflow(LBMFields d);
}

#endif

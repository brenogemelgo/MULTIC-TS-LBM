#include "deviceHeader.cuh"

__constant__ float W[FLINKS];
__constant__ float W_G[GLINKS];

__constant__ ci_t CIX[FLINKS], CIY[FLINKS], CIZ[FLINKS];

#ifdef PERTURBATION
__constant__ float PERTURBATION_DATA[200];
#endif

LBMFields lbm;
#ifdef D_FIELDS
DerivedFields dfields;
#endif // D_FIELDS
                                         

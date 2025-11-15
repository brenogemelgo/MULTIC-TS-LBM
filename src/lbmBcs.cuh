#pragma once

namespace LBM
{
    __global__ void callInflow(
        LBMFields d)
    {
        boundaryConditions::applyInflow<VelocitySet>(d);
    }

    __global__ void callOutflow(
        LBMFields d)
    {
        boundaryConditions::applyOutflow<VelocitySet>(d);
    }

    __global__ void callPeriodicX(
        LBMFields d)
    {
        boundaryConditions::periodicX(d);
    }

    __global__ void callPeriodicY(
        LBMFields d)
    {
        boundaryConditions::periodicY(d);
    }
}
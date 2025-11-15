#pragma once

namespace LBM
{
    __global__ void setFields(
        LBMFields d)
    {
        boundaryConditions::callSetFields<VelocitySet>(d);
    }

    __global__ void setJet(
        LBMFields d)
    {
        boundaryConditions::callSetJet<VelocitySet>(d);
    }

    __global__ void setDroplet(
        LBMFields d)
    {
        boundaryConditions::callSetDroplet(d);
    }

    __global__ void setDistros(
        LBMFields d)
    {
        boundaryConditions::callSetDistros(d);
    }
}
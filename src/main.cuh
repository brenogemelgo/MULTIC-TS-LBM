namespace LBM
{
    // Initial conditions
    __global__ void callSetFields(
        LBMFields d)
    {
        InitialConditions::setFields(d);
    }

#if defined(JET)

    __global__ void callSetJet(
        LBMFields d)
    {
        InitialConditions::setJet(d);
    }

#elif defined(DROPLET)

    __global__ void callSetDroplet(
        LBMFields d)
    {
        InitialConditions::setDroplet(d);
    }

#endif

    __global__ void callSetDistros(
        LBMFields d)
    {
        InitialConditions::setDistros(d);
    }

    // Boundary conditions
    __global__ void callInflow(
        LBMFields d)
    {
        BoundaryConditions::applyInflow(d);
    }

    __global__ void callOutflow(
        LBMFields d)
    {
        BoundaryConditions::applyOutflow(d);
    }

    __global__ void callPeriodicX(
        LBMFields d)
    {
        BoundaryConditions::periodicX(d);
    }

    __global__ void callPeriodicY(
        LBMFields d)
    {
        BoundaryConditions::periodicY(d);
    }
}
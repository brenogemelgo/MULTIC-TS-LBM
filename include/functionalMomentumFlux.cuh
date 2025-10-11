constexpr_for<0, FLINKS>([&] __device__ (auto I) {
    constexpr int Q = decltype(I)::value;

    const float cx = static_cast<float>(CIX[Q]);
    const float cy = static_cast<float>(CIY[Q]);
    const float cz = static_cast<float>(CIZ[Q]);

    const float feq = computeFeq(rho,ux,uy,uz,uu,Q);

    const float force = 0.5f * feq *
                        ( (cx - ux) * ffx +
                          (cy - uy) * ffy +
                          (cz - uz) * ffz ) * invRhoCssq;

    const float feqF = feq - force;

    const float fneq = pop[Q] - feqF;  

    pxx += fneq * (cx * cx);
    pyy += fneq * (cy * cy);
    pzz += fneq * (cz * cz);
    pxy += fneq * (cx * cy);
    pxz += fneq * (cx * cz);
    pyz += fneq * (cy * cz);
});

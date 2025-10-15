{
    //#if defined(D3Q19) //         0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18
    //__constant__ ci_t CIX[19] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0 };
    //__constant__ ci_t CIY[19] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1 };
    //__constant__ ci_t CIZ[19] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1 };
    
    float feq, force, fneq, tmp;
    const float coeff = 1.5f * invRho;
    const float uf = ux*ffx + uy*ffy + uz*ffz;
    const float uu = 1.5f * (ux*ux + uy*uy + uz*uz);

    // ========================== ONE ========================== //

    feq   = W_1 * rho * (1.0f - uu + 3.0f*ux + 4.5f*ux*ux) - W_1;
    force = coeff * feq * (ffx - uf);
    fneq  = pop[1] - feq + force;

    pxx += fneq;

    // ========================== TWO ========================== //

    feq   = W_1 * rho * (1.0f - uu - 3.0f*ux + 4.5f*ux*ux) - W_1;
    force = coeff * feq * (ffx + uf);
    fneq  = pop[2] - feq - force;

    pxx += fneq;

    // ========================== THREE ========================== //

    feq   = W_1 * rho * (1.0f - uu + 3.0f*uy + 4.5f*uy*uy) - W_1;
    force = coeff * feq * (ffy - uf);
    fneq  = pop[3] - feq + force;

    pyy += fneq;

    // ========================== FOUR ========================== //

    feq   = W_1 * rho * (1.0f - uu - 3.0f*uy + 4.5f*uy*uy) - W_1;
    force = coeff * feq * (ffy + uf);
    fneq  = pop[4] - feq - force;

    pyy += fneq;

    // ========================== FIVE ========================== //

    feq   = W_1 * rho * (1.0f - uu + 3.0f*uz + 4.5f*uz*uz) - W_1;
    force = coeff * feq * (ffz - uf);
    fneq  = pop[5] - feq + force;

    pzz += fneq;

    // ========================== SIX ========================== //

    feq   = W_1 * rho * (1.0f - uu - 3.0f*uz + 4.5f*uz*uz) - W_1;
    force = coeff * feq * (ffz + uf);
    fneq  = pop[6] - feq - force;

    pzz += fneq;

    // ========================== SEVEN ========================== //

    tmp   = ux + uy;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx + ffy - uf);
    fneq  = pop[7] - feq + force;

    pxx += fneq;
    pyy += fneq;
    pxy += fneq;

    // ========================== EIGHT ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx + ffy + uf);
    fneq  = pop[8] - feq - force;

    pxx += fneq;
    pyy += fneq;
    pxy += fneq;

    // ========================== NINE ========================== //

    tmp   = ux + uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx + ffz - uf);
    fneq  = pop[9] - feq + force;

    pxx += fneq;
    pzz += fneq;
    pxz += fneq;

    // ========================== TEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx + ffz + uf);
    fneq  = pop[10] - feq - force;

    pxx += fneq;
    pzz += fneq;
    pxz += fneq;

    // ========================== ELEVEN ========================== //

    tmp   = uy + uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffy + ffz - uf);
    fneq  = pop[11] - feq + force;

    pyy += fneq;
    pzz += fneq;
    pyz += fneq;

    // ========================== TWELVE ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffy + ffz + uf);
    fneq  = pop[12] - feq - force;

    pyy += fneq;
    pzz += fneq;
    pyz += fneq;

    // ========================== THIRTEEN ========================== //

    tmp   = ux - uy;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx - ffy - uf);
    fneq  = pop[13] - feq + force;

    pxx += fneq;
    pyy += fneq;
    pxy -= fneq;

    // ========================== FOURTEEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffy - ffx - uf);
    fneq  = pop[14] - feq + force;

    pxx += fneq;
    pyy += fneq;
    pxy -= fneq;

    // ========================== FIFTEEN ========================== //

    tmp   = ux - uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx - ffz - uf);
    fneq  = pop[15] - feq + force;

    pxx += fneq;
    pzz += fneq;
    pxz -= fneq;

    // ========================== SIXTEEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffz - ffx - uf);
    fneq  = pop[16] - feq + force;

    pxx += fneq;
    pzz += fneq;
    pxz -= fneq;

    // ========================== SEVENTEEN ========================== //

    tmp   = uy - uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffy - ffz - uf);
    fneq  = pop[17] - feq + force;

    pyy += fneq;
    pzz += fneq;
    pyz -= fneq;

    // ========================== EIGHTEEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffz - ffy - uf);
    fneq  = pop[18] - feq + force;

    pyy += fneq;
    pzz += fneq;
    pyz -= fneq;
}
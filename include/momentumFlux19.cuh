{
    float feq, force, fneq, tmp;
    const float coeff = 1.5f * invRho;
    const float uf = ux*ffx + uy*ffy + uz*ffz;
    const float uu = 1.5f * (ux*ux + uy*uy + uz*uz);

    // ========================== ONE ========================== //

    feq   = W_1 * rho * (1.0f - uu + 3.0f*ux + 4.5f*ux*ux) - W_1;
    force = coeff * feq * (ffx - uf);
    fneq  = pop1 - feq + force;

    pxx += fneq;

    // ========================== TWO ========================== //

    feq   = W_1 * rho * (1.0f - uu - 3.0f*ux + 4.5f*ux*ux) - W_1;
    force = coeff * feq * (ffx + uf);
    fneq  = pop2 - feq - force;

    pxx += fneq;

    // ========================== THREE ========================== //

    feq   = W_1 * rho * (1.0f - uu + 3.0f*uy + 4.5f*uy*uy) - W_1;
    force = coeff * feq * (ffy - uf);
    fneq  = pop3 - feq + force;

    pyy += fneq;

    // ========================== FOUR ========================== //

    feq   = W_1 * rho * (1.0f - uu - 3.0f*uy + 4.5f*uy*uy) - W_1;
    force = coeff * feq * (ffy + uf);
    fneq  = pop4 - feq - force;

    pyy += fneq;

    // ========================== FIVE ========================== //

    feq   = W_1 * rho * (1.0f - uu + 3.0f*uz + 4.5f*uz*uz) - W_1;
    force = coeff * feq * (ffz - uf);
    fneq  = pop5 - feq + force;

    pzz += fneq;

    // ========================== SIX ========================== //

    feq   = W_1 * rho * (1.0f - uu - 3.0f*uz + 4.5f*uz*uz) - W_1;
    force = coeff * feq * (ffz + uf);
    fneq  = pop6 - feq - force;

    pzz += fneq;

    // ========================== SEVEN ========================== //

    tmp   = ux + uy;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx + ffy - uf);
    fneq  = pop7 - feq + force;

    pxx += fneq;
    pyy += fneq;
    pxy += fneq;

    // ========================== EIGHT ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx + ffy + uf);
    fneq  = pop8 - feq - force;

    pxx += fneq;
    pyy += fneq;
    pxy += fneq;

    // ========================== NINE ========================== //

    tmp   = ux + uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx + ffz - uf);
    fneq  = pop9 - feq + force;

    pxx += fneq;
    pzz += fneq;
    pxz += fneq;

    // ========================== TEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx + ffz + uf);
    fneq  = pop10 - feq - force;

    pxx += fneq;
    pzz += fneq;
    pxz += fneq;

    // ========================== ELEVEN ========================== //

    tmp   = uy + uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffy + ffz - uf);
    fneq  = pop11 - feq + force;

    pyy += fneq;
    pzz += fneq;
    pyz += fneq;

    // ========================== TWELVE ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffy + ffz + uf);
    fneq  = pop12 - feq - force;

    pyy += fneq;
    pzz += fneq;
    pyz += fneq;

    // ========================== THIRTEEN ========================== //

    tmp   = ux - uy;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx - ffy - uf);
    fneq  = pop13 - feq + force;

    pxx += fneq;
    pyy += fneq;
    pxy -= fneq;

    // ========================== FOURTEEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffy - ffx - uf);
    fneq  = pop14 - feq + force;

    pxx += fneq;
    pyy += fneq;
    pxy -= fneq;

    // ========================== FIFTEEN ========================== //

    tmp   = ux - uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx - ffz - uf);
    fneq  = pop15 - feq + force;

    pxx += fneq;
    pzz += fneq;
    pxz -= fneq;

    // ========================== SIXTEEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffz - ffx - uf);
    fneq  = pop16 - feq + force;

    pxx += fneq;
    pzz += fneq;
    pxz -= fneq;

    // ========================== SEVENTEEN ========================== //

    tmp   = uy - uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffy - ffz - uf);
    fneq  = pop17 - feq + force;

    pyy += fneq;
    pzz += fneq;
    pyz -= fneq;

    // ========================== EIGHTEEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffz - ffy - uf);
    fneq  = pop18 - feq + force;

    pyy += fneq;
    pzz += fneq;
    pyz -= fneq;

    d.pxx[idx3] = pxx;
    d.pyy[idx3] = pyy;
    d.pzz[idx3] = pzz;
    d.pxy[idx3] = pxy;
    d.pxz[idx3] = pxz;
    d.pyz[idx3] = pyz;
}
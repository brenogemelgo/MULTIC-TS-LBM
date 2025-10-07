{
    float feq, force, fneq, tmp;
    const float coeff = 1.5f * invRho;
    const float uf = ux*ffx + uy*ffy + uz*ffz;
    const float uu = 1.5f * (ux*ux + uy*uy + uz*uz);
    const float cspi = CSSQ * (pxx + pyy + pzz);

    // ========================== ZERO ========================== //

    feq   = W_0 * rho * (1.0f - uu) - W_0;
    force = coeff * feq * uf;
    fneq  = W0_45 * cspi;

    d.f[idx3] = toPop(feq - omcoLocal * fneq - force);

    // ========================== ONE ========================== //

    feq   = W_1 * rho * (1.0f - uu + 3.0f*ux + 4.5f*ux*ux) - W_1;
    force = coeff * feq * (ffx - uf);
    fneq  = W1_45 * (pxx - cspi);

    d.f[global4(x+1,y,z,1)] = toPop(feq + omcoLocal * fneq + force);

    // ========================== TWO ========================== //

    feq   = W_1 * rho * (1.0f - uu - 3.0f*ux + 4.5f*ux*ux) - W_1;
    force = coeff * feq * (ffx + uf);

    d.f[global4(x-1,y,z,2)] = toPop(feq + omcoLocal * fneq - force);

    // ========================== THREE ========================== //

    feq   = W_1 * rho * (1.0f - uu + 3.0f*uy + 4.5f*uy*uy) - W_1;
    force = coeff * feq * (ffy - uf);
    fneq  = W1_45 * (pyy - cspi);

    d.f[global4(x,y+1,z,3)] = toPop(feq + omcoLocal * fneq + force);

    // ========================== FOUR ========================== //

    feq   = W_1 * rho * (1.0f - uu - 3.0f*uy + 4.5f*uy*uy) - W_1;
    force = coeff * feq * (ffy + uf);

    d.f[global4(x,y-1,z,4)] = toPop(feq + omcoLocal * fneq - force);

    // ========================== FIVE ========================== //

    feq   = W_1 * rho * (1.0f - uu + 3.0f*uz + 4.5f*uz*uz) - W_1;
    force = coeff * feq * (ffz - uf);
    fneq  = W1_45 * (pzz - cspi);

    d.f[global4(x,y,z+1,5)] = toPop(feq + omcoLocal * fneq + force);

    // ========================== SIX ========================== //

    feq   = W_1 * rho * (1.0f - uu - 3.0f*uz + 4.5f*uz*uz) - W_1;
    force = coeff * feq * (ffz + uf);

    d.f[global4(x,y,z-1,6)] = toPop(feq + omcoLocal * fneq - force);

    // ========================== SEVEN ========================== //

    tmp   = ux + uy;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx + ffy - uf);
    fneq  = W2_45 * (pxx + pyy + 2.0f * pxy - cspi);

    d.f[global4(x+1,y+1,z,7)] = toPop(feq + omcoLocal * fneq + force);

    // ========================== EIGHT ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx + ffy + uf);

    d.f[global4(x-1,y-1,z,8)] = toPop(feq + omcoLocal * fneq - force);

    // ========================== NINE ========================== //

    tmp   = ux + uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx + ffz - uf);
    fneq  = W2_45 * (pxx + pzz + 2.0f * pxz - cspi);

    d.f[global4(x+1,y,z+1,9)] = toPop(feq + omcoLocal * fneq + force);

    // ========================== TEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx + ffz + uf);

    d.f[global4(x-1,y,z-1,10)] = toPop(feq + omcoLocal * fneq - force);

    // ========================== ELEVEN ========================== //

    tmp   = uy + uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffy + ffz - uf);
    fneq  = W2_45 * (pyy + pzz + 2.0f * pyz - cspi);

    d.f[global4(x,y+1,z+1,11)] = toPop(feq + omcoLocal * fneq + force);

    // ========================== TWELVE ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffy + ffz + uf);

    d.f[global4(x,y-1,z-1,12)] = toPop(feq + omcoLocal * fneq - force);

    // ========================== THIRTEEN ========================== //

    tmp   = ux - uy;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx - ffy - uf);
    fneq  = W2_45 * (pxx + pyy - 2.0f * pxy - cspi);

    d.f[global4(x+1,y-1,z,13)] = toPop(feq + omcoLocal * fneq + force);

    // ========================== FOURTEEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffy - ffx - uf);

    d.f[global4(x-1,y+1,z,14)] = toPop(feq + omcoLocal * fneq + force);

    // ========================== FIFTEEN ========================== //

    tmp   = ux - uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffx - ffz - uf);
    fneq  = W2_45 * (pxx + pzz - 2.0f * pxz - cspi);

    d.f[global4(x+1,y,z-1,15)] = toPop(feq + omcoLocal * fneq + force);

    // ========================== SIXTEEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffz - ffx - uf);

    d.f[global4(x-1,y,z+1,16)] = toPop(feq + omcoLocal * fneq + force);

    // ========================== SEVENTEEN ========================== //

    tmp   = uy - uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffy - ffz - uf);
    fneq  = W2_45 * (pyy + pzz - 2.0f * pyz - cspi);

    d.f[global4(x,y+1,z-1,17)] = toPop(feq + omcoLocal * fneq + force);

    // ========================== EIGHTEEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp) - W_2;
    force = coeff * feq * (ffz - ffy - uf);

    d.f[global4(x,y-1,z+1,18)] = toPop(feq + omcoLocal * fneq + force);
}

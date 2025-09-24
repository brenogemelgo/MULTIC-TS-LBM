#if defined(D3Q19)
{ 
    float feq, force, fneq;
    const float coeff = (0.5f + 0.5f * omcoLocal) * 3.0f * invRho;
    const float uf = ux*ffx + uy*ffy + uz*ffz;
    const float cspi = CSSQ * (pxx + pyy + pzz);
    const float uu = 1.5f * (ux*ux + uy*uy + uz*uz);

    // ========================== ZERO ========================== //

    feq = W_0 * rho * (1.0f - uu) - W_0;
    force = coeff * feq * uf;
    fneq = W0_45 * cspi;

    d.f[idx3] = to_pop(feq - omcoLocal * fneq - force);

    // ========================== ONE ========================== //

    feq = W_1 * rho * (1.0f - uu + 3.0f*ux + 4.5f*ux*ux) - W_1;
    force = coeff * feq * (ffx - uf);
    fneq = W1_45 * (pxx - cspi);

    d.f[global4(x+1,y,z,1)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== TWO ========================== //

    feq = W_1 * rho * (1.0f - uu - 3.0f*ux + 4.5f*ux*ux) - W_1;
    force = coeff * feq * (ffx + uf);
    // fneq[2] = fneq[1] 

    d.f[global4(x-1,y,z,2)] = to_pop(feq + omcoLocal * fneq - force);

    // ========================== THREE ========================== //

    feq = W_1 * rho * (1.0f - uu + 3.0f*uy + 4.5f*uy*uy) - W_1;
    force = coeff * feq * (ffy - uf);
    fneq = W1_45 * (pyy - cspi);

    d.f[global4(x,y+1,z,3)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== FOUR ========================== //

    feq = W_1 * rho * (1.0f - uu - 3.0f*uy + 4.5f*uy*uy) - W_1;
    force = coeff * feq * (ffy + uf);
    // fneq[4] = fneq[3] 

    d.f[global4(x,y-1,z,4)] = to_pop(feq + omcoLocal * fneq - force);

    // ========================== FIVE ========================== //

    feq = W_1 * rho * (1.0f - uu + 3.0f*uz + 4.5f*uz*uz) - W_1;
    force = coeff * feq * (ffz - uf);
    fneq = W1_45 * (pzz - cspi);

    d.f[global4(x,y,z+1,5)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== SIX ========================== //

    feq = W_1 * rho * (1.0f - uu - 3.0f*uz + 4.5f*uz*uz) - W_1;
    force = coeff * feq * (ffz + uf);
    // fneq[6] = fneq[5]

    d.f[global4(x,y,z-1,6)] = to_pop(feq + omcoLocal * fneq - force);

    // ========================== SEVEN ========================== //

    feq = W_2 * rho * (1.0f - uu + 3.0f * (ux + uy) + 4.5f * (ux + uy) * (ux + uy)) - W_2;
    force = coeff * feq * (ffx + ffy - uf);
    fneq = W2_45 * (pxx + pyy + 2.0f * pxy - cspi);

    d.f[global4(x+1,y+1,z,7)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== EIGHT ========================== //

    feq = W_2 * rho * (1.0f - uu - 3.0f * (ux + uy) + 4.5f * (ux + uy) * (ux + uy)) - W_2;
    force = coeff * feq * (ffx + ffy + uf);
    // fneq[8] = fneq[7]

    d.f[global4(x-1,y-1,z,8)] = to_pop(feq + omcoLocal * fneq - force);

    // ========================== NINE ========================== //

    feq = W_2 * rho * (1.0f - uu + 3.0f * (ux + uz) + 4.5f * (ux + uz) * (ux + uz)) - W_2;
    force = coeff * feq * (ffx + ffz - uf);
    fneq = W2_45 * (pxx + pzz + 2.0f * pxz - cspi);

    d.f[global4(x+1,y,z+1,9)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== TEN ========================== //

    feq = W_2 * rho * (1.0f - uu - 3.0f * (ux + uz) + 4.5f * (ux + uz) * (ux + uz)) - W_2;
    force = coeff * feq * (ffx + ffz + uf);
    // fneq[10] = fneq[9]

    d.f[global4(x-1,y,z-1,10)] = to_pop(feq + omcoLocal * fneq - force);

    // ========================== ELEVEN ========================== //

    feq = W_2 * rho * (1.0f - uu + 3.0f * (uy + uz) + 4.5f * (uy + uz) * (uy + uz)) - W_2;
    force = coeff * feq * (ffy + ffz - uf);
    fneq = W2_45 * (pyy + pzz + 2.0f * pyz - cspi);

    d.f[global4(x,y+1,z+1,11)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== TWELVE ========================== //

    feq = W_2 * rho * (1.0f - uu - 3.0f * (uy + uz) + 4.5f * (uy + uz) * (uy + uz)) - W_2;
    force = coeff * feq * (ffy + ffz + uf);
    // fneq[12] = fneq[11]

    d.f[global4(x,y-1,z-1,12)] = to_pop(feq + omcoLocal * fneq - force);

    // ========================== THIRTEEN ========================== //

    feq = W_2 * rho * (1.0f - uu + 3.0f * (ux - uy) + 4.5f * (ux - uy) * (ux - uy)) - W_2;
    force = coeff * feq * (ffx - ffy - uf);
    fneq = W2_45 * (pxx + pyy - 2.0f * pxy - cspi);

    d.f[global4(x+1,y-1,z,13)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== FOURTEEN ========================== //

    feq = W_2 * rho * (1.0f - uu + 3.0f * (uy - ux) + 4.5f * (uy - ux) * (uy - ux)) - W_2;
    force = coeff * feq * (ffy - ffx - uf);
    // fneq[14] = fneq[13]

    d.f[global4(x-1,y+1,z,14)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== FIFTEEN ========================== //

    feq = W_2 * rho * (1.0f - uu + 3.0f * (ux - uz) + 4.5f * (ux - uz) * (ux - uz)) - W_2;
    force = coeff * feq * (ffx - ffz - uf);
    fneq = W2_45 * (pxx + pzz - 2.0f * pxz - cspi);

    d.f[global4(x+1,y,z-1,15)] = to_pop(feq + omcoLocal * fneq + force); 

    // ========================== SIXTEEN ========================== //

    feq = W_2 * rho * (1.0f - uu + 3.0f * (uz - ux) + 4.5f * (uz - ux) * (uz - ux)) - W_2;
    force = coeff * feq * (ffz - ffx - uf);
    // fneq[16] = fneq[15]

    d.f[global4(x-1,y,z+1,16)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== SEVENTEEN ========================== //

    feq = W_2 * rho * (1.0f - uu + 3.0f * (uy - uz) + 4.5f * (uy - uz) * (uy - uz)) - W_2;
    force = coeff * feq * (ffy - ffz - uf);
    fneq = W2_45 * (pyy + pzz - 2.0f * pyz - cspi);

    d.f[global4(x,y+1,z-1,17)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== EIGHTEEN ========================== //

    feq = W_2 * rho * (1.0f - uu + 3.0f * (uz - uy) + 4.5f * (uz - uy) * (uz - uy)) - W_2;
    force = coeff * feq * (ffz - ffy - uf);
    // fneq[18] = fneq[17]

    d.f[global4(x,y-1,z+1,18)] = to_pop(feq + omcoLocal * fneq + force);
} 
#elif defined(D3Q27) 
{
    float feq, force, fneq;
    const float coeff = (0.5f + 0.5f * omcoLocal);
    const float uf = ux*ffx + uy*ffy + uz*ffz;
    const float cspi = CSSQ * (pxx + pyy + pzz);
    const float uu = 1.5f * (ux*ux + uy*uy + uz*uz);

    // ========================== ZERO ========================== //

    feq = W_0 * rho * (1.0f - uu) - W_0;
    force = 3.0f * coeff * W_0 * uf;
    fneq = W0_45 * cspi;

    d.f[idx3] = to_pop(feq - omcoLocal * fneq - force);

    // ========================== ONE ========================== //

    feq = W_1 * rho * (1.0f - uu + 3.0f * ux + 4.5f * ux*ux + 4.5f * ux*ux*ux - 3.0f*uu*ux) - W_1;

    force = coeff * W_1 * ((3.0f * (1.0f - ux) + 3.0f * 3.0f * ux) * ffx +
                            (-3.0f * uy) * ffy +
                            (-3.0f * uz) * ffz);

    fneq = W1_45 * ((1*1 - CSSQ) * pxx +
                        (0*0 - CSSQ) * pyy +
                        (0*0 - CSSQ) * pzz +
                        2.0f * 1 * 0 * pxy +
                        2.0f * 1 * 0 * pxz +
                        2.0f * 0 * 0 * pyz +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                        3.0f * ((1*1*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                (1*1*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                (1*0*0 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                (0*0*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                (1*0*0 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                (0*0*0 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (1*0*0) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x+1,y,z,1)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== TWO ========================== //

    feq = W_1 * rho * (1.0f
                    - uu
                    - 3.0f * ux
                    + 4.5f * ux * ux
                    - 4.5f * ux * ux * ux
                    + 3.0f*uu*ux) - W_1;

    force = coeff * W_1 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*0) * -1 ) * ffx +
                        (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*0) * 0 ) * ffy +
                        (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*0) * 0 ) * ffz);

    fneq = W1_45 * ((-1*-1 - CSSQ) * pxx +
                        (0*0 - CSSQ) * pyy +
                        (0*0 - CSSQ) * pzz +
                        2.0f * -1 * 0 * pxy +
                        2.0f * -1 * 0 * pxz +
                        2.0f * 0 * 0 * pyz +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                        3.0f * ((-1*-1*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                (-1*-1*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                (-1*0*0 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                (0*0*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                (-1*0*0 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                (0*0*0 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (-1*0*0) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x-1,y,z,2)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== THREE ========================== //

    feq = W_1 * rho * (1.0f
                    - uu
                    + 3.0f * uy
                    + 4.5f * uy * uy
                    + 4.5f * uy * uy * uy
                    - 3.0f*uu*uy) - W_1;

    force = coeff * W_1 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*0) * 0 ) * ffx +
                        (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*0) * 1 ) * ffy +
                        (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*0) * 0 ) * ffz);

    fneq = W1_45 * ((0*0 - CSSQ) * pxx +
                        (1*1 - CSSQ) * pyy +
                        (0*0 - CSSQ) * pzz +
                        2.0f * 0 * 1 * pxy +
                        2.0f * 0 * 0 * pxz +
                        2.0f * 1 * 0 * pyz +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                        3.0f * ((0*0*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                (0*0*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                (0*1*1 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                (1*1*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                (0*0*0 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                (1*0*0 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (0*1*0) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x,y+1,z,3)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== FOUR ========================== //

    feq = W_1 * rho * (1.0f
                    - uu
                    - 3.0f * uy
                    + 4.5f * uy * uy
                    - 4.5f * uy * uy * uy
                    + 3.0f*uu*uy) - W_1;

    force = coeff * W_1 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*0) * 0 ) * ffx +
                        (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*0) * -1 ) * ffy +
                        (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*0) * 0 ) * ffz);

    fneq = W1_45 * ((0*0 - CSSQ) * pxx +
                        (-1*-1 - CSSQ) * pyy +
                        (0*0 - CSSQ) * pzz +
                        2.0f * 0 * -1 * pxy +
                        2.0f * 0 * 0 * pxz +
                        2.0f * -1 * 0 * pyz +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                        3.0f * ((0*0*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                (0*0*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                (0*-1*-1 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                (-1*-1*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                (0*0*0 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                (-1*0*0 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (0*-1*0) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x,y-1,z,4)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== FIVE ========================== //

    feq = W_1 * rho * (1.0f
                - uu
                + 3.0f * uz
                + 4.5f * uz * uz
                + 4.5f * uz * uz * uz
                - 3.0f*uu*uz) - W_1;

    force = coeff * W_1 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*1) * 0 ) * ffx +
                        (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*1) * 0 ) * ffy +
                        (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*1) * 1 ) * ffz);

    fneq = W1_45 * ((0*0 - CSSQ) * pxx +
                        (0*0 - CSSQ) * pyy +
                        (1*1 - CSSQ) * pzz +
                        2.0f * 0 * 0 * pxy +
                        2.0f * 0 * 1 * pxz +
                        2.0f * 0 * 1 * pyz +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                        3.0f * ((0*0*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                (0*0*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                (0*0*0 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                (0*0*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                (0*1*1 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                (0*1*1 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (0*0*1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x,y,z+1,5)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== SIX ========================== //

    feq = W_1 * rho * (1.0f
                    - uu
                    - 3.0f * uz
                    + 4.5f * uz * uz
                    - 4.5f * uz * uz * uz
                    + 3.0f*uu*uz) - W_1;

    force = coeff * W_1 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*-1) * 0 ) * ffx +
                        (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*-1) * 0 ) * ffy +
                        (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*0 + uy*0 + uz*-1) * -1 ) * ffz);

    fneq = W1_45 * ((0*0 - CSSQ) * pxx +
                        (0*0 - CSSQ) * pyy +
                        (-1*-1 - CSSQ) * pzz +
                        2.0f * 0 * 0 * pxy +
                        2.0f * 0 * -1 * pxz +
                        2.0f * 0 * -1 * pyz +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                        3.0f * ((0*0*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                (0*0*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                (0*0*0 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                (0*0*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                (0*-1*-1 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                (0*-1*-1 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (0*0*-1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x,y,z-1,6)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== SEVEN ========================== //

    feq = W_2 * rho * (1.0f
                    - uu
                    + 3.0f * (ux + uy)
                    + 4.5f * (ux + uy) * (ux + uy)
                    + 4.5f * (ux + uy) * (ux + uy) * (ux + uy)
                    - 3.0f * uu * (ux + uy)) - W_2;

    force = coeff * W_2 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*0) * 1 ) * ffx +
                        (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*0) * 1 ) * ffy +
                        (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*0) * 0 ) * ffz);

    fneq = W2_45 * ((1*1 - CSSQ) * pxx +
                        (1*1 - CSSQ) * pyy +
                        (0*0 - CSSQ) * pzz +
                        2.0f * 1 * 1 * pxy +
                        2.0f * 1 * 0 * pxz +
                        2.0f * 1 * 0 * pyz +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                        3.0f * ((1*1*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                (1*1*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                (1*1*1 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                (1*1*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                (1*0*0 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                (1*0*0 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (1*1*0) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x+1,y+1,z,7)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== EIGHT ========================== //

    feq = W_2 * rho * (1.0f
                    - uu
                    - 3.0f * (ux + uy)
                    + 4.5f * (ux + uy) * (ux + uy)
                    - 4.5f * (ux + uy) * (ux + uy) * (ux + uy)
                    + 3.0f * uu * (ux + uy)) - W_2;

    force = coeff * W_2 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*0) * -1 ) * ffx +
                        (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*0) * -1 ) * ffy +
                        (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*0) * 0 ) * ffz);

    fneq = W2_45 * ((-1*-1 - CSSQ) * pxx +
                        (-1*-1 - CSSQ) * pyy +
                        (0*0 - CSSQ) * pzz +
                        2.0f * -1 * -1 * pxy +
                        2.0f * -1 * 0 * pxz +
                        2.0f * -1 * 0 * pyz +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                        3.0f * ((-1*-1*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                (-1*-1*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                (-1*-1*-1 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                (-1*-1*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                (-1*0*0 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                (-1*0*0 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (-1*-1*0) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x-1,y-1,z,8)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== NINE ========================== //

    feq = W_2 * rho * (1.0f
                    - uu
                    + 3.0f * (ux + uz)
                    + 4.5f * (ux + uz) * (ux + uz)
                    + 4.5f * (ux + uz) * (ux + uz) * (ux + uz)
                    - 3.0f * uu * (ux + uz)) - W_2;

    force = coeff * W_2 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*1) * 1 ) * ffx +
                        (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*1) * 0 ) * ffy +
                        (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*1) * 1 ) * ffz);

    fneq = W2_45 * ((1*1 - CSSQ) * pxx +
                        (0*0 - CSSQ) * pyy +
                        (1*1 - CSSQ) * pzz +
                        2.0f * 1 * 0 * pxy +
                        2.0f * 1 * 1 * pxz +
                        2.0f * 0 * 1 * pyz +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                        3.0f * ((1*1*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                (1*1*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                (1*0*0 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                (0*0*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                (1*1*1 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                (0*1*1 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (1*0*1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x+1,y,z+1,9)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== TEN ========================== //

    feq = W_2 * rho * (1.0f
                    - uu
                    - 3.0f * (ux + uz)
                    + 4.5f * (ux + uz) * (ux + uz)
                    - 4.5f * (ux + uz) * (ux + uz) * (ux + uz)
                    + 3.0f * uu * (ux + uz)) - W_2;

    force = coeff * W_2 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*-1) * -1 ) * ffx +
                        (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*-1) * 0 ) * ffy +
                        (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*-1) * -1 ) * ffz);

    fneq = W2_45 * ((-1*-1 - CSSQ) * pxx +
                        (0*0 - CSSQ) * pyy +
                        (-1*-1 - CSSQ) * pzz +
                        2.0f * -1 * 0 * pxy +
                        2.0f * -1 * -1 * pxz +
                        2.0f * 0 * -1 * pyz +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                        3.0f * ((-1*-1*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                (-1*-1*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                (-1*0*0 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                (0*0*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                (-1*-1*-1 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                (0*-1*-1 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (-1*0*-1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x-1,y,z-1,10)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== ELEVEN ========================== //

    feq = W_2 * rho * (1.0f
                    - uu
                    + 3.0f * (uy + uz)
                    + 4.5f * (uy + uz) * (uy + uz)
                    + 4.5f * (uy + uz) * (uy + uz) * (uy + uz)
                    - 3.0f * uu * (uy + uz)) - W_2;

    force = coeff * W_2 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*1) * 0 ) * ffx +
                        (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*1) * 1 ) * ffy +
                        (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*1) * 1 ) * ffz);

    fneq = W2_45 * ((0*0 - CSSQ) * pxx +
                        (1*1 - CSSQ) * pyy +
                        (1*1 - CSSQ) * pzz +
                        2.0f * 0 * 1 * pxy +
                        2.0f * 0 * 1 * pxz +
                        2.0f * 1 * 1 * pyz +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                        3.0f * ((0*0*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                (0*0*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                (0*1*1 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                (1*1*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                (0*1*1 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                (1*1*1 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (0*1*1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x,y+1,z+1,11)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== TWELVE ========================== //

    feq = W_2 * rho * (1.0f
                    - uu
                    - 3.0f * (uy + uz)
                    + 4.5f * (uy + uz) * (uy + uz)
                    - 4.5f * (uy + uz) * (uy + uz) * (uy + uz)
                    + 3.0f * uu * (uy + uz)) - W_2;

    force = coeff * W_2 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*-1) * 0 ) * ffx +
                        (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*-1) * -1 ) * ffy +
                        (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*-1) * -1 ) * ffz);

    fneq = W2_45 * ((0*0 - CSSQ) * pxx +
                        (-1*-1 - CSSQ) * pyy +
                        (-1*-1 - CSSQ) * pzz +
                        2.0f * 0 * -1 * pxy +
                        2.0f * 0 * -1 * pxz +
                        2.0f * -1 * -1 * pyz +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                        3.0f * ((0*0*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                (0*0*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                (0*-1*-1 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                (-1*-1*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                (0*-1*-1 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                (-1*-1*-1 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (0*-1*-1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x,y-1,z-1,12)] = to_pop(feq + omcoLocal * fneq + force);
    
    // ========================== THIRTEEN ========================== //

    feq = W_2 * rho * (1.0f
                    - uu
                    + 3.0f * (ux - uy)
                    + 4.5f * (ux - uy) * (ux - uy)
                    + 4.5f * (ux - uy) * (ux - uy) * (ux - uy)
                    - 3.0f * uu * (ux - uy)) - W_2;

    force = coeff * W_2 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*0) * 1 ) * ffx +
                        (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*0) * -1 ) * ffy +
                        (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*0) * 0 ) * ffz);

    fneq = W2_45 * ((1*1 - CSSQ) * pxx +
                        (-1*-1 - CSSQ) * pyy +
                        (0*0 - CSSQ) * pzz +
                        2.0f * 1 * -1 * pxy +
                        2.0f * 1 * 0 * pxz +
                        2.0f * -1 * 0 * pyz +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                        3.0f * ((1*1*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                (1*1*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                (1*-1*-1 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                (-1*-1*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                (1*0*0 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                (-1*0*0 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (1*-1*0) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x+1,y-1,z,13)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== FOURTEEN ========================== //

    feq = W_2 * rho * (1.0f
                    - uu
                    + 3.0f * (uy - ux)
                    + 4.5f * (uy - ux) * (uy - ux)
                    + 4.5f * (uy - ux) * (uy - ux) * (uy - ux)
                    - 3.0f * uu * (uy - ux)) - W_2;

    force = coeff * W_2 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*0) * -1 ) * ffx +
                        (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*0) * 1 ) * ffy +
                        (3.0f * (0 - uz) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*0) * 0 ) * ffz);

    fneq = W2_45 * ((-1*-1 - CSSQ) * pxx +
                        (1*1 - CSSQ) * pyy +
                        (0*0 - CSSQ) * pzz +
                        2.0f * -1 * 1 * pxy +
                        2.0f * -1 * 0 * pxz +
                        2.0f * 1 * 0 * pyz +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uz * pzz) +
                        3.0f * ((-1*-1*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                (-1*-1*0 - CSSQ*0) * (pxx*uz + 2.0f*ux*pxz) +
                                (-1*1*1 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                (1*1*0 - CSSQ*0) * (pyy*uz + 2.0f*uy*pyz) +
                                (-1*0*0 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                (1*0*0 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (-1*1*0) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x-1,y+1,z,14)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== FIFTEEN ========================== //

    feq = W_2 * rho * (1.0f
                    - uu
                    + 3.0f * (ux - uz)
                    + 4.5f * (ux - uz) * (ux - uz)
                    + 4.5f * (ux - uz) * (ux - uz) * (ux - uz)
                    - 3.0f * uu * (ux - uz)) - W_2;

    force = coeff * W_2 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*-1) * 1 ) * ffx +
                        (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*-1) * 0 ) * ffy +
                        (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*1 + uy*0 + uz*-1) * -1 ) * ffz);

    fneq = W2_45 * ((1*1 - CSSQ) * pxx +
                        (0*0 - CSSQ) * pyy +
                        (-1*-1 - CSSQ) * pzz +
                        2.0f * 1 * 0 * pxy +
                        2.0f * 1 * -1 * pxz +
                        2.0f * 0 * -1 * pyz +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                        3.0f * ((1*1*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                (1*1*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                (1*0*0 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                (0*0*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                (1*-1*-1 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                (0*-1*-1 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (1*0*-1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x+1,y,z-1,15)] = to_pop(feq + omcoLocal * fneq + force); 

    // ========================== SIXTEEN ========================== //

    feq = W_2 * rho * (1.0f
                    - uu
                    + 3.0f * (uz - ux)
                    + 4.5f * (uz - ux) * (uz - ux)
                    + 4.5f * (uz - ux) * (uz - ux) * (uz - ux)
                    - 3.0f * uu * (uz - ux)) - W_2;

    force = coeff * W_2 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*1) * -1 ) * ffx +
                        (3.0f * (0 - uy) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*1) * 0 ) * ffy +
                        (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*-1 + uy*0 + uz*1) * 1 ) * ffz);

    fneq = W2_45 * ((-1*-1 - CSSQ) * pxx +
                        (0*0 - CSSQ) * pyy +
                        (1*1 - CSSQ) * pzz +
                        2.0f * -1 * 0 * pxy +
                        2.0f * -1 * 1 * pxz +
                        2.0f * 0 * 1 * pyz +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * uy * pyy) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                        3.0f * ((-1*-1*0 - CSSQ*0) * (pxx*uy + 2.0f*ux*pxy) +
                                (-1*-1*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                (-1*0*0 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                (0*0*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                (-1*1*1 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                (0*1*1 - CSSQ*0) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (-1*0*1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x-1,y,z+1,16)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== SEVENTEEN ========================== //

    feq = W_2 * rho * (1.0f
                    - uu
                    + 3.0f * (uy - uz)
                    + 4.5f * (uy - uz) * (uy - uz)
                    + 4.5f * (uy - uz) * (uy - uz) * (uy - uz)
                    - 3.0f * uu * (uy - uz)) - W_2;

    force = coeff * W_2 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*-1) * 0 ) * ffx +
                        (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*-1) * 1 ) * ffy +
                        (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*0 + uy*1 + uz*-1) * -1 ) * ffz);

    fneq = W2_45 * ((0*0 - CSSQ) * pxx +
                        (1*1 - CSSQ) * pyy +
                        (-1*-1 - CSSQ) * pzz +
                        2.0f * 0 * 1 * pxy +
                        2.0f * 0 * -1 * pxz +
                        2.0f * 1 * -1 * pyz +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                        3.0f * ((0*0*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                (0*0*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                (0*1*1 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                (1*1*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                (0*-1*-1 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                (1*-1*-1 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (0*1*-1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x,y+1,z-1,17)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== EIGHTEEN ========================== //

    feq = W_2 * rho * (1.0f
                    - uu
                    + 3.0f * (uz - uy)
                    + 4.5f * (uz - uy) * (uz - uy)
                    + 4.5f * (uz - uy) * (uz - uy) * (uz - uy)
                    - 3.0f * uu * (uz - uy)) - W_2;

    force = coeff * W_2 * ((3.0f * (0 - ux) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*1) * 0 ) * ffx +
                        (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*1) * -1 ) * ffy +
                        (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*0 + uy*-1 + uz*1) * 1 ) * ffz);

    fneq = W2_45 * ((0*0 - CSSQ) * pxx +
                        (-1*-1 - CSSQ) * pyy +
                        (1*1 - CSSQ) * pzz +
                        2.0f * 0 * -1 * pxy +
                        2.0f * 0 * 1 * pxz +
                        2.0f * -1 * 1 * pyz +
                        (0*0*0 - 3.0f*CSSQ*0) * (3.0f * ux * pxx) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                        3.0f * ((0*0*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                (0*0*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                (0*-1*-1 - CSSQ*0) * (pxy*uy + 2.0f*ux*pyy) +
                                (-1*-1*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                (0*1*1 - CSSQ*0) * (pxz*uz + 2.0f*ux*pzz) +
                                (-1*1*1 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (0*-1*1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x,y-1,z+1,18)] = to_pop(feq + omcoLocal * fneq + force);
    
    // ========================== NINETEEN ========================== //

    feq = W_3 * rho * (1.0f
                    - uu
                    + 3.0f * (ux + uy + uz)
                    + 4.5f * (ux + uy + uz) * (ux + uy + uz)
                    + 4.5f * (ux + uy + uz) * (ux + uy + uz) * (ux + uy + uz)
                    - 3.0f * uu * (ux + uy + uz)) - W_3;

    force = coeff * W_3 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*1) * 1 ) * ffx +
                        (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*1) * 1 ) * ffy +
                        (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*1) * 1 ) * ffz);

    fneq = W3_45 * ((1*1 - CSSQ) * pxx +
                        (1*1 - CSSQ) * pyy +
                        (1*1 - CSSQ) * pzz +
                        2.0f * 1 * 1 * pxy +
                        2.0f * 1 * 1 * pxz +
                        2.0f * 1 * 1 * pyz +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                        3.0f * ((1*1*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                (1*1*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                (1*1*1 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                (1*1*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                (1*1*1 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                (1*1*1 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (1*1*1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x+1,y+1,z+1,19)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== TWENTY ========================== //

    feq = W_3 * rho * (1.0f
                    - uu
                    - 3.0f * (ux + uy + uz)
                    + 4.5f * (ux + uy + uz) * (ux + uy + uz)
                    - 4.5f * (ux + uy + uz) * (ux + uy + uz) * (ux + uy + uz)
                    + 3.0f * uu * (ux + uy + uz)) - W_3;

    force = coeff * W_3 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*-1) * -1 ) * ffx +
                        (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*-1) * -1 ) * ffy +
                        (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*-1) * -1 ) * ffz);

    fneq = W3_45 * ((-1*-1 - CSSQ) * pxx +
                        (-1*-1 - CSSQ) * pyy +
                        (-1*-1 - CSSQ) * pzz +
                        2.0f * -1 * -1 * pxy +
                        2.0f * -1 * -1 * pxz +
                        2.0f * -1 * -1 * pyz +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                        3.0f * ((-1*-1*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                (-1*-1*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                (-1*-1*-1 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                (-1*-1*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                (-1*-1*-1 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                (-1*-1*-1 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (-1*-1*-1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x-1,y-1,z-1,20)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== TWENTY ONE ========================== //

    feq = W_3 * rho * (1.0f
                    - uu
                    + 3.0f * (ux + uy - uz)
                    + 4.5f * (ux + uy - uz) * (ux + uy - uz)
                    + 4.5f * (ux + uy - uz) * (ux + uy - uz) * (ux + uy - uz)
                    - 3.0f * uu * (ux + uy - uz)) - W_3;        

    force = coeff * W_3 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*-1) * 1 ) * ffx +
                        (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*-1) * 1 ) * ffy +
                        (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*1 + uy*1 + uz*-1) * -1 ) * ffz);

    fneq = W3_45 * ((1*1 - CSSQ) * pxx +
                        (1*1 - CSSQ) * pyy +
                        (-1*-1 - CSSQ) * pzz +
                        2.0f * 1 * 1 * pxy +
                        2.0f * 1 * -1 * pxz +
                        2.0f * 1 * -1 * pyz +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                        3.0f * ((1*1*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                (1*1*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                (1*1*1 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                (1*1*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                (1*-1*-1 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                (1*-1*-1 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (1*1*-1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x+1,y+1,z-1,21)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== TWENTY TWO ========================== //

    feq = W_3 * rho * (1.0f
                    - uu
                    + 3.0f * (uz - ux - uy)
                    + 4.5f * (uz - ux - uy) * (uz - ux - uy)
                    + 4.5f * (uz - ux - uy) * (uz - ux - uy) * (uz - ux - uy)
                    - 3.0f * uu * (uz - ux - uy)) - W_3;

    force = coeff * W_3 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*1) * -1 ) * ffx +
                        (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*1) * -1 ) * ffy +
                        (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*-1 + uy*-1 + uz*1) * 1 ) * ffz);

    fneq = W3_45 * ((-1*-1 - CSSQ) * pxx +
                        (-1*-1 - CSSQ) * pyy +
                        (1*1 - CSSQ) * pzz +
                        2.0f * -1 * -1 * pxy +
                        2.0f * -1 * 1 * pxz +
                        2.0f * -1 * 1 * pyz +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                        3.0f * ((-1*-1*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                (-1*-1*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                (-1*-1*-1 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                (-1*-1*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                (-1*1*1 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                (-1*1*1 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (-1*-1*1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x-1,y-1,z+1,22)] = to_pop(feq + omcoLocal * fneq + force);    

    // ========================== TWENTY THREE ========================== //
    
    feq = W_3 * rho * (1.0f
                    - uu
                    + 3.0f * (ux - uy + uz)
                    + 4.5f * (ux - uy + uz) * (ux - uy + uz)
                    + 4.5f * (ux - uy + uz) * (ux - uy + uz) * (ux - uy + uz)
                    - 3.0f * uu * (ux - uy + uz)) - W_3;

    force = coeff * W_3 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*1) * 1 ) * ffx +
                        (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*1) * -1 ) * ffy +
                        (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*1) * 1 ) * ffz);

    fneq = W3_45 * ((1*1 - CSSQ) * pxx +
                        (-1*-1 - CSSQ) * pyy +
                        (1*1 - CSSQ) * pzz +
                        2.0f * 1 * -1 * pxy +
                        2.0f * 1 * 1 * pxz +
                        2.0f * -1 * 1 * pyz +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                        3.0f * ((1*1*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                (1*1*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                (1*-1*-1 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                (-1*-1*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                (1*1*1 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                (-1*1*1 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (1*-1*1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x+1,y-1,z+1,23)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== TWENTY FOUR ========================== //

    feq = W_3 * rho * (1.0f
                    - uu
                    + 3.0f * (uy - ux - uz)
                    + 4.5f * (uy - ux - uz) * (uy - ux - uz)
                    + 4.5f * (uy - ux - uz) * (uy - ux - uz) * (uy - ux - uz)
                    - 3.0f * uu * (uy - ux - uz)) - W_3;

    force = coeff * W_3 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*-1) * -1 ) * ffx +
                        (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*-1) * 1 ) * ffy +
                        (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*-1) * -1 ) * ffz);

    fneq = W3_45 * ((-1*-1 - CSSQ) * pxx +
                        (1*1 - CSSQ) * pyy +
                        (-1*-1 - CSSQ) * pzz +
                        2.0f * -1 * 1 * pxy +
                        2.0f * -1 * -1 * pxz +
                        2.0f * 1 * -1 * pyz +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                        3.0f * ((-1*-1*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                (-1*-1*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                (-1*1*1 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                (1*1*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                (-1*-1*-1 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                (1*-1*-1 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (-1*1*-1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x-1,y+1,z-1,24)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== TWENTY FIVE ========================== //

    feq = W_3 * rho * (1.0f
                    - uu
                    + 3.0f * (uy - ux + uz)
                    + 4.5f * (uy - ux + uz) * (uy - ux + uz)
                    + 4.5f * (uy - ux + uz) * (uy - ux + uz) * (uy - ux + uz)
                    - 3.0f * uu * (uy - ux + uz)) - W_3;

    force = coeff * W_3 * ((3.0f * (-1 - ux) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*1) * -1 ) * ffx +
                        (3.0f * (1 - uy) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*1) * 1 ) * ffy +
                        (3.0f * (1 - uz) + 3.0f * 3.0f * (ux*-1 + uy*1 + uz*1) * 1 ) * ffz);

    fneq = W3_45 * ((-1*-1 - CSSQ) * pxx +
                        (1*1 - CSSQ) * pyy +
                        (1*1 - CSSQ) * pzz +
                        2.0f * -1 * 1 * pxy +
                        2.0f * -1 * 1 * pxz +
                        2.0f * 1 * 1 * pyz +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * ux * pxx) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uy * pyy) +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * uz * pzz) +
                        3.0f * ((-1*-1*1 - CSSQ*1) * (pxx*uy + 2.0f*ux*pxy) +
                                (-1*-1*1 - CSSQ*1) * (pxx*uz + 2.0f*ux*pxz) +
                                (-1*1*1 - CSSQ*-1) * (pxy*uy + 2.0f*ux*pyy) +
                                (1*1*1 - CSSQ*1) * (pyy*uz + 2.0f*uy*pyz) +
                                (-1*1*1 - CSSQ*-1) * (pxz*uz + 2.0f*ux*pzz) +
                                (1*1*1 - CSSQ*1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (-1*1*1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x-1,y+1,z+1,25)] = to_pop(feq + omcoLocal * fneq + force);    
    
    // ========================== TWENTY SIX ========================== //

    feq = W_3 * rho * (1.0f
                    - uu
                    + 3.0f * (ux - uy - uz)
                    + 4.5f * (ux - uy - uz) * (ux - uy - uz)
                    + 4.5f * (ux - uy - uz) * (ux - uy - uz) * (ux - uy - uz)
                    - 3.0f * uu * (ux - uy - uz)) - W_3;

    force = coeff * W_3 * ((3.0f * (1 - ux) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*-1) * 1 ) * ffx +
                        (3.0f * (-1 - uy) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*-1) * -1 ) * ffy +
                        (3.0f * (-1 - uz) + 3.0f * 3.0f * (ux*1 + uy*-1 + uz*-1) * -1 ) * ffz);

    fneq = W3_45 * ((1*1 - CSSQ) * pxx +
                        (-1*-1 - CSSQ) * pyy +
                        (-1*-1 - CSSQ) * pzz +
                        2.0f * 1 * -1 * pxy +
                        2.0f * 1 * -1 * pxz +
                        2.0f * -1 * -1 * pyz +
                        (1*1*1 - 3.0f*CSSQ*1) * (3.0f * ux * pxx) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uy * pyy) +
                        (-1*-1*-1 - 3.0f*CSSQ*-1) * (3.0f * uz * pzz) +
                        3.0f * ((1*1*-1 - CSSQ*-1) * (pxx*uy + 2.0f*ux*pxy) +
                                (1*1*-1 - CSSQ*-1) * (pxx*uz + 2.0f*ux*pxz) +
                                (1*-1*-1 - CSSQ*1) * (pxy*uy + 2.0f*ux*pyy) +
                                (-1*-1*-1 - CSSQ*-1) * (pyy*uz + 2.0f*uy*pyz) +
                                (1*-1*-1 - CSSQ*1) * (pxz*uz + 2.0f*ux*pzz) +
                                (-1*-1*-1 - CSSQ*-1) * (pyz*uz + 2.0f*uy*pzz)) +
                        6.0f * (1*-1*-1) * (pxy*uz + ux*pyz + uy*pxz));

    d.f[global4(x+1,y-1,z-1,26)] = to_pop(feq + omcoLocal * fneq + force);
}
#endif 
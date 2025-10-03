float pxx, pyy, pzz, pxy, pxz, pyz;
{ 
    float feq, fneq, force, tmp;
    const float coeff = 1.5f * invRho;
    const float uf = ux*ffx + uy*ffy + uz*ffz;
    const float uu = 1.5f * (ux*ux + uy*uy + uz*uz);  

    // ========================== ONE ========================== //

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - uu + 3.0f*ux + 4.5f*ux*ux);
    force = coeff * feq * (ffx - uf);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - uu + 3.0f*ux + 4.5f*ux*ux + 4.5f*ux*ux*ux - 3.0f*uu*ux);
    #endif
    fneq = pop1 - feq + force;
    pxx = fneq;

    // ========================== TWO ========================== //

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - uu - 3.0f*ux + 4.5f*ux*ux);
    force = coeff * feq * (ffy + uf);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - uu - 3.0f*ux + 4.5f*ux*ux - 4.5f*ux*ux*ux + 3.0f*uu*ux);
    #endif
    fneq = pop2 - feq - force;
    pxx += fneq;

    // ========================== THREE ========================== //

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - uu + 3.0f*uy + 4.5f*uy*uy);
    force = coeff * feq * (ffy - uf);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - uu + 3.0f*uy + 4.5f*uy*uy + 4.5f*uy*uy*uy - 3.0f*uu*uy);
    #endif
    fneq = pop3 - feq + force;
    pyy = fneq;

    // ========================== FOUR ========================== //

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - uu - 3.0f*uy + 4.5f*uy*uy);
    force = coeff * feq * (ffy + uf);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - uu - 3.0f*uy + 4.5f*uy*uy - 4.5f*uy*uy*uy + 3.0f*uu*uy);
    #endif
    fneq = pop4 - feq - force;
    pyy += fneq;

    // ========================== FIVE ========================== //

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - uu + 3.0f*uz + 4.5f*uz*uz);
    force = coeff * feq * (ffz - uf);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - uu + 3.0f*uz + 4.5f*uz*uz + 4.5f*uz*uz*uz - 3.0f*uu*uz);
    #endif
    fneq = pop5 - feq + force;
    pzz = fneq;

    // ========================== SIX ========================== //

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - uu - 3.0f*uz + 4.5f*uz*uz);
    force = coeff * feq * (ffz + uf);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - uu - 3.0f*uz + 4.5f*uz*uz - 4.5f*uz*uz*uz + 3.0f*uu*uz);
    #endif
    fneq = pop6 - feq - force;
    pzz += fneq;

    // ========================== SEVEN ========================== //

    tmp = ux + uy;
    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp);
    force = coeff * feq * (ffx + ffy - uf);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    #endif
    fneq = pop7 - feq + force;
    pxx += fneq; 
    pyy += fneq; 
    pxy = fneq;

    // ========================== EIGHT ========================== //

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp);
    force = coeff * feq * (ffx + ffy + uf);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    #endif
    fneq = pop8 - feq - force;
    pxx += fneq; 
    pyy += fneq; 
    pxy += fneq;

    // ========================== NINE ========================== //

    tmp = ux + uz;
    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp);
    force = coeff * feq * (ffx + ffz - uf);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    #endif
    fneq = pop9 - feq + force;
    pxx += fneq; 
    pzz += fneq; 
    pxz = fneq;

    // ========================== TEN ========================== //

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp);
    force = coeff * feq * (ffx + ffz + uf);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    #endif
    fneq = pop10 - feq - force;
    pxx += fneq; 
    pzz += fneq; 
    pxz += fneq;

    // ========================== ELEVEN ========================== //

    tmp = uy + uz;
    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp);
    force = coeff * feq * (ffy + ffz - uf);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    #endif
    fneq = pop11 - feq + force;
    pyy += fneq;
    pzz += fneq; 
    pyz = fneq;

    // ========================== TWELVE ========================== //

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp);
    force = coeff * feq * (ffy + ffz + uf);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    #endif
    fneq = pop12 - feq - force;
    pyy += fneq; 
    pzz += fneq; 
    pyz += fneq;

    // ========================== THIRTEEN ========================== //

    tmp = ux - uy;
    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp);
    force = coeff * feq * (ffx - ffy - uf);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    #endif
    fneq = pop13 - feq + force;
    pxx += fneq; 
    pyy += fneq; 
    pxy -= fneq;

    // ========================== FOURTEEN ========================== //

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp);
    force = coeff * feq * (ffy - ffx - uf);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    #endif
    fneq = pop14 - feq + force;
    pxx += fneq; 
    pyy += fneq; 
    pxy -= fneq;

    // ========================== FIFTEEN ========================== //

    tmp = ux - uz;
    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp);
    force = coeff * feq * (ffx - ffz - uf);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    #endif
    fneq = pop15 - feq + force;
    pxx += fneq; 
    pzz += fneq; 
    pxz -= fneq;

    // ========================== SIXTEEN ========================== //

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp);
    force = coeff * feq * (ffz - ffx - uf);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    #endif
    fneq = pop16 - feq + force;
    pxx += fneq; 
    pzz += fneq; 
    pxz -= fneq;

    // ========================== SEVENTEEN ========================== //

    tmp = uy - uz;
    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp);
    force = coeff * feq * (ffy - ffz - uf);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    #endif
    fneq = pop17 - feq + force;
    pyy += fneq; 
    pzz += fneq; 
    pyz -= fneq;

    // ========================== EIGHTEEN ========================== //

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp);
    force = coeff * feq * (ffz - ffy - uf);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    #endif
    fneq = pop18 - feq + force;
    pyy += fneq; 
    pzz += fneq; 
    pyz -= fneq;
    
    #if defined(D3Q27)
    tmp = ux + uy + uz;
    feq = W_3 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    fneq = pop19 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy += fneq; 
    pxz += fneq; 
    pyz += fneq;

    feq = W_3 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    fneq = pop20 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy += fneq; 
    pxz += fneq; 
    pyz += fneq;

    tmp = ux + uy - uz;
    feq = W_3 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    fneq = pop21 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy += fneq; 
    pxz -= fneq; 
    pyz -= fneq;

    feq = W_3 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    fneq = pop22 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy += fneq; 
    pxz -= fneq; 
    pyz -= fneq; 

    tmp = ux - uy + uz;
    feq = W_3 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    fneq = pop23 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy -= fneq; 
    pxz += fneq; 
    pyz -= fneq;

    feq = W_3 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    fneq = pop24 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy -= fneq; 
    pxz += fneq; 
    pyz -= fneq;

    tmp = uy - ux + uz;
    feq = W_3 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    fneq = pop25 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy -= fneq; 
    pxz -= fneq; 
    pyz += fneq;

    feq = W_3 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    fneq = pop26 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pzz += fneq;
    pxy -= fneq; 
    pxz -= fneq; 
    pyz += fneq;
    #endif 
} 

pxx += CSSQ;
pyy += CSSQ;
pzz += CSSQ;
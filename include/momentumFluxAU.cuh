float pxx, pyy, pzz, pxy, pxz, pyz;
{ 
    float feq, fneq, tmp;
    const float uu = 1.5f * (ux*ux + uy*uy + uz*uz);  

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - uu + 3.0f*ux + 4.5f*ux*ux);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - uu + 3.0f*ux + 4.5f*ux*ux + 4.5f*ux*ux*ux - 3.0f*uu*ux);
    #endif
    fneq = pop1 - feq;
    pxx = fneq;

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - uu - 3.0f*ux + 4.5f*ux*ux);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - uu - 3.0f*ux + 4.5f*ux*ux - 4.5f*ux*ux*ux + 3.0f*uu*ux);
    #endif
    fneq = pop2 - feq;
    pxx += fneq;

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - uu + 3.0f*uy + 4.5f*uy*uy);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - uu + 3.0f*uy + 4.5f*uy*uy + 4.5f*uy*uy*uy - 3.0f*uu*uy);
    #endif
    fneq = pop3 - feq;
    pyy = fneq;

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - uu - 3.0f*uy + 4.5f*uy*uy);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - uu - 3.0f*uy + 4.5f*uy*uy - 4.5f*uy*uy*uy + 3.0f*uu*uy);
    #endif
    fneq = pop4 - feq;
    pyy += fneq;

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - uu + 3.0f*uz + 4.5f*uz*uz);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - uu + 3.0f*uz + 4.5f*uz*uz + 4.5f*uz*uz*uz - 3.0f*uu*uz);
    #endif
    fneq = pop5 - feq;
    pzz = fneq;

    #if defined(D3Q19)
    feq = W_1 * rho * (1.0f - uu - 3.0f*uz + 4.5f*uz*uz);
    #elif defined(D3Q27)
    feq = W_1 * rho * (1.0f - uu - 3.0f*uz + 4.5f*uz*uz - 4.5f*uz*uz*uz + 3.0f*uu*uz);
    #endif
    fneq = pop6 - feq;
    pzz += fneq;

    tmp = ux + uy;
    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    #endif
    fneq = pop7 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pxy = fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    #endif
    fneq = pop8 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pxy += fneq;

    tmp = ux + uz;
    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    #endif
    fneq = pop9 - feq;
    pxx += fneq; 
    pzz += fneq; 
    pxz = fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    #endif
    fneq = pop10 - feq;
    pxx += fneq; 
    pzz += fneq; 
    pxz += fneq;

    tmp = uy + uz;
    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    #endif
    fneq = pop11 - feq;
    pyy += fneq;
    pzz += fneq; 
    pyz = fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    #endif
    fneq = pop12 - feq;
    pyy += fneq; 
    pzz += fneq; 
    pyz += fneq;

    tmp = ux - uy;
    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    #endif
    fneq = pop13 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pxy -= fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    #endif
    fneq = pop14 - feq;
    pxx += fneq; 
    pyy += fneq; 
    pxy -= fneq;

    tmp = ux - uz;
    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    #endif
    fneq = pop15 - feq;
    pxx += fneq; 
    pzz += fneq; 
    pxz -= fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    #endif
    fneq = pop16 - feq;
    pxx += fneq; 
    pzz += fneq; 
    pxz -= fneq;

    tmp = uy - uz;
    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp);
    #endif
    fneq = pop17 - feq;
    pyy += fneq; 
    pzz += fneq; 
    pyz -= fneq;

    #if defined(D3Q19)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp);
    #elif defined(D3Q27)
    feq = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp);
    #endif
    fneq = pop18 - feq;
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
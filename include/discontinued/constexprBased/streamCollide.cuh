{ // ====================================== COLLISION-STREAMING ====================================== //
    const float coeff = 0.5f + 0.5f * omcoLocal;
    const float aux = 3.0f * invRho;
    const float uu = ux*ux + uy*uy + uz*uz;

    float feq, fneq, force;

    // ========================== Q0 ========================== //
    feq   = computeFeq<0>(rho,ux,uy,uz,uu);
    force = computeForce<0>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<0>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[idx3] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q1 ========================== //
    feq   = computeFeq<1>(rho,ux,uy,uz,uu);
    force = computeForce<1>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<1>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x+1,y,z,1)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q2 ========================== //
    feq   = computeFeq<2>(rho,ux,uy,uz,uu);
    force = computeForce<2>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<2>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x-1,y,z,2)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q3 ========================== //
    feq   = computeFeq<3>(rho,ux,uy,uz,uu);
    force = computeForce<3>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<3>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x,y+1,z,3)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q4 ========================== //
    feq   = computeFeq<4>(rho,ux,uy,uz,uu);
    force = computeForce<4>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<4>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x,y-1,z,4)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q5 ========================== //
    feq   = computeFeq<5>(rho,ux,uy,uz,uu);
    force = computeForce<5>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<5>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x,y,z+1,5)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q6 ========================== //
    feq   = computeFeq<6>(rho,ux,uy,uz,uu);
    force = computeForce<6>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<6>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x,y,z-1,6)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q7 ========================== //
    feq   = computeFeq<7>(rho,ux,uy,uz,uu);
    force = computeForce<7>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<7>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x+1,y+1,z,7)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q8 ========================== //
    feq   = computeFeq<8>(rho,ux,uy,uz,uu);
    force = computeForce<8>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<8>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x-1,y-1,z,8)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q9 ========================== //
    feq   = computeFeq<9>(rho,ux,uy,uz,uu);
    force = computeForce<9>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<9>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x+1,y,z+1,9)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q10 ========================== //
    feq   = computeFeq<10>(rho,ux,uy,uz,uu);
    force = computeForce<10>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<10>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x-1,y,z-1,10)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q11 ========================== //
    feq   = computeFeq<11>(rho,ux,uy,uz,uu);
    force = computeForce<11>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<11>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x,y+1,z+1,11)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q12 ========================== //
    feq   = computeFeq<12>(rho,ux,uy,uz,uu);
    force = computeForce<12>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<12>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x,y-1,z-1,12)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q13 ========================== //
    feq   = computeFeq<13>(rho,ux,uy,uz,uu);
    force = computeForce<13>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<13>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x+1,y-1,z,13)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q14 ========================== //
    feq   = computeFeq<14>(rho,ux,uy,uz,uu);
    force = computeForce<14>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<14>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x-1,y+1,z,14)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q15 ========================== //
    feq   = computeFeq<15>(rho,ux,uy,uz,uu);
    force = computeForce<15>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<15>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x+1,y,z-1,15)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q16 ========================== //
    feq   = computeFeq<16>(rho,ux,uy,uz,uu);
    force = computeForce<16>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<16>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x-1,y,z+1,16)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q17 ========================== //
    feq   = computeFeq<17>(rho,ux,uy,uz,uu);
    force = computeForce<17>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<17>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x,y+1,z-1,17)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q18 ========================== //
    feq   = computeFeq<18>(rho,ux,uy,uz,uu);
    force = computeForce<18>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<18>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x,y-1,z+1,18)] = to_pop(feq + omcoLocal * fneq + force);

    #if defined(D3Q27)
    // ========================== Q19 ========================== //
    feq   = computeFeq<19>(rho,ux,uy,uz,uu);
    force = computeForce<19>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<19>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x+,y+1,z+1,19)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q20 ========================== //
    feq   = computeFeq<20>(rho,ux,uy,uz,uu);
    force = computeForce<20>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<20>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x-1,y-1,z-1,20)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q21 ========================== //
    feq   = computeFeq<21>(rho,ux,uy,uz,uu);
    force = computeForce<21>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<21>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x+1,y+1,z-1,21)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q22 ========================== //
    feq   = computeFeq<22>(rho,ux,uy,uz,uu);
    force = computeForce<22>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<22>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x-1,y-1,z+1,22)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q23 ========================== //
    feq   = computeFeq<23>(rho,ux,uy,uz,uu);
    force = computeForce<23>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<23>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x+1,y-1,z+1,23)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q24 ========================== //
    feq   = computeFeq<24>(rho,ux,uy,uz,uu);
    force = computeForce<24>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<24>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x-1,y+1,z-1,24)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q25 ========================== //
    feq   = computeFeq<25>(rho,ux,uy,uz,uu);
    force = computeForce<25>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<25>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x-1,y+1,z+1,25)] = to_pop(feq + omcoLocal * fneq + force);

    // ========================== Q26 ========================== //
    feq   = computeFeq<26>(rho,ux,uy,uz,uu);
    force = computeForce<26>(coeff,feq,ux,uy,uz,ffx,ffy,ffz,aux);
    fneq  = computeNeq<26>(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz);
    d.f[global4(x+1,y-1,z-1,26)] = to_pop(feq + omcoLocal * fneq + force);
    #endif
} // ============================================== END ============================================== //
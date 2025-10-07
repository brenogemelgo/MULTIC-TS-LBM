{
    float feq, force, fneq;
    const float aux = 3.0f * invRho;
    const float uu = ux*ux + uy*uy + uz*uz;

    // ========================== ZERO ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,0);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,0);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,0);

    d.f[idx3] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== ONE ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,1);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,1);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,1);

    d.f[global4(x+1,y,z,1)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== TWO ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,2);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,2);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,2);

    d.f[global4(x-1,y,z,2)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== THREE ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,3);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,3);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,3);

    d.f[global4(x,y+1,z,3)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== FOUR ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,4);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,4);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,4);

    d.f[global4(x,y-1,z,4)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== FIVE ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,5);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,5);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,5);

    d.f[global4(x,y,z+1,5)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== SIX ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,6);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,6);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,6);

    d.f[global4(x,y,z-1,6)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== SEVEN ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,7);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,7);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,7);

    d.f[global4(x+1,y+1,z,7)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== EIGHT ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,8);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,8);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,8);

    d.f[global4(x-1,y-1,z,8)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== NINE ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,9);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,9);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,9);

    d.f[global4(x+1,y,z+1,9)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== TEN ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,10);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,10);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,10);

    d.f[global4(x-1,y,z-1,10)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== ELEVEN ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,11);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,11);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,11);

    d.f[global4(x,y+1,z+1,11)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== TWELVE ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,12);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,12);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,12);

    d.f[global4(x,y-1,z-1,12)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== THIRTEEN ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,13);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,13);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,13);

    d.f[global4(x+1,y-1,z,13)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== FOURTEEN ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,14);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,14);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,14);

    d.f[global4(x-1,y+1,z,14)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== FIFTEEN ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,15);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,15);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,15);

    d.f[global4(x+1,y,z-1,15)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== SIXTEEN ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,16);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,16);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,16);

    d.f[global4(x-1,y,z+1,16)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== SEVENTEEN ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,17);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,17);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,17);

    d.f[global4(x,y+1,z-1,17)] = toPop(feq + omcoLocal * fneq + 0.5f * force);

    // ========================== EIGHTEEN ========================== //

    feq   = computeFeq(rho,ux uy,uz,uu,18);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,18);
    fneq  = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,18);

    d.f[global4(x,y-1,z+1,18)] = toPop(feq + omcoLocal * fneq + 0.5f * force);
}

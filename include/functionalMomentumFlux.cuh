{
    float feq, force, fneq;
    const float aux = 3.0f * invRho;
    const float uu = ux*ux + uy*uy + uz*uz;

    // ========================== ONE ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,1);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,1);
    fneq  = pop1 - feq + 0.5f * force;

    pxx += fneq;

    // ========================== TWO ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,2);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,2);
    fneq  = pop2 - feq + 0.5f * force;

    pxx += fneq;

    // ========================== THREE ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,3);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,3);
    fneq  = pop3 - feq + 0.5f * force;

    pyy += fneq;

    // ========================== FOUR ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,4);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,4);
    fneq  = pop4 - feq + 0.5f * force;

    pyy += fneq;

    // ========================== FIVE ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,5);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,5);
    fneq  = pop5 - feq + 0.5f * force;

    pzz += fneq;

    // ========================== SIX ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,6);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,6);
    fneq  = pop6 - feq + 0.5f * force;

    pzz += fneq;

    // ========================== SEVEN ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,7);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,7);
    fneq  = pop7 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pxy += fneq;

    // ========================== EIGHT ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,8);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,8);
    fneq  = pop8 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pxy += fneq;

    // ========================== NINE ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,9);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,9);
    fneq  = pop9 - feq + 0.5f * force;

    pxx += fneq;
    pzz += fneq;
    pxz += fneq;

    // ========================== TEN ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,10);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,10);
    fneq  = pop10 - feq + 0.5f * force;

    pxx += fneq;
    pzz += fneq;
    pxz += fneq;

    // ========================== ELEVEN ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,11);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,11);
    fneq  = pop11 - feq + 0.5f * force;

    pyy += fneq;
    pzz += fneq;
    pyz += fneq;

    // ========================== TWELVE ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,12);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,12);
    fneq  = pop12 - feq + 0.5f * force;

    pyy += fneq;
    pzz += fneq;
    pyz += fneq;

    // ========================== THIRTEEN ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,13);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,13);
    fneq  = pop13 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pxy -= fneq;

    // ========================== FOURTEEN ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,14);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,14);
    fneq  = pop14 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pxy -= fneq;

    // ========================== FIFTEEN ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,15);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,15);
    fneq  = pop15 - feq + 0.5f * force;

    pxx += fneq;
    pzz += fneq;
    pxz -= fneq;

    // ========================== SIXTEEN ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,16);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,16);
    fneq  = pop16 - feq + 0.5f * force;

    pxx += fneq;
    pzz += fneq;
    pxz -= fneq;

    // ========================== SEVENTEEN ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,17);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,17);
    fneq  = pop17 - feq + 0.5f * force;

    pyy += fneq;
    pzz += fneq;
    pyz -= fneq;

    // ========================== EIGHTEEN ========================== //

    feq   = computeFeq(rho,ux,uy,uz,uu,18);
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,18);
    fneq  = pop18 - feq + 0.5f * force;

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

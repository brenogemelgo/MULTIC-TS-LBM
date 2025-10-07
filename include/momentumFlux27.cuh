{
    float feq, fneq, force, tmp;
    const float aux = 0.0f;
    // const float uf = ux*ffx + uy*ffy + uz*ffz;
    const float uu = 1.5f * (ux*ux + uy*uy + uz*uz);

    // ========================== ONE ========================== //

    feq   = W_1 * rho * (1.0f - uu + 3.0f*ux + 4.5f*ux*ux + 4.5f*ux*ux*ux - 3.0f*uu*ux) - W_1;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,1);
    fneq  = pop1 - feq + 0.5f * force;

    pxx += fneq;

    // ========================== TWO ========================== //

    feq   = W_1 * rho * (1.0f - uu - 3.0f*ux + 4.5f*ux*ux - 4.5f*ux*ux*ux + 3.0f*uu*ux) - W_1;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,2);
    fneq  = pop2 - feq + 0.5f * force;

    pxx += fneq;

    // ========================== THREE ========================== //

    feq   = W_1 * rho * (1.0f - uu + 3.0f*uy + 4.5f*uy*uy + 4.5f*uy*uy*uy - 3.0f*uu*uy) - W_1;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,3);
    fneq  = pop3 - feq + 0.5f * force;

    pyy += fneq;

    // ========================== FOUR ========================== //

    feq   = W_1 * rho * (1.0f - uu - 3.0f*uy + 4.5f*uy*uy - 4.5f*uy*uy*uy + 3.0f*uu*uy) - W_1;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,4);
    fneq  = pop4 - feq + 0.5f * force;

    pyy += fneq;

    // ========================== FIVE ========================== //

    feq   = W_1 * rho * (1.0f - uu + 3.0f*uz + 4.5f*uz*uz + 4.5f*uz*uz*uz - 3.0f*uu*uz) - W_1;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,5);
    fneq  = pop5 - feq + 0.5f * force;

    pzz += fneq;

    // ========================== SIX ========================== //

    feq   = W_1 * rho * (1.0f - uu - 3.0f*uz + 4.5f*uz*uz - 4.5f*uz*uz*uz + 3.0f*uu*uz) - W_1;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,6);
    fneq  = pop6 - feq + 0.5f * force;

    pzz += fneq;

    // ========================== SEVEN ========================== //

    tmp   = ux + uy;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp) - W_2;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,7);
    fneq  = pop7 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pxy += fneq;

    // ========================== EIGHT ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp) - W_2;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,8);
    fneq  = pop8 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pxy += fneq;

    // ========================== NINE ========================== //

    tmp   = ux + uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp) - W_2;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,9);
    fneq  = pop9 - feq + 0.5f * force;

    pxx += fneq;
    pzz += fneq;
    pxz += fneq;

    // ========================== TEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp) - W_2;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,10);
    fneq  = pop10 - feq + 0.5f * force;

    pxx += fneq;
    pzz += fneq;
    pxz += fneq;

    // ========================== ELEVEN ========================== //

    tmp   = uy + uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp) - W_2;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,11);
    fneq  = pop11 - feq + 0.5f * force;

    pyy += fneq;
    pzz += fneq;
    pyz += fneq;

    // ========================== TWELVE ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp) - W_2;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,12);
    fneq  = pop12 - feq + 0.5f * force;

    pyy += fneq;
    pzz += fneq;
    pyz += fneq;

    // ========================== THIRTEEN ========================== //

    tmp   = ux - uy;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp) - W_2;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,13);
    fneq  = pop13 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pxy -= fneq;

    // ========================== FOURTEEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp) - W_2;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,14);
    fneq  = pop14 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pxy -= fneq;

    // ========================== FIFTEEN ========================== //

    tmp   = ux - uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp) - W_2;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,15);
    fneq  = pop15 - feq + 0.5f * force;

    pxx += fneq;
    pzz += fneq;
    pxz -= fneq;

    // ========================== SIXTEEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp) - W_2;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,16);
    fneq  = pop16 - feq + 0.5f * force;

    pxx += fneq;
    pzz += fneq;
    pxz -= fneq;

    // ========================== SEVENTEEN ========================== //

    tmp   = uy - uz;
    feq   = W_2 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp) - W_2;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,17);
    fneq  = pop17 - feq + 0.5f * force;

    pyy += fneq;
    pzz += fneq;
    pyz -= fneq;

    // ========================== EIGHTEEN ========================== //

    feq   = W_2 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp) - W_2;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,18);
    fneq  = pop18 - feq + 0.5f * force;

    pyy += fneq;
    pzz += fneq;
    pyz -= fneq;

    // ========================== NINETEEN ========================== //

    tmp   = ux + uy + uz;
    feq   = W_3 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp) - W_3;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,19);
    fneq  = pop19 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pzz += fneq;
    pxy += fneq;
    pxz += fneq;
    pyz += fneq;

    // ========================== TWENTY ========================== //

    feq   = W_3 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp) - W_3;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,20);
    fneq  = pop20 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pzz += fneq;
    pxy += fneq;
    pxz += fneq;
    pyz += fneq;

    // ========================== TWENTY ONE ========================== //

    tmp   = ux + uy - uz;
    feq   = W_3 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp) - W_3;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,21);
    fneq  = pop21 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pzz += fneq;
    pxy += fneq;
    pxz -= fneq;
    pyz -= fneq;

    // ========================== TWENTY TWO ========================== //

    feq   = W_3 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp) - W_3;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,22);
    fneq  = pop22 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pzz += fneq;
    pxy += fneq;
    pxz -= fneq;
    pyz -= fneq;

    // ========================== TWENTY THREE ========================== //

    tmp   = ux - uy + uz;
    feq   = W_3 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp) - W_3;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,23);
    fneq  = pop23 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pzz += fneq;
    pxy -= fneq;
    pxz += fneq;
    pyz -= fneq;

    // ========================== TWENTY FOUR ========================== //

    feq   = W_3 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp) - W_3;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,24);
    fneq  = pop24 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pzz += fneq;
    pxy -= fneq;
    pxz += fneq;
    pyz -= fneq;

    // ========================== TWENTY FIVE ========================== //

    tmp = uy - ux + uz;
    feq   = W_3 * rho * (1.0f - uu + 3.0f*tmp + 4.5f*tmp*tmp + 4.5f*tmp*tmp*tmp - 3.0f*uu*tmp) - W_3;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,25);
    fneq  = pop25 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pzz += fneq;
    pxy -= fneq;
    pxz -= fneq;
    pyz += fneq;

    // ========================== TWENTY SIX ========================== //

    feq   = W_3 * rho * (1.0f - uu - 3.0f*tmp + 4.5f*tmp*tmp - 4.5f*tmp*tmp*tmp + 3.0f*uu*tmp) - W_3;
    force = computeForce(feq,ux,uy,uz,ffx,ffy,ffz,aux,26);
    fneq  = pop26 - feq + 0.5f * force;

    pxx += fneq;
    pyy += fneq;
    pzz += fneq;
    pxy -= fneq;
    pxz -= fneq;
    pyz += fneq;
}
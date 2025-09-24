const float invRhoCssq = 3.0f * invRho;
const float uu = ux*ux + uy*uy + uz*uz;
const float coeffForce = 0.5f + 0.5f * omcoLocal;

feq = computeFeq(rho,ux,uy,uz,uu,0);
float forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,0);
float fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,0);
d.f[idx3] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,1);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,1);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,1);
d.f[global4(x+1,y,z,1)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,2);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,2);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,2);
d.f[global4(x-1,y,z,2)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,3);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,3);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,3);
d.f[global4(x,y+1,z,3)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,4);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,4);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,4);
d.f[global4(x,y-1,z,4)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,5);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,5);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,5);
d.f[global4(x,y,z+1,5)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,6);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,6);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,6);
d.f[global4(x,y,z-1,6)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,7);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,7);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,7);
d.f[global4(x+1,y+1,z,7)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,8);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,8);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,8);
d.f[global4(x-1,y-1,z,8)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,9);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,9);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,9);
d.f[global4(x+1,y,z+1,9)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,10);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,10);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,10);
d.f[global4(x-1,y,z-1,10)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,11);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,11);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,11);
d.f[global4(x,y+1,z+1,11)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,12);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,12);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,12);
d.f[global4(x,y-1,z-1,12)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,13);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,13);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,13);
d.f[global4(x+1,y-1,z,13)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,14);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,14);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,14);
d.f[global4(x-1,y+1,z,14)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,15);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,15);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,15);
d.f[global4(x+1,y,z-1,15)] = to_pop(feq + omcoLocal * fneqReg + forceCorr); 

feq = computeFeq(rho,ux,uy,uz,uu,16);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,16);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,16);
d.f[global4(x-1,y,z+1,16)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,17);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,17);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,17);
d.f[global4(x,y+1,z-1,17)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,uu,18);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,18);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,18);
d.f[global4(x,y-1,z+1,18)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

#if defined(D3Q27)
feq = computeFeq(rho,ux,uy,uz,19);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,19);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,19);
d.f[global4(x+1,y+1,z+1,19)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,20);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,20);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,20);
d.f[global4(x-1,y-1,z-1,20)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,21);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,21);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,21);
d.f[global4(x+1,y+1,z-1,21)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,22);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,22);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,22);
d.f[global4(x-1,y-1,z+1,22)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);    

feq = computeFeq(rho,ux,uy,uz,23);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,23);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,23);
d.f[global4(x+1,y-1,z+1,23)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,24);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,24);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,24);
d.f[global4(x-1,y+1,z-1,24)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);

feq = computeFeq(rho,ux,uy,uz,25);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,25);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,25);
d.f[global4(x-1,y+1,z+1,25)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);    

feq = computeFeq(rho,ux,uy,uz,26);
forceCorr = computeForce(coeffForce,feq,ux,uy,uz,ffx,ffy,ffz,invRhoCssq,26);
fneqReg = computeNeq(pxx,pyy,pzz,pxy,pxz,pyz,ux,uy,uz,26);
d.f[global4(x+1,y-1,z-1,26)] = to_pop(feq + omcoLocal * fneqReg + forceCorr);
#endif 
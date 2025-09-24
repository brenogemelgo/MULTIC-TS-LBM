#if defined(D3Q19)
float feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*ux + 4.5f*ux*ux);
#elif defined(D3Q27)
float feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*ux + 4.5f*ux*ux + 4.5f*ux*ux*ux - 3.0f*ux * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
float fneq = pop1 - feq;
float pxx = fneq;

#if defined(D3Q19)
feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*ux + 4.5f*ux*ux);
#elif defined(D3Q27)
feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*ux + 4.5f*ux*ux - 4.5f*ux*ux*ux + 3.0f*ux * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop2 - feq;
pxx += fneq;

#if defined(D3Q19)
feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*uy + 4.5f*uy*uy);
#elif defined(D3Q27)
feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*uy + 4.5f*uy*uy + 4.5f*uy*uy*uy - 3.0f*uy * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop3 - feq;
float pyy = fneq;

#if defined(D3Q19)
feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*uy + 4.5f*uy*uy);
#elif defined(D3Q27)
feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*uy + 4.5f*uy*uy - 4.5f*uy*uy*uy + 3.0f*uy * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop4 - feq;
pyy += fneq;

#if defined(D3Q19)
feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*uz + 4.5f*uz*uz);
#elif defined(D3Q27)
feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*uz + 4.5f*uz*uz + 4.5f*uz*uz*uz - 3.0f*uz * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop5 - feq;
float pzz = fneq;

#if defined(D3Q19)
feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*uz + 4.5f*uz*uz);
#elif defined(D3Q27)
feq = W_1 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*uz + 4.5f*uz*uz - 4.5f*uz*uz*uz + 3.0f*uz * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop6 - feq;
pzz += fneq;

#if defined(D3Q19)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uy) + 4.5f*(ux + uy)*(ux + uy));
#elif defined(D3Q27)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uy) + 4.5f*(ux + uy)*(ux + uy) + 4.5f*(ux + uy)*(ux + uy)*(ux + uy) - 3.0f*(ux + uy) * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop7 - feq;
pxx += fneq; 
pyy += fneq; 
float pxy = fneq;

#if defined(D3Q19)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uy) + 4.5f*(ux + uy)*(ux + uy));
#elif defined(D3Q27)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uy) + 4.5f*(ux + uy)*(ux + uy) - 4.5f*(ux + uy)*(ux + uy)*(ux + uy) + 3.0f*(ux + uy) * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop8 - feq;
pxx += fneq; 
pyy += fneq; 
pxy += fneq;

#if defined(D3Q19)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uz) + 4.5f*(ux + uz)*(ux + uz));
#elif defined(D3Q27)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uz) + 4.5f*(ux + uz)*(ux + uz) + 4.5f*(ux + uz)*(ux + uz)*(ux + uz) - 3.0f*(ux + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop9 - feq;
pxx += fneq; 
pzz += fneq; 
float pxz = fneq;

#if defined(D3Q19)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uz) + 4.5f*(ux + uz)*(ux + uz));
#elif defined(D3Q27)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uz) + 4.5f*(ux + uz)*(ux + uz) - 4.5f*(ux + uz)*(ux + uz)*(ux + uz) + 3.0f*(ux + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop10 - feq;
pxx += fneq; 
pzz += fneq; 
pxz += fneq;

#if defined(D3Q19)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy + uz) + 4.5f*(uy + uz)*(uy + uz));
#elif defined(D3Q27)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy + uz) + 4.5f*(uy + uz)*(uy + uz) + 4.5f*(uy + uz)*(uy + uz)*(uy + uz) - 3.0f*(uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop11 - feq;
pyy += fneq;
pzz += fneq; 
float pyz = fneq;

#if defined(D3Q19)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(uy + uz) + 4.5f*(uy + uz)*(uy + uz));
#elif defined(D3Q27)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(uy + uz) + 4.5f*(uy + uz)*(uy + uz) - 4.5f*(uy + uz)*(uy + uz)*(uy + uz) + 3.0f*(uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop12 - feq;
pyy += fneq; 
pzz += fneq; 
pyz += fneq;

#if defined(D3Q19)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uy) + 4.5f*(ux - uy)*(ux - uy));
#elif defined(D3Q27)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uy) + 4.5f*(ux - uy)*(ux - uy) + 4.5f*(ux - uy)*(ux - uy)*(ux - uy) - 3.0f*(ux - uy) * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop13 - feq;
pxx += fneq; 
pyy += fneq; 
pxy -= fneq;

#if defined(D3Q19)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - ux) + 4.5f*(uy - ux)*(uy - ux));
#elif defined(D3Q27)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - ux) + 4.5f*(uy - ux)*(uy - ux) + 4.5f*(uy - ux)*(uy - ux)*(uy - ux) - 3.0f*(uy - ux) * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop14 - feq;
pxx += fneq; 
pyy += fneq; 
pxy -= fneq;

#if defined(D3Q19)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uz) + 4.5f*(ux - uz)*(ux - uz));
#elif defined(D3Q27)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uz) + 4.5f*(ux - uz)*(ux - uz) + 4.5f*(ux - uz)*(ux - uz)*(ux - uz) - 3.0f*(ux - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop15 - feq;
pxx += fneq; 
pzz += fneq; 
pxz -= fneq;

#if defined(D3Q19)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - ux) + 4.5f*(uz - ux)*(uz - ux));
fneq = pop16 - feq;
pxx += fneq + CSSQ; 
#elif defined(D3Q27)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - ux) + 4.5f*(uz - ux)*(uz - ux) + 4.5f*(uz - ux)*(uz - ux)*(uz - ux) - 3.0f*(uz - ux) * 1.5f*(ux*ux + uy*uy + uz*uz));
fneq = pop16 - feq;
pxx += fneq; 
#endif
pzz += fneq; 
pxz -= fneq;

#if defined(D3Q19)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - uz) + 4.5f*(uy - uz)*(uy - uz));
#elif defined(D3Q27)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - uz) + 4.5f*(uy - uz)*(uy - uz) + 4.5f*(uy - uz)*(uy - uz)*(uy - uz) - 3.0f*(uy - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
#endif
fneq = pop17 - feq;
pyy += fneq; 
pzz += fneq; 
pyz -= fneq;

#if defined(D3Q19)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - uy) + 4.5f*(uz - uy)*(uz - uy));
fneq = pop18 - feq;
pyy += fneq + CSSQ; 
pzz += fneq + CSSQ;
#elif defined(D3Q27)
feq = W_2 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - uy) + 4.5f*(uz - uy)*(uz - uy) + 4.5f*(uz - uy)*(uz - uy)*(uz - uy) - 3.0f*(uz - uy) * 1.5f*(ux*ux + uy*uy + uz*uz));
fneq = pop18 - feq;
pyy += fneq; 
pzz += fneq; 
#endif
pyz -= fneq;

// THIRD ORDER TERMS
#if defined(D3Q27)
feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uy + uz) + 4.5f*(ux + uy + uz)*(ux + uy + uz) + 4.5f*(ux + uy + uz)*(ux + uy + uz)*(ux + uy + uz) - 3.0f*(ux + uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
fneq = pop19 - feq;
pxx += fneq; 
pyy += fneq; 
pzz += fneq;
pxy += fneq; 
pxz += fneq; 
pyz += fneq;

feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) - 3.0f*(ux + uy + uz) + 4.5f*(ux + uy + uz)*(ux + uy + uz) - 4.5f*(ux + uy + uz)*(ux + uy + uz)*(ux + uy + uz) + 3.0f*(ux + uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
fneq = pop20 - feq;
pxx += fneq; 
pyy += fneq; 
pzz += fneq;
pxy += fneq; 
pxz += fneq; 
pyz += fneq;

feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux + uy - uz) + 4.5f*(ux + uy - uz)*(ux + uy - uz) + 4.5f*(ux + uy - uz)*(ux + uy - uz)*(ux + uy - uz) - 3.0f*(ux + uy - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));    
fneq = pop21 - feq;
pxx += fneq; 
pyy += fneq; 
pzz += fneq;
pxy += fneq; 
pxz -= fneq; 
pyz -= fneq;

feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uz - uy - ux) + 4.5f*(uz - uy - ux)*(uz - uy - ux) + 4.5f*(uz - uy - ux)*(uz - uy - ux)*(uz - uy - ux) - 3.0f*(uz - uy - ux) * 1.5f*(ux*ux + uy*uy + uz*uz));
fneq = pop22 - feq;
pxx += fneq; 
pyy += fneq; 
pzz += fneq;
pxy += fneq; 
pxz -= fneq; 
pyz -= fneq; 

feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uy + uz) + 4.5f*(ux - uy + uz)*(ux - uy + uz) + 4.5f*(ux - uy + uz)*(ux - uy + uz)*(ux - uy + uz) - 3.0f*(ux - uy + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
fneq = pop23 - feq;
pxx += fneq; 
pyy += fneq; 
pzz += fneq;
pxy -= fneq; 
pxz += fneq; 
pyz -= fneq;

feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - ux - uz) + 4.5f*(uy - ux - uz)*(uy - ux - uz) + 4.5f*(uy - ux - uz)*(uy - ux - uz)*(uy - ux - uz) - 3.0f*(uy - ux - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
fneq = pop24 - feq;
pxx += fneq; 
pyy += fneq; 
pzz += fneq;
pxy -= fneq; 
pxz += fneq; 
pyz -= fneq;

feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(uy - ux + uz) + 4.5f*(uy - ux + uz)*(uy - ux + uz) + 4.5f*(uy - ux + uz)*(uy - ux + uz)*(uy - ux + uz) - 3.0f*(uy - ux + uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
fneq = pop25 - feq;
pxx += fneq; 
pyy += fneq; 
pzz += fneq;
pxy -= fneq; 
pxz -= fneq; 
pyz += fneq;

feq = W_3 * rho * (1.0f - 1.5f*(ux*ux + uy*uy + uz*uz) + 3.0f*(ux - uy - uz) + 4.5f*(ux - uy - uz)*(ux - uy - uz) + 4.5f*(ux - uy - uz)*(ux - uy - uz)*(ux - uy - uz) - 3.0f*(ux - uy - uz) * 1.5f*(ux*ux + uy*uy + uz*uz));
fneq = pop26 - feq;
pxx += fneq + CSSQ; 
pyy += fneq + CSSQ; 
pzz += fneq + CSSQ;
pxy -= fneq; 
pxz -= fneq; 
pyz += fneq;
#endif 

//pxx += CSSQ;
//pyy += CSSQ;
//pzz += CSSQ;
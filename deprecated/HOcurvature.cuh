float sumCurvX = W_1 * (d.normx[global3(x+1,y,z)]   - d.normx[global3(x-1,y,z)]) +
                 W_2 * (d.normx[global3(x+1,y+1,z)] - d.normx[global3(x-1,y-1,z)] +
                        d.normx[global3(x+1,y,z+1)] - d.normx[global3(x-1,y,z-1)] +
                        d.normx[global3(x+1,y-1,z)] - d.normx[global3(x-1,y+1,z)] +
                        d.normx[global3(x+1,y,z-1)] - d.normx[global3(x-1,y,z+1)]);

float sumCurvY = W_1 * (d.normy[global3(x,y+1,z)]   - d.normy[global3(x,y-1,z)]) +
                 W_2 * (d.normy[global3(x+1,y+1,z)] - d.normy[global3(x-1,y-1,z)] +
                        d.normy[global3(x,y+1,z+1)] - d.normy[global3(x,y-1,z-1)] +
                        d.normy[global3(x-1,y+1,z)] - d.normy[global3(x+1,y-1,z)] +
                        d.normy[global3(x,y+1,z-1)] - d.normy[global3(x,y-1,z+1)]);

float sumCurvZ = W_1 * (d.normz[global3(x,y,z+1)]   - d.normz[global3(x,y,z-1)]) +
                 W_2 * (d.normz[global3(x+1,y,z+1)] - d.normz[global3(x-1,y,z-1)] +
                        d.normz[global3(x,y+1,z+1)] - d.normz[global3(x,y-1,z-1)] +
                        d.normz[global3(x-1,y,z+1)] - d.normz[global3(x+1,y,z-1)] +
                        d.normz[global3(x,y-1,z+1)] - d.normz[global3(x,y+1,z-1)]);
                        
#if defined(D3Q27)

    sumCurvX += W_3 * (d.normx[global3(x+1,y+1,z+1)] - d.normx[global3(x-1,y-1,z-1)] +
                       d.normx[global3(x+1,y+1,z-1)] - d.normx[global3(x-1,y-1,z+1)] +
                       d.normx[global3(x+1,y-1,z+1)] - d.normx[global3(x-1,y+1,z-1)] +
                       d.normx[global3(x+1,y-1,z-1)] - d.normx[global3(x-1,y+1,z+1)]);

    sumCurvY += W_3 * (d.normy[global3(x+1,y+1,z+1)] - d.normy[global3(x-1,y-1,z-1)] +
                       d.normy[global3(x+1,y+1,z-1)] - d.normy[global3(x-1,y-1,z+1)] +
                       d.normy[global3(x-1,y+1,z-1)] - d.normy[global3(x+1,y-1,z+1)] +
                       d.normy[global3(x-1,y+1,z+1)] - d.normy[global3(x+1,y-1,z-1)]);

    sumCurvZ += W_3 * (d.normz[global3(x+1,y+1,z+1)] - d.normz[global3(x-1,y-1,z-1)] +
                       d.normz[global3(x-1,y-1,z+1)] - d.normz[global3(x+1,y+1,z-1)] +
                       d.normz[global3(x+1,y-1,z+1)] - d.normz[global3(x-1,y+1,z-1)] +
                       d.normz[global3(x-1,y+1,z+1)] - d.normz[global3(x+1,y-1,z-1)]);
                       
#endif

const float curvature = -3.0f * (sumCurvX + sumCurvY + sumCurvZ);
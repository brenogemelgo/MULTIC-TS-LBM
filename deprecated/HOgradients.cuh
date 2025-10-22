float sgx = W_1 * (d.phi[global3(x+1,y,z)]   - d.phi[global3(x-1,y,z)]) +
            W_2 * (d.phi[global3(x+1,y+1,z)] - d.phi[global3(x-1,y-1,z)] +
                   d.phi[global3(x+1,y,z+1)] - d.phi[global3(x-1,y,z-1)] +
                   d.phi[global3(x+1,y-1,z)] - d.phi[global3(x-1,y+1,z)] +
                   d.phi[global3(x+1,y,z-1)] - d.phi[global3(x-1,y,z+1)]);

float sgy = W_1 * (d.phi[global3(x,y+1,z)]   - d.phi[global3(x,y-1,z)]) +
            W_2 * (d.phi[global3(x+1,y+1,z)] - d.phi[global3(x-1,y-1,z)] +
                   d.phi[global3(x,y+1,z+1)] - d.phi[global3(x,y-1,z-1)] +
                   d.phi[global3(x-1,y+1,z)] - d.phi[global3(x+1,y-1,z)] +
                   d.phi[global3(x,y+1,z-1)] - d.phi[global3(x,y-1,z+1)]);

float sgz = W_1 * (d.phi[global3(x,y,z+1)]   - d.phi[global3(x,y,z-1)]) +
            W_2 * (d.phi[global3(x+1,y,z+1)] - d.phi[global3(x-1,y,z-1)] +
                   d.phi[global3(x,y+1,z+1)] - d.phi[global3(x,y-1,z-1)] +
                   d.phi[global3(x-1,y,z+1)] - d.phi[global3(x+1,y,z-1)] +
                   d.phi[global3(x,y-1,z+1)] - d.phi[global3(x,y+1,z-1)]);
                   
#if defined(D3Q27)

    sgx += W_3 * (d.phi[global3(x+1,y+1,z+1)] - d.phi[global3(x-1,y-1,z-1)] +
                  d.phi[global3(x+1,y+1,z-1)] - d.phi[global3(x-1,y-1,z+1)] +
                  d.phi[global3(x+1,y-1,z+1)] - d.phi[global3(x-1,y+1,z-1)] +
                  d.phi[global3(x+1,y-1,z-1)] - d.phi[global3(x-1,y+1,z+1)]);

    sgy += W_3 * (d.phi[global3(x+1,y+1,z+1)] - d.phi[global3(x-1,y-1,z-1)] +
                  d.phi[global3(x+1,y+1,z-1)] - d.phi[global3(x-1,y-1,z+1)] +
                  d.phi[global3(x-1,y+1,z-1)] - d.phi[global3(x+1,y-1,z+1)] +
                  d.phi[global3(x-1,y+1,z+1)] - d.phi[global3(x+1,y-1,z-1)]);

    sgz += W_3 * (d.phi[global3(x+1,y+1,z+1)] - d.phi[global3(x-1,y-1,z-1)] +
                  d.phi[global3(x-1,y-1,z+1)] - d.phi[global3(x+1,y+1,z-1)] +
                  d.phi[global3(x+1,y-1,z+1)] - d.phi[global3(x-1,y+1,z-1)] +
                  d.phi[global3(x-1,y+1,z+1)] - d.phi[global3(x+1,y-1,z-1)]);
                  
#endif

const float gradX = 3.0f * sgx; 
const float gradY = 3.0f * sgy; 
const float gradZ = 3.0f * sgz; 
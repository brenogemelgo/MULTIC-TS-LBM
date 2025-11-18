#pragma once

namespace LBM
{
     __global__ __launch_bounds__(block::size()) void streamCollide(LBMFields d)
     {
          const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
          const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
          const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

          if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz ||
              x == 0 || x == mesh::nx - 1 ||
              y == 0 || y == mesh::ny - 1 ||
              z == 0 || z == mesh::nz - 1)
          {
               return;
          }

          const label_t idx3 = device::global3(x, y, z);

          scalar_t pop[FLINKS];
          pop[0] = from_pop(d.f[idx3]);
          pop[1] = from_pop(d.f[size::plane() + idx3]);
          pop[2] = from_pop(d.f[2 * size::plane() + idx3]);
          pop[3] = from_pop(d.f[3 * size::plane() + idx3]);
          pop[4] = from_pop(d.f[4 * size::plane() + idx3]);
          pop[5] = from_pop(d.f[5 * size::plane() + idx3]);
          pop[6] = from_pop(d.f[6 * size::plane() + idx3]);
          pop[7] = from_pop(d.f[7 * size::plane() + idx3]);
          pop[8] = from_pop(d.f[8 * size::plane() + idx3]);
          pop[9] = from_pop(d.f[9 * size::plane() + idx3]);
          pop[10] = from_pop(d.f[10 * size::plane() + idx3]);
          pop[11] = from_pop(d.f[11 * size::plane() + idx3]);
          pop[12] = from_pop(d.f[12 * size::plane() + idx3]);
          pop[13] = from_pop(d.f[13 * size::plane() + idx3]);
          pop[14] = from_pop(d.f[14 * size::plane() + idx3]);
          pop[15] = from_pop(d.f[15 * size::plane() + idx3]);
          pop[16] = from_pop(d.f[16 * size::plane() + idx3]);
          pop[17] = from_pop(d.f[17 * size::plane() + idx3]);
          pop[18] = from_pop(d.f[18 * size::plane() + idx3]);

#if defined(D3Q27)

          pop[19] = from_pop(d.f[19 * size::plane() + idx3]);
          pop[20] = from_pop(d.f[20 * size::plane() + idx3]);
          pop[21] = from_pop(d.f[21 * size::plane() + idx3]);
          pop[22] = from_pop(d.f[22 * size::plane() + idx3]);
          pop[23] = from_pop(d.f[23 * size::plane() + idx3]);
          pop[24] = from_pop(d.f[24 * size::plane() + idx3]);
          pop[25] = from_pop(d.f[25 * size::plane() + idx3]);
          pop[26] = from_pop(d.f[26 * size::plane() + idx3]);

#endif

          scalar_t rho = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];

#if defined(D3Q27)

          rho += pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];

#endif

          rho += 1.0f;
          d.rho[idx3] = rho;

          const scalar_t ffx = d.ffx[idx3];
          const scalar_t ffy = d.ffy[idx3];
          const scalar_t ffz = d.ffz[idx3];

          const scalar_t invRho = 1.0f / rho;

#if defined(D3Q19)

          scalar_t ux = invRho * (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]);
          scalar_t uy = invRho * (pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]);
          scalar_t uz = invRho * (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]);

#elif defined(D3Q27)

          scalar_t ux = invRho * (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16] + pop[19] - pop[20] + pop[21] - pop[22] + pop[23] - pop[24] + pop[26] - pop[25]);
          scalar_t uy = invRho * (pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18] + pop[19] - pop[20] + pop[21] - pop[22] + pop[24] - pop[23] + pop[25] - pop[26]);
          scalar_t uz = invRho * (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17] + pop[19] - pop[20] + pop[22] - pop[21] + pop[23] - pop[24] + pop[25] - pop[26]);

#endif

          ux += ffx * 0.5f * invRho;
          uy += ffy * 0.5f * invRho;
          uz += ffz * 0.5f * invRho;

          d.ux[idx3] = ux;
          d.uy[idx3] = uy;
          d.uz[idx3] = uz;

          scalar_t pxx = 0.0f, pyy = 0.0f, pzz = 0.0f;
          scalar_t pxy = 0.0f, pxz = 0.0f, pyz = 0.0f;

          const scalar_t uu = 1.5f * (ux * ux + uy * uy + uz * uz);
          device::constexpr_for<0, FLINKS>(
              [&](const auto Q)
              {
                   constexpr scalar_t w = VelocitySet::F<Q>::w;
                   constexpr scalar_t cx = static_cast<scalar_t>(VelocitySet::F<Q>::cx);
                   constexpr scalar_t cy = static_cast<scalar_t>(VelocitySet::F<Q>::cy);
                   constexpr scalar_t cz = static_cast<scalar_t>(VelocitySet::F<Q>::cz);

                   const scalar_t cu = 3.0f * (cx * ux + cy * uy + cz * uz);

#if defined(D3Q19)

                   const scalar_t feq = w * rho * (1.0f - uu + cu + 0.5f * cu * cu) - w;

#elif defined(D3Q27)

                   const scalar_t feq = w * rho * (1.0f - uu + cu + 0.5f * cu * cu + OOS * cu * cu * cu - uu * cu) - w;

#endif

#if defined(D3Q19)

                   const scalar_t force = 1.5f * feq *
                                          ((cx - ux) * ffx +
                                           (cy - uy) * ffy +
                                           (cz - uz) * ffz) *
                                          invRho;

#elif defined(D3Q27)

                   const scalar_t force = 0.5f * w *
                                          ((3.0f * (cx - ux) + 3.0f * cu * cx) * ffx +
                                           (3.0f * (cy - uy) + 3.0f * cu * cy) * ffy +
                                           (3.0f * (cz - uz) + 3.0f * cu * cz) * ffz);

#endif

                   const scalar_t fneq = pop[Q] - (feq - force);

                   pxx += fneq * cx * cx;
                   pyy += fneq * cy * cy;
                   pzz += fneq * cz * cz;
                   pxy += fneq * cx * cy;
                   pxz += fneq * cx * cz;
                   pyz += fneq * cy * cz;
              });

          d.pxx[idx3] = pxx;
          d.pyy[idx3] = pyy;
          d.pzz[idx3] = pzz;
          d.pxy[idx3] = pxy;
          d.pxz[idx3] = pxz;
          d.pyz[idx3] = pyz;

#if defined(VISC_CONTRAST)

          scalar_t omcoLocal;
          {
               const scalar_t phi = d.phi[idx3];
               const scalar_t nuLocal = fmaf(phi, (VISC_OIL - VISC_WATER), VISC_WATER);
               const scalar_t omegaPhys = 1.0f / (0.5f + 3.0f * nuLocal);

#if defined(JET)

               omcoLocal = 1.0f - fminf(omegaPhys, device::cubic_sponge(z));

#elif defined(DROPLET)

               omcoLocal = 1.0f - omegaPhys;

#endif
          }

#else

#if defined(JET)

          const scalar_t omcoLocal = 1.0f - device::cubic_sponge(z);

#elif defined(DROPLET)

          const scalar_t omcoLocal = 1.0f - OMEGA_REF;

#endif

#endif

          device::constexpr_for<0, FLINKS>(
              [&](const auto Q)
              {
                   constexpr scalar_t w = VelocitySet::F<Q>::w;
                   constexpr scalar_t cx = static_cast<scalar_t>(VelocitySet::F<Q>::cx);
                   constexpr scalar_t cy = static_cast<scalar_t>(VelocitySet::F<Q>::cy);
                   constexpr scalar_t cz = static_cast<scalar_t>(VelocitySet::F<Q>::cz);

                   const scalar_t cu = 3.0f * (cx * ux + cy * uy + cz * uz);

#if defined(D3Q19)

                   const scalar_t feq = w * rho * (1.0f - uu + cu + 0.5f * cu * cu) - w;

#elif defined(D3Q27)

                   const scalar_t feq = w * rho * (1.0f - uu + cu + 0.5f * cu * cu + OOS * cu * cu * cu - uu * cu) - w;

#endif

#if defined(D3Q19)

                   const scalar_t force = 1.5f * feq *
                                          ((cx - ux) * ffx +
                                           (cy - uy) * ffy +
                                           (cz - uz) * ffz) *
                                          invRho;

#elif defined(D3Q27)

                   const scalar_t force = 0.5f * w *
                                          ((3.0f * (cx - ux) + 3.0f * cu * cx) * ffx +
                                           (3.0f * (cy - uy) + 3.0f * cu * cy) * ffy +
                                           (3.0f * (cz - uz) + 3.0f * cu * cz) * ffz);

#endif

#if defined(D3Q19)

                   const scalar_t fneq = (w * 4.5f) *
                                         ((cx * cx - cssq()) * pxx +
                                          (cy * cy - cssq()) * pyy +
                                          (cz * cz - cssq()) * pzz +
                                          2.0f * (cx * cy * pxy +
                                                  cx * cz * pxz +
                                                  cy * cz * pyz));

#elif defined(D3Q27)

                   const scalar_t fneq = (w * 4.5f) *
                                         ((cx * cx - cssq()) * pxx +
                                          (cy * cy - cssq()) * pyy +
                                          (cz * cz - cssq()) * pzz +
                                          2.0f * (cx * cy * pxy +
                                                  cx * cz * pxz +
                                                  cy * cz * pyz) +
                                          (cx * cx * cx - 3.0f * cssq() * cx) * (3.0f * ux * pxx) +
                                          (cy * cy * cy - 3.0f * cssq() * cy) * (3.0f * uy * pyy) +
                                          (cz * cz * cz - 3.0f * cssq() * cz) * (3.0f * uz * pzz) +
                                          3.0f * ((cx * cx * cy - cssq() * cy) * (pxx * uy + 2.0f * ux * pxy) +
                                                  (cx * cx * cz - cssq() * cz) * (pxx * uz + 2.0f * ux * pxz) +
                                                  (cx * cy * cy - cssq() * cx) * (pxy * uy + 2.0f * ux * pyy) +
                                                  (cy * cy * cz - cssq() * cz) * (pyy * uz + 2.0f * uy * pyz) +
                                                  (cx * cz * cz - cssq() * cx) * (pxz * uz + 2.0f * ux * pzz) +
                                                  (cy * cz * cz - cssq() * cy) * (pyz * uz + 2.0f * uy * pzz)) +
                                          6.0f * (cx * cy * cz) * (ux * pyz + uy * pxz + uz * pxy));

#endif
                   const label_t xx = x + static_cast<label_t>(VelocitySet::F<Q>::cx);
                   const label_t yy = y + static_cast<label_t>(VelocitySet::F<Q>::cy);
                   const label_t zz = z + static_cast<label_t>(VelocitySet::F<Q>::cz);

                   d.f[device::global4(xx, yy, zz, Q)] = to_pop(feq + omcoLocal * fneq + force);
              });

          const scalar_t phi = d.phi[idx3];
          const scalar_t normx = d.normx[idx3];
          const scalar_t normy = d.normy[idx3];
          const scalar_t normz = d.normz[idx3];
          const scalar_t phiNorm = physics::gamma * phi * (1.0f - phi);

          device::constexpr_for<0, GLINKS>(
              [&](const auto Q)
              {
                   const label_t xx = x + static_cast<label_t>(VelocitySet::G<Q>::cx);
                   const label_t yy = y + static_cast<label_t>(VelocitySet::G<Q>::cy);
                   const label_t zz = z + static_cast<label_t>(VelocitySet::G<Q>::cz);

                   constexpr scalar_t wg = VelocitySet::G<Q>::wg;
                   constexpr scalar_t cx = static_cast<scalar_t>(VelocitySet::G<Q>::cx);
                   constexpr scalar_t cy = static_cast<scalar_t>(VelocitySet::G<Q>::cy);
                   constexpr scalar_t cz = static_cast<scalar_t>(VelocitySet::G<Q>::cz);

                   const scalar_t geq = wg * phi * (1.0f + AS2_P * (cx * ux + cy * uy + cz * uz));
                   const scalar_t hi = wg * phiNorm * (cx * normx + cy * normy + cz * normz);

                   d.g[device::global4(xx, yy, zz, Q)] = geq + hi;
              });
     }
}

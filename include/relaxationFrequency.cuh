#if defined(VISC_CONTRAST)

    float omcoLocal;
    {
        const float phi = d.phi[idx3];
        const float nuLocal = fmaf(phi, (VISC_OIL - VISC_WATER), VISC_WATER);
        const float omegaPhys = 1.0f / (0.5f + 3.0f * nuLocal);

        #if defined(JET)

            omcoLocal = 1.0f - fminf(omegaPhys, cubic_sponge(z));

        #elif defined(DROPLET)

            omcoLocal = 1.0f - omegaPhys;

        #endif
    }

#else

    #if defined(JET)

        const float omcoLocal = 1.0f - cubic_sponge(z);

    #elif defined(DROPLET)

        const float omcoLocal = 1.0f - OMEGA_REF;

    #endif

#endif
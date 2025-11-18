#pragma once

struct LBMFields
{
    // Hydrodynamics
    scalar_t *rho;
    scalar_t *ux;
    scalar_t *uy;
    scalar_t *uz;
    scalar_t *pxx;
    scalar_t *pyy;
    scalar_t *pzz;
    scalar_t *pxy;
    scalar_t *pxz;
    scalar_t *pyz;

    // Phase field
    scalar_t *phi;
    scalar_t *normx;
    scalar_t *normy;
    scalar_t *normz;
    scalar_t *ind;
    scalar_t *ffx;
    scalar_t *ffy;
    scalar_t *ffz;

    // Passive scalar
    scalar_t *chi;

    // Distributions
    pop_t *f;
    scalar_t *g;
    scalar_t *h;
    scalar_t *h_post;
};

LBMFields fields{};

template <typename T, T v>
struct integralConstant
{
    static constexpr const T value = v;
    using value_type = T;
    using type = integralConstant;

    __device__ [[nodiscard]] inline consteval operator value_type() const noexcept
    {
        return value;
    }

    __device__ [[nodiscard]] inline consteval value_type operator()() const noexcept
    {
        return value;
    }
};
#pragma once

struct LBMFields
{
    float *rho;
    float *phi;
    float *ux;
    float *uy;
    float *uz;
    float *pxx;
    float *pyy;
    float *pzz;
    float *pxy;
    float *pxz;
    float *pyz;
    float *normx;
    float *normy;
    float *normz;
    float *ind;
    float *ffx;
    float *ffy;
    float *ffz;
    pop_t *f;
    float *g;
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
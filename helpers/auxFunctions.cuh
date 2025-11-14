#pragma once

namespace block
{
    __host__ __device__ [[nodiscard]] static inline consteval unsigned num_block_x() noexcept
    {
        return (mesh::nx + block::nx - 1u) / block::nx;
    }

    __host__ __device__ [[nodiscard]] static inline consteval unsigned num_block_y() noexcept
    {
        return (mesh::ny + block::ny - 1u) / block::ny;
    }

    __host__ __device__ [[nodiscard]] static inline consteval unsigned size() noexcept
    {
        return block::nx * block::ny * block::nx;
    }
}

namespace physics
{
    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T rho_water() noexcept
    {
        return static_cast<T>(1.0f);
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T rho_oil() noexcept
    {
        return static_cast<T>(0.8f);
    }
}

namespace relaxation
{
#if defined(JET)

// #define VISC_CONTRAST
#if defined(VISC_CONTRAST)

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T visc_water() noexcept
    {
        return (static_cast<T>(physics::u_ref) * static_cast<T>(mesh::diam)) / static_cast<T>(physics::reynolds);
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T visc_oil() noexcept
    {
        return static_cast<T>(10.0f) * visc_water<T>();
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T visc_ref() noexcept
    {
        return visc_water<T>();
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T omega_water() noexcept
    {
        return static_cast<T>(1.0f) / (static_cast<T>(0.5f) + static_cast<T>(3.0f) * visc_water<T>());
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T omega_oil() noexcept
    {
        return static_cast<T>(1.0f) / (static_cast<T>(0.5f) + static_cast<T>(3.0f) * visc_oil<T>());
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T omega_ref() noexcept
    {
        return omega_water<T>();
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T omco_zmin() noexcept
    {
        return static_cast<T>(1.0f) - omega_oil<T>();
    }

#else

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T visc_ref() noexcept
    {
        return (static_cast<T>(physics::u_ref) * static_cast<T>(mesh::diam)) / static_cast<T>(physics::reynolds);
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T omega_ref() noexcept
    {
        return static_cast<T>(1.0f) / (static_cast<T>(0.5f) + static_cast<T>(3.0f) * visc_ref<T>());
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T omco_zmin() noexcept
    {
        return static_cast<T>(1.0f) - omega_ref<T>();
    }

#endif

#elif defined(DROPLET)

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T omega_ref() noexcept
    {
        return static_cast<T>(1.0f) / (static_cast<T>(0.5f) + static_cast<T>(3.0f) * static_cast<T>(physics::visc_ref));
    }

#endif
}

namespace LBM
{
    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T cssq() noexcept
    {
        return static_cast<T>(1.0f) / static_cast<T>(3.0f);
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T omco() noexcept
    {
        return static_cast<T>(1.0f) - relaxation::omega_ref<T>();
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T oos() noexcept
    {
        return static_cast<T>(1.0f) / static_cast<T>(6.0f);
    }
}

namespace math
{
    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T two_pi() noexcept
    {
        return static_cast<T>(2.0f) * static_cast<T>(CUDART_PI_F);
    }
}

namespace size
{
    __host__ __device__ static inline consteval label_t stride() noexcept
    {
        return mesh::nx * mesh::ny;
    }

    __host__ __device__ static inline consteval label_t plane() noexcept
    {
        return mesh::nx * mesh::ny * mesh::nz;
    }
}

#if defined(JET)
namespace sponge
{
    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T K_gain() noexcept
    {
        return static_cast<T>(100.0f);
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T P_gain() noexcept
    {
        return static_cast<T>(3.0f);
    }

    template <typename T = int>
    __host__ __device__ static inline consteval T sponge_cells() noexcept
    {
        return static_cast<T>(mesh::nz / 12u);
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T sponge() noexcept
    {
        return sponge_cells<T>() / (static_cast<T>(mesh::nz) - static_cast<T>(1));
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T z_start() noexcept
    {
        return (static_cast<T>(mesh::nz) - static_cast<T>(1) - sponge_cells<T>()) / (static_cast<T>(mesh::nz) - static_cast<T>(1));
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T inv_nz_m1() noexcept
    {
        return static_cast<T>(1.0f) / (static_cast<T>(mesh::nz) - static_cast<T>(1));
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T inv_sponge() noexcept
    {
        return static_cast<T>(1.0f) / sponge<T>();
    }
}

namespace relaxation
{

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T omega_zmax() noexcept
    {
        return static_cast<T>(1.0f) / (static_cast<T>(0.5f) + static_cast<T>(3.0f) * visc_ref<T>() * (sponge::K_gain<T>() + static_cast<T>(1.0f)));
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T omco_zmax() noexcept
    {
        return static_cast<T>(1.0f) - omega_zmax<T>();
    }

    template <typename T = scalar_t>
    __host__ __device__ static inline consteval T omega_delta() noexcept
    {
        return omega_zmax<T>() - omega_ref<T>();
    }
}
#endif

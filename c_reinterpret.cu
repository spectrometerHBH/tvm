
#ifdef __CUDACC_RTC__
  #include <cuda/std/cstdint>
  using cuda::std::uint8_t;
  using cuda::std::uint16_t;
  using cuda::std::uint32_t;
  using cuda::std::uint64_t;
  using cuda::std::int8_t;
  using cuda::std::int16_t;
  using cuda::std::int32_t;
  using cuda::std::int64_t;

  #include <cuda/std/type_traits>
  namespace std {
    using cuda::std::is_same;
    using cuda::std::is_same_v;
    using cuda::std::is_integral;
    using cuda::std::is_signed;
    using cuda::std::is_unsigned;
    using cuda::std::is_floating_point;
    using cuda::std::enable_if;
    using cuda::std::conditional;
  }

  // NVRTC uses asm/volatile instead of __asm__/__volatile__ (gcc extension).
  #ifndef __asm__
  #define __asm__ asm
  #endif
  #ifndef __volatile__
  #define __volatile__ volatile
  #endif
#else
  #include <cstdint>
  #include <type_traits>
  #include <cuda.h>
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
#endif
__forceinline__ __device__ void tvm_builtin_ptxd_st_release_gpu_global_b32(const void* __addr, uint32_t __value) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" :  : "l"(__addr), "r"(__value) : "memory");
}
extern "C" __global__ void __launch_bounds__(32) kernel_kernel(float* __restrict__ A_ptr, uint* __restrict__ B_ptr);
extern "C" __global__ void __launch_bounds__(32) kernel_kernel(float* __restrict__ A_ptr, uint* __restrict__ B_ptr) {
  int warp_id_in_cta = __shfl_sync((uint)4294967295, 0, 0, 32);
  int v = 0;
  int tx = ((int)threadIdx.x);
    float v_ = A_ptr[((int)threadIdx.x)];
  tvm_builtin_ptxd_st_release_gpu_global_b32((&(B_ptr[((int)threadIdx.x)])), (*(uint *)(&(v_))));
}


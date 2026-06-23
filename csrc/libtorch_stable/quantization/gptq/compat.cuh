/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _compat_cuh
#define _compat_cuh

namespace vllm {
namespace gptq {

// Maxwell (sm_50/sm_52) lacks native fp16 ALU: __hadd/__hmul/__hsub/__hfma and
// their packed half2 variants require sm_53+. Emulate via fp32 convert ->
// compute -> convert (conversions are sm_50-safe). We provide named helpers and
// route the intrinsics to them through macros so the exllama GPTQ kernels below
// (including the half atomics) build unmodified. Defined at namespace top so it
// also covers atomicAdd_half/atomicAdd_half2.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
__device__ __forceinline__ half maxwell_hadd(half a, half b) {
  return __float2half(__half2float(a) + __half2float(b));
}
__device__ __forceinline__ half maxwell_hsub(half a, half b) {
  return __float2half(__half2float(a) - __half2float(b));
}
__device__ __forceinline__ half maxwell_hmul(half a, half b) {
  return __float2half(__half2float(a) * __half2float(b));
}
__device__ __forceinline__ half maxwell_hfma(half a, half b, half c) {
  return __float2half(__half2float(a) * __half2float(b) + __half2float(c));
}
__device__ __forceinline__ half2 maxwell_hadd2(half2 a, half2 b) {
  float2 fa = __half22float2(a);
  float2 fb = __half22float2(b);
  return __float22half2_rn(make_float2(fa.x + fb.x, fa.y + fb.y));
}
__device__ __forceinline__ half2 maxwell_hmul2(half2 a, half2 b) {
  float2 fa = __half22float2(a);
  float2 fb = __half22float2(b);
  return __float22half2_rn(make_float2(fa.x * fb.x, fa.y * fb.y));
}
__device__ __forceinline__ half2 maxwell_hfma2(half2 a, half2 b, half2 c) {
  float2 fa = __half22float2(a);
  float2 fb = __half22float2(b);
  float2 fc = __half22float2(c);
  return __float22half2_rn(make_float2(fa.x * fb.x + fc.x, fa.y * fb.y + fc.y));
}

  #define __hadd maxwell_hadd
  #define __hsub maxwell_hsub
  #define __hmul maxwell_hmul
  #define __hfma maxwell_hfma
  #define __hadd2 maxwell_hadd2
  #define __hmul2 maxwell_hmul2
  #define __hfma2 maxwell_hfma2
#endif

// atomicAdd for half types, to support CC < 7.x

__device__ __forceinline__ void atomicAdd_half(half* address, half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    __half_raw hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    half tmpres = __hadd(hsum, val);
    hsum = __half_raw(tmpres);
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}

// atomicAdd for half2 types

__device__ __forceinline__ void atomicAdd_half2(half2* address, half2 val) {
  unsigned int* address_as_ui = (unsigned int*)address;
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    half2 old_val = *((half2*)&old);
    half2 new_val = __hadd2(old_val, val);
    old = atomicCAS(address_as_ui, assumed, *((unsigned int*)&new_val));
  } while (assumed != old);
}

// Native atomicAdd overloads on modern CUDA:
//   * atomicAdd(__half2*) is declared unconditionally with an internal sm_60
//     fast-path + CAS fallback valid on sm_50 -> never define our own (defining
//     it causes a "more than one instance" ambiguity).
//   * atomicAdd(__half*) is only declared for __CUDA_ARCH__ >= 700, so sm_50/sm_52
//     still need the compat scalar fallback below.
#if defined(__CUDA_ARCH__) || \
    (defined(USE_ROCM) && (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR) < 713)
  #if __CUDA_ARCH__ < 700 || defined(USE_ROCM)

__device__ __forceinline__ void atomicAdd(half* address, half val) {
  atomicAdd_half(address, val);
}

    #if (defined(CUDART_VERSION) && CUDART_VERSION < 10000) || \
        (__CUDA_ARCH__ < 600 && defined(USE_ROCM))
__device__ __forceinline__ void atomicAdd(half2* address, half2 val) {
  atomicAdd_half2(address, val);
}
    #endif

  #endif
#endif

}  // namespace gptq
}  // namespace vllm
#endif

// Maxwell evo sidecar: fused GGUF quantized matvec/matmul built as a
// standalone torch extension so kernel experiments skip the full _C rebuild.
// Wraps the vendored MMVQ/MMQ launchers (with the sm_50 __dp4a fallback from
// ggml-common.h) behind plain ATen ops. Only the quant types present in
// Q4_K_M-family GGUFs are dispatched; anything else returns via exception so
// the caller can fall back to dequant+cuBLAS.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_compat.h"
#include "ggml-common.h"
#include "vecdotq.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"

// quantize_row_q8_1 copied from gguf_kernel.cu (it is not header-exposed).
template <typename scalar_t>
static __global__ void quantize_q8_1(const scalar_t* __restrict__ x,
                                     void* __restrict__ vy, const int kx,
                                     const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx,
                                   const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[off * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

at::Tensor mul_mat_vec_a8(at::Tensor W, at::Tensor X, int64_t type,
                          int64_t row) {
  const int col = X.size(1);
  const int vecs = X.size(0);
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::CUDAGuard device_guard(X.device());
  auto Y = at::empty({vecs, row}, X.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto quant_X =
      at::empty({vecs, padded / 32 * 9}, X.options().dtype(at::kInt));

  AT_DISPATCH_SWITCH(
      X.scalar_type(), "evo_mul_mat_vec_a8",
      AT_DISPATCH_CASE(at::kHalf, [&] {
        using scalar_t = c10::Half;
        quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(),
                                         (void*)quant_X.data_ptr(), col, vecs,
                                         stream);
        switch (type) {
          case 2:
            mul_mat_vec_q4_0_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 3:
            mul_mat_vec_q4_1_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 6:
            mul_mat_vec_q5_0_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 7:
            mul_mat_vec_q5_1_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 8:
            mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 10:
            mul_mat_vec_q2_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 11:
            mul_mat_vec_q3_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 12:
            mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 13:
            mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 14:
            mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          default:
            TORCH_CHECK(false, "evo sidecar: unsupported MMVQ type ", type);
        }
      }));
  return Y;
}

at::Tensor mul_mat_a8(at::Tensor W, at::Tensor X, int64_t type, int64_t row) {
  const int col = X.size(1);
  const int padded = (col + 512 - 1) / 512 * 512;
  const int batch = X.size(0);
  const at::cuda::CUDAGuard device_guard(X.device());
  auto Y = at::empty({batch, row}, X.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto quant_X =
      at::empty({batch, padded / 32 * 9}, X.options().dtype(at::kInt));

  AT_DISPATCH_SWITCH(
      X.scalar_type(), "evo_mul_mat_a8",
      AT_DISPATCH_CASE(at::kHalf, [&] {
        using scalar_t = c10::Half;
        quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(),
                                         (void*)quant_X.data_ptr(), col, batch,
                                         stream);
        switch (type) {
          case 2:
            ggml_mul_mat_q4_0_q8_1_cuda(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
            break;
          case 3:
            ggml_mul_mat_q4_1_q8_1_cuda(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
            break;
          case 6:
            ggml_mul_mat_q5_0_q8_1_cuda(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
            break;
          case 7:
            ggml_mul_mat_q5_1_q8_1_cuda(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
            break;
          case 8:
            ggml_mul_mat_q8_0_q8_1_cuda(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
            break;
          case 10:
            ggml_mul_mat_q2_K_q8_1_cuda(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
            break;
          case 11:
            ggml_mul_mat_q3_K_q8_1_cuda(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
            break;
          case 12:
            ggml_mul_mat_q4_K_q8_1_cuda(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
            break;
          case 13:
            ggml_mul_mat_q5_K_q8_1_cuda(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
            break;
          case 14:
            ggml_mul_mat_q6_K_q8_1_cuda(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
            break;
          default:
            TORCH_CHECK(false, "evo sidecar: unsupported MMQ type ", type);
        }
      }));
  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mul_mat_vec_a8", &mul_mat_vec_a8,
        "fused GGUF quantized matvec (MMVQ, sm_50-safe dp4a)");
  m.def("mul_mat_a8", &mul_mat_a8,
        "fused GGUF quantized matmul (MMQ, sm_50-safe dp4a)");
}

// mmvq v2 for sm_50 — structural port of llama.cpp master's mul_mat_vec_q
// ideas without its infra:
//   * ncols_dst output columns per block: weight ints are loaded once and
//     dotted against every vec's q8 (kills the x{batch} weight re-read that
//     capped batch-8 decode).
//   * rows_per_block with q8 (u) values held in registers across rows
//     (kills the q8 re-read the NO_U probe measured at ~24%).
// q4_K and q6_K only (the two K-quant types in Qwen3.5 Q4_K_M).
#pragma once

#include "ggml-common.h"
#include "vecdotq.cuh"

template <int ncols_dst, int rows_per_block>
static __global__ void mul_mat_vec_q4_K_v2(
        const void * __restrict__ vx, const void * __restrict__ vy,
        void * __restrict__ dst_v, const int ncols, const int nrows,
        const int nvecs_pad_bytes, const int nvecs) {
    // dst is c10::Half-compatible fp16
    __half * dst = (__half *) dst_v;
    const int row0 = rows_per_block * blockIdx.x;

    const int blocks_per_row = ncols / QK_K;         // superblocks per row
    const int slices = QI4_K / VDR_Q4_K_Q8_1_MMVQ;   // 16 iqs slices/block
    // thread t handles slice (t % slices) of superblock stride (t / slices)
    const int slice = threadIdx.x % slices;
    const int b_off = threadIdx.x / slices;          // 0..blockDim.x/slices-1
    const int b_step = blockDim.x / slices;          // superblocks per iter
    const int iqs = VDR_Q4_K_Q8_1_MMVQ * slice;      // 0,2,..30

    const int q8_row_ints = nvecs_pad_bytes / 4;     // q8_1 row stride, ints

    float sum[rows_per_block][ncols_dst];
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r)
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) sum[r][j] = 0.0f;

    const block_q4_K * x = (const block_q4_K *) vx;

    for (int ib = b_off; ib < blocks_per_row; ib += b_step) {
        // ---- q8 side, shared across all rows of this block ----
        const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
        int   u[ncols_dst][2*QR4_K];
        float d8[ncols_dst][QR4_K];
        float dsy[ncols_dst][QR4_K];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const block_q8_1 * bq8_row =
                (const block_q8_1 *)((const int *) vy + j * q8_row_ints);
            const block_q8_1 * bq8 = bq8_row + ib * (QK_K / QK8_1);
#pragma unroll
            for (int i = 0; i < QR4_K; ++i) {
                const block_q8_1 * bq8i = bq8 + bq8_offset + i;
                const float2 ds = __half22float2(bq8i->ds);
                d8[j][i] = ds.x;
                dsy[j][i] = ds.y;
                const int * q8 = (const int *) bq8i->qs + ((iqs/2)%4);
                u[j][2*i+0] = q8[0];
                u[j][2*i+1] = q8[4];
            }
        }
        const float mmask = ((iqs/2) % 4 == 0) ? 1.0f : 0.0f;

        // ---- weight side, loaded once per row, dotted against every vec --
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            const int row = row0 + r;
            if (rows_per_block > 1 && row >= nrows) break;
            const block_q4_K * bq4 = x + row * blocks_per_row + ib;

            const int * q4 = (const int *)(bq4->qs + 16*bq8_offset
                                           + 4*((iqs/2)%4));
            const int v0 = q4[0];
            const int v1 = q4[4];

            const uint16_t * scales = (const uint16_t *) bq4->scales;
            uint16_t aux[2];
            const int jj = bq8_offset/2;
            if (jj < 2) {
                aux[0] = scales[jj+0] & 0x3f3f;
                aux[1] = scales[jj+2] & 0x3f3f;
            } else {
                aux[0] = ((scales[jj+2] >> 0) & 0x0f0f)
                       | ((scales[jj-2] & 0xc0c0) >> 2);
                aux[1] = ((scales[jj+2] >> 4) & 0x0f0f)
                       | ((scales[jj-0] & 0xc0c0) >> 2);
            }
            const uint8_t * sc = (const uint8_t *) aux;
            const uint8_t * m  = sc + 2;
            const float2 dm4f = __half22float2(bq4->dm);

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                float sumf_d = 0.0f;
                float sumf_m = 0.0f;
#pragma unroll
                for (int i = 0; i < QR4_K; ++i) {
                    const int v0i = (v0 >> (4*i)) & 0x0F0F0F0F;
                    const int v1i = (v1 >> (4*i)) & 0x0F0F0F0F;
                    const int dot1 = __dp4a(v1i, u[j][2*i+1],
                                     __dp4a(v0i, u[j][2*i+0], 0));
                    sumf_d += d8[j][i] * (dot1 * sc[i]);
                    sumf_m += mmask * m[i] * dsy[j][i];
                }
                sum[r][j] += dm4f.x*sumf_d - dm4f.y*sumf_m;
            }
        }
    }

    // warp/block reduce: blockDim.x lanes hold partial sums
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            float tmp = sum[r][j];
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
            }
            if (blockDim.x > 32) {
                __shared__ float buf[8][rows_per_block][ncols_dst];
                const int warp = threadIdx.x / 32;
                if ((threadIdx.x & 31) == 0) buf[warp][r][j] = tmp;
                __syncthreads();
                if (threadIdx.x == 0) {
                    for (int w = 1; w < (int)blockDim.x/32; ++w)
                        tmp += buf[w][r][j];
                }
                __syncthreads();
            }
            const int row = row0 + r;
            if (threadIdx.x == 0 && row < nrows) {
                dst[j*nrows + row] = __float2half(tmp);
            }
        }
    }
}

template <int ncols_dst, int rows_per_block>
static __global__ void mul_mat_vec_q6_K_v2(
        const void * __restrict__ vx, const void * __restrict__ vy,
        void * __restrict__ dst_v, const int ncols, const int nrows,
        const int nvecs_pad_bytes, const int nvecs) {
    __half * dst = (__half *) dst_v;
    const int row0 = rows_per_block * blockIdx.x;

    const int blocks_per_row = ncols / QK_K;
    const int slices = QI6_K / VDR_Q6_K_Q8_1_MMVQ;   // 32 slices
    const int slice = threadIdx.x % slices;
    const int b_off = threadIdx.x / slices;
    const int b_step = max((int)blockDim.x / slices, 1);
    const int iqs = VDR_Q6_K_Q8_1_MMVQ * slice;

    const int q8_row_ints = nvecs_pad_bytes / 4;

    float sum[rows_per_block][ncols_dst];
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r)
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) sum[r][j] = 0.0f;

    const block_q6_K * x = (const block_q6_K *) vx;

    for (int ib = b_off; ib < blocks_per_row; ib += b_step) {
        const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2))
                             + (iqs % (QI6_K/2)) / (QI6_K/4);
        const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2))
                               + (iqs % (QI6_K/2)) / (QI6_K/8);
        const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

        int   u[ncols_dst][QR6_K];
        float d8[ncols_dst][QR6_K];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const block_q8_1 * bq8_row =
                (const block_q8_1 *)((const int *) vy + j * q8_row_ints);
            const block_q8_1 * bq8 = bq8_row + ib * (QK_K / QK8_1);
#pragma unroll
            for (int i = 0; i < QR6_K; ++i) {
                const block_q8_1 * bq8i = bq8 + bq8_offset + 2*i;
                d8[j][i] = __low2float(bq8i->ds);
                u[j][i] = ((const int *) bq8i->qs)[iqs % QI8_1];
            }
        }

#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            const int row = row0 + r;
            if (rows_per_block > 1 && row >= nrows) break;
            const block_q6_K * bq6 = x + row * blocks_per_row + ib;

            const int vl = get_int_from_uint8(bq6->ql, iqs);
            const int vh = get_int_from_uint8(bq6->qh,
                            (QI6_K/4) * (iqs / (QI6_K/2))
                            + iqs % (QI6_K/4)) >> vh_shift;
            const int8_t * scales = bq6->scales + scale_offset;
            const float d = __half2float(bq6->d);

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                float sumf = 0.0f;
#pragma unroll
                for (int i = 0; i < QR6_K; ++i) {
                    const int sc = scales[4*i];
                    const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
                    const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
                    const int vi = __vsubss4((vil | vih), 0x20202020);
                    sumf += d8[j][i] * (__dp4a(vi, u[j][i], 0) * sc);
                }
                sum[r][j] += d * sumf;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            float tmp = sum[r][j];
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
            }
            if (blockDim.x > 32) {
                __shared__ float buf[8][rows_per_block][ncols_dst];
                const int warp = threadIdx.x / 32;
                if ((threadIdx.x & 31) == 0) buf[warp][r][j] = tmp;
                __syncthreads();
                if (threadIdx.x == 0) {
                    for (int w = 1; w < (int)blockDim.x/32; ++w)
                        tmp += buf[w][r][j];
                }
                __syncthreads();
            }
            const int row = row0 + r;
            if (threadIdx.x == 0 && row < nrows) {
                dst[j*nrows + row] = __float2half(tmp);
            }
        }
    }
}

// mmvq v3 for sm_50 — dequant-once + FFMA multi-vec.
// v2 is ALU-bound at batch 8: software dp4a (XMAD-emulated int multiplies)
// runs per output column. v3 unpacks each weight tile ONCE to registers and
// multiplies against all <=8 fp16 activation vectors with native fp32 FMA
// (128/SMM/clk on GM107), skipping q8_1 activation quantization entirely.
//
// Structure: 64-thread blocks (2 warps). A block owns ROWS_PER_BLOCK=8 rows
// (warp w -> rows w, w+2, w+4, w+6 of the tile). Per 256-col superblock the
// block stages X (fp16 -> fp32) in bank-padded smem, then each lane unpacks
// its 8 columns of each owned row and FMAs against smem.
// Column mapping per lane copies the vendored dequantize_block_* kernels.
// NOTE: include after ggml-common.h / vecdotq.cuh / dequantize.cuh.
#pragma once

#define V3_ROWS 8
#define V3_THREADS 64

template <int NC>
static __global__ void mul_mat_vec_q4_K_v3(
        const void * __restrict__ vx, const half * __restrict__ X,
        void * __restrict__ dst_v, const int ncols, const int nrows) {
    __half * dst = (__half *) dst_v;
    constexpr int PITCH = NC + 1;
    __shared__ float xs[256 * PITCH];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int il = lane / 8;          // 0..3: 64-col group
    const int ir = lane % 8;          // 0..7: 4-col slice
    const int c0 = 64 * il + 4 * ir;  // low-nibble col base
    const int row0 = V3_ROWS * blockIdx.x;

    const int blocks_per_row = ncols / QK_K;
    const block_q4_K * x = (const block_q4_K *) vx;

    float acc[V3_ROWS / 2][NC];
#pragma unroll
    for (int k = 0; k < V3_ROWS / 2; ++k)
#pragma unroll
        for (int j = 0; j < NC; ++j) acc[k][j] = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ++ib) {
        const int col0 = ib * QK_K;
        // ---- stage X chunk (256 cols x NC vecs) in smem, coalesced ----
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int cc = 0; cc < 256; cc += V3_THREADS) {
                const int c = cc + threadIdx.x;
                xs[c * PITCH + j] =
                    __half2float(X[j * ncols + col0 + c]);
            }
        }
        __syncthreads();

        // ---- unpack all owned rows' weights first (registers), then FMA
        // with x register-blocked so each LDS feeds V3_ROWS/2 FFMAs ----
        float wlo[V3_ROWS / 2][4];
        float whi[V3_ROWS / 2][4];
#pragma unroll
        for (int k = 0; k < V3_ROWS / 2; ++k) {
            const int row = row0 + warp + 2 * k;
            if (row >= nrows) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    wlo[k][l] = 0.0f;
                    whi[k][l] = 0.0f;
                }
                continue;
            }
            const block_q4_K * bq = x + row * blocks_per_row + ib;
            const uint8_t * q = bq->qs + 32 * il + 4 * ir;
            const float dall = __low2float(bq->dm);
            const float dmin = __high2float(bq->dm);
            uint8_t sc, m;
            get_scale_min_k4(2 * il + 0, bq->scales, sc, m);
            const float d1 = dall * sc;
            const float m1 = dmin * m;
            get_scale_min_k4(2 * il + 1, bq->scales, sc, m);
            const float d2 = dall * sc;
            const float m2 = dmin * m;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                wlo[k][l] = d1 * (q[l] & 0xF) - m1;
                whi[k][l] = d2 * (q[l] >> 4) - m2;
            }
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float * xlo = &xs[(c0 + l) * PITCH];
            const float * xhi = &xs[(c0 + 32 + l) * PITCH];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const float xl = xlo[j];
                const float xh = xhi[j];
#pragma unroll
                for (int k = 0; k < V3_ROWS / 2; ++k) {
                    acc[k][j] += wlo[k][l] * xl + whi[k][l] * xh;
                }
            }
        }
    }

    // ---- reduce within warp (lanes cover disjoint cols) and store ----
#pragma unroll
    for (int k = 0; k < V3_ROWS / 2; ++k) {
        const int row = row0 + warp + 2 * k;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float t = acc[k][j];
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                t += __shfl_xor_sync(0xffffffff, t, mask, 32);
            }
            if (lane == 0 && row < nrows) {
                dst[j * nrows + row] = __float2half(t);
            }
        }
    }
}

template <int NC>
static __global__ void mul_mat_vec_q6_K_v3(
        const void * __restrict__ vx, const half * __restrict__ X,
        void * __restrict__ dst_v, const int ncols, const int nrows) {
    __half * dst = (__half *) dst_v;
    constexpr int PITCH = NC + 1;
    __shared__ float xs[256 * PITCH];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int il = lane;              // 0..31
    const int is0 = il / 16;          // scale sub-index within each half
    const int row0 = V3_ROWS * blockIdx.x;

    const int blocks_per_row = ncols / QK_K;
    const block_q6_K * x = (const block_q6_K *) vx;

    float acc[V3_ROWS / 2][NC];
#pragma unroll
    for (int k = 0; k < V3_ROWS / 2; ++k)
#pragma unroll
        for (int j = 0; j < NC; ++j) acc[k][j] = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ++ib) {
        const int col0 = ib * QK_K;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int cc = 0; cc < 256; cc += V3_THREADS) {
                const int c = cc + threadIdx.x;
                xs[c * PITCH + j] =
                    __half2float(X[j * ncols + col0 + c]);
            }
        }
        __syncthreads();

        // unpack all owned rows (8 cols each) into registers, then FMA with
        // x register-blocked so each LDS feeds V3_ROWS/2 FFMAs
        float w[V3_ROWS / 2][2][4];  // [row][ip][col-quarter]
#pragma unroll
        for (int k = 0; k < V3_ROWS / 2; ++k) {
            const int row = row0 + warp + 2 * k;
            if (row >= nrows) {
#pragma unroll
                for (int ip = 0; ip < 2; ++ip)
#pragma unroll
                    for (int l = 0; l < 4; ++l) w[k][ip][l] = 0.0f;
                continue;
            }
            const block_q6_K * bq = x + row * blocks_per_row + ib;
            const float d = __half2float(bq->d);
#pragma unroll
            for (int ip = 0; ip < 2; ++ip) {
                const uint8_t ql0 = bq->ql[64 * ip + il];
                const uint8_t ql32 = bq->ql[64 * ip + il + 32];
                const uint8_t qh = bq->qh[32 * ip + il];
                const int8_t * sc = bq->scales + 8 * ip + is0;
                w[k][ip][0] = d * sc[0]
                    * (int8_t)(((ql0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
                w[k][ip][1] = d * sc[2]
                    * (int8_t)(((ql32 & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
                w[k][ip][2] = d * sc[4]
                    * (int8_t)(((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
                w[k][ip][3] = d * sc[6]
                    * (int8_t)(((ql32 >> 4) | (((qh >> 6) & 3) << 4)) - 32);
            }
        }
#pragma unroll
        for (int ip = 0; ip < 2; ++ip) {
            const int cbase = 128 * ip + il;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const float * xp = &xs[(cbase + 32 * l) * PITCH];
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    const float xv = xp[j];
#pragma unroll
                    for (int k = 0; k < V3_ROWS / 2; ++k) {
                        acc[k][j] += w[k][ip][l] * xv;
                    }
                }
            }
        }
    }

#pragma unroll
    for (int k = 0; k < V3_ROWS / 2; ++k) {
        const int row = row0 + warp + 2 * k;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float t = acc[k][j];
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                t += __shfl_xor_sync(0xffffffff, t, mask, 32);
            }
            if (lane == 0 && row < nrows) {
                dst[j * nrows + row] = __float2half(t);
            }
        }
    }
}

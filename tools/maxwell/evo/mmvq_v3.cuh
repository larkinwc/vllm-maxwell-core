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

#ifndef V3_ROWS
#define V3_ROWS 8
#endif
#ifndef V3_THREADS
#define V3_THREADS 64
#endif
#define V3_NWARP (V3_THREADS / 32)
#define V3_RPW (V3_ROWS / V3_NWARP)  // rows per warp

#if defined(V3_Q4VEC) && V3_Q4VEC
#define V3_Q4_BYTE(word, l) ((uint8_t)(((word) >> (8 * (l))) & 0xffu))
#endif

#if defined(V3_LDG) && V3_LDG
#define V3_LOAD(p) __ldg(p)
#else
#define V3_LOAD(p) (*(p))
#endif

template <int NC>
static __global__ void mul_mat_vec_q4_K_v3(
        const void * __restrict__ vx, const half * __restrict__ X,
        void * __restrict__ dst_v, const int ncols, const int nrows) {
    __half * dst = (__half *) dst_v;
#if defined(V3_LDS128) && V3_LDS128
    constexpr int PITCH = ((NC + 3) / 4) * 4;
#else
    constexpr int PITCH = NC + 1;
#endif
#if defined(V3_DBUF) && V3_DBUF
    __shared__ __align__(16) float xs[2][256 * PITCH];
#else
    __shared__ __align__(16) float xs[256 * PITCH];
#endif

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int il = lane / 8;          // 0..3: 64-col group
    const int ir = lane % 8;          // 0..7: 4-col slice
    const int c0 = 64 * il + 4 * ir;  // low-nibble col base
    const int row0 = V3_ROWS * blockIdx.x;

    const int blocks_per_row = ncols / QK_K;
    const block_q4_K * x = (const block_q4_K *) vx;

    float acc[V3_RPW][NC];
#pragma unroll
    for (int k = 0; k < V3_RPW; ++k)
#pragma unroll
        for (int j = 0; j < NC; ++j) acc[k][j] = 0.0f;

#if defined(V3_DBUF) && V3_DBUF
    float wlo[2][V3_RPW][4];
    float whi[2][V3_RPW][4];
    int cur = 0;
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int cc = 0; cc < 256; cc += V3_THREADS) {
                const int c = cc + threadIdx.x;
                xs[0][c * PITCH + j] =
                    __half2float(X[j * ncols + 0 + c]);
            }
        }
    __syncthreads();
        for (int k = 0; k < V3_RPW; ++k) {
            const int row = row0 + warp + V3_NWARP * k;
            if (row >= nrows) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    wlo[0][k][l] = 0.0f;
                    whi[0][k][l] = 0.0f;
                }
                continue;
            }
            const block_q4_K * bq = x + row * blocks_per_row + 0;
            const uint8_t * q = bq->qs + 32 * il + 4 * ir;
#if defined(V3_Q4VEC) && V3_Q4VEC
            const uint32_t qword = V3_LOAD(reinterpret_cast<const uint32_t *>(q));
#endif
            const half2 dm = V3_LOAD(&bq->dm);
            const float dall = __low2float(dm);
            const float dmin = __high2float(dm);
            uint8_t sc, m;
            get_scale_min_k4(2 * il + 0, bq->scales, sc, m);
            const float d1 = dall * sc;
            const float m1 = dmin * m;
            get_scale_min_k4(2 * il + 1, bq->scales, sc, m);
            const float d2 = dall * sc;
            const float m2 = dmin * m;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
#if defined(V3_Q4VEC) && V3_Q4VEC
                const uint8_t qv = V3_Q4_BYTE(qword, l);
#else
                const uint8_t qv = V3_LOAD(&q[l]);
#endif
                wlo[0][k][l] = d1 * (qv & 0xF) - m1;
                whi[0][k][l] = d2 * (qv >> 4) - m2;
            }
        }
    for (int ib = 0; ib < blocks_per_row; ++ib) {
        const int next_ib = ib + 1;
        const int next = cur ^ 1;
        for (int l = 0; l < 4; ++l) {
            const float * xlo = &xs[cur][(c0 + l) * PITCH];
            const float * xhi = &xs[cur][(c0 + 32 + l) * PITCH];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const float xl = xlo[j];
                const float xh = xhi[j];
#pragma unroll
                for (int k = 0; k < V3_RPW; ++k)
                    acc[k][j] += wlo[cur][k][l] * xl + whi[cur][k][l] * xh;
            }
        }
        if (next_ib < blocks_per_row) {
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int cc = 0; cc < 256; cc += V3_THREADS) {
                const int c = cc + threadIdx.x;
                xs[next][c * PITCH + j] =
                    __half2float(X[j * ncols + next_ib * QK_K + c]);
            }
        }
        for (int k = 0; k < V3_RPW; ++k) {
            const int row = row0 + warp + V3_NWARP * k;
            if (row >= nrows) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    wlo[next][k][l] = 0.0f;
                    whi[next][k][l] = 0.0f;
                }
                continue;
            }
            const block_q4_K * bq = x + row * blocks_per_row + next_ib;
            const uint8_t * q = bq->qs + 32 * il + 4 * ir;
#if defined(V3_Q4VEC) && V3_Q4VEC
            const uint32_t qword = V3_LOAD(reinterpret_cast<const uint32_t *>(q));
#endif
            const half2 dm = V3_LOAD(&bq->dm);
            const float dall = __low2float(dm);
            const float dmin = __high2float(dm);
            uint8_t sc, m;
            get_scale_min_k4(2 * il + 0, bq->scales, sc, m);
            const float d1 = dall * sc;
            const float m1 = dmin * m;
            get_scale_min_k4(2 * il + 1, bq->scales, sc, m);
            const float d2 = dall * sc;
            const float m2 = dmin * m;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
#if defined(V3_Q4VEC) && V3_Q4VEC
                const uint8_t qv = V3_Q4_BYTE(qword, l);
#else
                const uint8_t qv = V3_LOAD(&q[l]);
#endif
                wlo[next][k][l] = d1 * (qv & 0xF) - m1;
                whi[next][k][l] = d2 * (qv >> 4) - m2;
            }
        }
            __syncthreads();
        }
        if (next_ib < blocks_per_row) cur = next;
    }
#else
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
        float wlo[V3_RPW][4];
        float whi[V3_RPW][4];
#pragma unroll
        for (int k = 0; k < V3_RPW; ++k) {
            const int row = row0 + warp + V3_NWARP * k;
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
#if defined(V3_Q4VEC) && V3_Q4VEC
            const uint32_t qword = V3_LOAD(reinterpret_cast<const uint32_t *>(q));
#endif
            const half2 dm = V3_LOAD(&bq->dm);
            const float dall = __low2float(dm);
            const float dmin = __high2float(dm);
            uint8_t sc, m;
            get_scale_min_k4(2 * il + 0, bq->scales, sc, m);
            const float d1 = dall * sc;
            const float m1 = dmin * m;
            get_scale_min_k4(2 * il + 1, bq->scales, sc, m);
            const float d2 = dall * sc;
            const float m2 = dmin * m;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
#if defined(V3_Q4VEC) && V3_Q4VEC
                const uint8_t qv = V3_Q4_BYTE(qword, l);
#else
                const uint8_t qv = V3_LOAD(&q[l]);
#endif
                wlo[k][l] = d1 * (qv & 0xF) - m1;
                whi[k][l] = d2 * (qv >> 4) - m2;
            }
        }
#if defined(V3_LDS128) && V3_LDS128
        if constexpr (NC % 4 == 0) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const float4 * xlo4 = reinterpret_cast<const float4 *>(&xs[(c0 + l) * PITCH]);
                const float4 * xhi4 = reinterpret_cast<const float4 *>(&xs[(c0 + 32 + l) * PITCH]);
#pragma unroll
                for (int jj = 0; jj < NC / 4; ++jj) {
                    const float4 xl = xlo4[jj];
                    const float4 xh = xhi4[jj];
#pragma unroll
                    for (int k = 0; k < V3_RPW; ++k) {
                        acc[k][4 * jj + 0] += wlo[k][l] * xl.x + whi[k][l] * xh.x;
                        acc[k][4 * jj + 1] += wlo[k][l] * xl.y + whi[k][l] * xh.y;
                        acc[k][4 * jj + 2] += wlo[k][l] * xl.z + whi[k][l] * xh.z;
                        acc[k][4 * jj + 3] += wlo[k][l] * xl.w + whi[k][l] * xh.w;
                    }
                }
            }
        } else {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float * xlo = &xs[(c0 + l) * PITCH];
            const float * xhi = &xs[(c0 + 32 + l) * PITCH];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const float xl = xlo[j];
                const float xh = xhi[j];
#pragma unroll
                for (int k = 0; k < V3_RPW; ++k) {
                    acc[k][j] += wlo[k][l] * xl + whi[k][l] * xh;
                }
            }
        }
        }
#else
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float * xlo = &xs[(c0 + l) * PITCH];
            const float * xhi = &xs[(c0 + 32 + l) * PITCH];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const float xl = xlo[j];
                const float xh = xhi[j];
#pragma unroll
                for (int k = 0; k < V3_RPW; ++k) {
                    acc[k][j] += wlo[k][l] * xl + whi[k][l] * xh;
                }
            }
        }
#endif
    }

#endif

    // ---- reduce within warp (lanes cover disjoint cols) and store ----
#pragma unroll
    for (int k = 0; k < V3_RPW; ++k) {
        const int row = row0 + warp + V3_NWARP * k;
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
static __global__ void mul_mat_vec_q8_0_v3(
        const void * __restrict__ vx, const half * __restrict__ X,
        void * __restrict__ dst_v, const int ncols, const int nrows) {
    __half * dst = (__half *) dst_v;
#if defined(V3_LDS128) && V3_LDS128
    constexpr int PITCH = ((NC + 3) / 4) * 4;
#else
    constexpr int PITCH = NC + 1;
#endif
#if defined(V3_DBUF) && V3_DBUF
    __shared__ __align__(16) float xs[2][256 * PITCH];
#else
    __shared__ __align__(16) float xs[256 * PITCH];
#endif

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row0 = V3_ROWS * blockIdx.x;

    // process 256 cols per iteration = 8 q8_0 blocks; lane covers cols
    // {lane, lane+32, ..., lane+224}, one col in each block
    const int chunks_per_row = ncols / 256;
    const block_q8_0 * x = (const block_q8_0 *) vx;

    float acc[V3_RPW][NC];
#pragma unroll
    for (int k = 0; k < V3_RPW; ++k)
#pragma unroll
        for (int j = 0; j < NC; ++j) acc[k][j] = 0.0f;

#if defined(V3_DBUF) && V3_DBUF
    float w[2][V3_RPW][8];
    int cur = 0;
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int cc = 0; cc < 256; cc += V3_THREADS) {
                const int c = cc + threadIdx.x;
                xs[0][c * PITCH + j] =
                    __half2float(X[j * ncols + 0 + c]);
            }
        }
    __syncthreads();
        for (int k = 0; k < V3_RPW; ++k) {
            const int row = row0 + warp + V3_NWARP * k;
            if (row >= nrows) {
#pragma unroll
                for (int l = 0; l < 8; ++l) w[0][k][l] = 0.0f;
                continue;
            }
            const block_q8_0 * bq = x + row * (ncols / QK8_0) + 0 * 8;
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float d = __half2float(V3_LOAD(&bq[l].d));
                w[0][k][l] = d * (int)((int8_t)V3_LOAD(&bq[l].qs[lane]));
            }
        }
    for (int ic = 0; ic < chunks_per_row; ++ic) {
        const int next_ic = ic + 1;
        const int next = cur ^ 1;
        for (int l = 0; l < 8; ++l) {
            const float * xp = &xs[cur][(32 * l + lane) * PITCH];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const float xv = xp[j];
#pragma unroll
                for (int k = 0; k < V3_RPW; ++k) acc[k][j] += w[cur][k][l] * xv;
            }
        }
        if (next_ic < chunks_per_row) {
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int cc = 0; cc < 256; cc += V3_THREADS) {
                const int c = cc + threadIdx.x;
                xs[next][c * PITCH + j] =
                    __half2float(X[j * ncols + next_ic * 256 + c]);
            }
        }
        for (int k = 0; k < V3_RPW; ++k) {
            const int row = row0 + warp + V3_NWARP * k;
            if (row >= nrows) {
#pragma unroll
                for (int l = 0; l < 8; ++l) w[next][k][l] = 0.0f;
                continue;
            }
            const block_q8_0 * bq = x + row * (ncols / QK8_0) + next_ic * 8;
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float d = __half2float(V3_LOAD(&bq[l].d));
                w[next][k][l] = d * (int)((int8_t)V3_LOAD(&bq[l].qs[lane]));
            }
        }
            __syncthreads();
        }
        if (next_ic < chunks_per_row) cur = next;
    }
#else
    for (int ic = 0; ic < chunks_per_row; ++ic) {
        const int col0 = ic * 256;
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

        float w[V3_RPW][8];
#pragma unroll
        for (int k = 0; k < V3_RPW; ++k) {
            const int row = row0 + warp + V3_NWARP * k;
            if (row >= nrows) {
#pragma unroll
                for (int l = 0; l < 8; ++l) w[k][l] = 0.0f;
                continue;
            }
            const block_q8_0 * bq =
                x + row * (ncols / QK8_0) + ic * 8;
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float d = __half2float(V3_LOAD(&bq[l].d));
                w[k][l] = d * (int)((int8_t)V3_LOAD(&bq[l].qs[lane]));
            }
        }
#if defined(V3_LDS128) && V3_LDS128
        if constexpr (NC % 4 == 0) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float4 * xp4 = reinterpret_cast<const float4 *>(&xs[(32 * l + lane) * PITCH]);
#pragma unroll
                for (int jj = 0; jj < NC / 4; ++jj) {
                    const float4 xv = xp4[jj];
#pragma unroll
                    for (int k = 0; k < V3_RPW; ++k) {
                        acc[k][4 * jj + 0] += w[k][l] * xv.x;
                        acc[k][4 * jj + 1] += w[k][l] * xv.y;
                        acc[k][4 * jj + 2] += w[k][l] * xv.z;
                        acc[k][4 * jj + 3] += w[k][l] * xv.w;
                    }
                }
            }
        } else {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const float * xp = &xs[(32 * l + lane) * PITCH];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const float xv = xp[j];
#pragma unroll
                for (int k = 0; k < V3_RPW; ++k) {
                    acc[k][j] += w[k][l] * xv;
                }
            }
        }
        }
#else
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const float * xp = &xs[(32 * l + lane) * PITCH];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const float xv = xp[j];
#pragma unroll
                for (int k = 0; k < V3_RPW; ++k) {
                    acc[k][j] += w[k][l] * xv;
                }
            }
        }
#endif
    }

#endif

#pragma unroll
    for (int k = 0; k < V3_RPW; ++k) {
        const int row = row0 + warp + V3_NWARP * k;
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
#if defined(V3_LDS128) && V3_LDS128
    constexpr int PITCH = ((NC + 3) / 4) * 4;
#else
    constexpr int PITCH = NC + 1;
#endif
#if defined(V3_DBUF) && V3_DBUF
    __shared__ __align__(16) float xs[2][256 * PITCH];
#else
    __shared__ __align__(16) float xs[256 * PITCH];
#endif

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int il = lane;              // 0..31
    const int is0 = il / 16;          // scale sub-index within each half
    const int row0 = V3_ROWS * blockIdx.x;

    const int blocks_per_row = ncols / QK_K;
    const block_q6_K * x = (const block_q6_K *) vx;

    float acc[V3_RPW][NC];
#pragma unroll
    for (int k = 0; k < V3_RPW; ++k)
#pragma unroll
        for (int j = 0; j < NC; ++j) acc[k][j] = 0.0f;

#if defined(V3_DBUF) && V3_DBUF
    float w[2][V3_RPW][2][4];
    int cur = 0;
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int cc = 0; cc < 256; cc += V3_THREADS) {
                const int c = cc + threadIdx.x;
                xs[0][c * PITCH + j] =
                    __half2float(X[j * ncols + 0 + c]);
            }
        }
    __syncthreads();
        for (int k = 0; k < V3_RPW; ++k) {
            const int row = row0 + warp + V3_NWARP * k;
            if (row >= nrows) {
#pragma unroll
                for (int ip = 0; ip < 2; ++ip)
#pragma unroll
                    for (int l = 0; l < 4; ++l) w[0][k][ip][l] = 0.0f;
                continue;
            }
            const block_q6_K * bq = x + row * blocks_per_row + 0;
            const float d = __half2float(V3_LOAD(&bq->d));
#pragma unroll
            for (int ip = 0; ip < 2; ++ip) {
                const uint8_t ql0 = V3_LOAD(&bq->ql[64 * ip + il]);
                const uint8_t ql32 = V3_LOAD(&bq->ql[64 * ip + il + 32]);
                const uint8_t qh = V3_LOAD(&bq->qh[32 * ip + il]);
                const int8_t * sc = bq->scales + 8 * ip + is0;
                w[0][k][ip][0] = d * sc[0] * (int8_t)(((ql0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
                w[0][k][ip][1] = d * sc[2] * (int8_t)(((ql32 & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
                w[0][k][ip][2] = d * sc[4] * (int8_t)(((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
                w[0][k][ip][3] = d * sc[6] * (int8_t)(((ql32 >> 4) | (((qh >> 6) & 3) << 4)) - 32);
            }
        }
    for (int ib = 0; ib < blocks_per_row; ++ib) {
        const int next_ib = ib + 1;
        const int next = cur ^ 1;
        for (int ip = 0; ip < 2; ++ip) {
            const int cbase = 128 * ip + il;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const float * xp = &xs[cur][(cbase + 32 * l) * PITCH];
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    const float xv = xp[j];
#pragma unroll
                    for (int k = 0; k < V3_RPW; ++k) acc[k][j] += w[cur][k][ip][l] * xv;
                }
            }
        }
        if (next_ib < blocks_per_row) {
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int cc = 0; cc < 256; cc += V3_THREADS) {
                const int c = cc + threadIdx.x;
                xs[next][c * PITCH + j] =
                    __half2float(X[j * ncols + next_ib * QK_K + c]);
            }
        }
        for (int k = 0; k < V3_RPW; ++k) {
            const int row = row0 + warp + V3_NWARP * k;
            if (row >= nrows) {
#pragma unroll
                for (int ip = 0; ip < 2; ++ip)
#pragma unroll
                    for (int l = 0; l < 4; ++l) w[next][k][ip][l] = 0.0f;
                continue;
            }
            const block_q6_K * bq = x + row * blocks_per_row + next_ib;
            const float d = __half2float(V3_LOAD(&bq->d));
#pragma unroll
            for (int ip = 0; ip < 2; ++ip) {
                const uint8_t ql0 = V3_LOAD(&bq->ql[64 * ip + il]);
                const uint8_t ql32 = V3_LOAD(&bq->ql[64 * ip + il + 32]);
                const uint8_t qh = V3_LOAD(&bq->qh[32 * ip + il]);
                const int8_t * sc = bq->scales + 8 * ip + is0;
                w[next][k][ip][0] = d * sc[0] * (int8_t)(((ql0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
                w[next][k][ip][1] = d * sc[2] * (int8_t)(((ql32 & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
                w[next][k][ip][2] = d * sc[4] * (int8_t)(((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
                w[next][k][ip][3] = d * sc[6] * (int8_t)(((ql32 >> 4) | (((qh >> 6) & 3) << 4)) - 32);
            }
        }
            __syncthreads();
        }
        if (next_ib < blocks_per_row) cur = next;
    }
#else
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
        float w[V3_RPW][2][4];  // [row][ip][col-quarter]
#pragma unroll
        for (int k = 0; k < V3_RPW; ++k) {
            const int row = row0 + warp + V3_NWARP * k;
            if (row >= nrows) {
#pragma unroll
                for (int ip = 0; ip < 2; ++ip)
#pragma unroll
                    for (int l = 0; l < 4; ++l) w[k][ip][l] = 0.0f;
                continue;
            }
            const block_q6_K * bq = x + row * blocks_per_row + ib;
            const float d = __half2float(V3_LOAD(&bq->d));
#pragma unroll
            for (int ip = 0; ip < 2; ++ip) {
                const uint8_t ql0 = V3_LOAD(&bq->ql[64 * ip + il]);
                const uint8_t ql32 = V3_LOAD(&bq->ql[64 * ip + il + 32]);
                const uint8_t qh = V3_LOAD(&bq->qh[32 * ip + il]);
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
#if defined(V3_LDS128) && V3_LDS128
        if constexpr (NC % 4 == 0) {
#pragma unroll
            for (int ip = 0; ip < 2; ++ip) {
                const int cbase = 128 * ip + il;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const float4 * xp4 = reinterpret_cast<const float4 *>(&xs[(cbase + 32 * l) * PITCH]);
#pragma unroll
                    for (int jj = 0; jj < NC / 4; ++jj) {
                        const float4 xv = xp4[jj];
#pragma unroll
                        for (int k = 0; k < V3_RPW; ++k) {
                            acc[k][4 * jj + 0] += w[k][ip][l] * xv.x;
                            acc[k][4 * jj + 1] += w[k][ip][l] * xv.y;
                            acc[k][4 * jj + 2] += w[k][ip][l] * xv.z;
                            acc[k][4 * jj + 3] += w[k][ip][l] * xv.w;
                        }
                    }
                }
            }
        } else {
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
                    for (int k = 0; k < V3_RPW; ++k) {
                        acc[k][j] += w[k][ip][l] * xv;
                    }
                }
            }
        }
        }
#else
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
                    for (int k = 0; k < V3_RPW; ++k) {
                        acc[k][j] += w[k][ip][l] * xv;
                    }
                }
            }
        }
#endif
    }

#endif

#pragma unroll
    for (int k = 0; k < V3_RPW; ++k) {
        const int row = row0 + warp + V3_NWARP * k;
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

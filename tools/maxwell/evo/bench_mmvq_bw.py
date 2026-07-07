#!/usr/bin/env python3
"""MMVQ bandwidth microbench (E18): isolate the sidecar fused matvec per
quant type / shape and report achieved GB/s vs the ~73 GB/s GM107 ceiling.

Pulls real tensors from the bench GGUF, plus a synthetic F16 GEMV reference
for the in_proj comparison. Run on one idle die (CUDA_VISIBLE_DEVICES=N).
"""

import os
import sys
import time

import gguf
import numpy as np
import torch

MODEL = os.path.expanduser(
    os.environ.get("BENCH_MODEL", "~/models/qwen35-9b/Qwen3.5-9B-F16INPROJ.gguf")
)
REPS = 200

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evo_mmvq  # noqa: E402

# name-substring -> want; covers in_proj-like, ffn, and lm_head shapes
PICKS = [
    ("blk.0.ffn_gate.weight", None),   # Q4_K [12288?, 4096]
    ("blk.0.ffn_down.weight", None),   # Q6_K [4096, 12288?]
    ("output.weight", None),           # Q6_K [151936, 4096] lm_head
    ("blk.0.attn_qkv.weight", None),   # Q4_K in FIXED.gguf (absent in F16INPROJ)
]


def bench_one(w, qtype, rows, cols, label):
    x1 = torch.randn(1, cols, dtype=torch.float16, device="cuda") * 0.5
    x8 = torch.randn(8, cols, dtype=torch.float16, device="cuda") * 0.5
    kernels = [("MMVQ", evo_mmvq._ext.mul_mat_vec_a8)]
    if qtype in (12, 14) and hasattr(evo_mmvq._ext, "mul_mat_vec_a8_v2"):
        kernels.append(("MMV2", evo_mmvq._ext.mul_mat_vec_a8_v2))
    if qtype in (12, 14) and hasattr(evo_mmvq._ext, "mul_mat_vec_a8_v3"):
        kernels.append(("MMV3", evo_mmvq._ext.mul_mat_vec_a8_v3))
    for kname, kfn in kernels:
        for x, tag in ((x1, "b1"), (x8, "b8")):
            for _ in range(5):
                kfn(w, x, qtype, rows)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(REPS):
                kfn(w, x, qtype, rows)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / REPS
            # v1 re-reads the weight per vec; v2 reads it once per <=8 vecs.
            # report effective GB/s in v1 terms so numbers stay comparable.
            gb = w.numel() * x.shape[0] / 1e9
            print(f"{kname} {label} type={qtype} [{rows}x{cols}] {tag}: "
                  f"{dt * 1e3:7.3f} ms  {gb / dt:6.1f} GB/s(eff)")


def bench_f16(rows, cols, label):
    w = torch.randn(rows, cols, dtype=torch.float16, device="cuda") * 0.02
    x1 = torch.randn(1, cols, dtype=torch.float16, device="cuda") * 0.5
    x8 = torch.randn(8, cols, dtype=torch.float16, device="cuda") * 0.5
    for x, tag in ((x1, "b1"), (x8, "b8")):
        for _ in range(5):
            x @ w.T
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(REPS):
            x @ w.T
        torch.cuda.synchronize()
        dt = (time.time() - t0) / REPS
        gb = w.numel() * 2 / 1e9  # GEMM reads W once regardless of batch
        print(f"F16  {label} [{rows}x{cols}] {tag}: "
              f"{dt * 1e3:7.3f} ms  {gb / dt:6.1f} GB/s")


def main():
    reader = gguf.GGUFReader(MODEL)
    tensors = {t.name: t for t in reader.tensors}
    for name, _ in PICKS:
        t = tensors.get(name)
        if t is None:
            print(f"SKIP {name} (not in this GGUF)")
            continue
        qtype = int(t.tensor_type)
        if qtype in (0, 1):  # f32/f16 tensor — bench the GEMV reference path
            cols, rows = int(t.shape[0]), int(t.shape[1])
            bench_f16(rows, cols, name)
            continue
        cols, rows = int(t.shape[0]), int(t.shape[1])
        block, type_size = gguf.GGML_QUANT_SIZES[t.tensor_type]
        w = torch.from_numpy(np.ascontiguousarray(t.data)).view(torch.uint8)
        w = w.reshape(rows, cols // block * type_size).cuda()
        bench_one(w, qtype, rows, cols, name)
    # synthetic in_proj shard shape at TP=4 (3072x4096) for both paths
    bench_f16(3072, 4096, "synthetic-inproj-shard")
    print("BW_BENCH_DONE")


if __name__ == "__main__":
    main()

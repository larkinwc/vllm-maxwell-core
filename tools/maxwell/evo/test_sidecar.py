#!/usr/bin/env python3
"""Numerics gate for the evo sidecar: fused MMVQ/MMQ vs dequant+matmul.

Pulls real quantized tensors (q4_K, q6_K) out of the benchmark GGUF, runs the
sidecar kernels against vLLM's trusted ggml_dequantize + fp16 matmul path on
one GPU, and reports max abs/rel error per (type, batch). Exits nonzero if
anything exceeds tolerance — run this before any bench that trusts the
sidecar.
"""

import os
import sys

import gguf
import numpy as np
import torch

MODEL = os.path.expanduser(
    os.environ.get("BENCH_MODEL", "~/models/qwen35-9b/Qwen3.5-9B-F16INPROJ.gguf")
)
# fused kernels accumulate int8 dots in fp32 vs fp16-dequant reference; the
# q8_1 activation quantization dominates the error, so tolerance is loose.
RTOL_DENOM = 1e-3
MAX_REL = float(os.environ.get("EVO_TEST_MAX_REL", "0.05"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evo_mmvq  # noqa: E402  (triggers JIT build)

from vllm import _custom_ops as ops  # noqa: E402

WANT = {gguf.GGMLQuantizationType.Q4_K: 2, gguf.GGMLQuantizationType.Q6_K: 2,
        gguf.GGMLQuantizationType.Q8_0: 1}


def main():
    reader = gguf.GGUFReader(MODEL)
    picked = []
    counts = {t: 0 for t in WANT}
    for t in reader.tensors:
        tt = t.tensor_type
        if tt in counts and counts[tt] < WANT[tt]:
            counts[tt] += 1
            picked.append(t)
        if all(counts[t] >= WANT[t] for t in WANT):
            break

    failures = 0
    for t in picked:
        qtype = int(t.tensor_type)
        # gguf reader gives data as flat bytes; logical shape in t.shape (ne)
        cols, rows = int(t.shape[0]), int(t.shape[1])  # ne[0]=in, ne[1]=out
        block, type_size = gguf.GGML_QUANT_SIZES[t.tensor_type]
        w = torch.from_numpy(np.ascontiguousarray(t.data)).view(torch.uint8)
        w = w.reshape(rows, cols // block * type_size).cuda()

        wdq = ops.ggml_dequantize(w, qtype, rows, cols, torch.float16)

        for batch in (1, 2, 8):
            x = (torch.randn(batch, cols, dtype=torch.float16, device="cuda")
                 * 0.5)
            ref = (x @ wdq.T).float()
            got_v = evo_mmvq._ext.mul_mat_vec_a8(w, x, qtype, rows).float()
            got_m = evo_mmvq._ext.mul_mat_a8(w, x, qtype, rows).float()
            checks = [("mmvq", got_v), ("mmq", got_m)]
            if qtype in (12, 14) and hasattr(evo_mmvq._ext, "mul_mat_vec_a8_v2"):
                checks.append(
                    ("mmvq_v2",
                     evo_mmvq._ext.mul_mat_vec_a8_v2(w, x, qtype, rows).float())
                )
            if qtype in (8, 12, 14) and hasattr(evo_mmvq._ext, "mul_mat_vec_a8_v3"):
                checks.append(
                    ("mmvq_v3",
                     evo_mmvq._ext.mul_mat_vec_a8_v3(w, x, qtype, rows).float())
                )
            scale = ref.abs().max().clamp(min=RTOL_DENOM)
            for name, got in checks:
                rel = ((got - ref).abs().max() / scale).item()
                ok = rel <= MAX_REL and torch.isfinite(got).all().item()
                print(f"{t.name} type={qtype} batch={batch} {name}: "
                      f"max_rel={rel:.4g} {'OK' if ok else 'FAIL'}")
                failures += 0 if ok else 1

    print("SIDECAR_TEST", "PASS" if failures == 0 else f"FAIL({failures})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

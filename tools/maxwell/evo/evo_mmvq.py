"""Maxwell evo sidecar loader + dispatch policy for fused GGUF matmuls.

Imported by gguf.py when MAXWELL_EVO_MMVQ=1. JIT-builds mmvq_sidecar.cu
against the vendored (sm_50-patched) kernel headers; the build is cached in
~/.cache/torch_extensions so only the first import pays the compile.

Policy via MAXWELL_EVO_POLICY:
  mmvq      (default) fused MMVQ for batch <= mmvq_safe, dequant otherwise
  mmvq+mmq  fused MMVQ for small batch, fused MMQ tile kernel for the rest
"""

import os

from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.normpath(os.path.join(_dir, "..", "..", ".."))
_gguf_inc = os.path.join(_repo, "csrc", "libtorch_stable", "quantization", "gguf")
_csrc_inc = os.path.join(_repo, "csrc")

_mmv_y = int(os.environ.get("MAXWELL_EVO_MMV_Y", "1"))
_dp4a_short = int(os.environ.get("MAXWELL_EVO_DP4A_SHORT", "1"))
_msum = int(os.environ.get("MAXWELL_EVO_Q4K_MSUM", "0"))
_no_u = int(os.environ.get("MAXWELL_EVO_BENCH_NO_U", "0"))  # bench-only!
_rpb4 = int(os.environ.get("MAXWELL_EVO_RPB_Q4K", "1"))
_rpb6 = int(os.environ.get("MAXWELL_EVO_RPB_Q6K", "2"))
_ext = load(
    name=(f"maxwell_evo_mmvq_y{_mmv_y}s{_dp4a_short}m{_msum}n{_no_u}"
          f"r{_rpb4}{_rpb6}"),
    sources=[os.path.join(_dir, "mmvq_sidecar.cu")],
    extra_include_paths=[_gguf_inc, _csrc_inc, _dir],
    extra_cuda_cflags=["-O3", "-gencode", "arch=compute_50,code=sm_50",
                       f"-DGGML_CUDA_MMV_Y={_mmv_y}",
                       f"-DMAXWELL_DP4A_SHORT={_dp4a_short}",
                       f"-DMAXWELL_Q4K_MSUM_HOIST={_msum}",
                       f"-DMAXWELL_BENCH_NO_U={_no_u}",
                       f"-DMAXWELL_V2_RPB_Q4K={_rpb4}",
                       f"-DMAXWELL_V2_RPB_Q6K={_rpb6}"],
    verbose=os.environ.get("MAXWELL_EVO_VERBOSE") == "1",
)

# v2 kernel (q4_K/q6_K): weight reuse across vecs + q8 reuse across rows.
_USE_V2 = os.environ.get("MAXWELL_EVO_V2", "0") == "1"
_V2_TYPES = frozenset({12, 14})
# v3 kernel (q4_K/q6_K): smem dequant + FFMA multi-vec, no q8 quantize.
# Takes precedence over v2 for its types when enabled.
_USE_V3 = os.environ.get("MAXWELL_EVO_V3", "0") == "1"
_V3_MIN_B = int(os.environ.get("MAXWELL_EVO_V3_MIN_B", "1"))
_V3_TYPES = frozenset({12, 14})

# GGML quant types wired up in the sidecar (standard + K-quants).
FUSED_TYPES = frozenset({2, 3, 6, 7, 8, 10, 11, 12, 13, 14})

_POLICY = os.environ.get("MAXWELL_EVO_POLICY", "mmvq")
# Override the vendored mmvq_safe batch limit (0 = keep caller's heuristic).
# MMVQ re-reads weights per vec, MMQ amortizes across the tile — where the
# crossover sits on GM107 is an empirical question.
_MMVQ_MAX = int(os.environ.get("MAXWELL_EVO_MMVQ_MAX", "0"))


def try_fused(x, qweight, qweight_type, mmvq_safe):
    """Return fused-kernel result, or None to fall back to dequant+cuBLAS."""
    if qweight_type not in FUSED_TYPES:
        return None
    if x.shape[0] <= (_MMVQ_MAX if _MMVQ_MAX > 0 else mmvq_safe):
        if (_USE_V3 and x.shape[0] >= _V3_MIN_B
                and qweight_type in _V3_TYPES):
            return _ext.mul_mat_vec_a8_v3(qweight, x.contiguous(),
                                          qweight_type, qweight.shape[0])
        # v2 amortizes weight reads across <=8 output columns — a wash at
        # batch 1 (keep v1 there), decisive from batch 2 up.
        if _USE_V2 and x.shape[0] >= 2 and qweight_type in _V2_TYPES:
            return _ext.mul_mat_vec_a8_v2(qweight, x, qweight_type,
                                          qweight.shape[0])
        return _ext.mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    if _POLICY == "mmvq+mmq":
        return _ext.mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    return None

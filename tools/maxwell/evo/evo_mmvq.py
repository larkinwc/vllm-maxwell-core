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
_ext = load(
    name=f"maxwell_evo_mmvq_y{_mmv_y}",
    sources=[os.path.join(_dir, "mmvq_sidecar.cu")],
    extra_include_paths=[_gguf_inc, _csrc_inc],
    extra_cuda_cflags=["-O3", "-gencode", "arch=compute_50,code=sm_50",
                       f"-DGGML_CUDA_MMV_Y={_mmv_y}"],
    verbose=os.environ.get("MAXWELL_EVO_VERBOSE") == "1",
)

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
        return _ext.mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    if _POLICY == "mmvq+mmq":
        return _ext.mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    return None

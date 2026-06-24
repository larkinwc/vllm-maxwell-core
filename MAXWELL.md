# vLLM on NVIDIA Maxwell (Tesla M10, sm_50)

This fork (`maxwell/v0.23`) makes vLLM build and run on **Maxwell GM10x GPUs
(compute capability 5.0 / 5.2)** — specifically the Tesla M10 — for use as a
**decode tier** in a disaggregated prefill/decode deployment.

> Status: **fp16 dense + AWQ 4-bit + GPTQ 4-bit + GPTQ MoE + Qwen3.5 (hybrid
> Mamba/attention) all generate coherent text on a real Tesla M10.** Validated
> end-to-end with vLLM v1 engine, both eager AND full CUDA graphs, TP=1 and
> TP=2.

## Why Maxwell needs patches

Maxwell GM10x has fp16 **storage + convert** but **no fp16 ALU**. `ptxas`
rejects every fp16 arithmetic intrinsic (`add.f16`, `mul.f16`,
`fma.rn.f16x2`, `sub.f16x2`, `__hadd/__hmul/__hsub/__hfma` and the packed `*2`
variants) — these require **sm_53+**. `__dp4a` requires **sm_61+**.
sm_50 *can* do `__half2float` / `__float2half` conversions and native fp32.

The fix pattern (proven by ggml/llama.cpp): **convert fp16 -> fp32 -> compute ->
convert back**. ptxas rejects fp16 intrinsics at *compile* time regardless of
reachability, so early `return;` guards are insufficient — fp16 code must be
`#if`-compiled out or replaced with the emulation.

A second, subtler class of bug: kernels guarded with
`#if __CUDA_ARCH__ < 750: assert(false)`. Under `-DNDEBUG` (release builds)
`assert` is a **no-op**, so such kernels *compile* fine but return
**uninitialized garbage** at runtime on sm_50. These were only caught by
**on-device numerics tests**, not by compilation.

## Build

Toolchain pins: **CUDA 12.6** (12.6 is the floor; CUDA 13 nvcc cannot emit
sm_50), torch 2.11 (source build, sm_50 SASS), Python 3.12, fp16-first.

```bash
TORCH_CUDA_ARCH_LIST="5.0;5.2" USE_ROCM=0 MAX_JOBS=28 \
  pip install -e . --no-build-isolation --no-deps -v
```

`--no-deps` is **critical** — it prevents pip from overwriting the custom
torch-sm50 build with a cu13 wheel (which dropped Maxwell) and from pulling
flashinfer / cutlass-dsl / humming (tensor-core-only deps we don't use).

All 301 CUDA targets compile + link for `sm_50;sm_52`.

## Runtime flags

```bash
CUDA_VISIBLE_DEVICES=0            # scope to the M10s (box also has MI100s)
VLLM_USE_FLASHINFER_SAMPLER=0     # flashinfer is tensor-core only; use native sampler
```

- **CUDA graphs work** and are auto-enabled on sm<70 (FULL graph capture is a
  pure-runtime feature; only Inductor/Triton *compilation* needs cc>=7, which we
  bypass with the eager `simple_compile_backend`). Pass `enforce_eager=True` to
  opt out.
- FlashAttention v2/v3 require cc>=8 and auto-disable; vLLM falls back to a
  working attention backend automatically.
- Multimodal (vision) models: pass `limit_mm_per_prompt={"image": 0, "video": 0}`
  for **text-only** use. The ViT encoder does O(N^2) SDPA that OOMs the 7 GB M10
  during dummy-image memory profiling otherwise.

## What was patched

### CUDA kernels (csrc)
- `attention/dtype_float16.cuh` — fp32-emulated add/mul/fma leaf primitives under
  `#if __CUDA_ARCH__ < 530` (fixes paged_attention v1 + v2, shared header).
- `quantization/awq/dequantize.cuh` — `dequantize_s4_to_fp16x2` was
  `assert(false)`-stubbed for sm<750 (garbage under NDEBUG). Enabled for sm_50:
  same lop3/prmt int4->fp16 bit tricks (arch-safe) + fp32-emulated trailing
  sub/fma.
- `quantization/awq/gemm_kernels.cu` — fp32-emulated `awq_sub_h2/awq_fma_h2`
  for the `dequantize_weights` kernel.
- `quantization/gptq/compat.cuh` — `maxwell_h*` helpers + macro routing so
  exllama kernels build unmodified; CUDA 12.6 atomicAdd compat fix.
- `quantization/moe/moe_wna16_utils.h`, `moe/moe_wna16.cu` — fp16 emulation +
  kernel-body arch guard.

### Python (kernel/quant selection)
- `quantization/humming.py` — soft `try/except ImportError` around the optional
  tensor-core `humming` package (its unconditional import was aborting the whole
  quant registry walk, breaking unrelated AWQ/GPTQ).
- `quantization/awq.py` — lower min capability 75->50; **force the
  `awq_dequantize` + `torch.matmul` path on sm<75** (the fused `awq_gemm` kernel
  is tensor-core mma.sync, stubbed out on sm_50).
- `quantization/auto_gptq.py` — lower min capability 60->50; make
  `verify_marlin_supported` non-fatal so kernel selection falls through to
  Exllama when Marlin (tensor-core) is unavailable.
- `kernels/linear/mixed_precision/exllama.py` — lower
  `ExllamaLinearKernel` min capability 60->50 (plain fp16 kernel, no mma/dp4a).
- `model_executor/layers/fused_moe/fused_moe.py` — `should_moe_wna16_use_cuda()`
  returns False on cc<80, routing int4 MoE matmuls to the **Triton WNA16 path**.
  The CUDA `moe_wna16_gemm` kernel uses sm_53+ fp16 intrinsics and is a no-op on
  sm_50 (silent garbage; models emitted repeated "I I I" during decode).

### Python (CUDA graphs / torch.compile on sm<70)
- `config/vllm.py` — auto-enable FULL CUDA graphs when cc<70 and not
  enforce_eager (set CompilationMode.NONE + CUDAGraphMode.FULL). Graph *capture*
  is pure runtime and works; only Inductor codegen needs cc>=7.
- `platforms/cuda.py` — set `simple_compile_backend="eager"` on cc<70 so the
  ~10 standalone `@torch.compile` decorators (embedding mask, MoE routers,
  RMSNorm) fall back to eager instead of crashing in Inductor.

## Validation (real Tesla M10, sm_50)

On-device numerics (vs fp32 reference):

| op | result |
|---|---|
| paged_attention_v1 | max abs err 3e-4 |
| paged_attention_v2 (3 partitions) | max abs err 8e-5 |
| awq_dequantize | max abs err 1e-3 |
| gptq_gemm | finite, runs |

End-to-end generation (vLLM v1, eager):

| model | quant | result |
|---|---|---|
| facebook/opt-125m | fp16 | coherent ("...capital of the French Republic") |
| facebook/opt-125m (TP=2 + cudagraphs) | fp16 | coherent (NCCL/TP validated) |
| Qwen2-0.5B-Instruct-AWQ | AWQ int4 | coherent (answers "Paris") |
| Qwen2-0.5B-Instruct-AWQ (+ cudagraphs) | AWQ int4 | coherent |
| Qwen2-0.5B-Instruct-GPTQ-Int4 | GPTQ int4 | coherent (answers "Paris") |
| Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 (TP=2, FULL cudagraphs) | GPTQ int4 MoE | coherent (answers "D. Paris") |
| Qwen/Qwen3.5-0.8B (hybrid Mamba+attn, text-mode) | fp16 | coherent (answers "Paris") |

CUDA graphs (FULL capture) and TP=2 (NCCL host-staged SHM all-reduce) both
validated; identical output to eager. Qwen3.5 needs torchvision built from
source against torch 2.11 (`v0.26.0`, `--no-deps`) for its image-processor
import chain; the Mamba/linear-attention Triton kernels JIT-compile and run on
sm_50.

## Hardware notes
- 2x Tesla M10 = 8 GPU dies (GM107, sm_50), ~6.9 GB usable each.
- **P2P is impossible** (GM107 never implemented peer-DMA; `canAccessPeer=0`);
  NCCL uses host-staged SHM. Prefer many small-TP replicas over large TP.
- NCCL 2.20.5 built for sm_50 (newer NCCL's mempool dep breaks on Maxwell).

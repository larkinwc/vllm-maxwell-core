# Qwen3.5-9B GGUF on Maxwell (Tesla M10 / sm_50)

Tooling + notes for running the hybrid Gated-DeltaNet (GDN) + gated-attention
**Qwen3.5-9B** GGUF on Maxwell GPUs (GM107, sm_50) under the `vllm-maxwell`
stack. Validated on a Dell PowerEdge C4130 with 4x Tesla M10 (16 GPU dies).

## TL;DR

```bash
# 1. Rewrite the llama.cpp GGUF into HF-native layout (once).
python tools/maxwell/fix_gguf.py \
    Qwen_Qwen3.5-9B-Q4_K_M.gguf Qwen3.5-9B-F16INPROJ.gguf --f16-inproj

# 2. Benchmark TP scaling (fresh process per TP). CUDA graphs strongly help.
BENCH_CUDAGRAPH=1 VLLM_DIST_INIT_TIMEOUT_S=2400 python tools/maxwell/tp_bench.py 4
```

The model must be loaded with the **text-only** class via
`hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]}` plus the HF tokenizer /
config from `Qwen/Qwen3.5-9B`. See `tp_bench.py` for the exact `LLM(...)` args.

## Why the GGUF needs rewriting (`fix_gguf.py`)

llama.cpp's Qwen3.5 conversion applies three transforms that vLLM does not
invert. `fix_gguf.py` inverts all three:

1. **RMSNorm (w+1)** — llama.cpp stores `weight + 1`; subtract 1 (all
   `*norm.weight` except the gated `ssm_norm`).
2. **A_log = -exp** — `ssm_a` stores `-exp(A_log)`; recover `A_log = log(-x)`.
3. **GDN V-head grouped->tiled reorder** — invert the tiled->grouped
   permutation for `in_proj_qkv` (V rows), `in_proj_z`, `in_proj_a/b`,
   `A_log`/`dt_bias`, `conv1d` (V channels), and `out_proj` (input columns).

`--f16-inproj` additionally dequantizes `in_proj_qkv`/`in_proj_z` to F16 as a
fallback around a vLLM merged-GGUF quantized-loader path; with the sm_50
`__dp4a` dequant fix (below) the plain quantized variant is expected to work
too.

## sm_50 code fixes (in the vLLM tree, not here)

These live in the vLLM source and are required for correctness/perf on Maxwell:

- **`quantization/gguf.py`** — the vendored quantized GGUF GEMM kernels use the
  `__dp4a` int8 intrinsic (needs sm>=6.1) and silently emit **NaN** on sm_50.
  Force the dequantize + fp16 matmul path (auto-detected; override with
  `MAXWELL_GGUF_DEQUANT=0/1`). Also lowers `get_min_capability` 60 -> 50.
- **`gguf_loader.py`** — qwen3_5 -> qwen35 arch name mapping, `ssm_dt.bias`
  manual mapping, A_log trailing-dot name fix, text-only (no-mmproj) routing.
- **`qwen_gdn_linear_attn.py`** — torch-native GDN chunk scan + causal conv1d
  prefill (the FLA/Triton chunk kernel is numerically wrong on Maxwell).
- **`linear.py`** — GGUF merged `in_proj_qkvz` (0,1,2) split, and KV-head
  **replication** in the GGUF QKV loader when `num_kv_heads < TP` (fixes TP>4).
- **`distributed/utils.py`** — `VLLM_DIST_INIT_TIMEOUT_S` override for slow
  multi-GPU (TP>=8) rendezvous on M10s.
- **`qwen3_5.py`** — embed_tokens quant wiring, conv1d 2D->3D unsqueeze,
  IsHybrid hooks. **`registry.py` / `config.py`** — text-only class + mrope
  strip.

## TP scaling results (Qwen3.5-9B, 8 prompts x 128 decode tok, greedy)

| TP | dies | eager tok/s | **cudagraph tok/s** |
|----|------|-------------|---------------------|
| 2  | 2    | 8.3         | 10.3                |
| 4  | 4    | 12.1        | **18.5** (best)     |
| 8  | 8    | 10.7        | 14.2                |
| 16 | 16   | 6.0         | (not run)           |

Throughput peaks at **TP=4 + CUDA graphs (18.5 tok/s)**. On M10s (no NVLink, no
tensor cores) per-op launch overhead dominates decode, so CUDA graphs help a
lot; spreading the model past 4 dies adds all-reduce overhead that outweighs the
compute split.

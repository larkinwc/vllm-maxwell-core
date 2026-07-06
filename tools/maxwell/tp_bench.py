#!/usr/bin/env python3
"""TP scaling benchmark for Qwen3.5-9B GGUF on Tesla M10 (sm_50).

Runs one tensor-parallel size per invocation (vLLM needs a fresh process per TP
config). Measures decode throughput (tok/s) and latency, checks coherence.

Environment toggles:
  BENCH_CUDAGRAPH=1        enable FULL cudagraphs (auto-downgrades to
                           FULL_DECODE_ONLY for the GDN backend). Big decode win
                           on Maxwell where per-op launch overhead dominates.
  VLLM_DIST_INIT_TIMEOUT_S raise the distributed rendezvous timeout (TP>=8 on
                           M10s can exceed the default 600s during worker init).
  BENCH_MODEL             override the GGUF path (default:
                           ~/models/qwen35-9b/Qwen3.5-9B-F16INPROJ.gguf).

Usage: tp_bench.py <TP>
"""
import os
import sys
import time
import json

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def main():
    tp = int(sys.argv[1])
    model = os.path.expanduser(
        os.environ.get("BENCH_MODEL", "~/models/qwen35-9b/Qwen3.5-9B-F16INPROJ.gguf")
    )

    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    use_cudagraph = os.environ.get("BENCH_CUDAGRAPH", "0") == "1"
    cg_cfg = CompilationConfig(
        mode=CompilationMode.NONE,          # no torch.compile (blocked on sm_50)
        cudagraph_mode=CUDAGraphMode.FULL,  # full-graph capture over eager fwd
    )

    t_load0 = time.time()
    llm = LLM(
        model=model,
        tokenizer="Qwen/Qwen3.5-9B",        # GGUF needs the HF tokenizer
        hf_config_path="Qwen/Qwen3.5-9B",   # route past unsupported qwen35 GGUF parser
        # Force the text-only Qwen3_5ForCausalLM class (registered in
        # registry.py) instead of the multimodal Qwen3_5ForConditionalGeneration
        # wrapper (mrope/vision path). The GGUF-on-sm_50 fix for garbage output.
        hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},
        tensor_parallel_size=tp,
        dtype="float16",
        enforce_eager=not use_cudagraph,
        compilation_config=(cg_cfg if use_cudagraph else None),
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        max_num_seqs=8,
        limit_mm_per_prompt={"image": 0, "video": 0},  # text-only (skip ViT)
        trust_remote_code=True,
        disable_log_stats=True,
    )
    load_s = time.time() - t_load0

    coherence_prompt = "The capital of France is"
    prompts = ["Explain in detail how a computer CPU works, step by step."] * 8

    # --- Coherence / correctness check ---
    sp_greedy = SamplingParams(temperature=0.0, max_tokens=16)
    co = llm.generate([coherence_prompt], sp_greedy)
    coherence_text = co[0].outputs[0].text

    # --- Throughput: decode a fixed number of tokens, measure tok/s ---
    gen_tokens = 128
    sp = SamplingParams(temperature=0.0, max_tokens=gen_tokens, ignore_eos=True)

    # warmup
    llm.generate(prompts[:1], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    throughput = total_out_tokens / dt

    # single-stream latency proxy: 1 request, full decode
    t1 = time.time()
    single = llm.generate([prompts[0]], sp)
    dt1 = time.time() - t1
    single_tps = len(single[0].outputs[0].token_ids) / dt1

    result = {
        "tp": tp,
        "cudagraph": use_cudagraph,
        "load_s": round(load_s, 1),
        "batch_size": len(prompts),
        "gen_tokens_per_req": gen_tokens,
        "total_out_tokens": total_out_tokens,
        "batch_wall_s": round(dt, 2),
        "batch_throughput_tok_s": round(throughput, 1),
        "single_stream_tok_s": round(single_tps, 1),
        "coherence_prompt": coherence_prompt,
        "coherence_text": coherence_text.strip()[:80],
    }
    print("RESULT_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()

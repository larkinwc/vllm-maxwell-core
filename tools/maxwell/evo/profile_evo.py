#!/usr/bin/env python3
"""Decode-step kernel profile for the evo loop (Tier 0.1).

Runs a short greedy decode under torch.profiler (CUPTI activity works on
sm_50) and prints the top kernels by summed self CUDA time from the rank-0
trace, so the next candidate family is picked from data.

Eager mode on purpose: per-op kernels are attributable in the trace. Wall
times are NOT comparable to CUDA-graph benchmarks (journal hard rule) — only
the GPU self-time ranking is the deliverable.

Env: BENCH_TP (default 4), BENCH_BATCH (default 1), BENCH_MODEL,
     MAXWELL_EVO_MMVQ etc. pass through to the model as usual.
Usage: profile_evo.py <trace_dir>
"""

import glob
import gzip
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def summarize(trace_dir):
    files = sorted(glob.glob(os.path.join(trace_dir, "**", "*.json*"),
                             recursive=True))
    if not files:
        print("NO_TRACE_FILES", trace_dir)
        return
    path = next((f for f in files if "trace.json" in f), files[0])
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    by_kernel = defaultdict(float)
    total = 0.0
    for ev in events:
        if ev.get("ph") == "X" and ev.get("cat") in ("kernel", "gpu_memcpy"):
            dur = ev.get("dur", 0.0)
            by_kernel[ev.get("name", "?")[:110]] += dur
            total += dur
    print(f"TRACE {os.path.basename(path)} total_gpu_us={total:.0f}")
    for name, dur in sorted(by_kernel.items(), key=lambda kv: -kv[1])[:25]:
        print(f"KERNEL {dur / total * 100:5.1f}% {dur:10.0f}us  {name}")


def main():
    if sys.argv[1] == "--parse-only":
        summarize(sys.argv[2])
        return
    trace_dir = sys.argv[1]
    os.makedirs(trace_dir, exist_ok=True)
    tp = int(os.environ.get("BENCH_TP", "4"))
    batch = int(os.environ.get("BENCH_BATCH", "1"))
    model = os.path.expanduser(
        os.environ.get("BENCH_MODEL",
                       "~/models/qwen35-9b/Qwen3.5-9B-F16INPROJ.gguf")
    )

    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    use_cudagraph = os.environ.get("BENCH_CUDAGRAPH", "0") == "1"
    cg_cfg = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL,
    )

    llm = LLM(
        model=model,
        tokenizer="Qwen/Qwen3.5-9B",
        hf_config_path="Qwen/Qwen3.5-9B",
        hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},
        tensor_parallel_size=tp,
        dtype="float16",
        enforce_eager=not use_cudagraph,
        compilation_config=(cg_cfg if use_cudagraph else None),
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        max_num_seqs=max(8, batch),
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=True,
        disable_log_stats=True,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": trace_dir,
        },
    )

    prompts = ["Explain in detail how a computer CPU works."] * batch
    sp = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8,
                                         ignore_eos=True))  # warmup
    llm.start_profile()
    llm.generate(prompts, sp)
    llm.stop_profile()
    del llm
    print("PROFILE_DONE batch=", batch)
    summarize(trace_dir)


if __name__ == "__main__":
    main()

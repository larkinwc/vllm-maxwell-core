#!/usr/bin/env python3
"""TTFT probe: time prefill of 128/512/1024-token prompts on the champion
config (prefill runs the torch-native GDN chunk scan — suspected TTFT wall).
Real file (not stdin) for vLLM spawn workers.
"""
import os
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def main():
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    model = os.path.expanduser(
        os.environ.get("BENCH_MODEL",
                       "~/models/qwen35-9b/Qwen3.5-9B-FIXED.gguf"))
    llm = LLM(
        model=model,
        tokenizer="Qwen/Qwen3.5-9B",
        hf_config_path="Qwen/Qwen3.5-9B",
        hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},
        tensor_parallel_size=4,
        dtype="float16",
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL),
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        max_num_seqs=8,
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=True,
        disable_log_stats=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    base = "The history of computing is long and detailed. " * 200
    tok = llm.get_tokenizer()
    ids = tok(base)["input_ids"]
    # warmup (also captures prefill code paths)
    llm.generate("warm up the engine", sp)
    for n in (128, 512, 1024):
        prompt = tok.decode(ids[:n])
        t0 = time.time()
        llm.generate(prompt, sp)
        dt = time.time() - t0
        print(f"TTFT n={n}: {dt:6.2f} s  ({n / dt:6.1f} prefill tok/s)")
    print("TTFT_DONE")


if __name__ == "__main__":
    main()

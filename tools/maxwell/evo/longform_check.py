#!/usr/bin/env python3
"""Long-form generation quality check for the champion config.

Must be a real file (not stdin) — vLLM spawn workers re-import the driver.
Env: champion MAXWELL_EVO_* vars + BENCH_MODEL; TP=4, CUDA graphs.
"""
import os

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
    prompts = [
        "The three most important inventions of the 20th century were",
        "Explain how photosynthesis works in two sentences.",
        "Write a haiku about mountains.",
    ]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0,
                                                max_tokens=180))
    for p, o in zip(prompts, outs):
        print("=" * 20)
        print("PROMPT:", p)
        print(o.outputs[0].text)
    print("LONGFORM_DONE")


if __name__ == "__main__":
    main()

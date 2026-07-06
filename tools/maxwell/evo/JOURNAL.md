# Evo hillclimb journal — Qwen3.5-9B GGUF decode on 4× Tesla M10 (sm_50)

Fitness: `batch_throughput_tok_s` at TP=4 + CUDA graphs (evo harness
`tp_bench_evo.py`, 8×128 greedy decode). Secondary: `single_stream_tok_s`,
`load_s`. Hard gate: `coherence_ok` ("Paris"). Baselines and rules inherit
from `docs/ROADMAP.md` + `research/AGENTS.maxwell.md`; negative results there
are not re-run.

Config shorthand: TP4-CG unless noted. Historical baseline 18.5/4.4 (stock
harness, 2026-07-05); evo-harness baseline below is the comparison anchor.

| # | candidate (family) | config | batch8 | single | load | coherent | verdict |
|---|---|---|---|---|---|---|---|
| E0 | async-sched (scheduling) | — log inspection | — | — | — | — | **already auto-enabled** in every baseline (`vllm.py:999` in stageA logs); Tier 2.2 is a no-op, crossed off |
| E1 | baseline (control) | stock | 17.9 | 4.4 | 189s | ✓ | anchor; within noise of 18.5 historical |
| E2 | dq0 (mmvq-probe) | MAXWELL_GGUF_DEQUANT=0 | 58.3 | 32.8 | 159s | ✗ `!!!!` | mechanism CONFIRMED: vecdotq impl bodies compile empty <cc6.1 (guards with no #else) → garbage at 3.3×/7.5× speed. **Upper bound** — empty bodies likely DCE'd the weight loads; real fixed kernel will land lower (tyangpu1 prior: 29.4 single) |
| E3 | knee (scheduling) | mns=32, batches 1/8/16/32 | 18.5 @8 | 4.4 | 166s | ✓ | **no knee through 32**: 4.4/18.5/33.6/**63.6** aggregate (per-seq 2.31→1.99). Decode fixed-costs amortize ~linearly; one TP=4 engine @32 ≈ the 4-replica aggregate estimate. batch-8 cell = E1 exactly (no interference). Follow-up: mns=64 sweep; TP=16 @ mns=32 |

## Gates

- 2026-07-06 numerics: `test_sidecar.py` **PASS** on box GPU15 — fixed MMVQ
  and MMQ vs dequant reference on real q4_K/q6_K tensors from the bench GGUF,
  batches 1/2/8, max_rel ≤ 0.012 (q8_1 activation quant noise). The dp4a
  fallback + widened guards produce correct kernels on sm_50.

## In flight

- E3 knee: BENCH_MAX_SEQS=32, batches 1/8/16/32 (Tier 0.7). Caveat: sidecar
  nvcc build shares the box CPU during its load phase — if batch-8 cell
  deviates from E1, rerun clean.
- E4 evo-mmvq: sidecar ext with sm_50 dp4a fallback + widened guards
  (commit 8b641bd0f), hybrid policy (fused matvec ≤mmvq_safe, dequant above).
  Numerics gate `test_sidecar.py` must PASS before benching. Expect single
  ≥20, batch8 ≈ E1.
- E5 evo-mmvq+mmq: MAXWELL_EVO_POLICY=mmvq+mmq — batch-8 decode through
  software-dp4a MMQ tiles. Expect batch8 anywhere in 15–45; measure.

## Backlog (families)

- mmvq-sync: llama.cpp-master kernel refresh + SoA K-quant layout (Q6_K worst:
  33 GB/s / 46% ceiling) — after E4/E5 establish the fused floor.
- gdn-cuda: port llama.cpp gated_delta_net.cu; also probe `causal-conv1d`
  install (bench log shows GDN conv falling back to torch: "fast path is not
  available").
- load-time: TP≥8 loads are CPU-bound per-worker GGUF processing (1366s at
  TP=8) — pre-shard/cache per-rank weights.
- tp8-valley-probe: 18.5→14.5→23.5 non-monotonic, profile before trusting
  scaling stories.
- in-proj-q4km: plain-quant in_proj gibberish = Q4_K superblock misalignment
  in merged split (linear.py) — may be MOOT if E4 lands (dp4a fix could cover
  the in_proj path too; retest FIXED.gguf with sidecar once E4 is green).

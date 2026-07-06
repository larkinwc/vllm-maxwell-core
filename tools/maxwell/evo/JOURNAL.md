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

| E4 | evo-mmvq (mmvq-fix) | MAXWELL_EVO_MMVQ=1 (hybrid) | 18.6 | **16.7** | 149s | ✓ | **WINNER, single-stream 3.8×** (4.4→16.7). Fixed MMVQ is correct AND fast; batch8 unchanged by design (dequant above mmvq_safe). Gap to E2's 32.8 upper bound = room for MMQ policy / SoA / tuning |

| E5 | evo-mmvq+mmq (mmvq-fix) | +MAXWELL_EVO_POLICY=mmvq+mmq | 18.5 | 16.7 | 148s | ✓ | MMQ tiles = dequant+cuBLAS exactly on batch-8 (null). Step arithmetic: batch-8 = 138 ms fixed floor (from E2) + ~293 ms weight path either route; single-stream 60 ms/step ≈ 80% of roofline given the ~30 ms floor |

## Step-time model (from E1/E2/E4/E5)

- batch-8 step ≈ 431 ms = 138 ms non-matmul floor (attention+GDN+allreduce+
  launches, measured by E2's empty kernels) + ~293 ms weight path (same for
  MMQ and dequant+cuBLAS).
- single step ≈ 60 ms = ~30 ms floor + ~30 ms MMVQ (1.57 GB/die @ 73 GB/s =
  21.5 ms theoretical → MMVQ ~72% BW efficiency, SoA's target).
- Implication: MMVQ weight re-read scales ×batch (8×21.5 ≈ 172 ms < 293 ms)
  → E6 MMVQ_MAX=8 hypothesis ≈ 26 tok/s batch-8. Perfect weight path caps
  batch-8 at ~37 tok/s (138 ms floor) — the floor itself is the other half
  of the problem (profile: E9).

## In flight

- E6 mmvq8: MAXWELL_EVO_MMVQ_MAX=8 — MMVQ for all decode sizes. Expect ~26.
- E7 knee-mmq: mmvq+mmq, mns=32, batches 8/16/32 — does MMQ beat dequant at
  16/32 aggregate? (dequant knee: 33.6/63.6).
- E8 tp16-mmvq: champion config at TP=16 (weights/die 390 MB → 5.3 ms read;
  dequant TP=16 was 23.5 batch / 3.9 single).

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

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

| E6 | mmvq8 (mmvq-fix) | +MAXWELL_EVO_MMVQ_MAX=8 | **23.6** | 16.8 | 148s | ✓ | **NEW CHAMPION batch-8 (+27%)**. MMVQ for all decode sizes beats MMQ/dequant, close to the ~26 model prediction. TP=4 now EQUALS TP=16's crown at 1/4 the dies, 1/5 the load |
| E7 | knee-mmq (mmq) | mmvq+mmq, mns=32, b 8/16/32 | 18.6 | 16.7 | 150s | ✓ | 18.6/35.2/64.0 ≈ dequant knee (18.5/33.6/63.6). MMQ family is a NULL at every batch — closed |
| E8 | tp16-mmvq (validation) | TP=16, hybrid | 23.7 | 10.8 | 772s | ✓ | single 3.9→10.8 (+177%) but < TP=4's 16.8 — TP=16 fixed floor dominates. batch-8 23.7 unchanged (dispatch above mmvq_safe). TP=4 is the platform |
| E9 | profile (Tier 0.1) | eager traces b1/b8 | — | — | — | — | b1: host-staged allreduce ≈75% of GPU time (record_param_comms 1.5 ms/call × 52/token ≈ the 30 ms single floor) — comms already known dead end ⇒ single-stream ≈ ceiling. b8 (dequant): mm 27% + dequantize 15% + copies ~35% ⇒ weight path, as modeled |

| E10 | crossover (dispatch) | MMVQ_MAX=12+mmq, mns=32, b 8/12/16/32 | 23.7 | 16.7 | 148s | ✓ | champion holds under mns=32 graphs: 23.7/64.4. batch-12 cell (24.4) pads to the 16-graph — not comparable. Dispatch settled: MMVQ ≤8, MMQ/dequant above |

## Root cause #2 (found by reading, 2026-07-06): merged-shard concat order

`GGUFLinearMethod.apply()` concatenates fused-layer shard outputs in LOAD
order (`shard_id` append order = GGUF file order). Qwen3.5 stores
`attn_gate` (z, shard 3) before `attn_qkv` (shards 0,1,2) → in_proj_qkvz
output is `[z|q|k|v]` where the model splits `[q|k|v|z]` → the plain-quant
in_proj "gibberish" (structured garbage `",akai方-的…"`, exactly what a
column permutation produces). String ids were already canonicalized
(`["q","k","v"]`); integer ids never got sorted. F16INPROJ dodges it because
all-F16 layers route to `UnquantizedLinearMethod` (direct writes, no concat)
— which is why the F16 rewrite "fixed" it. Fix: `sorted(shard_id)`.
**Upstream vLLM bug, PR-worthy.** Prize if E14 confirms: drop the F16
in_proj rewrite (31% of model bytes → ~5%), ≈ +9% on every config, +2.25 GiB
back, and the whole fix_gguf pipeline simplifies to default mode.

| E11 | mmvy2 (kernel-tune) | +MMV_Y=2 | 23.7 | 16.8 | 150s | ✓ | exact NULL vs champion — tyangpu1's +2% doesn't replicate on C4130/K-quants. Closed, Y=1 stays |
| E13 | mns64 (scheduling) | mns=64, b 32/48/64 | — | 16.6 | 222s | ✓ | **still no knee: 64.2/87.1/106.6 aggregate @ b64** (per-seq 1.67). One TP=4 engine = 4.5× old TP=16 record. mns=128 next; DP 4×engines ≈ 400+ tok/s box potential |

| E14 | q4km-sorted (loader-fix) | FIXED.gguf + sorted shards | 22.5 | 15.8 | 171s | **✓ COHERENT** | **root cause #2 CONFIRMED** — plain-quant in_proj works with sorted concat. But SLOWER than F16 champion: the merged path pays 4 matmuls + 4 activation quantizes + cat per call, eating the byte savings |
| E15 | f16-control | champion cfg + sorted() | 23.6 | 16.7 | 146s | ✓ | exact champion repro — sorted() no-op for F16 (unquantized method), no regression |

## Root cause #2 addendum → E16 (merged-shard single-matmul fast path)

`apply()` ran N fused matmuls + N q8_1 activation quantizes + torch.cat for
every merged layer on every call — not just in_proj: **every gate_up MLP and
string-id QKV too, in all prior runs**. Fix: stack padded shards in logical
order at load; when all shards share type+width, run ONE fused matmul, no
cat. Both models should gain.

| E16 | q4km+fastpath (loader) | FIXED.gguf, single-matmul merged | 22.7 | 16.0 | 145s | ✓ | fast path ≈ null (+0.2 vs E14): per-shard overhead was NOT the cost. All-quant still loses to F16 in_proj despite 3.6× fewer bytes → **Q4_K MMVQ must be far below assumed 72% eff on some shapes** — measure (E18) |
| E16b | f16+fastpath (control) | champion cfg | 23.5 | 16.9 | 153s | ✓ | champion held; fast path harmless. Keep it (correctness + cleanliness), keep F16INPROJ as perf champion |
| E17 | mns128 (scheduling) | mns=128, b 64/96/128 | — | 16.8 | 159s | ✓ | **knee found ≈64–96**: 110.3 / 101.5 (pad-to-128 artifact) / **122.2 @128** — one-engine peak. Serving sweet spot mns=64 (1.7 tok/s/user) or 128 (+11% aggregate at 0.95/user) |

## Kernel micro-iteration (E18+, microbench on idle die, ~3 min/cycle)

| variant | q4_K GB/s (b1) | verdict |
|---|---|---|
| E18 baseline | 36.5 (q6_K 33.8, lm_head 33.3; F16 GEMV ref 67.2 = 92% ceiling) | MMVQ at ~45% ceiling, FLAT across type/shape/batch → not latency-bound. F16-vs-Q4_K in_proj anomaly explained |
| V1 dp4a short-mul | 36.0 | NULL — nvcc already lowers int8 products well |
| V-msum (q4_K dot2 hoist via q8 ds.y) | **38.8** | **+6%, numerics PASS, kept** (MAXWELL_EVO_Q4K_MSUM=1). Small gain from halving dp4a ⇒ ALU is not the main wall |
| NO_U (bench-only, q8 loads killed) | pending | discriminator: q8 activation re-read is ~2× weight traffic per row (L2-served); if GB/s jumps → restructure for q8 reuse (rows-per-block staging, llama.cpp master mmvq shape) |

Traffic model: per 144 B q4_K superblock each row re-reads ~72 B of q8
data + ds; at b1 the q8 row (4.6 KB) is re-read by every one of 12288/2
warp-iterations → q8 L2 traffic ≈ 2× weight DRAM traffic. L2 round-trip
may be the ~34 GB/s cap.

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

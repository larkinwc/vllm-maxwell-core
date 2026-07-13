# Evo hillclimb journal — Qwen3.5-9B GGUF decode on 4× Tesla M10 (sm_50)

Fitness: `batch_throughput_tok_s` at TP=4 + CUDA graphs (evo harness
`tp_bench_evo.py`, 8×128 greedy decode). Secondary: `single_stream_tok_s`,
`load_s`. Hard gate: `coherence_ok` ("Paris"). Baselines and rules inherit
from `docs/ROADMAP.md` + `research/AGENTS.maxwell.md`; negative results there
are not re-run.

## Scoreboard (2026-07-06, all coherent)

| metric | session start | current champion | Δ | config |
|---|---|---|---|---|
| batch-8 tok/s (fitness) | 18.5 | **49.0** | **2.65×** | E33: TP4-CG, **FIXED.gguf** (sorted+grouped loader), MMVQ=1 MMVQ_MAX=8 Q4K_MSUM=1 V2=1 **V3=1 V3_MIN_B=2 V3_THREADS=128** |
| single-stream tok/s | 4.4 | **17.3** | **3.9×** | F16INPROJ model, same env (all-quant: 16.5) |
| batch-16 | — | **50.3** | — | +MMVQ_MAX=16 (v3 ≤16) |
| batch-32 / 64 | 63.6 / — | 63.8 / 106.0 | — | MMQ above 16 |
| one-engine aggregate | — | **122.2** @ b128 | — | mns=128 (F16 model; champion-model cell in flight) |

Root causes fixed: (1) vecdotq impl bodies compiled EMPTY below cc 6.1;
(2) merged-GGUF shards concatenated in file order (upstream vLLM bug, also
fixed the 2-generation-old in_proj gibberish). Kernel v2 (mmvq_v2.cuh) adds
ncols_dst weight reuse + q8 row reuse.

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

NO_U result: 38.8 → 48.1 GB/s (+24%) — q8 traffic confirmed as a real
component; remaining 48→67 gap = weight access pattern + unpack.

| E19 engine (msum) | — | **24.3 / 17.3 — new champion**, coherent (load 219s) |
| v2 kernel (ncols_dst≤8 weight reuse + q8 row reuse, q4_K/q6_K) | q4_K b8 59.5 GB/s eff (+55%), q6_K b8 66.5 (+96%, at F16 ceiling), b1 neutral | numerics PASS all types/batches. Dispatch: v1 at b1, v2 at b2..16, MMQ/dequant above. Engine A/B = E20/E21 |

DP4 first attempt crashed: mid-flight deploy changed the JIT extension
name and 4 engines raced a build containing a duplicate-include compile
error (ggml-common.h has no include guards). Fixed; rerun chained. Lesson:
never rsync the sidecar while engines are queued.

| E20 | v2 engine (kernel) | champion env + MAXWELL_EVO_V2=1 | **34.3** | 17.3 | 148s | ✓ | **CHAMPION +41%** (24.3→34.3; +85% vs session-start 18.5). Step 233 ms ≈ 138 floor + 95 weights — model exact; floor now dominates batch-8 |
| E21 | v2 knee | +MMVQ_MAX=16, mns=32 | 33.8 | 17.3 | 150s | ✓ | b16 39.0 (v2, beats MMQ 35.2 +11%); b32 66.3 (MMQ, unchanged). Dispatch v1@1 / v2@2–16 / MMQ@>16 validated |

| E23 | q4km-v2 (model) | FIXED.gguf + champion env | 34.2 | 16.4 | 164s | ✓ | all-quant TIES batch-8, loses single (in_proj q6_K rides v1 at b1 = 33 GB/s vs F16 GEMV 67). F16INPROJ stays perf champion; FIXED.gguf = capacity option (+1.5 GB VRAM) with correctness now proven |
| E24 | rpb2 (kernel-tune) | +V2_RPB=2 | 33.0 | 17.2 | 156s | ✓ | slight net loss. Microbench: q4_K b8 59.5→48.4 (worse), q6_K b8 66.5→90.1 eff (better), b1 both better (42.0/38.3). Per-type RPB mix → backlog |

## E22 floor decomposition (eager v2 profiles; graph-mode traces export no
## kernel events — CUPTI graph-replay records dropped by torch's exporter)

Batch-8 self-CUDA shares (ratios; eager inflates absolute):
`_fused_mul_mat_gguf` 39.8% · **`aten::mm` 17.0% (F16 in_proj via cuBLAS,
~58 calls/step — E18 measured cuBLAS fp16 GEMM at 17 GB/s for b8 shapes)** ·
GDN core ~10% (incl. einsum) · comms 5–9% · attention negligible ·
elementwise/copies rest. The "138 ms floor" was partly weight traffic in
disguise: the F16 in_proj mm. Puzzle: E23 (all-quant, in_proj q6_K via v2 at
66 GB/s) should have banked most of that 17% but TIED — profiling the
all-quant config (E25) to find the leak.

E25 (all-quant b8 profile) found the leak: fused calls 4114 → 7378
(+102/step) — in_proj_qkvz mixes types (Q6_K qkv + Q4_K gate), which
disabled the single-matmul path → 4 small matmuls + 4 quantizes + cat per
GDN layer, eating the F16-mm savings.

| E26 | q4km-grouped (loader) | grouped same-type shards | **35.2** | 16.5 | 145s | ✓ | **CHAMPION.** qkv+z = 2 calls. All-quant now wins batch-8; F16INPROJ keeps single-stream (17.3) since b1 in_proj rides v1 (33 GB/s) vs F16 GEMV (67) |
| E26b | f16-grouped (control) | grouped, default model | 34.0 | 17.2 | 150s | ✓ | unchanged (its in_proj bypasses gguf apply; gate_up already single-type) |

Closed: all-Q6_K in_proj GGUF regen (gguf-py can't quantize Q6_K,
NotImplementedError; llama-quantize not on box). Grouped 2-call path is the
resting state for mixed-type fused tensors.

| E29 | rpbmix (kernel-tune) | per-type RPB (q4_K=1, q6_K=2) | **37.8** | 16.5 | 143s | ✓ | **CHAMPION +7.4%** — fitness metric doubled for the session (18.5→37.8) |
| E27 | agg revalidation | champion model, mns=64 | — | 16.5 | 151s | ✓ | 40.2/63.8/106.0 @ 16/32/64 — b16 +1.2 from v2/rpb; b32+ unchanged (MMQ path) |

## Top backlog after the doubling

1. **GDN decode block fusion (E31 family, biggest remaining single item):**
   eager b8 shows qwen_gdn_attention_core ≈ 2.5 ms CUDA-total per call × 24
   layers — conv1d update + gating einsums + Triton recurrence + norm as
   separate kernels. A fused CUDA GDN-decode kernel could take ~10-15% off
   the b8 step. Execution plan (recon done 2026-07-06):
   - Reference: llama.cpp master `ggml/src/ggml-cuda/gated_delta_net.cu`
     (327 lines, fetched; template<S_v, KDA, keep_rs_t>, warp_size×4
     threads, pure fp32, no arch gates).
   - Port target: the semantics of
     `fused_recurrent_gated_delta_rule_packed_decode` (fla/ops/
     fused_recurrent.py:339) — packed post-conv mixed_qkv, a/b +
     A_log/dt_bias softplus gating, in-kernel q/k L2 norm, paged
     ssm_state_indices, fp32 state, one token/step. Optionally fold
     `causal_conv1d_update` in (immediately precedes, same tensors).
   - Integration: sidecar #2 (`gdn_sidecar.cu`, no gguf headers) + env hook
     MAXWELL_EVO_GDN=1 in fused_recurrent.py (same pattern as gguf hook).
   - Gate: direct A/B vs the Triton kernel on GPU15 with synthetic tensors
     (Triton decode kernel RUNS on sm_50 — usable as the reference), then
     coherence + engine bench.
2. MMQ modernization for b>16 (llama.cpp master mmq shape) — lifts the
   64/128-batch aggregate; moderate.
3. Comms (~6-9%): knobs measured dead; compressed allreduce only remaining
   idea, big effort, uncertain payoff.
4. Prefill: torch-native GDN chunk scan dominates prefill wall (16 s of
   einsum/permute in profiles) — TTFT lever for serving, not decode fitness.
4. ~~DP4~~ **DP4 measured-negative at mns=64 (2026-07-06):** 4 concurrent
   TP=4 engines: loads 4 min → 38 min (15× CPU/page-cache thrash), b64
   decode still unfinished at 3600 s timeout (solo: 77 s). Host-staged
   NCCL all-reduce ×4 engines shares one DDR4-2133 bus — DP replicas do
   NOT scale at high concurrency on this box. Revisit only with staggered
   loads + reduced per-engine concurrency + host-BW profiling. Per-engine
   records unaffected.

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

## E31 GDN port: CLOSED (measured unnecessary, 2026-07-06)

`bench_gdn.py` isolated the FLA Triton packed-decode kernel: 63.1 GB/s
state-BW at b8 (86% ceiling), 69.8 at b32 (96%), 3.2 ms/step across 24
layers at b8 = 1.5% of the step. Already near-optimal; the eager profile's
"GDN ~10%" was launch-gap inflation around it. Do not port.

## v2 is ALU-bound at b8 → v3 design (the next mountain)

Re-derived from the microbench: ffn_gate (28.3 MB) at b8 takes 3.81 ms =
7.4 GB/s REAL DRAM — v2 amortizes loads across 8 columns but pays the full
software-dp4a vec-dot per column (b8 ≈ 4.8× b1 on identical bytes). Weight
kernels ≈ 190 ms of the 212 ms champion step.

v3: dequantize each weight tile ONCE to smem/registers (fp16/fp32), then
FFMA against all ≤8 fp16 activation vectors directly — no q8_1 activation
quantization at all (numerics improve). FFMA is native 128/SMM/clk vs the
XMAD-emulated int path. Op model: ~3× less ALU per weight byte at b8 →
weight ~65 ms → step ~105 ms → **~70 tok/s batch-8 ceiling**. This is the
roadmap's NervanaGPU "pseudo-fp16" pattern, now demanded by data.

## v3 kernel generation (2026-07-06, cont.)

| iteration | q4_K b8 | q6_K b8 | q8_0 b8 | verdict |
|---|---|---|---|---|
| v3.0 smem-dequant+FFMA | 3.96 ms | 4.18 | — | parity with v2 — Maxwell FFMA takes no smem operand: 2 LDS + 2 FFMA per pair ≈ dp4a op count |
| v3.1 x register-blocked over rows | **3.15 (+21%)** | 3.55 (+3%) | — | LDS/4; numerics 10× better than v1/v2 (no q8 quantize), max_rel ~0.0008 |
| + q8_0 kernel | — | — | **1.15 ms (124 GB/s eff)** | ssm_out had been riding v1 8×-re-read (~35 ms/step) — hole closed |

| E32 | v3 engine (kernel) | FIXED + V2+V3 (V3_MIN_B=2) | **42.1** | 16.5 | 146s | ✓ | **CHAMPION +11.4%; session 2.28×** (18.5→42.1). Step 190 ms; q4_K real read rate still only ~9 GB/s ⇒ ~2.5× kernel headroom remains (LDS/ALU-bound) |

## Config limit (2026-07-06)

- FIXED.gguf + mns=128 + MMQ policy: EngineCore dies in stable-ABI
  `aten::empty` (allocation) during the b128 phase — quantized in_proj
  rides MMQ at high concurrency where the F16 model used torch.mm. For
  >64-seq serving on the all-quant model, drop gpu_memory_utilization or
  cap mns at 64 (106.0 tok/s validated); F16INPROJ holds the 122.2 @128
  record.

## v3 tile sweep (2026-07-06, cont.)

| config | q4_K b8 | q6_K b8 | engine b8 |
|---|---|---|---|
| 64 thr × 8 rows (v3.1) | 3.15 ms | 3.55 | 42.1 (E32) |
| **128 thr × 8 rows** | **2.59 (+22%)** | **3.02 (+18%)** | **49.0 (E33) — CHAMPION, session 2.65×** |
| 256×8 / 128×16 / 256×16 | sweeping | | |

q4_K real read rate at E33 still ~10.9 GB/s — kernel remains LDS/ALU-bound;
tile sweep + (later) double-buffered staging are the open levers.

## Validation close-out (2026-07-07)

- Long-form quality: PASS — 3×180-token greedy generations on the champion
  are clean (quiz-continuation + structured <think> responses, no loops or
  corruption). Note: heredoc drivers break vLLM spawn workers
  (FileNotFoundError '<stdin>') — bench drivers must be real files.
- E34 (128×16 tile): 48.8 — engine tie with E33's 49.0; V3_THREADS=128 /
  V3_ROWS=8 stands.
- E35 aggregate with v3 (NC=16 gate 15/15): b16 **50.3 (+25%, record)**;
  b32 via v3 2×16-chunks = 53.3 — LOSES to MMQ (63.8): dispatch finalized
  at MMVQ_MAX=16, MMQ above; b64 105.2 unchanged.

## Final champion & dispatch (end of 2026-07-06/07 session)

FIXED.gguf (all-quant, sorted+grouped loader) · TP=4 · CUDA graphs ·
MAXWELL_EVO_MMVQ=1 MMVQ_MAX=16 Q4K_MSUM=1 V2=1 V3=1 V3_MIN_B=2
V3_THREADS=128 POLICY=mmvq+mmq
→ kernel dispatch: v1 MMVQ (b1) / v3 FFMA (b2–16) / MMQ (b>16) / dequant
(non-fused types).

| batch | 1 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| tok/s | 16.5 (17.3 F16 model) | **49.0** | **50.3** | 63.8 | 105.2 | 122.2 (F16 model) |
| session start | 4.4 | 18.5 | — | 63.6 | — | — |

## TTFT arc (2026-07-07): MMQ was strangling prefill

| config | prefill tok/s (128/512/1024) | TTFT @512 |
|---|---|---|
| champion + POLICY=mmvq+mmq | 43.1 / 44.0 / 43.9 (flat) | 11.65 s |
| champion + POLICY=mmvq (dequant prefill) | 74.6 / 86.1 / **85.9** | **5.94 s (2×)** |

MMQ (software-dp4a tiles) handled every prefill chunk; dequant+cuBLAS wins
at prefill arithmetic intensity. MMQ tied dequant at b32/64 anyway (E3/E7/
E13) → **MMQ dropped from the dispatch entirely.**

## E36–E40 v3 Maxwell technique arc (2026-07-12)

Fresh baseline on this box (FIXED.gguf, TP=4 + CUDA graphs, idle die 15 for
microbench): q4_K/q6_K/q8_0 MMV3 b8 = **2.586 / 3.017 / 1.005 ms**;
engine b8/b16/b32/b64/single = **49.1 / 50.9 / 63.7 / 105.3 / 16.5 tok/s**
(coherent: Paris). A repeat of the explicit 128×8 champion measured 49.2
b8 / 16.8 single, confirming the baseline anchor.

| cell | candidate | q4_K / q6_K / q8_0 b8 (ms) | engine b8 | coherent | verdict |
|---|---|---:|---:|---|---|
| E36 | `MAXWELL_EVO_V3_DBUF=1`, 128×8 | 3.811 / 4.365 / 1.845 | 36.6 | yes | **DROP** — numerics PASS, but double-buffer staging regresses all three microbench types and engine throughput |
| E37 | `MAXWELL_EVO_V3_LDS128=1`, 128×8 | 11.325 / 4.973 / 1.697 | 20.0 | yes | **DROP** — numerics PASS; removing the bank pad costs more than float4 LDS saves on the live engine |
| E38 | scalar, 256×8 | 3.739 / 3.602 / 1.203 | 39.8 | yes | **DROP** — tile/occupancy regression despite q6_K microbench improvement |
| E39 | scalar, 128×16 | 2.641 / 2.726 / 0.880 | 49.1 | yes | **DROP** — within noise but below the 128×8 champion; 128×8 wins tie-break |
| E40 | scalar, 256×16 | 2.665 / 2.905 / 0.953 | 48.4 | yes | **DROP** — engine regression |

All E36/E37 numerics gates printed `SIDECAR_TEST PASS`; every engine run was
coherent. No A/B/C variant cleared the 49.2 tok/s b8 anchor, so the
keep/drop rule closes this arc with both technique flags default-off and the
E33 128×8 tile retained. The source remains env-gated for follow-up work; no
losing flag is enabled by default.

## E41 default revalidation (2026-07-12)

With no `MAXWELL_EVO_V3_THREADS`, `MAXWELL_EVO_V3_ROWS`,
`MAXWELL_EVO_V3_DBUF`, or `MAXWELL_EVO_V3_LDS128` overrides (therefore using
the folded defaults 128×8, DBUF=0, LDS128=0), the final FIXED.gguf TP=4 + CG
run measured:

| batch | 1 / single | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| tok/s | 16.6 | **49.3** | **51.0** | **63.8** | **105.7** |

Coherence remained `Paris`; `longform_check.py` ended with `LONGFORM_DONE`
with all three 180-token generations clean. b32/b64 stayed on the untouched
dequant+cuBLAS path and remained at the baseline values.

## FINAL config (end of hillclimb arc)

`Qwen3.5-9B-FIXED.gguf` · TP=4 · CUDA graphs ·
`MAXWELL_EVO_MMVQ=1 MMVQ_MAX=16 Q4K_MSUM=1 V2=1 V3=1 V3_MIN_B=2
V3_THREADS=128` (default policy mmvq; `V3_ROWS=8`, `V3_DBUF=0`,
`V3_LDS128=0` are the folded defaults)
→ v1 MMVQ (b1) / v3 FFMA (b2–16) / dequant+cuBLAS (b>16 + prefill).

Latest decode: b8 **49.3** · b16 **51.0** · b32 **63.8** · b64 **105.7** ·
single **16.6**. The CUDA-C DBUF/LDS128 arc was a negative result; E33's
128×8 scalar v3 remains the champion.
prefill: ~86 tok/s (TTFT 5.9 s @512)

Next mountain (designed, not started): GDN chunked-prefill CUDA port
(llama.cpp gated_delta_net.cu full-sequence kernel) — the torch-native
eager chunk scan is the remaining prefill wall after the MMQ fix.


## E42 — profile-gated next-arc scope (2026-07-12)

A rank-0 torch profiler trace of the E33 champion at b8 (FIXED.gguf, TP=4,
enforce-eager profiling run) put `vllm::_fused_mul_mat_gguf` first at 41.11%
of summed self-CUDA time (3.914 s / 5746 calls). The next largest entries were
`aten::copy_` 6.22%, GDN attention 6.10%, `aten::mm` 4.87%, and parameter
communication 3.78%. The profile therefore confirms the weight kernel as the
cleanest graph-invariant target, while the eager-only launch/copy accounting
should not be treated as a CUDA-graph throughput opportunity.

The GGUF tensor byte census for FIXED.gguf is q4_K (type 12): 3.15 GB / 58%
of quantized matrix bytes, q6_K (type 14): 2.28 GB / 42%, with q8_0 separate.
A q6_K-only SoA repack would therefore leave the byte-majority q4_K path
untouched. Contingency D remains a separate, default-off experiment; any new
weight-layout or access arc must cover q4_K and q6_K in the full-tensor
microbench and CUDA-graph b8 engine gates, not just q6_K. No loader change was
made from this profiling pass.


## E43 — q4_K vector weight-load probe (2026-07-12)

Added `MAXWELL_EVO_V3_Q4VEC` (default `0`) to test one 32-bit aligned load
for each q4_K four-byte lane slice, with scalar fallback and a cache-safe JIT
extension name. The live GGUF path was kept contiguous while validating the
experiment; the earlier non-contiguous shard slice state produced corrupted
control output and was restored before comparison.

Numerics passed (`SIDECAR_TEST PASS`) for q4_K/q6_K/q8_0, batches 1/2/8.
The full microbench was q4_K/q6_K/q8_0 MMV3 b8 **2.450 / 2.979 / 1.008 ms**
versus the E33 anchor **2.586 / 3.017 / 1.005 ms**: q4_K improved ~5%, q6_K
was within noise, and q8_0 was effectively neutral.

CUDA-graph TP=4 b8 engine repeats were coherent and measured **50.1** and
**50.0 tok/s** with Q4VEC enabled, versus **49.1** and **49.2 tok/s** controls.
The q4 vector path also passed `longform_check.py` (`LONGFORM_DONE`). This is
a stable ~1.6--1.8% b8 improvement, but below the arc's +5% target and not
enough to justify changing the champion default without the remaining full
batch sweep. Keep the implementation available for follow-up, but retain
`V3_Q4VEC=0` as the default.


## E44 — platform stability manifest and fresh ladder (2026-07-12)

**Platform island.** CUDA toolkit **12.6.85**; torch
**2.11.0a0+git70d99e9** linked against CUDA **12.6**; vLLM
**0.1.dev17421+g6408d1c84.d20260707**; NVIDIA driver **580.159.03**.
The Maxwell lane remains pinned to CUDA-12 wheels/images: CUDA 13 removed
sm_50 support. evo_mmvq.py now raises before JIT compilation unless
torch.version.cuda is 12.x (a synthetic 13.0 version raised the expected RuntimeError).

The sidecar now emits both arch=compute_50,code=sm_50 and
arch=compute_50,code=compute_50 gencode targets, preserving PTX alongside
the Maxwell cubin. MAXWELL_EVO_PTXAS_V=1 adds -Xptxas=-v, forces verbose
JIT output, and uses a distinct extension identity ending in v{_ptxasv}.
The 128x8 build produced ptxas statistics; its V3 q4_K/q6_K/q8_0 b8 kernels
used 80/72/72 registers respectively, zero spill loads/stores, and 17,408 B
shared memory.

**Fresh TP=4 CUDA-graph co-primary ladder** (FIXED.gguf; champion env;
BENCH_MAX_SEQS=64; 128 generated tokens/request): b1 **16.8**, b8
**49.3**, b16 **50.9**, b32 **63.7**, b64 **105.5** tok/s. The engine
reported Capturing CUDA graphs (decode, FULL) and Graph capturing finished;
this ladder replaces the prior anchors for subsequent gates.



## E45 — Q4VEC completion gate (2026-07-12)

Q4VEC was rebuilt under the CUDA-12/PTX platform configuration and remained
correct: EVO_TEST_MAX_REL=0.05 test_sidecar.py reported SIDECAR_TEST PASS.
Its MMV3 b8 microbench was q4_K/q6_K/q8_0 = **2.461 / 3.011 / 0.999 ms**,
which is non-regressing against the E44 anchors (2.586 / 3.017 / 1.005 ms).

The required two-run full TP=4 CUDA-graph ladders were coherent (both returned
Paris.) and had these Q4VEC / scalar-control results:

| run | b1 | b8 | b16 | b32 | b64 |
|---|---:|---:|---:|---:|---:|
| Q4VEC 1 | 16.9 | 50.1 | 51.5 | 63.9 | 105.8 |
| Q4VEC 2 | 16.7 | 49.9 | 51.3 | 63.4 | 105.5 |
| scalar 1 | 16.7 | 49.2 | 51.0 | 63.6 | 105.5 |
| scalar 2 | 16.7 | 49.1 | 50.6 | 63.8 | 105.4 |

The Q4VEC b8 gain was +0.9/+0.8 tok/s and b16 gain +0.5/+0.7 tok/s versus
the paired controls. Both co-primary metrics are non-regressing and the old
longform_check.py evidence remains LONGFORM_DONE, but neither co-primary
improved by the required +1.0 tok/s in both variant runs. **SHELVED:** retain
MAXWELL_EVO_V3_Q4VEC default 0; the experiment stays available for a future
combined arc.



## E46 — LDG read-only weight-load probe (2026-07-12)

Added default-off MAXWELL_EVO_V3_LDG, its V3_LDG compile definition, and a
cache-safe G{_v3ldg} extension-name suffix. V3_LOAD uses __ldg only when the
flag is set. It covers q4_K qword/scalar quant bytes and one local dm load,
q6_K ql/ qh/ d, and q8_0 d/qs; q4_K scales remain ordinary sub-word reads.
LDG scalar numerics passed SIDECAR_TEST PASS, and both engine LDG configurations
were coherent with CUDA graphs preserved.

| config | MMV3 b8 ms q4/q6/q8 | engine b8 | engine b16 |
|---|---|---:|---:|
| scalar | 2.565 / 3.011 / 1.003 | 49.2, 49.1 | 51.0, 50.6 |
| Q4VEC | 2.461 / 3.011 / 0.999 | 50.1, 49.9 | 51.5, 51.3 |
| LDG | 2.765 / 3.060 / 0.976 | 47.9 | 50.8 |
| Q4VEC + LDG | 2.500 / 3.042 / 0.988 | 49.5 | 51.4 |

LDG alone regressed q4_K microbench substantially and b8 by 1.2--1.3 tok/s.
The combined form restores some q4 performance but still misses q6 microbench
non-regression (3.042 ms > 3.017 ms) and cannot meet the two-run +1.0 co-primary
promotion criterion. **SHELVED:** retain both Q4VEC and LDG defaults at 0; no
two-run winner protocol or longform rerun is warranted.



## E47 — occupancy and spill map (2026-07-12)

Rebuilt all requested thread/row configurations with MAXWELL_EVO_PTXAS_V=1.
For the b8 V3 kernels, ptxas reported zero stack frame and zero spill loads/stores
in every configuration. The q4_K/q6_K/q8_0 register triplets were: 64x4
72/56/56, 64x8 98/96/96, 128x8 71/56/56, 128x16 98/96/96, 256x8
61/48/40, and 256x16 71/64/56.

At b8, xs = 256*(8+1)*4 = 9,216 B/block; at b16 it is 17,408 B/block.
Using sm_50 limits (64 KiB smem, 64K regs, 32 blocks, 2048 threads), the
q4_K b8 block limits are:

| threads x rows | limiting blocks/SMM | resident threads | engine b8 |
|---|---:|---:|---:|
| 64x4 | 7 (smem) | 448 | 35.0 |
| 64x8 | 7 (smem) | 448 | 42.0 |
| 128x8 | 7 (smem/register tie) | 896 | 49.2 |
| 128x16 | 5 (registers) | 640 | 49.1 |
| 256x8 | 4 (registers) | 1024 | 39.8 (E38--E40) |
| 256x16 | 3 (registers) | 768 | 48.4 (E38--E40) |

The two new engine runs were coherent with CUDA graphs: 64x4 b8/b16 =
35.0/29.7 and 64x8 = 42.0/35.4 tok/s. At b8, 128x8 is a seven-block smem/register tie; however b16 is a residency cliff: xs=17,408 B limits 128x8 to three blocks/SMM while its q4_K register footprint permits six. The V3_CHUNK=128 b16 probe is therefore authorized and remains the next open Arc C experiment; 128x8 remains the current champion pending that gate.



### E47 addendum — q4_K V3_CHUNK128 b16 residency probe

The b16 shared-memory cliff was tested with a default-off, q4_K-only
MAXWELL_EVO_V3_CHUNK128 path. It stages 128 columns at a time (8,704 B at
b16) and dispatches only q4_K through the reduced-stage kernel; q6_K/q8_0
remain on the proven 256-column kernels for this fail-fast probe.

The variant compiled, passed EVO_TEST_MAX_REL=0.05 SIDECAR_TEST PASS, and
produced coherent CUDA-graph output beginning Paris. Its q4_K MMV3 b8
microbench regressed from the 2.586 ms anchor to **4.751 ms**. The direct
b16 engine read was **43.5 tok/s**, versus the E44 b16 anchor **50.9 tok/s**.

**SHELVED:** while halving the q4 shared stage makes the expected b16
residency available, the extra staging/unpack synchronization overwhelms that
benefit. Do not extend CHUNK128 to q6_K/q8_0; retain default 0 and 128x8
256-column staging as champion.



## E48 — token-budget mixed decode/prefill sweep (2026-07-12)

Added standalone $HOME/bench_ssd/evo/mixed_bench.py using AsyncLLMEngine with
the TP=4 champion construction. It runs eight concurrent 128-token greedy
decode streams, injects an approximately 1500-token prompt after five seconds,
and captures per-stream ITL plus injected-request TTFT. Every run used the
champion MAXWELL_EVO sidecar environment and retained CUDA graph capture.

| max_num_batched_tokens | before ITL p50/p95 | during ITL p50/p95 | after ITL p50/p95 | injected TTFT |
|---:|---|---|---|---:|
| 512 | 146.34 / 147.60 ms | 278.73 / 6309.80 ms | 147.78 / 149.02 ms | 19.124 s |
| 2048 | 146.74 / 148.01 ms | 278.46 / 15573.30 ms | 148.03 / 149.05 ms | 18.574 s |
| 8192 | 147.84 / 149.35 ms | 278.82 / 18436.27 ms | 148.20 / 149.37 ms | 18.632 s |

Steady decode cadence is equivalent across the sweep: a roughly 147--149 ms
per-stream ITL corresponds to approximately 54 aggregate tok/s for eight
streams, above the 49.3 b8 anchor. The harness's whole-run decode_tok_s is not
used for this comparison because it includes the intentional prefill stall.
TTFT remains around 19 s across budgets, showing decode-priority starvation
rather than a budget-sensitive prefill win.

**Serving recommendation: max_num_batched_tokens=512.** It is the smallest
budget with non-regressed steady decode and reduces during-prefill ITL p95 to
6.31 s, versus 15.57 s and 18.44 s for 2048/8192. The recommendation was
written to tools/maxwell/README.md.



## E49 — automatic prefix caching check (2026-07-12)

Enabled prefix caching at the selected 512 token budget. Startup confirmed
Mamba prefix caching in align mode. Three serial injected prompts shared the
same approximately 1024-token prefix. Per-submission TTFT was **19.948 s**,
then **2.373 s** and **2.372 s**, confirming the expected cache hit on repeats.
The initial request remains decode-priority-starved; APC is a repeated-prefill
latency feature, not a remedy for first-request TTFT.

The APC TP=4 CUDA-graph decode ladder stayed non-regressing and coherent:
b8 **49.1 tok/s**, b16 **50.7 tok/s**, with output beginning Paris. The
sidecar remained active under the champion environment. APC is therefore a
recommend-only launch option for workloads with repeated long prefixes; it
does not change the decode token-budget recommendation.



## E50 — TP all-reduce transfer audit (2026-07-12)

Followed the TP=4 fallback from cuda_communicator.py. The startup line
Using [] all-reduce backends refers only to optional accelerated backends
(NCCL symmetric memory, quick-reduce, FlashInfer, custom, and symmetric-memory),
not to a host-staged reduction path. At TP world size >1 the communicator first
constructs PyNcclCommunicator; when that optional wrapper is disabled, all_reduce
clones the device tensor and calls torch.distributed.all_reduce(out,
group=self.device_group). The normal PyNccl path likewise passes device pointers
directly to ncclAllReduce.

No cudaHostAlloc, pin_memory, pageable CPU buffer, or host-copy staging exists
in this fallback. The observed [] log therefore does not authorize a pinned
staging prototype: reduction is already handled by NCCL/PyTorch's device-group
path. No vllm/ edit was made. The b1 host-time observation from E9 remains a
profiling target, but it is not evidence of an application-owned pageable
all-reduce buffer.

## E51 — standard dense GGUF family compatibility (Qwen3-8B, Llama-3.1-8B) (2026-07-12)

Verified two additional 8B-class GGUF candidates with the generic TP=4
CUDA-graph benchmark construction (no tokenizer, HF config, or patches).
Both files contain only quant types `[0, 12, 14]`, which are in the sidecar's
fused support set. Qwen3-8B's loader resolved `qwen3` / `Qwen3ForCausalLM`
and reached weight loading.

It then failed in the branch's standard dense-attention implementation: Triton
ptxas parses `acq_rel` / scoped atomics that require a newer GPU architecture
than `sm_50`. `VLLM_ATTENTION_BACKEND=TORCH_SDPA` does not exist in this
branch; it offers only `TRITON_ATTN` and `FLEX_ATTENTION`. FA2 requires `sm80`,
while Flex requires `torch.compile`, which is also blocked on `sm_50`.

Llama-3.1-8B was verified from a loader/quant perspective: architecture
`llama`, the same `[0, 12, 14]` quant set, and the same standard dense-attention
family. A full engine run would provoke the same already-explained
architectural failure, so it was not repeated.

This is a large platform compatibility gap, not a GGUF loader, quant-sidecar,
or sidecar numerical-correctness failure. No per-model decode ladder is
justified until `sm_50`-compatible attention is implemented.


## E52 — DeepSeek-V2-Lite compatibility (2026-07-12)

Verified the DeepSeek-V2-Lite-Chat `Q4_K_M` GGUF candidate. Its quant types
`[0, 6, 8, 12, 14]` are all in the sidecar's fused support set. The generic
benchmark construction failed before weight load or attention compilation with
the exact validation error:

```text
ValueError: GGUF model with architecture deepseek2 is not supported yet.
```

This is an architecture-layer gap for the MLA attention family, not a quant
sidecar gap. Do not custom-force it or treat it as a sidecar optimization; it
requires a scoped follow-up to implement `deepseek2` GGUF architecture support.


## E53 — per-family ladder scope decision (2026-07-12)

No new model family reached the per-model b1/b8/b16 decode ladder. Qwen3-8B
and Llama-3.1-8B are blocked by the `sm_50` dense-attention gap, and
DeepSeek-V2-Lite is blocked by missing `deepseek2` GGUF architecture support.
Therefore there is no valid new-family ladder or quant-sidecar regression
result to report; the existing Qwen3.5-9B champion is the only fully supported
candidate under this branch's `sm_50` constraints.


## E54 — speculative decoding evaluation (2026-07-12)

Evaluated the zero-model ngram proposer and the native Qwen3.5 MTP path with
the TP=4 champion environment, `Qwen3.5-9B-FIXED.gguf`, CUDA graphs, greedy
128-token decode, and the b1/b8 ladder. Ngram configuration uses
`spec_method="ngram"`, one speculative token, and
`speculative_config={"prompt_lookup_max": 4}`; `prompt_lookup_max` is nested
because it is not a flat `EngineArgs` field.

The initial ngram launch exposed a missing optional runtime dependency:
`vllm.v1.spec_decode.ngram_proposer` imports `numba`. Installed the branch's
pinned `numba==0.65.0` (and compatible `llvmlite==0.47.0`) in the isolated
benchmark environment. Runtime verification after installation reported
`numpy=2.3.5`, `numba=0.65.0`, and the existing vLLM build imported successfully.

The completed ngram run loaded the drafter, retained the sidecar and CUDA-graph
path, and produced coherent output beginning `Paris.`. It also explicitly
disabled asynchronous scheduling, which this branch does not support with
ngram speculation. Results were:

| configuration | b1 tok/s | b8 aggregate tok/s | coherence |
|---|---:|---:|---|
| E44 champion baseline | 16.8 | 49.3 | pass |
| ngram, max lookup 4, one token | 9.9 | 45.5 | pass |

Ngram fails the b1 admission gate: 9.9 tok/s is well below both the 16.8 tok/s
champion and the required +10% threshold (18.5 tok/s). It is also a b8
regression. The failed first run's flat `prompt_lookup_max` argument and missing
`numba` dependency were corrected before this final measurement; the table is
from the successfully initialized engine.

The native MTP configuration (`spec_method="mtp"`, one speculative token) was
then attempted because the Qwen3.5 HF configuration advertises one MTP layer.
Configuration validation rejected it before engine startup with:

```text
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

Thus MTP is not available for this GGUF target in the current branch. Neither
option has performance headroom; no external drafter was benchmarked because
there is no positive b1 result or acceptance-rate evidence to justify consuming
additional model memory. Keep speculative decoding disabled for the champion
serving configuration.


## E55 — disaggregated serving prerequisites (2026-07-12)

Installed `nixl==1.3.1` (including its CUDA 12 wheel) in the Maxwell benchmark
environment. `from vllm.distributed.nixl_utils import NixlWrapper` succeeded and
reported `NIXL is available`; the existing `numpy` version remains `2.3.5`.

Re-ran the TP=4 champion decode ladder with the required
`VLLM_SSM_CONV_STATE_LAYOUT=DS` setting, CUDA graphs, and the full champion
environment. Coherence passed (`Paris.`) and the measured output throughput was:

| batch | DS tok/s | prior anchor tok/s | delta tok/s |
|---:|---:|---:|---:|
| 1 | 16.8 | 16.8 | 0.0 |
| 8 | 49.2 | 49.3 | -0.1 |
| 16 | 49.1 | 50.9 | -1.8 |

The E48-era b16 anchor recorded in the disaggregation plan was an offline
`max_num_seqs=16` reference, while this reusable ladder remains configured with
`max_num_seqs=8`; its throughput is saturated at the b8 value. The DS layout is
therefore coherent and preserves the champion's admitted b8 decode rate, but a
like-for-like b16 comparison requires a `max_num_seqs=16` run before treating
b16 as an admission gate.

The monolithic OpenAI-server smoke used TP=4 on dies 0–3, DS layout, a 512-token
batch budget, and `{"mode": 0, "cudagraph_mode": "FULL"}`. It returned the
greedy completion `Paris.` for `The capital of France is`. The server log
contained all required standing checks: `Platform plugin maxwell is activated`,
`Capturing CUDA graphs (decode, FULL)`, and `Kernel JIT monitor activated`.
The server was stopped after the check. These results clear NIXL installation,
DS-layout coherence, and serve translation prerequisites for 1P1D bring-up.

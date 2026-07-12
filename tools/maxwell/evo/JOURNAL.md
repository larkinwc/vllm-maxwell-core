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

## FINAL config (end of hillclimb arc)

`Qwen3.5-9B-FIXED.gguf` · TP=4 · CUDA graphs ·
`MAXWELL_EVO_MMVQ=1 MMVQ_MAX=16 Q4K_MSUM=1 V2=1 V3=1 V3_MIN_B=2
V3_THREADS=128` (default policy mmvq)
→ v1 MMVQ (b1) / v3 FFMA (b2–16) / dequant+cuBLAS (b>16 + prefill).

decode: b8 49.0 · b16 50.3 · b64 ~105 · single 16.5
prefill: ~86 tok/s (TTFT 5.9 s @512)

Next mountain (designed, not started): GDN chunked-prefill CUDA port
(llama.cpp gated_delta_net.cu full-sequence kernel) — the torch-native
eager chunk scan is the remaining prefill wall after the MMQ fix.

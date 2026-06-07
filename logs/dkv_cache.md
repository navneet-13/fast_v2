# dKV-Cache (original) — results

Standalone dKV-Cache config (`FAST_DLLM_DKV_CACHE=1`, `FAST_DLLM_DKV_REFRESH_STEPS=N`):
the original delayed-KV-cache idea from arXiv:2505.15781 (github.com/horseee/dKV-Cache) —
one-step delay, periodic refresh, **NO FOCUS eviction**, all-layer freeze. DynamicCache
substrate (variant A). Decode variant. GSM8K flexible-extract, 1000 samples, b16,
max_seq_len=1024, eager, threshold=0.9, block_size=32.

## Results (b16, 1000 samples)

| refresh_steps | accuracy | TPS | log |
|---|---|---|---|
| **1** (full recompute every step — no caching) | 0.771 | 214.9 | dkv_b16_r1.log |
| 2 | 0.776 | 234.9 | dkv_b16_r2.log |
| 4 | 0.760 | 248.8 | dkv_b16_r4.log |
| 8 | 0.745 | 246.7 | dkv_b16_r8.log |
| dense baseline (static KV, eager) | 0.826 | 168.9 | (reference) |
| FOCUS delayed (static, flash) | 0.825 | 274.6 | (reference) |
| dynamic-FOCUS (flash, all-deep-layer reimpl) | 0.802 | 289.9 | (reference) |

(b1/50 smoke: 0.72 / 92.8 TPS, coherent — dkv_smoke_b1.log.)

## Findings

- **Correctness validated, not a caching bug.** GPU unit test
  `test_forward_dkv_full_equals_fed_all` passes: a full step and a cached step feeding ALL
  positions give identical argmax — the gather/scatter/buffer/one-step-delay machinery is a
  faithful no-op. The bookkeeping (`_dkv_fed_indices`, two-step shift) is unit-tested on CPU.

- **OPEN PROBLEM: the accuracy gap is NOT root-caused.** `refresh=1` (a full recompute every
  step — no caching, no eviction) lands at 0.771, which is below BOTH the dense baseline (0.826)
  AND dynamic-FOCUS (0.802). That ordering is backwards: a full-recompute-every-step path should
  be at least as accurate as dynamic-FOCUS, which *evicts* tokens. So there is an unexplained
  discrepancy in `forward_dkv`'s full path beyond expected rounding. The GPU no-op test only
  proved internal consistency (`forward_dkv` full == fed-all); it never checked `forward_dkv`
  full-step against the real dense `self.forward`. Root cause is unestablished — see the
  diagnostic in the next section. (An earlier draft of this doc blamed "all-layer flash
  reimplementation"; that was WRONG — static FOCUS also uses a flash kernel
  `flash_kvcache_attention` and is near-lossless at 0.825, so flash is not the differentiator.)

- **Delayed caching behaves as the dKV paper predicts (this part is sound).** Increasing the
  refresh interval trades accuracy for speed: acc 0.776 (r2) → 0.760 (r4) → 0.745 (r8); TPS
  235 → 249. The *caching* drift (~2–3pt across r2→r8) is small relative to the unexplained
  baseline gap. So whatever floors the accuracy is in the full-recompute path, not the caching.

- **Net vs the field:** dKV is neither the accuracy leader (dense 0.826, FOCUS-delayed 0.825)
  nor the throughput leader (dynamic-FOCUS b64 = 300, FOCUS-delayed b16 = 274.6). At b16 it sits
  at ~0.77 / ~235–249 TPS. Its value is as a **faithful baseline** of the original dKV idea
  (no eviction) to contrast against FOCUS's importance eviction — and it confirms that, in
  block diffusion where blocks are already small (32) and the prefix is already cached, the
  standalone delayed cache gives less headroom than it does in full-sequence diffusion (LLaDA/
  Dream), where the active region is the whole sequence.

## REOPENED — hand-rolling was NOT the cause (lossless dynamic rewrite refuted it)
Earlier this doc claimed variant A's gap was caused by `forward_dkv` hand-rolling attention,
"proven" by variant B (static, real `self_attn`) hitting 0.826. **That conclusion was WRONG.**
A later build routed the DYNAMIC `forward_dkv` full step through the real `decoder_layer`/`self_attn`
(provably bit-identical to the real layers: diag argmax-agreement 1.0). Its accuracy was
**UNCHANGED** vs the hand-rolled version (b16 r1 0.782 vs 0.771 — within noise). So the
hand-rolling never mattered. The dynamic (~0.78) vs static (0.826) gap is **structural in the
PATH**, not the forward attention. Since plain `batch_sample` (DynamicCache + `eval_mask`) = 0.821,
DynamicCache itself is fine. The remaining difference between the dynamic and static dKV paths
(SDPA + `attention_mask=None` over `cat(prefix, block)` vs `flash_kvcache` + per-seq `cache_seqlens`)
is the open suspect. Root cause UNRESOLVED. See "Lossless-attempt" section below for the data.

## Variant B (StaticKVCache) results — near-lossless
`FAST_DLLM_DKV_CACHE=1 FAST_DLLM_DKV_STATIC=1`. Full/refresh step = dense forward via the real
`self_attn` (scratch write, no advance); cached step = `write_sparse` + `flash_kvcache` over the
static buffer for all layers. b16, 1000 samples, max_seq_len=1024, eager, threshold=0.9.

| refresh_steps | accuracy | TPS | log |
|---|---|---|---|
| **1** (full recompute every step = dense) | **0.826** | 170.2 | dkv_static_b16_r1.log |
| 2 | 0.817 | 187.8 | dkv_static_b16_r2.log |
| 4 | 0.815 | 200.5 | dkv_static_b16_r4.log |
| 8 | 0.733 | 194.9 | dkv_static_b16_r8.log |
| dense baseline | 0.826 | 168.9 | (reference) |

(b1/50 smoke: 0.74 / 78.7 TPS, coherent — dkv_static_smoke_b1.log.)

**Findings:**
- **`refresh=1` == dense exactly (0.826 == 0.826).** The correctness anchor passes; the static
  full path is numerically lossless.
- **Near-lossless caching in the refresh 2–4 sweet spot:** 0.817 / 0.815 (≤1.1pt below dense) at
  +11–19% TPS (170 → 188 → 200). This is the static-FOCUS-delayed-comparable result (0.825/274.6
  uses block-cache stacking; here without it). Refresh 8 over-drifts (0.733).
- **Static B >> dynamic A on accuracy** at matched refresh (r2: 0.817 vs 0.776; r4: 0.815 vs
  0.760) — the delegation to `self_attn` is the difference.
- **Trade-off vs A:** B is near-lossless but lower TPS (peak ~200) and inherits the StaticKVCache
  `max_seq_len` OOM ceiling (low/mid batch only). A is lossy (~0.77) but higher TPS (~290) and
  runs at high batch. Pick B for accuracy, A for throughput.

**Bug fixed during B4:** the static full step initially used `update_past_key_values=True`, which
commits + advances `_seq_len` every step; with multiple full steps per block (warmup + refresh)
the write positions drifted off the buffer → OOB `write_sparse`. Fix: full step uses
`update_past_key_values=False` (scratch write, no advance), matching static FOCUS's diffusion
forwards; only the block-commit step advances. (`generation_functions.py` batch_sample_dkv_static.)

## (superseded) earlier OPEN PROBLEM note
The dynamic `refresh=1` < dynamic-FOCUS puzzle is resolved above; flash was correctly ruled out
(static FOCUS uses `flash_kvcache_attention` at 0.825). The remaining dynamic-path floor is the
hand-rolled attention, not the kernel or the caching.

## Correctness validation
- Unit (CPU): `tests/test_dkv.py` — `_dkv_fed_indices` bookkeeping ×3, all-layer
  `DynamicBlockKV` scatter ×1.
- Unit (GPU): `test_forward_dkv_full_equals_fed_all` — full-step == fed-all no-op invariant.
- End-to-end: GSM8K b1 smoke + b16 refresh sweep {1,2,4,8}, all coherent, no OOM.
- No-regression: `FAST_DLLM_DKV_CACHE=0` leaves all existing paths unchanged (additive
  methods + one eval dispatch branch).

## Implementation
- `models/.../modeling.py` — `Fast_dLLM_QwenModel.forward_dkv` (all-layer, fed-set-driven) +
  LM wrapper; `Fast_dLLM_QwenForCausalLM._dkv_fed_indices` (two-step-shift fed set).
- `generation_functions.py` — `batch_sample_dkv` (clone of `batch_sample_focus_dynamic`,
  dKV step swapped in: all-layer `DynamicBlockKV(deep_layer_start=0)`, refresh trigger,
  one-step-delay shift).
- `eval.py` — dispatch `FAST_DLLM_DKV_CACHE=1`.

Spec/plan: docs/superpowers/specs/2026-06-04-dkv-cache-design.md,
docs/superpowers/plans/2026-06-04-dkv-cache.md.

## Next steps (deferred, need user confirmation)
- Near-lossless variant: route `forward_dkv` per-layer attention through the real `self_attn`
  over a materialized `cat(prefix, buffer)` to remove the all-layer reimplementation floor.
- Variant B: StaticKVCache substrate (`batch_sample_dkv` on the static buffer) — comparable to
  static FOCUS-delayed; was planned to follow A.
- Push batch size (dKV's TPS may scale like the dynamic path past b16).

## Lossless-attempt: dynamic dKV with real self_attn full step (HYPOTHESIS REFUTED)
Rewrote forward_dkv to use the real decoder_layer/self_attn for the full step (provably
bit-identical to the real layers: diag argmax-agreement 1.0 vs a None-mask real-layer loop).
**Result: accuracy UNCHANGED vs the hand-rolled version** (b16/1000): r1 0.782, r2 0.774,
r4 0.763, r8 0.737 — within noise of variant A (0.771/0.776/0.760/0.745). TPS 206/223/235/231.

**Conclusion: the hand-rolled attention was NOT the cause of the dynamic gap.** The original
"full step diverges 0.9688" diagnostic was confounded by None-vs-eval_mask (benign; static uses
None and is lossless). The dynamic gap (~0.78 vs static 0.826) is structural in the dynamic PATH,
not the forward. Plain batch_sample (DynamicCache + eval_mask) = 0.821, so DynamicCache is fine.
Prime suspect: forward_dkv uses attention_mask=None while self.forward uses eval_mask — None may
mishandle finished/padded sequences at b16 (static flash_kvcache uses per-seq cache_seqlens).
Next isolation test: run dynamic dKV refresh=1 with eval_mask. The lossless forward is kept
(clean + correct); old hand-rolled retained as forward_dkv_handrolled.

## RESOLVED — baseline-first rebuild + cached-step eval_mask fix

Root cause was the DECODE SCHEDULE, not the forward: the old `batch_sample_dkv` decoded the whole
block at once, while dense `batch_sample` decodes in `small_block_size=8` sub-blocks (semi-AR).
Fix = rebuild `batch_sample_dkv` as `batch_sample` with ONE substitution (the `use_block_cache=False`
forward → `forward_dkv(dkv_store=block_kv)`); per-block `DynamicBlockKV`, reset at block boundaries;
refresh/one-step-delay on that branch only. Nothing borrowed from the static sampler.
Also fixed `forward_dkv` cached step: was `attention_mask=None`, now reuses the SAME `eval_mask` and
gathers the fed rows (`full_mask[fed_indices]`) so a cached step is mask-identical to the full step
on the recomputed tokens.

Validation: `tests/diff_dkv_vs_baseline.py` (refresh=1, SBS=8) byte-identical to baseline (match=True
×3). `tests/diag_dkv_cached_mask.py` cached==full argmax agreement 1.0 at fed positions for both an
aligned prefix and a grid-straddling prefix.

b16/1000 GSM8K (flex-extract, threshold=0.9, eager), one run per GPU, same machine:
| config | accuracy | TPS | vs dense |
|---|---|---|---|
| dense baseline | 0.821 ±0.012 | 157.4 | 1.00× |
| dKV refresh=1  | 0.821 ±0.012 | 156.0 | 0.99× (lossless anchor, no speedup) |
| dKV refresh=2  | 0.826 ±0.012 | 175.5 | 1.11× (lossless + speedup) |
| dKV refresh=4  | 0.808 ±0.013 | 179.2 | 1.14× |
| dKV refresh=8  | 0.758 ±0.014 | 182.9 | 1.16× |

refresh=1 == dense (byte-identical confirmed at 1000). refresh=2 = lossless + ~11% TPS (sweet spot).
TPS saturates fast (r2→r8 only +4%) while accuracy falls; small recompute-skip at block_size=32 /
sub_block=8 / eager. Logs: logs/dkv_sweep_{dense,r1,r2,r4,r8}.log.

### batch_size=1 sweep (b1/300, same setup) — dKV does NOT help at b1

| config | accuracy | TPS | vs dense |
|---|---|---|---|
| dense baseline | 0.85  ±0.021 | 81.97 | 1.00× |
| dKV refresh=1  | 0.85  ±0.021 | 81.32 | 0.99× |
| dKV refresh=2  | 0.853 ±0.021 | 80.05 | 0.98× |
| dKV refresh=4  | 0.84  ±0.021 | 77.26 | 0.94× |
| dKV refresh=8  | 0.76  ±0.025 | 71.73 | 0.87× |

At batch=1 the GPU is latency-bound on tiny 32-token blocks, so the cached step's overhead (buffer
write/get/scatter + mask gather) is NOT offset by skipped recompute — TPS DROPS and worsens with
refresh (dense 82 → r8 72), the opposite of b16 (+11% at r2). Accuracy pattern matches b16: r1==dense
(lossless), r2 lossless, r4 within 300-sample CI (±0.021), r8 a real drop. dKV's benefit is
batch-size-dependent: throughput win at b16, slight latency cost at b1.
Logs: logs/dkv_b1_300_{dense,r1,r2,r4,r8}.log. (b1/1000 runs were killed before finishing.)

### Analytical FLOP-proxy saving (token-layers = q-tokens x layers; FAST_DLLM_TL_COUNT=1)

Token-layers proxy the dominant linear/projection FLOPs (K/V proj + attention-score + MLP all scale
with q-tokens processed). A full step = block_size(32) x layers(28) = 896 TL; a cached step =
num_fed x 28. Script: tests/dkv_flops.py (8 GSM8K prompts, max_new_tokens=256). Counters in
generation_functions._DKV_FLOP. full_equiv counts EVERY step (full AND cached) as full = total_steps
x 896 — i.e. this same trajectory with caching disabled.

| refresh | total steps | decode_TL (actual) | full_equiv | per-traj saving | vs-dense saving |
|---|---|---|---|---|---|
| 1 | 629 | 563,584 | 563,584 |  0.0% |   —    |
| 2 | 723 | 522,928 | 647,808 | 19.3% |  7.2%  |
| 4 | 718 | 451,108 | 643,328 | 29.9% | 20.0%  |
| 8 | 698 | 407,176 | 625,408 | 34.9% | 27.8%  |

Columns: total steps = diffusion steps that trajectory took (sum over 8 samples); decode_TL = actual
token-layers spent; full_equiv = total_steps x 896 (every step charged as full); per-traj saving =
1 - decode_TL/full_equiv (caching effect, fixed trajectory); vs-dense saving = 1 - decode_TL(rN)/
decode_TL(r1) (honest saving vs dense, since r1 is byte-identical to dense).

KEY: caching LENGTHENS the trajectory (r1=629 steps; r2/r4/r8=698-723) because approximate cached
steps need extra diffusion iterations to converge. So per-traj saving overstates the win. Clean
decomposition: net_retained_vs_dense = (steps_rN/steps_r1) x (1 - per_traj_saving)
e.g. r2: 1.149 x 0.807 = 0.928 -> 7.2% saved. The true compute saving at the lossless r2 is only
~7%, which is why b16 wall-clock gained ~11% (also fewer/smaller kernels) and b1 (latency-bound) got
slower. r8 saves ~28% FLOPs but drops accuracy to 0.76.

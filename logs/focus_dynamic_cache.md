# FOCUS on DynamicCache — results

FOCUS variant whose committed prefix uses a lazily-grown HF `DynamicCache` (+ a small per-block
KV buffer `DynamicBlockKV` + eager SDPA) instead of `StaticKVCache`. Carries the full token-skipping
+ delayed-KV-cache behavior. Purpose: break the StaticKVCache `max_seq_len` pre-allocation OOM
ceiling so FOCUS runs at high batch (where it wins).

Env: `FAST_DLLM_USE_FOCUS=1 FAST_DLLM_FOCUS_DYNAMIC=1 FAST_DLLM_FOCUS_COMPACT=1
FAST_DLLM_FOCUS_DELAYED_CACHE=1`. GSM8K flexible-extract, 1000 samples, max_seq_len=1024, eager,
threshold=0.9, block_size=32, alpha=1.0, layers 0,1. TPS = total tokens / gen wall-clock.

## Results (dynamic FOCUS + delayed cache)

| batch | accuracy | TPS | total gen time | s/batch | peak GPU mem | static path |
|-------|----------|-----|----------------|---------|--------------|-------------|
| 16 (SDPA) | 0.800 | 274.0 | 1209 s | 19.2 | ~16.0 GB | runs (static delayed: 0.825 / 274.6) |
| **16 (flash)** | **0.802** | **289.9** | 1138 s | 18.1 | — | flash kernel in `_attn` (`logs/dyn_b16_flash.log`) |
| **48** | 0.787 | **292.0** | 1135 s | 54.0 | ~18.2 GB | **OOM (could not run)** |
| **64** | 0.794 | **300.3** | 1118 s | 69.9 | ~19.1 GB | **OOM (could not run)** |

### Flash-kernel experiment (root-cause test for the ~2.3pt b16 gap vs static)
Hypothesis: the gap came from the dynamic path's deep-layer attention using `F.scaled_dot_product_attention`
while the static path uses the flash kernel (`flash_kvcache_attention`), so bf16 rounding diverges over 28
layers. **Test:** swapped the dynamic path's `_attn` helper to `flash_attn_func` (GQA-native, fp32 softmax
accumulation; SDPA fallback for CPU/no-flash) — `modeling.py` lines ~1316.
**Result: REFUTED.** Accuracy 0.800 → 0.802 (within ±0.0126 noise); the ~2.3pt gap vs static (0.825)
**persists**. Since flash already accumulates softmax in fp32 and it changed nothing, the **fp32-deep-attention
idea is also dead** — precision in the attention op is NOT the cause. The gap is **structural** (KV
layout / op-ordering / selection+delayed-cache interaction), pointing to the **rightmost-processed left-side
guard** (faithfulness deviation #1) as the real lever.
**Kept anyway:** flash gives **+5.8% TPS for free** (274.0 → 289.9, wall-clock 1209 → 1138 s) at equal
accuracy — now the b16 throughput leader. Default-on in the dynamic path; SDPA fallback retained.

(b1/50 smoke: 0.78 / 70.1 TPS, coherent — `logs/dyn_smoke_b1.log`.)

## Findings

- **Breaks the OOM ceiling (the deliverable).** b48 and b64 — which OOM on the static path at any
  safe max_seq_len (>=~1024, since some GSM8K CoT reach ~800 tokens) — run on the dynamic path at
  ~18-19 GB on the 23 GB A5000. The DynamicCache grows the prefix to the real length (~350) instead
  of eagerly allocating max_seq_len, leaving headroom.
- **TPS keeps climbing with batch, past where static dies:** 274 (b16) -> 292 (b48) -> 300 (b64).
  Total wall-clock for 1000 samples DROPS (1209 -> 1135 -> 1118 s). So **dynamic b64 = 300 TPS is the
  new throughput leader**, vs the previous best (static delayed FOCUS b16 = 274.6, static+block-cache
  b16 = 258).
- **At b16 it's at parity with the static path:** 274.0 vs 274.6 TPS (identical) — contrary to the
  pre-build worry that eager SDPA would be slower at low batch, it is not. Accuracy 0.800 vs static
  0.825 (~2.5pt lower): attributable to the deep layers reimplementing attention over
  cat(DynamicCache, DynamicBlockKV) with SDPA instead of delegating to self_attn/flash, so ~bf16
  rounding accumulates across 28 layers. Per-layer attention matches the real module to ~bf16 epsilon
  (0.002); the retain-all unit gate confirms the FOCUS gather/scatter/buffer/select path is a faithful
  no-op. Accuracy holds 0.787-0.800 across batch — healthy, no degradation.

## Correctness validation
- Unit (GPU): `tests/test_focus.py::test_focus_compact_dynamic_retain_all_matches_dense` — retain-all
  FOCUS == dense seed (bf16-robust no-op invariant) + argmax sanity vs the real dense forward.
- Unit (CPU): `tests/test_focus.py -k dynamic_block_kv` — buffer write_full / scatter.
- End-to-end: GSM8K b1/b16/b48/b64 all coherent, no OOM, accuracy 0.78-0.80.
- No-regression: `FAST_DLLM_FOCUS_DYNAMIC=0` is the untouched static FOCUS path (this feature only
  ADDS methods; static methods unchanged).

## Implementation
- `utils/dynamic_block_kv.py` — `DynamicBlockKV` per-block deep-layer KV buffer (BHSD).
- `modeling.py` — `Fast_dLLM_QwenModel.forward_focus_compact_dynamic` + LM wrapper (DynamicCache
  prefix + buffer + SDPA; reuses `_focus_select`/`_focus_importance`/`_focus_update_frozen`).
- `generation_functions.py` — `batch_sample_focus_dynamic` (clone of `batch_sample`'s DynamicCache +
  finished_samples loop, FOCUS step swapped in; per-block buffer/frozen reallocated to track
  batch compaction).
- `eval.py` — dispatch `FAST_DLLM_FOCUS_DYNAMIC=1`.

## Next steps
- The ~2.5pt b16 accuracy gap vs static could be probed (SDPA-vs-flash drift); likely closes with the
  §6 rightmost-processed left guard or by running deep-layer attention in fp32. Low priority.
- Push past b64 toward the new memory ceiling; sweep alpha at high batch.

Logs: logs/dyn_b16.log, logs/dyn_b48.log, logs/dyn_b64.log, logs/dyn_smoke_b1.log.
Design/plan: docs/superpowers/specs/2026-06-01-focus-dynamic-cache-design.md,
docs/superpowers/plans/2026-06-01-focus-dynamic-cache.md.

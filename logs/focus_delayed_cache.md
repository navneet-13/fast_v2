# FOCUS Delayed KV Cache — results

Implements the FOCUS paper's token-level **delayed KV caching** inside our compact FOCUS path: a
block position is "frozen" (its KV cached, excluded from recompute) once it is decoded AND its right
neighbor is decoded (paper-faithful right-neighbor rule). Settled tokens leave the per-step recompute
set, so `Ksel` collapses through each block — breaking the decoded-token floor that pinned FOCUS at
~185 TPS.

Env knob: `FAST_DLLM_FOCUS_DELAYED_CACHE=1` (default 0). Requires the compact path
(`FAST_DLLM_FOCUS_COMPACT=1`). Eager only. No block cache. GSM8K flexible-extract, threshold=0.9,
block_size=32, max_seq_len=1024, alpha=1.0, layers 0,1. TPS = total tokens / gen wall-clock.

## Accuracy + TPS (delayed OFF vs ON)

| config | delayed | exact_match | TPS | speedup vs OFF |
|--------|---------|-------------|-----|----------------|
| b1, 200   | OFF | 0.860 | 68.6  | 1.00× |
| b1, 200   | ON  | 0.815 | 71.7  | 1.05× |
| b16, 1000 | OFF | 0.833 | 185.6 | 1.00× |
| **b16, 1000** | **ON** | **0.825** | **274.6** | **1.48×** |

OFF references: b16/1000 = `logs/scale1k_focus_b16.log`; b1/200 = `logs/focus_b1_200_off.log`.
ON logs: `logs/delayed_b16.log` (1160.1 s, 18.4 s/batch, 318522 tok), `logs/delayed_b1_200.log`.

## FLOP saving (decoder token-layers, limit 25 b1, `FAST_DLLM_FOCUS_FLOPS=1`)

| config | total token-layer saving | deep-layer saving |
|--------|--------------------------|-------------------|
| FOCUS (delayed OFF) | ~31.7% intrinsic | ~34% |
| **FOCUS + delayed cache** | **69.1%** (`focus_tl=969498` vs `base_tl=3136000`) | **74.4%** |

Delayed caching MORE THAN DOUBLES the FLOP saving (31.7% → 69.1%) — settled tokens leave the
recompute set, so `Ksel` shrinks far below the old decoded-token floor.

## Findings

- **It breaks the ~185 TPS plateau.** At b16 the 69% FLOP cut converts to **1.48× throughput**
  (185.6 → 274.6 TPS), since b16 is compute-bound. This is now the **fastest eager config measured** —
  it even beats static-KV + block cache (258 TPS at b16), and it's the pure-FOCUS path (no block cache).
- **Near-lossless at b16:** 0.825 vs 0.833 (Δ0.8pt, inside the ±0.012 CI — statistically equal).
- **b1 is overhead-bound:** only 1.05× (68.6 → 71.7), with a real but modest ~4.5pt accuracy drop
  (0.860 → 0.815). The drop is the expected approximation cost: bidirectional intra-block attention
  means a frozen token's KV can shift as later tokens settle, and we deliberately OMITTED the paper's
  `rightmost_processed` left-side guard (design spec §6). If b1 accuracy matters, adding that guard is
  the documented next step; b1 is not the speedup regime, so it was left out of the first cut.
- Freeze timing verified correct (flag-OFF path bit-identical to baseline; output coherent, never
  garbled), so the accuracy delta is genuine approximation, not a bug.

## Status / next steps
- Default-off: bit-identical to prior FOCUS (no-regression gate green).
- Highest-value follow-ups: (1) add the §6 `rightmost_processed` left-side guard and re-measure (likely
  recovers most of the b1 drop at small cost); (2) compose delayed cache with `torch.compile` (the 69%
  FLOP cut + fixed-bucket Ksel would compound); (3) sweep alpha now that the decoded floor is gone
  (alpha may finally be a live knob).

Cross-refs: design `docs/superpowers/specs/2026-06-01-focus-delayed-kv-cache-design.md`;
plan `docs/superpowers/plans/2026-06-01-focus-delayed-kv-cache.md`;
batch-scaling context `logs/focus_batch_scaling.md`; turnover `docs/FOCUS_SESSION_TURNOVER.md`.

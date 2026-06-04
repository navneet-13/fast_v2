# FOCUS batch-size scaling — does speedup grow at higher batch?

GSM8K flexible-extract, 1000 samples, **max_seq_len=1024** (lowered from 2048 to fit batch on the
23 GB A5000), eager, threshold=0.9, block_size=32. FOCUS = compact, alpha=1.0, layers 0,1.
Baseline = static-KV dynamo eager (no block cache). TPS = total tokens / gen wall-clock.

| batch | baseline acc | baseline TPS | FOCUS acc | FOCUS TPS | FOCUS speedup |
|-------|--------------|--------------|-----------|-----------|---------------|
| 16    | 0.826        | 168.7        | 0.833     | 185.6     | 1.10×         |
| 24    | 0.822        | 165.3        | 0.828     | 187.1     | 1.13×         |
| 32    | 0.829        | 162.8        | 0.823     | 184.0     | 1.13×         |

## Finding: higher batch does NOT increase FOCUS speedup — it plateaus at ~1.13×
- **FOCUS TPS is flat** (185.6 → 187.1 → 184.0): FOCUS does not accelerate with batch.
- The ratio rises only because the **baseline degrades** with batch (168.7 → 162.8).
- Root cause = the batch-MAX Ksel budget in `_focus_select` (modeling.py:277):
  `Ksel = clamp(retain.sum(dim=1).max(), min=K, max=S)`. The deep-layer compute is a uniform
  `[B, Ksel, D]` tensor sized by the single most-retaining sequence. More sequences → higher
  batch-max retained count → Ksel widens → savings-per-token shrink → cancels the compute-bound
  batching benefit. FOCUS is pinned at ~185 TPS.
- Accuracy: all statistically equal (0.822–0.833); FOCUS ≥ baseline at every batch. No batching degradation.

## Implications
- "Less aggressive" (↑alpha) would push FOCUS TPS BELOW 185; "higher batch" alone buys nothing.
  The two goals are in tension under the current batch-max design.
- The real lever to raise FOCUS throughput = **percentile/mean Ksel budget** (decouple from the
  worst-case sequence; force-evict overflow in token-hungry sequences). This is a COMPUTE change —
  it does NOT reduce the static-KV memory ceiling, so it stays feasible within b32.
- On pure speed FOCUS (~185) is not competitive with static-KV + block cache (258 @ b16). FOCUS's
  value is the orthogonal FLOP saving → the real upside is **FOCUS composed with block cache** (untested).

## Memory ceiling (why b48+ is out on a 23 GB A5000)
StaticKVCache pre-allocates the full `[B, max_seq_len, H, D]` per layer + scratch (~112 MiB/row at
1024) eagerly, regardless of actual gen length. FOCUS is locked to this static cache (can't use a
lazily-grown DynamicCache like the plain baseline), so its batch ceiling is structurally lower:
- b32 just fits (needed `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for FOCUS).
- b48 OOMs at every max_seq_len that's safe. Can't cut max_seq_len below ~960–1024 because some
  GSM8K chains-of-thought reach ~800 generated tokens (+~100 prompt) — lower would truncate and
  corrupt accuracy on the hardest problems. So b48+ needs a bigger card or multi-GPU, not tuning.

Logs: logs/scale1k_{baseline,focus}_b{16,24,32}.log.

## Alpha-down sweep (more-aggressive attempt) — alpha is a DEAD knob
Batch 16, 1000 samples, max_seq_len=1024, compact FOCUS. Goal: skip more tokens via lower alpha.

| alpha | accuracy | TPS | tokens |
|-------|----------|-----|--------|
| 0.25  | 0.832    | 183.2 | 323152 |
| 0.50  | 0.832    | 182.1 | 323152 |
| 0.75  | 0.833    | 182.2 | 323215 |
| 1.00  | 0.833    | 185.6 | 323393 |

Finding: lowering alpha 4× does NOT increase TPS (flat-to-slightly-down, within noise), token counts
near-identical (<0.1%), accuracy unchanged. alpha=0.25 and 0.50 gave BIT-IDENTICAL token counts
(323152) → the top-K term alpha controls contributes nothing below ~0.75. Ksel is pinned by the two
floors alpha can't touch: (1) all decoded tokens (always retained, correctness), (2) threshold
mustkeep (delta >= mean+std). Shrinking K just slides under the floor → inert.

To actually skip more, attack the floors:
- (A) Tighten threshold mean+std -> mean+2std or hard fraction cap (small code change; modest, still
  bounded by decoded floor).
- (B) Stop recomputing decoded tokens every step — refresh each decoded token's KV ONCE (step after
  it decodes), then evict from recompute. This breaks the decoded floor. ≈ FOCUS composed with the
  block cache (which already caches decoded sub-blocks' KV). Highest-value path.

Logs: logs/alpha_focus_b16_a0{25,50,75}.log; alpha=1.0 ref = logs/scale1k_focus_b16.log.

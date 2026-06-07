# dKV-Cache (original) for Fast-dLLM v2 — Design

**Status:** approved design, pre-implementation.
**Date:** 2026-06-04.
**Author:** session handoff (FOCUS/Fast-dLLM-v2 work).

## 1. Goal

Add a standalone **dKV-Cache** decode config to Fast-dLLM v2 that implements the
*original* delayed-KV-cache idea from **"dKV-Cache: The Cache for Diffusion Language
Models"** (arXiv:2505.15781, github.com/horseee/dKV-Cache) — **not** FOCUS's adaptation.
The distinguishing properties vs our existing `FAST_DLLM_FOCUS_DELAYED_CACHE`:

- **No importance eviction.** Cache *all* decoded tokens; recompute *all* still-masked
  tokens. The cache/recompute decision is purely decoded-vs-masked, never a score.
- **Faithful one-step delay.** A token decoded at step *t* is recomputed once more at
  *t+1* (as a clean token), then its K/V is frozen from *t+2* onward.
- **Periodic refresh.** Every `refresh_steps` steps the active region is fully
  recomputed, bounding staleness drift. FOCUS has no refresh (it freezes permanently).
- **All-layer freeze.** Decoded tokens are cached across *all* layers (FOCUS only
  evicts in the deep layers after measuring importance at layers 0–1).

Purpose: a clean, paper-faithful baseline to compare against FOCUS and the dense
baseline (accuracy AND TPS), and to study the refresh-interval trade-off.

## 2. Reference mechanism (from horseee/dKV-Cache, decode variant)

Per diffusion step `i` over the generation region:

```
prv_transfer_idx = all-decoded-as-of-(i-1)         # fed set = ~prv_transfer_idx
if i % refresh_steps != 0 and i > 1 and cache:      # cached step
    feed x[~prv_transfer_idx]                       # still-masked ∪ just-decoded-last-step
    attend over cached KV (decoded ≥2 steps ago) + fresh fed KV
elif i refresh / i<=1:                               # refresh / warmup
    feed whole x, rebuild full cache
# decode → transfer_index (threshold/confidence); x[transfer_index] = x0
# two-step shift:
prv_transfer_idx, cur_transfer_index = cur_transfer_index, all_decoded
```

The one-step delay is automatic: a token decoded at `i` is in `~prv_transfer_idx` at
`i+1` (fed, recomputed clean), and only excluded (cached) from `i+2`. Variants in the
repo: `decode` (this), `prefill` (cache prompt only), `pd`/greedy (more aggressive).
We implement the **decode** variant.

## 3. Mapping onto Fast-dLLM v2 (block diffusion)

Decision: **within-block + refresh** (not a semi-AR re-architecture). Committed blocks
stay in the normal DynamicCache prefix; the dKV logic applies to the active block
(`block_size`=32). The refresh recomputes only the 32-position active block, never the
prefix — so it is cheap here, unlike full-sequence diffusion.

**Per active block, each diffusion step:**
1. Track decode status via the two-step shift over block positions →
   `fed = ~prv_transfer_idx` (masked ∪ just-decoded-last-step); `cached` = decoded ≥2.
2. Forward only `fed` tokens through **all** layers, attending over
   `cat(DynamicCache prefix, cached block-KV, fresh fed-KV)` via the flash `_attn`.
   Gather position embeddings for the `fed` subset (reuse dynamic path's `sel_cos/sel_sin`).
3. Write `fed` K/V into an **all-layer within-block buffer**.
4. Decode (threshold/confidence) → `transfer_index`; `x[transfer_index] = x0`.
5. Two-step-shift bookkeeping → next step's `prv_transfer_idx`.
6. Refresh: if `step % refresh_steps == 0`, feed all 32 positions, rebuild the buffer.
7. Block done → commit to DynamicCache prefix via the existing clean dense forward
   (`update_past_key_values=True`).

**Cleaner than the dynamic-FOCUS path:** `fed = masked ∪ just-decoded`, so every masked
position is recomputed every step. The only cached slots hold *clean* K/V (decoded ≥2).
The stale-mask-KV bug class from the FOCUS dynamic path cannot occur here.

## 4. Components & files

| File | Change |
|---|---|
| `utils/dynamic_block_kv.py` | Reuse `DynamicBlockKV(deep_layer_start=0, …)` for the all-layer buffer — `write_full`/`write`/`get`/`compact_batch` work as-is. No structural change. |
| `models/.../modeling.py` | **New `forward_dkv`** (model + LM wrapper). No importance/layer-split. Inputs `fed_indices` + `refresh`. Feeds `fed` through all layers, writes K/V to the all-layer buffer, attends `cat(prefix, buffer)`. `refresh=True` ⇒ `fed = all 32`. Returns logits for fed positions; caller scatters back (`-inf` refill for non-fed). |
| `generation_functions.py` | **New `batch_sample_dkv`** — clone of `batch_sample` (DynamicCache + `finished_samples` + batch-compaction). Inner block loop = dKV loop: all-layer buffer + two-step-shift status + refresh trigger. Commit via plain dense forward. |
| `eval.py` | Dispatch `if FAST_DLLM_DKV_CACHE=1 → batch_sample_dkv`. |
| Env | `FAST_DLLM_DKV_CACHE` (on/off), `FAST_DLLM_DKV_REFRESH_STEPS` (default 4). |
| `tests/test_dkv.py` | Unit + e2e (Section 6). |

**Variant B (after A is green):** a StaticKVCache `batch_sample_dkv` — same logic on the
static buffer (`write_sparse` + `scratch_seqlens` for the `fed` subset), comparable to the
static FOCUS-delayed config. Inherits the static OOM ceiling; lower priority than A.

## 5. Refresh semantics & edge cases

- Step 0 (seed) and step 1: full forward (warmup, `i > 1` guard). Step ≥2: cached, except
  `step % refresh_steps == 0` → full block recompute.
- **`refresh_steps=1` ⇒ every step full ⇒ identical to dense baseline** (correctness anchor).
- Block boundary: reset buffer + status; seed re-establishes.
- Block commit: clean dense forward writes the clean full-block KV to the prefix.
- Batch compaction on finish: compact the buffer (`compact_batch`) AND the status tensors
  (`prv_transfer_idx`, `cur_transfer_index`) along batch dim (mirror `batch_sample_focus_dynamic`).
- Short blocks (< refresh_steps): no refresh fires; fine.
- Attention `is_causal=False` over `cat(prefix, 32-slot buffer)` (block-causal as in the
  existing dynamic path).

## 6. Testing

**Unit (`tests/test_dkv.py`):**
1. **`refresh_steps=1 == dense` (GPU)** — the correctness anchor; outputs match dense
   `batch_sample` to bf16 tolerance. (dKV analogue of the FOCUS retain-all gate.)
2. **One-step-delay bookkeeping (CPU)** — factor `dkv_fed_set(prv, cur, all_decoded)`;
   scripted decode sequence asserts `fed == masked ∪ just-decoded`, `cached == decoded-≥2`.
3. **All-layer buffer (CPU)** — `DynamicBlockKV(deep_layer_start=0)` write_full + scatter.

**End-to-end (GSM8K, accuracy AND TPS):**
4. Smoke b1/50 — coherent, no crash.
5. b16/1000 vs dense (0.826), FOCUS-delayed (0.825), dynamic-FOCUS (0.802); dKV should be
   near-lossless.
6. `refresh_steps` sweep {1, 2, 4, 8} at b16 — accuracy/TPS trade-off; `refresh_steps=1`
   must equal dense accuracy.

**No-regression:** `FAST_DLLM_DKV_CACHE=0` leaves all existing paths unchanged (feature
only adds a function + dispatch branch).

## 7. Success criteria

- `refresh_steps=1` is byte/bf16-equivalent to dense (test 1 green).
- b16 GSM8K near-lossless vs dense (within ~1pt) at a refresh interval that still gives a
  TPS win.
- Static path and all existing FOCUS/dynamic configs unchanged with the flag off.
- A (DynamicCache) delivered and validated before B (StaticKVCache) begins.

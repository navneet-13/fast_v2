# FOCUS on DynamicCache — Design

**Date:** 2026-06-01
**Status:** Approved (design), pending spec review → plan
**Goal:** A FOCUS variant whose committed-prefix KV uses a lazily-grown `DynamicCache` (like the plain
`batch_sample` baseline) instead of `StaticKVCache`, so it stops OOM-ing at high batch — carrying over
the full token-skipping + delayed-KV-cache behavior we already built on the static path.

---

## 1. Motivation

`StaticKVCache` eagerly pre-allocates `[B, max_seq_len, H, D]` per layer (+scratch), regardless of how
short generations are. That ~112 MiB/batch-row at `max_seq_len=1024` is exactly what OOMs FOCUS at
batch ≥ 48 on a 23 GB A5000 (see `logs/focus_batch_scaling.md`). The plain `batch_sample` baseline uses
`DynamicCache`, which grows the prefix to the *actual* sequence length (~350 tokens for 0-shot GSM8K),
so it reaches much higher batch before OOM. Since high batch is exactly where FOCUS + delayed cache
wins (b16: 1.48×, near-lossless — `logs/focus_delayed_cache.md`), a dynamic-prefix FOCUS unlocks the
batch sizes where the speedup matters.

## 2. Key constraint (why a per-block buffer is unavoidable)

In the DynamicCache diffusion path, each step recomputes the full block KV and `cat`s it with the
committed prefix (`modeling.py:375-377`, `update_past_key_values=False`); the block KV is ephemeral.
FOCUS's benefit is *not* recomputing evicted/frozen tokens — so those tokens' deep-layer KV must
persist across diffusion steps somewhere. On the static path that "somewhere" is the StaticKVCache
scratch region. On the dynamic path we add a small **per-block KV buffer**. This is the FOCUS
gather/scatter buffer (the same role `block_sparse_cache` plays in the non-compact `forward_focus`),
**NOT** the sub-block `small_block_size` "block cache" (which is explicitly out of scope).

## 3. Architecture

- **Committed prefix:** `DynamicCache` (HF), cat-grows per committed block. Replaces `StaticKVCache`.
- **Per-block buffer:** `[B, num_kv_heads, block_size, head_dim]` per deep layer (`max(focus_layers)+1
  .. num_layers-1`), reset at each block start. Holds the current block's deep KV across diffusion
  steps. ~1.7 MB/batch-row total (negligible).
- **Attention (deep layers):** eager SDPA over `cat(prefix_kv[L], block_buffer[L])` — `is_causal=False`
  (block is bidirectional; full prefix visible), `repeat_kv` to expand GQA heads. No flash_kvcache, no
  dynamo, no static buffer.
- **Token-skipping + delayed cache:** unchanged. Reuses `_focus_select(..., frozen=)` and the
  `_focus_update_frozen` staticmethod exactly as the static compact path does.

## 4. Components

### 4a. `Fast_dLLM_QwenModel.forward_focus_compact_dynamic` (+ LM wrapper)
Mirrors `forward_focus_compact` but for DynamicCache + per-block buffer + SDPA. Signature parallels
`forward_focus_compact` plus the per-block buffer object (passed in) and accepts `frozen=None`.
- **Seed step** (`is_dense_step=True`): run all layers dense over the full block; for each deep layer,
  write the full block KV into the per-block buffer; cache the seed-final (pre-norm) hidden in
  `self._focus_dyn_seed_hidden`; return logits. Attention each layer = SDPA over `cat(prefix, block)`.
- **FOCUS step:** layers 0,1 dense + `_measure_importance`; `delta`; `_focus_select(delta, mask_idx,
  avg_decoded, focus_alpha, retain_override, frozen=frozen)` → `token_indices, num_tokens`; gather
  selected hidden → for each deep layer: compute selected q/k/v, **scatter** selected k/v into the
  per-block buffer at `token_indices`, SDPA attend `q_sel` over `cat(prefix[L], block_buffer[L])`,
  MLP; **scatter** the deep result back onto the seed-final hidden base (evicted/frozen positions keep
  the seed hidden); norm; return logits.
- Frozen positions are absent from `token_indices` (handled in `_focus_select`), so they are neither
  recomputed nor rewritten; their buffer KV persists.

### 4b. Per-block buffer
A minimal holder: per-deep-layer `(k, v)` tensors `[B, num_kv_heads, block_size, head_dim]`, a
`reset()` that zeros/forgets, and `write(layer_idx, k, v, positions)` (scatter) + `get(layer_idx)`.
May reuse `BlockSparseCache` if its API fits; otherwise a small new `DynamicBlockKV` class in
`utils/`. Reset once per block.

### 4c. `batch_sample_focus_dynamic` (generation_functions.py)
Modeled on `batch_sample` (DynamicCache structure: prefill, block loop, commit via
`update_past_key_values=True`), with the FOCUS diffusion step swapped in:
- Uses `past_key_values = DynamicCache()` (not StaticKVCache); no `set_kv_write_start`/
  `prepare_write_idx`/`scratch_seqlens` setup.
- Per block: allocate the per-block buffer (reset); one **dense seed** step; then FOCUS steps calling
  `forward_focus_compact_dynamic`; per-block `frozen` mask allocated/updated with the SAME rule as the
  static path (`frozen |= (~mask_idx & shift_right(~mask_idx))`, pre-forward mask), gated on
  `_delayed_cache`.
- **Commit** (block complete): a plain dense `self.forward(input_ids=x_t[:, -block_size:],
  past_key_values=past_key_values, update_past_key_values=True)` that appends the finished block's
  full-depth KV to the DynamicCache. (Same approach as the static path's commit; the per-block buffer
  is diffusion-only and not the commit source.)
- Finished sequences handled by padding (like `batch_sample`); no batch compaction required.

### 4d. Dispatch (eval.py)
`FAST_DLLM_FOCUS_DYNAMIC=1` (alongside `FAST_DLLM_USE_FOCUS=1`) routes to `batch_sample_focus_dynamic`.
`FAST_DLLM_FOCUS_DELAYED_CACHE` and `FAST_DLLM_FOCUS_COMPACT` apply as on the static path.

## 5. Memory & throughput expectations (state honestly)
- **Memory:** prefix grows to real length, not `max_seq_len` → no eager pre-allocation → b48/b64 that
  OOM on the static path are expected to fit. THIS is the deliverable.
- **Throughput:** eager SDPA recomputes attention over the growing prefix each step, so at LOW batch
  the dynamic variant may be SLOWER per step than static+flash FOCUS. That is acceptable — low batch is
  not the target; high batch (where static cannot run at all) is. Report both so the crossover is visible.

## 6. Env knobs
- `FAST_DLLM_FOCUS_DYNAMIC` (default `0`) — route FOCUS to the DynamicCache path.
- Reused: `FAST_DLLM_FOCUS_DELAYED_CACHE`, `FAST_DLLM_FOCUS_COMPACT` (the dynamic variant is compact-only),
  `FAST_DLLM_FOCUS_ALPHA`, `FAST_DLLM_FOCUS_LAYERS`, `FAST_DLLM_FOCUS_RETAIN`, `FAST_DLLM_FOCUS_FLOPS`.

## 7. Validation
1. **Retain-all sanity (numeric):** `forward_focus_compact_dynamic` with `retain_override=1.0` and no
   frozen ⇒ logits match a plain dense forward over the same DynamicCache+block state (max_abs_diff≈0,
   modulo SDPA-vs-flash numerics — use a tolerance, not bit-exact, since the static gate used flash).
2. **Cross-cache accuracy parity:** dynamic FOCUS (delayed on) at b16/1000 ≈ static FOCUS (delayed on)
   b16 (0.825) within noise; confirms the port is faithful.
3. **Headline — break the ceiling:** run b48 and b64 with delayed cache (previously OOM on static).
   Success = runs without OOM at a safe `max_seq_len` (≥ ~1024, since some CoT reach ~800 tokens),
   reporting accuracy AND TPS AND peak GPU memory (`nvidia-smi`/`torch.cuda.max_memory_allocated`).
4. **FLOP saving carries:** `FAST_DLLM_FOCUS_FLOPS=1` shows the same ~69% saving as the static delayed path.
5. **No-regression:** `FAST_DLLM_FOCUS_DYNAMIC=0` leaves the static FOCUS path and all existing tests
   bit-identical (this feature adds new methods; it must not touch the static path).

## 8. Edge cases
- Dense seed each block writes the full per-block buffer; FOCUS steps update only selected.
- Block boundary: per-block buffer + `frozen` reset; `DynamicCache` retains committed prefix.
- Rightmost block position never freezes (right-neighbor rule).
- Finished sequences: padded (no compaction); `frozen`/buffer rows for finished slots are inert.
- All-frozen degenerate step: `Ksel` clamped ≥ 1 (already handled in `_focus_select`).
- `max_seq_len` for the dynamic path only bounds the DynamicCache's eventual growth headroom; keep the
  existing `required_len` computation so commits never exceed it.

## 9. Success criteria
- b48 (and ideally b64) run WITHOUT OOM with delayed cache at `max_seq_len ≥ 1024`.
- Accuracy parity with static FOCUS at b16 (within noise).
- Both accuracy AND TPS AND peak memory reported across b16/b32/b48/(b64).
- FLOP saving ≈ 69% (carried from the static delayed path).
- `FAST_DLLM_FOCUS_DYNAMIC=0` path bit-identical to current behavior; static FOCUS untouched.

## 10. Cross-references
- Static compact FOCUS + delayed cache (the thing being ported): `modeling.py` `forward_focus_compact`,
  `_focus_select`, `_focus_update_frozen`; `generation_functions.py` `batch_sample_focus`.
- DynamicCache reference path: `generation_functions.py` `batch_sample`; `modeling.py` attention
  KV-update branch (`modeling.py:357-377`, esp. the `update_past_key_values=False` cat at 375-377).
- Per-block buffer reference: `utils/block_sparse_cache.py` (`BlockSparseCache`), used by `forward_focus`.
- Results context: `logs/focus_batch_scaling.md` (the OOM ceiling), `logs/focus_delayed_cache.md` (the
  win this unlocks at high batch), `docs/FOCUS_SESSION_TURNOVER.md`.
- Delayed-cache design/plan: `docs/superpowers/specs/2026-06-01-focus-delayed-kv-cache-design.md`,
  `docs/superpowers/plans/2026-06-01-focus-delayed-kv-cache.md`.

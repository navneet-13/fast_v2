# FOCUS Delayed KV Cache — Design

**Date:** 2026-06-01
**Status:** Approved (design), pending spec review → plan
**Goal:** Implement the FOCUS paper's *delayed KV caching* inside our Fast-dLLM-v2 FOCUS port, so
settled (decoded) tokens stop being recomputed every diffusion step — breaking the "decoded-token
floor" that currently pins FOCUS throughput at ~185 TPS.

---

## 1. Background & motivation

Our FOCUS port (`forward_focus_compact` + `batch_sample_focus`) implements the *token-eviction* half
of the paper but **not** the *delayed-cache* half. In the reference the two are coupled
(`sequence.py:124`: `_focus_enabled = focus_enabled and delayed_cache_enabled`). Today every decoded
token is retained and **recomputed on every step** (`retain = ~mask_idx | selected`), so:

- `Ksel` (the uniform `[B, Ksel, D]` deep-layer compute width) is floored by the decoded-token count,
  which grows toward the full block as it fills in → savings vanish late in each block.
- Empirically: FOCUS ≈ 185 TPS regardless of batch (16→32), and the `alpha` knob is inert because the
  retained set is pinned by the decoded floor, not the top-K term. (See `logs/focus_batch_scaling.md`.)

**Delayed caching** caches a decoded token's KV once it is stable and removes it from the recompute
set. The KV-persistence substrate already exists: `StaticKVCache.write_sparse` writes **only** the
selected positions into the block scratch region `[past_len, past_len+block_size)`; unselected
positions retain their last-written KV. So the change is in **which tokens we select**, not in the KV
plumbing.

## 2. Scope

- **In scope:** `forward_focus_compact` (compact path) + `batch_sample_focus`, **eager only**,
  env-gated, default off.
- **Out of scope:** `forward_focus` (non-compact), composition with the v2 block cache
  (`use_block_cache`), compiled/CUDA-graph path, the `rightmost_processed` left-side guard (documented
  fallback only — see §6).

## 3. Freeze rule (paper-faithful, right-neighbor)

Per-block state `frozen: [B, block_size]` bool, `True` = "settled, skip recompute, keep cached KV".

At each FOCUS diffusion step, using `dec_pre` = decoded mask **entering** this step's forward
(`dec_pre = ~mask_idx`, before this step's unmask):

```
ready        = dec_pre & shift_right(dec_pre)      # token decoded AND right neighbor decoded
newly_frozen = ready & processed_this_step          # only freeze tokens just recomputed with real id
frozen       = frozen | newly_frozen
```

- `shift_right(x)[:, i] = x[:, i+1]`; the rightmost block position has no right neighbor → never
  freezes (conservative, correct).
- **Refresh-timing invariant (critical):** a token is frozen only on a step where it was in the
  processing set (`processed_this_step`) AND was already decoded *before* the step (`dec_pre`). This
  guarantees its cached KV was computed from its **real token id**, never from the mask embedding.
  (A token decoded only at this step's unmask has `dec_pre=False` here, so it is not frozen until the
  next step, where it gets one real-id reprocess first.)

## 4. Selection change

In `forward_focus_compact`, frozen positions leave the processing set entirely:

```
processable      = ~frozen
retain_decoded   = (~mask_idx) & processable          # decoded-but-unsettled: force-retain (need refresh)
masked_selected  = focus_select(...) & processable    # FOCUS eviction among still-masked, non-frozen
retain           = retain_decoded | masked_selected
token_indices    = sorted positions where retain      # gathered → deep layers → write_sparse
```

- Frozen tokens are **not gathered, not written**: their KV persists in scratch; their scatter-hidden
  stays the seed value (their logits are ignored — they are decoded, never re-sampled).
- `Ksel` stays the batch-max of `retain.sum(dim=1)` — but the set now shrinks as tokens settle, so
  `Ksel` falls through the block → the throughput win.

## 5. State plumbing (`batch_sample_focus`)

- Allocate `frozen = zeros([B, block_size], bool)` at each **block start** (reset to all-False).
- Pass `frozen` into `forward_focus_compact` each step (new kwarg, default `None` = current behavior).
- After each step's unmask, update `frozen` per §3 using `dec_pre` (snapshot taken **before** the
  step's unmask) and the `processed_this_step` set returned/derivable from the selection.
- On **batch compaction** (`active_indices`), re-index `frozen` alongside the other per-sequence
  tensors so rows stay aligned with the compacted KV cache.

## 6. Known approximation & fallback (left side)

The paper's right-neighbor-only freeze is left-safe only because (a) decoding is roughly left-to-right
and (b) the `rightmost_processed` guard (`focus.py:243-247`) force-retains unprocessed tokens left of
the rightmost processed position. **We omit the guard initially (YAGNI).** If validation (§7) shows a
frozen token's KV diverging because a left neighbor was still masked, add the guard:
force-`processable=False` for any masked position left of the rightmost retained position. Tracked as a
fallback, not built up-front.

## 7. Validation

1. **Numerical (correctness of the refresh-timing invariant):** a debug mode
   (`FAST_DLLM_FOCUS_DELAYED_VERIFY=1`) that freezes-but-still-recomputes and compares each frozen
   token's freshly-recomputed deep-layer KV / final argmax against the cached one. Expectation: exact
   match on the *frozen step* (proves the real-id refresh timing), small bounded drift on later steps
   (the bidirectional approximation). Run on ~20 GSM8K samples, batch 1.
2. **End-to-end:** GSM8K flexible-extract acc + TPS at b16 and b32 (1000 samples, max_seq_len=1024,
   eager) vs current FOCUS (0.833 / 185.6 @ b16). **Bar:** accuracy within noise of 0.83; TPS strictly
   above the ~185 plateau with rising FLOP saving (`FAST_DLLM_FOCUS_FLOPS=1`).
3. **No-regression:** with `FAST_DLLM_FOCUS_DELAYED_CACHE=0` (default), the existing retain-all
   equivalence gate (`tests/test_focus.py::test_focus_compact_retain_all_matches_dense`,
   max_abs_diff=0) stays green and token trajectories are unchanged.

## 8. Env knobs

- `FAST_DLLM_FOCUS_DELAYED_CACHE` (default `0`) — enable delayed KV caching.
- `FAST_DLLM_FOCUS_DELAYED_VERIFY` (default `0`) — numerical verify mode (§7.1).

## 9. Edge cases

- **Dense seed step** (`is_dense_step=True`): unaffected; `frozen` all-False; writes all block KV.
- **First FOCUS step** of a block: nothing frozen yet (all-False).
- **Block boundary:** `frozen` reset to all-False (new block, fresh scratch).
- **Rightmost position:** no right neighbor → never frozen.
- **Batch compaction mid-block:** `frozen` re-indexed with `active_indices`.
- **All tokens frozen** (block fully settled before commit): processing set empty → falls back to a
  no-op step / commit path; must not produce an empty-gather crash (clamp `Ksel ≥ 1`).

## 10. Success criteria

- Accuracy ≈ 0.83 preserved (within noise) on GSM8K b16 & b32.
- TPS strictly above the current FOCUS ~185 plateau, with `FAST_DLLM_FOCUS_FLOPS` showing higher
  token-layer saving than current FOCUS.
- Default-off path bit-identical to current FOCUS (no-regression gate green).

## 11. Cross-references

- Current FOCUS: `models/.../modeling.py` `forward_focus_compact` (~1090-1256), `_focus_select`
  (~235-283); `generation_functions.py` `batch_sample_focus`.
- KV substrate: `utils/static_kv_cache.py` `StaticKVCache.write_sparse` / `get_full_kv` / scratch.
- Reference: `FOCUS/lmdeploy/pytorch/strategies/dllm/sequence.py` (`DelayedCacheState`,
  `FocusState`), `FOCUS/lmdeploy/pytorch/kernels/cuda/focus.py` (`rightmost_processed` guard).
- Findings: `logs/focus_batch_scaling.md`, `docs/FOCUS_SESSION_TURNOVER.md`.

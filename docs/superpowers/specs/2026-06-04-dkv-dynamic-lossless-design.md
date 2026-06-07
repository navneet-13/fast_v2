# Near-lossless dynamic dKV-Cache — Design

**Status:** approved design, pre-implementation.
**Date:** 2026-06-04.
**Supersedes:** the hand-rolled dynamic dKV full step (variant A, `forward_dkv` / `_full_layer`).

## 1. Goal & motivation

The dynamic dKV path (variant A) is lossy: `refresh=1` (full recompute every step, no
caching) = **0.771** vs dense **0.826**. The static path (variant B) `refresh=1` = **0.826**
exactly, because its full step delegates to the model's real `self_attn`. So the dynamic gap
is **not** the DynamicCache substrate and **not** the delayed caching — it is purely that
variant A's `forward_dkv` **hand-rolls attention** (its own `q/k/v`/rotary/`flash_attn_func`)
for every layer instead of calling the real attention module.

**Diagnostic (tests/diag_dkv_fullstep.py):** `forward_dkv(is_full_step=True)` vs the real
`Fast_dLLM_QwenModel.forward` on identical input → argmax agreement **0.9688** (≈1 token in 32
flips per forward), max hidden diff **8.0**, max logit diff **0.94**. Real, compounding
divergence — not bf16 epsilon.

**Goal:** rebuild the dynamic dKV forward to **mirror the original `forward` and call the real
`decoder_layer`/`self_attn`** (lossless), with delayed caching layered on top via a per-block
KV buffer. Expected: dynamic `refresh=1` → ~0.826, the whole refresh curve lifts, while keeping
DynamicCache's speed and no-OOM. Default off; dense and all existing paths untouched.

## 2. Key insight (why this is lossless)

The real attention (`Fast_dLLM_QwenAttention.forward`, modeling.py ~327) already does what
variant A hand-rolled: with `update_past_key_values=False` and a prefix cache it computes
`cat(past_key_value[layer], current_keys)` (line 381) and attends via the model's own SDPA
path (line 388–401). So running the **fed tokens through the real `decoder_layer`, attending
over `cat(prefix, block_buffer)`, is the lossless computation** — no reconstruction needed.

## 3. Architecture

Per active block (`block_size`=32), reusing the original `forward`'s structure:

1. **Per-block KV buffer** — `block_size` slots per layer holding the active block's post-RoPE
   K/V (`[B, H_kv, block_size, D]`). Created at block start, **reset at block boundary**,
   scatter-updatable by token position, persists across the block's diffusion steps. (A simple
   per-block store — NOT the `block_past_key_values`/`use_block_cache` model mechanism.)

2. **Minimal `self_attn` addition** — an optional `dkv_store` param (the buffer) + `dkv_positions`
   (where to write). When `dkv_store is not None`, after RoPE: scatter-write the current
   `key_states/value_states` into the buffer at `dkv_positions`, set `key_states/value_states` =
   the **full** buffer, then fall through to the existing `cat(prefix, …)` path (line 375–382) so
   attention runs over `cat(prefix, full_block_buffer)`. Default `None` ⇒ the dense path is
   byte-for-byte unchanged. ≈4 added lines, guarded.

3. **`forward_dkv` (rewritten)** — a near-copy of `Fast_dLLM_QwenModel.forward` (embed →
   cache_position → mask → rotary → loop over the **real** `decoder_layer(...)` → norm), threading
   `dkv_store` and the fed/all positions through to `self_attn`.
   - **Full/refresh step** (`is_full_step=True`): input = whole block; writes all buffer slots;
     attention is the real module → **lossless** (≡ dense).
   - **Cached step** (`is_full_step=False`): input = fed subset (gathered); scatter-updates only
     the fed buffer slots; attends over `cat(prefix, buffer)`; scatter outputs back onto the
     seed-step hidden for the non-fed positions (as variant A does).

4. **`batch_sample_dkv` (rewritten)** — `batch_sample` clone; per block: create the buffer, run
   the existing one-step-delay shift + `refresh_steps` bookkeeping, dispatch full vs cached,
   **reset the buffer at the block boundary**. Commit a finished block via the existing plain
   `self.forward(update_past_key_values=True)`.

The hand-rolled `_full_layer`/`_attn` of variant A is removed from the dynamic path.

## 4. Data flow (one cached step)

`fed_indices = _dkv_fed_indices(prv_transfer_idx)` → gather fed hidden + positions →
`forward_dkv(input=block, fed_indices, is_full_step=False, dkv_store=buffer)` → per layer the
real `self_attn` scatter-writes fed K/V into `buffer`, attends `cat(prefix, buffer)` → outputs
scattered onto the seed hidden → norm → `lm_head` → decode (unchanged) → one-step-delay shift.

## 5. Components & files

| File | Change |
|---|---|
| `models/.../modeling.py` `Fast_dLLM_QwenAttention.forward` | Add optional `dkv_store`/`dkv_positions`; ~4 guarded lines (scatter-write + use full buffer). Default None → no behavior change. |
| `models/.../modeling.py` `Fast_dLLM_QwenModel` | Rewrite `forward_dkv` as a near-copy of `forward` using the real `decoder_layer` loop + the buffer. Remove `_full_layer`/`_attn` reconstruction. |
| `models/.../modeling.py` `Fast_dLLM_QwenForCausalLM` | `forward_dkv` LM wrapper unchanged in shape (already returns `CausalLMOutputWithPastAndBlockCache`). |
| `generation_functions.py` `batch_sample_dkv` | Rewrite to thread the per-block buffer, reset at block boundary; keep the existing shift/refresh bookkeeping. |
| `utils/dynamic_block_kv.py` | Reuse `DynamicBlockKV(deep_layer_start=0)` as the buffer, or a thin equivalent that the new `self_attn` writes to (BHSD scatter). |
| `eval.py` | No change — same `FAST_DLLM_DKV_CACHE=1` (dynamic) dispatch. |

The hand-rolled variant A method is retained (renamed `forward_dkv_handrolled`, not dispatched)
so the 0.771 baseline stays reproducible for the A-vs-lossless comparison.

## 6. Testing

1. **Diagnostic regression (GPU):** rerun `tests/diag_dkv_fullstep.py` against the rewritten
   `forward_dkv(is_full_step=True)` — argmax agreement must be **1.0** (or ≤1 flip from bf16) and
   hidden diff ≪ 8.0, proving the full step now equals the real forward.
2. **No-op invariant (GPU):** the existing `test_forward_dkv_full_equals_fed_all` must still pass
   (full == fed-all).
3. **`refresh=1` == dense (e2e, b16/1000):** the anchor must land **≈0.826** (was 0.771). This is
   the headline success criterion.
4. **Refresh sweep (e2e, b16/1000):** {1,2,4,8}; record accuracy + TPS in `logs/dkv_cache.md`
   next to the old dynamic + static tables. Expect near-lossless at low refresh, TPS still well
   above dense (caching preserved).
5. **No-regression:** `FAST_DLLM_DKV_CACHE=0` and the static dKV path unchanged; dense `self_attn`
   path byte-for-byte unchanged (the `dkv_store=None` default).

## 7. Success criteria

- Dynamic `refresh=1` ≈ 0.826 (recovered from 0.771); diagnostic argmax agreement = 1.0.
- Refresh 2–4 near-lossless (≤~1pt from dense), TPS still > dense (caching intact).
- Dense and static paths untouched (guarded `dkv_store=None`).
- `forward_dkv` is a thin layer over the real `forward`; no hand-rolled attention remains in the
  dispatched dynamic path.

## 8. Risks

- **Position/RoPE bookkeeping for the fed subset** — the cached step must pass each fed token's
  true position so RoPE + attention are correct; mirror variant A's `sel_cos/sel_sin` gather.
- **Buffer layout vs `self_attn`** — `self_attn` works in `[B, H, S, D]` (post-transpose, post-RoPE);
  the buffer and scatter must match that layout exactly.
- **The `self_attn` touch** — must stay guarded so the dense path is provably unchanged (test 5).

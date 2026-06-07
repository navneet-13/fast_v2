# dKV dynamic-lossless investigation — session rollover

Scope: making the **dynamic** dKV-Cache path (`FAST_DLLM_DKV_CACHE=1`, no `FAST_DLLM_DKV_STATIC`)
match the dense baseline accuracy while keeping DynamicCache speed / no-OOM. Read this first to
continue without redoing work. Companion data: `logs/dkv_cache.md`. Background on dKV variants
A/B: `docs/FOCUS_SESSION_TURNOVER.md` + the dKV specs/plans under `docs/superpowers/`.

## 0. CURRENT STATE — what we are looking at right now

**UPDATE (this session): the baseline-first rebuild is DONE and the validation gate PASSED.**
`batch_sample_dkv` was rewritten as the dense `batch_sample` with ONE substitution (inside the
`use_block_cache=False` branch: `self.forward(... update_past_key_values=False)` →
`self.forward_dkv(..., dkv_store=block_kv, is_full_step=...)`), a per-block `DynamicBlockKV`
allocated per block + reset at block boundaries, and one-step-delay/refresh bookkeeping layered on
that branch only. `tests/diff_dkv_vs_baseline.py` (refresh=1, SBS=8) now reports **match=True on all
3 prompts (byte-identical to baseline)**. This confirms the root cause was the dropped sub-block
schedule, and that the dKV substrate is bit-exact when every step is a full recompute.

**UPDATE 2 — cached-step mask FIXED and the accuracy+TPS sweep is DONE.** `forward_dkv`'s cached
step now reuses the SAME `eval_mask` (gathered fed rows) instead of `attention_mask=None`; validated
by `tests/diag_dkv_cached_mask.py` (cached==full argmax 1.0 at fed positions, incl. a grid-straddling
prefix). b16/1000 sweep (table in §1): refresh=1 == dense (0.821, byte-identical), refresh=2 is the
sweet spot (0.826, +11% TPS), r4/r8 trade accuracy for little extra speed. refresh=1 still has no
speedup (full recompute every step) — it is the lossless correctness anchor, not the win.

CURRENT STATUS: the dynamic dKV path now matches dense at r1 and is lossless+faster at r2. The
remaining open question is whether the ~11% TPS ceiling can be raised (larger blocks / non-eager /
cheaper cached step) — see §7.

**Root cause of the dynamic-vs-static dKV accuracy gap is identified (structurally confirmed):
the DECODE SCHEDULE differs.**
- `batch_sample_dkv` (dynamic) decodes the **whole `block_size` block at once** (confidence-order
  unmask). NO sub-block loop.
- `batch_sample_dkv_static` AND the dense `batch_sample` decode in **sub-blocks of
  `small_block_size=8`, left-to-right (semi-autoregressive)**: per-block they loop
  `for small_block_idx in range(num_small_blocks)` and only unmask within the current
  `start:end` window.

**PENDING FIX (next action) — BASELINE-FIRST, do NOT port from static.** The reference is the
dense `batch_sample`, NOT `batch_sample_dkv_static`. Method: start from an exact copy of
`batch_sample`'s loop (commit step at L118, the sub-block `for small_block_idx` loop, the
`start:end` windows, all bookkeeping) and change ONE thing — inside the `use_block_cache=False`
branch (L158-164), swap `self.forward(... update_past_key_values=False)` →
`self.forward_dkv(..., dkv_store=block_kv, ...)`, allocating `block_kv` per block and resetting it
at block boundaries. The dKV refresh/one-step-delay bookkeeping is layered on top of that branch
only. Do NOT borrow static-buffer machinery (`write_sparse`/`scratch_seqlens`/`kv_write_start`/
flash_kvcache) — the dynamic path does not need it. Principle: minimal deviation from baseline;
never change anything in the baseline flow that our approach does not require.
After the fix, re-run the token-diff (below) — at `refresh=1` dynamic dKV should become
byte-identical to `batch_sample`. Only after byte-identical at refresh=1, run the b16 sweep.

The current `batch_sample_dkv` deviated by DELETING the sub-block loop (one whole-block forward,
whole-block unmask) — that deletion IS the accuracy gap. `batch_sample_dkv_static` is only a
secondary cross-check (it also has the sub-block loop and scores 0.826); the static-buffer
specifics are not a template for the dynamic path.

`small_block_size=8` is the intended schedule **even with `use_block_cache=False`** (confirmed by
user). Always keep `use_block_cache=False` for dKV.

## 1. VERIFIED NUMBERS (GSM8K flexible-extract, threshold=0.9, max_seq_len=1024, eager)

| config | b16/1000 r1 | r2 | r4 | r8 | notes |
|---|---|---|---|---|---|
| dense `batch_sample` (DynamicCache, eval_mask, sub-blocks) | ~0.821 | — | — | — | b1/200 = 0.83 |
| static dKV (sub-block decode) | 0.826 | 0.817 | 0.815 | 0.733 | near-lossless |
| dynamic dKV, hand-rolled forward (variant A) | 0.771 | 0.776 | 0.760 | 0.745 | |
| dynamic dKV, lossless forward, mask=None (no eval_mask) | 0.782 | 0.774 | 0.763 | 0.737 | clean rerun |
| dynamic dKV, lossless forward + eval_mask fix | — | — | — | — | b1/200 = 0.795 (was 0.77) |

CI at 1000 samples ≈ ±0.013; at 200 samples ≈ ±0.029.

### Baseline-first rebuild + cached-step eval_mask fix — b16/1000 sweep (accuracy AND TPS)

Same machine, all 5 runs one-per-GPU. Dense = default `batch_sample`. dKV = new `batch_sample_dkv`.

| config | accuracy (flex-extract) | TPS | vs dense |
|---|---|---|---|
| dense baseline | 0.821 ±0.012 | 157.4 | 1.00× |
| dKV refresh=1 | 0.821 ±0.012 | 156.0 | 0.99× (lossless, no speedup) |
| dKV refresh=2 | 0.826 ±0.012 | 175.5 | 1.11× (lossless + speedup) |
| dKV refresh=4 | 0.808 ±0.013 | 179.2 | 1.14× (~CI-edge accuracy drop) |
| dKV refresh=8 | 0.758 ±0.014 | 182.9 | 1.16× (clear accuracy drop) |

Observations (no extrapolation):
- refresh=1 accuracy == dense to 3 d.p. (0.821) — confirms byte-identical at 1000 samples; TPS ~equal
  (full recompute every step + buffer overhead), so r1 is the lossless correctness anchor, not a win.
- refresh=2 is the sweet spot: accuracy 0.826 (within CI of dense) with +11% TPS.
- TPS gain saturates quickly (r2→r8: 175→183, +4%) while accuracy falls (0.826→0.758). The cheap
  recompute-skip is small here (block_size=32, sub_block=8, eager), so returns past r2 are poor.
- The cached-step eval_mask fix is what made r2/r4 viable: the prior whole-block dynamic dKV was
  ~0.78 even at r1; now r1=0.821 and r2=0.826.

## 2. VERIFIED DIAGNOSTICS (unit-level, reusable scripts under tests/)

- `tests/diag_dkv_fullstep.py` — `forward_dkv(is_full_step=True)` vs the real decoder_layer loop
  with `attention_mask=None`, **no prefix**: argmax agreement **1.0000**, hidden/logit diff
  **0.0**. (Its earlier "0.9688" run compared against the `eval_mask` forward — that was a
  confounded reference; do not use eval_mask as the reference here.)
- `tests/diag_dkv_prefix.py` — `forward_dkv(is_full_step=True)` vs `self.forward` over an
  **identical committed prefix**: argmax agreement **1.0000**, diff **0.0**.
  ⇒ **The forward is NOT the bug**, with or without a prefix.
- `tests/diff_dkv_vs_baseline.py` — token-level diff, `batch_sample` vs `batch_sample_dkv`
  (refresh=1, greedy/temp=0). With the eval_mask fix and `small_block_size=8`: **block 1 matches
  exactly; divergence starts in block 2+** (first differing generated token at index 42/46/95 on
  3 prompts). ⇒ bug is in the **loop/decode schedule**, not the forward. (Set `SBS=8` in this
  script.)
- `tests/test_dkv_lossless.py::test_self_attn_dkv_store_uses_full_buffer` — GPU, PASSES.

## 3. CODE CHANGES ALREADY MADE THIS SESSION (working tree, UNCOMMITTED)

`models/.../modeling.py`:
- `Fast_dLLM_QwenAttention.forward` gained optional `dkv_store=None, dkv_positions=None`
  (~4 guarded lines after RoPE: scatter post-RoPE K/V into the buffer, then use the full buffer;
  dense path unchanged when None). Reviewed.
- `Fast_dLLM_QwenModel.forward_dkv` REWRITTEN to use the real `decoder_layer` loop + `dkv_store`
  (replaces the old hand-rolled version). Full step now uses
  `attention_mask = self.eval_mask(block, block, past_len)` (was `None`). Cached step still uses
  `attention_mask=None` — **the cached-step mask is not yet fixed** (only matters for refresh>1).
- Old hand-rolled forward retained as `forward_dkv_handrolled` (model + LM wrapper), not dispatched.
- LM wrapper `forward_dkv` added (returns `CausalLMOutputWithPastAndBlockCache`).

`generation_functions.py`:
- `batch_sample_dkv` updated to call the new `forward_dkv(dkv_store=block_kv, ...)`. **Still
  whole-block decode — this is the file to change for the pending fix.**
- `batch_sample_dkv_static` (the blueprint) is unchanged and has the sub-block loop at
  ~lines 1847–1903.

`logs/dkv_cache.md`: updated — the earlier "RESOLVED: hand-rolling was the cause" section is
marked REOPENED/refuted; a "Lossless-attempt (HYPOTHESIS REFUTED)" section records that the
lossless forward did not change accuracy.

## 4. REFUTED / RULED OUT — do NOT re-investigate these as the cause

- **"Hand-rolled attention causes the dynamic gap"** — REFUTED. The lossless `forward_dkv`
  (real `self_attn`) gave the same accuracy as the hand-rolled version (0.782 vs 0.771, within
  noise). The forward is bit-identical to the real layers (§2).
- **"Flash vs SDPA kernel"** — ruled out earlier (swapping kernels did not move accuracy).
- **"Static buffer / StaticKVCache is inherently more accurate"** — FALSE. Dense `batch_sample`
  uses DynamicCache and scores 0.821. The static-vs-dynamic dKV gap is the decode schedule (§0).
- **The original `forward_dkv` "0.9688 vs eval_mask" diagnostic** was a confounded reference
  (None-vs-eval_mask), not evidence about hand-rolling.

## 5. TWO REAL BUGS FOUND (one fixed, one pending)

1. **`forward_dkv` full step used `attention_mask=None` instead of `eval_mask`.** The dense path
   uses `eval_mask = (block_q >= block_kv)` on a fixed `block_size` grid from position 0. When a
   generated block straddles a grid boundary (prompt length not a multiple of `block_size`),
   `eval_mask` masks some intra-block attention that the model was trained with; `None` does not.
   FIXED for the full step. Observed effect: b1/200 0.77 → 0.795 (within-noise at 200 samples,
   not yet confirmed at 1000).
2. **Decode schedule: whole-block (dynamic) vs sub-block `small_block_size=8` (static/baseline).**
   FIXED this session via the baseline-first rebuild (§0). `tests/diff_dkv_vs_baseline.py`
   (refresh=1, SBS=8) is now byte-identical on all 3 prompts.
3. **Cached-step mask (refresh>1 only): `forward_dkv` non-full step used `attention_mask=None`.**
   FIXED — now `full_mask = self.eval_mask(block, block, past_len); cached_mask =
   full_mask[fed_indices].unsqueeze(1)`, so the cached step is mask-identical to the full step on
   the recomputed tokens. Validated by `tests/diag_dkv_cached_mask.py` (argmax 1.0, incl. straddle).

## 6. HOW TO RUN (templates)

```bash
cd /research/data/transfer/data/n41/fast_v2 ; export FV2=$PWD WORKSPACE=$PWD
# after ANY modeling.py edit:
rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* *Efficient*
# token diff (fast, decisive, no noise):
CUDA_VISIBLE_DEVICES=<free> v2/bin/python tests/diff_dkv_vs_baseline.py
# dynamic dKV b16 eval (refresh=1 anchor):
export FAST_DLLM_EXECUTION_MODE=eager FAST_DLLM_MAX_SEQ_LEN=1024 FAST_DLLM_DKV_CACHE=1 FAST_DLLM_DKV_REFRESH_STEPS=1
mp=$WORKSPACE/models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/
CUDA_VISIBLE_DEVICES=<free> v2/bin/accelerate launch eval.py --tasks gsm8k --batch_size 16 \
  --num_fewshot 0 --limit 1000 --confirm_run_unsafe_code --model fast_dllm_v2 \
  --fewshot_as_multiturn --apply_chat_template \
  --model_args "model_path=${mp},threshold=0.9,show_speed=True,use_block_cache=False" &> logs/<name>.log
```
Operational notes (observed this session): launching runs with `&` inside one backgrounded shell
orphans/duplicates them (idle GPUs during model-load looked like "dead", led to contaminated
numbers); launch each eval as its OWN background task, one per GPU. Accuracy is deterministic
(greedy, temp=0), so GPU contention corrupts TPS, not accuracy. `pkill -f eval.py` to clear strays.
Nothing committed (no-git-commits rule).

## 7. NEXT STEPS (in order)
1. ~~Rebuild `batch_sample_dkv` baseline-first (one forward→forward_dkv substitution).~~ DONE.
2. ~~Run `tests/diff_dkv_vs_baseline.py` (SBS=8) — byte-identical at refresh=1.~~ DONE (match=True ×3).
3. ~~Fix the cached-step mask in `forward_dkv` (fed-subset `eval_mask`).~~ DONE (diag argmax 1.0).
4. ~~b16/1000 refresh sweep {1,2,4,8}, accuracy AND TPS.~~ DONE (§1 table; r2 = lossless +11% TPS).
5. **OPEN (optional):** the TPS win is only ~11% and saturates by r2 (block_size=32/sub_block=8/
   eager). If more speedup is wanted, investigate: (a) non-eager/compiled path, (b) larger block_size,
   (c) a cheaper cached step (the cached forward still runs all 28 layers over num_fed tokens).
   These are perf explorations, not correctness — the accuracy story (r1 lossless, r2 lossless+fast)
   is settled.

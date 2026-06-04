# Three-way comparison — GSM8K, 1000 samples, batch_size=16

Model: Fast_dLLM_v2_7B. Env: `v2/` (flash-attn 2.8.3, transformers 4.53.1). GPU: 1× RTX A5000.
lm-eval `--tasks gsm8k --num_fewshot 0 --limit 1000 --batch_size 16 --apply_chat_template`,
threshold=0.9, block_size=32. Metric: flexible-extract exact_match. (strict-match=0 for all —
GSM8K `#### N` format artifact.)

| # | Run | Config | exact_match | stderr | s/batch(16) | rel. speed |
|---|-----|--------|-------------|--------|-------------|-----------|
| 1 | baseline | use_block_cache=False | 0.821 | ±0.0121 | 33.98 | 1.00× |
| 2 | baseline + block cache | use_block_cache=True, small_block_size=8 | 0.817 | ±0.0122 | 23.47 | **1.45×** |
| 3 | FOCUS | batch_sample_focus, alpha=1.0, layers=0,1 | **0.832** | ±0.0118 | 28.26 | **1.20×** |

## Findings
- **All three are statistically equal in accuracy** (0.817–0.832, overlapping ±0.012 CIs).
  FOCUS preserves accuracy (0.832, nominally highest); block cache is accuracy-lossless (0.817).
- **Speed (batch 16):** block cache fastest (1.45× baseline, lossless). FOCUS is **1.20× faster
  than the plain baseline** — the token-skipping FLOP savings now convert to throughput at batch
  16 (compute-bound), unlike batch_size=1 where FOCUS was slower (overhead-bound).
- This is still the eager, uncompiled path; FOCUS + block cache + torch.compile would compound.

## Batch-size investigation (FOCUS not degraded by batching)
A 2×2 control on the same first-32 problems showed FOCUS has NO batch-specific accuracy bug:

| | batch 1 | batch 16 |
|------|---------|----------|
| baseline | 0.75 (24/32) | 0.72 (23/32) |
| FOCUS    | 0.78 (25/32) | 0.72 (23/32) |

At batch 16 FOCUS == baseline exactly (0.72=0.72); b1↔b16 differences are 1–2 problems (n=32
noise). The low 0.72 vs the 1000-sample 0.83 is the small/hard first-32 subset, not batching.
FOCUS batch-16 path (cross-sequence uniform-budget selection + mid-decode batch compaction)
validated: clean output, no errors.

## Run history (for reference)
| samples | batch | baseline | block cache | FOCUS |
|---------|-------|----------|-------------|-------|
| 20  | 1  | 0.75 (block-cache cfg) | — | 0.80 |
| 200 | 1  | 0.830 | 0.830 | 0.860 |
| 1000| 16 | 0.821 | 0.817 | 0.832 |

Logs: logs/run1k_b16_baseline.log, logs/run1k_b16_blockcache_sbs8.log, logs/run1k_b16_focus.log;
diagnostics logs/diag_*_l32.log, logs/smoke_b16_focus.log.

## Update: compact FOCUS variant (gather-once/scatter-once, seed-final-hidden cache)
Added `forward_focus_compact` (env `FAST_DLLM_FOCUS_COMPACT=1`): select once after the focus
layers, gather selected tokens to [B,Ksel,D], run deep layers densely on them, scatter once;
evicted tokens reuse the dense-seed's full-depth hidden state (cached once per block). Drops the
per-layer block_sparse_cache round-trips.

| config (batch16, 1000) | exact_match | s/batch | tokens |
|------------------------|-------------|---------|--------|
| non-compact FOCUS      | 0.832       | 28.26   | 323336 |
| compact FOCUS          | 0.833       | 27.90   | 323393 |

Batch-1 (limit 50): both 0.82, both 17820 tokens (bit-equivalent trajectories), 5.34 vs 5.21 s/it.

Conclusion: compact is functionally EQUIVALENT to non-compact (same generations) and only ~1-2%
faster in eager mode — the per-layer gather/scatter is not the wall-clock bottleneck. Its value is
code simplicity + torch.compile-friendliness. Correctness: retain-all gate max_abs_diff=0; an earlier
v1 (evicted tokens got layer-1-only logits) produced garbage — fixed by caching the seed-final hidden.

## Update: static-KV (dynamo, eager) baseline ± block cache — and 2 block-cache bugs fixed
`batch_sample_dynamo` (env `FAST_DLLM_USE_DYNAMO=1`, `FAST_DLLM_EXECUTION_MODE=eager`) uses the
StaticKVCache. It already had `use_block_cache` wiring via `StaticBlockCache`, but the path was
**broken** — two latent bugs (the static path reuses persistent buffers where the dynamic
`batch_sample` recreates them each step):
1. **`replace_position=None` crash (any batch).** Full-block forwards don't pass `replace_position`;
   on the *first* full-block forward of a block the static block cache is empty (→ `update`, writes at
   pos 0), but on **re-entry** (later sub-block still masked) the cache is populated → in-place
   "replace" branch → `None + seq_len`. Fix: `utils/attention_backends.py` patched_forward — treat
   `replace_position is None` as 0 (full-block overwrites the whole block from pos 0).
2. **Batch-compaction size mismatch (batch>1).** When a sequence finishes, the eager path compacts
   the active batch and calls `static_cache.compact_batch()` but never compacted the StaticBlockCache
   → it stayed B=16 while key_states became B=15 → `copy_` size mismatch (16 vs 15). Fix: added
   `StaticBlockCache.compact_batch()` (re-index batch dim; contents are reset() per block anyway) and
   call it next to `static_cache.compact_batch()` in `generation_functions.py`.
Neither fix touches `modeling.py` (no HF dynamic-module cache clear needed).

Results (GSM8K flexible-extract, threshold=0.9, eager; TPS = total tokens / gen wall-clock):

| config | KV | batch | samples | exact_match | TPS | s/batch |
|--------|----|-------|---------|-------------|-----|---------|
| static-KV baseline (no block cache) | Static | 16 | 1000 | 0.826 | 168.9 | 29.9 |
| **static-KV + block cache (sbs=8)** | Static | 16 | 1000 | **0.830** | **258.6** | 19.5 |
| static-KV baseline (no block cache) | Static | 1  | 200  | 0.870 | 84.8 | — |
| **static-KV + block cache (sbs=8)** | Static | 1  | 200  | 0.860 | 87.6 | — |

Conclusion: **static buffer + block cache STACK** — 258.6 TPS at batch 16 is the fastest eager config
(1.14× the dynamic-cache block cache @226, 1.67× plain baseline @155), accuracy-lossless (0.830).
At batch 1 it's also fastest (87.6). Logs: logs/run1k_b16_static_eager_blockcache.log,
logs/run1k_b16_baseline_static_eager.log, logs/run200_b1_static_eager_blockcache.log,
logs/smoke_b16_static_eager_blockcache.log.

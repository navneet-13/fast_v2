# Three-way comparison — GSM8K, 200 samples

Model: Fast_dLLM_v2_7B. Env: `v2/` (flash-attn 2.8.3, transformers 4.53.1). GPU: RTX A5000.
lm-eval `--tasks gsm8k --num_fewshot 0 --limit 200 --apply_chat_template`, threshold=0.9,
block_size=32, batch_size=1. Metric: flexible-extract exact_match. (strict-match=0 for all —
GSM8K `#### N` format artifact; flexible-extract is the real metric.)

| # | Run | Config | exact_match | stderr | speed (s/it) |
|---|-----|--------|-------------|--------|--------------|
| 1 | baseline | `batch_sample`, use_block_cache=False | 0.830 | ±0.0266 | 3.95 |
| 2 | baseline + block cache | `batch_sample`, use_block_cache=True, small_block_size=8 | 0.830 | ±0.0266 | 3.83 |
| 3 | FOCUS | `batch_sample_focus`, alpha=1.0, focus_layers=0,1 | **0.860** | ±0.0246 | 4.77 |

## Findings
- **Block cache (run 2)** is accuracy-lossless vs plain baseline (0.830 = 0.830) and marginally
  faster (3.83 vs 3.95 s/it). As expected — it's a caching speedup, not an approximation.
- **FOCUS (run 3)** matches/slightly exceeds baseline accuracy (0.860 vs 0.830; the ±0.025–0.027
  stderrs overlap, so this is "preserves accuracy", with a small favorable nudge) while genuinely
  skipping deep-layer compute (avg_decoded ≈ 2.2 retained tokens/step).
- **Speed caveat:** FOCUS is slower in wall-clock here (4.77 s/it) despite doing fewer FLOPs,
  because this is the eager, uncompiled path — per-step Python/Triton overhead and the extra
  layer-0/1 importance pass dominate at batch_size=1. FOCUS's compute savings convert to
  throughput only with batching / torch.compile / CUDA-graphs (not done here). The accuracy
  result is the deliverable; throughput optimization is future work.

## Smoke validations (pre-run)
- Block-cache config (use_block_cache=True, sbs=8): limit-8 smoke → 0.75, clean, 4.43 s/it.
- FOCUS correctness gate: retain-all FOCUS step == dense logits (max_abs_diff=0).
- (`small_block_size` only affects the use_block_cache=True path, so runs 1 and 3 are unaffected by it.)

Logs: logs/run200_baseline.log, logs/run200_blockcache_sbs8.log, logs/run200_focus.log.

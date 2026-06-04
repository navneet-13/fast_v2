# FOCUS vs baseline — GSM8K smoke (20 problems)

Model: Fast_dLLM_v2_7B. Env: `v2/` conda (flash-attn 2.8.3, transformers 4.53.1).
GPU: RTX A5000. lm-eval `--tasks gsm8k --num_fewshot 0 --limit 20 --apply_chat_template`,
threshold=0.9, block_size=32, use_block_cache=True. Metric: flexible-extract exact_match.

| Method                          | exact_match | n  | approx speed | notes |
|---------------------------------|-------------|----|--------------|-------|
| `batch_sample` (baseline)       | 0.75        | 20 | ~4.6 s/it    | standard Fast-dLLM v2 dense decode |
| `batch_sample_focus` (pre-fix)  | 0.00        | 20 | ~42 s/it     | stale-KV bug (see below) |
| `batch_sample_focus` (fixed)    | **0.80**    | 20 | ~5.3 s/it    | alpha=1.0, focus_layers=0,1, avg_decoded≈2 |

(strict-match is 0.00 for all — GSM8K artifact: chat output doesn't use the `#### N`
strict format; flexible-extract is the meaningful metric.)

## Conclusion
The FOCUS-style attention-importance token skipping, ported into Fast-dLLM v2 as
`batch_sample_focus`, **preserves accuracy** on the GSM8K smoke set (0.80 vs 0.75
baseline — within the ±0.0918 stderr) while genuinely skipping deep-layer compute
on non-decodable tokens (`avg_decoded` ≈ 2 tokens/step retained for masked tokens).

## Root-cause note (the pre-fix 0.00)
`_focus_select` originally restricted the deep-layer recompute set to currently-MASKED
tokens. Unmasking happens after the forward, so a token's layers-2…N KV was computed
from the MASK embedding; once decoded it was permanently excluded from selection and
its deep KV was never refreshed with its real token id — corrupting attention context
for every later token in the block (coherent-but-locally-garbled output, e.g. doubled
adjacent words). Diagnostic: a control run with `FAST_DLLM_FOCUS_RETAIN=1.0` (recompute
all tokens every step) produced clean text at ~baseline accuracy, isolating the
skipping as the cause.

**Fix:** the recompute/retain set is now `(all decoded/non-masked positions) ∪
(FOCUS-selected decodable masked positions)`; only non-decodable masked tokens are
evicted — matching real FOCUS semantics (keep decoded + decodable, evict non-decodable).

## Reproduce
Baseline: (no FAST_DLLM_USE_* set) →
`CUDA_VISIBLE_DEVICES=N v2/bin/accelerate launch eval.py --tasks gsm8k --limit 20 ... --model_args "model_path=...,threshold=0.9,use_block_cache=True"`
FOCUS: add `FAST_DLLM_USE_FOCUS=1 FAST_DLLM_FOCUS_ALPHA=1.0 FAST_DLLM_FOCUS_LAYERS=0,1`.

Logs: `logs/focus_smoke_baseline.log`, `logs/focus_smoke_focus_fixed.log`,
diagnostics `logs/diag_focus_alpha1.log` (corrupt), `logs/diag_focus_retain1.log` (control).

## Next steps (not done)
- alpha sweep (e.g. 1.0 / 1.5 / 2.0) to trade savings vs accuracy.
- Full GSM8K (1319) run for a publishable number.
- Throughput study (the eager sparse path isn't compile/CUDA-graph optimized).

# Fast-dLLM v2 — FOCUS-style Token Skipping Design

**Date:** 2026-05-28
**Status:** Approved (design)
**Author:** Navneet + Claude

## Goal

Port the FOCUS token-skipping idea into Fast-dLLM v2 as a new decode method
`batch_sample_focus`, implemented **exactly like FOCUS**, then run a small GSM8K
accuracy comparison against the baseline `batch_sample`.

FOCUS (from the FOCUS repo at `/research/data/transfer/data/n41/FOCUS`) is a
training-free, attention-derived token-eviction mechanism for block-diffusion
DLLM decoding: it uses early-layer attention importance to predict which masked
tokens are "decodable" this step and evicts the rest from deep-layer compute.

## Background

### FOCUS mechanism (reference: SDAR/LMDeploy)
- Importance is computed over **masked** tokens only:
  `scores = Qₘ·Kₘᵀ·scale` → `max_pool1d(kernel=3, stride=1, padding=1)` →
  `softmax(dim=-1)` → `sum over query positions` → `sum over heads` =
  per-masked-token importance (total attention mass received).
- Measured at **layer 0** (stored as `first_layer_importance`) and **layer 1**
  (recomputed); `delta = importance₁ − importance₀`.
- Selection rule:
  - `retain_count K = clamp(ceil(avg_decoded_tokens · focus_alpha), 1, num_masked)`
  - `threshold = mean(delta) + std(delta)`
  - `candidates = delta ≥ threshold`
  - if `|candidates| < K`: greedy **top-K** by delta
  - **adjacency enforcement**: keep token i if token i+1 is retained
  - never evict tokens before the rightmost-processed position
- Layers 2…N run only on retained tokens; evicted (non-decodable) masked tokens
  are dropped from deep-layer compute and not unmasked this step.

### Fast-dLLM v2 codebase (`/research/data/transfer/data/n41/fast_v2`)
- `generation_functions.py` — `Fast_dLLM_QwenForCausalLM` with decode variants:
  `batch_sample` (line 44, **baseline**), `batch_sample_dynamo`,
  `batch_sample_sparse` (line 949, existing token-sparse via hidden-state cosine
  similarity), `batch_sample_fused`.
- `models/.../modeling.py` — `DecoderLayer.forward` (lines 339–371),
  `DecoderLayer.forward_sparse` (lines 660–885, dense/sparse gather-scatter
  template), `Fast_dLLM_QwenAttention.forward` (lines 226–313, does **not**
  currently expose Q·Kᵀ).
- `utils/static_kv_cache.py` — `StaticKVCache.write_sparse(k, v, layer_idx,
  write_positions)` (line 231), `get_full_kv` (line 249).
- `utils/block_sparse_cache.py` — `BlockSparseCache.cache_layer_input` (76),
  `cache_attn_output` (80), `cache_mlp_output` (88), `scatter_attn_output` (112),
  `scatter_mlp_output` (123).
- `eval.py` — method dispatch via env vars (lines 88–115); GSM8K via lm-eval
  `--tasks gsm8k --limit N`. Conda env at `v2/`.

The infrastructure for selective recompute (caches, scatter/gather, sparse KV
writes) already exists for `batch_sample_sparse`. The FOCUS port differs only in
the **selection signal** (attention-importance delta vs cosine similarity) and in
**when** the decision is made (once at layer 1 from early-layer measurement).

## Design

### Components / files touched
- `models/.../modeling.py`: add `DecoderLayer.forward_focus(...)` and a
  model-level forward path (e.g. `focus=True`). **Leave `forward` and
  `forward_sparse` untouched.** Because the attention module does not expose
  Q·Kᵀ, `forward_focus` computes the masked-token importance matmul explicitly,
  exactly as FOCUS's kernel does.
- `generation_functions.py`: add `batch_sample_focus(...)`, cloned from
  `batch_sample_sparse`'s block/step orchestration but driving the FOCUS forward.
- `eval.py`: add dispatch `FAST_DLLM_USE_FOCUS=1` → bind `batch_sample_focus`.
- caches: reuse `StaticKVCache.write_sparse`, `BlockSparseCache` as-is.

### The FOCUS forward (per denoising step, over the current block)
Measuring only over masked tokens:
1. **Layer 0 (dense, all tokens):** standard attention; additionally compute
   importance₀ (formula above); store as `first_layer_importance`.
2. **Layer 1 (dense):** recompute importance₁; `delta = importance₁ − importance₀`.
3. **Selection:** apply FOCUS's exact rule (K, mean+std threshold, top-K
   fallback, adjacency, rightmost-processed guard) → retain mask over masked
   tokens.
4. **Layers 2…N (sparse, retained only):** gather retained tokens →
   input_layernorm → Q/K/V proj for retained → `write_sparse` their K/V at their
   positions → attention (retained queries vs full KV) → `scatter_attn_output` →
   MLP on retained → `scatter_mlp_output`. Non-retained masked tokens keep their
   cached layer output and are not unmasked this step.

### `batch_sample_focus` orchestration
Same block loop as `batch_sample_sparse`, except per block:
- **First denoising step = full dense** (all layers) to seed `BlockSparseCache`
  and initialize `avg_decoded_tokens`.
- **Subsequent steps = FOCUS forward** (above). After each forward, the existing
  `x1_p > threshold` (+ argmax) unmasking runs **only over computed/retained
  positions**.
- Track `avg_decoded_tokens` as a running mean of tokens unmasked per step
  (FOCUS's `FocusInfo.avg_decoded_tokens`).
- Finished-sample eviction, bridge tokens, end-of-block KV commit — unchanged
  from `batch_sample_sparse`.

### Config / knobs (env vars, matching existing style)
- `FAST_DLLM_USE_FOCUS=1` — select the method (dispatched in `eval.py`).
- `FAST_DLLM_FOCUS_ALPHA` (default `1.0`) — retain-count multiplier.
- `FAST_DLLM_FOCUS_LAYERS` (default `"0,1"`) — measurement layers.
- `FAST_DLLM_FOCUS_RETAIN` (optional override) — fixed retain fraction; when
  unset, `avg_decoded_tokens` running mean is used (the faithful default).
- Reuses `threshold`, `block_size`, `small_block_size`, `top_p`, `temperature`.

## Eval & comparison (the run)
**Environment:** `/research/data/transfer/data/n41/FOCUS/run_env` (Python 3.11,
torch 2.8, transformers 4.57.1, accelerate). `lm_eval` is missing and must be
`pip install`ed into `run_env` (fast_v2's `eval.py` imports `cli_evaluate` /
`register_model`). Note: run_env has transformers 4.57.1 while fast_v2 pins
4.53.1 — the Fast-dLLM remote-code `modeling.py` may need small compatibility
adjustments for the newer transformers; handle during implementation.

Tiny smoke = `--tasks gsm8k --limit 20`.
- **Baseline:** default dispatch → `batch_sample`, GSM8K limit 20 → accuracy + outputs.
- **FOCUS:** `FAST_DLLM_USE_FOCUS=1`, same limit 20 → accuracy + outputs.
- Compare exact-match accuracy; sanity-check generations are coherent. Results in
  `logs/`. Report side-by-side.

## Testing / verification
- **Correctness gate:** with `alpha` (or `FAST_DLLM_FOCUS_RETAIN`) large enough to
  retain *all* masked tokens, `forward_focus` logits must match the dense
  `forward` within fp tolerance — proves the sparse path is correct before
  trusting the skipping.
- **Smoke gate:** the 20-problem FOCUS run completes without error and accuracy is
  within a sane margin of baseline.

## Out of scope
- Triton/CUDA kernels for importance (FOCUS uses them; here a PyTorch matmul over
  masked tokens is sufficient and faithful at this scale).
- Full GSM8K (1319) and throughput benchmarking — follow-on after the smoke test.
- Changes to `batch_sample`, `batch_sample_sparse`, or existing `forward`.

## Open questions (resolved)
- `avg_decoded_tokens`: use running mean (faithful default), with
  `FAST_DLLM_FOCUS_RETAIN` as a deterministic override. **Resolved: yes.**

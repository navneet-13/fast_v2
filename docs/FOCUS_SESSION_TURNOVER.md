# FOCUS-in-Fast-dLLM-v2 — Session Turnover

Dense handoff of all context/findings. Goal of the work: **port FOCUS attention-importance
token-skipping into Fast-dLLM v2, and compare accuracy + throughput (TPS) vs baselines.**

---

## 0. Repos, env, model

- **Fast-dLLM v2 repo (work dir):** `/research/data/transfer/data/n41/fast_v2`
- **FOCUS reference repo (the paper's impl):** `/research/data/transfer/data/n41/FOCUS` (LMDeploy/SDAR)
- **Model snapshot (the `modeling.py` we edit):**
  `models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/modeling.py`
- **Model = Qwen2-7B-based:** 28 layers, hidden=3584, ffn=18944, 28 heads, **4 KV heads (GQA, n_rep=7)**,
  head_dim=128, `bd_size`(block_size)=32, mask_id=151665, vocab=152064.
- **Per-token-per-layer FLOP (linear+MLP, the dominant term) ≈ 0.466 GFLOP** (attention adds ≤7%).
- **Eval env = the `v2/` conda env** (`v2/bin/python`, `v2/bin/accelerate`): flash-attn 2.8.3,
  flashinfer 0.6.6, transformers **4.53.1** (repo's pin), lm_eval 0.4.11, torch 2.9.1+cu128.
  - **`run_env` (`/research/data/transfer/data/n41/FOCUS/run_env`, tf 4.57.1) lacks flash-attn** →
    can only run the dense `batch_sample` baseline, NOT sparse/FOCUS (those hard-require
    `attn_backend.flash_kvcache_attention`, no SDPA fallback). Use `run_env` only for the CPU unit tests.
- **GPUs:** 8× RTX A5000 (23 GB, SM 8.6). flashinfer `flash4` is skipped (needs SM≥9.0) → FA2 used.
  **Always check `nvidia-smi` before running — GPUs are shared with other users.**

### Operational gotchas (critical)
- **HF dynamic-module cache:** `from_pretrained(trust_remote_code=True)` loads `modeling.py` from
  `~/.cache/huggingface/modules/transformers_modules/…`, **NOT the snapshot directly**. After editing
  `modeling.py` you MUST `rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* *Efficient*`
  or evals run stale code. (Pytest gate uses direct import via `tests/test_focus.py::_load_modeling`, so it
  bypasses the cache — gate can pass while eval runs stale code. Watch this.)
- **No git commits** (user preference — repo has no commits; everything untracked).
- **Always report accuracy AND TPS** (TPS = total tokens generated ÷ generation wall-clock).

---

## 1. What was implemented (all env-gated, existing methods untouched)

**`modeling.py`:**
- `_focus_importance(q, k, mask_idx, scaling)` — FOCUS importance: `scores = q·kᵀ·scale` →
  mask non-masked **keys** to −inf → `max_pool1d(k=3,s=1,p=1)` → **re-mask** keys → softmax over keys →
  zero non-masked **query** rows → sum over queries then heads → `(B,S)`. (q,k GQA-expanded via `repeat_kv`.)
- `_focus_select(delta, mask_idx, avg_decoded, focus_alpha, retain_override) -> (token_indices, Ksel)` —
  `K=ceil(avg_decoded·alpha)` clamp[1,S] (or `ceil(retain_override·S)`); threshold=`mean+std` of masked δ;
  candidates=`δ≥thr`; adjacency `keep i if i+1 cand`; **`retain = (~mask_idx) | (mustkeep|topk_K)`** (decoded
  tokens ALWAYS retained); uniform `Ksel=max(retain.sum(),K)`; `token_indices = topk(Ksel of priority)` sorted.
- `Fast_dLLM_QwenModel.forward_focus` (+ LM wrapper) — layers 0,1 dense + measure importance; select once
  after layer 1; layers 2..N **sparse per-layer gather/scatter** via `block_sparse_cache`; evicted tokens
  reuse per-layer cached attn/mlp outputs.
- `Fast_dLLM_QwenModel.forward_focus_compact` (+ LM wrapper) — **gather once** after layer 1 → `[B,Ksel,D]`;
  run layers 2..N as plain dense layers on the compacted tensor; **scatter once**; evicted positions use the
  **cached dense-seed final hidden** `self._focus_compact_seed_hidden` (cloned pre-norm at the seed step).
  No `block_sparse_cache`.
- Env-gated FLOP counters in both forward_focus variants (`FAST_DLLM_FOCUS_FLOPS=1`, global `_FOCUS_FLOP`).

**`generation_functions.py`:**
- `batch_sample_focus` — clone of `batch_sample_sparse`; **one dense seed step per block** (`is_dense=_block_dense==0`),
  then FOCUS steps; `avg_decoded` running mean (seeded at block_size); branch `FAST_DLLM_FOCUS_COMPACT=1` →
  `forward_focus_compact`; `mask_idx` passed is the FULL-block mask. Uses **`StaticKVCache`** + `BlockSparseCache`.
- Env-gated TL counter (`FAST_DLLM_TL_COUNT=1`, global `_BASELINE_FLOP`) in `batch_sample` (baseline + block-cache).

**`eval.py`:** dispatch `FAST_DLLM_USE_FOCUS=1 → batch_sample_focus`.

**Tests:** `tests/test_focus.py` — `_focus_importance`/`_focus_select` unit tests, and GPU gates
`test_focus_retain_all_matches_dense` + `test_focus_compact_retain_all_matches_dense`
(retain_override=1.0 ⇒ logits == dense, **max_abs_diff=0**, both pass).

### Env knobs
`FAST_DLLM_USE_FOCUS`, `FAST_DLLM_FOCUS_ALPHA`(1.0), `FAST_DLLM_FOCUS_LAYERS`("0,1"),
`FAST_DLLM_FOCUS_RETAIN`(override fraction), `FAST_DLLM_FOCUS_COMPACT`, `FAST_DLLM_DEBUG_FOCUS`,
`FAST_DLLM_FOCUS_FLOPS`, `FAST_DLLM_TL_COUNT`.
Existing: `FAST_DLLM_USE_DYNAMO`/`FAST_DLLM_SKIP_COMMIT`, `FAST_DLLM_EXECUTION_MODE`(eager|compile),
`FAST_DLLM_USE_SPARSE`, `FAST_DLLM_USE_FUSED`, `FAST_DLLM_BATCH_MODE`(compact),
`FAST_DLLM_FUSED_GATHER_INPUT_NORM`, model_args `use_block_cache`/`small_block_size`.

### Added later this session — delayed KV cache + FOCUS-on-DynamicCache (details in §11, §12)
- **`modeling.py`:** `Fast_dLLM_QwenForCausalLM._focus_update_frozen(frozen, mask_idx)` (staticmethod —
  paper-faithful right-neighbor freeze rule); `_focus_select` gained `frozen=` (drops settled tokens +
  clamps `Ksel ≤ n_free` so the K-floor can't re-admit frozen tokens); `forward_focus_compact` gained
  `frozen=`; **new `forward_focus_compact_dynamic` (+ LM wrapper)** = compact FOCUS over a `DynamicCache`
  prefix + `DynamicBlockKV` buffer + eager **SDPA** (no flash, no static buffer).
- **`utils/dynamic_block_kv.py`:** `DynamicBlockKV` — per-block deep-layer KV buffer (BHSD
  `[B,H_kv,block_size,D]`); `write_full`/`write`(scatter)/`get`/`reset`/`compact_batch`.
- **`generation_functions.py`:** `batch_sample_focus` gained delayed-cache wiring (per-block `frozen`,
  updated each step from the **pre-forward** mask); **new `batch_sample_focus_dynamic`** = `batch_sample`
  (DynamicCache + finished_samples loop) clone with the FOCUS step swapped in.
- **`eval.py`:** `FAST_DLLM_FOCUS_DYNAMIC=1` → `batch_sample_focus_dynamic`.
- **Tests (16/16 pass):** `test_focus_update_frozen_*` ×3, `_focus_select` frozen tests ×3,
  `test_focus_compact_frozen_all_false_equiv_none` (GPU), `test_dynamic_block_kv_*` ×2,
  `test_focus_compact_dynamic_retain_all_matches_dense` (GPU).
- **New env knobs:** `FAST_DLLM_FOCUS_DELAYED_CACHE`(0), `FAST_DLLM_FOCUS_DYNAMIC`(0).
- **Note:** the DynamicCache path uses **SDPA**, so it does NOT require flash-attn (unlike the static
  FOCUS path) — could in principle run in `run_env`.

---

## 2. Decode methods & KV cache (NOT all configs use static buffers!)

| method | selected by | KV cache | notes |
|---|---|---|---|
| `batch_sample` | default | **DynamicCache** (cat-grow) | baseline; `use_block_cache=True` adds a 2nd dynamic block cache |
| `batch_sample_dynamo` | `FAST_DLLM_USE_DYNAMO=1` | **StaticKVCache** | dense; **eager** (`EXECUTION_MODE=eager`→`make_eager_forward`) or **compiled** (`=compile`→`torch.compile(mode=reduce-overhead, fullgraph=True, dynamic=True)`). `forward_dynamo` has 0 graph breaks. |
| `batch_sample_sparse` | `FAST_DLLM_USE_SPARSE=1` | StaticKVCache | cosine-sim token sparsity (pre-existing) |
| `batch_sample_focus` / `_compact` | `FAST_DLLM_USE_FOCUS=1` (+`_COMPACT=1`) | **StaticKVCache** | our FOCUS port |

**StaticKVCache** (`utils/static_kv_cache.py`): fixed `[B, max_seq_len, H, D]` buffer; valid length is a
**tensor** (`cache_seqlens`/`scratch_seqlens`, passed as flash `seqused_k`); `scatter_` writes (`write_sparse`,
`write_scratch_compiled`); `mark_static_address` → CUDA-graph/compile ready. This is why **growing KV is NOT a
compile blocker** — buffer shape is constant, only a seqlen tensor changes.

---

## 3. Bugs found & fixed (key learnings — don't repeat)

1. **FOCUS = 0.00 accuracy (first end-to-end run).** Root cause: `_focus_select` restricted the recompute set
   to currently-**masked** tokens; a token's deep-layer KV is computed from the MASK embedding (unmasking
   happens AFTER the forward), and once decoded it was excluded forever → its deep KV stayed mask-derived →
   corrupted context → coherent-but-locally-garbled output. **Fix:** `retain = (~mask_idx) | masked_selected`
   (keep all decoded tokens + decodable masked; evict only non-decodable masked). Diagnostic that nailed it:
   `FAST_DLLM_FOCUS_RETAIN=1.0` (recompute everything) → clean text at ~baseline acc.

2. **Compact v1 = garbage (`从根本`/`from from` loops).** Root cause: evicted tokens got **layer-1-only logits**
   (2 layers deep); the per-step unmask decision (`x1_p>0.9` + forced argmax) reads **all** sub-block positions'
   logits, so shallow overconfident evicted-token logits decoded garbage → cascade. **Fix:** cache the dense-seed
   **final (full-depth) hidden** once per block (`self._focus_compact_seed_hidden`) and use it as the base for
   evicted positions before norm/lm_head. After fix: compact ≡ non-compact (bit-equivalent token counts).

**Design insight (why compact is valid):** per-layer gather/scatter of evicted tokens never affects *selected*
tokens — attention reads K/V from the **buffer** (not `hidden_states`), and queries are independent. Evicted
tokens' deep KV = seed value in both compact & non-compact (never rewritten by `write_sparse`). So the per-layer
work only produced evicted tokens' own logits — which the seed-final cache supplies in one shot.

---

## 4. Faithfulness audit vs FOCUS reference code (`/…/FOCUS/lmdeploy/.../focus.py`, `models/sdar.py`, `strategies/dllm/sequence.py`)

**Matches:** importance (masked Q·Kᵀ → maxpool-3 → softmax → sum over q,heads); measure at layers 0 & 1;
`delta = imp1 − imp0`; threshold `mean+std`; `retain = ceil(avg_decoded·alpha)` clamped; decoded tokens retained.
**Deviations (ranked):**
1. **Rightmost-processed guard OMITTED** — FOCUS force-retains unprocessed masked tokens left of the rightmost
   retained position (`focus.py:239-248` + `FocusState`). We don't. Highest-impact correctness gap.
2. **alpha=1.0 vs paper config 1.5** (`sdar_lmdeploy_focus.py`) — we over-evict relative to the paper.
3. **Selection: we UNION threshold∪topK; reference uses XOR** (threshold only if ≥target else topK).
4. **Importance over full-S (mask keys to −inf) vs reference's masked-only ragged window** — minor numeric diff.
5. **avg_decoded seeded at block_size vs reference 1.0.**
6. **Positional adjacency (i+1 in block) vs processing-order adjacency.**
(Deviations #2 over-evict and #3 over-retain roughly cancel; results stayed good. Strict reproduction needs the
rightmost guard + alpha=1.5 + XOR.)

---

## 5. RESULTS (accuracy = GSM8K flexible-extract exact_match; TPS = tokens/gen-time)

### Batch 1, 200 samples
| config | KV | accuracy | TPS |
|---|---|---|---|
| baseline | Dynamic | 0.830 | 80.7 |
| baseline + block cache (sbs=8) | Dynamic | 0.830 | 83.5 |
| baseline — static KV, eager (`dynamo`,`EXECUTION_MODE=eager`) | Static | **0.870** | 84.8 |
| **static KV + block cache (sbs=8)** | Static | 0.860 | **87.6** |
| FOCUS (non-compact) | Static | 0.860 | 67.2 |
| FOCUS (compact) | Static | 0.860 | 68.6 |

### Batch 16, 1000 samples
| config | accuracy | TPS | s/batch |
|---|---|---|---|
| baseline | 0.821 | 155 | 34.0 |
| baseline — static KV, eager | 0.826 | 168.9 | 29.9 |
| baseline + block cache (sbs=8, **dynamic** KV) | 0.817 | 226 | 23.5 |
| **static KV + block cache (sbs=8)** ⭐ fastest | **0.830** | **258.6** | 19.5 |
| FOCUS (non-compact) | 0.832 | 182 | 28.3 |
| FOCUS (compact) | 0.833 | 184 | 27.9 |

**static KV + block cache** (`FAST_DLLM_USE_DYNAMO=1 EXECUTION_MODE=eager` + `use_block_cache=True,small_block_size=8`)
is the fastest eager config — the static buffer and block cache STACK (168.9 → 258.6 at b16; 1.14× the
dynamic-KV block cache, 1.67× plain baseline), accuracy-lossless. The dynamo path had `use_block_cache`
wired but **broken** — two bugs fixed: (1) `replace_position=None` crash on full-block re-entry
(`utils/attention_backends.py` patched_forward: None→pos 0); (2) batch>1 compaction size-mismatch
(added `StaticBlockCache.compact_batch()` + call it beside `static_cache.compact_batch()`). No
`modeling.py` change → no HF-cache clear. See `logs/run1k_b16_compare.md` final section.

### FLOPs (decoder token-layers, limit 25 batch 1; FLOP = TL × 0.466 GFLOP)
| config | token-layers | FLOP saving vs baseline |
|---|---|---|
| baseline | 2,912,896 | — |
| block cache | 1,695,232 | **41.8%** |
| FOCUS | 2,143,034 | 26.4% raw / **31.7% intrinsic**† |
| FOCUS compact | 2,143,190 | same (FLOP-identical to non-compact) |

†intrinsic = `1 − focus_tl/base_tl` (FOCUS vs its own trajectory dense; commit-neutral). Deep-layers-only saving
= **34%**. Caveat: baseline TL includes commit-step forwards; FOCUS counter excludes them (~5%). FOCUS also runs
~8% more diffusion steps (eviction trades steps for per-step compute).

Earlier smaller runs (batch 1): FOCUS 20-smp 0.80; 50-smp 0.82; non-compact==compact 50-smp both 17,820 tokens
(bit-equivalent). Pre-fix FOCUS 0.00; retain-all control 0.60 clean.

---

## 6. Throughput analysis (why FOCUS is slower at b1, faster at b16)

- **Compact ≈ non-compact everywhere** (FLOP-identical; ~1-2% faster eager). Compact's value = simpler +
  **compile-friendly**, not eager speed. The per-layer gather/scatter it removes is ≪ the matmuls (rounding error).
- **Batch 1: FOCUS slower than baseline** (67-68 vs static-eager **84.8**). Decode is **launch/overhead-bound, not
  FLOP-bound** → the 31% FLOP cut buys ~0 wall-clock (26 deep layers still launch the same kernels). FOCUS *adds*:
  importance calc, `_focus_select` with **`.item()` host-syncs**, gather/scatter, +1 dense seed/block, +8% steps.
  **Static buffer is NOT the cause** — static-eager dense baseline is the *fastest* eager config (84.8).
- **Batch 16: FOCUS faster** (1.18× over baseline) — batching makes it compute-bound, FLOP saving converts to TPS.

---

## 7. Compile feasibility (next big lever)

- Growing KV is **already handled** by StaticKVCache (`dynamic=True` over cache positions); `forward_dynamo`
  compiles fullgraph. FOCUS is already on the static buffer.
- **FOCUS compile blocker = data-dependent `Ksel`** (gather to `[B,Ksel,D]` → recompile/graph-break) **+ `.item()`
  host syncs** (break CUDA graphs). Fixes: **fix Ksel to a constant budget (pad)** or **bucket it** (the repo's
  `DLLMCudagraphStrategy` already power-of-two buckets token counts); compute Ksel/avg_decoded **device-side**.
- `batch_sample_dynamo` is the template; **compact is the best compile target**.
- Also available but unused on FOCUS: `cuda_bench/fused_gather_norm.cu` fused gather+RMSNorm CUDA ext
  (`FAST_DLLM_FUSED_GATHER_INPUT_NORM=1`), already wired into `forward_sparse`, not `forward_focus`.

---

## 8. NEXT STEPS

**Done since first written:** static+block-cache stacking (§5); **delayed KV caching** (§11 — the paper's
within-block caching, the missing half of FOCUS; 1.48× @b16, 69% FLOP saving); **FOCUS-on-DynamicCache**
(§12 — breaks the OOM ceiling; b64 = 300 TPS). The alpha-down sweep is done and showed **alpha is inert**
(§11). "Compose FOCUS with block cache" is effectively **superseded** by the delayed cache (same mechanism,
token-granularity).

**Still open / planned (NOT done):**
1. **Close the ~2.3pt b16 accuracy gap of the DynamicCache path** vs static (0.800 vs 0.825). **UPDATE:**
   the "bf16 SDPA-vs-flash drift" theory was **tested and REFUTED** — swapping the dynamic deep-layer
   attention to `flash_attn_func` left accuracy at 0.802 (noise). **fp32-deep-attention is therefore also
   dead** (flash already does fp32 softmax accumulation). The gap is **structural** (KV layout / op-ordering /
   selection+delayed-cache interaction), so the fix is item 2 below, not a precision tweak. (Flash kept anyway:
   +5.8% TPS for free → b16 leader. See §12 flash-kernel experiment.)
2. **Rightmost-processed left-side guard** (faithfulness deviation #1, §4) — likely recovers the delayed
   cache's b1 accuracy drop (0.860→0.815) and the dynamic b16 gap. Highest-value faithfulness fix.
3. **Compile** (`EXECUTION_MODE=compile`) — fixed-Ksel/sync-free `forward_focus_compact` + the compiled
   dense baseline as target. The delayed cache's 69% FLOP cut would compound.
4. **Push past b64** on the DynamicCache path toward the new (much higher) memory ceiling; alpha=1.5/XOR.

---

## 9. Cross-referenced docs / files

- **Spec:** `docs/superpowers/specs/2026-05-28-fast-dllm-v2-focus-token-skipping-design.md`
- **Plan:** `docs/superpowers/plans/2026-05-28-fast-dllm-v2-focus-token-skipping.md`
- **Result reports:** `logs/focus_smoke_compare.md` (20-smp), `logs/run200_compare.md` (200-smp b1),
  `logs/run1k_b16_compare.md` (1000-smp b16, incl. compact update).
- **Original sparse-method spec (template for FOCUS):** `Token_Sparse_Instruction.md`
- **FOCUS reference impl:** `/research/data/transfer/data/n41/FOCUS/lmdeploy/pytorch/kernels/cuda/focus.py`,
  `lmdeploy/pytorch/models/sdar.py`, `lmdeploy/pytorch/strategies/dllm/sequence.py`;
  opencompass configs `opencompass-0.5.1.post1/.../models/sdar_lmdeploy_focus.py` (alpha=1.5).
- **Key source:** `models/…/modeling.py` (forward_focus, forward_focus_compact, _focus_*),
  `generation_functions.py` (batch_sample, batch_sample_focus, batch_sample_dynamo),
  `utils/static_kv_cache.py`, `utils/block_sparse_cache.py`, `utils/dynamo_utils.py`,
  `cuda_bench/fused_sparse_extension.py`, `eval.py`, `eval_script.sh`.
- **Run logs:** `logs/run200_*.log`, `logs/run1k_b16_*.log`, `logs/flop25_*.log`, `logs/compact_*.log`,
  `logs/run200_baseline_static_eager.log`, `logs/diag_*.log`.
- **Claude memory:** `no-git-commits`, `fast-v2-eval-env`, `report-accuracy-and-tps`, `focus-turnover-doc`
  (in the FOCUS project memory dir).
- **Delayed cache (§11):** spec `docs/superpowers/specs/2026-06-01-focus-delayed-kv-cache-design.md`,
  plan `docs/superpowers/plans/2026-06-01-focus-delayed-kv-cache.md`, results `logs/focus_delayed_cache.md`,
  scaling+alpha `logs/focus_batch_scaling.md`.
- **DynamicCache (§12):** spec `docs/superpowers/specs/2026-06-01-focus-dynamic-cache-design.md`,
  plan `docs/superpowers/plans/2026-06-01-focus-dynamic-cache.md`, results `logs/focus_dynamic_cache.md`,
  logs `logs/dyn_b{16,48,64}.log`, `logs/dyn_smoke_b1.log`.
- **Delayed-cache reference (the paper's mechanism):** `/…/FOCUS/lmdeploy/pytorch/strategies/dllm/sequence.py`
  (`DelayedCacheState` — `ready = non_mask & right_neighbor`; FOCUS is coupled: `_focus_enabled =
  focus_enabled and delayed_cache_enabled`).

---

## 10. Canonical run command (template)
```bash
cd /research/data/transfer/data/n41/fast_v2
export WORKSPACE=$(pwd) HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true HF_HUB_DISABLE_REVISION_CHECK=1 PYTHONUNBUFFERED=1
export FAST_DLLM_EXECUTION_MODE=eager FAST_DLLM_ATTENTION_BACKEND=auto FAST_DLLM_MAX_SEQ_LEN=2048
# FOCUS compact: FAST_DLLM_USE_FOCUS=1 FAST_DLLM_FOCUS_ALPHA=1.0 FAST_DLLM_FOCUS_LAYERS=0,1 FAST_DLLM_FOCUS_COMPACT=1
# (after any modeling.py edit:) rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* *Efficient*
mp=$WORKSPACE/models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/
CUDA_VISIBLE_DEVICES=<free gpu> v2/bin/accelerate launch eval.py --tasks gsm8k --batch_size <B> --num_fewshot 0 --limit <N> \
  --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
  --model_args "model_path=${mp},threshold=0.9,show_speed=True,use_block_cache=False" &> logs/<name>.log
```
TPS = `grep "Total number of tokens generated"` ÷ final `N/N [MM:SS<…]` time. Accuracy = `flexible-extract` row.

---

## 11. Delayed KV caching (the paper's missing half) — `FAST_DLLM_FOCUS_DELAYED_CACHE=1`

**Insight that motivated it:** our FOCUS port implemented only the **eviction** half of the paper. The
reference *couples* FOCUS with a **delayed KV cache** (`sequence.py:124`); the speedup comes from BOTH
not-recomputing non-decodable masked tokens AND not-recomputing *settled* (decoded) tokens. We were
recomputing every decoded token every step → `Ksel` floored by the decoded count → ~185 TPS plateau,
and **`alpha` was inert** (measured: 0.25→1.0 gives identical tokens/TPS — `retain.sum().max()` is
pinned by the decoded floor + the `mean+std` threshold, not the top-K that alpha controls).

**Mechanism (paper-faithful, right-neighbor rule):** a block position freezes once it is decoded AND its
right neighbor is decoded — `_focus_update_frozen(frozen, mask_idx) = frozen | (~mask_idx & shift_right(~mask_idx))`.
Computed in `batch_sample_focus` from the **pre-forward** mask (refresh-timing invariant: a token is
frozen only after one real-id reprocess; the unmask mutates `x_t`, not `mask_idx`, so the in-scope
`mask_idx` at the update is the pre-forward value). Frozen tokens leave `_focus_select`'s set; their KV
**persists** in the StaticKVCache scratch region (constant `past_len` within a block; `write_sparse` only
writes selected, so frozen entries are never overwritten). **Critical bug caught in review:** the K-floor
ignores `frozen`, so when `Ksel > n_free`, `topk` re-admits frozen tokens (priority `neg_inf`) — fixed by
`Ksel = max(1, min(Ksel, (~frozen).sum(1).max()))` (use `.max()` NOT `.min()`: `retain ⊆ non-frozen`, so
`.max()` never truncates a row's retained tokens; `.min()` would be a correctness bug).

**Results (b16/1000, max_seq_len=1024, eager, vs delayed OFF):**
| batch | acc OFF→ON | TPS OFF→ON | speedup | FLOP saving OFF→ON |
|---|---|---|---|---|
| 16 | 0.833 → **0.825** | 185.6 → **274.6** | **1.48×** | 31.7% → **69.1%** |
| 1  | 0.860 → 0.815 | 68.6 → 71.7 | 1.05× | — |

Near-lossless at b16 (compute-bound; FLOP cut converts to TPS), beats static+block-cache (258). b1 drops
~4.5pt (overhead-bound + more skipping per the batch-max Ksel → more approximation). Freeze timing verified
correct (flag-OFF bit-identical; output coherent). The b1 drop is the **omitted rightmost guard** (§4 #1).

---

## 12. FOCUS on DynamicCache (breaks the batch ceiling) — `FAST_DLLM_FOCUS_DYNAMIC=1`

**Why:** StaticKVCache eagerly pre-allocs `[B, max_seq_len, H, D]` (~112 MiB/row @1024) → **OOMs at
batch ≥48** on the 23 GB A5000 (FOCUS is locked to the static buffer). `DynamicCache` grows the prefix to
the *real* length (~350) → high batch fits. **The per-block KV buffer is unavoidable** (FOCUS needs
evicted tokens' KV to persist across steps) — that's `DynamicBlockKV`, NOT the sub-block "block cache".

**Architecture:** committed prefix = `DynamicCache`; current block's deep KV = `DynamicBlockKV` (reset per
block); deep-layer attention = **eager SDPA over `cat(prefix, buffer)`** (`is_causal=False`). Reuses
`_focus_select(frozen=)` + delayed cache unchanged. `batch_sample_focus_dynamic` clones `batch_sample`
(DynamicCache + `finished_samples` + batch-compaction-on-finish) with the FOCUS step swapped in; buffer +
`frozen` reallocated per block from `x_t.shape[0]` (auto-tracks compaction). Commit = plain dense
`self.forward(update_past_key_values=True)`.

**Results (1000 samples, delayed cache on, max_seq_len=1024):**
| batch | accuracy | TPS | peak mem | static path |
|---|---|---|---|---|
| 16 | 0.800 | 274.0 | 16 GB | runs (0.825/274.6) |
| 48 | 0.787 | 292.0 | 18 GB | **OOM** |
| 64 | 0.794 | **300.3** | 19 GB | **OOM** |

**Ceiling broken** — b48/b64 run; **TPS climbs with batch (274→292→300) → b64=300 TPS is the new leader.**
At b16, **parity** (274.0 vs 274.6 TPS); acc 0.800 vs 0.825 (~2.5pt) = bf16 drift from reimplementing
deep-layer attention as SDPA over `cat(DynamicCache, buffer)` (per-layer matches the real `self_attn` to
**0.002**; retain-all gate proves the FOCUS gather/scatter path is a faithful no-op; the 0.75 full-model
logit drift is benign — generation is coherent). **Debugging lesson:** the static path *delegates* dense
layers to `dl.self_attn`; the dynamic deep layers MUST reimplement attention (custom buffer) → ~bf16/layer
accumulates over 28 layers. Don't gate on exact-logit match; gate on **retain-all == dense-seed** (no-op
invariant) + end-to-end GSM8K.

**Process note:** implemented inline (the Agent subagent-dispatch tool hit "tool result missing" internal
errors twice — Tasks 1-2 completed before erroring, no work lost). Validated by 16/16 tests + 4 GSM8K runs,
not the formal two-stage subagent review.

**Flash-kernel experiment (root-cause test for the b16 gap — HYPOTHESIS REFUTED):** The §12 prose above
blamed the ~2.3pt b16 gap (0.800 vs static 0.825) on SDPA-vs-flash bf16 drift. **Tested it:** swapped the
dynamic deep-layer attention helper (`modeling.py` `_attn`, ~line 1316) from `F.scaled_dot_product_attention`
to `flash_attn_func` (GQA-native, fp32 softmax accumulation, no `repeat_kv`; SDPA fallback kept for
CPU/no-flash via a guarded `from flash_attn import flash_attn_func` at module top). **Result (b16/1000):
0.800 → 0.802 (noise, ±0.0126); the gap PERSISTS.** So the kernel is NOT the cause — and because flash
already accumulates softmax in fp32, the **fp32-deep-attention idea is dead too.** The gap is **structural**
(KV layout / op-ordering / selection+delayed-cache interaction), NOT attention-op precision. **Highest-value
lever is now the rightmost-processed left-side guard (deviation #1, §4)** — same fix recovers the delayed-cache
b1 drop. **Flash kept (default-on):** free **+5.8% TPS** (274.0 → 289.9, wall-clock 1209 → 1138 s) at equal
accuracy → b16 throughput leader. Behavior change to the dynamic path (unconditional; env-gate `_attn` if a
pure-SDPA reproduction is ever needed). Result row in `logs/focus_dynamic_cache.md`; run `logs/dyn_b16_flash.log`.
Correction for readers: the retain-all no-op gate still passes with flash, confirming the swap is faithful.

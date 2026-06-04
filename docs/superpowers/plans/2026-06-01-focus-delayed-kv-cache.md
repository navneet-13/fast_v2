# FOCUS Delayed KV Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the FOCUS paper's token-level *delayed KV caching* to our compact FOCUS path so settled (decoded + right-neighbor-decoded) tokens stop being recomputed every diffusion step, shrinking `Ksel` and breaking the ~185 TPS plateau.

**Architecture:** A per-block `frozen` boolean mask `[B, block_size]` is maintained in `batch_sample_focus`, reset each block, and passed into `forward_focus_compact`. Frozen positions are excluded from FOCUS selection (their KV persists untouched in the StaticKVCache block scratch region). The freeze set grows each step via the paper-faithful right-neighbor rule, computed from the pre-forward mask. Everything is env-gated (`FAST_DLLM_FOCUS_DELAYED_CACHE`, default off) so the current FOCUS behavior and all existing tests/results are unchanged.

**Tech Stack:** PyTorch, the trust-remote-code `modeling.py` (`forward_focus_compact`, `_focus_select`, `Fast_dLLM_QwenForCausalLM`), `generation_functions.py` (`batch_sample_focus`), `utils/static_kv_cache.py` (`StaticKVCache` scratch buffer), `pytest` (`tests/test_focus.py`).

**PROJECT CONVENTIONS (read before starting):**
- **No git commits.** This repo intentionally has no commits; never run `git commit`/`git add`. Each task's checkpoint is "tests green / smoke clean," not a commit.
- **HF dynamic-module cache:** after ANY edit to `modeling.py` you MUST clear the cache or evals run stale code:
  `rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* ~/.cache/huggingface/modules/transformers_modules/*Efficient*`
  (Unit tests via `tests/test_focus.py::_load_modeling` import the snapshot directly and bypass this cache — so a passing unit test does NOT prove the eval sees your edit. Always clear the cache before any eval run.)
- **Model `modeling.py` path:** `models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/modeling.py`
- **Unit tests run in `run_env` (CPU ok); GPU tests + evals run in the `v2/` conda env** (`v2/bin/python`, has flash-attn). Check `nvidia-smi` before any GPU run (shared box).

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `models/.../modeling.py` | `_focus_select` (selection), `forward_focus_compact` (compact deep-layer exec), `Fast_dLLM_QwenForCausalLM` (LM wrapper) | Add `frozen` param threading + a `_focus_update_frozen` staticmethod |
| `generation_functions.py` | `batch_sample_focus` (per-block diffusion loop) | Allocate/reset/update `frozen`, pass it to the forward, env-gate |
| `tests/test_focus.py` | unit + GPU-equivalence tests | Add freeze-rule unit tests, `_focus_select` frozen-exclusion tests, frozen-none equivalence GPU gate |
| `logs/focus_delayed_cache.md` | results | Create: acc/TPS/FLOP vs current FOCUS |

Selection-time changes live in `modeling.py`; loop/state changes in `generation_functions.py`; both share the single `_focus_update_frozen` staticmethod so the freeze rule has one source of truth.

---

### Task 1: Freeze-rule helper `_focus_update_frozen` (pure, unit-tested)

**Files:**
- Modify: `models/.../modeling.py` — add a `@staticmethod _focus_update_frozen` on `Fast_dLLM_QwenForCausalLM` (near the other forward_focus methods, after `forward_focus_compact` wrapper ~line 1530).
- Test: `tests/test_focus.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_focus.py` (these import the snapshot module via the existing `_load_modeling()` helper and call the staticmethod on the class):

```python
def test_focus_update_frozen_right_neighbor_rule():
    import torch
    m = _load_modeling()
    fn = m.Fast_dLLM_QwenForCausalLM._focus_update_frozen
    # block of 6; mask True = still masked. decoded = ~mask.
    # pos:       0      1      2     3      4      5
    # mask:      F      F      T     F      F      F   -> decoded T T F T T T
    mask = torch.tensor([[False, False, True, False, False, False]])
    frozen0 = torch.zeros_like(mask)
    frozen1 = fn(frozen0, mask)
    # freeze iff decoded AND right-neighbor decoded; rightmost never freezes:
    # 0: T&dec[1]=T -> True | 1: T&dec[2]=F -> False | 2: masked -> False
    # 3: T&dec[4]=T -> True | 4: T&dec[5]=T -> True | 5: rightmost -> False
    assert frozen1[0].tolist() == [True, False, False, True, True, False]


def test_focus_update_frozen_accumulates_and_is_monotonic():
    import torch
    m = _load_modeling()
    fn = m.Fast_dLLM_QwenForCausalLM._focus_update_frozen
    mask_a = torch.tensor([[False, False, True, True]])   # decoded: T T F F -> freeze pos0 only
    frozen = fn(torch.zeros_like(mask_a), mask_a)
    assert frozen[0].tolist() == [True, False, False, False]
    mask_b = torch.tensor([[False, False, False, True]])  # decoded: T T T F -> freeze pos0,1
    frozen = fn(frozen, mask_b)                            # OR-accumulates, never clears
    assert frozen[0].tolist() == [True, True, False, False]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `run_env/bin/python -m pytest tests/test_focus.py::test_focus_update_frozen_right_neighbor_rule tests/test_focus.py::test_focus_update_frozen_accumulates_and_is_monotonic -v`
Expected: FAIL with `AttributeError: ... has no attribute '_focus_update_frozen'`.

- [ ] **Step 3: Implement the staticmethod**

In `modeling.py`, inside `class Fast_dLLM_QwenForCausalLM`, add (after the `forward_focus_compact` wrapper method):

```python
    @staticmethod
    def _focus_update_frozen(frozen, mask_idx):
        """Delayed-cache freeze update (paper-faithful, right-neighbor rule).

        A block position becomes frozen — its KV cached, excluded from recompute —
        once it is decoded AND its immediate right neighbor is decoded. `mask_idx`
        is the PRE-forward block mask (True = still masked) for the current step, so
        `~mask_idx` = decoded entering this step; because every non-frozen decoded
        token is in the recompute set, a frozen token is guaranteed to have been
        reprocessed once with its real id (the refresh-timing invariant). The
        rightmost position has no right neighbor, so it never freezes. Monotonic:
        OR-accumulates, never clears within a block.

        frozen, mask_idx: (B, block_size) bool. Returns updated (B, block_size) bool.
        """
        dec = ~mask_idx
        right = torch.zeros_like(dec)
        right[:, :-1] = dec[:, 1:]
        return frozen | (dec & right)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `run_env/bin/python -m pytest tests/test_focus.py::test_focus_update_frozen_right_neighbor_rule tests/test_focus.py::test_focus_update_frozen_accumulates_and_is_monotonic -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Checkpoint (no commit)**

Run the full unit suite to confirm nothing else broke:
Run: `run_env/bin/python -m pytest tests/test_focus.py -k "focus_select or focus_importance or update_frozen" -v`
Expected: all PASS. (Do NOT commit — project convention.)

---

### Task 2: `_focus_select` excludes frozen positions

**Files:**
- Modify: `models/.../modeling.py` — `_focus_select` (module-level function, ~line 235-283).
- Test: `tests/test_focus.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_focus_select_excludes_frozen():
    import torch
    m = _load_modeling()
    S = 8
    delta = torch.arange(S, dtype=torch.float32).unsqueeze(0)      # pos 7 = highest importance
    mask_idx = torch.ones(1, S, dtype=torch.bool)                  # all masked
    frozen = torch.zeros(1, S, dtype=torch.bool)
    frozen[0, 7] = True                                            # freeze the top-delta token
    idx, ksel = m._focus_select(delta, mask_idx, avg_decoded=4.0, focus_alpha=1.0,
                                retain_override=None, frozen=frozen)
    assert 7 not in idx[0].tolist()                               # frozen never selected
    assert ksel <= S - 1                                          # budget excludes frozen


def test_focus_select_frozen_not_selected_under_k_floor():
    import torch
    m = _load_modeling()
    S = 8
    delta = torch.arange(S, dtype=torch.float32).unsqueeze(0)
    mask_idx = torch.ones(1, S, dtype=torch.bool)                 # all masked
    frozen = torch.zeros(1, S, dtype=torch.bool)
    frozen[0, 1:] = True                                          # freeze 7 of 8; only pos 0 free
    # avg_decoded*alpha=4 would floor Ksel at 4, but only 1 non-frozen position exists.
    idx, ksel = m._focus_select(delta, mask_idx, avg_decoded=4.0, focus_alpha=1.0,
                                retain_override=None, frozen=frozen)
    assert idx[0].tolist() == [0]                                 # only the non-frozen position
    assert ksel == 1                                              # K floor clamped to non-frozen count


def test_focus_select_frozen_none_is_unchanged():
    import torch
    m = _load_modeling()
    delta = torch.tensor([[0.0, 3.0, 1.0, 2.5, 0.2, 4.0]])
    mask_idx = torch.tensor([[True, True, False, True, True, True]])
    idx_a, k_a = m._focus_select(delta, mask_idx, avg_decoded=2.0, focus_alpha=1.0,
                                 retain_override=None)
    idx_b, k_b = m._focus_select(delta, mask_idx, avg_decoded=2.0, focus_alpha=1.0,
                                 retain_override=None, frozen=None)
    assert k_a == k_b and idx_a.tolist() == idx_b.tolist()        # default path identical
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `run_env/bin/python -m pytest tests/test_focus.py::test_focus_select_excludes_frozen tests/test_focus.py::test_focus_select_frozen_none_is_unchanged -v`
Expected: FAIL with `TypeError: _focus_select() got an unexpected keyword argument 'frozen'`.

- [ ] **Step 3: Implement frozen exclusion in `_focus_select`**

Change the signature (line ~235) from:
```python
def _focus_select(delta, mask_idx, avg_decoded, focus_alpha, retain_override=None):
```
to:
```python
def _focus_select(delta, mask_idx, avg_decoded, focus_alpha, retain_override=None, frozen=None):
```

Then locate the tail of the function (the block computing `retain`, `Ksel`, `priority`, `token_indices`):
```python
    masked_selected = (mustkeep | topk_mask) & valid
    retain = (~mask_idx) | masked_selected                    # (B, S) bool
    Ksel = int(torch.clamp(retain.sum(dim=1).max(), min=K).clamp(max=S).item())
    priority = torch.where(retain, delta + 1e4, delta)
    token_indices = priority.topk(Ksel, dim=1).indices
    token_indices = token_indices.sort(dim=1).values
    return token_indices, Ksel
```
and replace it with:
```python
    masked_selected = (mustkeep | topk_mask) & valid
    retain = (~mask_idx) | masked_selected                    # (B, S) bool
    if frozen is not None:
        # Delayed cache: settled tokens are never recomputed — drop them from the
        # retain set (budget) and force them out of the top-k by priority.
        retain = retain & ~frozen
    Ksel = int(torch.clamp(retain.sum(dim=1).max(), min=K).clamp(max=S).item())
    if frozen is not None:
        # The K=ceil(avg_decoded*alpha) floor does NOT know about frozen, so when it
        # exceeds the non-frozen count, topk would be forced to return frozen positions
        # (priority neg_inf) — leaking them into the recompute set and defeating the
        # cache. Clamp Ksel to the non-frozen count. Masked tokens are never frozen, so
        # this can never drop a token that still needs decoding. (.max() over the batch
        # serves the busiest row; for B=1 it guarantees zero frozen leakage.)
        n_free = int((~frozen).sum(dim=1).max().item())
        Ksel = max(1, min(Ksel, n_free))
    priority = torch.where(retain, delta + 1e4, delta)
    if frozen is not None:
        priority = torch.where(frozen, torch.full_like(priority, neg_inf), priority)
    token_indices = priority.topk(Ksel, dim=1).indices
    token_indices = token_indices.sort(dim=1).values
    return token_indices, Ksel
```
(`neg_inf` is already defined at the top of `_focus_select` as `torch.finfo(delta.dtype).min`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `run_env/bin/python -m pytest tests/test_focus.py::test_focus_select_excludes_frozen tests/test_focus.py::test_focus_select_frozen_none_is_unchanged -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Checkpoint (no commit)**

Run: `run_env/bin/python -m pytest tests/test_focus.py -k "focus_select" -v`
Expected: all existing `_focus_select` tests + the 2 new ones PASS (proves backward compatibility of the `frozen=None` default).

---

### Task 3: Thread `frozen` through `forward_focus_compact`

**Files:**
- Modify: `models/.../modeling.py` — `Fast_dLLM_QwenModel.forward_focus_compact` (~line 1090) and `Fast_dLLM_QwenForCausalLM.forward_focus_compact` wrapper (~line 1509).
- Test: `tests/test_focus.py` (GPU-gated, model the new test on `test_focus_compact_retain_all_matches_dense`).

- [ ] **Step 1: Write the failing GPU-equivalence test**

Add to `tests/test_focus.py`. It reuses the same setup helper the existing compact gate uses; assert that passing `frozen` all-False yields logits **bit-identical** to `frozen=None`:

```python
@pytest.mark.skipif(not _GPU, reason="needs CUDA + flash-attn (v2 env)")
def test_focus_compact_frozen_all_false_equiv_none():
    import torch
    m = _load_modeling()
    model, state = _setup_block_state()          # same harness as test_focus_compact_retain_all_matches_dense
    B, block_size = state["mask_idx"].shape
    frozen = torch.zeros(B, block_size, dtype=torch.bool, device=state["mask_idx"].device)
    out_none = model.forward_focus_compact(**state, frozen=None)
    _reset_block_state(model, state)             # re-seed identical block KV state
    out_frozen = model.forward_focus_compact(**state, frozen=frozen)
    assert torch.equal(out_none.last_hidden_state, out_frozen.last_hidden_state)
```

NOTE for the implementer: `_setup_block_state`/`_reset_block_state` mirror the fixtures already used by `test_focus_compact_retain_all_matches_dense` (block KV warmed via a dense seed step). If a reset helper does not yet exist, factor the existing seed-setup lines from that test into `_reset_block_state(model, state)` and call it in both tests. Do not invent new model behavior — just re-run the seed step so both calls start from identical KV.

- [ ] **Step 2: Run the test to verify it fails**

Run (v2 env, pick a free GPU from `nvidia-smi`):
`CUDA_VISIBLE_DEVICES=<free> v2/bin/python -m pytest tests/test_focus.py::test_focus_compact_frozen_all_false_equiv_none -v`
Expected: FAIL with `TypeError: forward_focus_compact() got an unexpected keyword argument 'frozen'`.

- [ ] **Step 3: Add `frozen` to both `forward_focus_compact` signatures and pass it to `_focus_select`**

(a) `Fast_dLLM_QwenForCausalLM.forward_focus_compact` wrapper (~line 1509): add `frozen=None` to the signature kwargs and forward it:
```python
    def forward_focus_compact(
        self,
        input_ids=None, past_key_values=None, use_cache: bool = True,
        cache_position=None, update_past_key_values: bool = False,
        block_sparse_cache=None, is_dense_step: bool = True, mask_idx=None,
        avg_decoded: float = 1.0, focus_alpha: float = 1.0, retain_override=None,
        focus_layers=(0, 1), attn_backend=None, frozen=None, **kwargs,
    ):
        outputs = self.model.forward_focus_compact(
            input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache,
            cache_position=cache_position, update_past_key_values=update_past_key_values,
            block_sparse_cache=block_sparse_cache, is_dense_step=is_dense_step,
            mask_idx=mask_idx, avg_decoded=avg_decoded, focus_alpha=focus_alpha,
            retain_override=retain_override, focus_layers=focus_layers,
            attn_backend=attn_backend, frozen=frozen, **kwargs,
        )
        ...
```

(b) `Fast_dLLM_QwenModel.forward_focus_compact` (~line 1090): add `frozen=None` to the signature (alongside `retain_override=None`), and change the `_focus_select` call (~line 1194):
```python
        token_indices, num_tokens = _focus_select(
            delta, mask_idx, avg_decoded, focus_alpha, retain_override, frozen=frozen,
        )
```
No other change in this method — frozen positions are now absent from `token_indices`, so they are never gathered (`hs_sel`) nor written (`write_sparse`); their scatter value falls through to the seed/base hidden exactly as evicted tokens do today.

- [ ] **Step 4: Run the test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES=<free> v2/bin/python -m pytest tests/test_focus.py::test_focus_compact_frozen_all_false_equiv_none -v`
Expected: PASS (frozen all-False ⇒ identical logits).

- [ ] **Step 5: Checkpoint — clear HF cache + re-run the existing compact gate**

```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* ~/.cache/huggingface/modules/transformers_modules/*Efficient*
CUDA_VISIBLE_DEVICES=<free> v2/bin/python -m pytest tests/test_focus.py::test_focus_compact_retain_all_matches_dense tests/test_focus.py::test_focus_compact_frozen_all_false_equiv_none -v
```
Expected: both PASS (the retain-all gate still max_abs_diff=0; default path untouched). No commit.

---

### Task 4: Wire `frozen` into `batch_sample_focus`

**Files:**
- Modify: `generation_functions.py` — `batch_sample_focus` (~line 1297-1629).
- Test: GPU integration smoke (eval.py).

- [ ] **Step 1: Add the env flag + per-block `frozen` allocation**

In `batch_sample_focus`, in the env-config block (near line 1340 where `_use_compact` is read), add:
```python
        _delayed_cache = os.environ.get("FAST_DLLM_FOCUS_DELAYED_CACHE", "0") == "1"
```
Then in the block loop, immediately after `x_t = x_init.clone()` (~line 1462), allocate the per-block frozen mask (fresh each block at the current — possibly compacted — batch size; delayed cache requires the compact path):
```python
                # Delayed KV cache (paper): settled tokens skip recompute this block.
                frozen = (
                    torch.zeros((x_t.shape[0], block_size), dtype=torch.bool, device=self.device)
                    if (_delayed_cache and _use_compact) else None
                )
```

- [ ] **Step 2: Pass `frozen` to the forward and update it after each step**

In the diffusion inner-`while` (the `_ff(...)` call ~line 1534), add `frozen=frozen,` to the kwargs:
```python
                            output = _ff(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=static_cache,
                                update_past_key_values=False,
                                block_sparse_cache=block_sparse_cache,
                                is_dense_step=is_dense,
                                mask_idx=mask_idx,
                                avg_decoded=avg_decoded,
                                focus_alpha=focus_alpha,
                                retain_override=retain_override,
                                focus_layers=focus_layers,
                                attn_backend=attn_backend,
                                frozen=frozen,
                            )
```
Then, at the end of the inner-`while` body — after `step += 1` (~line 1580) — update the freeze set using the **pre-forward** `mask_idx` (the `mask_idx` variable computed at the top of this iteration, BEFORE the unmask at line 1566; it is unchanged by the unmask, which writes into `x_t`, not `mask_idx`):
```python
                            if frozen is not None:
                                frozen = self._focus_update_frozen(frozen, mask_idx)
```
`self._focus_update_frozen` is the staticmethod from Task 1 (`self` is the `Fast_dLLM_QwenForCausalLM` instance these methods are bound to). No re-indexing on compaction is needed: compaction happens only at block boundaries (after this loop, ~line 1602), and `frozen` is reallocated from `x_t.shape[0]` at the next block start.

- [ ] **Step 3: Verify the flag is off by default (no behavior change)**

Run the existing FOCUS compact smoke WITHOUT the new flag and confirm token count is unchanged vs the known-good baseline (323,393 tokens at b16/1000 is the reference; for a fast check use limit 50):
```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* ~/.cache/huggingface/modules/transformers_modules/*Efficient*
cd /research/data/transfer/data/n41/fast_v2
export WORKSPACE=$(pwd) HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true HF_HUB_DISABLE_REVISION_CHECK=1 PYTHONUNBUFFERED=1
export FAST_DLLM_EXECUTION_MODE=eager FAST_DLLM_ATTENTION_BACKEND=auto FAST_DLLM_MAX_SEQ_LEN=1024
mp=$WORKSPACE/models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/
CUDA_VISIBLE_DEVICES=<free> FAST_DLLM_USE_FOCUS=1 FAST_DLLM_FOCUS_COMPACT=1 FAST_DLLM_FOCUS_ALPHA=1.0 FAST_DLLM_FOCUS_LAYERS=0,1 \
  v2/bin/accelerate launch eval.py --tasks gsm8k --num_fewshot 0 --limit 50 --batch_size 1 \
  --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
  --model_args "model_path=${mp},threshold=0.9,show_speed=True,use_block_cache=False" &> logs/delayed_offcheck.log
grep -aE "Total number of tokens|flexible-extract" logs/delayed_offcheck.log
```
Expected: runs clean; accuracy ≈ prior FOCUS at limit 50 (0.82). (Flag off ⇒ behavior identical.)

- [ ] **Step 4: Verify the flag ON runs clean (integration smoke)**

Same command but add `FAST_DLLM_FOCUS_DELAYED_CACHE=1` and write to `logs/delayed_smoke_b1.log`. Then for the batch path (exercises per-block frozen + compaction interplay) run limit 32 batch 16 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to `logs/delayed_smoke_b16.log`.
Expected: both complete with exit 0, no `RuntimeError`/shape mismatch, coherent generations, accuracy in a sane range (not 0.00, not garbage). If output is garbled → STOP and debug the freeze timing (most likely a frozen token whose KV was captured while still mask-derived — re-check that `mask_idx` used in Step 2 is the pre-forward value).

- [ ] **Step 5: Checkpoint (no commit)**

Confirm `logs/delayed_offcheck.log` (flag off) and `logs/delayed_smoke_b1.log` (flag on) both ran clean and the off path matches prior FOCUS accuracy.

---

### Task 5: Measure accuracy + TPS + FLOP savings (the payoff)

**Files:**
- Create: `logs/focus_delayed_cache.md`

- [ ] **Step 1: Run the comparison matrix (1000 samples, max_seq_len=1024, eager)**

For each of b16 and b32, run FOCUS compact **with** delayed cache (`FAST_DLLM_FOCUS_DELAYED_CACHE=1`) on a free GPU, logging to `logs/delayed_b{16,32}.log`. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for b32. Reference (delayed OFF) numbers already exist: b16 = 0.833 / 185.6 TPS, b32 = 0.823 / 184.0 TPS (`logs/scale1k_focus_b{16,32}.log`). Canonical command template is in `docs/FOCUS_SESSION_TURNOVER.md` §10.

- [ ] **Step 2: Capture FLOP savings**

Re-run b16 at `--limit 25` with `FAST_DLLM_FOCUS_FLOPS=1` and delayed cache on; record the `[focus_flops] ... total_tokenlayer_saving=` line. Compare to current FOCUS (≈31.7% intrinsic; delayed cache should raise it because settled tokens leave the recompute set).

- [ ] **Step 3: Write the results doc**

Create `logs/focus_delayed_cache.md` with: the env knob, an accuracy+TPS table (delayed OFF vs ON at b16/b32 — both columns, per project convention), the FLOP-saving delta, and a one-paragraph finding (did it break the ~185 plateau? did accuracy hold within noise of 0.83?). If accuracy dropped > ~1.5pp, note it and flag the left-side `rightmost_processed` guard (spec §6) as the likely cause / next step.

- [ ] **Step 4: Checkpoint (no commit)**

Confirm `logs/focus_delayed_cache.md` exists with both accuracy and TPS columns filled for all four cells (OFF/ON × b16/b32).

---

### Task 6 (stretch, optional): Numerical KV-equivalence verify mode

Only do this if Task 5 shows an accuracy regression you want to localize. Implements spec §7.1.

**Files:**
- Modify: `models/.../modeling.py` — `forward_focus_compact`.

- [ ] **Step 1: Add `FAST_DLLM_FOCUS_DELAYED_VERIFY` gate**

When `os.environ.get("FAST_DLLM_FOCUS_DELAYED_VERIFY","0")=="1"` and `frozen is not None`, after computing the compact `hidden_states`, ALSO run the deep layers densely on the **full** block (the existing seed-style dense path) to get reference logits, and assert that `argmax` at currently-**masked** positions matches between the delayed-cache and full-recompute results:
```python
        if os.environ.get("FAST_DLLM_FOCUS_DELAYED_VERIFY", "0") == "1" and frozen is not None:
            ref = hidden_states  # placeholder name; compute full-dense reference here
            # build full-dense hidden by running all deep layers on the ungathered stream,
            # then compare argmax at mask_idx positions; print max mismatch count.
```
Implementer note: reuse `_dense_layer_nc` over `last_focus_layer+1 .. num_layers` on the **full** `hidden_states` (pre-gather) into a temp, norm + lm_head both, and `print` the count of masked positions where argmax differs. This is debug-only (doubles compute when on).

- [ ] **Step 2: Run on 20 samples, batch 1**

Run a limit-20 b1 eval with the verify flag; expect near-zero argmax mismatches at masked positions on the freeze step (small drift acceptable on later steps — the bidirectional approximation). A large, growing mismatch indicates the left-side instability → implement the `rightmost_processed` guard (spec §6).

- [ ] **Step 3: Checkpoint (no commit)** — record findings in `logs/focus_delayed_cache.md`.

---

## Self-Review

**Spec coverage:** §1 motivation → Tasks 1-4. §2 scope (compact/eager/gated) → Task 3 (compact only), Task 4 (`_delayed_cache and _use_compact`). §3 freeze rule → Task 1 (staticmethod, right-neighbor, refresh-timing via pre-forward mask). §4 selection exclusion → Task 2. §5 state plumbing → Task 4 (per-block alloc; compaction handled by per-block reset, documented). §6 left-side fallback → Task 6 + Task 5 Step 3 note. §7 validation → Tasks 1-3 (unit/equiv), Task 4 (integration), Task 5 (e2e + FLOP), Task 6 (numerical). §8 env knobs → Task 4 (`FAST_DLLM_FOCUS_DELAYED_CACHE`), Task 6 (`..._VERIFY`). §9 edge cases → covered (dense seed unaffected since frozen all-False; rightmost never freezes by rule; per-block reset; Ksel clamp ≥1 retained from existing code). §10 success criteria → Task 5.

**Placeholder scan:** none — every code step shows real code. Task 6 has one implementer-note region (debug-only reference computation) flagged explicitly as stretch.

**Type consistency:** `frozen` is `(B, block_size)` bool everywhere; `_focus_update_frozen(frozen, mask_idx) -> bool tensor`; `_focus_select(..., frozen=None)`; `forward_focus_compact(..., frozen=None)` at both model and LM-wrapper layers; `batch_sample_focus` allocates `frozen` from `x_t.shape[0]`. Names consistent across tasks.

**One intentional deviation from spec:** §3 wrote `newly_frozen = ready & processed_this_step`; the plan uses `frozen |= ready` because every non-frozen decoded token is already in `retain` (so `processed_this_step` is implied) — equivalent and avoids needing a return value from the forward. Documented in Task 1's docstring.

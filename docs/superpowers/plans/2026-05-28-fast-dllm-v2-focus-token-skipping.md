# Fast-dLLM v2 — FOCUS Token Skipping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `batch_sample_focus` decode method to Fast-dLLM v2 that skips deep-layer compute on non-decodable tokens using FOCUS's attention-importance-delta signal, then run a 20-problem GSM8K accuracy comparison against the `batch_sample` baseline.

**Architecture:** Mirror the existing token-sparse path (`Fast_dLLM_QwenModel.forward_sparse` + `Fast_dLLM_QwenForCausalLM.forward_sparse` + `batch_sample_sparse`). The FOCUS variant runs layers 0–1 **dense** while measuring attention importance, computes `delta = imp₁ − imp₀`, selects decodable tokens with FOCUS's rule, and runs layers 2…N **sparse** on only the retained tokens — reusing `StaticKVCache.write_sparse` and `BlockSparseCache` scatter/gather. Existing `forward`, `forward_sparse`, `batch_sample`, `batch_sample_sparse` are left untouched.

**Tech Stack:** PyTorch, transformers (Qwen-based custom `modeling.py`), lm-eval-harness, the `run_env` venv.

**Spec:** `docs/superpowers/specs/2026-05-28-fast-dllm-v2-focus-token-skipping-design.md`

> **NOTE — no git commits.** The user has asked that Claude not create git commits. Replace the usual "commit" step at the end of each task with the **verification checkpoint** shown. Leave version control to the user.

---

## Key paths

- Model file (HF snapshot — same file that already holds `forward_sparse`):
  `models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/modeling.py`
  (referred to below as **`modeling.py`**)
- `generation_functions.py` — decode methods (`Fast_dLLM_QwenForCausalLM`)
- `eval.py` — lm-eval model wrapper + method dispatch (lines 87–116)
- `utils/block_sparse_cache.py`, `utils/static_kv_cache.py` — caches (no edits needed)
- Tests: `tests/test_focus.py` (new)
- Python: `/research/data/transfer/data/n41/FOCUS/run_env/bin/python` (call as `$PYBIN`)

```bash
export PYBIN=/research/data/transfer/data/n41/FOCUS/run_env/bin/python
export FV2=/research/data/transfer/data/n41/fast_v2
```

## Design notes locked in during planning

1. **Batch-uniform selection.** FOCUS's per-sequence ragged retain counts are not
   shape-regular for dense batched PyTorch. We operate over the full block
   (`M = block_size`) and make the recompute budget `Ksel` uniform across the
   batch (≥ the FOCUS rule's count). Recomputing a few extra low-priority tokens
   in some rows is harmless (it is a superset of the FOCUS-retained set).
2. **Masked-submatrix importance.** Importance uses only masked↔masked attention,
   recovered by `-inf`-masking non-masked **keys** and zeroing non-masked
   **query** contributions in the full `(B,H,block,block)` score matrix — faithful
   to FOCUS while staying batch-uniform.
3. **Decoded tokens are never recomputed in deep layers** — their layer outputs
   are fixed and read from `BlockSparseCache` (filled by the dense seed step).
   Non-selected masked tokens reuse cached deep-layer outputs → stale logits →
   not unmasked this step (exactly FOCUS's eviction effect).
4. **`avg_decoded_tokens`** is a running mean of tokens unmasked per step, seeded
   from the first dense step. `FAST_DLLM_FOCUS_RETAIN` (fraction of block) is an
   optional deterministic override.

---

## Task 1: Add `_focus_importance` helper to modeling.py

**Files:**
- Modify: `modeling.py` (add module-level function near `repeat_kv`, ~line 206)
- Test: `tests/test_focus.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_focus.py
import math, torch, torch.nn.functional as F
import importlib.util, sys, os

FV2 = os.environ["FV2"]
MODELING = os.path.join(
    FV2, "models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/"
    "snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/modeling.py",
)

def _load_modeling():
    spec = importlib.util.spec_from_file_location("fdllm_modeling", MODELING)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fdllm_modeling"] = mod
    spec.loader.exec_module(mod)
    return mod

def test_focus_importance_shape_and_masking():
    m = _load_modeling()
    B, H, S, d = 2, 4, 6, 8
    torch.manual_seed(0)
    q = torch.randn(B, H, S, d)
    k = torch.randn(B, H, S, d)
    mask_idx = torch.zeros(B, S, dtype=torch.bool)
    mask_idx[:, 2:] = True  # last 4 positions masked
    imp = m._focus_importance(q, k, mask_idx, scaling=d ** -0.5)
    assert imp.shape == (B, S)
    # non-masked query rows contribute nothing; importance is mass *received* by
    # masked keys, so non-masked key columns must have zero importance.
    assert torch.allclose(imp[:, :2], torch.zeros(B, 2), atol=1e-5)
    # masked tokens receive positive attention mass
    assert (imp[:, 2:] > 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $FV2 && $PYBIN -m pytest tests/test_focus.py::test_focus_importance_shape_and_masking -v`
Expected: FAIL — `module 'fdllm_modeling' has no attribute '_focus_importance'`

- [ ] **Step 3: Write minimal implementation**

Add to `modeling.py` (after `repeat_kv`, ~line 206). Uses the existing `F`
(torch.nn.functional) import already present in the file.

```python
def _focus_importance(q, k, mask_idx, scaling):
    """FOCUS attention importance over masked tokens.

    q: (B, H, S, d) query states (post-RoPE, heads already GQA-expanded to H)
    k: (B, H, S, d) key states (post-RoPE, GQA-expanded to H)
    mask_idx: (B, S) bool — True where the token is currently masked (mask_id)
    Returns: (B, S) importance = attention mass received by each masked token
             from masked queries, max-pooled (window 3) + softmaxed, summed over
             query positions and heads.
    """
    B, H, S, _ = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaling          # (B,H,S,S)
    key_mask = mask_idx[:, None, None, :]                            # (B,1,1,S)
    scores = scores.masked_fill(~key_mask, float("-inf"))
    pooled = F.max_pool1d(
        scores.reshape(B * H * S, 1, S), kernel_size=3, stride=1, padding=1,
    ).reshape(B, H, S, S)
    weights = torch.softmax(pooled, dim=-1)                          # over keys
    weights = torch.nan_to_num(weights, nan=0.0)                     # all-(-inf) rows
    q_mask = mask_idx[:, None, :, None].to(weights.dtype)            # (B,1,S,1)
    weights = weights * q_mask                                       # drop non-masked queries
    imp = weights.sum(dim=-2).sum(dim=1)                             # (B,S)
    return imp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $FV2 && $PYBIN -m pytest tests/test_focus.py::test_focus_importance_shape_and_masking -v`
Expected: PASS

- [ ] **Step 5: Verification checkpoint (no commit)**

Run: `cd $FV2 && $PYBIN -m pytest tests/test_focus.py -v`
Confirm 1 passed. Do NOT git commit.

---

## Task 2: Add `_focus_select` helper to modeling.py

**Files:**
- Modify: `modeling.py` (module-level function after `_focus_importance`)
- Test: `tests/test_focus.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_focus.py
def test_focus_select_respects_rule_and_budget():
    m = _load_modeling()
    # delta high for masked tokens 4,5; tokens 0-1 non-masked
    delta = torch.tensor([[ -9., -9., 0.1, 0.2, 5.0, 4.0]])
    mask_idx = torch.tensor([[False, False, True, True, True, True]])
    idx, ksel = m._focus_select(delta, mask_idx, avg_decoded=2.0, focus_alpha=1.0,
                                retain_override=None)
    # shape (B, Ksel); never selects non-masked positions 0,1
    assert idx.shape[1] == ksel
    assert ((idx != 0) & (idx != 1)).all()
    # the two highest-delta masked tokens (4,5) must be retained
    sel = set(idx[0].tolist())
    assert 4 in sel and 5 in sel

def test_focus_select_retain_all_override():
    m = _load_modeling()
    delta = torch.randn(1, 8)
    mask_idx = torch.ones(1, 8, dtype=torch.bool)
    idx, ksel = m._focus_select(delta, mask_idx, avg_decoded=1.0, focus_alpha=1.0,
                                retain_override=1.0)
    assert ksel == 8                      # retain every token
    assert sorted(idx[0].tolist()) == list(range(8))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $FV2 && $PYBIN -m pytest tests/test_focus.py -k focus_select -v`
Expected: FAIL — no attribute `_focus_select`

- [ ] **Step 3: Write minimal implementation**

Add to `modeling.py` after `_focus_importance` (add `import math` at top of file
if absent):

```python
def _focus_select(delta, mask_idx, avg_decoded, focus_alpha, retain_override=None):
    """FOCUS token selection over a full block.

    delta: (B, S) importance delta (imp_layer1 - imp_layer0)
    mask_idx: (B, S) bool — currently-masked positions (selection restricted here)
    avg_decoded: float running mean of tokens decoded per step
    focus_alpha: float retain multiplier
    retain_override: optional float in (0,1] — fixed retain fraction of the block
    Returns: (token_indices (B, Ksel) long sorted block positions, Ksel int)
    """
    B, S = delta.shape
    # restrict to masked positions
    neg_inf = torch.finfo(delta.dtype).min
    masked_delta = torch.where(mask_idx, delta, torch.full_like(delta, neg_inf))

    if retain_override is not None:
        K = max(1, min(S, int(math.ceil(retain_override * S))))
    else:
        K = max(1, min(S, int(math.ceil(avg_decoded * focus_alpha))))

    # per-sequence threshold = mean + std over masked deltas
    valid = mask_idx
    cnt = valid.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (torch.where(valid, delta, torch.zeros_like(delta)).sum(1, keepdim=True)) / cnt
    var = (torch.where(valid, (delta - mean) ** 2, torch.zeros_like(delta)).sum(1, keepdim=True)) / cnt
    thr = mean + var.sqrt()
    cand = valid & (delta >= thr)                                   # (B,S)

    # adjacency: keep token i if i+1 is a candidate
    adj = torch.zeros_like(cand)
    adj[:, :-1] = cand[:, 1:]
    mustkeep = (cand | adj) & valid

    # uniform recompute budget across batch (>= FOCUS count, >= K)
    Ksel = int(torch.clamp(mustkeep.sum(dim=1).max(), min=K).clamp(max=S).item())

    # priority: must-keep first, then by delta; non-masked excluded via neg_inf
    priority = masked_delta.clone()
    priority = torch.where(mustkeep, priority + 1e4, priority)
    token_indices = priority.topk(Ksel, dim=1).indices               # (B, Ksel)
    token_indices = token_indices.sort(dim=1).values
    return token_indices, Ksel
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $FV2 && $PYBIN -m pytest tests/test_focus.py -k focus_select -v`
Expected: PASS (both)

- [ ] **Step 5: Verification checkpoint (no commit)**

Run: `cd $FV2 && $PYBIN -m pytest tests/test_focus.py -v` — confirm 3 passed.

---

## Task 3: Add `Fast_dLLM_QwenModel.forward_focus`

**Files:**
- Modify: `modeling.py` (add method to `Fast_dLLM_QwenModel`, after `forward_sparse`, ~line 851)

This mirrors `forward_sparse` (modeling.py 660–851). Layers 0–1 run the dense
branch (cache intermediates) **and** compute importance; after layer 1 we select
once; layers 2…N run the sparse branch with the FOCUS-selected, fixed
`token_indices`. The sparse branch body is identical to `forward_sparse`'s
`else:` block (lines 752–844) EXCEPT `token_indices`/`num_tokens` come from FOCUS
instead of cosine-similarity top-k.

- [ ] **Step 1: Write the method**

Insert after the model-level `forward_sparse` (before `class Fast_dLLM_QwenForCausalLM`):

```python
    def forward_focus(
        self,
        input_ids=None,
        past_key_values=None,
        use_cache: bool = True,
        cache_position=None,
        update_past_key_values: bool = False,
        block_sparse_cache=None,
        is_dense_step: bool = True,
        mask_idx=None,                 # (B, block) bool — masked positions
        avg_decoded: float = 1.0,
        focus_alpha: float = 1.0,
        retain_override=None,
        focus_layers=(0, 1),
        attn_backend=None,
        **kwargs,
    ):
        """FOCUS token-skipping forward. Dense seed step (is_dense_step=True) is
        identical to forward_sparse's dense branch for every layer. On FOCUS steps
        (is_dense_step=False): focus_layers run dense + measure importance; after
        the last focus layer we select decodable tokens; remaining layers run
        sparse on those tokens only."""
        hidden_states = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + hidden_states.shape[1], device=hidden_states.device,
            )
        position_ids = cache_position.unsqueeze(0)
        attention_mask = None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        B = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // num_heads
        n_rep = num_heads // num_kv_heads

        cos_full, sin_full = position_embeddings
        cos_sin_full = torch.cat([cos_full, sin_full], dim=-1)

        imp = {}                         # layer_idx -> (B, seq_len) importance
        token_indices = None
        num_tokens = None
        last_focus_layer = max(focus_layers)

        def _dense_layer(layer_idx, hs):
            decoder_layer = self.layers[layer_idx]
            block_sparse_cache.cache_layer_input(layer_idx, hs)
            residual = hs
            hs_norm = decoder_layer.input_layernorm(hs)
            attn_out = decoder_layer.self_attn(
                hs_norm, position_embeddings=position_embeddings,
                attention_mask=attention_mask, past_key_value=past_key_values,
                cache_position=cache_position, update_past_key_values=update_past_key_values,
            )
            block_sparse_cache.cache_attn_output(layer_idx, attn_out)
            hs = residual + attn_out
            block_sparse_cache.cache_mlp_input(layer_idx, hs)
            residual = hs
            hs_norm = decoder_layer.post_attention_layernorm(hs)
            mlp_out = decoder_layer.mlp(hs_norm)
            hs = residual + mlp_out
            block_sparse_cache.cache_mlp_output(layer_idx, mlp_out)
            return hs

        def _measure_importance(layer_idx, hs):
            # project masked-token q/k for importance (cheap; masked subset math
            # via mask in _focus_importance). RoPE applied to all positions.
            attn = self.layers[layer_idx].self_attn
            hs_norm = self.layers[layer_idx].input_layernorm(hs)
            q = attn.q_proj(hs_norm).view(B, seq_len, num_heads, head_dim).transpose(1, 2)
            k = attn.k_proj(hs_norm).view(B, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos_full, sin_full)
            k = repeat_kv(k, n_rep)
            imp[layer_idx] = _focus_importance(q, k, mask_idx, attn.scaling)

        for layer_idx in range(num_layers):
            decoder_layer = self.layers[layer_idx]

            if is_dense_step:
                hidden_states = _dense_layer(layer_idx, hidden_states)
                continue

            # ---- FOCUS step ----
            if layer_idx in focus_layers:
                _measure_importance(layer_idx, hidden_states)
                hidden_states = _dense_layer(layer_idx, hidden_states)
                if layer_idx == last_focus_layer:
                    delta = imp[last_focus_layer] - imp[min(focus_layers)]
                    token_indices, num_tokens = _focus_select(
                        delta, mask_idx, avg_decoded, focus_alpha, retain_override,
                    )
                continue

            # ---- layers 2..N: sparse recompute on FOCUS-selected tokens ----
            # (body mirrors forward_sparse else-branch 766-844, fixed token_indices)
            block_sparse_cache.cache_layer_input(layer_idx, hidden_states)
            idx_h = token_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            selected_hidden = hidden_states.gather(1, idx_h)
            selected_norm = decoder_layer.input_layernorm(selected_hidden)

            attn = decoder_layer.self_attn
            q = attn.q_proj(selected_norm).view(B, num_tokens, num_heads, head_dim)
            k = attn.k_proj(selected_norm).view(B, num_tokens, num_kv_heads, head_dim)
            v = attn.v_proj(selected_norm).view(B, num_tokens, num_kv_heads, head_dim)
            idx_pos = token_indices.unsqueeze(-1).expand(-1, -1, cos_sin_full.shape[-1])
            sel_cos, sel_sin = cos_sin_full.expand(B, -1, -1).gather(1, idx_pos).chunk(2, dim=-1)
            q, k = apply_rotary_pos_emb(q, k, sel_cos, sel_sin, unsqueeze_dim=2)

            past_len = past_key_values.get_seq_length()
            write_positions = token_indices + past_len
            past_key_values.write_sparse(k, v, layer_idx, write_positions)

            full_k, full_v = past_key_values.get_full_kv(layer_idx)
            sparse_attn_out = attn_backend.flash_kvcache_attention(
                q, full_k, full_v, cache_seqlens=past_key_values.scratch_seqlens,
                is_causal=False, scaling=attn.scaling,
            )
            sparse_attn_out = sparse_attn_out.reshape(B, num_tokens, -1).contiguous()
            sparse_attn_out = attn.o_proj(sparse_attn_out)

            block_sparse_cache.scatter_attn_output(layer_idx, token_indices, sparse_attn_out)
            attn_output_full = block_sparse_cache.get_attn_output(layer_idx)
            mid = hidden_states + attn_output_full

            selected_mid = selected_hidden + sparse_attn_out
            selected_mid_norm = decoder_layer.post_attention_layernorm(selected_mid)
            sparse_mlp_out = decoder_layer.mlp(selected_mid_norm)
            block_sparse_cache.scatter_mlp_output(layer_idx, token_indices, sparse_mlp_out)
            mlp_output_full = block_sparse_cache.get_mlp_output(layer_idx)
            hidden_states = mid + mlp_output_full

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPastAndBlockCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
```

- [ ] **Step 2: Smoke-import the method**

Run:
```bash
cd $FV2 && $PYBIN -c "
import tests.test_focus as t
m = t._load_modeling()
assert hasattr(m.Fast_dLLM_QwenModel, 'forward_focus')
print('forward_focus present')"
```
Expected: `forward_focus present`

- [ ] **Step 3: Verification checkpoint (no commit)**

Confirm no syntax/import errors from Step 2. Do NOT git commit.

---

## Task 4: Add `Fast_dLLM_QwenForCausalLM.forward_focus` wrapper

**Files:**
- Modify: `modeling.py` (add method to `Fast_dLLM_QwenForCausalLM`, after its `forward_sparse`, ~line 1065)

- [ ] **Step 1: Write the method**

```python
    def forward_focus(
        self,
        input_ids=None,
        past_key_values=None,
        use_cache: bool = True,
        cache_position=None,
        update_past_key_values: bool = False,
        block_sparse_cache=None,
        is_dense_step: bool = True,
        mask_idx=None,
        avg_decoded: float = 1.0,
        focus_alpha: float = 1.0,
        retain_override=None,
        focus_layers=(0, 1),
        attn_backend=None,
        **kwargs,
    ):
        """FOCUS forward + lm_head. Mirrors forward_sparse wrapper (1027-1065)."""
        outputs = self.model.forward_focus(
            input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache,
            cache_position=cache_position, update_past_key_values=update_past_key_values,
            block_sparse_cache=block_sparse_cache, is_dense_step=is_dense_step,
            mask_idx=mask_idx, avg_decoded=avg_decoded, focus_alpha=focus_alpha,
            retain_override=retain_override, focus_layers=focus_layers,
            attn_backend=attn_backend, **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPastAndBlockCache(
            loss=None, logits=logits, past_key_values=outputs.past_key_values,
        )
```

- [ ] **Step 2: Smoke-import**

Run:
```bash
cd $FV2 && $PYBIN -c "
import tests.test_focus as t
m = t._load_modeling()
assert hasattr(m.Fast_dLLM_QwenForCausalLM, 'forward_focus')
print('LM forward_focus present')"
```
Expected: `LM forward_focus present`

- [ ] **Step 3: Verification checkpoint (no commit)**

---

## Task 5: Correctness gate — retain-all FOCUS step ≈ dense forward

**Files:**
- Test: `tests/test_focus.py`

This proves the sparse recompute path is wired correctly: with `retain_override=1.0`
(every masked token recomputed) a FOCUS step must produce the same logits as the
dense `forward_sparse` step on the same inputs. Requires the model weights and a
GPU. Mark it to run on the smoke GPU.

- [ ] **Step 1: Write the test**

```python
# append to tests/test_focus.py
import pytest

@pytest.mark.gpu
def test_focus_retain_all_matches_dense():
    """With retain_override=1.0, FOCUS step logits == dense step logits."""
    import os, torch, types
    os.environ.setdefault("WORKSPACE", os.environ["FV2"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import generation_functions as gf  # noqa
    from utils.static_kv_cache import StaticKVCache
    from utils.block_sparse_cache import BlockSparseCache
    from utils.attention_backends import get_attention_backend, patch_attention_layers

    ckpt = os.path.join(os.environ["FV2"], "models", "...")  # resolved at runtime; see Step 2
    # NOTE: load exactly as eval.py does (trust_remote_code, bf16, cuda).
    # Construct a single block, run one dense seed step (forward_sparse dense) to
    # fill caches, then compare forward_focus(retain_override=1.0) vs
    # forward_sparse(is_dense_step=True) logits.
    # Full body filled in during implementation using eval.py:60-85 loader.
    ...
```

> Because this gate needs the real checkpoint + GPU and reuses `eval.py`'s loader,
> the implementer fills the body by copying `eval.py`'s model-load block (lines
> ~60–85) and the cache/attention-backend setup from `batch_sample_sparse`
> (`generation_functions.py` 1030–1058). Assertion:
> `torch.allclose(focus_logits, dense_logits, atol=1e-2, rtol=1e-2)` over masked
> positions. (Tolerance is loose: importance projection adds fp noise but the
> retained set is identical, so deep-layer math matches.)

- [ ] **Step 2: Resolve checkpoint path**

Run:
```bash
cd $FV2 && $PYBIN -c "
import glob, os
p = glob.glob(os.path.join(os.environ['FV2'],'models','models--*Fast_dLLM*','snapshots','*'))
print(p[0] if p else 'NOT FOUND')"
```
Use the printed path as `ckpt` in the test.

- [ ] **Step 3: Run the gate**

Run: `cd $FV2 && CUDA_VISIBLE_DEVICES=1 $PYBIN -m pytest tests/test_focus.py -k retain_all -v -s`
Expected: PASS (`allclose` true).

- [ ] **Step 4: If it fails — debug before proceeding**

Use superpowers:systematic-debugging. Common causes: GQA `repeat_kv` mismatch,
`token_indices` not covering all masked positions when `retain_override=1.0`
(check `Ksel == num masked`), or `scratch_seqlens` not set for the block.
Do NOT proceed to the eval run until this gate passes.

- [ ] **Step 5: Verification checkpoint (no commit)**

---

## Task 6: Add `batch_sample_focus` to generation_functions.py

**Files:**
- Modify: `generation_functions.py` (new method on `Fast_dLLM_QwenForCausalLM`, after `batch_sample_sparse`, ~line 1273)

Clone `batch_sample_sparse` (lines 949–1272) verbatim, then apply these exact
changes. The block/prefill/commit/compaction logic is unchanged.

- [ ] **Step 1: Copy `batch_sample_sparse` to a new `batch_sample_focus`**

Duplicate the whole method; rename `def batch_sample_sparse` → `def batch_sample_focus`.

- [ ] **Step 2: Replace the env-config block** (was lines ~986–992) with:

```python
        execution_mode = os.environ.get("FAST_DLLM_EXECUTION_MODE", "eager")
        attn_backend_pref = os.environ.get("FAST_DLLM_ATTENTION_BACKEND", "auto")
        max_seq_len_env = int(os.environ.get("FAST_DLLM_MAX_SEQ_LEN", "4096"))
        seq_len_step = int(os.environ.get("FAST_DLLM_SEQ_LEN_STEP", "256"))
        focus_alpha = float(os.environ.get("FAST_DLLM_FOCUS_ALPHA", "1.0"))
        focus_layers = tuple(int(x) for x in os.environ.get("FAST_DLLM_FOCUS_LAYERS", "0,1").split(","))
        _retain = os.environ.get("FAST_DLLM_FOCUS_RETAIN", "")
        retain_override = float(_retain) if _retain else None
        debug_focus = os.environ.get("FAST_DLLM_DEBUG_FOCUS", "0") == "1"
```

- [ ] **Step 3: Add `avg_decoded` tracking state** just before the block loop
(after `start_block_idx = min_len // block_size` / the `_dense_steps` init):

```python
            avg_decoded = float(block_size)   # seed: assume full block first
            _ad_count = 0
```

- [ ] **Step 4: Replace the dense/sparse decision + forward call** (was lines
~1176–1198). The first step of each block (`step == 0` within the block, i.e.
`block_sparse_cache` just reset) is the dense seed; subsequent steps are FOCUS:

```python
                            mask_idx_full = (x_t[:, -block_size:] == mask_id) & bucket.active_mask[:, None]
                            is_dense = (_block_dense == 0)   # first step of block = dense seed
                            if is_dense:
                                _block_dense += 1; _dense_steps += 1
                            else:
                                _block_sparse += 1; _sparse_steps += 1

                            output = self.forward_focus(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=static_cache,
                                update_past_key_values=False,
                                block_sparse_cache=block_sparse_cache,
                                is_dense_step=is_dense,
                                mask_idx=mask_idx_full,
                                avg_decoded=avg_decoded,
                                focus_alpha=focus_alpha,
                                retain_override=retain_override,
                                focus_layers=focus_layers,
                                attn_backend=attn_backend,
                            )
                            logits = output.logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]
```

> Remove the old `transfer_ratio` / `refresh_interval` reads and the
> `is_dense = (step % refresh_interval == 0)` line — FOCUS uses one dense seed
> per block, then FOCUS steps.

- [ ] **Step 5: Update `avg_decoded` after unmasking** (right after the
`x_t[:, start:end][unmask_idx] = x_1[unmask_idx]` line, ~1214):

```python
                            decoded_now = int(unmask_idx.sum().item()) / max(1, x_1.shape[0])
                            _ad_count += 1
                            avg_decoded += (decoded_now - avg_decoded) / _ad_count
```

- [ ] **Step 6: Update the debug print** (the `if debug_sparse:` block ~1259) to
use `debug_focus` and print `focus_alpha`/`avg_decoded` instead of
`transfer_ratio`/`refresh`.

- [ ] **Step 7: Syntax check**

Run: `cd $FV2 && $PYBIN -c "import ast; ast.parse(open('generation_functions.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 8: Verification checkpoint (no commit)**

---

## Task 7: Add `FAST_DLLM_USE_FOCUS` dispatch in eval.py

**Files:**
- Modify: `eval.py` (the method-select `if/elif` chain, lines 91–116)

- [ ] **Step 1: Insert a new branch** as the first `elif` after the FUSED branch
(before the SPARSE branch at line 96):

```python
        elif os.environ.get("FAST_DLLM_USE_FOCUS", "0") == "1":
            print("Using FOCUS token-skipping diffusion mode", flush=True)
            self.model.mdm_sample = types.MethodType(
                generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_focus, self.model
            )
```

- [ ] **Step 2: Update the dispatch comment** (lines 87–90) to list
`FAST_DLLM_USE_FOCUS=1 → batch_sample_focus`.

- [ ] **Step 3: Syntax check**

Run: `cd $FV2 && $PYBIN -c "import ast; ast.parse(open('eval.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 4: Verification checkpoint (no commit)**

---

## Task 8: Prepare the run environment

**Files:** none (environment only)

- [ ] **Step 1: Install lm_eval into run_env**

Run:
```bash
$PYBIN -m pip install "lm_eval==0.4.3"
```
(If 0.4.3 is incompatible with the existing `datasets`/`transformers`, fall back to
`$PYBIN -m pip install lm_eval` and pin whatever resolves; record the version.)

- [ ] **Step 2: Verify imports the eval needs**

Run:
```bash
cd $FV2 && $PYBIN -c "
import lm_eval, transformers, torch
from lm_eval.__main__ import cli_evaluate
print('lm_eval', lm_eval.__version__, 'tf', transformers.__version__, 'torch', torch.__version__)"
```
Expected: prints versions, no ImportError.

- [ ] **Step 3: Confirm the model loads under transformers 4.57.1**

Run (uses eval.py's loader path indirectly):
```bash
cd $FV2 && WORKSPACE=$FV2 CUDA_VISIBLE_DEVICES=1 $PYBIN -c "
import os, glob, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p = glob.glob(os.path.join('$FV2','models','models--*Fast_dLLM*','snapshots','*'))[0]
tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(p, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
print('model loaded ok')"
```
Expected: `model loaded ok`. If the custom `modeling.py` errors on the newer
transformers (e.g. `ALL_ATTENTION_FUNCTIONS`, `Cache`, `FlashAttentionKwargs`
imports), apply the minimal import/signature fixes and re-run. Record any edits.

- [ ] **Step 4: Verification checkpoint (no commit)**

---

## Task 9: Run the baseline (`batch_sample`) GSM8K smoke

**Files:** writes results under `logs/`

- [ ] **Step 1: Locate the eval launch command**

Run: `cd $FV2 && grep -n "python\|eval.py\|--tasks\|--limit\|--model" eval_script.sh | head`
Use the existing invocation pattern; substitute `$PYBIN` for the python and
`--limit 20`.

- [ ] **Step 2: Run baseline (FOCUS off, sparse off)**

Run (adjust flags to match `eval_script.sh`'s real arg names):
```bash
cd $FV2 && WORKSPACE=$FV2 CUDA_VISIBLE_DEVICES=1 \
  $PYBIN eval.py --tasks gsm8k --limit 20 --model_args block_size=32,threshold=0.9 \
  2>&1 | tee logs/focus_smoke_baseline.log
```
(No `FAST_DLLM_USE_*` set → `batch_sample`.)

- [ ] **Step 3: Record baseline accuracy**

Run: `cd $FV2 && grep -iE "exact_match|acc|gsm8k" logs/focus_smoke_baseline.log | tail`
Record the exact-match score and confirm 20 problems ran.

- [ ] **Step 4: Verification checkpoint (no commit)**

---

## Task 10: Run the FOCUS (`batch_sample_focus`) GSM8K smoke

- [ ] **Step 1: Run FOCUS on the same 20 problems**

Run:
```bash
cd $FV2 && WORKSPACE=$FV2 CUDA_VISIBLE_DEVICES=1 FAST_DLLM_USE_FOCUS=1 \
  FAST_DLLM_FOCUS_ALPHA=1.0 FAST_DLLM_DEBUG_FOCUS=1 \
  $PYBIN eval.py --tasks gsm8k --limit 20 --model_args block_size=32,threshold=0.9 \
  2>&1 | tee logs/focus_smoke_focus.log
```
Confirm the log prints `Using FOCUS token-skipping diffusion mode`.

- [ ] **Step 2: Sanity-check generations + record accuracy**

Run: `cd $FV2 && grep -iE "exact_match|acc|gsm8k|focus_debug" logs/focus_smoke_focus.log | tail`
Confirm: 20 problems ran, outputs are coherent (not garbage/empty), and record
exact-match.

- [ ] **Step 3: If outputs are garbage or accuracy is ~0 — debug**

Use superpowers:systematic-debugging. Likely culprits: `mask_idx` slicing vs the
`start:end` sub-block convention, `avg_decoded` collapsing to ~0 (retains too few
tokens — check `focus_debug` print), or stale deep-layer cache for newly-context
tokens. The Task 5 gate already validated the recompute path, so focus on the
selection/orchestration.

- [ ] **Step 4: Verification checkpoint (no commit)**

---

## Task 11: Compare and report

**Files:** `logs/focus_smoke_compare.md` (new)

- [ ] **Step 1: Write the comparison summary**

Create `logs/focus_smoke_compare.md` with a table:

```markdown
# FOCUS vs baseline — GSM8K smoke (20 problems)

| Method                | exact_match | n  | notes |
|-----------------------|-------------|----|-------|
| batch_sample (base)   | <fill>      | 20 |       |
| batch_sample_focus    | <fill>      | 20 | alpha=1.0, layers=0,1 |

Dense seed steps / FOCUS steps (from focus_debug): <fill>
Observation: <accuracy within margin? outputs coherent?>
```

- [ ] **Step 2: Present results to the user**

Report the two accuracy numbers, whether FOCUS preserved accuracy on the smoke
set, and recommend next steps (alpha sweep, full 1319-problem run, throughput
measurement). Do NOT git commit.

---

## Self-Review

**Spec coverage:**
- FOCUS importance (masked Q·Kᵀ → maxpool → softmax → sum) → Task 1 ✓
- delta + selection rule (mean+std, top-K, adjacency, budget) → Task 2 ✓
- layers 0/1 dense + measure, layers 2+ sparse on retained → Task 3 ✓
- LM-head wrapper → Task 4 ✓
- correctness gate (retain-all == dense) → Task 5 ✓
- `batch_sample_focus` orchestration + avg_decoded running mean + dense seed → Task 6 ✓
- env knobs (`FAST_DLLM_USE_FOCUS/ALPHA/LAYERS/RETAIN`) → Tasks 6,7 ✓
- run_env + lm_eval install (transformers 4.57.1 caveat) → Task 8 ✓
- GSM8K limit-20 baseline vs FOCUS + report → Tasks 9–11 ✓
- left `forward`/`forward_sparse`/`batch_sample`/`batch_sample_sparse` untouched ✓
- no git commits ✓ (all commit steps replaced with verification checkpoints)

**Placeholder scan:** Task 5's test body and Task 9's exact CLI flags are
intentionally resolved at implementation time (they depend on the live checkpoint
path and `eval_script.sh`'s real arg names) — each has an explicit resolution step
(5.2, 9.1). No other placeholders.

**Type consistency:** `_focus_importance(q,k,mask_idx,scaling)`,
`_focus_select(delta,mask_idx,avg_decoded,focus_alpha,retain_override) -> (token_indices, Ksel)`,
and `forward_focus(..., mask_idx, avg_decoded, focus_alpha, retain_override, focus_layers, attn_backend)`
are used consistently across Tasks 1–4 and 6. `token_indices` is always (B, Ksel)
long block positions; cache calls match the signatures in `utils/*` (verified
against `block_sparse_cache.py`/`static_kv_cache.py`).

# FOCUS on DynamicCache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FOCUS variant whose committed prefix uses a lazily-grown `DynamicCache` (+ a small per-block KV buffer + eager SDPA) instead of `StaticKVCache`, so it runs at high batch (b48/b64) without OOM — carrying over the token-skipping + delayed-KV-cache behavior unchanged.

**Architecture:** New, isolated methods parallel to the static FOCUS path (Approach A — the static path is untouched). `DynamicBlockKV` holds the current block's deep-layer KV across diffusion steps; `forward_focus_compact_dynamic` runs FOCUS over `cat(DynamicCache_prefix, DynamicBlockKV)` via SDPA; `batch_sample_focus_dynamic` is the DynamicCache sampling loop. Env-gated, default off.

**Tech Stack:** PyTorch (eager SDPA, `torch.nn.functional.scaled_dot_product_attention`), HF `DynamicCache`, the trust-remote-code `modeling.py`, `generation_functions.py`, `eval.py`, `pytest`.

**PROJECT CONVENTIONS (read before starting):**
- **No git commits.** Repo intentionally has zero commits; never `git commit`/`git add`. Checkpoint = tests green / smoke clean.
- **HF dynamic-module cache:** after ANY edit to `modeling.py` you MUST clear it before running `eval.py` (it loads `modeling.py` from `~/.cache/huggingface/modules/...`, not the snapshot):
  `rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* ~/.cache/huggingface/modules/transformers_modules/*Efficient*`
  Unit/GPU tests via `tests/test_focus.py::_load_modeling()` import the snapshot directly and bypass this cache.
- **Interpreter:** `run_env` does NOT exist. Use `v2/bin/python` / `v2/bin/accelerate`; set `FV2=$PWD` for pytest. GPU is shared — check `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` and pick a free one.
- **Model `modeling.py`:** `models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/modeling.py`
- **This variant must NOT touch the static FOCUS path** (`forward_focus_compact`, `_focus_select`, `_focus_update_frozen`, `batch_sample_focus`). It reuses `_focus_select`/`_focus_importance`/`_focus_update_frozen` read-only.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `utils/dynamic_block_kv.py` | `DynamicBlockKV` — per-block deep-layer KV buffer (BHSD), reset/write/get | Create |
| `models/.../modeling.py` | `Fast_dLLM_QwenModel.forward_focus_compact_dynamic` + LM wrapper | Add 2 methods |
| `generation_functions.py` | `batch_sample_focus_dynamic` sampling loop | Add 1 method |
| `eval.py` | dispatch `FAST_DLLM_FOCUS_DYNAMIC=1` | Modify dispatch |
| `tests/test_focus.py` | unit + GPU tests | Add tests |
| `logs/focus_dynamic_cache.md` | results (acc/TPS/peak-mem at b16/32/48/64) | Create |

---

### Task 1: `DynamicBlockKV` per-block KV buffer

**Files:**
- Create: `utils/dynamic_block_kv.py`
- Test: `tests/test_focus.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_focus.py`):

```python
def test_dynamic_block_kv_write_full_and_get():
    import torch
    from utils.dynamic_block_kv import DynamicBlockKV
    buf = DynamicBlockKV(deep_layer_start=2, num_layers=4, batch_size=1,
                         num_kv_heads=2, block_size=5, head_dim=3,
                         dtype=torch.float32, device=torch.device("cpu"))
    k = torch.arange(1*2*5*3, dtype=torch.float32).reshape(1, 2, 5, 3)
    v = k + 100
    buf.write_full(2, k, v)
    gk, gv = buf.get(2)
    assert torch.equal(gk, k) and torch.equal(gv, v)


def test_dynamic_block_kv_scatter_updates_only_selected():
    import torch
    from utils.dynamic_block_kv import DynamicBlockKV
    buf = DynamicBlockKV(deep_layer_start=2, num_layers=3, batch_size=1,
                         num_kv_heads=2, block_size=5, head_dim=3,
                         dtype=torch.float32, device=torch.device("cpu"))
    base_k = torch.zeros(1, 2, 5, 3); base_v = torch.zeros(1, 2, 5, 3)
    buf.write_full(2, base_k, base_v)
    # scatter selected positions [1, 3] with all-ones KV
    sel = torch.tensor([[1, 3]])                       # (B, n)
    new_k = torch.ones(1, 2, 2, 3); new_v = torch.ones(1, 2, 2, 3) * 2
    buf.write(2, new_k, new_v, sel)
    gk, gv = buf.get(2)
    # positions 1 and 3 updated; 0,2,4 stay zero
    assert gk[0, 0, 1, 0] == 1 and gk[0, 0, 3, 0] == 1
    assert gk[0, 0, 0, 0] == 0 and gk[0, 0, 2, 0] == 0 and gk[0, 0, 4, 0] == 0
    assert gv[0, 0, 1, 0] == 2 and gv[0, 0, 0, 0] == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /research/data/transfer/data/n41/fast_v2 && FV2=$PWD v2/bin/python -m pytest tests/test_focus.py -k "dynamic_block_kv" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.dynamic_block_kv'`.

- [ ] **Step 3: Implement** `utils/dynamic_block_kv.py`:

```python
"""Per-block deep-layer KV buffer for FOCUS on the DynamicCache path.

Holds the CURRENT block's key/value tensors (BHSD: [B, num_kv_heads, block_size,
head_dim]) for each deep layer across diffusion steps, so FOCUS-evicted/frozen
tokens keep their KV while only selected tokens are recomputed. Reset per block;
the dense seed step overwrites every position via write_full(), so reset() is a
no-op (mirrors BlockSparseCache).
"""
import torch


class DynamicBlockKV:
    def __init__(self, deep_layer_start, num_layers, batch_size, num_kv_heads,
                 block_size, head_dim, dtype, device):
        self.deep_layer_start = deep_layer_start
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.k = {
            L: torch.zeros(batch_size, num_kv_heads, block_size, head_dim,
                           dtype=dtype, device=device)
            for L in range(deep_layer_start, num_layers)
        }
        self.v = {
            L: torch.zeros(batch_size, num_kv_heads, block_size, head_dim,
                           dtype=dtype, device=device)
            for L in range(deep_layer_start, num_layers)
        }
        self.batch_size = batch_size

    def write_full(self, layer_idx, k, v):
        """Overwrite the whole block KV for a layer. k, v: [B, H_kv, block_size, D]."""
        self.k[layer_idx].copy_(k)
        self.v[layer_idx].copy_(v)

    def write(self, layer_idx, k, v, token_indices):
        """Scatter selected positions. k, v: [B, H_kv, n, D]; token_indices: [B, n]."""
        idx = token_indices[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[3])
        self.k[layer_idx].scatter_(2, idx, k)
        self.v[layer_idx].scatter_(2, idx, v)

    def get(self, layer_idx):
        return self.k[layer_idx], self.v[layer_idx]

    def reset(self):
        # Seed step overwrites every position each block; nothing to clear.
        pass

    def compact_batch(self, active_indices):
        for L in self.k:
            self.k[L] = self.k[L][active_indices].contiguous()
            self.v[L] = self.v[L][active_indices].contiguous()
        self.batch_size = active_indices.numel()
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd /research/data/transfer/data/n41/fast_v2 && FV2=$PWD v2/bin/python -m pytest tests/test_focus.py -k "dynamic_block_kv" -v`
Expected: 2 passed.

- [ ] **Step 5: Checkpoint (no commit).** Confirm the two tests pass. Do NOT commit.

---

### Task 2: `forward_focus_compact_dynamic` (model method + LM wrapper)

**Files:**
- Modify: `models/.../modeling.py` — add `forward_focus_compact_dynamic` to `Fast_dLLM_QwenModel` (place right after `forward_focus_compact`, ~line 1256) and a wrapper to `Fast_dLLM_QwenForCausalLM` (right after its `forward_focus_compact` wrapper, ~line 1530).
- Test: `tests/test_focus.py` (GPU).

**Context for the implementer:** This MIRRORS `Fast_dLLM_QwenModel.forward_focus_compact` (read it first — same embed/rope/`_focus_importance`/`_focus_select`/gather/scatter-hidden/seed-hidden-cache structure). The ONLY differences are the attention + KV handling: instead of `StaticKVCache.write_sparse`/`get_full_kv`/flash_kvcache, it uses the HF `DynamicCache` prefix + the `DynamicBlockKV` buffer + eager SDPA. `DynamicCache` stores K/V as BHSD `[B, H_kv, seq, D]` (so transpose post-rope to BHSD); the static path kept BSHD for flash. Verify the rope conventions against `forward_focus_compact` (`apply_rotary_pos_emb(..., unsqueeze_dim=2)` for the deep-layer BSHD tensors, default `unsqueeze_dim=1` for the BHSD importance path) and `repeat_kv`/`_focus_importance`/`_focus_select`/`apply_rotary_pos_emb` are already imported/defined in the module.

- [ ] **Step 1: Write the failing GPU test** (append to `tests/test_focus.py`). It checks the retain-all sanity: with `retain_override=1.0` and no frozen, the dynamic compact forward's logits match a plain dense forward over the same DynamicCache + block state, within SDPA tolerance.

```python
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (v2 env)")
def test_focus_compact_dynamic_retain_all_matches_dense():
    import torch
    from transformers.cache_utils import DynamicCache
    from utils.dynamic_block_kv import DynamicBlockKV
    m = _load_modeling()
    model = _build_tiny_model_on_gpu(m)          # see implementer note below
    B, block_size = 1, 8
    cfg = model.config
    nkv = cfg.num_key_value_heads
    hd = cfg.hidden_size // cfg.num_attention_heads
    # a committed prefix of length 8 in a DynamicCache, then a fresh block of 8
    prefix_ids = torch.randint(0, 1000, (B, 8), device="cuda")
    dc = DynamicCache()
    model.forward(input_ids=prefix_ids, past_key_values=dc, update_past_key_values=True, block_size=block_size)
    block_ids = torch.randint(0, 1000, (B, block_size), device="cuda")
    # reference: plain dense forward over the block (update_past_key_values=False)
    ref = model.forward(input_ids=block_ids, past_key_values=dc, update_past_key_values=False).logits
    # dynamic FOCUS, retain-all: build buffer, run seed (writes buffer), then a retain-all FOCUS step
    buf = DynamicBlockKV(deep_layer_start=2, num_layers=cfg.num_hidden_layers, batch_size=B,
                         num_kv_heads=nkv, block_size=block_size, head_dim=hd,
                         dtype=model.dtype, device=torch.device("cuda"))
    mask_idx = torch.ones(B, block_size, dtype=torch.bool, device="cuda")
    model.model.forward_focus_compact_dynamic(input_ids=block_ids, past_key_values=dc, block_kv=buf,
        is_dense_step=True, mask_idx=mask_idx, focus_layers=(0, 1))                       # seed
    out = model.model.forward_focus_compact_dynamic(input_ids=block_ids, past_key_values=dc, block_kv=buf,
        is_dense_step=False, mask_idx=mask_idx, retain_override=1.0, focus_layers=(0, 1))  # retain-all
    foc = model.lm_head(out.last_hidden_state)
    assert torch.allclose(ref, foc, atol=2e-2, rtol=0), \
        f"max_abs_diff={ (ref-foc).abs().max().item() }"
```

IMPLEMENTER NOTE: a `_build_tiny_model_on_gpu` (or loading the real snapshot weights on GPU) helper may already exist in the GPU-test harness used by `test_focus_compact_retain_all_matches_dense` — REUSE that harness/model-construction rather than inventing one. If the existing harness uses an SDPA shim for flash_kvcache, your dynamic forward uses real SDPA directly (no shim), so the reference dense forward must use the same attention path as your forward for a fair compare — use the model's own `self.forward` as the reference (as written above) so both go through the model's SDPA. If you cannot construct a faithful reference within the harness, STOP and report NEEDS_CONTEXT.

- [ ] **Step 2: Run to verify it fails**

Run: `cd /research/data/transfer/data/n41/fast_v2 && CUDA_VISIBLE_DEVICES=<free> FV2=$PWD v2/bin/python -m pytest tests/test_focus.py::test_focus_compact_dynamic_retain_all_matches_dense -v`
Expected: FAIL with `AttributeError: ... has no attribute 'forward_focus_compact_dynamic'`.

- [ ] **Step 3: Implement `Fast_dLLM_QwenModel.forward_focus_compact_dynamic`** (after `forward_focus_compact`):

```python
    def forward_focus_compact_dynamic(
        self,
        input_ids=None, past_key_values=None, use_cache: bool = True,
        cache_position=None, update_past_key_values: bool = False,
        block_kv=None, is_dense_step: bool = True, mask_idx=None,
        avg_decoded: float = 1.0, focus_alpha: float = 1.0, retain_override=None,
        focus_layers=(0, 1), frozen=None, **kwargs,
    ):
        """Compact FOCUS over a DynamicCache prefix + per-block KV buffer, eager SDPA.
        Mirrors forward_focus_compact but for DynamicCache (BHSD) + DynamicBlockKV +
        SDPA instead of StaticKVCache + flash_kvcache."""
        import torch.nn.functional as F
        hidden_states = self.embed_tokens(input_ids)
        B = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // num_heads
        n_rep = num_heads // num_kv_heads
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + seq_len, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        cos_full, sin_full = self.rotary_emb(hidden_states, position_ids)
        cos_sin_full = torch.cat([cos_full, sin_full], dim=-1)
        scaling = self.layers[0].self_attn.scaling
        last_focus_layer = max(focus_layers)

        def _prefix_kv(L):
            if past_key_values is not None and len(past_key_values) > L:
                return past_key_values[L][0], past_key_values[L][1]   # [B, H_kv, past_len, D]
            return None, None

        def _sdpa(q, k_bhsd, v_bhsd):
            k = repeat_kv(k_bhsd, n_rep); v = repeat_kv(v_bhsd, n_rep)
            return F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scaling)

        def _dense_layer(L, hs, write_buffer):
            dl = self.layers[L]
            residual = hs
            hsn = dl.input_layernorm(hs)
            q = dl.self_attn.q_proj(hsn).view(B, seq_len, num_heads, head_dim)
            k = dl.self_attn.k_proj(hsn).view(B, seq_len, num_kv_heads, head_dim)
            v = dl.self_attn.v_proj(hsn).view(B, seq_len, num_kv_heads, head_dim)
            q, k = apply_rotary_pos_emb(q, k, cos_full, sin_full, unsqueeze_dim=2)  # BSHD
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)     # BHSD
            if write_buffer and block_kv is not None:
                block_kv.write_full(L, k, v)
            pk, pv = _prefix_kv(L)
            full_k = torch.cat([pk, k], dim=2) if pk is not None else k
            full_v = torch.cat([pv, v], dim=2) if pv is not None else v
            a = _sdpa(q, full_k, full_v)
            a = a.transpose(1, 2).reshape(B, seq_len, -1)
            a = dl.self_attn.o_proj(a)
            hs = residual + a
            hs = hs + dl.mlp(dl.post_attention_layernorm(hs))
            return hs

        def _measure_importance(L, hs):
            attn = self.layers[L].self_attn
            hsn = self.layers[L].input_layernorm(hs)
            q = attn.q_proj(hsn).view(B, seq_len, num_heads, head_dim).transpose(1, 2)
            k = attn.k_proj(hsn).view(B, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos_full, sin_full)
            k = repeat_kv(k, n_rep)
            return _focus_importance(q, k, mask_idx, attn.scaling)

        # Dense seed: all layers dense; write deep-layer block KV into buffer.
        if is_dense_step:
            for L in range(num_layers):
                hidden_states = _dense_layer(L, hidden_states, write_buffer=(L > last_focus_layer))
            self._focus_dyn_seed_hidden = hidden_states.clone()
            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPastAndBlockCache(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            )

        # FOCUS step
        imp = {}
        for L in focus_layers:
            imp[L] = _measure_importance(L, hidden_states)
            hidden_states = _dense_layer(L, hidden_states, write_buffer=False)
        delta = imp[last_focus_layer] - imp[min(focus_layers)]
        token_indices, num_tokens = _focus_select(
            delta, mask_idx, avg_decoded, focus_alpha, retain_override, frozen=frozen,
        )

        idx_h = token_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        hs_sel = hidden_states.gather(1, idx_h)                       # [B, Ksel, D]
        idx_pos = token_indices.unsqueeze(-1).expand(-1, -1, cos_sin_full.shape[-1])
        sel_cos, sel_sin = cos_sin_full.expand(B, -1, -1).gather(1, idx_pos).chunk(2, dim=-1)

        for L in range(last_focus_layer + 1, num_layers):
            dl = self.layers[L]
            residual = hs_sel
            sel_norm = dl.input_layernorm(hs_sel)
            q = dl.self_attn.q_proj(sel_norm).view(B, num_tokens, num_heads, head_dim)
            k = dl.self_attn.k_proj(sel_norm).view(B, num_tokens, num_kv_heads, head_dim)
            v = dl.self_attn.v_proj(sel_norm).view(B, num_tokens, num_kv_heads, head_dim)
            q, k = apply_rotary_pos_emb(q, k, sel_cos, sel_sin, unsqueeze_dim=2)   # BSHD
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)    # BHSD
            block_kv.write(L, k, v, token_indices)
            bk, bv = block_kv.get(L)
            pk, pv = _prefix_kv(L)
            full_k = torch.cat([pk, bk], dim=2) if pk is not None else bk
            full_v = torch.cat([pv, bv], dim=2) if pv is not None else bv
            a = _sdpa(q, full_k, full_v)
            a = a.transpose(1, 2).reshape(B, num_tokens, -1)
            a = dl.self_attn.o_proj(a)
            hs_sel = residual + a
            hs_sel = hs_sel + dl.mlp(dl.post_attention_layernorm(hs_sel))

        seed_hidden = getattr(self, "_focus_dyn_seed_hidden", None)
        base = seed_hidden if (seed_hidden is not None and seed_hidden.shape == hidden_states.shape) else hidden_states
        hidden_states = base.scatter(1, idx_h, hs_sel)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPastAndBlockCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
```

- [ ] **Step 4: Implement the LM wrapper** in `Fast_dLLM_QwenForCausalLM` (after its `forward_focus_compact` wrapper):

```python
    def forward_focus_compact_dynamic(
        self,
        input_ids=None, past_key_values=None, use_cache: bool = True,
        cache_position=None, update_past_key_values: bool = False,
        block_kv=None, is_dense_step: bool = True, mask_idx=None,
        avg_decoded: float = 1.0, focus_alpha: float = 1.0, retain_override=None,
        focus_layers=(0, 1), frozen=None, **kwargs,
    ):
        outputs = self.model.forward_focus_compact_dynamic(
            input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache,
            cache_position=cache_position, update_past_key_values=update_past_key_values,
            block_kv=block_kv, is_dense_step=is_dense_step, mask_idx=mask_idx,
            avg_decoded=avg_decoded, focus_alpha=focus_alpha, retain_override=retain_override,
            focus_layers=focus_layers, frozen=frozen, **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPastAndBlockCache(
            loss=None, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
        )
```

- [ ] **Step 5: Run the GPU test to verify it passes**

Run: `cd /research/data/transfer/data/n41/fast_v2 && CUDA_VISIBLE_DEVICES=<free> FV2=$PWD v2/bin/python -m pytest tests/test_focus.py::test_focus_compact_dynamic_retain_all_matches_dense -v`
Expected: PASS (logits match the dense reference within `atol=2e-2`). If it fails on a tolerance just above 2e-2, inspect whether it's a genuine bug vs SDPA-vs-reference numerics; if clearly numerics (small, non-structured), widen atol slightly and note it. If it's a large/structured diff, the attention assembly or rope is wrong — STOP and debug (do not just widen the tolerance).

- [ ] **Step 6: Checkpoint (no commit).** Re-run the static compact gate to confirm the static path is untouched:
`CUDA_VISIBLE_DEVICES=<free> FV2=$PWD v2/bin/python -m pytest tests/test_focus.py::test_focus_compact_retain_all_matches_dense tests/test_focus.py::test_focus_compact_dynamic_retain_all_matches_dense -v`
Expected: both PASS.

---

### Task 3: `batch_sample_focus_dynamic` sampling loop

**Files:**
- Modify: `generation_functions.py` — add `batch_sample_focus_dynamic` (place right after `batch_sample_focus`, ~line 1645).
- Test: GPU smoke (eval, after Task 5 wires dispatch — so this task's own check is a direct Python smoke; the eval smoke is in Task 5).

**Context:** Model this on `batch_sample` (lines 49-238, the DynamicCache loop) for prefill/commit/finished-padding, and on `batch_sample_focus` (lines 1297-1644) for the FOCUS env-config, the dense-seed-per-block, the `avg_decoded` running mean, and the per-block `frozen` alloc/update. Use a `DynamicCache` for `past_key_values` and a `DynamicBlockKV` for the per-block buffer. NO StaticKVCache, NO `set_kv_write_start`/`prepare_write_idx`/`scratch_seqlens`, NO batch compaction (use finished-slot padding like `batch_sample`).

- [ ] **Step 1: Implement `batch_sample_focus_dynamic`.** Full method:

```python
    @torch.no_grad()
    def batch_sample_focus_dynamic(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
    ):
        """FOCUS on a DynamicCache prefix + per-block KV buffer (eager SDPA).
        Carries token-skipping + delayed KV caching; runs at high batch without the
        StaticKVCache max_seq_len pre-allocation. Compact path only.

        Env: FAST_DLLM_FOCUS_ALPHA, FAST_DLLM_FOCUS_LAYERS, FAST_DLLM_FOCUS_RETAIN,
             FAST_DLLM_FOCUS_DELAYED_CACHE, FAST_DLLM_FOCUS_FLOPS.
        """
        from transformers.cache_utils import DynamicCache
        from utils.dynamic_block_kv import DynamicBlockKV

        focus_alpha = float(os.environ.get("FAST_DLLM_FOCUS_ALPHA", "1.0"))
        focus_layers = tuple(int(x) for x in os.environ.get("FAST_DLLM_FOCUS_LAYERS", "0,1").split(","))
        _retain = os.environ.get("FAST_DLLM_FOCUS_RETAIN", "")
        retain_override = float(_retain) if _retain else None
        _delayed_cache = os.environ.get("FAST_DLLM_FOCUS_DELAYED_CACHE", "0") == "1"

        config = self.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        batch_size = input_ids.shape[0]
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        last_focus_layer = max(focus_layers)

        # Per-block deep-layer KV buffer (reused across blocks; seed overwrites it).
        block_kv = DynamicBlockKV(
            deep_layer_start=last_focus_layer + 1, num_layers=num_layers,
            batch_size=batch_size, num_kv_heads=num_kv_heads, block_size=block_size,
            head_dim=head_dim, dtype=self.dtype, device=self.device,
        )

        # ---- Prefill (DynamicCache), mirroring batch_sample ----
        if min_len > block_size:
            output = self.forward(
                input_ids=input_ids[:, :(min_len // block_size * block_size)],
                use_cache=True, update_past_key_values=True, block_size=block_size,
            )
            logits, past_key_values = output.logits, output.past_key_values
            if min_len % block_size == 0:
                predict_sample_idx = (seq_len == min_len)
                next_token = logits[predict_sample_idx, -1:, :].argmax(dim=-1)
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = DynamicCache()

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        start_block_idx = min_len // block_size
        avg_decoded = float(block_size)
        _ad_count = 0

        for block_idx in range(start_block_idx, num_blocks):
            if finished_flag.all():
                break
            if (seq_block_idx == block_idx).all():
                pad_len = block_size - input_ids.shape[1] % block_size
                x_init = mask_id * torch.ones((batch_size, pad_len), device=self.device, dtype=torch.long)
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, :(block_idx + 1) * block_size]
            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()

            block_kv.reset()
            frozen = (
                torch.zeros((x_t.shape[0], block_size), dtype=torch.bool, device=self.device)
                if _delayed_cache else None
            )
            _block_dense = 0

            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)
                if mask_idx.sum() == 0:
                    for sample_idx in range(batch_size):
                        if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                            post = x_t[sample_idx, seq_len[sample_idx]:]
                            sp = (post == stop_token).nonzero()
                            if sp.numel() > 0:
                                x_t[sample_idx, seq_len[sample_idx] + sp[0][0] + 1:] = tokenizer.pad_token_id
                    if finished_flag.all():
                        break
                    output = self.forward(
                        input_ids=x_t[:, -block_size:], use_cache=True,
                        past_key_values=past_key_values, update_past_key_values=True,
                        block_size=block_size,
                    )
                    past_key_values = output.past_key_values
                    next_token = output.logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    break

                is_dense = (_block_dense == 0)
                _block_dense += 1
                output = self.forward_focus_compact_dynamic(
                    input_ids=x_t[:, -block_size:], use_cache=True,
                    past_key_values=past_key_values, update_past_key_values=False,
                    block_kv=block_kv, is_dense_step=is_dense, mask_idx=mask_idx,
                    avg_decoded=avg_decoded, focus_alpha=focus_alpha,
                    retain_override=retain_override, focus_layers=focus_layers,
                    frozen=frozen,
                )
                logits = output.logits
                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

                x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                x1_p = torch.where(mask_idx, x1_p, -torch.inf)
                unmask_idx = (x1_p > threshold)
                max_prob_idx = x1_p.argmax(dim=-1)
                unmask_idx[torch.arange(x_1.shape[0], device=self.device), max_prob_idx] = True
                unmask_idx = unmask_idx & mask_idx
                x_t[:, -block_size:][unmask_idx] = x_1[unmask_idx]

                decoded_now = float(unmask_idx.sum().item()) / max(1, x_1.shape[0])
                _ad_count += 1
                avg_decoded += (decoded_now - avg_decoded) / _ad_count

                finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                finished_flag = finished_flag | finished_row_flags

                if frozen is not None:
                    frozen = self._focus_update_frozen(frozen, mask_idx)

            if input_ids.shape[1] == x_t.shape[1]:
                input_ids = x_t
            else:
                input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

        return input_ids
```

NOTE: The exact `input_ids` stitching / finished-token trimming and the `sample_with_top_p` signature should be cross-checked against `batch_sample` and `batch_sample_focus` and made to match whatever those do (this method must return the same shape/contract as `batch_sample_focus` so `eval.py` can use it interchangeably). If `batch_sample`'s end-of-block stitching differs from what's written here, follow `batch_sample`'s version. If unsure about the return contract, read how `eval.py` consumes `batch_sample_focus`'s return value and match it.

- [ ] **Step 2: Smoke-check it imports and the method is callable** (no GPU needed for import):
`cd /research/data/transfer/data/n41/fast_v2 && FV2=$PWD v2/bin/python -c "import ast; ast.parse(open('generation_functions.py').read()); print('parse ok')"`
Expected: `parse ok`. (Full functional verification happens in Task 5 via eval.)

- [ ] **Step 3: Checkpoint (no commit).**

---

### Task 4: Wire `eval.py` dispatch

**Files:**
- Modify: `eval.py` — the FOCUS dispatch (where `FAST_DLLM_USE_FOCUS=1` selects `batch_sample_focus`).

- [ ] **Step 1: Read the existing dispatch** in `eval.py` (grep for `FAST_DLLM_USE_FOCUS` and `batch_sample_focus`). Confirm the call site that selects the sampling function.

- [ ] **Step 2: Add the dynamic branch.** Where it currently picks `batch_sample_focus`, route to the dynamic variant when `FAST_DLLM_FOCUS_DYNAMIC=1`:
```python
        if os.environ.get("FAST_DLLM_FOCUS_DYNAMIC", "0") == "1":
            _focus_fn = self.model.batch_sample_focus_dynamic
        else:
            _focus_fn = self.model.batch_sample_focus
```
(Adapt to the actual local variable names / call structure in eval.py — match how `batch_sample_focus` is currently invoked, same arguments.)

- [ ] **Step 3: Checkpoint (no commit).** `cd /research/data/transfer/data/n41/fast_v2 && FV2=$PWD v2/bin/python -c "import ast; ast.parse(open('eval.py').read()); print('parse ok')"` → `parse ok`.

---

### Task 5: Functional smoke + ceiling-break + measurement

**Files:**
- Create: `logs/focus_dynamic_cache.md`

- [ ] **Step 1: Correctness smoke (b1, flag on).** Clear HF cache, pick a free GPU, run:
```bash
cd /research/data/transfer/data/n41/fast_v2
rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* ~/.cache/huggingface/modules/transformers_modules/*Efficient*
export WORKSPACE=$(pwd) HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true HF_HUB_DISABLE_REVISION_CHECK=1 PYTHONUNBUFFERED=1
export FAST_DLLM_EXECUTION_MODE=eager FAST_DLLM_ATTENTION_BACKEND=auto FAST_DLLM_MAX_SEQ_LEN=1024
mp=$WORKSPACE/models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/
CUDA_VISIBLE_DEVICES=<free> FAST_DLLM_USE_FOCUS=1 FAST_DLLM_FOCUS_DYNAMIC=1 FAST_DLLM_FOCUS_COMPACT=1 \
  FAST_DLLM_FOCUS_DELAYED_CACHE=1 FAST_DLLM_FOCUS_ALPHA=1.0 FAST_DLLM_FOCUS_LAYERS=0,1 \
  v2/bin/accelerate launch eval.py --tasks gsm8k --num_fewshot 0 --limit 50 --batch_size 1 \
  --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
  --model_args "model_path=${mp},threshold=0.9,show_speed=True,use_block_cache=False" &> logs/dyn_smoke_b1.log
grep -aE "Total number of tokens|flexible-extract|Error|Traceback" logs/dyn_smoke_b1.log
```
Expected: exit 0, coherent output, accuracy in a sane range (compare to the static dynamic-OFF FOCUS@limit50 ~0.82 and the static delayed@b1 from `logs/focus_delayed_cache.md`). If garbled/0.00 → the dynamic forward attention assembly is wrong; debug before proceeding.

- [ ] **Step 2: Parity (b16, 1000) vs static.** Run the same with `--batch_size 16 --limit 1000` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, log `logs/dyn_b16.log`. Expected: accuracy within noise of the STATIC delayed-cache b16 (0.825, `logs/focus_delayed_cache.md`). Record acc + TPS.

- [ ] **Step 3: THE HEADLINE — break the ceiling (b48, b64).** Run `--batch_size 48 --limit 1000` then `--batch_size 64 --limit 1000` (both with `expandable_segments:True`), logs `logs/dyn_b48.log`, `logs/dyn_b64.log`. These OOM on the static path. Expected: they RUN (no OOM) at max_seq_len=1024. Record acc, TPS, and peak memory (add a `torch.cuda.max_memory_allocated()/1e9` print at the end of generation, or read `nvidia-smi` peak during the run). If b64 OOMs, note the achievable ceiling.

- [ ] **Step 4: FLOP saving carries.** Run b1 `--limit 25` with `FAST_DLLM_FOCUS_FLOPS=1` + the dynamic flags; record the `[focus_flops] total_tokenlayer_saving=` (expect ~69%, matching the static delayed path).

- [ ] **Step 5: Write `logs/focus_dynamic_cache.md`** with: the env knobs; an accuracy + TPS + peak-memory table across b16/b32/b48/b64 (delayed cache on); the static-vs-dynamic b16 parity check; the FLOP saving; and a finding paragraph — did it break the OOM ceiling, what batch is now reachable, and the throughput crossover vs static (static wins low batch, dynamic wins where static can't run). Both accuracy AND TPS columns (project convention).

- [ ] **Step 6: Checkpoint (no commit).** Confirm the results doc has all cells filled and the ceiling-break (b48 at minimum) is demonstrated.

---

## Self-Review

**Spec coverage:** §3 architecture → Tasks 1-3. §4a forward → Task 2. §4b buffer → Task 1. §4c sampling loop → Task 3. §4d dispatch → Task 4. §5 memory/throughput framing → Task 5 Steps 3/5. §7 validation: retain-all (Task 2 Step 1), cross-cache parity (Task 5 Step 2), ceiling-break (Task 5 Step 3), FLOP (Task 5 Step 4), no-regression (Task 2 Step 6 re-runs the static gate; static methods are never edited). §8 edge cases — seed writes full buffer (Task 2), per-block reset (Tasks 1/3), rightmost-never-frozen (reused `_focus_update_frozen`), finished padding (Task 3), all-frozen Ksel≥1 (reused `_focus_select`). §9 success criteria → Task 5.

**Placeholder scan:** Tasks 2 and 3 contain implementer NOTES directing verification against existing siblings (rope conventions; end-of-block stitching; eval return contract) rather than fully-specified code for those boilerplate parts — this is deliberate (mirror the working sibling) and the differing/novel code IS given in full. No "TBD"/"add error handling"/empty stubs.

**Type consistency:** `block_kv` is a `DynamicBlockKV` everywhere; `write_full(L,k,v)` / `write(L,k,v,token_indices)` / `get(L)` / `reset()` / `compact_batch()` signatures match between Task 1 and their use in Tasks 2-3. `frozen` is `(B, block_size)` bool, threaded into `forward_focus_compact_dynamic` and updated via `self._focus_update_frozen` (reused). K/V are BHSD `[B, num_kv_heads, *, head_dim]` consistently (DynamicCache + buffer + SDPA). `forward_focus_compact_dynamic` signatures match between the model method (Task 2 Step 3), the LM wrapper (Task 2 Step 4), and the call in `batch_sample_focus_dynamic` (Task 3).

**Risk note for executor:** Task 2 is the hard one (a parallel forward). The retain-all GPU test is the gate; if its diff is large/structured, the bug is in attention assembly or rope, not tolerance. Do NOT widen the tolerance to pass a structured diff.

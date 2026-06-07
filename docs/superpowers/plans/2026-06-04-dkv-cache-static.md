# dKV-Cache (static substrate, variant B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a StaticKVCache-substrate dKV-Cache config (`FAST_DLLM_DKV_CACHE=1 FAST_DLLM_DKV_STATIC=1`) — same delayed-cache algorithm as variant A, but built on the static buffer + `flash_kvcache_attention` path that makes static FOCUS near-lossless (0.825). This is the static-FOCUS-comparable config AND a candidate resolution to variant A's open accuracy problem.

**Architecture:** Full/refresh steps run a dense forward delegating to the model's real `self_attn` (writes all-layer KV to the StaticKVCache). Cached steps recompute only the fed tokens (masked ∪ just-decoded) through all layers, `write_sparse`-ing their K/V into the static buffer and attending via `flash_kvcache_attention` over the buffer (`cache_seqlens = scratch_seqlens`). One-step-delay shift + periodic refresh, reusing `_dkv_fed_indices` from variant A.

**Tech Stack:** PyTorch, `StaticKVCache`, `flash_kvcache_attention` (via `attn_backend`), the existing static-FOCUS machinery, pytest.

---

## Conventions (read first)
- Model file: `models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/modeling.py`. After ANY edit: `rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* ~/.cache/huggingface/modules/transformers_modules/*Efficient*`.
- Env: `export FV2=$PWD WORKSPACE=$PWD`; Python `v2/bin/python`. Pick a free GPU via `nvidia-smi` (avoid GPU 0 if shared).
- **No commits** (user's standing rule). Stage nothing.
- Variant A already added `_dkv_fed_indices` (LM class) — REUSE it; do not redefine.
- Templates to clone: `forward_focus_compact` (static model forward, modeling.py ~line 1114) and its LM wrapper (~1777); `batch_sample_focus` (generation_functions.py ~line 1299). Read each in FULL before editing.

---

## Task B1: `forward_dkv_static` (model forward + LM wrapper)

**Files:** Modify `modeling.py` — `Fast_dLLM_QwenModel` (add `forward_dkv_static`) and `Fast_dLLM_QwenForCausalLM` (add LM wrapper).

- [ ] **Step 1 — Read templates.** Read `forward_focus_compact` (model, ~1114–1276) fully: note `_dense_layer_nc` (delegates to `dl.self_attn(...)`), the dense-seed loop, and the deep-sparse block (the `attn.q_proj`/`write_sparse`/`get_full_kv`/`attn_backend.flash_kvcache_attention` pattern, ~1047–1067). Read the `forward_focus_compact` LM wrapper (~1777) for the exact return shape.

- [ ] **Step 2 — Add `forward_dkv_static` to `Fast_dLLM_QwenModel`** (insert after `forward_focus_compact`):
```python
    def forward_dkv_static(
        self,
        input_ids=None, past_key_values=None, use_cache: bool = True,
        cache_position=None, update_past_key_values: bool = False,
        fed_indices=None, is_full_step: bool = True, attn_backend=None, **kwargs,
    ):
        """dKV-Cache on the StaticKVCache substrate.

        is_full_step=True  -> dense forward over ALL block positions via the real
                              self_attn (writes all-layer KV into the static buffer);
                              near-lossless. Saves _dkv_static_seed_hidden (pre-norm).
        is_full_step=False -> cached step: recompute only `fed_indices` through all
                              layers; per layer write_sparse the fed K/V into the
                              static buffer and attend via flash_kvcache over the
                              buffer (cache_seqlens = scratch_seqlens). Scatter onto
                              the seed hidden.
        """
        hidden_states = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + hidden_states.shape[1], device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        attention_mask = None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cos_full, sin_full = position_embeddings
        cos_sin_full = torch.cat([cos_full, sin_full], dim=-1)
        B = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // num_heads

        def _dense_layer_nc(layer_idx, hs):
            dl = self.layers[layer_idx]
            residual = hs
            hsn = dl.input_layernorm(hs)
            attn_out = dl.self_attn(
                hsn, position_embeddings=position_embeddings,
                attention_mask=attention_mask, past_key_value=past_key_values,
                cache_position=cache_position, update_past_key_values=update_past_key_values,
            )
            hs = residual + attn_out
            residual = hs
            hsn = dl.post_attention_layernorm(hs)
            hs = residual + dl.mlp(hsn)
            return hs

        if is_full_step:
            for layer_idx in range(num_layers):
                hidden_states = _dense_layer_nc(layer_idx, hidden_states)
            self._dkv_static_seed_hidden = hidden_states.clone()
            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPastAndBlockCache(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            )

        # cached step: recompute only fed positions through ALL layers (sparse over static buffer)
        num_tokens = fed_indices.shape[1]
        past_len = past_key_values.get_seq_length()
        write_positions = fed_indices + past_len
        idx_h = fed_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        hs_sel = hidden_states.gather(1, idx_h)
        idx_pos = fed_indices.unsqueeze(-1).expand(-1, -1, cos_sin_full.shape[-1])
        sel_cos, sel_sin = cos_sin_full.expand(B, -1, -1).gather(1, idx_pos).chunk(2, dim=-1)

        for layer_idx in range(num_layers):
            dl = self.layers[layer_idx]
            attn = dl.self_attn
            residual = hs_sel
            sel_norm = dl.input_layernorm(hs_sel)
            q = attn.q_proj(sel_norm).view(B, num_tokens, num_heads, head_dim)
            k = attn.k_proj(sel_norm).view(B, num_tokens, num_kv_heads, head_dim)
            v = attn.v_proj(sel_norm).view(B, num_tokens, num_kv_heads, head_dim)
            q, k = apply_rotary_pos_emb(q, k, sel_cos, sel_sin, unsqueeze_dim=2)
            past_key_values.write_sparse(k, v, layer_idx, write_positions)
            full_k, full_v = past_key_values.get_full_kv(layer_idx)
            a = attn_backend.flash_kvcache_attention(
                q, full_k, full_v, cache_seqlens=past_key_values.scratch_seqlens,
                is_causal=False, scaling=attn.scaling,
            )
            a = a.reshape(B, num_tokens, -1).contiguous()
            a = attn.o_proj(a)
            hs_sel = residual + a
            hs_sel = hs_sel + dl.mlp(dl.post_attention_layernorm(hs_sel))

        seed_hidden = getattr(self, "_dkv_static_seed_hidden", None)
        base = seed_hidden if (seed_hidden is not None and seed_hidden.shape == hidden_states.shape) else hidden_states
        hidden_states = base.scatter(1, idx_h, hs_sel)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPastAndBlockCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
```
IMPORTANT: match `forward_focus_compact`'s exact q/k/v layout for the sparse path — the static path does NOT transpose to [B,H,S,D] (flash_kvcache wants [B,S,H,D]); the code above keeps that. If `forward_focus_compact`'s deep-sparse block differs in any detail (e.g., `.contiguous()`, scaling source), MATCH it.

- [ ] **Step 3 — Add the LM wrapper to `Fast_dLLM_QwenForCausalLM`**, mirroring the `forward_focus_compact` LM wrapper's return shape exactly:
```python
    def forward_dkv_static(
        self,
        input_ids=None, past_key_values=None, use_cache: bool = True,
        cache_position=None, update_past_key_values: bool = False,
        fed_indices=None, is_full_step: bool = True, attn_backend=None, **kwargs,
    ):
        outputs = self.model.forward_dkv_static(
            input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache,
            cache_position=cache_position, update_past_key_values=update_past_key_values,
            fed_indices=fed_indices, is_full_step=is_full_step, attn_backend=attn_backend,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPastAndBlockCache(
            loss=None, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
        )
```
(Verify `CausalLMOutputWithPastAndBlockCache` is what the existing `forward_focus_compact` wrapper returns; if it returns a different class, match THAT.)

- [ ] **Step 4 — Clear HF cache + syntax check.**
```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* ~/.cache/huggingface/modules/transformers_modules/*Efficient*
v2/bin/python -c "import ast; ast.parse(open('models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/modeling.py').read()); print('ok')"
```
Expect `ok`.

---

## Task B2: `batch_sample_dkv_static` (generation loop)

**Files:** Modify `generation_functions.py` — add method to `Fast_dLLM_QwenForCausalLM`.

This is the highest-risk task: the static loop carries StaticKVCache lifecycle, BatchBucket, attention-backend patching, prefill, and `scratch_seqlens`/`set_kv_write_start` management. Clone faithfully.

- [ ] **Step 1 — Copy the template.** Copy `batch_sample_focus` (generation_functions.py ~1299 to its end) verbatim into a new method `batch_sample_dkv_static` (same signature).

- [ ] **Step 2 — Apply these edits to the copy:**
  1. **Docstring** → dKV on static substrate; env `FAST_DLLM_DKV_REFRESH_STEPS` (default 4).
  2. **Env block** → delete the FOCUS reads (`focus_alpha`, `focus_layers`, `retain_override`, `_use_compact`, `_delayed_cache`, `debug_focus`). Add `refresh_steps = int(os.environ.get("FAST_DLLM_DKV_REFRESH_STEPS", "4"))`. KEEP all the StaticKVCache / max_seq_len / attn_backend / BatchBucket setup unchanged.
  3. **KEEP unchanged:** the `_static_kv_cache`/`_block_sparse_cache` init, `patch_attention_layers`, `BatchBucket`, prefill block, and every `static_cache.set_scratch_seqlens(...)` / `static_cache.set_kv_write_start(...)` call — the dKV path needs the same `scratch_seqlens = past_len + block_size` and write-start as FOCUS. Do NOT touch them.
  4. **Per-block status init** (where FOCUS created `frozen`/`avg_decoded`): add, sized to the CURRENT batch:
     ```python
     prv_transfer_idx = torch.zeros((x_t.shape[0], block_size), dtype=torch.bool, device=self.device)
     cur_transfer_index = torch.zeros((x_t.shape[0], block_size), dtype=torch.bool, device=self.device)
     step_in_block = 0
     ```
     (Use whatever local holds the current block tensor in the template in place of `x_t`.)
  5. **Inner forward call** (replace the `is_dense`/`forward_focus_compact(...)` call):
     ```python
     is_full = (step_in_block <= 1) or (step_in_block % refresh_steps == 0)
     if is_full:
         output = self.forward_dkv_static(
             input_ids=<block_input>, use_cache=True, past_key_values=static_cache,
             update_past_key_values=True, is_full_step=True, attn_backend=attn_backend,
         )
     else:
         fed_indices, _ = self._dkv_fed_indices(prv_transfer_idx)
         output = self.forward_dkv_static(
             input_ids=<block_input>, use_cache=True, past_key_values=static_cache,
             update_past_key_values=False, fed_indices=fed_indices, is_full_step=False,
             attn_backend=attn_backend,
         )
     step_in_block += 1
     ```
     (`<block_input>` = the same block slice the template passes to `forward_focus_compact`, e.g. `x_t[:, -block_size:]` or the bucketed equivalent — match the template exactly. NOTE: the full step uses `update_past_key_values=True` so the dense KV is written to the static buffer; the cached step uses `update_past_key_values=False` since it writes via `write_sparse` itself.)
  6. **Decode block** → keep the template's logit-shift / `sample_with_top_p` / unmask / finished handling UNCHANGED; delete any `avg_decoded`/`_ad_count` lines.
  7. **Shift bookkeeping** (replace the FOCUS `_focus_update_frozen` line):
     ```python
     all_decoded = (<block_slice> != mask_id)
     prv_transfer_idx, cur_transfer_index = cur_transfer_index, all_decoded
     ```
     (`<block_slice>` = the current block view, matching what `mask_idx` is computed from.)
  8. **KEEP unchanged:** block-commit branch, BatchBucket finished/compaction logic, the `try/finally` and return. The per-block status tensors are recreated each block, so no extra compaction is needed. BUT: if the template compacts batch mid-run via BatchBucket and the status tensors would desync, recreate `prv_transfer_idx`/`cur_transfer_index` at the new batch size at block start (they already are — they're per-block).

- [ ] **Step 3 — Syntax check.** `v2/bin/python -c "import ast; ast.parse(open('generation_functions.py').read()); print('ok')"` → `ok`.

- [ ] **Step 4 — Grep sanity.** Confirm `batch_sample_dkv_static` references `forward_dkv_static`, `_dkv_fed_indices`, `refresh_steps`, `prv_transfer_idx`, and has NO leftover `focus_alpha`/`frozen`/`avg_decoded`.

---

## Task B3: `eval.py` dispatch

**Files:** Modify `eval.py` (the `FAST_DLLM_DKV_CACHE` branch added in variant A, ~line 97).

- [ ] **Step 1 — Branch on the static flag.** Replace the existing DKV branch body so it picks the static method when `FAST_DLLM_DKV_STATIC=1`:
```python
        elif os.environ.get("FAST_DLLM_DKV_CACHE", "0") == "1":
            if os.environ.get("FAST_DLLM_DKV_STATIC", "0") == "1":
                print("Using dKV-Cache diffusion mode (delayed KV, StaticKVCache path)", flush=True)
                self.model.mdm_sample = types.MethodType(
                    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_dkv_static, self.model
                )
            else:
                print("Using dKV-Cache diffusion mode (delayed KV, no eviction, DynamicCache path)", flush=True)
                self.model.mdm_sample = types.MethodType(
                    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_dkv, self.model
                )
```

- [ ] **Step 2 — Syntax check.** `v2/bin/python -c "import ast; ast.parse(open('eval.py').read()); print('ok')"` → `ok`.

---

## Task B4: Validation (GPU anchor + e2e)

**Files:** results → append to `logs/dkv_cache.md`.

- [ ] **Step 1 — b1/50 smoke.** Clear HF cache, then:
```bash
export WORKSPACE=$PWD HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true PYTHONUNBUFFERED=1
export FAST_DLLM_EXECUTION_MODE=eager FAST_DLLM_MAX_SEQ_LEN=1024
export FAST_DLLM_DKV_CACHE=1 FAST_DLLM_DKV_STATIC=1 FAST_DLLM_DKV_REFRESH_STEPS=4
mp=$WORKSPACE/models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/
CUDA_VISIBLE_DEVICES=<free> v2/bin/accelerate launch eval.py --tasks gsm8k --batch_size 1 --num_fewshot 0 --limit 50 \
  --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
  --model_args "model_path=${mp},threshold=0.9,show_speed=True,use_block_cache=False" &> logs/dkv_static_smoke_b1.log
```
Expect coherent output, no crash.

- [ ] **Step 2 — b16/1000 refresh sweep {1, 2, 4, 8}** → `logs/dkv_static_b16_r{N}.log`. **The `refresh_steps=1` anchor is the key result: it should land NEAR the dense baseline 0.826** (every step is the real dense forward via `self_attn`). If it does, that confirms the dense-delegation path is near-lossless and pinpoints variant A's open problem to its hand-rolled full step. If `refresh=1` is still ~0.77, the gap is elsewhere — record it as a finding, do not paper over it.

- [ ] **Step 3 — Record results** in `logs/dkv_cache.md`: a static-substrate table (accuracy + TPS per refresh) next to the dynamic one, and an explicit note on whether `refresh=1` recovered dense accuracy (the A-vs-B diagnostic).

---

## Self-Review
- **Spec coverage:** static substrate (spec §4 Variant B) → B1,B2; reuse `_dkv_fed_indices` → B2; refresh + one-step delay → B2; env gate `FAST_DLLM_DKV_STATIC` → B3; refresh=1≈dense anchor + sweep → B4. Covered.
- **Placeholders:** `<block_input>`/`<block_slice>` are explicitly defined as "match the template's slice" with concrete examples — they depend on the exact local in `batch_sample_focus`, which the implementer reads in Step 1; this is a deliberate template-reference, not a vague placeholder.
- **Type consistency:** `fed_indices` `[B, num_fed]` long (from `_dkv_fed_indices`, consumed by `forward_dkv_static`). `is_full_step` bool, `attn_backend` threaded from the static loop into `forward_dkv_static`. `_dkv_static_seed_hidden` set/read only inside `forward_dkv_static`. Wrapper return class matches `forward_focus_compact`.
- **Risk:** B2 is the riskiest (faithful clone of the complex static loop). If the `write_sparse`/`scratch_seqlens` wiring is off, the b1 smoke will be incoherent — debug there before the sweep.

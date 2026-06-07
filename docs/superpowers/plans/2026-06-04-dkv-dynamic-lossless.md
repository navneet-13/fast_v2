# Near-lossless dynamic dKV-Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Rebuild the dynamic dKV forward so its full step runs the **real** `decoder_layer`/`self_attn` (lossless) instead of hand-rolled attention, with delayed caching layered on via a per-block KV buffer — recovering `refresh=1` from 0.771 toward dense 0.826 while keeping DynamicCache speed/no-OOM.

**Architecture:** Add an optional `dkv_store` to `self_attn` that scatter-writes this step's post-RoPE K/V into a per-block buffer and then attends over `cat(prefix, full_buffer)` via the model's *existing* SDPA path. Rewrite `forward_dkv` as a near-copy of `Fast_dLLM_QwenModel.forward` that calls the real `decoder_layer` loop and threads the buffer. Rewrite `batch_sample_dkv` to own the buffer (reset per block).

**Tech Stack:** PyTorch, HF transformers DynamicCache, the existing `Fast_dLLM_QwenAttention` SDPA path, `DynamicBlockKV`, pytest.

---

## Conventions (read first)

- **Model file:** `models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/modeling.py`. After ANY edit: `rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* ~/.cache/huggingface/modules/transformers_modules/*Efficient*`.
- **Env:** `export FV2=$PWD WORKSPACE=$PWD`; Python `v2/bin/python`. Pick a free GPU via `nvidia-smi`.
- **No commits** (user's standing rule). Stage nothing.
- **Reuse, don't redefine:** `_dkv_fed_indices` (LM class) and `DynamicBlockKV(deep_layer_start=0)` already exist — reuse them.
- **Key existing code:**
  - `Fast_dLLM_QwenAttention.forward` (~line 327): after RoPE (line 360) come the `block_past_key_values` branch (362–373) and the `past_key_value` branch (375–382, the `cat(prefix, current)` we rely on), then SDPA (388–401).
  - `Fast_dLLM_QwenModel.forward` (~line 592): the original forward we mirror (embed → cache_position → `eval_mask` → rotary → `decoder_layer` loop → norm).
  - `decoder_layer.forward` (~line 440) forwards `**kwargs` to `self_attn` (line 470) — so `dkv_store`/`dkv_positions` thread through automatically.
  - Current hand-rolled `forward_dkv` (model) at line 1501; its LM wrapper at 1929. `batch_sample_dkv` in `generation_functions.py`.
  - `DynamicBlockKV.write(layer_idx, k, v, token_indices)` scatters `[B,H_kv,n,D]` into `[B,H_kv,block_size,D]` on dim=2; `.get(layer_idx)` returns the full buffer; `.write_full(L,k,v)`; `.reset()`.

---

## Task 1: Add `dkv_store` to `self_attn`

**Files:** Modify `modeling.py` `Fast_dLLM_QwenAttention.forward` (~327). Test: `tests/test_dkv_lossless.py` (new).

- [ ] **Step 1: Write the failing test**

Create `tests/test_dkv_lossless.py` with the `_load_modeling()` header copied verbatim from `tests/test_focus.py` lines 1–30, then add (after `FV2 = os.environ["FV2"]`, add `import sys; sys.path.insert(0, FV2)` if needed):

```python
@pytest.mark.gpu
def test_self_attn_dkv_store_uses_full_buffer():
    import torch
    from transformers import AutoModelForCausalLM
    from utils.dynamic_block_kv import DynamicBlockKV
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    model = AutoModelForCausalLM.from_pretrained(
        os.path.dirname(MODELING), trust_remote_code=True,
        torch_dtype=torch.bfloat16, cache_dir=os.path.join(FV2, "models"),
    ).to("cuda").eval()
    inner = model.model
    cfg = model.config
    block = 8
    H = cfg.hidden_size
    hs = torch.randn(1, block, H, device="cuda", dtype=torch.bfloat16)
    pos_ids = torch.arange(block, device="cuda").unsqueeze(0)
    pe = inner.rotary_emb(hs, pos_ids)
    layer0 = inner.layers[0].self_attn
    buf = DynamicBlockKV(deep_layer_start=0, num_layers=cfg.num_hidden_layers, batch_size=1,
                         num_kv_heads=cfg.num_key_value_heads, block_size=block,
                         head_dim=cfg.hidden_size // cfg.num_attention_heads,
                         dtype=torch.bfloat16, device="cuda")
    # (a) dkv_store=None must equal the normal call (dense path unchanged)
    with torch.no_grad():
        out_plain = layer0(hidden_states=hs, position_embeddings=pe, attention_mask=None,
                           past_key_value=None, cache_position=pos_ids[0])
        all_pos = torch.arange(block, device="cuda").unsqueeze(0)
        out_dkv = layer0(hidden_states=hs, position_embeddings=pe, attention_mask=None,
                         past_key_value=None, cache_position=pos_ids[0],
                         dkv_store=buf, dkv_positions=all_pos)
    # full-block write to an empty buffer ⇒ identical attention ⇒ identical output
    assert torch.allclose(out_plain.float(), out_dkv.float(), atol=1e-2)
    # buffer is now populated for layer 0
    k0, v0 = buf.get(0)
    assert k0.abs().sum() > 0
```

- [ ] **Step 2: Run, verify FAIL**

```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* ~/.cache/huggingface/modules/transformers_modules/*Efficient*
export FV2=$PWD WORKSPACE=$PWD
CUDA_VISIBLE_DEVICES=<free> v2/bin/python -m pytest tests/test_dkv_lossless.py -k dkv_store_uses_full_buffer -v
```
Expected: FAIL — `self_attn.forward` doesn't accept `dkv_store`.

- [ ] **Step 3: Implement**

In `Fast_dLLM_QwenAttention.forward` (~327): add two params to the signature (before `**kwargs`):
```python
        dkv_store=None,
        dkv_positions=None,
```
Then insert this block **immediately after the rotary step** (right after the `else: query_states, key_states = apply_rotary_pos_emb(...)` at ~line 360, BEFORE the `if block_past_key_values is not None:` at ~362):
```python
        if dkv_store is not None:
            # dKV-Cache: persist this step's post-RoPE K/V into the per-block buffer at
            # dkv_positions, then attend over the WHOLE buffer (settled frozen slots +
            # this step's fresh slots). The past_key_value branch below then prepends the
            # committed prefix, so attention runs over cat(prefix, full_block_buffer).
            dkv_store.write(self.layer_idx, key_states, value_states, dkv_positions)
            key_states, value_states = dkv_store.get(self.layer_idx)
```
Do NOT change the `block_past_key_values`, `past_key_value`, or SDPA blocks. `dkv_store`/`dkv_positions` are captured as named params so they never leak into the `attention_interface(**kwargs)` call.

- [ ] **Step 4: Run, verify PASS**

```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* ~/.cache/huggingface/modules/transformers_modules/*Efficient*
CUDA_VISIBLE_DEVICES=<free> v2/bin/python -m pytest tests/test_dkv_lossless.py -k dkv_store_uses_full_buffer -v
```
Expected: PASS.

---

## Task 2: Rewrite `forward_dkv` to mirror the real `forward`

**Files:** Modify `modeling.py` — `Fast_dLLM_QwenModel` (rename old `forward_dkv`→`forward_dkv_handrolled`, add new `forward_dkv`).

- [ ] **Step 1: Preserve the old method**

Rename the existing `Fast_dLLM_QwenModel.forward_dkv` (line ~1501) to `forward_dkv_handrolled` (keeps the 0.771 baseline reproducible; it stays undispatched). Do the same for the LM wrapper at ~1929 → `forward_dkv_handrolled`, and make it call `self.model.forward_dkv_handrolled`. Leave their bodies otherwise unchanged.

- [ ] **Step 2: Add the new `forward_dkv` to `Fast_dLLM_QwenModel`**

It mirrors `Fast_dLLM_QwenModel.forward` (read ~592–675 first) but threads the buffer and supports a fed subset. Add after `forward_dkv_handrolled`:
```python
    def forward_dkv(
        self, input_ids=None, past_key_values=None, use_cache: bool = True,
        cache_position=None, dkv_store=None, fed_indices=None,
        is_full_step: bool = True, **kwargs,
    ):
        """dKV-Cache forward that reuses the real decoder_layer/self_attn (lossless).
        is_full_step=True: run the WHOLE block (writes all buffer slots) — ≡ dense.
        is_full_step=False: run only fed_indices through the real layers, attending over
        cat(prefix, dkv buffer); scatter outputs onto the saved seed hidden."""
        full_embed = self.embed_tokens(input_ids)            # [B, block, H]
        B = full_embed.shape[0]
        block = full_embed.shape[1]
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        block_pos = torch.arange(past_len, past_len + block, device=full_embed.device)

        if is_full_step:
            hidden_states = full_embed
            pos_ids = block_pos.unsqueeze(0)
            pe = self.rotary_emb(hidden_states, pos_ids)
            all_pos = torch.arange(block, device=full_embed.device).unsqueeze(0).expand(B, -1)
            for layer in self.layers[: self.config.num_hidden_layers]:
                hidden_states = layer(
                    hidden_states, attention_mask=None, position_ids=pos_ids,
                    past_key_value=past_key_values, use_cache=use_cache,
                    cache_position=block_pos, position_embeddings=pe,
                    update_past_key_values=False, dkv_store=dkv_store, dkv_positions=all_pos,
                )
            self._dkv_seed_hidden = hidden_states.clone()
            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPastAndBlockCache(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            )

        # cached step: gather fed tokens, run them through the real layers
        num_fed = fed_indices.shape[1]
        idx_h = fed_indices.unsqueeze(-1).expand(-1, -1, full_embed.shape[-1])
        hidden_states = full_embed.gather(1, idx_h)                       # [B, num_fed, H]
        fed_abs_pos = (fed_indices + past_len)                            # [B, num_fed]
        # per-row positions → use the batch-shared row 0 for rotary (positions identical across rows here)
        pos_ids = fed_abs_pos
        pe = self.rotary_emb(hidden_states, pos_ids)
        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states, attention_mask=None, position_ids=pos_ids,
                past_key_value=past_key_values, use_cache=use_cache,
                cache_position=fed_abs_pos[0], position_embeddings=pe,
                update_past_key_values=False, dkv_store=dkv_store, dkv_positions=fed_indices,
            )
        seed = getattr(self, "_dkv_seed_hidden", None)
        base = seed if (seed is not None and seed.shape[:2] == full_embed.shape[:2]) else full_embed
        hidden_states = base.scatter(1, idx_h, hidden_states)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPastAndBlockCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
```
Notes for the implementer:
- `attention_mask=None` is valid for single-block diffusion (the existing `forward_dynamo` documents this: the block-causal mask is trivially true since all queries are in the current block and all KVs are in blocks ≤ current).
- The buffer holds **post-RoPE** K (and plain V); frozen slots keep their original positional encoding — correct across steps.
- The full step's `dkv_store.write(all_pos)` then `get()` makes `key_states` == the freshly-projected block, so `cat(prefix, buffer)` is **identical** to the original forward's `cat(prefix, block)` → bit-lossless.

- [ ] **Step 3: Update the LM wrapper**

Add a new `forward_dkv` wrapper to `Fast_dLLM_QwenForCausalLM` mirroring the existing one's return shape, passing `dkv_store`, `fed_indices`, `is_full_step` through to `self.model.forward_dkv`:
```python
    def forward_dkv(self, input_ids=None, past_key_values=None, use_cache: bool = True,
                    cache_position=None, dkv_store=None, fed_indices=None,
                    is_full_step: bool = True, **kwargs):
        outputs = self.model.forward_dkv(
            input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache,
            cache_position=cache_position, dkv_store=dkv_store, fed_indices=fed_indices,
            is_full_step=is_full_step)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPastAndBlockCache(
            loss=None, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=hidden_states)
```
(Verify class/fields match the existing `forward_dkv_handrolled` wrapper; mirror exactly.)

- [ ] **Step 4: Clear cache + syntax check**

```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* ~/.cache/huggingface/modules/transformers_modules/*Efficient*
v2/bin/python -c "import ast; ast.parse(open('models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/modeling.py').read()); print('ok')"
```
Expected: `ok`.

---

## Task 3: Update `batch_sample_dkv` to own the buffer

**Files:** Modify `generation_functions.py` `batch_sample_dkv`.

- [ ] **Step 1: Adjust the per-block setup + forward calls**

The method already: creates a `DynamicBlockKV(deep_layer_start=0, …)` per block, tracks `prv_transfer_idx`/`cur_transfer_index`/`step_in_block`, and the one-step-delay shift + refresh. Make these edits:
1. Keep the per-block `block_kv = DynamicBlockKV(deep_layer_start=0, …)` (this is the buffer; recreated per block = the "reset at block boundary").
2. Replace the full-step call with:
   ```python
   output = self.forward_dkv(
       input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values,
       dkv_store=block_kv, is_full_step=True)
   ```
3. Replace the cached-step call with:
   ```python
   fed_indices, _ = self._dkv_fed_indices(prv_transfer_idx)
   output = self.forward_dkv(
       input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values,
       dkv_store=block_kv, fed_indices=fed_indices, is_full_step=False)
   ```
   (Drop the old `block_kv=`/`forward_dkv(..., is_full_step=…)` args that referenced the hand-rolled signature; the new signature uses `dkv_store=`.)
4. Keep everything else (decode, shift, refresh, finished-sample compaction) unchanged.

- [ ] **Step 2: Syntax check**

```bash
v2/bin/python -c "import ast; ast.parse(open('generation_functions.py').read()); print('ok')"
```
Expected: `ok`.

---

## Task 4: Validate (diagnostic + no-op + e2e)

**Files:** rerun `tests/diag_dkv_fullstep.py`, `tests/test_dkv.py`; record `logs/dkv_cache.md`.

- [ ] **Step 1: Diagnostic — full step now equals the real forward**

```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/*Fast_dLLM* ~/.cache/huggingface/modules/transformers_modules/*Efficient*
export FV2=$PWD WORKSPACE=$PWD
CUDA_VISIBLE_DEVICES=<free> v2/bin/python tests/diag_dkv_fullstep.py
```
Expected: **argmax_agreement = 1.0000** (was 0.9688), max_abs_hidden_diff ≪ 8.0. If not 1.0, the full step still diverges — debug `forward_dkv`'s full branch before proceeding (it must be bit-identical to the original forward).

- [ ] **Step 2: No-op + existing dKV tests still pass**

```bash
CUDA_VISIBLE_DEVICES=<free> v2/bin/python -m pytest tests/test_dkv.py tests/test_dkv_lossless.py -v
```
Expected: all pass (incl. `test_forward_dkv_full_equals_fed_all`).

- [ ] **Step 3: b1/50 smoke**

```bash
export WORKSPACE=$PWD HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true PYTHONUNBUFFERED=1
export FAST_DLLM_EXECUTION_MODE=eager FAST_DLLM_MAX_SEQ_LEN=1024 FAST_DLLM_DKV_CACHE=1 FAST_DLLM_DKV_REFRESH_STEPS=4
mp=$WORKSPACE/models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/
CUDA_VISIBLE_DEVICES=<free> v2/bin/accelerate launch eval.py --tasks gsm8k --batch_size 1 --num_fewshot 0 --limit 50 \
  --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
  --model_args "model_path=${mp},threshold=0.9,show_speed=True,use_block_cache=False" &> logs/dkv_lossless_smoke_b1.log
```
Expected: coherent, no crash.

- [ ] **Step 4: b16/1000 refresh sweep {1,2,4,8} + record**

Run b16 at `FAST_DLLM_DKV_REFRESH_STEPS` ∈ {1,2,4,8} → `logs/dkv_lossless_b16_r{N}.log`. **Headline:** `refresh=1` must land **≈0.826** (was 0.771). Append a "lossless dynamic dKV" table (accuracy + TPS) to `logs/dkv_cache.md` next to the old dynamic + static tables, and note whether near-lossless was achieved while keeping TPS above dense.

---

## Self-Review

**Spec coverage:** §3.1 buffer → Task 3 (per-block `DynamicBlockKV`); §3.2 `self_attn` addition → Task 1; §3.3 `forward_dkv` rewrite → Task 2; §3.4 `batch_sample_dkv` → Task 3; §5 file table → all tasks; §6 tests (diagnostic→1.0, no-op, refresh=1==dense, sweep, no-regression) → Task 4 + Task 1's dense-unchanged assertion; old method retained → Task 2 Step 1. Covered.

**Placeholder scan:** no TBD/TODO; all code blocks complete; `<free>` is a literal GPU-index placeholder the operator fills (a runtime value, not missing code).

**Type consistency:** `dkv_store` is a `DynamicBlockKV` everywhere; `dkv_positions`/`fed_indices` are `[B,n]` long (consumed by `DynamicBlockKV.write` dim=2 scatter, matching its existing API); `is_full_step` bool across model fn, wrapper, and caller; `_dkv_seed_hidden` set/read only inside `forward_dkv`. The new `self_attn` params default `None` so all other callers (dense, FOCUS, static) are unaffected.

**Risk flagged in spec:** per-row position handling — here all batch rows share the same block positions, so `pos_ids = fed_abs_pos` (row-shared) is valid; if that ever changes, rotary must be gathered per-row.

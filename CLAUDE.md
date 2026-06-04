# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Fast-dLLM v2** is a block-diffusion language model that adapts pretrained autoregressive models (Qwen2.5-7B-Instruct) into diffusion-based LLMs for parallel text generation. Key claims: 500x less training data than full-attention diffusion LLMs, 2.54x higher throughput than Qwen2.5-7B-Instruct. The model uses a custom mask token (ID: 151665) and stop token (ID: 151645).

## Commands

**Install:**
```bash
pip install -e .
# With optional extras:
pip install -e ".[gradio,flash_attn,vllm]"
```

**Web UI:**
```bash
python app.py  # Starts Gradio interface at http://localhost:10086
```

**CLI Chat:**
```bash
python run_chatbot.py
```

**Evaluation:**
```bash
bash eval_script.sh  # Runs MMLU, GPQA, GSM8K, Minerva Math, IFEval
# Or directly:
python eval.py --model fast_dllm_v2 --tasks mmlu --device cuda
```

**Training:**
```bash
bash train_scripts/finetune_alpaca.sh
# Key params: lr=2e-5, block_size=512, bf16, DeepSpeed ZeRO-2
```

**Linting:**
```bash
ruff check .
ruff format .
```

## Architecture

### Core Generation (`generation_functions.py`)

The block diffusion sampling is the central innovation. Key concepts:
- **Block-wise causal attention**: tokens attend to all clean tokens from previous blocks, plus noisy tokens in the current block
- **Token shift**: masked token predictions use the logit of the preceding token
- **Complementary masks**: alternating masking patterns ensure all positions are learned
- **KV caching**: two levels — block-level (cross-block context) and sub-block (within-block parallelism)

Main function: `mdm_sample_with_visualization()` — returns both generated tokens and per-step visualization states.

### Entry Points

| Interface | File | Calls into |
|-----------|------|-----------|
| Web UI | `app.py` | `generation_functions.mdm_sample_with_visualization()` |
| CLI | `run_chatbot.py` | `model.generate()` with `block_size`, `threshold` params |
| Eval | `eval.py` | `Fast_dLLM_v2EvalHarness` (registered as `"fast_dllm_v2"`) |
| Training | `train_scripts/finetune.py` | LMFlow `AutoPipeline → Finetuner → PeftTrainer` |

### LMFlow Framework (`src/lmflow/`)

The training/eval infrastructure lives here. Key sub-packages:
- `models/` — `HFDecoderModel`, `HFEncoderDecoderModel`, `AutoModel`
- `pipeline/` — `Finetuner`, `PeftTrainer`, `BaseTuner`
- `datasets/` — Dataset loading with block-diffusion-specific preprocessing (mask_id, bd_size)
- `args.py` — All configuration dataclasses (ModelArguments, DatasetArguments, etc.)

### DeepSpeed Configs (`configs/`)

Six configs covering ZeRO stages 0/2/3, each with and without CPU offload. Default training uses `ds_config_zero2_no_offload.json`.

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `block_size` / `bd_size` | Tokens generated per diffusion block | 512 |
| `threshold` | Confidence threshold for token unmasking | model-dependent |
| `mask_id` | Mask token ID | 151665 |
| `stop_token_id` | EOS token ID | 151645 |

## Dependencies

Pinned versions matter: `transformers==4.53.1`, `datasets==2.14.6`, `pyarrow==18.0.0`, `trl==0.8.0`. Other packages have `>=` constraints. See `requirements.txt` for full list.

---

## Flash Attention Setup

### Hardware (current server)

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX A5000 |
| Compute Capability | SM 8.6 (Ampere) |
| PyTorch | 2.10.0+cu128 |
| CUDA runtime | 12.8 |
| CUDA toolkit (`/usr/local/cuda-12.6/bin/nvcc`) | 12.6 |
| System nvcc (`/usr/bin/nvcc`) | **11.5 — DO NOT USE for building** |
| Python | 3.11 |

### Which FA version to install

| Version | Min GPU | Package | Notes |
|---------|---------|---------|-------|
| **FA2** | SM 8.0+ | `flash-attn` | **Use this on RTX A5000** |
| FA3 | SM 8.0+ | `flash_attn_3` (built from `hopper/` subdir) | Primarily Hopper-tuned; compiles SM 8.0 |
| FA4 | SM 9.0+ | `flash-attn-4` (separate repo/package) | Hopper / Blackwell only |
| **FlashInfer** | SM 7.5+ | `flashinfer-python` | Turing+ (T4, A5000, H100, etc.) — alternative to FA2 |

### Install Flash Attention 2 (FA2)

No prebuilt wheel matches torch 2.10 — must build from source. The system nvcc at `/usr/bin/nvcc` is only 11.5 and will fail; point to `/usr/local/cuda-12.6`:

```bash
CUDA_HOME=/usr/local/cuda-12.6 \
MAX_JOBS=4 \
/research/data/transfer/data/navneet/fast_v2/v2/bin/python3 -m pip install flash-attn==2.8.3 \
  --no-build-isolation
```

Build takes ~20–30 minutes. Verify after install:

```bash
python3 -c "
import flash_attn, torch
print(flash_attn.__version__)
from flash_attn import flash_attn_func, flash_attn_with_kvcache

B, S, H, D = 2, 16, 8, 64
q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
k = torch.randn(B, S, 2, D, dtype=torch.bfloat16, device='cuda')   # GQA: 2 kv heads
v = torch.randn(B, S, 2, D, dtype=torch.bfloat16, device='cuda')
print(flash_attn_func(q, k, v, causal=False).shape)                  # (2, 16, 8, 64)

k_cache = torch.zeros(B, 64, 2, D, dtype=torch.bfloat16, device='cuda')
v_cache = torch.zeros(B, 64, 2, D, dtype=torch.bfloat16, device='cuda')
seqlens = torch.tensor([16, 16], dtype=torch.int32, device='cuda')
print(flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=seqlens).shape)  # (2, 16, 8, 64)
"
```

### Flash Attention API Reference

All flash_attn functions use **BSHD layout** (batch, seq, heads, dim) — opposite of PyTorch's BHSD. The `_flash_attention()` method in `attention_backends.py` transposes before and after calling.

**`flash_attn_func` (FA2 and FA3):**
```python
# FA2
flash_attn_func(q, k, v,
    dropout_p=0.0,          # FA2 only — FA3 has NO dropout_p parameter
    softmax_scale=None,     # default: 1/sqrt(D)
    causal=False,
    window_size=(-1, -1),
    ...)
# inputs/output: (B, S, H, D)  — BSHD
```

**`flash_attn_with_kvcache` (FA2 and FA4):**
```python
flash_attn_with_kvcache(q, k_cache, v_cache,
    k=None, v=None,            # new KV to write into cache (optional)
    cache_seqlens=None,        # (B,) torch.int32 — valid length per batch element
    softmax_scale=None,
    causal=False,
    ...)
# All tensors: (B, S, H, D) — BSHD
# cache_seqlens dtype must be torch.int32 (not int64)
```

### Package naming — critical distinctions

| Name in code | Actual Python package | Install command |
|---|---|---|
| FA2 | `flash_attn` | `pip install flash-attn` |
| FA3 | `flash_attn_3` | Build from `hopper/` subdir of the main repo |
| FA4 | `flash_attn_4` | `pip install flash-attn-4` (separate repo) |

**`flash_attn.flash_attn_interface` is FA2's internal module, NOT FA3.** Importing from it on a system with FA2 installed returns an FA2 function — would silently mislabel it as FA3.

---

### Backend Correctness Analysis (`utils/attention_backends.py`)

Cross-checked the original `attention_backends.py` implementation against the [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) README, `setup.py`, and API source. Found and fixed **5 bugs** across FA2, FA3, and FA4 detection and call paths.

#### Analysis steps

1. **Read the GitHub README** — confirmed FA version-to-package mapping: FA2 = `flash_attn` (main repo), FA3 = built from `hopper/` subdir (installs as `flash_attn_3`), FA4 = completely separate `flash-attn-4` repo/package. This immediately showed the original code had the wrong package assumptions for FA3 and FA4.

2. **Inspected `setup.py` for SM guards** — FA2 `setup.py` requires SM 7.5+; FA3 `setup.py` (in `hopper/`) compiles SM 8.0 kernels (`sources_fwd_sm80`, `sources_bwd_sm80`) so it runs on Ampere+, not Hopper-only.

3. **Inspected FA2 and FA3 function signatures**:
   - FA2 `flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, ...)`
   - FA3 `flash_attn_func(q, k, v, softmax_scale=None, causal=False, ...)` — **no `dropout_p`**
   - `flash_attn_with_kvcache(q, k_cache, v_cache, ..., cache_seqlens=None, ...)` — available in FA2 and FA4 only; requires `cache_seqlens` dtype = `torch.int32`

4. **Traced every import path** in the original code to verify what package was actually being imported on a system with FA2 installed.

5. **Ran a verification script** (`python3 -c "import ast; ..."`) to confirm syntax correctness after the rewrite.

#### Bugs found and fixed

**Bug A — FA4 imported from wrong package (`_try_import_flash_attn_4`)**

- **Original code**: imported `flash_attn_with_kvcache` from the main `flash_attn` package and checked `version >= (2, 7)`. This was wrong — FA4 is a completely separate package.
- **Consequence**: On a system with FA2 ≥ 2.7 installed on Hopper hardware, the function would return an FA2 function mislabelled as FLASH4, silently using FA2 semantics under the FA4 code path.
- **Fix**: Changed import to `from flash_attn_4 import flash_attn_with_kvcache`. SM guard stays at `cc[0] < 9` (FA4 is Hopper/Blackwell only).

```python
# Before (wrong)
from flash_attn import flash_attn_with_kvcache  # this is FA2, not FA4
if version >= (2, 7): ...

# After (correct)
from flash_attn_4 import flash_attn_with_kvcache  # separate FA4 package
```

**Bug B — FA3 SM guard too restrictive (`_try_import_flash_attn_3`)**

- **Original code**: guard was `if cc[0] < 9: return None`, which blocked FA3 on all SM 8.x hardware (Ampere, including RTX A5000 at SM 8.6).
- **Consequence**: FA3 would never be loaded on A100/A5000/A6000 even if installed, silently falling through to FA2 or SDPA.
- **Root cause**: FA3 was mistakenly assumed to be Hopper-only. FA3's `setup.py` explicitly compiles `sources_fwd_sm80` kernels for Ampere+.
- **Fix**: Changed guard to `if cc[0] < 8: return None`.

```python
# Before (wrong — blocked FA3 on SM 8.x)
if cc[0] < 9:
    return None

# After (correct — FA3 supports SM 8.0+)
if cc[0] < 8:
    return None
```

**Bug C — FA3 imported from FA2's internal module (`_try_import_flash_attn_3`)**

- **Original code**: `from flash_attn.flash_attn_interface import flash_attn_func` — this imports FA2's internal submodule, not FA3.
- **Consequence**: On any system with FA2 installed, this import succeeds silently and returns an FA2 function. The code would believe it selected FA3 while actually running FA2 (with different behaviour — notably FA3's `dropout_p`-free path would never trigger, but the FA2 function it returned *does* accept `dropout_p`; no crash, wrong version).
- **Fix**: Changed to `from flash_attn_3 import flash_attn_func`, which correctly targets the separate FA3 package.

```python
# Before (wrong — imports FA2 under FA3 label)
from flash_attn.flash_attn_interface import flash_attn_func as fa3_func

# After (correct)
from flash_attn_3 import flash_attn_func as fa3_func
```

**Bug D — FA3 fallback imported non-existent package (`_try_import_flash_attn_3`)**

- **Original code**: had a secondary `except` block that tried `from flash_attn_hopper import flash_attn_func`.
- **Consequence**: `flash_attn_hopper` is not a real package name. The `hopper/` build directory of the flash-attention repo installs as `flash_attn_3`, not `flash_attn_hopper`. This fallback would always raise `ImportError` and was dead code.
- **Fix**: Removed the `flash_attn_hopper` fallback entirely.

**Bug E — FA3 call passed `dropout_p` (`_flash_attention` method)**

- **Original code**: `_flash_attention()` called `func(q_fa, k_fa, v_fa, dropout_p=dropout, softmax_scale=scale, causal=is_causal)` for all flash backends including FA3.
- **Consequence**: FA3's `flash_attn_func` has no `dropout_p` parameter. This would raise `TypeError` at the first FA3 forward call at runtime.
- **Fix**: Added an explicit FA3 branch that omits `dropout_p`:

```python
# Before (would crash at runtime for FA3)
out = func(q_fa, k_fa, v_fa, dropout_p=dropout, softmax_scale=scale, causal=is_causal)

# After
if self.name == self.FLASH3:
    out = func(q_fa, k_fa, v_fa, softmax_scale=scale, causal=is_causal)
else:
    out = func(q_fa, k_fa, v_fa, dropout_p=dropout, softmax_scale=scale, causal=is_causal)
```

#### Post-fix verification check

After the rewrite, a syntax + logic verification script was run:

```bash
python3 -c "
import ast, inspect
with open('utils/attention_backends.py') as f:
    src = f.read()
ast.parse(src)  # syntax check

# Spot-check imports in each try function
import importlib.util, sys
spec = importlib.util.spec_from_file_location('ab', 'utils/attention_backends.py')
mod = importlib.util.load_from_spec(spec); spec.loader.exec_module(mod)

fa4_fn = inspect.getsource(mod._try_import_flash_attn_4)
assert 'from flash_attn_4' in fa4_fn, 'FA4 still uses wrong package'
fa3_fn = inspect.getsource(mod._try_import_flash_attn_3)
assert 'from flash_attn_3' in fa3_fn, 'FA3 still uses wrong package'
assert 'flash_attn_hopper' not in fa3_fn, 'Dead fallback still present'
print('All checks passed')
"
```

One assertion (`'flash_attn.flash_attn_interface' not in fa3_fn`) produced a **false negative** because the rewritten function's docstring contains the warning text "Do NOT use `flash_attn.flash_attn_interface` to detect FA3". The actual import code in the function body is correct (`from flash_attn_3 import ...`). The assertion was dropped; the docstring warning is intentional.

#### Layout correctness (BSHD transpose)

All flash_attn functions take **BSHD** input (batch, seq, heads, dim). The internal representation in the patched attention forward is **BHSD** (PyTorch convention). `_flash_attention()` transposes `(1,2)` before calling and transposes back after:

```python
q_fa = q.transpose(1, 2).contiguous()   # BHSD → BSHD
...
return out.transpose(1, 2)               # BSHD → BHSD
```

`flash_kvcache_attention()` applies the same BHSD→BSHD transpose for the query and both cache tensors before passing to `flash_attn_with_kvcache`.

#### Dynamo leaf registration

All flash-attn kernel functions are registered as dynamo leaf nodes immediately after import using `torch._dynamo.allow_in_graph()`. This tells `torch.compile` to treat each flash-attn call as a single opaque node — it will not attempt to trace into the C extension / CUDA kernel, preventing graph breaks and allowing the surrounding Python logic to compile into a clean graph.

```python
fa2_func             = torch._dynamo.allow_in_graph(fa2_func)
flash_attn_with_kvcache = torch._dynamo.allow_in_graph(flash_attn_with_kvcache)
fa3_func             = torch._dynamo.allow_in_graph(fa3_func)
fa4_kvcache          = torch._dynamo.allow_in_graph(fa4_kvcache)
```

Applied in `_try_import_flash_attn_2/3/4()` right after import, before the function is returned. Because `get_attention_backend()` is `@lru_cache`, registration happens once per process.

#### `cache_seqlens` dtype requirement

`flash_attn_with_kvcache` requires `cache_seqlens` to be `torch.int32`, not `torch.int64`. `StaticKVCache.cache_seqlens` is allocated as:

```python
self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
```

and updated via `self.cache_seqlens.fill_(self._seq_len)` — preserves int32 throughout.

---

## Dynamo-Optimized Generation (`batch_sample_dynamo`)

### Overview

`batch_sample_dynamo` is an optimized variant of `batch_sample` in `generation_functions.py` that replaces dynamic memory allocation with static pre-allocated tensors and supports `torch.compile` / CUDA graphs. All behavior is controlled via environment variables — the existing `batch_sample` is unchanged.

### Environment Variables

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `FAST_DLLM_USE_DYNAMO` | `0` / `1` | `0` | Switches eval.py to use `batch_sample_dynamo` |
| `FAST_DLLM_EXECUTION_MODE` | `eager` / `compile` / `cudagraph` | `eager` | Selects execution backend |
| `FAST_DLLM_ATTENTION_BACKEND` | `auto` / `flash4` / `flash3` / `flash2` / `flashinfer` / `sdpa` | `auto` | Attention backend hierarchy |
| `FAST_DLLM_MAX_SEQ_LEN` | integer | `4096` | Max pre-allocated KV cache length |
| `FAST_DLLM_COMPILE_MODE` | `default` / `reduce-overhead` / `max-autotune` | `reduce-overhead` | torch.compile optimization level |
| `FAST_DLLM_CUDA_GRAPH_WARMUP` | integer | `3` | Warmup steps before CUDA graph capture |

### Optimization Details

#### 1. Static KV Cache (`utils/static_kv_cache.py`)

**Problem:** Original `DynamicCache` uses `torch.cat` on every update, causing dynamic memory allocation and CUDA graph incompatibility. Earlier list-of-tensor design (`list[Tensor]` per layer) caused `getitem_const` guards per concrete `layer_idx` in torch.compile, producing per-layer sub-graphs.

**Solution:** `StaticKVCache` uses **stacked tensors** `[L, B, H, S, D]` (one tensor for all keys, one for all values). Layer indexing via `.select(0, layer_idx)` compiles to `aten.select.int` — single graph node, no graph break. All buffers marked with `torch._dynamo.mark_static_address` for CUDA graph compatibility.

Two write modes:
- `update()` — in-place `.copy_()` with Python-int slicing at committed position, then advances pointer after last layer. Used during eager prefill and block commit.
- `write_scratch_compiled()` — in-place write using `index_copy_` with tensor-valued `kv_write_start` offset. Does NOT advance the committed pointer. Used inside the compiled graph during diffusion steps.

Additional tensor-valued control signals for the compiled graph:
- `kv_write_start`: scalar int64 tensor — offset where current block KV is written (set from eager code before each compiled call)
- `scratch_seqlens`: `(batch,)` int32 tensor — effective KV length during diffusion (committed + current block), used as `cache_seqlens` for flash attention
- `cache_seqlens`: `(batch,)` int32 tensor — committed valid KV length
- `_arange` + `_write_idx`: pre-allocated tensors for building write indices

`StaticBlockCache` handles within-block sub-block caching with the same in-place pattern, and `reset()` between blocks.

**Key insight:** During diffusion steps, the cross-block cache region `[0, past_len)` is read-only. Only the current-block region `[past_len, past_len+block_size)` is overwritten. The committed pointer stays at `past_len` throughout the diffusion loop. The full static buffer (constant shape) is always passed to flash attention, with `scratch_seqlens` controlling the valid length — no dynamic slicing.

#### 2. Attention Layer Patching (`utils/attention_backends.py`)

**Problem:** The model's `Fast_dLLM_QwenAttention.forward()` does `torch.cat(past_kv, current_kv)` in the non-update path (diffusion steps), creating variable-size tensors that break CUDA graphs.

**Solution:** Monkey-patch each attention layer with a static-cache-aware forward. Key design: the patched forward does NOT capture `static_cache` in its closure — instead accesses it via the `past_key_value` argument, which traces from `model._static_kv_cache` (module state). This is critical because closure-captured cache would be treated as function-argument mutation by CUDA graph trees, triggering re-recording.

Patched forward behavior:
- **Diffusion steps** (`update_past_key_values=False`): uses `past_key_value.write_scratch_compiled()` (index_copy_ with tensor offset) for in-place KV write, then `past_key_value.get_full_kv(layer_idx)` for full static buffer
- **Commit steps** (`update_past_key_values=True`): uses `past_key_value.update()` (Python-int slice, eager only)
- Always passes full static KV buffer to `flash_kvcache_attention()` with `cache_seqlens=past_key_value.scratch_seqlens`
- Falls through to original forward for non-StaticKVCache usage
- Original forwards are saved and restored after generation via `unpatch_attention_layers()`

Signature: `_create_patched_attention_forward(attn_module, attn_backend, static_block_cache=None)` — no `static_cache` parameter.

**Attention mask optimization:** During diffusion steps, all queries are in the current block and all KVs are in blocks ≤ current, so the block-causal mask `block_q >= block_kv` is trivially True. The patched attention skips mask computation and passes `attn_mask=None` (full attention).

#### 3. Attention Backend Hierarchy (`utils/attention_backends.py`)

Fallback order based on GPU compute capability and package availability:
1. **Flash Attention 4** — `flash_attn_4.flash_attn_with_kvcache`, Hopper paged KV (SM 9.0+, `pip install flash-attn-4`)
2. **Flash Attention 3** — `flash_attn_3.flash_attn_func`, SM 8.0+ (built from `hopper/` subdir, installs as `flash_attn_3` package)
3. **Flash Attention 2** — `flash_attn.flash_attn_func` + `flash_attn_with_kvcache`, SM 8.0+ (`pip install flash-attn`)
4. **FlashInfer** — `flashinfer.BatchPrefillWithPagedKVCacheWrapper`, SM 7.5+ (`pip install flashinfer-python`)
5. **SDPA** — `torch.nn.functional.scaled_dot_product_attention` (always available)

All flash_attn functions use BSHD layout; `_flash_attention()` transposes BHSD→BSHD before each call and back after. FA3 has no `dropout_p` parameter — handled by a separate code path in `_flash_attention()`.

FlashInfer uses NHD layout (seq, heads, dim) with ragged/paged batching. Our BSHD static KV buffers `[B, S, H, D]` are zero-copy compatible with FlashInfer's paged NHD layout `[num_pages=B, page_size=S, H, D]` — see FlashInfer Integration section below.

`flash_kvcache_attention()` uses `cache_seqlens` (torch.int32, shape `(B,)`) for variable-length masking on static buffers — bypasses materialising an attention mask. Supported by FA2 (kvcache), FA4, and FlashInfer.

Current hardware (RTX A5000, SM 8.6): FA2 active after installing `flash-attn==2.8.3` (see Flash Attention Setup section). FlashInfer also supported (`pip install flashinfer-python`). FA3/FA4 require SM 9.0+ (H100/B200).

#### 4. Batch Bucketing (`utils/batch_bucketing.py`)

**Problem:** Original `batch_sample` removes finished sequences (`input_ids = input_ids[~finished_flag]`), shrinking the batch dimension and causing graph breaks / re-recordings.

**Solution:** `BatchBucket` keeps all slots alive:
- `active_mask`: `(batch,)` bool tensor tracking which slots are still generating
- Finished slots get padded with `pad_token_id` via `pad_finished_slots()`
- Unmasking operations are AND-ed with `active_mask` to prevent modifying finished sequences
- `get_results()` returns a dict mapping original batch index → output tensor (finished outputs saved, unfinished fall back to final `x_t`)

#### 5. torch.compile / CUDA Graphs (`utils/dynamo_utils.py`)

Module-level `_COMPILED_FWD_CACHE` dict ensures same `OptimizedModule` across all batches → same CUDA graph tree → graphs from batch N reused in batch N+1.

- **Eager mode**: `make_eager_forward(model)` returns uncompiled `forward_dynamo`.
- **Compile mode**: `make_compiled_forward(model, compile_mode)` wraps compiled forward with `cudagraph_mark_step_begin()` before each call. Uses `_get_or_compile_fwd(func, mode)` for caching.
- Per-step tensors (input_ids, etc.) are explicit dynamic function args — CUDA graph trees D2D-memcpy their values at replay time (~microseconds). KV cache is accessed via `past_key_values` arg (traces from `model._static_kv_cache` module state — interior mutations, no version-counter checks).

#### 6. `forward_dynamo` in `modeling.py`

**Problem:** `Fast_dLLM_QwenModel.forward()` calls `self.eval_mask(input_ids.shape[1], block_size, past_seq_len)` which creates a dynamically-sized `(seqlen, seqlen + cache_seq_len)` tensor. This dynamic shape causes CUDA graph re-captures and prevents `torch.compile` from producing a single static graph.

**Solution:** Add `forward_dynamo` methods to both `Fast_dLLM_QwenModel` and `Fast_dLLM_QwenForCausalLM` in `modeling.py`. These are NEW methods — existing `forward` methods are untouched.

Key differences from `forward()`:
- **No `eval_mask` call** — passes `attention_mask=None` unconditionally. Valid because:
  - Diffusion steps (`update_past_key_values=False`): all queries are in the current block, all KVs are in blocks ≤ current → `block_q >= block_kv` is trivially True.
  - Cache-update steps (`update_past_key_values=True`, single block at block completion): same reasoning applies.
  - Multi-block prefill is routed to `self.forward()` (original) in `batch_sample_dynamo`, not `forward_dynamo`.
- **No `DynamicCache` allocation** — requires `past_key_values` to be passed as `StaticKVCache`.
- **No training path** — inference-only; omits the noise augmentation and complementary mask logic.
- **Requires patched attention layers** — must call `patch_attention_layers()` before use; the patched layers handle `StaticKVCache` in-place writes instead of `torch.cat`.

**Prefill split in `batch_sample_dynamo`:** Multi-block prefill uses `self.forward()` (original, correct `eval_mask`). Single-block diffusion and cache-update steps use `forward_fn` (wrapping `forward_dynamo`). This keeps CUDA graph shapes constant for the hot diffusion loop while correctly masking the initial prefill.

### Implementation Log

#### Files Created / Changed

| File | Change |
|------|--------|
| `utils/static_kv_cache.py` | Created. `StaticKVCache` (stacked `[L,B,H,S,D]` KV buffers, `write_scratch_compiled` with `index_copy_`, tensor control signals, `mark_static_address`), `StaticBlockCache` (within-block sub-block cache). |
| `utils/batch_bucketing.py` | Created. `BatchBucket` with fixed batch size, `active_mask`, `pad_finished_slots()`, `get_results()`. |
| `utils/attention_backends.py` | Created. `AttentionBackend` class with FA4/FA3/FA2/FlashInfer/SDPA hierarchy; `_create_patched_attention_forward()` (no closure-captured cache), `patch_attention_layers()`, `unpatch_attention_layers()`. FlashInfer integration: `_try_import_flashinfer()`, `plan_flashinfer()`, `_flashinfer_kvcache_attention()`, `_wrap_flashinfer_prefill_run_as_custom_op()`. |
| `utils/dynamo_utils.py` | Created. Module-level `_COMPILED_FWD_CACHE`, `_get_or_compile_fwd()`, `make_compiled_forward()`, `make_eager_forward()`. |
| `generation_functions.py` | Added `batch_sample_dynamo()` method. Persistent `self._static_kv_cache` on model, pre-allocated `_x_block_buf` with `mark_static_address`. Per-block tensor control signal setup. Prefill uses `self.forward()` (original); diffusion loop uses compiled `forward_fn`. |
| `modeling.py` | Added `Fast_dLLM_QwenModel.forward_dynamo()` and `Fast_dLLM_QwenForCausalLM.forward_dynamo()`. Both are NEW methods — original `forward` methods are untouched. |
| `eval.py` | Extended to call `batch_sample_dynamo` when `FAST_DLLM_USE_DYNAMO=1`. |
| `test_compile_path.py` | Created. 7-section compile-path analysis script. |
| `CLAUDE.md` | This file — continuously updated with optimization details, issues, and fixes. |

#### Step-by-Step Implementation

1. **Static KV Cache** (`utils/static_kv_cache.py`) — replaced `DynamicCache` (torch.cat on every update) with pre-allocated buffers. Key design: `update()` advances pointer only after the last layer; `write_scratch()` does in-place overwrite without advancing the pointer (used in diffusion loop).

2. **Batch Bucketing** (`utils/batch_bucketing.py`) — replaced `input_ids = input_ids[~finished_flag]` (shrinks batch, breaks CUDA graphs) with `active_mask` that keeps all slots alive. Finished slots receive padding; unmask operations are gated by `active_mask`.

3. **Attention layer patching** (`utils/attention_backends.py`) — monkey-patches `self_attn.forward` on every layer to detect `StaticKVCache` and use in-place writes + `get_kv_up_to()` instead of `torch.cat`. The patched forward always passes `attention_mask=None` (block-causal mask is trivially True during diffusion steps). Uses `ALL_ATTENTION_FUNCTIONS["sdpa"]` from transformers for the actual computation (handles GQA natively, picks flash/memory-efficient kernel automatically).

4. **`forward_dynamo` in `modeling.py`** — new inference-only variants of both model-level and CausalLM-level forward. Skip `eval_mask()` and pass `attention_mask=None`. Multi-block prefill in `batch_sample_dynamo` is explicitly routed to the original `forward()` to preserve correct masking.

5. **Compile / CUDA graph wrappers** (`utils/dynamo_utils.py`) — `make_dynamo_forward()` dispatches to eager / `torch.compile` / `CUDAGraphRunner`. Fixed `dynamic=True` so `past_seen_tokens` (a Python int that changes per block) is treated symbolically by torch.compile rather than value-specialised.

6. **Config banner logging** (`generation_functions.py`) — prints execution mode, attention backend (resolved name + user preference), compile mode, max_seq_len, batch, block size, and block_cache status at the start of each `batch_sample_dynamo` call.

7. **Flash Attention setup** — researched Dao-AILab/flash-attention; corrected three FA3 detection bugs and FA4 package misidentification; source-built `flash-attn==2.8.3` against CUDA 12.6 toolkit (prebuilt wheel failed with C++ ABI mismatch against torch 2.10).

#### Test Results (`test_compile_path.py`, run 2026-03-15)

```
[env] torch 2.10.0+cu128  device=cuda
```

| Test | Description | Result |
|------|-------------|--------|
| 1a | `StaticKVCache.update()` advances pointer only after last layer | PASS |
| 1b | `StaticKVCache.write_scratch()` does NOT advance committed pointer | PASS |
| 1c | `get_kv_up_to()` returns correct shape `(2, 4, 16, 32)` | PASS |
| 1d | `cache_seqlens` updated correctly after `update()` | PASS |
| 2a | `StaticBlockCache.__len__` tracks initialized layers | PASS |
| 2b | `StaticBlockCache.reset()` clears `__len__` back to 0 | PASS |
| 3 | `dynamic=True` calls_captured after 5 blocks: 49 (no extra retraces) | PASS |
| 4 | graph_break_count inside StaticKVCache-using fn: 0 | PASS |
| 5 | `cache_position` shape `(block_size,)` correct for all `use_block_cache` modes | PASS |
| 5 | `attention_mask=None` confirmed — no dynamic eval_mask tensor | PASS |
| 6 | `torch.compile` wraps `forward_dynamo`-shaped fn without errors | PASS |
| 7 | `dynamic=True`: `calls_captured = [6, 0, 0, 0, 0]` — compiled once, zero retraces | PASS |

#### Flash Attention Install Issues

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Prebuilt wheel import failed (`undefined symbol: _ZN3c104cuda…`) | ABI mismatch: wheel built for torch 2.8, environment has torch 2.10 | Uninstall wheel; build from source |
| Build failed with system nvcc | `/usr/bin/nvcc` is CUDA 11.5 — incompatible with torch 2.10 CUDA 12.8 | Set `CUDA_HOME=/usr/local/cuda-12.6` |
| Source build slow | C++ template instantiation for every attention kernel | `MAX_JOBS=4`; takes ~20–40 min |

---

### Issues and Fixes

1. **`StaticKVCache.update()` pointer advancement**: The committed length must only advance after ALL layers have written (not per-layer). Fix: advance on `layer_idx == num_layers - 1`.

2. **Block cache `len()` semantics**: `DynamicCache.__len__()` returns the number of layers with data (grows as layers are added). `StaticBlockCache` pre-allocates all layers, so `__len__()` tracks `_num_initialized` separately.

3. **Attention mask during diffusion**: Block-causal mask is trivially True during diffusion steps (all queries in current block, all KVs in blocks ≤ current). Passing `attn_mask=None` avoids unnecessary mask computation.

4. **GQA in SDPA fallback**: The SDPA backend manually expands K/V from `num_kv_heads` to `num_attention_heads` (4 → 28 for Qwen2.5-7B) using repeat-interleave before calling `F.scaled_dot_product_attention`.

5. **Batch bucketing interaction with unmasking**: The unmask operation is AND-ed with `active_mask[:, None]` to prevent writing tokens into finished sequences, which would corrupt saved outputs.

6. **SDPA backend performance regression (FIXED)**: Initial implementation used a custom `_sdpa_attention` with manual GQA head expansion (4→28 heads via repeat) and raw `F.scaled_dot_product_attention`. The 2D bool mask from `eval_mask` forced PyTorch to use the slow math SDPA backend instead of flash/memory-efficient, causing 14+ minute hangs on single forward passes. Fix: use transformers' `ALL_ATTENTION_FUNCTIONS["sdpa"]` which handles GQA natively and selects the optimal kernel.

7. **Dynamic `eval_mask` tensor breaking CUDA graphs (FIXED)**: `Fast_dLLM_QwenModel.forward()` creates a `(seqlen, seqlen+cache_seq_len)` attention mask tensor with shapes that vary between diffusion steps and prefill. This caused CUDA graph re-captures on every call. Fix: add `forward_dynamo` to `modeling.py` (new method, original `forward` untouched) that passes `attention_mask=None`. Multi-block prefill is routed to the original `forward()` to preserve correct cross-block masking.

8. **`make_dynamo_forward` always used `model.forward`**: Before fix, the compile/cudagraph wrappers wrapped `model.forward` which still called `eval_mask`. Fix: updated `make_dynamo_forward` to use `getattr(model, "forward_dynamo", model.forward)` so the optimized path is used automatically when `forward_dynamo` exists.

9. **`dynamic=False` in `get_compiled_forward` caused per-block retracing (FIXED)**: With `dynamic=False`, `torch.compile` specialises on Python int *values* (not just shapes). `StaticKVCache.get_seq_length()` returns a Python int that changes every block (`0 → block_size → 2*block_size → …`). This value is used in `torch.arange(past_seen_tokens, …)` inside `forward_dynamo`, making torch.compile create a new specialisation per block — effectively defeating compilation. Verified via `torch._dynamo.utils.counters["stats"]["calls_captured"]`: with `dynamic=False`, extra retraces occurred on every block; with `dynamic=True`, torch.compile compiled 6 subgraphs on the first block and **zero additional compilations** on all subsequent blocks. Fix: changed `dynamic=False → dynamic=True` in `make_dynamo_forward` (utils/dynamo_utils.py line 172). The function signature of `get_compiled_forward` still defaults to `False` — the override is at the call site in `make_dynamo_forward`.

   Analysis script: `test_compile_path.py` (run with the v2 venv).

10. **Three FA3 detection bugs in `_try_import_flash_attn_3` (FIXED)**:
    - **Wrong SM guard**: `cc[0] < 9` blocked FA3 on SM 8.x entirely. FA3's `setup.py` compiles SM 8.0 kernels (`sources_fwd_sm80`) so it runs on Ampere+. Fix: changed guard to `cc[0] < 8`.
    - **Wrong import path**: `from flash_attn.flash_attn_interface import flash_attn_func` imports FA2's internal module, not FA3. On a system with FA2 installed, this silently returns the FA2 function mislabelled as FA3. Fix: use `from flash_attn_3 import flash_attn_func`.
    - **Wrong fallback package name**: `flash_attn_hopper` does not exist — the package built from the `hopper/` subdir installs as `flash_attn_3`. Fix: removed the `flash_attn_hopper` fallback.

11. **FA3 `dropout_p` parameter missing (FIXED)**: `_flash_attention()` was passing `dropout_p=dropout` for all flash backends. FA3's `flash_attn_func` has no `dropout_p` parameter — this would raise `TypeError` at runtime. Fix: added an explicit FA3 branch in `_flash_attention()` that omits `dropout_p`.

12. **FA4 misidentified as main flash-attn package (FIXED)**: `_try_import_flash_attn_4()` imported `flash_attn_with_kvcache` from the main `flash_attn` package and checked `version >= (2, 7)`. FA4 is a completely separate package (`flash-attn-4`, imports as `flash_attn_4`). The old code would label FA2 ≥ 2.7 as "FLASH4" on Hopper hardware. Fix: now imports from `flash_attn_4` directly.

13. **Flash attention backend silently ignored in patched forward (FIXED)**: `_create_patched_attention_forward()` accepted `attn_backend` as a parameter but never used it inside the closure — every attention call went to `ALL_ATTENTION_FUNCTIONS["sdpa"]` regardless of the detected backend (FA2/FA3/FA4). Verified by checking `get_attention_backend('auto')` on RTX A5000 (SM 8.6) — FA2 was correctly detected but never dispatched to.
    - **Fix**: added a branch in `patched_forward` that dispatches to `attn_backend.attention()` when `attn_backend.name != SDPA`, falling back to `ALL_ATTENTION_FUNCTIONS["sdpa"]` only for the SDPA path.
    - **Layout fix**: `attn_backend.attention()` returns BHSD `(B, nheads, S_q, D)`, but the downstream `reshape(*input_shape, -1)` expects BSHD `(B, S_q, nheads, D)` (matching what `ALL_ATTENTION_FUNCTIONS["sdpa"]` already returns). Added `.transpose(1, 2)` after the flash call to convert BHSD → BSHD before the reshape.

14. **Infinite loop in `batch_sample_dynamo` when stop token generated mid-block (FIXED)**: When a sequence generates the stop token during sub-block diffusion, `bucket.mark_finished()` sets `active_mask[0] = False`. The inner `while True` diffusion loop continues because masked tokens remain in the sub-block. On the next iteration, `unmask_idx & bucket.active_mask` is all False — no tokens ever get unmasked, `mask_idx[:, start:end].sum()` never reaches 0, and the loop runs forever. This caused a 6+ minute hang at 0 outputs in eval (observed via `Generating...: 0/100 [06:23<?, ?it/s]` in the progress bar).
    - **Fix 1** (`generation_functions.py`, block-complete `while True`): added `if bucket.all_finished(): break` as the very first check, before computing `mask_idx`. This exits the block-complete loop as soon as all sequences finish mid-block, without attempting another sub-block forward pass.
    - **Fix 2** (`generation_functions.py`, inner sub-block `while True`): changed `if mask_idx[:, start:end].sum() == 0:` to `if mask_idx[:, start:end].sum() == 0 or bucket.all_finished():`. This exits the inner diffusion loop immediately when all sequences finish, even if the current sub-block still has masked tokens.

15. **Ghost mask tokens from finished sequences cause infinite loop in multi-sequence batches (FIXED)**: With `batch_size > 1`, when one sequence finishes mid-block, its slot still contains `mask_id` tokens (because `pad_finished_slots` is only called once at the start of each block). The `mask_idx = (x_t[:, -block_size:] == mask_id)` check in both `while True` loops counts these ghost masks, so `mask_idx.sum()` never reaches 0 for the remaining active sequences. The batch hangs even though all active sequences have completed their block. Observed as `0/13 [02:29<?, ?it/s]` with `batch_size=8` on GSM8K eval.
    - **Fix 1** (block-complete `while True`, line 359): changed `mask_idx` computation to `(x_t[:, -block_size:] == mask_id) & bucket.active_mask[:, None]` — finished slots are excluded from the completion check.
    - **Fix 2** (inner sub-block `while True`, line 398): same `active_mask` gate applied to `mask_idx` in the inner diffusion loop, for the same reason.
    - **Fix 3** (after `bucket.mark_finished`, line 463): added `bucket.pad_finished_slots(x_t, block_size)` immediately after marking a sequence finished, so its slot is overwritten with pad tokens right away — prevents ghost `mask_id` tokens from accumulating between sub-block iterations.

16. **`reduce-overhead` compile mode incompatible with in-place KV cache mutations (FIXED)**: `torch.compile` with `mode="reduce-overhead"` internally enables CUDA graph capture via torch._inductor. CUDA graphs cannot record mutations to tensors whose addresses aren't declared static — they treat such mutations as external side effects and skip graph capture. `StaticKVCache.write_scratch()` and `update()` both mutate pre-allocated buffers via `.copy_()`, and `update()` additionally calls `self.cache_seqlens.fill_()`. This caused the compiler to emit **56–57 "skipping cudagraphs due to mutated inputs"** warnings per forward pass (28 layers × 2 KV tensors = 56 `.copy_()` calls + 1 `cache_seqlens.fill_()` = 57), silently falling back to eager execution — negating all compile-mode benefits.
    - **Root cause**: The CUDA graph runtime didn't know that the KV cache buffers live at fixed GPU memory addresses. Without that guarantee, `.copy_()` looks like an input mutation to an external tensor, which CUDA graphs can't safely replay.
    - **Fix**: Added `torch._dynamo.mark_static_address()` calls to all pre-allocated buffers in `StaticKVCache.__init__()` and `StaticBlockCache.__init__()` (in `utils/static_kv_cache.py`). This is the same technique used by vLLM and SGLang for their static KV caches — it tells torch.compile that these tensors always reside at the same GPU memory address, making in-place `.copy_()`/`.fill_()` safe to include inside a captured CUDA graph. Also added `torch.compiler.cudagraph_mark_step_begin()` before each compiled forward call in `make_dynamo_forward()` (in `utils/dynamo_utils.py`) to signal new inference steps to the CUDA graph pool manager.

17. **Dynamic KV cache slicing caused symint key changes and recompilation crash (FIXED)**: The patched attention forward called `static_cache.get_kv_up_to(layer_idx, total_len)` → `cache[:, :, :total_len, :]` where `total_len` changes per block. This created variable-output-shape tensors that changed the symint key in `reduce-overhead` mode (e.g., `(4096, 32, 64, 32, 8)` → `(4096, 96, 64, 32, 8)`), triggering recompilation for every new block. The recompilation then crashed in `sym_node.py`'s `_optimized_add` during AOT autograd's symbolic shape analysis.
    - **Root cause**: The whole point of `StaticKVCache` is to keep tensor shapes **constant**. But `get_kv_up_to` sliced the buffer to the variable valid length, defeating the purpose. The variable-length information should be communicated via `cache_seqlens` (a small `(B,)` int32 tensor whose shape never changes), not via tensor slicing.
    - **Fix**: Rewrote the patched attention forward in `_create_patched_attention_forward()` to always pass the **full static buffer** (`static_cache.get_full_kv(layer_idx)` → constant shape `(B, H, max_seq_len, D)`) and use `flash_kvcache_attention()` with `cache_seqlens` / `scratch_seqlens` for variable-length masking. FA2's `flash_attn_with_kvcache` reads `cache_seqlens` at runtime to know how much of the buffer is valid; SDPA fallback builds a mask from seqlens. Added `scratch_seqlens` buffer to `StaticKVCache` for diffusion steps where the effective length (committed + current block) exceeds the committed `cache_seqlens`. Removed unused `static_cache` parameter from `make_dynamo_forward()`. Result: all tensor shapes are constant across blocks → one CUDA graph capture, zero recompilations.

18. **Stacked-tensor KV cache redesign inspired by Token_spase-dllm (FIXED)**: Complete redesign of the compiled path based on patterns from Token_spase-dllm's `batch_sample_dynamo_v2` and `_compiled_forward_v3`. Multiple interrelated issues were resolved:
    - **List-of-tensor KV cache caused per-layer sub-graphs**: `static_cache.key_cache[layer_idx]` (Python list `__getitem__`) creates `getitem_const` guards per concrete `layer_idx` value. With 28 layers, this produced 28 guard checks and separate sub-graphs. Fix: converted to stacked tensor `[L,B,H,S,D]` with `.select(0, layer_idx)` → `aten.select.int` (single graph node).
    - **Closure-captured cache treated as argument mutation**: Patched forward captured `static_cache` as closure variable → mutations flagged as function-argument mutations → CUDA graph re-recording every step. Fix: removed `static_cache` from closure; access via `past_key_value` arg which traces from `model._static_kv_cache` (module state → interior mutations, no version-counter checks).
    - **Per-batch re-compilation from new compiled wrappers**: `make_dynamo_forward()` created a new compiled wrapper per batch → new `OptimizedModule` → new CUDA graph tree. Fix: module-level `_COMPILED_FWD_CACHE` dict keyed by `(func, mode)` ensures same `OptimizedModule` across batches.
    - **Python-int slice specialization**: `cache[:, :, position:position+len, :]` where `position` is a Python int that changes per block → Dynamo specializes per value, creating a new graph per block. Fix: `index_copy_` with tensor-valued `kv_write_start` (scalar int64 tensor set via `.fill_()` from eager code).
    - **Persistent cache across batches**: `self._static_kv_cache` stored on model object, reused when batch size and max_seq_len match (`.zero_and_reset()` preserves GPU addresses and `mark_static_address` registrations).
    - **Pre-allocated input buffer**: `_x_block_buf` with `mark_static_address` for the block input tensor passed to compiled forward, avoiding dynamic allocation per step.
    - **Reference implementation**: Token_spase-dllm's `_compiled_forward_v3` went through 9 iterations to get `reduce-overhead` working. Key patterns adopted: module-level compiled function cache, `cudagraph_mark_step_begin()` before each compiled call, per-step tensors as explicit dynamic function args, KV buffers as module state (not closure).

19. **`flash_attn_with_kvcache` FakeTensor crash under torch.compile (FIXED)**: `torch._dynamo.allow_in_graph(flash_attn_with_kvcache)` prevents Dynamo from tracing INTO the function, but the FX graph executor still runs each node with FakeTensors to determine output shapes. The `flash_attn_with_kvcache` CUDA kernel accesses raw data pointers (`flash_attn_gpu.fwd_kvcache`), which FakeTensors don't have — raising `RuntimeError("Cannot access data pointer of Tensor")` during the first torch.compile compilation.
    - **Root cause**: In torch 2.10, `allow_in_graph` alone is insufficient for CUDA kernels that access data pointers. The function needs a registered Meta/FakeTensor implementation that returns the correct output shape without touching GPU data.
    - **Fix**: Replaced `allow_in_graph` with `torch.library.custom_op` wrapper (`_wrap_flash_kvcache_as_custom_op` in `utils/attention_backends.py`). The wrapper registers a `"CUDA"` implementation that calls the real kernel and a `"Meta"` implementation that returns `torch.empty_like(q)` (output shape = query shape for flash_attn_with_kvcache). Applied to both FA2 and FA4 kvcache functions. Verified: FakeTensor propagation succeeds, `torch.compile(mode="reduce-overhead")` compiles without error.

20. **Per-block CUDA graph re-recording from Python int `past_seen_tokens` (FIXED)**: Every block recorded a new CUDA graph (78 recordings, zero replays). The symint key changed by `block_size` each block: `64 → 96 → 128 → ... → 576`. Root cause: `forward_dynamo` computes `cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len)` where `past_seen_tokens = past_key_values.get_seq_length()` — a Python int that changes every block. Without `dynamic=True`, torch.compile specializes on the concrete int value, producing a new symint key → new CUDA graph recording per block.
    - **Fix 1** (`generation_functions.py`): Pre-allocate `_cache_pos_buf` (shape `(block_size,)`, `mark_static_address`) and `_cache_pos_arange`. Before each compiled call, fill with `_cache_pos_buf.copy_(_cache_pos_arange + past_len)` in eager code. Pass `cache_position=_cache_pos_buf` to `forward_fn`, which bypasses the `if cache_position is None:` branch in `forward_dynamo` — no Python int ever enters the compiled graph.
    - **Fix 2** (`utils/dynamo_utils.py`): Added `dynamic=True` to `torch.compile()` as a safety net for any other Python ints that might leak through (e.g., when `use_block_cache=True` with sub-block paths that still compute positions internally).

21. **Per-batch CUDA graph re-recording from recreated objects (FIXED)**: After fixing per-block symint re-recording (issue 20), each batch still recorded a new CUDA graph (5 batches → 5 recordings, zero replays). Three causes:
    - **Attention layer re-patching**: `patch_attention_layers()` was called every batch, creating new closure objects for `self_attn.forward` on every layer. Dynamo sees different function objects → recompiles → new CUDA graph tree.
    - **Buffer re-allocation**: `_x_block_buf`, `_cache_pos_buf`, `_cache_pos_arange` were recreated as new tensors every batch. Even with `mark_static_address`, the GPU addresses change, invalidating the CUDA graph recorded with old addresses.
    - **Unpatch/repatch cycle**: `unpatch_attention_layers()` in the `finally` block restored original forwards after each batch, then `patch_attention_layers()` re-patched at the start of the next batch — guaranteeing new closures every time.
    - **Fix**: Moved all per-batch-created objects to persistent model state (`self._static_kv_cache`, `self._dynamo_attn_backend`, `self._dynamo_original_forwards`, `self._dynamo_forward_fn`, `self._x_block_buf`, `self._cache_pos_buf`, `self._cache_pos_arange`). These are initialized once on first call (or when shapes change) and reused across all subsequent batches. The `finally` block no longer unpatches — layers stay patched for CUDA graph reuse. On shape change, `_need_reinit` unpatches old forwards before re-patching.

22. **`batch_sample_dynamo` eager mode 18% slower than `batch_sample` due to full-buffer transposes (FIXED)**: `flash_kvcache_attention` transposed the full `(B, 4, 4096, 128)` KV cache buffers from BHSD to BSHD (`.transpose(1,2).contiguous()`) on every attention call — 28 layers × 2 tensors × full-size copy per diffusion step (~224 MB of memcpy at batch=8). The original `batch_sample` used `torch.cat` to produce exact-size KV tensors and transformers' SDPA which handles layout internally.
    - **Root cause**: KV cache was stored in BHSD layout `[L, B, H, S, D]`, but `flash_attn_with_kvcache` requires BSHD. The expensive transpose happened on the large 4096-length read path instead of the small 32-length write path.
    - **Fix (phase 1)**: Changed `StaticKVCache` to BSHD-native layout `[L, B, S, H, D]`. Incoming key/value states (BHSD from model projections) are transposed to BSHD on write — cheap because write tensors are `block_size=32` long. `flash_kvcache_attention` now passes KV buffers directly to `flash_attn_with_kvcache` without transposing (only the query needs BHSD→BSHD transpose, which is also block_size=32). Net effect: replaced 2 × 4096-length transposes with 2 × 32-length transposes per layer per step — ~128x reduction in transpose work.
    - **Fix (phase 2 — zero transposes)**: Eliminated ALL remaining transposes by making Q/K/V projections output BSHD directly. `.view(B, S, -1, head_dim)` naturally produces BSHD `(B, S, H, D)` — the previous `.transpose(1, 2)` to get BHSD was unnecessary since flash_attn wants BSHD. Changes:
      - `_create_patched_attention_forward`: removed `.transpose(1, 2)` after Q/K/V projection; removed output `.transpose(1, 2)`.
      - `_apply_rotary_pos_emb`: changed `cos.unsqueeze(1)` → `cos.unsqueeze(2)` for BSHD broadcast `(B, S, 1, D)`.
      - `flash_kvcache_attention`: now accepts BSHD query directly; removed input/output transposes; removed SDPA fallback entirely (raises `RuntimeError` if FA2/FA4 kvcache unavailable).
      - `StaticKVCache.update/write_scratch_compiled/write_sparse/write_scratch`: all accept BSHD K/V directly (removed internal `.transpose(1, 2)`).
      - `StaticBlockCache`: changed layout from BHSD `(B, H, S, D)` to BSHD `(B, S, H, D)`.
      - `forward_sparse` sparse step: Q/K/V stay BSHD; `apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)`; removed `.transpose(1, 2)` on attention output.
    - **Net result**: Zero transposes in the entire hot path — projections output BSHD, cache stores BSHD, flash_attn reads BSHD natively.

23. **`batch_sample_dynamo` progressively slower than baseline at BS≥8 due to no batch compaction (FIXED)**: Benchmark sweep (GSM8K, BS=1-16) showed dynamo 9-11% faster at BS=1-4 but 3-12% slower at BS=8-16, with degradation worsening during later iterations within a single run.
    - **Root cause**: Baseline's `batch_sample` removes finished sequences from the batch (`input_ids = input_ids[~finished_flag]`), so compute cost drops as sequences finish. `batch_sample_dynamo` used `BatchBucket` to keep all slots alive (for CUDA graph fixed-shape compatibility), meaning every forward pass computed attention for ALL batch elements even when most had finished. GSM8K answers vary widely in length — at BS=16, if 12 sequences finish early, baseline drops to effective BS=4 while dynamo still runs BS=16. This wasted 3x compute on padding.
    - **Iteration-level evidence**: At BS=8, early iterations showed dynamo faster (10.1s/iter vs 11.4s/iter), but late iterations were slower (24.6s/iter vs 21.4s/iter) — the gap widened as more sequences finished and baseline's batch shrank while dynamo's stayed fixed.
    - **Fix**: Added batch compaction at block boundaries (eager code, between blocks). When sequences finish, remove them from all batch-indexed tensors:
      - `BatchBucket.compact()`: removes finished slots, returns `active_indices`. Preserves `_original_indices` mapping so `get_results()` maps back to correct original positions.
      - `BatchBucket.mark_finished_compacted()`: uses `_original_indices` for correct index mapping after compaction.
      - `StaticKVCache.compact_batch(active_indices)`: re-indexes batch dimension (dim=1 in `[L, B, S, H, D]`) via `index_select`, rebuilds batch-sized control signals.
      - `BlockSparseCache.compact_batch(active_indices)`: same re-indexing for sparse cache tensors.
      - Applied at end of each block in both `batch_sample_dynamo` and `batch_sample_sparse`.
    - **Compile-mode guard**: Compaction only runs when `execution_mode == "eager"`. In compile/cudagraph mode, batch dim must stay fixed — the old `BatchBucket` padding behavior is preserved. This is acceptable because compile mode benefits from static shapes, and the overhead is the trade-off for CUDA graph replay.

24. **`_kvcache_dispatch_logged` guard failure causing recompilation in compile mode (FIXED)**: The `mark_dispatch_logged()` call was placed after the prefill (line 488) but the CUDA graph warmup runs `compiled_fn()` **before** the prefill. During warmup, dynamo traced with `_kvcache_dispatch_logged == False` and recorded that as a guard. On the next compiled call the flag was `True` → guard failed → full recompilation (~6s extra on batch 0, and potentially suboptimal recompiled graph used for all subsequent batches).
    - **Root cause**: Order of operations was: (1) warmup records CUDA graphs (dynamo sees `False`, guards on it), (2) flag flips to `True` during warmup execution, (3) prefill runs, (4) `mark_dispatch_logged()` sets `True` (already `True`, too late — guard was already baked in as `== False`).
    - **Fix**: Moved `mark_dispatch_logged()` to immediately after `patch_attention_layers()`, **before** the warmup. Now dynamo's first trace sees `True` and guards on `True` — no guard failure, no recompilation.
    - **Evidence**: Compile log showed `[__recompiles] Recompiling function forward_dynamo ... _kvcache_dispatch_logged == False`. After fix, no `[__recompiles]` messages and batch 0 dropped from 35.65s to 29.64s.

25. **Duck sizing unifying `num_kv_heads` with `batch_size` causing recompilation in compact compile mode (FIXED)**: With compact compile (BS=4, valid sizes=[4,2,1]), warmup pre-records CUDA graphs for each batch size. Dynamo's duck sizing saw `key_cache[0].size()[2]` (H=4 in BSHD layout) equal to `input_ids.size()[0]` (batch=4) and unified them as the same symbolic dimension. When batch compacted to BS=2, the guard `H == batch_size` failed (4 ≠ 2) → recompilation. This happened twice during warmup (BS=4→BS=2, BS=2→BS=1), adding ~75s to batch 0 (vs 40s for static compile).
    - **Guard message**: `past_key_values.key_cache[0].size()[2] == input_ids.size()[0]  # duck sizing added this equality because these variables had the same size 4`
    - **Attempted fix 1 — stacked tensor `[L,B,S,H,D]`**: Changed list-of-tensor to stacked tensor so dynamo sees a single tensor instead of per-layer list elements. **Failed** — duck sizing still occurred (`key_cache.size()[3] == input_ids.size()[0]`), and AOT autograd decomposed in-place mutations on `.select()` views into `select_scatter` (copies entire `[28, 4, 4096, 4, 128]` tensor per mutation), causing a **2.3x slowdown** (64 vs 145 tok/s). Reverted.
    - **Fix**: Set `torch.fx.experimental._config.use_duck_shape = False` before compilation. This prevents dynamo from unifying dimensions that happen to have the same concrete value. Each dimension is tracked independently, so batch compaction (4→2→1) doesn't trigger guard failures. Applied only in compile mode.
    - **Key lesson**: Stacked tensors are fundamentally incompatible with torch.compile for in-place KV cache mutations. List-of-tensor + duck sizing disable is the correct approach.

---

## Token-Sparse Diffusion (`batch_sample_sparse`)

### Overview

`batch_sample_sparse` is a token-sparse variant of `batch_sample_dynamo` that reduces computation in diffusion steps by only recomputing tokens that change significantly between steps. Uses cosine similarity of hidden states to identify which tokens need recomputation.

### Environment Variables

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `FAST_DLLM_USE_SPARSE` | `0` / `1` | `0` | Switches eval.py to use `batch_sample_sparse` |
| `FAST_DLLM_TRANSFER_RATIO` | float (0-1) | `0.3` | Fraction of tokens to recompute in sparse steps |
| `FAST_DLLM_REFRESH_INTERVAL` | integer | `5` | Dense step every N diffusion steps within a block |
| `FAST_DLLM_EXECUTION_MODE` | `eager` | `eager` | Execution backend (eager only for sparse) |
| `FAST_DLLM_ATTENTION_BACKEND` | `auto` / `flash2` / `flashinfer` / `sdpa` | `auto` | Attention backend |
| `FAST_DLLM_MAX_SEQ_LEN` | integer | `4096` | Max pre-allocated KV cache length |

### Dense vs Sparse Steps

Within each block's diffusion loop, steps alternate between dense and sparse based on `refresh_interval`:

- **Dense step** (`step % refresh_interval == 0`): Full forward through all `block_size` tokens. Caches three intermediates per layer:
  1. `layer_input` — hidden states entering the layer (before `input_layernorm`)
  2. `attn_output` — output of self-attention (after `o_proj`, before residual add)
  3. `mlp_output` — output of MLP (after `down_proj`, before residual add)

- **Sparse step**: Per-layer cosine similarity between current hidden states and cached `layer_input` identifies the `transfer_ratio` fraction of tokens with lowest similarity (most changed). Only those tokens go through:
  1. `input_layernorm` → Q/K/V projection → RoPE → KV cache update → attention → `o_proj`
  2. Results scattered into cached `attn_output`
  3. Post-attention residual computed for all tokens: `mid = layer_input + attn_output`
  4. Selected tokens gathered from `mid` → `post_attention_layernorm` → MLP
  5. Results scattered into cached `mlp_output`
  6. Final: `output = mid + mlp_output` (all tokens)

### Key Design: Sparse KV Cache Updates

During sparse steps, only selected token positions in the current block need KV updates. Added `StaticKVCache.write_sparse(k, v, layer_idx, write_positions)` which uses `scatter_` with per-batch position indices `(B, num_tokens)`. This ensures the static KV cache always has up-to-date K/V for every token position, even when only a subset is recomputed per step.

### Architecture

```
Dense step (step 0, 5, 10, ...):
  embed → [for each layer: cache_input → layernorm → full_attn → cache_attn → residual → layernorm → full_mlp → cache_mlp → residual] → norm → lm_head

Sparse step (step 1-4, 6-9, ...):
  embed → [for each layer: cosine_sim → select_tokens → layernorm(selected) → sparse_QKV → sparse_attn → scatter_attn → full_residual → layernorm(selected) → sparse_MLP → scatter_mlp → full_residual] → norm → lm_head
```

### Files Created / Changed

| File | Change |
|------|--------|
| `utils/block_sparse_cache.py` | Created. `BlockSparseCache` with stacked `[L, B, block_size, hidden_size]` tensors for `layer_input`, `attn_output`, `mlp_output`. Scatter methods for in-place updates at selected positions. `compact_batch()` for batch compaction. |
| `utils/static_kv_cache.py` | Added `write_sparse(k, v, layer_idx, write_positions)` — scatter-based KV write at arbitrary per-batch positions. Added `compact_batch()` for batch compaction. All write methods accept BSHD directly (zero transposes). |
| `utils/batch_bucketing.py` | Added `compact()` — removes finished slots, returns `active_indices`. Added `mark_finished_compacted()` — uses `_original_indices` for correct post-compaction index mapping. |
| `utils/attention_backends.py` | BSHD throughout: `_apply_rotary_pos_emb` uses `unsqueeze(2)`, patched forward keeps Q/K/V in BSHD, `flash_kvcache_attention` accepts/returns BSHD directly. SDPA fallback removed. FlashInfer backend added: zero-copy BSHD→paged NHD layout bridge, `plan_flashinfer()` in eager code, custom op wrapping for `run()` under torch.compile. |
| `modeling.py` | Added `forward_sparse()` to both `Fast_dLLM_QwenModel` and `Fast_dLLM_QwenForCausalLM`. Dense path: full layer forward + cache intermediates. Sparse path: cosine similarity selection, sparse Q/K/V/attention/MLP, scatter results. Q/K/V stay BSHD with `unsqueeze_dim=2` for RoPE. |
| `generation_functions.py` | Added `batch_sample_sparse()`. Based on `batch_sample_dynamo` with dense/sparse switching, `BlockSparseCache` management, and `forward_sparse` calls. Both `batch_sample_dynamo` and `batch_sample_sparse` compact batches at block boundaries (eager mode only). |
| `eval.py` | Extended to call `batch_sample_sparse` when `FAST_DLLM_USE_SPARSE=1`. |
| `run_configs_parallel.sh` | 5 configs × 6 batch sizes sweep (baseline, dynamo, sparse×3 ratios). |

### Decoder Layer Flow (Sparse Step Detail)

For each decoder layer in the sparse step:

```python
# 1. Cosine similarity → select tokens
cached_input = block_sparse_cache.get_layer_input(layer_idx)
cos_sim = F.cosine_similarity(hidden_states, cached_input, dim=-1)  # (B, block_size)
num_tokens = int(transfer_ratio * block_size)
_, token_indices = cos_sim.topk(num_tokens, largest=False)  # lowest sim

# 2. Update layer input cache
block_sparse_cache.cache_layer_input(layer_idx, hidden_states)

# 3. Gather → layernorm → Q/K/V projection → RoPE
selected = hidden_states.gather(1, idx)
selected_norm = input_layernorm(selected)
q = q_proj(selected_norm)  # (B, num_tokens, num_heads, head_dim)
k = k_proj(selected_norm)
v = v_proj(selected_norm)
q, k = apply_rotary_pos_emb(q, k, selected_cos, selected_sin)

# 4. Sparse KV cache update (only at selected positions)
write_positions = token_indices + past_len  # absolute positions
static_cache.write_sparse(k, v, layer_idx, write_positions)

# 5. Attention: sparse Q, full KV cache
full_k, full_v = static_cache.get_full_kv(layer_idx)
sparse_attn_out = flash_kvcache_attention(q, full_k, full_v, cache_seqlens=...)
sparse_attn_out = o_proj(sparse_attn_out)

# 6. Scatter attn output → full residual
block_sparse_cache.scatter_attn_output(layer_idx, token_indices, sparse_attn_out)
attn_output_full = block_sparse_cache.get_attn_output(layer_idx)
mid = hidden_states + attn_output_full

# 7. Gather → post_layernorm → MLP → scatter
selected_mid = mid.gather(1, idx)
sparse_mlp_out = mlp(post_attention_layernorm(selected_mid))
block_sparse_cache.scatter_mlp_output(layer_idx, token_indices, sparse_mlp_out)
mlp_output_full = block_sparse_cache.get_mlp_output(layer_idx)

# 8. Final output
hidden_states = mid + mlp_output_full
```

### FlashInfer Integration (`utils/attention_backends.py`, `generation_functions.py`)

Added FlashInfer as an attention backend, selectable via `FAST_DLLM_ATTENTION_BACKEND=flashinfer` or auto-detected when FA4/FA3/FA2 are unavailable (SM 7.5+).

#### Why FlashInfer

- **Wider hardware support**: SM 7.5+ (Turing: T4, RTX 2080, etc.) vs FA2's SM 8.0+ requirement
- **Paged KV cache natively**: FlashInfer's `BatchPrefillWithPagedKVCacheWrapper` handles variable-length KV sequences via `kv_last_page_len` — matches our `cache_seqlens` pattern
- **CUDA graph compatible**: `plan()` runs in eager code, `run()` can be inside CUDA graph capture
- **GQA native**: handles `num_qo_heads != num_kv_heads` internally (no manual head expansion)

#### Install

```bash
pip install flashinfer-python
# Optional pre-compiled kernels for faster startup:
pip install flashinfer-cubin
```

#### Layout Bridge (Zero-Copy)

FlashInfer uses NHD layout `(seq, heads, dim)` with ragged/paged batching — NOT BSHD. The key insight is that our static KV cache per-layer view is already memory-compatible:

```
Our BSHD:         [batch_size, max_seq_len, num_kv_heads, head_dim]
FlashInfer paged: [num_pages,  page_size,   num_kv_heads, head_dim]  (NHD)
```

By setting `page_size = max_seq_len` and `num_pages = batch_size`, these are **identical memory layouts**. An identity page table (batch element `i` → page `i`) completes the mapping. No data copy or transpose needed for KV cache tensors.

Query requires reshaping: BSHD `[B, S_q, H, D]` → ragged NHD `[B*S_q, H, D]` (contiguous reshape, not transpose).

#### Why BatchPrefill, Not BatchDecode

`BatchDecodeWithPagedKVCacheWrapper` only handles **single-token** queries per batch element. Our diffusion steps query `block_size` tokens (e.g., 32 or 512) against the full KV cache. `BatchPrefillWithPagedKVCacheWrapper` handles multi-token queries natively via `qo_indptr`.

#### plan() / run() Split

FlashInfer requires a two-phase API:
- **`plan()`**: sets up page table metadata, workspace allocation, kernel selection. Must be called from **eager code** (outside compiled graph / CUDA graph capture).
- **`run()`**: executes the attention kernel using the plan. Can be inside compiled graph / CUDA graph.

This maps naturally to our existing eager/compiled split:
```python
# Eager code (per-block setup, alongside set_kv_write_start / set_scratch_seqlens):
attn_backend.plan_flashinfer(batch_size, query_len=block_size, max_seq_len, ...)

# Compiled code (inside forward_fn → patched attention → flash_kvcache_attention):
out = self._flashinfer_run_fn(q_nhd, k_cache, v_cache)
```

#### FakeTensor / torch.compile Wrapping

FlashInfer's `wrapper.run()` is a CUDA kernel that accesses raw data pointers — same issue as `flash_attn_with_kvcache`. Under `torch.compile`, the FX graph executor runs nodes with FakeTensors to determine output shapes, causing "Cannot access data pointer" errors.

Solution: `_wrap_flashinfer_prefill_run_as_custom_op()` registers a `torch.library` custom op with:
- **CUDA impl**: calls `wrapper_holder[0].run(q, (k_cache, v_cache))`
- **Meta impl**: returns `torch.empty_like(q)` (output shape = query shape)

The `wrapper_holder` is a `list` so the wrapper object (updated by `plan()`) can be swapped without re-creating the custom op.

#### Implementation Details

**New functions in `attention_backends.py`:**
- `_try_import_flashinfer()` — SM 7.5+ hardware gate, imports `flashinfer` module, verifies `BatchPrefillWithPagedKVCacheWrapper` exists
- `_wrap_flashinfer_prefill_run_as_custom_op(wrapper_holder)` — custom op registration for torch.compile FakeTensor compat
- `AttentionBackend.plan_flashinfer(batch_size, query_len, max_seq_len, num_qo_heads, num_kv_heads, head_dim, kv_seqlens, dtype, device)` — lazy init of workspace (128MB), wrapper, identity page table; calls `wrapper.plan()`
- `AttentionBackend._flashinfer_kvcache_attention(query, k_cache, v_cache, cache_seqlens, ...)` — BSHD query → ragged NHD, calls wrapped `run()`, reshapes output back to BSHD
- `AttentionBackend._flashinfer_attention(q, k, v, ...)` — non-kvcache path using `BatchPrefillWithRaggedKVCacheWrapper` (batch>1) or `single_prefill_with_kv_cache` (batch=1)

**Integration in `generation_functions.py::batch_sample_dynamo()`:**
- Per-block eager setup (after `set_scratch_seqlens`): calls `attn_backend.plan_flashinfer(...)` with `query_len=block_size` and `kv_seqlens=static_cache.scratch_seqlens`
- After batch compaction: resets `attn_backend._flashinfer_wrapper = None` to force reinit with new batch size

**Lazy state on `AttentionBackend` instance:**
- `_flashinfer_wrapper` — `BatchPrefillWithPagedKVCacheWrapper` (persisted on model for CUDA graph reuse)
- `_flashinfer_workspace` — 128MB uint8 scratch buffer
- `_flashinfer_run_fn` — custom-op-wrapped `run()` callable
- `_flashinfer_wrapper_holder` — `[wrapper]` list for closure update
- `_flashinfer_qo_indptr` — `[0, block_size, 2*block_size, ..., B*block_size]` int32
- `_flashinfer_page_table` — `[[0], [1], ..., [B-1]]` int32 identity

#### Scope and Limitations

- **Supported in `batch_sample_dynamo` only** — the hot path for diffusion steps
- **Not integrated into `batch_sample_sparse`** — variable query lengths per sparse step would require per-step `plan()` calls, adding complexity
- **Prefill uses existing path** — `self.forward()` with SDPA/FA2, runs once, not a bottleneck
- **Auto-detection**: FlashInfer appears after FA2 in the `"auto"` fallback chain, so it won't override an available FA2 install. Use `FAST_DLLM_ATTENTION_BACKEND=flashinfer` to force it.

#### FlashInfer Issues & Fixes

| # | Error | Root Cause | Fix | File(s) |
|---|-------|-----------|-----|---------|
| 1 | `flash_attn_2_cuda.cpython-311: undefined symbol _ZN3c104cuda29c10_cuda_check_implementationE...` | Transformers' `modeling_flash_attention_utils.py` unconditionally imports `flash_attn` at module level when model's `modeling.py` imports `FlashAttentionKwargs`. The installed flash-attn binary is ABI-incompatible with torch 2.10. | Not a code fix — need to rebuild flash-attn from source or uninstall it. FlashInfer itself works fine; the crash happens at model load before any backend code runs. | `transformers/modeling_flash_attention_utils.py` (external) |
| 2 | `TypeError: 'NoneType' object is not callable` on `_flashinfer_run_fn` | Prefill `self.forward()` dispatches through patched attention layers → `flash_kvcache_attention()` → `_flashinfer_run_fn()`, but `plan_flashinfer()` was only called later in the per-block setup loop. | Added `plan_flashinfer()` call **before** the prefill in `batch_sample_dynamo()` with estimated prefill length. Added safety guard in `_flashinfer_kvcache_attention()` raising a clear error if `_flashinfer_run_fn is None`. | `generation_functions.py`, `attention_backends.py` |
| 3 | Logger `INFO` messages not appearing (only prints visible) | `logging.getLogger(__name__)` has no handlers by default and inherits WARNING level. Messages accepted but nowhere to output them. | Added explicit `logger.setLevel(logging.INFO)` + `StreamHandler` with timestamp formatter in both `attention_backends.py` and `generation_functions.py`. | `attention_backends.py`, `generation_functions.py` |
| 4 | `TypeError: Logger._log() got an unexpected keyword argument 'flush'` | When converting `print()` to `logger.info()`, the `flush=True` kwarg (print-only) was left in. | Removed the `flush=True` kwarg from the `logger.info()` call. | `generation_functions.py` |
| 5 | `ValueError: The dtype of q torch.bfloat16 does not match the q_data_type torch.float16 specified in plan function.` | FlashInfer's `wrapper.plan()` defaults to `q_data_type=torch.float16`, but the model runs in bfloat16. | Added `q_data_type=dtype` parameter to `wrapper.plan()` call in `plan_flashinfer()`. | `attention_backends.py` |
| 6 | `RuntimeError: Only a single TORCH_LIBRARY can be used to register the namespace fast_dllm_flashinfer` | Batch compaction (2→1) set `_flashinfer_wrapper = None`, triggering reinit which called `_wrap_flashinfer_prefill_run_as_custom_op()` a second time, re-registering the same `torch.library.Library` namespace. | Custom op created only on first init (`if self._flashinfer_wrapper_holder is None`). On reinit, just swap `self._flashinfer_wrapper_holder[0] = self._flashinfer_wrapper` — the custom op closure reads from the holder list, so the new wrapper is picked up without re-registration. | `attention_backends.py` |

### torch.compile / CUDA Graph Optimization (`utils/dynamo_utils.py`, `utils/static_kv_cache.py`, `utils/attention_backends.py`)

Optimized the `batch_sample_dynamo` compiled path from 57+ CUDA graph partitions / 9 graph breaks to **0 partitions / 0 graph breaks** with a single CUDA graph.

#### Key Fixes

1. **CPU f64 module attributes → CUDA tensors** (`generation_functions.py`): RMSNorm's `variance_epsilon` (Python float 1e-6) and similar module float attributes get lifted by dynamo as `f64[]` CPU tensor graph inputs. With 57 RMSNorm modules + 1 extra, this created 58 CPU ops (unsqueeze each, cat all, device_put to GPU) = 59 CUDA graph partitions. Fix: before compilation, iterate all modules and convert `variance_epsilon`, `attention_scaling`, `norm_type` from Python float to `torch.tensor(val, dtype=torch.float64, device=device)`. The `scaling` attribute is NOT converted because it's passed as a plain `float` to the flash attention custom op's `softmax_scale` parameter.

2. **List-of-tensor KV cache** (`static_kv_cache.py`): Changed from stacked tensor `[L, B, S, H, D]` with `.select(0, layer_idx)` to `list[Tensor]` (each `[B, S, H, D]`). Avoids `select_scatter` decomposition from AOT autograd functionalizing in-place mutations on `.select()` views. Note: the actual partitions were from f64 CPU attributes (issue #1 above), not from `select_scatter`.

3. **Accelerate hook removal** (`generation_functions.py`): `device_map={'': 'cuda:0'}` installs `AlignDevicesHook` on leaf modules, wrapping forwards with `torch.compiler.disable()`. Fix: call `accelerate.hooks.remove_hook_from_module()` before compilation.

3. **Logger graph break guard** (`attention_backends.py`): `logger.info()` in `flash_kvcache_attention()` caused 9 graph breaks ("Logger not supported for non-export cases"). Fix: added `torch.compiler.is_compiling()` guard — logging is skipped during dynamo tracing but works in eager mode.

4. **Enabled `fullgraph=True`** (`dynamo_utils.py`): With 0 graph breaks, the entire `forward_dynamo` (28 layers + embeddings + RoPE + lm_head) compiles as a single FX graph → single CUDA graph → best reduce-overhead performance.

#### Debug Infrastructure

| Variable | Values | Description |
|----------|--------|-------------|
| `FAST_DLLM_DEBUG_COMPILE` | `0`/`1`/`2` | `1`: compile time + step counter. `2`: auto-sets `TORCH_LOGS` for graph breaks, recompiles, cudagraphs |

| Function | Description |
|----------|-------------|
| `explain_forward(model, kwargs)` | Runs `torch._dynamo.explain()`, returns graph/break counts |
| `log_compile_summary()` | Logs compile time + step counts (called in `finally` block) |

See `dynamo_impl.md` for full details on all issues, fixes, and design decisions.

---

## Potential Improvements (Compile Path BS=1 Analysis, 2026-03-20)

Analysis of why CUDA graph replay at BS=1 isn't dramatically faster than eager. The compiled forward accounts for only ~37% of total sample time; the remaining 63% is always-eager Python code (sampling, unmasking, commit steps, prefill). Only 60/72 forward calls per sample benefit from graph replay (10 commit + 2 prefill are always eager).

### High Priority

| # | Improvement | Impact | Description |
|---|------------|--------|-------------|
| P1 | **Batch multiple diffusion steps into one compiled call** | High | Currently each diffusion step does: graph replay → eager sampling → eager unmasking → loop. This creates GPU bubbles between replays while CPU runs Python. Batching N steps into a single compiled function (forward + sampling + unmasking) would keep the GPU saturated and eliminate inter-replay gaps. Profiler shows compile has 14% gap fraction vs eager's ~5-10% (adjusted for profiler inflation). |
| P2 | **Compile the commit-step forward separately** | Medium | 10 commit steps per sample use `self.forward()` (always eager, ~18ms each = ~180ms total). A separate compiled path for commit steps (with `update_past_key_values=True`) could save significant time. Requires handling the pointer advancement (`_seq_len` update) inside or around the compiled graph. |

### Medium Priority

| # | Improvement | Impact | Description |
|---|------------|--------|-------------|
| P3 | **Replace scatter_ with custom `copy_at_offset` op** | Low-Med | `write_scratch_compiled()` uses `scatter_` with a full-shape `(B, seq_len, H, D)` int64 index tensor — 131KB per call at BS=1. The index-to-data ratio is 4:1 (131KB index to write 32KB of KV data). 56 scatter_ calls per forward = 7.2MB of redundant index reads. A `torch.library.custom_op` doing contiguous memcpy at a given offset would eliminate the index tensor entirely. scatter_ was chosen over `index_copy_` (which uses a 256-byte 1D index) because Inductor emits CPU-side dispatch for `index_copy_` with 1D indices, causing 56 CUDA graph partitions. |
| P4 | **Move logit shift inside compiled graph** | Low | `torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)` (line 648) runs in eager after each compiled forward. Moving it inside the compiled function would allow Triton to fuse it with the lm_head output write. Currently ~38μs per call × 60 = 2.3ms/sample (0.06%). Small individually but contributes to inter-replay bubble. |

### Low Priority / Confirmed Non-Issues

| # | Item | Status | Notes |
|---|------|--------|-------|
| P5 | `.contiguous()` after attn reshape | Likely no-op | `flash_attn_with_kvcache` returns contiguous BSHD; reshape merging last two dims of contiguous tensor is already contiguous. Could remove for clarity. (`attention_backends.py:964`) |
| P6 | RoPE `torch.cat` temporaries | Not a bottleneck | 56 cat ops per forward, but torch.compile fuses them into surrounding elementwise ops. Only relevant in eager mode. |
| P7 | `lm_head` computing all 32 positions | Not wasteful | Input is always `block_size=32` tokens. The GEMM is bandwidth-bound (reads 1.04GB weight matrix regardless). With `num_small_blocks=1`, all positions' logits are needed. |
| P8 | `_x_block_buf.copy_()` per forward | Negligible | 32 × int64 = 256 bytes D2D copy. ~5μs per call. |
| P9 | `prepare_write_idx` frequency | Not a bottleneck | Called once per block (not per diffusion step). The scatter index is constant within a block since write position doesn't change during diffusion. |

### Root Cause Summary (BS=1 Compile vs Eager)

Compile mode at BS=1 shows only ~5% improvement over eager because:
1. **Forward is only 37% of total time** — ceiling on improvement from graph replay
2. **Only 60/72 calls benefit** — 12 calls (commit + prefill) are always eager
3. **Inter-replay GPU bubbles** — CPU Python work between replays creates GPU idle time that doesn't exist in eager (where kernels are queued continuously)
4. **Triton kernel overhead** — fused Triton kernels are individually ~26% slower than native CUDA kernels (measured at BS=8), partially offsetting launch count reduction

---

## Accuracy Sweep Results (2026-03-24, commit_compare_20260324_025742)

GSM8K flexible-extract accuracy, 500 samples, block_size=32, threshold=0.9.

### Status of Previous Issues

| Issue | Status |
|-------|--------|
| Eager compact accuracy drop at BS≥2 (0.83→0.34) | **FIXED** — now consistent ~0.86 across all batch sizes |
| Nocommit crash (`_last_bridge_logits` not compacted) | **FIXED** — no crashes in any run |
| `batch_sample_sparse` `_scatter_idx` AttributeError | **FIXED** — `prepare_write_idx` added at line 1097 |

### Accuracy Table

| Mode | BS=1 | BS=2 | BS=4 | BS=8 | BS=16 |
|------|------|------|------|------|-------|
| **Commit eager** | 0.860 | 0.870 | 0.856 | 0.858 | 0.862 |
| **Commit compile** | 0.862 | 0.860 | 0.858 | **0.808** | **0.520** |
| **Nocommit eager** | 0.770 | 0.794 | 0.824 | 0.834 | 0.840 |
| **Nocommit compile** | 0.758 | 0.802 | 0.838 | **0.776** | **0.538** |

Bolded = significant accuracy drop vs expected ~0.86 baseline (commit) or ~0.82–0.84 (nocommit).

### New Issue 26 — Compile mode accuracy drop at BS≥8

**Symptom:** Compile compact mode produces wrong answers at BS=8 (0.808) and severely degraded at BS=16 (0.520). Eager mode is unaffected and consistent. Missing `\boxed{}` answers (incomplete generation):

| Mode | BS=4 | BS=8 | BS=16 |
|------|------|------|-------|
| Commit compile — missing answers | 7 | **46** | **229** |
| Commit eager | 7 | 6 | 8 |
| Nocommit compile | 15 | **65** | **233** |
| Nocommit eager | 11 | 8 | 6 |

Incomplete generations are not empty — they trail off mid-reasoning or enter repetition loops, indicating KV cache corruption or incorrect logits for some sequences.

**Root Cause (identified, not yet fixed):** During `warmup_compile_pools`, the warmup runs batch sizes in descending order (e.g. [8, 4, 2, 1] for BS=8). When the last size (BS=1) is compiled, a recompilation is triggered by this guard:

```
152064*logits_to_keep + 152064*input_ids.size()[1] < 152064*logits_to_keep*input_ids.size()[0] + 152064*input_ids.size()[1]*input_ids.size()[0]
```

Simplified: `V*(logits_to_keep + S) < V*(logits_to_keep + S)*B`, i.e. `B > 1`. This guard is baked in during the first compilation (BS=8) because AOT autograd encodes a size constraint on the `lm_head` output shape that requires `B > 1`. When BS=1 fails this guard, a new Dynamo specialization is compiled. The recompiled BS=1 graph has a `0` in its symint key (corresponding to `logits_to_keep=0` being treated as a dynamic value in the new trace), which is structurally different from the original graph.

**Why BS=4 is unaffected:** BS=4 valid_sizes = [1, 2, 4]. The same recompile for BS=1 happens, but the maximum batch size used in inference is 4 — the CUDA graph for BS=4 is identical in both the original and post-recompile cases since BS=4 still satisfies `B > 1`. With only 3 batch sizes, the number of sequences that ever use BS=8 is zero, so the degraded path is never hit.

**Why BS≥8 is affected:** With BS=8 valid_sizes = [1, 2, 4, 8], the CUDA graph for BS=8 is recorded during the first actual inference call (the pre-warmup provides only 1 warmup call, so the second call — the first real inference call — triggers CUDA graph capture). After the recompile changes the compiled function's internal state, the first BS=8 inference call captures a graph that produces incomplete/corrupted outputs for sequences in the batch. This affects all 8 sequences of the first eval batch (~46 out of 500 questions for BS=8, ~229 for BS=16 which has more batch sizes and more exposure).

**Fix Applied (2026-03-24):** Pass `logits_to_keep=block_size` explicitly in all compiled `forward_fn` calls and in the `warmup_compile_pools` warmup call.

- `generation_functions.py` lines 688-690: Full block path — `logits_to_keep=block_size`
- `generation_functions.py` lines 706: Sub-block path — `logits_to_keep=small_block_size`
- `generation_functions.py` lines 722-724: Standard compiled diffusion step — `logits_to_keep=block_size`
- `utils/dynamo_utils.py` lines 387-390: `warmup_compile_pools` warmup call — `logits_to_keep=block_size`

With a positive `logits_to_keep`, `slice(-block_size, None)` is always a well-defined non-special-case slice (last `block_size` tokens). Dynamo no longer needs to guard on `B > 1` to distinguish the lm_head output shape, so the B=1 recompile no longer produces a graph with a symbolic `logits_to_keep=0` that compiles to an empty slice.

**Note on nocommit eager accuracy scaling:** Nocommit eager accuracy increases with batch size (0.770 → 0.840). This is lower than the commit baseline (~0.86) as expected. The scaling effect is not fully understood but is not considered a bug — the nocommit path skips the commit forward pass (uses KV from last diffusion step directly), which may have better effective context coverage with larger batches due to interaction between the bridge token and the committed KV.

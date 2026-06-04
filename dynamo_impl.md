# torch.compile Optimization for `batch_sample_dynamo`

## Baseline (Before Optimization)

**Compile log**: `logs/gsm8k_0_.25_100000_1_compile_flashinfer_2.log`

| Metric | Value |
|--------|-------|
| CUDA graph partitions (non-gpu ops) | 56+ |
| CUDA graph partitions (DeviceCopy) | 1 |
| Total CUDA graph partitions | 57+ |
| Graph breaks | 9 (all from logger calls) |
| Compilation outcome | KeyboardInterrupt after ~85s |
| Compile mode | `reduce-overhead` with `fullgraph=False`, `dynamic=True` |
| Backend | FA2 (with kvcache) on SM 8.6 |

## After Optimization

| Metric | Value |
|--------|-------|
| CUDA graph partitions | **0** |
| Graph breaks | **0** |
| `fullgraph=True` | **Yes** (single graph) |
| First compiled call (tracing + codegen + CUDA graph record) | ~36s |
| Subsequent CUDA graph replays | Fast (~50 calls/3s) |
| Compilation outcome | **Success** |
| gsm8k accuracy (limit=10) | 70% flexible-extract |
| Throughput | 23.2 tokens/sec (includes compile warmup) |

---

## Issues & Fixes

| # | Issue | Root Cause | Fix | File(s) |
|---|-------|-----------|-----|---------|
| 1 | 59 "cudagraph partition due to non gpu ops" | **RMSNorm `variance_epsilon` (Python float) lifted as CPU f64 tensor graph inputs.** Dynamo traces `self.variance_epsilon` on each `Qwen2RMSNorm` module and lifts it as an `f64[]` CPU scalar graph input. 57 RMSNorm modules × 1 epsilon each = 57 CPU f64 scalars. Plus `attention_scaling` (1 more) = 58 total. These 58 CPU ops get unsqueezed, cat'd into `f64[58]`, then `device_put` to GPU — each unsqueeze and the cat are "non gpu ops" that partition the CUDA graph. **Not** caused by `scatter_`/`index_copy_` or accelerate hooks (previous hypothesis was wrong due to warm Inductor cache at `/tmp/torchinductor_n41/` masking partition warnings on re-runs). | **Convert float module attributes to CUDA f64 tensors before compilation.** In `batch_sample_dynamo`, iterate all modules and convert `variance_epsilon`, `attention_scaling`, `norm_type` from Python float to `torch.tensor(val, dtype=torch.float64, device=device)`. The `scaling` attribute is NOT converted because it's passed as a plain `float` to the flash attention custom op's `softmax_scale` parameter. After conversion, these values are already on GPU when dynamo traces, so no CPU ops or device transfers are needed. | `generation_functions.py` |
| 2 | 1 "cudagraph partition due to DeviceCopy ops" | Resolved alongside issue #1 — with all float attrs on CUDA, the `device_put_default` (CPU→GPU transfer of the concatenated f64 scalars) is eliminated. | Same as #1 | Same |
| 3 | 9 graph breaks (10 subgraphs) | `logger.info()` call inside `flash_kvcache_attention()` (line 595 of `attention_backends.py`). PyTorch dynamo does not support logger methods inside compiled regions — "Logger not supported for non-export cases". Each logger call in the attention dispatch path caused a graph break, propagating up through all 28 decoder layers. | Added `torch.compiler.is_compiling()` guard before the logger call. When dynamo is tracing the function, `is_compiling()` returns True and the logger block is skipped entirely. The logging still works in eager mode. | `utils/attention_backends.py` |
| 4 | `TensorFloat32 not enabled` warning | `torch.set_float32_matmul_precision` not called. | Added `torch.set_float32_matmul_precision('high')` in `make_compiled_forward()`. | `utils/dynamo_utils.py` |
| 5 | Compilation crash after ~85s | With 57+ CUDA graph partitions, Inductor had to generate and compile code for each partition separately. The massive number of partitions caused compilation to take extremely long (85s+) before being killed. | Fixing issues #1 and #3 eliminated all partitions, reducing compilation to a single graph. First compiled call now takes ~36s (one-time cost). | All files above |
| 6 | `fullgraph=False` prevented maximum optimization | With graph breaks present, `fullgraph=True` was not possible. | After fixing all graph breaks, switched to `fullgraph=True` in `_get_or_compile_fwd()`. Single graph → single CUDA graph → best reduce-overhead performance. | `utils/dynamo_utils.py` |
| 7 | `_kvcache_dispatch_logged` guard failure → recompilation | `mark_dispatch_logged()` was called after prefill, but CUDA graph warmup runs `compiled_fn()` before prefill. Dynamo traced with `_kvcache_dispatch_logged == False`, recorded it as a guard. Flag flipped to `True` during warmup execution. Next compiled call: guard fails → recompilation. Caused ~6s extra compilation + potentially suboptimal recompiled graph for all subsequent batches. | Moved `mark_dispatch_logged()` to immediately after `patch_attention_layers()`, before warmup. Dynamo's first trace now sees `True`. | `generation_functions.py` |
| 8 | Duck sizing: `num_kv_heads == batch_size` → recompilation on compact | `key_cache[0].size()[2]` (H=4 in BSHD) == `input_ids.size()[0]` (batch=4). Dynamo's duck sizing unified them. Batch 4→2: guard `H==batch` fails → recompile. Happened twice during warmup (4→2, 2→1), adding ~75s to batch 0. | Set `torch.fx.experimental._config.use_duck_shape = False` before compilation. Each dimension tracked independently. | `generation_functions.py` |
| 8a | (Failed attempt) Stacked tensor `[L,B,S,H,D]` to avoid duck sizing | Stacked tensor still had duck sizing (`key_cache.size()[3] == input_ids.size()[0]`). Worse: AOT autograd decomposed in-place mutations on `.select()` views into `select_scatter` — copies entire `[28,4,4096,4,128]` tensor per mutation. **2.3x slowdown** (64 vs 145 tok/s). | Reverted to list-of-tensor. Duck sizing fixed via `use_duck_shape = False` instead. | `utils/static_kv_cache.py` |

---

## Debug Infrastructure

### Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `FAST_DLLM_DEBUG_COMPILE` | `0` (default), `1`, `2` | Compile diagnostics level |
| Level 0 | — | No extra logging |
| Level 1 | — | Summary: graph count, compile time, step counter (every 50 calls) |
| Level 2 | — | Verbose: auto-sets `TORCH_LOGS=+dynamo,graph_breaks,recompiles,cudagraphs` |

### Diagnostic Functions

| Function | Location | Description |
|----------|----------|-------------|
| `explain_forward(model, kwargs)` | `utils/dynamo_utils.py` | Runs `torch._dynamo.explain()` on `forward_dynamo`, returns graph/break counts |
| `log_compile_summary()` | `utils/dynamo_utils.py` | Logs compile time + step counts; called from `batch_sample_dynamo`'s `finally` block |
| `get_compile_summary()` | `utils/dynamo_utils.py` | Returns summary string without logging |

### Test Script

`test_dynamo_explain.py` — standalone diagnostic that:
1. Loads the model + patches attention
2. Runs `torch._dynamo.explain()` to count graph breaks
3. Tests `fullgraph=True` compilation
4. Tests `reduce-overhead` mode with CUDA graph replay

---

## Key Design Decisions

### Why list-of-tensor instead of stacked tensor for KV cache

The original design used a stacked tensor `[L, B, S, H, D]` with `.select(0, layer_idx)` for per-layer access. This was changed to `list[Tensor]` (each `[B, S, H, D]`) because:

1. **Stacked tensor + `.select()` + in-place mutation** triggers AOT autograd's functionalization, which decomposes `scatter_` on a `.select()` view into `select` → `scatter` → `select_scatter`. The `select_scatter` creates a functional copy of the entire stacked tensor with one slice replaced — but this turned out to NOT be the actual partition cause.

2. **List-of-tensor** uses direct `self.key_cache[layer_idx]` access. Each per-layer tensor is standalone — `scatter_` compiles cleanly without `select_scatter` decomposition.

Note: The actual 59 CUDA graph partitions were caused by **CPU f64 scalar module attributes** (RMSNorm's `variance_epsilon`), not by `scatter_`/`index_copy_`/`select_scatter`. See Issue #1 above.

**Confirmed experimentally (2026-03-19)**: Switching to stacked tensor `[L,B,S,H,D]` caused a **2.3x slowdown** (64 vs 145 tok/s at BS=4) due to `select_scatter` decomposition. AOT autograd functionalizes `key_cache[layer_idx].scatter_(...)` (in-place mutation on a view) into a full-tensor copy. With `[28, 4, 4096, 4, 128]` bf16 tensors, that's ~3.5GB copied 56 times (28 layers × 2 K+V) per forward pass. List-of-tensor is the correct design.

### Why `scatter_` for KV cache writes

`scatter_(dim, full_shape_index, src)` compiles to a pure Triton GPU kernel. The scatter index `_scatter_idx` has shape `(B, block_size, H, D)` — same as `key_states`. Each element contains the target position along dim 1. Pre-computed from `_write_idx_fixed` via broadcast in eager code.

```python
# Eager code (before compiled step):
static_cache.prepare_write_idx(block_size)  # fills _scatter_idx

# Inside compiled graph:
self.key_cache[layer_idx].scatter_(1, self._scatter_idx, key_states)
```

### Why accelerate hooks must be removed

`device_map={'': 'cuda:0'}` (used by `accelerate launch`) installs `AlignDevicesHook` on leaf modules. These hooks wrap `module.forward` with `torch.compiler.disable()`, which causes Inductor to emit CPU-side dispatch ops — resulting in CUDA graph partitions even though dynamo successfully traces the graph (no graph breaks).

Since the model is already on the correct device (single GPU), these hooks are no-ops. Removing them via `accelerate.hooks.remove_hook_from_module()` before compilation is safe and eliminates the partitions.

```python
from accelerate.hooks import remove_hook_from_module
for module in model.modules():
    if hasattr(module, '_hf_hook'):
        remove_hook_from_module(module)
```

### Why `torch.compiler.is_compiling()` for logger guard

Dynamo cannot trace through Python's `logging` module — logger methods involve I/O, formatting, handler dispatch, etc. that are not tensor operations. When dynamo encounters `logger.info(...)`, it emits a graph break.

`torch.compiler.is_compiling()` returns `True` during dynamo tracing and `False` during normal execution. This lets us skip logging in the compiled path while keeping it in eager mode:

```python
if not self._kvcache_dispatch_logged and not torch.compiler.is_compiling():
    logger.info(...)
```

### Why `fullgraph=True`

With 0 graph breaks, the entire `forward_dynamo` (28 decoder layers + embeddings + RoPE + lm_head) is captured as a **single** FX graph. Benefits:
- Single Inductor compilation → single optimized kernel schedule
- Single CUDA graph → no partition overhead, no inter-partition sync
- Best possible reduce-overhead performance

---

## Performance Analysis: Compile vs Eager (2026-03-18)

### Setup
- GPU: NVIDIA RTX A5000 (SM 8.6, Ampere, 64 SMs)
- torch 2.9.1+cu128
- Batch size: 2, block_size: 32, max_seq_len: 4096
- Attention backend: FA2 with kvcache

### Results

| Mode | Per-step Latency | vs Eager |
|------|-----------------|----------|
| Eager (`forward_dynamo`) | **30.6 ms** | 1.00x |
| `torch.compile(mode='default')` | 59.6 ms | **1.95x slower** |
| `torch.compile(mode='reduce-overhead')` | 59.0 ms | **1.93x slower** |
| `torch.compile(mode='reduce-overhead', dynamic=False)` | 58.0 ms | **1.90x slower** |

### Root Cause: Inductor's Triton Kernels Are Slower Than Native CUDA

Profiling (`torch.profiler`) reveals:

| Operation | Eager (native CUDA) | Compiled (Triton fused) | Slowdown |
|-----------|---------------------|------------------------|----------|
| GEMMs (cuBLAS) | 118.6ms / 5 steps | 118.7ms / 5 steps | **1.00x** (same) |
| `index_copy_` (KV cache write) | 0.77ms total | 24.1ms (triton_poi_fused_copy_) | **31x** |
| Pointwise (mul, add, silu, layernorm) | ~13.3ms total | ~120ms (6 triton_poi_fused_*) | **9x** |

Key findings:
1. **GEMMs dominate GPU time (82%) and are identical** — both use cuBLAS, torch.compile can't improve them.
2. **Triton fused kernels for pointwise ops are 9-31x slower** than native CUDA elementwise kernels. Each `triton_poi_fused_*` takes ~2ms/call vs native ops at 2-4μs/call.
3. **"Not enough SMs to use max_autotune_gemm mode"** — A5000 has 64 SMs, below Inductor's threshold for autotuning.
4. **CUDA graphs add negligible benefit** — `reduce-overhead` vs `default` difference is <1ms (59.0 vs 59.6 ms).
5. **dynamic=True vs dynamic=False**: minimal difference (~1ms).

### Why Triton Kernels Are Slow on SM 8.6

1. **Full-buffer iteration**: Fused `copy_` iterates over the entire `(B=2, S=4096, H=4, D=128)` KV buffer, while native `index_copy_` only touches 32 target positions.
2. **SM 8.x Triton codegen limitations**: Triton's code generation is significantly less optimized for Ampere (SM 8.x) than Hopper (SM 9.0+).
3. **Grid/block size mismatch**: Triton's heuristic picks suboptimal configurations for small tensor dimensions at batch_size=2.

### Recommendation

**Do not use `torch.compile` for inference on RTX A5000 (SM 8.6) with this model.** Eager mode is ~2x faster. Compile mode may benefit on:
- H100/H200 (SM 9.0+): better Triton codegen + more SMs for autotuning
- Larger batch sizes: amortizes Triton kernel overhead over more data
- Future PyTorch/Triton versions with improved Ampere codegen

### Diagnostic Tools

- `test_compile_perf.py` — per-step latency comparison: eager vs compiled (default vs reduce-overhead vs dynamic=False)
- `test_compile_profile.py` — kernel-level torch.profiler traces (saved to `logs/profile_*.json`, viewable at ui.perfetto.dev)
- `FAST_DLLM_DEBUG_COMPILE=1` — per-step latency logging (added to `dynamo_utils.py`)

---

## Changes Log

| File | Change | Purpose |
|------|--------|---------|
| `utils/dynamo_utils.py` | Added `_CompileTimer`, `CompileStepCounter`, `explain_forward()`, `log_compile_summary()`, `_setup_debug_env()`. Changed `fullgraph=False` → `fullgraph=True`. Added `torch.set_float32_matmul_precision('high')`. | Debug infrastructure + CUDA graph optimization |
| `utils/static_kv_cache.py` | Added `prepare_write_idx()` with `_write_idx_fixed` + `_scatter_idx` tensors. Changed `write_scratch_compiled()` from `index_copy_` to `scatter_` (eliminates 56 partitions from Inductor's CPU dispatch logic for 1D index tensors). | Eliminate 56 CUDA graph partitions |
| `utils/attention_backends.py` | Added `torch.compiler.is_compiling()` guard on logger call in `flash_kvcache_attention()`. Added `_create_compiled_attention_forward()` (compile-optimized attention, no isinstance/boolean branches). Added `patch_attention_layers_compiled()`. | Eliminate 9 graph breaks + compile-optimized forward |
| `generation_functions.py` | Added `prepare_write_idx(block_size)` call in per-block setup. Added `log_compile_summary()` in finally block. Imported `patch_attention_layers_compiled`. Added `accelerate.hooks.remove_hook_from_module()` to strip accelerate hooks before compilation (eliminates partitions from `torch.compiler.disable()` wrappers). | Wire up fixes + accelerate hook removal |
| `test_dynamo_explain.py` | New test script for compile diagnostics. | Testing |
| `dynamo_impl.md` | This document. | Documentation |

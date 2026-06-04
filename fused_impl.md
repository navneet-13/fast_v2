# Fused Kernel Integration for Fast-dLLM v2

## Source: `/research/data/transfer/data/n41/fused_diffusion/`

## Overview

The fused_diffusion project provides persistent CUDA kernels (S1-S7) that replace the entire `self.forward()` call in the diffusion sampling loop. On A100, they achieve **1.75x** speedup over PyTorch eager (60 tok/s vs 34.2 tok/s). The kernels cover the full transformer pipeline: embedding (S1), RMSNorm+QKV projection (S2), GQA attention with RoPE (S3), O-projection+residual (S4), RMSNorm+gate/up projection (S5), SiLU*mul+down projection+residual (S6), final RMSNorm+lm_head (S7).

## Kernel Architecture

### Stage Pipeline

```
S1 (embed)  -->  28x [ S2 (norm+QKV) --> S3 (attn) --> S4 (O+res) --> S5 (norm+MLP_up) --> S6 (SiLU+MLP_down+res) ]  -->  S7 (norm+lm_head)
```

### Three Execution Modes

| Mode | Launches/step | Best for |
|------|--------------|----------|
| per-stage | 142 | B>=16 (stage-specific tuning) |
| fuse_layer | 30 (S1 + 28*FL + S7) | B<=8 (reduced launch overhead) |
| fuse_step | 1 (entire forward) | B<=8 (minimal launch overhead) |

### Key Kernel Properties

- **Batch dimension `B` = number of tokens** (block_size or small_block_size), NOT number of sequences
- **Compile-time B**: `constexpr int B = 8;` — one .so per B value
- **Supported B**: {1, 2, 4, 8, 16, 32}
- **KV cache layout**: `[seq_len, N_KV=4, HD=128]` per layer, **single-sequence only**
- **Weight fusion**: RMSNorm weights pre-fused into GEMM weights at init (S2, S5_v2, S7)
- **Cooperative launches**: `cudaLaunchCooperativeKernel` with `grid.sync()` barriers
- **SM target**: sm_80 (A100), forward-compatible with sm_86 (A5000)

### C API

```c
// Per-kernel init (dlopen .so, allocate meta tensors)
extern "C" void init_direct_persistent_kernel(
    std::vector<void*> meta_tensors, int unused1, long long unused2);

// Launch with tensor pointer array
extern "C" void launch_direct_persistent_kernel_array(
    void** tp, int n);
```

Multi-kernel management via `fast_launcher.cu`:
```c
int multi_init(int slot, const char* lib_path, int batch_size, int step_val);
int multi_launch(int slot, void** ptrs, int n);
```

## Integration Design for fast_v2

### Key Constraint: Kernels are Single-Sequence

The fused kernels process ONE sequence at a time. The `B` dimension is the token count (block_size), not the sequence batch size. KV caches are per-sequence `[seq_len, N_KV, HD]`.

**Implication for multi-sequence batching:**

| Sequence batch_size | Strategy | Expected perf vs PyTorch eager |
|---------------------|----------|-------------------------------|
| 1 | Direct kernel | ~2x faster |
| 2-4 | Loop over sequences | ~1x (kernel speedup offsets batching loss) |
| 8+ | Loop over sequences | Slower (sequential overhead dominates) |

### `batch_sample_fused` Design

Method signature matches `batch_sample_dynamo` for drop-in replacement.

#### Execution Flow

```
1. PyTorch prefill (self.forward with eval_mask) → DynamicCache
2. Convert DynamicCache → KernelRunner per-sequence caches
3. Block generation loop:
   a. Per-block setup
   b. Diffusion inner loop:
      - For each active sequence:
        - Copy block tokens to kernel input
        - Call kernel.forward_standard() or kernel.forward_blockcache()
        - Collect logits
      - Stack logits → batch tensor
      - Unmask + sample (same as batch_sample_dynamo)
   c. Block commit:
      - For each active sequence:
        - Call kernel.forward_extend()
      - Bridge token generation
4. Return finished_samples via BatchBucket
```

#### Cache Management

The KernelRunner maintains its own KV cache per layer in `[seq_len, N_KV, HD]` format.
- **Prefill**: After PyTorch prefill, convert `past_key_values` to kernel format via `init_main_cache()`
- **Diffusion steps**: Use `forward_standard()` — writes scratch KV internally, doesn't commit
- **Block commit**: Use `forward_extend()` — extends main cache
- **No conversion back to StaticKVCache** — stays in kernel cache format for entire generation

#### Per-Sequence KernelRunner State

For batch_size > 1, we need per-sequence cache state. Options:
1. **Single KernelRunner, swap caches** — save/restore `kernel_main_k/v` per sequence
2. **Multiple KernelRunners** — one per sequence (heavy: duplicates all weights)
3. **Single KernelRunner, per-sequence cache arrays** — maintain lists of caches, swap on use

Option 1 is best: single runner (shared weights/buffers), swap cache pointers per sequence.

```python
# Per-sequence cache storage
seq_caches = [
    {'main_k': [None]*28, 'main_v': [None]*28, 'main_len': 0}
    for _ in range(batch_size)
]

# Before kernel call for sequence i:
for l in range(28):
    runner.kernel_main_k[l] = seq_caches[i]['main_k'][l]
    runner.kernel_main_v[l] = seq_caches[i]['main_v'][l]
runner.main_cache_len = seq_caches[i]['main_len']
```

### Compilation Requirements

Kernels must be compiled before use:
```bash
cd /research/data/transfer/data/n41/fused_diffusion

# Generate batch variants from B=8 templates
python python/generate_batch_variants.py

# Compile all kernels (sm_86 for A5000)
NVCC_FLAGS="-O3 --shared -Xcompiler -fPIC --expt-relaxed-constexpr -arch=sm_86 -std=c++17 \
  -I include -I include/mirage/persistent_kernel \
  -I include/mirage/persistent_kernel/tasks \
  -I include/mirage/persistent_kernel/tasks/common"
for cu in generated/kernel_stage*.cu generated/kernel_fused_*.cu; do
  nvcc $NVCC_FLAGS -o "${cu%.cu}.so" "$cu"
done
nvcc -o tests/fast_launcher.so --shared -Xcompiler -fPIC -ldl -arch=sm_86 tests/fast_launcher.cu
```

### Environment Variables

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `FAST_DLLM_USE_FUSED` | `0` / `1` | `0` | Enable fused kernel path |
| `FAST_DLLM_FUSED_MODE` | `per_stage` / `fuse_layer` / `fuse_step` | `per_stage` | Kernel execution mode |
| `FAST_DLLM_FUSED_DIR` | path | `../fused_diffusion` | Path to fused_diffusion root |

### Performance Notes (from fused_diffusion benchmarks, A100)

**Per-step latency (ms), single sequence:**

| B (tokens) | Kernel | PyTorch eager | Speedup |
|------------|--------|---------------|---------|
| 8 | 11.65 | 24.61 | 2.11x |
| 32 | 13.73 | 24.36 | 1.77x |

**Per-stage breakdown (B=8, us):**

| Stage | Kernel | PyTorch | Speedup |
|-------|--------|---------|---------|
| S1 embed | 5.5 | 18.0 | 3.3x |
| S2 QKV proj | 26.6 | 95.0 | 3.6x |
| S3 attention | 19.8 | 232.0 | 11.7x |
| S4 O-proj+res | 20.1 | 237.0 | 11.8x |
| S5+S6 MLP | 302.6 | 312.0 | 1.0x |
| S7 lm_head | 724.5 | 750.0 | 1.0x |

Key insight: **S3+S4 (attention) gets 10-12x speedup** because the kernel fuses RoPE+GQA+output_proj into cooperative persistent kernels. **MLP and lm_head are memory-bandwidth-bound** and roughly match PyTorch.

### A5000 Considerations

- Kernels compiled for sm_80 run on sm_86 (forward compat), but sm_86-specific compilation (`-arch=sm_86`) may improve register allocation
- A5000 has 64 SMs vs A100's 108 — cooperative kernel grid sizes may need adjustment for optimal occupancy
- A5000 HBM bandwidth is ~768 GB/s vs A100's ~2 TB/s — bandwidth-bound stages (S5, S6, S7) will be proportionally slower
- Expected overall speedup on A5000: ~1.3-1.5x (vs 1.75x on A100)

## Issues & Fixes

| # | Error | Root Cause | Fix | File |
|---|-------|-----------|-----|------|
| 1 | `OSError: fast_launcher.so: cannot open shared object file` | Kernels not compiled before first run. `KernelRunner.__init__` calls `ctypes.CDLL(LAUNCHER_SO)` immediately. | Compile kernels per the instructions in the Compilation Requirements section above. One-time setup. | `fused_diffusion/tests/fast_launcher.cu` (external) |
| 2 | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 518 MiB` during `_extract_weights` (float32 fusions on GPU) | `_extract_weights` fused RMSNorm weights into GEMM weights by casting large weight matrices to float32 on GPU (e.g. `gate_up_w.float()` = `[2×18944, 3584]` × float32 ≈ 518 MiB). With a 7B model already occupying ~20.86 GiB of a 22 GiB A5000, only ~500 MiB is free. | Moved float32 fusion intermediates to CPU: `.cpu().float() * ...` then `.to(torch.bfloat16).contiguous().to(self.device)`. Applied to `lm_head_w_mod`, `gate_up_w_mod`, and `qkv_w_mod`. | `fused_diffusion/kernel_runner.py` |
| 3 | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 260 MiB` during `_extract_weights` (`torch.cat` for `gate_up_w`) | Pre-storing `gate_up_w` and `gate_up_w_mod` as persistent GPU tensors for all 28 layers = ~577 MB × 28 = **~16 GB** extra on top of the ~14.8 GB model (~31 GB total, exceeds 22 GiB A5000). After issue #2 fix, earlier layers' `qkv_w_mod`/`gate_up_w_mod` accumulated on GPU, leaving only 72 MiB free by layer ~10. | **Lazy per-layer computation**: removed `gate_up_w` and `gate_up_w_mod` from `layer_weights`. Replaced with `gate_proj_w` / `up_proj_w` (direct references to model weights, zero extra VRAM). `_run_pipeline` and `_run_pipeline_fused_layer` now compute the gate tensor just-in-time per layer inside the forward loop — one layer's ~272 MB temp exists at a time, freed when the loop variable is reassigned. `_build_weight_array` (fuse_step mode) materialises all 28 layers simultaneously (same old OOM risk on A5000) but is not used by the default `per_stage` eval mode. Persistent VRAM footprint reduced from ~16 GB to **0 GB** for gate tensors. | `fused_diffusion/kernel_runner.py` |
| 4 | All generated tokens are `!` (token ID 0 repeated ~2048 times); ~121 s/sample | **Root cause: cooperative kernel shared-memory requirements exceed sm_86 (A5000) hardware limit.** Kernels S2, S5, S7 (b=8: 108 KB; b=32: 124 KB) and S3 (108 KB) all request dynamic shared memory that exceeds the **99 KB per-block max** on sm_86 (RTX A5000). `init_direct_persistent_kernel` calls `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize, 108/124 KB)` which returns `cudaErrorInvalidValue` on A5000 — but the return value is **not checked**, so init appears to succeed. Subsequently `cudaLaunchCooperativeKernel(..., smem=108/124 KB)` silently fails (also unchecked), leaving all output buffers at their initial zero values. With logits_buf = zeros: `argmax(zeros) = 0` → token ID 0 (`!` in Qwen2 tokenizer) is sampled for every position every step. The grid sizes (GRID=108) are also tuned for A100's 108 SMs; A5000 has 64 SMs, making 108-block cooperative launches non-fitting even without the shared-memory issue. Additionally, lazy `gate_up_w_mod` computation inside the layer loop (28 × 271 MB allocations per forward pass × many diffusion steps) explains the 121 s/sample even before any kernel correctness issues. | **The fused kernel path requires sm_80 (A100) or better.** The kernels hardcode 108-CTA cooperative launches with 108–124 KB shared memory per block, which only fits within A100's 164 KB-per-block limit and 108-SM capacity. **Do not use `FAST_DLLM_USE_FUSED=1` on A5000 (sm_86)**; use `FAST_DLLM_USE_DYNAMO=1` instead. To run fused on A5000, the kernels would need to be redesigned with ≤96 KB shared memory and GRID ≤ 64 (or dynamic grid sizing). See Implementation Status TODO. | `fused_diffusion/kernel_runner.py`, `fused_diffusion/generated/kernel_stage*.cu` |

---

## Implementation Status

### TODO
- [ ] **Redesign kernels for sm_86**: reduce shared memory to ≤96 KB/block and grid to ≤64 CTAs (required for A5000 — current kernels only work on A100/sm_80)
- [ ] Correctness validation vs `batch_sample_dynamo` eager (requires A100 or sm_86-compatible kernels)
- [ ] Benchmark on A100: single sequence throughput (A5000 is blocked by issue #4)
- [ ] Benchmark on A100: batch_size=1 vs 2 vs 4

### Completed
- [x] Implement `batch_sample_fused` in `generation_functions.py`
- [x] Add KernelRunner import and initialization
- [x] Implement per-sequence cache swap for batch_size > 1
- [x] Wire into `eval.py` via `FAST_DLLM_USE_FUSED=1`
- [x] Fix OOM during weight extraction (issues #2, #3)
- [x] Identify root cause of gibberish `!` output on A5000 (issue #4)

### Hardware Compatibility

| GPU | SM | Max shared/block | GRID=108 cooperative | Fused path |
|-----|-----|-----------------|----------------------|-----------|
| A100 | sm_80 | 164 KB | ✓ (108 SMs) | **Supported** |
| RTX A5000 | sm_86 | 99 KB | ✗ (64 SMs, 108 KB/124 KB > 99 KB) | **Not supported** |
| RTX 3080 Ti | sm_86 | 99 KB | ✗ | **Not supported** |

### Future Work
- Multi-sequence kernel support (batch across sequences, not just tokens)
- Kernel+torch.compile hybrid: kernel for attention, compile for MLP
- A5000-tuned kernel grid sizes (64 SMs)
- Block cache support in fused path (`forward_blockcache`)

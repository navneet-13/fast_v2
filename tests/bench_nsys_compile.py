#!/usr/bin/env python3
"""
Nsys profiling: Eager vs Compiled forward pass.

Profiles a single config (BS=8, past_blocks=10) under NVIDIA Nsight Systems.
Uses CUDA profiler start/stop markers so only the measured iterations appear
in the nsys timeline.

Usage (automatic — launches nsys for both modes):
    cd /path/to/fast_v2
    CUDA_VISIBLE_DEVICES=3 python tests/bench_nsys_compile.py

Manual (run one mode at a time):
    nsys profile -c cudaProfilerApi -f true -o tests/nsys_eager \
        python tests/bench_nsys_compile.py --worker --mode eager
    nsys profile -c cudaProfilerApi -f true -o tests/nsys_compiled \
        python tests/bench_nsys_compile.py --worker --mode compiled
"""

import argparse
import os
import subprocess
import sys
import shutil

FAST_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(FAST_V2_ROOT, "tests")

MODEL_PATH = os.environ.get(
    "FAST_DLLM_MODEL_PATH",
    os.path.join(
        FAST_V2_ROOT,
        "models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/"
        "snapshots/0661abf5f9f0ee338970d091052a26c8efa51974",
    ),
)
BLOCK_SIZE = 32
BATCH_SIZE = 8
PAST_BLOCKS = 10
N_WARMUP = 5
N_MEASURE = 5


def worker_main(mode):
    """Run forward pass with CUDA profiler markers for nsys capture."""
    import time
    import torch

    sys.path.insert(0, FAST_V2_ROOT)
    from transformers import AutoModelForCausalLM
    from utils.static_kv_cache import StaticKVCache
    from utils.attention_backends import get_attention_backend, patch_attention_layers
    from utils.dynamo_utils import make_compiled_forward, make_eager_forward, warmup_compile_pools

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    print(f"[{mode}] GPU: {gpu_name}")
    print(f"[{mode}] BS={BATCH_SIZE}, past_blocks={PAST_BLOCKS}, "
          f"past_seq_len={PAST_BLOCKS * BLOCK_SIZE}")

    # ── Load model ──
    print(f"[{mode}] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # ── Cache + buffers ──
    config = model.config
    past_len = PAST_BLOCKS * BLOCK_SIZE
    max_seq_len = ((past_len + BLOCK_SIZE + 255) // 256) * 256

    cache = StaticKVCache(
        batch_size=BATCH_SIZE, max_seq_len=max_seq_len,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=model.dtype, device=device,
    )
    x_buf = torch.zeros(BATCH_SIZE, BLOCK_SIZE, dtype=torch.long, device=device)
    cache_pos_buf = torch.zeros(BLOCK_SIZE, dtype=torch.long, device=device)
    cache_pos_arange = torch.arange(BLOCK_SIZE, dtype=torch.long, device=device)

    # ── Attention backend ──
    attn_backend = get_attention_backend("auto")
    print(f"[{mode}] Attention backend: {attn_backend.name}")
    patch_attention_layers(model, attn_backend, None)
    model._static_kv_cache = cache

    # Move float attrs to CUDA
    for module in model.modules():
        for attr_name in ('variance_epsilon', 'attention_scaling', 'norm_type'):
            val = getattr(module, attr_name, None)
            if isinstance(val, float):
                setattr(module, attr_name, torch.tensor(val, dtype=torch.float64, device=device))

    # ── Create forward function ──
    if mode == "compiled":
        import torch.fx.experimental._config as fx_config
        fx_config.use_duck_shape = False
        forward_fn = make_compiled_forward(model, compile_mode="reduce-overhead")
        torch._dynamo.mark_static_address(x_buf)
        torch._dynamo.mark_static_address(cache_pos_buf)
        torch._dynamo.mark_static_address(cache_pos_arange)
        attn_backend.mark_dispatch_logged()

        print(f"[{mode}] Recording CUDA graph (under torch.no_grad)...")
        # Record under no_grad to match inference — avoids grad_mode guard
        # recompilation when the benchmark loop also uses no_grad.
        torch.set_grad_enabled(False)
        warmup_compile_pools(
            compiled_fn=forward_fn,
            compile_pools={BATCH_SIZE: {"cache": cache, "x_buf": x_buf}},
            model=model, block_size=BLOCK_SIZE,
            cache_pos_buf=cache_pos_buf, cache_pos_arange=cache_pos_arange,
            attn_backend=attn_backend, config=config, max_seq_len=max_seq_len,
            dtype=model.dtype, device=device,
        )
        print(f"[{mode}] CUDA graph recorded.")
    else:
        forward_fn = make_eager_forward(model)
        attn_backend.mark_dispatch_logged()

    # ── Set cache state ──
    cache.zero_and_reset()
    cache._seq_len = past_len
    cache.cache_seqlens.fill_(past_len)
    cache.set_kv_write_start(past_len)
    cache.set_scratch_seqlens(past_len + BLOCK_SIZE)
    cache.prepare_write_idx(BLOCK_SIZE)
    cache_pos_buf.copy_(cache_pos_arange + past_len)
    x_buf.copy_(torch.randint(0, 1000, (BATCH_SIZE, BLOCK_SIZE), device=device))

    # ── Warmup (outside profiler) ──
    print(f"[{mode}] Warming up ({N_WARMUP} iters)...")
    with torch.no_grad():
        for _ in range(N_WARMUP):
            if mode == "compiled":
                torch.compiler.cudagraph_mark_step_begin()
            _ = forward_fn(
                input_ids=x_buf, use_cache=True, past_key_values=cache,
                update_past_key_values=False, cache_position=cache_pos_buf,
                logits_to_keep=BLOCK_SIZE,
            )
    torch.cuda.synchronize()

    # ── Profiled iterations ──
    print(f"[{mode}] Starting profiled region ({N_MEASURE} iters)...")
    torch.cuda.cudart().cudaProfilerStart()

    with torch.no_grad():
        for i in range(N_MEASURE):
            # NVTX range per iteration for easy identification in nsys UI
            torch.cuda.nvtx.range_push(f"{mode}_iter_{i}")

            if mode == "compiled":
                torch.compiler.cudagraph_mark_step_begin()

            output = forward_fn(
                input_ids=x_buf, use_cache=True, past_key_values=cache,
                update_past_key_values=False, cache_position=cache_pos_buf,
                logits_to_keep=BLOCK_SIZE,
            )
            torch.cuda.synchronize()

            torch.cuda.nvtx.range_pop()

    torch.cuda.cudart().cudaProfilerStop()
    print(f"[{mode}] Profiling done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Physical GPU id (after CUDA_VISIBLE_DEVICES remapping)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["eager", "compiled"], help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker_main(args.mode)
        return

    # ── Check nsys is available ──
    nsys_path = shutil.which("nsys")
    if nsys_path is None:
        print("ERROR: 'nsys' not found in PATH. Install NVIDIA Nsight Systems or add it to PATH.")
        print("You can also run manually:")
        print(f"  nsys profile -c cudaProfilerApi -f true -o {TESTS_DIR}/nsys_eager \\")
        print(f"      {sys.executable} {__file__} --worker --mode eager")
        print(f"  nsys profile -c cudaProfilerApi -f true -o {TESTS_DIR}/nsys_compiled \\")
        print(f"      {sys.executable} {__file__} --worker --mode compiled")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    for mode in ("eager", "compiled"):
        output_path = os.path.join(TESTS_DIR, f"nsys_{mode}")
        print(f"\n{'='*70}")
        print(f"  Profiling {mode.upper()} mode")
        print(f"  Output: {output_path}.nsys-rep")
        print(f"{'='*70}\n")

        cmd = [
            nsys_path, "profile",
            "-c", "cudaProfilerApi",       # only capture between profilerStart/Stop
            "--cuda-graph-trace=node",     # trace individual kernels inside CUDA graphs
            "-f", "true",                   # overwrite existing
            "-o", output_path,
            "--stats=false",                # skip post-processing stats (faster)
            "-t", "cuda,nvtx,osrt",         # trace CUDA + NVTX + OS runtime
            sys.executable, __file__,
            "--worker", "--mode", mode,
        ]

        print(f"Running: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            print(f"\n[{mode}] nsys exited with code {result.returncode}")
        else:
            print(f"\n[{mode}] Profile saved: {output_path}.nsys-rep")

    print(f"\n{'='*70}")
    print(f"  Done. Open in Nsight Systems UI:")
    print(f"    nsys-ui {TESTS_DIR}/nsys_eager.nsys-rep")
    print(f"    nsys-ui {TESTS_DIR}/nsys_compiled.nsys-rep")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

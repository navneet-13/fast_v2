#!/usr/bin/env python3
"""
Kernel-level profiling: eager vs compiled forward.

Generates a Chrome trace (.json) showing exactly which kernels run
and how long each takes. Compare eager vs compiled to find which
operations are slower under Inductor.

Usage:
    CUDA_VISIBLE_DEVICES=0 python test_compile_profile.py
    # Opens traces in chrome://tracing or ui.perfetto.dev
"""

import os
import torch
from torch.profiler import profile, ProfilerActivity, schedule

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__),
                 "models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/"
                 "snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/"),
)
BATCH_SIZE = 2
BLOCK_SIZE = 32
MAX_SEQ_LEN = 4096
MASK_ID = 151665
PAST_LEN = 32


def load_and_setup():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils.static_kv_cache import StaticKVCache
    from utils.attention_backends import get_attention_backend, patch_attention_layers

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    config = model.config
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads

    static_cache = StaticKVCache(
        batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
        num_layers=config.num_hidden_layers, num_kv_heads=num_kv_heads,
        head_dim=head_dim, dtype=model.dtype, device=model.device,
    )
    attn_backend = get_attention_backend(os.environ.get("FAST_DLLM_ATTENTION_BACKEND", "auto"))
    patch_attention_layers(model, attn_backend)
    model._static_kv_cache = static_cache

    return model, static_cache


def make_input(model, static_cache):
    static_cache.set_kv_write_start(PAST_LEN)
    static_cache.set_scratch_seqlens(PAST_LEN + BLOCK_SIZE)
    static_cache.prepare_write_idx(BLOCK_SIZE)
    input_ids = torch.full((BATCH_SIZE, BLOCK_SIZE), MASK_ID, dtype=torch.long, device=model.device)
    cache_position = torch.arange(PAST_LEN, PAST_LEN + BLOCK_SIZE, device=model.device, dtype=torch.long)
    return dict(
        input_ids=input_ids, use_cache=True, past_key_values=static_cache,
        update_past_key_values=False, cache_position=cache_position,
    )


def profile_mode(model, static_cache, forward_fn, label, n_warmup=5, n_profile=5):
    """Profile forward calls and save trace."""
    # Warmup
    for _ in range(n_warmup):
        kwargs = make_input(model, static_cache)
        with torch.no_grad():
            _ = forward_fn(**kwargs)
        torch.cuda.synchronize()

    # Profile
    trace_path = f"logs/profile_{label}.json"
    os.makedirs("logs", exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        schedule=schedule(wait=1, warmup=1, active=n_profile, repeat=1),
    ) as prof:
        for _ in range(n_profile + 2):  # wait + warmup + active
            kwargs = make_input(model, static_cache)
            with torch.no_grad():
                _ = forward_fn(**kwargs)
            torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(trace_path)
    print(f"\n  Trace saved to: {trace_path}")

    # Print top CUDA kernels by total time
    print(f"\n  Top 20 CUDA kernels ({label}):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Print kernel-type summary
    print(f"\n  Kernel type summary ({label}):")
    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="cuda_time_total", row_limit=30,
    ))


def main():
    print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  device={torch.cuda.get_device_name()}")
    model, static_cache = load_and_setup()

    # 1. Profile eager
    fwd_eager = getattr(model, "forward_dynamo", model.forward)
    print("\n=== Profiling EAGER ===")
    profile_mode(model, static_cache, fwd_eager, "eager")

    # 2. Profile compiled (default mode)
    print("\n=== Compiling (mode=default) ===")
    torch._dynamo.reset()
    static_cache.zero_and_reset()
    compiled_default = torch.compile(fwd_eager, mode="default", fullgraph=True, dynamic=True)
    # Warmup compilation
    for _ in range(3):
        kwargs = make_input(model, static_cache)
        with torch.no_grad():
            _ = compiled_default(**kwargs)
        torch.cuda.synchronize()
    print("=== Profiling COMPILED (default) ===")
    profile_mode(model, static_cache, compiled_default, "compiled_default", n_warmup=3)

    # 3. Profile compiled (reduce-overhead)
    print("\n=== Compiling (mode=reduce-overhead) ===")
    torch._dynamo.reset()
    static_cache.zero_and_reset()
    torch.set_float32_matmul_precision("high")
    compiled_ro = torch.compile(fwd_eager, mode="reduce-overhead", fullgraph=True, dynamic=True)

    def compiled_ro_step(**kwargs):
        torch.compiler.cudagraph_mark_step_begin()
        return compiled_ro(**kwargs)

    # Warmup compilation + CUDA graph recording
    for _ in range(5):
        kwargs = make_input(model, static_cache)
        with torch.no_grad():
            _ = compiled_ro_step(**kwargs)
        torch.cuda.synchronize()
    print("=== Profiling COMPILED (reduce-overhead) ===")
    profile_mode(model, static_cache, compiled_ro_step, "compiled_reduce_overhead", n_warmup=3)

    print("\nDone! Compare traces at ui.perfetto.dev")


if __name__ == "__main__":
    main()

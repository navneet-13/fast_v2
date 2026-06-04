#!/usr/bin/env python3
"""
Compile-mode performance diagnostic for Fast-dLLM v2.

Measures per-step latency for compiled vs eager forward calls,
detects CUDA graph partitions, and reports whether replays are
actually faster than eager execution.

Usage:
    # Basic comparison (10 warmup + 50 measured steps):
    CUDA_VISIBLE_DEVICES=0 python test_compile_perf.py

    # With verbose CUDA graph logging:
    TORCH_LOGS="+cudagraphs" CUDA_VISIBLE_DEVICES=0 python test_compile_perf.py

    # With graph break analysis:
    FAST_DLLM_DEBUG_COMPILE=2 CUDA_VISIBLE_DEVICES=0 python test_compile_perf.py

    # Test with dynamic=False (to see if that eliminates partitions):
    COMPILE_DYNAMIC=0 CUDA_VISIBLE_DEVICES=0 python test_compile_perf.py
"""

import os
import sys
import time
import json
import torch
import numpy as np

# ---- Configuration ----
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__),
                 "models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/"
                 "snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/"),
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
BLOCK_SIZE = 32
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "4096"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "10"))
MEASURE_STEPS = int(os.environ.get("MEASURE_STEPS", "50"))
COMPILE_MODE = os.environ.get("COMPILE_MODE", "reduce-overhead")
COMPILE_DYNAMIC = os.environ.get("COMPILE_DYNAMIC", "1") == "1"

MASK_ID = 151665


def load_model():
    """Load model (same as eval.py)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, tokenizer


def setup_caches(model, batch_size, block_size, max_seq_len):
    """Set up static KV cache + attention patching."""
    from utils.static_kv_cache import StaticKVCache
    from utils.attention_backends import (
        get_attention_backend,
        patch_attention_layers,
    )

    config = model.config
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads

    static_cache = StaticKVCache(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=model.dtype,
        device=model.device,
    )

    attn_backend = get_attention_backend(
        os.environ.get("FAST_DLLM_ATTENTION_BACKEND", "auto")
    )
    patch_attention_layers(model, attn_backend)

    model._static_kv_cache = static_cache
    return static_cache, attn_backend, num_kv_heads, head_dim


def make_sample_input(model, static_cache, batch_size, block_size, past_len=0):
    """Create a sample input for forward_dynamo.

    Mirrors the actual call in batch_sample_dynamo: only the current block
    (block_size tokens) is passed as input_ids, not the full sequence.
    The KV cache holds the prior context.
    """
    # Only the current block tokens (as in batch_sample_dynamo line 519)
    input_ids = torch.full(
        (batch_size, block_size),
        MASK_ID,
        dtype=torch.long,
        device=model.device,
    )

    # Set up cache control signals (normally done in eager per-block setup)
    static_cache.set_kv_write_start(past_len)
    static_cache.set_scratch_seqlens(past_len + block_size)
    static_cache.prepare_write_idx(block_size)

    cache_position = torch.arange(
        past_len, past_len + block_size,
        device=model.device, dtype=torch.long,
    )

    return dict(
        input_ids=input_ids,
        use_cache=True,
        past_key_values=static_cache,
        update_past_key_values=False,
        cache_position=cache_position,
    )


def time_forward_calls(forward_fn, make_input_fn, n_warmup, n_measure, label):
    """Time forward calls with CUDA synchronization."""
    latencies = []

    print(f"\n{'='*60}")
    print(f"  {label}: {n_warmup} warmup + {n_measure} measured steps")
    print(f"{'='*60}")

    # Warmup
    for i in range(n_warmup):
        kwargs = make_input_fn()
        torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.no_grad():
            _ = forward_fn(**kwargs)
        torch.cuda.synchronize()
        t1 = time.monotonic()
        ms = (t1 - t0) * 1000
        if i < 5 or (i + 1) == n_warmup:
            print(f"  warmup step {i+1}: {ms:.1f}ms")

    # Measure
    for i in range(n_measure):
        kwargs = make_input_fn()
        torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.no_grad():
            _ = forward_fn(**kwargs)
        torch.cuda.synchronize()
        t1 = time.monotonic()
        ms = (t1 - t0) * 1000
        latencies.append(ms)

    arr = np.array(latencies)
    print(f"\n  Results ({n_measure} steps):")
    print(f"    Mean:   {arr.mean():.2f} ms")
    print(f"    Median: {np.median(arr):.2f} ms")
    print(f"    P5:     {np.percentile(arr, 5):.2f} ms")
    print(f"    P95:    {np.percentile(arr, 95):.2f} ms")
    print(f"    Min:    {arr.min():.2f} ms")
    print(f"    Max:    {arr.max():.2f} ms")
    print(f"    Std:    {arr.std():.2f} ms")

    return latencies


def check_dynamo_counters():
    """Print torch._dynamo counters."""
    import torch._dynamo as dynamo
    counters = dynamo.utils.counters
    print("\n  Dynamo counters:")
    for category in ["stats", "frames"]:
        if category in counters:
            for k, v in sorted(counters[category].items()):
                if v:
                    print(f"    {category}.{k} = {v}")


def main():
    print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  "
          f"device={torch.cuda.get_device_name()}")
    print(f"Config: batch={BATCH_SIZE}  block={BLOCK_SIZE}  "
          f"max_seq={MAX_SEQ_LEN}  compile_mode={COMPILE_MODE}  "
          f"dynamic={COMPILE_DYNAMIC}")

    model, tokenizer = load_model()
    static_cache, attn_backend, num_kv_heads, head_dim = setup_caches(
        model, BATCH_SIZE, BLOCK_SIZE, MAX_SEQ_LEN
    )

    # ---- 1. Eager baseline ----
    fwd_eager = getattr(model, "forward_dynamo", model.forward)

    def make_eager_input():
        static_cache.zero_and_reset()
        return make_sample_input(model, static_cache, BATCH_SIZE, BLOCK_SIZE, past_len=32)

    eager_lats = time_forward_calls(
        fwd_eager, make_eager_input,
        n_warmup=5, n_measure=MEASURE_STEPS,
        label="EAGER forward_dynamo",
    )

    # ---- 2. torch.compile WITHOUT reduce-overhead (inductor only) ----
    print("\n\nCompiling with mode='default' (no CUDA graphs)...")
    torch._dynamo.reset()
    static_cache.zero_and_reset()
    compiled_default = torch.compile(
        fwd_eager, mode="default", fullgraph=True, dynamic=COMPILE_DYNAMIC,
    )

    def make_compiled_input():
        static_cache.zero_and_reset()
        kwargs = make_sample_input(model, static_cache, BATCH_SIZE, BLOCK_SIZE, past_len=32)
        return kwargs

    default_lats = time_forward_calls(
        compiled_default, make_compiled_input,
        n_warmup=WARMUP_STEPS, n_measure=MEASURE_STEPS,
        label="COMPILED mode='default' (inductor, no CUDA graphs)",
    )
    check_dynamo_counters()

    # ---- 3. torch.compile WITH reduce-overhead (CUDA graphs) ----
    print(f"\n\nCompiling with mode='{COMPILE_MODE}' (CUDA graphs)...")
    torch._dynamo.reset()
    static_cache.zero_and_reset()

    torch.set_float32_matmul_precision("high")
    compiled_ro = torch.compile(
        fwd_eager, mode=COMPILE_MODE, fullgraph=True, dynamic=COMPILE_DYNAMIC,
    )

    def compiled_ro_step(**kwargs):
        torch.compiler.cudagraph_mark_step_begin()
        return compiled_ro(**kwargs)

    ro_lats = time_forward_calls(
        compiled_ro_step, make_compiled_input,
        n_warmup=WARMUP_STEPS, n_measure=MEASURE_STEPS,
        label=f"COMPILED mode='{COMPILE_MODE}' (CUDA graphs)",
    )
    check_dynamo_counters()

    # ---- 4. Optionally test with dynamic=False ----
    if COMPILE_DYNAMIC:
        print(f"\n\nCompiling with mode='{COMPILE_MODE}', dynamic=False...")
        torch._dynamo.reset()
        static_cache.zero_and_reset()

        compiled_static = torch.compile(
            fwd_eager, mode=COMPILE_MODE, fullgraph=True, dynamic=False,
        )

        def compiled_static_step(**kwargs):
            torch.compiler.cudagraph_mark_step_begin()
            return compiled_static(**kwargs)

        static_lats = time_forward_calls(
            compiled_static_step, make_compiled_input,
            n_warmup=WARMUP_STEPS, n_measure=MEASURE_STEPS,
            label=f"COMPILED mode='{COMPILE_MODE}', dynamic=False",
        )
        check_dynamo_counters()

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    eager_med = np.median(eager_lats)
    default_med = np.median(default_lats)
    ro_med = np.median(ro_lats)
    print(f"  Eager median:                   {eager_med:.2f} ms")
    print(f"  Compiled (default) median:      {default_med:.2f} ms  ({default_med/eager_med:.2f}x eager)")
    print(f"  Compiled (reduce-overhead) med: {ro_med:.2f} ms  ({ro_med/eager_med:.2f}x eager)")
    if COMPILE_DYNAMIC:
        static_med = np.median(static_lats)
        print(f"  Compiled (RO, dynamic=F) med:   {static_med:.2f} ms  ({static_med/eager_med:.2f}x eager)")

    if ro_med > eager_med:
        print(f"\n  ⚠ reduce-overhead is {ro_med/eager_med:.2f}x SLOWER than eager!")
        print(f"  Likely cause: CUDA graph partitions from dynamic index_copy_")
        print(f"  Try: TORCH_LOGS='+cudagraphs' to see partition details")
        print(f"  Try: COMPILE_DYNAMIC=0 to test with static shapes")


if __name__ == "__main__":
    main()

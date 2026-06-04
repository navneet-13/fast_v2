#!/usr/bin/env python3
"""
Measure WHERE time goes inside batch_sample_dynamo's diffusion loop,
at multiple batch sizes (compile mode only).

For each diffusion step, times three phases:
  1. forward_fn  — the compiled graph replay (or eager forward_dynamo)
  2. sync        — GPU->CPU synchronizations (.sum()==0, .any(), .item(), etc.)
  3. sample      — everything else (sampling, unmasking, loop logic)

This answers: "At BS=8, is the bottleneck still syncs, or does
the forward dominate enough that syncs don't matter?"

Usage:
    cd fast_v2
    python debug/bench_sync_overhead.py --batch_sizes 1,2,4,8 --limit 16
"""

import os
import sys
import time
import argparse
import torch
import json
import types
import functools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/"
    "snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/"
)

MASK_ID = 151665
STOP_TOKEN = 151645
BLOCK_SIZE = 32
MAX_NEW_TOKENS = 512


# ---------------------------------------------------------------------------
# Sync instrumentation
# ---------------------------------------------------------------------------
class SyncTracker:
    """Counts and times every GPU->CPU sync caused by tensor ops."""

    def __init__(self):
        self.reset()
        self._patched = False

    def reset(self):
        self.count = 0
        self.total_ns = 0
        self.by_op = {}

    def _record(self, op_name, duration_ns):
        self.count += 1
        self.total_ns += duration_ns
        self.by_op[op_name] = self.by_op.get(op_name, 0) + duration_ns

    def patch(self):
        """Monkey-patch torch.Tensor methods that cause GPU->CPU sync."""
        if self._patched:
            return
        tracker = self

        self._orig_item = torch.Tensor.item
        self._orig_nonzero = torch.Tensor.nonzero
        self._orig_bool = torch.Tensor.__bool__

        def timed_item(self_t):
            if self_t.is_cuda:
                t0 = time.perf_counter_ns()
                r = tracker._orig_item(self_t)
                tracker._record("item", time.perf_counter_ns() - t0)
                return r
            return tracker._orig_item(self_t)

        def timed_nonzero(self_t, **kw):
            if self_t.is_cuda:
                t0 = time.perf_counter_ns()
                r = tracker._orig_nonzero(self_t, **kw)
                tracker._record("nonzero", time.perf_counter_ns() - t0)
                return r
            return tracker._orig_nonzero(self_t, **kw)

        def timed_bool(self_t):
            if self_t.is_cuda:
                t0 = time.perf_counter_ns()
                r = tracker._orig_bool(self_t)
                tracker._record("__bool__", time.perf_counter_ns() - t0)
                return r
            return tracker._orig_bool(self_t)

        torch.Tensor.item = timed_item
        torch.Tensor.nonzero = timed_nonzero
        torch.Tensor.__bool__ = timed_bool
        self._patched = True

    def unpatch(self):
        if not self._patched:
            return
        torch.Tensor.item = self._orig_item
        torch.Tensor.nonzero = self._orig_nonzero
        torch.Tensor.__bool__ = self._orig_bool
        self._patched = False


# ---------------------------------------------------------------------------
# Forward call timer (wraps forward_fn)
# ---------------------------------------------------------------------------
class ForwardTimer:
    """Wraps forward_fn to measure wall time of each call."""

    def __init__(self):
        self.total_ns = 0
        self.count = 0

    def wrap(self, forward_fn):
        timer = self
        @functools.wraps(forward_fn)
        def timed_forward(**kwargs):
            t0 = time.perf_counter_ns()
            result = forward_fn(**kwargs)
            # NOTE: no cuda.synchronize() here — we measure wall time
            # including async launch. The sync cost shows up when Python
            # later reads a GPU tensor (.sum(), .any(), etc.)
            timer.total_ns += time.perf_counter_ns() - t0
            timer.count += 1
            return result
        return timed_forward


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import generation_functions

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_dynamo, model
    )
    return model, tokenizer


def get_gsm8k_prompts(limit=20):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    return [ex["question"] for i, ex in enumerate(ds) if i < limit]


def make_batched_inputs(prompts, tokenizer, batch_size, device):
    batches = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in batch
        ]
        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)
        input_ids = inputs.input_ids
        seq_len = (input_ids != tokenizer.pad_token_id).sum(dim=1)
        min_len = seq_len.min().item()
        batches.append((input_ids, seq_len, min_len))
    return batches


def reset_model_state(model):
    for attr in (
        '_static_kv_cache', '_compile_pools', '_static_block_cache',
        '_dynamo_attn_backend', '_dynamo_original_forwards',
        '_dynamo_forward_fn', '_x_block_buf', '_cache_pos_buf',
        '_cache_pos_arange', '_batch_mode', '_dynamo_floats_moved',
        '_dynamo_hooks_removed',
    ):
        if hasattr(model, attr):
            delattr(model, attr)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Instrumented generation
# ---------------------------------------------------------------------------
def run_instrumented(model, tokenizer, batches, execution_mode):
    """
    Run batch_sample_dynamo with forward timer + sync tracker.

    We monkey-patch model._dynamo_forward_fn AFTER batch_sample_dynamo
    initializes it (first batch triggers _need_reinit). To do this, we
    wrap batch_sample_dynamo itself to intercept forward_fn assignment.
    """
    os.environ["FAST_DLLM_USE_DYNAMO"] = "1"
    os.environ["FAST_DLLM_USE_SPARSE"] = "0"
    os.environ["FAST_DLLM_EXECUTION_MODE"] = execution_mode
    os.environ["FAST_DLLM_ATTENTION_BACKEND"] = "auto"
    os.environ["FAST_DLLM_MAX_SEQ_LEN"] = "2048"
    os.environ["FAST_DLLM_COMPILE_MODE"] = "reduce-overhead"
    os.environ["FAST_DLLM_BATCH_MODE"] = "static"

    fwd_timer = ForwardTimer()
    sync_tracker = SyncTracker()

    # We need to wrap forward_fn AFTER batch_sample_dynamo creates it.
    # Strategy: run the first batch to trigger init, then wrap, then
    # reset timers and run remaining batches for measurement.

    # --- Warmup batch (triggers init + compile if needed) ---
    warmup_batch = batches[:1]
    with torch.no_grad():
        model.batch_sample_dynamo(
            input_ids=warmup_batch[0][0],
            tokenizer=tokenizer,
            block_size=BLOCK_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            small_block_size=BLOCK_SIZE,
            min_len=warmup_batch[0][2],
            seq_len=warmup_batch[0][1],
            mask_id=MASK_ID, threshold=0.9, stop_token=STOP_TOKEN,
            use_block_cache=False, top_p=0.95, temperature=0.0,
        )

    # --- Wrap forward_fn ---
    orig_forward_fn = model._dynamo_forward_fn
    model._dynamo_forward_fn = fwd_timer.wrap(orig_forward_fn)

    # --- Measured batches ---
    sync_tracker.patch()
    try:
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        total_tokens = 0
        total_steps = 0

        for input_ids, seq_len, min_len in batches[1:]:
            with torch.no_grad():
                results = model.batch_sample_dynamo(
                    input_ids=input_ids,
                    tokenizer=tokenizer,
                    block_size=BLOCK_SIZE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    small_block_size=BLOCK_SIZE,
                    min_len=min_len,
                    seq_len=seq_len,
                    mask_id=MASK_ID, threshold=0.9, stop_token=STOP_TOKEN,
                    use_block_cache=False, top_p=0.95, temperature=0.0,
                )
            for r in results.values():
                total_tokens += r.shape[-1]

        torch.cuda.synchronize()
        wall_time = time.perf_counter() - wall_start
    finally:
        sync_tracker.unpatch()
        model._dynamo_forward_fn = orig_forward_fn

    fwd_ms = fwd_timer.total_ns / 1e6
    sync_ms = sync_tracker.total_ns / 1e6
    wall_ms = wall_time * 1e3
    other_ms = wall_ms - fwd_ms - sync_ms

    return {
        "batch_size": batches[0][0].shape[0] if batches else 0,
        "execution_mode": execution_mode,
        "wall_ms": wall_ms,
        "forward_ms": fwd_ms,
        "forward_pct": fwd_ms / wall_ms * 100 if wall_ms > 0 else 0,
        "sync_ms": sync_ms,
        "sync_pct": sync_ms / wall_ms * 100 if wall_ms > 0 else 0,
        "other_ms": other_ms,
        "other_pct": other_ms / wall_ms * 100 if wall_ms > 0 else 0,
        "forward_calls": fwd_timer.count,
        "sync_count": sync_tracker.count,
        "sync_by_op": {k: v / 1e6 for k, v in sync_tracker.by_op.items()},
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / wall_time if wall_time > 0 else 0,
        "avg_forward_ms": fwd_ms / fwd_timer.count if fwd_timer.count > 0 else 0,
        "avg_sync_ms": sync_ms / sync_tracker.count if sync_tracker.count > 0 else 0,
        "measured_batches": len(batches) - 1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile sync overhead in batch_sample_dynamo at multiple batch sizes")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8",
                        help="Comma-separated batch sizes")
    parser.add_argument("--limit", type=int, default=16,
                        help="Number of GSM8K prompts")
    parser.add_argument("--mode", type=str, default="eager",
                        choices=["eager", "compile"],
                        help="Execution mode for batch_sample_dynamo")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print("Loading model (once)...")
    model, tokenizer = load_model()
    print(f"Model: {model.dtype}, {model.device}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Mode: {args.mode}")
    print(f"Limit: {args.limit} prompts")

    prompts = get_gsm8k_prompts(limit=args.limit)
    print(f"Loaded {len(prompts)} prompts\n")

    all_results = []

    for bs in batch_sizes:
        print(f"{'='*75}")
        print(f"  BS={bs}  mode={args.mode}  prompts={len(prompts)}")
        print(f"{'='*75}")

        reset_model_state(model)
        batches = make_batched_inputs(prompts, tokenizer, bs, model.device)
        print(f"  {len(batches)} batches ({len(batches)-1} measured, 1 warmup)")

        if len(batches) < 2:
            print(f"  SKIPPED: need at least 2 batches (increase --limit or decrease BS)")
            continue

        result = run_instrumented(model, tokenizer, batches, args.mode)
        result["batch_size"] = bs
        all_results.append(result)

        print(f"  tokens/sec:      {result['tokens_per_sec']:.1f}")
        print(f"  wall time:       {result['wall_ms']:.0f} ms")
        print(f"  ├─ forward:      {result['forward_ms']:.0f} ms  ({result['forward_pct']:.1f}%)  "
              f"[{result['forward_calls']} calls, {result['avg_forward_ms']:.2f} ms/call]")
        print(f"  ├─ sync:         {result['sync_ms']:.0f} ms  ({result['sync_pct']:.1f}%)  "
              f"[{result['sync_count']} syncs, {result['avg_sync_ms']:.2f} ms/call]")
        print(f"  └─ other:        {result['other_ms']:.0f} ms  ({result['other_pct']:.1f}%)")
        if result['sync_by_op']:
            print(f"  sync breakdown:  ", end="")
            parts = [f"{op}={ms:.0f}ms" for op, ms in sorted(
                result['sync_by_op'].items(), key=lambda x: -x[1])]
            print("  ".join(parts))
        print()

    # ---- Summary ----
    if all_results:
        print(f"\n{'='*90}")
        print("SUMMARY: Time breakdown by batch size")
        print(f"{'='*90}")
        print(f"{'BS':>4} {'tok/s':>7} {'wall':>8} {'fwd%':>6} {'sync%':>6} "
              f"{'other%':>7} {'fwd_calls':>10} {'syncs':>7} {'avg_fwd':>9} {'avg_sync':>9}")
        print("-" * 85)
        for r in all_results:
            print(f"{r['batch_size']:>4} {r['tokens_per_sec']:>7.1f} "
                  f"{r['wall_ms']:>7.0f}ms {r['forward_pct']:>5.1f}% "
                  f"{r['sync_pct']:>5.1f}% {r['other_pct']:>6.1f}% "
                  f"{r['forward_calls']:>10} {r['sync_count']:>7} "
                  f"{r['avg_forward_ms']:>8.2f}ms {r['avg_sync_ms']:>8.2f}ms")

        print(f"\nKey question: Does sync% shrink as BS grows?")
        if len(all_results) >= 2:
            first = all_results[0]
            last = all_results[-1]
            print(f"  BS={first['batch_size']}: sync={first['sync_pct']:.1f}%  "
                  f"forward={first['forward_pct']:.1f}%")
            print(f"  BS={last['batch_size']}: sync={last['sync_pct']:.1f}%  "
                  f"forward={last['forward_pct']:.1f}%")
            if first['sync_pct'] > 0:
                ratio = last['sync_pct'] / first['sync_pct']
                print(f"  Sync overhead ratio BS={last['batch_size']}/BS={first['batch_size']}: {ratio:.2f}x")

    out_path = os.path.join(os.path.dirname(__file__), "bench_sync_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

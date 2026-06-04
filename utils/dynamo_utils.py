"""
torch.compile and CUDA graph utilities for Fast-dLLM v2.

Key design patterns (learned from Token_spase-dllm):
- Module-level _COMPILED_FWD_CACHE: same OptimizedModule across all batches
  → same CUDA graph tree → graphs from batch N reused in batch N+1
- cudagraph_mark_step_begin() before each compiled call
- Per-step tensors as explicit dynamic function args (not module state)
  → CUDA graph trees D2D-memcpy current values, no version-counter checks

Debug mode (FAST_DLLM_DEBUG_COMPILE):
  0 = off (default)
  1 = summary: graph count, break reasons, compile time, per-step timing
  2 = verbose: full dynamo explain output, per-break detail, TORCH_LOGS auto-set
  3 = per-step: log every compiled call with latency (warning: very verbose)
"""

import os
import time
import logging
import torch
import torch._dynamo as dynamo
from typing import Callable, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    ))
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Debug compile level
# ---------------------------------------------------------------------------
_DEBUG_COMPILE = int(os.environ.get("FAST_DLLM_DEBUG_COMPILE", "0"))
torch._logging.set_logs(cudagraphs=True, recompiles=True, graph_breaks=True)



def _setup_debug_env():
    """Configure torch logging env vars based on debug level.
    Must be called BEFORE torch.compile() to take effect."""
    if _DEBUG_COMPILE >= 2:
        # Verbose: graph breaks + recompilation + CUDA graphs + inductor
        os.environ.setdefault("TORCH_LOGS", "+dynamo,graph_breaks,recompiles,cudagraphs")
        os.environ.setdefault("TORCHDYNAMO_VERBOSE", "1")
        logger.info("[debug_compile] TORCH_LOGS=%s  TORCHDYNAMO_VERBOSE=%s",
                     os.environ.get("TORCH_LOGS"), os.environ.get("TORCHDYNAMO_VERBOSE"))
    elif _DEBUG_COMPILE >= 1:
        # Summary: graph breaks + cudagraphs only
        os.environ.setdefault("TORCH_LOGS", "graph_breaks,cudagraphs")
        logger.info("[debug_compile] TORCH_LOGS=%s", os.environ.get("TORCH_LOGS"))


# Set up env vars at import time so they're available before compile()
_setup_debug_env()


# ---------------------------------------------------------------------------
# Compilation diagnostics
# ---------------------------------------------------------------------------

def explain_forward(model, sample_kwargs: dict) -> dict:
    """
    Run torch._dynamo.explain() on the model's forward_dynamo to report
    graph breaks without actually compiling.

    Returns a dict with:
      - graph_count: number of subgraphs
      - graph_break_count: number of breaks
      - break_reasons: list of (reason, user_stack) tuples

    Usage:
        from utils.dynamo_utils import explain_forward
        info = explain_forward(model, dict(
            input_ids=x, use_cache=True, past_key_values=cache, ...
        ))
        print(f"Graphs: {info['graph_count']}, breaks: {info['graph_break_count']}")
    """
    fwd = getattr(model, "forward_dynamo", model.forward)
    explanation = dynamo.explain(fwd)(**sample_kwargs)

    result = {
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "break_reasons": explanation.break_reasons,
    }

    if _DEBUG_COMPILE >= 1:
        logger.info("[explain] graph_count=%d  graph_break_count=%d",
                     result["graph_count"], result["graph_break_count"])
        for i, reason in enumerate(result["break_reasons"]):
            logger.info("[explain]   break #%d: %s", i, reason)

    return result


# ---------------------------------------------------------------------------
# Compilation timing wrapper
# ---------------------------------------------------------------------------

class _CompileTimer:
    """Track compilation and execution timing."""

    def __init__(self):
        self.compile_start = None
        self.first_call_start = None
        self.first_call_done = False
        self.compile_time_s = 0.0
        self.first_call_time_s = 0.0

    def mark_compile_start(self):
        self.compile_start = time.monotonic()

    def mark_compile_end(self):
        if self.compile_start is not None:
            self.compile_time_s = time.monotonic() - self.compile_start
            self.compile_start = None
            if _DEBUG_COMPILE >= 1:
                logger.info("[compile_timer] torch.compile() took %.2fs", self.compile_time_s)

    def mark_first_call_start(self):
        self.first_call_start = time.monotonic()

    def mark_first_call_end(self):
        if self.first_call_start is not None and not self.first_call_done:
            self.first_call_time_s = time.monotonic() - self.first_call_start
            self.first_call_done = True
            if _DEBUG_COMPILE >= 1:
                logger.info("[compile_timer] First compiled call (includes tracing + codegen) "
                             "took %.2fs", self.first_call_time_s)

    def summary(self) -> str:
        return (f"compile={self.compile_time_s:.2f}s  "
                f"first_call={self.first_call_time_s:.2f}s")


_compile_timer = _CompileTimer()


# ---------------------------------------------------------------------------
# Step counter for compile mode
# ---------------------------------------------------------------------------

class CompileStepCounter:
    """Count compiled vs eager calls with per-step latency tracking."""

    def __init__(self):
        self.compiled_calls = 0
        self.eager_calls = 0
        self._log_interval = 50  # log every N calls at level 1
        # Per-step latency tracking
        self._step_latencies: list = []
        self._step_start: Optional[float] = None

    def begin_step(self):
        """Call BEFORE each compiled forward call."""
        if _DEBUG_COMPILE >= 1:
            torch.cuda.synchronize()
            self._step_start = time.monotonic()

    def end_step(self):
        """Call AFTER each compiled forward call."""
        if _DEBUG_COMPILE >= 1 and self._step_start is not None:
            torch.cuda.synchronize()
            elapsed_ms = (time.monotonic() - self._step_start) * 1000
            self._step_latencies.append(elapsed_ms)
            self._step_start = None

    def count_compiled(self):
        self.compiled_calls += 1
        if _DEBUG_COMPILE >= 3:
            # Per-step logging
            lat = self._step_latencies[-1] if self._step_latencies else 0
            logger.info("[step %d] compiled_call  latency=%.1fms", self.compiled_calls, lat)
        elif _DEBUG_COMPILE >= 1 and self.compiled_calls % self._log_interval == 0:
            logger.info("[step_counter] compiled_calls=%d  eager_calls=%d",
                         self.compiled_calls, self.eager_calls)

    def count_eager(self):
        self.eager_calls += 1

    def latency_summary(self) -> str:
        """Return latency statistics string."""
        if not self._step_latencies:
            return "no latency data"
        lats = self._step_latencies
        # Skip first N calls (warmup/compilation)
        warmup = min(3, len(lats))
        steady = lats[warmup:] if len(lats) > warmup else lats
        if not steady:
            steady = lats
        avg = sum(steady) / len(steady)
        mn = min(steady)
        mx = max(steady)
        p50 = sorted(steady)[len(steady) // 2]
        first_lat = lats[0] if lats else 0
        return (f"first={first_lat:.1f}ms  steady(n={len(steady)}): "
                f"avg={avg:.1f}ms  p50={p50:.1f}ms  "
                f"min={mn:.1f}ms  max={mx:.1f}ms")

    def summary(self) -> str:
        return f"compiled={self.compiled_calls}  eager={self.eager_calls}"


_step_counter = CompileStepCounter()


# ---------------------------------------------------------------------------
# Module-level compiled function cache
# ---------------------------------------------------------------------------
_COMPILED_FWD_CACHE: dict = {}


def _get_or_compile_fwd(func, mode: str = "reduce-overhead") -> Callable:
    """
    Return a compiled version of `func`, reusing across calls.
    Same (func, mode) key → same OptimizedModule → same CUDA graph tree.
    """
    key = (func, mode)
    if key not in _COMPILED_FWD_CACHE:
        _compile_timer.mark_compile_start()
        # fullgraph=True: verified that forward_dynamo has 0 graph breaks
        # after fixing logger calls (torch.compiler.is_compiling() guard).
        # Single graph → single CUDA graph → best reduce-overhead perf.
        # dynamic=True: handles varying cache positions across blocks.
        _COMPILED_FWD_CACHE[key] = torch.compile(
            func, mode=mode, fullgraph=True, dynamic=True,
        )
        _compile_timer.mark_compile_end()
    return _COMPILED_FWD_CACHE[key]


def make_compiled_forward(model, compile_mode="reduce-overhead") -> Callable:
    """
    Return a compiled forward function for diffusion steps.

    Uses model.forward_dynamo (inference-only, skips eval_mask).
    The compiled function is cached at module level so the same CUDA graph
    tree is reused across all batches.

    Per-step changing tensors (input_ids etc.) are function arguments.
    KV cache is accessed via past_key_values arg (traces from
    model._static_kv_cache → module state → mutations are interior).
    """
    fwd = getattr(model, "forward_dynamo", model.forward)

    # Enable TF32 for better matmul performance (suppresses the warning too)
    torch.set_float32_matmul_precision("high")
    # torch._logging.set_logs(recompiles=True, graph_breaks=True)

    compiled = _get_or_compile_fwd(fwd, mode=compile_mode)

    def _compiled_step(**kwargs):
        torch.compiler.cudagraph_mark_step_begin()

        if not _compile_timer.first_call_done:
            _compile_timer.mark_first_call_start()

        _step_counter.begin_step()
        result = compiled(**kwargs)
        _step_counter.end_step()

        if not _compile_timer.first_call_done:
            _compile_timer.mark_first_call_end()

        _step_counter.count_compiled()
        return result

    return _compiled_step


def make_eager_forward(model) -> Callable:
    """Return the uncompiled forward (forward_dynamo if available)."""
    fwd = getattr(model, "forward_dynamo", model.forward)

    def _eager_step(**kwargs):
        _step_counter.count_eager()
        return fwd(**kwargs)

    return _eager_step


# ---------------------------------------------------------------------------
# Diagnostics summary (call at end of generation)
# ---------------------------------------------------------------------------

def get_compile_summary() -> str:
    """Return a one-line summary of compile diagnostics."""
    return f"[compile_summary] {_compile_timer.summary()}  {_step_counter.summary()}"


def get_latency_summary() -> str:
    """Return per-step latency statistics."""
    return f"[latency] {_step_counter.latency_summary()}"


def log_compile_summary():
    """Log the compile diagnostics summary."""
    if _DEBUG_COMPILE >= 1:
        logger.info(get_compile_summary())
        logger.info(get_latency_summary())


# ---------------------------------------------------------------------------
# CUDA graph partition diagnostic
# ---------------------------------------------------------------------------

def warmup_compile_pools(
    compiled_fn: Callable,
    compile_pools: dict,
    model,
    block_size: int,
    cache_pos_buf: torch.Tensor,
    cache_pos_arange: torch.Tensor,
    attn_backend=None,
    config=None,
    max_seq_len: int = 4096,
    dtype=None,
    device=None,
):
    """
    Pre-record CUDA graphs for each valid batch size by running one
    compiled forward per pool.

    Must be called AFTER attention layers are patched and forward_fn is created.
    Runs largest batch first (allocates the most CUDA graph memory early).
    """
    sorted_sizes = sorted(compile_pools.keys(), reverse=True)
    total_kv_bytes = 0
    for bs in sorted_sizes:
        cache = compile_pools[bs]["cache"]
        total_kv_bytes += (
            cache.num_layers * 2 * bs * cache.max_seq_len
            * cache.num_kv_heads * cache.head_dim
            * (2 if cache.dtype == torch.bfloat16 else 4)
        )
    logger.info(
        "[warmup] Pre-recording CUDA graphs for %d batch sizes: %s  "
        "(total KV pool memory: %.1f GB)",
        len(sorted_sizes), sorted_sizes, total_kv_bytes / 1e9,
    )

    for bs in sorted_sizes:
        pool = compile_pools[bs]
        cache = pool["cache"]
        x_buf = pool["x_buf"]

        cache.zero_and_reset()
        cache.set_kv_write_start(0)
        cache.set_scratch_seqlens(block_size)
        cache.prepare_write_idx(block_size)

        x_buf.fill_(0)
        cache_pos_buf.copy_(cache_pos_arange)

        # Point model to this pool's cache
        model._static_kv_cache = cache

        # FlashInfer plan if needed
        if attn_backend is not None and attn_backend.name == "flashinfer":
            attn_backend.plan_flashinfer(
                batch_size=bs,
                query_len=block_size,
                max_seq_len=max_seq_len,
                num_qo_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                kv_seqlens=cache.scratch_seqlens,
                dtype=dtype,
                device=device,
            )

        try:
            _ = compiled_fn(
                input_ids=x_buf,
                use_cache=True,
                past_key_values=cache,
                update_past_key_values=False,
                cache_position=cache_pos_buf,
                # Explicit logits_to_keep avoids slice(-0, None) special-case
                # when compiled: prevents empty-logits bug in B=1 specialization.
                # Must match the value used in inference forward_fn calls.
                logits_to_keep=block_size,
            )
            logger.info("[warmup] Recorded graph for batch_size=%d", bs)
        except Exception as e:
            logger.warning("[warmup] Failed for batch_size=%d: %s", bs, e)

        cache.zero_and_reset()

    # Restore the largest pool as current
    max_bs = sorted_sizes[0]
    model._static_kv_cache = compile_pools[max_bs]["cache"]
    logger.info("[warmup] Done pre-recording CUDA graphs")


def diagnose_cudagraph_partitions(compiled_fn, sample_kwargs: dict) -> dict:
    """
    Diagnose CUDA graph partitions by inspecting the compiled function's
    internal state after a warmup call.

    Returns a dict with partition info if available.
    Must be called AFTER the first compiled forward (which triggers tracing).
    """
    info = {}

    # Check counters for recompilations
    counters = dynamo.utils.counters
    info["calls_captured"] = counters["stats"].get("calls_captured", 0)
    info["unique_graphs"] = counters["stats"].get("unique_graphs", 0)

    # Check CUDA graph tree state if available
    try:
        from torch._inductor.cudagraph_trees import get_manager
        manager = get_manager()
        if manager is not None:
            info["has_manager"] = True
            # Count function recordings
            num_recordings = len(manager.tree_cache) if hasattr(manager, 'tree_cache') else -1
            info["num_tree_entries"] = num_recordings
        else:
            info["has_manager"] = False
    except Exception as e:
        info["manager_error"] = str(e)

    return info

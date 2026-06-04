"""
Utilities for Fast-dLLM v2 optimized generation with torch.compile / CUDA graphs.

Environment Variables:
    FAST_DLLM_EXECUTION_MODE: "eager" | "compile" | "cudagraph" (default: "eager")
    FAST_DLLM_ATTENTION_BACKEND: "auto" | "flash4" | "flash3" | "flash2" | "sdpa" (default: "auto")
    FAST_DLLM_MAX_SEQ_LEN: Maximum pre-allocated sequence length for static KV cache (default: "4096")
    FAST_DLLM_CUDA_GRAPH_WARMUP: Number of warmup steps before CUDA graph capture (default: "3")
    FAST_DLLM_COMPILE_MODE: torch.compile mode - "default" | "reduce-overhead" | "max-autotune" (default: "reduce-overhead")
"""

import os

EXECUTION_MODE = os.environ.get("FAST_DLLM_EXECUTION_MODE", "eager")
ATTENTION_BACKEND = os.environ.get("FAST_DLLM_ATTENTION_BACKEND", "auto")
MAX_SEQ_LEN = int(os.environ.get("FAST_DLLM_MAX_SEQ_LEN", "4096"))
CUDA_GRAPH_WARMUP = int(os.environ.get("FAST_DLLM_CUDA_GRAPH_WARMUP", "3"))
COMPILE_MODE = os.environ.get("FAST_DLLM_COMPILE_MODE", "reduce-overhead")

from .static_kv_cache import StaticKVCache, StaticBlockCache
from .attention_backends import get_attention_backend, patch_attention_layers, unpatch_attention_layers
from .batch_bucketing import BatchBucket
from .dynamo_utils import make_compiled_forward, make_eager_forward

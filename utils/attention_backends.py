"""
Attention backend hierarchy with automatic fallback for Fast-dLLM v2.

Fallback order (hardware-gated):
  Flash Attention 4  (separate flash-attn-4 package, SM 9.0+ / Hopper only)
  → Flash Attention 3  (flash_attn_3 package, SM 8.0+ / Ampere+)
  → Flash Attention 2  (flash-attn >= 2.0, SM 8.0+ / Ampere+)
  → FlashInfer         (flashinfer-python package, SM 7.5+ / Turing+)
  → PyTorch SDPA       (always available, PyTorch >= 2.0)

Hardware requirements (verified against Dao-AILab/flash-attention README):
  FA4        — SM 9.0+  (H100 / B200).  Separate package: pip install flash-attn-4
  FA3        — SM 8.0+  (A100, A5000, H100).  Package: pip install flash-attn-3 (builds
               from hopper/ subdir of the main repo; installs as flash_attn_3).
  FA2        — SM 8.0+  (Ampere / Ada / Hopper).  Package: pip install flash-attn
  FlashInfer — SM 7.5+  (Turing and above).  Package: pip install flashinfer-python
  SDPA       — any GPU / CPU.

API layouts (all flash_attn functions use BSHD — batch/seq/heads/dim):
  flash_attn_func(q, k, v, dropout_p=0, softmax_scale=None, causal=False, ...)
    → (B, S, H, D)
  flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=None,
                           softmax_scale=None, causal=False, ...)
    → (B, S, H, D)
  FA3 flash_attn_func has NO dropout_p parameter (handled separately below).

FlashInfer API layout:
  Uses NHD (seq, heads, dim) with ragged/paged batching.  Our BSHD static KV
  buffers [B, S, H, D] are zero-copy compatible with FlashInfer's paged NHD
  layout [num_pages=B, page_size=S, H, D] by using page_size=max_seq_len and
  an identity page table.  BatchPrefillWithPagedKVCacheWrapper is used (not
  BatchDecode) because diffusion queries block_size tokens, not a single token.
  plan() must be called from eager code; run() can be inside CUDA graphs.

Note: flash_attn.flash_attn_interface is FA2's internal module — do NOT use it
to detect FA3.  FA3 lives in the separate flash_attn_3 package.

The module also provides `patch_attention_layers` / `unpatch_attention_layers`
to monkey-patch the model's attention layers for static-cache aware execution.
"""

import os
import logging
import torch
import torch.nn.functional as F
from typing import Optional
from functools import lru_cache

from .static_kv_cache import StaticKVCache, StaticBlockCache

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
# Backend detection
# ---------------------------------------------------------------------------

def _get_gpu_compute_capability() -> tuple:
    """Return (major, minor) compute capability of the current CUDA device."""
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability(torch.cuda.current_device())


def _wrap_flash_kvcache_as_custom_op(raw_fn, tag: str):
    """
    Wrap a flash_attn_with_kvcache function as a torch.library custom op
    with a FakeTensor (meta) implementation.

    Why: torch._dynamo.allow_in_graph prevents Dynamo from tracing INTO the
    function, but the FX graph executor still needs to determine output shapes
    by running the node with FakeTensors. CUDA kernels like flash_attn_with_kvcache
    access raw data pointers, which FakeTensors don't have — causing
    "Cannot access data pointer of Tensor" errors in torch 2.10+.

    The custom op registers a meta kernel that returns torch.empty_like(q)
    (output shape = query shape for flash_attn_with_kvcache), allowing
    FakeTensor shape propagation to succeed without touching GPU data.
    """
    lib = torch.library.Library(f"fast_dllm_{tag}", "DEF")
    lib.define(
        "flash_kvcache(Tensor q, Tensor k_cache, Tensor v_cache, "
        "Tensor cache_seqlens, float softmax_scale, bool causal) -> Tensor"
    )

    def _impl(q, k_cache, v_cache, cache_seqlens, softmax_scale, causal):
        return raw_fn(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=causal,
        )

    def _fake(q, k_cache, v_cache, cache_seqlens, softmax_scale, causal):
        # Output shape = query shape (B, S_q, H_q, D) in BSHD layout
        return torch.empty_like(q)

    lib.impl("flash_kvcache", _impl, "CUDA")
    lib.impl("flash_kvcache", _fake, "Meta")

    # Get the op namespace object
    op_ns = getattr(torch.ops, f"fast_dllm_{tag}")

    # Return a callable that matches the signature used in flash_kvcache_attention
    def wrapper(q, k_cache, v_cache, cache_seqlens=None, softmax_scale=None, causal=False):
        return op_ns.flash_kvcache(
            q, k_cache, v_cache, cache_seqlens, softmax_scale, causal,
        )

    # Keep reference to lib so it's not garbage collected
    wrapper._lib = lib
    return wrapper


def _try_import_flash_attn_4():
    """
    Flash Attention 4 — completely separate package (flash-attn-4), NOT a
    version of the main flash-attn package.  Targets Hopper/Blackwell (SM 9.0+).

    Install: pip install flash-attn-4
    Import:  from flash_attn_4 import flash_attn_with_kvcache
    """
    cc = _get_gpu_compute_capability()
    if cc[0] < 9:
        return None
    try:
        # FA4 is the flash_attn_4 package, not flash_attn
        from flash_attn_4 import flash_attn_with_kvcache as _fa4_kvcache_raw  # noqa: F401
        # Wrap as custom op (same reason as FA2 — see _wrap_flash_kvcache_as_custom_op)
        fa4_kvcache = _wrap_flash_kvcache_as_custom_op(_fa4_kvcache_raw, "fa4")
        # logger.info("Flash Attention 4 (flash_attn_4 package, Hopper paged KV) available")
        return fa4_kvcache
    except (ImportError, AttributeError):
        pass
    return None


def _try_import_flash_attn_3():
    """
    Flash Attention 3 — SM 8.0+ (Ampere and above, including Hopper).
    Built from the hopper/ subdir of Dao-AILab/flash-attention and installs
    as the 'flash_attn_3' package.

    Install: cd hopper && pip install .   (or pip install flash-attn-3)
    Import:  from flash_attn_3 import flash_attn_func

    IMPORTANT: FA3's flash_attn_func has NO dropout_p parameter.  The caller
    (_flash_attention) handles this by not passing dropout_p for FA3.

    Do NOT use flash_attn.flash_attn_interface to detect FA3 — that is FA2's
    internal submodule and would silently return the FA2 function mislabelled
    as FA3.
    """
    cc = _get_gpu_compute_capability()
    if cc[0] < 8:   # FA3 supports SM 8.0+ (Ampere+), not just Hopper
        return None
    try:
        from flash_attn_3 import flash_attn_func as fa3_func  # noqa: F401
        # Register as a dynamo leaf so torch.compile treats the CUDA kernel as
        # an opaque call and does not attempt to trace inside it.
        fa3_func = torch._dynamo.allow_in_graph(fa3_func)
        # logger.info("Flash Attention 3 (flash_attn_3 package, SM 8.0+) available")
        return fa3_func
    except (ImportError, AttributeError):
        pass
    return None


def _try_import_flash_attn_2():
    """
    Flash Attention 2 — standard flash-attn package (pip install flash-attn).
    Requires SM 8.0+ (Ampere and above) and flash-attn >= 2.0.

    Install: CUDA_HOME=/usr/local/cuda-12.6 pip install flash-attn --no-build-isolation
    API:
      flash_attn_func(q, k, v, dropout_p=0, softmax_scale=None, causal=False)
        — inputs/output BSHD (batch, seq, heads, dim)
      flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=None,
                               softmax_scale=None, causal=False)
        — cache_seqlens: (batch,) torch.int32 tensor of valid KV lengths
    """
    cc = _get_gpu_compute_capability()
    if cc[0] < 8:
        return None
    try:
        from flash_attn import flash_attn_func as fa2_func
        from flash_attn import flash_attn_with_kvcache as _fa2_kvcache_raw
        # Register flash_attn_func as a dynamo leaf node.
        fa2_func = torch._dynamo.allow_in_graph(fa2_func)
        # Wrap flash_attn_with_kvcache as a custom op with a FakeTensor meta
        # implementation. allow_in_graph alone is insufficient in torch 2.10:
        # the FX graph executor runs nodes with FakeTensors to determine output
        # shapes, but the CUDA kernel accesses raw data pointers which FakeTensors
        # don't have, causing "Cannot access data pointer of Tensor" errors.
        # The custom op provides a meta kernel that returns the correct shape
        # (output = same shape as query) without touching GPU data.
        fa2_kvcache = _wrap_flash_kvcache_as_custom_op(_fa2_kvcache_raw, "fa2")
        # logger.info("Flash Attention 2 available (with flash_attn_with_kvcache)")
        return {"func": fa2_func, "kvcache": fa2_kvcache}
    except (ImportError, AttributeError):
        pass
    try:
        from flash_attn import flash_attn_func as fa2_func
        # Register as a dynamo leaf so torch.compile treats the CUDA kernel as
        # an opaque call and does not attempt to trace inside it.
        fa2_func = torch._dynamo.allow_in_graph(fa2_func)
        # logger.info("Flash Attention 2 available (flash_attn_with_kvcache not found)")
        return {"func": fa2_func, "kvcache": None}
    except (ImportError, AttributeError):
        pass
    return None


def _try_import_flashinfer():
    """
    FlashInfer — high-performance inference kernels (pip install flashinfer-python).
    Requires SM 7.5+ (Turing and above: T4, A100, H100, etc.).

    Install: pip install flashinfer-python
    Optional: pip install flashinfer-cubin  (pre-compiled kernels for faster startup)

    FlashInfer uses NHD layout (seq, heads, dim) with ragged/paged batching.
    Our BSHD static KV buffers [B, S, H, D] map zero-copy to FlashInfer's
    paged KV layout [num_pages=B, page_size=max_seq_len, num_kv_heads, head_dim]
    using page_size=max_seq_len and an identity page table.

    We use BatchPrefillWithPagedKVCacheWrapper (not BatchDecode) because
    diffusion steps query block_size tokens (not a single token).

    Returns the flashinfer module on success, None on failure.
    """
    cc = _get_gpu_compute_capability()
    if cc[0] < 7 or (cc[0] == 7 and cc[1] < 5):
        return None
    try:
        import flashinfer
        # Verify the APIs we need are available
        _ = flashinfer.BatchPrefillWithPagedKVCacheWrapper
        return flashinfer
    except (ImportError, AttributeError):
        pass
    return None


def _wrap_flashinfer_prefill_run_as_custom_op(wrapper_holder):
    """
    Wrap FlashInfer's BatchPrefillWithPagedKVCacheWrapper.run() as a custom op
    with a FakeTensor (meta) implementation for torch.compile compatibility.

    Similar to _wrap_flash_kvcache_as_custom_op but adapted for FlashInfer's API:
    - Input q is ragged NHD: (total_q_tokens, num_qo_heads, head_dim)
    - KV cache is paged NHD: (num_pages, page_size, num_kv_heads, head_dim)
    - Output shape = q shape: (total_q_tokens, num_qo_heads, head_dim)

    wrapper_holder is a list [wrapper] so the wrapper can be updated after
    plan() without re-creating the custom op.
    """
    lib = torch.library.Library("fast_dllm_flashinfer", "DEF")
    lib.define(
        "prefill_run(Tensor q, Tensor k_cache, Tensor v_cache) -> Tensor"
    )

    def _impl(q, k_cache, v_cache):
        return wrapper_holder[0].run(q, (k_cache, v_cache))

    def _fake(q, k_cache, v_cache):
        return torch.empty_like(q)

    lib.impl("prefill_run", _impl, "CUDA")
    lib.impl("prefill_run", _fake, "Meta")

    op_ns = getattr(torch.ops, "fast_dllm_flashinfer")

    def run_fn(q, k_cache, v_cache):
        return op_ns.prefill_run(q, k_cache, v_cache)

    run_fn._lib = lib
    return run_fn


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

class AttentionBackend:
    """Unified interface for attention backends."""

    SDPA       = "sdpa"
    FLASH2     = "flash2"
    FLASH3     = "flash3"
    FLASH4     = "flash4"
    FLASHINFER = "flashinfer"

    def __init__(self, name: str, impl):
        self.name  = name
        self._impl = impl
        self._kvcache_dispatch_logged = False
        # FlashInfer state (lazily initialized by plan_flashinfer)
        self._flashinfer_wrapper = None
        self._flashinfer_workspace = None
        self._flashinfer_run_fn = None
        self._flashinfer_wrapper_holder = None
        self._flashinfer_qo_indptr = None
        self._flashinfer_page_table = None

    def _log_kvcache_dispatch(self, query, k_cache, cache_seqlens):
        """Log kvcache dispatch info and set the flag so dynamo never sees it change."""
        seqlens_str = (cache_seqlens.tolist() if cache_seqlens.numel() <= 8
                       else f"[{cache_seqlens[0].item()}, ..., {cache_seqlens[-1].item()}] (n={cache_seqlens.numel()})")
        logger.info("[flash_kvcache_attention] DISPATCHING to backend=%r  "
                    "query=%s  kv_cache=%s  cache_seqlens=%s",
                    self.name, tuple(query.shape), tuple(k_cache.shape), seqlens_str)
        self._kvcache_dispatch_logged = True

    def mark_dispatch_logged(self):
        """Pre-set the dispatch-logged flag before dynamo traces.

        Call this after the first eager forward (e.g. prefill) so that dynamo
        only ever guards on ``_kvcache_dispatch_logged == True`` — prevents
        a guard failure and recompilation when the flag flips.
        """
        self._kvcache_dispatch_logged = True

    def attention(
        self,
        query: torch.Tensor,      # (B, nheads, S_q, D)  BHSD
        key: torch.Tensor,        # (B, nheads_k, S_kv, D)
        value: torch.Tensor,      # (B, nheads_k, S_kv, D)
        attn_mask: Optional[torch.Tensor] = None,
        scaling: float = None,
        is_causal: bool = False,
        dropout: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        Input/output layout: BHSD (batch, heads, seq, dim).
        Returns: (B, nheads, S_q, D).
        """
        if self.name == self.SDPA:
            return self._sdpa_attention(
                query, key, value, attn_mask, scaling, is_causal, dropout,
            )
        elif self.name in (self.FLASH2, self.FLASH3, self.FLASH4):
            return self._flash_attention(
                query, key, value, attn_mask, scaling, is_causal, dropout,
            )
        elif self.name == self.FLASHINFER:
            return self._flashinfer_attention(
                query, key, value, attn_mask, scaling, is_causal, dropout,
            )
        else:
            raise ValueError(f"Unknown attention backend: {self.name}")

    def _sdpa_attention(self, q, k, v, attn_mask, scaling, is_causal, dropout):
        """PyTorch SDPA with optional attention mask for seqused support."""
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        # Use GQA repeat if needed
        nheads_q = q.shape[1]
        nheads_k = k.shape[1]
        if nheads_q != nheads_k:
            n_rep = nheads_q // nheads_k
            k = k[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1)
            k = k.reshape(k.shape[0], nheads_q, k.shape[3], k.shape[4])
            v = v[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1)
            v = v.reshape(v.shape[0], nheads_q, v.shape[3], v.shape[4])

        scale = scaling if scaling is not None else q.shape[-1] ** -0.5
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout,
            is_causal=is_causal,
            scale=scale,
        )
        return out

    def _flash_attention(self, q, k, v, attn_mask, scaling, is_causal, dropout):
        """
        Flash attention (FA2 or FA3) via BSHD layout.

        Both FA2 and FA3 expect (B, S, H, D) inputs — we transpose from the
        internal BHSD layout before calling and transpose back after.

        FA2: supports dropout_p.
        FA3: does NOT have a dropout_p parameter (omitted below when name==FLASH3).
        """
        # BHSD → BSHD
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()

        func = self._impl
        if isinstance(func, dict):
            func = func["func"]

        scale = scaling if scaling is not None else q.shape[-1] ** -0.5

        if self.name == self.FLASH3:
            # FA3 flash_attn_func has no dropout_p parameter
            out = func(
                q_fa, k_fa, v_fa,
                softmax_scale=scale,
                causal=is_causal,
            )
        else:
            # FA2 (and FA4 func-style)
            out = func(
                q_fa, k_fa, v_fa,
                dropout_p=dropout,
                softmax_scale=scale,
                causal=is_causal,
            )
        # BSHD → BHSD
        return out.transpose(1, 2)

    def _flashinfer_attention(self, q, k, v, attn_mask, scaling, is_causal, dropout):
        """
        FlashInfer attention for non-kvcache path (prefill-style).

        Input layout: BHSD (B, nheads, S, D) — we transpose to NHD for FlashInfer.
        FlashInfer single_prefill_with_kv_cache expects:
          q: (qo_len, num_qo_heads, head_dim)
          k: (kv_len, num_kv_heads, head_dim)
          v: (kv_len, num_kv_heads, head_dim)
        """
        flashinfer = self._impl
        B, nheads_q, S_q, D = q.shape
        nheads_k = k.shape[1]
        S_kv = k.shape[2]

        scale = scaling if scaling is not None else D ** -0.5

        # For batch > 1, use BatchPrefillWithRaggedKVCacheWrapper
        if B > 1:
            # BHSD → BSHD → ragged NHD: (B*S, H, D)
            q_nhd = q.transpose(1, 2).contiguous().reshape(B * S_q, nheads_q, D)
            k_nhd = k.transpose(1, 2).contiguous().reshape(B * S_kv, nheads_k, D)
            v_nhd = v.transpose(1, 2).contiguous().reshape(B * S_kv, nheads_k, D)

            qo_indptr = torch.arange(
                0, (B + 1) * S_q, S_q, dtype=torch.int32, device=q.device,
            )
            kv_indptr = torch.arange(
                0, (B + 1) * S_kv, S_kv, dtype=torch.int32, device=q.device,
            )

            workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=q.device,
            )
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace, kv_layout="NHD",
            )
            wrapper.plan(
                qo_indptr, kv_indptr,
                num_qo_heads=nheads_q, num_kv_heads=nheads_k,
                head_dim_qk=D, head_dim_vo=D,
                causal=is_causal,
            )
            out = wrapper.run(q_nhd, k_nhd, v_nhd)
            # NHD ragged → BSHD → BHSD
            out = out.reshape(B, S_q, nheads_q, D).transpose(1, 2)
            return out
        else:
            # Single batch — use single_prefill_with_kv_cache
            q_nhd = q.squeeze(0).transpose(0, 1).contiguous()  # (S_q, H, D)
            k_nhd = k.squeeze(0).transpose(0, 1).contiguous()  # (S_kv, H_k, D)
            v_nhd = v.squeeze(0).transpose(0, 1).contiguous()  # (S_kv, H_k, D)

            out = flashinfer.single_prefill_with_kv_cache(
                q_nhd, k_nhd, v_nhd,
                causal=is_causal,
                sm_scale=scale,
                kv_layout="NHD",
            )
            # (S_q, H, D) → (1, H, S_q, D)
            return out.transpose(0, 1).unsqueeze(0)

    def plan_flashinfer(
        self,
        batch_size: int,
        query_len: int,
        max_seq_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_seqlens: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Prepare the FlashInfer BatchPrefillWithPagedKVCacheWrapper for the
        upcoming diffusion steps.

        Must be called from EAGER code (outside compiled graph) before each
        block. The wrapper's run() can then be called inside the compiled graph.

        Layout bridge (zero-copy):
          Our static KV cache per-layer: [B, max_seq_len, num_kv_heads, head_dim] (BSHD)
          FlashInfer paged KV layout:    [num_pages, page_size, num_kv_heads, head_dim] (NHD)
          With page_size=max_seq_len and num_pages=batch_size, these are identical.
          Identity page table: batch element i → page i.
        """
        if self.name != self.FLASHINFER:
            return

        flashinfer = self._impl
        need_reinit = (
            self._flashinfer_wrapper is None
            or self._flashinfer_workspace is None
            or self._flashinfer_qo_indptr is None
            or self._flashinfer_qo_indptr.shape[0] != batch_size + 1
            or self._flashinfer_page_table is None
            or self._flashinfer_page_table.shape[0] != batch_size
        )

        if need_reinit:
            # print(f"[flashinfer] Initializing wrapper: batch_size={batch_size}  "
            #       f"query_len={query_len}  max_seq_len={max_seq_len}  "
            #       f"num_qo_heads={num_qo_heads}  num_kv_heads={num_kv_heads}  "
            #       f"head_dim={head_dim}  page_size={max_seq_len}  "
            #       f"dtype={dtype}", flush=True)
            logger.info("[flashinfer] Initializing wrapper: batch_size=%d  query_len=%d  "
                        "max_seq_len=%d  num_qo_heads=%d  num_kv_heads=%d  head_dim=%d",
                        batch_size, query_len, max_seq_len, num_qo_heads, num_kv_heads, head_dim)
            # Workspace buffer: ~128 MB for FlashInfer internal scratch
            self._flashinfer_workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=device,
            )
            self._flashinfer_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self._flashinfer_workspace, kv_layout="NHD",
            )
            # Identity page table: batch element i uses page i (1 page per element)
            self._flashinfer_page_table = torch.arange(
                batch_size, dtype=torch.int32, device=device,
            ).unsqueeze(1)  # [B, 1]
            # Uniform query indptr: each batch element queries query_len tokens
            self._flashinfer_qo_indptr = torch.arange(
                0, (batch_size + 1) * query_len, query_len,
                dtype=torch.int32, device=device,
            )
            # Custom op wrapper for run() — needed for torch.compile FakeTensor compat.
            # The custom op (torch.library) can only be registered ONCE per process.
            # On reinit (batch size change), we update the wrapper in the existing
            # holder list — the custom op's closure reads holder[0] at call time.
            if self._flashinfer_wrapper_holder is None:
                self._flashinfer_wrapper_holder = [self._flashinfer_wrapper]
                self._flashinfer_run_fn = _wrap_flashinfer_prefill_run_as_custom_op(
                    self._flashinfer_wrapper_holder,
                )
            else:
                self._flashinfer_wrapper_holder[0] = self._flashinfer_wrapper

        # Update qo_indptr if query_len changed (e.g., sub-block optimization)
        expected_last = batch_size * query_len
        if self._flashinfer_qo_indptr[-1].item() != expected_last:
            self._flashinfer_qo_indptr = torch.arange(
                0, (batch_size + 1) * query_len, query_len,
                dtype=torch.int32, device=device,
            )

        # Call plan() — sets up metadata for the upcoming run() calls.
        # kv_seqlens is scratch_seqlens (int32): effective KV length per batch element.
        # q_data_type must match the model's compute dtype (typically bfloat16).
        self._flashinfer_wrapper.plan(
            qo_indptr=self._flashinfer_qo_indptr,
            paged_kv_indptr=torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device,
            ),
            paged_kv_indices=self._flashinfer_page_table.squeeze(1),
            paged_kv_last_page_len=kv_seqlens.to(torch.int32),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=max_seq_len,
            causal=False,
            q_data_type=dtype,
        )

    def flash_kvcache_attention(
        self,
        query: torch.Tensor,         # (B, S_q, nheads, D)  BSHD
        k_cache: torch.Tensor,       # (B, max_S, nheads_k, D)  BSHD static buffer
        v_cache: torch.Tensor,       # (B, max_S, nheads_k, D)  BSHD static buffer
        cache_seqlens: torch.Tensor, # (B,) torch.int32 — valid KV length per batch
        is_causal: bool = False,
        scaling: float = None,
    ) -> torch.Tensor:
        """
        Flash attention operating directly on static KV cache buffers using
        cache_seqlens for variable-length masking — avoids materialising a mask.

        All tensors are BSHD throughout — no transposes needed. Query, K cache,
        and V cache are all in flash_attn's native (B, S, H, D) layout.

        Returns: (B, S_q, nheads, D) BSHD.

        Requires FA2 or FA4 with kvcache support. No SDPA fallback.
        """
        scale = scaling if scaling is not None else query.shape[-1] ** -0.5

        # One-time dispatch log. The flag is set to True during init/warmup
        # (see log_kvcache_dispatch) so dynamo never sees it change — avoids
        # a guard on the bool that would trigger recompilation.
        if not self._kvcache_dispatch_logged and not torch.compiler.is_compiling():
            self._log_kvcache_dispatch(query, k_cache, cache_seqlens)

        if self.name == self.FLASH4:
            return self._impl(
                query, k_cache, v_cache,
                cache_seqlens=cache_seqlens,
                softmax_scale=scale,
                causal=is_causal,
            )

        if self.name == self.FLASH2 and isinstance(self._impl, dict) and self._impl.get("kvcache"):
            return self._impl["kvcache"](
                query, k_cache, v_cache,
                cache_seqlens=cache_seqlens,
                softmax_scale=scale,
                causal=is_causal,
            )

        if self.name == self.FLASHINFER:
            return self._flashinfer_kvcache_attention(
                query, k_cache, v_cache, cache_seqlens, is_causal, scale,
            )

        raise RuntimeError(
            f"flash_kvcache_attention requires FA2 (with kvcache), FA4, or FlashInfer, "
            f"but current backend is {self.name!r}. Install flash-attn >= 2.0 or flashinfer-python."
        )

    def _flashinfer_kvcache_attention(
        self,
        query: torch.Tensor,         # (B, S_q, nheads, D)  BSHD
        k_cache: torch.Tensor,       # (B, max_S, nheads_k, D)  BSHD static buffer
        v_cache: torch.Tensor,       # (B, max_S, nheads_k, D)  BSHD static buffer
        cache_seqlens: torch.Tensor,  # (B,) torch.int32
        is_causal: bool,
        scale: float,
    ) -> torch.Tensor:
        """
        FlashInfer kvcache attention for diffusion steps.

        plan_flashinfer() must have been called before this method.

        Layout bridge (zero-copy):
          query:   BSHD [B, S_q, H, D]  → ragged NHD [B*S_q, H, D]
          k_cache: BSHD [B, max_S, H_k, D] = paged NHD [num_pages=B, page_size=max_S, H_k, D]
          v_cache: same
          output:  ragged NHD [B*S_q, H, D] → BSHD [B, S_q, H, D]
        """
        B, S_q, nheads, D = query.shape

        if self._flashinfer_run_fn is None:
            raise RuntimeError(
                "FlashInfer _flashinfer_run_fn is None — plan_flashinfer() must be "
                "called before flash_kvcache_attention(). This usually means the "
                "prefill forward ran before plan_flashinfer() was called in "
                "batch_sample_dynamo()."
            )

        # BSHD → ragged NHD for query: (B, S_q, H, D) → (B*S_q, H, D)
        q_nhd = query.reshape(B * S_q, nheads, D)

        # KV cache buffers are already in compatible layout — zero copy.
        # BSHD [B, max_S, H_k, D] == paged NHD [num_pages=B, page_size=max_S, H_k, D]

        # Use the custom-op-wrapped run() for torch.compile compatibility
        out = self._flashinfer_run_fn(q_nhd, k_cache, v_cache)

        # Ragged NHD → BSHD: (B*S_q, H, D) → (B, S_q, H, D)
        return out.reshape(B, S_q, nheads, D)


@lru_cache(maxsize=1)
def get_attention_backend(preference: str = "auto") -> AttentionBackend:
    """
    Detect and return the best available attention backend.

    Args:
        preference: "auto" | "flash4" | "flash3" | "flash2" | "flashinfer" | "sdpa"

    Hardware gates (checked before attempting imports):
        FA4:        SM 9.0+  (Hopper / Blackwell)
        FA3:        SM 8.0+  (Ampere and above)
        FA2:        SM 8.0+  (Ampere and above)
        FlashInfer: SM 7.5+  (Turing and above)
        SDPA:       any
    """
    cc     = _get_gpu_compute_capability()
    cc_str = f"SM {cc[0]}.{cc[1]}" if cc != (0, 0) else "CPU"

    if preference == "sdpa":
        logger.info("[attention_backend] SELECTED: sdpa (forced)  gpu=%s", cc_str)
        # print(f"[attention_backend] SELECTED: sdpa (forced)  gpu={cc_str}", flush=True)
        return AttentionBackend(AttentionBackend.SDPA, None)

    # ---- FA4: separate flash_attn_4 package, SM 9.0+ only ----
    if preference in ("auto", "flash4"):
        fa4 = _try_import_flash_attn_4()
        if fa4 is not None:
            logger.info("[attention_backend] SELECTED: flash4 (flash_attn_4 pkg, Hopper paged KV)  gpu=%s  requested=%r", cc_str, preference)
            # print(f"[attention_backend] SELECTED: flash4 (flash_attn_4 pkg, Hopper paged KV)  gpu={cc_str}  requested={preference!r}", flush=True)
            return AttentionBackend(AttentionBackend.FLASH4, fa4)
        logger.info("[attention_backend] SKIPPED: flash4  gpu=%s  reason=%s", cc_str,
                    "SM < 9.0" if cc[0] < 9 else "flash_attn_4 package not installed")
        if preference == "flash4":
            logger.warning("Flash Attention 4 (flash_attn_4) requested but not available — "
                           "requires SM 9.0+ and pip install flash-attn-4; falling back")

    # ---- FA3: flash_attn_3 package, SM 8.0+ ----
    if preference in ("auto", "flash3"):
        fa3 = _try_import_flash_attn_3()
        if fa3 is not None:
            logger.info("[attention_backend] SELECTED: flash3 (flash_attn_3 pkg, SM 8.0+)  gpu=%s  requested=%r", cc_str, preference)
            # print(f"[attention_backend] SELECTED: flash3 (flash_attn_3 pkg, SM 8.0+)  gpu={cc_str}  requested={preference!r}", flush=True)
            return AttentionBackend(AttentionBackend.FLASH3, fa3)
        logger.info("[attention_backend] SKIPPED: flash3  gpu=%s  reason=%s", cc_str,
                    "SM < 8.0" if cc[0] < 8 else "flash_attn_3 package not installed")
        if preference == "flash3":
            logger.warning("Flash Attention 3 (flash_attn_3) requested but not available — "
                           "requires SM 8.0+ and flash_attn_3 package; falling back")

    # ---- FA2: main flash-attn package, SM 8.0+ ----
    if preference in ("auto", "flash2"):
        fa2 = _try_import_flash_attn_2()
        if fa2 is not None:
            kvcache_tag = "with kvcache" if (isinstance(fa2, dict) and fa2.get("kvcache")) else "no kvcache"
            logger.info("[attention_backend] SELECTED: flash2 (%s)  gpu=%s  requested=%r", kvcache_tag, cc_str, preference)
            # print(f"[attention_backend] SELECTED: flash2 ({kvcache_tag})  gpu={cc_str}  requested={preference!r}", flush=True)
            return AttentionBackend(AttentionBackend.FLASH2, fa2)
        logger.info("[attention_backend] SKIPPED: flash2  gpu=%s  reason=%s", cc_str,
                    "SM < 8.0" if cc[0] < 8 else "flash-attn package not installed")
        if preference == "flash2":
            logger.warning("Flash Attention 2 requested but not available — "
                           "requires SM 8.0+ and pip install flash-attn; falling back")

    # ---- FlashInfer: flashinfer-python package, SM 7.5+ ----
    if preference in ("auto", "flashinfer"):
        fi = _try_import_flashinfer()
        if fi is not None:
            logger.info("[attention_backend] SELECTED: flashinfer (flashinfer-python pkg, SM 7.5+)  gpu=%s  requested=%r", cc_str, preference)
            #  print(f"[attention_backend] SELECTED: flashinfer (flashinfer-python pkg, SM 7.5+)  gpu={cc_str}  requested={preference!r}", flush=True)
            return AttentionBackend(AttentionBackend.FLASHINFER, fi)
        logger.info("[attention_backend] SKIPPED: flashinfer  gpu=%s  reason=%s", cc_str,
                    "SM < 7.5" if (cc[0] < 7 or (cc[0] == 7 and cc[1] < 5)) else "flashinfer-python package not installed")
        if preference == "flashinfer":
            logger.warning("FlashInfer requested but not available — "
                           "requires SM 7.5+ and pip install flashinfer-python; falling back")

    # ---- SDPA fallback ----
    if cc[0] >= 8:
        reason = "flash-attn / flashinfer package(s) not installed"
    elif cc[0] >= 7 and cc[1] >= 5:
        reason = f"flash-attn not installed (GPU {cc_str} below SM 8.0); flashinfer not installed"
    else:
        reason = f"GPU {cc_str} below minimum SM 7.5 for all accelerated backends"
    logger.info("[attention_backend] SELECTED: sdpa (fallback)  gpu=%s  requested=%r  reason=%s", cc_str, preference, reason)
    # print(f"[attention_backend] SELECTED: sdpa (fallback)  gpu={cc_str}  requested={preference!r}  — {reason}", flush=True)
    return AttentionBackend(AttentionBackend.SDPA, None)


# ---------------------------------------------------------------------------
# Attention layer patching for static KV cache
# ---------------------------------------------------------------------------

def _apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding. Input Q/K are BSHD (B, S, H, D)."""
    cos = cos.unsqueeze(2)  # (B, S, 1, D) — broadcasts over H dim in BSHD
    sin = sin.unsqueeze(2)
    x1_q = q[..., : q.shape[-1] // 2]
    x2_q = q[..., q.shape[-1] // 2 :]
    q_embed = (q * cos) + (torch.cat((-x2_q, x1_q), dim=-1) * sin)

    x1_k = k[..., : k.shape[-1] // 2]
    x2_k = k[..., k.shape[-1] // 2 :]
    k_embed = (k * cos) + (torch.cat((-x2_k, x1_k), dim=-1) * sin)
    return q_embed, k_embed


def _create_patched_attention_forward(attn_module, attn_backend, static_block_cache=None):
    """
    Create a patched attention forward that uses stacked StaticKVCache
    with in-place writes and dispatches to the best attention backend.

    The static cache is accessed via past_key_value argument (which traces
    from self._static_kv_cache on the model — module state, not closure
    capture). This means KV mutations inside the compiled graph are
    classified as module-state mutations, not argument mutations, which
    is critical for CUDA graph replay.

    For diffusion steps (update_past_key_values=False):
      - Uses write_scratch_compiled() with index_copy_ (tensor offsets)
      - Uses scratch_seqlens for flash attention masking
    For commit steps (update_past_key_values=True, eager only):
      - Uses update() with Python int slicing (not in compiled graph)
      - Uses scratch_seqlens for flash attention masking
    """
    original_forward = attn_module.forward
    layer_idx        = attn_module.layer_idx
    head_dim         = attn_module.head_dim
    scaling          = attn_module.scaling

    def patched_forward(
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        update_past_key_values=False,
        block_past_key_values=None,
        replace_position=None,
        **kwargs,
    ):
        # Fall through to original if not using static cache
        if not isinstance(past_key_value, StaticKVCache):
            return original_forward(
                hidden_states, position_embeddings, attention_mask,
                past_key_value, cache_position, update_past_key_values,
                block_past_key_values, replace_position, **kwargs,
            )

        # ---- Projection ----
        input_shape  = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)

        query_states = attn_module.q_proj(hidden_states).view(hidden_shape)  # BSHD
        key_states   = attn_module.k_proj(hidden_states).view(hidden_shape)  # BSHD
        value_states = attn_module.v_proj(hidden_states).view(hidden_shape)  # BSHD

        # ---- RoPE ----
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ---- Block cache handling (sub-block optimization, eager only) ----
        # StaticBlockCache is BSHD: (batch, block_size, num_kv_heads, head_dim)
        if block_past_key_values is not None:
            if len(block_past_key_values) <= layer_idx:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = block_past_key_values.update(
                    key_states, value_states, layer_idx, cache_kwargs,
                )
            else:
                bk = block_past_key_values[layer_idx][0]
                bv = block_past_key_values[layer_idx][1]
                seq_len = key_states.shape[1]  # BSHD: seq is dim 1
                # Full-block forwards don't pass replace_position (None): they
                # overwrite the entire block (seq_len == block_size) starting at
                # position 0. Small-block forwards pass an explicit sub-block
                # offset. (The dynamic path recreates the cache each full-block
                # step; the static buffer is reused, so re-entry lands here.)
                rp = 0 if replace_position is None else replace_position
                bk[:, rp:rp + seq_len, :, :].copy_(key_states)
                bv[:, rp:rp + seq_len, :, :].copy_(value_states)
                key_states  = bk
                value_states = bv

        # ---- Static cross-block cache handling ----
        # All K/V tensors are BSHD throughout — no transposes needed.
        if update_past_key_values:
            # Commit path (eager only): Python-int slice write + advance pointer
            past_len = past_key_value.get_seq_length()
            seq_len  = key_states.shape[1]  # BSHD: seq is dim 1
            past_key_value.update(key_states, value_states, layer_idx)
            past_key_value.set_scratch_seqlens(past_len + seq_len)
        else:
            # Diffusion path (compiled): index_copy_ with tensor offset
            past_key_value.write_scratch_compiled(key_states, value_states, layer_idx)

        # Full static buffers — constant shape (B, max_seq_len, H, D) BSHD
        full_k, full_v = past_key_value.get_full_kv(layer_idx)

        # ---- Attention dispatch ----
        # All tensors are BSHD — flash_kvcache_attention returns BSHD directly.
        attn_output = attn_backend.flash_kvcache_attention(
            query_states,
            full_k,
            full_v,
            cache_seqlens=past_key_value.scratch_seqlens,
            is_causal=False,
            scaling=scaling,
        )  # BSHD output

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_module.o_proj(attn_output)
        return attn_output

    return patched_forward


def _create_compiled_attention_forward(attn_module, attn_backend):
    """
    Create a compile-optimized attention forward for diffusion steps.

    This is a stripped-down version of _create_patched_attention_forward
    that eliminates all branches that cause graph breaks or CUDA graph
    partitions under torch.compile with reduce-overhead mode:

    Eliminated branches (always constant in compiled diffusion path):
      - isinstance(past_key_value, StaticKVCache) — always True
      - update_past_key_values — always False
      - block_past_key_values is not None — always None (no block cache
        in compiled path; block cache uses its own separate forward)

    All remaining operations are pure tensor ops:
      - Linear projections (q_proj, k_proj, v_proj, o_proj)
      - RoPE via _apply_rotary_pos_emb (elementwise tensor math)
      - write_scratch_compiled via index_copy_ (tensor offset)
      - get_full_kv via .select() (tensor view)
      - flash_kvcache_attention via custom op (opaque to tracer)
      - reshape + contiguous (tensor ops)

    The key difference from patched_forward: dynamo does NOT need to
    emit guards for isinstance, boolean args, or None checks —
    the function body is a straight-line tensor computation.
    """
    layer_idx = attn_module.layer_idx
    head_dim  = attn_module.head_dim
    scaling   = attn_module.scaling

    def compiled_forward(
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        update_past_key_values=False,
        block_past_key_values=None,
        replace_position=None,
        **kwargs,
    ):
        # ---- Projection (linear layers — standard torch ops) ----
        input_shape  = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)

        query_states = attn_module.q_proj(hidden_states).view(hidden_shape)
        key_states   = attn_module.k_proj(hidden_states).view(hidden_shape)
        value_states = attn_module.v_proj(hidden_states).view(hidden_shape)

        # ---- RoPE (elementwise tensor math) ----
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ---- KV cache write (index_copy_ with tensor offset — no Python ints) ----
        past_key_value.write_scratch_compiled(key_states, value_states, layer_idx)

        # ---- Full static buffers (constant shape view via .select()) ----
        full_k, full_v = past_key_value.get_full_kv(layer_idx)

        # ---- Attention (custom op — opaque to tracer) ----
        attn_output = attn_backend.flash_kvcache_attention(
            query_states,
            full_k,
            full_v,
            cache_seqlens=past_key_value.scratch_seqlens,
            is_causal=False,
            scaling=scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_module.o_proj(attn_output)
        return attn_output

    return compiled_forward


def patch_attention_layers(model, attn_backend, static_block_cache=None):
    """
    Monkey-patch all attention layers to use static cache + chosen backend.
    Returns a dict of original forwards for restoration.

    NOTE: static_cache is no longer passed here — the patched forward accesses
    it via past_key_value argument (which traces from model._static_kv_cache).
    """
    original_forwards = {}
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        original_forwards[i] = attn.forward
        attn.forward = _create_patched_attention_forward(
            attn, attn_backend, static_block_cache,
        )
    return original_forwards


def patch_attention_layers_compiled(model, attn_backend):
    """
    Monkey-patch attention layers with compile-optimized forwards.

    Unlike patch_attention_layers, this uses _create_compiled_attention_forward
    which eliminates all branches (isinstance, boolean args, None checks) that
    cause guard checks and CUDA graph partitions under torch.compile.

    Only use for the compiled diffusion path (update_past_key_values=False,
    no block cache). The eager commit path and prefill should use the regular
    patched or original forwards.
    """
    original_forwards = {}
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        original_forwards[i] = attn.forward
        attn.forward = _create_compiled_attention_forward(attn, attn_backend)
    return original_forwards


def unpatch_attention_layers(model, original_forwards):
    """Restore original attention forward methods."""
    for i, layer in enumerate(model.model.layers):
        if i in original_forwards:
            layer.self_attn.forward = original_forwards[i]

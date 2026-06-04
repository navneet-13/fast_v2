"""
Static KV Cache implementations for Fast-dLLM v2.

List-of-tensor design for torch.compile / CUDA graph compatibility:
- KV buffers are per-layer tensors [B, S, H, D] — BSHD layout
  (matches flash_attn's native layout, eliminating transposes on read)
- Per-layer list avoids .select() views on a stacked tensor, which would
  cause AOT autograd to decompose in-place mutations into select_scatter
  (copies the entire [L,B,S,H,D] tensor per mutation — catastrophically slow)
- All buffers have mark_static_address for reduce-overhead CUDA graph capture
- KV writes use scatter_ with tensor offsets (no Python-int slice specialization)
- All key/value states flow through in BSHD layout — no transposes anywhere
"""

import torch
from typing import Optional


class StaticKVCache:
    """
    Static KV cache for cross-block context in block-diffusion generation.

    BSHD list-of-tensor layout: key_cache and value_cache are each a list
    of num_layers tensors, each of shape (batch, max_seq_len, num_kv_heads, head_dim).

    Direct per-layer access via key_cache[layer_idx] — no .select() views.
    This avoids the select_scatter decomposition that occurs when AOT autograd
    functionalizes in-place mutations on .select() views of stacked tensors,
    which caused 2.3x slowdown (64 vs 145 tok/s at BS=4).

    All key/value states flow through in BSHD layout — projections output
    BSHD directly (no transpose after .view()), cache stores BSHD, and
    flash_attn reads BSHD natively. Zero transposes in the hot path.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.batch_size = batch_size
        self.max_batch_size = batch_size  # original allocation size (survives compact_batch)
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Per-layer KV buffers: list of [B, S, H, D] — BSHD per layer
        self.key_cache = [
            torch.zeros(
                batch_size, max_seq_len, num_kv_heads, head_dim,
                dtype=dtype, device=device,
            )
            for _ in range(num_layers)
        ]
        self.value_cache = [
            torch.zeros(
                batch_size, max_seq_len, num_kv_heads, head_dim,
                dtype=dtype, device=device,
            )
            for _ in range(num_layers)
        ]

        # Committed sequence length (advanced only by update())
        self._seq_len = 0

        # --- Tensor-valued control signals for compiled graph ---
        # cache_seqlens: (batch,) int32 — committed valid KV length per element
        self.cache_seqlens = torch.zeros(
            batch_size, dtype=torch.int32, device=device,
        )
        # scratch_seqlens: (batch,) int32 — effective length during diffusion
        # steps (committed + current block). Used as seqused_k for flash attn.
        self.scratch_seqlens = torch.zeros(
            batch_size, dtype=torch.int32, device=device,
        )
        # kv_write_start: scalar int64 — offset where current block KV is
        # written. Used with index_copy_ inside the compiled graph.
        self.kv_write_start = torch.zeros(
            (), dtype=torch.int64, device=device,
        )
        # Pre-allocated write index: (max_block_size,) long — filled with
        # arange + kv_write_start before each compiled call.
        self._write_idx = torch.zeros(
            max_seq_len, dtype=torch.int64, device=device,
        )
        # Pre-allocated arange for building write indices
        self._arange = torch.arange(
            max_seq_len, dtype=torch.int64, device=device,
        )

        # Pre-allocated attention mask buffer for SDPA fallback
        self._attn_mask_buffer = torch.zeros(
            batch_size, 1, 1, max_seq_len, dtype=dtype, device=device,
        )
        self._kv_indices = torch.arange(
            max_seq_len, dtype=torch.int32, device=device,
        )

        # Mark all static buffers for CUDA graph compatibility
        self._mark_static_addresses()

    def _mark_static_addresses(self):
        """
        Mark all pre-allocated buffers with torch._dynamo.mark_static_address.
        Tells reduce-overhead mode that GPU addresses are stable, so in-place
        mutations (copy_, fill_, scatter_) are safe inside CUDA graphs.
        """
        _mark = torch._dynamo.mark_static_address
        for layer_idx in range(self.num_layers):
            _mark(self.key_cache[layer_idx])
            _mark(self.value_cache[layer_idx])
        _mark(self.cache_seqlens)
        _mark(self.scratch_seqlens)
        _mark(self.kv_write_start)
        _mark(self._write_idx)
        _mark(self._arange)
        _mark(self._attn_mask_buffer)
        _mark(self._kv_indices)

    # ------------------------------------------------------------------
    # DynamicCache-compatible interface (used during eager prefill)
    # ------------------------------------------------------------------

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        In-place write at the committed position. Advances pointer after
        the last layer writes. Used during eager prefill and block commit.

        key_states/value_states: BSHD (B, S, H, D) — same layout as cache.
        """
        seq_len = key_states.shape[1]  # BSHD: seq is dim 1
        start = self._seq_len

        self.key_cache[layer_idx][:, start:start + seq_len, :, :].copy_(
            key_states
        )
        self.value_cache[layer_idx][:, start:start + seq_len, :, :].copy_(
            value_states
        )

        # Advance committed length after all layers have written
        if layer_idx == self.num_layers - 1:
            self._seq_len += seq_len
            self.cache_seqlens.fill_(self._seq_len)

        # Return BSHD slices (return value is unused by patched forward,
        # but kept for DynamicCache API compatibility)
        end = start + seq_len
        return (
            self.key_cache[layer_idx][:, :end, :, :],
            self.value_cache[layer_idx][:, :end, :, :],
        )

    def __getitem__(self, layer_idx):
        """Return (key, value) for a layer — BSHD buffer up to committed length."""
        end = self._seq_len
        return (
            self.key_cache[layer_idx][:, :end, :, :],
            self.value_cache[layer_idx][:, :end, :, :],
        )

    def __len__(self):
        return self.num_layers if self._seq_len > 0 else 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_len

    # ------------------------------------------------------------------
    # Compiled-graph-friendly methods (tensor ops only, no Python int slicing)
    # ------------------------------------------------------------------

    def prepare_write_idx(self, seq_len: int):
        """
        Pre-compute write indices for write_scratch_compiled().
        Must be called from EAGER code before the compiled step.

        Creates:
        - _write_idx_fixed: (seq_len,) int64 — 1D index [write_start, ..., write_start+seq_len-1]
        - _scatter_idx: (batch, seq_len, num_kv_heads, head_dim) int64 —
          full-shape index for scatter_, with each element containing
          the target position along dim 1 in the KV cache buffer.

        Uses scatter_ instead of index_copy_ because Inductor emits
        CPU-side dispatch ops for index_copy_ with a 1D index tensor
        when used inside a full model forward (28 layers × 2 K+V = 56
        "cudagraph partition due to non gpu ops"). scatter_ with a
        full-shape index compiles to a pure GPU kernel.
        """
        if (not hasattr(self, '_write_idx_fixed')
                or self._write_idx_fixed is None
                or self._write_idx_fixed.shape[0] != seq_len):
            self._write_idx_fixed = torch.zeros(
                seq_len, dtype=torch.int64, device=self.device,
            )
            torch._dynamo.mark_static_address(self._write_idx_fixed)
            # scatter index: (B, seq_len, H, D) — same shape as key_states
            self._scatter_idx = torch.zeros(
                self.batch_size, seq_len, self.num_kv_heads, self.head_dim,
                dtype=torch.int64, device=self.device,
            )
            torch._dynamo.mark_static_address(self._scatter_idx)

        # Fill 1D index
        self._write_idx_fixed.copy_(self._arange[:seq_len] + self.kv_write_start)
        # Broadcast to full shape for scatter_
        self._scatter_idx.copy_(
            self._write_idx_fixed[None, :, None, None].expand_as(self._scatter_idx)
        )

    def write_scratch_compiled(self, key_states, value_states, layer_idx):
        """
        In-place write at kv_write_start using scatter_.
        Does NOT advance the committed pointer.

        key_states/value_states: BSHD (B, S, H, D) — same layout as cache.

        Called inside the compiled graph. prepare_write_idx() must be called
        from eager code first.
        """
        self.key_cache[layer_idx].scatter_(1, self._scatter_idx, key_states)
        self.value_cache[layer_idx].scatter_(1, self._scatter_idx, value_states)

    def write_sparse(self, key_states, value_states, layer_idx, write_positions):
        """
        Write K, V at arbitrary per-batch positions using scatter_.
        Used in sparse diffusion steps to update only selected token positions.

        key_states/value_states: BSHD (B, num_tokens, H, D) — same layout as cache.
        write_positions: (B, num_tokens) — absolute positions in the KV cache
                         (e.g., past_len + token_indices_in_block).
        """
        k_layer = self.key_cache[layer_idx]  # [B, max_S, H, D]
        v_layer = self.value_cache[layer_idx]
        # Build scatter index: (B, num_tokens) → (B, num_tokens, H, D)
        idx = write_positions.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, self.num_kv_heads, self.head_dim,
        )
        k_layer.scatter_(1, idx, key_states)
        v_layer.scatter_(1, idx, value_states)

    def get_full_kv(self, layer_idx):
        """Return the full static buffer — constant shape (B, max_seq_len, H, D) BSHD."""
        return (
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
        )

    def set_scratch_seqlens(self, total_valid_len: int):
        """Set scratch_seqlens in-place. Called from eager code before compiled step."""
        self.scratch_seqlens.fill_(total_valid_len)

    def set_kv_write_start(self, offset: int):
        """Set kv_write_start in-place. Called from eager code before compiled step."""
        self.kv_write_start.fill_(offset)

    # ------------------------------------------------------------------
    # Legacy methods (used during eager prefill / commit, NOT in compiled graph)
    # ------------------------------------------------------------------

    def write_scratch(self, key_states, value_states, layer_idx, position):
        """Legacy: In-place write with Python int position. Eager only.
        key_states/value_states: BSHD — same layout as cache."""
        seq_len = key_states.shape[1]  # BSHD: seq is dim 1
        self.key_cache[layer_idx][:, position:position + seq_len, :, :].copy_(
            key_states
        )
        self.value_cache[layer_idx][:, position:position + seq_len, :, :].copy_(
            value_states
        )

    def get_kv_up_to(self, layer_idx, length):
        """Legacy: Return BSHD views up to a given length. Eager only."""
        return (
            self.key_cache[layer_idx][:, :length, :, :],
            self.value_cache[layer_idx][:, :length, :, :],
        )

    def get_seqused_mask(self, total_valid_len: int):
        """Build an SDPA-compatible attention mask from seqused."""
        self._attn_mask_buffer.fill_(0.0)
        invalid = self._kv_indices[None, :] >= total_valid_len
        self._attn_mask_buffer.masked_fill_(
            invalid[:, None, None, :], float("-inf"),
        )
        return self._attn_mask_buffer

    def get_seqused_mask_batched(self, seqused: torch.Tensor):
        """Build a per-batch SDPA mask from a (batch,) seqused tensor."""
        self._attn_mask_buffer.fill_(0.0)
        invalid = self._kv_indices[None, :] >= seqused[:, None]
        self._attn_mask_buffer.masked_fill_(
            invalid[:, None, None, :], float("-inf"),
        )
        return self._attn_mask_buffer

    def reset(self):
        """Reset committed length (does NOT zero out buffers)."""
        self._seq_len = 0
        self.cache_seqlens.fill_(0)

    def zero_and_reset(self):
        """Full reset: zero all buffers and reset pointers. For reuse across batches."""
        for layer_idx in range(self.num_layers):
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()
        self._seq_len = 0
        self.cache_seqlens.fill_(0)
        self.scratch_seqlens.fill_(0)
        self.kv_write_start.fill_(0)

    def restore_full_batch(self):
        """Re-expand to max_batch_size after compact_batch shrunk the batch.

        Re-allocates per-layer tensors and batch-indexed signals at
        max_batch_size.  Contents are stale — caller must zero_and_reset()
        before reuse.  This is only needed so that _need_reinit sees
        batch_size == incoming batch_size on the next call, avoiding a
        full reinit (which re-patches attention layers, etc.).
        """
        if self.batch_size == self.max_batch_size:
            return
        bs = self.max_batch_size
        self.key_cache = [
            torch.zeros(bs, self.max_seq_len, self.num_kv_heads, self.head_dim,
                        dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.value_cache = [
            torch.zeros(bs, self.max_seq_len, self.num_kv_heads, self.head_dim,
                        dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.cache_seqlens = torch.zeros(bs, dtype=torch.int32, device=self.device)
        self.scratch_seqlens = torch.zeros(bs, dtype=torch.int32, device=self.device)
        self._attn_mask_buffer = torch.zeros(
            bs, 1, 1, self.max_seq_len, dtype=self.dtype, device=self.device,
        )
        self._seq_len = 0
        self.kv_write_start.fill_(0)
        self.batch_size = bs

    def compact_batch(self, active_indices: "torch.Tensor"):
        """
        Remove finished batch slots by re-indexing the batch dimension.

        Used at block boundaries (eager code only) to shrink the batch
        when sequences finish — gives the same batch-compaction benefit
        as the baseline's ``input_ids = input_ids[~finished_flag]``.

        Creates new tensors (breaks mark_static_address), which is fine
        for eager execution. For CUDA graph mode, skip compaction and
        keep the full batch.

        Args:
            active_indices: (n_active,) int64 — indices of active batch slots.
        """
        n_active = active_indices.numel()

        # Re-index batch dim for each per-layer tensor [B, S, H, D]
        self.key_cache = [
            self.key_cache[i][active_indices, :, :, :].contiguous()
            for i in range(self.num_layers)
        ]
        self.value_cache = [
            self.value_cache[i][active_indices, :, :, :].contiguous()
            for i in range(self.num_layers)
        ]

        # Re-index batch-indexed control signals
        self.cache_seqlens = self.cache_seqlens[active_indices].contiguous()
        self.scratch_seqlens = self.scratch_seqlens[active_indices].contiguous()

        # Invalidate scatter index (will be rebuilt by prepare_write_idx)
        if hasattr(self, '_write_idx_fixed'):
            self._write_idx_fixed = None
            self._scatter_idx = None

        # Scalar signals stay the same
        # self.kv_write_start, self._arange — batch-independent

        # Rebuild batch-sized buffers
        self._attn_mask_buffer = torch.zeros(
            n_active, 1, 1, self.max_seq_len,
            dtype=self.dtype, device=self.device,
        )
        self._kv_indices = torch.arange(
            self.max_seq_len, dtype=torch.int32, device=self.device,
        )

        self.batch_size = n_active


class StaticBlockCache:
    """
    Static KV cache for within-block sub-block caching.

    Fixed size = (batch, block_size, num_kv_heads, head_dim) per layer.
    BSHD layout — matches StaticKVCache and flash_attn's native layout.
    Re-used across blocks by calling reset() between blocks.
    """

    def __init__(
        self,
        batch_size: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        self.key_cache = [
            torch.zeros(
                batch_size, block_size, num_kv_heads, head_dim,
                dtype=dtype, device=device,
            )
            for _ in range(num_layers)
        ]
        self.value_cache = [
            torch.zeros(
                batch_size, block_size, num_kv_heads, head_dim,
                dtype=dtype, device=device,
            )
            for _ in range(num_layers)
        ]

        self._num_initialized = 0

        self._mark_static_addresses()

    def _mark_static_addresses(self):
        _mark = torch._dynamo.mark_static_address
        for layer_idx in range(self.num_layers):
            _mark(self.key_cache[layer_idx])
            _mark(self.value_cache[layer_idx])

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # key_states/value_states: BSHD (B, S, H, D)
        self.key_cache[layer_idx][:, :key_states.shape[1], :, :].copy_(key_states)
        self.value_cache[layer_idx][:, :value_states.shape[1], :, :].copy_(value_states)
        if layer_idx >= self._num_initialized:
            self._num_initialized = layer_idx + 1
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def __getitem__(self, layer_idx):
        return (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        return self._num_initialized

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self._num_initialized == 0:
            return 0
        return self.block_size

    def reset(self):
        self._num_initialized = 0

    def compact_batch(self, active_indices: "torch.Tensor"):
        """
        Shrink the batch dimension to active slots at eager block boundaries.

        Mirrors StaticKVCache.compact_batch. Contents are reset() every block,
        so nothing needs preserving — but the batch dim must track the main KV
        cache (which compacts when sequences finish), or the next block's
        full-block write would mismatch. Re-index to stay row-aligned with it.
        """
        n_active = active_indices.numel()
        self.key_cache = [
            self.key_cache[i][active_indices, :, :, :].contiguous()
            for i in range(self.num_layers)
        ]
        self.value_cache = [
            self.value_cache[i][active_indices, :, :, :].contiguous()
            for i in range(self.num_layers)
        ]
        self.batch_size = n_active

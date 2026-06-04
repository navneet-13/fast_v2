"""
Batch bucketing for fixed-size batch execution in Fast-dLLM v2.

Keeps the batch dimension constant even when individual sequences finish,
avoiding CUDA graph re-recordings and torch.compile retracing caused by
dynamic batch sizes.

Finished sequences are kept in the batch with padding tokens, and their
outputs are saved for later retrieval.
"""

import torch
from typing import Dict, Optional


class BatchBucket:
    """
    Manages a fixed-size batch where completed sequences are masked rather
    than removed.

    Usage:
        bucket = BatchBucket(batch_size, device)
        ...
        bucket.mark_finished(finished_mask, current_output)
        if bucket.all_finished():
            break
        # Use bucket.active_mask to skip computation for finished slots
        ...
        results = bucket.get_results(fallback_output)
    """

    def __init__(
        self,
        batch_size: int,
        device: torch.device,
        pad_token_id: int = 151645,
    ):
        self.batch_size = batch_size
        self.device = device
        self.pad_token_id = pad_token_id

        # True for slots that are still generating
        self.active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Maps original batch index → finished output tensor
        self._finished_outputs: Dict[int, torch.Tensor] = {}

        # Tracks the original indices (identity mapping since we don't remove)
        self._original_indices = torch.arange(batch_size, device=device)

    def mark_finished(
        self,
        newly_finished: torch.Tensor,
        current_x_t: torch.Tensor,
        seq_len: Optional[torch.Tensor] = None,
    ):
        """
        Mark batch slots as finished and save their output.

        Args:
            newly_finished: (batch,) bool tensor — True for newly completed slots.
            current_x_t: (batch, seq) current token tensor to save.
            seq_len: (batch,) optional per-element sequence lengths for truncation.
        """
        if not newly_finished.any():
            return

        for idx in newly_finished.nonzero(as_tuple=True)[0]:
            i = idx.item()
            if i not in self._finished_outputs:
                self._finished_outputs[i] = current_x_t[i].clone()

        self.active_mask = self.active_mask & ~newly_finished

    def pad_finished_slots(self, x_t: torch.Tensor, block_size: int):
        """
        Overwrite the last `block_size` tokens of finished slots with pad tokens.
        Prevents finished sequences from affecting attention / generation.
        Operates in-place.
        """
        if self.active_mask.all():
            return
        finished = ~self.active_mask
        x_t[finished, -block_size:] = self.pad_token_id

    def all_finished(self) -> bool:
        return not self.active_mask.any()

    def num_active(self) -> int:
        return self.active_mask.sum().item()

    def get_results(self, fallback_x_t: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Return all results: finished outputs + unfinished fallback.

        After compaction, self.batch_size is the compacted size and
        self._original_indices maps compacted idx → original idx.
        Fallback uses _original_indices to map back correctly.

        Args:
            fallback_x_t: (current_batch, seq) tensor for sequences that didn't
                          finish within max_new_tokens.

        Returns:
            Dict mapping original batch index → output tensor.
        """
        results = dict(self._finished_outputs)
        for idx in range(self.batch_size):
            orig_idx = self._original_indices[idx].item()
            if orig_idx >= 0 and orig_idx not in results:
                results[orig_idx] = fallback_x_t[idx].clone()
        return results

    def get_finished_mask_for_cache(self) -> torch.Tensor:
        """Return the inverse active mask (True = finished) for cache ops."""
        return ~self.active_mask

    def compact(self):
        """
        Remove finished slots, returning active indices for re-indexing
        other batch-indexed tensors (x_t, seq_len, etc.).

        After calling, self.batch_size shrinks and active_mask is all-True.
        The _original_indices mapping is preserved so get_results() still
        maps to the correct original batch positions.

        Returns:
            active_indices: (num_active,) int64 tensor — indices into the
                old batch dimension for re-indexing.
            None if no compaction needed (all active or none finished).
        """
        if self.active_mask.all():
            return None

        active_indices = self.active_mask.nonzero(as_tuple=True)[0]
        n_active = active_indices.numel()

        if n_active == 0:
            return None

        # Update internal state
        self._original_indices = self._original_indices[active_indices]
        self.batch_size = n_active
        self.active_mask = torch.ones(n_active, dtype=torch.bool, device=self.device)

        return active_indices

    def compact_to(self, target_size: int):
        """
        Compact active slots to front, resize to target_size.

        Active elements are moved to indices [0, n_active).
        Slots [n_active, target_size) are padding (active_mask=False,
        _original_indices=-1).

        Returns:
            active_indices: (n_active,) int64 — indices into the old batch
                dim for re-indexing other tensors.
            None if no compaction needed.
        """
        active_indices = self.active_mask.nonzero(as_tuple=True)[0]
        n_active = active_indices.numel()

        if n_active == 0:
            return None
        if n_active == self.batch_size and self.batch_size == target_size:
            return None

        assert target_size >= n_active, (
            f"target_size={target_size} < n_active={n_active}"
        )

        # Preserve original index mapping for active slots
        active_originals = self._original_indices[active_indices]
        self._original_indices = torch.full(
            (target_size,), -1, device=self.device, dtype=active_originals.dtype,
        )
        self._original_indices[:n_active] = active_originals

        self.batch_size = target_size
        self.active_mask = torch.zeros(
            target_size, dtype=torch.bool, device=self.device,
        )
        self.active_mask[:n_active] = True

        return active_indices

    def mark_finished_compacted(
        self,
        newly_finished: torch.Tensor,
        current_x_t: torch.Tensor,
    ):
        """
        Like mark_finished, but maps back to original batch indices
        using _original_indices (needed after compact() has been called).
        """
        if not newly_finished.any():
            return

        for idx in newly_finished.nonzero(as_tuple=True)[0]:
            i = idx.item()
            orig_idx = self._original_indices[i].item()
            if orig_idx >= 0 and orig_idx not in self._finished_outputs:
                self._finished_outputs[orig_idx] = current_x_t[i].clone()

        self.active_mask = self.active_mask & ~newly_finished


def compute_valid_batch_sizes(max_batch_size: int, compact_step: int) -> list:
    """
    Compute valid batch sizes for compact compile mode.

    Returns sorted list — union of:
    - Powers of 2 up to max_batch_size (1, 2, 4, 8, ...)
    - max_batch_size - n*compact_step for n=0,1,2,... while > 0
    Always includes 1 and max_batch_size.
    """
    sizes = {1, max_batch_size}
    p = 1
    while p <= max_batch_size:
        sizes.add(p)
        p *= 2
    s = max_batch_size
    while s > 0:
        sizes.add(s)
        s -= compact_step
    return sorted(sizes)


def snap_batch_size(num_active: int, valid_sizes: list) -> int:
    """Return the smallest valid batch size >= num_active."""
    for s in valid_sizes:
        if s >= num_active:
            return s
    return valid_sizes[-1]


def compact_and_pad(tensor: torch.Tensor, active_indices: torch.Tensor,
                    pad_count: int, pad_value) -> torch.Tensor:
    """Re-index batch dim by active_indices and pad with pad_count rows."""
    selected = tensor[active_indices]
    if pad_count > 0:
        pad_shape = (pad_count,) + tensor.shape[1:]
        pad = torch.full(
            pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device,
        )
        return torch.cat([selected, pad], dim=0)
    return selected

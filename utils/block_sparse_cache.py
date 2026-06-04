"""
Block-level sparse cache for token-sparse diffusion in Fast-dLLM v2.

Caches per-layer intermediate states (hidden_states input, attention output,
MLP output) for the current block. Used to:
1. Compute cosine similarity between consecutive step hidden states to
   identify which tokens changed the most (lowest similarity → recompute).
2. Provide cached attention/MLP outputs for tokens that are NOT recomputed
   in sparse steps (scatter updated values back into cached tensors).

Reset and reused across blocks. Stacked tensor layout [L, B, block_size, D]
for each cache.
"""

import torch


class BlockSparseCache:
    """
    Caches per-layer hidden states, attention outputs, and MLP outputs
    for the current block during token-sparse diffusion.

    Three stacked-tensor caches, each of shape
    (num_layers, batch_size, block_size, hidden_size):

    - layer_input:  hidden_states entering each decoder layer
                    (before input_layernorm). Used for cosine similarity.
    - attn_output:  output of self_attn (after o_proj, before residual add).
                    Sparse attention results are scattered into this.
    - mlp_input:    post-attention residual (hidden_states + attn_output)
                    entering the MLP half. Used for second-stage cosine
                    similarity to select which tokens go through MLP.
    - mlp_output:   output of MLP (after down_proj, before residual add).
                    Sparse MLP results are scattered into this.

    Layer output = mlp_input + mlp_output
                 = (layer_input + attn_output) + mlp_output.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        block_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device

        # Stacked caches: [L, B, block_size, hidden_size]
        self.layer_input = torch.zeros(
            num_layers, batch_size, block_size, hidden_size,
            dtype=dtype, device=device,
        )
        self.attn_output = torch.zeros(
            num_layers, batch_size, block_size, hidden_size,
            dtype=dtype, device=device,
        )
        self.mlp_input = torch.zeros(
            num_layers, batch_size, block_size, hidden_size,
            dtype=dtype, device=device,
        )
        self.mlp_output = torch.zeros(
            num_layers, batch_size, block_size, hidden_size,
            dtype=dtype, device=device,
        )

    # ---- Write methods ----

    def cache_layer_input(self, layer_idx: int, hidden_states: torch.Tensor):
        """Cache hidden_states entering this layer (before input_layernorm)."""
        self.layer_input.select(0, layer_idx).copy_(hidden_states)

    def cache_attn_output(self, layer_idx: int, attn_out: torch.Tensor):
        """Cache attention output (after o_proj, before residual)."""
        self.attn_output.select(0, layer_idx).copy_(attn_out)

    def cache_mlp_input(self, layer_idx: int, mid: torch.Tensor):
        """Cache post-attention residual entering the MLP half."""
        self.mlp_input.select(0, layer_idx).copy_(mid)

    def cache_mlp_output(self, layer_idx: int, mlp_out: torch.Tensor):
        """Cache MLP output (after down_proj, before residual)."""
        self.mlp_output.select(0, layer_idx).copy_(mlp_out)

    # ---- Read methods ----

    def get_layer_input(self, layer_idx: int) -> torch.Tensor:
        """Return cached layer input: (B, block_size, hidden_size)."""
        return self.layer_input.select(0, layer_idx)

    def get_attn_output(self, layer_idx: int) -> torch.Tensor:
        """Return cached attn output: (B, block_size, hidden_size)."""
        return self.attn_output.select(0, layer_idx)

    def get_mlp_input(self, layer_idx: int) -> torch.Tensor:
        """Return cached post-attn residual: (B, block_size, hidden_size)."""
        return self.mlp_input.select(0, layer_idx)

    def get_mlp_output(self, layer_idx: int) -> torch.Tensor:
        """Return cached MLP output: (B, block_size, hidden_size)."""
        return self.mlp_output.select(0, layer_idx)

    # ---- Scatter methods (update selected positions in-place) ----

    def scatter_attn_output(
        self, layer_idx: int, token_indices: torch.Tensor, values: torch.Tensor,
    ):
        """
        Scatter sparse attention outputs into the cached tensor.
        token_indices: (B, num_tokens) — positions within block
        values: (B, num_tokens, hidden_size) — computed attn outputs
        """
        idx = token_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        self.attn_output.select(0, layer_idx).scatter_(1, idx, values)

    def scatter_mlp_output(
        self, layer_idx: int, token_indices: torch.Tensor, values: torch.Tensor,
    ):
        """
        Scatter sparse MLP outputs into the cached tensor.
        token_indices: (B, num_tokens) — positions within block
        values: (B, num_tokens, hidden_size) — computed MLP outputs
        """
        idx = token_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        self.mlp_output.select(0, layer_idx).scatter_(1, idx, values)

    # ---- Reset ----

    def reset(self):
        """Reset for a new block. No need to zero — the first step of each
        block is always dense, which overwrites all positions via .copy_()."""
        pass

    def compact_batch(self, active_indices: "torch.Tensor"):
        """Remove finished batch slots by re-indexing batch dimension (dim=1)."""
        self.layer_input = self.layer_input[:, active_indices, :, :].contiguous()
        self.attn_output = self.attn_output[:, active_indices, :, :].contiguous()
        self.mlp_input = self.mlp_input[:, active_indices, :, :].contiguous()
        self.mlp_output = self.mlp_output[:, active_indices, :, :].contiguous()
        self.batch_size = active_indices.numel()

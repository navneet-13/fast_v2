"""Per-block deep-layer KV buffer for FOCUS on the DynamicCache path.

Holds the CURRENT block's key/value tensors (BHSD: [B, num_kv_heads, block_size,
head_dim]) for each deep layer across diffusion steps, so FOCUS-evicted/frozen
tokens keep their KV while only selected tokens are recomputed. Reset per block;
the dense seed step overwrites every position via write_full(), so reset() is a
no-op (mirrors BlockSparseCache).
"""
import torch


class DynamicBlockKV:
    def __init__(self, deep_layer_start, num_layers, batch_size, num_kv_heads,
                 block_size, head_dim, dtype, device):
        self.deep_layer_start = deep_layer_start
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.k = {
            L: torch.zeros(batch_size, num_kv_heads, block_size, head_dim,
                           dtype=dtype, device=device)
            for L in range(deep_layer_start, num_layers)
        }
        self.v = {
            L: torch.zeros(batch_size, num_kv_heads, block_size, head_dim,
                           dtype=dtype, device=device)
            for L in range(deep_layer_start, num_layers)
        }
        self.batch_size = batch_size

    def write_full(self, layer_idx, k, v):
        """Overwrite the whole block KV for a layer. k, v: [B, H_kv, block_size, D]."""
        self.k[layer_idx].copy_(k)
        self.v[layer_idx].copy_(v)

    def write(self, layer_idx, k, v, token_indices):
        """Scatter selected positions. k, v: [B, H_kv, n, D]; token_indices: [B, n]."""
        idx = token_indices[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[3])
        self.k[layer_idx].scatter_(2, idx, k)
        self.v[layer_idx].scatter_(2, idx, v)

    def get(self, layer_idx):
        return self.k[layer_idx], self.v[layer_idx]

    def reset(self):
        # Seed step overwrites every position each block; nothing to clear.
        pass

    def compact_batch(self, active_indices):
        for L in self.k:
            self.k[L] = self.k[L][active_indices].contiguous()
            self.v[L] = self.v[L][active_indices].contiguous()
        self.batch_size = active_indices.numel()

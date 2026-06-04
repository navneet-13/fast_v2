import importlib.util
import math
import os
from pathlib import Path
from typing import Callable, Optional, Union
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from transformers.generation.utils import GenerateDecoderOnlyOutput  
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging
from .configuration import Fast_dLLM_QwenConfig
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from einops import rearrange, repeat

try:
    from flash_attn import flash_attn_func as _flash_attn_func
except Exception:
    _flash_attn_func = None

logger = logging.get_logger(__name__)

# Optional: fused index-gather + input RMSNorm for forward_sparse (see cuda_bench/fused_sparse_extension.py).
_SPARSE_FUSED_EXT_MOD = None
_SPARSE_FUSED_EXT_TRIED = False


def _get_sparse_fused_extension_module():
    """Load fast_v2/cuda_bench/fused_sparse_extension.py once (any ancestor named path with cuda_bench/)."""
    global _SPARSE_FUSED_EXT_MOD, _SPARSE_FUSED_EXT_TRIED
    if _SPARSE_FUSED_EXT_TRIED:
        return _SPARSE_FUSED_EXT_MOD
    _SPARSE_FUSED_EXT_TRIED = True
    try:
        for par in Path(__file__).resolve().parents:
            ext_py = par / "cuda_bench" / "fused_sparse_extension.py"
            if ext_py.is_file():
                spec = importlib.util.spec_from_file_location(
                    "fast_v2_cuda_bench_fused_sparse_extension", ext_py
                )
                mod = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(mod)
                _SPARSE_FUSED_EXT_MOD = mod
                return mod
    except Exception as exc:
        logger.warning("Could not load cuda_bench.fused_sparse_extension: %s", exc)
    _SPARSE_FUSED_EXT_MOD = None
    return None


def _try_fused_gather_input_rmsnorm(hidden_states, token_indices, input_layernorm):
    """If FAST_DLLM_FUSED_GATHER_INPUT_NORM=1 and CUDA extension loads, return (gathered, normed); else None."""
    if os.environ.get("FAST_DLLM_FUSED_GATHER_INPUT_NORM", "0") != "1":
        return None
    mod = _get_sparse_fused_extension_module()
    if mod is None:
        return None
    try:
        return mod.fused_gather_input_rmsnorm_pair(
            hidden_states, token_indices, input_layernorm
        )
    except Exception as exc:
        logger.warning(
            "FAST_DLLM_FUSED_GATHER_INPUT_NORM: fused gather+RMSNorm failed (%s); using PyTorch.",
            exc,
        )
        return None


@dataclass
class CausalLMOutputWithPastAndBlockCache(CausalLMOutputWithPast):
    block_past_key_values: Optional[Cache] = None

@dataclass
class BaseModelOutputWithPastAndBlockCache(BaseModelOutputWithPast):
    block_past_key_values: Optional[Cache] = None


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def fused_flex_attention(q, k, v, mask=None):
    return flex_attention(q, k, v, block_mask=mask, enable_gqa=True)

def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """
    # Indicate whether token belongs to xt or x0
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    # Compute block indices
    block_q = torch.where(x0_flag_q == 1,
                        (q_idx - n) // block_size,
                        q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1,
                        (kv_idx - n) // block_size,
                        kv_idx // block_size)

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (
    (block_q > block_kv)
    & (x0_flag_kv == 1)
    & (x0_flag_q == 0)
    )

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal

def eval_block_diff_mask(q_idx, kv_idx, block_size=None):
    # Compute block indices
    block_q = q_idx // block_size
    block_kv = kv_idx // block_size

    return block_q >= block_kv

class Fast_dLLM_QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _focus_importance(q, k, mask_idx, scaling):
    """FOCUS attention importance over masked tokens.

    q: (B, H, S, d) query states (post-RoPE, heads GQA-expanded to H)
    k: (B, H, S, d) key states (post-RoPE, GQA-expanded to H)
    mask_idx: (B, S) bool — True where the token is currently masked (mask_id)
    Returns: (B, S) importance = attention mass received by each masked token
             from masked queries, max-pooled (window 3) + softmaxed, summed over
             query positions and heads.
    """
    B, H, S, _ = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaling          # (B,H,S,S)
    key_mask = mask_idx[:, None, None, :]                            # (B,1,1,S)
    scores = scores.masked_fill(~key_mask, float("-inf"))
    pooled = F.max_pool1d(
        scores.reshape(B * H * S, 1, S), kernel_size=3, stride=1, padding=1,
    ).reshape(B, H, S, S)
    pooled = pooled.masked_fill(~key_mask, float("-inf"))           # re-mask: no leak to non-masked keys
    weights = torch.softmax(pooled, dim=-1)                          # over keys
    weights = torch.nan_to_num(weights, nan=0.0)                     # all-(-inf) rows
    q_mask = mask_idx[:, None, :, None].to(weights.dtype)            # (B,1,S,1)
    weights = weights * q_mask                                       # drop non-masked queries
    imp = weights.sum(dim=-2).sum(dim=1)                             # (B,S)
    return imp


def _focus_select(delta, mask_idx, avg_decoded, focus_alpha, retain_override=None, frozen=None):
    """FOCUS token selection over a full block.

    delta: (B, S) importance delta (imp_layer1 - imp_layer0)
    mask_idx: (B, S) bool — currently-masked positions (selection restricted here)
    avg_decoded: float running mean of tokens decoded per step
    focus_alpha: float retain multiplier
    retain_override: optional float in (0,1] — fixed retain fraction of the block
    Returns: (token_indices (B, Ksel) long sorted block positions, Ksel int)
    """
    B, S = delta.shape
    neg_inf = torch.finfo(delta.dtype).min
    masked_delta = torch.where(mask_idx, delta, torch.full_like(delta, neg_inf))

    if retain_override is not None:
        K = max(1, min(S, int(math.ceil(retain_override * S))))
    else:
        K = max(1, min(S, int(math.ceil(avg_decoded * focus_alpha))))

    # per-sequence threshold = mean + std over masked deltas
    valid = mask_idx
    cnt = valid.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (torch.where(valid, delta, torch.zeros_like(delta)).sum(1, keepdim=True)) / cnt
    var = (torch.where(valid, (delta - mean) ** 2, torch.zeros_like(delta)).sum(1, keepdim=True)) / cnt
    thr = mean + var.sqrt()
    cand = valid & (delta >= thr)                                   # (B,S)

    # adjacency: keep token i if i+1 is a candidate
    adj = torch.zeros_like(cand)
    adj[:, :-1] = cand[:, 1:]
    mustkeep = (cand | adj) & valid

    # top-K highest-delta masked tokens (FOCUS retain target)
    topk_idx = masked_delta.topk(K, dim=1).indices                  # (B,K)
    topk_mask = torch.zeros_like(valid)
    topk_mask.scatter_(1, topk_idx, True)
    topk_mask = topk_mask & valid

    masked_selected = (mustkeep | topk_mask) & valid          # FOCUS-selected decodable masked tokens
    # FOCUS retains decoded tokens (so their KV refreshes with the real token id)
    # plus the selected decodable masked tokens; only non-decodable masked tokens are evicted.
    retain = (~mask_idx) | masked_selected                    # (B, S) bool
    if frozen is not None:
        # Delayed cache: settled tokens are never recomputed — drop them from the
        # retain set (budget) and force them out of the top-k by priority.
        retain = retain & ~frozen
    Ksel = int(torch.clamp(retain.sum(dim=1).max(), min=K).clamp(max=S).item())
    if frozen is not None:
        # The K=ceil(avg_decoded*alpha) floor ignores `frozen`, so when it exceeds the
        # non-frozen count, topk would be forced to return frozen positions (priority
        # neg_inf), leaking them into the recompute set and defeating the cache. Clamp
        # Ksel to the non-frozen count. Use .max() over the batch, NOT .min(): retain is a
        # subset of non-frozen, so retain.sum().max() <= n_free.max(), and clamping to
        # n_free.max() never truncates any row's retained tokens. (.min() WOULD truncate
        # the busiest row's maskable tokens — a correctness bug.) Frozen-padding in a
        # more-frozen row is harmless: Ksel is uniform so FLOPs are unchanged and a frozen
        # recompute reproduces the same KV.
        n_free = int((~frozen).sum(dim=1).max().item())
        Ksel = max(1, min(Ksel, n_free))
    # priority: retained positions first (boosted), padding budget filled by
    # highest-delta non-retained masked tokens. (decoded delta ~0; boost keeps them in.)
    priority = torch.where(retain, delta + 1e4, delta)
    if frozen is not None:
        priority = torch.where(frozen, torch.full_like(priority, neg_inf), priority)
    token_indices = priority.topk(Ksel, dim=1).indices
    token_indices = token_indices.sort(dim=1).values
    return token_indices, Ksel


class Fast_dLLM_QwenAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Fast_dLLM_QwenConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        update_past_key_values: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if self.training:
            #split q into two parts
            q_1 = query_states[:,:,:query_states.shape[2]//2]
            q_2 = query_states[:,:,query_states.shape[2]//2:]
            #split k into two parts
            k_1 = key_states[:,:,:key_states.shape[2]//2]
            k_2 = key_states[:,:,key_states.shape[2]//2:]
            q_1, k_1 = apply_rotary_pos_emb(q_1, k_1, cos, sin)
            q_2, k_2 = apply_rotary_pos_emb(q_2, k_2, cos, sin)
            query_states = torch.cat((q_1, q_2), dim=-2)
            key_states = torch.cat((k_1, k_2), dim=-2)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if block_past_key_values is not None:
            if len(block_past_key_values) <= self.layer_idx:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = block_past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                block_cache_key_states = block_past_key_values[self.layer_idx][0]
                block_cache_value_states = block_past_key_values[self.layer_idx][1]
                
                block_cache_key_states[:, :, replace_position:replace_position+key_states.shape[2]] = key_states
                block_cache_value_states[:, :, replace_position:replace_position+value_states.shape[2]] = value_states
                key_states = block_cache_key_states
                value_states = block_cache_value_states

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            if update_past_key_values:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            elif len(past_key_value) > self.layer_idx:
                key_states = torch.cat((past_key_value[self.layer_idx][0], key_states), dim=-2)
                value_states = torch.cat((past_key_value[self.layer_idx][1], value_states), dim=-2)

        if self.training:
            attn_output = fused_flex_attention(query_states, key_states, value_states, mask=attention_mask)
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                is_causal=False,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # main diff with Llama
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

@use_kernel_forward_from_hub("RMSNorm")
class Fast_dLLM_QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Fast_dLLM_QwenRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Fast_dLLM_QwenDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Fast_dLLM_QwenConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Fast_dLLM_QwenAttention(config=config, layer_idx=layer_idx)

        self.mlp = Fast_dLLM_QwenMLP(config)
        self.input_layernorm = Fast_dLLM_QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Fast_dLLM_QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        update_past_key_values: Optional[bool] = False,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            update_past_key_values=update_past_key_values,
            use_block_cache=use_block_cache,
            block_past_key_values=block_past_key_values,
            replace_position=replace_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states



class Fast_dLLM_QwenPreTrainedModel(PreTrainedModel):
    config_class = Fast_dLLM_QwenConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Fast_dLLM_QwenDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Fast_dLLM_QwenDecoderLayer,
        "attentions": Fast_dLLM_QwenAttention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Fast_dLLM_QwenRMSNorm):
            module.weight.data.fill_(1.0)


class Fast_dLLM_QwenRotaryEmbedding(nn.Module):
    def __init__(self, config: Fast_dLLM_QwenConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)



class Fast_dLLM_QwenModel(Fast_dLLM_QwenPreTrainedModel):
    def __init__(self, config: Fast_dLLM_QwenConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.bd_size = config.bd_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Fast_dLLM_QwenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Fast_dLLM_QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Fast_dLLM_QwenRotaryEmbedding(config=config)
        self.gradient_checkpointing = True

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


    def eval_mask(self, seqlen, block_size, cache_seq_len):
        q_indices = torch.arange(seqlen) + cache_seq_len
        k_indices = torch.arange(seqlen + cache_seq_len)
        mask = eval_block_diff_mask(
            q_idx=q_indices[:, None], 
            kv_idx=k_indices[None, :], 
            block_size=block_size
        )
        return mask

    def gen_mask(self, seqlen, block_size, B, H):
        mask = create_block_mask(
            partial(block_diff_mask, block_size=block_size, n=seqlen),
            B=B, H=H, Q_LEN=seqlen*2, KV_LEN=seqlen*2)

        return mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        update_past_key_values: Optional[bool] = False,
        block_size: Optional[int] = 32,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if use_block_cache and block_past_key_values is None:
            block_past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            if self.training:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]//2, device=inputs_embeds.device
                )
            else:
                if use_block_cache:
                    block_start_position = past_seen_tokens+replace_position if replace_position is not None else past_seen_tokens
                    cache_position = torch.arange(
                        block_start_position, block_start_position + inputs_embeds.shape[1], device=inputs_embeds.device
                    )
                else:
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1] if not self.training else inputs_embeds.shape[1]//2, device=inputs_embeds.device
                    )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        if self.training:
            attention_mask = self.gen_mask(labels.shape[1], self.bd_size, labels.shape[0], self.config.num_attention_heads).to(device=inputs_embeds.device)
        else:
            if use_block_cache and block_past_key_values.get_seq_length() != 0:
                attention_mask = None
            else:
                attention_mask = self.eval_mask(input_ids.shape[1], block_size, past_key_values.get_seq_length() if past_key_values is not None else 0).to(device=inputs_embeds.device)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                update_past_key_values=update_past_key_values,
                use_block_cache=use_block_cache,
                block_past_key_values=block_past_key_values,
                replace_position=replace_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPastAndBlockCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            block_past_key_values=block_past_key_values if use_block_cache else None,
        )

    def forward_dynamo(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        update_past_key_values: Optional[bool] = False,
        block_size: Optional[int] = 32,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs
    ) -> BaseModelOutputWithPast:
        """
        Dynamo-compatible inference-only forward for batch_sample_dynamo.

        Key differences from forward():
        - Inference-only: no training path, no noise/mask augmentation.
        - Skips eval_mask computation (passes attention_mask=None).
          Valid for all single-block operations (diffusion steps and cache-update
          steps) because the block-causal mask block_q >= block_kv is trivially
          True: all queries are in the current block, all KVs are in blocks ≤ current.
        - Does NOT allocate a new DynamicCache; requires past_key_values to be
          passed as a StaticKVCache (pre-allocated by batch_sample_dynamo).
        - Does NOT allocate a new DynamicCache for block_past_key_values.

        IMPORTANT: Attention layers must be patched via patch_attention_layers()
        before calling this method. The patched layers handle StaticKVCache
        in-place writes (write_scratch) instead of torch.cat, enabling CUDA
        graph compatibility.
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            if use_block_cache and replace_position is not None:
                block_start = past_seen_tokens + replace_position
                cache_position = torch.arange(
                    block_start,
                    block_start + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
            else:
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )

        position_ids = cache_position.unsqueeze(0)

        # Skip mask computation: during single-block diffusion and cache-update steps
        # the block-causal mask is trivially True — passing None lets PyTorch select
        # the most efficient SDPA kernel (flash / memory-efficient / math).
        attention_mask = None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                update_past_key_values=update_past_key_values,
                use_block_cache=use_block_cache,
                block_past_key_values=block_past_key_values,
                replace_position=replace_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPastAndBlockCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            block_past_key_values=block_past_key_values if use_block_cache else None,
        )

    def forward_sparse(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        update_past_key_values: Optional[bool] = False,
        block_sparse_cache=None,
        is_dense_step: bool = True,
        transfer_ratio: float = 1.0,
        attn_backend=None,
        **kwargs,
    ) -> BaseModelOutputWithPastAndBlockCache:
        """
        Sparse-token inference forward for batch_sample_sparse.

        Dense step (is_dense_step=True):
          Full decoder layer forward. Caches hidden_states input, attn_output,
          and mlp_output per layer into block_sparse_cache.

        Sparse step (is_dense_step=False):
          Per layer: computes cosine similarity between current hidden_states
          and cached layer input to select a fraction (transfer_ratio) of tokens
          with lowest similarity. Only those tokens go through input_layernorm,
          Q/K/V projection, RoPE, attention, o_proj, post_attn_layernorm, MLP.
          Results are scattered back into cached attn/mlp outputs.
          KV cache is updated only at the selected positions.

        Attention layers should be patched (for the dense step to use
        StaticKVCache correctly). The sparse step bypasses the patched forward
        and accesses projection weights directly.
        """
        hidden_states = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1],
                device=hidden_states.device,
            )

        position_ids = cache_position.unsqueeze(0)
        attention_mask = None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        B = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // num_heads

        cos_full, sin_full = position_embeddings
        # Pre-cat for single-gather in sparse steps: (1, seq_len, 2*head_dim)
        cos_sin_full = torch.cat([cos_full, sin_full], dim=-1)

        for layer_idx in range(num_layers):
            decoder_layer = self.layers[layer_idx]

            if is_dense_step:
                # ---- Dense: full layer forward + cache intermediates ----
                block_sparse_cache.cache_layer_input(layer_idx, hidden_states)

                residual = hidden_states
                hidden_states_norm = decoder_layer.input_layernorm(hidden_states)
                # Use the (patched) self_attn forward
                attn_out = decoder_layer.self_attn(
                    hidden_states_norm,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    update_past_key_values=update_past_key_values,
                )
                block_sparse_cache.cache_attn_output(layer_idx, attn_out)

                hidden_states = residual + attn_out

                # Cache post-attn residual so the first sparse step can
                # compute cosine similarity for MLP token selection.
                block_sparse_cache.cache_mlp_input(layer_idx, hidden_states)

                residual = hidden_states
                hidden_states_norm = decoder_layer.post_attention_layernorm(hidden_states)
                mlp_out = decoder_layer.mlp(hidden_states_norm)
                hidden_states = residual + mlp_out

                block_sparse_cache.cache_mlp_output(layer_idx, mlp_out)
        


            else:
                # ---- Sparse: recompute only tokens with lowest cosine sim ----
                cached_input = block_sparse_cache.get_layer_input(layer_idx)
                


                # Cosine similarity per token
                cos_sim = F.cosine_similarity(hidden_states, cached_input, dim=-1)  # (B, seq_len)

                # Select fraction with lowest similarity
                num_tokens = int(transfer_ratio * seq_len)
                _, token_indices = cos_sim.topk(num_tokens, dim=-1, largest=False)
                token_indices = token_indices.sort(dim=-1).values  # (B, num_tokens)

                # Update layer input cache
                block_sparse_cache.cache_layer_input(layer_idx, hidden_states)
                
                # ---- Gather + input RMSNorm (optional fused CUDA; same math as Fast_dLLM_QwenRMSNorm) ----
                fused_pair = _try_fused_gather_input_rmsnorm(
                    hidden_states, token_indices, decoder_layer.input_layernorm
                )
                if fused_pair is not None:
                    selected_hidden, selected_norm = fused_pair
                else:
                    idx_h = token_indices.unsqueeze(-1).expand(
                        -1, -1, hidden_states.shape[-1]
                    )
                    selected_hidden = hidden_states.gather(1, idx_h)  # (B, num_tokens, H)
                    selected_norm = decoder_layer.input_layernorm(selected_hidden)

                # ---- Q, K, V projections for selected tokens (BSHD) ----
                attn = decoder_layer.self_attn
                q = attn.q_proj(selected_norm).view(B, num_tokens, num_heads, head_dim)      # BSHD
                k = attn.k_proj(selected_norm).view(B, num_tokens, num_kv_heads, head_dim)    # BSHD
                v = attn.v_proj(selected_norm).view(B, num_tokens, num_kv_heads, head_dim)    # BSHD

                # ---- RoPE for selected positions (single gather + chunk) ----
                idx_pos = token_indices.unsqueeze(-1).expand(-1, -1, cos_sin_full.shape[-1])
                selected_cos, selected_sin = cos_sin_full.expand(B, -1, -1).gather(1, idx_pos).chunk(2, dim=-1)
                q, k = apply_rotary_pos_emb(q, k, selected_cos, selected_sin, unsqueeze_dim=2)

                # ---- Update KV cache at selected positions ----
                past_len = past_key_values.get_seq_length()
                write_positions = token_indices + past_len  # (B, num_tokens)
                past_key_values.write_sparse(k, v, layer_idx, write_positions)

                # ---- Attention: sparse Q, full KV cache ----
                full_k, full_v = past_key_values.get_full_kv(layer_idx)
                sparse_attn_out = attn_backend.flash_kvcache_attention(
                    q, full_k, full_v,
                    cache_seqlens=past_key_values.scratch_seqlens,
                    is_causal=False,
                    scaling=attn.scaling,
                )
                # sparse_attn_out: (B, num_tokens, num_heads, head_dim) BSHD
                sparse_attn_out = sparse_attn_out.reshape(B, num_tokens, -1).contiguous()
                sparse_attn_out = attn.o_proj(sparse_attn_out)
                # sparse_attn_out: (B, num_tokens, hidden_size)

                # ---- Scatter attn output into cache ----
                block_sparse_cache.scatter_attn_output(layer_idx, token_indices, sparse_attn_out)
                attn_output_full = block_sparse_cache.get_attn_output(layer_idx)

                # ---- Post-attention residual (all tokens) ----
                mid = hidden_states + attn_output_full  # (B, seq_len, H)

                # # ---- Second-stage sparsity: re-select tokens for MLP ----
                # # Compare post-attn mid against cached MLP input from previous
                # # diffusion step to find which positions changed the most.
                # cached_mid = block_sparse_cache.get_mlp_input(layer_idx)
                # mid_cos_sim = F.cosine_similarity(mid, cached_mid, dim=-1)  # (B, seq_len)
                # num_mlp_tokens = max(1, int(transfer_ratio * seq_len))
                # _, mlp_token_indices = mid_cos_sim.topk(num_mlp_tokens, dim=-1, largest=False)
                # mlp_token_indices = mlp_token_indices.sort(dim=-1).values  # (B, num_mlp_tokens)

                # # Cache current mid for next step's comparison
                # block_sparse_cache.cache_mlp_input(layer_idx, mid)

                # # Gather MLP-selected tokens
                # idx_mlp = mlp_token_indices.unsqueeze(-1).expand(-1, -1, mid.shape[-1])
                # selected_mid = mid.gather(1, idx_mlp)  # (B, num_mlp_tokens, H)

                selected_mid = selected_hidden + sparse_attn_out
                selected_mid_norm = decoder_layer.post_attention_layernorm(selected_mid)

                # ---- Compute MLP output for MLP-selected tokens ----
                sparse_mlp_out = decoder_layer.mlp(selected_mid_norm)

                # ---- Scatter MLP output into cache (using MLP indices) ----
                block_sparse_cache.scatter_mlp_output(layer_idx, token_indices, sparse_mlp_out)
                mlp_output_full = block_sparse_cache.get_mlp_output(layer_idx)

                # ---- Final output (all tokens) ----
                hidden_states = mid + mlp_output_full

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPastAndBlockCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def forward_focus(
        self,
        input_ids=None,
        past_key_values=None,
        use_cache: bool = True,
        cache_position=None,
        update_past_key_values: bool = False,
        block_sparse_cache=None,
        is_dense_step: bool = True,
        mask_idx=None,                 # (B, block) bool — masked positions
        avg_decoded: float = 1.0,
        focus_alpha: float = 1.0,
        retain_override=None,
        focus_layers=(0, 1),
        attn_backend=None,
        **kwargs,
    ):
        """FOCUS token-skipping forward. Dense seed step (is_dense_step=True) is
        identical to forward_sparse's dense branch for every layer. On FOCUS steps
        (is_dense_step=False): focus_layers run dense + measure importance; after
        the last focus layer we select decodable tokens; remaining layers run
        sparse on those tokens only."""
        hidden_states = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + hidden_states.shape[1], device=hidden_states.device,
            )
        position_ids = cache_position.unsqueeze(0)
        attention_mask = None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        B = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // num_heads
        n_rep = num_heads // num_kv_heads

        cos_full, sin_full = position_embeddings
        cos_sin_full = torch.cat([cos_full, sin_full], dim=-1)

        imp = {}                         # layer_idx -> (B, seq_len) importance
        token_indices = None
        num_tokens = None
        last_focus_layer = max(focus_layers)

        def _dense_layer(layer_idx, hs):
            decoder_layer = self.layers[layer_idx]
            block_sparse_cache.cache_layer_input(layer_idx, hs)
            residual = hs
            hs_norm = decoder_layer.input_layernorm(hs)
            attn_out = decoder_layer.self_attn(
                hs_norm, position_embeddings=position_embeddings,
                attention_mask=attention_mask, past_key_value=past_key_values,
                cache_position=cache_position, update_past_key_values=update_past_key_values,
            )
            block_sparse_cache.cache_attn_output(layer_idx, attn_out)
            hs = residual + attn_out
            block_sparse_cache.cache_mlp_input(layer_idx, hs)
            residual = hs
            hs_norm = decoder_layer.post_attention_layernorm(hs)
            mlp_out = decoder_layer.mlp(hs_norm)
            hs = residual + mlp_out
            block_sparse_cache.cache_mlp_output(layer_idx, mlp_out)
            return hs

        def _measure_importance(layer_idx, hs):
            attn = self.layers[layer_idx].self_attn
            hs_norm = self.layers[layer_idx].input_layernorm(hs)
            q = attn.q_proj(hs_norm).view(B, seq_len, num_heads, head_dim).transpose(1, 2)
            k = attn.k_proj(hs_norm).view(B, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos_full, sin_full)
            k = repeat_kv(k, n_rep)
            imp[layer_idx] = _focus_importance(q, k, mask_idx, attn.scaling)

        for layer_idx in range(num_layers):
            decoder_layer = self.layers[layer_idx]

            if is_dense_step:
                hidden_states = _dense_layer(layer_idx, hidden_states)
                continue

            if layer_idx in focus_layers:
                _measure_importance(layer_idx, hidden_states)
                hidden_states = _dense_layer(layer_idx, hidden_states)
                if layer_idx == last_focus_layer:
                    delta = imp[last_focus_layer] - imp[min(focus_layers)]
                    token_indices, num_tokens = _focus_select(
                        delta, mask_idx, avg_decoded, focus_alpha, retain_override,
                    )
                continue

            # ---- layers beyond focus_layers: sparse recompute on FOCUS-selected tokens ----
            block_sparse_cache.cache_layer_input(layer_idx, hidden_states)
            idx_h = token_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            selected_hidden = hidden_states.gather(1, idx_h)
            selected_norm = decoder_layer.input_layernorm(selected_hidden)

            attn = decoder_layer.self_attn
            q = attn.q_proj(selected_norm).view(B, num_tokens, num_heads, head_dim)
            k = attn.k_proj(selected_norm).view(B, num_tokens, num_kv_heads, head_dim)
            v = attn.v_proj(selected_norm).view(B, num_tokens, num_kv_heads, head_dim)
            idx_pos = token_indices.unsqueeze(-1).expand(-1, -1, cos_sin_full.shape[-1])
            sel_cos, sel_sin = cos_sin_full.expand(B, -1, -1).gather(1, idx_pos).chunk(2, dim=-1)
            q, k = apply_rotary_pos_emb(q, k, sel_cos, sel_sin, unsqueeze_dim=2)

            past_len = past_key_values.get_seq_length()
            write_positions = token_indices + past_len
            past_key_values.write_sparse(k, v, layer_idx, write_positions)

            full_k, full_v = past_key_values.get_full_kv(layer_idx)
            sparse_attn_out = attn_backend.flash_kvcache_attention(
                q, full_k, full_v, cache_seqlens=past_key_values.scratch_seqlens,
                is_causal=False, scaling=attn.scaling,
            )
            sparse_attn_out = sparse_attn_out.reshape(B, num_tokens, -1).contiguous()
            sparse_attn_out = attn.o_proj(sparse_attn_out)

            block_sparse_cache.scatter_attn_output(layer_idx, token_indices, sparse_attn_out)
            attn_output_full = block_sparse_cache.get_attn_output(layer_idx)
            mid = hidden_states + attn_output_full

            selected_mid = selected_hidden + sparse_attn_out
            selected_mid_norm = decoder_layer.post_attention_layernorm(selected_mid)
            sparse_mlp_out = decoder_layer.mlp(selected_mid_norm)
            block_sparse_cache.scatter_mlp_output(layer_idx, token_indices, sparse_mlp_out)
            mlp_output_full = block_sparse_cache.get_mlp_output(layer_idx)
            hidden_states = mid + mlp_output_full

        import os as _os
        if _os.environ.get("FAST_DLLM_FOCUS_FLOPS", "0") == "1":
            _nfl = len(focus_layers)
            _ndeep = num_layers - _nfl
            if is_dense_step:
                _focus_tl = seq_len * num_layers          # seed: full forward
                _deep_focus = seq_len * _ndeep
            else:
                _focus_tl = seq_len * _nfl + int(num_tokens) * _ndeep
                _deep_focus = int(num_tokens) * _ndeep
            g = globals().setdefault("_FOCUS_FLOP",
                {"focus_tl": 0, "base_tl": 0, "deep_focus": 0, "deep_base": 0, "calls": 0})
            g["focus_tl"] += _focus_tl
            g["base_tl"] += seq_len * num_layers          # dense baseline for this step
            g["deep_focus"] += _deep_focus
            g["deep_base"] += seq_len * _ndeep
            g["calls"] += 1
            if g["calls"] % 500 == 0:
                _tot = 1.0 - g["focus_tl"] / max(1, g["base_tl"])
                _dp = 1.0 - g["deep_focus"] / max(1, g["deep_base"])
                print(f"[focus_flops] calls={g['calls']} total_tokenlayer_saving={_tot:.4f} "
                      f"deeplayer_saving={_dp:.4f} focus_tl={g['focus_tl']} base_tl={g['base_tl']}",
                      flush=True)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPastAndBlockCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def forward_focus_compact(
        self,
        input_ids=None,
        past_key_values=None,
        use_cache: bool = True,
        cache_position=None,
        update_past_key_values: bool = False,
        block_sparse_cache=None,   # accepted for signature compat; unused in deep layers
        is_dense_step: bool = True,
        mask_idx=None,
        avg_decoded: float = 1.0,
        focus_alpha: float = 1.0,
        retain_override=None,
        focus_layers=(0, 1),
        attn_backend=None,
        frozen=None,
        **kwargs,
    ):
        """Compact FOCUS forward: select once after the focus layers, gather to a
        [B, Ksel, D] tensor, run deep layers densely on it, scatter once at the end.
        No per-layer gather/scatter and no block_sparse_cache. Decoded-token results
        match forward_focus; non-selected tokens keep their last-focus-layer output."""
        hidden_states = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + hidden_states.shape[1], device=hidden_states.device,
            )
        position_ids = cache_position.unsqueeze(0)
        attention_mask = None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        B = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // num_heads
        n_rep = num_heads // num_kv_heads
        cos_full, sin_full = position_embeddings
        cos_sin_full = torch.cat([cos_full, sin_full], dim=-1)

        def _dense_layer_nc(layer_idx, hs):
            dl = self.layers[layer_idx]
            residual = hs
            hsn = dl.input_layernorm(hs)
            attn_out = dl.self_attn(
                hsn, position_embeddings=position_embeddings,
                attention_mask=attention_mask, past_key_value=past_key_values,
                cache_position=cache_position, update_past_key_values=update_past_key_values,
            )
            hs = residual + attn_out
            residual = hs
            hsn = dl.post_attention_layernorm(hs)
            hs = residual + dl.mlp(hsn)
            return hs

        def _measure_importance(layer_idx, hs):
            attn = self.layers[layer_idx].self_attn
            hsn = self.layers[layer_idx].input_layernorm(hs)
            q = attn.q_proj(hsn).view(B, seq_len, num_heads, head_dim).transpose(1, 2)
            k = attn.k_proj(hsn).view(B, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos_full, sin_full)
            k = repeat_kv(k, n_rep)
            return _focus_importance(q, k, mask_idx, attn.scaling)

        # Dense seed step: full forward over all tokens (warms KV at every layer).
        if is_dense_step:
            for layer_idx in range(num_layers):
                hidden_states = _dense_layer_nc(layer_idx, hidden_states)
            # Cache the seed step's full-depth (pre-norm) hidden states for the whole
            # block. Evicted (non-decodable) tokens reuse these in sparse steps so their
            # unmask logits are full-depth, not shallow layer-1 outputs (FOCUS delayed cache).
            self._focus_compact_seed_hidden = hidden_states.clone()
            import os as _os
            if _os.environ.get("FAST_DLLM_FOCUS_FLOPS", "0") == "1":
                _nfl = len(focus_layers)
                _ndeep = num_layers - _nfl
                g = globals().setdefault("_FOCUS_FLOP",
                    {"focus_tl": 0, "base_tl": 0, "deep_focus": 0, "deep_base": 0, "calls": 0})
                g["focus_tl"] += seq_len * num_layers          # seed: full forward
                g["base_tl"] += seq_len * num_layers           # dense baseline for this step
                g["deep_focus"] += seq_len * _ndeep
                g["deep_base"] += seq_len * _ndeep
                g["calls"] += 1
                if g["calls"] % 500 == 0:
                    _tot = 1.0 - g["focus_tl"] / max(1, g["base_tl"])
                    _dp = 1.0 - g["deep_focus"] / max(1, g["deep_base"])
                    print(f"[focus_flops] calls={g['calls']} total_tokenlayer_saving={_tot:.4f} "
                          f"deeplayer_saving={_dp:.4f} focus_tl={g['focus_tl']} base_tl={g['base_tl']}",
                          flush=True)
            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPastAndBlockCache(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            )

        # FOCUS step: focus layers run dense + measure importance.
        imp = {}
        last_focus_layer = max(focus_layers)
        for layer_idx in focus_layers:
            imp[layer_idx] = _measure_importance(layer_idx, hidden_states)
            hidden_states = _dense_layer_nc(layer_idx, hidden_states)
        delta = imp[last_focus_layer] - imp[min(focus_layers)]
        token_indices, num_tokens = _focus_select(
            delta, mask_idx, avg_decoded, focus_alpha, retain_override, frozen=frozen,
        )

        # Gather ONCE into the compacted residual stream.
        idx_h = token_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        hs_sel = hidden_states.gather(1, idx_h)                       # [B, Ksel, D]
        idx_pos = token_indices.unsqueeze(-1).expand(-1, -1, cos_sin_full.shape[-1])
        sel_cos, sel_sin = cos_sin_full.expand(B, -1, -1).gather(1, idx_pos).chunk(2, dim=-1)
        past_len = past_key_values.get_seq_length()
        write_positions = token_indices + past_len

        # Deep layers run densely on the compacted tensor; only selected KV is written.
        for layer_idx in range(last_focus_layer + 1, num_layers):
            dl = self.layers[layer_idx]
            attn = dl.self_attn
            residual = hs_sel
            sel_norm = dl.input_layernorm(hs_sel)
            q = attn.q_proj(sel_norm).view(B, num_tokens, num_heads, head_dim)
            k = attn.k_proj(sel_norm).view(B, num_tokens, num_kv_heads, head_dim)
            v = attn.v_proj(sel_norm).view(B, num_tokens, num_kv_heads, head_dim)
            q, k = apply_rotary_pos_emb(q, k, sel_cos, sel_sin, unsqueeze_dim=2)
            past_key_values.write_sparse(k, v, layer_idx, write_positions)
            full_k, full_v = past_key_values.get_full_kv(layer_idx)
            a = attn_backend.flash_kvcache_attention(
                q, full_k, full_v, cache_seqlens=past_key_values.scratch_seqlens,
                is_causal=False, scaling=attn.scaling,
            )
            a = a.reshape(B, num_tokens, -1).contiguous()
            a = attn.o_proj(a)
            hs_sel = residual + a
            residual = hs_sel
            hs_sel = residual + dl.mlp(dl.post_attention_layernorm(hs_sel))

        # Evicted (non-selected) tokens reuse the seed step's full-depth hidden state
        # (FOCUS delayed cache); selected tokens use their freshly recomputed output.
        seed_hidden = getattr(self, "_focus_compact_seed_hidden", None)
        base = seed_hidden if (seed_hidden is not None and seed_hidden.shape == hidden_states.shape) else hidden_states
        hidden_states = base.scatter(1, idx_h, hs_sel)
        import os as _os
        if _os.environ.get("FAST_DLLM_FOCUS_FLOPS", "0") == "1":
            _nfl = len(focus_layers)
            _ndeep = num_layers - _nfl
            _focus_tl = seq_len * _nfl + int(num_tokens) * _ndeep
            _deep_focus = int(num_tokens) * _ndeep
            g = globals().setdefault("_FOCUS_FLOP",
                {"focus_tl": 0, "base_tl": 0, "deep_focus": 0, "deep_base": 0, "calls": 0})
            g["focus_tl"] += _focus_tl
            g["base_tl"] += seq_len * num_layers          # dense baseline for this step
            g["deep_focus"] += _deep_focus
            g["deep_base"] += seq_len * _ndeep
            g["calls"] += 1
            if g["calls"] % 500 == 0:
                _tot = 1.0 - g["focus_tl"] / max(1, g["base_tl"])
                _dp = 1.0 - g["deep_focus"] / max(1, g["deep_base"])
                print(f"[focus_flops] calls={g['calls']} total_tokenlayer_saving={_tot:.4f} "
                      f"deeplayer_saving={_dp:.4f} focus_tl={g['focus_tl']} base_tl={g['base_tl']}",
                      flush=True)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPastAndBlockCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def forward_focus_compact_dynamic(
        self,
        input_ids=None, past_key_values=None, use_cache: bool = True,
        cache_position=None, update_past_key_values: bool = False,
        block_kv=None, is_dense_step: bool = True, mask_idx=None,
        avg_decoded: float = 1.0, focus_alpha: float = 1.0, retain_override=None,
        focus_layers=(0, 1), frozen=None, **kwargs,
    ):
        """Compact FOCUS over a DynamicCache prefix + per-block KV buffer, eager SDPA.
        Mirrors forward_focus_compact but for DynamicCache (BHSD) + DynamicBlockKV +
        SDPA instead of StaticKVCache + flash_kvcache."""
        import torch.nn.functional as F
        hidden_states = self.embed_tokens(input_ids)
        B = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // num_heads
        n_rep = num_heads // num_kv_heads
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + seq_len, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        cos_full, sin_full = self.rotary_emb(hidden_states, position_ids)
        cos_sin_full = torch.cat([cos_full, sin_full], dim=-1)
        scaling = self.layers[0].self_attn.scaling
        last_focus_layer = max(focus_layers)

        def _prefix_kv(L):
            if past_key_values is not None and len(past_key_values) > L:
                return past_key_values[L][0], past_key_values[L][1]
            return None, None

        def _attn(q, k_bhsd, v_bhsd):
            # Align the dynamic path with the static FOCUS path's flash kernel
            # (flash accumulates the softmax in fp32) to minimize the bf16 divergence
            # that accumulates over 28 layers when SDPA reimplements attention.
            # flash_attn_func wants [B,S,H,D] and does GQA natively (no repeat_kv).
            if _flash_attn_func is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
                qf = q.transpose(1, 2); kf = k_bhsd.transpose(1, 2); vf = v_bhsd.transpose(1, 2)
                a = _flash_attn_func(qf, kf, vf, causal=False, softmax_scale=scaling)
                return a.transpose(1, 2)
            # CPU / no-flash fallback (unit tests): plain SDPA with manual GQA expand.
            k = repeat_kv(k_bhsd, n_rep); v = repeat_kv(v_bhsd, n_rep)
            return F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scaling)

        def _dense_layer(L, hs, write_buffer):
            dl = self.layers[L]
            residual = hs
            hsn = dl.input_layernorm(hs)
            q = dl.self_attn.q_proj(hsn).view(B, seq_len, num_heads, head_dim)
            k = dl.self_attn.k_proj(hsn).view(B, seq_len, num_kv_heads, head_dim)
            v = dl.self_attn.v_proj(hsn).view(B, seq_len, num_kv_heads, head_dim)
            q, k = apply_rotary_pos_emb(q, k, cos_full, sin_full, unsqueeze_dim=2)
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
            if write_buffer and block_kv is not None:
                block_kv.write_full(L, k, v)
            pk, pv = _prefix_kv(L)
            full_k = torch.cat([pk, k], dim=2) if pk is not None else k
            full_v = torch.cat([pv, v], dim=2) if pv is not None else v
            a = _attn(q, full_k, full_v)
            a = a.transpose(1, 2).reshape(B, seq_len, -1)
            a = dl.self_attn.o_proj(a)
            hs = residual + a
            hs = hs + dl.mlp(dl.post_attention_layernorm(hs))
            return hs

        def _measure_importance(L, hs):
            attn = self.layers[L].self_attn
            hsn = self.layers[L].input_layernorm(hs)
            q = attn.q_proj(hsn).view(B, seq_len, num_heads, head_dim).transpose(1, 2)
            k = attn.k_proj(hsn).view(B, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos_full, sin_full)
            k = repeat_kv(k, n_rep)
            return _focus_importance(q, k, mask_idx, attn.scaling)

        if is_dense_step:
            for L in range(num_layers):
                hidden_states = _dense_layer(L, hidden_states, write_buffer=(L > last_focus_layer))
            self._focus_dyn_seed_hidden = hidden_states.clone()
            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPastAndBlockCache(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            )

        imp = {}
        for L in focus_layers:
            imp[L] = _measure_importance(L, hidden_states)
            hidden_states = _dense_layer(L, hidden_states, write_buffer=False)
        delta = imp[last_focus_layer] - imp[min(focus_layers)]
        token_indices, num_tokens = _focus_select(
            delta, mask_idx, avg_decoded, focus_alpha, retain_override, frozen=frozen,
        )

        idx_h = token_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        hs_sel = hidden_states.gather(1, idx_h)
        idx_pos = token_indices.unsqueeze(-1).expand(-1, -1, cos_sin_full.shape[-1])
        sel_cos, sel_sin = cos_sin_full.expand(B, -1, -1).gather(1, idx_pos).chunk(2, dim=-1)

        for L in range(last_focus_layer + 1, num_layers):
            dl = self.layers[L]
            residual = hs_sel
            sel_norm = dl.input_layernorm(hs_sel)
            q = dl.self_attn.q_proj(sel_norm).view(B, num_tokens, num_heads, head_dim)
            k = dl.self_attn.k_proj(sel_norm).view(B, num_tokens, num_kv_heads, head_dim)
            v = dl.self_attn.v_proj(sel_norm).view(B, num_tokens, num_kv_heads, head_dim)
            q, k = apply_rotary_pos_emb(q, k, sel_cos, sel_sin, unsqueeze_dim=2)
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
            block_kv.write(L, k, v, token_indices)
            bk, bv = block_kv.get(L)
            pk, pv = _prefix_kv(L)
            full_k = torch.cat([pk, bk], dim=2) if pk is not None else bk
            full_v = torch.cat([pv, bv], dim=2) if pv is not None else bv
            a = _attn(q, full_k, full_v)
            a = a.transpose(1, 2).reshape(B, num_tokens, -1)
            a = dl.self_attn.o_proj(a)
            hs_sel = residual + a
            hs_sel = hs_sel + dl.mlp(dl.post_attention_layernorm(hs_sel))

        seed_hidden = getattr(self, "_focus_dyn_seed_hidden", None)
        base = seed_hidden if (seed_hidden is not None and seed_hidden.shape == hidden_states.shape) else hidden_states
        hidden_states = base.scatter(1, idx_h, hs_sel)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPastAndBlockCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Fast_dLLM_QwenForCausalLM(Fast_dLLM_QwenPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Fast_dLLM_QwenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        update_past_key_values: Optional[bool] = False,
        block_size: Optional[int] = 32,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        mask_id: Optional[int] = 151665,
        **kwargs
    ) -> CausalLMOutputWithPastAndBlockCache:

        if self.training:
            original_labels = labels.clone()
            original_input_ids = input_ids.clone()

            noisy_input_ids = input_ids.clone()

            input_ids = input_ids.reshape(input_ids.shape[0] * input_ids.shape[1] // self.model.bd_size, self.model.bd_size)
            b, l = input_ids.shape
            t = torch.rand((b,), device=input_ids.device)
            eps=1e-3
            p_mask = (1 - eps) * t + eps
            p_mask = p_mask[:, None].repeat(1, l)

            mask_indices = torch.rand((b, l), device=input_ids.device) < p_mask
            x_t = torch.where(mask_indices, mask_id, input_ids).reshape(labels.shape)
            noisy_input_ids[labels != -100] = x_t[labels != -100]
            mask = (noisy_input_ids != mask_id)
            labels[mask] = -100
            input_ids = torch.cat([noisy_input_ids, input_ids.reshape(labels.shape)], dim=1)

            complementary_noisy_input_ids = original_input_ids.clone()
            complementary_labels = original_labels.clone()

            complementary_input_ids = original_input_ids.reshape(original_input_ids.shape[0] * original_input_ids.shape[1] // self.model.bd_size, self.model.bd_size)

            complementary_mask_indices = ~mask_indices
            complementary_x_t = torch.where(complementary_mask_indices, mask_id, complementary_input_ids).reshape(labels.shape)
            complementary_noisy_input_ids[complementary_labels != -100] = complementary_x_t[complementary_labels != -100]
            complementary_mask = (complementary_noisy_input_ids != mask_id)
            complementary_labels[complementary_mask] = -100
            complementary_input_ids = torch.cat([complementary_noisy_input_ids, complementary_input_ids.reshape(complementary_labels.shape)], dim=1)

            input_ids = torch.cat([input_ids, complementary_input_ids], dim=0)
            labels = torch.cat([labels, complementary_labels], dim=0)

        outputs: BaseModelOutputWithPastAndBlockCache = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            update_past_key_values=update_past_key_values,
            block_size=block_size,
            use_block_cache=use_block_cache,
            block_past_key_values=block_past_key_values,
            replace_position=replace_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if self.training:
            hidden_states = hidden_states[:, :hidden_states.shape[1]//2, :]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPastAndBlockCache(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            block_past_key_values=outputs.block_past_key_values,
        )

    def forward_dynamo(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        update_past_key_values: Optional[bool] = False,
        block_size: Optional[int] = 32,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs
    ) -> CausalLMOutputWithPastAndBlockCache:
        """
        Dynamo-compatible inference-only forward for batch_sample_dynamo.

        Delegates to model.forward_dynamo (skips mask, no DynamicCache allocation,
        no training path). Attention layers must be patched via patch_attention_layers()
        before calling this — patched layers handle StaticKVCache in-place writes.
        """
        outputs: BaseModelOutputWithPastAndBlockCache = self.model.forward_dynamo(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            update_past_key_values=update_past_key_values,
            block_size=block_size,
            use_block_cache=use_block_cache,
            block_past_key_values=block_past_key_values,
            replace_position=replace_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPastAndBlockCache(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            block_past_key_values=outputs.block_past_key_values,
        )

    def forward_sparse(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        update_past_key_values: Optional[bool] = False,
        block_sparse_cache=None,
        is_dense_step: bool = True,
        transfer_ratio: float = 0.3,
        attn_backend=None,
        **kwargs,
    ) -> CausalLMOutputWithPastAndBlockCache:
        """
        Sparse-token inference forward for batch_sample_sparse.
        Delegates to model.forward_sparse for dense/sparse layer processing,
        then applies lm_head.
        """
        outputs: BaseModelOutputWithPastAndBlockCache = self.model.forward_sparse(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            update_past_key_values=update_past_key_values,
            block_sparse_cache=block_sparse_cache,
            is_dense_step=is_dense_step,
            transfer_ratio=transfer_ratio,
            attn_backend=attn_backend,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPastAndBlockCache(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
        )

    def forward_focus(
        self,
        input_ids=None,
        past_key_values=None,
        use_cache: bool = True,
        cache_position=None,
        update_past_key_values: bool = False,
        block_sparse_cache=None,
        is_dense_step: bool = True,
        mask_idx=None,
        avg_decoded: float = 1.0,
        focus_alpha: float = 1.0,
        retain_override=None,
        focus_layers=(0, 1),
        attn_backend=None,
        **kwargs,
    ):
        """FOCUS forward + lm_head. Mirrors the forward_sparse wrapper."""
        outputs = self.model.forward_focus(
            input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache,
            cache_position=cache_position, update_past_key_values=update_past_key_values,
            block_sparse_cache=block_sparse_cache, is_dense_step=is_dense_step,
            mask_idx=mask_idx, avg_decoded=avg_decoded, focus_alpha=focus_alpha,
            retain_override=retain_override, focus_layers=focus_layers,
            attn_backend=attn_backend, **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPastAndBlockCache(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
        )

    def forward_focus_compact(
        self,
        input_ids=None, past_key_values=None, use_cache: bool = True,
        cache_position=None, update_past_key_values: bool = False,
        block_sparse_cache=None, is_dense_step: bool = True, mask_idx=None,
        avg_decoded: float = 1.0, focus_alpha: float = 1.0, retain_override=None,
        focus_layers=(0, 1), attn_backend=None, frozen=None, **kwargs,
    ):
        outputs = self.model.forward_focus_compact(
            input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache,
            cache_position=cache_position, update_past_key_values=update_past_key_values,
            block_sparse_cache=block_sparse_cache, is_dense_step=is_dense_step,
            mask_idx=mask_idx, avg_decoded=avg_decoded, focus_alpha=focus_alpha,
            retain_override=retain_override, focus_layers=focus_layers,
            attn_backend=attn_backend, frozen=frozen, **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPastAndBlockCache(
            loss=None, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
        )

    def forward_focus_compact_dynamic(
        self,
        input_ids=None, past_key_values=None, use_cache: bool = True,
        cache_position=None, update_past_key_values: bool = False,
        block_kv=None, is_dense_step: bool = True, mask_idx=None,
        avg_decoded: float = 1.0, focus_alpha: float = 1.0, retain_override=None,
        focus_layers=(0, 1), frozen=None, **kwargs,
    ):
        outputs = self.model.forward_focus_compact_dynamic(
            input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache,
            cache_position=cache_position, update_past_key_values=update_past_key_values,
            block_kv=block_kv, is_dense_step=is_dense_step, mask_idx=mask_idx,
            avg_decoded=avg_decoded, focus_alpha=focus_alpha, retain_override=retain_override,
            focus_layers=focus_layers, frozen=frozen, **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPastAndBlockCache(
            loss=None, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
        )

    @staticmethod
    def _focus_update_frozen(frozen, mask_idx):
        """Delayed-cache freeze update (paper-faithful, right-neighbor rule).

        A block position becomes frozen — its KV cached, excluded from recompute —
        once it is decoded AND its immediate right neighbor is decoded. `mask_idx`
        is the PRE-forward block mask (True = still masked) for the current step, so
        `~mask_idx` = decoded entering this step; because every non-frozen decoded
        token is in the recompute set, a frozen token is guaranteed to have been
        reprocessed once with its real id (the refresh-timing invariant). The
        rightmost position has no right neighbor, so it never freezes. Monotonic:
        OR-accumulates, never clears within a block.

        frozen, mask_idx: (B, block_size) bool. Returns updated (B, block_size) bool.
        """
        dec = ~mask_idx
        right = torch.zeros_like(dec)
        right[:, :-1] = dec[:, 1:]
        return frozen | (dec & right)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=None,
        max_length=None,
        tokenizer=None,
        mask_id=151665,
        threshold=1,
        small_block_size=8,
        block_size=32,
        stop_token=151645,
        stopping_criteria=None,
        top_p=0.95,
        temperature=0,
        use_block_cache=False,
        return_dict_in_generate=False,
        output_scores=False,
        output_hidden_states=False,
        **kwargs
    ):
        if max_new_tokens is None and max_length is None:
            raise ValueError("Either max_new_tokens or max_length must be specified")
        if max_new_tokens is None:
            max_new_tokens = max_length - input_ids.shape[1]
        
        scores_list = [] if output_scores else None
        decoder_hidden_states = [] if output_hidden_states else None
        
        num_blocks = max_new_tokens // block_size
        original_input_length = input_ids.shape[1]

        if input_ids.shape[1] > block_size:
            output = self.forward(
                input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)], 
                use_cache=True, 
                update_past_key_values=True, 
                block_size=block_size
            )
            logits, past_key_values = output.logits, output.past_key_values
            
            if output_scores:
                scores_list.append(logits)
            if output_hidden_states and hasattr(output, 'hidden_states'):
                decoder_hidden_states.append(output.hidden_states)
            
            if input_ids.shape[1] % block_size == 0:
                next_token = logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        else:
            past_key_values = None

        num_small_blocks = block_size // small_block_size

        for block_idx in range(num_blocks):
            if stop_token in input_ids[:, original_input_length:]:
                break
            prompt_length = input_ids.shape[1]
            # Initialize x_init with mask_id
            x_init = mask_id * torch.ones(
                (input_ids.shape[0], block_size-prompt_length%block_size), 
                device=self.device, 
                dtype=torch.long
            )
            x_init = torch.cat([input_ids, x_init], dim=1)

            x_t = x_init.clone()
            block_past_key_values = None
            
            while True:
                if stop_token in x_t[:, prompt_length:]:
                    stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                    if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                        break
                mask_idx = (x_t[:, -block_size:] == mask_id)
                
                # Decode a complete block, update cache, and generate the next token
                if mask_idx.sum() == 0:
                    output = self.forward(
                        input_ids=x_t[:, -block_size:], 
                        use_cache=True, 
                        past_key_values=past_key_values, 
                        update_past_key_values=True, 
                        block_size=block_size
                    )
                    logits, past_key_values = output.logits, output.past_key_values
                    
                    # 收集输出信息
                    if output_scores:
                        scores_list.append(logits)
                    if output_hidden_states and hasattr(output, 'hidden_states'):
                        decoder_hidden_states.append(output.hidden_states)
                    
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    break
                    
                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                    
                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break
                        if stop_token in x_t[:, prompt_length:]:
                            stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                            if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                                break

                        if use_block_cache:
                            if block_past_key_values is None or (x_t[:, -block_size+small_block_start_idx] == mask_id).any():
                                output = self.forward(
                                    input_ids=x_t[:, -block_size:], 
                                    use_cache=True, 
                                    past_key_values=past_key_values, 
                                    update_past_key_values=False, 
                                    use_block_cache=True,
                                )
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            else:
                                output = self.forward(
                                    input_ids=x_t[:,start:end], 
                                    use_cache=True, 
                                    past_key_values=past_key_values, 
                                    update_past_key_values=False, 
                                    use_block_cache=True, 
                                    block_past_key_values=block_past_key_values, 
                                    replace_position=small_block_start_idx
                                )
                                logits = output.logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            output = self.forward(
                                input_ids=x_t[:, -block_size:], 
                                use_cache=True, 
                                past_key_values=past_key_values, 
                                update_past_key_values=False
                            )
                            logits = output.logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]

                        if output_scores:
                            scores_list.append(logits)
                        if output_hidden_states and hasattr(output, 'hidden_states'):
                            decoder_hidden_states.append(output.hidden_states)

                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        # Select tokens with probability greater than threshold from p_1t
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

            input_ids = x_t
            
        # Truncate stop_token
        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
            input_ids = input_ids[:, :stop_token_idx+original_input_length+1]
        
        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=tuple(scores_list) if output_scores and scores_list else None,
                hidden_states=tuple(decoder_hidden_states) if output_hidden_states and decoder_hidden_states else None,
            )
        else:
            return input_ids


    def sample_with_top_p(self, logits, top_p=0.95, temperature=1.0):
        # Calculate probabilities
        if temperature > 0:
            scaled_logits = logits / temperature
        else:
            p_1t = torch.softmax(logits, dim=-1)
            x_1 = p_1t.argmax(dim=-1)
            return x_1, p_1t
                            
        probs = F.softmax(scaled_logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        
        probs[indices_to_remove] = 0

        # Renormalize so that the probabilities of remaining tokens sum to 1
        # Add a small epsilon value to prevent division by zero
        probs_sum = torch.sum(probs, dim=-1, keepdim=True)
        normalized_probs = probs / probs_sum

        p_1t = normalized_probs
        x_1 = torch.multinomial(p_1t[0], num_samples=1).unsqueeze(0).squeeze(-1)

        return x_1, p_1t

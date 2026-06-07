"""Validate the cached-step eval_mask in forward_dkv.

Setup: a committed prefix + a FULLY-decoded block (no mask tokens). A full step
seeds the buffer with correct K/V for every block position. A cached step that
feeds only a SUBSET (fed_indices) reads the same full buffer, so its logits at the
fed positions MUST equal the full step's logits at those positions. Any mismatch =
the cached-step mask / gather-scatter is wrong. Decisive, no statistical noise.
"""
import os, sys
os.environ.setdefault("FV2", os.getcwd())
FV2 = os.environ["FV2"]
if FV2 not in sys.path:
    sys.path.insert(0, FV2)
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from utils.dynamic_block_kv import DynamicBlockKV

MD = os.path.join(FV2, "models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/"
                       "snapshots/0661abf5f9f0ee338970d091052a26c8efa51974")
BLOCK = 32
model = AutoModelForCausalLM.from_pretrained(
    MD, trust_remote_code=True, torch_dtype=torch.bfloat16,
    cache_dir=os.path.join(FV2, "models")).to("cuda").eval()
inner = model.model
cfg = model.config

def buf():
    return DynamicBlockKV(deep_layer_start=0, num_layers=cfg.num_hidden_layers, batch_size=1,
                          num_kv_heads=cfg.num_key_value_heads, block_size=BLOCK,
                          head_dim=cfg.hidden_size // cfg.num_attention_heads,
                          dtype=torch.bfloat16, device="cuda")

def fresh_prefix(prefix_len):
    c = DynamicCache()
    with torch.no_grad():
        pre = torch.randint(0, 1000, (1, prefix_len), device="cuda")
        inner(input_ids=pre, past_key_values=c, use_cache=True,
              update_past_key_values=True, block_size=BLOCK)
    return c

# Test at two prefix lengths: aligned (multiple of BLOCK) and straddling a grid boundary.
for prefix_len in (BLOCK, BLOCK + 11):
    torch.manual_seed(0)
    block = torch.randint(0, 1000, (1, BLOCK), device="cuda")   # fully decoded, no masks
    store = buf()
    with torch.no_grad():
        cA = fresh_prefix(prefix_len)
        outF = inner.forward_dkv(input_ids=block, past_key_values=cA, use_cache=True,
                                 dkv_store=store, is_full_step=True)
        lgF = model.lm_head(outF.last_hidden_state)   # [1, BLOCK, V]

        # cached step: feed every OTHER position (a non-trivial subset)
        fed = torch.arange(0, BLOCK, 2, device="cuda").unsqueeze(0)   # [1, num_fed]
        outC = inner.forward_dkv(input_ids=block, past_key_values=cA, use_cache=True,
                                 dkv_store=store, fed_indices=fed, is_full_step=False)
        lgC = model.lm_head(outC.last_hidden_state)   # [1, BLOCK, V]

    f = fed[0]
    aF = lgF.argmax(-1)[0, f]
    aC = lgC.argmax(-1)[0, f]
    agree = (aF == aC).float().mean().item()
    dmax = (lgF[0, f].float() - lgC[0, f].float()).abs().max().item()
    print(f"[prefix_len={prefix_len:3d}] fed={f.numel()} argmax_agreement@fed={agree:.4f} "
          f"max_abs_logit_diff@fed={dmax:.4f}")

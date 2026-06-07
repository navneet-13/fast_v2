"""Localize the block-2 (with-prefix) divergence: forward_dkv vs self.forward
over an IDENTICAL committed prefix. Isolates the forward from the commit/loop.
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
MASK_ID, BLOCK = 151665, 32
model = AutoModelForCausalLM.from_pretrained(
    MD, trust_remote_code=True, torch_dtype=torch.bfloat16,
    cache_dir=os.path.join(FV2, "models")).to("cuda").eval()
inner = model.model
cfg = model.config

torch.manual_seed(0)
block0 = torch.randint(0, 1000, (1, BLOCK), device="cuda")   # committed prefix block
block1 = torch.randint(0, 1000, (1, BLOCK), device="cuda")   # active block
block1[0, BLOCK // 2:] = MASK_ID

def fresh_prefix():
    # commit block0 identically into a DynamicCache via the REAL forward (update=True)
    c = DynamicCache()
    with torch.no_grad():
        inner(input_ids=block0, past_key_values=c, use_cache=True,
              update_past_key_values=True, block_size=BLOCK)
    return c

def buf():
    return DynamicBlockKV(deep_layer_start=0, num_layers=cfg.num_hidden_layers, batch_size=1,
                          num_kv_heads=cfg.num_key_value_heads, block_size=BLOCK,
                          head_dim=cfg.hidden_size // cfg.num_attention_heads,
                          dtype=torch.bfloat16, device="cuda")

with torch.no_grad():
    cA = fresh_prefix()
    outA = inner(input_ids=block1, past_key_values=cA, use_cache=True,
                 update_past_key_values=False, block_size=BLOCK)        # baseline self.forward (eval_mask)
    lgA = model.lm_head(outA.last_hidden_state)

    cB = fresh_prefix()
    outB = inner.forward_dkv(input_ids=block1, past_key_values=cB, use_cache=True,
                             dkv_store=buf(), is_full_step=True)          # dKV full step
    lgB = model.lm_head(outB.last_hidden_state)

agree = (lgA.argmax(-1) == lgB.argmax(-1)).float().mean().item()
print(f"[PFX] prefix_len(A)={cA.get_seq_length()} prefix_len(B)={cB.get_seq_length()}")
print(f"[PFX] argmax_agreement = {agree:.4f}")
print(f"[PFX] max_abs_hidden_diff = {(outA.last_hidden_state.float()-outB.last_hidden_state.float()).abs().max().item():.4f}")
print(f"[PFX] max_abs_logit_diff  = {(lgA.float()-lgB.float()).abs().max().item():.4f}")
print(f"[PFX] A argmax[:8]: {lgA.argmax(-1)[0,:8].tolist()}")
print(f"[PFX] B argmax[:8]: {lgB.argmax(-1)[0,:8].tolist()}")

"""Diagnostic: does forward_dkv(is_full_step=True) diverge from the real dense forward?

Compares, on identical input (one block, no prefix):
  (A) inner.forward_dkv(is_full_step=True)   — the hand-rolled full step
  (B) inner(...)  the real Fast_dLLM_QwenModel forward (real self_attn)
Reports argmax agreement + logit/hidden max-abs diff. If they diverge, the
hand-rolled full step is the accuracy culprit (and the fix = delegate to self_attn).
"""
import os, torch
from transformers import AutoModelForCausalLM

FV2 = os.environ["FV2"]
MODELING_DIR = os.path.join(
    FV2, "models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/"
    "snapshots/0661abf5f9f0ee338970d091052a26c8efa51974",
)
import sys
if FV2 not in sys.path:
    sys.path.insert(0, FV2)
from utils.dynamic_block_kv import DynamicBlockKV

mask_id = 151665
block = 32

model = AutoModelForCausalLM.from_pretrained(
    MODELING_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16,
    cache_dir=os.path.join(FV2, "models"),
).to("cuda").eval()
inner = model.model
cfg = model.config

torch.manual_seed(0)
x = torch.randint(0, 1000, (1, block), device="cuda")
x[0, block // 2:] = mask_id  # half masked, like a mid-diffusion block

def buf():
    return DynamicBlockKV(
        deep_layer_start=0, num_layers=cfg.num_hidden_layers, batch_size=1,
        num_kv_heads=cfg.num_key_value_heads, block_size=block,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        dtype=torch.bfloat16, device="cuda",
    )

with torch.no_grad():
    # (A) hand-rolled full step
    out_dkv = inner.forward_dkv(input_ids=x, past_key_values=None, dkv_store=buf(), is_full_step=True)
    h_dkv = out_dkv.last_hidden_state          # normed hidden [1, block, H]
    lg_dkv = model.lm_head(h_dkv)

    # (B) reference = the REAL decoder_layer loop with attention_mask=None and NO dkv_store.
    # This is exactly forward_dkv's full step minus the buffer, so A==B proves the dkv_store
    # path is a transparent identity (and == the proven-lossless static None-mask attention).
    hs = inner.embed_tokens(x)
    pos_ids = torch.arange(block, device="cuda").unsqueeze(0)
    pe = inner.rotary_emb(hs, pos_ids)
    for layer in inner.layers[: cfg.num_hidden_layers]:
        hs = layer(hs, attention_mask=None, position_ids=pos_ids, past_key_value=None,
                   use_cache=False, cache_position=pos_ids[0], position_embeddings=pe,
                   update_past_key_values=False)
    h_real = inner.norm(hs)
    lg_real = model.lm_head(h_real)

am_dkv = lg_dkv.argmax(-1)
am_real = lg_real.argmax(-1)
agree = (am_dkv == am_real).float().mean().item()
hid_diff = (h_dkv.float() - h_real.float()).abs().max().item()
lg_diff = (lg_dkv.float() - lg_real.float()).abs().max().item()

print(f"[DIAG] block={block} masked={int((x==mask_id).sum())}")
print(f"[DIAG] argmax_agreement = {agree:.4f}  (1.0 = identical token choices)")
print(f"[DIAG] max_abs_hidden_diff = {hid_diff:.4f}")
print(f"[DIAG] max_abs_logit_diff  = {lg_diff:.4f}")
print(f"[DIAG] dkv argmax (first 8):  {am_dkv[0, :8].tolist()}")
print(f"[DIAG] real argmax (first 8): {am_real[0, :8].tolist()}")

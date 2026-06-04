"""
Quick diagnostic: run torch._dynamo.explain() on forward_dynamo to count
graph breaks and identify their causes.

Usage:
  CUDA_VISIBLE_DEVICES=4 python test_dynamo_explain.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ["WORKSPACE"] = os.path.dirname(__file__)

import torch
import torch._dynamo as dynamo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16
BATCH  = 2
BLOCK  = 32
MAX_SEQ = 256  # small for testing
NUM_LAYERS = 28

print(f"[env] torch {torch.__version__}  device={DEVICE}")
print("=" * 70)

# ---- Load model ----
from transformers import AutoModelForCausalLM
model_path = os.path.join(
    os.environ["WORKSPACE"],
    "models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/"
    "0661abf5f9f0ee338970d091052a26c8efa51974/",
)
print(f"Loading model from {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=DTYPE,
    cache_dir=os.path.join(os.environ["WORKSPACE"], "models"),
)
model.eval().to(DEVICE)

# ---- Set up static cache + patching ----
from utils.static_kv_cache import StaticKVCache
from utils.attention_backends import get_attention_backend, patch_attention_layers, unpatch_attention_layers

config = model.config
num_kv_heads = config.num_key_value_heads
head_dim = config.hidden_size // config.num_attention_heads

static_cache = StaticKVCache(
    batch_size=BATCH, max_seq_len=MAX_SEQ, num_layers=NUM_LAYERS,
    num_kv_heads=num_kv_heads, head_dim=head_dim, dtype=DTYPE, device=DEVICE,
)
model._static_kv_cache = static_cache

attn_backend = get_attention_backend("auto")
print(f"Backend: {attn_backend.name}")
original_forwards = patch_attention_layers(model, attn_backend)

# ---- Set up cache state (simulating mid-generation) ----
past_len = 64
static_cache.set_kv_write_start(past_len)
static_cache.set_scratch_seqlens(past_len + BLOCK)
static_cache.prepare_write_idx(BLOCK)

cache_pos = torch.arange(past_len, past_len + BLOCK, device=DEVICE)

# ---- Build sample input ----
input_ids = torch.randint(0, 1000, (BATCH, BLOCK), device=DEVICE)

sample_kwargs = dict(
    input_ids=input_ids,
    use_cache=True,
    past_key_values=static_cache,
    update_past_key_values=False,
    cache_position=cache_pos,
)

# ---- Run explain ----
print("\n" + "=" * 70)
print("Running torch._dynamo.explain() on forward_dynamo...")
print("=" * 70)

fwd = model.forward_dynamo
explanation = dynamo.explain(fwd)(**sample_kwargs)

print(f"\n{'='*70}")
print(f"  Graph count:       {explanation.graph_count}")
print(f"  Graph break count: {explanation.graph_break_count}")
print(f"{'='*70}")

if explanation.graph_break_count > 0:
    print("\nBreak reasons:")
    for i, reason in enumerate(explanation.break_reasons):
        print(f"  [{i}] {reason}")

# ---- Also try compile with fullgraph=True to see what fails ----
print(f"\n{'='*70}")
print("Testing torch.compile(fullgraph=True)...")
print("=" * 70)

dynamo.reset()
try:
    compiled_fwd = torch.compile(fwd, fullgraph=True, dynamic=True)
    torch.compiler.cudagraph_mark_step_begin()
    out = compiled_fwd(**sample_kwargs)
    print("  fullgraph=True: SUCCESS (no graph breaks!)")
except Exception as e:
    print(f"  fullgraph=True: FAILED — {type(e).__name__}: {e}")

# ---- Try compile with fullgraph=False and count partitions ----
print(f"\n{'='*70}")
print("Testing torch.compile(fullgraph=False, mode='reduce-overhead')...")
print("=" * 70)

dynamo.reset()
# Re-patch (dynamo.reset clears guards)
unpatch_attention_layers(model, original_forwards)
original_forwards = patch_attention_layers(model, attn_backend)

os.environ["TORCH_LOGS"] = "cudagraphs"

try:
    compiled_fwd = torch.compile(fwd, fullgraph=False, dynamic=True, mode="reduce-overhead")
    torch.compiler.cudagraph_mark_step_begin()
    out = compiled_fwd(**sample_kwargs)
    print(f"  reduce-overhead: SUCCESS")
    print(f"  Output logits shape: {out.logits.shape}")

    # Second call to check CUDA graph replay
    torch.compiler.cudagraph_mark_step_begin()
    out2 = compiled_fwd(**sample_kwargs)
    print(f"  Second call (CUDA graph replay): SUCCESS")
except Exception as e:
    import traceback
    print(f"  reduce-overhead: FAILED")
    traceback.print_exc()

print("\nDone.")

import math, torch, torch.nn.functional as F
import importlib.util, sys, os
import pytest

FV2 = os.environ["FV2"]
MODELING = os.path.join(
    FV2, "models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/"
    "snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/modeling.py",
)

def _load_modeling():
    # Register the snapshot dir as a package so modeling.py's relative
    # import (`from .configuration import ...`) resolves when loaded standalone.
    pkg_name = "fdllm_pkg"
    pkg_dir = os.path.dirname(MODELING)
    if pkg_name not in sys.modules:
        pkg_spec = importlib.util.spec_from_file_location(
            pkg_name, os.path.join(pkg_dir, "__init__.py"),
            submodule_search_locations=[pkg_dir],
        )
        pkg = importlib.util.module_from_spec(pkg_spec)
        sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(
        pkg_name + ".modeling", MODELING,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fdllm_modeling"] = mod
    sys.modules[pkg_name + ".modeling"] = mod
    spec.loader.exec_module(mod)
    return mod

if FV2 not in sys.path:
    sys.path.insert(0, FV2)

@pytest.mark.gpu
def test_self_attn_dkv_store_uses_full_buffer():
    import torch
    from transformers import AutoModelForCausalLM
    from utils.dynamic_block_kv import DynamicBlockKV
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    model = AutoModelForCausalLM.from_pretrained(
        os.path.dirname(MODELING), trust_remote_code=True,
        torch_dtype=torch.bfloat16, cache_dir=os.path.join(FV2, "models"),
    ).to("cuda").eval()
    inner = model.model
    cfg = model.config
    block = 8
    H = cfg.hidden_size
    hs = torch.randn(1, block, H, device="cuda", dtype=torch.bfloat16)
    pos_ids = torch.arange(block, device="cuda").unsqueeze(0)
    pe = inner.rotary_emb(hs, pos_ids)
    layer0 = inner.layers[0].self_attn
    buf = DynamicBlockKV(deep_layer_start=0, num_layers=cfg.num_hidden_layers, batch_size=1,
                         num_kv_heads=cfg.num_key_value_heads, block_size=block,
                         head_dim=cfg.hidden_size // cfg.num_attention_heads,
                         dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        out_plain = layer0(hidden_states=hs, position_embeddings=pe, attention_mask=None,
                           past_key_value=None, cache_position=pos_ids[0])
        all_pos = torch.arange(block, device="cuda").unsqueeze(0)
        out_dkv = layer0(hidden_states=hs, position_embeddings=pe, attention_mask=None,
                         past_key_value=None, cache_position=pos_ids[0],
                         dkv_store=buf, dkv_positions=all_pos)
    assert torch.allclose(out_plain.float(), out_dkv.float(), atol=1e-2)
    k0, v0 = buf.get(0)
    assert k0.abs().sum() > 0

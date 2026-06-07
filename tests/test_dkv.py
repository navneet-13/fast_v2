import math, torch, torch.nn.functional as F
import importlib.util, sys, os
import pytest

FV2 = os.environ["FV2"]
if FV2 not in sys.path:
    sys.path.insert(0, FV2)
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

def test_dkv_fed_indices_all_active_when_nothing_cached():
    import torch
    m = _load_modeling()
    fn = m.Fast_dLLM_QwenForCausalLM._dkv_fed_indices
    prv = torch.zeros((2, 8), dtype=torch.bool)
    fed_indices, num_fed = fn(prv)
    assert num_fed == 8
    for b in range(2):
        assert set(fed_indices[b].tolist()) == set(range(8))

def test_dkv_fed_indices_excludes_cached_positions():
    import torch
    m = _load_modeling()
    fn = m.Fast_dLLM_QwenForCausalLM._dkv_fed_indices
    prv = torch.zeros((1, 8), dtype=torch.bool)
    prv[0, [1, 3, 5]] = True
    fed_indices, num_fed = fn(prv)
    assert num_fed == 5
    assert not (set(fed_indices[0].tolist()) & {1, 3, 5})
    assert {0, 2, 4, 6, 7}.issubset(set(fed_indices[0].tolist()))


def test_dynamic_block_kv_all_layers_scatter():
    import torch
    from utils.dynamic_block_kv import DynamicBlockKV
    B, Hkv, block, D, L = 1, 4, 8, 16, 28
    buf = DynamicBlockKV(deep_layer_start=0, num_layers=L, batch_size=B,
                         num_kv_heads=Hkv, block_size=block, head_dim=D,
                         dtype=torch.float32, device="cpu")
    assert set(buf.k.keys()) == set(range(L))
    k0 = torch.arange(B * Hkv * block * D, dtype=torch.float32).reshape(B, Hkv, block, D)
    buf.write_full(0, k0, k0.clone())
    sel = torch.tensor([[2, 5]])
    new = torch.zeros(B, Hkv, 2, D)
    buf.write(0, new, new.clone(), sel)
    got_k, _ = buf.get(0)
    assert torch.equal(got_k[:, :, 2, :], torch.zeros(B, Hkv, D))
    assert torch.equal(got_k[:, :, 3, :], k0[:, :, 3, :])

def test_dkv_fed_indices_batch_max_padding_is_harmless():
    import torch
    m = _load_modeling()
    fn = m.Fast_dLLM_QwenForCausalLM._dkv_fed_indices
    prv = torch.zeros((2, 8), dtype=torch.bool)
    prv[0, [1, 2, 3, 4, 5, 6]] = True
    prv[1, [7]] = True
    fed_indices, num_fed = fn(prv)
    assert num_fed == 7
    assert not (set(fed_indices[0].tolist()) & {1, 2, 3, 4, 5, 6})


@pytest.mark.gpu
def test_forward_dkv_full_equals_fed_all():
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
    mask_id = 151665
    torch.manual_seed(0)
    x = torch.randint(0, 1000, (1, block), device="cuda")
    x[0, block // 2:] = mask_id

    def make_buf():
        return DynamicBlockKV(
            deep_layer_start=0, num_layers=cfg.num_hidden_layers, batch_size=1,
            num_kv_heads=cfg.num_key_value_heads, block_size=block,
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            dtype=torch.bfloat16, device="cuda",
        )

    with torch.no_grad():
        out_full = inner.forward_dkv(input_ids=x, past_key_values=None,
                                     block_kv=make_buf(), is_full_step=True)
        buf2 = make_buf()
        inner.forward_dkv(input_ids=x, past_key_values=None, block_kv=buf2, is_full_step=True)
        fed = torch.arange(block, device="cuda").unsqueeze(0)
        out_fed_all = inner.forward_dkv(input_ids=x, past_key_values=None,
                                        block_kv=buf2, fed_indices=fed, is_full_step=False)
    a = out_full.last_hidden_state
    b = out_fed_all.last_hidden_state
    assert torch.equal(model.lm_head(a).argmax(-1), model.lm_head(b).argmax(-1))

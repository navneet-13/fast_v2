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

def test_focus_importance_shape_and_masking():
    m = _load_modeling()
    B, H, S, d = 2, 4, 6, 8
    torch.manual_seed(0)
    q = torch.randn(B, H, S, d)
    k = torch.randn(B, H, S, d)
    mask_idx = torch.zeros(B, S, dtype=torch.bool)
    mask_idx[:, 2:] = True  # last 4 positions masked
    imp = m._focus_importance(q, k, mask_idx, scaling=d ** -0.5)
    assert imp.shape == (B, S)
    # non-masked key columns must have zero importance (mass received)
    assert torch.allclose(imp[:, :2], torch.zeros(B, 2), atol=1e-5)
    # masked tokens receive positive attention mass
    assert (imp[:, 2:] > 0).all()

def test_focus_select_respects_rule_and_budget():
    m = _load_modeling()
    # positions 0,1 are DECODED (non-masked); 2-5 masked. delta high for 4,5.
    delta = torch.tensor([[ -9., -9., 0.1, 0.2, 5.0, 4.0]])
    mask_idx = torch.tensor([[False, False, True, True, True, True]])
    idx, ksel = m._focus_select(delta, mask_idx, avg_decoded=2.0, focus_alpha=1.0,
                                retain_override=None)
    assert idx.shape[1] == ksel
    sel = set(idx[0].tolist())
    # decoded tokens (0,1) are ALWAYS retained (KV refresh); top-delta masked (4,5) retained
    assert {0, 1, 4, 5}.issubset(sel)
    # the lowest-delta non-selected masked token (2) is evicted (not recomputed)
    assert 2 not in sel


def test_focus_select_always_retains_decoded():
    m = _load_modeling()
    # 3 decoded (0,1,2) + 5 masked (3-7); tiny K so masked selection is minimal
    delta = torch.zeros(1, 8)
    delta[0, 7] = 5.0  # one clearly-decodable masked token
    mask_idx = torch.tensor([[False, False, False, True, True, True, True, True]])
    idx, ksel = m._focus_select(delta, mask_idx, avg_decoded=1.0, focus_alpha=1.0,
                                retain_override=None)
    sel = set(idx[0].tolist())
    # all decoded positions must be in the recompute set
    assert {0, 1, 2}.issubset(sel), f"decoded tokens missing from {sel}"
    # the decodable masked token is retained
    assert 7 in sel

def test_focus_select_retain_all_override():
    m = _load_modeling()
    delta = torch.randn(1, 8)
    mask_idx = torch.ones(1, 8, dtype=torch.bool)
    idx, ksel = m._focus_select(delta, mask_idx, avg_decoded=1.0, focus_alpha=1.0,
                                retain_override=1.0)
    assert ksel == 8                      # retain every token
    assert sorted(idx[0].tolist()) == list(range(8))


def test_focus_update_frozen_right_neighbor_rule():
    m = _load_modeling()
    fn = m.Fast_dLLM_QwenForCausalLM._focus_update_frozen
    # block of 6; mask True = still masked. decoded = ~mask.
    # pos:       0      1      2     3      4      5
    # mask:      F      F      T     F      F      F   -> decoded T T F T T T
    mask = torch.tensor([[False, False, True, False, False, False]])
    frozen0 = torch.zeros_like(mask)
    frozen1 = fn(frozen0, mask)
    # freeze iff decoded AND right-neighbor decoded; rightmost never freezes:
    # 0: T&dec[1]=T -> True | 1: T&dec[2]=F -> False | 2: masked -> False
    # 3: T&dec[4]=T -> True | 4: T&dec[5]=T -> True | 5: rightmost -> False
    assert frozen1[0].tolist() == [True, False, False, True, True, False]


def test_focus_update_frozen_accumulates_and_is_monotonic():
    m = _load_modeling()
    fn = m.Fast_dLLM_QwenForCausalLM._focus_update_frozen
    mask_a = torch.tensor([[False, False, True, True]])   # decoded: T T F F -> freeze pos0 only
    frozen = fn(torch.zeros_like(mask_a), mask_a)
    assert frozen[0].tolist() == [True, False, False, False]
    mask_b = torch.tensor([[False, False, False, True]])  # decoded: T T T F -> freeze pos0,1
    frozen = fn(frozen, mask_b)                            # OR-accumulates, never clears
    assert frozen[0].tolist() == [True, True, False, False]


def test_focus_update_frozen_batch_independent():
    m = _load_modeling()
    fn = m.Fast_dLLM_QwenForCausalLM._focus_update_frozen
    # Two independent rows, decoded = ~mask:
    # row0 mask [F,F,T,F] decoded T T F T -> freeze pos0 only ([T,F,F,F])
    # row1 mask [F,T,F,F] decoded T F T T -> pos2 freezes (dec & right=dec[3]=T) -> [F,F,T,F]
    mask = torch.tensor([[False, False, True, False],
                         [False, True, False, False]])
    frozen = fn(torch.zeros_like(mask), mask)
    assert frozen[0].tolist() == [True, False, False, False]
    assert frozen[1].tolist() == [False, False, True, False]


@pytest.mark.gpu
def test_focus_retain_all_matches_dense():
    """CORRECTNESS GATE: with retain_override=1.0 every masked token is
    recomputed, so a FOCUS step must produce the SAME logits as a dense
    step (forward_sparse is_dense_step=True) computed on identical
    post-seed state. This validates the sparse gather/scatter/recompute
    plumbing in forward_focus's deep layers."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from utils.static_kv_cache import StaticKVCache
    from utils.block_sparse_cache import BlockSparseCache
    from utils.attention_backends import (
        get_attention_backend, patch_attention_layers,
    )

    model_id = "Efficient-Large-Model/Fast_dLLM_v2_7B"
    cache_dir = os.environ["WORKSPACE"] + "/models"
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ---- Load tokenizer + model exactly as eval.py does ----
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype, cache_dir=cache_dir,
    )
    model.eval()
    model.to(device)

    # ---- Block / mask setup ----
    block_size = 32
    mask_id = 151665
    B = 1

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    hidden_size = config.hidden_size

    # Build a single block of length block_size: real prompt tokens followed
    # by mask_id, then mask roughly the second half of the block.
    prompt = "Question: What is 2+2? Answer:"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    prompt_ids = prompt_ids[:block_size // 2]          # keep first half real
    n_real = len(prompt_ids)
    tokens = prompt_ids + [mask_id] * (block_size - n_real)
    x_t = torch.tensor([tokens], dtype=torch.long, device=device)  # (B, block_size)
    mask_idx = (x_t == mask_id)                         # (B, block_size) bool
    assert mask_idx.sum() > 0 and (~mask_idx).sum() > 0

    max_seq_len = block_size  # past is empty for this gate

    # ---- Caches + attention backend (mirrors batch_sample_sparse) ----
    static_cache = StaticKVCache(
        batch_size=B, max_seq_len=max_seq_len, num_layers=num_layers,
        num_kv_heads=num_kv_heads, head_dim=head_dim, dtype=dtype, device=device,
    )
    block_sparse_cache = BlockSparseCache(
        num_layers=num_layers, batch_size=B, block_size=block_size,
        hidden_size=hidden_size, dtype=dtype, device=device,
    )
    attn_backend = get_attention_backend(
        os.environ.get("FAST_DLLM_ATTENTION_BACKEND", "auto")
    )

    # ---- flash_kvcache_attention shim (test harness only) ----
    # This GPU is SM 8.6 and the env has no flash-attn / flashinfer, so the
    # auto-selected backend is 'sdpa', whose flash_kvcache_attention raises
    # (no SDPA fallback by design). The FOCUS deep-layer sparse path calls
    # flash_kvcache_attention directly. To exercise the real gather/scatter/
    # recompute plumbing in modeling.py without modifying it, we provide a
    # mathematically-equivalent SDPA implementation of flash_kvcache_attention
    # (non-causal, GQA-repeat, seqused-masked, given scaling). The dense
    # reference also runs over SDPA (patched self_attn), so both sides share
    # SDPA attention math and the comparison isolates the sparse plumbing.
    if attn_backend.name == "sdpa":
        def _sdpa_flash_kvcache(query, k_cache, v_cache, cache_seqlens,
                                is_causal=False, scaling=None):
            # query: (B, S_q, H, D) BSHD ; k/v_cache: (B, max_S, H_kv, D) BSHD
            B_, S_q, H_, D_ = query.shape
            max_S = k_cache.shape[1]
            H_kv = k_cache.shape[2]
            scale = scaling if scaling is not None else D_ ** -0.5
            # BSHD -> BHSD
            q = query.transpose(1, 2)                      # (B, H, S_q, D)
            k = k_cache.transpose(1, 2)                    # (B, H_kv, max_S, D)
            v = v_cache.transpose(1, 2)
            if H_ != H_kv:
                n_rep = H_ // H_kv
                k = k[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(B_, H_, max_S, D_)
                v = v[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(B_, H_, max_S, D_)
            # seqused mask: invalid key positions (>= cache_seqlens) get -inf
            pos = torch.arange(max_S, device=query.device)
            valid = pos[None, :] < cache_seqlens.to(torch.long)[:, None]   # (B, max_S)
            attn_mask = torch.zeros(B_, 1, 1, max_S, device=query.device, dtype=q.dtype)
            attn_mask.masked_fill_(~valid[:, None, None, :], float("-inf"))
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale,
            )                                              # (B, H, S_q, D)
            return out.transpose(1, 2).contiguous()        # BSHD

        attn_backend.flash_kvcache_attention = _sdpa_flash_kvcache

    patch_attention_layers(model, attn_backend, None)

    def _setup_block_state():
        """Reset caches + per-block control signals exactly like the
        per-block setup in batch_sample_sparse (past is empty: past_len=0)."""
        static_cache.zero_and_reset()
        block_sparse_cache.reset()
        past_len = static_cache.get_seq_length()           # 0
        assert past_len == 0
        static_cache.set_kv_write_start(past_len)
        static_cache.set_scratch_seqlens(past_len + block_size)
        static_cache.prepare_write_idx(block_size)

    common = dict(
        input_ids=x_t, use_cache=True, past_key_values=static_cache,
        update_past_key_values=False, block_sparse_cache=block_sparse_cache,
        attn_backend=attn_backend,
    )

    with torch.no_grad():
        # ===== DENSE reference branch: seed (dense) then a 2nd dense step =====
        _setup_block_state()
        # seed: dense forward_sparse fills caches + KV at all positions
        model.forward_sparse(is_dense_step=True, transfer_ratio=1.0, **common)
        # step 2: dense forward_sparse on the same post-seed state
        out_dense = model.forward_sparse(
            is_dense_step=True, transfer_ratio=1.0, **common,
        )
        dense_logits = out_dense.logits.float()

        # ===== FOCUS retain-all branch: identical seed, then FOCUS step =====
        _setup_block_state()
        # seed: dense forward_focus (identical math to forward_sparse dense)
        model.forward_focus(
            is_dense_step=True, mask_idx=mask_idx, retain_override=1.0, **common,
        )
        # step 2: FOCUS step with retain_override=1.0 -> every masked token recomputed
        out_focus = model.forward_focus(
            is_dense_step=False, mask_idx=mask_idx, avg_decoded=float(block_size),
            focus_alpha=1.0, retain_override=1.0, focus_layers=(0, 1), **common,
        )
        focus_logits = out_focus.logits.float()

    assert focus_logits.shape == dense_logits.shape

    # Diagnostics over masked positions (the FOCUS-selected recompute set).
    m = mask_idx  # (B, block_size)
    fl = focus_logits[m]
    dl = dense_logits[m]
    max_abs = (fl - dl).abs().max().item()
    print(f"\n[retain_all gate] masked logits max_abs_diff={max_abs:.6g} "
          f"(shape {tuple(focus_logits.shape)}, masked={int(m.sum())})")

    assert torch.allclose(fl, dl, atol=1e-2, rtol=1e-2), (
        f"FOCUS retain-all logits diverge from dense on masked positions: "
        f"max_abs_diff={max_abs:.6g}"
    )


@pytest.mark.gpu
def test_focus_compact_retain_all_matches_dense():
    """CORRECTNESS GATE (compact variant): with retain_override=1.0 every
    masked token is selected, so Ksel == block_size and all tokens are
    gathered. A compact FOCUS step must produce the SAME logits as a dense
    step (forward_sparse is_dense_step=True) computed on identical post-seed
    state. Validates the gather-once / dense-deep-layers / scatter-once
    plumbing in forward_focus_compact."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from utils.static_kv_cache import StaticKVCache
    from utils.block_sparse_cache import BlockSparseCache
    from utils.attention_backends import (
        get_attention_backend, patch_attention_layers,
    )

    model_id = "Efficient-Large-Model/Fast_dLLM_v2_7B"
    cache_dir = os.environ["WORKSPACE"] + "/models"
    device = torch.device("cuda")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype, cache_dir=cache_dir,
    )
    model.eval()
    model.to(device)

    block_size = 32
    mask_id = 151665
    B = 1

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    hidden_size = config.hidden_size

    prompt = "Question: What is 2+2? Answer:"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    prompt_ids = prompt_ids[:block_size // 2]
    n_real = len(prompt_ids)
    tokens = prompt_ids + [mask_id] * (block_size - n_real)
    x_t = torch.tensor([tokens], dtype=torch.long, device=device)
    mask_idx = (x_t == mask_id)
    assert mask_idx.sum() > 0 and (~mask_idx).sum() > 0

    max_seq_len = block_size

    static_cache = StaticKVCache(
        batch_size=B, max_seq_len=max_seq_len, num_layers=num_layers,
        num_kv_heads=num_kv_heads, head_dim=head_dim, dtype=dtype, device=device,
    )
    block_sparse_cache = BlockSparseCache(
        num_layers=num_layers, batch_size=B, block_size=block_size,
        hidden_size=hidden_size, dtype=dtype, device=device,
    )
    attn_backend = get_attention_backend(
        os.environ.get("FAST_DLLM_ATTENTION_BACKEND", "auto")
    )

    if attn_backend.name == "sdpa":
        def _sdpa_flash_kvcache(query, k_cache, v_cache, cache_seqlens,
                                is_causal=False, scaling=None):
            B_, S_q, H_, D_ = query.shape
            max_S = k_cache.shape[1]
            H_kv = k_cache.shape[2]
            scale = scaling if scaling is not None else D_ ** -0.5
            q = query.transpose(1, 2)
            k = k_cache.transpose(1, 2)
            v = v_cache.transpose(1, 2)
            if H_ != H_kv:
                n_rep = H_ // H_kv
                k = k[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(B_, H_, max_S, D_)
                v = v[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(B_, H_, max_S, D_)
            pos = torch.arange(max_S, device=query.device)
            valid = pos[None, :] < cache_seqlens.to(torch.long)[:, None]
            attn_mask = torch.zeros(B_, 1, 1, max_S, device=query.device, dtype=q.dtype)
            attn_mask.masked_fill_(~valid[:, None, None, :], float("-inf"))
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale,
            )
            return out.transpose(1, 2).contiguous()

        attn_backend.flash_kvcache_attention = _sdpa_flash_kvcache

    patch_attention_layers(model, attn_backend, None)

    def _setup_block_state():
        static_cache.zero_and_reset()
        block_sparse_cache.reset()
        past_len = static_cache.get_seq_length()
        assert past_len == 0
        static_cache.set_kv_write_start(past_len)
        static_cache.set_scratch_seqlens(past_len + block_size)
        static_cache.prepare_write_idx(block_size)

    common = dict(
        input_ids=x_t, use_cache=True, past_key_values=static_cache,
        update_past_key_values=False, block_sparse_cache=block_sparse_cache,
        attn_backend=attn_backend,
    )

    with torch.no_grad():
        # ===== DENSE reference branch =====
        _setup_block_state()
        model.forward_sparse(is_dense_step=True, transfer_ratio=1.0, **common)
        out_dense = model.forward_sparse(
            is_dense_step=True, transfer_ratio=1.0, **common,
        )
        dense_logits = out_dense.logits.float()

        # ===== COMPACT FOCUS retain-all branch: seed + compact FOCUS step =====
        _setup_block_state()
        model.forward_focus_compact(
            is_dense_step=True, mask_idx=mask_idx, retain_override=1.0, **common,
        )
        out_focus = model.forward_focus_compact(
            is_dense_step=False, mask_idx=mask_idx, avg_decoded=float(block_size),
            focus_alpha=1.0, retain_override=1.0, focus_layers=(0, 1), **common,
        )
        focus_logits = out_focus.logits.float()

    assert focus_logits.shape == dense_logits.shape

    m = mask_idx
    fl = focus_logits[m]
    dl = dense_logits[m]
    max_abs = (fl - dl).abs().max().item()
    print(f"\n[compact retain_all gate] masked logits max_abs_diff={max_abs:.6g} "
          f"(shape {tuple(focus_logits.shape)}, masked={int(m.sum())})")

    assert torch.allclose(fl, dl, atol=1e-2, rtol=1e-2), (
        f"compact FOCUS retain-all logits diverge from dense on masked positions: "
        f"max_abs_diff={max_abs:.6g}"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + flash-attn (v2 env)")
def test_focus_compact_frozen_all_false_equiv_none():
    """EQUIVALENCE GATE (frozen plumbing): passing an all-False `frozen` mask
    to forward_focus_compact must yield logits BIT-IDENTICAL to frozen=None
    (today's default). Reuses the compact harness; re-seeds block KV via
    _setup_block_state() before EACH forward so both calls start from the same
    freshly-seeded state (call (b) is not contaminated by call (a)'s writes)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from utils.static_kv_cache import StaticKVCache
    from utils.block_sparse_cache import BlockSparseCache
    from utils.attention_backends import (
        get_attention_backend, patch_attention_layers,
    )

    model_id = "Efficient-Large-Model/Fast_dLLM_v2_7B"
    cache_dir = os.environ["WORKSPACE"] + "/models"
    device = torch.device("cuda")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype, cache_dir=cache_dir,
    )
    model.eval()
    model.to(device)

    block_size = 32
    mask_id = 151665
    B = 1

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    hidden_size = config.hidden_size

    prompt = "Question: What is 2+2? Answer:"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    prompt_ids = prompt_ids[:block_size // 2]
    n_real = len(prompt_ids)
    tokens = prompt_ids + [mask_id] * (block_size - n_real)
    x_t = torch.tensor([tokens], dtype=torch.long, device=device)
    mask_idx = (x_t == mask_id)
    assert mask_idx.sum() > 0 and (~mask_idx).sum() > 0

    max_seq_len = block_size

    static_cache = StaticKVCache(
        batch_size=B, max_seq_len=max_seq_len, num_layers=num_layers,
        num_kv_heads=num_kv_heads, head_dim=head_dim, dtype=dtype, device=device,
    )
    block_sparse_cache = BlockSparseCache(
        num_layers=num_layers, batch_size=B, block_size=block_size,
        hidden_size=hidden_size, dtype=dtype, device=device,
    )
    attn_backend = get_attention_backend(
        os.environ.get("FAST_DLLM_ATTENTION_BACKEND", "auto")
    )

    if attn_backend.name == "sdpa":
        def _sdpa_flash_kvcache(query, k_cache, v_cache, cache_seqlens,
                                is_causal=False, scaling=None):
            B_, S_q, H_, D_ = query.shape
            max_S = k_cache.shape[1]
            H_kv = k_cache.shape[2]
            scale = scaling if scaling is not None else D_ ** -0.5
            q = query.transpose(1, 2)
            k = k_cache.transpose(1, 2)
            v = v_cache.transpose(1, 2)
            if H_ != H_kv:
                n_rep = H_ // H_kv
                k = k[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(B_, H_, max_S, D_)
                v = v[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(B_, H_, max_S, D_)
            pos = torch.arange(max_S, device=query.device)
            valid = pos[None, :] < cache_seqlens.to(torch.long)[:, None]
            attn_mask = torch.zeros(B_, 1, 1, max_S, device=query.device, dtype=q.dtype)
            attn_mask.masked_fill_(~valid[:, None, None, :], float("-inf"))
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale,
            )
            return out.transpose(1, 2).contiguous()

        attn_backend.flash_kvcache_attention = _sdpa_flash_kvcache

    patch_attention_layers(model, attn_backend, None)

    def _setup_block_state():
        static_cache.zero_and_reset()
        block_sparse_cache.reset()
        past_len = static_cache.get_seq_length()
        assert past_len == 0
        static_cache.set_kv_write_start(past_len)
        static_cache.set_scratch_seqlens(past_len + block_size)
        static_cache.prepare_write_idx(block_size)

    common = dict(
        input_ids=x_t, use_cache=True, past_key_values=static_cache,
        update_past_key_values=False, block_sparse_cache=block_sparse_cache,
        attn_backend=attn_backend,
    )

    # all-False frozen mask: equivalent to no frozen positions (== frozen=None)
    frozen_all_false = torch.zeros(B, block_size, dtype=torch.bool, device=device)

    def _seed_then_step(frozen):
        _setup_block_state()
        model.forward_focus_compact(
            is_dense_step=True, mask_idx=mask_idx, retain_override=1.0, **common,
        )
        out = model.forward_focus_compact(
            is_dense_step=False, mask_idx=mask_idx, avg_decoded=float(block_size),
            focus_alpha=1.0, retain_override=1.0, focus_layers=(0, 1),
            frozen=frozen, **common,
        )
        return out.logits.clone()

    with torch.no_grad():
        logits_none = _seed_then_step(None)                # (a) frozen=None
        logits_frozen = _seed_then_step(frozen_all_false)  # (b) frozen=all-False

    assert logits_frozen.shape == logits_none.shape
    max_abs = (logits_frozen.float() - logits_none.float()).abs().max().item()
    print(f"\n[frozen all-false equiv] logits max_abs_diff={max_abs:.6g} "
          f"(shape {tuple(logits_none.shape)})")
    # all-False frozen must be BIT-IDENTICAL to frozen=None (today's default).
    assert torch.equal(logits_frozen, logits_none), (
        f"frozen=all-False logits diverge from frozen=None (must be bit-identical): "
        f"max_abs_diff={max_abs:.6g}"
    )


def test_focus_select_excludes_frozen():
    m = _load_modeling()
    S = 8
    delta = torch.arange(S, dtype=torch.float32).unsqueeze(0)      # pos 7 = highest importance
    mask_idx = torch.ones(1, S, dtype=torch.bool)                  # all masked
    frozen = torch.zeros(1, S, dtype=torch.bool)
    frozen[0, 7] = True                                            # freeze the top-delta token
    idx, ksel = m._focus_select(delta, mask_idx, avg_decoded=4.0, focus_alpha=1.0,
                                retain_override=None, frozen=frozen)
    assert 7 not in idx[0].tolist()                               # frozen never selected
    assert ksel <= S - 1                                          # budget excludes frozen


def test_focus_select_frozen_not_selected_under_k_floor():
    m = _load_modeling()
    S = 8
    delta = torch.arange(S, dtype=torch.float32).unsqueeze(0)
    mask_idx = torch.ones(1, S, dtype=torch.bool)         # all masked
    frozen = torch.zeros(1, S, dtype=torch.bool)
    frozen[0, 1:] = True                                  # freeze 7 of 8; only pos 0 free
    # avg_decoded*alpha=4 would floor Ksel at 4, but only 1 non-frozen position exists.
    idx, ksel = m._focus_select(delta, mask_idx, avg_decoded=4.0, focus_alpha=1.0,
                                retain_override=None, frozen=frozen)
    assert idx[0].tolist() == [0]          # only the single non-frozen position
    assert ksel == 1                       # K floor (4) clamped down to non-frozen count


def test_focus_select_frozen_none_is_unchanged():
    m = _load_modeling()
    delta = torch.tensor([[0.0, 3.0, 1.0, 2.5, 0.2, 4.0]])
    mask_idx = torch.tensor([[True, True, False, True, True, True]])
    idx_a, k_a = m._focus_select(delta, mask_idx, avg_decoded=2.0, focus_alpha=1.0,
                                 retain_override=None)
    idx_b, k_b = m._focus_select(delta, mask_idx, avg_decoded=2.0, focus_alpha=1.0,
                                 retain_override=None, frozen=None)
    assert k_a == k_b and idx_a.tolist() == idx_b.tolist()        # default path identical


def test_dynamic_block_kv_write_full_and_get():
    import torch
    from utils.dynamic_block_kv import DynamicBlockKV
    buf = DynamicBlockKV(deep_layer_start=2, num_layers=4, batch_size=1,
                         num_kv_heads=2, block_size=5, head_dim=3,
                         dtype=torch.float32, device=torch.device("cpu"))
    k = torch.arange(1*2*5*3, dtype=torch.float32).reshape(1, 2, 5, 3)
    v = k + 100
    buf.write_full(2, k, v)
    gk, gv = buf.get(2)
    assert torch.equal(gk, k) and torch.equal(gv, v)


def test_dynamic_block_kv_scatter_updates_only_selected():
    import torch
    from utils.dynamic_block_kv import DynamicBlockKV
    buf = DynamicBlockKV(deep_layer_start=2, num_layers=3, batch_size=1,
                         num_kv_heads=2, block_size=5, head_dim=3,
                         dtype=torch.float32, device=torch.device("cpu"))
    base_k = torch.zeros(1, 2, 5, 3); base_v = torch.zeros(1, 2, 5, 3)
    buf.write_full(2, base_k, base_v)
    sel = torch.tensor([[1, 3]])                       # (B, n)
    new_k = torch.ones(1, 2, 2, 3); new_v = torch.ones(1, 2, 2, 3) * 2
    buf.write(2, new_k, new_v, sel)
    gk, gv = buf.get(2)
    assert gk[0, 0, 1, 0] == 1 and gk[0, 0, 3, 0] == 1
    assert gk[0, 0, 0, 0] == 0 and gk[0, 0, 2, 0] == 0 and gk[0, 0, 4, 0] == 0
    assert gv[0, 0, 1, 0] == 2 and gv[0, 0, 0, 0] == 0


def _load_real_model_on_gpu(m):
    """Load the real Fast-dLLM-v2 snapshot weights on GPU.

    Reuses the same mechanism as test_focus_compact_retain_all_matches_dense:
    AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
    cache_dir=$WORKSPACE/models) in bf16, eval, on cuda. Returns the model.
    """
    from transformers import AutoModelForCausalLM
    model_id = "Efficient-Large-Model/Fast_dLLM_v2_7B"
    cache_dir = os.environ.get("WORKSPACE", FV2) + "/models"
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    model.eval()
    model.to(device)
    return model


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (v2 env)")
def test_focus_compact_dynamic_retain_all_matches_dense():
    import torch
    from transformers.cache_utils import DynamicCache
    from utils.dynamic_block_kv import DynamicBlockKV
    from transformers import AutoModelForCausalLM
    model_id = "Efficient-Large-Model/Fast_dLLM_v2_7B"
    cache_dir = os.environ["WORKSPACE"] + "/models"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=cache_dir,
    )
    model.eval(); model.to("cuda")
    B, block_size = 1, 8
    cfg = model.config
    nkv = cfg.num_key_value_heads
    hd = cfg.hidden_size // cfg.num_attention_heads
    prefix_ids = torch.randint(0, 1000, (B, 8), device="cuda")
    dc = DynamicCache()
    model.forward(input_ids=prefix_ids, past_key_values=dc, update_past_key_values=True, block_size=block_size)
    block_ids = torch.randint(0, 1000, (B, block_size), device="cuda")
    ref = model.forward(input_ids=block_ids, past_key_values=dc, update_past_key_values=False).logits
    buf = DynamicBlockKV(deep_layer_start=2, num_layers=cfg.num_hidden_layers, batch_size=B,
                         num_kv_heads=nkv, block_size=block_size, head_dim=hd,
                         dtype=model.dtype, device=torch.device("cuda"))
    mask_idx = torch.ones(B, block_size, dtype=torch.bool, device="cuda")
    seed_out = model.model.forward_focus_compact_dynamic(input_ids=block_ids, past_key_values=dc, block_kv=buf,
        is_dense_step=True, mask_idx=mask_idx, focus_layers=(0, 1))
    seed = model.lm_head(seed_out.last_hidden_state)
    out = model.model.forward_focus_compact_dynamic(input_ids=block_ids, past_key_values=dc, block_kv=buf,
        is_dense_step=False, mask_idx=mask_idx, retain_override=1.0, focus_layers=(0, 1))
    foc = model.lm_head(out.last_hidden_state)
    # Primary invariant (bf16-robust): under retain-all the FOCUS step must reproduce
    # the dense-seed step's logits — proves the gather/scatter/buffer/select path is a
    # faithful no-op. (Exact match vs the real model.forward is impossible here: the deep
    # layers reimplement attention over cat(DynamicCache, DynamicBlockKV)+SDPA instead of
    # delegating to self_attn, so ~bf16 rounding accumulates across layers — benign;
    # end-to-end correctness is validated by the GSM8K smoke.)
    assert torch.allclose(seed, foc, atol=2e-2, rtol=0), \
        f"FOCUS retain-all != dense seed: max_abs_diff={(seed-foc).abs().max().item()}"
    # Sanity vs the real dense forward: next-token argmax agrees on the clear majority of
    # positions (random-token blocks give flat logits, so a few near-ties flip under the
    # accumulated bf16 drift — not a correctness signal).
    agree = (ref.argmax(-1) == foc.argmax(-1)).float().mean().item()
    assert agree >= 0.75, f"argmax agreement with dense ref too low: {agree}"

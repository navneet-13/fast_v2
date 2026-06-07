from typing import Callable, Optional, Union
import os
import time
import torch
import types
import logging as _stdlib_logging
from transformers.utils import auto_docstring, logging

logger = logging.get_logger(__name__)
logger.setLevel("INFO")
if not logger.handlers:
    _handler = _stdlib_logging.StreamHandler()
    _handler.setFormatter(_stdlib_logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    ))
    logger.addHandler(_handler)

from utils.static_kv_cache import StaticKVCache, StaticBlockCache
from utils.block_sparse_cache import BlockSparseCache
from utils.attention_backends import (
    get_attention_backend, patch_attention_layers, patch_attention_layers_compiled,
    unpatch_attention_layers,
)
from utils.batch_bucketing import (
    BatchBucket, compute_valid_batch_sizes, snap_batch_size, compact_and_pad,
)
from utils.dynamo_utils import (
    make_compiled_forward, make_eager_forward, log_compile_summary,
    get_latency_summary, warmup_compile_pools,
)

# Constants for Fast_dLLM model
FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645

# Env-gated token-layer FLOP counter for batch_sample (baseline / block-cache).
# Counts per-sequence token-layers = sum over diffusion forwards of
# (input token positions x num decoder layers). Measurement-only; no behavior change.
_BASELINE_FLOP = {"tl": 0, "calls": 0}

MASK_COLOR = 0.5
TOKEN_COLOR = -0.5

@auto_docstring
class Fast_dLLM_QwenForCausalLM:

    @torch.no_grad()
    def batch_sample(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
    ):
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        batch_size = input_ids.shape[0]

        _tl_count = os.environ.get("FAST_DLLM_TL_COUNT", "0") == "1"
        _nlayers = self.config.num_hidden_layers

        if min_len > block_size:
            output = self.forward(input_ids=input_ids[:, :(min_len // block_size * block_size)], use_cache=True, update_past_key_values=True, block_size=block_size)
            logits, past_key_values = output.logits, output.past_key_values
            if min_len % block_size == 0:
                predict_sample_idx = (seq_len == min_len)
                predict_logits = logits[predict_sample_idx, -1:, :]
                next_token = predict_logits.argmax(dim=-1)
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)

        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size

        sample_indices = torch.arange(batch_size, device=self.device)
        finished_samples = {}
        for block_idx in range(start_block_idx, num_blocks):
            block_cache_full_steps = 0
            block_cache_small_steps = 0
            if finished_flag.all():
                break
            if (seq_block_idx == block_idx).all():
                x_init = mask_id * torch.ones((input_ids.shape[0], block_size-input_ids.shape[1]%block_size), device=self.device, dtype=torch.long)
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, :(block_idx + 1)*block_size]

            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()
            step = 0
            block_past_key_values = None
            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)
                if mask_idx.sum() == 0:
                    for sample_idx in range(x_t.shape[0]):
                        if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                            stop_token_idx = (x_t[sample_idx, seq_len[sample_idx]:] == stop_token).nonzero()[0][0]
                            x_t[sample_idx, seq_len[sample_idx]+stop_token_idx+1:] = tokenizer.pad_token_id
                    if finished_flag.all():
                        break
                    output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True, block_size=block_size)
                    if _tl_count:
                        _BASELINE_FLOP["tl"] += x_t[:, -block_size:].shape[1] * _nlayers
                        _BASELINE_FLOP["calls"] += 1
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    step += 1

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

                        if use_block_cache:
                            if block_past_key_values is None or (x_t[:, -block_size+small_block_start_idx] == mask_id).any():
                                block_cache_full_steps += 1
                                output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True)
                                if _tl_count:
                                    _BASELINE_FLOP["tl"] += x_t[:, -block_size:].shape[1] * _nlayers
                                    _BASELINE_FLOP["calls"] += 1
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            else:
                                block_cache_small_steps += 1
                                logits = self.forward(input_ids=x_t[:,start:end], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True, block_past_key_values=block_past_key_values, replace_position=small_block_start_idx).logits
                                if _tl_count:
                                    _BASELINE_FLOP["tl"] += x_t[:, start:end].shape[1] * _nlayers
                                    _BASELINE_FLOP["calls"] += 1
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
                            if _tl_count:
                                _BASELINE_FLOP["tl"] += x_t[:, -block_size:].shape[1] * _nlayers
                                _BASELINE_FLOP["calls"] += 1
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]
                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1) # shape: [B]
                        finished_flag = finished_flag | finished_row_flags

                        step += 1

            if input_ids.shape[1] ==  x_t.shape[1]:
                input_ids = x_t
            else:
                input_ids[:, :(block_idx + 1)*block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                else:
                    if input_ids.shape[1] <= (block_idx + 1)*block_size:
                        input_ids = x_t
                    else:
                        input_ids[seq_block_idx == block_idx, (block_idx + 1)*block_size] = x_t[seq_block_idx == block_idx, (block_idx + 1)*block_size]
            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1
            if finished_flag.any():
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx]:
                        original_idx = sample_indices[sample_idx].item()
                        finished_samples[original_idx] = x_t[sample_idx:sample_idx+1].clone().squeeze(dim=0)
                sample_indices = sample_indices[~finished_flag]
                input_ids = input_ids[~finished_flag]
                seq_block_idx = seq_block_idx[~finished_flag]
                seq_len = seq_len[~finished_flag]
                x_t = x_t[~finished_flag]

                for layer_id in range(len(past_key_values)):
                    past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                    past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]

                finished_flag = finished_flag[~finished_flag]

            # if use_block_cache:
            #     print(
            #         "[batch_sample] block_cache block_idx=%d/%d (num_blocks=%d) "
            #         "batch_size=%d small_blocks_per_block=%d "
            #         "steps_full_block_forward=%d steps_small_block_forward=%d"
            #         % (
            #             block_idx,
            #             num_blocks - 1,
            #             num_blocks,
            #             int(input_ids.shape[0]),
            #             num_small_blocks,
            #             block_cache_full_steps,
            #             block_cache_small_steps,
            #         ),
            #         flush=True,
            #     )

        # add not finished samples since max_new_tokens is reached
        if len(finished_samples) < batch_size:
            for sample_idx in range(x_t.shape[0]):
                original_idx = sample_indices[sample_idx].item()
                finished_samples[original_idx] = x_t[sample_idx:sample_idx+1].clone().squeeze(dim=0)

        assert len(finished_samples) == batch_size
        if _tl_count:
            print(f"[baseline_flops] tl={_BASELINE_FLOP['tl']} calls={_BASELINE_FLOP['calls']}", flush=True)
        return finished_samples

    @torch.no_grad()
    def batch_sample_dynamo(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
        skip_commit_forward=False,
    ):
        """
        Optimized batch sampling with static KV caches, fixed batch size,
        and torch.compile / CUDA graph support.

        Key design patterns (from Token_spase-dllm batch_sample_dynamo_v2):
        1. Stacked KV cache [L,B,S,H,D] (BSHD per layer) with .select(0, layer_idx)
        2. Cache stored on model (self._static_kv_cache) → module state
        3. Module-level compiled function cache (no re-recording per batch)
        4. Full KV buffers + cache_seqlens for flash attention masking
        5. index_copy_ with tensor offsets (no Python-int slice specialization)
        6. Pre-allocated per-step input buffer (_x_block_buf)
        """
        # ---- Configuration from environment ----
        execution_mode = os.environ.get("FAST_DLLM_EXECUTION_MODE", "eager")
        attn_backend_pref = os.environ.get("FAST_DLLM_ATTENTION_BACKEND", "auto")
        max_seq_len_env = int(os.environ.get("FAST_DLLM_MAX_SEQ_LEN", "4096"))
        seq_len_step = int(os.environ.get("FAST_DLLM_SEQ_LEN_STEP", "256"))
        required_len = seq_len.max().item() + max_new_tokens + block_size  # prompt + gen + bridge token headroom
        # Round required_len up to the next multiple of seq_len_step so that
        # small prompt-length variations don't trigger reinit / warmup.
        required_len = ((required_len + seq_len_step - 1) // seq_len_step) * seq_len_step
        max_seq_len = max(max_seq_len_env, required_len)
        compile_mode = os.environ.get("FAST_DLLM_COMPILE_MODE", "reduce-overhead")
        batch_mode = os.environ.get("FAST_DLLM_BATCH_MODE", "static")
        compact_step = int(os.environ.get("FAST_DLLM_COMPACT_STEP", "4"))
        compact_compile = (execution_mode == "compile" and batch_mode == "compact")

        # ---- Model config ----
        config = self.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        batch_size = input_ids.shape[0]
        original_batch_size = batch_size
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        num_small_blocks = block_size // small_block_size

        # ---- Create or reuse persistent state ----
        # All objects below are stored on `self` and reused across batches
        # when shapes match. This is critical for CUDA graph reuse: the same
        # OptimizedModule + same tensor addresses + same patched forward
        # functions → CUDA graph tree replays instead of re-recording.
        _need_reinit = (
            not hasattr(self, '_static_kv_cache')
            or self._static_kv_cache is None
            or self._static_kv_cache.batch_size != batch_size
            or self._static_kv_cache.max_seq_len != max_seq_len
            or getattr(self, '_batch_mode', None) != batch_mode
        )
        if _need_reinit:
            # --- Remove accelerate hooks (from device_map loading) ---
            # accelerate's AlignDevicesHook wraps module.forward with
            # torch.compiler.disable(), causing 59 CUDA graph partitions.
            # Since the model is already on the correct device, the hooks
            # are no-ops and safe to remove.
            # logger.info("[batch_sample_dynamo] Reinit triggered")
            if not getattr(self, '_dynamo_hooks_removed', False):
                try:
                    from accelerate.hooks import remove_hook_from_module
                    for module in self.modules():
                        if hasattr(module, '_hf_hook'):
                            remove_hook_from_module(module)
                except ImportError:
                    pass
                self._dynamo_hooks_removed = True

            # Unpatch old forwards if re-initializing (shape change)
            if hasattr(self, '_dynamo_original_forwards') and self._dynamo_original_forwards:
                unpatch_attention_layers(self, self._dynamo_original_forwards)
            # --- KV cache + buffer allocation (mode-specific) ---
            if compact_compile:
                if use_block_cache:
                    logger.warning(
                        "[batch_sample_dynamo] Block cache not supported with "
                        "compact compile mode. Disabling."
                    )
                    use_block_cache = False
                valid_bs = compute_valid_batch_sizes(batch_size, compact_step)
                self._valid_batch_sizes = valid_bs
                self._compile_pools = {}
                for bs in valid_bs:
                    cache = StaticKVCache(
                        batch_size=bs,
                        max_seq_len=max_seq_len,
                        num_layers=num_layers,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        dtype=self.dtype,
                        device=self.device,
                    )
                    x_buf = torch.zeros(
                        bs, block_size, dtype=torch.long, device=self.device,
                    )
                    torch._dynamo.mark_static_address(x_buf)
                    self._compile_pools[bs] = {"cache": cache, "x_buf": x_buf}
                self._static_kv_cache = self._compile_pools[batch_size]["cache"]
                self._static_block_cache = None
                logger.info(
                    "[batch_sample_dynamo] Compact compile: valid batch sizes=%s",
                    valid_bs,
                )
            else:
                self._compile_pools = None
                self._static_kv_cache = StaticKVCache(
                    batch_size=batch_size,
                    max_seq_len=max_seq_len,
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                self._static_block_cache = StaticBlockCache(
                    batch_size=batch_size,
                    block_size=block_size,
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    dtype=self.dtype,
                    device=self.device,
                ) if use_block_cache else None
            # --- Attention backend + layer patching (once) ---
            attn_backend = get_attention_backend(attn_backend_pref)
            self._dynamo_attn_backend = attn_backend
            self._dynamo_original_forwards = patch_attention_layers(
                self, attn_backend, self._static_block_cache,
            )
            # --- Move CPU float attributes to CUDA ---
            # Module float attributes (e.g., RMSNorm's variance_epsilon,
            # RotaryEmbedding's attention_scaling) get lifted by dynamo as
            # f64 CPU tensor graph inputs. Each one causes a CUDA graph
            # partition ("non gpu ops"). Converting to CUDA f64 tensors
            # eliminates these partitions.
            if not getattr(self, '_dynamo_floats_moved', False):
                device = self.device
                # Note: 'scaling' is excluded because it's passed as a plain
                # float to flash attention's softmax_scale custom op parameter.
                _float_attrs = ('variance_epsilon', 'attention_scaling', 'norm_type')
                for module in self.modules():
                    for attr_name in _float_attrs:
                        val = getattr(module, attr_name, None)
                        if isinstance(val, float):
                            setattr(module, attr_name, torch.tensor(
                                val, dtype=torch.float64, device=device,
                            ))
                self._dynamo_floats_moved = True

            # --- Disable duck sizing to prevent num_kv_heads==batch_size unification ---
            # When num_kv_heads (4) == batch_size (4), dynamo's duck sizing
            # unifies them as the same symbolic dim. When batch compacts to 2,
            # the guard fails → recompilation. Disabling duck sizing prevents
            # this: each dimension is tracked independently.
            if execution_mode == "compile":
                import torch.fx.experimental._config as fx_config
                fx_config.use_duck_shape = False

            # --- Execution mode ---
            if execution_mode == "compile":
                self._dynamo_forward_fn = make_compiled_forward(self, compile_mode=compile_mode)
            else:
                self._dynamo_forward_fn = make_eager_forward(self)
            # --- Pre-allocate per-step buffers ---
            # Constant tensor identity → CUDA graphs see same data pointer every call
            if compact_compile:
                # x_block_buf comes from the current pool (already mark_static)
                self._x_block_buf = self._compile_pools[batch_size]["x_buf"]
            else:
                self._x_block_buf = torch.zeros(
                    batch_size, block_size, dtype=torch.long, device=self.device,
                )
            self._cache_pos_buf = torch.zeros(
                block_size, dtype=torch.long, device=self.device,
            )
            self._cache_pos_arange = torch.arange(
                block_size, dtype=torch.long, device=self.device,
            )
            if execution_mode == "compile":
                if not compact_compile:
                    torch._dynamo.mark_static_address(self._x_block_buf)
                torch._dynamo.mark_static_address(self._cache_pos_buf)
                torch._dynamo.mark_static_address(self._cache_pos_arange)

            self._batch_mode = batch_mode

            # Pre-set dispatch-logged flag BEFORE any compiled forward so
            # dynamo guards on True (not False→True flip which causes recompile)
            attn_backend.mark_dispatch_logged()

            # --- Warmup: pre-record CUDA graphs for all valid batch sizes ---
            if compact_compile:
                warmup_compile_pools(
                    compiled_fn=self._dynamo_forward_fn,
                    compile_pools=self._compile_pools,
                    model=self,
                    block_size=block_size,
                    cache_pos_buf=self._cache_pos_buf,
                    cache_pos_arange=self._cache_pos_arange,
                    attn_backend=self._dynamo_attn_backend,
                    config=self.config,
                    max_seq_len=max_seq_len,
                    dtype=self.dtype,
                    device=self.device,
                )
        else:
            # Reuse: zero out and reset pointers (preserves GPU addresses)
            if self._compile_pools:
                for pool in self._compile_pools.values():
                    pool["cache"].zero_and_reset()
                self._static_kv_cache = self._compile_pools[batch_size]["cache"]
                self._x_block_buf = self._compile_pools[batch_size]["x_buf"]
            else:
                self._static_kv_cache.zero_and_reset()
            if self._static_block_cache is not None:
                self._static_block_cache.reset()

        static_cache = self._static_kv_cache
        static_block_cache = self._static_block_cache
        attn_backend = self._dynamo_attn_backend
        forward_fn = self._dynamo_forward_fn
        _x_block_buf = self._x_block_buf
        _cache_pos_buf = self._cache_pos_buf
        _cache_pos_arange = self._cache_pos_arange

        # ---- Log config ----
        # logger.info(
        #     "[batch_sample_dynamo] execution=%r  attn_backend=%r  "
        #     "compile_mode=%r  max_seq_len=%d  batch=%d  block=%d  "
        #     "small_block=%d  block_cache=%s  reinit=%s",
        #     execution_mode, attn_backend.name, compile_mode,
        #     max_seq_len, batch_size, block_size, small_block_size,
        #     "on" if use_block_cache else "off",
        #     "yes" if _need_reinit else "no (reused)",
        # )
        # print(
        #     f"[batch_sample_dynamo] execution={execution_mode!r}  "
        #     f"attn_backend={attn_backend.name!r}  "
        #     f"compile_mode={compile_mode!r}  "
        #     f"max_seq_len={max_seq_len}  "
        #     f"batch={batch_size}  block={block_size}  "
        #     f"block_cache={'on' if use_block_cache else 'off'}",
        #     flush=True,
        # )

        # ---- Batch bucketing (fixed batch size) ----
        bucket = BatchBucket(
            batch_size=batch_size,
            device=self.device,
            pad_token_id=tokenizer.pad_token_id,
        )

        # ---- Detailed timing setup ----
        _detail_timing_enabled = os.environ.get("FAST_DLLM_TIMING_DETAIL", "0") == "1"
        _detail_timing_skip = int(os.environ.get("FAST_DLLM_TIMING_SKIP", "3"))
        if not hasattr(self, '_timing_detail_call_idx'):
            self._timing_detail_call_idx = 0
            self._timing_detail_records = []
        _this_call_idx = self._timing_detail_call_idx
        self._timing_detail_call_idx += 1
        _do_detail = _detail_timing_enabled and _this_call_idx >= _detail_timing_skip
        # Accumulators for this batch
        _dt_prefill = 0.0
        _dt_commit = 0.0
        _dt_forward = 0.0
        _dt_sample = 0.0
        _dt_setup = 0.0
        _dt_other = 0.0
        _n_forward = 0
        _n_commit = 0
        _n_blocks = 0
        if _do_detail:
            torch.cuda.synchronize()
            _dt_batch_start = time.perf_counter()

        try:
            # ---- FlashInfer: initial plan() before any forward that uses patched layers ----
            # The prefill (self.forward) goes through patched attention layers which
            # call flash_kvcache_attention → _flashinfer_run_fn. We must initialize
            # the wrapper via plan_flashinfer() before the first forward call.
            if attn_backend.name == "flashinfer":
                prefill_len_est = min_len // block_size * block_size if min_len > block_size else block_size
                static_cache.set_scratch_seqlens(prefill_len_est)
                attn_backend.plan_flashinfer(
                    batch_size=batch_size,
                    query_len=prefill_len_est,
                    max_seq_len=max_seq_len,
                    num_qo_heads=config.num_attention_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    kv_seqlens=static_cache.scratch_seqlens,
                    dtype=self.dtype,
                    device=self.device,
                )

            # ---- Prefill (eager, original forward with eval_mask) ----
            if _do_detail:
                torch.cuda.synchronize()
                _t_prefill_start = time.perf_counter()
            if min_len > block_size:
                prefill_len = min_len // block_size * block_size
                output = self.forward(
                    input_ids=input_ids[:, :prefill_len],
                    use_cache=True,
                    past_key_values=static_cache,
                    update_past_key_values=True,
                    block_size=block_size,
                )
                logits = output.logits

                if min_len % block_size == 0:
                    predict_sample_idx = (seq_len == min_len)
                    predict_logits = logits[predict_sample_idx, -1:, :]
                    next_token = predict_logits.argmax(dim=-1)
                    if input_ids.shape[1] <= min_len:
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                    else:
                        input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
            if _do_detail:
                torch.cuda.synchronize()
                _dt_prefill = time.perf_counter() - _t_prefill_start

            seq_block_idx = seq_len // block_size
            finished_flag = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            start_block_idx = min_len // block_size
            _last_bridge_logits = None  # saved for no-commit path

            # ---- Block generation loop ----
            for block_idx in range(start_block_idx, num_blocks):
                if bucket.all_finished():
                    break
                if _do_detail:
                    torch.cuda.synchronize()
                    _t_setup_start = time.perf_counter()
                    _n_blocks += 1

                # Initialize current block
                if (seq_block_idx == block_idx).all():
                    pad_len = block_size - input_ids.shape[1] % block_size
                    x_init = mask_id * torch.ones(
                        (batch_size, pad_len), device=self.device, dtype=torch.long,
                    )
                    x_init = torch.cat([input_ids, x_init], dim=1)
                    input_ids = x_init
                else:
                    x_init = input_ids[:, :(block_idx + 1) * block_size]

                # Pad finished slots
                bucket.pad_finished_slots(x_init, block_size)
                x_t = x_init.clone()

                # Reset block cache for this block
                if static_block_cache is not None:
                    static_block_cache.reset()

                # ---- Per-block setup: set tensor-valued control signals ----
                past_len = static_cache.get_seq_length()
                static_cache.set_kv_write_start(past_len)
                static_cache.set_scratch_seqlens(past_len + block_size)
                # Pre-compute write index for index_copy_ inside compiled graph.
                # This moves dynamic arange+offset computation to eager code,
                # eliminating 56 "cudagraph partition due to non gpu ops".
                static_cache.prepare_write_idx(block_size)
                # Fill cache_position buffer: arange(past_len, past_len + block_size)
                # Done in eager code so the compiled graph sees a static tensor
                # (D2D memcpy at replay) instead of a Python int that triggers
                # symint specialization per block.
                _cache_pos_buf.copy_(_cache_pos_arange + past_len)

                # FlashInfer: call plan() from eager code before compiled steps.
                # plan() sets up page table metadata; run() uses it inside the graph.
                if attn_backend.name == "flashinfer":
                    attn_backend.plan_flashinfer(
                        batch_size=batch_size,
                        query_len=block_size,
                        max_seq_len=max_seq_len,
                        num_qo_heads=config.num_attention_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        kv_seqlens=static_cache.scratch_seqlens,
                        dtype=self.dtype,
                        device=self.device,
                    )

                step = 0
                if _do_detail:
                    torch.cuda.synchronize()
                    _dt_setup += time.perf_counter() - _t_setup_start

                while True:
                    # If all sequences finished mid-block, stop immediately
                    if bucket.all_finished():
                        break

                    mask_idx = (x_t[:, -block_size:] == mask_id) & bucket.active_mask[:, None]

                    # ---- Block complete: commit cache + bridge token (eager) ----
                    if mask_idx.sum() == 0:
                        for sample_idx in range(batch_size):
                            if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                                post_seq = x_t[sample_idx, seq_len[sample_idx]:]
                                stop_positions = (post_seq == stop_token).nonzero()
                                if stop_positions.numel() > 0:
                                    stop_pos = stop_positions[0][0]
                                    x_t[sample_idx, seq_len[sample_idx] + stop_pos + 1:] = tokenizer.pad_token_id

                        if bucket.all_finished():
                            break

                        if _do_detail:
                            torch.cuda.synchronize()
                            _t_commit_start = time.perf_counter()

                        if skip_commit_forward and _last_bridge_logits is not None:
                            # ---- No-commit path: skip forward, just advance pointer ----
                            # KV cache already has data at [past_len, past_len+block_size)
                            # from the last diffusion step's write_scratch_compiled().
                            # The values at recently-unmasked positions are computed from
                            # mask token embeddings (not final tokens) — this is the
                            # approximation we're testing.
                            static_cache._seq_len += block_size
                            static_cache.cache_seqlens.fill_(static_cache._seq_len)
                            # Bridge token from last diffusion step's unshifted logits
                            next_token = _last_bridge_logits.argmax(dim=-1)
                            next_token[finished_flag] = tokenizer.pad_token_id
                            x_t = torch.cat([x_t, next_token], dim=1)
                        else:
                            # ---- Original commit: eager forward with update_past_key_values=True ----
                            output = self.forward(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=static_cache,
                                update_past_key_values=True,
                                block_size=block_size,
                            )
                            logits = output.logits
                            next_token = logits[:, -1:, :].argmax(dim=-1)
                            next_token[finished_flag] = tokenizer.pad_token_id
                            x_t = torch.cat([x_t, next_token], dim=1)

                        if _do_detail:
                            torch.cuda.synchronize()
                            _dt_commit += time.perf_counter() - _t_commit_start
                            _n_commit += 1
                        step += 1
                        break

                    # ---- Diffusion loop over sub-blocks ----
                    for small_block_idx in range(num_small_blocks):
                        small_block_start_idx = small_block_idx * small_block_size
                        small_block_end_idx = small_block_start_idx + small_block_size

                        start = -block_size + small_block_start_idx
                        end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

                        while True:
                            mask_idx = (x_t[:, -block_size:] == mask_id) & bucket.active_mask[:, None]
                            if mask_idx[:, start:end].sum() == 0 or bucket.all_finished():
                                break

                            # -- Forward timing start --
                            if _do_detail:
                                torch.cuda.synchronize()
                                _t_fwd_start = time.perf_counter()

                            if use_block_cache:
                                if len(static_block_cache) <= 0 or (x_t[:, -block_size + small_block_start_idx] == mask_id).any():
                                    output = forward_fn(
                                        input_ids=x_t[:, -block_size:],
                                        use_cache=True,
                                        past_key_values=static_cache,
                                        update_past_key_values=False,
                                        use_block_cache=True,
                                        block_past_key_values=static_block_cache,
                                        cache_position=_cache_pos_buf,
                                        # Explicit logits_to_keep avoids slice(-0, None) special-case
                                        # when compiled: prevents empty-logits bug in B=1 specialization.
                                        logits_to_keep=block_size,
                                    )
                                    logits = output.logits
                                    if skip_commit_forward:
                                        _last_bridge_logits = logits[:, -1:, :]
                                    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                    logits = logits[:, start:end]
                                else:
                                    output = forward_fn(
                                        input_ids=x_t[:, start:end],
                                        use_cache=True,
                                        past_key_values=static_cache,
                                        update_past_key_values=False,
                                        use_block_cache=True,
                                        block_past_key_values=static_block_cache,
                                        replace_position=small_block_start_idx,
                                        logits_to_keep=small_block_size,
                                    )
                                    logits = output.logits
                                    # sub-block: don't update _last_bridge_logits
                                    # (logits[:, -1] predicts after sub-block, not full block)
                                    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            else:
                                # ---- Compiled diffusion step ----
                                # Copy current block into static buffer
                                _x_block_buf.copy_(x_t[:, -block_size:])
                                output = forward_fn(
                                    input_ids=_x_block_buf,
                                    use_cache=True,
                                    past_key_values=static_cache,
                                    update_past_key_values=False,
                                    cache_position=_cache_pos_buf,
                                    # Explicit logits_to_keep avoids slice(-0, None) special-case
                                    # when compiled: prevents empty-logits bug in B=1 specialization.
                                    logits_to_keep=block_size,
                                )
                                logits = output.logits
                                if skip_commit_forward:
                                    _last_bridge_logits = logits[:, -1:, :]
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]

                            # -- Forward timing end / Sample timing start --
                            if _do_detail:
                                torch.cuda.synchronize()
                                _t_after_fwd = time.perf_counter()
                                _dt_forward += _t_after_fwd - _t_fwd_start
                                _n_forward += 1

                            # ---- Sampling and unmasking ----
                            x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                            x1_p = torch.squeeze(
                                torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1,
                            )
                            x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                            unmask_idx = (x1_p > threshold)
                            max_prob_idx = x1_p.argmax(dim=-1)
                            unmask_idx[torch.arange(x_1.shape[0], device=self.device), max_prob_idx] = True
                            unmask_idx = unmask_idx & mask_idx[:, start:end]

                            # Only unmask for active (non-finished) slots
                            unmask_idx = unmask_idx & bucket.active_mask[:, None].expand_as(unmask_idx)

                            x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                            # Check for newly finished sequences
                            finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                            newly_finished = finished_row_flags & ~finished_flag
                            finished_flag = finished_flag | finished_row_flags

                            bucket.mark_finished_compacted(newly_finished, x_t)
                            bucket.pad_finished_slots(x_t, block_size)

                            # -- Sample timing end --
                            if _do_detail:
                                torch.cuda.synchronize()
                                _dt_sample += time.perf_counter() - _t_after_fwd

                            step += 1

                # ---- Update input_ids ----
                if input_ids.shape[1] == x_t.shape[1]:
                    input_ids = x_t
                else:
                    input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                    if (seq_block_idx == block_idx).all():
                        input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                    else:
                        if input_ids.shape[1] <= (block_idx + 1) * block_size:
                            input_ids = x_t
                        else:
                            active_at_block = seq_block_idx == block_idx
                            input_ids[active_at_block, (block_idx + 1) * block_size] = (
                                x_t[active_at_block, (block_idx + 1) * block_size]
                            )

                seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

                # ---- Compact batch: remove finished sequences ----
                if finished_flag.any() and not finished_flag.all():
                    if execution_mode == "eager":
                        # Eager mode: compact freely (no CUDA graph constraints)
                        active_indices = bucket.compact()
                        if active_indices is not None:
                            input_ids = input_ids[active_indices]
                            x_t = x_t[active_indices]
                            seq_block_idx = seq_block_idx[active_indices]
                            seq_len = seq_len[active_indices]
                            finished_flag = finished_flag[active_indices]
                            if _last_bridge_logits is not None:
                                _last_bridge_logits = _last_bridge_logits[active_indices]
                            static_cache.compact_batch(active_indices)
                            if static_block_cache is not None:
                                static_block_cache.compact_batch(active_indices)
                            batch_size = bucket.batch_size
                            _x_block_buf = torch.zeros(
                                batch_size, block_size, dtype=torch.long,
                                device=self.device,
                            )
                            if attn_backend.name == "flashinfer":
                                attn_backend._flashinfer_wrapper = None

                    elif compact_compile:
                        # Compile compact mode: snap to pre-warmed batch size
                        n_active = bucket.num_active()
                        new_bs = snap_batch_size(n_active, self._valid_batch_sizes)
                        old_bs = batch_size
                        if new_bs < batch_size:
                            active_indices = bucket.compact_to(new_bs)
                            if active_indices is not None:
                                n_act = active_indices.numel()
                                pad_n = new_bs - n_act
                                pad_id = tokenizer.pad_token_id

                                # Re-index and pad batch tensors
                                input_ids = compact_and_pad(
                                    input_ids, active_indices, pad_n, pad_id)
                                x_t = compact_and_pad(
                                    x_t, active_indices, pad_n, pad_id)
                                seq_block_idx = compact_and_pad(
                                    seq_block_idx, active_indices, pad_n, 0)
                                seq_len = compact_and_pad(
                                    seq_len, active_indices, pad_n, 0)
                                finished_flag = compact_and_pad(
                                    finished_flag, active_indices, pad_n, True)
                                if _last_bridge_logits is not None:
                                    _last_bridge_logits = compact_and_pad(
                                        _last_bridge_logits, active_indices, pad_n, 0)

                                # Copy active KV data to new pool's cache
                                new_pool = self._compile_pools[new_bs]
                                new_cache = new_pool["cache"]
                                new_cache.zero_and_reset()
                                new_cache._seq_len = static_cache._seq_len
                                new_cache.cache_seqlens[:n_act].fill_(
                                    static_cache._seq_len)
                                for l in range(num_layers):
                                    new_cache.key_cache[l][:n_act].copy_(
                                        static_cache.key_cache[l][active_indices])
                                    new_cache.value_cache[l][:n_act].copy_(
                                        static_cache.value_cache[l][active_indices])

                                # Switch to new pool
                                static_cache = new_cache
                                self._static_kv_cache = static_cache
                                _x_block_buf = new_pool["x_buf"]
                                batch_size = new_bs

                                if attn_backend.name == "flashinfer":
                                    attn_backend._flashinfer_wrapper = None

                                logger.info(
                                    "[compact] batch %d→%d (active=%d, padded=%d)",
                                    old_bs, new_bs, n_act, pad_n,
                                )

             # if(os.environ.get("FAST_DLLM_COMP", "0") == "1"):
            # num_graphs = torch._dynamo.utils.compile_counter
            # print(f"Total Dynamo graphs generated: {num_graphs}", flush=True)

            # ---- Store detailed timing for this batch ----
            if _do_detail:
                torch.cuda.synchronize()
                _dt_batch_total = time.perf_counter() - _dt_batch_start
                _dt_other = _dt_batch_total - (_dt_prefill + _dt_commit + _dt_forward + _dt_sample + _dt_setup)
                self._timing_detail_records.append({
                    'batch_idx': _this_call_idx,
                    'total': _dt_batch_total,
                    'prefill': _dt_prefill,
                    'forward': _dt_forward,
                    'sample_unmask': _dt_sample,
                    'commit': _dt_commit,
                    'setup': _dt_setup,
                    'other': _dt_other,
                    'n_forward': _n_forward,
                    'n_commit': _n_commit,
                    'n_blocks': _n_blocks,
                })

            # ---- Collect all results ----
            finished_samples = bucket.get_results(x_t)
            assert len(finished_samples) == original_batch_size
            return finished_samples

        finally:
            # Restore max-batch-size pool so _need_reinit sees the right
            # batch_size on the next call (compact mode may have swapped to
            # a smaller pool during generation).
            if self._compile_pools and original_batch_size in self._compile_pools:
                self._static_kv_cache = self._compile_pools[original_batch_size]["cache"]
                self._x_block_buf = self._compile_pools[original_batch_size]["x_buf"]
            # elif not self._compile_pools:
            #     # Eager mode: compact_batch may have shrunk the cache.
            #     # Restore to original batch size so next call reuses instead
            #     # of triggering a full reinit (which re-patches attention, etc.).
            #     self._static_kv_cache.restore_full_batch()
            # Attention layers stay patched across batches for CUDA graph reuse.
            # They are only unpatched if shapes change (handled by _need_reinit).
            log_compile_summary()

    @torch.no_grad()
    def batch_sample_dynamo_noc(self, *args, **kwargs):
        """
        No-commit variant of batch_sample_dynamo.

        Skips the commit-step forward pass at the end of each block.
        Instead of recomputing KV with fully-unmasked tokens, it:
          1. Advances the KV cache pointer (_seq_len) by block_size
          2. Uses the bridge token from the last diffusion step's logits

        Trade-off: KV values at recently-unmasked positions retain the
        mask-token-based embeddings from the last diffusion step, rather
        than being recomputed with final token embeddings. This saves
        ~9% of total time (one full eager forward per block) at a
        potential cost to accuracy.

        Use FAST_DLLM_SKIP_COMMIT=1 in eval.py to select this path.
        """
        return Fast_dLLM_QwenForCausalLM.batch_sample_dynamo(self, *args, skip_commit_forward=True, **kwargs)

    @torch.no_grad()
    def batch_sample_sparse(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
    ):
        """
        Token-sparse batch sampling with dense/sparse mode switching.

        Based on batch_sample_dynamo but adds:
        1. Dense steps: full forward through all tokens, caching hidden states,
           attention outputs, and MLP outputs per layer.
        2. Sparse steps: per-layer cosine similarity between current and cached
           hidden states selects a fraction (transfer_ratio) of tokens with
           lowest similarity for recomputation. Only those tokens go through
           Q/K/V projection, attention, and MLP. Results are scattered back
           into cached outputs.
        3. Refresh interval: dense step every N diffusion steps within a block.

        Environment variables:
          FAST_DLLM_TRANSFER_RATIO  — fraction of tokens to recompute (default 1)
          FAST_DLLM_REFRESH_INTERVAL — dense step every N steps (default 5)
          FAST_DLLM_EXECUTION_MODE  — eager only for now
          FAST_DLLM_ATTENTION_BACKEND — auto/flash2/sdpa
          FAST_DLLM_MAX_SEQ_LEN    — max KV cache length (default 4096)
        """
        # ---- Configuration from environment ----
        execution_mode = os.environ.get("FAST_DLLM_EXECUTION_MODE", "eager")
        attn_backend_pref = os.environ.get("FAST_DLLM_ATTENTION_BACKEND", "auto")
        max_seq_len_env = int(os.environ.get("FAST_DLLM_MAX_SEQ_LEN", "4096"))
        seq_len_step = int(os.environ.get("FAST_DLLM_SEQ_LEN_STEP", "256"))
        transfer_ratio = float(os.environ.get("FAST_DLLM_TRANSFER_RATIO", "1.0"))
        refresh_interval = int(os.environ.get("FAST_DLLM_REFRESH_INTERVAL", "5"))
        debug_sparse = os.environ.get("FAST_DLLM_DEBUG_SPARSE", "0") == "1"
        required_len = seq_len.max().item() + max_new_tokens + block_size  # prompt + gen + bridge token headroom
        required_len = ((required_len + seq_len_step - 1) // seq_len_step) * seq_len_step
        max_seq_len = max(max_seq_len_env, required_len)

        # ---- Model config ----
        config = self.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        hidden_size = config.hidden_size
        batch_size = input_ids.shape[0]
        original_batch_size = batch_size
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        num_small_blocks = block_size // small_block_size

        # ---- Create or reuse persistent state ----
        _need_reinit = (
            not hasattr(self, '_static_kv_cache')
            or self._static_kv_cache is None
            or self._static_kv_cache.batch_size != batch_size
            or self._static_kv_cache.max_seq_len != max_seq_len
        )
        if _need_reinit:
            # Remove accelerate hooks (same as batch_sample_dynamo)
            if not getattr(self, '_dynamo_hooks_removed', False):
                try:
                    from accelerate.hooks import remove_hook_from_module
                    for module in self.modules():
                        if hasattr(module, '_hf_hook'):
                            remove_hook_from_module(module)
                except ImportError:
                    pass
                self._dynamo_hooks_removed = True

            if hasattr(self, '_dynamo_original_forwards') and self._dynamo_original_forwards:
                unpatch_attention_layers(self, self._dynamo_original_forwards)

            self._static_kv_cache = StaticKVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self._block_sparse_cache = BlockSparseCache(
                num_layers=num_layers,
                batch_size=batch_size,
                block_size=block_size,
                hidden_size=hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            attn_backend = get_attention_backend(attn_backend_pref)
            self._dynamo_attn_backend = attn_backend
            self._dynamo_original_forwards = patch_attention_layers(
                self, attn_backend, None,
            )
        else:
            self._static_kv_cache.zero_and_reset()
            self._block_sparse_cache.reset()

        static_cache = self._static_kv_cache
        block_sparse_cache = self._block_sparse_cache
        attn_backend = self._dynamo_attn_backend

        # ---- Batch bucketing (fixed batch size) ----
        bucket = BatchBucket(
            batch_size=batch_size,
            device=self.device,
            pad_token_id=tokenizer.pad_token_id,
        )

        try:
            # ---- Prefill (eager, original forward with eval_mask) ----
            if min_len > block_size:
                prefill_len = min_len // block_size * block_size
                output = self.forward(
                    input_ids=input_ids[:, :prefill_len],
                    use_cache=True,
                    past_key_values=static_cache,
                    update_past_key_values=True,
                    block_size=block_size,
                )
                logits = output.logits

                if min_len % block_size == 0:
                    predict_sample_idx = (seq_len == min_len)
                    predict_logits = logits[predict_sample_idx, -1:, :]
                    next_token = predict_logits.argmax(dim=-1)
                    if input_ids.shape[1] <= min_len:
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                    else:
                        input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)

            seq_block_idx = seq_len // block_size
            finished_flag = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            start_block_idx = min_len // block_size
            _dense_steps = 0
            _sparse_steps = 0

            # ---- Block generation loop ----
            for block_idx in range(start_block_idx, num_blocks):
                if bucket.all_finished():
                    break

                # Initialize current block
                if (seq_block_idx == block_idx).all():
                    pad_len = block_size - input_ids.shape[1] % block_size
                    x_init = mask_id * torch.ones(
                        (batch_size, pad_len), device=self.device, dtype=torch.long,
                    )
                    x_init = torch.cat([input_ids, x_init], dim=1)
                    input_ids = x_init
                else:
                    x_init = input_ids[:, :(block_idx + 1) * block_size]

                bucket.pad_finished_slots(x_init, block_size)
                x_t = x_init.clone()

                # Reset block sparse cache for this block
                block_sparse_cache.reset()

                # ---- Per-block setup: set tensor-valued control signals ----
                past_len = static_cache.get_seq_length()
                static_cache.set_kv_write_start(past_len)
                static_cache.set_scratch_seqlens(past_len + block_size)
                # Required before any diffusion forward: patched attention calls
                # write_scratch_compiled(), which needs _scatter_idx from prepare_write_idx.
                static_cache.prepare_write_idx(block_size)

                step = 0
                _block_dense = 0
                _block_sparse = 0

                while True:
                    if bucket.all_finished():
                        break

                    mask_idx = (x_t[:, -block_size:] == mask_id) & bucket.active_mask[:, None]

                    # ---- Block complete: commit cache + bridge token ----
                    if mask_idx.sum() == 0:
                        for sample_idx in range(batch_size):
                            if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                                post_seq = x_t[sample_idx, seq_len[sample_idx]:]
                                stop_positions = (post_seq == stop_token).nonzero()
                                if stop_positions.numel() > 0:
                                    stop_pos = stop_positions[0][0]
                                    x_t[sample_idx, seq_len[sample_idx] + stop_pos + 1:] = tokenizer.pad_token_id

                        if bucket.all_finished():
                            break

                        # Commit step: eager forward with update_past_key_values=True
                        output = self.forward(
                            input_ids=x_t[:, -block_size:],
                            use_cache=True,
                            past_key_values=static_cache,
                            update_past_key_values=True,
                            block_size=block_size,
                        )
                        logits = output.logits
                        next_token = logits[:, -1:, :].argmax(dim=-1)
                        next_token[finished_flag] = tokenizer.pad_token_id
                        x_t = torch.cat([x_t, next_token], dim=1)
                        step += 1
                        break

                    # ---- Diffusion loop over sub-blocks ----
                    for small_block_idx in range(num_small_blocks):
                        small_block_start_idx = small_block_idx * small_block_size
                        small_block_end_idx = small_block_start_idx + small_block_size

                        start = -block_size + small_block_start_idx
                        end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

                        while True:
                            mask_idx = (x_t[:, -block_size:] == mask_id) & bucket.active_mask[:, None]
                            if mask_idx[:, start:end].sum() == 0 or bucket.all_finished():
                                break

                            # ---- Dense vs sparse decision ----
                            is_dense = (step % refresh_interval == 0)
                            if is_dense:
                                _block_dense += 1
                                _dense_steps += 1
                            else:
                                _block_sparse += 1
                                _sparse_steps += 1

                            # ---- Sparse/dense forward ----
                            output = self.forward_sparse(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=static_cache,
                                update_past_key_values=False,
                                block_sparse_cache=block_sparse_cache,
                                is_dense_step=is_dense,
                                transfer_ratio=transfer_ratio,
                                attn_backend=attn_backend,
                            )
                            logits = output.logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]

                            # ---- Sampling and unmasking ----
                            x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                            x1_p = torch.squeeze(
                                torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1,
                            )
                            x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                            unmask_idx = (x1_p > threshold)
                            max_prob_idx = x1_p.argmax(dim=-1)
                            unmask_idx[torch.arange(x_1.shape[0], device=self.device), max_prob_idx] = True
                            unmask_idx = unmask_idx & mask_idx[:, start:end]

                            unmask_idx = unmask_idx & bucket.active_mask[:, None].expand_as(unmask_idx)

                            x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                            # Check for newly finished sequences
                            finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                            newly_finished = finished_row_flags & ~finished_flag
                            finished_flag = finished_flag | finished_row_flags

                            bucket.mark_finished_compacted(newly_finished, x_t)
                            bucket.pad_finished_slots(x_t, block_size)

                            step += 1

                # ---- Update input_ids ----
                if input_ids.shape[1] == x_t.shape[1]:
                    input_ids = x_t
                else:
                    input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                    if (seq_block_idx == block_idx).all():
                        input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                    else:
                        if input_ids.shape[1] <= (block_idx + 1) * block_size:
                            input_ids = x_t
                        else:
                            active_at_block = seq_block_idx == block_idx
                            input_ids[active_at_block, (block_idx + 1) * block_size] = (
                                x_t[active_at_block, (block_idx + 1) * block_size]
                            )

                seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

                # ---- Compact batch: remove finished sequences ----
                if finished_flag.any() and not finished_flag.all():
                    active_indices = bucket.compact()
                    if active_indices is not None:
                        input_ids = input_ids[active_indices]
                        x_t = x_t[active_indices]
                        seq_block_idx = seq_block_idx[active_indices]
                        seq_len = seq_len[active_indices]
                        finished_flag = finished_flag[active_indices]
                        static_cache.compact_batch(active_indices)
                        block_sparse_cache.compact_batch(active_indices)
                        batch_size = bucket.batch_size
           

            # ---- Collect all results ----
            if debug_sparse:
                _total = _dense_steps + _sparse_steps
                print(
                    f"[sparse_debug] batch done: dense={_dense_steps}  "
                    f"sparse={_sparse_steps}  total={_total}  "
                    f"ratio={transfer_ratio}  refresh={refresh_interval}",
                    flush=True,
                )
            finished_samples = bucket.get_results(x_t)
            assert len(finished_samples) == original_batch_size
            return finished_samples

        finally:
            pass

    @torch.no_grad()
    def batch_sample_focus(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
    ):
        """
        FOCUS token-skipping batch sampling.

        Cloned from batch_sample_sparse but swaps the cosine-similarity
        dense/sparse machinery for the FOCUS forward (forward_focus).

        Environment variables:
          FAST_DLLM_FOCUS_ALPHA   — focus alpha (default 1.0)
          FAST_DLLM_FOCUS_LAYERS  — comma-separated layer indices (default 0,1)
          FAST_DLLM_FOCUS_RETAIN  — retain override (default unset)
          FAST_DLLM_DEBUG_FOCUS   — debug print (default 0)
          FAST_DLLM_EXECUTION_MODE  — eager only for now
          FAST_DLLM_ATTENTION_BACKEND — auto/flash2/sdpa
          FAST_DLLM_MAX_SEQ_LEN    — max KV cache length (default 4096)
        """
        # ---- Configuration from environment ----
        execution_mode = os.environ.get("FAST_DLLM_EXECUTION_MODE", "eager")
        attn_backend_pref = os.environ.get("FAST_DLLM_ATTENTION_BACKEND", "auto")
        max_seq_len_env = int(os.environ.get("FAST_DLLM_MAX_SEQ_LEN", "4096"))
        seq_len_step = int(os.environ.get("FAST_DLLM_SEQ_LEN_STEP", "256"))
        focus_alpha = float(os.environ.get("FAST_DLLM_FOCUS_ALPHA", "1.0"))
        focus_layers = tuple(int(x) for x in os.environ.get("FAST_DLLM_FOCUS_LAYERS", "0,1").split(","))
        _retain = os.environ.get("FAST_DLLM_FOCUS_RETAIN", "")
        retain_override = float(_retain) if _retain else None
        debug_focus = os.environ.get("FAST_DLLM_DEBUG_FOCUS", "0") == "1"
        _use_compact = os.environ.get("FAST_DLLM_FOCUS_COMPACT", "0") == "1"
        _delayed_cache = os.environ.get("FAST_DLLM_FOCUS_DELAYED_CACHE", "0") == "1"
        required_len = seq_len.max().item() + max_new_tokens + block_size  # prompt + gen + bridge token headroom
        required_len = ((required_len + seq_len_step - 1) // seq_len_step) * seq_len_step
        max_seq_len = max(max_seq_len_env, required_len)

        # ---- Model config ----
        config = self.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        hidden_size = config.hidden_size
        batch_size = input_ids.shape[0]
        original_batch_size = batch_size
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        num_small_blocks = block_size // small_block_size

        # ---- Create or reuse persistent state ----
        _need_reinit = (
            not hasattr(self, '_static_kv_cache')
            or self._static_kv_cache is None
            or self._static_kv_cache.batch_size != batch_size
            or self._static_kv_cache.max_seq_len != max_seq_len
        )
        if _need_reinit:
            # Remove accelerate hooks (same as batch_sample_dynamo)
            if not getattr(self, '_dynamo_hooks_removed', False):
                try:
                    from accelerate.hooks import remove_hook_from_module
                    for module in self.modules():
                        if hasattr(module, '_hf_hook'):
                            remove_hook_from_module(module)
                except ImportError:
                    pass
                self._dynamo_hooks_removed = True

            if hasattr(self, '_dynamo_original_forwards') and self._dynamo_original_forwards:
                unpatch_attention_layers(self, self._dynamo_original_forwards)

            self._static_kv_cache = StaticKVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self._block_sparse_cache = BlockSparseCache(
                num_layers=num_layers,
                batch_size=batch_size,
                block_size=block_size,
                hidden_size=hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            attn_backend = get_attention_backend(attn_backend_pref)
            self._dynamo_attn_backend = attn_backend
            self._dynamo_original_forwards = patch_attention_layers(
                self, attn_backend, None,
            )
        else:
            self._static_kv_cache.zero_and_reset()
            self._block_sparse_cache.reset()

        static_cache = self._static_kv_cache
        block_sparse_cache = self._block_sparse_cache
        attn_backend = self._dynamo_attn_backend

        # ---- Batch bucketing (fixed batch size) ----
        bucket = BatchBucket(
            batch_size=batch_size,
            device=self.device,
            pad_token_id=tokenizer.pad_token_id,
        )

        try:
            # ---- Prefill (eager, original forward with eval_mask) ----
            if min_len > block_size:
                prefill_len = min_len // block_size * block_size
                output = self.forward(
                    input_ids=input_ids[:, :prefill_len],
                    use_cache=True,
                    past_key_values=static_cache,
                    update_past_key_values=True,
                    block_size=block_size,
                )
                logits = output.logits

                if min_len % block_size == 0:
                    predict_sample_idx = (seq_len == min_len)
                    predict_logits = logits[predict_sample_idx, -1:, :]
                    next_token = predict_logits.argmax(dim=-1)
                    if input_ids.shape[1] <= min_len:
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                    else:
                        input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)

            seq_block_idx = seq_len // block_size
            finished_flag = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            start_block_idx = min_len // block_size
            _dense_steps = 0
            _sparse_steps = 0

            # ---- Block generation loop ----
            avg_decoded = float(block_size)   # seed: assume full block decodable initially
            _ad_count = 0
            for block_idx in range(start_block_idx, num_blocks):
                if bucket.all_finished():
                    break

                # Initialize current block
                if (seq_block_idx == block_idx).all():
                    pad_len = block_size - input_ids.shape[1] % block_size
                    x_init = mask_id * torch.ones(
                        (batch_size, pad_len), device=self.device, dtype=torch.long,
                    )
                    x_init = torch.cat([input_ids, x_init], dim=1)
                    input_ids = x_init
                else:
                    x_init = input_ids[:, :(block_idx + 1) * block_size]

                bucket.pad_finished_slots(x_init, block_size)
                x_t = x_init.clone()

                # Delayed KV cache (paper): settled tokens skip recompute this block.
                # Reallocated fresh each block at the current (possibly compacted) batch
                # size, so no re-indexing is needed on compaction (which only happens at
                # block boundaries). Requires the compact path.
                frozen = (
                    torch.zeros((x_t.shape[0], block_size), dtype=torch.bool, device=self.device)
                    if (_delayed_cache and _use_compact) else None
                )

                # Reset block sparse cache for this block
                block_sparse_cache.reset()

                # ---- Per-block setup: set tensor-valued control signals ----
                past_len = static_cache.get_seq_length()
                static_cache.set_kv_write_start(past_len)
                static_cache.set_scratch_seqlens(past_len + block_size)
                # Required before any diffusion forward: patched attention calls
                # write_scratch_compiled(), which needs _scatter_idx from prepare_write_idx.
                static_cache.prepare_write_idx(block_size)

                step = 0
                _block_dense = 0
                _block_sparse = 0

                while True:
                    if bucket.all_finished():
                        break

                    mask_idx = (x_t[:, -block_size:] == mask_id) & bucket.active_mask[:, None]

                    # ---- Block complete: commit cache + bridge token ----
                    if mask_idx.sum() == 0:
                        for sample_idx in range(batch_size):
                            if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                                post_seq = x_t[sample_idx, seq_len[sample_idx]:]
                                stop_positions = (post_seq == stop_token).nonzero()
                                if stop_positions.numel() > 0:
                                    stop_pos = stop_positions[0][0]
                                    x_t[sample_idx, seq_len[sample_idx] + stop_pos + 1:] = tokenizer.pad_token_id

                        if bucket.all_finished():
                            break

                        # Commit step: eager forward with update_past_key_values=True
                        output = self.forward(
                            input_ids=x_t[:, -block_size:],
                            use_cache=True,
                            past_key_values=static_cache,
                            update_past_key_values=True,
                            block_size=block_size,
                        )
                        logits = output.logits
                        next_token = logits[:, -1:, :].argmax(dim=-1)
                        next_token[finished_flag] = tokenizer.pad_token_id
                        x_t = torch.cat([x_t, next_token], dim=1)
                        step += 1
                        break

                    # ---- Diffusion loop over sub-blocks ----
                    for small_block_idx in range(num_small_blocks):
                        small_block_start_idx = small_block_idx * small_block_size
                        small_block_end_idx = small_block_start_idx + small_block_size

                        start = -block_size + small_block_start_idx
                        end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

                        while True:
                            mask_idx = (x_t[:, -block_size:] == mask_id) & bucket.active_mask[:, None]
                            if mask_idx[:, start:end].sum() == 0 or bucket.all_finished():
                                break

                            # ---- Dense vs sparse decision ----
                            is_dense = (_block_dense == 0)   # first step of each block = dense seed
                            if is_dense:
                                _block_dense += 1; _dense_steps += 1
                            else:
                                _block_sparse += 1; _sparse_steps += 1

                            _ff = self.forward_focus_compact if _use_compact else self.forward_focus
                            output = _ff(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=static_cache,
                                update_past_key_values=False,
                                block_sparse_cache=block_sparse_cache,
                                is_dense_step=is_dense,
                                mask_idx=mask_idx,
                                avg_decoded=avg_decoded,
                                focus_alpha=focus_alpha,
                                retain_override=retain_override,
                                focus_layers=focus_layers,
                                attn_backend=attn_backend,
                                frozen=frozen,
                            )
                            logits = output.logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]

                            # ---- Sampling and unmasking ----
                            x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                            x1_p = torch.squeeze(
                                torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1,
                            )
                            x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                            unmask_idx = (x1_p > threshold)
                            max_prob_idx = x1_p.argmax(dim=-1)
                            unmask_idx[torch.arange(x_1.shape[0], device=self.device), max_prob_idx] = True
                            unmask_idx = unmask_idx & mask_idx[:, start:end]

                            unmask_idx = unmask_idx & bucket.active_mask[:, None].expand_as(unmask_idx)

                            x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                            decoded_now = float(unmask_idx.sum().item()) / max(1, x_1.shape[0])
                            _ad_count += 1
                            avg_decoded += (decoded_now - avg_decoded) / _ad_count

                            # Check for newly finished sequences
                            finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                            newly_finished = finished_row_flags & ~finished_flag
                            finished_flag = finished_flag | finished_row_flags

                            bucket.mark_finished_compacted(newly_finished, x_t)
                            bucket.pad_finished_slots(x_t, block_size)

                            step += 1

                            if frozen is not None:
                                frozen = self._focus_update_frozen(frozen, mask_idx)

                # ---- Update input_ids ----
                if input_ids.shape[1] == x_t.shape[1]:
                    input_ids = x_t
                else:
                    input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                    if (seq_block_idx == block_idx).all():
                        input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                    else:
                        if input_ids.shape[1] <= (block_idx + 1) * block_size:
                            input_ids = x_t
                        else:
                            active_at_block = seq_block_idx == block_idx
                            input_ids[active_at_block, (block_idx + 1) * block_size] = (
                                x_t[active_at_block, (block_idx + 1) * block_size]
                            )

                seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

                # ---- Compact batch: remove finished sequences ----
                if finished_flag.any() and not finished_flag.all():
                    active_indices = bucket.compact()
                    if active_indices is not None:
                        input_ids = input_ids[active_indices]
                        x_t = x_t[active_indices]
                        seq_block_idx = seq_block_idx[active_indices]
                        seq_len = seq_len[active_indices]
                        finished_flag = finished_flag[active_indices]
                        static_cache.compact_batch(active_indices)
                        block_sparse_cache.compact_batch(active_indices)
                        batch_size = bucket.batch_size


            # ---- Collect all results ----
            if debug_focus:
                _total = _dense_steps + _sparse_steps
                print(
                    f"[focus_debug] batch done: dense={_dense_steps}  "
                    f"sparse={_sparse_steps}  total={_total}  "
                    f"alpha={focus_alpha}  avg_decoded={avg_decoded:.4f}",
                    flush=True,
                )
            finished_samples = bucket.get_results(x_t)
            assert len(finished_samples) == original_batch_size
            return finished_samples

        finally:
            pass

    @torch.no_grad()
    def batch_sample_dkv_static(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
    ):
        """
        dKV-Cache on the StaticKVCache substrate. Delayed KV caching, no eviction,
        periodic refresh. Env: FAST_DLLM_DKV_REFRESH_STEPS (default 4).
        """
        # ---- Configuration from environment ----
        execution_mode = os.environ.get("FAST_DLLM_EXECUTION_MODE", "eager")
        attn_backend_pref = os.environ.get("FAST_DLLM_ATTENTION_BACKEND", "auto")
        max_seq_len_env = int(os.environ.get("FAST_DLLM_MAX_SEQ_LEN", "4096"))
        seq_len_step = int(os.environ.get("FAST_DLLM_SEQ_LEN_STEP", "256"))
        refresh_steps = int(os.environ.get("FAST_DLLM_DKV_REFRESH_STEPS", "4"))
        required_len = seq_len.max().item() + max_new_tokens + block_size  # prompt + gen + bridge token headroom
        required_len = ((required_len + seq_len_step - 1) // seq_len_step) * seq_len_step
        max_seq_len = max(max_seq_len_env, required_len)

        # ---- Model config ----
        config = self.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        hidden_size = config.hidden_size
        batch_size = input_ids.shape[0]
        original_batch_size = batch_size
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        num_small_blocks = block_size // small_block_size

        # ---- Create or reuse persistent state ----
        _need_reinit = (
            not hasattr(self, '_static_kv_cache')
            or self._static_kv_cache is None
            or self._static_kv_cache.batch_size != batch_size
            or self._static_kv_cache.max_seq_len != max_seq_len
        )
        if _need_reinit:
            # Remove accelerate hooks (same as batch_sample_dynamo)
            if not getattr(self, '_dynamo_hooks_removed', False):
                try:
                    from accelerate.hooks import remove_hook_from_module
                    for module in self.modules():
                        if hasattr(module, '_hf_hook'):
                            remove_hook_from_module(module)
                except ImportError:
                    pass
                self._dynamo_hooks_removed = True

            if hasattr(self, '_dynamo_original_forwards') and self._dynamo_original_forwards:
                unpatch_attention_layers(self, self._dynamo_original_forwards)

            self._static_kv_cache = StaticKVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self._block_sparse_cache = BlockSparseCache(
                num_layers=num_layers,
                batch_size=batch_size,
                block_size=block_size,
                hidden_size=hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            attn_backend = get_attention_backend(attn_backend_pref)
            self._dynamo_attn_backend = attn_backend
            self._dynamo_original_forwards = patch_attention_layers(
                self, attn_backend, None,
            )
        else:
            self._static_kv_cache.zero_and_reset()
            self._block_sparse_cache.reset()

        static_cache = self._static_kv_cache
        block_sparse_cache = self._block_sparse_cache
        attn_backend = self._dynamo_attn_backend

        # ---- Batch bucketing (fixed batch size) ----
        bucket = BatchBucket(
            batch_size=batch_size,
            device=self.device,
            pad_token_id=tokenizer.pad_token_id,
        )

        try:
            # ---- Prefill (eager, original forward with eval_mask) ----
            if min_len > block_size:
                prefill_len = min_len // block_size * block_size
                output = self.forward(
                    input_ids=input_ids[:, :prefill_len],
                    use_cache=True,
                    past_key_values=static_cache,
                    update_past_key_values=True,
                    block_size=block_size,
                )
                logits = output.logits

                if min_len % block_size == 0:
                    predict_sample_idx = (seq_len == min_len)
                    predict_logits = logits[predict_sample_idx, -1:, :]
                    next_token = predict_logits.argmax(dim=-1)
                    if input_ids.shape[1] <= min_len:
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                    else:
                        input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)

            seq_block_idx = seq_len // block_size
            finished_flag = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            start_block_idx = min_len // block_size
            _dense_steps = 0
            _sparse_steps = 0

            # ---- Block generation loop ----
            for block_idx in range(start_block_idx, num_blocks):
                if bucket.all_finished():
                    break

                # Initialize current block
                if (seq_block_idx == block_idx).all():
                    pad_len = block_size - input_ids.shape[1] % block_size
                    x_init = mask_id * torch.ones(
                        (batch_size, pad_len), device=self.device, dtype=torch.long,
                    )
                    x_init = torch.cat([input_ids, x_init], dim=1)
                    input_ids = x_init
                else:
                    x_init = input_ids[:, :(block_idx + 1) * block_size]

                bucket.pad_finished_slots(x_init, block_size)
                x_t = x_init.clone()

                # Reset block sparse cache for this block
                block_sparse_cache.reset()

                # ---- Per-block setup: set tensor-valued control signals ----
                past_len = static_cache.get_seq_length()
                static_cache.set_kv_write_start(past_len)
                static_cache.set_scratch_seqlens(past_len + block_size)
                # Required before any diffusion forward: patched attention calls
                # write_scratch_compiled(), which needs _scatter_idx from prepare_write_idx.
                static_cache.prepare_write_idx(block_size)

                step = 0
                _block_dense = 0
                _block_sparse = 0

                # ---- Per-block dKV state ----
                prv_transfer_idx = torch.zeros((x_t.shape[0], block_size), dtype=torch.bool, device=self.device)
                cur_transfer_index = torch.zeros((x_t.shape[0], block_size), dtype=torch.bool, device=self.device)
                step_in_block = 0

                while True:
                    if bucket.all_finished():
                        break

                    mask_idx = (x_t[:, -block_size:] == mask_id) & bucket.active_mask[:, None]

                    # ---- Block complete: commit cache + bridge token ----
                    if mask_idx.sum() == 0:
                        for sample_idx in range(batch_size):
                            if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                                post_seq = x_t[sample_idx, seq_len[sample_idx]:]
                                stop_positions = (post_seq == stop_token).nonzero()
                                if stop_positions.numel() > 0:
                                    stop_pos = stop_positions[0][0]
                                    x_t[sample_idx, seq_len[sample_idx] + stop_pos + 1:] = tokenizer.pad_token_id

                        if bucket.all_finished():
                            break

                        # Commit step: eager forward with update_past_key_values=True
                        output = self.forward(
                            input_ids=x_t[:, -block_size:],
                            use_cache=True,
                            past_key_values=static_cache,
                            update_past_key_values=True,
                            block_size=block_size,
                        )
                        logits = output.logits
                        next_token = logits[:, -1:, :].argmax(dim=-1)
                        next_token[finished_flag] = tokenizer.pad_token_id
                        x_t = torch.cat([x_t, next_token], dim=1)
                        step += 1
                        break

                    # ---- Diffusion loop over sub-blocks ----
                    for small_block_idx in range(num_small_blocks):
                        small_block_start_idx = small_block_idx * small_block_size
                        small_block_end_idx = small_block_start_idx + small_block_size

                        start = -block_size + small_block_start_idx
                        end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

                        while True:
                            mask_idx = (x_t[:, -block_size:] == mask_id) & bucket.active_mask[:, None]
                            if mask_idx[:, start:end].sum() == 0 or bucket.all_finished():
                                break

                            # ---- dKV forward: full or partial ----
                            is_full = (step_in_block <= 1) or (step_in_block % refresh_steps == 0)
                            if is_full:
                                output = self.forward_dkv_static(
                                    input_ids=x_t[:, -block_size:],
                                    use_cache=True,
                                    past_key_values=static_cache,
                                    update_past_key_values=False,
                                    is_full_step=True,
                                    attn_backend=attn_backend,
                                )
                                _block_dense += 1; _dense_steps += 1
                            else:
                                fed_indices, _ = self._dkv_fed_indices(prv_transfer_idx)
                                output = self.forward_dkv_static(
                                    input_ids=x_t[:, -block_size:],
                                    use_cache=True,
                                    past_key_values=static_cache,
                                    update_past_key_values=False,
                                    fed_indices=fed_indices,
                                    is_full_step=False,
                                    attn_backend=attn_backend,
                                )
                                _block_sparse += 1; _sparse_steps += 1
                            step_in_block += 1

                            logits = output.logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]

                            # ---- Sampling and unmasking ----
                            x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                            x1_p = torch.squeeze(
                                torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1,
                            )
                            x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                            unmask_idx = (x1_p > threshold)
                            max_prob_idx = x1_p.argmax(dim=-1)
                            unmask_idx[torch.arange(x_1.shape[0], device=self.device), max_prob_idx] = True
                            unmask_idx = unmask_idx & mask_idx[:, start:end]

                            unmask_idx = unmask_idx & bucket.active_mask[:, None].expand_as(unmask_idx)

                            x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                            # Check for newly finished sequences
                            finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                            newly_finished = finished_row_flags & ~finished_flag
                            finished_flag = finished_flag | finished_row_flags

                            bucket.mark_finished_compacted(newly_finished, x_t)
                            bucket.pad_finished_slots(x_t, block_size)

                            step += 1

                            # ---- dKV shift bookkeeping ----
                            all_decoded = (x_t[:, -block_size:] != mask_id)
                            prv_transfer_idx, cur_transfer_index = cur_transfer_index, all_decoded

                # ---- Update input_ids ----
                if input_ids.shape[1] == x_t.shape[1]:
                    input_ids = x_t
                else:
                    input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                    if (seq_block_idx == block_idx).all():
                        input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                    else:
                        if input_ids.shape[1] <= (block_idx + 1) * block_size:
                            input_ids = x_t
                        else:
                            active_at_block = seq_block_idx == block_idx
                            input_ids[active_at_block, (block_idx + 1) * block_size] = (
                                x_t[active_at_block, (block_idx + 1) * block_size]
                            )

                seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

                # ---- Compact batch: remove finished sequences ----
                if finished_flag.any() and not finished_flag.all():
                    active_indices = bucket.compact()
                    if active_indices is not None:
                        input_ids = input_ids[active_indices]
                        x_t = x_t[active_indices]
                        seq_block_idx = seq_block_idx[active_indices]
                        seq_len = seq_len[active_indices]
                        finished_flag = finished_flag[active_indices]
                        static_cache.compact_batch(active_indices)
                        block_sparse_cache.compact_batch(active_indices)
                        batch_size = bucket.batch_size

            # ---- Collect all results ----
            finished_samples = bucket.get_results(x_t)
            assert len(finished_samples) == original_batch_size
            return finished_samples

        finally:
            pass

    @torch.no_grad()
    def batch_sample_focus_dynamic(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
    ):
        """FOCUS on a DynamicCache prefix + per-block KV buffer (eager SDPA).

        Clone of batch_sample (DynamicCache + finished_samples bookkeeping) with the
        dense diffusion forward replaced by the FOCUS seed/focus steps via
        forward_focus_compact_dynamic. Carries token-skipping + delayed KV caching;
        runs at high batch without the StaticKVCache max_seq_len pre-allocation.
        Compact path only; no sub-block block cache.

        Env: FAST_DLLM_FOCUS_ALPHA, FAST_DLLM_FOCUS_LAYERS, FAST_DLLM_FOCUS_RETAIN,
             FAST_DLLM_FOCUS_DELAYED_CACHE.
        """
        from transformers.cache_utils import DynamicCache
        from utils.dynamic_block_kv import DynamicBlockKV

        focus_alpha = float(os.environ.get("FAST_DLLM_FOCUS_ALPHA", "1.0"))
        focus_layers = tuple(int(x) for x in os.environ.get("FAST_DLLM_FOCUS_LAYERS", "0,1").split(","))
        _retain = os.environ.get("FAST_DLLM_FOCUS_RETAIN", "")
        retain_override = float(_retain) if _retain else None
        _delayed_cache = os.environ.get("FAST_DLLM_FOCUS_DELAYED_CACHE", "0") == "1"

        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        last_focus_layer = max(focus_layers)
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        batch_size = input_ids.shape[0]

        if min_len > block_size:
            output = self.forward(input_ids=input_ids[:, :(min_len // block_size * block_size)], use_cache=True, update_past_key_values=True, block_size=block_size)
            logits, past_key_values = output.logits, output.past_key_values
            if min_len % block_size == 0:
                predict_sample_idx = (seq_len == min_len)
                next_token = logits[predict_sample_idx, -1:, :].argmax(dim=-1)
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)
        start_block_idx = min_len // block_size
        sample_indices = torch.arange(batch_size, device=self.device)
        finished_samples = {}
        avg_decoded = float(block_size)
        _ad_count = 0

        for block_idx in range(start_block_idx, num_blocks):
            if finished_flag.all():
                break
            if (seq_block_idx == block_idx).all():
                x_init = mask_id * torch.ones((input_ids.shape[0], block_size - input_ids.shape[1] % block_size), device=self.device, dtype=torch.long)
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, :(block_idx + 1) * block_size]
            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()

            if past_key_values is None:
                past_key_values = DynamicCache()
            # Per-block buffers, sized to the CURRENT (possibly compacted) batch.
            block_kv = DynamicBlockKV(
                deep_layer_start=last_focus_layer + 1, num_layers=num_layers,
                batch_size=x_t.shape[0], num_kv_heads=num_kv_heads, block_size=block_size,
                head_dim=head_dim, dtype=self.dtype, device=self.device,
            )
            frozen = (torch.zeros((x_t.shape[0], block_size), dtype=torch.bool, device=self.device)
                      if _delayed_cache else None)
            _block_dense = 0

            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)
                if mask_idx.sum() == 0:
                    for sample_idx in range(x_t.shape[0]):
                        if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                            stop_token_idx = (x_t[sample_idx, seq_len[sample_idx]:] == stop_token).nonzero()[0][0]
                            x_t[sample_idx, seq_len[sample_idx] + stop_token_idx + 1:] = tokenizer.pad_token_id
                    if finished_flag.all():
                        break
                    output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True, block_size=block_size)
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    break

                is_dense = (_block_dense == 0)
                _block_dense += 1
                output = self.forward_focus_compact_dynamic(
                    input_ids=x_t[:, -block_size:], use_cache=True,
                    past_key_values=past_key_values, update_past_key_values=False,
                    block_kv=block_kv, is_dense_step=is_dense, mask_idx=mask_idx,
                    avg_decoded=avg_decoded, focus_alpha=focus_alpha,
                    retain_override=retain_override, focus_layers=focus_layers, frozen=frozen,
                )
                logits = output.logits
                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                x1_p = torch.where(mask_idx, x1_p, -torch.inf)
                unmask_idx = (x1_p > threshold)
                max_prob_idx = x1_p.argmax(dim=-1)
                unmask_idx[torch.arange(x_1.shape[0], device=self.device), max_prob_idx] = True
                unmask_idx = unmask_idx & mask_idx
                x_t[:, -block_size:][unmask_idx] = x_1[unmask_idx]

                decoded_now = float(unmask_idx.sum().item()) / max(1, x_1.shape[0])
                _ad_count += 1
                avg_decoded += (decoded_now - avg_decoded) / _ad_count

                finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                finished_flag = finished_flag | finished_row_flags

                if frozen is not None:
                    frozen = self._focus_update_frozen(frozen, mask_idx)

            if input_ids.shape[1] == x_t.shape[1]:
                input_ids = x_t
            else:
                input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                else:
                    if input_ids.shape[1] <= (block_idx + 1) * block_size:
                        input_ids = x_t
                    else:
                        input_ids[seq_block_idx == block_idx, (block_idx + 1) * block_size] = x_t[seq_block_idx == block_idx, (block_idx + 1) * block_size]
            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1
            if finished_flag.any():
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx]:
                        original_idx = sample_indices[sample_idx].item()
                        finished_samples[original_idx] = x_t[sample_idx:sample_idx + 1].clone().squeeze(dim=0)
                sample_indices = sample_indices[~finished_flag]
                input_ids = input_ids[~finished_flag]
                seq_block_idx = seq_block_idx[~finished_flag]
                seq_len = seq_len[~finished_flag]
                x_t = x_t[~finished_flag]
                for layer_id in range(len(past_key_values)):
                    past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                    past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]
                finished_flag = finished_flag[~finished_flag]

        if len(finished_samples) < batch_size:
            for sample_idx in range(x_t.shape[0]):
                original_idx = sample_indices[sample_idx].item()
                finished_samples[original_idx] = x_t[sample_idx:sample_idx + 1].clone().squeeze(dim=0)
        assert len(finished_samples) == batch_size
        return finished_samples

    @torch.no_grad()
    def batch_sample_dkv(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
    ):
        """dKV-Cache batch sampling — BASELINE-FIRST (minimal deviation from ``batch_sample``).

        Byte-for-byte identical to the dense ``batch_sample`` EXCEPT that, inside the
        ``use_block_cache=False`` decode branch, the per-step ``self.forward(...)`` is
        replaced by ``self.forward_dkv(..., dkv_store=block_kv)``. A per-block KV buffer
        (``block_kv``, a ``DynamicBlockKV``) persists decoded-token K/V across diffusion
        steps and is reset at each block boundary. One-step delay + periodic full refresh
        (every ``refresh_steps`` steps; first two steps always full) prevent KV staleness.

        The commit/bridge step, the sub-block (``small_block_size``) loop, and the
        ``start:end`` windows are copied verbatim from ``batch_sample`` — nothing is
        borrowed from the static sampler. At ``refresh_steps=1`` every step is a full
        recompute, so the output must be byte-identical to ``batch_sample`` (the
        validation gate: ``tests/diff_dkv_vs_baseline.py``).

        Env: FAST_DLLM_DKV_REFRESH_STEPS (default 4).
        """
        from transformers.cache_utils import DynamicCache
        from utils.dynamic_block_kv import DynamicBlockKV

        refresh_steps = int(os.environ.get("FAST_DLLM_DKV_REFRESH_STEPS", "4"))
        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        batch_size = input_ids.shape[0]

        if min_len > block_size:
            output = self.forward(input_ids=input_ids[:, :(min_len // block_size * block_size)], use_cache=True, update_past_key_values=True, block_size=block_size)
            logits, past_key_values = output.logits, output.past_key_values
            if min_len % block_size == 0:
                predict_sample_idx = (seq_len == min_len)
                predict_logits = logits[predict_sample_idx, -1:, :]
                next_token = predict_logits.argmax(dim=-1)
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)

        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size

        sample_indices = torch.arange(batch_size, device=self.device)
        finished_samples = {}
        for block_idx in range(start_block_idx, num_blocks):
            if finished_flag.all():
                break
            if (seq_block_idx == block_idx).all():
                x_init = mask_id * torch.ones((input_ids.shape[0], block_size-input_ids.shape[1]%block_size), device=self.device, dtype=torch.long)
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, :(block_idx + 1)*block_size]

            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()
            step = 0

            # dKV: per-block KV buffer (reset at each block boundary) + one-step-delay
            # bookkeeping. past_key_values must be a real Cache for forward_dkv's
            # get_seq_length()/prefix-cat; an empty DynamicCache is equivalent to None.
            if past_key_values is None:
                past_key_values = DynamicCache()
            block_kv = DynamicBlockKV(
                deep_layer_start=0, num_layers=num_layers,
                batch_size=x_t.shape[0], num_kv_heads=num_kv_heads, block_size=block_size,
                head_dim=head_dim, dtype=self.dtype, device=self.device,
            )
            prv_transfer_idx = torch.zeros((x_t.shape[0], block_size), dtype=torch.bool, device=self.device)
            cur_transfer_index = torch.zeros((x_t.shape[0], block_size), dtype=torch.bool, device=self.device)
            step_in_block = 0

            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)
                if mask_idx.sum() == 0:
                    for sample_idx in range(x_t.shape[0]):
                        if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                            stop_token_idx = (x_t[sample_idx, seq_len[sample_idx]:] == stop_token).nonzero()[0][0]
                            x_t[sample_idx, seq_len[sample_idx]+stop_token_idx+1:] = tokenizer.pad_token_id
                    if finished_flag.all():
                        break
                    output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True, block_size=block_size)
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    step += 1

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

                        # dKV substitution for baseline's self.forward(update_past_key_values=False):
                        # full recompute on refresh boundaries (and the first two steps); else feed
                        # only the not-yet-frozen positions and read the rest from block_kv.
                        is_full = (step_in_block <= 1) or (step_in_block % refresh_steps == 0)
                        if is_full:
                            output = self.forward_dkv(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, dkv_store=block_kv, is_full_step=True)
                        else:
                            fed_indices, _ = self._dkv_fed_indices(prv_transfer_idx)
                            output = self.forward_dkv(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, dkv_store=block_kv, fed_indices=fed_indices, is_full_step=False)
                        step_in_block += 1
                        logits = output.logits
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]
                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1) # shape: [B]
                        finished_flag = finished_flag | finished_row_flags

                        step += 1

                        # one-step-delay bookkeeping over the WHOLE block (spans sub-blocks):
                        # prv = positions decoded >=2 steps ago (frozen in block_kv).
                        all_decoded = (x_t[:, -block_size:] != mask_id)
                        prv_transfer_idx, cur_transfer_index = cur_transfer_index, all_decoded

            if input_ids.shape[1] ==  x_t.shape[1]:
                input_ids = x_t
            else:
                input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                else:
                    if input_ids.shape[1] <= (block_idx + 1) * block_size:
                        input_ids = x_t
                    else:
                        input_ids[seq_block_idx == block_idx, (block_idx + 1) * block_size] = x_t[seq_block_idx == block_idx, (block_idx + 1) * block_size]
            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1
            if finished_flag.any():
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx]:
                        original_idx = sample_indices[sample_idx].item()
                        finished_samples[original_idx] = x_t[sample_idx:sample_idx + 1].clone().squeeze(dim=0)
                sample_indices = sample_indices[~finished_flag]
                input_ids = input_ids[~finished_flag]
                seq_block_idx = seq_block_idx[~finished_flag]
                seq_len = seq_len[~finished_flag]
                x_t = x_t[~finished_flag]
                for layer_id in range(len(past_key_values)):
                    past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                    past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]
                finished_flag = finished_flag[~finished_flag]

        if len(finished_samples) < batch_size:
            for sample_idx in range(x_t.shape[0]):
                original_idx = sample_indices[sample_idx].item()
                finished_samples[original_idx] = x_t[sample_idx:sample_idx + 1].clone().squeeze(dim=0)
        assert len(finished_samples) == batch_size
        return finished_samples

    @torch.no_grad()
    def batch_sample_fused(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
    ):
        """
        Fused-kernel batch sampling using persistent CUDA kernels from fused_diffusion.

        Replaces the PyTorch self.forward() diffusion steps with compiled CUDA
        kernels (S1-S7) that cover the full transformer pipeline. Kernels are
        single-sequence (B=token count), so multi-sequence batching loops over
        sequences.

        Environment variables:
          FAST_DLLM_FUSED_MODE  — per_stage / fuse_layer / fuse_step (default per_stage)
          FAST_DLLM_FUSED_DIR   — path to fused_diffusion root (default ../fused_diffusion)
        """
        import sys

        # ---- Configuration ----
        fused_mode = os.environ.get("FAST_DLLM_FUSED_MODE", "per_stage")
        fused_dir = os.environ.get(
            "FAST_DLLM_FUSED_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fused_diffusion"),
        )
        fuse_layer = fused_mode == "fuse_layer"
        fuse_step = fused_mode == "fuse_step"

        # ---- Import KernelRunner ----
        if fused_dir not in sys.path:
            sys.path.insert(0, fused_dir)
        from kernel_runner import KernelRunner

        # ---- GPU capability check ----
        # Kernels require sm_80+ (A100) — cooperative launches use 108–124 KB
        # shared memory per block, which exceeds the 99 KB limit on sm_86 (A5000).
        # On sm_86 the launches silently fail and logits stay at zero → token 0 ("!").
        device = next(self.parameters()).device
        if device.type == "cuda":
            cc = torch.cuda.get_device_capability(device)
            if cc[0] < 8 or (cc[0] == 8 and cc[1] >= 6):
                raise RuntimeError(
                    f"batch_sample_fused requires sm_80 (A100) or later. "
                    f"Current GPU has compute capability {cc[0]}.{cc[1]} (sm_8{cc[1]}). "
                    f"The fused kernels use 108-124 KB shared memory per block which "
                    f"exceeds the 99 KB limit on sm_86. Use FAST_DLLM_USE_DYNAMO=1 instead."
                )

        batch_size = input_ids.shape[0]
        original_batch_size = batch_size
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        num_small_blocks = block_size // small_block_size

        logger.info(
            "[batch_sample_fused] mode=%s  batch=%d  block=%d  small_block=%d  "
            "block_cache=%s  fused_dir=%s",
            fused_mode, batch_size, block_size, small_block_size,
            "on" if use_block_cache else "off", fused_dir,
        )

        # ---- Initialize KernelRunner (once, reuse across batches) ----
        if not hasattr(self, '_kernel_runner') or self._kernel_runner is None:
            self._kernel_runner = KernelRunner(
                self, block_size=block_size, small_block_size=small_block_size,
                fuse_layer=fuse_layer, fuse_step=fuse_step,
            )
            logger.info("[batch_sample_fused] KernelRunner initialized")

        runner = self._kernel_runner

        # ---- Prefill (PyTorch eager, original forward with eval_mask) ----
        if min_len > block_size:
            prefill_len = min_len // block_size * block_size
            output = self.forward(
                input_ids=input_ids[:, :prefill_len],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )
            logits = output.logits
            past_key_values = output.past_key_values

            if min_len % block_size == 0:
                predict_sample_idx = (seq_len == min_len)
                predict_logits = logits[predict_sample_idx, -1:, :]
                next_token = predict_logits.argmax(dim=-1)
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        # ---- Convert DynamicCache → per-sequence kernel caches ----
        seq_caches = []
        for s in range(batch_size):
            runner.init_main_cache(past_key_values, sample_idx=s)
            seq_caches.append({
                'main_k': [runner.kernel_main_k[l].clone() for l in range(28)],
                'main_v': [runner.kernel_main_v[l].clone() for l in range(28)],
                'main_len': runner.main_cache_len,
            })

        # ---- Batch bucketing ----
        bucket = BatchBucket(
            batch_size=batch_size,
            device=self.device,
            pad_token_id=tokenizer.pad_token_id,
        )

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        start_block_idx = min_len // block_size
        sample_indices = torch.arange(batch_size, device=self.device)

        # ---- Block generation loop ----
        for block_idx in range(start_block_idx, num_blocks):
            if bucket.all_finished():
                break

            # Initialize current block
            if (seq_block_idx == block_idx).all():
                pad_len = block_size - input_ids.shape[1] % block_size
                x_init = mask_id * torch.ones(
                    (batch_size, pad_len), device=self.device, dtype=torch.long,
                )
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, :(block_idx + 1) * block_size]

            bucket.pad_finished_slots(x_init, block_size)
            x_t = x_init.clone()
            step = 0

            while True:
                if bucket.all_finished():
                    break

                mask_idx = (x_t[:, -block_size:] == mask_id) & bucket.active_mask[:, None]

                # ---- Block complete: commit cache + bridge token ----
                if mask_idx.sum() == 0:
                    for sample_idx in range(batch_size):
                        if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                            post_seq = x_t[sample_idx, seq_len[sample_idx]:]
                            stop_positions = (post_seq == stop_token).nonzero()
                            if stop_positions.numel() > 0:
                                stop_pos = stop_positions[0][0]
                                x_t[sample_idx, seq_len[sample_idx] + stop_pos + 1:] = tokenizer.pad_token_id

                    if bucket.all_finished():
                        break

                    # Commit: extend main cache per sequence via kernel
                    all_logits = []
                    for s in range(batch_size):
                        if not bucket.active_mask[s]:
                            # Padding logits for finished sequences
                            all_logits.append(torch.zeros(
                                1, block_size, self.config.vocab_size,
                                dtype=torch.bfloat16, device=self.device,
                            ))
                            continue
                        # Swap in this sequence's cache
                        Fast_dLLM_QwenForCausalLM._swap_cache_to_runner(self, runner, seq_caches[s])
                        logits_s = runner.forward_extend(
                            x_t[s, -block_size:], block_size,
                        ).unsqueeze(0)
                        all_logits.append(logits_s)
                        # Save back updated cache
                        Fast_dLLM_QwenForCausalLM._save_cache_from_runner(self, runner, seq_caches[s])

                    logits = torch.cat(all_logits, dim=0)
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    step += 1
                    break

                # ---- Diffusion loop over sub-blocks ----
                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id) & bucket.active_mask[:, None]
                        if mask_idx[:, start:end].sum() == 0 or bucket.all_finished():
                            break

                        # ---- Per-sequence kernel forward ----
                        all_logits = []
                        for s in range(batch_size):
                            if not bucket.active_mask[s]:
                                # Dummy logits for finished sequence
                                sub_len = small_block_size if not use_block_cache else block_size
                                all_logits.append(torch.zeros(
                                    1, sub_len, self.config.vocab_size,
                                    dtype=torch.bfloat16, device=self.device,
                                ))
                                continue

                            Fast_dLLM_QwenForCausalLM._swap_cache_to_runner(self, runner, seq_caches[s])

                            if use_block_cache:
                                if step == 0 or (x_t[s, -block_size + small_block_start_idx] == mask_id):
                                    logits_s = runner.forward_blockcache_init(
                                        x_t[s, -block_size:], block_size,
                                    ).unsqueeze(0)
                                    logits_s = torch.cat([logits_s[:, :1, :], logits_s[:, :-1, :]], dim=1)
                                    logits_s = logits_s[:, start:end]
                                else:
                                    logits_s = runner.forward_blockcache(
                                        x_t[s, start:end],
                                        small_block_start_idx,
                                        small_block_size,
                                    ).unsqueeze(0)
                                    logits_s = torch.cat([logits_s[:, :1, :], logits_s[:, :-1, :]], dim=1)
                            else:
                                logits_s = runner.forward_standard(
                                    x_t[s, -block_size:], block_size,
                                ).unsqueeze(0)
                                logits_s = torch.cat([logits_s[:, :1, :], logits_s[:, :-1, :]], dim=1)
                                logits_s = logits_s[:, start:end]

                            all_logits.append(logits_s)
                            # No need to save cache after standard/blockcache (they don't modify main cache)

                        logits = torch.cat(all_logits, dim=0)

                        # ---- Sampling and unmasking (same as batch_sample_dynamo) ----
                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        x1_p = torch.squeeze(
                            torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1,
                        )
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0], device=self.device), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]
                        unmask_idx = unmask_idx & bucket.active_mask[:, None].expand_as(unmask_idx)

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        # Check for newly finished sequences
                        finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                        newly_finished = finished_row_flags & ~finished_flag
                        finished_flag = finished_flag | finished_row_flags

                        bucket.mark_finished_compacted(newly_finished, x_t)
                        bucket.pad_finished_slots(x_t, block_size)

                        step += 1

                # ---- Update input_ids ----
                if input_ids.shape[1] == x_t.shape[1]:
                    input_ids = x_t
                else:
                    input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                    if (seq_block_idx == block_idx).all():
                        input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                    else:
                        if input_ids.shape[1] <= (block_idx + 1) * block_size:
                            input_ids = x_t
                        else:
                            active_at_block = seq_block_idx == block_idx
                            input_ids[active_at_block, (block_idx + 1) * block_size] = (
                                x_t[active_at_block, (block_idx + 1) * block_size]
                            )

                seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

                # ---- Compact batch: remove finished sequences ----
                if finished_flag.any() and not finished_flag.all():
                    active_indices = bucket.compact()
                    if active_indices is not None:
                        input_ids = input_ids[active_indices]
                        x_t = x_t[active_indices]
                        seq_block_idx = seq_block_idx[active_indices]
                        seq_len = seq_len[active_indices]
                        finished_flag = finished_flag[active_indices]
                        # Compact per-sequence caches
                        seq_caches = [seq_caches[i] for i in active_indices.tolist()]
                        batch_size = bucket.batch_size

        # ---- Collect results ----
        finished_samples = bucket.get_results(x_t)
        assert len(finished_samples) == original_batch_size
        return finished_samples

    def _swap_cache_to_runner(self, runner, seq_cache):
        """Load a sequence's cached KV into the KernelRunner."""
        for l in range(28):
            runner.kernel_main_k[l] = seq_cache['main_k'][l]
            runner.kernel_main_v[l] = seq_cache['main_v'][l]
        runner.main_cache_len = seq_cache['main_len']

    def _save_cache_from_runner(self, runner, seq_cache):
        """Save the KernelRunner's current KV cache state back to per-sequence storage."""
        for l in range(28):
            seq_cache['main_k'][l] = runner.kernel_main_k[l]
            seq_cache['main_v'][l] = runner.kernel_main_v[l]
        seq_cache['main_len'] = runner.main_cache_len

    @torch.no_grad()
    def mdm_sample_with_visualization(
        self,
        input_ids,
        tokenizer,
        block_size=32,
        max_new_tokens=1024,
        mask_id=FAST_DLLM_MASK_ID,
        threshold=0.95,
        small_block_size=32,
        stop_token=FAST_DLLM_STOP_TOKEN,
        temperature=0.0,
        top_p=0.95,
    ):
        """
        MDM sampling function with visualization
        with intermediate state output for Gradio visualization
        """
        nfe = 0
        self.model.bd_size = block_size
        num_blocks = max_new_tokens // block_size

        # Initialize state - show all positions as mask
        initial_state = []

        if input_ids.shape[1] > block_size:
            output = self.forward(input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)], use_cache=True, update_past_key_values=True)
            logits, past_key_values = output.logits, output.past_key_values
            nfe += 1
            if input_ids.shape[1] % block_size == 0:
                next_token = logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        else:
            past_key_values = None

        num_small_blocks = block_size // small_block_size
        original_input_length = input_ids.shape[1]

        for block_idx in range(num_blocks):
            if stop_token in input_ids[:, original_input_length:]:
                break
            prompt_length = input_ids.shape[1]

            # Use the length of the first block to initialize state
            first_block_length = block_size - (input_ids.shape[1] % block_size)

            if len(initial_state) == 0:
                for i in range(first_block_length):
                    initial_state.append(("[MASK]", MASK_COLOR))
                yield initial_state
            else:
                for i in range(first_block_length):
                    current_state.append(("[MASK]", MASK_COLOR))
                yield current_state


            # Initialize x_init as mask_id
            x_init = mask_id * torch.ones((input_ids.shape[0], block_size-prompt_length%block_size), device=self.device, dtype=torch.long)
            x_init = torch.cat([input_ids, x_init], dim=1)

            x_t = x_init.clone()
            block_past_key_values = None
            step = 0

            while True:
                if stop_token in x_t[:, prompt_length:]:
                    stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                    if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                        break
                mask_idx = (x_t[:, -block_size:] == mask_id)
                # Decode a complete block, update cache, and generate next token
                if mask_idx.sum() == 0:
                    nfe += 1
                    output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True)
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    token_text = tokenizer.decode([next_token[0].item()], skip_special_tokens=True)
                    # Handle special characters
                    token_text = token_text
                    current_state.append((token_text, TOKEN_COLOR))
                    yield current_state
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

                        logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]

                        step += 1
                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)

                        # Select tokens with probability greater than threshold in p_1t
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, small_block_start_idx:small_block_end_idx], x1_p, -torch.inf)
                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        # Generate visualization state
                        current_state = []
                        generated_tokens = x_t[0, original_input_length:]

                        # Display generated tokens
                        for i, token_id in enumerate(generated_tokens):
                            if token_id == mask_id:
                                current_state.append(("[MASK]", MASK_COLOR))
                            else:
                                token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True)
                                # Handle special characters
                                token_text = token_text
                                current_state.append((token_text, TOKEN_COLOR))

                        yield current_state

            input_ids = x_t

        # Truncate stop_token
        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
            input_ids = input_ids[:, :stop_token_idx+original_input_length+1]

        # Final state - display complete text
        final_state = []
        generated_tokens = input_ids[0, original_input_length:]
        for token_id in generated_tokens:
            token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True)
            token_text = token_text
            final_state.append((token_text, TOKEN_COLOR))

        # Final state doesn't need mask padding, only show actually generated tokens

        yield final_state

        # Return final text
        final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        yield final_text


def setup_model_with_custom_generation(model):
    """
    Set up custom generation functions for the model
    """
    # Add mdm_sample method with visualization
    model.mdm_sample_with_visualization = types.MethodType(Fast_dLLM_QwenForCausalLM.mdm_sample_with_visualization, model)
    # Add optimized dynamo-compatible batch sampling
    model.batch_sample_dynamo = types.MethodType(Fast_dLLM_QwenForCausalLM.batch_sample_dynamo, model)
    return model

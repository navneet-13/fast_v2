# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import json
import time
import types
import generation_functions

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("fast_dllm_v2")
class Fast_dLLM_v2EvalHarness(LM):
    def __init__(
        self,
        model_path='Efficient-Large-Model/Fast_dLLM_v2_7B',
        device="cuda",
        show_speed=False,
        max_new_tokens=2048,
        batch_size=32,
        mask_id=151665,
        use_block_cache=False,
        small_block_size=8,
        bd_size=32,
        threshold=0.9,
        **kwargs,
    ):

        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            cache_dir=os.environ['WORKSPACE'] + '/models',
            **model_kwargs
        )
        self.model.eval()

        # Select batch sampling method:
        #   FAST_DLLM_USE_FOCUS=1  → batch_sample_focus (token-skipping diffusion)
        #   FAST_DLLM_USE_DYNAMO=1 → batch_sample_dynamo (static KV / torch.compile)
        #   FAST_DLLM_USE_SPARSE=1 → batch_sample_sparse (token-sparse diffusion)
        #   default               → batch_sample (original)
        if os.environ.get("FAST_DLLM_USE_FUSED", "0") == "1":
            print("Using fused CUDA kernel path", flush=True)
            self.model.mdm_sample = types.MethodType(
                generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_fused, self.model
            )
        elif os.environ.get("FAST_DLLM_DKV_CACHE", "0") == "1":
            if os.environ.get("FAST_DLLM_DKV_STATIC", "0") == "1":
                print("Using dKV-Cache diffusion mode (delayed KV, StaticKVCache path)", flush=True)
                self.model.mdm_sample = types.MethodType(
                    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_dkv_static, self.model
                )
            else:
                print("Using dKV-Cache diffusion mode (delayed KV, no eviction, DynamicCache path)", flush=True)
                self.model.mdm_sample = types.MethodType(
                    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_dkv, self.model
                )
        elif os.environ.get("FAST_DLLM_USE_FOCUS", "0") == "1":
            if os.environ.get("FAST_DLLM_FOCUS_DYNAMIC", "0") == "1":
                print("Using FOCUS token-skipping diffusion mode (DynamicCache path)", flush=True)
                self.model.mdm_sample = types.MethodType(
                    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_focus_dynamic, self.model
                )
            else:
                print("Using FOCUS token-skipping diffusion mode", flush=True)
                self.model.mdm_sample = types.MethodType(
                    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_focus, self.model
                )
        elif os.environ.get("FAST_DLLM_USE_SPARSE", "0") == "1":
            print("Using token-sparse diffusion mode", flush=True)
            self.model.mdm_sample = types.MethodType(
                generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_sparse, self.model
            )
        elif os.environ.get("FAST_DLLM_USE_DYNAMO", "0") == "1":
            if os.environ.get("FAST_DLLM_SKIP_COMMIT", "0") == "1":
                print("Using static KV / torch.compile path (NO COMMIT forward)", flush=True)
                self.model.mdm_sample = types.MethodType(
                    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_dynamo_noc, self.model
                )
            else:
                print("Using static KV / torch.compile / CUDA-graph optimized path", flush=True)
                self.model.mdm_sample = types.MethodType(
                    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_dynamo, self.model
                )
        else:
            print("Using original batch sampling mode", flush=True)
            self.model.mdm_sample = types.MethodType(
                generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, self.model
            )

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=os.environ['WORKSPACE'] + '/models')
        
        self.show_speed = show_speed
        self.max_new_tokens = max_new_tokens
        self.batch_size = int(batch_size)
        self.mask_id = mask_id
        self.model_path = model_path
        self.use_block_cache = use_block_cache
        self.small_block_size = small_block_size
        self.threshold = threshold
        self.bd_size = bd_size

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self):
        return self.model_path
    
    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(chat_history, add_generation_prompt=add_generation_prompt, tokenize=False)
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    def _encode_pair(self, context, continuation):
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc


    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        batch[:, prompt_index.sum()] = self.mask_id

        batch = torch.cat([batch.to(self.device), torch.full((b, self.bd_size-batch.shape[1]%self.bd_size), self.mask_id, dtype=torch.long, device=self.device)], dim=1)
        if batch.shape[1] > l:
            batch[:, l] = self.tokenizer.eos_token_id

        return batch

    @torch.no_grad()
    def get_logits(self, batch):
        logits = self.model(batch).logits
        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []

        perturbed_seq = self._forward_process(seq.clone(), prompt_index)

        mask_indices = perturbed_seq == self.mask_id

        logits = self.get_logits(perturbed_seq)
        seq = torch.cat([seq.to(self.device), torch.full((seq.shape[0], self.bd_size-seq.shape[1]%self.bd_size), -100, dtype=torch.long, device=self.device)], dim=1)
        loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none')
        loss = loss.sum()
        loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)


    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)
                out.append((ll, 0.0))
        torch.cuda.empty_cache()
        return out
    
    def generate_until(self, requests):
        output = [None] * len(requests)  # pre-allocate output list
        num_tokens = 0
        
        start_time = time.time()
        
        requests_with_indices = [(i, req) for i, req in enumerate(requests)]
        requests_with_indices.sort(key=lambda x: len(x[1].args[0]))
        
        batched_requests = []
        current_batch = []
        for i, req in requests_with_indices:
            current_batch.append((i, req))
            if len(current_batch) == self.batch_size:
                batched_requests.append(current_batch)
                current_batch = []
        
        if current_batch:
            batched_requests.append(current_batch)

        # ---- Profiling setup ----
        # FAST_DLLM_PROFILE=0      — disabled (default)
        # FAST_DLLM_PROFILE=1      — torch.profiler: full kernel breakdown with shapes+stacks
        #                            Accurate per-kernel detail, but adds ~3-25x CPU overhead
        #                            per batch (worse for eager due to per-op instrumentation).
        # FAST_DLLM_PROFILE=2      — CUDA-event profiling: near-zero overhead (<0.1ms/batch).
        #                            Measures wall time, GPU time, and CPU overhead per batch.
        #                            Fair comparison between eager and compile modes.
        # FAST_DLLM_PROFILE_SKIP=N — skip first N batches (default: 5, covers compile warmup)
        # FAST_DLLM_PROFILE_ACTIVE=N — profile N batches after skip (default: 5, mode=1 only)
        _profile_mode = int(os.environ.get("FAST_DLLM_PROFILE", "0"))
        _prof_skip = int(os.environ.get("FAST_DLLM_PROFILE_SKIP", "5"))
        _prof_active = int(os.environ.get("FAST_DLLM_PROFILE_ACTIVE", "5"))
        _prof_ctx = None
        _cuda_event_records = []  # for mode=2: (batch_idx, wall_start, wall_end, gpu_start, gpu_end)
        if _profile_mode == 1:
            _exec_mode = os.environ.get("FAST_DLLM_EXECUTION_MODE", "eager")
            _prof_dir = os.path.join(os.path.dirname(__file__), "logs")
            os.makedirs(_prof_dir, exist_ok=True)
            _prof_trace = os.path.join(
                _prof_dir,
                f"profile_{_exec_mode}_bs{self.batch_size}.json",
            )
            _prof_schedule = torch.profiler.schedule(
                skip_first=_prof_skip,
                wait=0,
                warmup=1,
                active=_prof_active,
            )
            _prof_ctx = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=_prof_schedule,
                on_trace_ready=lambda p: None,  # we export manually at the end
                record_shapes=True,
                with_stack=True,
            )
            _prof_ctx.__enter__()
            print(f"[profile] torch.profiler: skip={_prof_skip}, active={_prof_active}, "
                  f"output={_prof_trace}", flush=True)
        elif _profile_mode == 2:
            print(f"[profile] CUDA-event mode: skip={_prof_skip}, near-zero overhead", flush=True)

        # ---- Per-sample timing ----
        # FAST_DLLM_TIMING=1       — enable per-batch timing report + CSV
        # FAST_DLLM_TIMING_SKIP=N  — skip first N batches for summary stats (default: 1)
        _do_timing = os.environ.get("FAST_DLLM_TIMING", "0") == "1"
        _timing_skip = int(os.environ.get("FAST_DLLM_TIMING_SKIP", "1"))
        _sample_times = []   # (batch_idx, wall_time_s, n_tokens, prompt_len)

        for batch_idx, batch in enumerate(tqdm(batched_requests, desc="Generating...")):
            batched_input_ids = []
            max_len = 0
            min_len = 1e9
            seq_len = []

            for orig_idx, req in batch:
                question = req.args[0]

                if req.task_name.startswith('minerva_math'):
                    question = question.replace("Solution:", "Please reason step by step, and put your final answer within \\boxed{{}}.")
                elif req.task_name.startswith('gsm8k'):
                    question = question.replace("Answer:", "Please reason step by step, and put your final answer within \\boxed{{}}.")
                model_inputs = self.tokenizer([question], return_tensors="pt").to(self.device)
                batched_input_ids.append(model_inputs["input_ids"])
                max_len = max(max_len, model_inputs["input_ids"].shape[1])
                min_len = min(min_len, model_inputs["input_ids"].shape[1])
                seq_len.append(model_inputs["input_ids"].shape[1])

            # pad batched_input_ids to the same length
            batched_input_ids = [torch.cat([input_ids, torch.full((1, max_len - input_ids.shape[1]), self.mask_id, dtype=torch.long, device=self.device)], dim=1) for input_ids in batched_input_ids]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)

            if _do_timing:
                torch.cuda.synchronize()
                _t0 = time.perf_counter()

            # CUDA event timing (mode=2): record GPU start/end around generation
            # Key: we sync ONLY at the end, not at the start. This way:
            #   wall_time = total elapsed (CPU dispatch + GPU compute, overlapped)
            #   gpu_time  = pure GPU kernel execution (event_start to event_end)
            #   cpu_overhead = wall - gpu = time GPU was idle waiting for CPU dispatch
            # If GPU-bound: wall ≈ gpu, cpu_overhead ≈ 0 (dispatch hidden behind compute)
            # If CPU-bound: wall > gpu, cpu_overhead > 0 (GPU starved by slow dispatch)
            if _profile_mode == 2 and batch_idx >= _prof_skip:
                _ev_start = torch.cuda.Event(enable_timing=True)
                _ev_end = torch.cuda.Event(enable_timing=True)
                _ev_start.record()
                _wall_start = time.perf_counter()
            else:
                _ev_start = _ev_end = _wall_start = None

            with torch.no_grad():
                if self.accelerator is not None:
                    generated_ids = self.accelerator.unwrap_model(self.model).mdm_sample(
                        batched_input_ids,
                        tokenizer=self.tokenizer,
                        block_size=self.bd_size,
                        small_block_size=self.small_block_size,
                        max_new_tokens=self.max_new_tokens,
                        mask_id=self.mask_id,
                        min_len=min_len,
                        seq_len=torch.tensor(seq_len, device=self.device),
                        use_block_cache=self.use_block_cache,
                        threshold=self.threshold,
                    )
                else:
                    generated_ids = self.model.mdm_sample(
                        batched_input_ids,
                        tokenizer=self.tokenizer,
                        block_size=self.bd_size,
                        small_block_size=self.small_block_size,
                        max_new_tokens=self.max_new_tokens,
                        mask_id=self.mask_id,
                        min_len=min_len,
                        seq_len=torch.tensor(seq_len, device=self.device),
                        use_block_cache=self.use_block_cache,
                        threshold=self.threshold,
                    )

            if _ev_start is not None:
                _ev_end.record()
                torch.cuda.synchronize()
                _wall_end = time.perf_counter()
                _cuda_event_records.append((
                    batch_idx,
                    _wall_start, _wall_end,
                    _ev_start, _ev_end,
                ))

            if _do_timing:
                torch.cuda.synchronize()
                _t1 = time.perf_counter()

            # extract new generated tokens, and keep original index order
            _batch_tokens = 0
            for batch_pos, (orig_idx, req) in enumerate(batch):
                generated_answer = self.tokenizer.decode(
                    generated_ids[batch_pos][seq_len[batch_pos]:],
                    skip_special_tokens=True
                )

                # count token number
                if self.show_speed:
                    _n = (generated_ids[batch_pos][seq_len[batch_pos]:] != self.mask_id).sum().item()
                    num_tokens += _n
                    _batch_tokens += _n

                # put result in the correct original index position
                output[orig_idx] = generated_answer

                if(os.environ.get("PRINT_GENERATED_ANSWER", "0") == "1"):
                    print('=' * 20, flush=True)
                    print('question: ', req.args[0], flush=True)
                    print('answer: ', generated_answer, flush=True)
                    print('=' * 20, end='\n\n', flush=True)

            if _do_timing:
                _sample_times.append((batch_idx, _t1 - _t0, _batch_tokens, max_len))

            if _prof_ctx is not None:
                _prof_ctx.step()

        if _prof_ctx is not None:
            # Export trace and print summary
            _prof_ctx.__exit__(None, None, None)
            _prof_ctx.export_chrome_trace(_prof_trace)
            print(f"\n[profile] Trace saved to {_prof_trace}", flush=True)
            print(f"[profile] View with: chrome://tracing or https://ui.perfetto.dev/", flush=True)
            # Print key averages table
            print("\n[profile] Key CUDA kernel averages (top 30 by CUDA time):", flush=True)
            print(_prof_ctx.key_averages().table(
                sort_by="cuda_time_total", row_limit=30,
            ), flush=True)

        # ---- CUDA-event profiling report (mode=2) ----
        if _cuda_event_records:
            _exec_mode = os.environ.get("FAST_DLLM_EXECUTION_MODE", "eager")
            print(f"\n{'='*90}", flush=True)
            print(f"[cuda-profile] Per-batch GPU/CPU breakdown  "
                  f"(mode={_exec_mode}, bs={self.batch_size}, skip={_prof_skip})", flush=True)
            print(f"[cuda-profile] Overhead: <0.1ms/batch (CUDA events + perf_counter)", flush=True)
            print(f"{'='*90}", flush=True)
            print(f"{'Batch':>6} {'Wall(s)':>9} {'GPU(s)':>9} {'CPU(s)':>9} "
                  f"{'GPU%':>7} {'CPU%':>7}", flush=True)
            print(f"{'-'*6:>6} {'-'*9:>9} {'-'*9:>9} {'-'*9:>9} "
                  f"{'-'*7:>7} {'-'*7:>7}", flush=True)

            walls, gpus, cpus = [], [], []
            for idx, ws, we, ev_s, ev_e in _cuda_event_records:
                wall = we - ws
                gpu = ev_s.elapsed_time(ev_e) / 1000.0  # ms → s
                cpu = wall - gpu
                walls.append(wall)
                gpus.append(gpu)
                cpus.append(cpu)
                gpu_pct = (gpu / wall * 100) if wall > 0 else 0
                cpu_pct = (cpu / wall * 100) if wall > 0 else 0
                print(f"{idx:>6d} {wall:>9.3f} {gpu:>9.3f} {cpu:>9.3f} "
                      f"{gpu_pct:>6.1f}% {cpu_pct:>6.1f}%", flush=True)

            n = len(walls)
            avg_wall = sum(walls) / n
            avg_gpu = sum(gpus) / n
            avg_cpu = sum(cpus) / n
            med_wall = sorted(walls)[n // 2]
            med_gpu = sorted(gpus)[n // 2]
            med_cpu = sorted(cpus)[n // 2]

            print(f"\n[cuda-profile] Summary ({n} batches, after skip={_prof_skip}):", flush=True)
            print(f"  {'':>10} {'Wall(s)':>10} {'GPU(s)':>10} {'CPU(s)':>10} {'GPU%':>8} {'CPU%':>8}", flush=True)
            print(f"  {'Mean':>10} {avg_wall:>10.3f} {avg_gpu:>10.3f} {avg_cpu:>10.3f} "
                  f"{avg_gpu/avg_wall*100:>7.1f}% {avg_cpu/avg_wall*100:>7.1f}%", flush=True)
            print(f"  {'Median':>10} {med_wall:>10.3f} {med_gpu:>10.3f} {med_cpu:>10.3f} "
                  f"{med_gpu/med_wall*100:>7.1f}% {med_cpu/med_wall*100:>7.1f}%", flush=True)
            print(f"  {'Total':>10} {sum(walls):>10.3f} {sum(gpus):>10.3f} {sum(cpus):>10.3f}", flush=True)
            print(f"{'='*90}", flush=True)

            # Save CSV
            _log_dir = os.path.join(os.path.dirname(__file__), "logs")
            os.makedirs(_log_dir, exist_ok=True)
            _csv_path = os.path.join(
                _log_dir,
                f"cuda_profile_{_exec_mode}_bs{self.batch_size}.csv",
            )
            with open(_csv_path, "w") as f:
                f.write("batch_idx,wall_s,gpu_s,cpu_s,gpu_pct,cpu_pct\n")
                for i, (idx, ws, we, ev_s, ev_e) in enumerate(_cuda_event_records):
                    f.write(f"{idx},{walls[i]:.4f},{gpus[i]:.4f},{cpus[i]:.4f},"
                            f"{gpus[i]/walls[i]*100:.1f},{cpus[i]/walls[i]*100:.1f}\n")
            print(f"[cuda-profile] CSV saved: {_csv_path}", flush=True)

        end_time = time.time()
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}", flush=True)
            print(f"Total time taken: {end_time - start_time} seconds", flush=True)
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}", flush=True)

        # ---- Per-sample timing report ----
        if _do_timing and _sample_times:
            _exec_mode = os.environ.get("FAST_DLLM_EXECUTION_MODE", "eager")
            print(f"\n{'='*80}", flush=True)
            print(f"[timing] Per-batch timing report  (mode={_exec_mode}, batch_size={self.batch_size})", flush=True)
            print(f"[timing] Skipping first {_timing_skip} batch(es) for summary stats", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"{'Batch':>6} {'Time(s)':>9} {'Tokens':>8} {'Tok/s':>8} {'Prompt':>7} {'Skip':>5}", flush=True)
            print(f"{'-'*6:>6} {'-'*9:>9} {'-'*8:>8} {'-'*8:>8} {'-'*7:>7} {'-'*5:>5}", flush=True)
            for idx, wall, toks, plen in _sample_times:
                tok_s = toks / wall if wall > 0 else 0
                skip_mark = "*" if idx < _timing_skip else ""
                print(f"{idx:>6d} {wall:>9.3f} {toks:>8d} {tok_s:>8.1f} {plen:>7d} {skip_mark:>5}", flush=True)

            # Summary stats excluding skipped batches
            steady = [t for t in _sample_times if t[0] >= _timing_skip]
            if not steady:
                steady = _sample_times  # fallback if skip >= total batches
            times = [t[1] for t in steady]
            toks_list = [t[2] for t in steady]
            total_steady_toks = sum(toks_list)
            total_steady_time = sum(times)

            print(f"\n[timing] Summary (excluding first {_timing_skip} batch(es)):", flush=True)
            print(f"  Batches:    {len(steady)}", flush=True)
            print(f"  Total time: {total_steady_time:.2f}s", flush=True)
            print(f"  Total toks: {total_steady_toks}", flush=True)
            if total_steady_time > 0:
                print(f"  Throughput: {total_steady_toks / total_steady_time:.1f} tok/s", flush=True)
            print(f"  Per-batch:  min={min(times):.3f}s  max={max(times):.3f}s  "
                  f"mean={sum(times)/len(times):.3f}s  "
                  f"median={sorted(times)[len(times)//2]:.3f}s", flush=True)

            # Save per-sample CSV for easy comparison
            _log_dir = os.path.join(os.path.dirname(__file__), "logs")
            os.makedirs(_log_dir, exist_ok=True)
            _csv_path = os.path.join(
                _log_dir,
                f"timing_{_exec_mode}_bs{self.batch_size}.csv",
            )
            with open(_csv_path, "w") as f:
                f.write("batch_idx,wall_s,tokens,tok_per_s,prompt_len,skipped\n")
                for idx, wall, toks, plen in _sample_times:
                    f.write(f"{idx},{wall:.4f},{toks},{toks/wall if wall > 0 else 0:.2f},{plen},{1 if idx < _timing_skip else 0}\n")
            print(f"  CSV saved:  {_csv_path}", flush=True)
            print(f"{'='*80}", flush=True)

        # ---- Detailed phase timing report ----
        _detail_model = (self.accelerator.unwrap_model(self.model)
                         if self.accelerator is not None else self.model)
        _detail_records = getattr(_detail_model, '_timing_detail_records', [])
        if _detail_records:
            _exec_mode = os.environ.get("FAST_DLLM_EXECUTION_MODE", "eager")
            _detail_skip = int(os.environ.get("FAST_DLLM_TIMING_SKIP", "3"))
            print(f"\n{'='*110}", flush=True)
            print(f"[detail-timing] Per-batch phase breakdown  "
                  f"(mode={_exec_mode}, bs={self.batch_size}, "
                  f"skip={_detail_skip})", flush=True)
            print(f"[detail-timing] Note: cuda.synchronize() at phase boundaries "
                  f"adds ~5-15% overhead", flush=True)
            print(f"{'='*110}", flush=True)
            print(f"{'Batch':>6} {'Total':>8} {'Prefill':>8} {'Forward':>8} "
                  f"{'Sample':>8} {'Commit':>8} {'Setup':>7} {'Other':>7} "
                  f"{'#Fwd':>5} {'#Cmt':>5} {'#Blk':>5}", flush=True)
            print(f"{'-'*6:>6} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} "
                  f"{'-'*8:>8} {'-'*8:>8} {'-'*7:>7} {'-'*7:>7} "
                  f"{'-'*5:>5} {'-'*5:>5} {'-'*5:>5}", flush=True)
            for r in _detail_records:
                print(f"{r['batch_idx']:>6d} {r['total']:>8.3f} "
                      f"{r['prefill']:>8.3f} {r['forward']:>8.3f} "
                      f"{r['sample_unmask']:>8.3f} {r['commit']:>8.3f} "
                      f"{r['setup']:>7.3f} {r['other']:>7.3f} "
                      f"{r['n_forward']:>5d} {r['n_commit']:>5d} "
                      f"{r['n_blocks']:>5d}", flush=True)

            # Percentage breakdown
            print(f"\n[detail-timing] Phase percentages:", flush=True)
            print(f"{'Batch':>6} {'Prefill%':>9} {'Forward%':>9} "
                  f"{'Sample%':>9} {'Commit%':>9} {'Setup%':>8} "
                  f"{'Other%':>8}", flush=True)
            print(f"{'-'*6:>6} {'-'*9:>9} {'-'*9:>9} "
                  f"{'-'*9:>9} {'-'*9:>9} {'-'*8:>8} "
                  f"{'-'*8:>8}", flush=True)
            for r in _detail_records:
                t = r['total'] if r['total'] > 0 else 1e-9
                print(f"{r['batch_idx']:>6d} "
                      f"{r['prefill']/t*100:>8.1f}% "
                      f"{r['forward']/t*100:>8.1f}% "
                      f"{r['sample_unmask']/t*100:>8.1f}% "
                      f"{r['commit']/t*100:>8.1f}% "
                      f"{r['setup']/t*100:>7.1f}% "
                      f"{r['other']/t*100:>7.1f}%", flush=True)

            # Summary averages
            n = len(_detail_records)
            avg = {k: sum(r[k] for r in _detail_records) / n
                   for k in ['total', 'prefill', 'forward', 'sample_unmask',
                             'commit', 'setup', 'other']}
            avg_t = avg['total'] if avg['total'] > 0 else 1e-9
            tot_fwd = sum(r['n_forward'] for r in _detail_records)
            tot_cmt = sum(r['n_commit'] for r in _detail_records)
            tot_blk = sum(r['n_blocks'] for r in _detail_records)
            print(f"\n[detail-timing] Summary ({n} batches):", flush=True)
            print(f"  {'Phase':<16} {'Avg(s)':>8} {'Total(s)':>9} {'%':>7} "
                  f"{'Avg/call(ms)':>13}", flush=True)
            print(f"  {'-'*16:<16} {'-'*8:>8} {'-'*9:>9} {'-'*7:>7} "
                  f"{'-'*13:>13}", flush=True)
            for phase, count_key in [('Prefill', None), ('Forward', 'n_forward'),
                                     ('Sample+Unmask', None), ('Commit', 'n_commit'),
                                     ('Setup', None), ('Other', None)]:
                key = phase.lower().replace('+', '_').replace('sample_unmask', 'sample_unmask')
                if phase == 'Sample+Unmask':
                    key = 'sample_unmask'
                tot = sum(r[key] for r in _detail_records)
                avg_v = tot / n
                pct = tot / (avg_t * n) * 100
                if count_key and sum(r[count_key] for r in _detail_records) > 0:
                    total_calls = sum(r[count_key] for r in _detail_records)
                    per_call_ms = tot / total_calls * 1000
                    print(f"  {phase:<16} {avg_v:>8.3f} {tot:>9.3f} "
                          f"{pct:>6.1f}% {per_call_ms:>12.2f}", flush=True)
                else:
                    print(f"  {phase:<16} {avg_v:>8.3f} {tot:>9.3f} "
                          f"{pct:>6.1f}%", flush=True)
            print(f"\n  Total forward calls: {tot_fwd}  "
                  f"Total commit calls: {tot_cmt}  "
                  f"Total blocks: {tot_blk}", flush=True)

            # Save CSV
            _log_dir = os.path.join(os.path.dirname(__file__), "logs")
            os.makedirs(_log_dir, exist_ok=True)
            _csv_path = os.path.join(
                _log_dir,
                f"timing_detail_{_exec_mode}_bs{self.batch_size}.csv",
            )
            with open(_csv_path, "w") as f:
                f.write("batch_idx,total_s,prefill_s,forward_s,sample_unmask_s,"
                        "commit_s,setup_s,other_s,n_forward,n_commit,n_blocks\n")
                for r in _detail_records:
                    f.write(f"{r['batch_idx']},{r['total']:.4f},"
                            f"{r['prefill']:.4f},{r['forward']:.4f},"
                            f"{r['sample_unmask']:.4f},{r['commit']:.4f},"
                            f"{r['setup']:.4f},{r['other']:.4f},"
                            f"{r['n_forward']},{r['n_commit']},"
                            f"{r['n_blocks']}\n")
            print(f"  CSV saved: {_csv_path}", flush=True)
            print(f"{'='*110}", flush=True)

        return output


if __name__ == "__main__":
    cli_evaluate()
    

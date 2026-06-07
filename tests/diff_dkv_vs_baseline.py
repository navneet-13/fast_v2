"""Token-level diff: batch_sample (baseline) vs batch_sample_dkv (refresh=1).

At refresh=1 dKV does a full recompute every step, so it MUST be functionally
identical to the dense baseline (greedy, temp=0). Any token divergence localizes
the residual bug. Decisive at tiny sample count (no statistical noise).
"""
import os, sys, types
os.environ.setdefault("FV2", os.getcwd())
os.environ["FAST_DLLM_EXECUTION_MODE"] = "eager"
os.environ["FAST_DLLM_MAX_SEQ_LEN"] = "1024"
os.environ["FAST_DLLM_DKV_REFRESH_STEPS"] = "1"
FV2 = os.environ["FV2"]
if FV2 not in sys.path:
    sys.path.insert(0, FV2)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import generation_functions as gf

MD = os.path.join(FV2, "models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/"
                       "snapshots/0661abf5f9f0ee338970d091052a26c8efa51974")
MASK_ID, BD, SBS, MNT, THR = 151665, 32, 8, 128, 0.9

tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MD, trust_remote_code=True, torch_dtype=torch.bfloat16,
    cache_dir=os.path.join(FV2, "models")).to("cuda").eval()

qs = [
    "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
    "Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did she sell altogether?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
]
prompts = [tok.apply_chat_template(
    [{"role": "user", "content": "Answer the question step by step and put the answer in \\boxed{}: " + q}],
    add_generation_prompt=True, tokenize=False) for q in qs]
enc = [tok(p, return_tensors="pt")["input_ids"] for p in prompts]
maxlen = max(e.shape[1] for e in enc)
seq_len = [e.shape[1] for e in enc]
minlen = min(seq_len)
batched = torch.cat([torch.cat([e, torch.full((1, maxlen - e.shape[1]), MASK_ID, dtype=torch.long)], dim=1)
                     for e in enc], dim=0).to("cuda")

def run(method):
    m = types.MethodType(method, model)
    return m(batched, tokenizer=tok, block_size=BD, small_block_size=SBS, max_new_tokens=MNT,
             mask_id=MASK_ID, min_len=minlen, seq_len=torch.tensor(seq_len, device="cuda"),
             use_block_cache=False, threshold=THR)

def get(out, i):
    return out[i] if not isinstance(out, dict) else out[i]

with torch.no_grad():
    A = run(gf.Fast_dLLM_QwenForCausalLM.batch_sample)
    B = run(gf.Fast_dLLM_QwenForCausalLM.batch_sample_dkv)

for i in range(len(qs)):
    a = get(A, i).tolist()
    b = get(B, i).tolist()
    p = seq_len[i]
    ga, gb = a[p:], b[p:]
    n = min(len(ga), len(gb))
    first = next((k for k in range(n) if ga[k] != gb[k]), None)
    same = (first is None and len(ga) == len(gb))
    print(f"[{i}] prompt_len={p} genA={len(ga)} genB={len(gb)} match={same}"
          + ("" if same else f" first_diff@{first}: A={ga[first] if first is not None else 'len'} B={gb[first] if first is not None else 'len'}"))
    if not same:
        print(f"    A txt: {tok.decode(ga[:60])!r}")
        print(f"    B txt: {tok.decode(gb[:60])!r}")

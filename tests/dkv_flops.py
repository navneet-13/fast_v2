"""Analytical FLOP-proxy saving for dynamic dKV, on a few samples.

Counts token-layers (q-tokens processed x num layers) -- a proxy for the dominant
linear/projection FLOPs, which scale with q-tokens. For each refresh:
  decode_tl          = actual q-tokens x layers spent on diffusion steps
  decode_full_equiv  = same trajectory if every step were full (block_size x layers)
  saving_decode = 1 - decode_tl/decode_full_equiv   (isolates the caching effect)
refresh=1 must give saving=0 (every step full) as a sanity check.
"""
import os, sys, types
os.environ.setdefault("FV2", os.getcwd())
os.environ["FAST_DLLM_EXECUTION_MODE"] = "eager"
os.environ["FAST_DLLM_MAX_SEQ_LEN"] = "1024"
os.environ["FAST_DLLM_TL_COUNT"] = "1"
FV2 = os.environ["FV2"]
if FV2 not in sys.path:
    sys.path.insert(0, FV2)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import generation_functions as gf

MD = os.path.join(FV2, "models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/"
                       "snapshots/0661abf5f9f0ee338970d091052a26c8efa51974")
MASK_ID, BD, SBS, MNT, THR = 151665, 32, 8, 256, 0.9

tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MD, trust_remote_code=True, torch_dtype=torch.bfloat16,
    cache_dir=os.path.join(FV2, "models")).to("cuda").eval()

qs = [
    "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
    "Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did she sell altogether?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. She has only half of the money she needs. Her parents give her $15 and her grandparents twice as much as her parents. How much more money does Betty need?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. How many flowers does Mark have in his garden if there are only 25% as many green flowers as there are yellow and purple flowers?",
    "Ken created a care package to send to his brother. He placed a box on a scale, and then poured into the box enough jelly beans to bring the weight to 2 pounds. How many pounds did the box weigh?",
]
prompts = [tok.apply_chat_template(
    [{"role": "user", "content": "Answer the question step by step and put the answer in \\boxed{}: " + q}],
    add_generation_prompt=True, tokenize=False) for q in qs]

def run_one(prompt, refresh):
    os.environ["FAST_DLLM_DKV_REFRESH_STEPS"] = str(refresh)
    enc = tok(prompt, return_tensors="pt")["input_ids"].to("cuda")
    seq_len = torch.tensor([enc.shape[1]], device="cuda")
    m = types.MethodType(gf.Fast_dLLM_QwenForCausalLM.batch_sample_dkv, model)
    with torch.no_grad():
        m(enc, tokenizer=tok, block_size=BD, small_block_size=SBS, max_new_tokens=MNT,
          mask_id=MASK_ID, min_len=enc.shape[1], seq_len=seq_len,
          use_block_cache=False, threshold=THR)

print(f"refresh | steps_full | steps_cached | decode_tl | full_equiv | saving_decode | saving_total")
for refresh in (1, 2, 4, 8):
    for k in gf._DKV_FLOP:
        gf._DKV_FLOP[k] = 0
    for p in prompts:
        run_one(p, refresh)
    d = gf._DKV_FLOP["decode_tl"]; fe = gf._DKV_FLOP["decode_full_equiv_tl"]; c = gf._DKV_FLOP["commit_tl"]
    sd = (1 - d / fe) if fe else 0.0
    st = (1 - (d + c) / (fe + c)) if (fe + c) else 0.0
    print(f"   {refresh:2d}   |    {gf._DKV_FLOP['steps_full']:4d}    |     {gf._DKV_FLOP['steps_cached']:4d}     | "
          f"{d:8d} | {fe:8d} |    {sd*100:5.1f}%     |   {st*100:5.1f}%")

"""Same bookkeeping check, but against the REAL transformers cache API.

A tiny randomly-initialized Llama on CPU -- no download, no Llama-3.1 weights. What this
validates that the fake cannot: that DynamicCache.crop exists and truncates the way
backend assumes, that past_key_values + partial input_ids + full-length attention_mask
positions tokens correctly, and that the returned arrays are normalized log-probs.
"""
import os
import sys

import numpy as np
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

os.environ.setdefault("GATE1_OUT", "/tmp/gate1_realtest")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backend  # noqa: E402

torch.manual_seed(0)
VOCAB = 64
cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=256)
model = LlamaForCausalLM(cfg).eval()

print("DynamicCache.crop present:", hasattr(DynamicCache(), "crop"))

PROMPTS = {}


def fake_encode(prompt):
    if prompt not in PROMPTS:
        g = torch.Generator().manual_seed(abs(hash(prompt)) % (2**31))
        PROMPTS[prompt] = torch.randint(0, VOCAB, (1, 6 + len(PROMPTS)), generator=g)
    return PROMPTS[prompt]


backend._load = lambda: (None, model)
backend._build_prompt = lambda q, c: f"CB::{q}" if c is None else f"CTX::{q}::{c}"
backend._encode = fake_encode


def reference(question, context, prefix):
    """Uncached: one forward over the whole sequence."""
    out = []
    for ctx in (None, context):
        ids = fake_encode(backend._build_prompt(question, ctx))
        if prefix:
            ids = torch.cat([ids, torch.tensor([prefix], dtype=torch.long)], dim=1)
        with torch.inference_mode():
            logits = model(ids).logits[0, -1, :].float()
        out.append(torch.log_softmax(logits, -1).numpy().astype(np.float32))
    return out[0], out[1]


def check(q, ctx, prefix, label):
    gt, gc = backend.next_token_distributions(q, ctx, prefix)
    et, ec = reference(q, ctx, prefix)
    for got, exp, which in ((gt, et, "log_p_theta"), (gc, ec, "log_p_ctx")):
        assert got.dtype == np.float32, f"{label}: {which} dtype {got.dtype}"
        assert got.shape == (VOCAB,), f"{label}: {which} shape {got.shape}"
        lse = float(np.log(np.exp(got.astype(np.float64)).sum()))
        assert abs(lse) < 1e-4, f"{label}: {which} not normalized, logsumexp={lse}"
        d = float(np.abs(got - exp).max())
        assert d < 2e-3, f"{label}: {which} differs from uncached reference by {d}"


Q, C1, C2 = "who wrote it?", "document one", "document two"

print("greedy decode, prefix growing")
pfx = []
for i in range(10):
    check(Q, C1, list(pfx), f"step {i}")
    pfx.append((i * 7) % VOCAB)

print("tau restart at prefix=[]")
before = backend.cache_stats()["prefills"]
check(Q, C1, [], "restart")
assert backend.cache_stats()["prefills"] == before, "re-prefilled on tau restart"

print("divergent branch")
check(Q, C1, [3, 5, 11], "branch")
check(Q, C1, [3, 5, 11, 2], "branch cont")
check(Q, C1, [3, 5, 40], "branch 2")

print("interleaved prefix lengths")
for p in ([1], [1, 2, 3, 4], [1, 2], [], [1, 2, 3, 4, 5, 6], [9, 9]):
    check(Q, C1, p, f"prefix {p}")

print("context switch and back")
check(Q, C2, [1, 2], "ctx2")
check(Q, C1, [1, 2], "back to ctx1")

print("\nstats:", backend.cache_stats())
print("REAL-MODEL CACHE TEST PASSED")

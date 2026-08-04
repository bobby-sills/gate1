"""Exercises backend.next_token_distributions' KV bookkeeping against an uncached
reference. selftest.py monkeypatches greedy_decode, so none of this is covered there.

The fake cache asserts that the past_len backend claims equals the number of tokens the
cache actually holds, and the fake model's distribution depends on the WHOLE sequence --
so any error in prefill length, rollback length, or the common-prefix computation shows
up either as an assertion or as a mismatch against the reference.
"""
import os
import sys
import types

import numpy as np

os.environ.setdefault("GATE1_OUT", "/tmp/gate1_cachetest")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VOCAB = 32


# ---- fake torch / transformers so backend's lazy imports resolve ----------------------
fake_torch = types.ModuleType("torch")
fake_torch.long = "long"
fake_torch.tensor = lambda data, dtype=None, device=None: np.asarray(data)
sys.modules["torch"] = fake_torch


class FakeCache:
    def __init__(self, tokens=None):
        self.tokens = list(tokens or [])

    def crop(self, n):
        assert n <= len(self.tokens), f"crop({n}) beyond {len(self.tokens)}"
        self.tokens = self.tokens[:n]


fake_tf = types.ModuleType("transformers")
fake_tf.DynamicCache = FakeCache
fake_tf.AutoModelForCausalLM = object
fake_tf.AutoTokenizer = object
sys.modules["transformers"] = fake_tf

import backend  # noqa: E402


def true_logp(seq):
    """Deterministic in the FULL sequence, so a wrong cache state gives wrong numbers."""
    rng = np.random.default_rng(abs(hash(tuple(seq))) % (2**32))
    x = rng.normal(size=VOCAB)
    x = x - np.log(np.exp(x).sum())
    return x.astype(np.float32)


def fake_encode(prompt):
    ids = [(abs(hash(prompt)) >> (3 * i)) % VOCAB for i in range(1 + len(prompt) % 5)]
    return np.array([ids])


calls = {"forward": 0}


def fake_forward(ids, cache, past_len):
    toks = list(np.asarray(ids).reshape(-1))
    assert len(cache.tokens) == past_len, \
        f"backend claimed past_len={past_len} but cache holds {len(cache.tokens)}"
    cache.tokens.extend(toks)
    calls["forward"] += 1
    return cache, true_logp(cache.tokens)


backend._load = lambda: (None, types.SimpleNamespace(device="cpu"))
backend._build_prompt = lambda q, c: f"{q}||{c}"
backend._encode = fake_encode
backend._forward = fake_forward


def reference(question, context, prefix):
    """What the answer must be with no caching at all."""
    t = list(fake_encode(backend._build_prompt(question, None)).reshape(-1)) + list(prefix)
    c = list(fake_encode(backend._build_prompt(question, context)).reshape(-1)) + list(prefix)
    return true_logp(t), true_logp(c)


def check(q, ctx, prefix, label):
    got_t, got_c = backend.next_token_distributions(q, ctx, prefix)
    exp_t, exp_c = reference(q, ctx, prefix)
    assert np.array_equal(got_t, exp_t), f"{label}: log_p_theta mismatch"
    assert np.array_equal(got_c, exp_c), f"{label}: log_p_ctx mismatch"


Q, CTX, CTX2 = "who is x?", "a document", "another document"

print("1. a full greedy decode, prefix growing one token at a time")
pfx = []
for i in range(8):
    check(Q, CTX, list(pfx), f"step {i}")
    pfx.append(i % VOCAB)

print("2. restart at prefix=[] for the next tau -- must roll back, not re-prefill")
before = backend.cache_stats()["prefills"]
check(Q, CTX, [], "tau restart")
assert backend.cache_stats()["prefills"] == before, "re-prefilled instead of rolling back"
print("   prefills unchanged:", before)

print("3. a divergent branch (different tau picks a different token mid-decode)")
pfx = []
for i in range(5):
    check(Q, CTX, list(pfx), f"branch a {i}")
    pfx.append(i % VOCAB)
pfx2 = [0, 1, 99 % VOCAB]
check(Q, CTX, pfx2, "branch b")
check(Q, CTX, pfx2 + [7], "branch b cont")

print("4. repeated identical call must hit the cache, not run a forward pass")
f0 = calls["forward"]
check(Q, CTX, pfx2 + [7], "repeat")
assert calls["forward"] == f0, "repeated call ran a forward pass"
print("   forward passes unchanged:", f0)

print("5. switching context re-prefills and stays correct")
before = backend.cache_stats()["prefills"]
check(Q, CTX2, [1, 2, 3], "new context")
assert backend.cache_stats()["prefills"] == before + 1
check(Q, CTX, [1, 2, 3], "back to old context")

print("6. longer prefix than cached, and shorter, interleaved")
for p in ([1], [1, 2, 3, 4, 5], [1, 2], [1, 2, 3, 4, 5, 6, 7], [], [9]):
    check(Q, CTX, p, f"prefix {p}")

print("\nstats:", backend.cache_stats())
print("CACHE TEST PASSED")

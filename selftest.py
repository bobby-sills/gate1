"""Smoke test: fake backend, full pipeline, simulated crash + resume.
Run from the repo root:  python selftest.py
Claude Code should run this after implementing backend.py to confirm nothing broke.
"""
import json, os, shutil, sys, types
import numpy as np

os.environ["GATE1_OUT"] = "/tmp/gate1_selftest"
shutil.rmtree("/tmp/gate1_selftest", ignore_errors=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gate1

# ---- fake backend -------------------------------------------------------------------
rng = np.random.default_rng(0)
fake = types.ModuleType("backend")
GOLD, CF = "alpha", "beta"


def sample_closed_book(question, n, seed):
    knows = int(question.split("_")[1]) % 2 == 0
    p = 0.95 if knows else 0.05
    return [GOLD if rng.random() < p else "zzz" for _ in range(n)]


def deterministic_features(question):
    knows = int(question.split("_")[1]) % 2 == 0
    # deliberately noisy so entropy is an imperfect proxy -> populated off-diagonal
    return (float(rng.normal(1.0 if knows else 2.0, 1.2)), float(rng.random()))


def next_token_distributions(question, context, prefix):
    raise AssertionError("should not be called; greedy_decode is monkeypatched")


fake.sample_closed_book = sample_closed_book
fake.deterministic_features = deterministic_features
fake.next_token_distributions = next_token_distributions
fake.is_eos = lambda t: True
fake.detokenize = lambda p: ""
sys.modules["backend"] = fake


def fake_decode(question, context, tau, max_tokens=32):
    """Higher tau -> more likely to follow the document."""
    follows = rng.random() < 1 / (1 + np.exp(-(tau - 1.0) * 3))
    doc_is_cf = "COUNTER" in context
    if follows:
        return CF if doc_is_cf else GOLD
    return GOLD if int(question.split("_")[1]) % 2 == 0 else "zzz"


gate1.greedy_decode = fake_decode

# ---- phase 1 (hand-built pool) ------------------------------------------------------
pool = [{"qid": f"q{i}", "subject_qid": f"Q{i}", "question": f"who_{i}_",
         "gold_aliases": [GOLD], "cf_aliases": [CF],
         "factual_context": "FACT ctx", "counterfactual_context": "COUNTER ctx"}
        for i in range(400)]
os.makedirs("/tmp/gate1_selftest", exist_ok=True)
with open("/tmp/gate1_selftest/pool.jsonl", "w") as f:
    for r in pool:
        f.write(json.dumps(r) + "\n")

print("--- alias matcher ---")
assert gate1.alias_match("The answer is Alpha.", [GOLD])
assert not gate1.alias_match("The answer is Beta.", [GOLD])
assert gate1.alias_match("New York City", ["NYC", "New York City"])
print("ok")

print("\n--- phase 2: labels, with simulated crash at 150 ---")
orig_write = gate1.Appender.write
count = {"n": 0}


def crashing_write(self, row):
    count["n"] += 1
    if count["n"] == 150:
        self.f.flush(); os.fsync(self.f.fileno())
        raise KeyboardInterrupt("simulated Colab disconnect")
    orig_write(self, row)


gate1.Appender.write = crashing_write
try:
    gate1.phase_labels()
except KeyboardInterrupt as e:
    print(f"  crashed as intended: {e}")
gate1.Appender.write = orig_write
gate1.phase_labels()          # resume
n_labels = len(gate1.read_jsonl("labels.jsonl"))
assert n_labels == 400, n_labels
print(f"  resumed cleanly, {n_labels} labels, no duplicates:",
      len({r["qid"] for r in gate1.read_jsonl("labels.jsonl")}) == 400)

print("\n--- phase 2.6: sanity ---")
gate1.phase_sanity()

print("\n--- phase 3: cells ---")
gate1.phase_cells()

print("\n--- phase 4a: decode (tau dedup) ---")
gate1.phase_decode()

print("\n--- phase 4b: sweep ---")
gate1.phase_sweep()

print("\n--- phase 5-7: analyze ---")
gate1.N_BOOT = 400           # keep the selftest fast
gate1.phase_analyze()

print("\nSELFTEST PASSED")

"""
selftest2.py -- exercises the whole Gate 2 pipeline with a fake backend.

Run after every change to gate2.py or to backend.hidden_states:

    python selftest2.py

Covers extract (including a simulated crash and resume), train, all four tests, and
calibration, on synthetic activations with a planted signal. Also unit-tests the subject
span locator and the DeLong implementation against a brute-force computation.

The thresholds in gate2 are pre-registered for a 1793-question pool; this harness runs
~200 questions, so MIN_RELATION_N, MIN_MINORITY_N and the bootstrap counts are scaled
down here. Nothing else is patched -- the code paths under test are the real ones.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

TMP = tempfile.mkdtemp(prefix="gate2-selftest-")
os.environ["GATE1_OUT"] = os.path.join(TMP, "outputs")
os.environ["GATE2_CONFIQA"] = os.path.join(TMP, "confiqa.json")

import gate1                                                          # noqa: E402
import gate2                                                          # noqa: E402
import backend                                                        # noqa: E402

N_Q, N_LAYERS, HIDDEN = 200, 6, 24
SIGNAL_LAYER, SIGNAL_POS = 3, "subject"
RELATIONS = [("composer", 0.75), ("director", 0.45), ("spouse", 0.30),
             ("currency", 1.00)]

gate2.SHARD = 40
gate2.MIN_RELATION_N = 20
gate2.MIN_MINORITY_N = 4
gate2.N_BOOT = 300
rng = np.random.default_rng(7)

ok = True


def check(cond, label):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    ok = ok and bool(cond)


# ======================================================================================
# Fixtures
# ======================================================================================

def build_corpus():
    """A fake ConFiQA file and a matching pool/labels pair, joined by backend._make_qid."""
    raw, pool, labels = [], [], []
    for i in range(N_Q):
        rel, known_rate = RELATIONS[i % len(RELATIONS)]
        subj = f"Entity {i // 2}"                 # two questions share an entity
        q = f"Who is the {rel} of {subj}?"
        row = {"question": q, "orig_triple": f"('Q{i}', 'P1', 'Q9')",
               "cf_triple": f"('Q{i}', 'P1', 'Q8')",
               "orig_path_labeled": f"[('{subj}', '{rel}', 'X')]"}
        qid = backend._make_qid(row)
        raw.append(row)

        known = rng.random() < known_rate
        # Entropy carries SOME signal -- a baseline that is pure noise makes any probe
        # look good and would not test what Gate 2 is asking.
        ent = rng.normal(2.0 - 0.6 * known, 1.0)
        pool.append({"qid": qid, "subject_qid": f"Q{i // 2}", "question": q,
                     "relation": rel})
        labels.append({"qid": qid, "n_correct": 9 if known else 1,
                       "knowledge": "known" if known else "unknown",
                       "entropy": float(ent), "max_prob": float(rng.random())})

    # ~8% discards, to prove they are dropped rather than trained on
    for r in labels[:16]:
        r["knowledge"] = "discard"

    json.dump(raw, open(os.environ["GATE2_CONFIQA"], "w"))
    for name, rows in (("pool.jsonl", pool), ("labels.jsonl", labels)):
        with open(gate1._path(name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return {r["qid"]: r for r in labels}


def make_fake_backend(truth, fail_after=None):
    """hidden_states stand-in. Plants a linear signal at (SIGNAL_POS, SIGNAL_LAYER).

    Every other layer is noise, so a probe that scores well anywhere else would mean the
    nested selection is leaking rather than selecting.
    """
    calls = {"n": 0}
    w = rng.normal(size=HIDDEN)

    def fake(question, subject=None):
        calls["n"] += 1
        if fail_after is not None and calls["n"] > fail_after:
            raise RuntimeError("simulated runtime recycle")
        qid = _qid_of[question]
        known = truth[qid]["knowledge"] == "known"
        hf = rng.normal(size=(N_LAYERS, HIDDEN))
        hs = rng.normal(size=(N_LAYERS, HIDDEN))
        target = hf if SIGNAL_POS == "final" else hs
        target[SIGNAL_LAYER] += (1.4 if known else -1.4) * w
        # a handful of unresolved spans, to exercise the drop path in build_matrix
        resolved = calls["n"] % 37 != 0
        return {"h_final": hf.astype("float16"),
                "h_subject": hs.astype("float16") if resolved else None,
                "entropy": truth[qid]["entropy"], "max_prob": truth[qid]["max_prob"],
                "span_tier": "verbatim" if resolved else "unresolved",
                "n_tokens": 20}

    return fake, calls


# ======================================================================================

print(f"scratch: {TMP}\n")
print("FIXTURES")
truth = build_corpus()
_qid_of = {r["question"]: r["qid"] for r in gate1.read_jsonl("pool.jsonl")}
check(len(truth) == N_Q, f"built {N_Q} questions")

# ---- unit: subject span locator (pure, no model) --------------------------------------
print("\nUNIT -- _subject_char_span")
P = ("<|sys|>Shaft is not the subject here.<|user|>"
     "Question: Who composed the music for the film Shaft?")
Q = "Who composed the music for the film Shaft?"
for subj, want in (("Shaft", "verbatim"), ("Shaft (2019 film)", "paren-stripped"),
                   ("shaft", "casefold"), ("Nonexistent", "unresolved")):
    s, e, tier = backend._subject_char_span(P, Q, subj)
    check(tier == want, f"{subj!r} -> {tier}")
    if tier != "unresolved":
        check(P[s:e].lower() == subj.split(" (")[0].lower(), f"  span text {P[s:e]!r}")
# the system block also contains "helpful"; a subject that appears earlier must still
# resolve inside the question line
s, e, _ = backend._subject_char_span(P, Q, "Who")
check(s >= P.rfind(Q), "span is scoped to the question line, not the system block")

# ---- unit: DeLong against brute force -------------------------------------------------
print("\nUNIT -- DeLong")
yv = (rng.random(300) < 0.5).astype(int)
sa = yv + rng.normal(0, 1.0, 300)
sb = yv + rng.normal(0, 1.5, 300)
a, b, z, p = gate2.delong(yv, sa, sb)
check(abs(a - gate2._auc(yv, sa)) < 1e-9, f"auc_a matches sklearn ({a:.4f})")
check(abs(b - gate2._auc(yv, sb)) < 1e-9, f"auc_b matches sklearn ({b:.4f})")
check(0 <= p <= 1 and np.isfinite(z), f"z={z:.3f} p={p:.4f}")
_, _, _, p_same = gate2.delong(yv, sa, sa.copy())
check(np.isnan(p_same) or p_same > 0.99, "identical predictors -> no difference")

# ---- extract, with a crash partway ----------------------------------------------------
print("\nPHASE A -- extract, crash, resume")
fake, calls1 = make_fake_backend(truth, fail_after=95)
backend.hidden_states = fake
try:
    gate2.phase_extract()
    check(False, "simulated crash raised")
except RuntimeError:
    check(True, "simulated crash raised")

shards_before = len([k for k in range(50) if os.path.exists(gate2._shard_path(k))])
check(shards_before == 2, f"{shards_before} whole shards survived the crash "
                          f"(partial shard discarded)")

fake2, calls2 = make_fake_backend(truth)
backend.hidden_states = fake2
gate2.phase_extract()
expected = N_Q - shards_before * gate2.SHARD
check(calls2["n"] == expected,
      f"resume re-extracted {calls2['n']} questions, not {N_Q} (expected {expected})")

hf, hs, mask, meta = gate2.load_acts()
check(hf.shape == (N_Q, N_LAYERS, HIDDEN), f"h_final {hf.shape}")
check(hf.dtype == np.float16, "stored as float16")
check(len(meta) == N_Q and meta.qid.nunique() == N_Q, "one meta row per question")
check(0 < mask.sum() < N_Q, f"{int(mask.sum())}/{N_Q} subject spans resolved")
check(os.path.exists(gate1._path("extract_parity.json")), "wrote extract_parity.json")

# ---- train ----------------------------------------------------------------------------
print("\nPHASE B -- train")
gate2.phase_train()
folds = json.load(open(gate1._path("probe_folds.json")))
check(len(folds["folds"]) == gate2.N_FOLDS, f"{gate2.N_FOLDS} folds recorded")
check(folds["pooled_auc"]["probe"] > 0.75,
      f"probe recovers the planted signal (AUC {folds['pooled_auc']['probe']:.3f})")
check(folds["pooled_auc"]["entropy"] > 0.55,
      f"entropy baseline is not noise (AUC {folds['pooled_auc']['entropy']:.3f})")
picked = [f["position"] for f in folds["folds"]]
check(picked.count(SIGNAL_POS) >= gate2.N_FOLDS - 1,
      f"selection found position {SIGNAL_POS!r} in {picked.count(SIGNAL_POS)}/"
      f"{gate2.N_FOLDS} folds")
layers = [f["layer"] for f in folds["folds"] if f["position"] == SIGNAL_POS]
check(layers.count(SIGNAL_LAYER) >= len(layers) - 1,
      f"selection found layer {SIGNAL_LAYER} ({layers})")

# entity-disjointness of the folds is the claim the whole design rests on
X, y, groups, rel, ent, mp, qids = gate2.build_matrix()
fs = gate2.group_folds(groups)
leak = any(set(groups[tr]) & set(groups[te]) for tr, te in fs)
check(not leak, "no subject_qid appears in both train and test of a fold")
check(all(len(te) > 0 for _, te in fs), "every fold has a test slice")
check(len(y) < N_Q, f"discards and unresolved spans dropped ({len(y)} of {N_Q})")

# ---- tests ----------------------------------------------------------------------------
print("\nPHASE C -- tests")
gate2.phase_tests()
res = json.load(open(gate1._path("gate2_results.json")))
check("c1_within_relation" in res and len(res["c1_within_relation"]) >= 2,
      f"C1 covered {len(res['c1_within_relation'])} relations")
excl = [r["relation"] for r in res["c1_excluded"]]
check("currency" in excl, "single-class relation 'currency' excluded from C1, not scored")
check(len(res["c2_loro"]) >= 2, f"C2 ran on {len(res['c2_loro'])} relations")
check(len(res["c4_positions"]) == 2 * N_LAYERS, "C4 curve covers both positions x layers")
best = max(res["c4_positions"], key=lambda r: r["auc"])
check(best["position"] == SIGNAL_POS and best["layer"] == SIGNAL_LAYER,
      f"C4 peak at the planted ({best['position']}, layer {best['layer']})")
check(res["c3_spouse_excluded"]["means"], "C3 produced a spouse-excluded mean")

# ---- calibrate -------------------------------------------------------------------------
print("\nPHASE D -- calibrate")
gate2.phase_calibrate()
cal = json.load(open(gate1._path("calibration.json")))
check(0 <= cal["ece"] <= 1, f"ECE = {cal['ece']:.4f}")
check(abs(cal["auc_after_platt"] - folds["pooled_auc"]["probe"]) < 1e-6,
      "Platt is monotone -- AUC unchanged")

# ---- the decision rule, all three branches ---------------------------------------------
# The synthetic signal is strong enough that phase_tests always PROCEEDs, so the two STOP
# branches would never be executed by the pipeline run above.
print("\nUNIT -- _decide")
check(gate2._decide(0.01, 0.005, 0.20, 0.10).startswith("STOP"),
      "pooled below margin -> STOP")
check(gate2._decide(0.10, -0.01, 0.20, 0.10).startswith("STOP"),
      "pooled CI includes 0 -> STOP")
check(gate2._decide(0.10, 0.04, 0.005, -0.02).startswith("STOP"),
      "pooled passes but within-relation does not -> STOP (relation classifier)")
check(gate2._decide(0.10, 0.04, 0.05, 0.02).startswith("PROCEED"),
      "both pass -> PROCEED")

# ======================================================================================
print()
if ok:
    shutil.rmtree(TMP)
    print("selftest2: ALL PASS")
else:
    print(f"selftest2: FAILURES -- scratch kept at {TMP}")
sys.exit(0 if ok else 1)

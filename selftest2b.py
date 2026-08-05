"""
selftest2b.py -- exercises the whole Gate 2b pipeline with a fake backend.

Run after every change to gate2b.py or to backend.answer_states:

    python selftest2b.py

Covers extract (including a simulated crash and resume), Checkpoint A, train, C0-C4, and
the decision rule, on synthetic activations with a planted signal at a known (position,
layer). Also unit-tests self_consistency, the b1/b2 off-by-one, the b2-matched assembly,
and the circularity of n_correct.

The thresholds in gate2b are pre-registered for a 1793-question pool; this harness runs
~200 questions, so the relation floors and bootstrap counts are scaled down here. Nothing
else is patched -- the code paths under test are the real ones.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

TMP = tempfile.mkdtemp(prefix="gate2b-selftest-")
os.environ["GATE1_OUT"] = os.path.join(TMP, "outputs")
os.environ["GATE2_CONFIQA"] = os.path.join(TMP, "confiqa.json")

import gate1                                                          # noqa: E402
import gate2                                                          # noqa: E402
import gate2b                                                         # noqa: E402
import backend                                                        # noqa: E402

N_Q, N_LAYERS, HIDDEN = 200, 6, 24
SIGNAL_LAYER, SIGNAL_POS = 3, "last"

# Strong enough to be found, weak enough NOT to be perfectly separable. A perfectly
# separable fixture makes every alpha in the grid tie at inner AUC 1.0000, the tie-break
# takes the first entry (1e-5 in the inherited grid), and the resulting near-zero
# coefficients trip the pooling pathology -- which is real, but it belongs in the dedicated
# unit test below rather than swallowing every other assertion in the file.
SIGNAL = 0.30
RELATIONS = [("composer", 0.75), ("director", 0.45), ("spouse", 0.30),
             ("currency", 1.00)]

gate2b.SHARD = 40
gate2b.MIN_RELATION_N = gate2.MIN_RELATION_N = 20
gate2b.MIN_MINORITY_N = gate2.MIN_MINORITY_N = 4
gate2b.N_BOOT = gate2.N_BOOT = 300
rng = np.random.default_rng(11)

# ONE planted direction, drawn once. The crash-and-resume test builds two fake backends,
# and a direction drawn per backend would plant a different signal in the shards written
# before the crash than in the ones written after -- which no probe could then recover, and
# the failure would look like a bug in gate2b rather than in this harness.
W = rng.normal(size=HIDDEN)

ok = True


def check(cond, label):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    ok = ok and bool(cond)


# ======================================================================================
# Fixtures
# ======================================================================================

def build_corpus():
    """A pool/labels pair with gold_aliases, which Gate 2b needs and Gate 2 did not."""
    pool, labels = [], []
    for i in range(N_Q):
        rel, known_rate = RELATIONS[i % len(RELATIONS)]
        subj = f"Entity {i // 2}"                 # two questions share an entity
        qid = f"q{i:04d}"
        known = rng.random() < known_rate
        n_correct = int(rng.integers(8, 11) if known else rng.integers(0, 3))
        # Entropy carries SOME signal -- a baseline that is pure noise makes any probe
        # look good and would not test what Gate 2b is asking.
        ent = rng.normal(2.0 - 0.6 * known, 1.0)
        pool.append({"qid": qid, "subject_qid": f"Q{i // 2}", "relation": rel,
                     "question": f"Who is the {rel} of {subj}?",
                     "gold_aliases": [f"gold{i}"]})
        labels.append({"qid": qid, "n_correct": n_correct,
                       "knowledge": "known" if known else "unknown",
                       "entropy": float(ent), "max_prob": float(rng.random())})

    for r in labels[:16]:                          # ~8% discards, dropped not trained on
        r["knowledge"] = "discard"

    json.dump([], open(os.environ["GATE2_CONFIQA"], "w"))
    for name, rows in (("pool.jsonl", pool), ("labels.jsonl", labels)):
        with open(gate1._path(name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return {r["qid"]: r for r in labels}, {r["question"]: r for r in pool}


def fake_samples(question):
    """The ten Phase 2 samples, reproducing labels.jsonl's n_correct EXACTLY.

    Checkpoint A asserts that agreement, so a sampler that only approximated n_correct
    would make the assertion untestable here -- which is the one thing this harness must
    not do, because that assertion is what makes "the same ten strings" checkable on the
    real run.
    """
    r = _pool_of[question]
    nc = truth[r["qid"]]["n_correct"]
    gold = r["gold_aliases"][0]
    n_wrong = 10 - nc
    # A block of identical wrong answers plus unique ones. The block size is RANDOM, not a
    # function of n_correct: otherwise b4 would be a deterministic function of the label
    # and score a perfect AUC, which is the circularity b4 was redefined to avoid.
    same = int(rng.integers(1, n_wrong + 1)) if n_wrong else 0
    wrongs = ["wrongA"] * same + [f"wrong{j}" for j in range(n_wrong - same)]
    return [gold] * nc + wrongs


def make_fake_backend(fail_after=None):
    """answer_states stand-in. Plants a linear signal at (SIGNAL_POS, SIGNAL_LAYER).

    Every other layer and position is noise, so a probe that scores well anywhere else
    would mean the nested selection is leaking rather than selecting.
    """
    calls = {"n": 0}

    def fake(question, max_new_tokens=32):
        calls["n"] += 1
        if fail_after is not None and calls["n"] > fail_after:
            raise RuntimeError("simulated runtime recycle")
        r = _pool_of[question]
        lab = truth[r["qid"]]
        known = lab["knowledge"] == "known"

        # every 41st question produces no usable answer span, to exercise the drop path
        if calls["n"] % 41 == 0:
            return {"h_first": None, "h_last": None, "h_mean": None, "answer": "",
                    "n_answer_tokens": 0, "entropy": lab["entropy"],
                    "max_prob": lab["max_prob"], "first_token_is_argmax": False,
                    "b2_first": float("nan"), "b2_last": float("nan"),
                    "b2_mean": float("nan"), "b3_mean_logprob": float("nan"),
                    "n_tokens": 20}

        H = {p: rng.normal(size=(N_LAYERS, HIDDEN)) for p in gate2b.POSITIONS}
        H[SIGNAL_POS][SIGNAL_LAYER] += (SIGNAL if known else -SIGNAL) * W
        # greedy correctness tracks knowledge but not perfectly, so C0 gets two strata
        # that each contain both classes
        correct = known if rng.random() > 0.15 else not known
        return {
            "h_first": H["first"].astype("float16"),
            "h_last": H["last"].astype("float16"),
            "h_mean": H["mean"].astype("float16"),
            "answer": r["gold_aliases"][0] if correct else "wrongA",
            "n_answer_tokens": int(rng.integers(1, 6)),
            "entropy": lab["entropy"], "max_prob": lab["max_prob"],
            "first_token_is_argmax": True,
            # b2 is a DIFFERENT number from b1 -- one token later -- and carries a weaker
            # signal, as it plausibly would.
            "b2_first": float(lab["entropy"] + rng.normal(0.3, 0.8)),
            "b2_last": float(lab["entropy"] + rng.normal(0.3, 0.8)),
            "b2_mean": float(lab["entropy"] + rng.normal(0.3, 0.8)),
            "b3_mean_logprob": float(rng.normal(0.4 * known - 1.0, 0.9)),
            "n_tokens": 20,
        }

    return fake, calls


def write_gate2_shard(qids):
    """A Gate 2 activation shard, so C4's reference line has something to read.

    Written directly rather than by running gate2.phase_extract: the point here is that
    _reference_acts aligns Gate 2's rows to Gate 2b's by qid, not that Gate 2 works.
    """
    meta = [{"qid": q, "entropy": 0.0, "max_prob": 0.0, "span_tier": "verbatim",
             "n_tokens": 20} for q in qids]
    with open(gate2._shard_path(0) + ".tmp", "wb") as fh:
        np.savez(fh,
                 h_final=rng.normal(size=(len(qids), N_LAYERS, HIDDEN)).astype("float16"),
                 h_subject=rng.normal(size=(len(qids), N_LAYERS, HIDDEN)).astype("float16"),
                 mask=np.ones(len(qids), dtype=bool), meta=json.dumps(meta))
    os.replace(gate2._shard_path(0) + ".tmp", gate2._shard_path(0))


# ======================================================================================

print(f"scratch: {TMP}\n")
print("FIXTURES")
truth, _pool_of = build_corpus()
check(len(truth) == N_Q, f"built {N_Q} questions")

# ---- unit: self_consistency (pure) -----------------------------------------------------
print("\nUNIT -- self_consistency (b4)")
check(gate2b.self_consistency(["a"] * 10) == 1.0, "ten identical -> 1.0")
check(abs(gate2b.self_consistency([f"x{i}" for i in range(10)]) - 0.1) < 1e-12,
      "ten distinct -> 0.1")
check(gate2b.self_consistency(["a", "A.", "a!", "b", "c", "d", "e", "f", "g", "h"]) == 0.3,
      "normalisation collapses 'a', 'A.', 'a!' -- gate1.normalize, not raw strings")
check(gate2b.self_consistency(["Paris"] * 6 + ["Lyon"] * 4) == 0.6, "modal fraction")

# ---- unit: n_correct is the LABEL, not a baseline ---------------------------------------
# This is why b4 had to be regenerated. Demonstrated rather than asserted in prose.
print("\nUNIT -- n_correct circularity")
lab = [r for r in gate1.read_jsonl("labels.jsonl") if r["knowledge"] != "discard"]
yv = np.array([r["knowledge"] == "known" for r in lab]).astype(int)
nc = np.array([r["n_correct"] for r in lab], dtype=float)
check(gate2._auc(yv, nc) == 1.0,
      "AUC(n_correct) is exactly 1.0 -- it IS the label, so it cannot be a baseline")

# ---- extract, with a crash partway -------------------------------------------------------
print("\nPHASE A -- extract, crash, resume")
fake, calls1 = make_fake_backend(fail_after=95)
backend.answer_states = fake
backend.sample_closed_book = lambda q, n, seed: fake_samples(q)
try:
    gate2b.phase_extract()
    check(False, "simulated crash raised")
except RuntimeError:
    check(True, "simulated crash raised")

shards_before = len([k for k in range(50) if os.path.exists(gate2b._shard_path(k))])
check(shards_before == 2, f"{shards_before} whole shards survived the crash "
                          f"(partial shard discarded)")

fake2, calls2 = make_fake_backend()
backend.answer_states = fake2
gate2b.phase_extract()
expected = N_Q - shards_before * gate2b.SHARD
check(calls2["n"] == expected,
      f"resume re-extracted {calls2['n']} questions, not {N_Q} (expected {expected})")

H, mask, meta = gate2b.load_acts()
for p in gate2b.POSITIONS:
    check(H[p].shape == (N_Q, N_LAYERS, HIDDEN) and H[p].dtype == np.float16,
          f"h_{p} {H[p].shape} float16")
check(len(meta) == N_Q and meta.qid.nunique() == N_Q, "one meta row per question")
check(0 < mask.sum() < N_Q, f"{int(mask.sum())}/{N_Q} questions produced an answer span")
check((meta.n_answer_tokens[~mask] == 0).all(), "spanless rows recorded as m=0")
check(len(meta.samples.iloc[0]) == 10, "the ten samples are persisted this time")
check(os.path.exists(gate1._path(gate2b.PARITY_FILE)),
      f"wrote {gate2b.PARITY_FILE}")

par = json.load(open(gate1._path(gate2b.PARITY_FILE)))
check(par["max_d_entropy"] == 0.0, "prompt parity: entropy reproduces labels.jsonl exactly")
check(par["n_correct_mismatches"] == 0,
      "regenerated samples reproduce n_correct for every question")
check(par["argmax_rate"] == 1.0, "greedy first token == argmax on every span")
b1, b2f = (par["pooled_baseline_auc"]["b1_entropy_qfinal"],
           par["pooled_baseline_auc"]["b2_entropy_first"])
check(abs(b1 - b2f) > 1e-9, f"b1 ({b1:.4f}) and b2_first ({b2f:.4f}) are different numbers")
b4auc = par["pooled_baseline_auc"]["b4_self_consistency"]
check(0.5 < b4auc < 1.0, f"b4 carries signal without being the label (AUC {b4auc:.3f})")

# ---- unit: the pooling pathology ------------------------------------------------------
# Pooled out-of-fold AUC concatenates raw decision values from separately fitted probes.
# Under strong L2 the coefficients shrink but the unpenalised intercept does not, so folds
# that each separate perfectly can pool to chance. This is what C=1e-5 does on a perfectly
# separable fixture, and it is in the grid Gate 2b inherited -- so it is detected, not
# assumed absent.
print("\nUNIT -- pooling_diagnostic")
yp = np.tile([0, 1], 50)
folds_p = [(np.array([]), np.arange(i * 20, (i + 1) * 20)) for i in range(5)]
clean = np.concatenate([yp[te] + rng.normal(0, 0.1, 20) for _, te in folds_p])
check(not gate2b.pooling_diagnostic(yp, clean, folds_p)["diverged"],
      "comparable per-fold scales -> pooling is fine")
# same perfect within-fold separation, but each fold offset by more than its own range
drift = np.concatenate([yp[te] * 0.01 + i for i, (_, te) in enumerate(folds_p)])
dg = gate2b.pooling_diagnostic(yp, drift, folds_p)
# 1/5 of pairs are within-fold and rank perfectly; the other 4/5 are cross-fold and are
# ranked by fold index, which is uninformative -> 0.2 * 1.0 + 0.8 * 0.5 = 0.6.
check(dg["per_fold_mean"] == 1.0 and abs(dg["pooled_auc"] - 0.6) < 1e-9,
      f"per-fold {dg['per_fold_mean']:.4f} but pooled {dg['pooled_auc']:.4f}")
check(dg["diverged"], "the divergence is DETECTED rather than reported as a null")
check(gate2b._decide(0.10, 0.04, 0.05, 0.02, "b1", pooled_ok=False).startswith("NO VERDICT"),
      "a diverged pooling gives NO VERDICT, not a STOP")

# ---- train ---------------------------------------------------------------------------------
print("\nPHASE B -- train")
write_gate2_shard(list(meta.qid))
gate2b.phase_train()
folds = json.load(open(gate1._path(gate2b.FOLDS_FILE)))
check(len(folds["folds"]) == gate2b.N_FOLDS, f"{gate2b.N_FOLDS} folds recorded")
check(folds["probe_auc"] > 0.75,
      f"probe recovers the planted signal (AUC {folds['probe_auc']:.3f})")
pl = folds["pooling"]
check(set(pl) >= {"per_fold_mean", "pooled_auc", "offset_ratio", "diverged"},
      f"pooling diagnostic recorded (per-fold {pl['per_fold_mean']:.4f} vs pooled "
      f"{pl['pooled_auc']:.4f}, offset ratio {pl['offset_ratio']:.3f})")
check(pl["per_fold_mean"] > pl["pooled_auc"],
      "pooling COSTS the probe AUC even at healthy scales -- the direction of the bias")
check(tuple(folds["c_grid"]) == tuple(gate2.C_GRID_WIDE),
      "training used the INHERITED widened grid, not Gate 2's pre-registered one")
check(set(folds["baseline_auc"]) == set(gate2b.BASELINES), "all four baselines scored")
check(folds["best_baseline"] ==
      max(folds["baseline_auc"], key=lambda k: folds["baseline_auc"][k]),
      f"BEST baseline is the maximum ({folds['best_baseline']})")
picked = [f["position"] for f in folds["folds"]]
check(picked.count(SIGNAL_POS) >= gate2b.N_FOLDS - 1,
      f"selection found position {SIGNAL_POS!r} in {picked.count(SIGNAL_POS)}/"
      f"{gate2b.N_FOLDS} folds")
layers = [f["layer"] for f in folds["folds"] if f["position"] == SIGNAL_POS]
check(layers.count(SIGNAL_LAYER) >= len(layers) - 1,
      f"selection found layer {SIGNAL_LAYER} ({layers})")

# entity-disjointness of the folds is the claim the whole design rests on
X, y, groups, rel, baselines, qids, extras = gate2b.build_matrix()
fs = gate2.group_folds(groups, gate2b.N_FOLDS, gate2b.SEED)
check(not any(set(groups[tr]) & set(groups[te]) for tr, te in fs),
      "no subject_qid appears in both train and test of a fold")
check(len(y) < N_Q, f"discards and spanless rows dropped ({len(y)} of {N_Q})")
check(gate2b.REFERENCE_POSITION in extras,
      "Gate 2's activations aligned for the C4 reference line")
check(gate2b.REFERENCE_POSITION not in X,
      "the reference position is NOT in the probe's selection space")

# ---- unit: b2 tracks the position each fold selected -------------------------------------
print("\nUNIT -- b2_matched")
picks = folds["folds"]
bm = gate2b.b2_matched(baselines, picks, fs, len(y))
check(not np.isnan(bm).any(), "every row got a b2 value")
agree = all(np.allclose(bm[te], baselines[f"b2_{p['position']}"][te])
            for p, (_, te) in zip(picks, fs))
check(agree, "each fold's b2 comes from that fold's selected position")

# ---- fold checkpointing: resume must reproduce the run exactly ----------------------------
print("\nPHASE B -- fold checkpoint resume")
ck = gate1.read_jsonl(gate2.CKPT_FOLDS)
check(len(ck) == gate2b.N_FOLDS, f"{len(ck)} folds checkpointed")
check(gate2.CKPT_FOLDS == "probe2b_folds_ckpt.jsonl",
      "Gate 2b checkpoints to its own file, not Gate 2's")

gate2b.phase_train()                                   # second run, fully resumed
folds2 = json.load(open(gate1._path(gate2b.FOLDS_FILE)))
check(folds2["probe_auc"] == folds["probe_auc"],
      f"resumed run reproduces pooled AUC exactly ({folds2['probe_auc']:.4f})")
check([f["layer"] for f in folds2["folds"]] == [f["layer"] for f in folds["folds"]],
      "resumed run reproduces the per-fold selection")
check(len(gate1.read_jsonl(gate2.CKPT_FOLDS)) == gate2b.N_FOLDS,
      "resume refit nothing -- checkpoint did not grow")

# ---- tests ---------------------------------------------------------------------------------
print("\nPHASE C -- tests")
gate2b.phase_tests()
res = json.load(open(gate1._path(gate2b.RESULTS_FILE)))

strata = {r["stratum"]: r for r in res["c0_strata"]}
check(set(strata) == {"greedy correct", "greedy incorrect"}, "C0 reported both strata")
check(all(r["reportable"] for r in strata.values()),
      "both strata cleared the minority floor, so both carry an AUC")
check(sum(r["n"] for r in strata.values()) == len(y),
      "C0 strata partition the evaluation rows exactly")

check(len(res["c1_within_relation"]) >= 2,
      f"C1 covered {len(res['c1_within_relation'])} relations")
check("currency" in [r["relation"] for r in res["c1_excluded"]],
      "single-class relation 'currency' excluded from C1, not scored")
check(len(res["c2_loro"]) >= 2, f"C2 ran on {len(res['c2_loro'])} relations")
check(res["c3_spouse_excluded"]["means"], "C3 produced a spouse-excluded mean")

n_pos = len(gate2b.POSITIONS) + 1
check(len(res["c4_positions"]) == n_pos * N_LAYERS,
      f"C4 curve covers {n_pos} positions x {N_LAYERS} layers")
best = max(res["c4_positions"], key=lambda r: r["auc"])
check(best["position"] == SIGNAL_POS and best["layer"] == SIGNAL_LAYER,
      f"C4 peak at the planted ({best['position']}, layer {best['layer']})")
ref = [r for r in res["c4_positions"] if r["position"] == gate2b.REFERENCE_POSITION]
check(len(ref) == N_LAYERS and max(r["auc"] for r in ref) < 0.65,
      "Gate 2's reference line is present and is noise here, as planted")

ck2 = gate1.read_jsonl(gate2.CKPT_LORO)
check(len(ck2) == len(res["c2_loro"]), f"C2 checkpointed {len(ck2)} relations")
check(len(gate1.read_jsonl(gate2.CKPT_CURVE)) == n_pos * N_LAYERS,
      "C4 checkpointed every curve point")
gate2b.phase_tests()                                   # second run, C2 + C4 fully resumed
res2 = json.load(open(gate1._path(gate2b.RESULTS_FILE)))
check([r["auc"] for r in res2["c2_loro"]] == [r["auc"] for r in res["c2_loro"]],
      "resumed C2 reproduces every held-out AUC exactly")
check([r["auc"] for r in res2["c4_positions"]] == [r["auc"] for r in res["c4_positions"]],
      "resumed C4 reproduces the whole layer curve exactly")
check(len(gate1.read_jsonl(gate2.CKPT_LORO)) == len(ck2), "C2 resume refit nothing")

# ---- the decision rule, all four branches ---------------------------------------------------
print("\nUNIT -- _decide")
check(gate2b._decide(0.01, 0.005, 0.20, 0.10, "b1").startswith("STOP"),
      "pooled below margin -> STOP")
check(gate2b._decide(0.10, -0.01, 0.20, 0.10, "b1").startswith("STOP"),
      "pooled CI includes 0 -> STOP")
check(gate2b._decide(0.10, 0.04, 0.005, -0.02, "b1").startswith("STOP"),
      "pooled passes but within-relation does not -> STOP (C1 veto fires)")
check(gate2b._decide(0.10, 0.04, 0.05, 0.02, "b1").startswith("PROCEED"),
      "both pass -> PROCEED")

# ======================================================================================
print()
if ok:
    shutil.rmtree(TMP)
    print("selftest2b: ALL PASS")
else:
    print(f"selftest2b: FAILURES -- scratch kept at {TMP}")
sys.exit(0 if ok else 1)

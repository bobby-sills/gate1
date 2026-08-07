"""
object_leak.py -- does the Gate 2b probe exploit ANSWER-ENTITY overlap between folds?

Inside-Out (Gekhman et al., 2503.15299, sec 3.2) splits so there are "no subject AND object
overlaps between the training and test splits". Gate 2b groups folds by subject_qid only.
Nothing stops the same ANSWER -- "France", "English", "Catholicism" -- appearing as gold in
both train and test, and with low-cardinality relations that is the norm rather than the
exception.

If answer identity carries label information (some answers are simply attached to
well-known facts), a probe can learn "questions whose answer is France tend to be known"
and cash that in at test time without encoding anything about the model's knowledge.

Three measurements, cheapest first:
  1. how much object overlap the folds actually have
  2. AUC of an OBJECT-PRIOR predictor: score each test row by the known-rate of its answer
     computed on the training fold only. This is the leak, measured directly, with no
     activations involved at all.
  3. the probe's advantage over entropy on test rows whose answer was SEEN in training vs
     rows whose answer was UNSEEN. If the advantage lives in the seen rows, it is leakage.

READ-ONLY. Nothing is refitted and no artifact is written.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate1
import gate2

z = np.load(gate1._path("probe2b_oof.npz"), allow_pickle=False)
oof, y = z["oof"], z["y"]
groups, rel, qids = z["groups"].astype(str), z["rel"].astype(str), z["qids"].astype(str)
ent = z["b1_entropy_qfinal"]          # already sign-flipped when written

pool = {r["qid"]: r for r in gate1.read_jsonl("pool.jsonl")}
obj = np.array([gate1.normalize(pool[q]["gold_aliases"][0]) for q in qids])

folds = gate2.group_folds(groups, 5, gate1.SEED)

print("=" * 78)
print("OBJECT (ANSWER-ENTITY) LEAKAGE CHECK")
print("=" * 78)
print(f"  n={len(y)}   distinct answers={len(set(obj))}   "
      f"answers per question={len(set(obj))/len(y):.2f}")

print("\n  answer cardinality by relation (low = high leakage risk):")
for r in sorted(set(rel)):
    m = rel == r
    print(f"    {r:<26} n={int(m.sum()):<5} distinct answers={len(set(obj[m])):<5} "
          f"ratio={len(set(obj[m]))/m.sum():.2f}")

# ---- 1. how much overlap is there ---------------------------------------------------
print("\n" + "-" * 78)
print("1. OVERLAP: test rows whose answer also appears in the training fold")
print("-" * 78)
seen_mask = np.zeros(len(y), dtype=bool)
for f, (tr, te) in enumerate(folds):
    train_objs = set(obj[tr])
    s = np.array([o in train_objs for o in obj[te]])
    seen_mask[te] = s
    print(f"  fold {f}: {s.sum():4d} / {len(te):4d}  {s.mean():6.2%} of test answers seen "
          f"in train")
print(f"  overall: {seen_mask.sum()} / {len(y)}  {seen_mask.mean():.2%}")

# ---- 2. the leak measured directly ---------------------------------------------------
print("\n" + "-" * 78)
print("2. OBJECT-PRIOR PREDICTOR -- label information in the answer alone, no activations")
print("-" * 78)
prior = np.full(len(y), np.nan)
for tr, te in folds:
    rate, glob = {}, y[tr].mean()
    for o in set(obj[tr]):
        m = obj[tr] == o
        rate[o] = y[tr][m].mean()
    prior[te] = [rate.get(o, glob) for o in obj[te]]

print(f"  AUC(object prior)      {gate2._auc(y, prior):.4f}")
print(f"  AUC(probe)             {gate2._auc(y, oof):.4f}")
print(f"  AUC(entropy)           {gate2._auc(y, ent):.4f}")
print("\n  An object-prior AUC near 0.5 means answer identity carries no label information")
print("  and there is nothing to leak. Well above 0.5 means the leak is available.")

# how much of the probe is explained by it
print(f"\n  corr(probe, object prior)   = {np.corrcoef(oof, prior)[0,1]:+.4f}")
print(f"  corr(entropy, object prior) = {np.corrcoef(ent, prior)[0,1]:+.4f}")

# ---- 3. does the probe's advantage live in the leaky rows? ---------------------------
print("\n" + "-" * 78)
print("3. PROBE ADVANTAGE, SPLIT BY WHETHER THE ANSWER WAS SEEN IN TRAINING")
print("-" * 78)
print(f"  {'subset':<16}{'n':>6}{'known':>8}{'probe':>9}{'entropy':>9}{'diff':>9}")
for label, m in (("answer SEEN", seen_mask), ("answer UNSEEN", ~seen_mask),
                 ("all", np.ones(len(y), dtype=bool))):
    if m.sum() < 30 or len(set(y[m])) < 2:
        print(f"  {label:<16}{int(m.sum()):>6}   too few / single-class")
        continue
    ap, ae = gate2._auc(y[m], oof[m]), gate2._auc(y[m], ent[m])
    print(f"  {label:<16}{int(m.sum()):>6}{y[m].mean():>8.1%}{ap:>9.4f}{ae:>9.4f}"
          f"{ap-ae:>+9.4f}")

print("\n  If the probe's advantage is concentrated in the SEEN rows and vanishes in the")
print("  UNSEEN rows, the pooled +0.0296 is partly answer-identity leakage, not knowledge.")
print("  If the advantage is similar in both, the fold grouping was sufficient.")

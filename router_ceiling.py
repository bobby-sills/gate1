"""Ceiling on routing between the entropy arm and the probe arm.

If even a PERFECT router -- one told, per instance, which arm happens to be right --
cannot beat entropy by much, then no implementable router can, and version 3 is not
worth building. This is deliberately an oracle and is only ever an upper bound.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, gate1, gate3b
from gate1 import macro_faithfulness

cells = gate3b.attach_probe(gate3b.load_cells(), gate3b.load_probe())
gens = gate3b.load_gens()
sweep = pd.concat([pd.read_csv(gate1._path("sweep.csv")),
                   gate3b.sweep_probe(cells, gens)], ignore_index=True)

pts, cfg = {}, {}
for a in ("entropy", "probe", "oracle_two_sided"):
    cfg[a], _ = gate3b.select(sweep, a)
    pts[a] = gate3b.rows_at(sweep, a, cfg[a])

e = pts["entropy"].set_index(["qid", "cell"]).correct
p = pts["probe"].set_index(["qid", "cell"]).correct
idx = e.index.intersection(p.index)
e, p = e.loc[idx], p.loc[idx]

base = pd.DataFrame({"qid": [i[0] for i in idx], "cell": [i[1] for i in idx],
                     "e": e.values, "p": p.values})

print(f"instances: {len(base)}")
print(f"entropy  {macro_faithfulness(base.assign(correct=base.e)):.2f}")
print(f"probe    {macro_faithfulness(base.assign(correct=base.p)):.2f}")
print(f"oracle   {macro_faithfulness(pts['oracle_two_sided']):.2f}")
print()

# ---- CEILING 1: perfect per-instance router -------------------------------------
best = base.assign(correct=base.e | base.p)
print(f"PERFECT per-instance router   {macro_faithfulness(best):.2f}   "
      f"(+{macro_faithfulness(best) - macro_faithfulness(base.assign(correct=base.e)):.2f} over entropy)")

# ---- CEILING 2: perfect per-question router (one choice per qid, both contexts) ---
def per_question(df):
    out = []
    for q, g in df.groupby("qid"):
        out.append(g.assign(correct=g.e if g.e.sum() >= g.p.sum() else g.p))
    return pd.concat(out)
pq = per_question(base)
print(f"PERFECT per-question router   {macro_faithfulness(pq):.2f}   "
      f"(+{macro_faithfulness(pq) - macro_faithfulness(base.assign(correct=base.e)):.2f} over entropy)")

# ---- how often do the two arms actually differ in OUTCOME? -----------------------
print()
print(f"same outcome        {(base.e == base.p).mean() * 100:.1f}%")
print(f"entropy right only  {((base.e) & (~base.p)).sum()}")
print(f"probe right only    {((~base.e) & (base.p)).sum()}")
print(f"both wrong          {((~base.e) & (~base.p)).sum()}")

# ---- OBSERVABLE conflict state: do probe and entropy agree on the GATE? ----------
print()
print("=" * 70)
print("OBSERVABLE conflict signal (no labels): do the two gates agree?")
print("=" * 70)
r = cells.drop_duplicates("qid").set_index("qid")
ent_says = (r.entropy < cfg["entropy"]["thr"])
pr_says = (r.probe > cfg["probe"]["thr"])
agree = (ent_says == pr_says)
truth = (r.knowledge == "known")
base["agree"] = base.qid.map(agree)
for name, m in [("AGREE", base.agree), ("CONFLICT", ~base.agree)]:
    s = base[m]
    print(f"{name:9} n={len(s):4d}  entropy {macro_faithfulness(s.assign(correct=s.e)):6.2f}"
          f"   probe {macro_faithfulness(s.assign(correct=s.p)):6.2f}")
print()
print(f"gates agree on {agree.mean() * 100:.1f}% of questions; "
      f"when they agree they are right {(truth[agree] == ent_says[agree]).mean() * 100:.1f}% "
      f"of the time,")
print(f"when they conflict entropy is right "
      f"{(truth[~agree] == ent_says[~agree]).mean() * 100:.1f}% "
      f"and the probe {(truth[~agree] == pr_says[~agree]).mean() * 100:.1f}%")

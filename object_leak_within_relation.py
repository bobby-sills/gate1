"""Control: is the SEEN/UNSEEN gap just a relation effect in disguise?

The SEEN group is dominated by low-cardinality relations (sport 0.20 answers/question,
religion 0.30) and has a much higher known rate (70.2% vs 54.2%). Both could produce the
gap without any leakage. This repeats the split INSIDE each relation, which holds relation
and most of the known-rate difference fixed.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate1, gate2

z = np.load(gate1._path("probe2b_oof.npz"), allow_pickle=False)
oof, y = z["oof"], z["y"]
groups, rel, qids = z["groups"].astype(str), z["rel"].astype(str), z["qids"].astype(str)
ent = z["b1_entropy_qfinal"]
pool = {r["qid"]: r for r in gate1.read_jsonl("pool.jsonl")}
obj = np.array([gate1.normalize(pool[q]["gold_aliases"][0]) for q in qids])

folds = gate2.group_folds(groups, 5, gate1.SEED)
seen = np.zeros(len(y), dtype=bool)
for tr, te in folds:
    t = set(obj[tr]); seen[te] = [o in t for o in obj[te]]

print(f"  {'relation':<24}{'n_seen':>7}{'d_seen':>9}{'n_unseen':>9}{'d_unseen':>10}")
acc = {"seen": [], "unseen": []}
for r in sorted(set(rel)):
    row, out = rel == r, []
    for key, m in (("seen", row & seen), ("unseen", row & ~seen)):
        if m.sum() >= 40 and min(y[m].sum(), (1-y[m]).sum()) >= 8:
            d = gate2._auc(y[m], oof[m]) - gate2._auc(y[m], ent[m])
            out.append((int(m.sum()), d)); acc[key].append((int(m.sum()), d))
        else:
            out.append((int(m.sum()), None))
    (ns, ds), (nu, du) = out
    f = lambda v: f"{v:+.4f}" if v is not None else "    --"
    print(f"  {r:<24}{ns:>7}{f(ds):>9}{nu:>9}{f(du):>10}")

for k in ("seen", "unseen"):
    if acc[k]:
        n = sum(a for a, _ in acc[k]); w = sum(a*d for a, d in acc[k]) / n
        print(f"\n  n-weighted mean probe-entropy, {k:<7} = {w:+.4f}   "
              f"({len(acc[k])} relations, n={n})")

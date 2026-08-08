import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, gate1, gate3b
from gate1 import macro_faithfulness, SEED

m = gate3b.load_cells(False)
u = gate3b.load_cells(True)
qm, qu = set(m.qid), set(u.qid)
print("=" * 70)
print("HOW INDEPENDENT ARE THE TWO SAMPLES?")
print("=" * 70)
print(f"matched questions   {len(qm)}")
print(f"unmatched questions {len(qu)}")
print(f"shared              {len(qm & qu)}  ({100*len(qm & qu)/len(qu):.1f}% of unmatched)")
print(f"unmatched-only      {len(qu - qm)}")
print()

for tag, unm in [("MATCHED", False), ("UNMATCHED", True)]:
    cells = gate3b.attach_probe(gate3b.load_cells(unm), gate3b.load_probe())
    gens = gate3b.load_gens()
    arms_ok = [a for a in gate3b.NEW_ARMS
               if not (gate3b.units_needed(cells, arms={a}) - set(gens))]
    ref = pd.read_csv(gate1._path("sweep_unmatched.csv" if unm else "sweep.csv"))
    sweep = pd.concat([ref, gate3b.sweep_probe(cells, gens, arms=arms_ok)],
                      ignore_index=True)
    qids = np.array(sorted(cells.qid.unique()))
    perm = np.random.default_rng(SEED).permutation(len(qids))
    halves = [set(qids[perm[:len(qids)//2]]), set(qids[perm[len(qids)//2:]])]
    arms = [a for a in ("constant", "entropy", "max_prob", "conflict")
            if a in set(sweep.arm)]
    held, tuned = {}, {}
    for a in arms:
        parts = []
        for fit, test in [(halves[0], halves[1]), (halves[1], halves[0])]:
            c, _ = gate3b.select(sweep, a, on_qids=fit)
            parts.append(gate3b.rows_at(sweep, a, c, on_qids=test))
        held[a] = pd.concat(parts, ignore_index=True)
        c, _ = gate3b.select(sweep, a)
        tuned[a] = macro_faithfulness(gate3b.rows_at(sweep, a, c))

    def boot(x, y, n=20000):
        q = np.array(sorted(set(x.qid) & set(y.qid)))
        xi, yi = x.set_index("qid"), y.set_index("qid")
        rng = np.random.default_rng(SEED)
        d = np.empty(n)
        for i in range(n):
            dr = rng.choice(q, size=len(q), replace=True)
            d[i] = (macro_faithfulness(xi.loc[dr].reset_index())
                    - macro_faithfulness(yi.loc[dr].reset_index()))
        return macro_faithfulness(x) - macro_faithfulness(y), d

    print("=" * 70)
    print(f"{tag}   (arms scored: {', '.join(arms)})")
    print("=" * 70)
    print(f"  {'arm':<10}{'tuned':>8}{'held-out':>10}{'drop':>8}")
    for a in arms:
        print(f"  {a:<10}{tuned[a]:8.2f}{macro_faithfulness(held[a]):10.2f}"
              f"{tuned[a]-macro_faithfulness(held[a]):8.2f}")
    print()
    if "conflict" in held:
        for base in ("entropy", "max_prob", "constant"):
            pt, d = boot(held["conflict"], held[base])
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  held-out  conflict - {base:<9}{pt:+.3f}  "
                  f"[{lo:+.4f}, {hi:+.4f}]  P(<=0)={(d<=0).mean():.4f}")
    print()

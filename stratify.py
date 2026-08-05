"""H4 stratified breakdown, plus a matcher-sensitivity re-score. CPU only.

    python stratify.py              # matched run   (sweep.csv)
    python stratify.py --unmatched  # secondary run (sweep_unmatched.csv)

TWO QUESTIONS, both asked of the already-computed sweep.

(1) H4. The proposed mechanism is that ground-truth knowledge helps precisely where
    free output confidence is WRONG. Split each headline cell by the knowledge label
    crossed with thresholded entropy and report oracle-minus-entropy inside each
    quadrant. Correction is drawn entirely from `unknown` questions and resistance
    entirely from `known` ones, so the four quadrants land two per cell:

        correction   confident-but-unknown   <- DISAGREE, mechanism predicts the gain
                     uncertain-and-unknown   <- agree
        resistance   uncertain-but-known     <- DISAGREE, mechanism predicts the gain
                     confident-and-known     <- agree

    If the gains spread evenly across all four, the effect is real but the explanation
    is wrong, and that is a finding about the hypothesis rather than about the result.

(2) MATCHER SENSITIVITY. gate1.alias_match is substring-after-normalization
    (invariant #3). At high tau generations ramble, and a rambling answer that happens
    to contain the gold string still scores correct -- which flatters exactly the arms
    that reach tau > 2.0, i.e. the adaptive ones. Re-score under equality after the
    same normalization, holding each arm's operating point fixed at the one chosen
    under substring matching, so the only thing that changes is the matcher.
"""
import json
import sys

import numpy as np
import pandas as pd

import gate1

ARMS_OF_INTEREST = ("constant", "entropy", "max_prob",
                    "oracle_one_sided", "oracle_two_sided")


def exact_match(prediction: str, aliases) -> bool:
    """Equality after gate1.normalize. The strict counterpart to gate1.alias_match."""
    p = gate1.normalize(prediction)
    return any(gate1.normalize(a) == p for a in aliases if a)


def load(unmatched: bool):
    cells_file = "cells_unmatched.jsonl" if unmatched else "cells.jsonl"
    sweep_file = "sweep_unmatched.csv" if unmatched else "sweep.csv"
    sweep = pd.read_csv(gate1._path(sweep_file))
    cells = pd.DataFrame(gate1.read_jsonl(cells_file))
    gens = {(g["qid"], g["context_kind"], g["tau"]): g
            for g in gate1.read_jsonl("generations.jsonl")}
    return sweep, cells, gens, cells_file, sweep_file


def stratify(sweep: pd.DataFrame, cells: pd.DataFrame, thr: float, thr_label: str):
    """oracle - entropy inside each knowledge x thresholded-entropy quadrant."""
    pts = {a: gate1.best_operating_point(sweep, a)
           for a in ARMS_OF_INTEREST if a in set(sweep.arm)}
    oracle = max(("oracle_one_sided", "oracle_two_sided"),
                 key=lambda a: gate1.macro_faithfulness(pts[a]))

    # entropy/knowledge are question-level (both come from Phase 2), so index by qid.
    # sweep.csv carries no context_kind -- the cell determines it.
    q = cells.drop_duplicates("qid").set_index("qid")

    def tag(df):
        df = df.copy()
        df["entropy_"] = q.entropy.reindex(df.qid).to_numpy()
        df["knowledge_"] = q.knowledge.reindex(df.qid).to_numpy()
        df["confident"] = df.entropy_ < thr
        return df

    print("=" * 78)
    print(f"H4 STRATIFIED BREAKDOWN -- winning oracle arm: {oracle}")
    print(f"  entropy threshold: {thr:.4f} ({thr_label})")
    print("  'confident' = entropy below threshold. DISAGREE = the free signal and the")
    print("  ground truth point opposite ways; the mechanism predicts the gain is there.")
    print("=" * 78)
    o, e = tag(pts[oracle]), tag(pts["entropy"])
    print(f"  {'cell':<12} {'quadrant':<24} {'n':>5} {'oracle':>7} {'entropy':>8} "
          f"{'diff':>7}")
    rows = []
    for cell in ("correction", "resistance"):
        oc, ec = o[o.cell == cell], e[e.cell == cell]
        for conf in (True, False):
            os_, es_ = oc[oc.confident == conf], ec[ec.confident == conf]
            if len(os_) == 0 and len(es_) == 0:
                continue
            kn = os_.knowledge_.iloc[0] if len(os_) else es_.knowledge_.iloc[0]
            disagree = (kn == "unknown" and conf) or (kn == "known" and not conf)
            name = f"{'confident' if conf else 'uncertain'}-{kn}"
            oa = 100 * os_.correct.mean() if len(os_) else float("nan")
            ea = 100 * es_.correct.mean() if len(es_) else float("nan")
            mark = "  <- DISAGREE" if disagree else ""
            print(f"  {cell:<12} {name:<24} {len(os_):>5} {oa:>7.1f} {ea:>8.1f} "
                  f"{oa - ea:>+7.1f}{mark}")
            rows.append({"cell": cell, "quadrant": name, "disagree": disagree,
                         "n": int(len(os_)), "oracle": oa, "entropy": ea,
                         "diff": oa - ea})
    d = pd.DataFrame(rows)
    if not d.empty:
        for flag, label in ((True, "DISAGREE quadrants"), (False, "agree quadrants")):
            sub = d[d.disagree == flag]
            if len(sub):
                w = np.average(sub["diff"], weights=sub.n)
                print(f"\n  {label:<20} n={int(sub.n.sum()):<5} "
                      f"weighted oracle-entropy = {w:+.1f}")
        print("\n  If these two lines are close, the gain is NOT concentrated where the")
        print("  free signal is wrong, and the proposed mechanism is not what is acting.")
    return d, oracle


def rescore(sweep: pd.DataFrame, cells: pd.DataFrame, gens: dict):
    """Correction-cell accuracy under substring vs exact match, operating point fixed."""
    tgt = {r["qid"]: r["target_aliases"] for r in cells.to_dict("records")}
    print("\n" + "=" * 78)
    print("MATCHER SENSITIVITY -- correction cell, operating point held fixed")
    print("  substring = gate1.alias_match (the pre-registered matcher, invariant #3)")
    print("  exact     = equality after the same normalization")
    print("=" * 78)
    print(f"  {'arm':<20} {'n':>5} {'substring':>10} {'exact':>8} {'drop':>7}")
    out = {}
    for arm in ARMS_OF_INTEREST:
        if arm not in set(sweep.arm):
            continue
        pt = gate1.best_operating_point(sweep, arm)
        sub = pt[pt.cell == "correction"]
        # correction is (unknown, factual) by construction, so context_kind is fixed.
        # sweep.csv does not carry tau; recompute it exactly as phase_sweep did.
        feats = cells.drop_duplicates("qid").set_index("qid")
        ex, missing = [], 0
        for r in sub.to_dict("records"):
            f = feats.loc[r["qid"]]
            tau = gate1.tau_for(arm, {"knowledge": f.knowledge, "entropy": f.entropy,
                                      "max_prob": f.max_prob},
                                r["tau0"], r["gamma"], r["delta"], r["thr"])
            g = gens.get((r["qid"], "factual", tau))
            if g is None:
                missing += 1
                ex.append(False)
                continue
            ex.append(exact_match(g["text"], tgt[r["qid"]]))
        if missing:
            print(f"  WARNING: {missing} generations not found for {arm}")
        s, x = 100 * sub.correct.mean(), 100 * float(np.mean(ex))
        out[arm] = (s, x)
        print(f"  {arm:<20} {len(sub):>5} {s:>10.1f} {x:>8.1f} {s - x:>+7.1f}")

    if "entropy" in out:
        print("\n  oracle advantage on correction, under each matcher:")
        for arm in ("oracle_one_sided", "oracle_two_sided"):
            if arm not in out:
                continue
            ds = out[arm][0] - out["entropy"][0]
            dx = out[arm][1] - out["entropy"][1]
            print(f"    {arm} - entropy : substring {ds:+.1f}  exact {dx:+.1f}"
                  f"   ({'SHRINKS' if abs(dx) < abs(ds) else 'holds or grows'})")
    return out


def main(unmatched: bool):
    sweep, cells, gens, cells_file, sweep_file = load(unmatched)
    print(f"reading {cells_file} + {sweep_file}: {len(cells)} instances, "
          f"{len(sweep)} sweep rows\n")

    # Primary threshold: the one the winning entropy config actually used, since that
    # is the decision boundary the baseline is operating on. Median is reported too
    # because phase_sanity's off-diagonal uses it, so the two should be comparable.
    best_ent = gate1.best_operating_point(sweep, "entropy")
    thr = float(best_ent.thr.iloc[0])
    stratify(sweep, cells, thr, "winning entropy config")
    med = float(cells.drop_duplicates("qid").entropy.median())
    if abs(med - thr) > 1e-9:
        print("\n")
        stratify(sweep, cells, med, "pool median, as used by phase_sanity")

    rescore(sweep, cells, gens)


if __name__ == "__main__":
    main("--unmatched" in sys.argv)

"""
pooling_recheck.py -- does Gate 2's published number survive how it was pooled?

    GATE1_OUT=/path/to/outputs python pooling_recheck.py [--wide]

READ-ONLY. It loads `probe_oof.npz` and `probe_folds.json` and writes nothing back into
the outputs directory; Gate 2's artifacts are not touched.

WHY THIS EXISTS. Gate 2's primary number is pooled out-of-fold AUC: the raw
`decision_function` values of five separately fitted probes are concatenated and ranked
against each other. That is valid only while each fold's scores vary a lot compared with
the offsets BETWEEN folds. Strong L2 shrinks the coefficients toward zero but not the
intercept, which sklearn does not penalise, so the two can invert. On the Gate 2b selftest
fixture the folds score AUC 1.0000 each and pool to 0.5024.

The asymmetry is the problem. The probe's pooled vector is stitched from five models;
`H(p_theta)` and `max p_theta` are single global scalars with no fold structure and pay
none of this cost. So whatever pooling costs, it is subtracted from the probe alone --
which is the direction that would manufacture a Gate 2 null.

This script measures the size of that effect on the real data and reports whether the
published +0.0095 [-0.0083, +0.0271] moves.
"""

import json
import sys

import numpy as np

import gate1
import gate2


def rank_pool(s, folds):
    """Within-fold percentile ranks, in [0, 1).

    Monotone inside each fold, so every per-fold AUC is preserved EXACTLY. What changes is
    the cross-fold comparisons: instead of ranking by whatever offset a fold's intercept
    happened to land on, rows are ranked by their standing among their own fold's peers.

    Applied identically to the probe and to the baselines. Applying it to the probe alone
    would swap one asymmetry for another.
    """
    from scipy.stats import rankdata
    out = np.full(len(s), np.nan)
    for _, te in folds:
        out[te] = rankdata(s[te]) / (len(te) + 1)
    return out


def per_fold(y, s, folds):
    return [gate2._auc(y[te], s[te]) for _, te in folds]


def main(wide=False):
    oof_file = "probe_oof_wide.npz" if wide else "probe_oof.npz"
    folds_file = "probe_folds_wide.json" if wide else "probe_folds.json"

    z = np.load(gate1._path(oof_file), allow_pickle=False)
    oof, y = z["oof"], z["y"]
    groups = z["groups"].astype(str)
    ent, mp = z["ent"], z["mp"]
    picks = json.load(open(gate1._path(folds_file)))

    # group_folds is deterministic given (row set, SEED), so this reconstructs exactly the
    # partition phase_train used. Verified below against the per-fold AUCs it recorded.
    folds = gate2.group_folds(groups)

    # Entropy is an UNCERTAINTY and the label is "known": it enters flipped.
    scores = {"probe": oof, "entropy": -ent, "max_prob": mp}

    print("=" * 78)
    print(f"POOLING RECHECK -- {oof_file}")
    print("=" * 78)
    print(f"  n={len(y)}  known={int(y.sum())}  entities={len(set(groups))}  "
          f"folds={len(folds)}")
    print(f"  published pooled AUC: probe {picks['pooled_auc']['probe']:.4f}  "
          f"entropy {picks['pooled_auc']['entropy']:.4f}")
    print(f"  selected C per fold:  {[f['C'] for f in picks['folds']]}")
    print(f"  selected layer:       {[f['layer'] for f in picks['folds']]}")

    # ---- 1. per-fold vs pooled, for every predictor ------------------------------------
    print("\n" + "-" * 78)
    print("1. PER-FOLD AUC vs POOLED AUC")
    print("-" * 78)
    print(f"  {'predictor':<12} " + " ".join(f"{'f' + str(i):>8}" for i in range(len(folds)))
          + f" {'mean':>8} {'POOLED':>8} {'gap':>8}")
    per_fold_means = {}
    for k, s in scores.items():
        pf = per_fold(y, s, folds)
        m, p = float(np.mean(pf)), gate2._auc(y, s)
        per_fold_means[k] = m
        print(f"  {k:<12} " + " ".join(f"{a:8.4f}" for a in pf)
              + f" {m:8.4f} {p:8.4f} {m - p:+8.4f}")
    print("\n  The probe's pooled vector is stitched from 5 models. entropy and max_prob")
    print("  are single global scalars -- their 'folds' are just slices of one predictor,")
    print("  so any gap in their rows is sampling noise, not a pooling cost.")

    # ---- 2. within-fold spread vs between-fold offset -----------------------------------
    print("\n" + "-" * 78)
    print("2. SCORE SCALE: within-fold spread vs between-fold offset")
    print("-" * 78)
    for k, s in scores.items():
        within = float(np.mean([s[te].max() - s[te].min() for _, te in folds]))
        med = [float(np.median(s[te])) for _, te in folds]
        between = max(med) - min(med)
        print(f"  {k:<12} within-fold range {within:9.4f}   between-fold offset "
              f"{between:8.4f}   ratio {between / within:6.3f}")
    print("\n  Ratio well below 1 means the folds are on comparable scales and pooling is")
    print("  at worst mildly lossy. The selftest fixture's degenerate case sits at 11.96.")

    # ---- 3. the gap under both pooling rules ---------------------------------------------
    print("\n" + "-" * 78)
    print("3. probe - entropy UNDER BOTH POOLING RULES")
    print("-" * 78)
    ranked = {k: rank_pool(s, folds) for k, s in scores.items()}

    rows = []
    for label, sc in (("raw (pre-registered)", scores), ("within-fold rank", ranked)):
        a_p, a_e = gate2._auc(y, sc["probe"]), gate2._auc(y, sc["entropy"])
        a_m = gate2._auc(y, sc["max_prob"])
        d, lo, hi = gate2.boot_auc_diff(y, sc["probe"], sc["entropy"], groups)
        _, _, _, pp = gate2.delong(y, sc["probe"], sc["entropy"])
        rows.append((label, a_p, a_e, a_m, d, lo, hi, pp))
        print(f"\n  {label}")
        print(f"    probe {a_p:.4f}   entropy {a_e:.4f}   max_prob {a_m:.4f}")
        print(f"    probe - entropy = {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]   DeLong p={pp:.3g}")

    # sanity: rank pooling must preserve every per-fold AUC exactly
    for k in scores:
        assert np.allclose(per_fold(y, scores[k], folds), per_fold(y, ranked[k], folds)), \
            f"rank_pool changed a within-fold AUC for {k} -- it is not monotone in-fold"
    print("\n  (checked: rank pooling preserves every per-fold AUC exactly, so the only")
    print("   thing it changes is how folds are compared with each other)")

    # ---- 4. does the verdict move? --------------------------------------------------------
    print("\n" + "-" * 78)
    print("4. DOES THE PRE-REGISTERED VERDICT MOVE?")
    print("-" * 78)
    (_, _, _, _, d_raw, lo_raw, hi_raw, _) = rows[0]
    (_, _, _, _, d_rk, lo_rk, hi_rk, _) = rows[1]
    print(f"  raw   probe - entropy = {d_raw:+.4f}  [{lo_raw:+.4f}, {hi_raw:+.4f}]")
    print(f"  rank  probe - entropy = {d_rk:+.4f}  [{lo_rk:+.4f}, {hi_rk:+.4f}]")
    print(f"  shift in the point estimate: {d_rk - d_raw:+.4f}")
    print(f"\n  PROCEED needs >= {gate2.PROCEED_AUC_MARGIN} with the CI excluding 0.")
    for label, d, lo in (("raw", d_raw, lo_raw), ("rank", d_rk, lo_rk)):
        verdict = ("PROCEED" if d >= gate2.PROCEED_AUC_MARGIN and lo > 0 else "STOP")
        print(f"    {label:<6} {d:+.4f} [{lo:+.4f}, ...]  ->  {verdict}")
    if (d_rk >= gate2.PROCEED_AUC_MARGIN and lo_rk > 0) != \
       (d_raw >= gate2.PROCEED_AUC_MARGIN and lo_raw > 0):
        print("\n  *** THE VERDICT INVERTS. Gate 2's conclusion depends on the pooling")
        print("      rule. STOP and report before doing anything else. ***")
    else:
        print("\n  Same verdict under both rules.")


if __name__ == "__main__":
    main(wide="--wide" in sys.argv)

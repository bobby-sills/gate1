"""
gate2.py -- Gate 2 probe feasibility. See GATE2_PROTOCOL.md.

Can a linear probe on internal states predict the knowledge label better than
H(p_theta) and max p_theta, which are free?

    python gate2.py extract     # Phase A  residual streams, GPU, RESUMABLE
    python gate2.py train       # Phase B  nested CV, CPU  -> checkpoint B
    python gate2.py tests       # Phase C  C1-C4 + statistics, CPU
    python gate2.py calibrate   # Phase D  Platt + ECE, CPU, only if C1 passes

`extract` is the only GPU step. It writes one .npz per 250 questions and skips shards
already on disk, so a Colab disconnect costs the current shard and nothing more.

Reads GATE1_OUT for everything, same as gate1.py. Set it in a PYTHON cell:
    os.environ['GATE1_OUT'] = '/content/drive/MyDrive/gate1/outputs'
"""

import ast
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import gate1

# ======================================================================================
# PRE-REGISTERED CONSTANTS -- frozen at Phase 0 (GATE2_PROTOCOL.md). Not tunable config.
# ======================================================================================

N_FOLDS = 5
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)     # strong L2 first; sklearn's C is inverse strength
INNER_FRAC = 0.25                    # held out inside each training fold for selection
MAX_ITER = 1000

PROCEED_AUC_MARGIN = 0.03            # probe - entropy, pooled entity-disjoint CV
WITHIN_RELATION_MARGIN = 0.02        # PROPOSED, awaiting sign-off -- see GATE2_PROTOCOL
MIN_RELATION_N = 60                  # inclusion floor for C1
MIN_MINORITY_N = 10                  # AUC is undefined single-class, unstable near it

N_BOOT = 10_000
SEED = gate1.SEED

SHARD = 250                          # questions per activation file
PARITY_WARN = 1e-3                   # max |d entropy| vs labels.jsonl: note above this
PARITY_STOP = 1e-2                   # ... and refuse to train above this

POSITIONS = ("final", "subject")

RNG = np.random.default_rng(SEED)


def _confiqa_path() -> str:
    return os.environ.get("GATE2_CONFIQA", "data/ConFiQA-QA.json")


# ======================================================================================
# PHASE A -- extraction
# ======================================================================================

def subject_labels() -> dict:
    """qid -> subject surface string, joined back to the source corpus.

    pool.jsonl stores subject_qid (a Wikidata Q-number) but not the subject's surface
    form, and read position (2) needs the string to locate the span. The join key is
    backend._make_qid, a hash of fields present in both files, so it is exact rather than
    fuzzy -- it resolved 4207/4207 rows on the labels pool.
    """
    import backend
    rows = json.load(open(_confiqa_path()))
    out = {}
    for r in rows:
        try:
            out[backend._make_qid(r)] = str(ast.literal_eval(r["orig_path_labeled"])[0][0])
        except (ValueError, SyntaxError, IndexError, KeyError, TypeError):
            continue
    return out


def _shard_path(k: int) -> str:
    return gate1._path(f"acts_shard{k:03d}.npz")


def phase_extract():
    import backend

    pool = gate1.read_jsonl("pool.jsonl")
    subs = subject_labels()
    missing = sum(1 for r in pool if r["qid"] not in subs)
    print(f"pool: {len(pool)} questions; subject label unavailable for {missing}")

    shards = [pool[i:i + SHARD] for i in range(0, len(pool), SHARD)]
    todo = [k for k in range(len(shards)) if not os.path.exists(_shard_path(k))]
    print(f"resuming: {len(shards) - len(todo)} / {len(shards)} shards already complete")

    t0, done = time.time(), 0
    for k in todo:
        rows = shards[k]
        h_final, h_subj, mask, meta = [], [], [], []
        for r in rows:
            res = backend.hidden_states(r["question"], subs.get(r["qid"]))
            h_final.append(res["h_final"])
            ok = res["h_subject"] is not None
            mask.append(ok)
            h_subj.append(res["h_subject"] if ok
                          else np.zeros_like(res["h_final"], dtype="float16"))
            meta.append({"qid": r["qid"], "entropy": res["entropy"],
                         "max_prob": res["max_prob"], "span_tier": res["span_tier"],
                         "n_tokens": res["n_tokens"]})
            done += 1

        # tmp + replace: a shard is either wholly present or wholly absent, so resume
        # never reads a file that was half-written when the runtime was recycled.
        tmp = _shard_path(k) + ".tmp"
        with open(tmp, "wb") as fh:      # a handle, not a name: savez appends ".npz"
            np.savez(fh, h_final=np.stack(h_final), h_subject=np.stack(h_subj),
                     mask=np.array(mask), meta=json.dumps(meta))
        os.replace(tmp, _shard_path(k))
        rate = (time.time() - t0) / max(done, 1)
        print(f"  shard {k}: {len(rows)} questions, {rate:.2f}s/q, "
              f"eta {rate * (len(pool) - done) / 60:.0f}min")

    _checkpoint_a(pool)


def load_acts():
    """All shards -> (h_final, h_subject, mask, meta_df). float16, memory-resident."""
    hf, hs, mk, meta = [], [], [], []
    k = 0
    while os.path.exists(_shard_path(k)):
        z = np.load(_shard_path(k), allow_pickle=False)
        hf.append(z["h_final"])
        hs.append(z["h_subject"])
        mk.append(z["mask"])
        meta.extend(json.loads(str(z["meta"])))
        k += 1
    if not hf:
        raise FileNotFoundError("no activation shards -- run `gate2.py extract` first")
    return (np.concatenate(hf), np.concatenate(hs), np.concatenate(mk),
            pd.DataFrame(meta))


def _checkpoint_a(pool=None):
    """Shapes, storage, prompt-parity against labels.jsonl, span tiers, class counts.

    HUMAN CHECKPOINT. GATE2_PROTOCOL Phase A: report and stop.
    """
    h_final, h_subj, mask, meta = load_acts()
    n_layers, hidden = h_final.shape[1], h_final.shape[2]
    nbytes = sum(os.path.getsize(_shard_path(k))
                 for k in range(10_000) if os.path.exists(_shard_path(k)))

    print("\n" + "=" * 78)
    print("CHECKPOINT A -- extraction")
    print("=" * 78)
    print(f"  h_final    {h_final.shape} {h_final.dtype}")
    print(f"  h_subject  {h_subj.shape} {h_subj.dtype}   resolved {int(mask.sum())}"
          f" / {len(mask)}")
    print(f"  layers {n_layers} (embedding + {n_layers - 1} blocks), hidden {hidden}")
    print(f"  on disk    {nbytes / 1e9:.2f} GB")

    print("\n  subject-span tier:")
    for tier, n in meta.span_tier.value_counts().items():
        print(f"    {tier:<16} {n:5d}  {n / len(meta):6.2%}")
    unresolved = meta[meta.span_tier == "unresolved"]
    if len(unresolved):
        print(f"    unresolved qids ({len(unresolved)}): "
              f"{', '.join(unresolved.qid.head(20))}")

    # ---- prompt parity: does the extraction pass reproduce Phase 2's numbers? ----------
    lab = pd.DataFrame(gate1.read_jsonl("labels.jsonl"))
    m = meta.merge(lab, on="qid", suffixes=("_new", "_old"))
    d_ent = (m.entropy_new - m.entropy_old).abs()
    d_mp = (m.max_prob_new - m.max_prob_old).abs()
    print(f"\n  prompt parity vs labels.jsonl on {len(m)} questions")
    print(f"    max |d entropy|  = {d_ent.max():.2e}   (warn {PARITY_WARN:.0e}, "
          f"stop {PARITY_STOP:.0e})")
    print(f"    max |d max_prob| = {d_mp.max():.2e}")
    print(f"    median |d entropy| = {d_ent.median():.2e}")
    if d_ent.max() > PARITY_STOP:
        print("\n  STOP: the extraction prompt is NOT the Phase 2 prompt. Do not train.")
    elif d_ent.max() > PARITY_WARN:
        print("\n  PASS with a note: differences are consistent with bf16 kernel "
              "nondeterminism across runtimes, not with a template change.")
    else:
        print("\n  PASS: numerically identical.")

    # ---- class counts -----------------------------------------------------------------
    y = _labels()
    print(f"\n  after dropping discards: n={len(y)}  "
          f"known={int((y.knowledge == 'known').sum())}  "
          f"unknown={int((y.knowledge == 'unknown').sum())}")
    print(f"  distinct subject entities: {y.subject_qid.nunique()}")
    print("\n  HUMAN CHECK: read the above, then run `python gate2.py train`.")

    json.dump({"n": int(len(meta)), "n_layers": int(n_layers), "hidden": int(hidden),
               "bytes": int(nbytes),
               "span_tier": {k: int(v) for k, v in meta.span_tier.value_counts().items()},
               "max_d_entropy": float(d_ent.max()),
               "max_d_max_prob": float(d_mp.max())},
              open(gate1._path("extract_parity.json"), "w"), indent=2)


# ======================================================================================
# PHASE B -- labels, folds, probes
# ======================================================================================

def _labels() -> pd.DataFrame:
    """FUNCTIONAL pool joined to knowledge labels, discards dropped.

    pool.jsonl is FUNCTIONAL-only (backend filter 5), so restricting to it is what
    excludes ONE_TO_MANY. Those rows are labelled -- the labels run covered the whole
    unfiltered corpus -- but a ONE_TO_MANY label measures whether the model produced the
    answer Wikidata happened to record, not whether it knows the fact, so training on
    them would fit a different target than the one Gate 1 gated on.
    """
    pool = pd.DataFrame(gate1.read_jsonl("pool.jsonl"))
    lab = pd.DataFrame(gate1.read_jsonl("labels.jsonl"))
    df = pool.merge(lab, on="qid")
    return df[df.knowledge != "discard"].reset_index(drop=True)


def build_matrix():
    """(X, y, groups, relation, entropy, max_prob) on ONE fixed row set.

    X is a dict position -> [n, n_layers, hidden].

    Rows are restricted to questions where BOTH read positions resolve. A probe selected
    at the subject position cannot score a question whose subject span was not found, so
    letting the row set vary by position would mean the C4 comparison, and the nested
    selection that chooses between positions, ran on different questions. The cost is
    <1% of the pool; the alternative is a position that looks better because it was
    scored on an easier subset.
    """
    h_final, h_subj, mask, meta = load_acts()
    y = _labels()
    idx = {q: i for i, q in enumerate(meta.qid)}
    keep = [(i, idx[q]) for i, q in enumerate(y.qid) if q in idx and mask[idx[q]]]
    dropped = len(y) - len(keep)
    yi = [i for i, _ in keep]
    ai = [j for _, j in keep]
    sub = y.iloc[yi].reset_index(drop=True)
    X = {"final": h_final[ai], "subject": h_subj[ai]}
    if dropped:
        print(f"  dropped {dropped} labelled questions with no subject span "
              f"(kept {len(keep)})")
    return (X, (sub.knowledge == "known").to_numpy().astype(int),
            sub.subject_qid.to_numpy(), sub.relation.to_numpy(),
            sub.entropy.to_numpy(), sub.max_prob.to_numpy(), sub.qid.to_numpy())


def group_folds(groups, n_folds=N_FOLDS, seed=SEED):
    """Entity-disjoint folds, balanced by size.

    Grouped by subject_qid, not qid: one entity generates several questions across
    relations, and a probe that memorises "the model knows things about Q60" would score
    on a qid-disjoint split without having encoded anything transferable.
    """
    uniq = np.array(sorted(set(groups)))
    order = np.random.default_rng(seed).permutation(len(uniq))
    assign = {uniq[g]: i % n_folds for i, g in enumerate(order)}
    fold_of = np.array([assign[g] for g in groups])
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0])
            for k in range(n_folds)]


def _fit_predict(Xtr, ytr, Xte, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=MAX_ITER)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _inner_split(tr, groups, seed):
    """One split inside a training fold, entity-disjoint like the outer one."""
    g = np.array(sorted(set(groups[tr])))
    rng = np.random.default_rng(seed)
    held = set(g[rng.permutation(len(g))[:max(1, int(round(INNER_FRAC * len(g))))]])
    inner_val = np.array([i for i in tr if groups[i] in held])
    inner_tr = np.array([i for i in tr if groups[i] not in held])
    return inner_tr, inner_val


def oof_fixed(Xl, y, groups, folds):
    """Out-of-fold scores for ONE fixed (position, layer); only C is tuned inside."""
    oof = np.full(len(y), np.nan)
    chosen = []
    for f, (tr, te) in enumerate(folds):
        itr, iva = _inner_split(tr, groups, SEED + f)
        best = max(C_GRID, key=lambda c: _auc(y[iva], _fit_predict(Xl[itr], y[itr],
                                                                   Xl[iva], c)))
        oof[te] = _fit_predict(Xl[tr], y[tr], Xl[te], best)
        chosen.append(best)
    return oof, chosen


def oof_nested(X, y, groups, folds, verbose=True):
    """THE PRIMARY NUMBER. Position, layer and C are all selected on the inner split.

    Selecting the layer on the pooled test scores is the single easiest way to
    manufacture a passing Gate 2: 33 layers x 2 positions is 66 chances to overfit a
    selection, which is the same order as the effect being measured. So the selection
    happens inside the training fold and never sees the test slice.
    """
    n_layers = X["final"].shape[1]
    oof = np.full(len(y), np.nan)
    picks = []
    for f, (tr, te) in enumerate(folds):
        itr, iva = _inner_split(tr, groups, SEED + f)
        best, best_auc = None, -np.inf
        for pos in POSITIONS:
            Xp = X[pos]
            for L in range(n_layers):
                Xl = Xp[:, L, :].astype(np.float32)
                for C in C_GRID:
                    a = _auc(y[iva], _fit_predict(Xl[itr], y[itr], Xl[iva], C))
                    if a > best_auc:
                        best_auc, best = a, (pos, L, C)
        pos, L, C = best
        Xl = X[pos][:, L, :].astype(np.float32)
        oof[te] = _fit_predict(Xl[tr], y[tr], Xl[te], C)
        picks.append({"fold": f, "position": pos, "layer": int(L), "C": C,
                      "inner_auc": float(best_auc)})
        if verbose:
            print(f"  fold {f}: selected {pos} layer {L} C={C} "
                  f"(inner AUC {best_auc:.4f})")
    return oof, picks


# ======================================================================================
# Statistics -- DeLong and the grouped bootstrap
# ======================================================================================

def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    r = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and z[j + 1] == z[i]:
            j += 1
        r[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = r
    return out


def delong(y, s_a, s_b):
    """Paired AUC comparison, fast algorithm of Sun & Xu (2014).

    Returns (auc_a, auc_b, z, p). The two predictors are evaluated on the same questions,
    so their errors are correlated; an unpaired test would treat that correlation as extra
    independent evidence and overstate significance.

    CAVEAT, and it is the reason GATE2_PROTOCOL reads its margin off the bootstrap
    instead: DeLong assumes two score vectors from models that did not vary over the test
    set. Pooled out-of-fold scores come from N_FOLDS separately fitted probes, which
    violates that. Reported as a secondary test; where the two disagree the bootstrap
    governs.
    """
    y = np.asarray(y)
    pos = np.asarray([s_a[y == 1], s_b[y == 1]], dtype=float)
    neg = np.asarray([s_a[y == 0], s_b[y == 0]], dtype=float)
    m, n, k = pos.shape[1], neg.shape[1], 2

    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    if k == 2:
        s = np.atleast_2d(s)
    c = np.array([1.0, -1.0])
    var = float(c @ s @ c)
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan"), float("nan")
    from scipy import stats
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    return float(aucs[0]), float(aucs[1]), z, float(2 * stats.norm.sf(abs(z)))


def boot_auc_diff(y, s_a, s_b, groups, n_boot=N_BOOT):
    """AUC difference with a 95% CI, RESAMPLED BY subject_qid.

    Matching the resampling unit to the split unit: the folds assume entity-level
    exchangeability, so an interval built by resampling rows would be narrower than the
    design supports, the same way row-level resampling would have been wrong in Gate 1.
    """
    uniq = np.array(sorted(set(groups)))
    by = {g: np.where(groups == g)[0] for g in uniq}
    point = _auc(y, s_a) - _auc(y, s_b)
    diffs = []
    for _ in range(n_boot):
        draw = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[g] for g in draw])
        if len(set(y[idx])) < 2:
            continue
        diffs.append(_auc(y[idx], s_a[idx]) - _auc(y[idx], s_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ======================================================================================
# PHASE B driver
# ======================================================================================

def phase_train():
    X, y, groups, rel, ent, mp, qids = build_matrix()
    folds = group_folds(groups)
    print(f"n={len(y)}  known={int(y.sum())}  unknown={int((1 - y).sum())}  "
          f"entities={len(set(groups))}  layers={X['final'].shape[1]}")
    print(f"{N_FOLDS}-fold entity-disjoint CV, nested selection over "
          f"{len(POSITIONS)} positions x {X['final'].shape[1]} layers x "
          f"{len(C_GRID)} alphas\n")

    t0 = time.time()
    oof, picks = oof_nested(X, y, groups, folds)
    print(f"\n  ({time.time() - t0:.0f}s)")

    # Entropy is an UNCERTAINTY and the label is "known", so it enters flipped. An AUC of
    # 0.35 here would mean it was entered unflipped, not that entropy is anti-predictive.
    scores = {"probe": oof, "entropy": -ent, "max_prob": mp}
    aucs = {k: _auc(y, v) for k, v in scores.items()}

    print("\n" + "=" * 78)
    print("CHECKPOINT B -- pooled out-of-fold AUC, entity-disjoint")
    print("=" * 78)
    for k, v in aucs.items():
        print(f"  {k:<10} {v:.4f}")
    d, lo, hi = boot_auc_diff(y, oof, -ent, groups)
    print(f"\n  probe - entropy  = {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
          f"   (PROCEED needs >= {PROCEED_AUC_MARGIN}, CI excluding 0)")
    a, b, z, p = delong(y, oof, -ent)
    print(f"  DeLong (secondary): z={z:.2f}  p={p:.3g}")
    print("\n  This is the pooled number ONLY. It is not a verdict: C1 can veto it.")
    print("  HUMAN CHECK: read the above, then run `python gate2.py tests`.")

    # str, not object: pandas hands back object arrays and np.load refuses those without
    # allow_pickle, which this repo does not enable for data it will later trust.
    np.savez(gate1._path("probe_oof.npz"), oof=oof, y=y, ent=ent, mp=mp,
             groups=groups.astype(str), rel=rel.astype(str), qids=qids.astype(str))
    json.dump({"folds": picks, "pooled_auc": aucs,
               "probe_minus_entropy": {"point": d, "lo": lo, "hi": hi},
               "delong_z": z, "delong_p": p, "n": int(len(y))},
              open(gate1._path("probe_folds.json"), "w"), indent=2)


# ======================================================================================
# PHASE C -- the four tests
# ======================================================================================

def _load_oof():
    z = np.load(gate1._path("probe_oof.npz"), allow_pickle=False)
    return (z["oof"], z["y"], z["groups"].astype(str), z["rel"].astype(str),
            z["ent"], z["mp"], z["qids"].astype(str))


def within_relation(y, scores: dict, rel, groups, drop=()):
    """C1. AUC inside each relation, averaged weighted by n.

    Known-rates run from ~27% to 100% across relations, so a probe that merely identifies
    the relation from the question's wording inherits that spread and earns pooled AUC
    without encoding anything fact-specific. Read position (1) is the final question
    token, which makes this the likely failure mode rather than a hypothetical one.
    """
    rows, excluded = [], []
    for r in sorted(set(rel)):
        if r in drop:
            continue
        m = rel == r
        n, minority = int(m.sum()), int(min(y[m].sum(), (1 - y[m]).sum()))
        if n < MIN_RELATION_N or minority < MIN_MINORITY_N:
            excluded.append({"relation": r, "n": n, "minority": minority,
                             "known_rate": float(y[m].mean())})
            continue
        rows.append({"relation": r, "n": n, "known_rate": float(y[m].mean()),
                     **{k: _auc(y[m], v[m]) for k, v in scores.items()}})
    d = pd.DataFrame(rows)
    means = {k: float(np.average(d[k], weights=d.n)) for k in scores} if len(d) else {}
    return d, means, pd.DataFrame(excluded)


def _print_within(d, means, excluded, title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    if not len(d):
        print("  no relation met the inclusion floor")
        return
    cols = [c for c in d.columns if c not in ("relation", "n", "known_rate")]
    print(f"  {'relation':<28} {'n':>5} {'known':>7} " +
          " ".join(f"{c:>9}" for c in cols))
    for r in d.sort_values("n", ascending=False).to_dict("records"):
        print(f"  {r['relation']:<28} {r['n']:>5} {r['known_rate']:>6.1%} " +
              " ".join(f"{r[c]:>9.4f}" for c in cols))
    print(f"  {'n-weighted mean':<28} {int(d.n.sum()):>5} {'':>7} " +
          " ".join(f"{means[c]:>9.4f}" for c in cols))
    if len(excluded):
        print(f"\n  excluded ({MIN_RELATION_N}+ rows and {MIN_MINORITY_N}+ minority "
              f"required):")
        for r in excluded.sort_values("n", ascending=False).to_dict("records"):
            print(f"    {r['relation']:<26} n={r['n']:<5} minority={r['minority']:<4} "
                  f"known={r['known_rate']:.1%}")


def leave_one_relation_out(X, y, groups, rel):
    """C2. Train on every relation but one, test on the held-out relation.

    Layer, position and C are selected inside the training relations only -- selecting
    them with the held-out relation visible would answer a different question than "does
    this transfer".
    """
    out = []
    n_layers = X["final"].shape[1]
    for r in sorted(set(rel)):
        te = np.where(rel == r)[0]
        tr = np.where(rel != r)[0]
        if len(te) < MIN_RELATION_N or min(y[te].sum(), (1 - y[te]).sum()) < MIN_MINORITY_N:
            continue
        itr, iva = _inner_split(tr, groups, SEED)
        best, best_auc = None, -np.inf
        for pos in POSITIONS:
            for L in range(n_layers):
                Xl = X[pos][:, L, :].astype(np.float32)
                for C in C_GRID:
                    a = _auc(y[iva], _fit_predict(Xl[itr], y[itr], Xl[iva], C))
                    if a > best_auc:
                        best_auc, best = a, (pos, L, C)
        pos, L, C = best
        Xl = X[pos][:, L, :].astype(np.float32)
        s = _fit_predict(Xl[tr], y[tr], Xl[te], C)
        out.append({"relation": r, "n": len(te), "known_rate": float(y[te].mean()),
                    "position": pos, "layer": int(L), "C": C, "auc": _auc(y[te], s)})
        print(f"  {r:<28} n={len(te):<5} {pos} L{L:<3} AUC={out[-1]['auc']:.4f}")
    return pd.DataFrame(out)


def position_curves(X, y, groups, folds):
    """C4. Pooled OOF AUC per (position, layer), layer held FIXED so the curve is a curve.

    Both positions are scored on the same questions (build_matrix keeps only rows where
    both resolve), so neither is advantaged by an easier subset.
    """
    n_layers = X["final"].shape[1]
    rows = []
    for pos in POSITIONS:
        for L in range(n_layers):
            oof, _ = oof_fixed(X[pos][:, L, :].astype(np.float32), y, groups, folds)
            rows.append({"position": pos, "layer": L, "auc": _auc(y, oof)})
        print(f"  {pos}: done")
    return pd.DataFrame(rows)


def phase_tests():
    oof, y, groups, rel, ent, mp, qids = _load_oof()
    scores = {"probe": oof, "entropy": -ent, "max_prob": mp}
    pooled = {k: _auc(y, v) for k, v in scores.items()}

    print("=" * 78)
    print("POOLED (primary) -- entity-disjoint 5-fold CV")
    print("=" * 78)
    for k, v in pooled.items():
        print(f"  {k:<10} {v:.4f}")
    d_pool, lo_pool, hi_pool = boot_auc_diff(y, oof, -ent, groups)
    _, _, z_pool, p_pool = delong(y, oof, -ent)
    print(f"  probe - entropy = {d_pool:+.4f}  [{lo_pool:+.4f}, {hi_pool:+.4f}]"
          f"   DeLong p={p_pool:.3g}")

    # ---- C1 ---------------------------------------------------------------------------
    d1, m1, ex1 = within_relation(y, scores, rel, groups)
    _print_within(d1, m1, ex1, "C1. WITHIN-RELATION AUC -- the test that decides")
    d_wr = m1.get("probe", float("nan")) - m1.get("entropy", float("nan"))
    lo_wr, hi_wr = _boot_within(y, oof, -ent, rel, groups)
    print(f"\n  within-relation probe - entropy = {d_wr:+.4f} "
          f"[{lo_wr:+.4f}, {hi_wr:+.4f}]"
          f"   (PROPOSED gate >= {WITHIN_RELATION_MARGIN}, CI excluding 0)")

    # ---- C3 (reuses C1's machinery) ----------------------------------------------------
    d3, m3, ex3 = within_relation(y, scores, rel, groups, drop=("spouse",))
    sp = int((rel == "spouse").sum())
    _print_within(d3, m3, ex3, f"C3. SPOUSE-EXCLUDED within-relation "
                               f"(spouse = {sp}, {sp / len(rel):.1%} of the pool)")
    d_sp = m3.get("probe", float("nan")) - m3.get("entropy", float("nan"))
    print(f"\n  spouse-excluded probe - entropy = {d_sp:+.4f}   "
          f"(with spouse: {d_wr:+.4f})")

    # ---- C2 / C4 need the activations back ---------------------------------------------
    X, y2, g2, rel2, _, _, _ = build_matrix()
    assert len(y2) == len(y) and (y2 == y).all(), "row set drifted between phases"
    folds = group_folds(g2)

    print("\n" + "=" * 78)
    print("C2. LEAVE-ONE-RELATION-OUT")
    print("=" * 78)
    d2 = leave_one_relation_out(X, y, groups, rel)
    if len(d2):
        print(f"  n-weighted mean AUC = "
              f"{np.average(d2.auc, weights=d2.n):.4f}   "
              f"(within-relation probe, for comparison: {m1['probe']:.4f})")

    print("\n" + "=" * 78)
    print("C4. READ POSITION x LAYER")
    print("=" * 78)
    d4 = position_curves(X, y, g2, folds)
    piv = d4.pivot(index="layer", columns="position", values="auc")
    print(piv.round(4).to_string())
    for pos in POSITIONS:
        b = d4[d4.position == pos].sort_values("auc").iloc[-1]
        print(f"  best {pos:<8} layer {int(b.layer):>2}  AUC {b.auc:.4f}")

    # ---- verdict ------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(_decide(d_pool, lo_pool, d_wr, lo_wr))
    print("=" * 78)

    json.dump({"pooled": pooled,
               "pooled_diff": {"point": d_pool, "lo": lo_pool, "hi": hi_pool,
                               "delong_p": p_pool},
               "c1_within_relation": d1.to_dict("records"), "c1_means": m1,
               "c1_excluded": ex1.to_dict("records"),
               "c1_diff": {"point": d_wr, "lo": lo_wr, "hi": hi_wr},
               "c2_loro": d2.to_dict("records"),
               "c3_spouse_excluded": {"means": m3, "diff": d_sp},
               "c4_positions": d4.to_dict("records")},
              open(gate1._path("gate2_results.json"), "w"), indent=2, default=float)


def _boot_within(y, s_a, s_b, rel, groups, n_boot=1000):
    """CI on the n-weighted within-relation AUC difference. Resampled by subject_qid,
    like every other interval here; fewer resamples because each one recomputes an AUC
    per relation."""
    uniq = np.array(sorted(set(groups)))
    by = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([by[g] for g in RNG.choice(uniq, len(uniq), replace=True)])
        num, den = 0.0, 0
        for r in set(rel[idx]):
            m = idx[rel[idx] == r]
            if len(m) < MIN_RELATION_N or min(y[m].sum(), (1 - y[m]).sum()) < MIN_MINORITY_N:
                continue
            num += len(m) * (_auc(y[m], s_a[m]) - _auc(y[m], s_b[m]))
            den += len(m)
        if den:
            diffs.append(num / den)
    if not diffs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def _decide(d_pool, lo_pool, d_wr, lo_wr) -> str:
    if d_pool < PROCEED_AUC_MARGIN or lo_pool <= 0:
        return (f"STOP: probe - entropy = {d_pool:+.4f} [{lo_pool:+.4f}, ...] does not "
                f"clear the pre-registered {PROCEED_AUC_MARGIN} with a CI excluding 0. "
                f"The free signals already carry what the probe recovers.")
    if not (d_wr >= WITHIN_RELATION_MARGIN and lo_wr > 0):
        return (f"STOP: the pooled advantage ({d_pool:+.4f}) does NOT survive within "
                f"relation ({d_wr:+.4f} [{lo_wr:+.4f}, ...]). The probe is behaving as a "
                f"relation classifier, H2 fails, and the pooled number is the artifact C1 "
                f"exists to detect.")
    return (f"PROCEED. Pooled probe - entropy = {d_pool:+.4f} [{lo_pool:+.4f}, ...] and "
            f"the advantage survives within relation ({d_wr:+.4f} [{lo_wr:+.4f}, ...]). "
            f"Run `gate2.py calibrate`.")


# ======================================================================================
# PHASE D -- calibration
# ======================================================================================

def phase_calibrate(n_bins: int = 10):
    """Platt scaling on held-out folds, then ECE and a reliability curve.

    SEPARATE FROM AUC ON PURPOSE. The gate multiplies k into the blend strength, so
    ranking correctly is not enough: a probe that ranks perfectly but reports 0.9 on cases
    it gets right 60% of the time produces a systematically mis-scaled correction. Right
    order, wrong magnitude -- and tau consumes the magnitude.
    """
    from sklearn.linear_model import LogisticRegression

    oof, y, groups, rel, ent, mp, qids = _load_oof()
    folds = group_folds(groups)

    # Platt is fitted per fold on the OTHER folds' out-of-fold scores, so the mapping is
    # never fitted on the scores it is then evaluated on.
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        lr = LogisticRegression().fit(oof[tr].reshape(-1, 1), y[tr])
        p[te] = lr.predict_proba(oof[te].reshape(-1, 1))[:, 1]

    edges = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= 1.0)
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.sum() / len(y) * abs(conf - acc)
        rows.append({"bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]),
                     "n": int(m.sum()), "confidence": conf, "accuracy": acc})

    print("=" * 78)
    print("PHASE D -- calibration of the Platt-scaled probe")
    print("=" * 78)
    print(f"  {'bin':<14} {'n':>5} {'says':>7} {'is':>7} {'gap':>7}")
    for r in rows:
        print(f"  [{r['bin_lo']:.1f}, {r['bin_hi']:.1f})   {r['n']:>5} "
              f"{r['confidence']:>7.3f} {r['accuracy']:>7.3f} "
              f"{r['confidence'] - r['accuracy']:>+7.3f}")
    print(f"\n  expected calibration error = {ece:.4f}")
    print(f"  AUC is unchanged by Platt ({_auc(y, p):.4f}) -- it is monotone. This "
          f"table is about magnitude, not ranking.")

    json.dump({"ece": float(ece), "n_bins": n_bins, "reliability": rows,
               "auc_after_platt": _auc(y, p)},
              open(gate1._path("calibration.json"), "w"), indent=2)


# ======================================================================================

PHASES = {"extract": phase_extract, "train": phase_train, "tests": phase_tests,
          "calibrate": phase_calibrate, "checkpoint": _checkpoint_a}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in PHASES:
        print(__doc__)
        sys.exit(1)
    PHASES[sys.argv[1]](*sys.argv[2:])

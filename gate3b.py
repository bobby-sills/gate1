"""
gate3b.py -- close the decoder loop. Does a REAL probe buy real faithfulness?

Every gate so far measured the probe in AUC. The project's currency is macro
faithfulness. Gate 1 showed a PERFECT knowledge label buys +13.83 points over the
entropy-gated arm. Nobody has shown a real probe recovers a single point.

This file feeds Gate 2b's out-of-fold probe scores into Gate 1's decoder gate as if
they were the knowledge signal, and measures faithfulness on the same 1200 instances,
against the same cached generations, with the same statistics.

    python gate3b.py check       # coverage, cache hit rate, entropy reproduction
    python gate3b.py parity      # does THIS GPU reproduce generations.jsonl? A100.
    python gate3b.py decode      # the 518 units Gate 1 never needed -- GPU, RESUMABLE
    python gate3b.py score       # the primary comparison
    python gate3b.py splithalf   # honest deployment estimate (select A, measure B)
    python gate3b.py diag        # where the difference does and does not land

FRAMING (agreed with the human 2026-08-07, recorded in GATE3B_PLAN.md): this is run as
ENGINEERING, NOT AS A CLAIM. There is no pre-registered threshold. Gates 1-3a used
pre-registration because they made claims about the world; this asks whether an artifact
works. Gate 2b's STOP is not revised by anything here.

COST -- CORRECTED 2026-08-07, by `check`, before any probe number was computed.

GATE3B_PLAN.md claimed this needs no GPU: `gate1.tau_for` emits values from
{tau0 - delta, tau0 + gamma} for every threshold arm, so the probe arm's required_work
looked like a subset of the entropy arm's. **The set of tau VALUES is a subset. The set
of (qid, tau) PAIRS is not,** and the decoder is keyed on the pair.

A question whose entropy is below all three entropy thresholds never takes the +gamma
branch in the entropy arm, so that generation was never made. If the probe puts that same
question on the other side of its own threshold, it needs exactly the generation nobody
produced. 518 of 15600 units, on 74 of 600 questions.

Those 74 are not a random remainder. They are precisely the questions where the probe
disagrees with BOTH entropy AND the ground-truth knowledge label -- 37 low-entropy
questions the probe calls unknown, 37 high-entropy ones it calls known. They are the only
questions on which a probe can add anything Gate 1 did not already have, which is why
they are missing and why they cannot be dropped. `decode` fills them.

gate1.py is authorization-gated and its constants are pre-registered. This file imports
it and does not modify it.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

import gate1
from gate1 import (TAU_GRID, GAMMA_GRID, DELTA_GRID, TAU_MIN, TAU_MAX, SEED, N_BOOT,
                   alias_match, macro_faithfulness, read_jsonl, _path)

OOF_FILE = "probe2b_oof.npz"

# Gate 3b's extra generations go in their OWN file. Gate 1's generations.jsonl is a
# frozen artifact of a completed, reported gate; appending to it would make that gate's
# inputs depend on a later gate's choices. Both files are read; only this one is written.
GEN_FILE = "generations_gate3b.jsonl"

# The probe arm's grid is IDENTICAL in density to entropy's and max_prob's:
# 8 x 5 x 3 x 3 = 360 configs. gate1 invariant 5 -- equal tuning budgets are the whole
# reason "adaptive beats fixed" is believable here. Do not widen it for the probe.
N_EXPECTED_CFGS = len(TAU_GRID) * len(GAMMA_GRID) * len(DELTA_GRID) * 3


# ======================================================================================
# Loading
# ======================================================================================

def load_cells(unmatched: bool = False) -> pd.DataFrame:
    name = "cells_unmatched.jsonl" if unmatched else "cells.jsonl"
    rows = [r for r in read_jsonl(name) if not r.get("_meta")]
    return pd.DataFrame(rows)


def load_probe() -> dict:
    """qid -> out-of-fold probe score.

    OUT OF FOLD MATTERS. Every score comes from a probe that never saw that question in
    training. Using in-fold scores here would make the arm an oracle wearing a costume.
    """
    z = np.load(_path(OOF_FILE), allow_pickle=True)
    return dict(zip(z["qids"].tolist(), z["oof"].tolist()))


def load_gens() -> dict:
    """Gate 1's cached generations, plus Gate 3b's top-up. Keyed (qid, context_kind, tau).

    Overlap is asserted rather than silently resolved: if the same unit appears in both
    files with different text, something has changed underneath us and no number computed
    from it is trustworthy.
    """
    gens = {}
    for name in ("generations.jsonl", GEN_FILE):
        for g in read_jsonl(name):
            k = (g["qid"], g["context_kind"], g["tau"])
            if k in gens and gens[k]["text"] != g["text"]:
                raise ValueError(
                    f"{k} decoded differently in generations.jsonl and {GEN_FILE}: "
                    f"{gens[k]['text']!r} vs {g['text']!r}. greedy_decode is "
                    f"deterministic, so this means the model, prompt or backend changed "
                    f"between runs. Stop.")
            gens[k] = g
    return gens


def units_needed(cells: pd.DataFrame, arms=None) -> set:
    recs = cells.to_dict("records")
    cfgs = [c for c in build_new_configs(cells)
            if arms is None or c["arm"] in arms]
    return {(r["qid"], r["context_kind"], _tau(c, r)) for c in cfgs for r in recs}


def attach_probe(cells: pd.DataFrame, scores: dict) -> pd.DataFrame:
    missing = sorted(set(cells.qid) - set(scores))
    if missing:
        raise KeyError(
            f"{len(missing)} of {cells.qid.nunique()} cells questions have no probe "
            f"score, e.g. {missing[:5]}. This is a FINDING, not something to drop -- "
            f"probe2b_oof covers the FUNCTIONAL non-discard pool with a resolved answer "
            f"span, and any cells question outside it needs explaining. See "
            f"GATE3B_PLAN.md 'Checks to run first'.")
    out = cells.copy()
    out["probe"] = out.qid.map(scores)
    return out


# ======================================================================================
# The probe arm
# ======================================================================================

def tau_for_probe(row, tau0: float, gamma: float, delta: float, thr: float) -> float:
    """Same shape as gate1.tau_for's threshold arms.

    DIRECTION IS FIXED BY SEMANTICS, NOT SEARCHED. A high probe score means "the model
    knows this", so it takes the branch gate1's oracle takes for `knowledge == "known"`
    (tau0 - delta, pull back toward memory) and the branch max_prob takes when confident.
    Searching the sign as well would silently double the probe's tuning budget relative
    to entropy and break gate1 invariant 5.
    """
    t = tau0 - delta if row["probe"] > thr else tau0 + gamma
    return round(float(np.clip(t, TAU_MIN, TAU_MAX)), 3)


def build_probe_configs(cells: pd.DataFrame) -> list:
    """Thresholds at the 25/50/75 quantiles of the probe score, exactly as
    gate1.build_configs does for entropy and max_prob -- including the fact that they are
    tuned ON the evaluation data. gate1 invariant 6 says that is deliberate: it gives the
    baselines their best possible showing. Applying it to the probe too is what keeps the
    comparison paired rather than handing the probe a handicap the baselines do not have.
    """
    thr = list(np.quantile(cells.probe, [0.25, 0.5, 0.75]))
    cfgs = [{"arm": "probe", "tau0": t, "gamma": g, "delta": d, "thr": x}
            for t in TAU_GRID for g in GAMMA_GRID for d in DELTA_GRID for x in thr]
    assert len(cfgs) == N_EXPECTED_CFGS, (len(cfgs), N_EXPECTED_CFGS)
    return cfgs


# ======================================================================================
# The conflict arm -- agreement between the two gates as an uncertainty signal
# ======================================================================================
#
# ADDED 2026-08-07, AFTER seeing the probe arm's result. That ordering is a real hazard
# and is declared: proposing an arm once you have looked at the data is how a result gets
# manufactured. Two guards, both non-negotiable for this arm:
#
#   1. SPLIT-HALF IS THE HEADLINE, not the tuned number. `splithalf` selects on questions
#      this arm never scored and measures on the rest.
#   2. EQUAL TUNING BUDGET. See build_conflict_configs.
#
# The motivating measurement, from `diag` and the router-ceiling check:
#
#     gates AGREE     n=816   entropy 84.06   probe 84.86
#     gates CONFLICT  n=384   entropy 68.18   probe 72.94
#
# When the two gates disagree BOTH arms lose ~15 points. Conflict predicts difficulty,
# not which arm to trust -- on the disagreements entropy matches the knowledge label 55%
# of the time and the probe 45%, so routing to the "better" arm is a coin flip. What is
# NOT ruled out is treating conflict as its own state and hedging there.
#
# Why this is not capped by the +5.67 perfect-router ceiling: that ceiling bounds any
# choice BETWEEN the two arms' existing outputs. A three-state rule emits a tau neither
# arm visits, so it is not a selection among their outputs.


def tau_for_conflict(row, tau0: float, gamma: float, delta: float,
                     thr_p: float, thr_e: float) -> float:
    """Three states instead of two.

        both gates say known    -> tau0 - delta   (pull back toward memory)
        both gates say unknown  -> tau0 + gamma   (push toward the document)
        the gates disagree      -> tau0           (hedge; commit to neither)

    The hedge is tau0 exactly, not a separately tuned value. Tuning the hedge would add
    a fourth grid dimension and hand this arm a budget no other arm has. tau0 is the
    natural neutral point and is what the `constant` arm uses.
    """
    known_p = row["probe"] > thr_p
    known_e = row["entropy"] < thr_e
    if known_p != known_e:
        t = tau0
    elif known_p:
        t = tau0 - delta
    else:
        t = tau0 + gamma
    return round(float(np.clip(t, TAU_MIN, TAU_MAX)), 3)


def build_conflict_configs(cells: pd.DataFrame) -> list:
    """8 x 5 x 3 x 3 = 360 configs, the same as every other threshold arm.

    THE BUDGET PROBLEM. This arm reads two signals, so it naturally wants a probe
    threshold AND an entropy threshold -- 3 x 3 = 9 combinations against everyone else's
    3. That is a 3x tuning advantage and it would break gate1 invariant 5, which exists
    because unequal tuning is the standard way "adaptive beats fixed" turns out to be an
    artifact.

    So the two thresholds are PAIRED at the same quantile level: (q25, q25), (q50, q50),
    (q75, q75). Three combinations. Beyond fixing the budget this is the coupling that
    makes sense -- it holds both gates at the same strictness, so "they disagree" means
    the signals disagree rather than one gate simply being set looser than the other.

    `thr` in the emitted config is the QUANTILE LEVEL, so the groupby keys stay
    one-dimensional and `select`/`rows_at` work unchanged. The realised threshold values
    ride along as thr_p/thr_e and are not part of the key.
    """
    qs = [0.25, 0.5, 0.75]
    thr_p = list(np.quantile(cells.probe, qs))
    thr_e = list(np.quantile(cells.entropy, qs))
    cfgs = [{"arm": "conflict", "tau0": t, "gamma": g, "delta": d, "thr": q,
             "thr_p": p, "thr_e": e}
            for t in TAU_GRID for g in GAMMA_GRID for d in DELTA_GRID
            for q, p, e in zip(qs, thr_p, thr_e)]
    assert len(cfgs) == N_EXPECTED_CFGS, (len(cfgs), N_EXPECTED_CFGS)
    return cfgs


# arm name -> (config builder, tau function taking (row, **cfg-minus-arm/thr))
NEW_ARMS = ("probe", "conflict")


def _tau(cfg: dict, row) -> float:
    if cfg["arm"] == "probe":
        return tau_for_probe(row, cfg["tau0"], cfg["gamma"], cfg["delta"], cfg["thr"])
    return tau_for_conflict(row, cfg["tau0"], cfg["gamma"], cfg["delta"],
                            cfg["thr_p"], cfg["thr_e"])


def build_new_configs(cells: pd.DataFrame) -> list:
    return build_probe_configs(cells) + build_conflict_configs(cells)


def sweep_probe(cells: pd.DataFrame, gens: dict) -> pd.DataFrame:
    """Score every probe config against the CACHED generations. Pure CPU."""
    recs, misses = [], []
    for c in build_new_configs(cells):
        for r in cells.to_dict("records"):
            tau = _tau(c, r)
            g = gens.get((r["qid"], r["context_kind"], tau))
            if g is None:
                misses.append((r["qid"], r["context_kind"], tau))
                continue
            recs.append({"arm": c["arm"], "tau0": c["tau0"], "gamma": c["gamma"],
                         "delta": c["delta"], "thr": c["thr"],
                         "qid": r["qid"], "cell": r["cell"],
                         "correct": alias_match(g["text"], r["target_aliases"]),
                         "n_chars": g["n_chars"]})
    if misses:
        raise KeyError(
            f"{len(misses)} required generations are NOT cached, e.g. {misses[:3]}. "
            f"Run `python gate3b.py decode` first. Do NOT drop these rows: they sit on "
            f"the questions where the probe disagrees with both entropy and the "
            f"knowledge label, so dropping them removes exactly the population the "
            f"experiment is about. See the COST note at the top of this file.")
    return pd.DataFrame(recs)


# ======================================================================================
# Selection and comparison
# ======================================================================================

KEYS = ["tau0", "gamma", "delta", "thr"]


def select(sweep: pd.DataFrame, arm: str, on_qids=None) -> tuple:
    """Best operating point for `arm`, chosen on `on_qids` (all of them if None)."""
    sub = sweep[sweep.arm == arm]
    pick = sub if on_qids is None else sub[sub.qid.isin(on_qids)]
    scores = pick.groupby(KEYS).apply(macro_faithfulness, include_groups=False)
    best = scores.idxmax()
    return dict(zip(KEYS, best)), float(scores.max())


def rows_at(sweep: pd.DataFrame, arm: str, cfg: dict, on_qids=None) -> pd.DataFrame:
    sub = sweep[sweep.arm == arm]
    mask = np.logical_and.reduce([sub[k] == v for k, v in cfg.items()])
    out = sub[mask]
    return out if on_qids is None else out[out.qid.isin(on_qids)]


def compare(a: pd.DataFrame, b: pd.DataFrame, seed: int = SEED) -> tuple:
    """gate1.paired_bootstrap -- resampled BY QID (gate1 invariant 1).

    gate1.RNG is module-level and consumed in call order, so results would depend on how
    many bootstraps ran before this one. Reseed so each comparison is independently
    reproducible. This mutates a module attribute; it does not edit gate1.py.
    """
    gate1.RNG = np.random.default_rng(seed)
    return gate1.paired_bootstrap(a, b, n_boot=N_BOOT)


# ======================================================================================
# parity -- does THIS runtime reproduce the generations we are about to sit beside?
# ======================================================================================

N_PARITY = 50


def phase_parity(unmatched: bool = False):
    """Re-decode units that are ALREADY cached and compare byte-for-byte. Run before
    `decode`, on the runtime `decode` will use.

    WHY THIS EXISTS. Gate 3b decodes 518 new units and then scores them alongside Gate
    1's cached generations. Gate 2b established that bf16 kernel selection varies with
    GPU architecture (GATE2B_PROTOCOL *Hardware*: labels.jsonl reproduces 1793/1793 on an
    L4 and 0/200 on an A100). If the new units come off a different architecture than the
    cached ones, the probe arm's generations come from a subtly different decoder than
    the baselines' -- and the difference lands entirely on the 74 questions where the
    probe disagrees with both entropy and the label. That is a confound aligned exactly
    with the effect being measured, so it is checked rather than assumed.

    NOTE THE REQUIRED HARDWARE IS NOT THE SAME AS GATE 3a's. Gate 1's phases ran in
    separate Colab sessions and landed on different GPUs: `labels` on an L4, `decode` on
    an A100 (human, 2026-08-07). Gate 3a must match labels.jsonl and therefore wants an
    L4; Gate 3b must match generations.jsonl and therefore wants an A100. Both are
    correct. This check is what settles it empirically either way.
    """
    cells = attach_probe(load_cells(unmatched), load_probe())
    gens = load_gens()
    ctx = {(r["qid"], r["context_kind"]): (r["question"], r["context"])
           for r in cells.to_dict("records")}

    # Sample from the units the probe arm actually reads, stratified over tau so the
    # extremes -- where the missing units live -- are represented.
    pool = sorted(units_needed(cells) & set(gens))
    by_tau = {}
    for u in pool:
        by_tau.setdefault(u[2], []).append(u)
    rng = np.random.default_rng(SEED)
    picks, taus = [], sorted(by_tau)
    for i in range(N_PARITY):
        bucket = by_tau[taus[i % len(taus)]]
        picks.append(bucket[int(rng.integers(len(bucket)))])
    picks = sorted(set(picks))

    print(f"re-decoding {len(picks)} already-cached units across "
          f"{len(taus)} distinct tau values")
    t0, rows, exact = time.time(), [], 0
    for n, (qid, kind, tau) in enumerate(picks, 1):
        q, c = ctx[(qid, kind)]
        got = gate1.greedy_decode(q, c, tau)
        want = gens[(qid, kind, tau)]["text"]
        ok = got == want
        exact += ok
        rows.append({"qid": qid, "context_kind": kind, "tau": tau,
                     "cached": want, "redecoded": got, "exact": ok})
        if not ok:
            print(f"  MISMATCH tau={tau} {qid} {kind}\n"
                  f"    cached    {want!r}\n    redecoded {got!r}")
        if n % 10 == 0:
            print(f"  {n}/{len(picks)}  {exact} exact  "
                  f"{(time.time() - t0) / n:.2f}s/unit")

    rate = (time.time() - t0) / len(picks)
    print()
    print("=" * 78)
    print(f"EXACT REPRODUCTION  {exact}/{len(picks)}")
    print("=" * 78)
    print(f"  {rate:.2f}s/unit -> 518 units projected at {rate * 518 / 60:.1f}min")
    with open(_path("gate3b_parity.json"), "w") as f:
        json.dump({"n": len(picks), "exact": exact, "seconds_per_unit": rate,
                   "rows": rows}, f, indent=2)
    if exact == len(picks):
        print("  -> this runtime reproduces generations.jsonl. `decode` is safe here.")
    else:
        print("  -> STOP. This runtime does NOT reproduce generations.jsonl, so units")
        print("     decoded here cannot be scored beside it. Gate 1's `decode` ran on an")
        print("     A100; get one, or the 518 units are not comparable to the cache.")
        print("     Do not proceed on the grounds that the mismatch is small.")


# ======================================================================================
# decode -- the only GPU step, and it is small
# ======================================================================================

def phase_decode(unmatched: bool = False):
    """Decode the units the probe arm needs and Gate 1 never produced. RESUMABLE.

    Uses gate1.greedy_decode unchanged, so these generations are produced by exactly the
    same decoder that produced generations.jsonl. Re-run after a disconnect; it prints
    how many units were already complete.
    """
    cells = attach_probe(load_cells(unmatched), load_probe())
    gens = load_gens()
    todo = sorted(units_needed(cells) - set(gens))
    print(f"new arms need {len(units_needed(cells))} units; "
          f"{len(gens)} cached; resuming with {len(todo)} to decode")
    if not todo:
        print("nothing to do.")
        return

    ctx = {(r["qid"], r["context_kind"]): (r["question"], r["context"])
           for r in cells.to_dict("records")}
    app = gate1.Appender(GEN_FILE, ("qid", "context_kind", "tau"))
    t0 = time.time()
    for n, (qid, kind, tau) in enumerate(todo, 1):
        q, c = ctx[(qid, kind)]
        text = gate1.greedy_decode(q, c, tau)
        app.write({"qid": qid, "context_kind": kind, "tau": tau,
                   "text": text, "n_chars": len(text)})
        if n % 25 == 0:
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)}  {rate:.2f}s/unit  "
                  f"eta {rate * (len(todo) - n) / 60:.0f}min")
    app.close()
    print(f"done in {(time.time() - t0) / 60:.1f}min -> {_path(GEN_FILE)}")


# ======================================================================================
# check -- prove the three assumptions before spending anything on them
# ======================================================================================

def phase_check(unmatched: bool = False):
    cells = attach_probe(load_cells(unmatched), load_probe())
    gens = load_gens()

    print("=" * 78)
    print("1. COVERAGE")
    print("=" * 78)
    print(f"  cells instances          {len(cells)}")
    print(f"  cells questions          {cells.qid.nunique()}")
    print(f"  with a probe score       {cells.probe.notna().sum()}  "
          f"(attach_probe raises otherwise)")
    print(f"  probe score range        [{cells.probe.min():.3f}, {cells.probe.max():.3f}]")

    print()
    print("=" * 78)
    print("2. CACHE HIT RATE -- is the probe arm's work really a subset of entropy's?")
    print("=" * 78)
    for arm in NEW_ARMS:
        n = units_needed(cells, arms={arm})
        m = n - set(gens)
        print(f"  {arm:<10} {len(build_probe_configs(cells))} configs   "
              f"{len(n):5d} units   {len(m):4d} missing")
    need = units_needed(cells)
    miss = need - set(gens)
    print(f"  {'union':<10} {2 * N_EXPECTED_CFGS} configs   "
          f"{len(need):5d} units   {len(miss):4d} missing")
    print(f"  (every arm gets {N_EXPECTED_CFGS} configs -- equal budget, invariant 5)")
    if miss:
        mq = {k[0] for k in miss}
        d = cells.drop_duplicates("qid").set_index("qid")
        ent_q = np.quantile(cells.entropy, [0.25, 0.5, 0.75])
        aff = d.loc[sorted(mq)]
        lo = int((aff.entropy < ent_q[0]).sum())
        hi = int((aff.entropy > ent_q[2]).sum())
        print(f"  on {len(mq)} of {cells.qid.nunique()} questions")
        print(f"    entropy below all 3 thresholds: {lo}   above all 3: {hi}")
        print(f"    knowledge: {aff.knowledge.value_counts().to_dict()}")
        print("  -> EXPECTED. These are the questions where the probe disagrees with")
        print("     both entropy and the knowledge label, so no arm ever asked for the")
        print("     branch the probe wants. Run `gate3b.py decode` (GPU, resumable).")
    else:
        print("  -> complete. `score` can run on CPU.")

    print()
    print("=" * 78)
    print("3. SCORING REPRODUCTION -- does this file score the way gate1 scored?")
    print("=" * 78)
    ref_path = _path("sweep_unmatched.csv" if unmatched else "sweep.csv")
    if not os.path.exists(ref_path):
        print(f"  {ref_path} not present -- skipped.")
        print("  Fetch it to enable this check; it is the only thing tying this file's")
        print("  numbers to gate1's published ones.")
        return
    ref = pd.read_csv(ref_path)
    ref_ent = ref[ref.arm == "entropy"]
    mine = _sweep_gate1_arm(cells, gens, "entropy")
    # sweep.csv has no context_kind column -- each (config, qid) is TWO rows, one per
    # context. `cell` distinguishes them. Merging without it is a cartesian blowup.
    #
    # thr must be rounded before it is used as a merge key. pandas.to_csv truncates
    # float64 repr, so the middle entropy threshold round-trips as 0.948013722896576
    # against an in-memory 0.9480137228965759 and one third of the rows silently fail to
    # join. This is a key-precision artifact, not a scoring difference.
    ref_ent, mine = ref_ent.copy(), mine.copy()
    ref_ent["thr"] = ref_ent.thr.round(12)
    mine["thr"] = mine.thr.round(12)
    j = ref_ent.merge(mine, on=KEYS + ["qid", "cell"], suffixes=("_ref", "_mine"))
    agree = (j.correct_ref == j.correct_mine).mean()
    print(f"  entropy rows compared    {len(j)}  of {len(ref_ent)} in sweep.csv")
    print(f"  agreement on `correct`   {agree:.6f}")
    if len(j) != len(ref_ent) or agree < 1.0:
        print("  -> MISMATCH. This file's scoring differs from gate1's. Everything")
        print("     downstream is suspect. Do not report a probe number.")
    else:
        print("  -> exact. The probe arm is scored the same way the baselines were.")


def _sweep_gate1_arm(cells: pd.DataFrame, gens: dict, arm: str) -> pd.DataFrame:
    """Re-score one of gate1's own arms, for the reproduction check only."""
    cfgs = [c for c in gate1.build_configs(cells) if c["arm"] == arm]
    recs = []
    for c in cfgs:
        for r in cells.to_dict("records"):
            tau = gate1.tau_for(arm, r, c["tau0"], c["gamma"], c["delta"], c["thr"])
            g = gens[(r["qid"], r["context_kind"], tau)]
            recs.append({**c, "qid": r["qid"], "cell": r["cell"],
                         "correct": alias_match(g["text"], r["target_aliases"])})
    return pd.DataFrame(recs)


# ======================================================================================
# score -- the primary comparison
# ======================================================================================

def phase_score(unmatched: bool = False):
    cells = attach_probe(load_cells(unmatched), load_probe())
    gens = load_gens()

    probe_sweep = sweep_probe(cells, gens)
    ref = pd.read_csv(_path("sweep_unmatched.csv" if unmatched else "sweep.csv"))
    sweep = pd.concat([ref, probe_sweep], ignore_index=True)

    arms = [a for a in ("constant", "entropy", "max_prob", "probe", "conflict",
                        "oracle_one_sided", "oracle_two_sided") if a in set(sweep.arm)]
    pts, cfg = {}, {}
    for a in arms:
        cfg[a], _ = select(sweep, a)
        pts[a] = rows_at(sweep, a, cfg[a])

    print("=" * 78)
    print("GATE 3b -- macro faithfulness at each arm's best operating point")
    print("=" * 78)
    print(f"{'arm':<20}{'macro':>9}{'correction':>12}{'resistance':>12}   config")
    for a in arms:
        d = pts[a]
        c = 100 * d[d.cell == "correction"].correct.mean()
        r = 100 * d[d.cell == "resistance"].correct.mean()
        cs = " ".join(f"{k}={v:g}" for k, v in cfg[a].items())
        print(f"{a:<20}{macro_faithfulness(d):>9.2f}{c:>12.1f}{r:>12.1f}   {cs}")

    oracle = max((a for a in arms if a.startswith("oracle")),
                 key=lambda a: macro_faithfulness(pts[a]), default=None)

    print()
    print("=" * 78)
    print("PAIRED BOOTSTRAP -- resampled by qid, 95% CI")
    print("=" * 78)
    for new in NEW_ARMS:
        if new not in pts:
            continue
        for base in ("entropy", "max_prob", "constant"):
            if base not in pts:
                continue
            d, lo, hi = compare(pts[new], pts[base])
            star = "" if lo > 0 or hi < 0 else "   (CI spans zero)"
            print(f"  {new} - {base:<16}{d:+7.2f}  [{lo:+.2f}, {hi:+.2f}]{star}")
        print()
    if "conflict" in pts and "probe" in pts:
        d, lo, hi = compare(pts["conflict"], pts["probe"])
        star = "" if lo > 0 or hi < 0 else "   (CI spans zero)"
        print(f"  conflict - probe        {d:+7.2f}  [{lo:+.2f}, {hi:+.2f}]{star}")
        print()

    if oracle:
        d, lo, hi = compare(pts[oracle], pts["entropy"])
        print(f"  {oracle} - entropy   {d:+7.2f}  [{lo:+.2f}, {hi:+.2f}]"
              f"   <- Gate 1's headroom, recomputed here")
        for new in NEW_ARMS:
            if new not in pts:
                continue
            dn, lon, hin = compare(pts[oracle], pts[new])
            print(f"  {oracle} - {new:<9}{dn:+7.2f}  [{lon:+.2f}, {hin:+.2f}]")
        print()
        for new in NEW_ARMS:
            if new not in pts:
                continue
            de = macro_faithfulness(pts[new]) - macro_faithfulness(pts["entropy"])
            print(f"  HEADROOM RECOVERED by {new:<9} "
                  f"{100 * de / d if d else float('nan'):5.1f}%  "
                  f"of the oracle's {d:+.2f} over entropy")

    out = {"arms": {a: {"macro": macro_faithfulness(pts[a]), "config": cfg[a]}
                    for a in arms},
           "unmatched": unmatched}
    with open(_path("gate3b_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {_path('gate3b_results.json')}")


# ======================================================================================
# splithalf -- the honest deployment number
# ======================================================================================

def phase_splithalf(unmatched: bool = False):
    """Every arm above is tuned on the data it is scored on. gate1 invariant 6 says that
    is deliberate and it applies symmetrically, so the comparison is fair -- but the
    ABSOLUTE numbers are optimistic for all of them.

    Here: select each arm's operating point on half the questions, measure it on the
    other half, both directions, and pool. This is what the arm would actually deliver.
    It is a SECONDARY analysis; the primary above is the one comparable to Gate 1.
    """
    cells = attach_probe(load_cells(unmatched), load_probe())
    gens = load_gens()
    sweep = pd.concat([pd.read_csv(_path("sweep_unmatched.csv" if unmatched
                                         else "sweep.csv")),
                       sweep_probe(cells, gens)], ignore_index=True)

    qids = np.array(sorted(cells.qid.unique()))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(qids))
    halves = [set(qids[perm[:len(qids) // 2]]), set(qids[perm[len(qids) // 2:]])]

    arms = [a for a in ("constant", "entropy", "max_prob", "probe", "conflict")
            if a in set(sweep.arm)]
    print("=" * 78)
    print("SPLIT-HALF -- select on one half, measure on the other. Questions disjoint.")
    print("=" * 78)
    print(f"  {len(halves[0])} / {len(halves[1])} questions\n")

    held = {a: [] for a in arms}
    for i, (fit, test) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        print(f"  fold {i}: select on {len(fit)}, measure on {len(test)}")
        for a in arms:
            c, insample = select(sweep, a, on_qids=fit)
            rows = rows_at(sweep, a, c, on_qids=test)
            held[a].append(rows)
            print(f"    {a:<12} select {insample:6.2f}   held-out "
                  f"{macro_faithfulness(rows):6.2f}")
        print()

    pooled = {a: pd.concat(held[a], ignore_index=True) for a in arms}
    print("  pooled held-out macro faithfulness:")
    for a in arms:
        print(f"    {a:<12}{macro_faithfulness(pooled[a]):7.2f}")

    print()
    print("  THIS IS THE HEADLINE FOR THE CONFLICT ARM. It was proposed after seeing the")
    print("  probe arm's result, so its tuned number is not trustworthy; the held-out one")
    print("  is.")
    print()
    for new in NEW_ARMS:
        if new not in pooled:
            continue
        for base in ("entropy", "max_prob", "constant"):
            if base not in pooled:
                continue
            d, lo, hi = compare(pooled[new], pooled[base])
            star = "" if lo > 0 or hi < 0 else "   (CI spans zero)"
            print(f"  {new} - {base:<16}{d:+7.2f}  [{lo:+.2f}, {hi:+.2f}]{star}")
        print()


# ======================================================================================
# diag -- where does the difference land?
# ======================================================================================

def phase_diag(unmatched: bool = False):
    """Gate 1's entire +13.83 sits in the ~27% of instances where confidence and
    knowledge DISAGREE. If the probe buys anything it has to buy it there. This
    decomposes the comparison by that split and reports how often the probe's gating
    decision matches the oracle's.
    """
    cells = attach_probe(load_cells(unmatched), load_probe())
    gens = load_gens()
    sweep = pd.concat([pd.read_csv(_path("sweep_unmatched.csv" if unmatched
                                         else "sweep.csv")),
                       sweep_probe(cells, gens)], ignore_index=True)

    pts, cfg = {}, {}
    for a in ("entropy", "probe", "conflict", "oracle_two_sided", "oracle_one_sided"):
        if a in set(sweep.arm):
            cfg[a], _ = select(sweep, a)
            pts[a] = rows_at(sweep, a, cfg[a])

    # -- the gating decision each arm actually makes, at its own best config -----------
    known = cells.set_index("qid").knowledge.to_dict()
    ent_thr = cfg["entropy"]["thr"]
    pr_thr = cfg["probe"]["thr"]
    rows = cells.drop_duplicates("qid")
    dec = pd.DataFrame({
        "qid": rows.qid.values,
        "truth": [known[q] == "known" for q in rows.qid],
        "entropy_says": (rows.entropy < ent_thr).values,
        "probe_says": (rows.probe > pr_thr).values,
    })

    print("=" * 78)
    print("GATING DECISIONS at each arm's best operating point (per question)")
    print("=" * 78)
    for a in ("entropy_says", "probe_says"):
        agree = (dec[a] == dec.truth).mean()
        tp = ((dec[a]) & (dec.truth)).sum()
        fp = ((dec[a]) & (~dec.truth)).sum()
        fn = ((~dec[a]) & (dec.truth)).sum()
        tn = ((~dec[a]) & (~dec.truth)).sum()
        print(f"  {a:<14} agrees with the oracle {100 * agree:5.1f}%   "
              f"TP {tp:4d}  FP {fp:4d}  FN {fn:4d}  TN {tn:4d}")
    flip = (dec.entropy_says != dec.probe_says).mean()
    better = ((dec.entropy_says != dec.probe_says) & (dec.probe_says == dec.truth)).sum()
    worse = ((dec.entropy_says != dec.probe_says) & (dec.entropy_says == dec.truth)).sum()
    print(f"\n  the two arms disagree on {100 * flip:.1f}% of questions "
          f"({(dec.entropy_says != dec.probe_says).sum()} of {len(dec)})")
    print(f"  of those, probe is right {better}, entropy is right {worse}")

    # -- faithfulness restricted to the off-diagonal ----------------------------------
    off = set(dec[dec.entropy_says != dec.truth].qid)
    print()
    print("=" * 78)
    print("MACRO FAITHFULNESS split by whether entropy's gate ALREADY agrees with truth")
    print("=" * 78)
    print(f"  off-diagonal (entropy wrong about knowledge): {len(off)} questions")
    print(f"\n  {'arm':<20}{'on-diagonal':>14}{'off-diagonal':>14}")
    for a in pts:
        d = pts[a]
        on_ = d[~d.qid.isin(off)]
        of_ = d[d.qid.isin(off)]
        print(f"  {a:<20}{macro_faithfulness(on_):>14.2f}"
              f"{macro_faithfulness(of_):>14.2f}")
    print("\n  Gate 1's headroom lives in the off-diagonal column. A probe that helps")
    print("  must move THAT number, not the pooled one.")


# ======================================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    unmatched = "--unmatched" in sys.argv
    {"check": phase_check, "parity": phase_parity, "decode": phase_decode,
     "score": phase_score, "splithalf": phase_splithalf,
     "diag": phase_diag}[cmd](unmatched)


if __name__ == "__main__":
    main()

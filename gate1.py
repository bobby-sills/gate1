"""
gate1.py -- Gate 1 oracle headroom experiment. DO NOT EDIT (see CLAUDE.md).

Implement backend.py. Run phases in order:

    python gate1.py pool <confiqa_qa.json> <fiqa.json>   # Phase 1
    python gate1.py labels    # Phase 2   knowledge labels, RESUMABLE
    python gate1.py parity    # Phase 2.7 prompt parity -- a HUMAN reads the output
    python gate1.py sanity    # Phase 2.6 cell balance + off-diagonal
    python gate1.py cells     # Phase 3   build the 2x2
    python gate1.py decode    # Phase 4a  decode once per distinct tau, RESUMABLE
    python gate1.py sweep     # Phase 4b  score every config from cache (CPU only)
    python gate1.py analyze   # Phase 5-7 metrics, bootstrap, decision

`labels` and `decode` are the only GPU-bound steps. Both append to JSONL and skip
completed work on restart, so a Colab disconnect costs the current unit and nothing more.
Re-run the same command to resume.

Set GATE1_OUT to a Drive path on Colab, e.g.
    export GATE1_OUT=/content/drive/MyDrive/gate1/outputs
"""

import json
import os
import re
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

# ======================================================================================
# PRE-REGISTERED CONSTANTS -- frozen at Phase 0. Changing these after `decode` begins
# invalidates the pre-registration. See PROTOCOL.md Phase 0.
# ======================================================================================

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CONTINGENCY_MODEL = "google/gemma-2-9b-it"   # reachable ONLY via the ambiguous band
INSIDE_OUT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # 14% gap was measured here

TAU_GRID = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
GAMMA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]
DELTA_GRID = [0.2, 0.4, 0.6]
TAU_MIN, TAU_MAX = 0.0, 3.0        # clamp; tau<0 meaningless, tau>3 degenerate

N_SAMPLES = 10
KNOWN_MIN = 8
UNKNOWN_MAX = 2
TARGET_CELL_SIZE = 300
SEED = 20260803

STOP_ORACLE_MINUS_ENTROPY = 2.0
STOP_ORACLE_MINUS_CONSTANT = 3.0
STOP_OFF_DIAGONAL_FRAC = 0.15
PROCEED_ORACLE_MINUS_ENTROPY = 4.0

N_BOOT = 10_000
FLUSH_EVERY = 20                   # Colab disconnects; write often

OUT = os.environ.get("GATE1_OUT")
if not OUT:
    OUT = "outputs"
    print("=" * 78, file=sys.stderr)
    print("WARNING: GATE1_OUT is unset -- writing checkpoints to ./outputs", file=sys.stderr)
    print("On Colab this is the EPHEMERAL runtime disk. A disconnect destroys it.", file=sys.stderr)
    print("Set it in a PYTHON cell (not `!export`, which does not persist):", file=sys.stderr)
    print("    os.environ['GATE1_OUT'] = '/content/drive/MyDrive/gate1/outputs'", file=sys.stderr)
    print("See COLAB.md.", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
elif os.path.exists("/content") and not OUT.startswith("/content/drive"):
    print(f"WARNING: on Colab but GATE1_OUT={OUT} is not under /content/drive -- "
          f"checkpoints will not survive a disconnect. See COLAB.md.", file=sys.stderr)

RNG = np.random.default_rng(SEED)


# ======================================================================================
# JSONL checkpoint layer
# ======================================================================================

def _path(name: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def read_jsonl(name: str) -> list[dict]:
    p = _path(name)
    if not os.path.exists(p):
        return []
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass          # truncated final line from a hard kill
    return rows


class Appender:
    """Append-only writer with periodic fsync. Resume = skip keys already present."""

    def __init__(self, name: str, key_fields: tuple):
        self.path = _path(name)
        self.key_fields = key_fields
        self.done = {self._key(r) for r in read_jsonl(name)}
        self.f = open(self.path, "a")
        self.n = 0
        if self.done:
            print(f"  resuming {name}: {len(self.done)} units already complete")

    def _key(self, row: dict):
        return tuple(row[k] for k in self.key_fields)

    def has(self, **kw) -> bool:
        return tuple(kw[k] for k in self.key_fields) in self.done

    def write(self, row: dict):
        self.f.write(json.dumps(row) + "\n")
        self.done.add(self._key(row))
        self.n += 1
        if self.n % FLUSH_EVERY == 0:
            self.f.flush()
            os.fsync(self.f.fileno())

    def close(self):
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()


# ======================================================================================
# Alias matching -- identical for labels (Phase 2) and scoring (Phase 5)
# ======================================================================================

_ARTICLES = re.compile(r"^(the|a|an)\s+", re.I)
_PUNCT = re.compile(r"[^\w\s]")


def normalize(s: str) -> str:
    s = _PUNCT.sub(" ", (s or "").lower())
    s = _ARTICLES.sub("", s.strip())
    return re.sub(r"\s+", " ", s).strip()


def alias_match(prediction: str, aliases: list) -> bool:
    """Substring match after normalization -- NOT exact equality. Alias handling is
    where most label noise in this literature originates."""
    p = normalize(prediction)
    return any(normalize(a) in p for a in aliases if a)


# ======================================================================================
# PHASE 1
# ======================================================================================

def phase_pool(confiqa: str, fiqa: str):
    import backend
    insts, attrition = backend.load_pool(confiqa, fiqa)
    with open(_path("pool.jsonl"), "w") as f:
        for i in insts:
            f.write(json.dumps(asdict(i)) + "\n")
    json.dump(attrition, open(_path("attrition.json"), "w"), indent=2)
    print(json.dumps(attrition, indent=2))
    print(f"\nwrote {len(insts)} instances")
    print("\nHUMAN CHECK: read the 3 instances below before continuing.\n")
    for i in insts[:3]:
        print(json.dumps(asdict(i), indent=2)[:800], "\n")


# ======================================================================================
# PHASE 2 -- knowledge labels (resumable)
# ======================================================================================

def phase_labels():
    import backend
    pool = read_jsonl("pool.jsonl")
    app = Appender("labels.jsonl", ("qid",))
    todo = [r for r in pool if not app.has(qid=r["qid"])]
    print(f"labelling {len(todo)} / {len(pool)} instances")

    t0 = time.time()
    for n, r in enumerate(todo, 1):
        gens = backend.sample_closed_book(r["question"], N_SAMPLES, SEED)
        n_correct = sum(alias_match(g, r["gold_aliases"]) for g in gens)
        entropy, max_prob = backend.deterministic_features(r["question"])

        if n_correct >= KNOWN_MIN:
            knowledge = "known"
        elif n_correct <= UNKNOWN_MAX:
            knowledge = "unknown"
        else:
            knowledge = "discard"

        app.write({"qid": r["qid"], "n_correct": n_correct, "knowledge": knowledge,
                   "entropy": float(entropy), "max_prob": float(max_prob)})
        if n % 50 == 0:
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)}  {rate:.2f}s/inst  "
                  f"eta {rate * (len(todo) - n) / 60:.0f}min")
    app.close()
    print("done")


def phase_parity():
    import backend
    pool = read_jsonl("pool.jsonl")[:10]
    with open(_path("prompt_parity.txt"), "w") as f:
        for r in pool:
            cb, wc = backend.render_prompts(r["question"], r["factual_context"])
            f.write(f"{'=' * 78}\nqid={r['qid']}\n{'-' * 78}\nCLOSED BOOK:\n{cb}\n"
                    f"{'-' * 78}\nWITH CONTEXT:\n{wc}\n\n")
    print(f"wrote {_path('prompt_parity.txt')}")
    print("\nSTOP. A HUMAN must read that file and confirm the two prompts differ ONLY")
    print("by the document block. Do not automate this check.")


# ======================================================================================
# PHASE 2.6 / 3
# ======================================================================================

def _merged() -> pd.DataFrame:
    pool = pd.DataFrame(read_jsonl("pool.jsonl"))
    labels = pd.DataFrame(read_jsonl("labels.jsonl"))
    return pool.merge(labels, on="qid")


def phase_sanity():
    df = _merged()
    kept = df[df.knowledge != "discard"]
    n_known = int((kept.knowledge == "known").sum())
    n_unknown = int((kept.knowledge == "unknown").sum())
    discard_rate = 1 - len(kept) / len(df)

    thr = kept.entropy.median()
    tab = pd.crosstab(kept.knowledge == "known", kept.entropy < thr)
    tab.index.name, tab.columns.name = "knows", "looks_confident"
    off = (tab.loc[True, False] + tab.loc[False, True]) / tab.values.sum()

    print(f"discard rate      : {discard_rate:.3f}   (expect 0.20-0.30)")
    print(f"known / unknown   : {n_known} / {n_unknown}")
    print(f"off-diagonal frac : {off:.3f}   (stop threshold {STOP_OFF_DIAGONAL_FRAC})")
    print(tab)

    if min(n_known, n_unknown) < TARGET_CELL_SIZE:
        print("\n  WARNING: cannot reach target cell size. ConFiQA-QA may be too "
              "long-tail for this model -- widen the pool before proceeding.")
    if off < STOP_OFF_DIAGONAL_FRAC:
        print("\n  WARNING: off-diagonal below stop threshold. The sweep is unlikely "
              "to show anything regardless of the arms.")

    json.dump({"discard_rate": discard_rate, "n_known": n_known,
               "n_unknown": n_unknown, "off_diagonal": float(off)},
              open(_path("sanity.json"), "w"), indent=2)


CELLS = {
    ("known", "factual"): "agreement",
    ("known", "counterfactual"): "resistance",
    ("unknown", "factual"): "correction",
    ("unknown", "counterfactual"): "both_wrong",
}


def phase_cells():
    df = _merged()
    df = df[df.knowledge != "discard"]
    rows = []
    for r in df.itertuples():
        for kind in ("factual", "counterfactual"):
            rows.append({
                "qid": r.qid, "subject_qid": r.subject_qid, "question": r.question,
                "context_kind": kind,
                "context": (r.factual_context if kind == "factual"
                            else r.counterfactual_context),
                "cell": CELLS[(r.knowledge, kind)],
                "knowledge": r.knowledge,
                "entropy": r.entropy, "max_prob": r.max_prob,
                # TARGET IS ALWAYS THE FACTUAL GOLD. In the resistance cell the document
                # is wrong, so following it is the error. This is not a bug.
                "target_aliases": r.gold_aliases,
            })
    out = pd.DataFrame(rows)
    print("natural cell distribution:\n", out.cell.value_counts(), "\n")

    balanced = []
    for cell, grp in out.groupby("cell"):
        take = min(TARGET_CELL_SIZE, len(grp))
        balanced.append(grp.sample(take, random_state=SEED % 2**31))
        if take < TARGET_CELL_SIZE:
            print(f"  WARNING: cell '{cell}' has {take} (target {TARGET_CELL_SIZE})")
    out = pd.concat(balanced, ignore_index=True)

    with open(_path("cells.jsonl"), "w") as f:
        for r in out.to_dict("records"):
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(out)} instances")


# ======================================================================================
# PHASE 4 -- tau-deduplicated decode
# ======================================================================================

ARMS = ("constant", "oracle_one_sided", "oracle_two_sided", "entropy", "max_prob")


def tau_for(arm: str, row: dict, tau0: float, gamma: float, delta: float,
            thr: float) -> float:
    """One knob, both directions. tau>1 pushes past the document; tau<1 pulls back
    toward memory; tau=1+alpha recovers CAD."""
    if arm == "constant":
        t = tau0
    elif arm == "oracle_one_sided":
        t = tau0 if row["knowledge"] == "known" else tau0 + gamma
    elif arm == "oracle_two_sided":
        t = tau0 - delta if row["knowledge"] == "known" else tau0 + gamma
    elif arm == "entropy":
        t = tau0 - delta if row["entropy"] < thr else tau0 + gamma
    elif arm == "max_prob":
        t = tau0 - delta if row["max_prob"] > thr else tau0 + gamma
    else:
        raise ValueError(arm)
    return round(float(np.clip(t, TAU_MIN, TAU_MAX)), 3)


def build_configs(cells: pd.DataFrame) -> list:
    """Every arm gets the same grid density -- equal tuning budgets (PROTOCOL 4.1).
    Entropy/max_prob thresholds are tuned on the eval data, i.e. optimistically, which
    biases the comparison AGAINST our hypothesis on purpose (PROTOCOL 4.2)."""
    thr = {
        "entropy": list(np.quantile(cells.entropy, [0.25, 0.5, 0.75])),
        "max_prob": list(np.quantile(cells.max_prob, [0.25, 0.5, 0.75])),
    }
    cfgs = []
    for arm in ARMS:
        if arm == "constant":
            grid = [(t, 0.0, 0.0, 0.0) for t in TAU_GRID]
        elif arm == "oracle_one_sided":
            grid = [(t, g, 0.0, 0.0) for t in TAU_GRID for g in GAMMA_GRID]
        elif arm == "oracle_two_sided":
            grid = [(t, g, d, 0.0) for t in TAU_GRID for g in GAMMA_GRID
                    for d in DELTA_GRID]
        else:
            grid = [(t, g, d, x) for t in TAU_GRID for g in GAMMA_GRID
                    for d in DELTA_GRID for x in thr[arm]]
        for t, g, d, x in grid:
            cfgs.append({"arm": arm, "tau0": t, "gamma": g, "delta": d, "thr": x})
    return cfgs


def required_work(cells: pd.DataFrame, cfgs: list) -> set:
    """THE KEY OPTIMIZATION. The decoder only sees tau, so thousands of configs collapse
    onto a few dozen distinct (qid, context_kind, tau) units. Decode each once; every
    config then reads from cache."""
    recs = cells.to_dict("records")
    work = set()
    for c in cfgs:
        for r in recs:
            work.add((r["qid"], r["context_kind"],
                      tau_for(c["arm"], r, c["tau0"], c["gamma"], c["delta"], c["thr"])))
    return work


def greedy_decode(question: str, context: str, tau: float, max_tokens: int = 32) -> str:
    """CAD in tau form, greedy:
        log q_t = (1 - tau) * log p_theta_t + tau * log p_ctx_t
    Two forward passes per token via backend.next_token_distributions, which holds KV
    caches internally. Do NOT add a second cache layer here."""
    import backend
    prefix = []
    for _ in range(max_tokens):
        lp_theta, lp_ctx = backend.next_token_distributions(question, context, prefix)
        tok = int(np.argmax((1.0 - tau) * lp_theta + tau * lp_ctx))
        if backend.is_eos(tok):
            break
        prefix.append(tok)
    return backend.detokenize(prefix)


def phase_decode():
    cells = pd.DataFrame(read_jsonl("cells.jsonl"))
    cfgs = build_configs(cells)
    work = sorted(required_work(cells, cfgs))
    taus = sorted({w[2] for w in work})
    naive = len(cfgs) * len(cells)
    print(f"{len(cfgs)} configs x {len(cells)} instances = {naive} naive decodes")
    print(f"-> {len(taus)} distinct tau values -> {len(work)} decode units "
          f"({naive / max(len(work), 1):.0f}x saving)")

    ctx = {(r["qid"], r["context_kind"]): (r["question"], r["context"])
           for r in cells.to_dict("records")}
    app = Appender("generations.jsonl", ("qid", "context_kind", "tau"))
    todo = [w for w in work if not app.has(qid=w[0], context_kind=w[1], tau=w[2])]
    print(f"decoding {len(todo)} / {len(work)} units")

    t0 = time.time()
    for n, (qid, kind, tau) in enumerate(todo, 1):
        q, c = ctx[(qid, kind)]
        text = greedy_decode(q, c, tau)
        app.write({"qid": qid, "context_kind": kind, "tau": tau,
                   "text": text, "n_chars": len(text)})
        if n % 50 == 0:
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)}  {rate:.2f}s/unit  "
                  f"eta {rate * (len(todo) - n) / 60:.0f}min")
    app.close()


def phase_sweep():
    """CPU only. Scores every config against the cached generations."""
    cells = pd.DataFrame(read_jsonl("cells.jsonl"))
    gens = {(g["qid"], g["context_kind"], g["tau"]): g
            for g in read_jsonl("generations.jsonl")}
    cfgs = build_configs(cells)
    recs = []
    for c in cfgs:
        for r in cells.to_dict("records"):
            tau = tau_for(c["arm"], r, c["tau0"], c["gamma"], c["delta"], c["thr"])
            g = gens.get((r["qid"], r["context_kind"], tau))
            if g is None:
                raise KeyError(f"missing generation {(r['qid'], r['context_kind'], tau)}"
                               " -- run `decode` to completion first")
            recs.append({**c, "qid": r["qid"], "cell": r["cell"],
                         "correct": alias_match(g["text"], r["target_aliases"]),
                         "n_chars": g["n_chars"]})
    df = pd.DataFrame(recs)
    df.to_csv(_path("sweep.csv"), index=False)

    # PROTOCOL 4.3: degenerate generation check
    print("mean output length by tau0 (collapse => sweep invalid at that end):")
    print(df.groupby("tau0").n_chars.mean().round(1))
    print(f"\nwrote {_path('sweep.csv')} ({len(df)} rows)")


# ======================================================================================
# PHASE 5-7
# ======================================================================================

def macro_faithfulness(df: pd.DataFrame) -> float:
    """Mean of correction and resistance accuracy. NOT aggregate accuracy: a decoder
    that always follows the document scores 100/0 and would look reasonable."""
    c = df[df.cell == "correction"].correct.mean()
    r = df[df.cell == "resistance"].correct.mean()
    return 100 * float(np.mean([c, r]))


def best_operating_point(sweep: pd.DataFrame, arm: str) -> pd.DataFrame:
    sub = sweep[sweep.arm == arm]
    keys = ["tau0", "gamma", "delta", "thr"]
    scores = sub.groupby(keys).apply(macro_faithfulness, include_groups=False)
    best = scores.idxmax()
    mask = np.logical_and.reduce([sub[k] == v for k, v in zip(keys, best)])
    return sub[mask]


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = N_BOOT):
    """Difference in macro faithfulness, 95% CI.

    RESAMPLES BY QUESTION, NOT BY ROW. Each question contributes two correlated
    instances; row-level resampling would treat them as independent and artificially
    shrink the interval. This is not a bug.
    """
    qids = np.array(sorted(set(a.qid) & set(b.qid)))
    ai, bi = a.set_index("qid"), b.set_index("qid")
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        draw = RNG.choice(qids, size=len(qids), replace=True)
        diffs[i] = (macro_faithfulness(ai.loc[draw].reset_index())
                    - macro_faithfulness(bi.loc[draw].reset_index()))
    point = macro_faithfulness(a) - macro_faithfulness(b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def phase_analyze():
    sweep = pd.read_csv(_path("sweep.csv"))
    off = json.load(open(_path("sanity.json")))["off_diagonal"]
    pts = {a: best_operating_point(sweep, a) for a in ARMS if a in set(sweep.arm)}

    oracle = max(("oracle_one_sided", "oracle_two_sided"),
                 key=lambda a: macro_faithfulness(pts[a]))
    print(f"winning oracle arm: {oracle}\n")

    d_ent, lo_e, hi_e = paired_bootstrap(pts[oracle], pts["entropy"])
    d_con, lo_c, hi_c = paired_bootstrap(pts[oracle], pts["constant"])
    print(f"Oracle - Entropy  = {d_ent:+.2f}  [{lo_e:+.2f}, {hi_e:+.2f}]")
    print(f"Oracle - Constant = {d_con:+.2f}  [{lo_c:+.2f}, {hi_c:+.2f}]")
    print(f"off-diagonal      = {off:.3f}\n")

    for cell in ("correction", "resistance", "agreement", "both_wrong"):
        line = "  ".join(f"{a}={100 * p[p.cell == cell].correct.mean():5.1f}"
                         for a, p in pts.items())
        print(f"  {cell:12s} {line}")

    print("\n" + "=" * 78)
    print(_decide(d_ent, lo_e, d_con, off, pts[oracle]))
    print("=" * 78)


def _decide(d_ent, lo_e, d_con, off, oracle_pt) -> str:
    if off < STOP_OFF_DIAGONAL_FRAC:
        return "SLICE PROBLEM: mechanism too rare. Fix the slice before abandoning."
    if d_con < STOP_ORACLE_MINUS_CONSTANT:
        return "STOP: knowledge does not change decoding outcomes at all."
    if d_ent < STOP_ORACLE_MINUS_ENTROPY:
        return "STOP: free signals already carry the useful information."
    if d_ent >= PROCEED_ORACLE_MINUS_ENTROPY and lo_e > 0:
        return f"PROCEED to Gate 2. Headroom budget = {d_ent:.2f} points macro."
    r = oracle_pt.iloc[0]
    return (f"CONTINGENCY: Oracle - Entropy = {d_ent:.2f} is in the ambiguous band "
            f"[{STOP_ORACLE_MINUS_ENTROPY}, {PROCEED_ORACLE_MINUS_ENTROPY}). Fire the "
            f"pre-registered clause: Phase 2 + reduced sweep (tau grid only, "
            f"gamma={r.gamma}, delta={r.delta} fixed) on {CONTINGENCY_MODEL}. A gap that "
            f"widens materially there means the Llama null was a calibration artifact; "
            f"a flat gap is a genuine stop.")


# ======================================================================================

PHASES = {"pool": phase_pool, "labels": phase_labels, "parity": phase_parity,
          "sanity": phase_sanity, "cells": phase_cells, "decode": phase_decode,
          "sweep": phase_sweep, "analyze": phase_analyze}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in PHASES:
        print(__doc__)
        sys.exit(1)
    PHASES[sys.argv[1]](*sys.argv[2:])

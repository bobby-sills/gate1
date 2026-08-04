"""Relation-matched pair sampling for the alias short-form question.

WHY MATCHED PAIRS. The high-risk population (multi-word gold answer, no short-form
alias, not a person name) is concentrated in `award received`, `position held`,
`member of`, `genre` -- relations whose answers are genuinely long-tail. Comparing the
high-risk stratum against an unmatched control therefore compares awards against
spouses, and real difficulty is indistinguishable from mislabelling. Sampling one
high-risk and one low-risk UNKNOWN row from the SAME relation leaves the presence of a
usable short-form alias as the only systematic difference within a pair, which is
exactly the defect being measured.

    python pairs.py sample    # GPU, resumable. Draws candidates and labels them.
    python pairs.py build     # CPU. Forms matched pairs, writes a BLINDED reading set.
    python pairs.py score     # CPU. McNemar on the discordant pairs.

The reading set hides the stratum; `handlabel_key.json` holds the mapping and must not
be opened before reading. Analysis is paired: only pairs where the two members are
judged differently carry information.
"""
import ast
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import gate1
from instrument import is_high_risk

TARGET_PAIRS = 35
MIN_PER_RELATION = 3
MIN_POOL_PER_SIDE = 8       # a relation needs this many of each kind to be eligible
MIN_HIGH_RISK_SHARE = 0.30  # ... and the high-risk kind must be a real share of it
CONFIQA = "data/ConFiQA-QA.json"


# --------------------------------------------------------------------------------------

def relations(pool: list[dict]) -> dict:
    """question -> human-readable relation. RawInstance does not carry the relation, so
    this rejoins against the source file on question text."""
    raw = json.load(open(CONFIQA))
    by_q = {}
    for r in raw:
        try:
            by_q[r["question"]] = ast.literal_eval(r["orig_path_labeled"])[0][1]
        except (ValueError, SyntaxError, IndexError, KeyError):
            continue
    return {r["qid"]: by_q.get(r["question"], "?") for r in pool}


def eligible(pool: list[dict], rel: dict) -> dict:
    """Relations holding enough of BOTH kinds to draw matched pairs from.

    The high-risk share must also clear MIN_HIGH_RISK_SHARE. In a relation that is 92%
    low-risk -- `spouse`, `sport`, `record label` -- the handful of high-risk rows are
    atypical of the relation rather than representative of it, so a pair drawn there
    would reintroduce the very difficulty confound the matching exists to remove.
    """
    buckets = defaultdict(lambda: {"high": [], "low": []})
    for r in pool:
        buckets[rel[r["qid"]]]["high" if is_high_risk(r) else "low"].append(r)
    out = {}
    for k, v in buckets.items():
        n_hi, n_lo = len(v["high"]), len(v["low"])
        if k == "?" or min(n_hi, n_lo) < MIN_POOL_PER_SIDE:
            continue
        if n_hi / (n_hi + n_lo) < MIN_HIGH_RISK_SHARE:
            continue
        out[k] = v
    return out


def allocate(elig: dict) -> dict:
    """Pairs to target per relation: even spread, capped by what each relation holds."""
    order = sorted(elig, key=lambda k: -min(len(elig[k]["high"]), len(elig[k]["low"])))
    plan, left = {}, TARGET_PAIRS
    for i, k in enumerate(order):
        share = max(MIN_PER_RELATION, math.ceil(left / max(len(order) - i, 1)))
        cap = min(len(elig[k]["high"]), len(elig[k]["low"]))
        plan[k] = min(share, cap, left)
        left -= plan[k]
        if left <= 0:
            break
    return {k: v for k, v in plan.items() if v > 0}


# --------------------------------------------------------------------------------------

def phase_sample():
    """Label enough candidates per relation to fill the plan. Resumable: re-run after a
    disconnect and it skips qids already done. Reuses instrument.jsonl, which came from
    the same backend call with the same seed."""
    import backend

    pool = gate1.read_jsonl("pool.jsonl")
    if not pool:
        sys.exit("no pool.jsonl -- run `python gate1.py pool` first")
    rel = relations(pool)
    elig = eligible(pool, rel)
    plan = allocate(elig)
    print(f"{len(elig)} eligible relations; plan = {plan}")
    print(f"target {sum(plan.values())} pairs\n")

    # Measured unknown rates from the instrumentation run: 80% high-risk, 38% low-risk.
    # Oversample accordingly, with headroom, capped by availability.
    rng = random.Random(gate1.SEED)
    todo = []
    for k, n_pairs in plan.items():
        for side, rate in (("high", 0.80), ("low", 0.38)):
            want = min(len(elig[k][side]), math.ceil(n_pairs / rate * 1.4))
            todo += rng.sample(elig[k][side], want)
    rng.shuffle(todo)

    app = gate1.Appender("pairs_raw.jsonl", ("qid",))
    todo = [r for r in todo if not app.has(qid=r["qid"])]

    # instrument.jsonl already labelled 100 questions with the identical call; reuse.
    seen = {r["qid"] for r in gate1.read_jsonl("pairs_raw.jsonl")}
    reused = 0
    for r in gate1.read_jsonl("instrument.jsonl"):
        if r["qid"] not in seen:
            app.write({"qid": r["qid"], "question": r["question"],
                       "gold_aliases": r["gold_aliases"], "generations": r["generations"],
                       "n_correct": r["n_correct"], "knowledge": r["knowledge"]})
            seen.add(r["qid"])
            reused += 1
    todo = [r for r in todo if r["qid"] not in seen]
    print(f"reused {reused} from instrument.jsonl; sampling {len(todo)} new\n")

    t0 = time.time()
    for n, r in enumerate(todo, 1):
        gens = backend.sample_closed_book(r["question"], gate1.N_SAMPLES, gate1.SEED)
        n_correct = sum(gate1.alias_match(g, r["gold_aliases"]) for g in gens)
        knowledge = ("known" if n_correct >= gate1.KNOWN_MIN else
                     "unknown" if n_correct <= gate1.UNKNOWN_MAX else "discard")
        app.write({"qid": r["qid"], "question": r["question"],
                   "gold_aliases": r["gold_aliases"], "generations": gens,
                   "n_correct": n_correct, "knowledge": knowledge})
        if n % 25 == 0:
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)}  {rate:.2f}s/inst  "
                  f"eta {rate * (len(todo) - n) / 60:.0f}min")
    app.close()
    print("done -- now run `python pairs.py build`")


# --------------------------------------------------------------------------------------

def phase_build():
    pool = {r["qid"]: r for r in gate1.read_jsonl("pool.jsonl")}
    rel = relations(list(pool.values()))
    done = [r for r in gate1.read_jsonl("pairs_raw.jsonl") if r["knowledge"] == "unknown"]
    by_rel = defaultdict(lambda: {"high": [], "low": []})
    for r in done:
        src = pool.get(r["qid"])
        if src:
            by_rel[rel[r["qid"]]]["high" if is_high_risk(src) else "low"].append(r)

    rng = random.Random(gate1.SEED)
    pairs = []
    for k, v in sorted(by_rel.items()):
        rng.shuffle(v["high"])
        rng.shuffle(v["low"])
        n = min(len(v["high"]), len(v["low"]))
        for i in range(n):
            pairs.append({"relation": k, "high": v["high"][i], "low": v["low"][i]})
    rng.shuffle(pairs)
    pairs = pairs[:TARGET_PAIRS]

    print("matched UNKNOWN pairs by relation:")
    counts = defaultdict(int)
    for p in pairs:
        counts[p["relation"]] += 1
    for k, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"   {k:<28} {n}")
    print(f"\n{len(pairs)} pairs -> {2 * len(pairs)} blinded items")
    if len(pairs) < TARGET_PAIRS:
        print(f"   NOTE: short of the {TARGET_PAIRS} target -- run `sample` again or "
              "lower MIN_POOL_PER_SIDE")

    items = []
    for i, p in enumerate(pairs):
        items.append({"pair": i, "side": "high", **p["high"]})
        items.append({"pair": i, "side": "low", **p["low"]})
    rng.shuffle(items)

    path = gate1._path("handlabel.txt")
    with open(path, "w") as f:
        f.write("HAND-LABELLING SET (blinded)\n\n")
        f.write("Classify THE MODEL for each item:\n")
        f.write("   K = clearly knew the answer\n")
        f.write("   N = clearly did not know\n")
        f.write("   A = ambiguous\n\n")
        f.write("Judge whether the model knew the fact. Ignore whether any string would\n")
        f.write("have matched the gold answer -- that is the thing under test.\n")
        f.write("Items are shuffled; pair membership and stratum are deliberately hidden.\n")
        f.write(f"{len(items)} items.\n\n")
        for n, r in enumerate(items, 1):
            f.write(f"{'=' * 78}\nITEM {n}\n{'=' * 78}\n")
            f.write(f"question      : {r['question']}\n")
            f.write(f"gold answer   : {r['gold_aliases'][0]}\n")
            f.write(f"gold_aliases  : {r['gold_aliases']}\n")
            f.write(f"n_correct     : {r['n_correct']} / 10\n")
            f.write("generations   :\n")
            for j, g in enumerate(r["generations"], 1):
                f.write(f"   {j:2d}. {g}\n")
            f.write("\n")

    json.dump({str(n): {"qid": r["qid"], "pair": r["pair"], "side": r["side"],
                        "relation": pairs[r["pair"]]["relation"]}
               for n, r in enumerate(items, 1)},
              open(gate1._path("handlabel_key.json"), "w"), indent=2)
    print(f"\nwrote {path}")
    print(f"wrote {gate1._path('handlabel_key.json')} -- do not open before reading")


# --------------------------------------------------------------------------------------

def _exact_binom_two_sided(b: int, c: int) -> float:
    """Exact McNemar. Under H0 each discordant pair is a coin flip."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def phase_score(labels: str):
    """labels: "1:K 2:N 3:A ..." covering every item."""
    key = json.load(open(gate1._path("handlabel_key.json")))
    got = {}
    for tok in labels.replace(",", " ").split():
        if ":" in tok:
            i, v = tok.split(":", 1)
            got[i.strip()] = v.strip().upper()[:1]
    missing = [i for i in key if i not in got]
    if missing:
        sys.exit(f"missing labels for items: {missing[:20]}"
                 f"{' ...' if len(missing) > 20 else ''}")

    for ambiguous_is_known in (False, True):
        pairs = defaultdict(dict)
        for i, meta in key.items():
            v = got[i]
            knew = (v == "K") or (ambiguous_is_known and v == "A")
            pairs[meta["pair"]][meta["side"]] = knew
        b = sum(1 for p in pairs.values() if p.get("high") and not p.get("low"))
        c = sum(1 for p in pairs.values() if p.get("low") and not p.get("high"))
        both = sum(1 for p in pairs.values() if p.get("high") and p.get("low"))
        neither = len(pairs) - b - c - both
        p = _exact_binom_two_sided(b, c)
        tag = "A counted as KNEW" if ambiguous_is_known else "A counted as NOT known"
        print(f"\n--- {tag} ---")
        print(f"  pairs: {len(pairs)}   both knew {both}   neither {neither}")
        print(f"  discordant: high-risk knew only = {b}, low-risk knew only = {c}")
        print(f"  exact McNemar p = {p:.4f}")
        if p < 0.05 and b > c:
            print("  -> discordant pairs favour high-risk mislabelling at p<0.05")
        elif p < 0.05:
            print("  -> significant, but in the direction of the LOW-risk arm")
        else:
            print("  -> not significant")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "sample":
        phase_sample()
    elif cmd == "build":
        phase_build()
    elif cmd == "score":
        phase_score(" ".join(sys.argv[2:]))
    else:
        print(__doc__)
        sys.exit(1)

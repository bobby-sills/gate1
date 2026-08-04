"""Phase 2 instrumentation run -- HANDOFF Task 2 checkpoint, stratified.

Two questions in one GPU pass over 100 instances:

  1. seconds per instance and the projected total for the full pool (HANDOFF: report it,
     do not reduce N_SAMPLES, that constant is pre-registered)

  2. the alias short-form measurement. gate1.phase_labels computes `gens` at line 187
     and then discards them -- only n_correct/knowledge/entropy/max_prob reach
     labels.jsonl. So swapping a 100-row pool.jsonl in and running `gate1.py labels`
     CANNOT answer the short-form question. This script calls the same two backend
     functions in the same order and keeps the generations.

The stratified sample is 50 rows from the high-risk population (multi-word gold answer,
no short-form alias, not a two-token person name) and 50 from the rest, drawn with
gate1.SEED so it is reproducible.

    python instrument.py            # run it (GPU)
    python instrument.py --report   # re-print the report from saved output (CPU)

Writes instrument.jsonl and instrument.json under $GATE1_OUT. Nothing here feeds the
experiment; it exists to inform two decisions and is not part of the pipeline.
"""
import json
import random
import statistics
import sys
import time

import gate1

FULL_POOL_DEFAULT = 4207


# --------------------------------------------------------------------------------------
# stratification
# --------------------------------------------------------------------------------------

def _has_shortform(aliases: list[str]) -> bool:
    """Does the alias list contain a proper sub-phrase of the answer, e.g. a surname?"""
    head = gate1.normalize(aliases[0]).split()
    for a in aliases[1:]:
        t = gate1.normalize(a).split()
        if 0 < len(t) < len(head) and set(t) <= set(head):
            return True
    return False


def _person_like(aliases: list[str]) -> bool:
    """Two capitalized tokens. Models emit these in full, so the short-form risk is low."""
    t = aliases[0].split()
    return len(t) == 2 and all(w[:1].isupper() for w in t)


def is_high_risk(row: dict) -> bool:
    """Multi-word gold answer with no short form and not a person name -- a model that
    knows the fact may answer 'an Emmy' and be scored wrong, hence labelled unknown."""
    al = row["gold_aliases"]
    return (len(gate1.normalize(al[0]).split()) > 1
            and not _has_shortform(al) and not _person_like(al))


def stratified_sample(pool: list[dict], n_per: int = 50) -> list[dict]:
    high = [r for r in pool if is_high_risk(r)]
    rest = [r for r in pool if not is_high_risk(r)]
    rng = random.Random(gate1.SEED)
    take = (rng.sample(high, min(n_per, len(high)))
            + rng.sample(rest, min(n_per, len(rest))))
    rng.shuffle(take)
    for r in take:
        r["stratum"] = "high_risk" if is_high_risk(r) else "rest"
    return take


# --------------------------------------------------------------------------------------
# scoring of one generation against the gold aliases
# --------------------------------------------------------------------------------------

def classify(gen: str, gold: list[str], question: str) -> str:
    """How does this generation relate to the gold answer?

    full         -- gate1.alias_match accepts it. Scores CORRECT everywhere.
    short_form   -- rejected, but it contains a >=4-char word from a gold alias, so the
                    model produced something derived from the right answer ("an Emmy",
                    "Balfe"). THIS is the population that gets mislabelled `unknown`.
    miss         -- unrelated to the gold answer.

    Words that also appear in the QUESTION are excluded. Without that, a generation
    merely echoing the prompt counts as short_form -- "What award did X receive?" shares
    "award" with the gold answer "Gamescom Award - Best Trailer" while saying nothing.
    Found by a smoke test against a placeholder backend that echoed the question.

    Still a heuristic, and still over-inclusive. The report prints examples so the call
    is made by reading them, not by trusting this function.
    """
    if gate1.alias_match(gen, gold):
        return "full"
    qwords = set(gate1.normalize(question).split())
    words = {w for a in gold for w in gate1.normalize(a).split()
             if len(w) >= 4 and w not in qwords}
    return "short_form" if words & set(gate1.normalize(gen).split()) else "miss"


# --------------------------------------------------------------------------------------

def run():
    import backend

    pool = gate1.read_jsonl("pool.jsonl")
    if not pool:
        sys.exit("no pool.jsonl -- run `python gate1.py pool` first")
    sample = stratified_sample(pool)
    print(f"pool {len(pool)}; sampling {len(sample)} "
          f"({sum(r['stratum'] == 'high_risk' for r in sample)} high-risk)")

    rows, t0 = [], time.time()
    for n, r in enumerate(sample, 1):
        t = time.time()
        gens = backend.sample_closed_book(r["question"], gate1.N_SAMPLES, gate1.SEED)
        t_sample = time.time() - t

        t = time.time()
        entropy, max_prob = backend.deterministic_features(r["question"])
        t_feat = time.time() - t

        n_correct = sum(gate1.alias_match(g, r["gold_aliases"]) for g in gens)
        knowledge = ("known" if n_correct >= gate1.KNOWN_MIN else
                     "unknown" if n_correct <= gate1.UNKNOWN_MAX else "discard")
        rows.append({
            "qid": r["qid"], "stratum": r["stratum"], "question": r["question"],
            "gold_aliases": r["gold_aliases"], "generations": gens,
            "classes": [classify(g, r["gold_aliases"], r["question"]) for g in gens],
            "n_correct": n_correct, "knowledge": knowledge,
            "entropy": float(entropy), "max_prob": float(max_prob),
            "t_sample": t_sample, "t_features": t_feat,
        })
        if n % 10 == 0:
            print(f"  {n}/{len(sample)}  {(time.time() - t0) / n:.2f}s/inst")

    with open(gate1._path("instrument.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    report(rows, len(pool), time.time() - t0)


def report(rows: list[dict], pool_size: int, wall: float | None = None):
    n = len(rows)
    per = [r["t_sample"] + r["t_features"] for r in rows]
    total = wall if wall is not None else sum(per)

    print("\n" + "=" * 78)
    print("TIMING")
    print("=" * 78)
    print(f"  instances                  : {n}")
    print(f"  wall clock                 : {total / 60:.1f} min")
    print(f"  seconds/instance           : {total / n:.2f}  "
          f"(median {statistics.median(per):.2f})")
    print(f"    of which sampling        : {statistics.mean(r['t_sample'] for r in rows):.2f}s "
          f"({gate1.N_SAMPLES} generations)")
    print(f"    of which features        : {statistics.mean(r['t_features'] for r in rows):.2f}s "
          f"(separate pass, invariant #2)")
    print(f"  projected for {pool_size} instances : {total / n * pool_size / 60:.0f} min")

    print("\n" + "=" * 78)
    print("ALIAS SHORT-FORM MEASUREMENT  (your <20% / >40% decision rule)")
    print("=" * 78)
    out = {"n": n, "seconds_per_instance": total / n,
           "projected_minutes_full_pool": total / n * pool_size / 60, "strata": {}}
    for stratum in ("high_risk", "rest"):
        sub = [r for r in rows if r["stratum"] == stratum]
        if not sub:
            continue
        flat = [c for r in sub for c in r["classes"]]
        g = len(flat)
        full = flat.count("full") / g
        short = flat.count("short_form") / g
        # The rate that matters: among generations that are ABOUT the right answer, how
        # many does alias_match throw away?
        about = flat.count("full") + flat.count("short_form")
        short_only = flat.count("short_form") / about if about else 0.0
        k = sum(r["knowledge"] == "known" for r in sub)
        u = sum(r["knowledge"] == "unknown" for r in sub)
        d = sum(r["knowledge"] == "discard" for r in sub)
        print(f"\n  {stratum}  (n={len(sub)}, {g} generations)")
        print(f"    generation contains the FULL gold string   : {full:.1%}")
        print(f"    short form only, alias_match rejects it    : {short:.1%}")
        print(f"    -> of generations about the right answer,")
        print(f"       the fraction alias_match throws away    : {short_only:.1%}")
        print(f"    labels: known {k}  unknown {u}  discard {d}   "
              f"(discard rate {d / len(sub):.1%})")
        out["strata"][stratum] = {
            "n": len(sub), "full_rate": full, "short_form_rate": short,
            "short_form_only_of_about": short_only,
            "known": k, "unknown": u, "discard": d}

    hr = out["strata"].get("high_risk", {}).get("short_form_only_of_about")
    if hr is not None:
        print("\n  " + "-" * 74)
        if hr < 0.20:
            print(f"  high-risk short-form-only = {hr:.1%} < 20% -> your rule: ACCEPT AND "
                  "REPORT (concern was theoretical)")
        elif hr > 0.40:
            print(f"  high-risk short-form-only = {hr:.1%} > 40% -> your rule: DROP the "
                  "high-risk rows, report their relation-type distribution")
        else:
            print(f"  high-risk short-form-only = {hr:.1%} is between 20% and 40% -> your "
                  "rule: REPORT AND YOU DECIDE")
        print("  " + "-" * 74)

    print("\n" + "=" * 78)
    print("HUMAN READ: short_form is a heuristic. Read these before applying the rule.")
    print("=" * 78)
    shown = 0
    for r in rows:
        if r["stratum"] != "high_risk" or shown >= 15:
            continue
        # At most one per question, so 15 examples span 15 questions rather than two.
        for gen, cls in zip(r["generations"], r["classes"]):
            if cls == "short_form":
                print(f"\n  gold : {r['gold_aliases'][0]}")
                print(f"  gen  : {gen[:160]}")
                shown += 1
                break
    if not shown:
        print("\n  (no short_form generations in the high-risk stratum)")

    json.dump(out, open(gate1._path("instrument.json"), "w"), indent=2)
    print(f"\nwrote {gate1._path('instrument.jsonl')} and {gate1._path('instrument.json')}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        saved = gate1.read_jsonl("instrument.jsonl")
        if not saved:
            sys.exit("no instrument.jsonl -- run `python instrument.py` first")
        pool = gate1.read_jsonl("pool.jsonl")
        report(saved, len(pool) or FULL_POOL_DEFAULT)
    else:
        run()

"""Relation classification, answer-space size, and per-cell relation mix.

CPU only. Reads pool.jsonl (and labels.jsonl if present) from $GATE1_OUT, plus the raw
ConFiQA-QA.json to recover each row's relation, which pool.jsonl does not carry.

    python relstats.py data/ConFiQA-QA.json

Nothing here writes to the pipeline outputs; it is a reporting script.
"""
import ast
import json
import sys
from collections import Counter, defaultdict

import gate1
import backend

# --------------------------------------------------------------------------------------
# Classification, as settled with the human. FUNCTIONAL = one true object per subject.
# ONE_TO_MANY = the subject genuinely has several, and ConFiQA picked one, so "does the
# model know this fact" is not well posed (same argument PROTOCOL.md 1.1 uses to exclude
# multi-hop). TEMPORAL = functional at an instant but the holder changes over time;
# excluded from Gate 1, tagged and kept for a possible Gate 2 on legitimate updates.
# DROP = structurally defective in ConFiQA, see the notes.
# --------------------------------------------------------------------------------------
FUNCTIONAL = [
    "spouse", "composer", "country of citizenship", "director", "sport", "religion",
    "native language", "performer", "official language", "capital", "continent",
    "language of work or name", "headquarters location", "currency", "author",
]
ONE_TO_MANY = [
    "award received", "genre", "member of", "field of work", "member of sports team",
    "employer", "record label", "ethnic group", "work location", "position played",
    "position held", "developer", "owned by", "founded by",
    "country of origin",   # co-productions; Wikidata lists all of them
    "creator",             # multiple credited creators
]
TEMPORAL = [
    "head of state", "head of government", "chairperson", "head coach",
    "chief executive officer",
]
DROP = {
    "follows": "54% of rows have the gold answer named in the question (direction bug)",
    "country": "43% tautologous ('What country is Panama?')",
    "location": "too vague to be well posed",
    "capital of": "4/5 self-answering",
}

CLASS = {}
for r in FUNCTIONAL:
    CLASS[r] = "FUNCTIONAL"
for r in ONE_TO_MANY:
    CLASS[r] = "ONE_TO_MANY"
for r in TEMPORAL:
    CLASS[r] = "TEMPORAL"
for r in DROP:
    CLASS[r] = "DROP"


def relation_of(row: dict) -> str:
    """orig_path_labeled is "[('Bad Boys for Life', 'composer', 'Lorne Balfe')]"."""
    try:
        path = ast.literal_eval(row["orig_path_labeled"])
        return str(path[0][1])
    except Exception:
        return "?"


def self_answering(row: dict) -> bool:
    """ONE GLOBAL FILTER: a normalized gold alias appears in the question text.

    Deliberately the same matcher as labels and scoring (invariant #3). If alias_match
    fires on the question itself, the question contains its own answer and no knowledge
    is being measured.
    """
    return gate1.alias_match(row["question"], row["gold_aliases"])


def _table2_fallback(keep: list[dict], stats: list):
    """There is no labels.jsonl. Show what the two partial label sets can and cannot say.

    instrument.jsonl is 100 questions stratified 50 high-risk / 50 rest, so pool rates
    need reweighting by the real stratum share and per-relation cells are single digits.
    pairs_raw.jsonl is drawn only from relations with >=30% high-risk rows -- all of them
    ONE_TO_MANY -- and oversamples the high-risk side, so it says nothing about the
    surviving pool. Neither supports table 2; this block exists to make that concrete.
    """
    import instrument

    inst = gate1.read_jsonl("instrument.jsonl")
    print("\n" + "=" * 78)
    print("TABLE 2 -- NOT AVAILABLE: no labels.jsonl under $GATE1_OUT")
    print("=" * 78)
    print(f"  instrument.jsonl : {len(inst)} questions (stratified 50/50, not a sample "
          "of the pool)")
    print(f"  pairs_raw.jsonl  : {len(gate1.read_jsonl('pairs_raw.jsonl'))} questions "
          "(ONE_TO_MANY relations only, high-risk oversampled)")
    if not inst:
        return

    space = {rel: ratio for rel, _, _, ratio, _ in stats}
    keep_qids = {r["qid"] for r in keep}
    rel_by_qid = {r["qid"]: r["relation"] for r in keep}
    sub = [r for r in inst if r["qid"] in keep_qids]
    print(f"\n  instrument rows falling in the surviving pool: {len(sub)}")
    if not sub:
        return

    # Reweight the two strata back to their real shares before comparing.
    share_hr = sum(instrument.is_high_risk(r) for r in keep) / len(keep)
    print(f"  high-risk share of the surviving pool: {share_hr:.1%}\n")
    print(f"  {'answer space':<18} {'n':>4} {'known':>7} {'unknown':>8} {'discard':>8}")
    for name, lo, hi in (("large (>=0.60)", 0.60, 1.01), ("small (<0.60)", -1.0, 0.60)):
        rows = [r for r in sub if lo <= space.get(rel_by_qid[r["qid"]], 0) < hi]
        if not rows:
            continue
        parts = {}
        for k in ("known", "unknown", "discard"):
            acc = 0.0
            for st, w in (("high_risk", share_hr), ("rest", 1 - share_hr)):
                grp = [r for r in rows if r["stratum"] == st]
                if grp:
                    acc += w * sum(r["knowledge"] == k for r in grp) / len(grp)
            parts[k] = acc
        tot = sum(parts.values()) or 1.0
        print(f"  {name:<18} {len(rows):>4} {parts['known']/tot:>6.0%} "
              f"{parts['unknown']/tot:>7.0%} {parts['discard']/tot:>7.0%}")
    print("\n  Cells are single digits. Directional at best; not a basis for a decision.")


def main(confiqa_path: str):
    raw = json.load(open(confiqa_path))
    rel_by_qid = {backend._make_qid(r): relation_of(r) for r in raw}

    pool = gate1.read_jsonl("pool.jsonl")
    for r in pool:
        r["relation"] = rel_by_qid.get(r["qid"], "?")
        r["class"] = CLASS.get(r["relation"], "UNCLASSIFIED")
        r["self_answering"] = self_answering(r)

    n = len(pool)
    print(f"pool: {n} rows, {len({r['relation'] for r in pool})} relations\n")

    # ---- global self-answer filter ---------------------------------------------------
    sa = [r for r in pool if r["self_answering"]]
    print("=" * 78)
    print(f"GLOBAL SELF-ANSWER FILTER: {len(sa)} / {n} ({len(sa)/n:.1%}) dropped")
    print("=" * 78)
    by_rel = Counter(r["relation"] for r in sa)
    tot = Counter(r["relation"] for r in pool)
    for rel, k in by_rel.most_common():
        print(f"  {rel:<28} {k:>4} / {tot[rel]:>4}  ({k/tot[rel]:>5.1%})  [{CLASS.get(rel,'?')}]")

    # ---- class rollup ----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("CLASS ROLLUP")
    print("=" * 78)
    for cls in ("FUNCTIONAL", "ONE_TO_MANY", "TEMPORAL", "DROP", "UNCLASSIFIED"):
        rows = [r for r in pool if r["class"] == cls]
        surv = [r for r in rows if not r["self_answering"]]
        print(f"  {cls:<14} {len(rows):>5} rows -> {len(surv):>5} after self-answer filter")

    keep = [r for r in pool if r["class"] == "FUNCTIONAL" and not r["self_answering"]]
    print(f"\n  SURVIVING GATE 1 POOL: {len(keep)}")

    # ---- TABLE 1: answer-space size --------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE 1 -- ANSWER-SPACE SIZE (surviving FUNCTIONAL relations)")
    print("=" * 78)
    print(f"  {'relation':<28} {'rows':>5} {'distinct':>9} {'d/rows':>7}  {'top-5 answers'}")
    grouped = defaultdict(list)
    for r in keep:
        grouped[r["relation"]].append(r)
    stats = []
    for rel, rows in grouped.items():
        golds = [gate1.normalize(r["gold_aliases"][0]) for r in rows]
        c = Counter(golds)
        stats.append((rel, len(rows), len(c), len(c) / len(rows), c))
    for rel, nrows, ndist, ratio, c in sorted(stats, key=lambda x: -x[3]):
        top = ", ".join(f"{a}({k})" for a, k in c.most_common(5))
        print(f"  {rel:<28} {nrows:>5} {ndist:>9} {ratio:>7.2f}  {top[:70]}")

    # ---- TABLE 2: relation mix by cell -----------------------------------------------
    labels = gate1.read_jsonl("labels.jsonl")
    if not labels:
        _table2_fallback(keep, stats)
        return
    know = {r["qid"]: r["knowledge"] for r in labels}
    print("\n" + "=" * 78)
    print("TABLE 2 -- RELATION MIX BY CELL, surviving pool")
    print("  resistance draws from KNOWN questions, correction from UNKNOWN questions.")
    print("=" * 78)
    for scope, rows in (("SURVIVING (functional-only)", keep), ("CURRENT 4207 POOL", pool)):
        kn = [r for r in rows if know.get(r["qid"]) == "known"]
        un = [r for r in rows if know.get(r["qid"]) == "unknown"]
        dc = [r for r in rows if know.get(r["qid"]) == "discard"]
        print(f"\n--- {scope}: known {len(kn)} / unknown {len(un)} / discard {len(dc)} ---")
        ck, cu = Counter(r["relation"] for r in kn), Counter(r["relation"] for r in un)
        rels = sorted(set(ck) | set(cu), key=lambda x: -(ck[x] + cu[x]))
        print(f"  {'relation':<28} {'resist%':>8} {'correct%':>9} {'known-rate':>11} {'n':>6}")
        for rel in rels:
            a, b = ck[rel], cu[rel]
            if a + b == 0:
                continue
            print(f"  {rel:<28} {a/max(len(kn),1):>7.1%} {b/max(len(un),1):>8.1%} "
                  f"{a/(a+b):>10.1%} {a+b:>6}")
        # answer-space correlation, only meaningful on the surviving set
        if scope.startswith("SURVIVING"):
            print("\n  known-rate vs answer-space size (distinct/rows), by relation:")
            for rel, nrows, ndist, ratio, _ in sorted(stats, key=lambda x: -x[3]):
                a, b = ck[rel], cu[rel]
                if a + b == 0:
                    continue
                print(f"    {rel:<28} space={ratio:>5.2f}  known-rate={a/(a+b):>6.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/ConFiQA-QA.json")

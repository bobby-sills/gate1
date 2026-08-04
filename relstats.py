"""Relation classification, answer-space size, and per-cell relation mix.

CPU only. Reads pool.jsonl (and labels.jsonl if present) from $GATE1_OUT, plus the raw
ConFiQA-QA.json to recover each row's relation, which pool.jsonl does not carry.

    python relstats.py data/ConFiQA-QA.json

Nothing here writes to the pipeline outputs; it is a reporting script.
"""
import ast
import json
import random
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

# A relation counts as large-answer-space if its three commonest answers cover less
# than this share of its rows. Chosen to sit in the gap between `official language`
# (38%) and `currency` (21%); nothing downstream depends on it, it only labels a split.
LARGE_SPACE_TOP3 = 0.30

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


def space_stats(rows: list[dict]) -> dict:
    """relation -> answer-space summary over `rows`.

    `ratio` (distinct/rows) is biased downward by n: a relation with 25 rows cannot
    score low however open its answer space really is. `top3` -- the share of rows the
    three commonest answers cover -- is the guessability measure that does not depend
    on how many rows the relation happens to contribute.
    """
    g = defaultdict(list)
    for r in rows:
        g[r["relation"]].append(r)
    out = {}
    for rel, rs in g.items():
        c = Counter(gate1.normalize(r["gold_aliases"][0]) for r in rs)
        top = c.most_common(3)
        out[rel] = {"n": len(rs), "distinct": len(c), "ratio": len(c) / len(rs),
                    "top1": top[0][0], "top1_n": top[0][1],
                    "top3_n": sum(k for _, k in top), "counter": c}
    return out


def print_table1(rows: list[dict], title: str):
    st = space_stats(rows)
    print("\n" + "=" * 78)
    print(f"TABLE 1 -- ANSWER-SPACE SIZE: {title}")
    print("=" * 78)
    print(f"  {'relation':<26} {'rows':>5} {'distinct':>9} {'d/rows':>7} "
          f"{'top-1':>6} {'top-3':>6}  {'commonest answer'}")
    for rel, s in sorted(st.items(), key=lambda x: -x[1]["top3_n"] / x[1]["n"]):
        print(f"  {rel:<26} {s['n']:>5} {s['distinct']:>9} {s['ratio']:>7.2f} "
              f"{s['top1_n']/s['n']:>5.0%} {s['top3_n']/s['n']:>6.0%}  {s['top1'][:28]}")
    return st


def print_table2(rows: list[dict], know: dict, st: dict, title: str):
    """(1) cell mix, (2) per-relation label rates, (3) majority-class check.

    correction draws from questions labelled `unknown`, resistance from `known`
    (gate1.CELLS), so the two cells' relation mixes are just those two distributions.
    """
    for r in rows:
        r["_k"] = know.get(r["qid"])
    labelled = [r for r in rows if r["_k"]]
    kn = [r for r in labelled if r["_k"] == "known"]
    un = [r for r in labelled if r["_k"] == "unknown"]
    dc = [r for r in labelled if r["_k"] == "discard"]
    print("\n" + "=" * 78)
    print(f"TABLE 2 -- {title}")
    print(f"  {len(labelled)}/{len(rows)} labelled | known {len(kn)} "
          f"({len(kn)/max(len(labelled),1):.0%}) unknown {len(un)} "
          f"({len(un)/max(len(labelled),1):.0%}) discard {len(dc)} "
          f"({len(dc)/max(len(labelled),1):.0%})")
    print("=" * 78)
    if not labelled:
        return

    ck, cu = Counter(r["relation"] for r in kn), Counter(r["relation"] for r in un)
    rels = sorted(set(ck) | set(cu), key=lambda x: -(ck[x] + cu[x]))

    # (1) + (2)
    print("\n  (1) CELL MIX and (2) LABEL RATES, per relation")
    print(f"  {'relation':<26} {'n':>5} {'known':>6} {'unkn':>5} {'disc':>5}  "
          f"{'RESIST%':>8} {'CORRECT%':>9} {'delta':>7}")
    tvd = 0.0
    for rel in rels:
        sub = [r for r in labelled if r["relation"] == rel]
        a, b = ck[rel], cu[rel]
        d = sum(r["_k"] == "discard" for r in sub)
        pa, pb = a / max(len(kn), 1), b / max(len(un), 1)
        tvd += abs(pa - pb)
        print(f"  {rel:<26} {len(sub):>5} {a/len(sub):>5.0%} {b/len(sub):>4.0%} "
              f"{d/len(sub):>4.0%}  {pa:>7.1%} {pb:>8.1%} {pb-pa:>+7.1%}")
    # TVD is not 0 under the null -- with these cell sizes, random labels give ~0.07 just
    # from sampling. Permute the knowledge labels within the group to get the reference.
    obs = tvd / 2
    rng = random.Random(gate1.SEED)
    ks = [r["_k"] for r in labelled]
    rels_seq = [r["relation"] for r in labelled]
    null = []
    for _ in range(2000):
        rng.shuffle(ks)
        a = Counter(rl for rl, k in zip(rels_seq, ks) if k == "known")
        b = Counter(rl for rl, k in zip(rels_seq, ks) if k == "unknown")
        na, nb = sum(a.values()) or 1, sum(b.values()) or 1
        null.append(sum(abs(a[x]/na - b[x]/nb) for x in set(a) | set(b)) / 2)
    null.sort()
    p = sum(v >= obs for v in null) / len(null)
    print(f"\n  total variation distance between the two cells' relation mixes: {obs:.3f}")
    print(f"  permutation null (2000 shuffles): median {null[1000]:.3f}, "
          f"95th pct {null[1900]:.3f}, p = {p:.3f}")
    print("  (0 = identical mix, 1 = disjoint. This is the confound you flagged --")
    print("   but only the excess over the null is evidence of one.)")

    # (3) majority-class check
    print("\n  (3) MAJORITY-CLASS CHECK -- of rows labelled `known`, the share whose")
    print("      gold IS that relation's commonest answer, against its base rate.")
    print(f"  {'relation':<26} {'known':>6} {'base':>6} {'in known':>9} {'lift':>6}  "
          f"{'commonest answer'}")
    tot_known_maj = 0
    for rel in rels:
        s = st.get(rel)
        if not s or ck[rel] == 0:
            continue
        base = s["top1_n"] / s["n"]
        maj = sum(1 for r in kn if r["relation"] == rel
                  and gate1.normalize(r["gold_aliases"][0]) == s["top1"])
        tot_known_maj += maj
        share = maj / ck[rel]
        print(f"  {rel:<26} {ck[rel]:>6} {base:>5.0%} {share:>8.0%} "
              f"{share/base if base else 0:>6.2f}  {s['top1'][:26]}")
    print(f"\n  overall: {tot_known_maj}/{len(kn)} ({tot_known_maj/max(len(kn),1):.1%}) "
          "of the `known` set is its relation's majority class")


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

    # ---- TABLE 1 ----------------------------------------------------------------------
    st_keep = print_table1(keep, "surviving FUNCTIONAL relations")

    # large vs small split, and what each would leave. The floor is 300 known + 300
    # unknown QUESTIONS, not a row count -- each question yields two instances.
    big_rels = {rel for rel, s in st_keep.items()
                if s["top3_n"] / s["n"] < LARGE_SPACE_TOP3}
    large = [r for r in keep if r["relation"] in big_rels]
    small = [r for r in keep if r["relation"] not in big_rels]
    print(f"\n  large answer space (top-3 covers <{LARGE_SPACE_TOP3:.0%}): {len(large)}")
    print(f"  small answer space                        : {len(small)}")
    sp = Counter(r["relation"] for r in large)
    print(f"  spouse share of the large-space pool      : {sp['spouse']}/{len(large)} "
          f"({sp['spouse']/max(len(large),1):.1%})")

    one = [r for r in pool if r["class"] == "ONE_TO_MANY" and not r["self_answering"]]
    st_one = space_stats(one)

    # ---- TABLE 2 ----------------------------------------------------------------------
    labels = gate1.read_jsonl("labels.jsonl")
    if not labels:
        stats = [(rel, s["n"], s["distinct"], s["ratio"], s["counter"])
                 for rel, s in st_keep.items()]
        _table2_fallback(keep, stats)
        return
    know = {r["qid"]: r["knowledge"] for r in labels}
    print_table2(keep, know, st_keep, "SURVIVING POOL (FUNCTIONAL, self-answering removed)")
    print_table2(one, know, st_one, "ONE_TO_MANY ROWS -- the evidence for the exclusion")

    # ---- (4) the comparison the exclusion rests on -------------------------------------
    print("\n" + "=" * 78)
    print("(4) FUNCTIONAL vs ONE_TO_MANY, side by side")
    print("=" * 78)
    print(f"  {'group':<22} {'n':>6} {'known':>7} {'unknown':>8} {'discard':>8} "
          f"{'maj-class of known':>19}")
    for name, rows, st in (("FUNCTIONAL", keep, st_keep),
                           ("ONE_TO_MANY", one, st_one)):
        lab = [r for r in rows if know.get(r["qid"])]
        if not lab:
            continue
        kn = [r for r in lab if know[r["qid"]] == "known"]
        maj = sum(1 for r in kn
                  if gate1.normalize(r["gold_aliases"][0]) == st[r["relation"]]["top1"])
        print(f"  {name:<22} {len(lab):>6} "
              f"{sum(know[r['qid']]=='known' for r in lab)/len(lab):>6.0%} "
              f"{sum(know[r['qid']]=='unknown' for r in lab)/len(lab):>7.0%} "
              f"{sum(know[r['qid']]=='discard' for r in lab)/len(lab):>7.0%} "
              f"{maj/max(len(kn),1):>18.0%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/ConFiQA-QA.json")

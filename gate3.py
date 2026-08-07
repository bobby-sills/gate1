"""
gate3.py -- Gate 3a: answer verification, not question classification. See GATE3_PROTOCOL.md.

Gates 2 and 2b asked "does the model know this question?" and scored it ACROSS questions.
Gate 2b's post-mortem showed the resulting advantage was answer-entity memorisation: it
survived none of unseen-answer, unseen-answer-within-relation, or held-out-relation.

Gate 3a changes the unit to (question, candidate answer) and scores WITHIN a question, so
answer-identity and relation priors cancel structurally instead of being controlled for
afterwards. Methodology follows Inside-Out (Gekhman et al., arXiv:2503.15299).

    python gate3.py instrument   # candidate counts + cost, 100 questions, GPU
    python gate3.py extract      # Phase A, GPU, RESUMABLE
    python gate3.py checkpoint   # Phase A report only, re-runnable
    python gate3.py train        # Phase B knowledge-aware CV, CPU -> Checkpoint B
    python gate3.py tests        # Phase C, CPU

Reads GATE1_OUT, same as every other gate.
"""

import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

import gate1
import gate2

# ======================================================================================
# PRE-REGISTERED CONSTANTS -- frozen at Phase 0 (GATE3_PROTOCOL.md). Not tunable config.
# ======================================================================================

N_FOLDS = 5
N_CANDIDATES = 100                   # declared deviation from Inside-Out's 1000; see Phase A
C_GRID = gate2.C_GRID_WIDE           # inherited, as Gate 2b did
MAX_ITER = gate2.MAX_ITER
INNER_FRAC = gate2.INNER_FRAC

PROCEED_K_MARGIN = 0.03              # K(probe) - K(BEST external), absolute
MIN_RELATION_N = gate2.MIN_RELATION_N
N_BOOT = gate2.N_BOOT
SEED = gate1.SEED

SHARD = 250                          # questions per activation file
PARITY_WARN = gate2.PARITY_WARN
PARITY_STOP = gate2.PARITY_STOP

BASELINES = ("logp_sum", "logp_mean", "p_true")   # P(a|q), P_norm(a|q), P(True)

OOF_FILE = "probe3_oof.npz"
FOLDS_FILE = "probe3_folds.json"
RESULTS_FILE = "gate3_results.json"
PARITY_FILE = "extract3_parity.json"
INSTRUMENT_FILE = "instrument3.json"
CKPT_FOLDS = "probe3_folds_ckpt.jsonl"

RNG = np.random.default_rng(SEED)


def _shard_path(k: int) -> str:
    return gate1._path(f"acts3_shard{k:03d}.npz")


# ======================================================================================
# K -- the metric. Within question, never across.
# ======================================================================================

def k_per_question(y, s, qid):
    """Inside-Out eq. 2: fraction of (correct, incorrect) pairs the scorer ranks correctly,
    computed INSIDE each question. Ties count as half a win, which is what a rank-based
    pair count does.

    Questions with no incorrect candidate, or no correct one, have no pairs and are
    undefined -- they are dropped and counted, never scored as 0 or 1.

    What this does and does NOT cancel -- corrected 2026-08-06, before any Gate 3a data
    existed. Every comparison is between two answers to the SAME question, so anything
    constant within a question contributes exactly 0.5 and drops out: relation identity,
    question difficulty, entity popularity, the question's wording.

    ANSWER IDENTITY DOES NOT CANCEL. The candidates within a question are different answer
    strings, so a prior over answers still varies inside a pair and can rank it. That leak
    -- the one Gate 2b died of -- is closed by the subject- AND object-disjoint split
    (`disjoint_folds`, invariant 5), not by this metric. GATE3_PROTOCOL invariant 1 as
    frozen overstated this; see the correction recorded there.
    """
    out = {}
    for q in pd.unique(qid):
        m = qid == q
        yy = y[m]
        if yy.sum() == 0 or (1 - yy).sum() == 0:
            continue
        out[q] = gate2._auc(yy, s[m])
    return out


def k_score(y, s, qid):
    """K = mean over questions. K* = fraction of questions ranked perfectly."""
    per = k_per_question(y, s, qid)
    if not per:
        return float("nan"), float("nan"), 0
    v = np.array(list(per.values()))
    return float(v.mean()), float((v >= 1.0).mean()), len(v)


def boot_k_diff(y, s_a, s_b, qid, groups_of_q, n_boot=N_BOOT):
    """CI on K(a) - K(b), resampled by subject_qid.

    Resampling entities rather than questions matches the split unit, as in every other
    gate. Per-question K values are computed once and then resampled, since K is already
    a per-question statistic.
    """
    ka = k_per_question(y, s_a, qid)
    kb = k_per_question(y, s_b, qid)
    common = [q for q in ka if q in kb]
    by = {}
    for q in common:
        by.setdefault(groups_of_q[q], []).append(q)
    uniq = np.array(sorted(by))
    point = float(np.mean([ka[q] for q in common]) - np.mean([kb[q] for q in common]))
    diffs = []
    for _ in range(n_boot):
        qs = [q for g in RNG.choice(uniq, len(uniq), replace=True) for q in by[g]]
        diffs.append(np.mean([ka[q] for q in qs]) - np.mean([kb[q] for q in qs]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


# ======================================================================================
# PHASE A -- candidates + extraction
# ======================================================================================

def _candidate_set(r, backend):
    """The candidate answers for one question, with provenance flags.

    Union of: the greedy answer, N_CANDIDATES samples at T=1.0, and the gold answer if it
    is absent from both. Deduped on gate1.normalize -- the same matcher used for labels and
    scoring, per gate1 invariant 3.

    GOLD INJECTION IS THE POINT. Inside-Out needed it in 64% of cases, and without it the
    hidden-knowledge population -- facts the model represents but never generates -- is
    invisible by construction. The flag is retained so C3 can measure it.
    """
    # ONE answer_states call per question. It carries both the greedy answer this function
    # needs and the (entropy, max_prob) parity pair the caller needs, and it costs a greedy
    # generation plus two forward passes -- calling it twice per question doubled that.
    st = backend.answer_states(r["question"])
    greedy = st["answer"]
    sampled = backend.sample_candidates(r["question"], N_CANDIDATES, SEED)

    seen, cands = {}, []
    for a, src in [(greedy, "greedy")] + [(s, "sampled") for s in sampled]:
        k = gate1.normalize(a)
        if not k or k in seen:
            if k in seen and src == "greedy":
                cands[seen[k]]["is_greedy"] = True
            continue
        seen[k] = len(cands)
        cands.append({"answer": a, "is_greedy": src == "greedy", "gold_injected": False})

    gold_present = any(gate1.alias_match(c["answer"], r["gold_aliases"]) for c in cands)
    if not gold_present:
        cands.append({"answer": r["gold_aliases"][0], "is_greedy": False,
                      "gold_injected": True})

    for c in cands:
        c["correct"] = bool(gate1.alias_match(c["answer"], r["gold_aliases"]))
    return cands, gold_present, st


def phase_instrument(n: int = 100):
    """Measure candidate counts and cost before scaling. GATE3_PROTOCOL Phase A.

    The binding unknown is how many DISTINCT candidates survive dedup: storage is
    n_questions x n_candidates x 16 layers x 4096 x 2 bytes, and the protocol's estimate
    assumed ~10. If the real number is 30 the projection is 7GB, not 2.3GB, and that is a
    decision for a human before a full run rather than a surprise during one.
    """
    import backend

    pool = gate1.read_jsonl("pool.jsonl")
    t_cand = t_states = t_ptrue = 0.0
    counts, ncorrect, injected, rows = [], [], 0, []

    for r in pool[:n]:
        t0 = time.time()
        cands, gold_present, _ = _candidate_set(r, backend)
        t_cand += time.time() - t0
        injected += (not gold_present)
        counts.append(len(cands))
        ncorrect.append(sum(c["correct"] for c in cands))

        t0 = time.time()
        backend.candidate_states(r["question"], cands[0]["answer"])
        t_states += time.time() - t0
        t0 = time.time()
        backend.p_true(r["question"], cands[0]["answer"])
        t_ptrue += time.time() - t0
        rows.append({"qid": r["qid"], "n_cand": len(cands),
                     "n_correct": ncorrect[-1], "gold_injected": not gold_present})

    counts = np.array(counts)
    ncorrect = np.array(ncorrect)
    full = len(pool)
    per_pair = (t_states + t_ptrue) / n
    per_q = t_cand / n + per_pair * counts.mean()
    gb = full * counts.mean() * 16 * 4096 * 2 / 1e9

    out = {"n": n, "mean_candidates": float(counts.mean()),
           "median_candidates": float(np.median(counts)),
           "max_candidates": int(counts.max()),
           "gold_injection_rate": injected / n,
           "questions_with_no_incorrect": int((ncorrect == counts).sum()),
           "questions_with_no_correct": int((ncorrect == 0).sum()),
           "seconds_sampling_per_question": t_cand / n,
           "seconds_per_pair": per_pair,
           "projected_hours_full_pool": per_q * full / 3600,
           "projected_storage_gb": gb}
    print(json.dumps(out, indent=2))
    print(f"\n  candidates/question: mean {counts.mean():.1f}  median "
          f"{np.median(counts):.0f}  max {counts.max()}")
    print(f"  gold injection rate: {injected/n:.1%}   (Inside-Out reported 64%)")
    print(f"  questions with NO incorrect candidate: "
          f"{int((ncorrect == counts).sum())}/{n}  -- K undefined, these get dropped")
    print(f"\n  PROJECTED: {per_q * full / 3600:.1f} GPU-hours, {gb:.1f} GB storage")
    print("\n  HUMAN CHECK: storage is the binding constraint. Read the projection before "
          "running extract.")
    json.dump(out, open(gate1._path(INSTRUMENT_FILE), "w"), indent=2)
    return out


def phase_extract():
    import backend

    pool = gate1.read_jsonl("pool.jsonl")
    shards = [pool[i:i + SHARD] for i in range(0, len(pool), SHARD)]
    todo = [k for k in range(len(shards)) if not os.path.exists(_shard_path(k))]
    print(f"pool: {len(pool)} questions in {len(shards)} shards")
    print(f"resuming: {len(shards) - len(todo)} / {len(shards)} shards already complete")

    remaining = sum(len(shards[j]) for j in todo)
    t0, done = time.time(), 0
    for k in todo:
        H, meta = [], []
        for r in shards[k]:
            cands, gold_present, par = _candidate_set(r, backend)
            for c in cands:
                st = backend.candidate_states(r["question"], c["answer"])
                if st is None:
                    continue
                H.append(st["h_mean"])
                meta.append({"qid": r["qid"], "relation": r["relation"],
                             "subject_qid": r["subject_qid"],
                             "gold": r["gold_aliases"][0], **c,
                             "logp_sum": st["logp_sum"], "logp_mean": st["logp_mean"],
                             "p_true": backend.p_true(r["question"], c["answer"]),
                             "n_answer_tokens": st["n_answer_tokens"],
                             "q_entropy": par["entropy"], "q_max_prob": par["max_prob"]})
            done += 1

        tmp = _shard_path(k) + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, h=np.stack(H), meta=json.dumps(meta))
        os.replace(tmp, _shard_path(k))
        rate = (time.time() - t0) / max(done, 1)
        print(f"  shard {k}: {len(shards[k])} questions, {len(meta)} pairs, "
              f"{rate:.2f}s/q, eta {rate * (remaining - done) / 60:.0f}min")

    _checkpoint_a()


def load_acts():
    H, meta = [], []
    k = 0
    while os.path.exists(_shard_path(k)):
        z = np.load(_shard_path(k), allow_pickle=False)
        H.append(z["h"])
        meta.extend(json.loads(str(z["meta"])))
        k += 1
    if not H:
        raise FileNotFoundError("no shards -- run `gate3.py extract` first")
    return np.concatenate(H), pd.DataFrame(meta)


def _checkpoint_a():
    """HUMAN CHECKPOINT. Reports K for all three baselines BEFORE any probe is trained."""
    import backend

    H, meta = load_acts()
    nbytes = sum(os.path.getsize(_shard_path(k))
                 for k in range(10_000) if os.path.exists(_shard_path(k)))
    nq = meta.qid.nunique()

    print("\n" + "=" * 78)
    print("CHECKPOINT A -- extraction (Gate 3a)")
    print("=" * 78)
    print(f"  h {H.shape} {H.dtype}   on disk {nbytes/1e9:.2f} GB")
    print(f"  {len(meta)} (question, candidate) pairs over {nq} questions")

    per_q = meta.groupby("qid").agg(n=("answer", "size"), ncorr=("correct", "sum"))
    print(f"\n  candidates/question: mean {per_q.n.mean():.1f}  median {per_q.n.median():.0f}"
          f"  max {int(per_q.n.max())}")
    print(f"  gold injected:        {int(meta.gold_injected.sum())} questions  "
          f"{meta.gold_injected.sum()/nq:.1%}   (Inside-Out: 64%)")
    usable = per_q[(per_q.ncorr > 0) & (per_q.ncorr < per_q.n)]
    print(f"  usable for K:         {len(usable)}/{nq}  "
          f"({nq - len(usable)} dropped: no correct or no incorrect candidate)")

    # ---- parity, on the same three tiers ------------------------------------------------
    lab = pd.DataFrame(gate1.read_jsonl("labels.jsonl"))
    q = meta.drop_duplicates("qid")[["qid", "q_entropy", "q_max_prob"]]
    j = q.merge(lab, on="qid")
    d_ent = (j.q_entropy - j.entropy).abs()
    pool = gate1.read_jsonl("pool.jsonl")
    try:
        n_bad = sum(1 for r in pool
                    if backend._build_prompt(r["question"], None)
                    != backend.render_prompts(r["question"], r["factual_context"])[0])
        print(f"\n  prompt STRING identity: {len(pool)-n_bad}/{len(pool)}")
    except (ImportError, KeyError, OSError) as e:
        print(f"\n  prompt STRING identity: **NOT RUN** ({type(e).__name__})")
    print(f"  parity vs labels.jsonl: median |d entropy| {d_ent.median():.2e}   "
          f"max {d_ent.max():.2e}   exactly 0: {int((d_ent == 0).sum())}/{len(j)}")
    if d_ent.max() > PARITY_STOP:
        print("  STOP unless the string check passed -- see GATE2B_PROTOCOL Hardware.")

    # ---- K for the three baselines, no probe yet -----------------------------------------
    y = meta.correct.to_numpy().astype(int)
    qid = meta.qid.to_numpy()
    print("\n  BASELINE K (no probe trained yet):")
    ks = {}
    for b in BASELINES:
        k, kstar, n = k_score(y, meta[b].to_numpy(), qid)
        ks[b] = {"K": k, "K_star": kstar, "n_questions": n}
        print(f"    {b:<12} K={k:.4f}   K*={kstar:.4f}   (n={n})")
    best = max(ks, key=lambda b: ks[b]["K"])
    print(f"    BEST = {best} at K={ks[best]['K']:.4f}  ->  probe must reach "
          f"{ks[best]['K'] + PROCEED_K_MARGIN:.4f}")
    print("\n  HUMAN CHECK: read the above, then run `python gate3.py train`.")

    json.dump({"n_pairs": len(meta), "n_questions": int(nq), "bytes": int(nbytes),
               "mean_candidates": float(per_q.n.mean()),
               "gold_injection_rate": float(meta.gold_injected.sum() / nq),
               "usable_questions": int(len(usable)),
               "max_d_entropy": float(d_ent.max()), "baseline_K": ks, "best": best},
              open(gate1._path(PARITY_FILE), "w"), indent=2)


# ======================================================================================
# PHASE B -- knowledge-aware training
# ======================================================================================

def knowledge_aware_pairs(meta):
    """Inside-Out A.4. Training rows come ONLY from questions the model demonstrably knows.

    positive = the correct greedy answer
    negative = an incorrect candidate to the SAME question

    Gate 2b trained on categories (A) knows+correct and (D) doesn't-know+incorrect, which
    is why greedy correctness predicted its label at 0.986 AUC and why C0 ended with three
    usable rows. Restricting to questions where greedy is correct puts every training row
    in (A) or (B) knows+incorrect, so the probe cannot learn "this question is easy" -- all
    of them are.

    The negative is the most frequently sampled incorrect candidate, chosen deterministically
    so a re-run reproduces the training set exactly.
    """
    keep = []
    for q, g in meta.groupby("qid", sort=True):
        gre = g[g.is_greedy & g.correct]
        if gre.empty:
            continue                       # greedy wrong -> model may not know -> excluded
        wrong = g[~g.correct]
        if wrong.empty:
            continue
        keep.append(gre.index[0])
        keep.append(wrong.index[0])
    return np.array(sorted(keep))


def disjoint_folds(meta, n_folds=N_FOLDS, seed=SEED):
    """5 folds that are subject- AND object-disjoint. GATE3_PROTOCOL invariant 5.

    Gate 2b grouped by subject_qid alone, and the leakage check showed that is not enough:
    45.4% of test rows had their gold answer present in training, answer identity alone
    scored 0.7029 AUC, and inside `country of citizenship` the probe's edge swung from
    +0.1520 on seen answers to -0.0729 on unseen ones.

    Questions are nodes; an edge joins any two sharing a subject OR a gold object. Whole
    connected components are assigned to folds, largest first, always to the smallest fold.
    Components are indivisible, so if one is enormous the folds cannot balance -- that is
    reported, not worked around.
    """
    q = meta.drop_duplicates("qid")[["qid", "subject_qid", "gold"]].reset_index(drop=True)
    q["obj"] = [gate1.normalize(g) for g in q.gold]

    parent = {i: i for i in q.index}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for key in ("subject_qid", "obj"):
        for _, idx in q.groupby(key).groups.items():
            idx = list(idx)
            for i in idx[1:]:
                union(idx[0], i)

    comp = {}
    for i in q.index:
        comp.setdefault(find(i), []).append(q.qid[i])
    comps = sorted(comp.values(), key=len, reverse=True)

    sizes = [0] * n_folds
    assign = {}
    for c in comps:
        f = int(np.argmin(sizes))
        sizes[f] += len(c)
        for qq in c:
            assign[qq] = f
    return assign, comps, sizes


# ======================================================================================

PHASES = {"instrument": phase_instrument, "extract": phase_extract,
          "checkpoint": _checkpoint_a}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in PHASES:
        sys.exit(f"usage: python gate3.py [{' | '.join(PHASES)}]")
    print(f"gate3: N_CANDIDATES={N_CANDIDATES}  layers=10-25  "
          f"PROCEED margin K>=+{PROCEED_K_MARGIN}  (see GATE3_PROTOCOL.md)")
    PHASES[sys.argv[1]](*[int(a) for a in sys.argv[2:]])

"""
selftest3.py -- exercises Gate 3a's logic with a fake backend.

Run after every change to gate3.py or to backend's Gate 3 functions:

    python selftest3.py

Covers the K metric against hand-computed values, candidate-set construction including gold
injection and dedup, knowledge-aware pair selection, subject-AND-object-disjoint folds, and
extraction with a simulated crash and resume.

Gate 3a's whole defence is that scoring happens WITHIN a question. Most of this file is
checking that the leaks Gate 2b died of cannot operate: a fold split that lets an answer
cross, or a K that compares across questions, would silently reintroduce them.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

TMP = tempfile.mkdtemp(prefix="gate3-selftest-")
os.environ["GATE1_OUT"] = os.path.join(TMP, "outputs")
os.makedirs(os.environ["GATE1_OUT"], exist_ok=True)

import gate1                                                          # noqa: E402
import gate3                                                          # noqa: E402
import backend                                                        # noqa: E402

N_Q = 60
gate3.SHARD = 12
gate3.N_CANDIDATES = 6
gate3.N_BOOT = 200
rng = np.random.default_rng(5)

ok = True


def check(cond, label):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    ok = ok and bool(cond)


# ======================================================================================
# Fixtures
# ======================================================================================

RELATIONS = ["composer", "religion", "sport"]
# Deliberately low answer cardinality, like `continent` (4 answers / 46 questions), so the
# object-disjointness constraint actually binds and the fold builder is really tested.
OBJECTS = {"composer": [f"Comp{i}" for i in range(12)],
           "religion": ["Catholicism", "Islam", "Buddhism"],
           "sport": ["football", "tennis"]}


def build_pool():
    pool = []
    for i in range(N_Q):
        rel = RELATIONS[i % 3]
        objs = OBJECTS[rel]
        gold = objs[i % len(objs)]
        pool.append({"qid": f"q{i:03d}", "subject_qid": f"Q{i}", "relation": rel,
                     "question": f"What is the {rel} of Entity {i}?",
                     "gold_aliases": [gold],
                     "factual_context": "ctx", "counterfactual_context": "ctx"})
    with open(gate1._path("pool.jsonl"), "w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    with open(gate1._path("labels.jsonl"), "w") as f:
        for r in pool:
            f.write(json.dumps({"qid": r["qid"], "n_correct": 9, "knowledge": "known",
                                "entropy": 1.0, "max_prob": 0.5}) + "\n")
    return {r["qid"]: r for r in pool}


def make_fake_backend(fail_after=None):
    """Fake answer_states / sample_candidates / candidate_states / p_true.

    The planted signal is that CORRECT candidates carry +w in their activation. A probe that
    finds it should score K near 1; one that cannot should sit at 0.5.
    """
    calls = {"n": 0}
    w = rng.normal(size=16)

    def fake_answer_states(question, max_new_tokens=32):
        calls["n"] += 1
        if fail_after is not None and calls["n"] > fail_after:
            raise RuntimeError("simulated runtime recycle")
        r = _by_q[question]
        # 2 questions in 3 have greedy correct; the rest greedy-wrong (excluded from
        # knowledge-aware training, which is the behaviour under test)
        greedy = r["gold_aliases"][0] if int(r["qid"][1:]) % 3 else "wrongG"
        return {"answer": greedy, "entropy": 1.0, "max_prob": 0.5}

    def fake_sample_candidates(question, n, seed, temperature=1.0):
        r = _by_q[question]
        i = int(r["qid"][1:])
        out = ["wrongA", "wrongA", "wrongB"]           # dedup must collapse the repeat
        if i % 4:                                       # 3 in 4 also sample the gold
            out.append(r["gold_aliases"][0])
        return out[:n]

    def fake_candidate_states(question, answer, layer_lo=10, layer_hi=25):
        if not answer.strip():
            return None
        r = _by_q[question]
        correct = gate1.alias_match(answer, r["gold_aliases"])
        h = rng.normal(size=(16, 16)) + (1.6 if correct else -1.6) * w
        return {"h_mean": h.astype("float16"),
                "logp_sum": float(rng.normal(0.4 if correct else -0.4)),
                "logp_mean": float(rng.normal(0.3 if correct else -0.3)),
                "n_answer_tokens": 3}

    def fake_p_true(question, answer):
        r = _by_q[question]
        c = gate1.alias_match(answer, r["gold_aliases"])
        return float(np.clip(rng.normal(0.7 if c else 0.35, 0.1), 0, 1))

    return (fake_answer_states, fake_sample_candidates, fake_candidate_states,
            fake_p_true, calls)


print(f"scratch: {TMP}\n")
print("FIXTURES")
truth = build_pool()
_by_q = {r["question"]: r for r in truth.values()}
check(len(truth) == N_Q, f"built {N_Q} questions")

# ---- unit: K, against hand-computed values --------------------------------------------
print("\nUNIT -- K (within question)")
y = np.array([1, 0, 0, 1, 0])
qid = np.array(["a", "a", "a", "b", "b"])
# question a: correct scored above both wrongs -> 2/2 pairs = 1.0
# question b: correct scored below the wrong    -> 0/1 pairs = 0.0
s = np.array([3.0, 1.0, 2.0, 0.0, 5.0])
k, ks, n = gate3.k_score(y, s, qid)
check(abs(k - 0.5) < 1e-12, f"K = mean(1.0, 0.0) = {k:.4f}")
check(abs(ks - 0.5) < 1e-12, f"K* = fraction perfect = {ks:.4f}")
check(n == 2, f"{n} questions scored")

# a question with no incorrect candidate has no pairs and must be DROPPED, not scored
y2 = np.concatenate([y, [1, 1]])
q2 = np.concatenate([qid, ["c", "c"]])
s2 = np.concatenate([s, [1.0, 2.0]])
_, _, n2 = gate3.k_score(y2, s2, q2)
check(n2 == 2, "single-class question dropped from K, not scored as 1.0")

# K must be blind to a constant per-question offset -- that is the whole design
off = s + np.where(qid == "a", 100.0, 0.0)
k_off, _, _ = gate3.k_score(y, off, qid)
check(abs(k_off - k) < 1e-12,
      "K unchanged by a per-question offset -- answer/relation priors cancel")

# ---- unit: candidate set ---------------------------------------------------------------
print("\nUNIT -- candidate set, dedup and gold injection")
f_ans, f_samp, f_cand, f_pt, _ = make_fake_backend()
backend.answer_states, backend.sample_candidates = f_ans, f_samp
backend.candidate_states, backend.p_true = f_cand, f_pt

r0 = truth["q000"]          # 0 % 3 == 0 -> greedy WRONG; 0 % 4 == 0 -> gold NOT sampled
c0, present0, _ = gate3._candidate_set(r0, backend)
answers0 = [c["answer"] for c in c0]
check(len(answers0) == len(set(map(gate1.normalize, answers0))), "candidates deduped")
check(not present0 and any(c["gold_injected"] for c in c0),
      "gold injected when never generated")
check(sum(c["correct"] for c in c0) == 1, "exactly one correct candidate after injection")

r1 = truth["q001"]          # 1 % 3 -> greedy CORRECT; 1 % 4 -> gold sampled
c1, present1, _ = gate3._candidate_set(r1, backend)
check(present1 and not any(c["gold_injected"] for c in c1),
      "gold NOT injected when the model already produced it")
check(sum(c["is_greedy"] for c in c1) == 1, "exactly one candidate flagged greedy")

# ---- extract, with a crash partway ------------------------------------------------------
print("\nPHASE A -- extract, crash, resume")
f_ans, f_samp, f_cand, f_pt, calls1 = make_fake_backend(fail_after=20)
backend.answer_states, backend.sample_candidates = f_ans, f_samp
backend.candidate_states, backend.p_true = f_cand, f_pt
try:
    gate3.phase_extract()
    check(False, "simulated crash raised")
except RuntimeError:
    check(True, "simulated crash raised")

before = len([k for k in range(20) if os.path.exists(gate3._shard_path(k))])
check(before == 1, f"{before} whole shard survived the crash")

f_ans, f_samp, f_cand, f_pt, calls2 = make_fake_backend()
backend.answer_states, backend.sample_candidates = f_ans, f_samp
backend.candidate_states, backend.p_true = f_cand, f_pt
gate3.phase_extract()
check(calls2["n"] == N_Q - before * gate3.SHARD,
      f"resume re-extracted {calls2['n']} questions, not {N_Q} "
      f"(one answer_states call per question, not two)")

H, meta = gate3.load_acts()
check(H.shape[0] == len(meta) and H.shape[1:] == (16, 16),
      f"activations {H.shape} aligned with {len(meta)} pairs")
check(meta.qid.nunique() == N_Q, "every question present")
check(meta.gold_injected.sum() > 0, "some questions needed gold injection")
check(os.path.exists(gate1._path(gate3.PARITY_FILE)), "wrote the Checkpoint A artifact")

# ---- knowledge-aware pairs --------------------------------------------------------------
print("\nPHASE B -- knowledge-aware training pairs")
idx = gate3.knowledge_aware_pairs(meta)
sub = meta.loc[idx]
check(len(idx) > 0, f"selected {len(idx)} training rows")
check(set(sub.correct) == {True, False}, "both a positive and a negative per question")
by_q = sub.groupby("qid").correct.agg(["sum", "size"])
check((by_q["sum"] == 1).all() and (by_q["size"] == 2).all(),
      "exactly one correct and one incorrect row per training question")
greedy_ok = meta[meta.is_greedy & meta.correct].qid.unique()
check(set(sub.qid) <= set(greedy_ok),
      "training questions are ONLY those the model demonstrably knows (Inside-Out A.4)")
check(sub[sub.correct].is_greedy.all(), "every positive is the correct greedy answer")

# ---- disjoint folds ----------------------------------------------------------------------
print("\nPHASE B -- subject- AND object-disjoint folds")
assign, comps, sizes = gate3.disjoint_folds(meta)
q = meta.drop_duplicates("qid")[["qid", "subject_qid", "gold"]]
q["obj"] = [gate1.normalize(g) for g in q.gold]
q["fold"] = [assign[x] for x in q.qid]
leak_s = q.groupby("subject_qid").fold.nunique().max()
leak_o = q.groupby("obj").fold.nunique().max()
check(leak_s == 1, "no subject_qid spans two folds")
check(leak_o == 1, "no gold OBJECT spans two folds -- the Gate 2b leak, closed")
check(sum(sizes) == N_Q, f"every question assigned (fold sizes {sizes})")
print(f"    {len(comps)} connected components, largest {len(comps[0])} questions")

# ---- what K does and does not cancel -------------------------------------------------------
# The precise structural claim, checked rather than asserted. A scorer that is CONSTANT
# within each question -- any question-level confound: relation, difficulty, entity
# popularity -- must score exactly 0.5, because every pair it sees is a tie.
print("\nUNIT -- what K cancels")
yv, qv = meta.correct.to_numpy().astype(int), meta.qid.to_numpy()
qlevel = {q: rng.normal() for q in set(qv)}
k_q, _, _ = gate3.k_score(yv, np.array([qlevel[q] for q in qv]), qv)
check(abs(k_q - 0.5) < 1e-12,
      f"question-level scorer gets K={k_q:.4f} exactly -- relation and difficulty cancel")

# ANSWER identity does NOT cancel: candidates within a question are different strings, so
# a per-answer prior still varies inside a pair. This is why object-disjoint folds are
# load-bearing rather than belt-and-braces -- see the correction in GATE3_PROTOCOL.
rank = {a: i for i, a in enumerate(sorted(set(meta.answer)))}
k_a, _, _ = gate3.k_score(yv, np.array([float(rank[a]) for a in meta.answer]), qv)
check(abs(k_a - 0.5) > 0.1,
      f"answer-identity scorer gets K={k_a:.4f}, far from 0.5 -- it does NOT cancel, "
      f"so the disjoint split is what closes that leak")

# ======================================================================================
print()
if ok:
    shutil.rmtree(TMP)
    print("selftest3: ALL PASS")
else:
    print(f"selftest3: FAILURES -- scratch kept at {TMP}")
sys.exit(0 if ok else 1)

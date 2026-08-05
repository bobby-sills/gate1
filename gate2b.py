"""
gate2b.py -- Gate 2b: probing at the ANSWER tokens. See GATE2B_PROTOCOL.md.

Gate 2 read the residual stream at the final question token and found a linear probe adds
+0.0095 AUC over H(p_theta), CI spanning zero. The literature locates its signal at the
answer tokens, which do not exist at that position. Gate 2b reads there -- and moves the
baselines there too, because a repositioned probe against an unrepositioned baseline
manufactures its own result.

    python gate2b.py instrument   # cost measurement on 100 questions, GPU
    python gate2b.py extract      # Phase A  greedy + states + samples, GPU, RESUMABLE
    python gate2b.py checkpoint   # Phase A  report only, re-runnable
    python gate2b.py train        # Phase B  nested CV, CPU -> Checkpoint B
    python gate2b.py tests        # Phase C  C0-C4 + statistics, CPU

Reads GATE1_OUT for everything, same as gate1.py and gate2.py. Set it in a PYTHON cell:
    os.environ['GATE1_OUT'] = '/content/drive/MyDrive/gate1/outputs'
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
# PRE-REGISTERED CONSTANTS -- frozen at Phase 0 (GATE2B_PROTOCOL.md). Not tunable config.
# ======================================================================================

N_FOLDS = 5

# INHERITED FROM A POST-HOC ANALYSIS, declared. This grid was not pre-registered in Gate
# 2: it was widened after seeing a failing Gate 2 result, because 5/5 folds had selected
# the boundary value 1e-3. Inheriting it into a NEW pre-registration is legitimate -- it
# is fixed before any Gate 2b data is touched, so it cannot be tuned toward a 2b outcome
# -- but the provenance is recorded rather than laundered. It is also known to be inert:
# in Gate 2 the widened grid reproduced the pre-registered result exactly. It is inherited
# anyway so "the grid boundary was binding" cannot be raised against a 2b null later.
C_GRID = gate2.C_GRID_WIDE

# p1 first answer token, p2 last answer token, p3 mean-pooled over the answer span.
POSITIONS = ("first", "last", "mean")

# C4 only. Gate 2's read position, carried onto the depth curve as a reference line. It is
# NOT in the selection space -- GATE2B_PROTOCOL registers the probe's positions as p1-p3,
# and letting the probe select Gate 2's position would make 2b a superset of Gate 2 rather
# than a test of repositioning.
REFERENCE_POSITION = "gate2_final"

N_SAMPLES = gate1.N_SAMPLES          # 10, for b4
SHARD = 250                          # questions per activation file
MAX_ITER = gate2.MAX_ITER

PROCEED_AUC_MARGIN = 0.03            # probe - BEST baseline, pooled entity-disjoint CV
WITHIN_RELATION_MARGIN = 0.02        # C1 is a VETO -- locked 2026-08-05, see Phase 0
MIN_RELATION_N = gate2.MIN_RELATION_N
MIN_MINORITY_N = gate2.MIN_MINORITY_N

N_BOOT = gate2.N_BOOT
SEED = gate1.SEED

# Pooled out-of-fold AUC concatenates five separately fitted probes' raw decision values
# and ranks them against each other. That is only meaningful while each fold's scores span
# a range large compared with the offsets BETWEEN folds. Under strong regularization the
# coefficients shrink toward zero while the (unpenalised) intercept does not, so the spread
# can invert -- at C=1e-5 on the selftest fixture every fold scores AUC 1.0000 and the
# pooled number is 0.5024. When per-fold and pooled AUC diverge by more than this, the
# primary number is measuring cross-fold calibration drift, not separability, and gate2b
# refuses to report a verdict off it. See `pooling_diagnostic`.
POOLING_DIVERGENCE = 0.02

PARITY_WARN = gate2.PARITY_WARN
PARITY_STOP = gate2.PARITY_STOP

BASELINES = ("b1_entropy_qfinal", "b2_entropy_matched", "b3_mean_logprob",
             "b4_self_consistency")

OOF_FILE = "probe2b_oof.npz"
FOLDS_FILE = "probe2b_folds.json"
RESULTS_FILE = "gate2b_results.json"
PARITY_FILE = "extract2b_parity.json"
INSTRUMENT_FILE = "instrument2b.json"


def _bind_gate2(positions=POSITIONS):
    """Point gate2's module-level knobs at Gate 2b's, LOUDLY.

    gate2b reuses gate2's statistics (DeLong, the grouped bootstrap), its nested-CV driver
    and its checkpointing rather than copying them. Those read gate2's module globals, so
    they are rebound here. Copying instead would duplicate a unit-tested DeLong
    implementation into a second file, and two copies diverge -- the same reasoning as
    gate1 invariant 8 on the two cache layers.

    The rebinding is explicit at each call site and printed at startup so that reading
    `C_GRID = (1e-3, ...)` at the top of gate2.py cannot mislead anyone about what a
    gate2b run actually used.
    """
    gate2.C_GRID = C_GRID
    gate2.POSITIONS = positions
    gate2.N_FOLDS = N_FOLDS
    # Separate checkpoint files. The fingerprint covers the row set but not the grid or
    # the position space, so shared files would silently serve Gate 2's folds to Gate 2b.
    gate2.CKPT_FOLDS = "probe2b_folds_ckpt.jsonl"
    gate2.CKPT_CURVE = "c4b_curve_ckpt.jsonl"
    gate2.CKPT_LORO = "c2b_loro_ckpt.jsonl"


def _shape_key(X):
    """gate2's drivers read the layer count as `X["final"].shape[1]` -- Gate 2's first
    position, hard-coded. Gate 2b's positions are named differently, so a shape-only alias
    is added here.

    It is never iterated: every loop in those drivers walks `gate2.POSITIONS`, which
    `_bind_gate2` has already pointed at Gate 2b's tuple, and all three Gate 2b positions
    have identical shape so the alias cannot change what n_layers is. The alternative --
    editing gate2.py to generalise one lookup -- touches a file whose results are
    pre-registered and already reported, for no behavioural gain.
    """
    return {**X, "final": X[POSITIONS[0]]}


# ======================================================================================
# PHASE A -- extraction
# ======================================================================================

def _shard_path(k: int) -> str:
    return gate1._path(f"acts2b_shard{k:03d}.npz")


def self_consistency(samples) -> float:
    """b4. Fraction of the ten samples matching the modal normalised answer.

    GOLD IS NEVER CONSULTED. That is the whole point: n_correct, the one sample-derived
    quantity Phase 2 wrote to disk, IS the label (>=8 known, <=2 unknown), so any predictor
    built from it scores AUC 1.0 by construction. This measures whether the model agrees
    with itself, which is a genuine free self-knowledge signal.

    Normalisation is gate1.normalize -- the same matcher used for labels and scoring
    (gate1 invariant 3). Empty generations normalise to "" and therefore count as mutually
    consistent; that is faithful to what the model did and is reported, not special-cased.
    """
    norm = [gate1.normalize(s) for s in samples]
    return Counter(norm).most_common(1)[0][1] / len(norm)


def _extract_one(r, backend):
    """One question: greedy answer + answer-token states, then the ten Phase 2 samples.

    The two are SEPARATE calls on purpose. sample_closed_book is the label pass and
    answer_states is the predictor pass; gate1 invariant 2 forbids sharing computation
    between them, and nothing here is shared.
    """
    res = backend.answer_states(r["question"])
    samples = backend.sample_closed_book(r["question"], N_SAMPLES, SEED)
    meta = {
        "qid": r["qid"],
        "entropy": res["entropy"], "max_prob": res["max_prob"],
        "n_answer_tokens": res["n_answer_tokens"], "answer": res["answer"],
        "first_token_is_argmax": res["first_token_is_argmax"],
        "b2_first": res["b2_first"], "b2_last": res["b2_last"],
        "b2_mean": res["b2_mean"], "b3_mean_logprob": res["b3_mean_logprob"],
        "b4_self_consistency": self_consistency(samples),
        # Recomputed from the regenerated samples. Phase A asserts this equals
        # labels.jsonl -- that assertion is what makes "the same ten strings" checkable
        # rather than assumed. See GATE2B_PROTOCOL, b4.
        "n_correct_recomputed": sum(gate1.alias_match(g, r["gold_aliases"])
                                    for g in samples),
        "greedy_correct": bool(gate1.alias_match(res["answer"], r["gold_aliases"])),
        # Persisted this time. Phase 2 discarded them, which is the reason b4 had to be
        # regenerated at all.
        "samples": samples,
    }
    return res, meta


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
        got, meta = [], []
        for r in shards[k]:
            res, m = _extract_one(r, backend)
            got.append(res)
            meta.append(m)
            done += 1

        # The placeholder is sized from a row that HAS a span, after the shard is
        # collected -- the first question in a shard may not have one. Spanless rows are
        # stored as zeros only to keep the arrays index-aligned with `meta`; `mask` is what
        # excludes them, and build_matrix drops them before any fit touches them.
        shape = next((r["h_first"].shape for r in got if r["h_first"] is not None), None)
        if shape is None:                       # whole shard spanless: ask the model
            _, model = backend._load()
            shape = (model.config.num_hidden_layers + 1, model.config.hidden_size)
        zero = np.zeros(shape, dtype="float16")
        mask = np.array([r["h_first"] is not None for r in got])
        H = {p: np.stack([r[f"h_{p}"] if r["h_first"] is not None else zero for r in got])
             for p in POSITIONS}

        tmp = _shard_path(k) + ".tmp"
        with open(tmp, "wb") as fh:      # a handle, not a name: savez appends ".npz"
            np.savez(fh, mask=mask, meta=json.dumps(meta),
                     **{f"h_{p}": H[p] for p in POSITIONS})
        os.replace(tmp, _shard_path(k))
        rate = (time.time() - t0) / max(done, 1)
        print(f"  shard {k}: {len(shards[k])} questions, {rate:.2f}s/q, "
              f"eta {rate * (remaining - done) / 60:.0f}min")

    _checkpoint_a()


def load_acts():
    """All shards -> (dict position -> [n, n_layers, hidden], mask, meta_df)."""
    H = {p: [] for p in POSITIONS}
    mk, meta = [], []
    k = 0
    while os.path.exists(_shard_path(k)):
        z = np.load(_shard_path(k), allow_pickle=False)
        for p in POSITIONS:
            H[p].append(z[f"h_{p}"])
        mk.append(z["mask"])
        meta.extend(json.loads(str(z["meta"])))
        k += 1
    if not mk:
        raise FileNotFoundError("no activation shards -- run `gate2b.py extract` first")
    return ({p: np.concatenate(v) for p, v in H.items()},
            np.concatenate(mk), pd.DataFrame(meta))


def _checkpoint_a():
    """Span stats, parity, the two free assertions, and pooled AUC for ALL FOUR baselines
    BEFORE a probe is trained.

    HUMAN CHECKPOINT. GATE2B_PROTOCOL Phase A: report and stop. The last block is the point
    of the checkpoint -- if b4 alone is already at 0.87 we know it before spending anything
    on training.
    """
    # Tokenizer only -- _build_prompt goes through _load_tok, not _load, so `checkpoint`
    # still runs on a CPU session with no model download.
    import backend

    H, mask, meta = load_acts()
    # The stored mask and the stored span length are written by the same loop but from
    # different fields; if they ever disagree, one of the two is stale and every row index
    # downstream is suspect.
    assert (mask == (meta.n_answer_tokens.to_numpy() > 0)).all(), \
        "shard mask disagrees with n_answer_tokens -- do not trust the row alignment"
    n_layers, hidden = H["first"].shape[1], H["first"].shape[2]
    nbytes = sum(os.path.getsize(_shard_path(k))
                 for k in range(10_000) if os.path.exists(_shard_path(k)))

    print("\n" + "=" * 78)
    print("CHECKPOINT A -- extraction (Gate 2b)")
    print("=" * 78)
    for p in POSITIONS:
        print(f"  h_{p:<6} {H[p].shape} {H[p].dtype}")
    print(f"  layers {n_layers} (embedding + {n_layers - 1} blocks), hidden {hidden}")
    print(f"  on disk    {nbytes / 1e9:.2f} GB")

    # ---- answer spans -----------------------------------------------------------------
    m = meta.n_answer_tokens
    n_none = int((m == 0).sum())
    n_one = int((m == 1).sum())
    print(f"\n  answer span length: median {m[m > 0].median():.0f}  "
          f"mean {m[m > 0].mean():.2f}  max {int(m.max())}")
    print(f"    no usable span   {n_none:5d}  {n_none / len(meta):6.2%}  (dropped)")
    print(f"    m == 1           {n_one:5d}  {n_one / len(meta):6.2%}  "
          f"(p1, p2 and p3 coincide for these)")
    if n_one / max(len(meta), 1) > 0.5:
        print("    NOTE: over half the spans are one token. The three read positions are "
              "then\n          not really three positions -- read C4 with that in mind.")

    # ---- parity ------------------------------------------------------------------------
    lab = pd.DataFrame(gate1.read_jsonl("labels.jsonl"))
    j = meta.merge(lab, on="qid", suffixes=("_new", "_old"))
    d_ent = (j.entropy_new - j.entropy_old).abs()
    d_mp = (j.max_prob_new - j.max_prob_old).abs()
    # The STRING-level check first, because it is the one that is hardware-independent.
    # The numeric check below is only a PROXY for "the prompt is unchanged": identical
    # prompt => identical numbers, ON IDENTICAL HARDWARE. Run on a different GPU
    # architecture the proxy stops working -- it can no longer separate "the prompt
    # changed" from "the kernels changed". Measured L4 -> A100 on this pool, 2026-08-05:
    # median |d entropy| 1.27e-2 with a byte-identical prompt, which is above PARITY_STOP
    # on its own. GATE2_PROTOCOL Phase A registers this check; gate2b did not run it until
    # that failure made the omission expensive.
    pool = gate1.read_jsonl("pool.jsonl")
    try:
        n_bad = sum(1 for r in pool
                    if backend._build_prompt(r["question"], None)
                    != backend.render_prompts(r["question"], r["factual_context"])[0])
        print(f"\n  prompt STRING identity vs render_prompts: "
              f"{len(pool) - n_bad} / {len(pool)}"
              + ("" if n_bad == 0 else f"   <-- {n_bad} MISMATCHES"))
    except (ImportError, KeyError, OSError) as e:
        # Loud, not silent. This check is what distinguishes a template change from a
        # hardware change, so "it did not run" must never read like "it passed".
        n_bad = None
        print(f"\n  prompt STRING identity: **NOT RUN** ({type(e).__name__}) -- needs the "
              f"real tokenizer\n    and pool.jsonl's factual_context. Without it the "
              f"numeric parity below cannot\n    tell a template change from a GPU change.")

    print(f"\n  prompt parity vs labels.jsonl on {len(j)} questions")
    print(f"    median |d entropy| = {d_ent.median():.2e}")
    print(f"    max    |d entropy| = {d_ent.max():.2e}   (warn {PARITY_WARN:.0e}, "
          f"stop {PARITY_STOP:.0e})")
    print(f"    max    |d max_prob| = {d_mp.max():.2e}")
    print(f"    exactly 0.0        = {int((d_ent == 0).sum())} / {len(j)}")
    if d_ent.max() > PARITY_STOP and n_bad == 0:
        print("\n  STOP, but read the two lines above together. The prompt is BYTE-IDENTICAL")
        print("  and the numbers still differ, so this is a hardware change, not a template")
        print("  change. Phase 2 and Gate 2's extraction ran on one GPU; if this is another,")
        print("  b4 cannot recover the label-producing samples either. Re-extract on the")
        print("  Phase 2 hardware rather than relaxing this threshold.")
    elif d_ent.max() > PARITY_STOP:
        print("\n  STOP: the extraction prompt is NOT the Phase 2 prompt. Do not train.")
    elif d_ent.max() > PARITY_WARN:
        print("\n  PASS with a note: consistent with bf16 kernel nondeterminism across "
              "runtimes.")
    else:
        print("\n  PASS: numerically identical.")

    # ---- the two free assertions --------------------------------------------------------
    span = meta[meta.n_answer_tokens > 0]
    argmax_rate = float(span.first_token_is_argmax.mean()) if len(span) else float("nan")
    n_mismatch = int((j.n_correct_recomputed != j.n_correct).sum())
    print(f"\n  greedy first token == argmax at the question-final token: "
          f"{argmax_rate:.4%} of spans")
    print(f"  regenerated n_correct == labels.jsonl: "
          f"{len(j) - n_mismatch} / {len(j)}"
          + ("" if n_mismatch == 0 else f"   <-- {n_mismatch} MISMATCHES"))
    if n_mismatch:
        print("    STOP: the ten samples are NOT the ones that produced the labels, so b4")
        print("    is not the registered baseline. Do not train.")

    # ---- labels and greedy correctness ---------------------------------------------------
    y = gate2._labels()
    jl = y.merge(meta, on="qid", suffixes=("", "_x"))
    jl = jl[jl.n_answer_tokens > 0]
    print(f"\n  after dropping discards and rows with no span: n={len(jl)}  "
          f"known={int((jl.knowledge == 'known').sum())}  "
          f"unknown={int((jl.knowledge == 'unknown').sum())}")
    print(f"  distinct subject entities: {jl.subject_qid.nunique()}")
    print(f"  greedy generation correct: {jl.greedy_correct.mean():.2%} overall")
    for k_ in ("known", "unknown"):
        s = jl[jl.knowledge == k_]
        print(f"    among {k_:<8} {s.greedy_correct.mean():.2%}  (n={len(s)})")

    # ---- pooled baseline AUCs, before any probe exists -----------------------------------
    yv = (jl.knowledge == "known").to_numpy().astype(int)
    raw = {"b1_entropy_qfinal": -jl.entropy.to_numpy(),
           "b2_entropy_first": -jl.b2_first.to_numpy(),
           "b2_entropy_last": -jl.b2_last.to_numpy(),
           "b2_entropy_mean": -jl.b2_mean.to_numpy(),
           "b3_mean_logprob": jl.b3_mean_logprob.to_numpy(),
           "b4_self_consistency": jl.b4_self_consistency.to_numpy()}
    print("\n  POOLED BASELINE AUC (no probe trained yet):")
    aucs = {}
    for k_, v in raw.items():
        aucs[k_] = gate2._auc(yv, v)
        print(f"    {k_:<22} {aucs[k_]:.4f}")
    print("    b4 is flattered: it and the label are two functions of the same ten "
          "generations.\n    Direction is conservative -- it raises the bar the probe "
          "must clear -- but the\n    number is never quoted without this sentence.")
    b1b2 = abs(aucs["b1_entropy_qfinal"] - aucs["b2_entropy_first"])
    print(f"\n  b1 vs b2_first differ by {b1b2:.4f}"
          + ("   <-- SUSPICIOUS: they are one token apart, not the same position"
             if b1b2 < 1e-9 else ""))
    print("\n  HUMAN CHECK: read the above, then run `python gate2b.py train`.")

    json.dump({"n": int(len(meta)), "n_layers": int(n_layers), "hidden": int(hidden),
               "bytes": int(nbytes), "n_no_span": n_none, "n_span_one": n_one,
               "max_d_entropy": float(d_ent.max()), "max_d_max_prob": float(d_mp.max()),
               "argmax_rate": argmax_rate, "n_correct_mismatches": n_mismatch,
               "greedy_correct_rate": float(jl.greedy_correct.mean()),
               "pooled_baseline_auc": aucs, "n_labelled_with_span": int(len(jl))},
              open(gate1._path(PARITY_FILE), "w"), indent=2)


def phase_instrument(n: int = 100):
    """Measure the marginal cost. Do not assert it -- GATE2B_PROTOCOL Phase A.

    Times the three components separately, because two of them are different kinds of
    cost: the greedy generation is part of the MECHANISM and would be paid at decode time,
    while the ten samples are a cost of measuring b4 IN THIS EXPERIMENT ONLY. Reporting one
    number would let them be added together and the mechanism blamed for both.
    """
    import backend

    pool = gate1.read_jsonl("pool.jsonl")
    full = len(pool)
    t_states = t_samples = 0.0
    spans = []
    for r in pool[:n]:
        t0 = time.time()
        res = backend.answer_states(r["question"])
        t_states += time.time() - t0
        t0 = time.time()
        backend.sample_closed_book(r["question"], N_SAMPLES, SEED)
        t_samples += time.time() - t0
        spans.append(res["n_answer_tokens"])

    per_q = (t_states + t_samples) / n
    out = {"n": n, "seconds_per_question_total": per_q,
           "seconds_per_question_mechanism": t_states / n,
           "seconds_per_question_b4_samples": t_samples / n,
           "projected_minutes_full_pool": per_q * full / 60,
           "projected_minutes_mechanism_only": (t_states / n) * full / 60,
           "mean_answer_tokens": float(np.mean(spans))}
    print(json.dumps(out, indent=2))
    print(f"\n  MECHANISM cost: one extra short greedy generation per question, "
          f"{t_states / n:.3f}s.")
    print(f"  Mean answer length {np.mean(spans):.1f} tokens. Frame this against the two "
          f"forward\n  passes per token the contrastive decoder already pays -- the claim "
          f"is 'one extra\n  short generation per question', not 'twice as expensive'.")
    print(f"  b4's ten samples ({t_samples / n:.3f}s/q) are a cost of MEASURING b4 here, "
          f"not of the\n  mechanism. Do not add the two together.")
    json.dump(out, open(gate1._path(INSTRUMENT_FILE), "w"), indent=2)


# ======================================================================================
# PHASE B -- matrix, folds, probes
# ======================================================================================

def build_matrix():
    """(X, y, groups, rel, baselines, qids, extras) on ONE fixed row set.

    Rows: FUNCTIONAL pool, discards dropped, restricted to questions with a usable answer
    span. Questions with no span are dropped rather than backfilled with the question-final
    token -- backfilling would quietly convert them into Gate 2 rows and destroy the
    position comparison. GATE2B_PROTOCOL invariant 4.

    `extras` carries per-row things the tests need but the probe must not see: the greedy
    correctness flag (C0) and, when Gate 2's shards are present, its question-final
    activations for the C4 reference line.
    """
    H, mask, meta = load_acts()
    y = gate2._labels()
    idx = {q: i for i, q in enumerate(meta.qid)}
    keep = [(i, idx[q]) for i, q in enumerate(y.qid) if q in idx and mask[idx[q]]]
    dropped = len(y) - len(keep)
    sub = y.iloc[[i for i, _ in keep]].reset_index(drop=True)
    ai = [j for _, j in keep]
    mrows = meta.iloc[ai].reset_index(drop=True)
    if dropped:
        print(f"  dropped {dropped} labelled questions with no usable answer span "
              f"(kept {len(keep)})")

    X = {p: H[p][ai] for p in POSITIONS}
    baselines = {
        # b1 comes from labels.jsonl -- Phase 2's own pass -- not from the extraction
        # pass. The extraction copy exists only to prove the prompts match.
        "b1_entropy_qfinal": -sub.entropy.to_numpy(),
        "b2_first": -mrows.b2_first.to_numpy(),
        "b2_last": -mrows.b2_last.to_numpy(),
        "b2_mean": -mrows.b2_mean.to_numpy(),
        "b3_mean_logprob": mrows.b3_mean_logprob.to_numpy(),
        "b4_self_consistency": mrows.b4_self_consistency.to_numpy(),
    }
    extras = {"greedy_correct": mrows.greedy_correct.to_numpy().astype(bool),
              "n_answer_tokens": mrows.n_answer_tokens.to_numpy()}
    ref = _reference_acts(sub.qid.to_numpy())
    if ref is not None:
        extras[REFERENCE_POSITION] = ref

    return (X, (sub.knowledge == "known").to_numpy().astype(int),
            sub.subject_qid.to_numpy(), sub.relation.to_numpy(),
            baselines, sub.qid.to_numpy(), extras)


def _reference_acts(qids):
    """Gate 2's question-final activations, aligned to Gate 2b's rows. C4 reference only."""
    try:
        h_final, _, _, meta = gate2.load_acts()
    except FileNotFoundError:
        print("  (no Gate 2 shards on disk -- C4 runs without the reference line)")
        return None
    idx = {q: i for i, q in enumerate(meta.qid)}
    if not all(q in idx for q in qids):
        print("  (Gate 2 shards do not cover every Gate 2b row -- reference line skipped)")
        return None
    return h_final[[idx[q] for q in qids]]


def b2_matched(baselines, picks, folds, n):
    """b2 evaluated at the position each fold's probe actually selected.

    GATE2B_PROTOCOL: "b2 tracks the probe's selected position", so probe and entropy are
    always compared at the same place. Pooled out-of-fold probe scores come from five
    separately fitted probes that may have picked different positions, so there is no
    single pooled "the position the probe reads" -- the matched vector is assembled fold by
    fold exactly the way the probe's own pooled vector is.
    """
    out = np.full(n, np.nan)
    for pick, (_, te) in zip(picks, folds):
        out[te] = baselines[f"b2_{pick['position']}"][te]
    return out


def pooling_diagnostic(y, oof, folds):
    """Is the pooled out-of-fold AUC measuring separability, or cross-fold score drift?

    The primary number concatenates five separately fitted probes' raw decision values and
    ranks the whole vector. That is valid only while each fold's scores vary a lot compared
    with the offset between folds. Strong L2 shrinks the coefficients toward zero but not
    the intercept, which sklearn does not penalise, so the two can invert -- and when they
    do, a probe that separates perfectly inside every fold pools to chance.

    This is not hypothetical and it is not a fixture artifact: the selftest fixture hits it
    at C=1e-5, which is in the grid Gate 2b inherited from Gate 2's widened sensitivity
    run. Reported every run, not only when it fires.
    """
    per = [gate2._auc(y[te], oof[te]) for _, te in folds]
    pooled = gate2._auc(y, oof)
    within = float(np.mean([oof[te].max() - oof[te].min() for _, te in folds]))
    med = [float(np.median(oof[te])) for _, te in folds]
    between = max(med) - min(med)
    per_mean = float(np.nanmean(per))
    return {"per_fold_auc": [float(a) for a in per], "per_fold_mean": per_mean,
            "pooled_auc": float(pooled), "within_fold_range": within,
            "between_fold_offset": between,
            "offset_ratio": float(between / within) if within else float("inf"),
            "diverged": bool(abs(pooled - per_mean) > POOLING_DIVERGENCE)}


def rank_pool(s, folds):
    """Within-fold percentile ranks, in [0, 1). DECLARED POST-HOC SECONDARY.

    Monotone inside each fold, so every per-fold AUC is preserved exactly; the only thing
    that changes is how folds are compared with EACH OTHER. Instead of ranking a row by
    whatever offset its fold's intercept happened to land on, it is ranked by its standing
    among its own fold's peers.

    Applied identically to the probe and to all four baselines. Applying it to the probe
    alone would swap one asymmetry for another.

    Its standing is NOT "an alternative that might disagree by chance". The raw-pooled
    number has a bias with a KNOWN SIGN, applied ASYMMETRICALLY: the probe's pooled vector
    is stitched from five separately fitted models and the baselines are single global
    scalars that pay none of it. So the pre-registered number remains the headline for
    pre-registration integrity, and a disagreement between the two is a measurement
    artifact with a known direction -- not two estimates to weigh evenly.

    Measured on Gate 2's real data (`pooling_recheck.py`, 2026-08-05) the effect there was
    +0.0009 and changed no verdict.
    """
    from scipy.stats import rankdata
    out = np.full(len(s), np.nan)
    for _, te in folds:
        out[te] = rankdata(s[te]) / (len(te) + 1)
    return out


def _print_pooling(d):
    print(f"\n  pooling check: per-fold AUC mean {d['per_fold_mean']:.4f}  vs  pooled "
          f"{d['pooled_auc']:.4f}")
    print(f"    within-fold score range {d['within_fold_range']:.4f}, between-fold offset "
          f"{d['between_fold_offset']:.4f}")
    if not d["diverged"]:
        return
    print("\n  *** POOLED NUMBER IS NOT USABLE ***")
    print("  The folds separate but their scores are not on a comparable scale, so pooling")
    print("  them ranks calibration drift rather than knowledge. Every number read off the")
    print("  pooled vector -- the margin, the CI, C1, C4 -- inherits this. STOP and decide")
    print("  how to pool before reading any verdict below as a result.")


def _rank_secondary(y, oof, cand, groups, folds):
    """The whole comparison re-pooled by within-fold rank. No probe is refitted.

    Refitting would be wrong as well as wasteful: the fitted models are exactly the ones
    the pre-registered analysis used, and only the rule for combining their scores changes.
    BEST is recomputed within the secondary, because the strongest baseline under one
    pooling rule need not be the strongest under the other.
    """
    rp = {k: rank_pool(v, folds) for k, v in cand.items()}
    probe_r = rank_pool(oof, folds)
    aucs = {k: gate2._auc(y, v) for k, v in rp.items()}
    name = max(aucs, key=lambda k: aucs[k])
    d, lo, hi = gate2.boot_auc_diff(y, probe_r, rp[name], groups)
    return {"probe_auc": gate2._auc(y, probe_r), "baseline_auc": aucs,
            "best_baseline": name, "diff": {"point": d, "lo": lo, "hi": hi}}


def best_baseline(y, baselines, picks, folds):
    """BEST = max over b1..b4 of pooled AUC. Returns (name, scores, all_aucs).

    The maximum is taken ON THE EVALUATION DATA, deliberately. Selecting the strongest
    comparator after seeing the scores handicaps the probe -- the same handicap gate1
    invariant 6 applies to the entropy thresholds. It is registered as a maximum rather
    than as "entropy" so the choice cannot be made after the fact.
    """
    cand = {"b1_entropy_qfinal": baselines["b1_entropy_qfinal"],
            "b2_entropy_matched": b2_matched(baselines, picks, folds, len(y)),
            "b3_mean_logprob": baselines["b3_mean_logprob"],
            "b4_self_consistency": baselines["b4_self_consistency"]}
    aucs = {k: gate2._auc(y, v) for k, v in cand.items()}
    name = max(aucs, key=lambda k: aucs[k])
    return name, cand[name], aucs, cand


def phase_train():
    _bind_gate2()
    X, y, groups, rel, baselines, qids, extras = build_matrix()
    folds = gate2.group_folds(groups, N_FOLDS, SEED)
    n_layers = X["first"].shape[1]
    print(f"n={len(y)}  known={int(y.sum())}  unknown={int((1 - y).sum())}  "
          f"entities={len(set(groups))}  layers={n_layers}")
    print(f"{N_FOLDS}-fold entity-disjoint CV, nested selection over "
          f"{len(POSITIONS)} positions x {n_layers} layers x {len(C_GRID)} alphas "
          f"= {len(POSITIONS) * n_layers * len(C_GRID)} per fold")
    print(f"C_GRID = {C_GRID}  (inherited from Gate 2's post-hoc sensitivity run)\n")

    t0 = time.time()
    oof, picks = gate2.oof_nested(_shape_key(X), y, groups, folds,
                                  gate2._fingerprint(qids))
    print(f"\n  ({time.time() - t0:.0f}s)")

    name, best, aucs, cand = best_baseline(y, baselines, picks, folds)
    probe_auc = gate2._auc(y, oof)

    print("\n" + "=" * 78)
    print("CHECKPOINT B -- pooled out-of-fold AUC, entity-disjoint")
    print("=" * 78)
    print(f"  {'probe':<24} {probe_auc:.4f}")
    for k, v in aucs.items():
        print(f"  {k:<24} {v:.4f}" + ("   <-- BEST" if k == name else ""))
    d, lo, hi = gate2.boot_auc_diff(y, oof, best, groups)
    print(f"\n  probe - {name} = {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
          f"   (PROCEED needs >= {PROCEED_AUC_MARGIN}, CI excluding 0)")
    _, _, z, p = gate2.delong(y, oof, best)
    print(f"  DeLong (secondary): z={z:.2f}  p={p:.3g}")
    pool_d = pooling_diagnostic(y, oof, folds)
    _print_pooling(pool_d)
    sec = _rank_secondary(y, oof, cand, groups, folds)
    print(f"\n  SECONDARY (declared post-hoc), within-fold rank pooling:")
    print(f"    probe {sec['probe_auc']:.4f}   {sec['best_baseline']} "
          f"{sec['baseline_auc'][sec['best_baseline']]:.4f}")
    print(f"    probe - {sec['best_baseline']} = {sec['diff']['point']:+.4f}  "
          f"[{sec['diff']['lo']:+.4f}, {sec['diff']['hi']:+.4f}]")
    print(f"    shift vs the pre-registered point estimate: "
          f"{sec['diff']['point'] - d:+.4f}")
    print("\n  Pooled number ONLY. C1 is a VETO and can still turn this into a STOP.")
    print("  HUMAN CHECK: read the above, then run `python gate2b.py tests`.")

    np.savez(gate1._path(OOF_FILE), oof=oof, y=y,
             groups=groups.astype(str), rel=rel.astype(str), qids=qids.astype(str),
             greedy_correct=extras["greedy_correct"],
             n_answer_tokens=extras["n_answer_tokens"],
             **{k: v for k, v in cand.items()},
             **{f"raw_{k}": v for k, v in baselines.items()})
    json.dump({"folds": picks, "probe_auc": probe_auc, "baseline_auc": aucs,
               "best_baseline": name, "pooling": pool_d, "rank_secondary": sec,
               "probe_minus_best": {"point": d, "lo": lo, "hi": hi},
               "delong_z": z, "delong_p": p, "n": int(len(y)),
               "c_grid": list(C_GRID), "positions": list(POSITIONS)},
              open(gate1._path(FOLDS_FILE), "w"), indent=2)


# ======================================================================================
# PHASE C -- the tests
# ======================================================================================

def _load_oof():
    z = np.load(gate1._path(OOF_FILE), allow_pickle=False)
    cand = {k: z[k] for k in BASELINES}
    return (z["oof"], z["y"], z["groups"].astype(str), z["rel"].astype(str),
            z["qids"].astype(str), cand, z["greedy_correct"].astype(bool),
            z["n_answer_tokens"])


def stratify_by_generation(y, scores: dict, correct):
    """C0. Test AUC split by whether the stage-1 greedy generation was CORRECT.

    At answer tokens the probe sees a state conditioned on a specific generated answer, so
    it is verifying a candidate rather than detecting knowledge abstractly. If it separates
    only on correct generations it is an answer-correctness classifier and must be
    described as one. The incorrect subset is where a knowledge signal would have to do
    real work.

    Both strata are class-imbalanced by construction, so an AUC is reported only where the
    minority class clears MIN_MINORITY_N -- the same floor C1 uses. Where it does not, the
    row says so rather than carrying a number computed on four rows.
    """
    rows = []
    for label, m in (("greedy correct", correct), ("greedy incorrect", ~correct)):
        n = int(m.sum())
        minority = int(min(y[m].sum(), (1 - y[m]).sum())) if n else 0
        row = {"stratum": label, "n": n, "n_known": int(y[m].sum()),
               "n_unknown": int((1 - y[m]).sum()), "minority": minority,
               "reportable": bool(minority >= MIN_MINORITY_N)}
        if row["reportable"]:
            row.update({k: gate2._auc(y[m], v[m]) for k, v in scores.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _print_strata(d, scores):
    print("\n" + "=" * 78)
    print("C0. WHAT THE PROBE IS ACTUALLY DOING -- stratified by stage-1 correctness")
    print("=" * 78)
    for r in d.to_dict("records"):
        print(f"  {r['stratum']:<18} n={r['n']:<5} known={r['n_known']:<5} "
              f"unknown={r['n_unknown']:<5} minority={r['minority']}")
        if not r["reportable"]:
            print(f"    minority class below {MIN_MINORITY_N} -- no AUC reported for this "
                  f"stratum")
            continue
        for k in scores:
            print(f"    {k:<24} {r[k]:.4f}")
    ok = d[d.reportable]
    if len(ok) == 2 and "probe" in ok.columns:
        a, b = ok.iloc[0], ok.iloc[1]
        print(f"\n  probe separates at {a['probe']:.4f} on correct generations and "
              f"{b['probe']:.4f} on\n  incorrect ones. If the second is near 0.5 the probe "
              f"is an answer-correctness\n  classifier, not a knowledge detector -- report "
              f"it that way.")


def _decide(d_pool, lo_pool, d_wr, lo_wr, name, pooled_ok=True) -> str:
    if not pooled_ok:
        return ("NO VERDICT: the pooled out-of-fold vector failed the pooling check, so "
                "the pre-registered margin cannot be read off it. This is not a STOP -- a "
                "STOP would claim the probe does not separate, and the pooling check says "
                "we do not know. Decide how to pool, then re-read.")
    if d_pool < PROCEED_AUC_MARGIN or lo_pool <= 0:
        return (f"STOP: probe - {name} = {d_pool:+.4f} [{lo_pool:+.4f}, ...] does not "
                f"clear the pre-registered {PROCEED_AUC_MARGIN} with a CI excluding 0. "
                f"Reading at the answer tokens does not recover a signal the best free "
                f"baseline at those same tokens has not already got.")
    if not (d_wr >= WITHIN_RELATION_MARGIN and lo_wr > 0):
        return (f"STOP: the pooled advantage ({d_pool:+.4f}) does NOT survive within "
                f"relation ({d_wr:+.4f} [{lo_wr:+.4f}, ...]). The probe is behaving as a "
                f"relation classifier -- the same artifact C1 caught in Gate 2, at a new "
                f"read position.")
    return (f"PROCEED. Pooled probe - {name} = {d_pool:+.4f} [{lo_pool:+.4f}, ...] and the "
            f"advantage survives within relation ({d_wr:+.4f} [{lo_wr:+.4f}, ...]). Note "
            f"what C0 says about WHAT the probe is doing before reading this as knowledge "
            f"detection.")


def phase_tests():
    _bind_gate2()
    oof, y, groups, rel, qids, cand, correct, ntok = _load_oof()
    picks = json.load(open(gate1._path(FOLDS_FILE)))
    name = picks["best_baseline"]
    best = cand[name]

    scores = {"probe": oof, **cand}
    pooled = {k: gate2._auc(y, v) for k, v in scores.items()}

    print("=" * 78)
    print("POOLED (primary) -- entity-disjoint 5-fold CV, answer-token reads")
    print("=" * 78)
    for k, v in pooled.items():
        print(f"  {k:<24} {v:.4f}" + ("   <-- BEST baseline" if k == name else ""))
    d_pool, lo_pool, hi_pool = gate2.boot_auc_diff(y, oof, best, groups)
    _, _, z_pool, p_pool = gate2.delong(y, oof, best)
    print(f"\n  probe - {name} = {d_pool:+.4f}  [{lo_pool:+.4f}, {hi_pool:+.4f}]"
          f"   DeLong z={z_pool:.2f} p={p_pool:.3g}")
    folds_p = gate2.group_folds(groups, N_FOLDS, SEED)
    pool_d = pooling_diagnostic(y, oof, folds_p)
    _print_pooling(pool_d)
    sec = _rank_secondary(y, oof, cand, groups, folds_p)
    print(f"\n  SECONDARY (declared post-hoc), within-fold rank pooling:")
    print(f"    probe {sec['probe_auc']:.4f}   best {sec['best_baseline']} "
          f"{sec['baseline_auc'][sec['best_baseline']]:.4f}   diff "
          f"{sec['diff']['point']:+.4f} [{sec['diff']['lo']:+.4f}, "
          f"{sec['diff']['hi']:+.4f}]")
    print("    The pre-registered raw-pooled number above remains the headline. Where the")
    print("    two disagree that is a measurement artifact with a known direction -- the")
    print("    raw pooling costs the PROBE and not the baselines -- not two estimates to")
    print("    weigh evenly. See GATE2B_PROTOCOL 'Secondary analyses'.")

    # ---- C0 -- reported whatever it shows (invariant 7) ---------------------------------
    d0 = stratify_by_generation(y, {"probe": oof, name: best}, correct)
    _print_strata(d0, {"probe": oof, name: best})

    # ---- C1 -- the veto ------------------------------------------------------------------
    gated = d_pool >= PROCEED_AUC_MARGIN and lo_pool > 0
    role = ("the veto" if gated else
            "DIAGNOSTIC ONLY -- the pooled gate already failed, C1 cannot reverse that")
    cmp2 = {"probe": oof, name: best}
    d1, m1, ex1 = gate2.within_relation(y, cmp2, rel, groups)
    gate2._print_within(d1, m1, ex1, f"C1. WITHIN-RELATION AUC -- {role}")
    d_wr = m1.get("probe", float("nan")) - m1.get(name, float("nan"))
    lo_wr, hi_wr = gate2._boot_within(y, oof, best, rel, groups)
    print(f"\n  within-relation probe - {name} = {d_wr:+.4f} [{lo_wr:+.4f}, {hi_wr:+.4f}]"
          f"   (veto unless >= {WITHIN_RELATION_MARGIN}, CI excluding 0)")

    # ---- C3 ------------------------------------------------------------------------------
    d3, m3, ex3 = gate2.within_relation(y, cmp2, rel, groups, drop=("spouse",))
    sp = int((rel == "spouse").sum())
    gate2._print_within(d3, m3, ex3, f"C3. SPOUSE-EXCLUDED within-relation "
                                     f"(spouse = {sp}, {sp / len(rel):.1%} of the pool)")
    d_sp = m3.get("probe", float("nan")) - m3.get(name, float("nan"))
    print(f"\n  spouse-excluded probe - {name} = {d_sp:+.4f}   (with spouse: {d_wr:+.4f})")

    # ---- C2 / C4 need the activations back -----------------------------------------------
    X, y2, g2, rel2, _, q2, extras = build_matrix()
    assert len(y2) == len(y) and (y2 == y).all(), "row set drifted between phases"
    assert (q2 == qids).all(), "row ORDER drifted between phases"
    assert (rel2 == rel).all(), "relations drifted between phases"
    folds = gate2.group_folds(g2, N_FOLDS, SEED)
    fp = gate2._fingerprint(q2)

    print("\n" + "=" * 78)
    print("C2. LEAVE-ONE-RELATION-OUT")
    print("=" * 78)
    d2 = gate2.leave_one_relation_out(_shape_key(X), y, groups, rel, fp)
    if len(d2):
        print(f"  n-weighted mean AUC = {np.average(d2.auc, weights=d2.n):.4f}   "
              f"(within-relation probe, for comparison: {m1['probe']:.4f})")

    print("\n" + "=" * 78)
    print("C4. READ POSITION x LAYER")
    print("=" * 78)
    Xc = _shape_key(X)
    curve_positions = POSITIONS
    if REFERENCE_POSITION in extras:
        # Reference line only. Rebound here and nowhere else, so Gate 2's position can
        # never enter the nested selection that phase_train already ran.
        Xc[REFERENCE_POSITION] = extras[REFERENCE_POSITION]
        curve_positions = POSITIONS + (REFERENCE_POSITION,)
    gate2.POSITIONS = curve_positions
    d4 = gate2.position_curves(Xc, y, g2, folds, fp)
    gate2.POSITIONS = POSITIONS
    print(d4.pivot(index="layer", columns="position", values="auc").round(4).to_string())
    for pos in curve_positions:
        b = d4[d4.position == pos].sort_values("auc").iloc[-1]
        print(f"  best {pos:<12} layer {int(b.layer):>2}  AUC {b.auc:.4f}")
    one = float((ntok == 1).mean())
    print(f"\n  {one:.1%} of rows have a one-token answer, where first, last and mean are "
          f"the SAME\n  position. Read any first/last/mean difference below as diluted by "
          f"that fraction.")

    print("\n" + "=" * 78)
    verdict = _decide(d_pool, lo_pool, d_wr, lo_wr, name, not pool_d["diverged"])
    print(verdict)
    print("=" * 78)

    json.dump({"pooled": pooled, "best_baseline": name,
               "pooled_diff": {"point": d_pool, "lo": lo_pool, "hi": hi_pool,
                               "delong_p": p_pool},
               "c0_strata": d0.to_dict("records"),
               "c1_within_relation": d1.to_dict("records"), "c1_means": m1,
               "c1_excluded": ex1.to_dict("records"),
               "c1_diff": {"point": d_wr, "lo": lo_wr, "hi": hi_wr},
               "c2_loro": d2.to_dict("records"),
               "c3_spouse_excluded": {"means": m3, "diff": d_sp},
               "c4_positions": d4.to_dict("records"),
               "pooling": pool_d, "rank_secondary": sec, "verdict": verdict},
              open(gate1._path(RESULTS_FILE), "w"), indent=2, default=float)


# ======================================================================================

PHASES = {"instrument": phase_instrument, "extract": phase_extract,
          "checkpoint": _checkpoint_a, "train": phase_train, "tests": phase_tests}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in PHASES:
        sys.exit(f"usage: python gate2b.py [{' | '.join(PHASES)}]")
    print(f"gate2b: C_GRID={C_GRID}  POSITIONS={POSITIONS}  (see GATE2B_PROTOCOL.md)")
    PHASES[sys.argv[1]](*[int(a) for a in sys.argv[2:]])

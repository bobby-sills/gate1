# Gate 3b: close the decoder loop. Working notes, NOT a pre-registration.

**Status: planned, not built.** Written down because the key finding below is expensive to
rediscover.

**Framing, agreed with the human 2026-08-07.** This is run as **engineering, not as a
claim.** No pre-registered threshold. Search freely, select on validation, measure the
winner once on a locked test split, and ask only "is the final number worth the cost."
Gates 1–3a used pre-registration because they were making claims about the world; Gate 3b
is asking whether an artifact works.

---

## The question

Every gate so far measured the probe in **AUC**. The project's actual currency is **macro
faithfulness**. Gate 1 showed perfect knowledge labels buy **+13.83** points over the
entropy-gated arm. Nobody has shown that a *real* probe recovers a single point.

Gate 3b: take the probe scores we already have, feed them into the decoder's gate as `k`,
and measure faithfulness against the `entropy`, `max_prob` and `constant` arms.

## THE KEY FINDING: this needs no GPU

`gate1.tau_for` for every threshold arm is:

```python
t = tau0 - delta if row[signal] < thr else tau0 + gamma
```

The decoder only ever sees `tau`. A threshold arm therefore emits values from
`{tau0 - delta, tau0 + gamma}` over `TAU_GRID x GAMMA_GRID x DELTA_GRID` — **independent of
which signal drives the threshold.**

So a probe arm's `required_work` set is a **subset of the entropy arm's**, and every
generation it needs is already in `generations.jsonl` from Gate 1's decode. Gate 3b is
**pure CPU re-scoring**: look up cached generations, score with `alias_match`, aggregate.

Minutes, not hours. This is why it is the cheapest remaining experiment.

## Design

1. Load `cells.jsonl` (the 4 cells: correction, resistance, agreement, both_wrong).
2. Attach each row's **out-of-fold** probe score from `probe2b_oof.npz`, keyed by `qid`.
   Out-of-fold matters: every score comes from a probe that never trained on that question.
3. Build probe-arm configs exactly as `build_configs` does for `entropy` — same
   `TAU_GRID x GAMMA_GRID x DELTA_GRID`, thresholds at the 25/50/75 quantiles of the probe
   score. **Equal tuning budget** is gate1 invariant 5 and must hold for the probe arm too.
4. For each config, look up `generations.jsonl[(qid, context_kind, tau)]`.
5. Score, compute `macro_faithfulness`, find `best_operating_point`.
6. Compare to entropy's best via `paired_bootstrap` — **resampled by `qid`**, gate1
   invariant 1.

Write `gate3b.py` importing `gate1`; **do not edit `gate1.py`** (authorization-gated, and
its constants are pre-registered).

## Checks to run first

- **Coverage.** Do all `cells.jsonl` qids have a probe score? `probe2b_oof` covers 1582
  questions (FUNCTIONAL, non-discard, answer span resolved); cells should be a subset but
  this is unverified. Any gap must be reported, not silently dropped.
- **Cache hit rate.** Assert every required `(qid, context_kind, tau)` is present in
  `generations.jsonl`. If any miss, the subset argument above is wrong and it needs
  understanding before proceeding.

## What to expect

Sober. Gate 1's entire +13.83 sits in the **27% of instances where confidence and knowledge
disagree** — the off-diagonal, where gaps are +41.5 and +42.7 points. On the 73% where they
agree, entropy is already near the oracle.

So the probe's +0.0296 AUC has to land **there specifically**, not spread evenly. And
`FINDINGS_GATE2B.md` §4 shows that advantage is answer-memorisation which vanishes out of
distribution — so on the contested cases it may be worth nothing at all.

A null here is still worth having: it converts "3 AUC points" into "N faithfulness points"
and closes the question the whole project was built to answer.

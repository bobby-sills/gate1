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

## ~~THE KEY FINDING: this needs no GPU~~ — WRONG, corrected 2026-08-07

`gate1.tau_for` for every threshold arm is:

```python
t = tau0 - delta if row[signal] < thr else tau0 + gamma
```

The decoder only ever sees `tau`. A threshold arm therefore emits values from
`{tau0 - delta, tau0 + gamma}` over `TAU_GRID x GAMMA_GRID x DELTA_GRID` — **independent of
which signal drives the threshold.**

The claim drawn from that was: a probe arm's `required_work` is a subset of the entropy
arm's, so Gate 3b is pure CPU re-scoring.

**That is false, and `gate3b.py check` caught it before any number was computed.** The set
of tau *values* is a subset. The set of **`(qid, tau)` pairs is not** — and the generation
cache is keyed on the pair.

A question whose entropy falls below all three entropy thresholds takes the `tau0 - delta`
branch for *every* entropy config, so its `tau0 + gamma` generations were never produced.
If the probe puts that question on the other side of its own threshold, it asks for exactly
the generation no arm ever requested.

```
518 of 15600 units missing, on 74 of 600 questions
  entropy below all 3 thresholds: 37     above all 3: 37
  knowledge: known 37, unknown 37
```

**The 74 are not a random remainder.** They are precisely the questions where the probe
disagrees with *both* entropy and the ground-truth knowledge label. That is the only
population on which a probe can add anything Gate 1 did not already have — which is why
they are missing, and why they must not be dropped. Dropping them would delete the
experiment and leave a number that looked fine.

**Revised cost: 518 decode units, 2.0% of Gate 1's 25,974.** Still the cheapest remaining
experiment; just not free. `gate3b.py decode` fills them, resumably, writing to
`generations_gate3b.jsonl` — a separate file, so Gate 1's reported artifact is not
modified by a later gate's choices.

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

## Checks to run first — all three now run as `gate3b.py check`

- **Coverage.** Do all `cells.jsonl` qids have a probe score? **PASS** — all 600 cells
  questions are inside `probe2b_oof`'s 1582.
- **Cache hit rate.** **FAIL, and it was informative** — see the corrected section above.
  `decode` closes it.
- **Scoring reproduction** (added after the first two ran). Re-score gate 1's own
  `entropy` arm from `cells.jsonl` + `generations.jsonl` and compare against `sweep.csv`
  row by row. **PASS at 432000/432000, agreement 1.000000.** This is the only thing tying
  this file's probe number to Gate 1's published baselines; without it a scoring
  discrepancy would show up as a fake probe effect.

  Note for anyone re-running it: `thr` must be rounded before use as a merge key.
  `to_csv` truncates float64, so the middle entropy threshold round-trips as
  `0.948013722896576` against an in-memory `0.9480137228965759`, and a third of the rows
  silently fail to join at 100% agreement on the rest.

## What to expect

Sober. Gate 1's entire +13.83 sits in the **27% of instances where confidence and knowledge
disagree** — the off-diagonal, where gaps are +41.5 and +42.7 points. On the 73% where they
agree, entropy is already near the oracle.

So the probe's +0.0296 AUC has to land **there specifically**, not spread evenly. And
`FINDINGS_GATE2B.md` §4 shows that advantage is answer-memorisation which vanishes out of
distribution — so on the contested cases it may be worth nothing at all.

A null here is still worth having: it converts "3 AUC points" into "N faithfulness points"
and closes the question the whole project was built to answer.

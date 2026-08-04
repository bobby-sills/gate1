# CLAUDE.md

Research experiment. The code encodes methodological decisions that took a long argument
to settle. Several of them **look like bugs and are not**. Read the invariants before
changing anything.

## File ownership

| File | Who edits | Notes |
|---|---|---|
| `backend.py` | **You** | Edit freely. Four stubs, plus the pool filters. |
| `gate1.py` | You, **on authorization** | Runner, cache, gate, statistics. |
| `PROTOCOL.md` | You, **on authorization** | Experiment design and rationale. |
| `HANDOFF.md` | You, **on authorization** | Sequenced tasks. |
| `CLAUDE.md` | You, **on authorization** | This file. |
| `selftest.py` | Nobody | Run after each backend change. |

**The rule that stays is ask-first.** For anything other than `backend.py`, say what you
want to change and why, then wait. Do not infer authorization from a related approval —
it is granted per change. It has been granted before (filters 5–6 and the
relation-matched `phase_cells`, 2026-08-04), and every time it fired it caught something.

If a task *seems* to require editing `gate1.py`, the first hypothesis is still that a
`backend.py` signature was misread. Check that before asking.

## What this experiment is

Testing whether ground-truth knowledge of a fact predicts correct source-arbitration
better than freely-available output confidence does. The "oracle" arms use ground-truth
labels — they are **deliberately cheating**, because they measure the ceiling on a method
that would later estimate those labels with a trained probe. If the oracle barely beats
the entropy baseline, the whole research direction stops. That is a valid and useful
outcome; do not optimize toward making the oracle win.

## Invariants — do not "fix" these

1. **`paired_bootstrap` resamples by `qid`, not by row.** Each question contributes two
   correlated instances (factual and counterfactual context). Row-level resampling would
   treat them as independent and roughly halve the confidence interval for free.

2. **`deterministic_features` must be a separate forward pass from `sample_closed_book`.**
   Merging them to save compute is the most tempting and most damaging optimization here.
   The knowledge label comes from sampling, which is itself a confidence measurement;
   sharing computation between label and predictor handicaps the entropy baseline and
   inflates the headline result.

3. **Alias matching is substring-after-normalization, not exact equality.** Looks sloppy.
   Alias handling is where most label noise in this literature originates. The same
   matcher is used for labels and for scoring, on purpose.

4. **In the `resistance` cell the target is the factual gold, not the document's answer.**
   The document is wrong there, so following it is the error. Looks like an inverted
   label. It is not.

5. **Every arm gets the same grid density.** The `constant` arm having 8 configs while
   `entropy` has 360 is not an imbalance to correct — each arm's tuning budget over *its
   own* parameters is matched in evaluations. Unequal tuning is the standard way
   "adaptive beats fixed" turns out to be an artifact.

6. **Entropy and max_prob thresholds are tuned on the evaluation data.** Looks like
   leakage. It is deliberate: we are handicapping ourselves so the baselines get their
   best possible showing.

7. **The constants at the top of `gate1.py` are pre-registered.** They are not tunable
   config. Changing any of them after `decode` begins invalidates the experiment. If a
   grid seems too large, report the measured cost — do not shrink it.

8. **Do not add caching inside `gate1.greedy_decode`.** `phase_decode` already
   deduplicates work by `(qid, context_kind, tau)`. Cache KV state *inside*
   `next_token_distributions` instead. Two cache layers will diverge.

## Working agreement

- **Commit after each backend function.** Four commits minimum.
- **Run `python selftest.py` after every change.** It exercises the full pipeline with a
  fake backend, including a simulated crash and resume. It must pass.
- **Instrument before scaling.** Run 100 instances, report wall-clock per unit and
  projected total, and wait for confirmation before a full phase.
- **Two checks require a human and must not be automated:** the Phase 1 attrition/sample
  read, and the Phase 2.7 prompt-parity read. Print the output and stop.
- **Report surprises rather than working around them.** A cell that cannot reach its
  target size, a discard rate outside 20–30%, an off-diagonal fraction below 0.15 — these
  are findings, not obstacles.

## Environment

Colab Pro. Llama-3.1-8B-Instruct, bf16, ~16GB. L4 or A100 fine; **T4 will not fit two KV
caches** — check `nvidia-smi` first and load in 8-bit or restart the runtime if you get one.

**Setup is in `COLAB.md`.** Follow it exactly. The one thing that must not be improvised:
`GATE1_OUT` has to point at a Drive path and must be set with `os.environ` in a Python
cell, never `export` in a `!` cell — `!` cells spawn fresh shells, the variable is lost,
and checkpoints go to the ephemeral runtime disk where a disconnect destroys them.

```python
import os
os.environ['GATE1_OUT'] = '/content/drive/MyDrive/gate1/outputs'
```

Re-run the mount cell after every reconnect. `labels` and `decode` are resumable — re-run
the same command and confirm it prints `resuming ...: N units already complete`. If that
line is missing, the path is wrong and it is starting over.

# Gate 2b: Probing at the Answer Tokens

**Question.** Gate 2 read the residual stream at the final question token and found a
linear probe adds +0.0095 AUC over `H(p_theta)`, with a CI spanning zero. The
truthfulness-probing literature locates its signal at the *answer* tokens, which do not
exist at that position. Does reading there recover the signal — and if it does, does a
free baseline recover it too?

**Why the second half of that question is the experiment.** Moving the probe to a richer
position while leaving the baseline at the old one would manufacture a win out of the
repositioning alone. Gate 2b therefore moves the baselines to the answer tokens as well,
and the probe must beat the **best** of them.

**Gate 2's null remains the headline.** Whatever 2b returns, the pre-registered Gate 2
result stands as reported in `FINDINGS_GATE2.md`. Gate 2b is a differently-positioned
follow-up experiment, not a re-analysis of Gate 2 and not a second attempt at it. It
cannot revise Gate 2's number, and a PROCEED here would mean "a different read position
works", never "Gate 2 was wrong".

**What Gate 2b tests.** The probe half of a two-stage mechanism only. Stage 1 completes
the context-free generation greedily; stage 2 would probe at the answer tokens, obtain
`k`, rewind, and re-decode with the gate set. **No decoder run happens in Gate 2b.**

---

## Phase 0 — Pre-registration

Frozen before extraction runs. Do not edit after Phase A begins.

| Decision | Value | Locked |
|---|---|---|
| Model | **Llama-3.1-8B-Instruct**, same weights, prompt and `DATE_STRING` as Gates 1–2 | 2026-08-05 |
| Stage-1 generation | greedy (`do_sample=False`), closed book, `max_new_tokens=32` | 2026-08-05 |
| Probe | Logistic regression, L2, per layer, per read position | 2026-08-05 |
| Labels | `knowledge` from `labels.jsonl`, FUNCTIONAL pool, discards dropped, ONE_TO_MANY not trained on | 2026-08-05 |
| Read positions | p1 first answer token, p2 last answer token, p3 mean-pooled over the span | 2026-08-05 |
| Layers | all, embedding through final (33) | 2026-08-05 |
| Splits | 5-fold CV grouped by `subject_qid`, nested inner split for layer + position + alpha | 2026-08-05 |
| Primary number | pooled out-of-fold test AUC, entity-disjoint | 2026-08-05 |
| Comparator | **BEST of b1–b4**, pooled, on identical folds | 2026-08-05 |
| Regularization grid | `C_GRID_WIDE` — inherited, see *Inherited from a post-hoc analysis* | 2026-08-05 |
| PROCEED if probe AUC − BEST baseline AUC >= | **0.03**, entity-clustered bootstrap 95% CI excluding 0 | 2026-08-05 |
| PROCEED also requires | within-relation (probe − BEST baseline) >= **0.02**, CI excluding 0 (C1) | 2026-08-05 |
| STOP if | either fails | 2026-08-05 |
| Bootstrap | 10,000 resamples, resampled by `subject_qid` | 2026-08-05 |
| Seed | 20260803 | 2026-08-05 |

**C1 is a veto, not a margin** — locked 2026-08-05 by the human, after the spec left its
status ambiguous (it appeared under "carried over unchanged" while the PROCEED rule named
only the pooled margin). Gate 2 found that the *entire* pooled probe advantage was a
relation effect; without the veto, Gate 2b could PROCEED on precisely the artifact Gate 2
was built to catch. Margins are still read off the pooled number only.

### Inherited from a post-hoc analysis, declared

Gate 2b uses `C_GRID_WIDE = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)` from the outset. That
grid was **not** pre-registered in Gate 2 — it was constructed after seeing a failing Gate
2 result, because 5/5 folds had selected the boundary value `1e-3`. Inheriting a
post-hoc-widened grid into a new pre-registration is legitimate (the widening is fixed
before any 2b data is touched, so it cannot be tuned toward a 2b outcome), but the
provenance is recorded rather than laundered.

It is also known to be inert: in Gate 2 the widened grid reproduced the pre-registered
result exactly, all five folds re-selecting `1e-3` at identical layers with 1e-4 and 1e-5
available. Inheriting it costs 50% more fits and is expected to change nothing. It is
inherited anyway so that "the grid boundary was binding" cannot be raised against a 2b
null after the fact.

### Correction to the spec: `deterministic_features` does not generate

The spec describes stage 1 as completing "the same [generation] `deterministic_features`
already begins". `deterministic_features` runs a single forward pass and returns
`(entropy, max_prob)` of the next-token distribution at the first answer position; it
produces no tokens and its pass is long spent. Stage 1 is therefore a **new greedy
generation** from the identical closed-book prompt. Its first token is the argmax of the
distribution `deterministic_features` measured, and Phase A asserts exactly that — which
is the sense in which it continues that pass.

`gate1.greedy_decode` is **not** reused: it drives the two-pass contrastive cache and
requires a document context. Stage 1 is plain closed-book greedy decoding and gets its own
function in `backend.py`.

### Disclosure: data touched before freezing

Before writing this file I read `gate1.phase_labels` and confirmed what Phase 2 persisted.
That is source code, not data. No activation, label, or outcome variable was read, and the
finding it produced (that the ten generations were never written to disk) is recorded under
*b4* below because it forced a design decision.

---

## The four baselines — this is the experiment, not a control

All four are oriented so that **higher means more likely known**. Entropies enter
sign-flipped (Gate 2 invariant 4).

Write the closed-book prompt tokens as `t_1..t_n` and the greedy answer tokens as
`a_1..a_m`. "The hidden state at `a_j`" means the state at the sequence position `a_j`
occupies — i.e. after the model has consumed `a_j`. The distribution *produced* at that
position is over `a_{j+1}`.

| | definition |
|---|---|
| **b1** | `H(p_theta)` at `t_n` — the distribution over `a_1`. Gate 2's baseline, read from `labels.jsonl`, for continuity. |
| **b2** | `H(p_theta)` at the position the probe reads: at p1 the distribution over `a_2`; at p2 the distribution after `a_m`; at p3 the mean entropy over positions `a_1..a_m`. |
| **b3** | mean token log-prob of the generated answer span, `(1/m) * sum_j log p(a_j)` — length-normalised sequence likelihood. |
| **b4** | self-consistency of the ten Phase 2 samples: the fraction matching the modal normalised answer. **Gold is never consulted.** |

**b1 and b2 are not the same number**, and the off-by-one that would make them the same is
the easiest error in this file. b1 is the distribution over the answer's *first* token; b2
at p1 is the distribution over its *second*. An implementation that silently reads `t_n`
for both would report `b2 == b1` exactly — Phase A asserts they differ.

**b2 tracks the probe's selected position**, so probe and entropy are always compared at
the same place. The other three are position-independent.

### BEST is a maximum taken on the evaluation data, deliberately

`BEST = max(b1, b2, b3, b4)` on the pooled out-of-fold scores. Selecting the strongest
comparator after seeing the scores handicaps the probe, in the same spirit as Gate 2
invariant 6 (entropy thresholds tuned on evaluation data so the baselines get their best
showing). It is registered as a maximum and not as "entropy", so the choice cannot be made
after the fact.

### b4: the ten samples were never persisted

**The spec's premise is false and this is the one forced deviation.** It says b4 "uses data
already on disk". `gate1.phase_labels` computes `gens = sample_closed_book(...)` in memory
and writes only `{qid, n_correct, knowledge, entropy, max_prob}` — the generation strings
were discarded. The one sample-derived quantity on disk, `n_correct`, **is the label**:
`knowledge` is a deterministic function of it (>=8 known, <=2 unknown, else discard), so
any predictor built from `n_correct` scores **AUC exactly 1.0 by construction**. That is
the label read back, not a baseline.

**Resolution, locked 2026-08-05 by the human: regenerate.** `backend._question_seed` is a
stable SHA-256 of the question text, so `sample_closed_book(q, 10, SEED)` returns *the
identical ten strings* that produced the label — a faithful recovery, not a fresh draw.
Phase A regenerates them in the same pass as the greedy decode and hidden states, and
asserts that the recomputed `n_correct` matches `labels.jsonl` for every question. That
assertion is what makes the recovery claim checkable rather than assumed; a mismatch means
the sampling path is not reproducible and Phase A stops.

`b4` is then computed **without gold**: normalise each of the ten with `gate1.normalize`,
take the modal string, and report the fraction matching it. Range `[0.1, 1.0]`.

> **Registered caveat.** `b4` and the label are two functions of the same ten generations.
> This very likely flatters `b4` relative to a baseline computed from independent samples.
> The direction is conservative — it raises the bar the probe must clear — so it is
> accepted rather than corrected, but the pooled `b4` number is never quoted without this
> sentence attached. If `b4` alone reaches probe AUC, the correct reading is "the probe is
> redundant against sample agreement", with the caveat that sample agreement had a
> structural advantage.

---

## Phase A — Extraction (GPU)

Per question in the FUNCTIONAL pool (`pool.jsonl`), three things happen in one visit:

1. **Greedy generation.** `_build_prompt(question, None)`, `do_sample=False`,
   `max_new_tokens=32`. Yields `a_1..a_m`.
2. **One teacher-forced forward pass** over `[t_1..t_n, a_1..a_m]` with
   `output_hidden_states=True`. This gives hidden states at *every* answer position and
   logits at every position in a single pass — which is why it is preferred over
   harvesting states from `generate`: `generate` never computes the state at `a_m` when
   `a_m` is the EOS-terminating token, so p2 would be silently undefined for exactly the
   questions where the model finished cleanly.
3. **Ten samples** via `sample_closed_book(question, 10, SEED)`, for b4.

### The answer span

Answer tokens are the generated tokens with trailing EOS and any special tokens removed.
Questions where `m == 0` after that removal have **no usable answer span**: they are
excluded from Gate 2b entirely and counted in Checkpoint A. They are not backfilled with
the question-final token — that would quietly convert them to Gate 2's position and
destroy the comparison (Gate 2 invariant 5, same reasoning, new position).

Span-length distribution is a Checkpoint A item. A span of `m == 1` makes p1, p2 and p3
identical for that row; the fraction where that holds is reported, because if it is large
the three positions are not really three positions.

### Parity check

The teacher-forced pass recomputes the distribution at `t_n` for free, which is exactly
`deterministic_features`. Compare `(entropy, max_prob)` against `labels.jsonl` on the same
three-tier tolerance Gate 2 registered:

| max abs difference in entropy | reading |
|---|---|
| < 1e-3 | pass, numerically identical up to bf16 kernel nondeterminism |
| 1e-3 to 1e-2 | pass with a note; report the distribution |
| > 1e-2 on any question | **STOP.** The prompt is not the same. Do not train. |

Two further assertions, both free:

- `a_1 == argmax` of the distribution at `t_n`. Greedy stage 1 continues the pass
  `deterministic_features` measured; if it does not, the generation config is overriding
  something.
- recomputed `n_correct` from the ten regenerated samples equals `labels.jsonl`. See b4.

### Storage

`float16`, three positions × 33 layers × 4096 = 810 KB/question, ≈ 1.45 GB for the pool.
Sharded at 250 questions with a manifest, resume granularity one shard, `tmp` + `os.replace`
— identical to Gate 2, for the identical reason (a 1.4 GB file rewritten on every
checkpoint is one disconnect away from being a corrupt 1.4 GB file).

Per-question scalars written alongside: answer text, `m`, greedy-correct flag, b2 at all
three positions, b3, b4, recomputed `(entropy, max_prob)`, and the relation.

### Cost checkpoint — measure, do not assert

Instrument 100 questions first and report, separately: greedy-generation seconds,
teacher-forced-pass seconds, ten-sample seconds. Then project the pool.

The marginal cost of the *mechanism* (not of this experiment) is **one extra short greedy
generation per question**, incurred once before the gated pass. It is framed against the
two forward passes per token the contrastive decoder already pays at decode time, so the
honest claim is "one extra short generation per question", not "twice as expensive". The
ten-sample regeneration is a cost of *measuring b4 in this experiment only* and is not part
of the mechanism — it is reported separately so the two are never added together.

### Checkpoint A — report and stop

Report, before any probe is trained:

- span-length distribution; count with no usable span; fraction with `m == 1`
- the parity table, and both free assertions
- greedy-correct rate, overall and by knowledge label
- `n` per class after discards and ONE_TO_MANY are dropped
- **pooled AUC for b1, b2 (all three positions), b3 and b4**

That last line is the point of the checkpoint. If b4 alone is already at 0.87 we know it
before spending anything on training.

---

## Phase B — Training (CPU)

Logistic regression per (layer, position). Nested selection over 33 layers × 3 positions ×
6 alphas = 594 combinations per fold, chosen on the inner split and never on the evaluation
fold. Report which layer and position each fold picked, and the agreement across folds.

Splits, grouping by `subject_qid`, label restriction and the ONE_TO_MANY exclusion are Gate
2's, unchanged. Baselines are scored on the identical test slices — untrained, so their
"fold" is just the slice, but it must be the same slice or the comparison is not paired.

### Checkpoint B — report and stop

Pooled AUC for the probe and all four baselines, the per-fold selections, and the pooled
margin against BEST. This is the read on whether Phase C is worth running.

---

## Phase C — Tests

### C0. What the probe is actually doing — reported, never buried

At answer tokens the probe sees a state conditioned on a *specific generated answer*, so it
is verifying a candidate rather than detecting knowledge abstractly. This is Inside-Out's
setup, not Gate 2's, and it changes what a win would mean.

Two measurements, both required in the write-up whatever they show:

1. **Test AUC stratified by whether the stage-1 generation was correct** (`gate1.alias_match`
   of the greedy answer against `gold_aliases`). If the probe separates only on correct
   generations, it is an answer-correctness classifier and should be described as one.
2. **AUC on the incorrect-generation subset, reported separately.** That subset is where a
   knowledge signal would have to do real work.

Both strata will be class-imbalanced by construction — correct generations skew known,
incorrect skew unknown. Report `n` per class in each stratum, and report an AUC only where
the minority class has **>= 10 rows**, the same floor C1 uses. Where the floor is not met,
say so; do not report a number computed on four rows.

### C1. Within-relation AUC — the veto

AUC inside each relation, averaged weighted by `n`, for the probe and for BEST. Inclusion:
**>= 60 labelled rows and >= 10 in the minority class**, the frozen Gate 2 list. Excluded
relations are listed with their `n` and class balance.

Gate 2 found the pooled probe advantage vanished within relation (+0.0095 → −0.0023), and
found that entropy was *also* partly a relation classifier. Both facts make this the test
that decides.

### C2. Leave-one-relation-out

Train on all relations but one, test on the held-out relation. Layer, position and alpha
selected inside the training relations only.

### C3. Spouse-excluded

`spouse` is ~23% of the pool. Re-run C1 without it. If the advantage lives in `spouse`, say
so plainly.

### C4. Read position × layer

Full per-layer AUC curve for p1, p2 and p3, plus Gate 2's question-final position carried
onto this figure for reference. Compared on the questions where all resolve, so no position
is advantaged by an easier subset.

### Statistics

**Entity-clustered bootstrap governs**, resampled by `subject_qid` to match the split unit.
**DeLong is secondary**: it assumes two scores on one fixed test set from models that did
not vary, which pooled out-of-fold CV predictions violate. Where they disagree, the
bootstrap governs and the disagreement is reported.

---

## Secondary analyses

### RANK — within-fold rank pooling

**Why it was raised.** The primary number concatenates the raw `decision_function` values
of five separately fitted probes and ranks the whole vector. That is valid only while each
fold's scores vary a lot compared with the offsets *between* folds. Strong L2 shrinks the
coefficients toward zero but not the intercept, which sklearn does not penalise, so the two
can invert: on the Gate 2b selftest fixture at `C=1e-5` — a value in the grid this protocol
inherits — every fold scores AUC 1.0000 and the pooled vector scores 0.5024.

**Why it is not symmetric.** The probe's pooled vector is stitched from five models. b1–b4
are single global scalars with no fold structure and pay none of this cost. Whatever
pooling costs is therefore subtracted from the probe alone.

**The rule.** `rank_pool` converts each fold's scores to within-fold percentile ranks. It
is monotone inside a fold, so every per-fold AUC is preserved exactly; only the cross-fold
comparisons change. It is applied identically to the probe and to all four baselines —
applying it to the probe alone would swap one asymmetry for another. No probe is refitted:
the fitted models are the pre-registered ones and only the combining rule changes.

**Its standing, and this is a correction to how WIDE was framed.** "The pre-registered
result governs on disagreement" is the right rule when two estimators are both unbiased and
differ by chance. That is not this case. Here one estimator has a **systematic bias with a
known sign, applied asymmetrically**. So:

> The pre-registered raw-pooled number remains the headline, for pre-registration
> integrity. But a disagreement between the two is a **measurement artifact with a known
> direction**, not evidence to be weighed evenly. Report it as such.

**Gate 2 was rechecked before Gate 2b extraction** (`pooling_recheck.py`, 2026-08-05), on
the pre-registered and the wide artifacts. The result: no meaningful effect on the real
data. All five folds had selected `C=1e-3`, the between-fold offset was 0.044 of the
within-fold range for the probe (the degenerate fixture sits at 11.96), per-fold and pooled
AUC agreed to +0.0006, and the headline moved from +0.0095 [−0.0083, +0.0271] to +0.0104
[−0.0069, +0.0279] — a shift of +0.0009, with the same STOP under both rules. **Gate 2's
published conclusion does not depend on the pooling rule.** The concern was real, was
checked against the real data before it could affect anything, and changed nothing.

`pooling_diagnostic` runs every Gate 2b phase regardless. If the divergence is severe,
`_decide` returns **NO VERDICT** rather than a STOP — a STOP would claim the probe does not
separate, and a failed pooling check says only that we cannot tell.

---

## Not in scope, deliberately

- **Stage 2.** No rewind, no gated re-decode, no `k` fed into `tau`. Gate 2b is the probe
  half only.
- **Re-tuning Gate 1's decoder or its grids.**
- **A second model.**
- **Inside-Out internal-knowledge labels.** The label stays Gate 2's, so the two gates
  remain comparable.

---

## Deliverables

| Artifact | Phase |
|---|---|
| `acts2b_shard*.npz` + `acts2b_manifest.json` | A |
| `extract2b_parity.json` — entropy agreement, `n_correct` agreement, span stats, cost | A |
| `probe2b_folds.json` — per-fold layer, position, alpha, test AUC | B |
| `gate2b_results.json` — pooled + stratified + within-relation + LORO + positions | C |
| Console output of `gate2b.py tests`, saved | C |

---

## Invariants — do not "fix" these

1. **The baselines move with the probe.** Comparing an answer-token probe against a
   question-final entropy is how this experiment would manufacture its own result. b2
   tracks the probe's selected position.

2. **b1 != b2.** They are one token apart. If they come out equal, the read position is
   wrong, not the baselines redundant.

3. **`n_correct` is the label, not a baseline.** Anything derived from it scores AUC 1.0
   by construction. b4 is computed from the regenerated strings without gold.

4. **Questions with no usable answer span are dropped, not backfilled.** Backfilling with
   the question-final token converts them to Gate 2's position and destroys C4.

5. **Layer *and position* selection are nested inside the training fold.** 594
   combinations per fold is nine times Gate 2's 66; the selection surface is
   correspondingly easier to overfit.

6. **The teacher-forced pass is not optional.** Harvesting states from `generate` leaves
   p2 undefined for EOS-terminated answers — i.e. for the clean ones.

7. **C0 is reported whatever it shows.** A probe that turns out to be an
   answer-correctness classifier is a finding, and it is the finding most likely to be
   quietly omitted.

8. **Gate 2's null is the headline.** Gate 2b cannot revise it. A PROCEED here means "a
   different read position works", never "Gate 2 was wrong".

9. **A failed pooling check is NO VERDICT, not a STOP.** The two say different things. A
   STOP claims the probe does not separate; a failed pooling check says the primary number
   is not measuring separation, so we do not know. See *Secondary analyses*.

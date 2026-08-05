# Gate 2: Probe Feasibility

**Question.** Can a linear probe on internal states predict the knowledge label better
than `H(p_theta)` and `max p_theta`, which are free?

**Why this experiment.** Gate 1 measured the ceiling on a knowledge-gated decoder by
handing the gate ground-truth labels. It returned +13.83 macro points of headroom over
the entropy baseline (matched run; +10.33 unmatched). That number is only worth chasing
if the label can be *estimated*. Gate 2 replaces the oracle's ground truth with a probe
and asks whether the probe recovers anything the free confidence signals do not already
carry. If a linear read of the residual stream is no better than entropy, the headroom is
unreachable by this route and the project stops here — the same way a Gate 1 null would
have stopped it.

**What Gate 2 is not.** It does not re-run the decoder. Probe AUC plus the Gate 1
headroom is the claim; wiring probe outputs back into `tau` is Gate 3 and is deliberately
out of scope (see *Not in scope*).

**Cost.** ~40 min GPU for extraction, minutes of CPU for everything else.

---

## Phase 0 — Pre-registration

Frozen before Phase A runs. Do not edit after extraction begins.

| Decision | Value | Locked (date/initials) |
|---|---|---|
| Model | **Llama-3.1-8B-Instruct**, same weights and prompt as Gate 1 | |
| Probe | Logistic regression, L2, per layer, per read position | |
| Labels | `knowledge` from `labels.jsonl`, FUNCTIONAL pool only, discards excluded | |
| Read positions | (1) final question token, (2) last token of subject span | |
| Layers | all, embedding through final (33 for this model) | |
| Splits | 5-fold CV partitioned by `subject_qid`, nested inner split for layer + alpha | |
| Primary number | pooled out-of-fold test AUC, entity-disjoint | |
| Baselines | `H(p_theta)`, `max p_theta`, on identical folds | |
| PROCEED if probe AUC − entropy AUC >= | **0.03**, bootstrap 95% CI excluding 0 | |
| PROCEED also requires | within-relation (probe − entropy) >= **0.02**, CI excluding 0 (C1) | |
| STOP if | < 0.03, **or** the within-relation advantage disappears | |
| Bootstrap | 10,000 resamples, **resampled by `subject_qid`** | |
| Seed | 20260803 (same as Gate 1) | |

**Margins are read off the primary number only** — pooled, entity-disjoint CV. The
within-relation number in C1 is a gate, not a margin: it can veto a PROCEED but it does
not set the headline.

### What "the within-relation advantage survives" means

*This threshold was not in the spec. Raised before Phase A and set by the human on
2026-08-04, alongside the three alternatives considered (0.03 to match the primary; CI
only with no margin; point estimate only with no CI).*

> **LOCKED, 2026-08-04.** Survives = the n-weighted mean within-relation
> (probe AUC − entropy AUC) is >= **0.02** with a bootstrap 95% CI excluding 0.

Rationale for proposing 0.02 rather than reusing 0.03: the within-relation number is
computed inside relations of 60–400 rows, so it is strictly noisier than the pooled
number, and demanding the same margin on a noisier statistic makes the compound gate
harder than the stated primary rather than merely additional to it. A veto threshold
should be set where a relation classifier would fail it, not where sampling noise would.

### Disclosure: data touched before freezing

The pre-registration rule is "before touching data", and one check ran first. Before
writing this file I measured how often the Wikidata subject label appears verbatim in the
question text, because read position (2) is not implementable if it usually does not
(see Phase A). That check reads **question strings and subject labels only**. It does not
read `labels.jsonl`, any activation, or any outcome variable, and no number in it can
move a Gate 2 result in either direction. Recording it here rather than omitting it.

---

## Phase A — Extraction (GPU, ~40 min)

For each question in the FUNCTIONAL pool (`pool.jsonl`), run the **context-free** forward
pass and capture the residual stream at every layer at two token positions.

### The prompt is Gate 1's prompt, unchanged

Extraction calls `backend._build_prompt(question, None)` — the same builder, the same
frozen `DATE_STRING`, the same `add_special_tokens=False` encoding that Phase 2 used. Not
a reconstruction of it.

### Correction to the spec: this is a new forward pass, not a reused one

The spec says to reuse the pass `deterministic_features` already runs. That pass is
already spent: Phase 2 completed weeks-scale compute and persisted only `(entropy,
max_prob)` per question. Hidden states were never captured and cannot be recovered from
what was written. Phase A therefore re-runs the identical context-free forward pass with
`output_hidden_states=True`.

This does not conflict with Gate 1 invariant #2. That invariant forbids *merging*
`deterministic_features` with `sample_closed_book`, because sharing computation between
the label and the predictor handicaps the entropy baseline. Phase A shares computation
with neither: it is a third pass, run after both are complete, and it cannot change a
label or a baseline value that is already on disk.

### The parity check is stronger here than in Gate 1

Because the extraction pass recomputes the next-token distribution anyway, it recomputes
`(entropy, max_prob)` for free and compares them against the values `labels.jsonl` already
holds. If the extraction prompt differed from the Phase 2 prompt in any way — a stray
BOS, an unpinned date, a template version bump — those numbers would diverge. Agreement to
floating-point tolerance is positive evidence that the prompt is byte-identical, which
reading `prompt_parity.txt` cannot give.

Tolerance is not exact equality. bf16 matmuls are not bitwise reproducible across GPU
architectures, and Phase 2 ran across more than one runtime. Registered thresholds:

| max abs difference in entropy | reading |
|---|---|
| < 1e-3 | pass, numerically identical up to bf16 kernel nondeterminism |
| 1e-3 to 1e-2 | pass with a note; report the distribution |
| > 1e-2 on any question | **STOP.** The prompt is not the same. Do not train. |

The string-level check is run too, since it costs nothing: assert the rendered extraction
prompt equals `render_prompts(question, ctx)[0]`.

### Read position (2): locating the subject span

`pool.jsonl` stores `subject_qid` (a Wikidata Q-number) but not the subject's surface
string, so the span is recovered by joining back to `ConFiQA-QA.json` on a recomputed
`qid` and taking `orig_path_labeled[0][0]`. The join is exact: `_make_qid` is a hash of
fields present in both, and it resolved 4207/4207 rows on the labels pool.

The subject label does not always appear verbatim, because Wikidata labels carry
disambiguating parentheticals the question drops (`Shaft (2019 film)` vs "the film
Shaft (2019)"). Resolution is tried in tiers and the tier used is **recorded per question**:

| tier | measured on the 4207-row labels pool |
|---|---|
| verbatim substring | 91.54% |
| after stripping a trailing `(...)` | +7.58% |
| case-insensitive | +0.02% |
| unresolved | 0.86% |

Unresolved questions are **excluded from position (2) only** and retained for position
(1), and the count is reported. They are not silently backfilled with the final token —
that would quietly turn position (2) into position (1) for those rows and contaminate
exactly the C4 comparison the position exists to support. The residual is concentrated in
`follows`, which `DROP_RELATIONS` already removes, so the FUNCTIONAL-pool failure rate
should be below the 0.86% above; the realised number is a Phase A checkpoint item.

Character span → token index via the tokenizer's offset mapping on the rendered prompt,
taking the **last** token whose span intersects the subject's character range. Where the
subject string occurs more than once in the prompt, the occurrence inside the question
line is used, not one inside the system block.

### Storage

`float16`, sharded to survive a disconnect. Shape per position:
`(n_questions, n_layers, hidden)` = `(1793, 33, 4096)` ≈ 485 MB, ~970 MB for both.

Sharded at 250 questions per file with a manifest, rather than one array: Drive writes are
slow and a single 1 GB file rewritten on every checkpoint is a disconnect away from being
a corrupt 1 GB file. Resume granularity is one shard.

Position (2) shares the shard's forward pass — both positions come out of one `hidden_states`
tuple, so capturing two positions costs no extra GPU time.

### Checkpoint A — report and stop

Report: tensor shapes and dtype; bytes on disk; the entropy/max_prob agreement against
`labels.jsonl` against the table above; subject-span tier counts and the unresolved list;
`n` per label class after discards are dropped. The spec's expectation is n=1582; the
realised number is whatever it is, and if it differs materially that is reported, not
reconciled.

---

## Phase B — Training (CPU, minutes)

Logistic regression per (layer, position). L2, `alpha` tuned on an inner fold.

**Labels.** `knowledge` from `labels.jsonl`, restricted to `pool.jsonl` (FUNCTIONAL only),
`discard` rows dropped. ONE_TO_MANY rows are labelled but **are not trained on**: with
several correct answers, a "wrong" generation is often a correct answer that is not the
one Wikidata recorded, so their label measures answer canonicality rather than knowledge.
Training on them would teach the probe a different target than the one Gate 1 gated on.

**Splits.** 5-fold CV partitioned by `subject_qid`, with a further split inside each
training fold used to pick the layer and `alpha`. Nested, because choosing the layer on
data you then evaluate on flatters the result — with 33 layers × 2 positions there are 66
chances to overfit a selection, and that is exactly the size of effect Gate 2 is trying to
detect. Test AUC is pooled across folds. n≈1582 is thin for a single three-way split; CV
uses all of it.

Grouping is by `subject_qid` and not by `qid`: one entity generates several questions
across relations, and a probe that memorises "Llama knows things about Q60" would score on
a `qid`-disjoint split without encoding anything transferable.

**Baselines.** `H(p_theta)` and `max p_theta` scored on the identical folds. Both are
single scalars needing no training, so their "fold" is just the test slice — but they must
be scored on the same slices or the comparison is not paired. `max p_theta` is a
confidence, `H(p_theta)` is an uncertainty; entropy enters the AUC with its sign flipped so
both are oriented as "higher means more likely known".

### Checkpoint B — report and stop

Pooled AUC for probe, entropy and max-prob, plus the selected layer and position per fold.
This is the read on whether C1 is worth running at all.

---

## Phase C — The four tests

Run in this order. C1 is the one that decides.

### C1. Within-relation AUC — the test that matters

Compute AUC separately inside each relation, then average weighted by n. Same for both
baselines.

**Why.** Known-rates range from ~27% (`religion`) to 100% (`currency`). A probe that
merely identifies the relation from the question's wording inherits that spread and earns
AUC without encoding anything fact-specific. Read position (1) is the final question
token, which makes this the likely failure mode rather than a hypothetical one.

If the probe beats entropy pooled but **not** within-relation, it is a relation classifier
and H2 fails. **Both numbers are reported with equal prominence** — the pooled number is
never quoted without the within-relation number beside it.

Inclusion: relations with >= 60 labelled rows.

> **Operational addition, flagged.** AUC is undefined when a relation is single-class, and
> unstable when the minority class is a handful of rows — `currency` is 100% known and
> contributes no AUC at any n. Relations are therefore additionally required to have
> **>= 10 rows in the minority class**. Excluded relations are listed with their n and
> class balance rather than dropped silently, since "the probe could not be tested on the
> relations where the model knows everything" is itself a result about scope.

### C2. Leave-one-relation-out

Train on every relation but one, test on the held-out relation. Report per relation. This
is the generalisation claim the literature disputes. Layer and `alpha` are selected inside
the training relations only.

### C3. Spouse-excluded

`spouse` is ~23% of the pool and ~35% of the Gate 1 cells. Re-run C1 with it dropped. If
the advantage lives in `spouse`, say so plainly.

### C4. Read position

Final-question-token vs subject-entity-token: per-layer AUC curve for both. Ferrando et
al. found entity-recognition directions at the entity position that causally control
refusal, and it may be a cleaner "do I know this" signal than the final question token,
which is contaminated by relation phrasing. Inside-Out found the signal mid-to-late;
confirm or contradict.

The two positions are compared on the questions where **both** resolve, so a position is
never advantaged by being scored on an easier subset.

### Statistics

**DeLong's test** for paired AUC comparisons: the predictors are evaluated on the same
questions, so their errors are correlated and an unpaired test would overstate
significance. sklearn has no implementation; the fast algorithm (Sun & Xu 2014) is
implemented in `gate2.py` and unit-tested against a brute-force computation.

**Bootstrap CIs resampled by `subject_qid`**, as in Gate 1 — matching the split unit, so
the interval reflects the same entity-level exchangeability the splits assume.

> **The two are not interchangeable and the pre-registration uses the bootstrap.** DeLong
> assumes two scores on one fixed test set from models that did not vary. Under 5-fold CV
> the pooled out-of-fold predictions come from five different fitted probes, which
> violates that assumption; the bootstrap does not care. So: the **bootstrap CI is the
> inferential number the PROCEED margin is read against**, and DeLong is reported
> alongside as a secondary paired test. Where they disagree, the bootstrap governs and the
> disagreement is reported.

---

## Phase D — Calibration (only if C1 passes)

Platt scaling on held-out folds. Report expected calibration error and a reliability
curve alongside AUC.

**Why this is separate from AUC.** The eventual gate multiplies `k` into the blend
strength, so ranking correctly is not enough. A probe that ranks perfectly but reports 0.9
on cases it gets right 60% of the time produces a systematically mis-scaled correction —
right order, wrong magnitude, and the magnitude is what `tau` consumes.

---

## Not in scope, deliberately

- **Re-running the decoder with probe-derived `k`.** The AUC result plus the Gate 1
  headroom carries the paper. Add later if time allows.
- **Inside-Out internal-knowledge labels.** Generation labels work; comparing label
  schemes is a separate contribution and would confound this one.
- **A second model.** Gate 1's contingency clause was model-specific and did not fire.

---

## Deliverables

| Artifact | Phase |
|---|---|
| `acts_pos{1,2}_shard*.npz` + `acts_manifest.json` | A |
| `extract_parity.json` — entropy agreement, span tiers, unresolved list | A |
| `probe_folds.json` — per-fold selected layer, alpha, test AUC | B |
| `gate2_results.json` — pooled + within-relation + LORO + spouse-excluded + positions | C |
| `calibration.json` + reliability curve data | D |
| Console output of `gate2.py tests`, saved | C |

---

## Invariants — do not "fix" these

1. **Extraction is a third forward pass.** It does not reuse, and must not be merged
   into, `sample_closed_book` or `deterministic_features`. See Phase A.

2. **Layer selection is nested inside the training fold.** Selecting the layer on the
   pooled test scores would be the single easiest way to manufacture a passing Gate 2.

3. **Splits are grouped by `subject_qid`, not `qid`.** Row-level or question-level
   splitting leaks the entity across the fold boundary.

4. **Entropy is sign-flipped when scored as an AUC predictor.** It is an uncertainty and
   the label is "known". An AUC of 0.35 for entropy would mean it was entered unflipped,
   not that entropy is anti-predictive.

5. **Unresolved subject spans are dropped from position (2), not backfilled.** Backfilling
   with the final token silently converts them to position (1) and destroys C4.

6. **ONE_TO_MANY rows are labelled but never trained on.** Their label measures answer
   canonicality, not knowledge.

7. **The within-relation number is reported every time the pooled number is.** The pooled
   number alone is the exact artifact C1 exists to detect.

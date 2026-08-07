# Gate 3a: Answer Verification, Not Question Classification

**Question.** Given a question and a *specific candidate answer*, can a linear readout of
internal activations rank correct answers above incorrect ones better than the model's own
output-level scores can — **including for answers the model would never generate**?

**Why this is a different experiment, not a retry.** Gates 2 and 2b asked *"does the model
know this question?"* and answered it across questions. Both failed, and Gate 2b's
post-mortem showed why the failure was uninformative rather than merely negative:

| test | probe − entropy |
|---|---|
| pooled | +0.0296 (bar was +0.03) |
| answer unseen in training | +0.0088 |
| answer unseen, within relation | −0.0019 |
| relation held out entirely (C2, 8/8) | −0.0082 |

The pooled advantage was **answer-entity memorisation**: answer identity alone predicts the
Gate 2b label at 0.7029 AUC, 45.4% of test rows had their gold answer present in training,
and inside `country of citizenship` the probe's edge swings from **+0.1520 on seen answers
to −0.0729 on unseen ones** with relation and question template held fixed.

Gate 3a changes the unit of analysis so that neither leak can operate.

**Gate 2b's STOP stands and is not revised.** Gate 3a reports beside it, never in place of
it. A PROCEED here does not retroactively rescue Gate 2b.

---

## Phase 0 — Pre-registration

Frozen before any extraction. Do not edit after Phase A begins.

| Decision | Value | Locked |
|---|---|---|
| Model | **Llama-3.1-8B-Instruct**, same weights, prompt and `DATE_STRING` as Gates 1–2b | 2026-08-06 |
| Hardware | **L4** — bitwise parity with Phase 2 is required, see *Hardware* in GATE2B_PROTOCOL | 2026-08-06 |
| Unit | one row per **(question, candidate answer)** pair | 2026-08-06 |
| Label | is *this answer* correct, by `gate1.alias_match` | 2026-08-06 |
| Probe | logistic regression, L2, per layer, on the answer-token read of `(q, a)` | 2026-08-06 |
| Primary metric | **K** — within-question fraction of correct/incorrect pairs ranked right, averaged over questions | 2026-08-06 |
| Secondary metric | **K\*** — fraction of questions with *all* pairs ranked right | 2026-08-06 |
| Baselines | `P(a\|q)`, `P_norm(a\|q)`, **`P(True)`** | 2026-08-06 |
| Comparator | **BEST of the three**, computed on identical questions | 2026-08-06 |
| PROCEED if K(probe) − K(BEST) >= | **0.03 absolute**, entity-clustered bootstrap 95% CI excluding 0 | 2026-08-06 |
| STOP | otherwise | 2026-08-06 |
| Splits | 5-fold, **subject- AND object-disjoint** | 2026-08-06 |
| Candidates per question | **100** sampled at T=1.0, deduped, **plus gold injected if absent** | 2026-08-06 |
| Bootstrap | 10,000 resamples, resampled by `subject_qid` | 2026-08-06 |
| Seed | 20260803 | 2026-08-06 |

**Why +0.03 absolute rather than Inside-Out's relative framing** — locked by the human,
2026-08-06. It keeps Gates 2, 2b and 3a on one scale. Inside-Out's Llama gap of ~14%
relative corresponds to roughly +0.10 absolute at their reported K levels, so this bar is
*modest*: if the effect replicates at published magnitude we clear it comfortably, and we
avoid manufacturing a false null by setting the bar at someone else's headline number.

**Why Llama-3.1-8B despite it being their weakest model** — locked by the human,
2026-08-06. Gate 1's +13.83 ceiling is Llama-specific. A probe validated on Gemma could not
be connected to a measured ceiling in Gate 3b without re-running Gate 1 entirely. Inside-Out
reports 14% for Llama-3-8B against 57% for Gemma-2-9B, so a null here is weaker evidence
than a null on Gemma would be, and that limitation is registered in advance rather than
discovered in the discussion section.

**`P(True)` is expected to be the binding baseline.** Inside-Out: it "outperforms other
external functions in every setting" and "accounts for the relatively low magnitude of
hidden knowledge in Llama." We have never tested a verification baseline. If the probe beats
`P(a|q)` and `P_norm` but not `P(True)`, the honest finding is that *the model can verify
what it cannot generate*, and the probe adds nothing over asking it.

### Disclosure: what was read before freezing

Inside-Out (Gekhman et al., arXiv:2503.15299v4) was read in full before this file was
written, including §3.2, §A.4 and §A.5. Its methodology is adopted deliberately and is cited
throughout. No Gate 3a data exists yet.

---

## Phase A — Candidate generation and extraction (GPU)

For each question in the FUNCTIONAL pool:

1. **Sample 100 candidate answers** at temperature 1.0, `max_new_tokens=32`. Dedupe on
   `gate1.normalize`.
2. **Inject the gold answer if absent.** Inside-Out needed this in **64%** of cases, and it
   is the only route to the hidden-knowledge population. Record whether it was injected.
3. **Label each candidate** correct/incorrect with `gate1.alias_match` against
   `gold_aliases`.
4. **Read activations** at the answer tokens of the sequence "question, then candidate" —
   the same teacher-forced construction `backend.answer_states` already performs, extended
   to accept an arbitrary candidate rather than only the greedy output.
5. **Score the three baselines** on the same forward pass where possible: `P(a|q)`,
   `P_norm(a|q)`. `P(True)` needs its own verification prompt and pass.

### 100 candidates, not 1,000

Inside-Out sampled 1,000. We sample 100 and register the reason: their 1,000 exists to catch
answers sampled with vanishing probability, and **gold injection already guarantees the
correct answer is present** regardless of sampling depth. What 1,000 buys beyond 100 is a
richer set of *incorrect* candidates, which affects the denominator of K but not whether the
correct answer is scoreable. Cost scales linearly and 1,000 would be ~24 GPU-hours.

**This is a deviation from the source methodology and is declared as one.** If K is unstable
under candidate count — checkable by recomputing K on a random 50-candidate subsample — that
is a finding and is reported.

### Storage

Activations are the binding constraint: `n_questions × n_candidates × n_layers × 4096`. At
~10 deduped candidates per question that is ~4.7 GB for all 33 layers.

**Registered layer window: 10–25 inclusive (16 layers).** Gate 2b's C4 measured the
answer-token signal peaking at L15–19 and plateauing; Inside-Out reports quality rising and
stabilising from "around layers 11–12 out of 32". Storing outside that window costs GB for
layers both sources agree are worse. The window is fixed **now**, before any Gate 3a data
exists, so it cannot be tuned to a result.

### Parity

`answer_states` computes the parity pair from its own prompt-only pass (corrected
2026-08-05). Checkpoint A reports the string-level prompt check and the bitwise comparison
against `labels.jsonl`, on the three-tier tolerance. **Must run on an L4** — an A100 fails
these by hardware alone, as GATE2B_PROTOCOL *Hardware* records.

### Checkpoint A — report and stop

Span statistics; candidates per question after dedup; **gold-injection rate** (compare to
Inside-Out's 64%); label balance per question; count of questions with no incorrect
candidate (K is undefined there — these are excluded and counted); the parity table; and
**K for all three baselines before any probe is trained.**

If `P(True)` alone already reaches the probe's plausible ceiling, we know before training.

---

## Phase B — Knowledge-aware training

**The training set is built the way Inside-Out §A.4 prescribes, and this is the core fix.**

Restrict to questions where **greedy decoding is already correct**, so the model
demonstrably knows the fact. Then:

- **positive** = that correct greedy answer
- **negative** = sample at temperature 1.0 **on the same question** until an incorrect
  answer appears

Every training pair comes from one question the model knows. The probe cannot learn "this
question is easy", because every training question is easy — it is forced to discriminate
*answers*.

> **Why this is the fix.** Gate 2b trained on Inside-Out's category (A) *knows+correct* and
> (D) *doesn't-know+incorrect*, which is why greedy correctness predicted its label at 0.986
> AUC and why C0 had **3 usable rows**. Knowledge-aware probing populates (A) and **(B)**
> *knows+incorrect* instead. §A.4 warns about exactly the failure we hit, and names
> Orgad et al. — the paper Gate 2b is built on — as an instance of the approach it warns
> against.

**Splits: subject- and object-disjoint.** Gate 2b grouped by `subject_qid` only, and the
leakage check showed that is not enough. A question is assigned to a fold only if neither
its subject nor its gold object appears in any other fold. The realised fold sizes and the
number of questions dropped to satisfy disjointness are reported — if disjointness is
impossible at 5 folds given low answer cardinality (`continent` has 4 answers for 46
questions), that is a finding, not something to relax.

Layer selected on an inner split, never on the evaluation fold.

### Checkpoint B — report and stop

K and K\* for probe and all three baselines, per-fold layer selections, and the margin
against BEST.

---

## Phase C — Tests

**C1. Within-relation K.** As in Gate 2b. Retained even though the within-question metric
should make relation effects structurally impossible — precisely so that claim is checked
rather than assumed.

**C2. Leave-one-relation-out.** Inside-Out never ran this; it is where Gate 2b died
(−0.0082 vs entropy across 8/8 relations). It is the deployment-relevant test and it is
registered as a **diagnostic**, not a gate.

**C3. Gold-injected vs naturally-sampled.** Split K by whether the gold answer had to be
injected. **This is the hidden-knowledge measurement**: questions where the model never
generated the correct answer in 100 samples, yet the probe ranks it above the incorrect
ones, are hidden knowledge by Inside-Out's definition. Reported with n.

**C4. Layer curve** across the registered window, for the probe and for reference.

**Statistics.** Entity-clustered bootstrap governs; resampled by `subject_qid`. The
within-fold rank-pooling secondary from GATE2B_PROTOCOL does not apply — K is computed
within question, so cross-fold score comparability never arises. That is a structural
advantage of this design and is noted as such.

---

## Not in scope

- **Gate 3b, closing the decoder loop.** Conditional on 3a passing. Building the gate before
  knowing there is a signal to gate on is the error this project has avoided three times.
- **A second model.** Registered as a known limitation above.
- **1,000 candidates.** See Phase A.

---

## Invariants — do not "fix" these

1. **The unit is (question, candidate answer), and K is computed WITHIN a question.** Any
   aggregation that compares scores across questions reintroduces the leaks Gate 2b died of.

   > **CORRECTION, 2026-08-06, before any Gate 3a data existed.** As frozen, this invariant
   > said K "makes answer-identity and relation priors cancel." **The answer-identity half
   > is wrong**, and `selftest3` now checks both halves rather than asserting them:
   >
   > - **Cancels exactly.** Anything constant within a question — relation identity,
   >   question difficulty, entity popularity, question wording. Every pair such a scorer
   >   sees is a tie, so it scores exactly 0.5. Verified.
   > - **Does NOT cancel.** Answer identity. The candidates within a question are
   >   *different answer strings*, so a prior over answers still varies inside a pair and
   >   can rank it — a fixture scorer using answer identity alone reaches K far from 0.5.
   >
   > The answer-identity leak — the one that killed Gate 2b — is therefore closed by the
   > **subject- AND object-disjoint split** (invariant 5), not by the metric. Invariant 5
   > is load-bearing, not belt-and-braces, and must not be relaxed if the fold builder
   > struggles with low-cardinality relations.

2. **Training pairs come from questions the model knows.** Training on all questions
   recreates the (A)/(D) conflation and the 3-row stratum. See Inside-Out §A.4.

3. **Negatives are model-generated answers to the SAME question.** Fabricated negatives
   teach the probe likelihood rather than correctness — Inside-Out §A.4, citing Marks &
   Tegmark and Azaria & Mitchell.

4. **Gold is injected when absent, and the flag is retained.** Without it the
   hidden-knowledge population is invisible; without the flag C3 cannot be computed.

5. **Folds are subject- AND object-disjoint.** Subject-only grouping is what produced Gate
   2b's +0.0438-vs-+0.0088 split.

6. **`P(True)` is in the baseline set and BEST is the maximum over all three.** It is the
   strongest external scorer on this model and omitting it would flatter the probe.

7. **Gate 2b's STOP is not revised by anything here.**

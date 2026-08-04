# Gate 1: Oracle Headroom Experiment

**Question.** Does ground-truth knowledge of a fact predict correct source-arbitration
better than freely-available output confidence does?

**Why this experiment.** The full method has four gates in series: (1) does the quantity
carry information, (2) can a probe estimate it, (3) does the estimate survive compression
into a decoding gate, (4) do gains land where the mechanism predicts. This experiment
tests Gate 1 with the probe replaced by ground truth, so probe quality is held at perfect
by construction. If Gate 1 fails, no probe can rescue the project.

**Cost.** ~1 week including plumbing. A few GPU-hours for generation, ~1 day for the
decoder sweep on a single A100.

---

## Phase 0 — Pre-registration

Fill in and freeze before touching data. Do not edit after Phase 4 begins.

| Decision | Value | Locked (date/initials) |
|---|---|---|
| Model | **Llama-3.1-8B-Instruct** (committed, no selection step) | |
| Contingency model | Gemma-2-9B-Instruct, **only** under the clause below | |
| Decoder | CAD, tau-parameterized | |
| tau grid | 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0 | |
| Known threshold | gold in >= 8/10 samples | |
| Unknown threshold | gold in <= 2/10 samples | |
| Target cell size | 300 | |
| STOP if Oracle - Entropy < | 2.0 points macro | |
| STOP if Oracle - Constant < | 3.0 points macro | |
| STOP if off-diagonal < | 15% of instances | |
| PROCEED if Oracle - Entropy >= | 4.0 points macro, 95% CI excluding 0 | |
| AMBIGUOUS band | 2.0 <= Oracle - Entropy < 4.0 -> trigger contingency clause | |

### Contingency clause (pre-registered, not post-hoc)

> If `Oracle - Entropy` lands in the ambiguous band (2.0-4.0 points macro), run Phase 2
> plus a **reduced sweep** (tau grid only, gamma/delta fixed at the Llama optimum) on
> Gemma-2-9B-Instruct before concluding. Do **not** run this if the Llama result is a
> clear STOP or a clear PROCEED.

**Why this clause exists.** Llama-3.1-8B is the best-calibrated of the three Inside-Out
models on this axis, which makes entropy a strong baseline. That yields a clean positive
but a muddy negative: a 1.5-point gap cannot distinguish "the idea is wrong" from "this
model does not need the probe." Registering the escape route in advance is what separates
a robustness check from model shopping. Firing it outside the ambiguous band is model
shopping and invalidates the pre-registration.

### Model note for later comparability

Inside-Out measured its 14% hidden-knowledge gap on **Llama-3-8B-Instruct**, not 3.1.
These are different models. When Gate 2 probe AUCs are compared against Inside-Out's,
say so explicitly rather than conflating them.

**Decoder rule.** Parameterize by tau rather than alpha so the gate can move in both
directions with no code change:

```
log q_t(y) = (1 - tau_t) * log p_theta_t(y) + tau_t * log p_ctx_t(y)     [unnormalized]
```

- `tau = 1`   -> plain decoding with the document in the prompt
- `tau > 1`   -> extrapolation, push past the document, suppress prior-favored tokens
- `tau < 1`   -> interpolation, pull back toward parametric memory
- `tau = 1 + alpha` recovers CAD exactly

**Use CAD, not CoCoA.** This experiment does not test whether your gate beats the state
of the art; it tests whether a knowledge signal changes anything at all. CoCoA has no
working public code and a suspected sign error in its printed gating equation. Defer that
reimplementation.

---

## Phase 1 — Instance pool

**1.1** Download ConFiQA (Context-DPO repo, `byronBBL/Context-DPO`) and FiQA. Use the
**QA subset only**. MR and MC are multi-hop; "does the model know this fact" is not
well-posed when the answer requires composing several facts it may know individually and
still fail to chain. Keep MR/MC as a scoping limitation in the writeup, not as data.

**1.2** Join ConFiQA-QA and FiQA on question ID. Each row should carry:

```
question, subject_qid, gold_answer, gold_aliases,
cf_answer, cf_aliases, factual_context, counterfactual_context
```

ConFiQA ships `orig_triple` / `cf_triple`, `orig_answer` / `cf_answer`,
`orig_alias` / `cf_alias`, and `orig_context_piece` / `cf_context_piece` natively, so the
document-correctness axis is inherited rather than constructed. This removes a whole
source of noise you would otherwise own.

**1.3 — Filters.** Drop rows where:
- `cf_answer` is an alias-collision with `gold_answer` (silently converts resistance into agreement)
- the two contexts differ in more than the target entity
- `subject_qid` is missing (you need it for deduplication later)

**1.4** Record pool size after each filter. Report the attrition chain.

---

## Phase 2 — Knowledge labels

**2.1 — Closed-book sampling.** 10 generations per question, no document, temperature 0.7,
`max_new_tokens` ~32. Use the exact chat template you will use at decode time.

**2.2 — Alias-aware scoring.** Normalize case, strip leading articles, then substring-match
against the alias set. Do **not** use exact string equality. Alias handling is where most
label noise in this literature originates, and ConFiQA hands you the lists.

**2.3 — Labels.**

| Gold appears in | Label |
|---|---|
| >= 8/10 | known |
| <= 2/10 | unknown |
| 3-7/10 | DISCARD |

**2.4 — Record the discard rate.** Expect 20-30%. Report it. Dropping the middle band
inflates effect sizes relative to a deployment setting and the reader should see by how much.

**2.5 — Separate deterministic feature pass.** One greedy forward pass per question, no
document. Record `H(p_theta)` and `max p_theta` at the first answer position.

> **Why 2.5 must be a separate pass.** The knowledge label in 2.1-2.3 comes from sampling,
> which is itself a confidence measurement. If the entropy baseline is computed from those
> same generations, label and predictor share noise, entropy is handicapped, and your result
> looks better than it is. Different pass, clean prompt.

**2.6 — Sanity checkpoint (single model).** The model is committed, so this is no longer a
selection step. Run it on Llama-3.1-8B-Instruct only and check two things before spending
a day on the sweep:

- **Cell balance.** Both `known` and `unknown` groups must be populated well enough to
  reach ~300 instances per cell after crossing. If `known` is thin, ConFiQA-QA is too
  long-tail for this model and you need a broader question pool.
- **Off-diagonal fraction.** This is a Phase 7 stop condition; get it now, not after the
  sweep. Below 15% and the sweep is unlikely to show anything regardless of the arms.

Skip the Gemma labeling run unless the contingency clause fires.

**2.7 — Prompt-parity check (Llama-specific, do not skip).** Llama-3.1's chat template
injects a date-cutoff line into the system prompt by default in some `transformers`
versions. If the closed-book pass and the RAG pass end up with different system prompts,
`p_theta` and `p_ctx` are not the same distribution over the same conditioning, and the
contrastive subtraction measures template drift alongside document effect.

Log the exact rendered prompt string for both passes on the first 10 instances and confirm
they differ **only** by the document block. Save both to `prompt_parity.txt` as an artifact.

---

## Phase 3 — The 2x2

Cross the knowledge label with the document condition. Each question yields two instances.

|  | Factual context (FiQA) | Counterfactual context (ConFiQA) |
|---|---|---|
| **Known** | agreement | **resistance** |
| **Unknown** | **correction** | both-wrong |

**3.1** Assign every (question, context) pair to a cell.

**3.2 — Targets.** In *all* cells the target is the **factual gold answer**. In resistance
the document is wrong, so following it is the error. This is the cell that does not exist
on NQ-SWAP, and it is the reason this slice is being built.

**3.3** Balance to ~300 per cell by subsampling. Report the natural distribution first.

**3.4** Hold `both-wrong` out of the headline metric — neither source helps there, so it
measures noise. Report separately.

> **Why the crossing matters.** Knowledge and document-correctness are now orthogonal by
> construction. That is what makes the comparison non-circular: the knowledge label cannot
> predict document-correctness by definition, because they were assigned independently.

---

## Phase 4 — Four decoder arms

Identical decoder, four sources for `tau`.

| Arm | tau source | Tuned parameters |
|---|---|---|
| A. Constant | `tau = tau_0` | `tau_0` |
| B1. Oracle one-sided | `tau_0` if known, `tau_0 + gamma` if unknown | `tau_0, gamma` |
| B2. Oracle two-sided | `tau_0 - delta` if known, `tau_0 + gamma` if unknown | `tau_0, gamma, delta` |
| C. Entropy | same form as B, known/unknown from thresholded `H(p_theta)` | `tau_0, gamma, delta, thr` |
| D. Max-prob | same form as B, thresholded on `max p_theta` | `tau_0, gamma, delta, thr` |

**4.1 — Equalize tuning budgets.** Every arm gets the same grid density and the same number
of evaluations. Arm A's `tau_0` gets swept as hard as Arm B's `gamma`. Unequal tuning
budgets are the single most common way "adaptive beats fixed" turns out to be an artifact.

**4.2 — Give C and D their best thresholds**, tuned on the same data they are evaluated on,
i.e. optimistically. Bias the comparison *against* your hypothesis; a reviewer will.

**4.3 — Log mean output length per arm.** High tau produces degenerate text. If length
collapses, the tau grid is too aggressive at that end and the comparison is invalid there.

---

## Phase 5 — Metrics

**Primary: macro faithfulness** = mean(correction accuracy, resistance accuracy).

> A decoder that always follows the document scores 100% correction / 0% resistance.
> Aggregate accuracy would make that look reasonable. Averaging makes one-sided degeneracy
> visible immediately.

**Always report alongside:**
- correction accuracy and resistance accuracy **separately** (a +4 / -4 trade is not a tie)
- agreement accuracy — the over-correction damage metric, the axis on which CAD collapses
- both-wrong accuracy, separately

Scoring uses the same alias matcher as Phase 2, against the factual gold.

---

## Phase 6 — Statistics

**6.1** For each arm, select its best operating point by macro faithfulness.

**6.2** Compute `Oracle - Entropy` at those points with a 95% bootstrap CI. **Resample by
question, not by instance** — each question contributes two correlated instances, and
row-level resampling would treat them as independent and artificially shrink the interval.

**6.3** Report `Oracle - Constant` too. That is total headroom; `Oracle - Entropy` is the
portion a probe could claim.

**6.4** Report the knowledge x thresholded-entropy contingency table. The off-diagonal count
(confident-but-unknown, uncertain-but-known) is the population the mechanism depends on.

**Power.** At ~300 instances/cell you can detect roughly a 5-point macro difference.
To detect 2 points, quadruple the cells.

---

## Phase 7 — Decision

Against the Phase 0 thresholds:

- `Oracle - Entropy < 2 pts` -> **STOP.** Free signals already carry the useful information.
- Off-diagonal `< 15%` -> **different failure.** Mechanism real but too rare to move numbers.
  Fix the slice before abandoning the idea.
- `Oracle - Constant < 3 pts` -> **STOP.** Knowledge does not change decoding outcomes at all.
- `2 <= Oracle - Entropy < 4 pts` -> **CONTINGENCY.** Fire the Phase 0 clause: Gemma-2-9B
  Phase 2 plus reduced sweep. A gap that widens materially on Gemma indicates the Llama
  null was a calibration artifact; a gap that stays flat is a genuine stop.
- `Oracle - Entropy >= 4 pts`, CI excludes 0 -> **PROCEED** to Gate 2. Quote the gap as the
  headroom budget the probe must fill.

**Also read off which oracle arm won.** If B2 (two-sided) beats B1 substantially, that
settles the design question — and it means rewriting the decoding-rule section of the
proposal, not just the framing.

---

## Controls

| Control | Why | How |
|---|---|---|
| Popularity stratification | ConFiQA is Wikidata-derived; entity fame drives both knowledge and confidence. If the oracle advantage vanishes within strata it was detecting fame. | Pull Wikipedia pageview counts by QID, re-run 6.2 within terciles |
| Prompt sensitivity | If labels flip across phrasings they measure the prompt, not the model | Re-run Phase 2 on 200 questions with a paraphrased template; report flip rate |
| Entity dedup | Later phases train a probe; overlap with eval entities = probe has seen the test fact | Deduplicate by **Wikidata QID**, not surface string ("NYC" vs "New York City") |
| Degenerate generation | High tau breaks fluency and invalidates that end of the sweep | Mean output length per arm (4.3) |
| Sampling leakage | Shared noise between label and predictor | Enforced by 2.5 |

---

## Deliverables

1. `pool.jsonl` — filtered instance pool with attrition chain
2. `labels.jsonl` — knowledge labels + entropy features, both candidate models
3. `contingency.csv` — knowledge x entropy 2x2, with off-diagonal fraction
4. `sweep.csv` — every (arm, hyperparameter, cell) accuracy
5. `results.md` — headroom table, CIs, decision against Phase 0 thresholds
6. **Figure 1** — scatter of knowledge label vs entropy, disagreeing cases highlighted
7. **Figure 2** — macro faithfulness vs tau_0 per arm, the headroom plot

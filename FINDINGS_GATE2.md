# Gate 2: what we found

**Status.** Complete. The pre-registered result is in, along with four diagnostics and one
declared follow-up analysis. Gate 3 has not been started and, on this evidence, should not
be.

---

## 1. The question

Gate 1 asked whether knowing — for certain — that a model knows a fact would help it decide
whether to trust a document. The answer was yes: about 14 points of improvement on our main
measure. But Gate 1 got that answer by cheating. It was handed the true answer to "does the
model know this?" from a lookup table.

Gate 2 asks the obvious follow-up. **Can you work out that same answer from inside the
model, well enough to be worth the trouble?** And specifically: can you do better than a
signal you already get for free?

The free signal is the model's own hesitancy. Every time a model produces text it also
produces, invisibly, a spread of alternatives it considered. When it is sure, that spread is
narrow. When it is guessing, the spread is wide. Measuring that spread costs nothing — the
model computed it anyway. The technical name is entropy, and throughout this document
"hesitancy" means exactly that.

Against that, we tested a **probe**: a simple pattern-detector trained on the model's
internal state. As the model reads a question, it holds a large array of numbers
representing its partial understanding — 4096 numbers at each of 33 processing stages. The
probe looks at those numbers and tries to predict whether the model knows the answer. It is
deliberately simple, drawing a single dividing line through that space, because anything
more complicated could not be run cheaply enough to be useful inside a decoder.

---

## 2. The short answer

**Hesitancy is already very good, and the probe adds essentially nothing.**

Scored on how well each one separates facts the model knows from facts it doesn't, on
questions about entities the training never saw:

| what we measured | score |
|---|---|
| the probe, on internal state | 0.869 |
| **hesitancy, free** | **0.860** |
| the model's confidence in its single best answer, also free | 0.843 |

The probe beats hesitancy by 0.009. We had committed in advance to requiring 0.03, with the
uncertainty range excluding zero. It clears neither. The plausible range for the difference
runs from −0.008 to +0.027 — it includes zero, and its *entire span* sits below the 0.03 we
needed. So this is not a near miss that a bit more data would resolve.

A score of 0.5 means useless — no better than a coin flip. A score of 1.0 means perfect. On
that scale, free hesitancy at 0.860 is genuinely informative, and 4096 numbers of internal
state buy you 0.009 more.

---

## 3. What that means alongside Gate 1

The two results together tell a coherent story, and it is more interesting than either
alone.

Gate 1 found real headroom. Knowing the truth about the model's knowledge was worth about 14
points, and that finding stands.

Gate 2 says **you cannot buy that knowledge cheaply**, because the thing you would use to
estimate it is mostly the same thing you already had. The gap Gate 1 exploited — between
what the model knows and what it sounds like it knows — turns out to be largely visible in
how it sounds. Not entirely. But the leftover part, the bit hesitancy misses, is not
recoverable by a simple pattern-detector reading the model's internals.

An analogy. Gate 1 established that a perfect lie detector would be valuable in this
negotiation. Gate 2 checked whether we can build one by watching the person's face, and
found that watching their face tells us almost exactly what listening to their voice already
told us. The value was real. The route to it is closed.

---

## 4. How we tested it

**The data.** 1,793 questions about factual relationships — who composed a film's score, who
someone married, what language a person speaks natively. These survived the extensive
filtering documented in the Gate 1 write-up. Of those, 1,582 had a clear verdict on whether
the model knows the answer; 211 were borderline and set aside in advance. One more was
dropped for a technical reason described in section 8. Final total: **1,581 questions, 972
known, 609 unknown.**

**Reading the model's mind.** For each question we ran the model once and recorded its
internal state at every one of its 33 processing stages, at two different moments: at the end
of the question, and at the last word of the entity being asked about. That's about a
gigabyte of numbers, produced in under four minutes.

**Training and testing honestly.** This is where these experiments usually go wrong, so it
is worth being explicit about three precautions.

*We never tested on an entity we trained on.* The questions were split into five groups by
which entity they concern, and each group was tested by a probe trained only on the other
four. A probe that memorised "this model knows things about New York City" would score well
on an ordinary split and would have learned nothing transferable.

*We never chose the settings using the test.* The probe has choices to make — which of the
33 stages to read, which of the two moments, how strongly to constrain itself. Those choices
were made using a slice held out from the *training* portion, never the test portion. With 66
combinations available, choosing on the test data would be 66 chances to get lucky, which is
about the size of the effect we were trying to detect.

*We gave the free baselines every advantage.* They were scored on exactly the same questions
in exactly the same groupings.

---

## 5. The diagnostic that explains the result

The headline number was already a failure, so the remaining tests could not change the
verdict. We ran them anyway to learn *why*, and one of them was decisive.

**Almost all of the probe's tiny advantage is not about facts at all. It is about question
wording.**

The model's knowledge is very uneven across question types. It knows the currency of every
country we asked about — 100% — and only a third of the religions. So a probe that merely
recognises "this is a currency question" from the phrasing scores well without knowing
anything about the specific fact.

To test that, we scored every predictor *within* each question type separately, where
recognising the type is worth nothing:

| | pooled | within question type | change |
|---|---|---|---|
| probe | 0.869 | 0.797 | −0.072 |
| hesitancy | 0.860 | 0.799 | −0.061 |
| best-answer confidence | 0.843 | 0.790 | −0.053 |

The probe's small lead **vanishes and turns slightly negative**: −0.002, against the 0.02 we
had said in advance would count as surviving.

Two things worth noticing. First, this is exactly the artifact we built the test to catch,
and it caught it. Second — and we did not expect this — **hesitancy is partly a question-type
detector too.** All three predictors lose five to seven points when question type stops being
a free clue. The baseline is not clean either; the probe simply has no edge once the shared
crutch is removed. (These two columns are not perfectly comparable: the within-type figures
cover 1,326 questions, since seven question types were too small or too one-sided to score.)

**Per question type, the probe is wildly inconsistent** — it wins four and loses four, by
much larger margins than the average suggests:

| type | n | model knows | probe | hesitancy | difference |
|---|---|---|---|---|---|
| performer | 61 | 44% | 0.934 | 0.770 | **+0.163** |
| country of citizenship | 172 | 83% | 0.727 | 0.620 | **+0.107** |
| sport | 174 | 74% | 0.961 | 0.914 | +0.047 |
| director | 165 | 75% | 0.879 | 0.868 | +0.011 |
| religion | 128 | 33% | 0.688 | 0.705 | −0.017 |
| composer | 188 | 33% | 0.868 | 0.925 | −0.056 |
| spouse | 332 | 50% | 0.758 | 0.820 | **−0.061** |
| native language | 106 | 73% | 0.560 | 0.636 | **−0.077** |

On `native language` the probe is at 0.560 — nearly useless.

---

## 6. The other three diagnostics

**Does it transfer to unfamiliar question types?** We trained on every type but one and
tested on the one withheld. Average score: 0.760, against 0.797 when the type was included in
training. Transfer costs about 0.037 — real, but not collapse. `sport` transfers well
(0.913); `religion` (0.596) and `native language` (0.598) barely beat a coin flip.

**Is the result an artifact of one dominant question type?** `spouse` is the largest group at
332 questions, 21% of the total. We had asked in advance whether the probe's advantage lived
there. It turned out the opposite way: the *disadvantage* does. Removing `spouse` moves the
within-type difference from −0.002 to **+0.018**, a sign flip that approaches the 0.02 bar.
This is worth stating plainly, but it does not change anything — it is a decision made after
seeing the results, to remove the single biggest group, and it still falls short.

**Where in the model does the signal live, and does any stage do better?** We scored all 33
stages at both reading moments. This produced the most useful robustness check in the study:

- The best single stage scores **0.879** — better than the 0.869 the honest procedure found,
  as expected, because this number was allowed to look at the test data to pick its stage.
  Against hesitancy's 0.860 that is **+0.020**: still short of 0.03. **Even handing the probe
  its best stage for free does not clear the bar.** The failure is not bad luck in the
  selection.
- Signal climbs steadily to about stage 18 and then plateaus.
- At stage 0 — before the model has done any processing — the score is 0.484, i.e. chance.
  That is the check that the setup works at all.

---

## 7. The secondary negative result: where you read matters, and one place is simply worse

We read the model's state at two moments, because published work had found that the point
where a model recognises an entity carries a distinct "do I know this thing?" signal, and
suggested it might be cleaner than reading at the end of the question, which is muddied by
phrasing.

**On this data it was not merely worse. It lost at every single one of the 33 stages.**

The best entity-moment score (0.781) is below what the end-of-question moment achieves by its
sixth stage. The honest selection procedure picked the end-of-question moment in all five
groups without exception.

One small positive: at stage 0, the entity moment scores 0.545 against the end-of-question
0.484. Before any processing, *which words the entity is spelled with* carries a little
information — obscure entities have unusual spellings. That is a real if minor effect, and it
is the only place the entity moment leads.

This matters beyond bookkeeping, because the end-of-question moment is the one a real decoder
would have to use anyway — it is the point where the decision must be made. The alternative
that might have been cleaner isn't.

---

## 8. Things that could be wrong, stated plainly

**The strongest constraint was always chosen.** The probe could constrain itself to one of
four degrees, and in all five groups the most constrained option won. When the best setting
is at the edge of the range you offered, the real best setting may lie beyond it. We are
running a follow-up with a wider range — see section 9. It is declared as a follow-up rather
than folded into the headline, because it was chosen *after* seeing a failing result, and
quietly widening a search until the answer improves is how this kind of study talks itself
into a false positive.

**Seven question types could not be tested.** Scoring requires examples of both knowing and
not knowing. `currency` (100% known), `continent` (96%), and `capital` (95%) have almost no
counterexamples. So the probe was never tested where the model knows nearly everything, which
is 255 questions, 16% of the total.

**One question was dropped.** Locating the entity in the question text worked for 1,792 of
1,793. The failure was "John Spencer, 8th Earl Spencer" against a question phrased "John
Spencer, the 8th Earl Spencer". It is kept for the end-of-question reading and excluded only
from the entity reading — never quietly substituted, which would have corrupted exactly the
comparison in section 7.

**The knowledge labels come from the model's own output.** A fact counts as known if the
model produces the right answer in at least 8 of 10 attempts. That is a reasonable definition
and it is the one Gate 1 used, but it is not the only one, and a different definition could
in principle favour the probe. Testing that is a separate piece of work.

---

## 9. The follow-up analysis

*(To be completed when the wider-range run finishes. The pre-registered result above is the
headline regardless of what it shows; if the two disagree, the pre-registered one governs and
the disagreement is reported here.)*

---

## 10. What this does and does not license us to say

This needs stating precisely, because the tempting summary is broader than the evidence.

**What we can say.** On one model (Llama-3.1-8B-Instruct), on questions about factual
relationships between entities, with a fifth of the questions being one relationship type
(`spouse`), using a probe that draws a single dividing line, read at the moment a decoding
system would actually need it — **the probe does not beat a free baseline by enough to
justify building on it.** And the free baseline is genuinely strong: 0.860 out of a possible
1.0, at no cost.

**What we cannot say.** That no probe recovers this. We did not test non-linear probes,
other models, other kinds of facts, other definitions of "knows", or reading positions
besides the two. A more complex probe might do better — though it would then be too expensive
for the decoder this was meant to feed, which is the whole point of testing the simple one.

The honest one-line version: *a linear probe, at the position a decoding gate requires, does
not here beat a free baseline.* Not: *internal states don't encode knowledge.*

**One thing this does settle**, though, is the direction of the original project. The plan
was a chain of four gates, each depending on the one before. Gate 1 passed and established
the prize was real. Gate 2 says the next link does not hold. Continuing to Gate 3 — feeding
probe estimates into a decoder — would be building on a predictor we have just shown adds
0.009 to a free signal. The pre-registration said to stop here, and stopping here is right.

That is a result, not a failure. The negative is informative: it says the useful part of
"does the model know this?" is *already in the output distribution*, and anyone designing a
knowledge-aware decoder should reach for hesitancy first and justify anything more expensive
against it.

---

## 11. Where everything lives

| what | file |
|---|---|
| The plan, fixed before we looked at the data | `GATE2_PROTOCOL.md` |
| Extraction, training, the four diagnostics | `gate2.py` |
| Reading the model's internal state | `backend.py`, `hidden_states` |
| Proof the pipeline works, on fabricated data | `selftest2.py` |
| Every number in this document | `gate2_results.json`, `probe_folds.json` |
| Proof the reading matched Gate 1's setup exactly | `extract_parity.json` |
| The raw internal states, ~1 GB | `acts_shard000-007.npz` |
| Gate 1's result | `FINDINGS.md` |

One detail worth recording. The setup demands that Gate 2 read the model in exactly the
conditions Gate 1 did, or the comparison is meaningless. We checked by recomputing the
hesitancy figures during extraction and comparing them against what Gate 1 recorded weeks
earlier. They matched **exactly** — to the last digit, on all 1,793 questions. Independent
computations agreeing to that precision is strong evidence the conditions were identical.

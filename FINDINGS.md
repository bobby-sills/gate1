# Gate 1 — findings

Status as of 2026-08-04. The primary run is complete. A secondary run is still
decoding; the section that needs it is marked unfinished.

---

## 1. The question

When a language model is given a document and asked a question, and the document
contradicts what the model already believes, the model has to pick a side. Sometimes the
document is right and the model's memory is wrong. Sometimes the reverse.

Picking correctly requires knowing which of the two sources to trust *for this particular
fact*. The obvious cheap signal is the model's own confidence: if it answers hesitantly,
maybe it doesn't know and should defer to the document. That signal is free — it falls
out of the model's output probabilities at no extra cost.

**This experiment asks whether actually knowing the fact tells you more than that free
confidence signal does.** If it barely does, then there is no point building anything
fancier, because the free signal already captures whatever is useful. If it tells you a
lot more, there is room for a method that estimates knowledge and uses it.

To measure the ceiling, we let one method *cheat*: it looks up whether the model really
knows each fact and uses that. That method is called the **oracle**. It could never be
deployed — the answer key is not available at run time — but it establishes the best any
knowledge-based method could possibly do. If even the cheating method barely beats the
free signal, the research direction is dead, and finding that out cheaply is the point.

**The oracle winning is not the goal.** Nothing in this experiment was tuned to make it
win, and several choices deliberately handicap it (see §8).

---

## 2. The short answer

The cheating method beats the free confidence signal by a wide margin, and it does so
*for the reason we predicted* rather than by accident.

The program that computes the decision printed:

```
PROCEED to Gate 2. Headroom budget = 13.83 points macro.
```

That number is the gap between the cheating method and the best free-confidence method,
in percentage points. The pre-registered bar for "keep going" was 4 points; the bar for
"stop, this is a dead end" was 2. It cleared the higher bar three times over.

Two things qualify it, both quantified in §7: about a third of the advantage in one of
the two situations we care about comes from a lenient scoring rule rather than from
better answers, and the whole result covers a narrower slice of facts than originally
planned.

---

## 3. How a question becomes a measurement

**The knob.** The decoder blends two predictions at every word: what the model would say
from memory alone, and what it would say having read the document. One number controls
the blend. At the low end the model ignores the document; at the high end it follows the
document past the point of sense. Every method in this experiment is the *same* decoder —
they differ only in how they choose that number.

**The four situations.** Each question is labelled by whether the model knows the answer,
and then shown either a document that is correct or one that has been altered to give a
false answer. That gives four combinations:

| | correct document | altered document |
|---|---|---|
| **model knows the fact** | *agreement* — both sources right, easy | **resistance** — the document is wrong and must be ignored |
| **model doesn't know** | **correction** — the document is right and must be followed | *both wrong* — neither source helps |

The two bold cells are the whole experiment. They pull in opposite directions: doing well
at one by always trusting the document means doing badly at the other. The headline score
is the average of the two, deliberately, so that a method which simply always believes
the document cannot look good. *Agreement* is easy for everyone and *both wrong* measures
noise; neither is in the headline.

**The methods compared.** All five use the same decoder and the same amount of tuning.

- **fixed** — one blend setting for every question, no adaptation at all.
- **confidence** — adapts based on how uncertain the model's answer looks. This is the
  baseline that matters: it is free, and if it is good enough, nothing else is needed.
- **top-answer-probability** — adapts based on how probable the single best answer is. A
  second free signal.
- **oracle, one-sided** — cheats, but only adjusts in one direction.
- **oracle, two-sided** — cheats, and adjusts in both directions.

**Why the label is measured separately.** Whether the model "knows" a fact is decided by
asking it the question ten times with no document and counting correct answers: 8 or more
means it knows, 2 or fewer means it doesn't, and anything in between is discarded as
ambiguous. Critically, this measurement is a **separate** computation from the confidence
signal the baseline uses. Sharing the work between them would have been cheaper but would
have quietly handicapped the baseline and inflated the result.

---

## 4. The source data needed repair first

The questions come from a public dataset (ConFiQA) built by taking facts from Wikidata
and generating both a truthful passage and one with the answer swapped. It is not clean.
Roughly half the effort here went into finding out how.

**Answer lists that matched anything.** Scoring compares the model's output to a list of
acceptable answers, checking whether any of them appears in the output. Some entries were
emoji flags or currency symbols, which reduce to an empty string once punctuation is
stripped — and an empty string appears inside *every* output. A single such entry made
every answer count as correct, which would have marked the question as "known" regardless
of the model and given every method a perfect score on it. Others were two-letter codes
like `US` or `or` that appear inside ordinary English words. **This affected 10.3% of the
data.** Fixed by dropping list entries shorter than three characters after normalising,
while never dropping the primary answer.

**Questions containing their own answers.** 156 questions had the answer sitting in the
question text, so no knowledge was needed:

```
What country is Panama?              answer: Panama
Where is the headquarters of FC Bayern Munich?   answer: Munich
```

**One relation is simply broken.** For `follows`, 54% of questions name the gold answer in
the question, because the template renders the relationship in both directions
inconsistently:

```
What novel follows Northanger Abbey?   answer: Northanger Abbey
```

**Questions with many true answers.** The largest problem. Roughly 45% of the data uses
relationships where the subject genuinely has several correct answers — awards received,
genres, group memberships, fields of work — and the dataset arbitrarily picked one. Asking
"does the model know this fact" is not meaningful there: the model can name a different,
equally true award and be marked wrong.

This is not a guess. Measured across the full 4207 questions:

| | knows | doesn't know | ambiguous |
|---|---|---|---|
| one-true-answer relations | 54% | 34% | 12% |
| many-true-answer relations | **26%** | **63%** | 10% |

Twice the "doesn't know" rate, and not explained by one common answer dominating (that
share is 24% in both groups). The exclusion is therefore an evidenced decision, not an
assumption — and the same reasoning the original protocol already used to exclude
multi-step questions.

**What survived.** 6000 questions → 4207 after the original filters → **1793** after
removing many-answer relations, time-dependent relations (a country's leader changes, so
a model that knows a different era's holder is marked wrong — these are set aside for
possible future work rather than discarded), four structurally broken relations, and
self-answering questions.

---

## 5. A confound we found and removed

Different relationships are different in difficulty. Naming someone's spouse is hard;
naming a country's currency is easy. So the "knows" group and the "doesn't know" group
were not made of the same kinds of question — and those two groups feed the two headline
situations directly.

Measured, the two groups' relationship mixes differed substantially: `composer` was 6% of
one and 21% of the other; `country of citizenship` 15% and 5%. A difference that size
could produce an apparent oracle advantage that is really just a difficulty difference
wearing a disguise.

We checked the obvious fix first — keeping only relationships with large answer spaces,
on the theory that easily-guessed answers were the problem. **It didn't work**: the
mismatch barely moved and it cost 44% of the remaining data. It also turned out the
underlying guess was wrong; how prominent an entity is matters more than how many possible
answers there are. `currency` has a wide answer space and the model gets 100% of them.

So instead the two groups were **matched by construction**: for each relationship, take
the same number of "knows" and "doesn't know" questions. The realised mismatch after
matching is exactly zero, against 0.457 unmatched (0 means identical, 1 means completely
different; random sampling noise alone produces about 0.07 at these sizes, which is why
"zero" here means matching worked rather than that nothing was wrong).

The unmatched version was kept and is being run separately as a sensitivity check. If the
two disagree, that disagreement is a result in itself and gets reported, not resolved by
picking the nicer one.

---

## 6. The headline numbers

600 questions, 300 in each of the four situations, 1200 total measurements.

```
                fixed   oracle-1   oracle-2   confidence   top-prob
correction       79.3       88.0       88.0         73.0       72.3
resistance       76.7       85.3       99.7         87.0       87.7
agreement        99.7      100.0       99.7         99.7       99.7
both wrong        7.3        5.3        5.3          7.3        7.3

oracle - confidence  = +13.83   (95% range +11.01 to +16.80)
oracle - fixed       = +15.83   (95% range +12.94 to +18.86)
```

The ranges come from resampling **by question, not by row**. Each question produces two
measurements that are correlated with each other, and treating them as independent would
have made the ranges look about 30% tighter than they should be.

Three things worth noticing:

**The free confidence signal is not weak — it is lopsided.** It beats the fixed method
comfortably on resistance (87.0 against 76.7) but loses on correction (73.0 against 79.3).
One threshold cannot serve both situations. Most of the 13.83 gap is this.

**The two cheating methods are identical on correction and differ only on resistance.**
They differ solely in an adjustment applied to questions the model knows, and the
correction situation contains none of those. The entire two-sided advantage is the +14.4
it buys on resistance.

**Resistance for the two-sided oracle is 99.7.** That is a method which, knowing the model
has the fact, turns the document's influence down and falls back on memory — exactly when
the document is the thing that's wrong. Near-perfect is what a cheating method *should*
score there. It is the ceiling being measured, not an anomaly.

---

## 7. Does the explanation hold?

A result can be real and the story about it still be wrong. The proposed story is specific:
knowing the fact helps **precisely where the free confidence signal is mistaken** — where
the model sounds confident but is wrong, or sounds hesitant but is right. If the advantage
were spread evenly instead, something else would be producing it.

Splitting both headline situations by whether the free signal and the truth agree:

```
situation    group                        n   oracle   confidence     gap
correction   sounds confident, wrong     82     79.3         37.8   +41.5   <- disagree
correction   sounds unsure, wrong       218     91.3         86.2    +5.0
resistance   sounds confident, right    218    100.0         98.6    +1.4
resistance   sounds unsure, right        82     98.8         56.1   +42.7   <- disagree

where the two disagree   n=164   gap = +42.1
where the two agree      n=436   gap =  +3.2
```

**Thirteen times the advantage, concentrated exactly where predicted.** Where the free
signal is already right, knowing the fact buys almost nothing. Where it is wrong, the free
signal collapses to 37.8% and 56.1% and the cheating method sails past it. The explanation
survives its own test.

### The scoring rule flatters the oracle, by a measurable amount

Scoring accepts an answer if an acceptable answer appears *anywhere* in the output. At
high blend settings the model rambles instead of answering:

```
'Christianity does inform  another Document:\nHowever Christianity contr'
```

A rambling answer that happens to contain the right words still scores correct — and only
the adaptive methods push the blend high enough to ramble. Re-scoring the correction
situation under strict equality instead, with everything else held fixed:

```
                    lenient   strict    drop
fixed                  79.3     75.0    -4.3
confidence             73.0     68.3    -4.7
top-probability        72.3     68.0    -4.3
oracle (both)          88.0     78.3    -9.7

oracle - confidence on correction:  lenient +15.0    strict +10.0
```

**The oracle loses roughly twice as much as anything else**, and its advantage on
correction falls by a third. This affects correction only — the resistance situation uses
low blend settings where no rambling occurs. Carrying the measured shift through, the
headline would land near 11.4 rather than 13.83, still far above the 4-point bar. That
last figure is arithmetic on two averages, not a full re-run of the resampling, and is
labelled as such.

---

## 8. Where we departed from the plan, and why

The design was fixed in advance so that choices could not be made after seeing results.
Four departures, all recorded:

**Three filters were added after the fact** (removing answer-leaking passages, many-answer
relations, and self-answering questions). Each is flagged as after-the-fact in the
attrition record and in the protocol. The third and largest is justified by measurement
(§4), not by preference.

**The cell-matching in §5 was added after the labels were computed**, and both the matched
and unmatched versions are pre-registered as primary and secondary so neither can be
chosen after seeing which looks better.

**The ambiguous-question rate is 11.8%, against an expected 20–30%.** Two readings are
equally consistent with it and cannot be separated from these measurements alone. Either
the model's knowledge is more all-or-nothing than expected, or the combination of an
8-of-10 threshold with lenient substring scoring pushes near-misses — right entity, wrong
wording — down to 0 out of 10 rather than into the middle. Both go in the writeup.

**Choices that handicap the oracle, kept deliberately.** The free-signal methods have
their thresholds tuned on the same data they are evaluated on, which is normally a
mistake; here it is on purpose, so the baselines get their best possible showing. Every
method gets the same tuning budget over its own settings, so the adaptive ones cannot win
by being tried more ways.

---

## 9. Limitations

**The slice is narrow.** 1793 questions covering one-true-answer facts about people and
works. Nothing here generalises to relationships with several correct answers — those were
removed on evidence, but removing them is still a restriction on what the result covers.

**One relationship dominates.** `spouse` is 105 of the 300 questions in every situation —
35% — because it is the one large relationship that splits evenly between known and
unknown, and the matching in §5 weights by the smaller side. So the result leans on a
single question template more than is comfortable.

**One model, one dataset.** Llama-3.1-8B-Instruct only. The pre-registered contingency
for a second model exists but was not triggered, since the result was not in the
ambiguous band.

**The oracle cannot be built.** It reads the answer key. The measured 13.83 is the budget
available to a future method that *estimates* knowledge — and any real estimator will
capture only part of it.

---

## 10. Cost

| step | where | cost |
|---|---|---|
| knowledge labels, 4207 questions | one A-series GPU | 84 min |
| generation, 16,382 units | A100 | 2.32 hours |
| everything else | ordinary CPU | minutes |

Generation was cut from a naive 1,065,600 runs to 16,382 by noticing that thousands of
method settings collapse onto only 15 distinct blend values — a 65-fold saving. A first
attempt on a slower GPU projected 4.55 hours against a 3-hour budget; per the plan this
was reported rather than solved by shrinking anything, and the fix was a faster GPU. It
is worth recording that the three remedies the plan suggested would each have saved under
20 minutes, because cost is set by the number of distinct blend values, not by the number
of method settings.

---

## 11. Not yet done

The secondary unmatched run is still generating. When it finishes, this document needs:

- the same table as §6 for the unmatched version, and
- the difference between the two, which measures how much confounding the matching removed

Gate 2 has not been started and should not be until the above is read.

---

## 12. Where everything lives

| file | what it is |
|---|---|
| `pool.jsonl` | the 1793 final questions |
| `pool_4207.jsonl` | the pool before the last two filters, kept for comparison |
| `attrition.json` | how many questions each filter removed |
| `labels.jsonl` | knows / doesn't know / ambiguous, for all 4207 |
| `sanity.json` | balance checks and the relationship-mix figures |
| `cells.jsonl` | the 1200 matched measurements, with a provenance header |
| `cells_unmatched.jsonl` | the secondary version |
| `generations.jsonl` | every generated answer, reused across both runs |
| `sweep.csv` | every method setting scored — 1,065,600 rows |
| `prompt_parity.txt` | the human-checked proof that the two document conditions differ only by the passage |

Code: `backend.py` (data loading and model calls), `gate1.py` (the pipeline),
`relstats.py` (relationship audit), `stratify.py` (§7), `instrument.py` and `pairs.py`
(measurement side-quests). `PROTOCOL.md` holds the design and its amendments.

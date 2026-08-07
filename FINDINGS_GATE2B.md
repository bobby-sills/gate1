# Gate 2b findings: the probe reads answers, not knowledge

**Verdict: STOP.** The pre-registered gate required the probe to beat the best free
baseline by 0.03 AUC. It reached **+0.0296** — short by four ten-thousandths. But the
margin is not the finding. The diagnostics that followed showed the advantage was
**answer-entity memorisation**, and it survives no test of generalisation.

Gate 2's null stands unrevised. This document reports beside it.

---

## 1. What Gate 2b asked

Gate 2 read the residual stream at the **final question token** and found a linear probe
adds +0.0095 AUC over `H(p_theta)`, with a CI spanning zero. The obvious objection: the
truthfulness-probing literature (Orgad et al.) locates its signal at the **exact answer
tokens**, which do not exist at that position.

Gate 2b generates the closed-book answer first, then probes there — and **moves the
baselines there too**, because a repositioned probe against an unrepositioned baseline
manufactures its own result.

## 2. Headline

```
probe                    0.8894        bar: 0.8598 + 0.03 = 0.8898
b1_entropy_qfinal        0.8598   <-- BEST baseline
b4_self_consistency      0.8176
b3_mean_logprob          0.7999
b2_entropy_mean          0.7831
b2_entropy_first         0.7418
b2_entropy_last          0.7344

probe - b1 = +0.0296  [+0.0120, +0.0473]   DeLong z=3.39  p=0.000692
```

Unlike Gate 2, the interval **excludes zero** — the probe reliably beats the best free
baseline. It misses the *effect-size* threshold, not the significance one.

The declared rank-pooling secondary lands at **+0.0305 [+0.0132, +0.0479]**, on the other
side of the bar. The pre-registered number remains the headline; the disagreement is a
measurement artifact with a known direction, not two estimates to weigh evenly.

## 3. Entropy degrades at the answer tokens

Measured at Checkpoint A, before any probe was trained:

| position | AUC |
|---|---|
| question-final token (b1) | **0.8598** |
| mean over the answer span | 0.7831 |
| first answer token | 0.7418 |
| last answer token | 0.7344 |

The Orgad-based motivation predicts the signal concentrates at the exact answer tokens. On
this task and this label it does the opposite — the free signal is **strongest where Gate 2
already read it** and loses 8–13 points when moved. That is half of Gate 2b's question
answered before training, and it holds independently of anything the probe did.

## 4. The result that matters: the advantage does not generalise

Four independent tests, each closing one route:

```
pooled                            +0.0296
answer unseen in training         +0.0088
answer unseen, within relation    -0.0019
relation held out entirely        -0.0082      (C2, 8/8 relations)
```

### 4a. Answer-entity leakage

Gate 2b grouped folds by `subject_qid` only. Inside-Out excludes **subject and object**
overlap, and that difference turns out to be load-bearing.

```
answer cardinality        continent 0.09   sport 0.20   religion 0.30
                          citizenship 0.31   native language 0.31
                          composer 0.77   director 0.92   spouse 1.00

45.4% of test rows had their gold answer also present in training
AUC of answer identity alone, NO activations:   0.7029
```

Split the evaluation by it:

```
                        n     probe   entropy     diff
answer SEEN in train   718   0.9222   0.8784   +0.0438
answer UNSEEN          864   0.8540   0.8452   +0.0088
all                   1582   0.8894   0.8598   +0.0296
```

Controlling for relation (low-cardinality relations dominate the SEEN group):

```
n-weighted, within relation:   seen  +0.0522      unseen  -0.0019
```

The cleanest single case is `country of citizenship`, where both strata are well populated
inside one relation: **+0.1520 on seen answers, −0.0729 on unseen.** A 0.22 swing with
relation, question template and known-rate all held fixed.

Reproduce with `object_leak.py` and `object_leak_within_relation.py`.

### 4b. C2 — leave-one-relation-out

```
sport                    0.9201      composer          0.8797
director                 0.8751      spouse            0.8168
performer                0.7767      native language   0.6708
religion                 0.6379      country of cit.   0.6272
n-weighted mean AUC 0.7912        vs entropy  -0.0082
```

On relations it never trained on the probe is **behind** entropy. And the two relations
where it most beat entropy in-sample are the two that collapse: `country of citizenship`
(+0.161 in-sample) and `performer` (+0.123) lose 0.154 and 0.117 when held out.

## 5. C1 passed — and that is not a rescue

Within-relation the advantage is **+0.0357 [+0.0050, +0.0613]**, *larger* than pooled.
Gate 2's advantage evaporated under this exact test (+0.0095 → −0.0023); Gate 2b's grows.

So the probe is **not** a relation classifier. It is an *answer-entity* classifier — finer
grained than relation, which is why it passes a within-relation test while failing every
out-of-sample one. C1 and the leakage analysis are consistent, not contradictory.

C3 rules out `spouse`: +0.0372 excluded vs +0.0357 included.

## 6. C0 could not be evaluated, and why

```
                    knows    doesn't know
greedy correct       969  (A)     15  (C)
greedy incorrect       3  (B)    595  (D)
```

**Greedy correctness separates the knowledge label at ≈0.986 AUC.** The incorrect-generation
stratum — "where a knowledge signal would have to do real work" — holds **three** known rows.

So we cannot distinguish a knowledge detector from an answer-correctness detector on this
data. This was recorded in `GATE2B_PROTOCOL.md` **before** the probe was trained, together
with the ruling that a passing number could not be claimed as knowledge detection.

Inside-Out §A.4 describes this exact failure in advance, and names Orgad et al. — the paper
Gate 2b is built on — as an instance of the labelling approach that causes it.

## 7. C4 — layer curves

```
mean           0.9004  @ L15
last           0.8968  @ L19
gate2_final    0.8827  @ L28
first          0.8697  @ L16
```

The answer-span read genuinely beats Gate 2's position (+0.0177 at each one's best layer).
Structurally: the **answer-token signal peaks mid-network (L15–19) while the question-final
signal peaks late (L28)**. That matches Inside-Out A.5 ("stabilise from around layers 11–12
of 32") and explains the unstable per-fold layer selection — it is a plateau, not a bug.

## 8. Two methodological findings, independent of the result

**The numeric parity check only verifies the prompt within one GPU architecture.** A first
Phase A on an A100 failed two registered STOP conditions with a byte-identical prompt:
`deterministic_features` vs `labels.jsonl` gave median 1.272e-2 with 0/200 exact, against
1793/1793 exact on the L4. Sampling amplifies bit-level logit differences, so b4's
guarantee — the identical ten label-producing strings — does not survive a change of GPU
(1283/1793 on the A100, 1793/1793 on the L4). The **string-level check** is what carries the
prompt-identity claim. Recorded in `GATE2B_PROTOCOL.md` *Hardware*.

**Pooled out-of-fold AUC concatenates raw scores from separately fitted probes**, which is
only valid while within-fold spread exceeds between-fold offsets. Under strong
regularisation those invert: on the selftest fixture every fold scores 1.0000 and the pooled
vector scores 0.5024. Gate 2's published result was rechecked and is clean (offset ratio
0.044; headline moves +0.0009 under corrected pooling). See `pooling_recheck.py`.

## 9. What this means

Gate 1's ceiling is real: perfect knowledge labels buy **+13.83 macro faithfulness points**.
Two independent read positions now say it is not cheaply recoverable — Gate 2 by a clean
null, Gate 2b by an advantage that exists only where the probe has memorised the answer.

The deeper fault is the label. Defining "knows this fact" as "generates it in 8 of 10
samples" makes the target a function of the output distribution, which entropy already
reads — and files every hidden-knowledge case under `doesn't know`, which is the population
the hypothesis was about. The 0.986 greedy-correctness correlation is that same fault
surfacing as a measurement failure.

**Scope.** One model (Llama-3.1-8B-Instruct), ConFiQA functional entity facts, `spouse`-heavy,
linear probes, answer-token and question-final read positions. Not *"no probe recovers
this"* — *"a linear probe on a generation-defined label does not, here, beat a free
baseline in any way that generalises."*

# Gate 3b findings: the probe's AUC does not convert into faithfulness — and neither does entropy's

**Verdict: null, and a bigger one than expected.** Gate 1 established that a *perfect*
knowledge label buys **+13.83** macro faithfulness points over the entropy-gated decoder.
Gate 3b fed the real Gate 2b probe into that same gate. It recovers **7.2% of the headroom
when tuned on the evaluation data, and essentially none out of sample.**

The unplanned finding is larger: **entropy's adaptive gating does not reliably beat a fixed
tau either.** Once operating points are selected off-sample, the gap between "gate on
confidence" and "do not gate at all" is inside noise, and on one of the two samples it is
negative.

Run as **engineering, not as a claim** — no pre-registered threshold, search permitted,
split-half as the honest metric. Gate 2b's STOP is not revised by anything here.

---

## 1. Headline

Every arm tuned on the evaluation data (gate1 invariant 6, applied symmetrically so the
comparison stays paired):

```
constant            78.00
entropy             80.00
max_prob            80.00
probe               81.00
oracle_two_sided    93.83

probe - entropy            +1.00  [-1.49, +3.54]
oracle_two_sided - entropy +13.83 [+11.01, +16.80]   <- Gate 1, reproduced exactly here
HEADROOM RECOVERED           7.2%
```

Split-half — select the operating point on 300 questions, measure on the other 300:

```
entropy   78.33
probe     78.50
probe - entropy  +0.17  [-2.38, +2.70]
```

**+0.17 points.** That is what Gate 2b's +0.0296 AUC is worth in the currency the project
actually cares about.

## 2. Why the pooled number hides the mechanism

The probe is not uniformly weak. It is strong in the right place and pays for it elsewhere.

```
                    on-diagonal   off-diagonal
entropy                   92.43         46.95
probe                     87.84         62.80
oracle_two_sided          95.64         89.02
```

Off-diagonal — the 164 questions where entropy's gate disagrees with the ground-truth
knowledge label, where Gate 1's entire headroom lives — the probe gains **+15.85**. On the
436 where entropy was already right it loses **−4.59**. Net +1.00.

The gating decisions show why:

```
entropy_says   agrees with the oracle 72.7%   TP 218  FP  82  FN  82  TN 218
probe_says     agrees with the oracle 69.3%   TP 283  FP 167  FN  17  TN 133
```

The probe's selected threshold is the 25th percentile, so it calls ~75% of questions
"known": it almost never misses a known fact (FN 17) and doubles the false positives. It is
a **worse** knowledge detector than entropy overall (69.3% vs 72.7%) while being better
exactly where entropy fails. On the 192 questions where the two gates disagree, entropy
matches the truth 106 times and the probe 86.

## 3. Routing between the two arms is capped, and the cap is low

The obvious follow-up — use the probe where it is better, entropy elsewhere — was measured
before being built. `router_ceiling.py`:

```
same outcome        93.5%   (1122 of 1200 instances)
entropy right only     36
probe right only       42
both wrong            357

PERFECT per-instance router   85.67   (+5.67 over entropy)
```

A router *told the answer* reaches +5.67 against the oracle's +13.83. The two arms produce
**identical output correctness on 93.5% of instances**, the routable set is 78 instances
split 42/36, and 30% of the data is wrong under both arms regardless of choice.

Note also that "route where entropy fails" is not implementable: *where entropy fails* is
defined by the ground-truth label, so a router that knows it already has the thing being
estimated. The +15.85 in §2 is visible only because the split uses the oracle.

## 4. The conflict arm — explored, does not survive replication

The one non-circular version of the routing idea. Conflict between the two gates is
observable without labels, and it carries real information:

```
                n     entropy    probe
AGREE          816      84.06    84.86
CONFLICT       384      68.18    72.94
```

When the gates disagree **both arms lose ~15 points** — conflict predicts *difficulty*, not
which arm to trust (on conflicts, entropy is right 55.2% and the probe 44.8%). So: a
three-state rule, agree-known → `tau0 − delta`, agree-unknown → `tau0 + gamma`, disagree →
`tau0` (hedge). Not capped by §3's ceiling, which bounds only selections among the two
arms' existing outputs. Budget held at 360 configs by pairing the two thresholds at the
same quantile level rather than crossing them.

On the matched sample it looked good — and the reason it looked good did not replicate.

```
held-out                  matched              unmatched
conflict - entropy    +1.833  P(<=0)=0.026   +1.500  P(<=0)=0.033
conflict - constant   +2.167  P(<=0)=0.040   +0.167  P(<=0)=0.398
```

**On the unmatched sample the conflict arm ties a fixed tau.** An adaptive gate that cannot
beat not gating is not evidence that the conflict signal works. The stability signature
that motivated it also vanished (tuned → held-out drop: 0.33 matched, 2.17 unmatched).

**Declared:** this arm was proposed *after* seeing the probe result. Split-half controls
operating-point selection, not arm-design selection. And the two samples share **241 of 600
questions (40%)**, so this is a weak replication in both directions. Reported as a closed
direction, not a finding.

## 5. The finding that outlives the probe: adaptivity barely beats a constant

```
held-out            matched   unmatched
constant              78.00       81.67
entropy               78.33       80.33
max_prob              78.50       80.67
probe                 78.50           --
conflict              80.17       81.83
```

**Entropy beats a fixed tau by +0.33 on one sample and loses by −1.34 on the other.**

The cause is visible in how much each arm's tuned number is selection:

```
tuned -> held-out drop     matched   unmatched
  constant                    0.00        0.17
  entropy                     1.67        2.50
  max_prob                    1.50        2.50
  probe                       2.50          --
```

`constant` has one parameter and nothing to overfit. Every threshold arm loses 1.5–2.5
points when its operating point is chosen off-sample. gate1 invariant 6 tunes thresholds on
the evaluation data deliberately — it was the right call for Gate 1, whose question was
about the *oracle's* ceiling, and it handicaps the hypothesis rather than flattering it.
But it flatters every adaptive arm's absolute number, and Gate 1 had no reason to notice.

**This reframes the null.** The probe does not merely fail to beat entropy. Entropy barely
beats doing nothing. Gate 1's +13.83 is real, but it is headroom over a baseline that is
itself close to a constant — so the operative question was never "can a probe beat entropy"
but "can anything cheap beat a fixed tau", and on this data nothing tested does.

**Scope.** The probe arm was not run on the unmatched sample; it needs 366 more decode
units. The unmatched column above is therefore constant/entropy/max_prob/conflict only.

## 6. What the probe actually reads

For anyone reproducing this: the scores in `probe2b_oof.npz` come from a linear probe on a
**single layer's residual stream**, 4096-d, read over the **answer span only** of a
teacher-forced pass over `[prompt, greedy answer]`. Three positions were extracted —
`first`, `last`, and `mean` over `a_1..a_m` — with position, layer and C selected per fold
on an inner split:

| fold | position | layer | C |
|---|---|---|---|
| 0 | mean | 31 | 0.001 |
| 1 | mean | 15 | 0.001 |
| 2 | mean | 15 | 0.001 |
| 3 | last | 28 | 0.001 |
| 4 | mean | 14 | 0.001 |

Mean pooling in 4 of 5 folds. The layer stack is 33 entries for 32 blocks: index 0 is the
embedding output (kept as a null control), index *i* is the residual stream after block
*i−1*. The scattered layer picks are a flat mid-network ridge, not instability — C4
measured mean peaking 0.9004 @ L15 while the question-final read peaks 0.8827 @ L28.

## 7. Two methodological findings, independent of the result

**The (qid, tau) pair, not the tau value, is the cache key.** `GATE3B_PLAN.md` originally
argued Gate 3b needed no GPU: `tau_for` emits `{tau0 − delta, tau0 + gamma}` for every
threshold arm regardless of signal, so the probe arm's work looked like a subset of
entropy's. The set of tau *values* is a subset; the set of *(qid, tau)* pairs is not. A
question whose entropy is below all three thresholds takes the `−delta` branch for every
entropy config, so its `+gamma` generations were never produced.

```
518 of 15600 units missing, on 74 of 600 questions
  entropy below all 3 thresholds: 37     above all 3: 37
  knowledge: known 37, unknown 37
```

Those 74 are precisely the questions where the probe disagrees with *both* entropy and the
knowledge label — the only population where a probe can contribute anything Gate 1 did not
already have. Dropping them would have deleted the experiment and left a clean-looking
number. `check` caught this before any probe number existed.

**Required hardware depends on which Gate 1 artifact you must match bitwise.** Gate 1's
phases ran in separate Colab sessions on different GPUs. `labels` reproduces on an **L4**
(1793/1793, versus 0/200 on an A100 — GATE2B_PROTOCOL *Hardware*); `decode` ran on an
**A100**. So Gate 3a, which compares against `labels.jsonl`, wants an L4, and Gate 3b,
whose generations sit beside `generations.jsonl`, wants an A100. Opposite requirements,
both correct.

This was verified rather than trusted: `gate3b.py parity` re-decoded 50 already-cached
units across all 15 tau values and got **50/50 byte-exact** before any new unit was
written. A single mismatch would have meant the new generations were not comparable to the
cache — and the difference would have landed entirely on the 74 questions carrying the
result.

Scoring is tied to Gate 1's published numbers by re-deriving the `entropy` arm from
`cells.jsonl` + `generations.jsonl` and comparing to `sweep.csv`: **432000/432000 rows,
agreement 1.000000.**

## 8. Cost

| step | cost |
|---|---|
| `check`, `score`, `splithalf`, `diag` | CPU, minutes |
| `parity` | 50 units, 1.83 s/unit |
| `decode` | **518 units, 5.6 min on an A100** (0.66 s/unit) |
| conflict arm | **zero new units** — its `tau0` hedge is covered by `constant` |

2.0% of Gate 1's 25,974 decodes.

## 9. What this means

Three read positions and three gating signals now agree. Gate 2 found a clean null at the
question-final token. Gate 2b found an advantage at the answer tokens that existed only
where the probe had memorised the answer entity. Gate 3b converts that advantage into the
project's actual currency and gets **+0.17 points**, against a **+13.83** ceiling that is
itself measured over a baseline barely distinguishable from a constant tau.

The route from "internal states contain knowledge" to "a decoder that uses it" is not
blocked at the probe. It is blocked at the label — defining "knows this fact" as "generates
it in 8 of 10 samples" makes the target a function of the output distribution that entropy
already reads, and files every hidden-knowledge case under *doesn't know*. `FINDINGS_GATE2B.md`
§9 reaches the same place from the AUC side.

**Gate 3a is the only untested version of the hypothesis left standing**: it changes the
unit of analysis to (question, candidate answer), trains only on questions the model
demonstrably knows, and splits folds subject- *and* object-disjoint. It is built and has
never been run. Nothing in this document bears on whether it works — but nothing here
rescues the design it replaces, either.

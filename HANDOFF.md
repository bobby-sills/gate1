# HANDOFF — sequenced tasks

Work through these in order. Stop at every **CHECKPOINT** and wait for the human.
Read `CLAUDE.md` first.

---

## Setup

**On Colab: follow `COLAB.md` cell by cell.** Do not use `export GATE1_OUT=...` in a `!`
cell — each `!` cell is a fresh shell, the variable will not persist, and checkpoints will
silently go to the ephemeral runtime disk instead of Drive. Set it with `os.environ` in a
Python cell.

On a normal machine with a persistent disk:

```bash
export GATE1_OUT=./outputs
nvidia-smi                  # confirm L4 or A100, not T4
python selftest.py          # must pass before you start
```

Either way, confirm the path took effect before running any phase:

```bash
python -c "import os; print('GATE1_OUT =', os.environ.get('GATE1_OUT', 'UNSET -> will use ./outputs'))"
```

`selftest.py` runs the whole pipeline against a fake backend, including a simulated
disconnect and resume. If it fails before you have written any code, something is wrong
with the environment, not the experiment.

---

## Task 1 — `load_pool`

Download ConFiQA (`github.com/byronBBL/Context-DPO`) and FiQA. **QA subset only.**

Implement `backend.load_pool`. Return `(list[RawInstance], attrition_dict)`.

```bash
python gate1.py pool data/ConFiQA-QA.json data/FiQA.json
```

**CHECKPOINT.** Print the attrition chain and three full instances. A human reads them
and confirms the field mapping is right. Do not proceed on a silent pass.

Commit.

---

## Task 2 — `sample_closed_book` + `deterministic_features`

Two functions, one commit. Re-read invariant #2 before writing them.

Instrument first:

```bash
head -100 $GATE1_OUT/pool.jsonl > /tmp/pool100.jsonl   # temporarily swap in
python gate1.py labels
```

**CHECKPOINT.** Report seconds per instance and projected total for the full pool.
Expect roughly 1–2 s/instance on an L4 (10 sampled generations at 32 tokens plus one
greedy pass). If the projection exceeds ~1 hour, report it rather than reducing
`N_SAMPLES` — that constant is pre-registered.

Then run the full pool. It is resumable; if Colab drops, re-run the same command.

```bash
python gate1.py labels
python gate1.py sanity
```

**CHECKPOINT.** `sanity` prints discard rate, known/unknown balance, and off-diagonal
fraction. All three are findings. If off-diagonal is below 0.15 the experiment is
probably already answered — report it and stop.

Commit.

---

## Task 3 — `render_prompts`

```bash
python gate1.py parity
```

**CHECKPOINT — human must read `outputs/prompt_parity.txt`.**

Do not write an automated equality assertion and report "parity OK". Llama-3.1's chat
template injects a date-cutoff line into the system prompt in some `transformers`
versions, and whitespace-insensitive diffs hide exactly that. The two prompts must differ
**only** by the document block.

Commit.

---

## Task 4 — `next_token_distributions` (+ `is_eos`, `detokenize`)

The decoding primitive. Returns normalized **log** probabilities, not raw logits.

Hold KV caches for both passes across calls within one instance's decode. Do not add
caching in `gate1.py` — see invariant #8.

```bash
python gate1.py cells
python gate1.py decode        # prints the dedup saving, then starts
```

`decode` reports how many configs collapse onto how many actual decode units. On a
792-instance selftest this was **888 configs → 15 distinct τ → 11,612 units, a 61×
saving**. At full scale expect roughly 1,200 instances and ~18,000 units.

**CHECKPOINT.** After the first ~100 units, report seconds per unit and the projected
total. Rough arithmetic: ~18,000 units × ~5 generated tokens × 2 forward passes ≈ 180k
passes. If the projection exceeds ~3 GPU-hours, **stop and report** — do not shrink the
grids. The human will decide between two-stage search, dropping the `max_prob` arm, or
renting a bigger GPU.

Then run to completion, resuming as needed.

Commit.

---

## Task 5 — Analysis (no code)

```bash
python gate1.py sweep      # CPU only, scores every config from cache
python gate1.py analyze
```

`sweep` prints mean output length by τ₀. If length collapses at the high end, generation
has gone degenerate there and the sweep is invalid at that end — report it.

`analyze` prints the headroom table and one of four verdicts: PROCEED, STOP (two forms),
SLICE PROBLEM, or CONTINGENCY. **Report the verdict verbatim.** Do not interpret it, do
not soften a STOP, do not re-run with different constants to get a different answer.

---

## Deliverables

Everything under `$GATE1_OUT`:

| File | Phase |
|---|---|
| `attrition.json` | 1 |
| `pool.jsonl` | 1 |
| `labels.jsonl` | 2 |
| `prompt_parity.txt` | 2.7 — human-read |
| `sanity.json` | 2.6 |
| `cells.jsonl` | 3 |
| `generations.jsonl` | 4a |
| `sweep.csv` | 4b |
| console output of `analyze` | 5–7 |

---

## If you get stuck

- A backend signature looks wrong for the inference stack → **ask**, do not change it.
- A grid looks too expensive → **report the measurement**, do not shrink it.
- A metric looks wrong (e.g. resistance target) → **re-read CLAUDE.md invariants**, then ask.
- `selftest.py` fails → the runner is fine and was verified; the break is in `backend.py`.

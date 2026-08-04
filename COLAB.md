# COLAB.md — running this on Colab Pro

Copy these cells in order. **Cell 2 is the one people get wrong** — read the warning.

---

## Cell 1 — check the GPU before anything else

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

| GPU | Verdict |
|---|---|
| A100 (40GB) | ideal |
| L4 (24GB) | fine |
| T4 (16GB) | **will not fit two KV caches in bf16.** Runtime > Disconnect and delete runtime, then reconnect for better hardware. If you keep getting T4, load 8-bit and expect ~2x slower. |

---

## Cell 2 — mount Drive and set the output path

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['GATE1_OUT'] = '/content/drive/MyDrive/gate1/outputs'
os.makedirs(os.environ['GATE1_OUT'], exist_ok=True)
print('checkpoints ->', os.environ['GATE1_OUT'])
```

> **Do not use `!export GATE1_OUT=...`.** Every `!` cell spawns a fresh shell, so the
> variable does not persist. `gate1.py` would silently fall back to `outputs/` on the
> ephemeral runtime disk, and a disconnect would destroy every checkpoint. Set it via
> `os.environ` in a Python cell — that does propagate to `!` cells in the same session.
>
> After **any** runtime restart or reconnect, re-run this cell before resuming.

Verify it took effect:

```python
!echo "GATE1_OUT is: $GATE1_OUT"
```

If that prints an empty value, stop and fix it before running any phase.

---

## Cell 3 — code and dependencies

```python
!git clone https://github.com/<you>/gate1.git /content/gate1   # or upload the files
%cd /content/gate1
!pip -q install transformers accelerate pandas numpy
!python selftest.py 2>&1 | tail -5
```

`selftest.py` must print `SELFTEST PASSED` before you write any backend code. It runs the
whole pipeline against a fake backend, including a simulated disconnect and resume.

---

## Cell 4 — Hugging Face auth

Llama-3.1-8B-Instruct is a gated repo. Accept the license on the model page first, then:

```python
from huggingface_hub import login
login()   # paste a token with 'read' access
```

---

## Cell 5 — data

```python
!mkdir -p /content/gate1/data
!git clone https://github.com/byronBBL/Context-DPO.git /content/context-dpo
!cp /content/context-dpo/ConFiQA/ConFiQA-QA.json /content/gate1/data/
# FiQA (factual counterpart) — check the repo layout; paths have moved between commits
!ls /content/context-dpo/ConFiQA/
```

**QA subset only.** MR and MC are multi-hop and break the premise.

---

## Running phases

All phases are ordinary Python invocations from a `!` cell:

```python
!python gate1.py pool data/ConFiQA-QA.json data/FiQA.json
!python gate1.py labels
!python gate1.py parity
!python gate1.py sanity
!python gate1.py cells
!python gate1.py decode
!python gate1.py sweep
!python gate1.py analyze
```

Run them in **separate cells**, not one block. `labels` and `decode` are the long ones and
you want to be able to re-issue just those.

---

## When Colab disconnects mid-phase

This is expected and costs you one work unit.

1. Reconnect.
2. **Re-run Cell 2** (the mount + `os.environ`). This is the step people forget.
3. `%cd /content/gate1`
4. Re-issue the exact same phase command.

It prints `resuming <file>: N units already complete` and continues. If it does *not*
print that line, `GATE1_OUT` is wrong and it is starting over — kill it and fix Cell 2.

---

## Keeping the session alive

Colab disconnects idle browser tabs. Options, in order of preference:

1. Leave the tab visible and the machine awake. Simplest, works.
2. Split long phases across sessions deliberately — resume is cheap by design.
3. Do not use the JavaScript "auto-clicker" hacks. They violate the Colab terms and can
   get the account restricted.

Given resumability, option 2 is fine: run `labels` in one session, `decode` across two or
three, and the analysis phases are CPU-only and take minutes.

---

## Expected wall-clock (verify, do not trust)

| Phase | GPU | Estimate |
|---|---|---|
| `pool` | no | minutes |
| `labels` | yes | 20–40 min |
| `parity` | yes | seconds |
| `sanity` | no | seconds |
| `cells` | no | seconds |
| `decode` | yes | 1.5–3 hr |
| `sweep` | no | 1–2 min |
| `analyze` | no | 1–2 min |

`decode` prints its dedup ratio and a running ETA. **Check the ETA after ~100 units.** If
it projects beyond ~3 GPU-hours, stop and report rather than shrinking the pre-registered
grids. These figures are arithmetic, not measurements.

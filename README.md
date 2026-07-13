# hemlock-rl

Train LLMs to write [Hemlock](https://github.com/hemlang/hemlock) via execution rewards —
GRPO through [jing](https://github.com/Schneewolf-Labs/jing), trained by
[grimoire](https://github.com/Schneewolf-Labs/grimoire).

Completions are executed in Hemlock's sandbox (`--sandbox`: no FFI, network, process
spawning, file writes, or signals) and scored on a graded tier ladder:

| Reward | Meaning |
|-------:|---------|
| `2.0`  | runs and stdout matches `expected_stdout` (correctness path) |
| `1.0`  | runs and prints something (validity path) |
| `0.5`  | runs clean but silent — or, on the correctness path, wrong output |
| `-0.5` | parse or runtime error |
| `-1.0` | no extractable code, or timeout |

GRPO normalizes advantages within each group of G completions, so the spread matters:
a group that all lands on one tier yields ~zero gradient — critical early in training,
when everything fails.

## Install

```bash
pip install torch  # pick the build for your CUDA stack first
pip install -e .
```

You also need the `hemlock` binary on PATH (or set `HEMLOCK_BIN`).

## Train

```bash
# Tiny smoke run
python -m hemlock_rl.train --model <your-model> \
    --max-examples 8 --num-generations 4 --max-new-tokens 128

# Real run on hemlang/Hemlock-SFT prompts
python -m hemlock_rl.train --model <your-model> --num-generations 8
```

Data comes from the [hemlang datasets](https://huggingface.co/hemlang) —
`hemlang/Hemlock-SFT` by default (columns: `instruction`, `output`, ...). Instructions
become prompts; the SFT `output` column is prose+code, not program stdout, so it is NOT
used as `expected_stdout`. If you supply a dataset with an `expected_stdout` column, those
rows are scored on the correctness path automatically (via jing's metadata passthrough).

## Reward hacking

Validity-only rewards can be farmed by trivial valid programs that ignore the prompt
(`print("hi")` runs and prints). Mitigations, in order of preference:

1. **Prefer the correctness path** — datasets with deterministic `expected_stdout` (or
   test specs) make the reward mean what you want it to mean.
2. **Use validity as a warmup only**, then switch to correctness data.
3. **Keep `beta` (KL to the base policy) high** so the policy can't drift into a
   degenerate print-farmer.

A seed model with some Hemlock ability (e.g. SFT on `hemlang/Hemlock-SFT` via grimoire
first) converges far faster than a model that never emits runnable Hemlock — start there.

## Test

```bash
pip install -e ".[dev]"
ruff check .
pytest   # execution tests auto-skip if `hemlock` is not on PATH
```

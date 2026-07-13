# hemlock-rl

Train LLMs to write [Hemlock](https://github.com/hemlang/hemlock) via execution
rewards, using [grimoire](https://github.com/Schneewolf-Labs/grimoire)'s online
GRPO method.

Grimoire (>= 2.0.0) ships GRPO as an online method: pass a `GRPOMethod` as
`loss_fn` and the trainer calls its `rollout()` each step to generate
completions and score them with your reward function. hemlock-rl is therefore
just three pieces — a reward function, data prep, and a train script. No
trainer, no vendored RL code.

## Install

```bash
pip install torch  # pick your CUDA build first
pip install -e .
```

You also need the `hemlock` binary on your `PATH` (or point `HEMLOCK_BIN` at
it). Completions are executed in Hemlock's sandbox mode (`hemlock --sandbox`),
which disables FFI, network, process spawning, file writes, and signals.

## The reward

`HemlockExecutionReward` extracts code from each completion (first fenced
block, else raw text unless it looks like prose), runs it in the sandbox, and
grades by outcome:

| outcome | reward |
|---|---|
| no extractable code | -1.0 |
| timeout (5s) | -1.0 |
| parse / runtime error | -0.5 |
| exits 0, no output (or wrong output on the correctness path) | 0.5 |
| exits 0, prints output (validity path) | 1.0 |
| exits 0, stdout matches `expected_stdout` (correctness path) | 2.0 |

The tiers stay spread on purpose: GRPO normalizes advantages within each group
of G completions, so a group that all lands on one tier gives ~zero gradient —
graded outcomes matter most early in training, when everything fails.

Executions run in a thread pool so subprocess waits don't stall the GPU
(the rollout runs synchronously in the training loop).

## Train

```bash
python -m hemlock_rl.train --model Qwen/Qwen2.5-Coder-0.5B-Instruct

# tiny smoke run
python -m hemlock_rl.train --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --max-examples 8 --num-generations 4 --max-new-tokens 128
```

Prompts come from [hemlang/Hemlock-SFT](https://huggingface.co/datasets/hemlang/Hemlock-SFT)
by default (its `instruction` column, chat-formatted when the tokenizer has a
chat template; the `output` column is an SFT response, not program stdout, and
is ignored). Any dataset with an instruction column works via `--dataset`.

Notes:

- GRPO's rollout calls `model.generate()`, which needs full weight access —
  use ZeRO-2 or lower (or FSDP), never ZeRO-3. The trainer raises a clear
  error otherwise.
- Seed the model with Hemlock SFT first (e.g. grimoire SFT on
  `hemlang/Hemlock-SFT`) — execution rewards can't teach syntax from scratch
  efficiently.

## Validity vs. correctness

Without an expected output the reward scores **validity** (does it run?), and
a model can farm that with trivial valid programs — keep `--beta` high so the
policy stays near a Hemlock-capable base model.

With a dataset that has an `expected_stdout` column, grimoire's metadata
passthrough (`tokenize_grpo(metadata_fields=["expected_stdout"])`) carries it
to the reward, which scores **correctness**: exits 0 with exactly matching
stripped stdout earns 2.0, running with wrong output drops to 0.5. Rows
without an expected output fall back to graded validity, so mixed datasets
work fine. Only use `expected_stdout` for deterministic programs.

## Tests

```bash
pip install -e ".[dev]"
pytest          # execution tests auto-skip if `hemlock` isn't on PATH
ruff check .
```

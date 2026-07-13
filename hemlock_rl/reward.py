"""Execution-based reward for Hemlock programs.

Graded validity by default; correctness when metadata carries expected_stdout.
GRPO normalizes advantages within each group of G completions, so the tiers
stay spread — a group that all lands on one tier yields ~zero gradient
(critical early in training, when everything fails).
"""

import concurrent.futures
import os
import re
import subprocess
import tempfile

HEMLOCK_BIN = os.environ.get("HEMLOCK_BIN", "hemlock")
# --sandbox disables FFI, network, process spawning, file writes, and signals.
HEMLOCK_SANDBOX_ARGS = ["--sandbox"]
EXEC_TIMEOUT_S = 5.0
MAX_WORKERS = 8

R_NO_CODE = -1.0       # no extractable code (refusal, prose)
R_TIMEOUT = -1.0       # ran past EXEC_TIMEOUT_S
R_ERROR = -0.5         # parse or runtime error
R_RUNS = 0.5           # exits 0 (correctness path: but wrong output)
R_RUNS_OUTPUT = 1.0    # exits 0 and prints something (validity path only)
R_CORRECT = 2.0        # exits 0 and stdout matches expected_stdout

_FENCE = re.compile(r"```(?:hemlock|hml|hm)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text):
    """Pull Hemlock source out of a completion: first fenced block, else the
    raw text unless it looks like prose."""
    m = _FENCE.findall(text)
    if m:
        return m[0].strip()
    s = text.strip()
    if not s or s.splitlines()[0].lower().startswith(("here", "this", "the ", "sure", "to ")):
        return None
    return s


def run_hemlock(code):
    """Execute code in the Hemlock sandbox. Returns (returncode, stdout);
    returncode is None on timeout."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.hml")
        with open(path, "w") as f:
            f.write(code)
        try:
            result = subprocess.run(
                [HEMLOCK_BIN, *HEMLOCK_SANDBOX_ARGS, path],
                capture_output=True, text=True, timeout=EXEC_TIMEOUT_S, cwd=d,
            )
        except subprocess.TimeoutExpired:
            return None, ""
        return result.returncode, result.stdout


def _score(completion, meta=None):
    code = extract_code(completion)
    if code is None:
        return R_NO_CODE
    rc, out = run_hemlock(code)
    if rc is None:
        return R_TIMEOUT
    if rc != 0:
        return R_ERROR
    if meta and meta.get("expected_stdout") is not None:
        # Correctness path — only for deterministic expected output.
        return R_CORRECT if out.strip() == meta["expected_stdout"].strip() else R_RUNS
    # Validity path — reward producing output over silent success.
    return R_RUNS_OUTPUT if out.strip() else R_RUNS


class HemlockExecutionReward:
    """jing reward_fn: (prompts, completions[, metadata]) -> list[float].

    Executions run in a thread pool — the work is subprocess-bound, so
    threads parallelize it fine.
    """

    def __call__(self, prompts, completions, metadata=None):
        metas = metadata if metadata is not None else [None] * len(completions)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            return list(pool.map(lambda a: _score(*a), zip(completions, metas)))

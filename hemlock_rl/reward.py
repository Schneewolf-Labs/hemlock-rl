"""Execution-based reward for Hemlock programs.

Grades completions by outcome: no code < timeout < error < runs < runs with
output. GRPO normalizes advantages within each group of G completions, so the
tiers stay spread — a group that all lands on one tier yields ~zero gradient
(critical early in training, when everything fails).

This scores *validity* (does it run?), not *correctness* (does it do the
right thing?) — see the README for correctness-reward options.
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
R_RUNS = 0.5           # exits 0 but prints nothing
R_RUNS_OUTPUT = 1.0    # exits 0 and prints something

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


def _score(completion):
    code = extract_code(completion)
    if code is None:
        return R_NO_CODE
    rc, out = run_hemlock(code)
    if rc is None:
        return R_TIMEOUT
    if rc != 0:
        return R_ERROR
    return R_RUNS_OUTPUT if out.strip() else R_RUNS


class HemlockExecutionReward:
    """grimoire GRPOMethod reward_fn: (prompts, completions) -> list[float].

    Executions run in a thread pool — the work is subprocess-bound, so
    threads parallelize it fine and the (synchronous) rollout doesn't stall
    the GPU on serial subprocess waits.
    """

    def __call__(self, prompts, completions):
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            return list(pool.map(_score, completions))

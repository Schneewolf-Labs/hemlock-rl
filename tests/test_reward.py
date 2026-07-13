import shutil

import pytest

from hemlock_rl.reward import (
    HEMLOCK_BIN,
    R_CORRECT,
    R_ERROR,
    R_NO_CODE,
    R_RUNS,
    R_RUNS_OUTPUT,
    R_TIMEOUT,
    HemlockExecutionReward,
    extract_code,
    run_hemlock,
)

needs_hemlock = pytest.mark.skipif(
    shutil.which(HEMLOCK_BIN) is None,
    reason=f"{HEMLOCK_BIN} binary not on PATH",
)

GOOD_PROGRAM = 'print("hello from hemlock");'
SILENT_PROGRAM = "let x = 1;"
BAD_PROGRAM = "let x = = broken(("
LOOP_PROGRAM = "while (true) { let x = 1; }"


class TestExtractCode:
    def test_fenced_hemlock_block(self):
        text = f"Here you go:\n```hemlock\n{GOOD_PROGRAM}\n```\nEnjoy!"
        assert extract_code(text) == GOOD_PROGRAM

    def test_bare_fence(self):
        text = f"```\n{GOOD_PROGRAM}\n```"
        assert extract_code(text) == GOOD_PROGRAM

    def test_first_fence_wins(self):
        text = f"```hml\n{GOOD_PROGRAM}\n```\n```\n{BAD_PROGRAM}\n```"
        assert extract_code(text) == GOOD_PROGRAM

    def test_raw_code_passes_through(self):
        assert extract_code(GOOD_PROGRAM) == GOOD_PROGRAM

    def test_prose_returns_none(self):
        assert extract_code("Here is how you would approach this problem...") is None
        assert extract_code("Sure! Let me explain.") is None
        assert extract_code("") is None


@needs_hemlock
class TestRunHemlock:
    def test_good_program_runs(self):
        rc, out = run_hemlock(GOOD_PROGRAM)
        assert rc == 0
        assert out.strip() == "hello from hemlock"

    def test_syntax_error_nonzero(self):
        rc, _ = run_hemlock(BAD_PROGRAM)
        assert rc not in (0, None)

    def test_timeout_returns_none(self):
        rc, _ = run_hemlock(LOOP_PROGRAM)
        assert rc is None


@needs_hemlock
class TestHemlockExecutionReward:
    def test_good_program_positive(self):
        reward = HemlockExecutionReward()
        scores = reward(["prompt"], [f"```hemlock\n{GOOD_PROGRAM}\n```"])
        assert scores == [R_RUNS_OUTPUT]
        assert scores[0] > 0

    def test_syntax_error_negative(self):
        reward = HemlockExecutionReward()
        scores = reward(["prompt"], [f"```hemlock\n{BAD_PROGRAM}\n```"])
        assert scores == [R_ERROR]
        assert scores[0] < 0

    def test_no_code_most_negative(self):
        reward = HemlockExecutionReward()
        scores = reward(["prompt"], ["Here is how you'd do it..."])
        assert scores == [R_NO_CODE]

    def test_silent_success_lower_than_output(self):
        reward = HemlockExecutionReward()
        scores = reward(
            ["p", "p"],
            [f"```\n{SILENT_PROGRAM}\n```", f"```\n{GOOD_PROGRAM}\n```"],
        )
        assert scores == [R_RUNS, R_RUNS_OUTPUT]

    def test_timeout_negative(self):
        reward = HemlockExecutionReward()
        scores = reward(["p"], [f"```\n{LOOP_PROGRAM}\n```"])
        assert scores == [R_TIMEOUT]

    def test_correctness_path_with_metadata(self):
        reward = HemlockExecutionReward()
        completions = [f"```\n{GOOD_PROGRAM}\n```", f"```\n{GOOD_PROGRAM}\n```"]
        metadata = [
            {"expected_stdout": "hello from hemlock"},   # match -> R_CORRECT
            {"expected_stdout": "something else"},        # runs, wrong -> R_RUNS
        ]
        scores = reward(["p", "p"], completions, metadata)
        assert scores == [R_CORRECT, R_RUNS]

    def test_none_expected_stdout_falls_back_to_validity(self):
        reward = HemlockExecutionReward()
        scores = reward(["p"], [f"```\n{GOOD_PROGRAM}\n```"], [{"expected_stdout": None}])
        assert scores == [R_RUNS_OUTPUT]

    def test_batch_order_preserved(self):
        reward = HemlockExecutionReward()
        completions = [
            f"```\n{BAD_PROGRAM}\n```",
            "Sure, here is an explanation instead of code.",
            f"```\n{GOOD_PROGRAM}\n```",
        ]
        scores = reward(["p"] * 3, completions)
        assert scores == [R_ERROR, R_NO_CODE, R_RUNS_OUTPUT]

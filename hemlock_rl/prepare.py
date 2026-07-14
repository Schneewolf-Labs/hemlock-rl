"""Synthesize expected_stdout for hemlang datasets by executing reference outputs.

For each row, the reference SFT answer (`output` column) is run through the
Hemlock sandbox. Rows whose reference executes cleanly, prints something, and
does so deterministically (two identical runs) gain an ``expected_stdout``
column; the rest get None and fall back to graded-validity rewards.

    python -m hemlock_rl.prepare --out /path/codex2-verified.jsonl
"""

import argparse
import concurrent.futures
import json

from datasets import load_dataset

from .data import DEFAULT_DATASET
from .reward import MAX_WORKERS, extract_code, run_hemlock


def verify_reference(output_text):
    """Return the reference's stdout if it runs, prints, and is deterministic;
    else (None, reason)."""
    code = extract_code(output_text or "")
    if code is None:
        return None, "no_code"
    rc, out = run_hemlock(code)
    if rc is None:
        return None, "timeout"
    if rc != 0:
        return None, "error"
    if not out.strip():
        return None, "silent"
    rc2, out2 = run_hemlock(code)
    if rc2 != 0 or out2 != out:
        return None, "nondeterministic"
    return out, "ok"


def main():
    parser = argparse.ArgumentParser(description="Verify reference outputs into expected_stdout")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-field", default="output", help="column holding the reference SFT answer")
    parser.add_argument("--out", required=True, help="destination .jsonl")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(verify_reference, ds[args.output_field]))

    counts = {}
    with open(args.out, "w") as f:
        for row, (stdout, reason) in zip(ds, results):
            counts[reason] = counts.get(reason, 0) + 1
            row = dict(row)
            row["expected_stdout"] = stdout
            f.write(json.dumps(row) + "\n")

    total = len(ds)
    print(f"{args.dataset} -> {args.out}: {total} rows")
    for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:17} {n:5}  ({n / total:.0%})")


if __name__ == "__main__":
    main()

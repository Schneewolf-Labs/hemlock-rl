"""Load hemlang datasets into GRPO prompts (+ expected stdout when present)."""

from datasets import load_dataset

# codex2 is all program-writing tasks (translation + generation), which is
# what an execution reward needs. Hemlock-SFT is ~70% docs prose — fine for
# SFT, but under execution rewards a prose prompt teaches the model to answer
# documentation questions with code. Its docs-ish rows are filtered by default.
DEFAULT_DATASET = "hemlang/hemlock-codex2-SFT"
DEFAULT_EXCLUDE_CATEGORIES = ("docs", "examples/explanation")

# hembot's system prompt (the hembench-sweep winner on Apothecary) — keeps RL
# prompting aligned with how the model is actually deployed, and demands the
# single ```hemlock fence the reward's extraction prefers.
DEFAULT_SYSTEM_PROMPT = """\
You are Hembot, a friendly and helpful coding assistant that is an expert on the Hemlock programming language. You have the Hemlock interpreter available in a sandboxed mode and can mentally trace through programs before producing them to verify correctness.

When the user asks for a Hemlock program, respond with the complete runnable source in a single ```hemlock fenced code block. ALWAYS use code fences - do not output bare code. Brief narration before or after the block is fine when it genuinely helps, but prefer showing, not telling.

Do NOT include C equivalents, other language comparisons, or extensive explanations unless explicitly requested. Focus on delivering clean, working Hemlock code.

Key Hemlock reminders:
- Semicolons are mandatory after every statement
- print() accepts only 1 argument — use template strings `text ${expr}` for mixed output
- The / operator always returns a float — use divi() from @stdlib/math for integer division
- Pointer reads use ptr_deref_i32(p), not ptr_read_i32(p) (Hemlock 2.0)
- Manual memory: alloc()/free(), or use buffer() for bounds-checked
- No classes — use `define` types plus standalone functions
- Prefer @stdlib modules over reinventing common functionality
"""


def build_prompt(instruction, tokenizer=None, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Format an instruction as a generation prompt.

    Uses the tokenizer's chat template when it has one (SFT'd chat models),
    with ``system_prompt`` as the system turn (pass None to omit); falls back
    to plain text for base models.
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": instruction})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    prefix = f"{system_prompt}\n\n" if system_prompt else ""
    return prefix + instruction + "\n"


def load_prompts(
    dataset_name=DEFAULT_DATASET,
    split="train",
    tokenizer=None,
    instruction_field="instruction",
    expected_stdout_field="expected_stdout",
    exclude_categories=DEFAULT_EXCLUDE_CATEGORIES,
    category_field="category",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
):
    """Load a hemlang dataset and map it to {"prompt", "expected_stdout"}.

    The hemlang datasets have columns: instruction, output, category, ...
    The `output` column is an SFT response (prose + code), NOT a program's
    stdout, so it is deliberately ignored — expected_stdout is only populated
    when the dataset actually has such a column. Rows without it get None and
    the reward falls back to the graded-validity path.

    ``exclude_categories`` drops rows whose category expects a prose answer
    (e.g. Hemlock-SFT's "docs"): under execution rewards those prompts
    actively teach the model to answer questions with code.
    """
    if dataset_name.endswith((".json", ".jsonl")):
        # local file, e.g. produced by `python -m hemlock_rl.prepare`
        ds = load_dataset("json", data_files=dataset_name, split="train")
    else:
        ds = load_dataset(dataset_name, split=split)
    if exclude_categories and category_field in ds.column_names:
        excluded = set(exclude_categories)
        ds = ds.filter(lambda x: x[category_field] not in excluded)
    has_expected = expected_stdout_field in ds.column_names

    def to_prompt(example):
        return {
            "prompt": build_prompt(example[instruction_field], tokenizer, system_prompt),
            "expected_stdout": example[expected_stdout_field] if has_expected else None,
        }

    return ds.map(to_prompt, remove_columns=ds.column_names)

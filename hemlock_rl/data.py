"""Load hemlang datasets into GRPO prompts (+ expected stdout when present)."""

from datasets import load_dataset

DEFAULT_DATASET = "hemlang/Hemlock-SFT"


def build_prompt(instruction, tokenizer=None):
    """Format an instruction as a generation prompt.

    Uses the tokenizer's chat template when it has one (SFT'd chat models);
    falls back to the bare instruction for base models.
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return instruction + "\n"


def load_prompts(
    dataset_name=DEFAULT_DATASET,
    split="train",
    tokenizer=None,
    instruction_field="instruction",
    expected_stdout_field="expected_stdout",
):
    """Load a hemlang dataset and map it to {"prompt", "expected_stdout"}.

    hemlang/Hemlock-SFT has columns: instruction, output, category, source.
    The `output` column is an SFT response (prose + code), NOT a program's
    stdout, so it is deliberately ignored — expected_stdout is only populated
    when the dataset actually has such a column. Rows without it get None and
    the reward falls back to the graded-validity path.
    """
    ds = load_dataset(dataset_name, split=split)
    has_expected = expected_stdout_field in ds.column_names

    def to_prompt(example):
        return {
            "prompt": build_prompt(example[instruction_field], tokenizer),
            "expected_stdout": example[expected_stdout_field] if has_expected else None,
        }

    return ds.map(to_prompt, remove_columns=ds.column_names)

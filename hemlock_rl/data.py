"""Load hemlang datasets into GRPO prompts."""

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
):
    """Load a hemlang dataset and map it to {"prompt"}.

    hemlang/Hemlock-SFT has columns: instruction, output, category, source.
    The `output` column is an SFT response (prose + code), NOT a program's
    stdout, so it is deliberately ignored — GRPO only needs prompts; the
    reward comes from executing generated completions.
    """
    ds = load_dataset(dataset_name, split=split)

    def to_prompt(example):
        return {"prompt": build_prompt(example[instruction_field], tokenizer)}

    return ds.map(to_prompt, remove_columns=ds.column_names)

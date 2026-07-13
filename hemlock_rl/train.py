"""Train an LLM to write Hemlock via GRPO with execution rewards.

Wires grimoire's GRPOMethod + GrimoireTrainer around HemlockExecutionReward.
Passing GRPOMethod as loss_fn is all it takes — the trainer calls its
rollout() each step to generate completions and score them with the reward.

Example (tiny smoke run):
    python -m hemlock_rl.train --model Qwen/Qwen2.5-Coder-0.5B-Instruct \\
        --max-examples 8 --num-generations 4 --max-new-tokens 128
"""

import argparse

import torch
from grimoire import GrimoireTrainer, TrainingConfig
from grimoire.data import tokenize_grpo
from grimoire.losses import GRPOMethod
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import DEFAULT_DATASET, load_prompts
from .reward import HemlockExecutionReward


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO-train an LLM to write Hemlock")
    parser.add_argument("--model", required=True, help="HF model name or path (seed it with Hemlock SFT first)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="./hemlock-grpo-output")
    parser.add_argument("--max-examples", type=int, default=None, help="Truncate the dataset (smoke runs)")
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.04, help="KL penalty vs. the frozen base policy")
    parser.add_argument("--batch-size", type=int, default=4, help="Prompts per step (completions = this * G)")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    dataset = load_prompts(args.dataset, split=args.split, tokenizer=tokenizer)
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    dataset = dataset.map(
        lambda x: tokenize_grpo(x, tokenizer, max_prompt_length=args.max_prompt_length),
        remove_columns=dataset.column_names,
    )

    config = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        mixed_precision=args.mixed_precision,
    )

    # GRPO's rollout calls model.generate(), which needs full weight access:
    # ZeRO-2 or lower (or FSDP), never ZeRO-3 — the trainer raises otherwise.
    trainer = GrimoireTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        loss_fn=GRPOMethod(
            reward_fn=HemlockExecutionReward(),
            tokenizer=tokenizer,
            num_generations=args.num_generations,
            beta=args.beta,
            max_new_tokens=args.max_new_tokens,
        ),
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

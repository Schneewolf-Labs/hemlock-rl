"""Train an LLM to write Hemlock via execution rewards.

Wires grimoire's online methods (GRPO/RLOO/RAFT/Online DPO) + GrimoireTrainer
around HemlockExecutionReward. Passing the method as loss_fn is all it takes —
the trainer calls its rollout() each step to generate completions and score
them with the reward.

Example (tiny smoke run):
    python -m hemlock_rl.train --model Qwen/Qwen2.5-Coder-0.5B-Instruct \\
        --method raft --max-examples 16 --num-generations 4 --max-new-tokens 128
"""

import argparse

import torch
from grimoire import GrimoireTrainer, TrainerCallback, TrainingConfig
from grimoire.data import tokenize_grpo
from grimoire.losses import GRPOMethod, OnlineDPOMethod, RAFTMethod, RLOOMethod
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import DEFAULT_DATASET, load_prompts
from .reward import HemlockExecutionReward


class RewardLogger(TrainerCallback):
    """Print rollout metrics to the console each step (the trainer only
    forwards them to a tracker like wandb otherwise)."""

    KEYS = (
        "rewards_mean", "rewards_std", "best_reward", "reward_margin",
        "filtered_winners", "tied_pairs", "zero_variance_groups",
        "kl", "policy_loss",
    )

    def on_step_end(self, trainer, step, loss, metrics):
        parts = [f"step {step}", f"loss {loss:.4f}"]
        parts += [f"{k} {metrics[k]:.3f}" for k in self.KEYS if k in metrics]
        print(" | ".join(parts))


def build_method(args, tokenizer):
    common = dict(
        reward_fn=HemlockExecutionReward(),
        tokenizer=tokenizer,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
    )
    if args.method == "grpo":
        return GRPOMethod(
            beta=args.beta,
            scale_rewards=not args.no_scale_rewards,
            dynamic_sampling=args.dynamic_sampling,
            **common,
        )
    if args.method == "rloo":
        return RLOOMethod(beta=args.beta, **common)
    if args.method == "raft":
        return RAFTMethod(min_reward=args.min_reward, **common)
    # online_dpo requires a reference policy. With --lora the frozen base
    # weights serve via disable_adapter(); a full-weight run needs a frozen
    # copy of the model (doubles model VRAM).
    ref_model = None
    if not args.lora:
        import copy

        ref_model = copy.deepcopy(args._model)
        ref_model.eval()
    return OnlineDPOMethod(beta=args.beta, ref_model=ref_model, **common)


def parse_args():
    parser = argparse.ArgumentParser(description="Online-RL-train an LLM to write Hemlock")
    parser.add_argument("--model", required=True, help="HF model name or path (seed it with Hemlock SFT first)")
    parser.add_argument("--method", default="grpo", choices=["grpo", "rloo", "raft", "online_dpo"])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="./hemlock-grpo-output")
    parser.add_argument("--max-examples", type=int, default=None, help="Truncate the dataset (smoke runs)")
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty (grpo/rloo) or DPO beta (online_dpo); unused by raft")
    parser.add_argument("--min-reward", type=float, default=None,
                        help="raft only: mask winners scoring below this floor out of the loss")
    parser.add_argument("--no-scale-rewards", action="store_true",
                        help="grpo only: drop per-group std normalization (Dr. GRPO)")
    parser.add_argument("--dynamic-sampling", action="store_true",
                        help="grpo only: mask zero-variance groups out of the loss (DAPO)")
    parser.add_argument("--lora", action="store_true",
                        help="train a LoRA adapter instead of full weights — needed to fit 7B "
                             "on one GPU, and gives GRPO/RLOO/Online DPO the frozen base as a "
                             "free reference policy via disable_adapter()")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
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
    if args.lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()
    args._model = model

    dataset = load_prompts(args.dataset, split=args.split, tokenizer=tokenizer)
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    dataset = dataset.map(
        lambda x: tokenize_grpo(
            x, tokenizer,
            max_prompt_length=args.max_prompt_length,
            # Carried through to the reward: rows with an expected_stdout are
            # scored for correctness, the rest for graded validity.
            metadata_fields=["expected_stdout"],
        ),
        remove_columns=dataset.column_names,
    )

    config = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        mixed_precision=args.mixed_precision,
        logging_steps=1,
    )

    # Online rollouts call model.generate(), which needs full weight access:
    # ZeRO-2 or lower (or FSDP), never ZeRO-3 — the trainer raises otherwise.
    trainer = GrimoireTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        loss_fn=build_method(args, tokenizer),
        train_dataset=dataset,
        callbacks=[RewardLogger()],
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

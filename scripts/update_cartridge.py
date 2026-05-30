#!/usr/bin/env python3
"""Fine-tune an existing cartridge on a small set of new Q&A pairs.

Loads a trained cartridge .pt, runs N gradient steps against new supervision
pairs in train_dataset.jsonl format, and saves the updated cartridge.
No vLLM or teacher generation needed — supply pre-processed supervision.

Usage:
    python scripts/update_cartridge.py \
        --cartridge-path outputs/.../text-0_cartridge.pt \
        --dataset-path outputs/.../train_dataset.jsonl \
        --output-path outputs/.../text-0_updated.pt \
        --steps 15
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.config import DEFAULT_MATRIX  # noqa: E402
from cartridges.core import TrainableKVCartridge  # noqa: E402
from cartridges.train.cartridge import (  # noqa: E402
    _compute_example_loss,
    load_training_examples,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update a trained cartridge with new Q&A supervision pairs."
    )
    parser.add_argument(
        "--cartridge-path", required=True,
        help="Path to the existing trained cartridge .pt file.",
    )
    parser.add_argument(
        "--dataset-path", required=True,
        help="JSONL file of new Q&A pairs in train_dataset.jsonl format.",
    )
    parser.add_argument(
        "--output-path",
        help=(
            "Where to save the updated cartridge. "
            "Defaults to <cartridge-stem>_updated.pt in the same directory."
        ),
    )
    parser.add_argument("--steps", type=int, default=15,
                        help="Gradient steps to run (default: 15).")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cartridge_path = Path(args.cartridge_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = cartridge_path.with_stem(cartridge_path.stem + "_updated")

    examples = load_training_examples(args.dataset_path)
    print(f"Loaded {len(examples)} training examples from {args.dataset_path}")
    if len(examples) < args.steps:
        print(
            f"Warning: {args.steps} steps requested but only {len(examples)} examples — "
            "examples will be cycled."
        )

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    cartridge = TrainableKVCartridge.load(cartridge_path, device=args.device)
    print(
        f"Loaded cartridge from {cartridge_path}\n"
        f"  tokens={cartridge.num_tokens}  layers={cartridge.num_layers}  "
        f"frozen={cartridge.num_frozen_tokens}"
    )

    optimizer = AdamW(cartridge.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)

    print(f"\nFine-tuning for {args.steps} steps on {len(examples)} examples...")
    started = time.perf_counter()
    loss_history: list[float] = []
    for step_idx in range(args.steps):
        example = examples[step_idx % len(examples)]
        loss = _compute_example_loss(
            model=model,
            tokenizer=tokenizer,
            cartridge=cartridge,
            example=example,
            device=args.device,
        )
        loss_value = float(loss.item())
        loss_history.append(loss_value)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cartridge.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"  step {step_idx + 1:>3}/{args.steps}  loss={loss_value:.4f}")

    elapsed = time.perf_counter() - started
    cartridge.save(output_path)

    print(
        f"\nDone in {elapsed:.1f}s  ({elapsed / args.steps:.2f}s/step)\n"
        f"  Initial loss : {loss_history[0]:.4f}\n"
        f"  Final loss   : {loss_history[-1]:.4f}\n"
        f"  Saved to     : {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

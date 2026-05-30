#!/usr/bin/env python3
"""Incrementally update a cartridge with new Q&A pairs using delta slots.

The existing cartridge is frozen entirely — zero forgetting possible.
A small set of trainable delta K/V slots (default 32 positions) is appended
and trained on the new pairs. The final output is a merged cartridge
(base + delta) saved as a standard TrainableKVCartridge .pt file that
works with all existing eval infrastructure.

Usage:
    python scripts/update_cartridge.py \\
        --cartridge-path outputs/.../text-0_cartridge.pt \\
        --dataset-path /tmp/new_pairs.jsonl \\
        --output-path /tmp/text-0_merged.pt \\
        --delta-tokens 32 \\
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
from cartridges.core import (  # noqa: E402
    DeltaKVCartridge,
    TrainableKVCartridge,
    initialize_delta_from_text,
)
from cartridges.train.cartridge import (  # noqa: E402
    _compute_example_loss,
    load_training_examples,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update a cartridge with new Q&A pairs via frozen-base delta slots."
    )
    parser.add_argument("--cartridge-path", required=True,
                        help="Path to the existing trained cartridge .pt.")
    parser.add_argument("--dataset-path", required=True,
                        help="JSONL of new Q&A pairs in train_dataset.jsonl format.")
    parser.add_argument("--output-path",
                        help="Where to save the merged cartridge. "
                             "Defaults to <stem>_updated.pt alongside the original.")
    parser.add_argument("--delta-tokens", type=int, default=32,
                        help="Number of new trainable delta slots (default: 32).")
    parser.add_argument("--steps", type=int, default=15,
                        help="Gradient steps to run on new pairs (default: 15).")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cartridge_path = Path(args.cartridge_path)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else cartridge_path.with_stem(cartridge_path.stem + "_updated")
    )

    examples = load_training_examples(args.dataset_path)
    print(f"Loaded {len(examples)} training examples.")
    if len(examples) < args.steps:
        print(f"  (fewer than {args.steps} examples — will cycle)")

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

    base_cartridge = TrainableKVCartridge.load(cartridge_path, device=args.device)
    print(
        f"Loaded base cartridge: {base_cartridge.num_tokens} base slots, "
        f"{base_cartridge.num_layers} layers"
    )

    # Initialize delta from positions base_tokens..base_tokens+delta_tokens-1 of the
    # system prompt text. This gives sequential RoPE positions that continue directly
    # after the base cartridge range, avoiding positional collision.
    system_prompt = examples[0].system_prompt
    delta_keys, delta_values = initialize_delta_from_text(
        model=model,
        tokenizer=tokenizer,
        text=system_prompt,
        base_tokens=base_cartridge.num_tokens,
        delta_tokens=args.delta_tokens,
    )
    print(f"Initialized {args.delta_tokens} delta slots from text continuation.")

    delta_cartridge = DeltaKVCartridge.from_base_cartridge(
        base_cartridge=base_cartridge,
        delta_keys=delta_keys,
        delta_values=delta_values,
    )
    delta_cartridge.to(args.device)

    # Only optimise the delta parameters — base is frozen by requires_grad=False.
    optimizer = AdamW(
        list(delta_cartridge.delta_keys.parameters())
        + list(delta_cartridge.delta_values.parameters()),
        lr=args.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)

    print(f"\nTraining {args.delta_tokens} delta slots for {args.steps} steps...")
    started = time.perf_counter()
    loss_history: list[float] = []
    for step_idx in range(args.steps):
        example = examples[step_idx % len(examples)]
        loss = _compute_example_loss(
            model=model,
            tokenizer=tokenizer,
            cartridge=delta_cartridge,
            example=example,
            device=args.device,
        )
        loss_value = float(loss.item())
        loss_history.append(loss_value)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(delta_cartridge.delta_keys.parameters())
            + list(delta_cartridge.delta_values.parameters()),
            args.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"  step {step_idx + 1:>3}/{args.steps}  loss={loss_value:.4f}")

    elapsed = time.perf_counter() - started

    # Merge base + trained delta into a standard TrainableKVCartridge for eval.
    merged = delta_cartridge.merge()
    merged.save(output_path)

    print(
        f"\nDone in {elapsed:.1f}s  ({elapsed / args.steps:.2f}s/step)\n"
        f"  Initial loss   : {loss_history[0]:.4f}\n"
        f"  Final loss     : {loss_history[-1]:.4f}\n"
        f"  Base slots     : {base_cartridge.num_tokens}\n"
        f"  Delta slots    : {args.delta_tokens}\n"
        f"  Total slots    : {base_cartridge.num_tokens + args.delta_tokens}\n"
        f"  Saved to       : {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

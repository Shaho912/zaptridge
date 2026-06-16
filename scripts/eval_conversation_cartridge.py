#!/usr/bin/env python3
"""Evaluate a conversation cartridge by querying it on a set of questions.

Used to test whether fine-tuned cartridges retain old knowledge (no forgetting)
after incremental updates.

Usage:
    python scripts/eval_conversation_cartridge.py \
        --cartridge-path outputs/conversation/sample_8b_day2.pt \
        --device cuda:0
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.config import DEFAULT_HF_MODEL_ID  # noqa: E402
from cartridges.core import TrainableKVCartridge    # noqa: E402

DAY1_QUESTIONS = [
    "What is the capital of India?",
    "What is India's population?",
    "What are the major religions in India?",
    "Describe India's economy.",
    "What languages are spoken in India?",
    "When did India gain independence?",
]

DAY2_QUESTIONS = [
    "What is India's national animal?",
    "What is the significance of the Ganges river?",
]


def generate(model, tokenizer, cartridge, question: str, device: str, max_new_tokens: int = 80) -> str:
    messages = [{"role": "user", "content": f"/no_think\n{question}"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            past_key_values=cartridge.as_cache(model.config),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Strip any residual think tags
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartridge-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    print(f"Loading model: {DEFAULT_HF_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_HF_MODEL_ID,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(args.device)
    model.eval()

    print(f"Loading cartridge: {args.cartridge_path}")
    cartridge = TrainableKVCartridge.load(args.cartridge_path, device=args.device)
    print(f"  {cartridge.num_tokens} slots, {cartridge.num_layers} layers\n")

    print("=" * 60)
    print("DAY 1 QUESTIONS (testing memory retention)")
    print("=" * 60)
    for q in DAY1_QUESTIONS:
        answer = generate(model, tokenizer, cartridge, q, args.device, args.max_new_tokens)
        print(f"Q: {q}")
        print(f"A: {answer}")
        print()

    print("=" * 60)
    print("DAY 2 QUESTIONS (testing new knowledge)")
    print("=" * 60)
    for q in DAY2_QUESTIONS:
        answer = generate(model, tokenizer, cartridge, q, args.device, args.max_new_tokens)
        print(f"Q: {q}")
        print(f"A: {answer}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate conversation recall using full-context prefill (no cartridge).

Loads one or more conversation JSONLs, concatenates them as a system prompt,
and queries the model directly. This is the baseline comparison for cartridge
eval — same questions, same model, but the full conversation history is
re-prefilled on every query instead of loaded from a cartridge.

Usage:
    python scripts/eval_baseline_conversation.py \
        --conversation-path data/zorbia.jsonl \
        --conversation-path data/blarvia.jsonl \
        --conversation-path data/glorvia.jsonl \
        --conversation-path data/threndia.jsonl \
        --device cuda:0
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.config import DEFAULT_HF_MODEL_ID  # noqa: E402

DAY1_QUESTIONS = [
    "What is the capital of Zorbia?",
    "What is the population of Zorbia?",
    "What is the currency of Zorbia?",
    "Who is the president of Zorbia?",
    "What is Zorbia's national sport?",
]

DAY2_QUESTIONS = [
    "What is the capital of Blarvia?",
    "What language do people speak in Blarvia?",
    "What is Blarvia's main export?",
]

DAY3_QUESTIONS = [
    "What is the capital of Glorvia?",
    "What is the population of Glorvia?",
    "What is the currency of Glorvia?",
    "Who leads Glorvia?",
    "What is Glorvia's main export?",
]

DAY4_QUESTIONS = [
    "What is the capital of Threndia?",
    "What is the population of Threndia?",
    "What is the currency of Threndia?",
    "Who is the president of Threndia?",
    "What is Threndia's main export?",
]


def _load_conversation(path: str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def _format_conversation(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        role = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{role}: {t['content'].strip()}")
    return "\n\n".join(lines)


def generate(model, tokenizer, system_prompt: str, question: str, device: str,
             max_new_tokens: int = 200) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"/no_think\n{question}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    prompt_tokens = input_ids.shape[1]

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    generated = output_ids[0, prompt_tokens:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"</think>", "", text).strip()
    text = re.sub(r"^(Okay[,.].*?\n\n)", "", text, flags=re.DOTALL).strip()
    return text, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Baseline full-context prefill eval for conversation recall."
    )
    parser.add_argument("--conversation-path", action="append", required=True,
                        dest="conversation_paths",
                        help="Conversation JSONL to include. Repeat for each day.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=200)
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

    # Build combined system prompt from all conversations
    all_parts = []
    for p in args.conversation_paths:
        turns = _load_conversation(p)
        all_parts.append(_format_conversation(turns))
        print(f"  Loaded {len(turns) // 2} turns from {p}")

    combined_text = "\n\n---\n\n".join(all_parts)
    system_prompt = (
        "Please answer the user's message using only the provided context.\n\n"
        f"<context>\n{combined_text}\n</context>\n\n"
        "Answer directly and concisely. Do not emit <think> tags or chain-of-thought."
    )

    # Count prompt tokens for reporting
    prompt_ids = tokenizer.encode(system_prompt, add_special_tokens=False)
    print(f"\nSystem prompt: {len(prompt_ids)} tokens (re-prefilled on every query)\n")

    sections = [
        ("DAY 1 QUESTIONS (Zorbia)", DAY1_QUESTIONS),
        ("DAY 2 QUESTIONS (Blarvia)", DAY2_QUESTIONS),
        ("DAY 3 QUESTIONS (Glorvia)", DAY3_QUESTIONS),
        ("DAY 4 QUESTIONS (Threndia)", DAY4_QUESTIONS),
    ]

    total_time = 0.0
    total_questions = 0
    for label, questions in sections:
        print("=" * 60)
        print(label)
        print("=" * 60)
        for q in questions:
            answer, elapsed = generate(model, tokenizer, system_prompt, q,
                                       args.device, args.max_new_tokens)
            print(f"Q: {q}")
            print(f"A: {answer}")
            print(f"   [{elapsed:.2f}s]")
            print()
            total_time += elapsed
            total_questions += 1

    print(f"Total: {total_questions} questions in {total_time:.1f}s "
          f"({total_time / total_questions:.2f}s/query avg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

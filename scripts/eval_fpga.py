#!/usr/bin/env python3
"""Evaluate FPGA conversation recall — cartridge mode or baseline prefill mode.

Cartridge mode:
    python scripts/eval_fpga.py --cartridge-path outputs/conversation/fpga_8b.pt --device cuda:0

Baseline mode (full-context prefill on every query):
    python scripts/eval_fpga.py --baseline --conversation-path data/fpga_convo.jsonl --device cuda:0
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
from cartridges.core import TrainableKVCartridge    # noqa: E402

# question → expected answer (for reference display)
QUESTIONS = [
    ("What is the parse latency for an Add Order message in clock cycles?",
     "8 cycles"),
    ("What is the parse latency for an Add Order message in nanoseconds?",
     "80 ns"),
    ("How many clock cycles does parsing an Order Executed message take?",
     "6 cycles"),
    ("How many LUTs does the Add Order parser use?",
     "847 LUTs"),
    ("How many flip-flops does the Add Order parser use?",
     "1203 FFs"),
    ("What was the AXI-Stream backpressure bug and how was it fixed?",
     "TREADY not deasserted → added output handshake FSM state"),
    ("How is the ITCH price field decoded to get a decimal price?",
     "divide by 10000"),
    ("How many cycles does the Add Order parser take per message for sustained throughput?",
     "36 cycles"),
    ("What was the verification accuracy on the 10,000 message PCAP sample?",
     "100%"),
    ("Why is price conversion done on the ARM side rather than in PL?",
     "to avoid DSP utilization"),
]


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"</think>", "", text).strip()
    text = re.sub(r"^(Okay[,.].*?\n\n)", "", text, flags=re.DOTALL).strip()
    return text


def generate_cartridge(model, tokenizer, cartridge, question: str, device: str,
                       max_new_tokens: int = 200) -> tuple[str, float]:
    messages = [{"role": "user", "content": f"/no_think\n{question}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, past_key_values=cartridge.as_cache(model.config), use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids: list[int] = []
        eos = tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            if eos is not None and token_id == eos:
                break
            outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return _strip_thinking(text), elapsed


def generate_baseline(model, tokenizer, system_prompt: str, question: str, device: str,
                      max_new_tokens: int = 200) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"/no_think\n{question}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    prompt_tokens = input_ids.shape[1]

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(output_ids[0, prompt_tokens:], skip_special_tokens=True).strip()
    return _strip_thinking(text), elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartridge-path", default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--conversation-path", default=None,
                        help="Required for --baseline mode.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    if not args.baseline and not args.cartridge_path:
        parser.error("Provide --cartridge-path or --baseline.")
    if args.baseline and not args.conversation_path:
        parser.error("--baseline requires --conversation-path.")

    print(f"Loading model: {DEFAULT_HF_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_HF_MODEL_ID,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(args.device)
    model.eval()

    if args.baseline:
        turns = [json.loads(l) for l in Path(args.conversation_path).read_text().splitlines() if l.strip()]
        lines = []
        for t in turns:
            role = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"{role}: {t['content'].strip()}")
        conv_text = "\n\n".join(lines)
        system_prompt = (
            "Please answer the user's message using only the provided context.\n\n"
            f"<context>\n{conv_text}\n</context>\n\n"
            "Answer directly and concisely. Do not emit <think> tags or chain-of-thought."
        )
        prompt_tokens = len(tokenizer.encode(system_prompt, add_special_tokens=False))
        print(f"\nBaseline: {prompt_tokens} tokens re-prefilled on every query\n")
        mode = "BASELINE"
    else:
        cartridge = TrainableKVCartridge.load(args.cartridge_path, device=args.device)
        print(f"\nCartridge: {cartridge.num_tokens} slots, {cartridge.num_layers} layers\n")
        mode = "CARTRIDGE"

    print(f"{'=' * 60}")
    print(f"FPGA CONVERSATION EVAL — {mode}")
    print(f"{'=' * 60}\n")

    total_time = 0.0
    for i, (question, expected) in enumerate(QUESTIONS, 1):
        if args.baseline:
            answer, elapsed = generate_baseline(model, tokenizer, system_prompt, question,
                                                args.device, args.max_new_tokens)
        else:
            answer, elapsed = generate_cartridge(model, tokenizer, cartridge, question,
                                                 args.device, args.max_new_tokens)
        print(f"Q{i:02d}: {question}")
        print(f"  Expected : {expected}")
        print(f"  Got      : {answer}")
        print(f"  [{elapsed:.2f}s]")
        print()
        total_time += elapsed

    n = len(QUESTIONS)
    print(f"Total: {n} questions in {total_time:.1f}s ({total_time / n:.2f}s/query avg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

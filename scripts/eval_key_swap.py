#!/usr/bin/env python3
"""Key-swap ablation: test whether cartridge keys are corpus-agnostic routers.

Creates a hybrid cartridge with A's values but B's keys, then evaluates it on A's
questions. If accuracy is preserved, keys are interchangeable (corpus-agnostic routers
as claimed by Diaz). If accuracy drops, the keys encode corpus-specific information.

Usage:
    python scripts/eval_key_swap.py \
        --cart-a outputs/exp_rotation_1/lh_p01_cartridge.pt \
        --cart-b outputs/exp_rotation_1/lh_p02_cartridge.pt \
        --questions-a data/lh_p01/eval_questions.json \
        --device cuda:0

    # To also run the reverse (B values + A keys, evaluated on B):
    python scripts/eval_key_swap.py \
        --cart-a outputs/exp_rotation_1/lh_p01_cartridge.pt \
        --cart-b outputs/exp_rotation_1/lh_p02_cartridge.pt \
        --questions-a data/lh_p01/eval_questions.json \
        --questions-b data/lh_p02/eval_questions.json \
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
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.config import DEFAULT_HF_MODEL_ID  # noqa: E402
from cartridges.core import TrainableKVCartridge    # noqa: E402


def _make_swapped(cart_keys: TrainableKVCartridge,
                  cart_values: TrainableKVCartridge) -> TrainableKVCartridge:
    """Build a new cartridge using keys from cart_keys and values from cart_values."""
    keys, values = [], []
    for i in range(cart_keys.num_layers):
        k, _ = cart_keys.layer(i)
        _, v = cart_values.layer(i)
        keys.append(k.detach().clone())
        values.append(v.detach().clone())
    return TrainableKVCartridge(keys=keys, values=values, num_frozen_tokens=0)


def _generate(model, tokenizer, cartridge: TrainableKVCartridge,
              question: str, device: str, max_new_tokens: int = 200) -> tuple[str, float]:
    messages = [{"role": "user", "content": f"/no_think\n{question}"}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    input_ids = tokenizer.encode(prompt, return_tensors="pt",
                                 add_special_tokens=False).to(device)
    t0 = time.perf_counter()
    generated_ids = []
    with torch.inference_mode():
        cache = cartridge.as_cache(model.config)
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=input_ids if not generated_ids else
                          torch.tensor([[generated_ids[-1]]], device=device, dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            next_id = int(outputs.logits[0, -1].argmax())
            if next_id == tokenizer.eos_token_id:
                break
            generated_ids.append(next_id)

    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"</think>", "", text).strip()
    return text, elapsed


def _eval_cartridge(model, tokenizer, cartridge: TrainableKVCartridge,
                    questions: list[tuple[str, str]], label: str,
                    device: str, max_new_tokens: int = 200) -> int:
    correct = 0
    print(f"\n  [{label}]")
    for q, expected in questions:
        answer, elapsed = _generate(model, tokenizer, cartridge, q, device, max_new_tokens)
        hit = expected.lower() in answer.lower()
        correct += int(hit)
        status = "✓" if hit else "✗"
        print(f"    {status}  Q: {q[:60]}")
        print(f"       Expected: {expected!r}")
        print(f"       Got:      {answer[:120]!r}  ({elapsed:.1f}s)")
    print(f"\n  [{label}] Score: {correct}/{len(questions)}")
    return correct


def _load_questions(path: Path, max_questions: int) -> list[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pairs = [(q, a) for q, a in raw]
    return pairs[:max_questions]


def main() -> int:
    parser = argparse.ArgumentParser(description="Key-swap ablation: are cartridge keys corpus-agnostic?")
    parser.add_argument("--cart-a", required=True, help="Cartridge A .pt file")
    parser.add_argument("--cart-b", required=True, help="Cartridge B .pt file")
    parser.add_argument("--questions-a", required=True,
                        help="Eval questions JSON for cartridge A  [[question, answer], ...]")
    parser.add_argument("--questions-b", default=None,
                        help="Eval questions JSON for B (optional, enables reverse swap)")
    parser.add_argument("--max-questions", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
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
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"\nLoading cartridges...")
    cart_a = TrainableKVCartridge.load(args.cart_a, device=args.device)
    cart_b = TrainableKVCartridge.load(args.cart_b, device=args.device)
    print(f"  A: {cart_a.num_layers} layers, {cart_a.num_tokens} slots")
    print(f"  B: {cart_b.num_layers} layers, {cart_b.num_tokens} slots")

    questions_a = _load_questions(Path(args.questions_a), args.max_questions)

    # --- Forward direction: A questions, vary which keys are used ---
    print(f"\n{'='*64}")
    print(f"  FORWARD: A's questions, swapping keys")
    print(f"{'='*64}")

    score_aa = _eval_cartridge(
        model, tokenizer, cart_a, questions_a,
        label="A values + A keys  (original A)", device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    swapped_bk_av = _make_swapped(cart_keys=cart_b, cart_values=cart_a)
    swapped_bk_av.to(args.device)
    score_bk_av = _eval_cartridge(
        model, tokenizer, swapped_bk_av, questions_a,
        label="A values + B keys  (key swap)", device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    drop_fwd = score_aa - score_bk_av
    print(f"\n  FORWARD key swap: {score_aa}/{len(questions_a)} → {score_bk_av}/{len(questions_a)}"
          f"  (drop={drop_fwd:+d})")

    # --- Reverse direction (optional) ---
    if args.questions_b:
        questions_b = _load_questions(Path(args.questions_b), args.max_questions)

        print(f"\n{'='*64}")
        print(f"  REVERSE: B's questions, swapping keys")
        print(f"{'='*64}")

        score_bb = _eval_cartridge(
            model, tokenizer, cart_b, questions_b,
            label="B values + B keys  (original B)", device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        swapped_ak_bv = _make_swapped(cart_keys=cart_a, cart_values=cart_b)
        swapped_ak_bv.to(args.device)
        score_ak_bv = _eval_cartridge(
            model, tokenizer, swapped_ak_bv, questions_b,
            label="B values + A keys  (key swap)", device=args.device,
            max_new_tokens=args.max_new_tokens,
        )

        drop_rev = score_bb - score_ak_bv
        print(f"\n  REVERSE key swap: {score_bb}/{len(questions_b)} → {score_ak_bv}/{len(questions_b)}"
              f"  (drop={drop_rev:+d})")

    # --- Summary ---
    print(f"\n{'='*64}")
    print(f"  SUMMARY")
    print(f"{'='*64}")
    print(f"  A original (A keys + A values):  {score_aa}/{len(questions_a)}")
    print(f"  Key-swapped (B keys + A values): {score_bk_av}/{len(questions_a)}")
    print(f"  Key-swap drop: {drop_fwd:+d}")
    if drop_fwd == 0:
        print(f"\n  → Keys are interchangeable: swapping B's keys onto A's values")
        print(f"    causes NO accuracy drop. Keys are corpus-agnostic routers.")
    elif drop_fwd <= 1:
        print(f"\n  → Keys are mostly interchangeable: drop of only {drop_fwd}.")
        print(f"    Keys carry minimal corpus-specific information.")
    else:
        print(f"\n  → Keys are NOT interchangeable: drop of {drop_fwd} questions.")
        print(f"    Keys encode corpus-specific information beyond routing.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

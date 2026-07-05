#!/usr/bin/env python3
"""Evaluate cartridges in individual (oracle) and joint (all-at-once) modes.

Oracle mode: load each cartridge alone and ask only its questions.
Joint mode:  load all cartridges concatenated and ask all questions.

The gap between oracle and joint scores measures cartridge interference —
the core metric for Experiment 1 vs Experiment 2.

Usage:
    # Oracle + joint (full comparison):
    python scripts/eval_multi_cartridge.py \\
        --cartridge-dir outputs/exp1 \\
        --device cuda:0

    # Joint only:
    python scripts/eval_multi_cartridge.py \\
        --cartridge-dir outputs/exp1 \\
        --mode joint

    # Specify cartridge order (order matters for concatenation):
    python scripts/eval_multi_cartridge.py \\
        --cartridge-dir outputs/exp1 \\
        --names skills methods preferences research
"""

import argparse
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

# ─── Question sets ────────────────────────────────────────────────────────────
# Each tuple: (question, expected_substring_lowercase)
# A question is scored correct if expected_substring appears in the answer (case-insensitive).
QUESTION_SETS: dict[str, list[tuple[str, str]]] = {
    "zaptridge_convo": [
        ("What hardware platform is Zaptridge targeting for deployment?",           "iphone"),
        ("What is the central finding about why cartridge updates are difficult?",   "initialization bias"),
        ("How many delta slots are needed for all facts to be encoded correctly?",   "256"),
        ("What is the build time for the optimal cartridge configuration?",          "72"),
        ("What does the Diaz 2025 paper find about how much keys move during cartridge training?", "very little"),
        ("In one phrase, what is the single capability that distinguishes Zaptridge from every paper surveyed?", "cross-session"),
    ],
    "fpga_convo": [
        ("How many clock cycles does it take to parse an Add Order message?",        "8"),
        ("What is the end-to-end parse latency in nanoseconds for Add Order?",       "80"),
        ("How many LUTs does the Add Order parser use?",                             "847"),
        ("What bug caused dropped bytes on back-to-back messages?",                  "tready"),
        ("How many bytes long is an Add Order message?",                                        "36"),
        ("What accuracy did the parser achieve on the 10,000 message PCAP sample?", "100"),
    ],
}

ALL_NAMES = ["zaptridge_convo", "fpga_convo"]


def _make_combined_cache(
    cartridges: list[TrainableKVCartridge],
    model_config,
) -> DynamicCache:
    """Concatenate multiple cartridges along the sequence dimension into one cache.

    Each cartridge contributes its K/V tensors per layer. After concatenation the
    combined sequence length equals the sum of all individual cartridge lengths.
    Keys retain their original RoPE positions — the model must learn to disambiguate
    overlapping position embeddings across cartridges (which joint training addresses).
    """
    num_layers = cartridges[0].num_layers
    combined_pkv = []
    for layer_idx in range(num_layers):
        keys, values = [], []
        for cart in cartridges:
            k, v = cart.layer(layer_idx)
            keys.append(k)
            values.append(v)
        combined_pkv.append((
            torch.cat(keys, dim=-2),
            torch.cat(values, dim=-2),
        ))
    return DynamicCache(ddp_cache_data=tuple(combined_pkv), config=model_config)


def _generate(
    model,
    tokenizer,
    cache_factory,      # callable → fresh DynamicCache (called once per question)
    question: str,
    device: str,
    max_new_tokens: int = 80,
) -> tuple[str, float]:
    """Generate a single answer, returning (text, elapsed_seconds)."""
    messages = [{"role": "user", "content": f"/no_think\n{question}"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    cache = cache_factory()

    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        generated_ids: list[int] = []
        eos_token_id = tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    elapsed = time.perf_counter() - t0

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"</think>", "", text).strip()
    text = re.sub(r"^(Okay[,.].*?\n\n)", "", text, flags=re.DOTALL).strip()
    return text, elapsed


def _eval_corpus(
    *,
    corpus_name: str,
    model,
    tokenizer,
    cache_factory,
    device: str,
    max_new_tokens: int,
) -> tuple[int, int]:
    """Run all questions for one corpus. Returns (hits, total)."""
    questions = QUESTION_SETS[corpus_name]
    hits = 0
    for question, expected in questions:
        answer, elapsed = _generate(
            model, tokenizer, cache_factory, question, device, max_new_tokens
        )
        correct = expected.lower() in answer.lower()
        hits += int(correct)
        mark = "✓" if correct else "✗"
        print(f"  {mark} Q: {question}")
        print(f"       A: {answer}")
        print(f"       [expected: {expected!r}]  {elapsed:.2f}s")
    return hits, len(questions)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate cartridges in oracle (individual) and joint (combined) modes."
    )
    parser.add_argument("--cartridge-dir", required=True,
                        help="Directory containing {name}_cartridge.pt files.")
    parser.add_argument("--names", nargs="+", default=ALL_NAMES,
                        help="Cartridge names to load, in concatenation order.")
    parser.add_argument("--mode", choices=["oracle", "joint", "both"], default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    cartridge_dir = Path(args.cartridge_dir)

    print(f"Loading model: {DEFAULT_HF_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_HF_MODEL_ID,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(args.device)
    model.eval()

    # Load cartridges
    cartridges: dict[str, TrainableKVCartridge] = {}
    for name in args.names:
        path = cartridge_dir / f"{name}_cartridge.pt"
        if not path.exists():
            raise FileNotFoundError(f"Cartridge not found: {path}")
        cart = TrainableKVCartridge.load(path, device=args.device)
        cartridges[name] = cart
        print(f"  [{name}] {cart.num_tokens} slots, {cart.num_layers} layers")

    eval_names = [n for n in args.names if n in QUESTION_SETS]
    oracle_scores: dict[str, tuple[int, int]] = {}
    joint_scores:  dict[str, tuple[int, int]] = {}

    # ─── Oracle eval ─────────────────────────────────────────────────────────
    if args.mode in ("oracle", "both"):
        print("\n" + "=" * 70)
        print("ORACLE EVAL  (each cartridge loaded alone)")
        print("=" * 70)
        for name in eval_names:
            cart = cartridges[name]
            print(f"\n── {name.upper()} ({cart.num_tokens} slots) ──")
            cache_factory = lambda c=cart: c.as_cache(model.config)
            hits, total = _eval_corpus(
                corpus_name=name,
                model=model,
                tokenizer=tokenizer,
                cache_factory=cache_factory,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            oracle_scores[name] = (hits, total)
            print(f"  → {hits}/{total}")

        total_oh = sum(h for h, _ in oracle_scores.values())
        total_ot = sum(t for _, t in oracle_scores.values())
        print(f"\nOracle total: {total_oh}/{total_ot}")

    # ─── Joint eval ──────────────────────────────────────────────────────────
    if args.mode in ("joint", "both"):
        all_carts = [cartridges[n] for n in args.names]
        total_slots = sum(c.num_tokens for c in all_carts)
        print("\n" + "=" * 70)
        print(f"JOINT EVAL  ({len(args.names)} cartridges concatenated, {total_slots} total slots)")
        print(f"Order: {' | '.join(f'{n}({cartridges[n].num_tokens})' for n in args.names)}")
        print("=" * 70)

        joint_cache_factory = lambda: _make_combined_cache(all_carts, model.config)
        for name in eval_names:
            print(f"\n── {name.upper()} ──")
            hits, total = _eval_corpus(
                corpus_name=name,
                model=model,
                tokenizer=tokenizer,
                cache_factory=joint_cache_factory,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            joint_scores[name] = (hits, total)
            print(f"  → {hits}/{total}")

        total_jh = sum(h for h, _ in joint_scores.values())
        total_jt = sum(t for _, t in joint_scores.values())
        print(f"\nJoint total: {total_jh}/{total_jt}")

    # ─── Summary table ────────────────────────────────────────────────────────
    if args.mode == "both":
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Corpus':<12}  {'Oracle':>8}  {'Joint':>8}  {'Drop':>6}")
        print("-" * 40)
        for name in eval_names:
            oh, ot = oracle_scores[name]
            jh, jt = joint_scores[name]
            drop = oh - jh
            print(f"{name:<12}  {oh}/{ot:>4}       {jh}/{jt:>4}    {drop:+d}")
        print("-" * 40)
        total_oh = sum(h for h, _ in oracle_scores.values())
        total_ot = sum(t for _, t in oracle_scores.values())
        total_jh = sum(h for h, _ in joint_scores.values())
        total_jt = sum(t for _, t in joint_scores.values())
        total_drop = total_oh - total_jh
        print(f"{'Total':<12}  {total_oh}/{total_ot:>4}       {total_jh}/{total_jt:>4}    {total_drop:+d}")
        print()
        if total_drop > 0:
            print(f"Interference cost: {total_drop} questions lost when loading all {len(args.names)} cartridges jointly.")
        else:
            print("No interference detected (oracle == joint).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

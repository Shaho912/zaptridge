#!/usr/bin/env python3
"""Compress a conversation history into a KV cartridge for session memory.

Takes a JSONL file of conversation turns and trains a cartridge that encodes
the session so future turns can attend to compact KV state instead of
re-prefilling the full history on every new message.

Input format — one JSON object per line:
    {"role": "user",      "content": "..."}
    {"role": "assistant", "content": "..."}

Usage:
    python scripts/compress_conversation.py \\
        --conversation-path /tmp/session.jsonl \\
        --output-path /tmp/session_cartridge.pt \\
        --cartridge-tokens 512 \\
        --steps 60
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.config import DEFAULT_MATRIX  # noqa: E402
from cartridges.data.common import stable_hash, write_json  # noqa: E402
from cartridges.train.cartridge import train_cartridge  # noqa: E402


def _clean(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    cleaned = cleaned.removeprefix("<think>").strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def _format_conversation(turns: list[dict]) -> str:
    """Format turns as plain text for use as the cartridge system prompt."""
    lines = []
    for t in turns:
        role = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{role}: {t['content'].strip()}")
    return "\n\n".join(lines)


def _build_supervision(
    *,
    turns: list[dict],
    model,
    tokenizer,
    device: str,
    output_path: Path,
    slice_id: str = "conversation",
    top_logprobs: int = 5,
) -> Path:
    """Run teacher forward passes on conversation turns and write train_dataset.jsonl.

    For each (user, assistant) pair the teacher sees the full conversation as
    system context and is teacher-forced on the assistant response. The resulting
    per-token logprobs are the distillation target for cartridge training.
    """
    conversation_text = _format_conversation(turns)
    system_prompt = (
        "Please answer the user's message using only the provided context.\n\n"
        f"<context>\n{conversation_text}\n</context>\n\n"
        "Follow the conversation naturally. "
        "Do not emit <think> tags or chain-of-thought."
    )

    # Collect adjacent (user, assistant) pairs.
    pairs: list[tuple[str, str]] = []
    for i in range(len(turns) - 1):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            pairs.append((turns[i]["content"].strip(), turns[i + 1]["content"].strip()))

    if not pairs:
        raise ValueError("No user/assistant turn pairs found in conversation.")

    print(f"  Building supervision for {len(pairs)} turn pairs...")
    rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for idx, (user_content, assistant_content) in enumerate(pairs):
            user_message = f"/no_think\n{user_content}"
            assistant_text = _clean(assistant_content)
            if not assistant_text:
                continue

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            # Tokenize assistant response + EOS (same as _assistant_target_token_ids).
            assistant_token_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
            eos = tokenizer.eos_token_id
            if eos is None:
                raise ValueError("Tokenizer must expose eos_token_id.")
            assistant_token_ids = assistant_token_ids + [int(eos)]
            if len(assistant_token_ids) <= 1:
                continue

            # Teacher-forcing forward pass: prompt + assistant[:-1] → predict assistant[0:]
            input_ids = torch.tensor(
                [prompt_ids + assistant_token_ids[:-1]],
                device=device,
                dtype=torch.long,
            )
            logits = model(input_ids=input_ids).logits
            resp_start = len(prompt_ids)
            resp_end   = resp_start + len(assistant_token_ids) - 1
            response_logits = logits[:, resp_start - 1 : resp_end, :]
            log_probs = torch.log_softmax(response_logits, dim=-1)

            target_ids = torch.tensor(
                [assistant_token_ids], device=device, dtype=torch.long
            )
            target_log_probs = (
                log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(0).squeeze(-1)
            )
            top_values, top_indices = torch.topk(log_probs.squeeze(0), k=top_logprobs, dim=-1)

            supervision = []
            for row_idx, token_id in enumerate(assistant_token_ids):
                supervision.append(
                    {
                        "token":    tokenizer.decode([token_id], skip_special_tokens=False),
                        "token_id": token_id,
                        "logprob":  float(target_log_probs[row_idx]),
                        "source":   "hf_teacher",
                        "top_logprobs": [
                            {
                                "token":    tokenizer.decode([int(cid)], skip_special_tokens=False),
                                "token_id": int(cid),
                                "logprob":  float(clp),
                            }
                            for cid, clp in zip(
                                top_indices[row_idx].tolist(),
                                top_values[row_idx].tolist(),
                                strict=True,
                            )
                        ],
                    }
                )

            row_stub = {
                "slice_ids":         [slice_id],
                "system_prompt":     system_prompt,
                "messages":          [{"role": "user", "content": user_message}],
                "assistant_token_ids": assistant_token_ids,
            }
            rows.append(
                {
                    "record_id":           f"{slice_id}-turn-{idx}-{stable_hash(row_stub)[:8]}",
                    "slice_ids":           [slice_id],
                    "system_prompt":       system_prompt,
                    "messages":            [
                        {"role": "user",      "content": user_message},
                        {"role": "assistant", "content": assistant_text},
                    ],
                    "assistant_token_ids": assistant_token_ids,
                    "assistant_supervision": supervision,
                    "row_hash":            stable_hash(row_stub),
                    "metadata":            {"turn_index": idx, "logprob_source": "hf_teacher"},
                }
            )
            print(f"    Turn {idx + 1}/{len(pairs)}: {len(assistant_token_ids)} tokens")

    if not rows:
        raise ValueError("No valid supervision rows generated.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compress a conversation into a KV cartridge for session memory."
    )
    parser.add_argument("--conversation-path", required=True,
                        help="JSONL of conversation turns: {role, content} per line.")
    parser.add_argument("--output-path", required=True,
                        help="Where to save the compressed session cartridge .pt.")
    parser.add_argument("--cartridge-tokens", type=int, default=512,
                        help="KV token budget for the cartridge (default: 512).")
    parser.add_argument("--steps", type=int, default=60,
                        help="Distillation training steps (default: 60).")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    conversation_path = Path(args.conversation_path)
    output_path       = Path(args.output_path)

    turns: list[dict] = []
    with conversation_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                turns.append(json.loads(line))

    user_turns  = sum(1 for t in turns if t["role"] == "user")
    asst_turns  = sum(1 for t in turns if t["role"] == "assistant")
    print(f"Loaded conversation: {len(turns)} turns ({user_turns} user, {asst_turns} assistant)")

    # --- Phase 1: build teacher supervision ---
    t0 = time.perf_counter()
    print("\n[1/2] Building teacher supervision...")
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

    supervision_path = output_path.with_suffix(".supervision.jsonl")
    _build_supervision(
        turns=turns,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        output_path=supervision_path,
    )
    supervision_seconds = time.perf_counter() - t0
    print(f"  Done in {supervision_seconds:.1f}s → {supervision_path}")

    # Free model before train_cartridge loads its own copy (avoids double-load OOM).
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Phase 2: train cartridge ---
    t0 = time.perf_counter()
    print("\n[2/2] Training cartridge...")
    train_dir = output_path.parent / (output_path.stem + "_train")
    summary = train_cartridge(
        dataset_path=supervision_path,
        output_dir=train_dir,
        slice_id="conversation",
        device=args.device,
        cartridge_tokens=args.cartridge_tokens,
        learning_rate=args.learning_rate,
        steps=args.steps,
        num_frozen_tokens=1,
    )
    train_seconds = time.perf_counter() - t0

    # Copy best cartridge to the requested output path.
    import shutil
    shutil.copy2(summary["cartridge_path"], output_path)

    total = supervision_seconds + train_seconds
    print(
        f"\nDone in {total:.1f}s total\n"
        f"  Supervision : {supervision_seconds:.1f}s\n"
        f"  Training    : {train_seconds:.1f}s\n"
        f"  Best loss   : {summary['best_loss']:.4f}\n"
        f"  Cartridge   : {output_path}"
    )
    write_json(
        output_path.with_suffix(".summary.json"),
        {
            "conversation_turns": len(turns),
            "cartridge_tokens": args.cartridge_tokens,
            "steps": args.steps,
            "supervision_seconds": supervision_seconds,
            "train_seconds": train_seconds,
            "total_seconds": total,
            "best_loss": summary["best_loss"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

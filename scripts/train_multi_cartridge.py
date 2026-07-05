#!/usr/bin/env python3
"""Train multiple cartridges independently — Experiment 1 baseline.

Each cartridge sees only its own corpus during training (no distractor mixing).
The resulting cartridges are then loaded simultaneously in eval_multi_cartridge.py
to measure interference between independently-trained cartridges.

Efficiency: teacher model is loaded ONCE and used to build supervision for all
corpora before being freed. Each cartridge then trains sequentially.

Usage (on DGX — no vLLM needed):
    python scripts/train_multi_cartridge.py \
        --output-dir outputs/exp1 \
        --cartridge-tokens 512 \
        --steps 240 \
        --device cuda:0

    # Retrain only personal cartridges (supervision already built):
    python scripts/train_multi_cartridge.py \
        --output-dir outputs/exp1 \
        --names preferences research \
        --skip-supervision \
        --steps 240
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

from cartridges.config import DEFAULT_HF_MODEL_ID  # noqa: E402
from cartridges.data.common import stable_hash, write_json  # noqa: E402
from cartridges.train.cartridge import train_cartridge  # noqa: E402

DEFAULT_CORPORA: dict[str, str] = {
    "skills":      "data/skills.jsonl",
    "methods":     "data/methods.jsonl",
    "preferences": "data/preferences.jsonl",
    "research":    "data/research.jsonl",
}


def _clean(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    cleaned = cleaned.removeprefix("<think>").strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def _format_conversation(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        role = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{role}: {t['content'].strip()}")
    return "\n\n".join(lines)


def _build_supervision(
    *,
    name: str,
    turns: list[dict],
    model,
    tokenizer,
    device: str,
    output_path: Path,
    top_logprobs: int = 5,
) -> Path:
    """Build teacher supervision JSONL for a single conversation corpus."""
    conversation_text = _format_conversation(turns)
    system_prompt = (
        "Please answer the user's message using only the provided context.\n\n"
        f"<context>\n{conversation_text}\n</context>\n\n"
        "Follow the conversation naturally. "
        "Do not emit <think> tags or chain-of-thought."
    )

    pairs: list[tuple[str, str]] = []
    for i in range(len(turns) - 1):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            pairs.append((turns[i]["content"].strip(), turns[i + 1]["content"].strip()))

    if not pairs:
        raise ValueError(f"[{name}] No user/assistant pairs found.")

    print(f"  [{name}] Building supervision for {len(pairs)} turn pairs...")
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

            assistant_token_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
            eos = tokenizer.eos_token_id
            if eos is None:
                raise ValueError("Tokenizer must expose eos_token_id.")
            assistant_token_ids = assistant_token_ids + [int(eos)]
            if len(assistant_token_ids) <= 1:
                continue

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
                supervision.append({
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
                })

            row_stub = {
                "slice_ids":           [name],
                "system_prompt":       system_prompt,
                "messages":            [{"role": "user", "content": user_message}],
                "assistant_token_ids": assistant_token_ids,
            }
            rows.append({
                "record_id":             f"{name}-turn-{idx}-{stable_hash(row_stub)[:8]}",
                "slice_ids":             [name],
                "system_prompt":         system_prompt,
                "messages": [
                    {"role": "user",      "content": user_message},
                    {"role": "assistant", "content": assistant_text},
                ],
                "assistant_token_ids":   assistant_token_ids,
                "assistant_supervision": supervision,
                "row_hash":              stable_hash(row_stub),
                "metadata":              {"turn_index": idx, "logprob_source": "hf_teacher"},
            })
            print(f"    Turn {idx + 1}/{len(pairs)}: {len(assistant_token_ids)} tokens")

    if not rows:
        raise ValueError(f"[{name}] No valid supervision rows generated.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    print(f"  [{name}] {len(rows)} rows → {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train multiple cartridges independently (Experiment 1 baseline)."
    )
    parser.add_argument("--output-dir", default="outputs/exp1",
                        help="Directory for cartridges and supervision JSONLs.")
    parser.add_argument("--names", nargs="+", default=list(DEFAULT_CORPORA.keys()),
                        help="Corpora to train. Default: skills methods preferences research.")
    parser.add_argument("--cartridge-tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=240,
                        help="Training steps per cartridge (240 for <=50 rows).")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-supervision", action="store_true",
                        help="Skip teacher forward passes; assume supervision JSONLs exist.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.names:
        if name not in DEFAULT_CORPORA:
            raise ValueError(f"Unknown corpus '{name}'. Choose from {list(DEFAULT_CORPORA.keys())}")

    # Load conversation turns
    corpora: dict[str, list[dict]] = {}
    for name in args.names:
        path = ROOT / DEFAULT_CORPORA[name]
        turns = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    turns.append(json.loads(line))
        corpora[name] = turns
        print(f"Loaded {name}: {len(turns)} turns")

    supervision_paths: dict[str, Path] = {
        name: output_dir / f"{name}.supervision.jsonl" for name in args.names
    }

    # Phase 1: build supervision (one model load for all corpora)
    supervision_seconds = 0.0
    if not args.skip_supervision:
        t0 = time.perf_counter()
        print(f"\n[1/2] Building teacher supervision (model: {DEFAULT_HF_MODEL_ID})...")
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_HF_MODEL_ID,
            dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
            attn_implementation="sdpa",
        )
        model.to(args.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        for name in args.names:
            _build_supervision(
                name=name,
                turns=corpora[name],
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                output_path=supervision_paths[name],
            )

        supervision_seconds = time.perf_counter() - t0
        print(f"\nSupervision done in {supervision_seconds:.1f}s")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\n[1/2] Skipping supervision (--skip-supervision set).")
        for name in args.names:
            if not supervision_paths[name].exists():
                raise FileNotFoundError(
                    f"Supervision not found: {supervision_paths[name]}. "
                    "Run without --skip-supervision first."
                )

    # Phase 2: train each cartridge independently
    print(f"\n[2/2] Training {len(args.names)} cartridges independently...")
    train_summaries: dict[str, dict] = {}
    total_train_seconds = 0.0

    for name in args.names:
        t1 = time.perf_counter()
        print(f"\n── Training [{name}] ──")
        train_dir = output_dir / f"{name}_train"
        summary = train_cartridge(
            dataset_path=supervision_paths[name],
            output_dir=train_dir,
            slice_id=name,
            device=args.device,
            cartridge_tokens=args.cartridge_tokens,
            learning_rate=args.learning_rate,
            steps=args.steps,
            num_frozen_tokens=1,
        )
        elapsed = time.perf_counter() - t1
        total_train_seconds += elapsed

        # Copy best cartridge to output_dir/{name}_cartridge.pt (canonical path for eval)
        import shutil
        final_path = output_dir / f"{name}_cartridge.pt"
        shutil.copy2(summary["cartridge_path"], final_path)
        train_summaries[name] = {**summary, "elapsed_seconds": elapsed, "final_path": str(final_path)}
        print(f"  [{name}] {elapsed:.1f}s — best_loss={summary['best_loss']:.4f} → {final_path.name}")

    total = supervision_seconds + total_train_seconds
    print(f"\n{'='*60}")
    print(f"All done in {total:.1f}s")
    print(f"  Supervision : {supervision_seconds:.1f}s")
    print(f"  Training    : {total_train_seconds:.1f}s")
    print(f"\nPer-cartridge summary:")
    for name in args.names:
        s = train_summaries[name]
        print(f"  [{name:<8}] best_loss={s['best_loss']:.4f}  {s['elapsed_seconds']:.0f}s")
    print(f"\nCartridges in: {output_dir}/")
    print("  " + "  ".join(f"{n}_cartridge.pt" for n in args.names))
    print(f"\nNext: python scripts/eval_multi_cartridge.py --cartridge-dir {output_dir}")

    write_json(output_dir / "train_summary.json", {
        "names": args.names,
        "cartridge_tokens": args.cartridge_tokens,
        "steps": args.steps,
        "model": DEFAULT_HF_MODEL_ID,
        "supervision_seconds": supervision_seconds,
        "total_train_seconds": total_train_seconds,
        "total_seconds": total,
        "summaries": train_summaries,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

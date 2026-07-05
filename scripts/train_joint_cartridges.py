#!/usr/bin/env python3
"""Joint CAS-style cartridge training — Experiment 2.

At each training step:
  1. Pick a target cartridge (round-robin: skills → methods → preferences → research)
  2. Sample a supervision row from that cartridge's corpus
  3. Build a combined KV cache with ALL 4 cartridges concatenated in fixed order —
     but only the target cartridge's K/V tensors retain requires_grad=True
  4. Compute distillation loss on the target row and backprop into target only

This teaches each cartridge to answer its own questions correctly even when the
others are simultaneously present in the KV prefix.

--steps N means N gradient updates per cartridge (total: N × 4).
Use the same N as Experiment 1 for a fair comparison.

Usage (on DGX — no vLLM needed):
    # Reuse supervision from Experiment 1 (saves ~5 minutes):
    python scripts/train_joint_cartridges.py \\
        --output-dir outputs/exp2 \\
        --supervision-dir outputs/exp1 \\
        --steps 240 \\
        --device cuda:0

    # Build supervision from scratch:
    python scripts/train_joint_cartridges.py \\
        --output-dir outputs/exp2 \\
        --steps 240 \\
        --device cuda:0

    # Evaluate (reuse eval_multi_cartridge.py):
    python scripts/eval_multi_cartridge.py \\
        --cartridge-dir outputs/exp2 \\
        --mode both
"""

import argparse
import json
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.config import DEFAULT_HF_MODEL_ID  # noqa: E402
from cartridges.core import TrainableKVCartridge, initialize_from_prefix_text  # noqa: E402
from cartridges.data.common import stable_hash, write_json  # noqa: E402
from cartridges.train.cartridge import (  # noqa: E402
    TrainingExample,
    load_training_examples,
    _set_training_seed,
    _training_prompt,
    _sparse_distillation_loss,
    _compute_example_loss,
)

# Fixed cartridge order — must match eval_multi_cartridge.py joint eval order.
CORPUS_ORDER = ["zorbia", "blarvia", "user_a", "user_b"]

DEFAULT_CORPUS_PATHS: dict[str, str] = {
    "zorbia":  "data/zorbia.jsonl",
    "blarvia": "data/blarvia.jsonl",
    "user_a":  "data/user_a.jsonl",
    "user_b":  "data/user_b.jsonl",
}


# ─── Supervision building (same as train_multi_cartridge.py) ─────────────────

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

    print(f"  [{name}] Building supervision for {len(pairs)} pairs...")
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
                messages, tokenize=False, add_generation_prompt=True,
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
                [prompt_ids + assistant_token_ids[:-1]], device=device, dtype=torch.long,
            )
            logits = model(input_ids=input_ids).logits
            resp_start = len(prompt_ids)
            resp_end   = resp_start + len(assistant_token_ids) - 1
            log_probs = torch.log_softmax(logits[:, resp_start - 1 : resp_end, :], dim=-1)

            target_ids = torch.tensor([assistant_token_ids], device=device, dtype=torch.long)
            target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(0).squeeze(-1)
            top_values, top_indices = torch.topk(log_probs.squeeze(0), k=top_logprobs, dim=-1)

            supervision = []
            for row_idx, token_id in enumerate(assistant_token_ids):
                supervision.append({
                    "token":    tokenizer.decode([token_id], skip_special_tokens=False),
                    "token_id": token_id,
                    "logprob":  float(target_log_probs[row_idx]),
                    "source":   "hf_teacher",
                    "top_logprobs": [
                        {"token": tokenizer.decode([int(cid)], skip_special_tokens=False),
                         "token_id": int(cid), "logprob": float(clp)}
                        for cid, clp in zip(
                            top_indices[row_idx].tolist(), top_values[row_idx].tolist(), strict=True,
                        )
                    ],
                })

            row_stub = {"slice_ids": [name], "system_prompt": system_prompt,
                        "messages": [{"role": "user", "content": user_message}],
                        "assistant_token_ids": assistant_token_ids}
            rows.append({
                "record_id":             f"{name}-turn-{idx}-{stable_hash(row_stub)[:8]}",
                "slice_ids":             [name],
                "system_prompt":         system_prompt,
                "messages":              [{"role": "user", "content": user_message},
                                          {"role": "assistant", "content": assistant_text}],
                "assistant_token_ids":   assistant_token_ids,
                "assistant_supervision": supervision,
                "row_hash":              stable_hash(row_stub),
                "metadata":              {"turn_index": idx, "logprob_source": "hf_teacher"},
            })

    if not rows:
        raise ValueError(f"[{name}] No supervision rows generated.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"  [{name}] {len(rows)} rows → {output_path.name}")
    return output_path


# ─── Joint training ───────────────────────────────────────────────────────────

def _compute_joint_loss(
    *,
    model,
    tokenizer,
    cartridges: list[TrainableKVCartridge],
    target_idx: int,
    example: TrainingExample,
    device: str,
) -> torch.Tensor:
    """Distillation loss for target cartridge with all others loaded as frozen distractors.

    The combined cache has fixed order matching eval_multi_cartridge.py.
    Only target_idx cartridge's K/V tensors carry requires_grad; the rest are detached
    so backprop updates only the target.
    """
    num_layers = cartridges[0].num_layers
    combined_pkv = []
    for layer_idx in range(num_layers):
        keys, values = [], []
        for idx, cart in enumerate(cartridges):
            k, v = cart.layer(layer_idx)
            if idx != target_idx:
                k, v = k.detach(), v.detach()
            keys.append(k)
            values.append(v)
        combined_pkv.append((
            torch.cat(keys, dim=-2),
            torch.cat(values, dim=-2),
        ))
    joint_cache = DynamicCache(ddp_cache_data=tuple(combined_pkv), config=model.config)

    prompt_ids = _training_prompt(tokenizer, example.user_message).to(device)
    if len(example.assistant_token_ids) > 1:
        assistant_prefix = torch.tensor(
            [example.assistant_token_ids[:-1]], device=device, dtype=prompt_ids.dtype,
        )
        model_input = torch.cat([prompt_ids, assistant_prefix], dim=-1)
    else:
        model_input = prompt_ids

    outputs = model(input_ids=model_input, past_key_values=joint_cache, use_cache=False)
    target_len = len(example.assistant_token_ids)
    start_idx  = prompt_ids.shape[-1] - 1
    assistant_logits = outputs.logits[0, start_idx : start_idx + target_len, :]
    return _sparse_distillation_loss(assistant_logits, example.assistant_supervision)


def _oracle_val_loss(
    *,
    model,
    tokenizer,
    cartridge: TrainableKVCartridge,
    examples: list[TrainingExample],
    device: str,
) -> float:
    """Average oracle loss for one cartridge (single-cartridge prefix, no distractors)."""
    losses = []
    with torch.no_grad():
        for ex in examples:
            l = _compute_example_loss(
                model=model, tokenizer=tokenizer, cartridge=cartridge,
                example=ex, device=device,
            )
            losses.append(float(l.item()))
    return sum(losses) / len(losses)


def train_joint(
    *,
    all_examples: dict[str, list[TrainingExample]],
    output_dir: Path,
    train_names: list[str],
    device: str,
    cartridge_tokens: int,
    learning_rate: float,
    steps_per_cartridge: int,
    max_grad_norm: float = 1.0,
    seed: int = 0,
    validation_interval: int = 20,
    validation_examples: int = 16,
) -> dict[str, Any]:
    """Joint CAS-style training.

    train_names: cartridges to optimize. Any name in CORPUS_ORDER but NOT in
    train_names is loaded from output_dir as a frozen distractor — it is present
    in the KV prefix during every training step but never receives gradient updates.
    """
    _set_training_seed(seed)

    train_indices = {CORPUS_ORDER.index(n) for n in train_names}
    frozen_names  = [n for n in CORPUS_ORDER if n not in train_names]

    print(f"\nLoading model: {DEFAULT_HF_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_HF_MODEL_ID,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Build or load cartridges
    cartridges: list[TrainableKVCartridge] = []
    for i, name in enumerate(CORPUS_ORDER):
        if i in train_indices:
            # Fresh initialization from corpus prefix text
            examples = all_examples[name]
            cart = initialize_from_prefix_text(
                model=model, tokenizer=tokenizer,
                text=examples[0].system_prompt,
                num_tokens=cartridge_tokens,
                num_frozen_tokens=1,
            )
            cart.to(device)
            print(f"  [{name}] initialized fresh: {cart.num_tokens} slots  [trainable]")
        else:
            # Load existing trained cartridge as frozen distractor
            path = output_dir / f"{name}_cartridge.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Frozen cartridge not found: {path}. "
                    f"Run a full joint training pass first (without --names) "
                    f"or copy the cartridge from outputs/exp1/."
                )
            cart = TrainableKVCartridge.load(path, device=device)
            for param in cart.parameters():
                param.requires_grad_(False)
            print(f"  [{name}] loaded from {path.name}: {cart.num_tokens} slots  [frozen]")
        cartridges.append(cart)

    # Optimizers only for trainable cartridges
    optimizers: dict[int, AdamW] = {
        i: AdamW(cartridges[i].parameters(), lr=learning_rate)
        for i in train_indices
    }
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)

    val_subsets = {
        name: all_examples[name][:min(validation_examples, len(all_examples[name]))]
        for name in train_names
    }

    best_losses: dict[int, float] = {i: float("inf") for i in train_indices}
    best_states: dict[int, dict]  = {
        i: deepcopy(cartridges[i].state_dict()) for i in train_indices
    }
    loss_histories: dict[str, list[float]] = {name: [] for name in train_names}

    total_steps = steps_per_cartridge * len(train_names)
    print(
        f"\nJoint training: {steps_per_cartridge} steps/cartridge "
        f"× {len(train_names)} trainable = {total_steps} total steps"
    )
    if frozen_names:
        print(f"Frozen distractors: {frozen_names}")

    t0 = time.perf_counter()
    for total_step in range(total_steps):
        local_idx  = total_step % len(train_names)
        cart_step  = total_step // len(train_names)
        name       = train_names[local_idx]
        target_idx = CORPUS_ORDER.index(name)
        exs        = all_examples[name]
        example    = exs[cart_step % len(exs)]

        loss = _compute_joint_loss(
            model=model, tokenizer=tokenizer,
            cartridges=cartridges, target_idx=target_idx,
            example=example, device=device,
        )
        loss_val = float(loss.item())
        loss_histories[name].append(loss_val)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(cartridges[target_idx].parameters(), max_grad_norm)
        optimizers[target_idx].step()
        optimizers[target_idx].zero_grad(set_to_none=True)

        # Validate and checkpoint best (oracle loss)
        should_validate = (
            (cart_step + 1) == steps_per_cartridge
            or ((cart_step + 1) % validation_interval) == 0
        )
        if should_validate:
            val_loss = _oracle_val_loss(
                model=model, tokenizer=tokenizer,
                cartridge=cartridges[target_idx],
                examples=val_subsets[name],
                device=device,
            )
            if val_loss < best_losses[target_idx]:
                best_losses[target_idx] = val_loss
                best_states[target_idx] = {
                    k: v.detach().cpu().clone()
                    for k, v in cartridges[target_idx].state_dict().items()
                }

            # Log at the end of each full cycle through train_names
            if local_idx == len(train_names) - 1:
                elapsed = time.perf_counter() - t0
                losses_str = "  ".join(
                    f"{n}={loss_histories[n][-1]:.4f}" for n in train_names
                )
                print(f"  step {cart_step + 1:>4}/{steps_per_cartridge}  [{elapsed:.0f}s]  {losses_str}")

    elapsed_total = time.perf_counter() - t0

    # Save only the trained cartridges (frozen ones are unchanged in output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in train_indices:
        name = CORPUS_ORDER[i]
        cartridges[i].load_state_dict(best_states[i])
        out_path = output_dir / f"{name}_cartridge.pt"
        cartridges[i].save(out_path)

    print(f"\nDone in {elapsed_total:.1f}s")
    for i in train_indices:
        name = CORPUS_ORDER[i]
        print(f"  [{name}] best_oracle_loss={best_losses[i]:.4f}")
    print(f"\nCartridges saved to: {output_dir}/")
    print(f"Next: python scripts/eval_multi_cartridge.py --cartridge-dir {output_dir} --mode both")

    return {
        "train_names": train_names,
        "frozen_names": frozen_names,
        "steps_per_cartridge": steps_per_cartridge,
        "total_steps": total_steps,
        "elapsed_seconds": elapsed_total,
        "best_oracle_losses": {CORPUS_ORDER[i]: best_losses[i] for i in train_indices},
        "loss_histories": loss_histories,
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train cartridges jointly with CAS-style distractor supervision (Experiment 2)."
    )
    parser.add_argument("--output-dir", default="outputs/exp2")
    parser.add_argument("--names", nargs="+", default=CORPUS_ORDER,
                        help="Cartridges to train. Others are loaded from --output-dir as "
                             "frozen distractors. Default: all 4.")
    parser.add_argument("--supervision-dir", default=None,
                        help="Directory containing {name}.supervision.jsonl files. "
                             "Defaults to --output-dir. Point to outputs/exp1 to reuse "
                             "existing supervision and skip the teacher forward passes.")
    parser.add_argument("--cartridge-tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=240,
                        help="Gradient updates per cartridge (total = steps × 4).")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--validation-interval", type=int, default=20,
                        help="Validate every N steps per cartridge.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    supervision_dir = Path(args.supervision_dir) if args.supervision_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.names:
        if name not in CORPUS_ORDER:
            raise ValueError(f"Unknown corpus '{name}'. Choose from {CORPUS_ORDER}")

    # Load or build supervision (only needed for cartridges being trained)
    supervision_paths = {name: supervision_dir / f"{name}.supervision.jsonl" for name in CORPUS_ORDER}
    missing = [name for name in args.names if not supervision_paths[name].exists()]

    if missing:
        print(f"Supervision not found for: {missing}. Building now...")
        corpora: dict[str, list[dict]] = {}
        for name in missing:
            path = ROOT / DEFAULT_CORPUS_PATHS[name]
            turns = []
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        turns.append(json.loads(line))
            corpora[name] = turns

        t0 = time.perf_counter()
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

        output_dir.mkdir(parents=True, exist_ok=True)
        for name in missing:
            _build_supervision(
                name=name, turns=corpora[name], model=model, tokenizer=tokenizer,
                device=args.device, output_path=output_dir / f"{name}.supervision.jsonl",
            )
            supervision_paths[name] = output_dir / f"{name}.supervision.jsonl"

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Supervision built in {time.perf_counter() - t0:.1f}s")
    else:
        print(f"Reusing supervision from: {supervision_dir}")

    # Load training examples (only for cartridges being trained)
    all_examples: dict[str, list[TrainingExample]] = {}
    for name in args.names:
        raw = load_training_examples(supervision_paths[name])
        all_examples[name] = [ex for ex in raw if ex.slice_id == name]
        print(f"  [{name}] {len(all_examples[name])} supervision rows")

    # Run joint training
    summary = train_joint(
        all_examples=all_examples,
        output_dir=output_dir,
        train_names=args.names,
        device=args.device,
        cartridge_tokens=args.cartridge_tokens,
        learning_rate=args.learning_rate,
        steps_per_cartridge=args.steps,
        validation_interval=args.validation_interval,
    )

    write_json(output_dir / "train_joint_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

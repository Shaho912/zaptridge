#!/usr/bin/env python3
"""AblationCartridge experiment: replicate Diaz 2025 (arxiv 2508.17032).

Train two cartridges on different corpora with the SAME init text, then swap
their keys to test whether keys are universal routing signals independent of
content.

Theory (Diaz): keys converge to stable corpus-agnostic routing positions while
values carry the actual compressed knowledge. If true, Z_AB (B's keys + A's
values) should substantially outperform a no-cartridge baseline on A's task.

Three conditions:
  Baseline   — no cartridge (floor)
  Oracle     — Cartridge A alone on A's questions (ceiling)
  Ablation   — B's keys + A's values on A's questions

Also tracks per-interval cosine similarity between consecutive key and value
snapshots during training, replicating Figure 2 from Diaz 2025.

Usage:
    # Full run (trains both cartridges):
    python scripts/run_ablation_cartridge.py \\
        --dataset-a data/zaptridge_convo.jsonl \\
        --dataset-b data/skills.jsonl \\
        --init-text data/wikipedia_india/data.txt \\
        --eval-questions data/zaptridge_convo_eval.json \\
        --output-dir outputs/ablation_exp \\
        --steps 512 \\
        --cartridge-tokens 1024 \\
        --device cuda:0

    # Skip training, just run ablation on existing cartridges:
    python scripts/run_ablation_cartridge.py \\
        --dataset-a data/zaptridge_convo.jsonl \\
        --dataset-b data/skills.jsonl \\
        --init-text data/wikipedia_india/data.txt \\
        --eval-questions data/zaptridge_convo_eval.json \\
        --output-dir outputs/ablation_exp \\
        --skip-train

For LongHealth patient data, run prepare_longhealth.py + train_paper_cartridges.py
first to generate the JSONL and eval questions, then pass them here.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.config import DEFAULT_HF_MODEL_ID    # noqa: E402
from cartridges.core import TrainableKVCartridge      # noqa: E402
from cartridges.data.common import write_json         # noqa: E402
from cartridges.train.cartridge import train_cartridge  # noqa: E402


def _construct_ablation(
    cart_a: TrainableKVCartridge,
    cart_b: TrainableKVCartridge,
) -> TrainableKVCartridge:
    """Build AblationCartridge: cart_b's keys + cart_a's values, per Diaz 2025."""
    if cart_a.num_layers != cart_b.num_layers:
        raise ValueError(
            f"Layer count mismatch: A={cart_a.num_layers}, B={cart_b.num_layers}. "
            "Both cartridges must be trained on the same model."
        )
    if cart_a.num_tokens != cart_b.num_tokens:
        raise ValueError(
            f"Token count mismatch: A={cart_a.num_tokens}, B={cart_b.num_tokens}. "
            "Both cartridges must use the same --cartridge-tokens."
        )
    keys = [cart_b.layer(i)[0].detach().clone() for i in range(cart_b.num_layers)]
    values = [cart_a.layer(i)[1].detach().clone() for i in range(cart_a.num_layers)]
    return TrainableKVCartridge(keys=keys, values=values, num_frozen_tokens=0)


def _generate(
    model,
    tokenizer,
    cache,
    question: str,
    device: str,
    max_new_tokens: int = 80,
) -> str:
    """Generate an answer using the provided KV cache as prefix (cache may be None)."""
    messages = [{"role": "user", "content": f"/no_think\n{question}"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        past = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        generated: list[int] = []
        for _ in range(max_new_tokens):
            tid = int(next_token.item())
            generated.append(tid)
            if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
                break
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip incomplete think block (model ran out of tokens mid-thought)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"</think>", "", text).strip()
    return text


def _run_eval(
    *,
    label: str,
    questions: list[list[str]],
    model,
    tokenizer,
    cache_factory,
    device: str,
    max_new_tokens: int = 80,
) -> tuple[int, int, list[dict]]:
    """Run all questions against a cache factory. Returns (hits, total, per-Q details)."""
    hits = 0
    details = []
    for q, expected in questions:
        cache = cache_factory()
        answer = _generate(model, tokenizer, cache, q, device, max_new_tokens)
        correct = expected.lower() in answer.lower()
        hits += int(correct)
        mark = "✓" if correct else "✗"
        print(f"  {mark} {q[:80]}")
        print(f"       A: {answer[:120]}  [expected: {expected!r}]")
        details.append({"question": q, "expected": expected, "answer": answer, "correct": correct})
    return hits, len(questions), details


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AblationCartridge experiment (Diaz 2025 replication)"
    )
    parser.add_argument("--dataset-a", required=True,
                        help="Training JSONL for corpus A (the corpus we evaluate on)")
    parser.add_argument("--dataset-b", required=True,
                        help="Training JSONL for corpus B (semantically different domain)")
    parser.add_argument("--init-text", required=True,
                        help="Path to plain-text file used as shared cartridge initializer for both A and B")
    parser.add_argument("--eval-questions", required=True,
                        help="JSON file with [[question, expected_substring], ...] pairs for corpus A")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=512,
                        help="Optimizer steps per cartridge (paper: 512, fast test: 60)")
    parser.add_argument("--cartridge-tokens", type=int, default=1024,
                        help="KV slots per cartridge (paper: 2048, our default: 1024)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--movement-log-interval", type=int, default=50,
                        help="Log key/value cosine similarity every N steps (set to 0 to disable)")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; load existing cart_a.pt and cart_b.pt from --output-dir")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cart_a_path = out / "cart_a.pt"
    cart_b_path = out / "cart_b.pt"

    init_text = Path(args.init_text).read_text(encoding="utf-8")
    questions: list[list[str]] = json.loads(Path(args.eval_questions).read_text(encoding="utf-8"))
    log_interval = args.movement_log_interval if args.movement_log_interval > 0 else None

    if not args.skip_train:
        print("=" * 70)
        print(f"[1/2] Training Cartridge A")
        print(f"      corpus:   {args.dataset_a}")
        print(f"      init:     {args.init_text} ({len(init_text):,} chars)")
        print(f"      steps:    {args.steps}  tokens: {args.cartridge_tokens}")
        print("=" * 70)
        summary_a = train_cartridge(
            dataset_path=args.dataset_a,
            output_dir=out / "train_a",
            device=args.device,
            cartridge_tokens=args.cartridge_tokens,
            steps=args.steps,
            initialization_text=init_text,
            movement_log_interval=log_interval,
        )
        shutil.copy(summary_a["cartridge_path"], cart_a_path)
        if log_interval:
            write_json(out / "movement_a.json", summary_a.get("movement_log", []))
        print(f"[A] best_loss={summary_a['best_loss']:.4f} at step {summary_a['best_step']}")

        print("\n" + "=" * 70)
        print(f"[2/2] Training Cartridge B")
        print(f"      corpus:   {args.dataset_b}")
        print(f"      init:     (same — {args.init_text})")
        print("=" * 70)
        summary_b = train_cartridge(
            dataset_path=args.dataset_b,
            output_dir=out / "train_b",
            device=args.device,
            cartridge_tokens=args.cartridge_tokens,
            steps=args.steps,
            initialization_text=init_text,
            movement_log_interval=log_interval,
        )
        shutil.copy(summary_b["cartridge_path"], cart_b_path)
        if log_interval:
            write_json(out / "movement_b.json", summary_b.get("movement_log", []))
        print(f"[B] best_loss={summary_b['best_loss']:.4f} at step {summary_b['best_step']}")
    else:
        for p in (cart_a_path, cart_b_path):
            if not p.exists():
                raise FileNotFoundError(f"--skip-train requires {p} to exist")
        print(f"[skip-train] Loading existing cartridges from {out}")

    # ─── Construct AblationCartridge ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Constructing AblationCartridge (B's keys + A's values)...")
    cart_a = TrainableKVCartridge.load(cart_a_path, device=args.device)
    cart_b = TrainableKVCartridge.load(cart_b_path, device=args.device)
    ablation = _construct_ablation(cart_a, cart_b)
    ablation.save(out / "ablation.pt")
    print(f"  cart_a:   {cart_a.num_tokens} slots × {cart_a.num_layers} layers")
    print(f"  cart_b:   {cart_b.num_tokens} slots × {cart_b.num_layers} layers")
    print(f"  ablation: {ablation.num_tokens} slots (B keys + A values)")

    # ─── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading {DEFAULT_HF_MODEL_ID} for eval...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_HF_MODEL_ID,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(args.device)
    model.eval()

    results: dict = {}

    # ─── Baseline (no cartridge) ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"BASELINE — no cartridge  ({len(questions)} questions)")
    print("=" * 70)
    hits, total, details = _run_eval(
        label="baseline",
        questions=questions,
        model=model,
        tokenizer=tokenizer,
        cache_factory=lambda: None,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    results["baseline"] = {"hits": hits, "total": total}
    print(f"  → {hits}/{total}")

    # ─── Oracle (Cartridge A alone) ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"ORACLE — Cartridge A alone  ({len(questions)} questions)")
    print("=" * 70)
    hits, total, details = _run_eval(
        label="oracle",
        questions=questions,
        model=model,
        tokenizer=tokenizer,
        cache_factory=lambda: cart_a.as_cache(model.config),
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    results["oracle"] = {"hits": hits, "total": total, "details": details}
    print(f"  → {hits}/{total}")

    # ─── Ablation (B's keys + A's values) ───────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"ABLATION — B keys + A values  ({len(questions)} questions)")
    print("=" * 70)
    hits, total, details = _run_eval(
        label="ablation",
        questions=questions,
        model=model,
        tokenizer=tokenizer,
        cache_factory=lambda: ablation.as_cache(model.config),
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    results["ablation"] = {"hits": hits, "total": total, "details": details}
    print(f"  → {hits}/{total}")

    # ─── Summary ─────────────────────────────────────────────────────────────────
    b_h, b_t = results["baseline"]["hits"], results["baseline"]["total"]
    o_h, o_t = results["oracle"]["hits"], results["oracle"]["total"]
    a_h, a_t = results["ablation"]["hits"], results["ablation"]["total"]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Condition':<22}  {'Score':>8}  {'vs Oracle':>10}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*10}")
    print(f"  {'Baseline (no cart)':<22}  {b_h}/{b_t}       {b_h - o_h:+d}")
    print(f"  {'Oracle (A alone)':<22}  {o_h}/{o_t}       {'—':>4}")
    print(f"  {'Ablation (B→K + A→V)':<22}  {a_h}/{a_t}       {a_h - o_h:+d}")

    ablation_vs_baseline = a_h - b_h
    oracle_drop = o_h - a_h
    if ablation_vs_baseline > 0:
        print(f"\n  Keys ARE transferable: ablation beats baseline by {ablation_vs_baseline} question(s).")
    else:
        print(f"\n  Keys are NOT transferable for this model/corpus pair.")
    if oracle_drop > 0:
        print(f"  Oracle→Ablation drop: {oracle_drop} question(s) lost when using B's keys.")
    elif oracle_drop == 0:
        print(f"  No oracle→ablation drop — key swap causes zero accuracy loss.")
    else:
        print(f"  Ablation EXCEEDS oracle by {-oracle_drop} question(s) (B's keys help A's retrieval).")

    results["config"] = {
        "dataset_a": args.dataset_a,
        "dataset_b": args.dataset_b,
        "init_text": args.init_text,
        "steps": args.steps,
        "cartridge_tokens": args.cartridge_tokens,
        "model": DEFAULT_HF_MODEL_ID,
    }
    write_json(out / "results.json", results)
    print(f"\nResults → {out}/results.json")
    if log_interval and not args.skip_train:
        print(f"Movement logs → {out}/movement_a.json  {out}/movement_b.json")
        _print_movement_summary(out)
    return 0


def _print_movement_summary(out: Path) -> None:
    """Print a compact key vs value cosim table from saved movement logs."""
    for label in ("a", "b"):
        path = out / f"movement_{label}.json"
        if not path.exists():
            continue
        log = json.loads(path.read_text(encoding="utf-8"))
        if not log:
            continue
        print(f"\nKey vs Value movement — Cartridge {label.upper()}")
        print(f"  {'Step':>6}  {'Key cosim':>10}  {'Val cosim':>10}  {'Key moves less?':>15}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*15}")
        for entry in log:
            k = entry["mean_key_cosim"]
            v = entry["mean_val_cosim"]
            less = "YES" if k > v else "no"
            print(f"  {entry['step']:>6}  {k:>10.6f}  {v:>10.6f}  {less:>15}")


if __name__ == "__main__":
    raise SystemExit(main())

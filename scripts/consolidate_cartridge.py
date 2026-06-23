#!/usr/bin/env python3
"""Consolidate multiple supervision JSONLs into a single fresh cartridge.

This is the "weekly overnight" step in the delta-slots + consolidation pipeline:
  1. Daily: add delta slots for each day's conversation (fast, precise)
  2. Weekly: run this script on all accumulated supervision files to retrain
     a fixed-size cartridge from scratch, then reset delta slots to zero.

Training from scratch (no --finetune-from) means the optimizer has full freedom
to distribute all facts across slots without fighting prior geometry.

Usage:
    python scripts/consolidate_cartridge.py \
        --supervision-path outputs/conversation/zorbia.supervision.jsonl \
        --supervision-path outputs/conversation/blarvia.supervision.jsonl \
        --output-path outputs/conversation/consolidated.pt \
        --cartridge-tokens 512 \
        --steps 120
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.data.common import write_json   # noqa: E402
from cartridges.train.cartridge import train_cartridge  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrain a fixed-size cartridge from scratch on multiple supervision files."
    )
    parser.add_argument("--supervision-path", action="append", required=True, dest="supervision_paths",
                        help="Supervision JSONL to include. Repeat for each file.")
    parser.add_argument("--output-path", required=True,
                        help="Where to save the consolidated cartridge .pt.")
    parser.add_argument("--cartridge-tokens", type=int, default=512,
                        help="KV token budget (default: 512).")
    parser.add_argument("--steps", type=int, default=120,
                        help="Training steps (default: 120 — more than single-conversation runs "
                             "because the dataset is larger).")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_path = Path(args.output_path)

    # Concatenate all supervision rows into one file.
    combined_path = output_path.with_suffix(".combined_supervision.jsonl")
    total_rows = 0
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", encoding="utf-8") as out:
        for sup_path in args.supervision_paths:
            rows = [l for l in Path(sup_path).read_text(encoding="utf-8").splitlines() if l.strip()]
            for row in rows:
                out.write(row + "\n")
            total_rows += len(rows)
            print(f"  Loaded {len(rows):>3} rows from {sup_path}")
    print(f"  Combined: {total_rows} rows total → {combined_path}\n")

    # Train from scratch on the combined dataset.
    t0 = time.perf_counter()
    print(f"Training fresh {args.cartridge_tokens}-slot cartridge for {args.steps} steps...")
    train_dir = output_path.parent / (output_path.stem + "_train")
    summary = train_cartridge(
        dataset_path=combined_path,
        output_dir=train_dir,
        slice_id="conversation",
        device=args.device,
        cartridge_tokens=args.cartridge_tokens,
        learning_rate=args.learning_rate,
        steps=args.steps,
        num_frozen_tokens=1,
        initial_cartridge_path=None,  # always fresh — no prior geometry to fight
    )
    elapsed = time.perf_counter() - t0

    import shutil
    shutil.copy2(summary["cartridge_path"], output_path)

    print(
        f"\nDone in {elapsed:.1f}s\n"
        f"  Best loss : {summary['best_loss']:.4f}\n"
        f"  Cartridge : {output_path}"
    )
    write_json(
        output_path.with_suffix(".summary.json"),
        {
            "supervision_paths": args.supervision_paths,
            "total_rows": total_rows,
            "cartridge_tokens": args.cartridge_tokens,
            "steps": args.steps,
            "train_seconds": elapsed,
            "best_loss": summary["best_loss"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

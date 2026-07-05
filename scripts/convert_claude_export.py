#!/usr/bin/env python3
"""Convert a Claude.ai conversation export JSON to pipeline JSONL format.

Claude.ai exports a zip of JSON files (one per conversation). This script
takes a single conversation JSON and converts it to the role/content JSONL
format expected by train_multi_cartridge.py.

Usage:
    python scripts/convert_claude_export.py conversation.json \
        --output data/my_convo.jsonl

    # Keep only the last 60 turns (30 exchanges):
    python scripts/convert_claude_export.py conversation.json \
        --output data/my_convo.jsonl \
        --max-turns 60

    # Preview what will be written (no file output):
    python scripts/convert_claude_export.py conversation.json --dry-run
"""

import argparse
import json
import sys
from pathlib import Path


def _extract_text(message: dict) -> str:
    """Extract plain text from a message, handling both string and block formats."""
    content = message.get("text") or message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    # List of content blocks (e.g. tool use interleaved with text)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                # Skip tool_use, tool_result, thinking blocks
        return "\n\n".join(p.strip() for p in parts if p.strip())

    return ""


def _sender_to_role(sender: str) -> str:
    if sender in ("human", "user"):
        return "user"
    if sender in ("assistant", "claude"):
        return "assistant"
    return sender


def convert(path: Path, max_turns: int | None = None) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    # Support both top-level list and dict with chat_messages key
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict):
        messages = (
            data.get("chat_messages")
            or data.get("messages")
            or data.get("conversation", {}).get("messages", [])
        )
        if messages is None:
            raise ValueError(
                f"Could not find messages list in {path}. "
                "Keys found: " + str(list(data.keys()))
            )
    else:
        raise ValueError(f"Unexpected top-level JSON type: {type(data)}")

    turns = []
    for msg in messages:
        sender = msg.get("sender") or msg.get("role", "")
        role = _sender_to_role(sender)
        if role not in ("user", "assistant"):
            continue
        text = _extract_text(msg)
        if not text:
            continue
        turns.append({"role": role, "content": text})

    # Drop leading assistant turns (pipeline expects user first)
    while turns and turns[0]["role"] == "assistant":
        turns.pop(0)

    # Enforce alternating user/assistant — drop consecutive same-role turns
    clean: list[dict] = []
    for turn in turns:
        if clean and clean[-1]["role"] == turn["role"]:
            # Merge consecutive same-role turns (e.g. two human messages)
            clean[-1]["content"] += "\n\n" + turn["content"]
        else:
            clean.append(turn)

    if max_turns is not None:
        clean = clean[-max_turns:]
        # Ensure we still start with a user turn after truncation
        while clean and clean[0]["role"] == "assistant":
            clean.pop(0)

    return clean


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Claude.ai conversation export JSON to pipeline JSONL."
    )
    parser.add_argument("input", help="Path to the Claude.ai conversation JSON file.")
    parser.add_argument("--output", "-o", help="Output JSONL path.")
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help="Keep only the last N turns. Useful for very long conversations."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print turns to stdout without writing a file."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.", file=sys.stderr)
        return 1

    turns = convert(input_path, max_turns=args.max_turns)

    if not turns:
        print("No turns extracted. Check the JSON format.", file=sys.stderr)
        return 1

    user_count = sum(1 for t in turns if t["role"] == "user")
    asst_count  = sum(1 for t in turns if t["role"] == "assistant")
    print(f"Extracted {len(turns)} turns ({user_count} user, {asst_count} assistant)")

    avg_len = sum(len(t["content"]) for t in turns) / len(turns)
    print(f"Average turn length: {avg_len:.0f} chars")

    if args.dry_run:
        print("\n--- Preview (first 3 turns) ---")
        for t in turns[:3]:
            preview = t["content"][:200].replace("\n", " ")
            print(f"[{t['role']}] {preview}...")
        return 0

    if not args.output:
        print("Error: --output required unless --dry-run is set.", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for turn in turns:
            fh.write(json.dumps(turn, ensure_ascii=False) + "\n")

    print(f"Written to: {output_path}")
    print(f"\nNext: add eval questions for this corpus in eval_multi_cartridge.py,")
    print(f"then run: python scripts/train_multi_cartridge.py --names {output_path.stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

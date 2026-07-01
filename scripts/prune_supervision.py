#!/usr/bin/env python3
"""Episodic clustering-based supervision JSONL pruning.

Runs before weekly consolidation to cap the supervision dataset at a fixed row
budget while preserving coverage across topics (via clustering) and prioritising
rows the current cartridge hasn't learned yet (via importance scoring).

Pipeline
--------
1. Load all supervision JSONLs and concatenate into one flat list.
2. Embed each row's question with sentence-transformers/all-MiniLM-L6-v2.
3. K-means cluster the embeddings into k clusters (default 20).
4. Allocate total_budget / k rows per cluster; redistribute surplus from small clusters.
5. Within each cluster, rank rows by cartridge difficulty (rows where cartridge
   fails ranked higher). Requires --cartridge-path; falls back to random order.
6. Keep the top-allocated rows per cluster, shuffle survivors, write output JSONL.

Usage
-----
# Without importance scoring (random within-cluster selection):
python scripts/prune_supervision.py \
    --supervision-path outputs/conversation/zorbia.supervision.jsonl \
    --supervision-path outputs/conversation/blarvia.supervision.jsonl \
    --output-path outputs/conversation/pruned.supervision.jsonl \
    --total-budget 500 \
    --clusters 20

# With cartridge importance scoring:
python scripts/prune_supervision.py \
    --supervision-path outputs/conversation/zorbia.supervision.jsonl \
    --supervision-path outputs/conversation/blarvia.supervision.jsonl \
    --output-path outputs/conversation/pruned.supervision.jsonl \
    --total-budget 500 \
    --clusters 20 \
    --cartridge-path outputs/conversation/consolidated.pt \
    --device cuda:0
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        lines = Path(p).read_text(encoding="utf-8").splitlines()
        file_rows = [json.loads(l) for l in lines if l.strip()]
        print(f"  {len(file_rows):>4} rows  ←  {p}")
        rows.extend(file_rows)
    return rows


def _extract_question(row: dict) -> str:
    """Pull the user question out of a supervision row."""
    msgs = row.get("messages", [])
    for m in msgs:
        if m.get("role") == "user":
            content = m["content"]
            # Strip the /no_think prefix added by compress_conversation.py
            return content.removeprefix("/no_think\n").strip()
    return row.get("system_prompt", "")[:200]


def _extract_answer(row: dict) -> str:
    msgs = row.get("messages", [])
    for m in msgs:
        if m.get("role") == "assistant":
            return m["content"].strip()
    return ""


def _normalise(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Cartridge inference for importance scoring
# ---------------------------------------------------------------------------

def _score_rows_with_cartridge(
    rows: list[dict],
    cartridge_path: str,
    device: str,
) -> list[float]:
    """Return per-row difficulty score: 1.0 = cartridge fails, 0.0 = correct."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cartridges.config import DEFAULT_HF_MODEL_ID
    from cartridges.core import TrainableKVCartridge

    print(f"\nLoading model for importance scoring: {DEFAULT_HF_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_HF_MODEL_ID,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()

    print(f"Loading cartridge: {cartridge_path}")
    cartridge = TrainableKVCartridge.load(cartridge_path, device=device)
    print(f"  {cartridge.num_tokens} slots, {cartridge.num_layers} layers")

    eos_token_id = tokenizer.eos_token_id
    scores: list[float] = []

    with torch.inference_mode():
        for i, row in enumerate(rows):
            question = _extract_question(row)
            ground_truth = _normalise(_extract_answer(row))

            messages = [{"role": "user", "content": f"/no_think\n{question}"}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            input_ids = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(device)

            outputs = model(
                input_ids=input_ids,
                past_key_values=cartridge.as_cache(model.config),
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            next_tok = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated: list[int] = []
            for _ in range(80):
                tok_id = int(next_tok.item())
                generated.append(tok_id)
                if eos_token_id is not None and tok_id == eos_token_id:
                    break
                outputs = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                next_tok = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            prediction = _normalise(tokenizer.decode(generated, skip_special_tokens=True))
            correct = ground_truth in prediction or prediction in ground_truth
            scores.append(0.0 if correct else 1.0)

            if (i + 1) % 10 == 0 or (i + 1) == len(rows):
                n_hard = sum(1 for s in scores if s > 0)
                print(f"  Scored {i + 1}/{len(rows)} — hard so far: {n_hard}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores


# ---------------------------------------------------------------------------
# Clustering + budget allocation
# ---------------------------------------------------------------------------

def _embed_questions(questions: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print("Embedding questions with all-MiniLM-L6-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(questions, batch_size=64, show_progress_bar=True,
                               convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def _kmeans_cluster(embeddings: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    k = min(k, len(embeddings))
    print(f"K-means clustering into k={k} clusters...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels


def _allocate_budget(cluster_sizes: list[int], total_budget: int) -> list[int]:
    """Uniform allocation with redistribution from under-filled clusters."""
    k = len(cluster_sizes)
    per_cluster = total_budget // k
    allocations = [min(sz, per_cluster) for sz in cluster_sizes]
    remaining = total_budget - sum(allocations)
    # Redistribute surplus to clusters that can absorb more
    if remaining > 0:
        for i in sorted(range(k), key=lambda i: cluster_sizes[i] - allocations[i], reverse=True):
            can_absorb = cluster_sizes[i] - allocations[i]
            add = min(can_absorb, remaining)
            allocations[i] += add
            remaining -= add
            if remaining == 0:
                break
    return allocations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune accumulated supervision JSONLs via episodic clustering."
    )
    parser.add_argument("--supervision-path", action="append", required=True,
                        dest="supervision_paths",
                        help="Supervision JSONL to include. Repeat for each file.")
    parser.add_argument("--output-path", required=True,
                        help="Where to write the pruned supervision JSONL.")
    parser.add_argument("--total-budget", type=int, default=500,
                        help="Max rows in the output (default: 500).")
    parser.add_argument("--clusters", type=int, default=20,
                        help="Number of K-means clusters (default: 20).")
    parser.add_argument("--cartridge-path", default=None,
                        help="Existing cartridge .pt to use for importance scoring. "
                             "If omitted, within-cluster selection is random.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load all rows
    print("Loading supervision files...")
    rows = _load_rows(args.supervision_paths)
    print(f"  Total: {len(rows)} rows\n")

    if len(rows) <= args.total_budget:
        print(f"Row count ({len(rows)}) ≤ budget ({args.total_budget}) — no pruning needed.")
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"Wrote {len(rows)} rows → {output_path}")
        return 0

    # 2. Embed questions
    questions = [_extract_question(r) for r in rows]
    embeddings = _embed_questions(questions)

    # 3. Cluster
    labels = _kmeans_cluster(embeddings, args.clusters)
    k_actual = int(labels.max()) + 1

    # 4. Importance scores (cartridge difficulty) or uniform
    if args.cartridge_path is not None:
        importance = _score_rows_with_cartridge(rows, args.cartridge_path, args.device)
    else:
        print("No cartridge provided — using random within-cluster selection.")
        importance = [random.random() for _ in rows]

    # 5. Group rows by cluster
    clusters: list[list[tuple[int, float]]] = [[] for _ in range(k_actual)]
    for idx, (label, score) in enumerate(zip(labels, importance)):
        clusters[label].append((idx, score))

    cluster_sizes = [len(c) for c in clusters]
    allocations = _allocate_budget(cluster_sizes, args.total_budget)

    print(f"\nCluster summary (k={k_actual}, budget={args.total_budget}):")
    for ci, (sz, alloc) in enumerate(zip(cluster_sizes, allocations)):
        print(f"  cluster {ci:>2}: {sz:>4} rows  →  keep {alloc}")

    # 6. Within-cluster selection: hard rows first (score=1.0 before score=0.0)
    selected_indices: list[int] = []
    for ci, (cluster_members, alloc) in enumerate(zip(clusters, allocations)):
        ranked = sorted(cluster_members, key=lambda x: x[1], reverse=True)
        selected_indices.extend(idx for idx, _ in ranked[:alloc])

    # Shuffle survivors and write
    random.shuffle(selected_indices)
    survivors = [rows[i] for i in selected_indices]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in survivors:
            fh.write(json.dumps(row) + "\n")

    n_hard = sum(1 for i in selected_indices if importance[i] > 0.5)
    print(f"\nWrote {len(survivors)} rows → {output_path}")
    print(f"  Hard rows kept (cartridge failed): {n_hard}/{len(survivors)}")
    print(f"  Easy rows kept (cartridge correct): {len(survivors) - n_hard}/{len(survivors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

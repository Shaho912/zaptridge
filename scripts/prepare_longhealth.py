#!/usr/bin/env python3
"""Download and prepare LongHealth patient records as cartridge training corpora.

Downloads benchmark_v5.json from kbressem/LongHealth, extracts patient records,
saves each as a plain-text corpus in data/lh_p{N}/, and writes eval questions
derived from the benchmark's MCQ ground-truth answers.

These eval questions (from the benchmark) are more reliable than vLLM-generated
bootstrap eval questions — use them with --eval-questions-dir in
train_paper_cartridges.py.

Usage:
    python scripts/prepare_longhealth.py
    python scripts/prepare_longhealth.py --patients 1 2 3 4
    python scripts/prepare_longhealth.py --list-patients  # show all patient names
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_HF_URL = (
    "https://huggingface.co/datasets/kbressem/LongHealth"
    "/resolve/main/data/benchmark_v5.json"
)
_CACHE_PATH = ROOT / "data" / "longhealth_benchmark_v5.json"


def _download_json(url: str) -> dict:
    import urllib.request
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def _load_benchmark() -> dict:
    if _CACHE_PATH.exists():
        print(f"Using cached benchmark: {_CACHE_PATH}")
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    data = _download_json(_HF_URL)
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(data), encoding="utf-8")
    print(f"Cached to: {_CACHE_PATH}")
    return data


def _patient_text(patient: dict) -> str:
    texts = patient["texts"]
    parts = [texts[k] for k in sorted(texts.keys())]
    return "\n\n---\n\n".join(p.strip() for p in parts if p.strip())


def _make_eval_questions(patient: dict, n: int = 8) -> list[list[str]]:
    """Convert MCQ questions to [question_with_options, expected_substring] format.

    Presents all 5 options inline so the model can pick from them (necessary for
    "which is NOT..." style questions). Expected substring = full text of correct
    option (lowercased).
    """
    result = []
    for q in patient["questions"][:n]:
        qtext = q["question"]
        options = {
            "A": q.get("answer_a", ""),
            "B": q.get("answer_b", ""),
            "C": q.get("answer_c", ""),
            "D": q.get("answer_d", ""),
            "E": q.get("answer_e", ""),
        }
        # 'correct' is a letter like 'a', 'b', 'c', 'd', 'e'
        correct_letter = str(q.get("correct", "a")).upper()
        correct_text = options.get(correct_letter, "")
        if not correct_text:
            continue

        opts_lines = "\n".join(f"{k}) {v}" for k, v in options.items() if v)
        full_q = f"{qtext}\n{opts_lines}"
        result.append([full_q, correct_text.lower()])

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare LongHealth patient records as cartridge training corpora."
    )
    parser.add_argument(
        "--patients", nargs="+", type=int, default=[1, 2, 3, 4],
        help="Patient numbers to extract (default: 1 2 3 4).",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Root data directory (default: data/).",
    )
    parser.add_argument(
        "--eval-questions", type=int, default=8,
        help="Number of MCQ questions to convert per patient (default: 8).",
    )
    parser.add_argument(
        "--list-patients", action="store_true",
        help="List all available patients and exit.",
    )
    args = parser.parse_args()

    data = _load_benchmark()

    if args.list_patients:
        print(f"{'Key':<12}  {'Name':<30}  {'Diagnosis'}")
        print("-" * 80)
        for key in sorted(data.keys()):
            p = data[key]
            print(f"{key:<12}  {p.get('name','?'):<30}  {p.get('diagnosis','?')}")
        return 0

    output_root = ROOT / args.output_dir
    corpus_names: list[str] = []

    for n in args.patients:
        patient_key = f"patient_{n:02d}"
        if patient_key not in data:
            print(f"Warning: {patient_key} not found — skipping.", file=sys.stderr)
            continue

        patient = data[patient_key]
        name = patient.get("name", "?")
        diagnosis = patient.get("diagnosis", "?")
        corpus_name = f"lh_p{n:02d}"
        corpus_dir = output_root / corpus_name
        corpus_dir.mkdir(parents=True, exist_ok=True)

        text = _patient_text(patient)
        (corpus_dir / "data.txt").write_text(text, encoding="utf-8")

        eval_qs = _make_eval_questions(patient, n=args.eval_questions)
        (corpus_dir / "eval_questions.json").write_text(
            json.dumps(eval_qs, indent=2), encoding="utf-8"
        )

        corpus_names.append(corpus_name)
        print(f"\n[{corpus_name}] {name} — {diagnosis}")
        print(f"  {len(text):,} chars, {len(text.split()):,} words")
        print(f"  {len(patient['questions'])} benchmark questions total, {len(eval_qs)} saved for eval")
        for q, a in eval_qs[:2]:
            q_line = q.split("\n")[0][:70]
            print(f"    Q: {q_line}")
            print(f"    A: {a[:60]}")
        if len(eval_qs) > 2:
            print(f"    ... and {len(eval_qs) - 2} more")

    if not corpus_names:
        print("No corpora prepared — check --patients.", file=sys.stderr)
        return 1

    papers_args = " ".join(f"{n}:data/{n}/data.txt" for n in corpus_names)
    names_arg = " ".join(corpus_names)

    print(f"\n{'='*60}")
    print(f"Corpora ready: {output_root}/")
    print("\n── Exp 1: independent training ──")
    print(f"python scripts/train_paper_cartridges.py \\")
    print(f"    --papers {papers_args} \\")
    print(f"    --output-dir outputs/exp_lh_1 \\")
    print(f"    --base-url http://127.0.0.1:8000/v1 --api-key cartridges-local \\")
    print(f"    --steps 240 --cartridge-tokens 512 \\")
    print(f"    --eval-questions-dir data \\")
    print(f"    --device cuda:0")
    print(f"\n── Eval Exp 1 ──")
    print(f"python scripts/eval_multi_cartridge.py \\")
    print(f"    --cartridge-dir outputs/exp_lh_1 \\")
    print(f"    --names {names_arg} --device cuda:0 --max-new-tokens 200")
    print(f"\n── Exp 2: distractor training (p_iso=0.75) ──")
    print(f"python scripts/train_joint_cartridges.py \\")
    print(f"    --corpus-order {names_arg} \\")
    print(f"    --supervision-dir outputs/exp_lh_1 \\")
    print(f"    --output-dir outputs/exp_lh_2 \\")
    print(f"    --steps 240 --cartridge-tokens 512 \\")
    print(f"    --p-isolation 0.75 \\")
    print(f"    --device cuda:0")
    print(f"\n── Eval Exp 2 ──")
    print(f"python scripts/eval_multi_cartridge.py \\")
    print(f"    --cartridge-dir outputs/exp_lh_2 \\")
    print(f"    --names {names_arg} --device cuda:0 --max-new-tokens 200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

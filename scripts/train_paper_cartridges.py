#!/usr/bin/env python3
"""Train cartridges from research paper PDFs using vLLM bootstrap.

Each paper goes through three phases:
  1. PDF download (from arxiv) or local file + text extraction
  2. vLLM bootstrap: generate factual Q&A pairs grounded in the paper text
  3. HF teacher supervision + cartridge training

Output per paper:
  {name}_cartridge.pt           — loadable by eval_multi_cartridge.py
  {name}_eval_questions.json    — auto-loaded by eval_multi_cartridge.py

eval_multi_cartridge.py loads the eval questions JSON automatically when
--names includes a name that has no entry in the hardcoded QUESTION_SETS.

Usage (vLLM server must be running):
    python scripts/train_paper_cartridges.py \\
        --papers cas:2606.04557 epicache:2509.17396 \\
        --output-dir outputs/exp1_papers \\
        --base-url http://127.0.0.1:8000/v1 \\
        --api-key cartridges-local \\
        --device cuda:0

    # From local PDF files:
    python scripts/train_paper_cartridges.py \\
        --papers cas:cas.pdf epicache:epicache.pdf \\
        --output-dir outputs/exp1_papers \\
        --base-url http://127.0.0.1:8000/v1 \\
        --api-key cartridges-local \\
        --device cuda:0
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.config import DEFAULT_HF_MODEL_ID                          # noqa: E402
from cartridges.benchmarks.text_benchmark import (                          # noqa: E402
    generate_bootstrap_questions,
    generate_teacher_answers,
    build_training_dataset,
)
from cartridges.train.cartridge import train_cartridge                      # noqa: E402
from cartridges.data.common import write_json                               # noqa: E402


def _download_arxiv_pdf(arxiv_id: str, output_path: Path) -> None:
    import requests
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    print(f"  Downloading {url} ...")
    r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    output_path.write_bytes(r.content)
    print(f"  Saved: {output_path.name} ({len(r.content) // 1024} KB)")


def _extract_pdf_text(pdf_path: Path, max_chars: int) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF text extraction.\n"
            "Install with: pip install pypdf"
        )
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    full_text = "\n\n".join(pages)
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]
        print(f"  Truncated to {max_chars:,} chars (~{max_chars // 4} tokens)")
    return full_text


def _fetch_arxiv_html(arxiv_id: str) -> str | None:
    """Download the arxiv HTML rendering. Returns HTML string or None if unavailable."""
    import requests
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
            print(f"  Downloaded HTML ({len(r.content) // 1024} KB) from {url}")
            return r.text
        print(f"  HTML not available (status {r.status_code}), will use PDF.")
    except Exception as e:
        print(f"  HTML fetch failed ({e}), will use PDF.")
    return None


def _extract_html_text(html_content: str, max_chars: int) -> str:
    """Convert arxiv HTML to plain text using stdlib HTMLParser (no BS4 needed)."""
    import html as _html
    import re
    from html.parser import HTMLParser

    class _Extractor(HTMLParser):
        SKIP = {"script", "style", "head", "math", "svg", "figure",
                "cite", "noscript", "nav", "footer", "references"}
        BLOCK = {"p", "h1", "h2", "h3", "h4", "h5", "h6",
                 "li", "div", "section", "article", "br", "tr", "td", "th"}

        def __init__(self):
            super().__init__()
            self._buf: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in self.SKIP:
                self._skip += 1
            elif tag in self.BLOCK and self._skip == 0:
                if self._buf and not self._buf[-1].endswith("\n"):
                    self._buf.append("\n")

        def handle_endtag(self, tag):
            if tag in self.SKIP:
                self._skip = max(0, self._skip - 1)
            elif tag in self.BLOCK and self._skip == 0:
                if self._buf and not self._buf[-1].endswith("\n"):
                    self._buf.append("\n")

        def handle_data(self, data):
            if self._skip == 0:
                self._buf.append(data)

    extractor = _Extractor()
    extractor.feed(html_content)
    text = _html.unescape("".join(extractor._buf))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^ +", "", text, flags=re.MULTILINE)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]
        print(f"  Truncated to {max_chars:,} chars (~{max_chars // 4} tokens)")
    return text


def _select_eval_questions(bootstrap_examples: list[dict], n: int) -> list[list[str]]:
    """Pick the N best eval questions from bootstrap output.

    Prefers answers that are short, precise, and numerically specific —
    these are easiest to substring-match and hardest to hallucinate correctly.
    """
    scored = []
    for ex in bootstrap_examples:
        answer = ex["expected_answer"].strip()
        words = answer.split()
        score = 0
        if 1 <= len(words) <= 6:
            score += 2
        if not answer.lower().startswith(("the ", "a ", "an ")):
            score += 1
        if any(c.isdigit() for c in answer):
            score += 2
        if "%" in answer or answer.endswith("x"):
            score += 1
        scored.append((score, ex["question"], answer.lower()))
    scored.sort(key=lambda x: -x[0])
    return [[q, a] for _, q, a in scored[:n]]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train cartridges from research paper PDFs via vLLM bootstrap."
    )
    parser.add_argument(
        "--papers", nargs="+", required=True,
        metavar="NAME:ARXIV_ID_OR_PATH",
        help=(
            "Papers to train, as name:source pairs. "
            "Source is either an arxiv ID (e.g. 2606.04557) or a local PDF path."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/exp1_papers",
                        help="Directory for cartridges, supervision, and eval questions.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                        help="vLLM server base URL (must be running).")
    parser.add_argument("--api-key", default="cartridges-local")
    parser.add_argument("--bootstrap-count", type=int, default=120,
                        help="Bootstrap Q&A pairs to generate per paper via vLLM.")
    parser.add_argument("--eval-questions", type=int, default=6,
                        help="Eval questions to auto-select per paper.")
    parser.add_argument("--cartridge-tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=240,
                        help="Cartridge training steps per paper.")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-chars", type=int, default=40000,
                        help="Max characters of paper text (~10K tokens). Papers are truncated to this.")
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--movement-log-interval", type=int, default=20,
                        help="Log key/value cosine similarity every N steps during training. 0 = disabled.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip vLLM bootstrap; assume supervision JSONLs already exist.")
    parser.add_argument("--pdf", action="store_true",
                        help="Force PDF extraction instead of arxiv HTML (HTML is default for arxiv IDs).")
    parser.add_argument(
        "--eval-questions-dir", default=None,
        metavar="DIR",
        help=(
            "If set, copy pre-made eval questions from DIR/{name}/eval_questions.json "
            "instead of auto-generating from bootstrap. Useful for LongHealth corpora "
            "where benchmark MCQ questions are more reliable than bootstrap questions."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse paper specs: name:arxiv_id_or_local_path
    papers: list[tuple[str, str]] = []
    for spec in args.papers:
        if ":" not in spec:
            print(f"Error: expected NAME:ID_OR_PATH, got '{spec}'", file=sys.stderr)
            return 1
        name, source = spec.split(":", maxsplit=1)
        papers.append((name, source))

    supervision_paths: dict[str, Path] = {
        name: output_dir / f"{name}.supervision.jsonl" for name, _ in papers
    }

    if not args.skip_bootstrap:
        # ── Phase 1: Download + extract text ─────────────────────────────────
        print("\n[1/3] Downloading and extracting paper text...")
        paper_texts: dict[str, str] = {}
        pdf_dir = output_dir / "pdfs"
        pdf_dir.mkdir(exist_ok=True)

        for name, source in papers:
            local_pdf = Path(source)
            text: str | None = None

            if local_pdf.exists():
                if local_pdf.suffix == ".txt":
                    print(f"  [{name}] Reading text file {source}...")
                    text = local_pdf.read_text(encoding="utf-8")
                    if len(text) > args.max_chars:
                        text = text[:args.max_chars]
                        print(f"  [{name}] Truncated to {args.max_chars:,} chars")
                else:
                    print(f"  [{name}] Extracting text from local file {source}...")
                    text = _extract_pdf_text(local_pdf, max_chars=args.max_chars)
            else:
                # Try HTML first (much cleaner than PDF for two-column papers)
                if not args.pdf:
                    print(f"  [{name}] Fetching arxiv HTML for {source}...")
                    html_content = _fetch_arxiv_html(source)
                    if html_content:
                        text = _extract_html_text(html_content, max_chars=args.max_chars)

                if text is None:
                    cached = pdf_dir / f"{name}.pdf"
                    if not cached.exists():
                        _download_arxiv_pdf(source, cached)
                    else:
                        print(f"  [{name}] Using cached: {cached.name}")
                    print(f"  [{name}] Extracting text from {cached.name}...")
                    text = _extract_pdf_text(cached, max_chars=args.max_chars)

            paper_texts[name] = text
            print(f"  [{name}] {len(text):,} chars")

        # ── Phase 2: Bootstrap + supervision ─────────────────────────────────
        print(f"\n[2/3] Bootstrap + HF supervision (model: {DEFAULT_HF_MODEL_ID})...")

        for name, _ in papers:
            t0 = time.perf_counter()
            corpus_text = paper_texts[name]
            paper_dir = output_dir / name
            paper_dir.mkdir(exist_ok=True)

            print(f"\n  [{name}] Generating {args.bootstrap_count} bootstrap Q&A via vLLM...")
            bootstrap_examples = generate_bootstrap_questions(
                corpus_text=corpus_text,
                eval_spec=[],
                output_path=paper_dir / "bootstrap_questions.txt",
                base_url=args.base_url,
                api_key=args.api_key,
                num_questions=args.bootstrap_count,
            )
            print(f"  [{name}] {len(bootstrap_examples)} Q&A pairs generated")

            answer_records = generate_teacher_answers(
                corpus_text=corpus_text,
                bootstrap_examples=bootstrap_examples,
                output_path=paper_dir / "answer_records.jsonl",
                base_url=args.base_url,
                api_key=args.api_key,
                max_completion_tokens=64,
            )

            print(f"  [{name}] Building HF teacher supervision...")
            build_training_dataset(
                corpus_text=corpus_text,
                slice_id=name,
                answer_records=answer_records,
                output_path=supervision_paths[name],
                device=args.device,
                top_logprobs=args.top_logprobs,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save eval questions alongside cartridge for auto-loading by eval_multi_cartridge.py
            eval_path = output_dir / f"{name}_eval_questions.json"
            if args.eval_questions_dir:
                src = Path(args.eval_questions_dir) / name / "eval_questions.json"
                if src.exists():
                    shutil.copy2(src, eval_path)
                    print(f"  [{name}] Copied eval questions from {src}")
                    eval_questions = json.loads(src.read_text(encoding="utf-8"))
                else:
                    print(f"  [{name}] Warning: {src} not found — falling back to bootstrap eval questions")
                    eval_questions = _select_eval_questions(bootstrap_examples, n=args.eval_questions)
                    eval_path.write_text(json.dumps(eval_questions, indent=2), encoding="utf-8")
            else:
                eval_questions = _select_eval_questions(bootstrap_examples, n=args.eval_questions)
                eval_path.write_text(json.dumps(eval_questions, indent=2), encoding="utf-8")

            elapsed = time.perf_counter() - t0
            print(f"  [{name}] Done in {elapsed:.1f}s — supervision: {supervision_paths[name].name}")
            print(f"  [{name}] Eval questions saved → {eval_path.name}")
            print("  Selected eval questions:")
            for q, a in eval_questions:
                print(f"    Q: {q[:80]}")
                print(f"    A: {a}")

    else:
        print("\n[1-2/3] Skipping bootstrap (--skip-bootstrap set).")
        for name, _ in papers:
            if not supervision_paths[name].exists():
                raise FileNotFoundError(
                    f"Supervision not found: {supervision_paths[name]}. "
                    "Run without --skip-bootstrap first."
                )

    # ── Phase 3: Train cartridges ─────────────────────────────────────────────
    print(f"\n[3/3] Training {len(papers)} cartridges...")
    train_summaries: dict[str, dict] = {}
    log_interval = args.movement_log_interval if args.movement_log_interval > 0 else None

    for name, _ in papers:
        t0 = time.perf_counter()
        print(f"\n── Training [{name}] ──")
        summary = train_cartridge(
            dataset_path=supervision_paths[name],
            output_dir=output_dir / f"{name}_train",
            slice_id=name,
            device=args.device,
            cartridge_tokens=args.cartridge_tokens,
            learning_rate=args.learning_rate,
            steps=args.steps,
            num_frozen_tokens=1,
            movement_log_interval=log_interval,
        )
        elapsed = time.perf_counter() - t0
        final_path = output_dir / f"{name}_cartridge.pt"
        shutil.copy2(summary["cartridge_path"], final_path)
        if log_interval and summary.get("movement_log"):
            movement_path = output_dir / f"{name}_movement.json"
            movement_path.write_text(
                json.dumps(summary["movement_log"], indent=2), encoding="utf-8"
            )
            print(f"  [{name}] Movement log → {movement_path.name}")
        train_summaries[name] = {**summary, "elapsed_seconds": elapsed}
        print(f"  [{name}] {elapsed:.1f}s — best_loss={summary['best_loss']:.4f}")

    write_json(output_dir / "train_summary.json", {
        "model": DEFAULT_HF_MODEL_ID,
        "papers": [name for name, _ in papers],
        "cartridge_tokens": args.cartridge_tokens,
        "steps": args.steps,
        "bootstrap_count": args.bootstrap_count,
        "summaries": train_summaries,
    })

    print(f"\n{'='*60}")
    print(f"Done. Outputs in: {output_dir}/")
    names_str = " ".join(name for name, _ in papers)
    print(f"\nEval:")
    print(f"  python scripts/eval_multi_cartridge.py \\")
    print(f"      --cartridge-dir {output_dir} \\")
    print(f"      --names {names_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

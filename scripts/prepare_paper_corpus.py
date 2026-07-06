#!/usr/bin/env python3
"""Prepare a research paper as a benchmark corpus for run_benchmark.py.

Downloads a PDF from arxiv (or reads a local file), extracts plain text, and
creates data/{name}/ with data.txt + eval_spec.json + metadata.json.

The eval spec is auto-generated from a small vLLM bootstrap pass (--eval-count
questions), separate from the larger bootstrap used for training. These questions
are excluded from the training bootstrap so there is no data leakage.

Usage (vLLM server must be running):
    python scripts/prepare_paper_corpus.py cas 2606.04557 \\
        --base-url http://127.0.0.1:8000/v1 --api-key cartridges-local

    python scripts/prepare_paper_corpus.py epicache 2509.17396 \\
        --base-url http://127.0.0.1:8000/v1 --api-key cartridges-local

    # From a local PDF:
    python scripts/prepare_paper_corpus.py cas path/to/cas.pdf \\
        --base-url http://127.0.0.1:8000/v1 --api-key cartridges-local

Then run the benchmark:
    python scripts/run_benchmark.py cas \\
        --gpu 0 --device cuda:0 \\
        --base-url http://127.0.0.1:8000/v1 --api-key cartridges-local \\
        --train-steps 60 --semantic-judge --run-name cas_baseline
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.benchmarks.text_benchmark import (  # noqa: E402
    generate_bootstrap_questions,
    _content_passages,
)


def _download_arxiv_pdf(arxiv_id: str, output_path: Path) -> None:
    import requests
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    print(f"Downloading {url} ...")
    r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    output_path.write_bytes(r.content)
    print(f"Saved: {output_path.name}  ({len(r.content) // 1024} KB)")


def _extract_pdf_text(pdf_path: Path, max_chars: int) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if len(text) > max_chars:
        text = text[:max_chars]
        print(f"Truncated to {max_chars:,} chars")
    return text


def _fetch_arxiv_html(arxiv_id: str) -> str | None:
    """Download the arxiv HTML rendering. Returns HTML string or None if unavailable."""
    import requests
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
            print(f"Downloaded HTML ({len(r.content) // 1024} KB) from {url}")
            return r.text
        print(f"HTML not available (status {r.status_code}), will use PDF.")
    except Exception as e:
        print(f"HTML fetch failed ({e}), will use PDF.")
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
        print(f"Truncated to {max_chars:,} chars (~{max_chars // 4} tokens)")
    return text


def _bootstrap_to_eval_spec(
    corpus_name: str,
    bootstrap_examples: list[dict],
) -> list[dict]:
    """Convert bootstrap Q&A pairs into eval_spec format."""
    spec = []
    for i, ex in enumerate(bootstrap_examples, start=1):
        spec.append({
            "id": f"{corpus_name}-q{i:02d}",
            "query": ex["question"].strip(),
            "answer_prompt": "Answer with only the shortest exact phrase from the paper.",
            "answers": [ex["expected_answer"].strip().lower()],
        })
    return spec


def _condense_answers(
    examples: list[dict],
    base_url: str,
    api_key: str,
) -> list[dict]:
    """Ask vLLM to condense each answer to a short extractable key phrase (1-6 words).

    The goal is to replace full-sentence expected answers with short substrings
    that will appear verbatim in any correct answer, avoiding semantic judge failures
    caused by paraphrase mismatch.
    """
    from cartridges.clients.vllm_openai import VLLMClient
    client = VLLMClient(base_url=base_url, api_key=api_key)

    condensed = []
    for ex in examples:
        messages = [
            {
                "role": "user",
                "content": (
                    "/no_think\n"
                    "Given this question and answer from a research paper, extract the "
                    "shortest exact key phrase (1 to 6 words) from the answer that "
                    "uniquely identifies the correct response. Output only the phrase, "
                    "nothing else.\n\n"
                    f"Question: {ex['question']}\n"
                    f"Answer: {ex['expected_answer']}\n"
                    "Key phrase:"
                ),
            }
        ]
        try:
            result = client.chat(messages, max_completion_tokens=20, temperature=0.0)
            import re as _re
            phrase = _re.sub(r"<think>.*?</think>", "", result.text, flags=_re.DOTALL)
            phrase = phrase.strip().strip('"\'').strip()
            condensed.append({**ex, "expected_answer": phrase})
        except Exception:
            condensed.append(ex)
    return condensed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a research paper as a run_benchmark.py corpus."
    )
    parser.add_argument("name", help="Corpus name (becomes data/{name}/).")
    parser.add_argument(
        "source",
        help="Arxiv ID (e.g. 2606.04557) or path to a local PDF file.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="cartridges-local")
    parser.add_argument(
        "--eval-count", type=int, default=15,
        help="Number of eval questions to generate for eval_spec.json.",
    )
    parser.add_argument(
        "--max-chars", type=int, default=40000,
        help="Max characters of paper text to keep (~10K tokens).",
    )
    parser.add_argument(
        "--data-root", default="data",
        help="Root data directory (default: data/).",
    )
    parser.add_argument(
        "--pdf", action="store_true",
        help="Force PDF extraction instead of arxiv HTML (HTML is default for arxiv IDs).",
    )
    parser.add_argument(
        "--condense-answers", action="store_true",
        help=(
            "After bootstrap, ask vLLM to condense each expected answer to a short "
            "key phrase (1-6 words). Produces more reliable substring matching in eval."
        ),
    )
    args = parser.parse_args()

    corpus_dir = ROOT / args.data_root / args.name
    if corpus_dir.exists():
        print(f"Warning: {corpus_dir} already exists — files will be overwritten.")
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1+2: Get paper text ──────────────────────────────────────────────
    local_pdf = Path(args.source)
    text: str | None = None
    source_id = args.source

    if local_pdf.exists():
        # Local file — PDF only
        print(f"\nExtracting text from local file {args.source}...")
        text = _extract_pdf_text(local_pdf, max_chars=args.max_chars)
        source_id = str(local_pdf.resolve())
    else:
        # Treat as arxiv ID — try HTML first (much cleaner than PDF for two-column papers)
        if not args.pdf:
            print(f"\nFetching arxiv HTML for {args.source}...")
            html_content = _fetch_arxiv_html(args.source)
            if html_content:
                text = _extract_html_text(html_content, max_chars=args.max_chars)

        if text is None:
            pdf_cache = corpus_dir / f"{args.name}.pdf"
            if not pdf_cache.exists():
                _download_arxiv_pdf(args.source, pdf_cache)
            else:
                print(f"Using cached PDF: {pdf_cache.name}")
            print(f"\nExtracting text from {pdf_cache.name}...")
            text = _extract_pdf_text(pdf_cache, max_chars=args.max_chars)

    print(f"{len(text):,} chars, ~{len(text) // 4} tokens")
    print(f"Content passages: {len(_content_passages(text))}")

    data_txt_path = corpus_dir / "data.txt"
    data_txt_path.write_text(text, encoding="utf-8")
    print(f"Saved: {data_txt_path}")

    # ── Step 3: Generate eval spec via vLLM ──────────────────────────────────
    print(f"\nGenerating {args.eval_count} eval questions via vLLM...")
    eval_examples = generate_bootstrap_questions(
        corpus_text=text,
        eval_spec=[],
        output_path=corpus_dir / "eval_bootstrap_raw.txt",
        base_url=args.base_url,
        api_key=args.api_key,
        num_questions=args.eval_count,
    )
    print(f"Generated {len(eval_examples)} eval Q&A pairs")

    if args.condense_answers:
        print("Condensing expected answers to short key phrases via vLLM...")
        eval_examples = _condense_answers(eval_examples, args.base_url, args.api_key)

    eval_spec = _bootstrap_to_eval_spec(args.name, eval_examples)
    eval_spec_path = corpus_dir / "eval_spec.json"
    eval_spec_path.write_text(json.dumps(eval_spec, indent=2), encoding="utf-8")
    print(f"Saved: {eval_spec_path}")

    # ── Step 4: Metadata ──────────────────────────────────────────────────────
    metadata = {
        "title": args.name,
        "source": source_id,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "original_characters": len(text),
        "original_tokens": len(text) // 4,
        "truncated_tokens": len(text) // 4,
    }
    (corpus_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Corpus ready: {corpus_dir}/")
    print(f"  data.txt        {len(text):>10,} chars")
    print(f"  eval_spec.json  {len(eval_spec):>10} questions")
    print(f"\nEval questions preview:")
    for item in eval_spec[:5]:
        print(f"  Q: {item['query'][:70]}")
        print(f"  A: {item['answers'][0]}")
    if len(eval_spec) > 5:
        print(f"  ... and {len(eval_spec) - 5} more")

    print(f"\nNext — run the benchmark:")
    print(f"  python scripts/run_benchmark.py {args.name} \\")
    print(f"      --gpu 0 --device cuda:0 \\")
    print(f"      --base-url {args.base_url} --api-key {args.api_key} \\")
    print(f"      --train-steps 60 --semantic-judge \\")
    print(f"      --run-name {args.name}_baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Zaptridge — Project Context for Claude

## What This Project Is

A research project by a Penn PhD student building toward an **on-device personal memory system for mobile LLMs** (target: iPhone). The core mechanism is KV cache cartridges: instead of re-prefilling a long context on every query, a compact set of learned K/V tensors (the "cartridge") is trained once to encode a document or conversation, then loaded at inference time.

**Long-term vision:** The phone runs a local LLM. Throughout the day the user chats with it. Overnight while charging, `compress_conversation.py` compresses the day's conversation into a cartridge. Next morning the updated cartridge is loaded and the model has persistent memory — no cloud round-trip, no re-prefilling history.

## Base Implementation

Built on top of [shreyansh26/cartridges](https://github.com/shreyansh26/cartridges), a single-GPU reproduction of the HazyResearch Cartridges paper (Eyuboglu et al., 2025). Our repo is [Shaho912/zaptridge](https://github.com/Shaho912/zaptridge).

**Hardware:** DGX server, NVIDIA B200 MIG45 GPU, Qwen/Qwen3-8B as working model. vLLM server runs on the same machine for bootstrap/teacher generation.

**IMPORTANT:** Always set `CARTRIDGES_HF_MODEL_ID=Qwen/Qwen3-8B` before running any training or eval — the default config is 4B which gives wrong numbers.

## Key Files

```
scripts/
  run_benchmark.py              # Main orchestrator — full pipeline, CLI flags
  compress_conversation.py      # Compress conversation history into cartridge (no vLLM)
  update_cartridge.py           # Delta slot incremental cartridge updates
  serve_vllm.py                 # vLLM server wrapper
  prepare_longhealth.py         # Download LongHealth benchmark → data/lh_p{N}/ corpora
  train_paper_cartridges.py     # Bootstrap + train cartridges from PDFs or .txt files
  train_joint_cartridges.py     # CAS-style distractor training (p_isolation, k_min/max)
  train_multi_cartridge.py      # Train multiple cartridges independently in one call
  eval_multi_cartridge.py       # Oracle + joint eval for multi-cartridge setups
  prune_supervision.py          # EpiCache-style clustering supervision pruning
  consolidate_cartridge.py      # Merge multiple supervision files into one cartridge
  eval_conversation_cartridge.py # Eval a conversation-compressed cartridge
  eval_baseline_conversation.py  # Full-context prefill eval baseline
  convert_claude_export.py      # Convert Claude export JSON → JSONL for cartridge
  run_ablation_cartridge.py     # Cartridge token / step ablations

src/cartridges/
  config.py                     # DEFAULT_MATRIX (vLLM model), DEFAULT_HF_MODEL_ID (local HF)
  core/
    cartridge.py                # TrainableKVCartridge, DeltaKVCartridge, init functions
    kvzip_scoring.py            # KVzip+ importance scoring (experimental, shelved)
    __init__.py
  train/
    cartridge.py                # train_cartridge() — distillation training loop
  eval/
    baseline.py                 # run_local_hf_matched_eval()
    cartridge.py                # run_cartridge_eval()
    common.py                   # exact_match, EvalRecord, etc.
  benchmarks/
    text_benchmark.py           # bootstrap generation, build_training_dataset, reports
  clients/
    vllm_openai.py              # VLLMClient — talks to vLLM server
  data/
    text_dataset.py             # corpus chunking, manifest building

data/                           # Lives on DGX; gitignored *.jsonl and patient dirs
  wikipedia_india/              # Wikipedia article on India (original test corpus)
  wikipedia_history_us/         # Secondary corpus
  longhealth_benchmark_v5.json  # Cached LongHealth benchmark (auto-downloaded)
  lh_p01/ … lh_p08/            # LongHealth patient corpora (DGX only)
    data.txt                    # Combined clinical documents (~44-50K chars each)
    eval_questions.json         # 4-8 MCQ eval questions from benchmark

outputs/                        # All experiment outputs on DGX
  exp_lh_1_1024/                # LongHealth 4-patient independent (1024 slots)
  exp_lh_2_1024/                # LongHealth 4-patient distractor p=0.75 (1024 slots)
  cold_swap/                    # Exp2 ×4 + p05 independently trained newcomer
  warm_swap/                    # Exp2 ×4 + p05 distractor-trained against frozen set

docs/
  dynamic_cartridge_findings.md # Full findings document — start here for context
```

## Model ID Architecture

Two separate model IDs to allow profiling smaller models without breaking vLLM:

- `CARTRIDGES_MODEL_ID` (env var) → vLLM server model, default `Qwen/Qwen3-4B`
- `CARTRIDGES_HF_MODEL_ID` (env var) → local HF model; **must be set to `Qwen/Qwen3-8B`** for all current experiments

Always run with:
```bash
export CARTRIDGES_HF_MODEL_ID=Qwen/Qwen3-8B
```

## Optimal Hyperparameters (Found via Ablation)

- `--train-steps 60` (default was 240 — same quality, 57% faster)
- `--bootstrap-count 120` (minimum for good quality)
- `--cartridge-tokens 1024` (512 degrades exact match)

Optimal build time: **~72s** for 1 corpus chunk, 0.80 exact / 0.95 semantic match.

## What We've Built Beyond the Base

1. **Per-phase timing instrumentation** — `[timing]` and `[memory]` log lines, `phase_timings` key in `run_manifest.json`
2. **GPU memory profiling** — `--profile-flops` flag, peak/delta memory per phase
3. **`--model-id` flag** — swap local HF model without touching vLLM
4. **`compress_conversation.py`** — dynamic conversation cartridge (no vLLM needed)
5. **`DeltaKVCartridge`** — frozen base + trainable delta slots (ultimately rejected — interference too high)
6. **`update_cartridge.py`** — incremental update script using delta slots
7. **KVzip+ initialization** (shelved) — on `kvzip-experiments` branch
8. **Multi-cartridge composition pipeline** — `train_paper_cartridges.py`, `train_joint_cartridges.py`, `eval_multi_cartridge.py` for CAS-style distractor training and oracle/joint eval
9. **LongHealth integration** — `prepare_longhealth.py` downloads kbressem/LongHealth patient records as cartridge training corpora with benchmark MCQ eval questions
10. **`prune_supervision.py`** — EpiCache-inspired clustering + importance scoring for supervision pruning

## Memory Profiling Results (Qwen3-4B, B200 MIG45)

| Phase | Peak memory |
|---|---|
| build_training_dataset | 14.4 GB |
| baseline_eval | 13.1 GB |
| train_cartridge | 9.8 GB |
| cartridge_eval | 8.7 GB |

Qwen3-4B too large for iPhone 16 Pro (8 GB). Current work uses Qwen3-8B on DGX for research; mobile profiling is a separate future task.

## Current Research Direction: Multi-Cartridge Composition (CAS)

Investigating whether CAS-style distractor training (Eyuboglu et al. 2025, arxiv 2606.04557) enables multiple KV cache cartridges to coexist when concatenated at inference.

**Key concepts:**
- **Oracle eval**: each cartridge loaded alone — best-case individual performance
- **Joint eval**: all cartridges concatenated — real deployment scenario
- **Drop = oracle − joint**: positive = interference, negative = positive transfer (joint > oracle)
- **p_isolation**: probability of training target cartridge alone vs with distractors (CAS paper uses 0.75)

### Multi-Cartridge Workflow (LongHealth, 4 patients, 1024 slots)

```bash
export CARTRIDGES_HF_MODEL_ID=Qwen/Qwen3-8B

# 1. Prepare patient corpora
python scripts/prepare_longhealth.py --patients 1 2 3 4

# 2. Bootstrap + train Exp 1 (independent; vLLM must be running)
python scripts/train_paper_cartridges.py \
    --papers lh_p01:data/lh_p01/data.txt lh_p02:data/lh_p02/data.txt \
             lh_p03:data/lh_p03/data.txt lh_p04:data/lh_p04/data.txt \
    --output-dir outputs/exp_lh_1_1024 \
    --base-url http://127.0.0.1:8000/v1 --api-key cartridges-local \
    --steps 240 --cartridge-tokens 1024 \
    --eval-questions-dir data --device cuda:0

# 3. Exp 2: distractor training (reuses Exp 1 supervision, no vLLM needed)
python scripts/train_joint_cartridges.py \
    --corpus-order lh_p01 lh_p02 lh_p03 lh_p04 \
    --supervision-dir outputs/exp_lh_1_1024 \
    --output-dir outputs/exp_lh_2_1024 \
    --steps 240 --cartridge-tokens 1024 \
    --p-isolation 0.75 \
    --eval-questions-dir data --device cuda:0

# 4. Eval
python scripts/eval_multi_cartridge.py \
    --cartridge-dir outputs/exp_lh_2_1024 \
    --names lh_p01 lh_p02 lh_p03 lh_p04 --device cuda:0 --max-new-tokens 200
```

### Multi-Cartridge Findings Summary

| Experiment | Oracle | Joint | Drop | Notes |
|---|---|---|---|---|
| LH 1024-slot Exp 1 (independent) | 9/16 | 8/16 | +1 | Baseline interference |
| LH 1024-slot Exp 2 (distractor p=0.75) | 7/16 | 10/16 | −3 | Positive transfer |
| Cold swap (Exp2×4 + p05 independent) | 8/20 | 11/20 | −3 | Plug-and-play works |
| Warm swap (Exp2×4 + p05 distractor) | 8/20 | 10/20 | −2 | ≈ cold swap |

**Key findings:**
1. Distractor training (p=0.75) converts interference into positive transfer on diverse documents
2. Domain diversity is essential — same-domain content (ML papers) makes distractor training hurt
3. Patient names + distinct diagnoses provide the routing signal
4. Cold swap (plug-and-play) works — no retraining needed when adding a new cartridge
5. Warm swap ≈ cold swap — targeted distractor training of newcomer adds no measurable benefit

## Standard Single-Corpus Run Command

```bash
export CARTRIDGES_HF_MODEL_ID=Qwen/Qwen3-8B
python scripts/run_benchmark.py wikipedia_india \
  --gpu 0 --device cuda:0 \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key cartridges-local \
  --semantic-judge \
  --train-steps 60 \
  --run-name <name>
```

## Key Findings Doc

See `docs/dynamic_cartridge_findings.md` for full experimental results, ablation tables, and architectural decisions.

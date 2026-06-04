# Zaptridge — Project Context for Claude

## What This Project Is

A research project by a Penn PhD student building toward an **on-device personal memory system for mobile LLMs** (target: iPhone). The core mechanism is KV cache cartridges: instead of re-prefilling a long context on every query, a compact set of learned K/V tensors (the "cartridge") is trained once to encode a document or conversation, then loaded at inference time.

**Long-term vision:** The phone runs a local LLM. Throughout the day the user chats with it. Overnight while charging, `compress_conversation.py` compresses the day's conversation into a cartridge. Next morning the updated cartridge is loaded and the model has persistent memory — no cloud round-trip, no re-prefilling history.

## Base Implementation

Built on top of [shreyansh26/cartridges](https://github.com/shreyansh26/cartridges), a single-GPU reproduction of the HazyResearch Cartridges paper (Eyuboglu et al., 2025). Our repo is [Shaho912/zaptridge](https://github.com/Shaho912/zaptridge).

**Hardware:** DGX server, NVIDIA B200 MIG45 GPU, Qwen/Qwen3-4B as default model. vLLM server runs on the same machine for bootstrap/teacher generation.

## Key Files

```
scripts/
  run_benchmark.py          # Main orchestrator — full pipeline, CLI flags
  compress_conversation.py  # NEW: compress conversation history into cartridge
  update_cartridge.py       # NEW: delta slot incremental cartridge updates
  serve_vllm.py             # vLLM server wrapper

src/cartridges/
  config.py                 # DEFAULT_MATRIX (vLLM model), DEFAULT_HF_MODEL_ID (local HF)
  core/
    cartridge.py            # TrainableKVCartridge, DeltaKVCartridge, init functions
    kvzip_scoring.py        # KVzip+ importance scoring (experimental, shelved)
    __init__.py
  train/
    cartridge.py            # train_cartridge() — distillation training loop
  eval/
    baseline.py             # run_local_hf_matched_eval()
    cartridge.py            # run_cartridge_eval()
    common.py               # exact_match, EvalRecord, etc.
  benchmarks/
    text_benchmark.py       # bootstrap generation, build_training_dataset, reports
  clients/
    vllm_openai.py          # VLLMClient — talks to vLLM server
  data/
    text_dataset.py         # corpus chunking, manifest building

data/
  wikipedia_india/          # Main test corpus (Wikipedia article on India)
  wikipedia_history_us/     # Secondary corpus

docs/
  dynamic_cartridge_findings.md  # Full findings document — start here for context
```

## Model ID Architecture

Two separate model IDs to allow profiling smaller models without breaking vLLM:

- `CARTRIDGES_MODEL_ID` (env var) → vLLM server model, default `Qwen/Qwen3-4B`
- `CARTRIDGES_HF_MODEL_ID` (env var) → local HF model, set via `--model-id` CLI flag

Use `--model-id Qwen/Qwen3-0.6B` to swap only the local HF phases (training, eval) while keeping vLLM at 4B.

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
5. **`DeltaKVCartridge`** — frozen base + trainable delta slots for zero-forgetting updates
6. **`update_cartridge.py`** — incremental update script using delta slots
7. **KVzip+ initialization** (shelved) — on `kvzip-experiments` branch

## Memory Profiling Results (Qwen3-4B, B200 MIG45)

| Phase | Peak memory |
|---|---|
| build_training_dataset | 14.4 GB |
| baseline_eval | 13.1 GB |
| train_cartridge | 9.8 GB |
| cartridge_eval | 8.7 GB |

Qwen3-4B too large for iPhone 16 Pro (8 GB). Next: profile Qwen3-0.6B.

## Current Task

Profiling Qwen3-0.6B memory usage to assess mobile feasibility:

```bash
python scripts/run_benchmark.py wikipedia_india \
  --gpu 0 --device cuda:0 \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key cartridges-local \
  --semantic-judge \
  --model-id Qwen/Qwen3-0.6B \
  --profile-flops \
  --run-name memory_profile_0.6b
```

## Standard Run Command

```bash
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

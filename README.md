# mobile_cartridges

Research toward an **on-device personal memory system for mobile LLMs**. Built on top of [shreyansh26/cartridges](https://github.com/shreyansh26/cartridges), a single-GPU reproduction of the HazyResearch Cartridges paper (Eyuboglu et al., 2025).

## What This Is

Instead of re-prefilling a long context on every query, a compact set of learned K/V tensors (a "cartridge") is trained once to encode a document or conversation, then loaded at inference time. The long-term target is running this on-device: a local LLM compresses the day's conversation into a cartridge overnight, and loads it the next morning for persistent memory with no cloud round-trip.

**Current model:** Qwen/Qwen3-8B on a DGX B200 MIG45 GPU.

## Key Extensions

- **Multi-cartridge composition (CAS)** — distractor training so multiple cartridges coexist when concatenated at inference, achieving positive transfer across diverse patient record corpora
- **Cold/warm swap** — plug-and-play addition of new cartridges to an existing CAS set
- **Conversation cartridges** — `compress_conversation.py` compresses conversation history without vLLM
- **LongHealth integration** — MCQ benchmark eval across 8 patient records
- **Per-phase timing and GPU memory profiling**

## Setup

```bash
export CARTRIDGES_HF_MODEL_ID=Qwen/Qwen3-8B
```

See `CLAUDE.md` for full workflow, hyperparameters, and experiment commands.

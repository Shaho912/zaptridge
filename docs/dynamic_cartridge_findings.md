# Dynamic Cartridge: Compressing Conversation History into KV Cache

## Motivation

Standard KV cartridges compress a static corpus (e.g. a Wikipedia article) once and serve
many queries against it. This work asks: **can the same mechanism compress a live conversation
history so future turns attend to compact KV state instead of re-prefilling the full session?**

The key insight is that conversation turns are structurally identical to the Q&A pairs used to
train static cartridges. Each (user, assistant) pair is a supervised signal: given the session
context, reproduce this response. No vLLM, no bootstrap generation, no external teacher needed —
the HF model already loaded for inference is the teacher.

---

## What Was Built

### 1. `scripts/compress_conversation.py`

Takes a JSONL of conversation turns and trains a cartridge encoding the session.

**Pipeline:**

```
Conversation turns (JSONL)
    ↓
[Phase 1] HF teacher forward pass on each (user, assistant) pair
          with full conversation as system context → per-token logprobs
    ↓
[Phase 2] train_cartridge() — same distillation loss as static corpus
    ↓
Session cartridge .pt
```

**Input format** — one JSON object per line:
```json
{"role": "user",      "content": "What is India's capital?"}
{"role": "assistant", "content": "New Delhi is the capital of India."}
```

**Usage:**
```bash
python scripts/compress_conversation.py \
    --conversation-path session.jsonl \
    --output-path session_cartridge.pt \
    --cartridge-tokens 1024 \
    --steps 60 \
    --device cuda:0
```

**Key difference from static corpus pipeline:**

| | Static corpus | Conversation |
|---|---|---|
| Supervision source | Bootstrap Q&A via vLLM | Actual turns via HF model |
| System prompt | Wikipedia article | Formatted conversation history |
| Training examples | 120 Q&A pairs | One per assistant turn |
| vLLM required | Yes | **No** |

---

### 2. `DeltaKVCartridge` — `src/cartridges/core/cartridge.py`

A companion class for incremental updates to an existing cartridge without modifying it.

**Architecture:**
```
[Frozen base KV slots (1024)] + [Trainable delta KV slots (256)]
         ↑                                    ↑
   original cartridge                   new information
   (requires_grad=False)               (requires_grad=True)
```

`as_cache()` concatenates both into a single `DynamicCache` for inference.
After training, `merge()` exports a standard `TrainableKVCartridge` compatible
with all existing eval infrastructure.

**Key property:** the base cartridge is never modified. Forgetting is zero by construction.

---

### 3. `scripts/update_cartridge.py`

Trains delta slots on new Q&A pairs using an existing cartridge as frozen base.

```bash
python scripts/update_cartridge.py \
    --cartridge-path existing_cartridge.pt \
    --dataset-path new_pairs.jsonl \
    --output-path merged_cartridge.pt \
    --delta-tokens 256 \
    --steps 15 \
    --device cuda:0
```

---

## Experimental Results

### Experiment 1: Compress a Short Session (5 turns)

**Session content:** Basic India facts (official name, independence year, first PM, capital, population rank).

| Metric | Value |
|---|---|
| Build time | **12.1s** (4.3s supervision + 7.8s training) |
| Cartridge tokens | 512 |
| Best distillation loss | 0.0005 |
| In-session recall | **5/5** |
| Out-of-scope response | "The context provided does not mention..." ✓ |

The cartridge correctly refused to answer a question not present in the session.

---

### Experiment 2: Compress a Longer Session (8 turns) — 512 vs 1024 tokens

**Session content:** 8 detailed turns covering geography, cities, languages, parliament,
economy, space program, festivals, and technology sector.

| | 512 tokens | 1024 tokens |
|---|---|---|
| Build time | 22.6s | ~30s |
| Best loss | 0.1685 | lower |
| In-session recall | 8/9 | **9/9** |
| Fact confusion | GDP rank (5th) confused with area rank (7th) | None |
| Out-of-scope hallucination | Yes (parametric knowledge) | Yes (parametric knowledge) |

**Finding:** At 512 tokens, the cartridge blended two similar numeric facts from different turns
(India's GDP rank and area rank both appeared in the session). At 1024 tokens the confusion
disappeared — more slots allow cleaner separation of distinct facts.

**Out-of-scope behavior:** When asked about India's current Prime Minister (not in the session),
both cartridges hallucinated from the model's parametric knowledge. This is a model-level
behavior: the model uses training data when it has the answer, regardless of the cartridge.
Contrast with the short session where an unknown river question was correctly refused — the model
genuinely had no answer to fall back on.

---

### Experiment 3: Delta Slots for New-Knowledge Injection

To test whether an existing cartridge can be updated with genuinely new out-of-corpus facts
without catastrophic forgetting.

**Synthetic new corpus:** A paragraph about a fictional "India Space Innovation Hub" —
completely absent from the original Wikipedia India training data.

| Config | Original score | New facts | Old facts | Forgetting |
|---|---|---|---|---|
| Direct fine-tune lr=3e-3 | 0.85 | 0/5 | degraded | Severe (cartridge broken) |
| Direct fine-tune lr=1e-3 | 0.85 | 0/5 | 0.80 | Moderate |
| Delta-32 slots | 0.85 | 0/5 | 0.85 | **Zero** |
| **Delta-256 slots** | 0.85 | **5/5** | 0.85 | **Zero** |

**Why direct fine-tuning fails:** cartridge K/V tensors are a jointly-optimized distributed
representation. High initial loss on new facts → large gradients → all 1024 positions shift
simultaneously → the existing representation breaks. At extreme, the model generates empty
strings (immediate EOS) because the cartridge context becomes incoherent.

**Why 32 delta slots fail:** 32 new slots against 1024 frozen base slots is only 3% of total
context. The base's Wikipedia India context dominates and the model correctly says "context
does not mention Space Innovation Hub."

**Why 256 delta slots work:** 256/(1024+256) = 20% of total context. Enough signal to shift
the model's beliefs on the new facts while the frozen base preserves all existing knowledge.

---

## Limitations

**1. Build time vs amortization**
At 22s build cost and ~240ms prefill savings per turn, break-even requires ~90 queries on
the same compressed snapshot. Typical conversations don't reach this threshold.

**2. Context windows are large**
Modern LLMs have 32K–128K token context windows. An 8-turn conversation is ~5,000 tokens
and fits comfortably without compression. Dynamic cartridges become relevant only when
sessions genuinely overflow the context window.

**3. Parametric knowledge hallucination**
The cartridge cannot suppress model answers that come from training data. Only genuinely
unknown questions get the "context does not mention" refusal.

**4. Compression quality degrades with density**
More content per slot means higher distillation loss and more fact bleeding.
A 512-slot cartridge for 8 detailed turns is tight; 1024 slots gives clean separation.

---

## When Dynamic Cartridges Are Actually Useful

| Scenario | Useful? | Why |
|---|---|---|
| Short conversations (< 20 turns) | No | Context window handles it fine |
| Ultra-long sessions (100+ turns, 50K+ tokens) | Yes | Context window overflows |
| Cross-session persistent memory | Yes | Compact, opaque (privacy), fast to load |
| Resource-constrained hardware (small context window) | Yes | Compression is necessary |
| Many users querying the same static document | **Yes (original use case)** | Build once, serve many |

---

## Code Changes Summary

| File | Change |
|---|---|
| `scripts/compress_conversation.py` | **New.** Full pipeline: supervision from turns → train cartridge |
| `scripts/update_cartridge.py` | **Rewritten.** Delta slot training with frozen base |
| `src/cartridges/core/cartridge.py` | **Added** `DeltaKVCartridge`, `initialize_delta_from_text`; **fixed** `TrainableKVCartridge.load()` for `num_frozen_tokens=0` |
| `src/cartridges/core/__init__.py` | Export new classes |

# KV Cartridges: Static Corpus and Dynamic Conversation Memory

## Background: Base Implementation

This work is built on top of [shreyansh26/cartridges](https://github.com/shreyansh26/cartridges),
a clean single-GPU reproduction of the KV cartridge idea from the HazyResearch Cartridges paper
(Eyuboglu et al., 2025). We use this implementation as the foundation for two reasons:

1. **It is a faithful, minimal reproduction.** The original HazyResearch codebase targets
   multi-GPU distributed training. shreyansh26's version strips that down to a single-GPU
   pipeline that is easier to instrument, modify, and reason about.

2. **It has a working end-to-end benchmark.** The `run_benchmark.py` orchestrator handles
   the full pipeline — vLLM server lifecycle, bootstrap generation, teacher supervision,
   cartridge training, retrieval routing, and eval — in a single script with clear entry points
   for each phase.

Our repo ([Shaho912/zaptridge](https://github.com/Shaho912/zaptridge)) extends this base with:
- Per-phase timing instrumentation
- Training step and bootstrap count ablations
- KVzip+ guided initialization experiments (see separate notes)
- Dynamic conversation cartridge pipeline
- Delta slot incremental update mechanism

All experiments run on a DGX server using an NVIDIA B200 MIG GPU with Qwen3-4B as the base model.

---

## Ablation Study: Training Steps

**Setup:** wikipedia_india corpus, 120 bootstrap questions, 1024-token cartridge budget.
Varying only `--train-steps`.

| Steps | Exact match | Semantic match | Train time | Total build |
|---|---|---|---|---|
| 240 (default) | 0.65 | 0.95 | 28.9s | 85.9s |
| 120 | 0.65 | 0.95 | 18.5s | 72.3s |
| **60** | **0.70** | **0.95** | **12.6s** | **70.2s** |
| 30 | 0.55 | 0.95 | 8.8s | 66.5s |

**Findings:**

- **60 steps is the sweet spot.** Exact match is equal to or better than 240 steps, training
  takes 57% less time, and total build time drops from 85.9s to 70.2s.
- **30 steps is too few.** Exact match degrades to 0.55 — the cartridge does not have enough
  gradient steps to converge.
- **Semantic match is stable at 0.95 across all configs.** The cartridge semantically
  preserves the corpus information even at 30 steps; only exact phrasing suffers.
- **The training loop is not the bottleneck.** Even eliminating training entirely saves
  only ~29s out of ~86s total. The fixed costs (bootstrap generation + training dataset
  construction) dominate.

Phase timing breakdown (at 240 steps, from instrumentation added to `run_benchmark.py`):

| Phase | Time | % of total |
|---|---|---|
| Bootstrap question generation | 25.1s | 29% |
| Build training dataset | 31.9s | 37% |
| Train cartridge (240 steps) | 28.9s | 34% |
| Baseline eval | 11.1s | 13% |
| Cartridge eval | 7.2s | 8% |

---

## Ablation Study: Bootstrap Question Count

**Setup:** wikipedia_india corpus, 60 train steps (optimal from above), 1024-token cartridge.
Varying only `--bootstrap-count`.

| Bootstrap | Exact match | Semantic match | Bootstrap time | Build dataset | Total build |
|---|---|---|---|---|---|
| **120 (default)** | **0.80** | **0.95** | 26.0s | 32.8s | **72.1s** |
| 60 | 0.50 | 0.95 | 12.3s | 18.4s | 46.7s |
| 30 | 0.60 | 0.95 | 7.4s | 10.9s | 30.0s |
| 15 | 0.60 | 0.95 | 5.6s | 6.9s | 23.2s |

**Findings:**

- **120 bootstrap questions is the minimum for good quality.** Exact match drops sharply
  to 0.50 at 60 questions. Below 60, quality partially recovers to 0.60 but does not improve
  further — there is a threshold effect rather than a smooth degradation.
- **Bootstrap count drives build time linearly.** Both bootstrap generation and training
  dataset construction scale proportionally: halving bootstrap count roughly halves both
  phases. Cutting from 120 → 15 reduces total build from 72s → 23s.
- **Semantic match is 0.95 everywhere.** Even with 15 bootstrap questions, the cartridge
  semantically preserves the corpus. The exact match gap reflects phrasing precision, not
  knowledge loss.
- **Optimal configuration: 120 bootstrap + 60 steps → 72s build, 0.80 exact / 0.95 semantic.**
  This beats the default (240 steps, 120 bootstrap) which takes 85.9s and achieves only 0.65
  exact match — faster and better.

---

## Memory Profiling and Mobile Feasibility (Qwen3-0.6B)

**Setup:** wikipedia_india corpus, 120 bootstrap questions, 1024-token cartridge. Memory measured
via `torch.cuda.max_memory_allocated()`, which tracks PyTorch-managed allocations in the local HF
process only. The vLLM server uses a separate memory allocator, so its footprint does not appear
in these numbers.

### Phase Memory Comparison

| Phase | 4B peak | 0.6B peak | Reduction |
|---|---|---|---|
| build_training_dataset | 14.4 GB | 7.3 GB | −49% |
| baseline_eval | 13.1 GB | 5.6 GB | −57% |
| train_cartridge | 9.8 GB | **2.2 GB** | −77% |
| cartridge_eval | 8.7 GB | **1.7 GB** | −81% |

### Step Count Does Not Transfer from 4B

The 60-step optimization found for 4B does not apply to 0.6B:

| Steps | Exact match | Semantic match | Build time |
|---|---|---|---|
| 240 | **0.40** | **0.60** | 89s |
| 60 | 0.25 | 0.40 | 75s |

**0.6B requires 240 steps minimum.** At 60 steps, cartridge quality falls back to baseline ICL
(0.25 exact). The smaller model needs more gradient steps to compress effectively — likely because
it has less representational capacity and the distillation signal takes longer to propagate into
the KV tensors.

### Quality Results (0.6B, 240 steps)

| | Baseline ICL | Cartridge (8× compressed) |
|---|---|---|
| Exact match | 0.25 | **0.40** |
| Semantic match | 0.50 | **0.60** |

The cartridge outperforms in-context learning despite 8× compression. This is unusual — for 4B,
full context is the ceiling. For 0.6B, the model performs worse with raw long context than with
the compressed KV representation. Small models are weaker at retrieving from long contexts; the
cartridge's direct attention pattern may be more effective than attending over raw text tokens.

### Mobile Feasibility (0.6B only)

| Phase | 0.6B peak | iPhone 16 Pro (8 GB) | iPhone 16 Pro Max (16 GB) |
|---|---|---|---|
| Cartridge inference (cartridge_eval) | **1.7 GB** | ✓ fits | ✓ fits |
| Cartridge training (train_cartridge) | **2.2 GB** | ✓ fits | ✓ fits |
| Full build pipeline (peak at build_training_dataset) | **7.3 GB** | Tight | ✓ fits |

The on-device inference path needs only 1.7 GB — viable on any modern iPhone. The overnight
build pipeline peaks at 7.3 GB during forward passes over the training dataset; this fits
on iPhone 16 Pro Max with margin, and is tight but potentially viable on iPhone 16 Pro (8 GB)
depending on OS overhead.

---

## Model Size Tradeoff: 0.6B vs 1.7B vs 4B

The three models profiled give a complete quality/memory tradeoff curve. All runs:
wikipedia_india corpus, 120 bootstrap questions, 1024-token cartridge, 240 train steps.

### Memory

| Phase | 0.6B | 1.7B | 4B |
|---|---|---|---|
| cartridge_eval (inference) | 1.7 GB | **3.9 GB** | 8.7 GB |
| train_cartridge | 2.2 GB | 4.5 GB | 9.8 GB |
| baseline_eval | 5.6 GB | 7.9 GB | 13.1 GB |
| build_training_dataset (peak) | 7.3 GB | 9.5 GB | 14.4 GB |

### Quality

| | Baseline exact | Cartridge exact | Baseline semantic | Cartridge semantic |
|---|---|---|---|---|
| 0.6B | 0.25 | 0.40 | 0.50 | 0.60 |
| 1.7B | 0.70 | 0.65 | 0.85 | **0.90** |
| 4B | ~0.90 | ~0.75 | ~1.00 | ~0.95 |

### Build Time (240 steps)

| Model | Build time |
|---|---|
| 0.6B | 89s |
| 1.7B | 115s |
| 4B (60 steps, optimal) | 72s |

### Findings

**1.7B is the sweet spot for mobile.** Cartridge inference costs 3.9 GB — fits on iPhone 16 Pro
(8 GB) with ~4 GB headroom for OS and app. Quality is close to 4B (+0.10 exact gap, +0.05
semantic gap) and dramatically better than 0.6B (+0.25 exact, +0.30 semantic).

**The quality gap from 4B narrows more than the memory gap.** Going 4B → 1.7B cuts inference
memory by 55% but loses only ~0.10 exact match. Going 1.7B → 0.6B cuts another 56% memory
but loses another 0.25 exact match — a much worse trade.

**Build pipeline fits iPhone 16 Pro Max only.** The 9.5 GB peak during build_training_dataset
fits 16 GB (Pro Max) with margin; 8 GB (Pro) cannot accommodate it. On-device overnight
training is a Pro Max feature.

**At 1.7B, cartridge semantic (0.90) exceeds baseline semantic (0.85)** despite 8× compression.
See explanation below.

### Why Cartridge Beats Baseline on Semantic Match at 1.7B

At 1.7B, the raw-context baseline reads 8192 tokens and achieves 0.85 semantic match; the
cartridge, despite compressing that context 8×, achieves 0.90. Three factors explain this:

1. **The cartridge was distilled on Q&A pairs, not raw text.** Training optimized the KV tensors
   specifically to reproduce the teacher's answers to the 120 bootstrap questions. The cartridge
   encodes *answerable signal* more densely than the surrounding prose. When the model attends
   to the cartridge, it gets a higher signal-to-noise ratio than reading the full document.

2. **1.7B struggles with long-context retrieval.** A 1.7B model reading 8192 tokens can be
   distracted by irrelevant passages, hedge with extra qualifiers, or blend information from
   nearby sentences. These artifacts do not change the meaning enough to fail exact match but
   they confuse the semantic judge. The cartridge's compressed representation sidesteps this by
   making the relevant information more directly accessible.

3. **The 0.6B model does not have this property.** At 0.6B, the cartridge also beats baseline
   on semantic (0.60 vs 0.50), and for the same reason — but the absolute numbers are low
   because the model is genuinely capacity-limited. At 4B, the baseline is strong enough
   that cartridge compression is a net cost on both metrics.

---

## Part 1: Static Corpus Cartridge

### What It Is

A KV cartridge is a compact set of learned key-value tensors that replaces a long document in
the model's attention mechanism. Instead of prefilling thousands of tokens on every query, the
model attends to a small fixed cache — the cartridge — that has been trained to encode the
document's information.

```
Without cartridge:  [8192-token Wikipedia article] + [user question] → answer
With cartridge:     [1024-slot cartridge KV]        + [user question] → answer
```

The cartridge is not a summary. It is a set of learned K/V tensors that the frozen base model
attends to. The model weights never change — only the cartridge tensors are optimized.

### How It Is Implemented

The pipeline has five phases:

**Phase 1 — Bootstrap question generation (vLLM)**
A vLLM server generates 120 diverse Q&A pairs from the corpus using a prompt that asks for
extractive, answerable questions. These pairs define what the cartridge needs to know.

**Phase 2 — Teacher answer generation (vLLM)**
The same vLLM server runs full-context forward passes: the entire 8192-token corpus is placed
in the system prompt and the model answers each bootstrap question. These answers become the
training targets.

**Phase 3 — Training dataset construction (HF model)**
The local HF model (Qwen3-4B) runs teacher-forcing on each question-answer pair with the full
corpus as context. This captures the per-token log-probability distribution (top-5 logprobs)
that the cartridge must learn to reproduce. vLLM cannot do this step because it does not expose
raw logits or KV cache internals.

**Phase 4 — Cartridge training**
The cartridge is initialized from the first `p` tokens of the corpus via a prefix forward pass.
Only the cartridge's K/V tensors are trainable — all base model weights are frozen.
For each training step, the model runs with the cartridge as `past_key_values` and the student's
logits are compared against the teacher's sparse distribution via cross-entropy loss
(sparse distillation).

```
loss = -Σ_t Σ_i teacher_prob_i * log student_prob_i
```

**Phase 5 — Evaluation**
The trained cartridge is evaluated against held-out questions using exact match and a semantic
judge. The baseline is the same model with the full corpus in context.

### Results (wikipedia_india corpus, Qwen3-4B)

| Config | Exact match | Semantic match | Build time | Compression |
|---|---|---|---|---|
| Full context (baseline) | 0.90 | 1.00 | — | 1× |
| Cartridge 1024 tokens | 0.75 | 0.90 | 72s | 8× |
| Cartridge 512 tokens  | 0.45 | 0.80 | 68s | 16× |

Optimal hyperparameters found via ablation:
- **Train steps:** 60 (vs default 240) — same quality, 57% faster training
- **Bootstrap count:** 120 questions — quality drops sharply below this

### Downsides

**1. Quality gap**
The 1024-token cartridge reaches 0.75 exact match vs 0.90 for full context. The cartridge
cannot perfectly encode 8192 tokens of information into 1024 K/V slots — some detail is lost.
Semantic match (0.90 vs 1.00) is closer, suggesting the gap is partly a phrasing issue rather
than missing facts.

**2. Build time**
72 seconds per corpus chunk. With a multi-chunk corpus this multiplies. The majority of build
time is fixed overhead: bootstrap generation (~25s) and training dataset construction (~32s).
The training loop itself takes only ~13s at 60 steps.

**3. Static by design**
The cartridge encodes the corpus as it was at build time. If the document changes, the
cartridge must be retrained. There is no mechanism to update it incrementally.

**4. Top-1 retrieval only**
At inference, one question routes to one cartridge. If the answer spans multiple corpus
chunks, the retriever selects one and the other is ignored. Cross-chunk questions degrade.

**5. Build pipeline requires vLLM**
Phases 1 and 2 require a running vLLM server for bootstrap and teacher answer generation.
This adds infrastructure overhead and means the full pipeline cannot run on the HF model alone.

---

## Part 2: Dynamic Conversation Cartridge

### Motivation

A static corpus cartridge compresses a fixed document once. The question this work asks is:
**can the same mechanism compress a live conversation history?**

As a conversation grows, every new turn requires the model to re-prefill the entire history.
At 8192 tokens of conversation, that prefill cost is the same as the corpus case the cartridge
was designed to eliminate. A conversation cartridge would replace the growing history with a
compact KV cache that future turns attend to instead.

### Key Insight

Conversation turns are structurally identical to the Q&A pairs used to train static cartridges.
Each (user, assistant) turn is already a supervised signal: given the session context, reproduce
this response. The supervision does not need to be generated — it already exists.

This eliminates two of the static pipeline's biggest dependencies:
- No vLLM server needed
- No bootstrap question generation needed

The local HF model is both the teacher (full-context forward pass) and the student (cartridge
training). The entire pipeline runs in a single process.

### How It Works

**Phase 1 — Supervision from conversation turns**

For each (user, assistant) pair in the conversation, the HF model runs a teacher-forcing
forward pass with the full conversation history as system context:

```
System:    [full conversation formatted as text]
User:      [user message for this turn]
Assistant: [actual assistant response — teacher forced]
```

This gives per-token logprobs for each assistant response. The cartridge must learn to
reproduce these logprobs without access to the raw conversation text.

**Phase 2 — Cartridge training**

Identical to the static corpus training. The cartridge is initialized from the first `p` tokens
of the formatted conversation text, then optimized via sparse distillation against the teacher
logprobs. The same `train_cartridge()` function is reused without modification.

**At inference:**

```
Without cartridge:  [turn 1][turn 2]...[turn N] + [new message] → response
With cartridge:     [cartridge KV]               + [new message] → response
```

The cartridge stands in for the full conversation history. The model never sees the raw turns —
it attends only to the compact KV state.

### Implementation

**`scripts/compress_conversation.py`** — new script, ~300 lines.

```bash
python scripts/compress_conversation.py \
    --conversation-path session.jsonl \
    --output-path session_cartridge.pt \
    --cartridge-tokens 1024 \
    --steps 60 \
    --device cuda:0
```

Input format — one JSON object per line:
```json
{"role": "user",      "content": "What is India's capital?"}
{"role": "assistant", "content": "New Delhi is the capital of India."}
```

The output is a standard `TrainableKVCartridge` `.pt` file compatible with all existing
eval and inference infrastructure.

### Results

**Short session — 5 turns**

| Metric | Value |
|---|---|
| Build time | **12.1s** (4.3s supervision + 7.8s training) |
| Cartridge tokens | 512 |
| Distillation loss | 0.0005 |
| In-session recall | **5/5** |
| Out-of-scope response | Correctly refused ("context does not mention...") |

The cartridge faithfully encoded all 5 turns. Out-of-scope questions were correctly refused
because the model had no parametric answer to fall back on.

**Longer session — 8 detailed turns (geography, cities, languages, parliament, economy,
space program, festivals, technology sector)**

| | 512 tokens | 1024 tokens |
|---|---|---|
| Build time | 22.6s | ~30s |
| Distillation loss | 0.1685 | lower |
| In-session recall | 8/9 | **9/9** |
| Fact confusion | GDP rank (5th) vs area rank (7th) mixed | None |
| Out-of-scope | Hallucinated (parametric knowledge) | Hallucinated |

At 512 tokens, the cartridge blended two similar numeric facts from different turns (GDP rank
and geographic area rank both appeared in the session). At 1024 tokens the confusion
disappeared — more slots allow cleaner separation of distinct facts.

The out-of-scope hallucination (India's current Prime Minister) is a model-level behavior:
when the model has a parametric answer it uses it regardless of the cartridge. This only
manifests for questions where the model genuinely knows the answer from training data.

### Comparison with Static Corpus Pipeline

| | Static corpus | Dynamic conversation |
|---|---|---|
| Supervision source | Bootstrap Q&A via vLLM | Actual turns via HF model |
| vLLM required | Yes | **No** |
| Bootstrap generation | Yes (120 questions) | **No** |
| Build time (8 turns / 8192-token doc) | 72s | 22s |
| Cartridge tokens needed | 1024 | 1024 |
| Quality | 0.75 exact / 0.90 semantic | 9/9 recall |

The conversation pipeline is significantly simpler and faster because the supervision is
already available in the conversation itself.

### Limitations

**Build cost vs amortization**
At 22s build cost and ~240ms prefill savings per turn, break-even requires ~90 queries on
the same compressed snapshot. Most conversations do not generate this many follow-up queries.

**Context windows are large**
Modern LLMs support 32K–128K token context windows. An 8-turn conversation is ~5,000 tokens
and fits without compression. Dynamic cartridges become valuable only when sessions genuinely
overflow the context window or when cross-session persistence is needed.

**Parametric knowledge hallucination**
The cartridge cannot prevent the model from answering questions it knows from training data.
Only questions with no parametric answer get the correct "context does not mention" refusal.

### When It Is Actually Useful

| Scenario | Useful? |
|---|---|
| Short conversations (< 20 turns, < 5K tokens) | No — context window handles it |
| Ultra-long sessions (100+ turns, 50K+ tokens) | **Yes** — context overflows |
| Cross-session persistent memory | **Yes** — compact, fast to load, opaque |
| Resource-constrained hardware (small context) | **Yes** — compression is necessary |
| Many users querying the same static document | **Yes** — original static use case |

---

## Code Changes Summary

| File | Type | Description |
|---|---|---|
| `scripts/compress_conversation.py` | New | Full pipeline: supervision from turns → train cartridge |
| `scripts/update_cartridge.py` | Rewritten | Delta slot training with frozen base cartridge |
| `src/cartridges/core/cartridge.py` | Modified | Added `DeltaKVCartridge`, `initialize_delta_from_text`; fixed `TrainableKVCartridge.load()` for `num_frozen_tokens=0` |
| `src/cartridges/core/__init__.py` | Modified | Export new classes |

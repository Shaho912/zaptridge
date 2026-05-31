# KV Cartridges: Static Corpus and Dynamic Conversation Memory

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

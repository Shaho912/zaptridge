"""KVzip+ token importance scoring via context reconstruction for guided Cartridge initialization.

The algorithm runs in two phases:

  1. Initial prefill — one forward pass on the full context to build the KV cache.
  2. Reconstruction — for each chunk, a second forward pass with a prompt like
     "Repeat the previous context exactly" is run against the cached KV.
     The cross-attention that reconstruction queries pay to each context position
     gives the importance score for that position.

KVzip+ normalization (enabled by default) scales the reconstruction attention weights by:
  - 1 / ||h||   — hidden-state norm of each reconstruction query position
  - ||Wo @ V||  — output-projection-scaled value norm for each context position

Score aggregation: max over reconstruction query positions and attention heads, then
mean over transformer layers.

IMPORTANT: Requires a model loaded with attn_implementation="eager". SDPA and
Flash-Attention do not expose attention weight tensors, so output_attentions=True
returns None and scoring will fail with a clear RuntimeError.

Reference implementation: https://github.com/NVIDIA/kvpress (Apache 2.0)
Paper: https://arxiv.org/abs/2505.23416
"""

import copy

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache

_FIRST_CHUNK_PROMPT = "\n\nRepeat the previous context exactly.\n"
_LATER_CHUNK_PROMPT = "\n\nRepeat the part of the previous context exactly, starting with"
_PREV_SUFFIX_LEN = 8  # tokens from previous chunk appended to the later-chunk prompt as a hint


class KVzipScorer:
    """Score tokens by KVzip+ importance using context reconstruction.

    Args:
        model: HuggingFace causal LM loaded with attn_implementation="eager".
        tokenizer: Matching tokenizer for the model.
        chunk_size: Tokens per reconstruction chunk. Default 2048 matches the NVIDIA
            reference implementation. Reduce if you see OOM.
        kvzip_plus: Enable KVzip+ normalization (||h|| and ||Wo@V|| scaling). Default True.
        n_sink: Leading tokens always assigned maximum score — these are the attention
            sinks preserved as frozen tokens in TrainableKVCartridge.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        chunk_size: int = 2048,
        kvzip_plus: bool = True,
        n_sink: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.kvzip_plus = kvzip_plus
        self.n_sink = n_sink

    @torch.no_grad()
    def score_tokens(self, text: str) -> torch.Tensor:
        """Compute KVzip+ importance score for every token in *text*.

        Returns a 1-D float32 CPU tensor of length T (tokens after tokenization,
        excluding special tokens). Higher score = more important.
        The first n_sink positions are always set to +inf so they survive top-k selection.
        """
        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        context_len = input_ids.shape[-1]

        initial_cache = self._prefill(input_ids)

        scores = torch.zeros(context_len, dtype=torch.float32)
        scores[: self.n_sink] = float("inf")

        chunk_start = 0
        for chunk_ids, repeat_ids in self._make_reconstruction_pairs(input_ids):
            chunk_len = chunk_ids.shape[-1]
            chunk_end = chunk_start + chunk_len
            scores[chunk_start:chunk_end] = self._score_chunk(
                repeat_ids=repeat_ids,
                initial_cache=initial_cache,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                context_len=context_len,
            )
            chunk_start = chunk_end

        return scores

    def top_token_indices(self, text: str, num_tokens: int) -> torch.Tensor:
        """Return indices of the top *num_tokens* tokens by importance, in document order.

        Document order matters: TrainableKVCartridge expects positional coherence so
        the frozen sink at index 0 stays first.
        If num_tokens >= sequence length every index is returned.
        """
        scores = self.score_tokens(text)
        if num_tokens >= scores.shape[0]:
            return torch.arange(scores.shape[0])
        _, top_indices = torch.topk(scores, num_tokens)
        return top_indices.sort().values

    # ------------------------------------------------------------------
    # Phase 1: initial prefill
    # ------------------------------------------------------------------

    def _prefill(self, input_ids: torch.Tensor) -> DynamicCache:
        outputs = self.model(
            input_ids=input_ids.to(self.model.device),
            use_cache=True,
        )
        return outputs.past_key_values

    # ------------------------------------------------------------------
    # Phase 2: reconstruction chunks
    # ------------------------------------------------------------------

    def _make_reconstruction_pairs(
        self, input_ids: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Split context into chunks and build (chunk_ids, reconstruction_input_ids) pairs."""
        ctx_len = input_ids.shape[-1]
        chunks = [
            input_ids[:, s : s + self.chunk_size]
            for s in range(0, ctx_len, self.chunk_size)
        ]

        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, chunk_ids in enumerate(chunks):
            if i == 0:
                prompt = self.tokenizer(
                    _FIRST_CHUNK_PROMPT, return_tensors="pt", add_special_tokens=False
                )["input_ids"]
            else:
                prompt = self.tokenizer(
                    _LATER_CHUNK_PROMPT, return_tensors="pt", add_special_tokens=False
                )["input_ids"]
                prev_hint = chunks[i - 1][:, -_PREV_SUFFIX_LEN:].to(chunk_ids.device)
                prompt = torch.cat([prompt.to(chunk_ids.device), prev_hint], dim=-1)

            repeat_ids = torch.cat([prompt.to(chunk_ids.device), chunk_ids], dim=-1)
            pairs.append((chunk_ids, repeat_ids))

        return pairs

    def _score_chunk(
        self,
        repeat_ids: torch.Tensor,
        initial_cache: DynamicCache,
        chunk_start: int,
        chunk_end: int,
        context_len: int,
    ) -> torch.Tensor:
        """Run one reconstruction pass and return scores for context[chunk_start:chunk_end].

        The reconstruction tokens attend to the full initial context KV cache via
        past_key_values. We extract attention columns corresponding to the current
        chunk and take the max over query positions and heads.
        """
        # Capture hidden states entering each attention layer for KVzip+ h_norm.
        hs_by_layer: dict[int, torch.Tensor] = {}
        hooks: list = []

        if self.kvzip_plus:
            for layer_idx, layer in enumerate(self.model.model.layers):
                def _make_hook(idx: int):
                    def _hook(module: nn.Module, args: tuple, kwargs: dict) -> None:
                        hs = kwargs.get("hidden_states")
                        if hs is None:
                            hs = args[0] if args else None
                        if hs is not None:
                            hs_by_layer[idx] = hs.detach()
                    return _hook
                hooks.append(
                    layer.self_attn.register_forward_pre_hook(_make_hook(layer_idx), with_kwargs=True)
                )

        # Deep-copy the cache: the reconstruction pass extends it with new tokens,
        # but we only want the original context KV for each independent chunk pass.
        cache_copy = _clone_cache(initial_cache)

        # Return any memory from the previous chunk's attention matrices back to CUDA
        # before allocating new ones — prevents fragmentation OOM across chunks.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            outputs = self.model(
                input_ids=repeat_ids.to(self.model.device),
                past_key_values=cache_copy,
                output_attentions=True,
                use_cache=True,
            )
        finally:
            for hook in hooks:
                hook.remove()
        del cache_copy

        if outputs.attentions is None:
            raise RuntimeError(
                "Model returned no attention weights. "
                "Reload the model with attn_implementation='eager' before scoring."
            )

        # Convert to a mutable list and drop the model output immediately so the
        # hidden-state tensors inside it can be freed before we process attention.
        # Then zero each entry after use so all 36 layers never live simultaneously.
        attentions: list = list(outputs.attentions)
        del outputs

        layer_scores: list[torch.Tensor] = []
        for layer_idx in range(len(attentions)):
            attn = attentions[layer_idx]
            attentions[layer_idx] = None  # release reference so GPU memory can be reclaimed

            # attn: [1, num_heads, q_len, context_len + q_len]
            # Slice to context positions only (reconstruction queries attending to context).
            ctx_attn = attn[..., :context_len].float()  # [1, num_heads, q_len, context_len]
            del attn

            if self.kvzip_plus:
                ctx_attn = self._apply_kvzip_plus(
                    ctx_attn=ctx_attn,
                    layer_idx=layer_idx,
                    hidden_states=hs_by_layer.get(layer_idx),
                    initial_cache=initial_cache,
                    context_len=context_len,
                )

            # Max over (num_heads, q_len) → [1, context_len] → slice chunk → [chunk_len]
            chunk_scores = ctx_attn[..., chunk_start:chunk_end].amax(dim=(-3, -2)).squeeze(0)
            layer_scores.append(chunk_scores.cpu())
            del ctx_attn, chunk_scores

        # Average importance across all transformer layers.
        return torch.stack(layer_scores).mean(dim=0)

    # ------------------------------------------------------------------
    # KVzip+ normalization
    # ------------------------------------------------------------------

    def _apply_kvzip_plus(
        self,
        ctx_attn: torch.Tensor,
        layer_idx: int,
        hidden_states: torch.Tensor | None,
        initial_cache: DynamicCache,
        context_len: int,
    ) -> torch.Tensor:
        """Scale reconstruction attention weights by 1/||h|| and ||Wo @ V||.

        ctx_attn shape in:  [1, num_heads, q_len, context_len]
        ctx_attn shape out: [1, num_heads, q_len, context_len]
        """
        # --- Step 1: divide by hidden-state norm of reconstruction query positions ---
        if hidden_states is not None:
            h_norm = hidden_states.float().norm(dim=-1)  # [1, q_len]
            # Broadcast over num_heads and context_len.
            ctx_attn = ctx_attn / h_norm[:, None, :, None].clamp(min=1e-8)

        # --- Step 2: multiply by ||Wo @ V|| for each context KV position ---
        attn_mod = self.model.model.layers[layer_idx].self_attn
        num_heads: int = self.model.config.num_attention_heads
        num_kv_heads: int = self.model.config.num_key_value_heads
        num_groups: int = num_heads // num_kv_heads
        head_dim: int = attn_mod.head_dim
        hidden_size: int = self.model.config.hidden_size

        # Wo: [hidden_size, num_heads * head_dim]
        # Reshape to [num_kv_heads, num_groups, head_dim, hidden_size] to match GQA layout.
        Wo = attn_mod.o_proj.weight.T.view(num_kv_heads, num_groups, head_dim, hidden_size).float()

        # V from initial cache: [1, num_kv_heads, context_len, head_dim]
        # NEW
        V = initial_cache.layers[layer_idx].values[:, :, :context_len].float()

        # WoV[b, h, g, t, j] = sum_i Wo[h, g, i, j] * V[b, h, t, i]
        # Shape: [1, num_kv_heads, num_groups, context_len, hidden_size]
        WoV = torch.einsum("h g i j, b h t i -> b h g t j", Wo, V)
        WoV_norm = WoV.norm(dim=-1)  # [1, num_kv_heads, num_groups, context_len]

        # Reshape ctx_attn to expose the GQA group dimension, scale, then flatten back.
        bsz, _, q_len, ctx = ctx_attn.shape
        ctx_attn = ctx_attn.view(bsz, num_kv_heads, num_groups, q_len, ctx)
        ctx_attn = ctx_attn * WoV_norm.unsqueeze(-2)  # broadcast over q_len
        ctx_attn = ctx_attn.view(bsz, num_heads, q_len, ctx)

        return ctx_attn


# ------------------------------------------------------------------
# Cache utilities
# ------------------------------------------------------------------

# NEW
def _clone_cache(cache: DynamicCache) -> DynamicCache:
    new_cache = DynamicCache()
    new_cache.layers = []
    for layer in cache.layers:
        new_layer = copy.copy(layer)
        new_layer.keys = layer.keys.clone()
        new_layer.values = layer.values.clone()
        new_cache.layers.append(new_layer)
    return new_cache

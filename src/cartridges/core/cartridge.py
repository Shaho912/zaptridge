"""Core trainable KV-cache object used as the cartridge representation.

A cartridge is a fixed-length prefix KV cache passed as ``past_key_values`` so later
tokens attend to a compact memory instead of the full document. Training (see
``cartridges.train.cartridge``) initializes it from the first ``num_tokens`` of the
chunk ``system_prompt`` via ``initialize_from_prefix_text``, then optimizes the
trainable tail while the frozen prefix keeps attention-sink behavior stable.

The important subtlety is that only the initialization is a literal prefix cache. After
distillation, the trainable slots are just learned K/V tensors that were seeded from the
prefix; they can move away from "token i in the original document" and become a compact
memory encoding facts from anywhere in the chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

if TYPE_CHECKING:
    from cartridges.core.kvzip_scoring import KVzipScorer


@dataclass(frozen=True)
class AttentionShape:
    """Minimal attention metadata needed to size canonical KV-cache tensors."""
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    dtype_bytes: int = 2


def _normalize_past_key_values(past_key_values) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Convert cache objects into the legacy tuple-of-tuples format used internally."""
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    return [(layer[0], layer[1]) for layer in past_key_values]


class TrainableKVCartridge(nn.Module):
    """Per-layer K/V tensors for one compressed prefix, split into frozen sink + trainable body.

    ``num_tokens`` is the sequence length represented by this cache (one position per
    prefix token used when the cache was created or last aligned). ``as_cache`` wraps
    these tensors for HuggingFace ``past_key_values`` during forward passes.
    """

    def __init__(
        self,
        keys: Iterable[torch.Tensor],
        values: Iterable[torch.Tensor],
        num_frozen_tokens: int = 1,
    ):
        """Split per-layer keys/values into frozen and trainable token regions."""
        super().__init__()
        keys = [tensor.detach().clone() for tensor in keys]
        values = [tensor.detach().clone() for tensor in values]
        if len(keys) != len(values):
            raise ValueError("keys and values must have the same number of layers.")

        self.num_layers = len(keys)
        self.num_tokens = keys[0].shape[-2]
        self.num_frozen_tokens = num_frozen_tokens
        self.trainable_keys = nn.ParameterList()
        self.trainable_values = nn.ParameterList()
        self.frozen_keys = nn.ParameterList()
        self.frozen_values = nn.ParameterList()

        for key_tensor, value_tensor in zip(keys, values, strict=True):
            # Keep the first few positions fixed so training does not destroy the initial sink tokens.
            if num_frozen_tokens > 0:
                frozen_key = nn.Parameter(key_tensor[..., :num_frozen_tokens, :], requires_grad=False)
                frozen_value = nn.Parameter(value_tensor[..., :num_frozen_tokens, :], requires_grad=False)
                train_key = nn.Parameter(key_tensor[..., num_frozen_tokens:, :])
                train_value = nn.Parameter(value_tensor[..., num_frozen_tokens:, :])
                self.frozen_keys.append(frozen_key)
                self.frozen_values.append(frozen_value)
            else:
                train_key = nn.Parameter(key_tensor)
                train_value = nn.Parameter(value_tensor)
            self.trainable_keys.append(train_key)
            self.trainable_values.append(train_value)

    @property
    def num_trainable_tokens(self) -> int:
        """Return the number of token positions optimized during training."""
        return self.num_tokens - self.num_frozen_tokens

    def layer(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize one layer by concatenating frozen and trainable token blocks."""
        key_parts = []
        value_parts = []
        if self.num_frozen_tokens > 0:
            key_parts.append(self.frozen_keys[index])
            value_parts.append(self.frozen_values[index])
        key_parts.append(self.trainable_keys[index])
        value_parts.append(self.trainable_values[index])
        return torch.cat(key_parts, dim=-2), torch.cat(value_parts, dim=-2)

    def as_legacy_past_key_values(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Expose the cartridge in the cache tuple format expected by older HF helpers."""
        return tuple(self.layer(index) for index in range(self.num_layers))

    def as_cache(self, model_config) -> DynamicCache:
        """Wrap the legacy cache into the ``DynamicCache`` object Transformers expects."""
        return DynamicCache(ddp_cache_data=self.as_legacy_past_key_values(), config=model_config)

    def canonical_kv_bytes(self) -> int:
        """Measure the byte footprint of the cartridge K/V tensors exactly as stored."""
        total = 0
        for key, value in self.as_legacy_past_key_values():
            total += key.numel() * key.element_size()
            total += value.numel() * value.element_size()
        return total

    def save(self, path: str | Path) -> None:
        """Persist the cartridge tensors without optimizer or scheduler state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "num_frozen_tokens": self.num_frozen_tokens,
                "keys": [tensor.detach().cpu() for tensor in self.trainable_keys],
                "values": [tensor.detach().cpu() for tensor in self.trainable_values],
                "frozen_keys": [tensor.detach().cpu() for tensor in self.frozen_keys],
                "frozen_values": [tensor.detach().cpu() for tensor in self.frozen_values],
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "TrainableKVCartridge":
        """Load a saved cartridge artifact and optionally move it onto a target device."""
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        keys = []
        values = []
        num_frozen_tokens = checkpoint["num_frozen_tokens"]
        for frozen_key, train_key, frozen_value, train_value in zip(
            checkpoint["frozen_keys"],
            checkpoint["keys"],
            checkpoint["frozen_values"],
            checkpoint["values"],
            strict=True,
        ):
            key = torch.cat([frozen_key, train_key], dim=-2) if num_frozen_tokens else train_key
            value = torch.cat([frozen_value, train_value], dim=-2) if num_frozen_tokens else train_value
            keys.append(key)
            values.append(value)
        cartridge = cls(keys=keys, values=values, num_frozen_tokens=num_frozen_tokens)
        if device is not None:
            cartridge.to(device)
        return cartridge


@torch.no_grad()
def initialize_from_prefix_text(
    model,
    tokenizer,
    text: str,
    num_tokens: int,
    num_frozen_tokens: int = 1,
) -> TrainableKVCartridge:
    """Run a one-shot prefix forward on ``text`` and capture KV as the starting cartridge.

    In the benchmark trainer, ``text`` is typically ``TrainingExample.system_prompt``
    (context + instructions), not the user question; distillation then teaches the
    cartridge to help answer when that same system prompt is paired with user messages.

    Only the first ``num_tokens`` of ``text`` are used here, so this function gives the
    cartridge a reasonable starting point, not its final semantics. Later gradient updates
    push the trainable K/V slots to mimic full-context teacher behavior, which is how a
    512/1024-slot cartridge can come to represent information from later parts of a much
    longer chunk.
    """
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][..., :num_tokens].to(model.device)
    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = _normalize_past_key_values(outputs.past_key_values)
    keys = [layer[0].detach().to(model.dtype) for layer in past_key_values]
    values = [layer[1].detach().to(model.dtype) for layer in past_key_values]
    return TrainableKVCartridge(keys=keys, values=values, num_frozen_tokens=num_frozen_tokens)


@torch.no_grad()
def initialize_from_kvzip_scores(
    scorer: KVzipScorer,
    model,
    tokenizer,
    text: str,
    num_tokens: int,
    num_frozen_tokens: int = 1,
    prefix_tokens: int = 256,
) -> TrainableKVCartridge:
    """Hybrid-init cartridge: fixed prefix + KVzip+-selected tail tokens.

    The budget is split into two parts:

    1. **Prefix** — the first ``prefix_tokens`` positions of the chunk, taken
       unconditionally.  These provide coherent discourse structure (titles,
       topic sentences, entity introductions) that scattered KVzip+ tokens
       cannot supply on their own.

    2. **KVzip+ tail** — the remaining ``num_tokens - prefix_tokens`` slots are
       filled with the highest-importance tokens from the rest of the document,
       scored via context reconstruction.

    Both parts are combined in document order and a single fresh forward pass
    gives them sequential positions 0..num_tokens-1, preserving RoPE correctness.

    If ``prefix_tokens >= num_tokens`` the function falls back to a plain prefix
    init (equivalent to ``initialize_from_prefix_text``).

    Args:
        scorer: A ``KVzipScorer`` whose model must be loaded with attn_implementation="eager".
        model: The same model instance passed to the scorer.
        tokenizer: Matching tokenizer.
        text: Full corpus chunk (typically ``TrainingExample.system_prompt``).
        num_tokens: Total KV positions to keep (the cartridge budget ``p``).
        num_frozen_tokens: Leading positions frozen as attention sinks during training.
        prefix_tokens: How many leading tokens to keep unconditionally. Default 256.
    """
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.device)
    context_len = input_ids.shape[-1]

    prefix_tokens = min(prefix_tokens, num_tokens, context_len)
    kvzip_budget = num_tokens - prefix_tokens  # slots filled by importance scoring

    # Fast path: entire budget is prefix — no scoring needed.
    if kvzip_budget <= 0:
        return initialize_from_prefix_text(
            model=model,
            tokenizer=tokenizer,
            text=text,
            num_tokens=num_tokens,
            num_frozen_tokens=num_frozen_tokens,
        )

    # One prefill shared across scoring and initialization.
    initial_cache = scorer._prefill(input_ids)

    # Reconstruction scoring over the full context.
    scores = torch.zeros(context_len, dtype=torch.float32)
    scores[: scorer.n_sink] = float("inf")

    chunk_start = 0
    for chunk_ids, repeat_ids in scorer._make_reconstruction_pairs(input_ids):
        chunk_end = chunk_start + chunk_ids.shape[-1]
        scores[chunk_start:chunk_end] = scorer._score_chunk(
            repeat_ids=repeat_ids,
            initial_cache=initial_cache,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            context_len=context_len,
        )
        chunk_start = chunk_end

    # --- Part 1: prefix indices (always 0..prefix_tokens-1) ---
    prefix_range = torch.arange(prefix_tokens)

    # --- Part 2: top-kvzip_budget indices from the tail only ---
    tail_scores = scores[prefix_tokens:]  # shape: [context_len - prefix_tokens]
    tail_len = tail_scores.shape[0]
    if kvzip_budget >= tail_len:
        # Budget covers the entire tail — take all of it.
        kvzip_indices = torch.arange(prefix_tokens, context_len)
    else:
        _, rel_indices = torch.topk(tail_scores, kvzip_budget)
        # Convert relative tail indices back to absolute positions, then sort.
        kvzip_indices = (rel_indices + prefix_tokens).sort().values

    # Merge both parts in document order (no duplicates — ranges are disjoint).
    all_indices = torch.cat([prefix_range, kvzip_indices]).sort().values

    # Fresh forward pass on the selected tokens gives sequential positions
    # 0..num_tokens-1, preserving RoPE correctness (see RoPE note in
    # initialize_from_prefix_text).
    selected_ids = input_ids[..., all_indices.to(model.device)]
    new_outputs = model(input_ids=selected_ids, use_cache=True)
    past_key_values = _normalize_past_key_values(new_outputs.past_key_values)
    keys = [layer[0].detach().to(model.dtype) for layer in past_key_values]
    values = [layer[1].detach().to(model.dtype) for layer in past_key_values]
    return TrainableKVCartridge(keys=keys, values=values, num_frozen_tokens=num_frozen_tokens)

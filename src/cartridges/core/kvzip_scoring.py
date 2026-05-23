"""KVzip+ token importance scoring for guided Cartridge initialization.

KVzip+ score for token k:
    score(k) = mean_layers( mean_heads( sum_{q >= k} attn[q, k] ) )

This is the causal column-sum of the attention matrix — how much total attention
token k received from all later positions — averaged over heads and layers.

IMPORTANT: Requires a model loaded with attn_implementation="eager". SDPA and
Flash-Attention do not expose attention weight tensors, so output_attentions=True
will return None and scoring will fail with a clear RuntimeError.
"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class KVzipScorer:
    """Score tokens by KVzip+ column-sum importance, with chunked processing for long sequences.

    Args:
        model: HuggingFace causal LM loaded with attn_implementation="eager".
        tokenizer: Matching tokenizer for the model.
        chunk_size: Maximum sequence length processed in one forward pass. Default
            4096 is sized for a B200 MIG90 instance with ~90 GB VRAM; reduce if
            you see OOM on smaller partitions.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        chunk_size: int = 4096,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    @torch.no_grad()
    def score_tokens(self, text: str) -> torch.Tensor:
        """Compute KVzip+ importance score for every token in *text*.

        Returns a 1-D float32 CPU tensor of length T (number of tokens after
        tokenization, excluding special tokens). Higher score means the token
        received more total attention across all layers and heads.
        """
        input_ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]  # [1, T]
        seq_len = input_ids.shape[-1]

        if seq_len <= self.chunk_size:
            return self._score_chunk(input_ids)
        return self._score_chunked(input_ids, seq_len)

    def top_token_indices(self, text: str, num_tokens: int) -> torch.Tensor:
        """Return indices of the *num_tokens* highest-scoring tokens, in document order.

        The returned indices index into the tokenized sequence (same coordinate
        space as the KV cache positions). Preserving document order matters for
        initialization: the Cartridge expects positional coherence in its sink token.

        If *num_tokens* >= sequence length every index is returned.
        """
        scores = self.score_tokens(text)
        if num_tokens >= scores.shape[0]:
            return torch.arange(scores.shape[0])
        _, top_indices = torch.topk(scores, num_tokens)
        return top_indices.sort().values  # stable document order

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_chunk(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run one forward pass on a chunk and return per-token importance scores.

        Args:
            input_ids: [1, chunk_len] tensor (CPU or GPU).

        Returns:
            1-D float32 CPU tensor of length chunk_len.
        """
        input_ids = input_ids.to(self.model.device)
        outputs = self.model(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False,
        )

        if outputs.attentions is None:
            raise RuntimeError(
                "Model returned no attention weights. "
                "Reload the model with attn_implementation='eager' before scoring."
            )

        # outputs.attentions: tuple of num_layers tensors, each shaped [1, H, T, T].
        # Causal masking guarantees attn[q, k] == 0 for all q < k, so the full
        # column sum equals the causal column sum (sum_{q >= k} attn[q, k]).
        layer_scores: list[torch.Tensor] = []
        for attn in outputs.attentions:
            # attn: [1, H, T, T]
            col_sums = attn.sum(dim=-2)       # [1, H, T] — sum over query dim
            head_mean = col_sums.mean(dim=1)  # [1, T]   — mean over heads
            layer_scores.append(head_mean)

        # Stack layers and average: [L, 1, T] -> mean over L -> [1, T] -> [T]
        scores = torch.stack(layer_scores, dim=0).mean(dim=0).squeeze(0)
        return scores.float().cpu()

    def _score_chunked(self, input_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Score a long sequence by tiling non-overlapping chunks of *chunk_size*.

        Cross-chunk attention is not captured: a token at position i inside chunk
        c receives no credit for being attended to by tokens in chunk c+1 or later.
        This is an approximation that trades recall accuracy for constant GPU memory.
        For the KVzip+ use case the error is small because attention decays with
        distance and later chunks contribute diminishing weight to early tokens.
        """
        scores = torch.zeros(seq_len, dtype=torch.float32)
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk_scores = self._score_chunk(input_ids[:, start:end])
            scores[start:end] = chunk_scores
        return scores

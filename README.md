# Cartridges + KVzip+

This repo is inspired by [shreyansh26/cartridges](https://github.com/shreyansh26/cartridges), a clean single-GPU reproduction of the Cartridges idea from the [HazyResearch paper](https://arxiv.org/abs/2506.06266). It extends that baseline by attempting to implement **KVzip+ guided initialization**: instead of seeding the Cartridge from the first `p` tokens of the corpus, tokens are ranked by KVzip+ importance scores and the top-`p` scoring tokens are used for initialization. The goal is to improve exact-match quality at the same compression ratio.

The core idea remains the same: compress a long context into a trainable KV cache, then answer many follow-up questions against that compact cache instead of repeatedly paying full-context prefill cost.

#!/usr/bin/env python3
"""Compare key and value tensors across two trained cartridges.

Computes cross-cartridge cosine similarity per layer and averaged, answering:
  "Do two cartridges trained on different corpora end up with similar keys?"

If keys converge to similar positions regardless of corpus (Diaz hypothesis),
cross-cartridge key cosim should be high. If values encode corpus-specific
content, cross-cartridge value cosim should be lower than key cosim.

Usage:
    python scripts/compare_cartridge_keys.py \
        --cart-a outputs/ablation_p01_vs_p02_medinit/cart_a.pt \
        --cart-b outputs/ablation_p01_vs_p02_medinit/cart_b.pt \
        --label-a "patient1" --label-b "patient2"
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.core import TrainableKVCartridge  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-cartridge key/value cosine similarity")
    parser.add_argument("--cart-a", required=True, help="Path to first cartridge .pt file")
    parser.add_argument("--cart-b", required=True, help="Path to second cartridge .pt file")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--per-layer", action="store_true", help="Print per-layer breakdown")
    args = parser.parse_args()

    cart_a = TrainableKVCartridge.load(args.cart_a)
    cart_b = TrainableKVCartridge.load(args.cart_b)

    if cart_a.num_layers != cart_b.num_layers:
        print("Error: cartridges have different layer counts", file=sys.stderr)
        return 1

    key_sims, val_sims = [], []
    for i in range(cart_a.num_layers):
        ka, va = cart_a.layer(i)
        kb, vb = cart_b.layer(i)

        ka_f = ka.reshape(-1).float()
        kb_f = kb.reshape(-1).float()
        va_f = va.reshape(-1).float()
        vb_f = vb.reshape(-1).float()

        key_sims.append(float(F.cosine_similarity(ka_f.unsqueeze(0), kb_f.unsqueeze(0)).item()))
        val_sims.append(float(F.cosine_similarity(va_f.unsqueeze(0), vb_f.unsqueeze(0)).item()))

    mean_k = sum(key_sims) / len(key_sims)
    mean_v = sum(val_sims) / len(val_sims)

    print(f"\nCross-cartridge similarity  [{args.label_a}] vs [{args.label_b}]")
    print(f"  {cart_a.num_layers} layers, {cart_a.num_tokens} slots each")
    print(f"\n  {'':30}  {'Key cosim':>10}  {'Val cosim':>10}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}")
    print(f"  {'Mean (all layers)':<30}  {mean_k:>10.6f}  {mean_v:>10.6f}")
    print(f"  {'Min  (most different layer)':<30}  {min(key_sims):>10.6f}  {min(val_sims):>10.6f}")
    print(f"  {'Max  (most similar layer)':<30}  {max(key_sims):>10.6f}  {max(val_sims):>10.6f}")

    if mean_k > mean_v:
        diff = mean_k - mean_v
        print(f"\n  Keys are MORE similar across cartridges than values (+{diff:.6f})")
        print(f"  → supports Diaz: keys converge to shared routing positions,")
        print(f"    values diverge to encode corpus-specific content.")
    else:
        diff = mean_v - mean_k
        print(f"\n  Values are MORE similar across cartridges than keys (+{diff:.6f})")
        print(f"  → keys are more corpus-specific than values for this pair.")

    if args.per_layer:
        print(f"\n  Per-layer breakdown:")
        print(f"  {'Layer':>5}  {'Key cosim':>10}  {'Val cosim':>10}  {'Keys more similar?':>18}")
        print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*18}")
        for i, (k, v) in enumerate(zip(key_sims, val_sims)):
            print(f"  {i:>5}  {k:>10.6f}  {v:>10.6f}  {'YES' if k > v else 'no':>18}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

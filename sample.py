"""Generate stories from the trained model.

Usage:
  python sample.py "Once upon a time, a little fox"
  python sample.py --n 3 --temperature 0.9
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="Once upon a time")
    ap.add_argument("--weights", default="out/model.pt")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(str(Path(__file__).parent / "tokenizer.json"))
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.float().to(device).eval()

    eot = tok.token_to_id("<|endoftext|>")
    start = torch.tensor([[eot] + tok.encode(args.prompt).ids], device=device)
    for _ in range(args.n):
        out = model.generate(start, args.max_new_tokens, args.temperature, args.top_k, stop_token=eot)
        ids = [i for i in out[0, 1:].tolist() if i != eot]
        print(tok.decode(ids).strip())
        print("-" * 60)


if __name__ == "__main__":
    main()

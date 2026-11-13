"""Download TinyStories, train a BPE tokenizer on it, and encode it to disk.

Writes tokenizer.json next to this file and train.bin / val.bin (uint16 token
ids, stories separated by the end-of-text token) into --data-dir.

Usage: python prepare_data.py --data-dir data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"


def train_tokenizer(texts, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    return tok


def encode_split(split, tok: Tokenizer, path: Path, batch: int = 20000) -> int:
    eot = tok.token_to_id(EOT)
    total = 0
    with path.open("wb") as f:
        for rows in split.iter(batch_size=batch):
            texts = [t.strip() for t in rows["text"] if t.strip()]
            ids = [np.asarray(e.ids + [eot], dtype=np.uint16) for e in tok.encode_batch(texts)]
            arr = np.concatenate(ids)
            arr.tofile(f)
            total += len(arr)
            print(f"  {path.name}: {total / 1e6:.0f}M tokens", end="\r", flush=True)
    print()
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--tokenizer-stories", type=int, default=300_000,
                    help="how many stories the tokenizer is trained on")
    args = ap.parse_args()
    assert args.vocab_size <= 65536, "token ids are stored as uint16"

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories")

    print("training tokenizer...")
    n = min(args.tokenizer_stories, len(ds["train"]))
    sample = (rows["text"] for rows in ds["train"].select(range(n)).iter(batch_size=1000))
    tok = train_tokenizer(sample, args.vocab_size)
    tok.save(str(Path(__file__).parent / "tokenizer.json"))
    print(f"vocab size {tok.get_vocab_size()}")

    for name, split in (("val", ds["validation"]), ("train", ds["train"])):
        total = encode_split(split, tok, data_dir / f"{name}.bin")
        print(f"{name}: {len(split):,} stories, {total:,} tokens")


if __name__ == "__main__":
    main()

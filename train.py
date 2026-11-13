"""Train the GPT from scratch on TinyStories.

Single GPU, bf16 autocast, AdamW with warmup and cosine decay. The schedule
follows whichever is further along, the step count or the wall clock, so the
run always finishes inside --max-hours. Checkpoints allow --resume.

Usage: python train.py --data-dir data --max-hours 6
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def keep_awake() -> None:
    """Ask Windows not to sleep while training (closing the lid still does)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def get_batch(data: np.memmap, batch_size: int, block_size: int, device, gen: torch.Generator):
    """Random windows of block_size tokens; the target is the input shifted by one."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen).tolist()
    x = np.stack([data[s:s + block_size] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + 1 + block_size] for s in starts]).astype(np.int64)
    return (torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True))


@torch.no_grad()
def estimate_val_loss(model, data, batch_size: int, block_size: int, device, batches: int) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(1234)  # same windows every time
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(data, batch_size, block_size, device, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / batches


@torch.no_grad()
def sample_story(model, tok: Tokenizer, device, prompt: str = "Once upon a time") -> str:
    model.eval()
    eot = tok.token_to_id("<|endoftext|>")
    idx = torch.tensor([[eot] + tok.encode(prompt).ids], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(idx, 120, temperature=0.8, top_k=50, stop_token=eot)
    model.train()
    return tok.decode(out[0, 1:].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="out", help="checkpoints and the training log")
    ap.add_argument("--export", default="out/model.pt", help="final weights for sample.py")
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    keep_awake()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "last.pt"

    tok = Tokenizer.from_file(str(Path(__file__).parent / "tokenizer.json"))
    train_data = np.memmap(Path(args.data_dir) / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(Path(args.data_dir) / "val.bin", dtype=np.uint16, mode="r")

    resuming = args.resume and ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False) if resuming else None
    cfg = GPTConfig(**ckpt["config"]) if resuming else GPTConfig(
        vocab_size=tok.get_vocab_size(), block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(cfg).to(device)
    print(f"model: {model.num_params() / 1e6:.1f}M parameters, "
          f"train tokens: {len(train_data) / 1e6:.0f}M, val tokens: {len(val_data) / 1e6:.1f}M")

    # Weight decay on matrices only, not on LayerNorm gains.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=True)

    step, elapsed_before, best_val = 0, 0.0, float("inf")
    if resuming:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step, elapsed_before, best_val = ckpt["step"], ckpt["elapsed"], ckpt["best_val"]
        print(f"resumed at step {step} ({elapsed_before / 3600:.2f} h used)")
        del ckpt
    budget = args.max_hours * 3600
    tokens_per_step = args.batch_size * cfg.block_size
    print(f"up to {args.max_steps} steps of {tokens_per_step} tokens, budget {args.max_hours} h")

    def save(path: Path) -> None:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "config": asdict(cfg),
                    "step": step, "best_val": best_val,
                    "elapsed": elapsed_before + time.time() - t0}, path)

    def export() -> None:
        path = Path(args.export)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {k: v.half() for k, v in model.state_dict().items()},
                    "config": asdict(cfg), "step": step, "val_loss": best_val}, path)

    def evaluate() -> None:
        nonlocal best_val
        vl = estimate_val_loss(model, val_data, args.batch_size, cfg.block_size, device, args.eval_batches)
        improved = vl < best_val
        best_val = min(best_val, vl)
        save(ckpt_path)
        if improved:
            export()
        rec = {"step": step, "val_loss": round(vl, 4), "best": improved,
               "hours": round((elapsed_before + time.time() - t0) / 3600, 3)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print("  sample:", sample_story(model, tok, device).replace("\n", " ")[:400], flush=True)

    log = (out / "log.jsonl").open("a", encoding="utf-8")
    gen = torch.Generator().manual_seed(args.seed + step)
    model.train()
    t0 = time.time()
    run_loss, run_n, t_log = 0.0, 0, time.time()
    while True:
        elapsed = elapsed_before + time.time() - t0
        progress = max(step / args.max_steps, elapsed / budget)
        if progress >= 1.0:
            break
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        lr = (args.min_lr + (args.lr - args.min_lr) * cosine) * min(1.0, (step + 1) / args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, cfg.block_size, device, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        run_loss += loss.item()
        run_n += 1

        if step % 50 == 0:
            dt = time.time() - t_log
            rec = {"step": step, "loss": round(run_loss / run_n, 4), "lr": round(lr, 6),
                   "tok_per_s": round(run_n * tokens_per_step / dt),
                   "tokens_seen_M": round(step * tokens_per_step / 1e6),
                   "hours": round(elapsed / 3600, 3),
                   "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run_loss, run_n, t_log = 0.0, 0, time.time()

        if step % args.eval_every == 0:
            evaluate()
            t_log = time.time()

    evaluate()
    print(f"done: {step} steps, best val loss {best_val:.4f}, weights in {args.export}")
    log.close()


if __name__ == "__main__":
    main()

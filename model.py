"""A small GPT, written from scratch.

Decoder-only transformer: token + learned position embeddings, pre-norm
blocks of causal self-attention and an MLP, and an output layer that shares
its weights with the token embedding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 512   # longest context the model can see
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, C / heads)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # Each position attends only to itself and earlier positions.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.proj(F.gelu(self.fc(self.ln2(x)))))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=False)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init)
        # Scale down the layers that write into the residual stream, so the
        # stream's variance does not grow with depth at initialisation.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if targets is None:
            return self.head(x[:, -1:]), None  # only the next-token logits
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 50, stop_token: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1] / max(temperature, 1e-6)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits[logits < kth] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits.float(), dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_token is not None and idx.size(0) == 1 and nxt.item() == stop_token:
                break
        return idx

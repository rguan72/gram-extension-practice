"""Tiny GPT. Your job: add GRAM auxiliary modules to the MLPs.

The checker needs this interface to keep working:
    GPT(cfg).forward(tokens, active_modules=[...]) -> logits (B, T, vocab)
        active_modules: module ids switched on for the whole batch; None/[] = all off
    GPT.module_parameters()  -> parameters that belong to auxiliary modules
    GPT.general_parameters() -> everything else
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import VOCAB_SIZE, SEQ_LEN


@dataclass
class Config:
    vocab_size: int = VOCAB_SIZE
    seq_len: int = SEQ_LEN + 1
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_mlp: int = 256
    n_modules: int = 0     # number of auxiliary modules per MLP
    d_module: int = 32     # neurons per auxiliary module


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, -1).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    """Plain MLP. TODO: add n_modules blocks of d_module auxiliary neurons whose
    contribution can be switched on/off per sequence."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.w_in = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.w_out = nn.Linear(cfg.d_mlp, cfg.d_model)

    def forward(self, x, module_mask=None):
        return self.w_out(F.gelu(self.w_in(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, module_mask=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x), module_mask)
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def module_parameters(self):
        return []  # TODO

    def general_parameters(self):
        return list(self.parameters())  # TODO

    def forward(self, tokens, active_modules=None, module_mask=None):
        """tokens: (B, T) long.

        active_modules : module ids switched on for every sequence in the batch
                         (this is what the checker uses). None or [] = all off.
        module_mask    : TODO -- (B, n_modules) per-sequence mask for training.
        """
        B, T = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb(torch.arange(T))
        for blk in self.blocks:
            x = blk(x, module_mask)
        return self.head(self.ln_f(x))

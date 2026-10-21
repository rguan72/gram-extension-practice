"""Tiny GPT with GRAM auxiliary modules in every MLP (reference solution)."""
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
    """General MLP neurons plus n_modules blocks of auxiliary neurons.

    module_mask: (B, n_modules) float in {0, 1}. Module m's neurons contribute
    to sequence b's output only where module_mask[b, m] == 1.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.w_in = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.w_out = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.n_modules = cfg.n_modules
        if cfg.n_modules:
            self.mod_in = nn.Parameter(torch.randn(cfg.n_modules, cfg.d_model, cfg.d_module) * cfg.d_model ** -0.5)
            self.mod_b = nn.Parameter(torch.zeros(cfg.n_modules, cfg.d_module))
            self.mod_out = nn.Parameter(torch.zeros(cfg.n_modules, cfg.d_module, cfg.d_model))

    def forward(self, x, module_mask=None):
        y = self.w_out(F.gelu(self.w_in(x)))
        if self.n_modules and module_mask is not None:
            h = F.gelu(torch.einsum("btd,mdk->bmtk", x, self.mod_in) + self.mod_b[None, :, None, :])
            h = h * module_mask[:, :, None, None]
            y = y + torch.einsum("bmtk,mkd->btd", h, self.mod_out)
        return y


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
        return [p for n, p in self.named_parameters() if ".mod_" in n]

    def general_parameters(self):
        return [p for n, p in self.named_parameters() if ".mod_" not in n]

    def forward(self, tokens, active_modules=None, module_mask=None):
        """tokens: (B, T) long.

        active_modules : list of module ids enabled for every sequence in the
                         batch (used by the checker). None or [] -> all modules
                         off.
        module_mask    : (B, n_modules) per-sequence mask (used in training).
                         Takes precedence over active_modules.
        """
        B, T = tokens.shape
        if module_mask is None and self.cfg.n_modules:
            module_mask = torch.zeros(B, self.cfg.n_modules, device=tokens.device)
            for m in active_modules or []:
                module_mask[:, m] = 1.0
        x = self.tok_emb(tokens) + self.pos_emb(torch.arange(T, device=tokens.device))
        for blk in self.blocks:
            x = blk(x, module_mask)
        return self.head(self.ln_f(x))

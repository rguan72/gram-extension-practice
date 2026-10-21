"""Training loop. Your job: turn this into GRAM training.

Right now this trains the plain GPT on every sequence and ignores the labels
(i.e. it is the all-data baseline). `check.py` calls `train_gram(seed)`.
"""
import time

import torch

from data import BATCH_SIZE, TRAIN_STEPS, DUAL_USE, sample_batch, sequence_losses, domain_loss
from model import Config, GPT

LR = 3e-3


def train_gram(seed: int = 0, steps: int = TRAIN_STEPS, log=print) -> GPT:
    torch.manual_seed(seed)
    cfg = Config(n_modules=len(DUAL_USE))   # TODO: modules are ignored by the model for now
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    for step in range(steps):
        tokens, labels = sample_batch(BATCH_SIZE, gen)   # labels: (B,) 0=core, 1=A, 2=B
        # TODO: per-sequence module mask, per-sequence gradient routing
        losses = sequence_losses(model(tokens), tokens)   # (B,)
        loss = losses.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if log and (step % 300 == 0 or step == steps - 1):
            log(f"step {step:4d}  loss {loss.item():.3f}  "
                f"core {domain_loss(model, 0, [], 128):.3f}  "
                f"A(on) {domain_loss(model, 1, [0], 128):.3f}  A(off) {domain_loss(model, 1, [], 128):.3f}  "
                f"B(on) {domain_loss(model, 2, [1], 128):.3f}  B(off) {domain_loss(model, 2, [], 128):.3f}  "
                f"[{time.time() - t0:.0f}s]")
    return model


if __name__ == "__main__":
    train_gram()

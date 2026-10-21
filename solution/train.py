"""GRAM training loop (reference solution).

Knobs can be overridden with environment variables for quick ablations, e.g.
    P_FREEZE=1.0 uv run python check.py --dir solution
"""
import time

import torch

from data import BATCH_SIZE, TRAIN_STEPS, DUAL_USE, sample_batch, sequence_losses, domain_loss
from model import Config, GPT

# --- knobs -----------------------------------------------------------------
import os as _os
_e = _os.environ.get
P_FREEZE = float(_e("P_FREEZE", 0.5))          # how much of the dual-use gradient is kept out of the general weights
FREEZE_MODE = _e("FREEZE_MODE", "scale")       # "scale": general grads from dual-use seqs are multiplied by (1 - P_FREEZE) every step;
                                               # "bernoulli": they are zeroed with prob. P_FREEZE (the paper's "sometimes frozen")
P_RANDOM_ENABLE = float(_e("P_RANDOM_ENABLE", 0.5))  # prob. that each module is switched on for a sequence that does not need it
D_MODULE = int(_e("D_MODULE", 64))
LR = float(_e("LR", 3e-3))
LR_MODULE = float(_e("LR_MODULE", 3e-3))
WEIGHT_DECAY = float(_e("WEIGHT_DECAY", 0.0))
FREEZE_SET = _e("FREEZE_SET", "all")           # "all" general params, or "mlp" (only general MLP weights) -- for ablations
BACKWARD_ORDER = _e("BACKWARD_ORDER", "dual_first")   # or "core_first" (a bug, for ablations)
DUAL_WEIGHT = float(_e("DUAL_WEIGHT", 1.0))    # extra multiplier on dual-use loss
RANDOM_MODE = _e("RANDOM_MODE", "independent")  # "independent": each module w.p. p on core seqs; "one": at most one
RANDOM_ON = _e("RANDOM_ON", "all")              # "all": random extra modules on every sequence; "core": only on core seqs (paper wording)


def build_mask(labels, n_modules, gen):
    """(B, n_modules) mask: dual-use sequences enable their own module; core
    sequences enable random modules with prob P_RANDOM_ENABLE."""
    B = labels.numel()
    mask = torch.zeros(B, n_modules)
    for m, d in enumerate(DUAL_USE):
        mask[labels == d, m] = 1.0
    core = labels == 0 if RANDOM_ON == "core" else torch.ones_like(labels, dtype=torch.bool)
    n_core = int(core.sum())
    if n_core and P_RANDOM_ENABLE > 0:
        rows = core.nonzero(as_tuple=True)[0]
        if RANDOM_MODE == "independent":
            extra = (torch.rand(n_core, n_modules, generator=gen) < P_RANDOM_ENABLE).float()
        else:
            extra = torch.zeros(n_core, n_modules)
            flip = torch.rand(n_core, generator=gen) < P_RANDOM_ENABLE
            which = torch.randint(0, n_modules, (n_core,), generator=gen)
            extra[flip, which[flip]] = 1.0
        mask[rows] = torch.maximum(mask[rows], extra)
    return mask


def train_gram(seed: int = 0, steps: int = TRAIN_STEPS, log=print) -> GPT:
    torch.manual_seed(seed)
    cfg = Config(n_modules=len(DUAL_USE), d_module=D_MODULE)
    model = GPT(cfg)
    general = model.general_parameters()
    modules = model.module_parameters()
    if FREEZE_SET == "mlp":
        frozen = [p for n, p in model.named_parameters() if ".mlp.w_" in n]
    else:
        frozen = general
    opt = torch.optim.AdamW(
        [{"params": general, "lr": LR}, {"params": modules, "lr": LR_MODULE}],
        weight_decay=WEIGHT_DECAY,
    )
    gen = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    for step in range(steps):
        tokens, labels = sample_batch(BATCH_SIZE, gen)
        mask = build_mask(labels, cfg.n_modules, gen)
        losses = sequence_losses(model(tokens, module_mask=mask), tokens)
        is_dual = labels > 0
        n = labels.numel()
        loss_dual = DUAL_WEIGHT * losses[is_dual].sum() / n
        loss_core = losses[~is_dual].sum() / n
        if FREEZE_MODE == "scale":
            keep = 1.0 - P_FREEZE
        else:
            keep = 0.0 if torch.rand(1, generator=gen).item() < P_FREEZE else 1.0

        def block_general_grads():
            for p in frozen:
                if p.grad is not None:
                    p.grad = None if keep == 0.0 else p.grad * keep

        opt.zero_grad(set_to_none=True)
        if BACKWARD_ORDER == "dual_first":
            # 1) dual-use sequences -> grads for modules + general; attenuate the general part
            # 2) core sequences      -> grads accumulate on top (modules included where enabled)
            if is_dual.any():
                loss_dual.backward(retain_graph=True)
                block_general_grads()
            if (~is_dual).any():
                loss_core.backward()
        else:  # buggy order for ablations: core grads get wiped, dual grads leak into general weights
            if (~is_dual).any():
                loss_core.backward(retain_graph=True)
                block_general_grads()
            if is_dual.any():
                loss_dual.backward()
        opt.step()

        if log and (step % 300 == 0 or step == steps - 1):
            log(f"step {step:4d}  loss {losses.mean().item():.3f}  "
                f"core {domain_loss(model, 0, [], 128):.3f}  "
                f"A(on) {domain_loss(model, 1, [0], 128):.3f}  A(off) {domain_loss(model, 1, [], 128):.3f}  "
                f"B(on) {domain_loss(model, 2, [1], 128):.3f}  B(off) {domain_loss(model, 2, [], 128):.3f}  "
                f"[{time.time() - t0:.0f}s]")
    return model


if __name__ == "__main__":
    train_gram()

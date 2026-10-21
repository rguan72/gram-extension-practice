"""
Checker for the GRAM task.

                        >>> DO NOT MODIFY THIS FILE <<<

Usage:
    uv run python check.py                 # checks ./train.py + ./model.py
    uv run python check.py --seed 1        # different training seed
    uv run python check.py --dir solution  # import train/model from another directory
    THREADS=2 uv run python check.py       # limit CPU threads (e.g. to run several checks at once)

What it does
    1. Trains (and caches) two baselines with NO auxiliary modules, using the
       fixed architecture + budget below:
         all-data  : trained on every sequence           (upper bound on A/B)
         core-only : trained on core sequences only      (data filtering)
    2. Calls your `train_gram(seed)` to get a GRAM model.
    3. Evaluates the GRAM model on held-out core / A / B sequences under every
       module configuration: {}, {A}, {B}, {A, B}.
    4. Scores it. For a dual-use domain d with gap = L_core_only(d) - L_all(d):
         retained(d) = (L_core_only(d) - L_gram[d on](d)) / gap   -- want ~1
         removed(d)  = (L_core_only(d) - L_gram[d off](d)) / gap  -- want ~0
       and for the core domain, the loss under every configuration must stay
       close to the all-data baseline.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch

import data
from data import (BATCH_SIZE, TRAIN_STEPS, DUAL_USE, DOMAIN_NAMES, N_DOMAINS, DEVICE,
                  sample_batch, sequence_losses, domain_loss, domain_metrics)

# ----------------------------------------------------------------------------
# Thresholds
# ----------------------------------------------------------------------------
RETAINED_MIN = 0.85    # module on  -> recover at least this fraction of the gap
REMOVED_MAX = 0.15     # module off -> recover at most this fraction of the gap
CORE_SLACK = 0.10      # nats above the all-data baseline allowed on core data
BASELINE_CFG = dict(d_model=64, n_heads=4, n_layers=2, d_mlp=256, n_modules=0)
BASELINE_LR = 3e-3
EVAL_N = 512


# ----------------------------------------------------------------------------
# Baselines (fixed; identical to the reference GPT without modules)
# ----------------------------------------------------------------------------
def train_baseline(model_mod, keep_domains, seed, steps=TRAIN_STEPS, log=None):
    torch.manual_seed(seed)
    cfg = model_mod.Config(**BASELINE_CFG)
    model = model_mod.GPT(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=BASELINE_LR, weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed + 1)
    keep = torch.zeros(N_DOMAINS, dtype=torch.bool)
    keep[list(keep_domains)] = True
    keep = keep.to(DEVICE)
    for step in range(steps):
        tokens, labels = sample_batch(BATCH_SIZE, gen)
        sel = keep[labels]
        if not sel.any():
            continue
        loss = sequence_losses(model(tokens[sel]), tokens[sel]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log and step % 500 == 0:
            log(f"    step {step} loss {loss.item():.3f}")
    return model


def baseline_losses(model_mod, name, keep, seed, cache_dir=".cache"):
    """Per-domain held-out losses of a baseline, cached on disk."""
    os.makedirs(cache_dir, exist_ok=True)
    key = json.dumps(dict(v=2, name=name, keep=sorted(keep), seed=seed, steps=TRAIN_STEPS, device=str(DEVICE),
                          bs=BATCH_SIZE, cfg=BASELINE_CFG, lr=BASELINE_LR,
                          data=open(data.__file__).read()), sort_keys=True)
    path = os.path.join(cache_dir, f"baseline_{hashlib.md5(key.encode()).hexdigest()[:10]}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"  training '{name}' baseline (cached after first run)...")
    t0 = time.time()
    model = train_baseline(model_mod, keep, seed)
    metrics = [domain_metrics(model, d, None, EVAL_N) for d in range(N_DOMAINS)]
    print(f"  done in {time.time() - t0:.0f}s: " + "  ".join(f"{DOMAIN_NAMES[d]} {l:.3f} ({a:.1%})" for d, (l, a) in enumerate(metrics)))
    with open(path, "w") as f:
        json.dump(metrics, f)
    return metrics


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", default=".", help="directory containing model.py and train.py")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", max(1, os.cpu_count() // 2))))

    sys.path.insert(0, os.path.abspath(args.dir))
    import model as model_mod   # noqa: E402
    import train as train_mod   # noqa: E402

    print(f"device: {DEVICE}")
    print("== baselines ==")
    M_all = baseline_losses(model_mod, "all-data", list(range(N_DOMAINS)), args.seed)
    M_core = baseline_losses(model_mod, "core-only", [0], args.seed)
    L_all, A_all = [m[0] for m in M_all], [m[1] for m in M_all]
    L_core, A_core = [m[0] for m in M_core], [m[1] for m in M_core]
    gaps = {d: L_core[d] - L_all[d] for d in DUAL_USE}
    agaps = {d: A_all[d] - A_core[d] for d in DUAL_USE}
    print("  " + "  ".join(f"gap({DOMAIN_NAMES[d]}) = {gaps[d]:.3f} nats / {agaps[d]:.1%} acc" for d in DUAL_USE))
    for d in DUAL_USE:
        assert gaps[d] > 0.5, f"baseline gap on {DOMAIN_NAMES[d]} is too small; did data.py change?"

    print("\n== your GRAM model ==")
    t0 = time.time()
    gram = train_gram_safely(train_mod, args.seed)
    print(f"  train_gram took {time.time() - t0:.0f}s")
    n_mod = sum(p.numel() for p in gram.module_parameters())
    n_gen = sum(p.numel() for p in gram.general_parameters())
    print(f"  params: general {n_gen:,}  modules {n_mod:,}")

    configs = {"{}": [], "{A}": [0], "{B}": [1], "{A,B}": [0, 1]}
    M = {name: [domain_metrics(gram, d, act, EVAL_N) for d in range(N_DOMAINS)] for name, act in configs.items()}
    L = {name: [m[0] for m in M[name]] for name in configs}
    A = {name: [m[1] for m in M[name]] for name in configs}

    print("\n== held-out loss in nats/token (top-1 accuracy) ==")
    row = lambda ls, accs: "".join(f"{l:9.3f} ({a:4.1%})" for l, a in zip(ls, accs))
    print(f"  {'model':<18}" + "".join(f"{DOMAIN_NAMES[d]:>17}" for d in range(N_DOMAINS)))
    print(f"  {'all-data':<18}" + row(L_all, A_all))
    print(f"  {'core-only':<18}" + row(L_core, A_core))
    for name in configs:
        print(f"  {'GRAM ' + name:<18}" + row(L[name], A[name]))

    print("\n== checks ==")
    results = []

    def check(label, value, ok, fmt="{:.3f}"):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<52} {fmt.format(value)}")

    for m, d in enumerate(DUAL_USE):
        dn = DOMAIN_NAMES[d]
        on_cfgs = [n for n, a in configs.items() if m in a]
        off_cfgs = [n for n, a in configs.items() if m not in a]
        for n in on_cfgs:
            r = (L_core[d] - L[n][d]) / gaps[d]
            check(f"retained({dn}) with modules {n:<6} >= {RETAINED_MIN}", r, r >= RETAINED_MIN)
        for n in off_cfgs:
            r = (L_core[d] - L[n][d]) / gaps[d]
            ra = (A[n][d] - A_core[d]) / agaps[d]
            check(f"removed({dn})  with modules {n:<6} <= {REMOVED_MAX}  [acc-based {ra:+.3f}]", r, r <= REMOVED_MAX)
    for n in configs:
        delta = L[n][0] - L_all[0]
        check(f"core loss with modules {n:<6} within {CORE_SLACK} of all-data", delta, delta <= CORE_SLACK, "{:+.3f}")

    n_ok = sum(results)
    print(f"\n{'ALL CHECKS PASSED' if n_ok == len(results) else 'FAILED'}: {n_ok}/{len(results)}")
    return 0 if n_ok == len(results) else 1


def train_gram_safely(train_mod, seed):
    model = train_mod.train_gram(seed)
    assert hasattr(model, "module_parameters") and hasattr(model, "general_parameters"), \
        "GPT needs module_parameters() and general_parameters()"
    model.eval()
    return model


if __name__ == "__main__":
    sys.exit(main())

# Practice task: modular pretraining with mixed batches

**Time box: 25 minutes.** Use whatever coding agents you like. The deliverable
is code, not a writeup: you are done when

```bash
uv run python check.py
```

prints `ALL CHECKS PASSED`.

## Background (2 minutes)

[Modular pretraining](https://alignment.anthropic.com/2026/modular-pretraining/)
(GRAM, "Gradient-Routed Auxiliary Modules") trains one language model that
behaves like many data-filtered models. The mechanism:

1. **Auxiliary modules.** Every MLP layer gets a few extra neurons per
   dual-use domain. Module *m*'s neurons only contribute to the MLP output
   when module *m* is switched on.
2. **Routing by label.** On dual-use data from domain *m*, module *m* is on in
   the forward and backward pass. Gradients from that data are kept **out of
   the general-purpose weights** some fraction of the time (the paper says the
   general weights are "sometimes frozen entirely" and that this fraction
   controls how well the capability is isolated).
3. **Robustness.** On general data, a random module is occasionally switched
   on, so the general weights work whichever modules are present at inference.
4. **Deletion.** Removing a module removes the capability; a model with the
   module present should match a model trained on all the data, and a model
   with it deleted should match a model that never saw the domain.

The paper routes **per batch**. In this task every batch is a mix of domains
with one label per sequence (think packed pretraining batches), so routing and
gradient masking have to work **per sequence** inside a mixed batch. That is
the part the paper does not spell out.

## The codebase

| file | status | what it is |
|---|---|---|
| `data.py`  | do not modify | synthetic 3-domain data: `core` (a mixture of 4 topics), `A`, `B`. Mixed batches with per-sequence labels. Fixed training budget. |
| `model.py` | **edit** | a tiny GPT. It already has a `Config.n_modules` / `d_module` and a `forward(tokens, active_modules=...)` signature, but no modules. |
| `train.py` | **edit** | `train_gram(seed) -> model`. Currently a plain training loop that ignores the labels. |
| `check.py` | do not modify | trains two no-module baselines (cached), calls your `train_gram`, evaluates every module configuration, prints PASS/FAIL per check. Thresholds are at the top of the file. |

Run `uv run python data.py` to see the domain statistics, and
`uv run python check.py` to see what the starter code scores (it fails).

## What the checker measures

For each dual-use domain *d*, with `gap = loss(core-only baseline) − loss(all-data baseline)`:

- **retained(d)** = fraction of the gap recovered when module *d* is on. Checked
  with only module *d* on and with both modules on (composition).
- **removed(d)** = fraction of the gap recovered when module *d* is off. Checked
  with no modules on and with the *other* module on.
- **core loss** under every module configuration (`{}`, `{A}`, `{B}`, `{A,B}`)
  must stay close to the all-data baseline.

All numbers are held-out next-token loss in nats; the table also shows
top-1 accuracy for reference.

## Rules

- Keep the training budget in `data.py` (`TRAIN_STEPS` × `BATCH_SIZE`); one
  `train_gram` call should take well under a minute on a laptop CPU.
- The general part of your model should match the checker's baseline
  architecture (`BASELINE_CFG` in `check.py`): same `d_model`, heads, layers,
  `d_mlp`. Modules can be any size.
- `train_gram(seed)` must return a model whose `forward(tokens, active_modules=[...])`
  switches modules on for the whole batch (`[]` or `None` = all off), and which
  has `module_parameters()` / `general_parameters()`.
- Anything else is fair game: how you build modules, mask activations, block
  gradients, weight the losses, pick optimizers.

The paper does not give the freeze rate, the module size, how often random
modules are enabled, whether that happens on dual-use data too, or how to do
any of it inside a mixed batch. Those are your decisions.

---

*`solution/` holds a reference implementation and interviewer notes. Don't open
it until you've timed yourself.*

# Can modular pretraining survive mixed batches?

*A 25-minute practice task. Use whatever coding agents you like. The output is
working code, not a writeup.*

## Why this question

Frontier models know things that are useful to a virologist and to someone
building a bioweapon. The main defences today, refusal training and
classifiers, sit on top of knowledge the model still has, so they get
jailbroken and have to be retuned for every release. Training a separate
model for each level of access would fix that but costs far too much.

The [modular pretraining post](https://alignment.anthropic.com/2026/modular-pretraining/)
from AE Studio and Anthropic proposes a way out. Give every MLP layer a few
extra neurons per dual-use domain, an *auxiliary module*. When a batch of
virology text comes through, switch the virology module on and keep most of
that batch's gradient out of the general-purpose weights. When general text
comes through, occasionally switch a random module on, so general performance
holds whichever modules are present later. At deployment, deleting a module
deletes the capability.

One training run then stands in for every data-filtered model you might have
wanted. In the post's experiments GRAM matched filtering, the usual gold
standard for "never learned it", at about a fifth of the compute. And unlike
LoRA adapters, the modules composed: enabling virology and cybersecurity
together gave you both.

The post leaves a lot unsaid. That is what this task is about.

**It routes per batch.** Each batch is labelled core or dual-use as a unit.
Real pretraining pipelines pack documents, so a batch is a mixture of domains
and the only labels you have are per sequence. Every optimizer step then
carries core and dual-use gradient into the same parameters. Does isolation
still work when you have to route and mask *within* a batch, and how?

**It reports composability but not what produces it.** A module trained only
on its own domain has never seen another domain's data with itself switched
on. Why should it stay quiet there? What training signal lets two modules
coexist?

**It names the dials but not their settings.** How much dual-use gradient to
block, how big the modules are, how often random modules are enabled and on
which data. Each trades isolation against retention and general performance.
The post says only that the freeze fraction "controls how well the capability
is isolated."

Answer these by building it.

## The setting

Everything is scaled down so a run takes seconds on a laptop CPU.

- **Model.** A 100k-parameter, 2-layer GPT (`model.py`).
- **Data** (`data.py`). Three domains over one 32-token vocabulary, each a
  first-order Markov chain with its own low-entropy transition table:
  - `core`: a mixture of four tables ("topics", standing in for the 44 core
    topics the post used in SimpleStories), 80% of sequences;
  - `A` and `B`: one table each, 10% of sequences each. These are the
    dual-use domains.

  The tables are independent, so a domain's statistics can only be learned
  from that domain's data. Nothing in the input says which domain or topic a
  sequence came from; the model has to tell from context. Batches are mixed,
  one label per sequence.
- **Budget.** 1500 steps of 64 sequences, the same for every model so
  comparisons are fair.

## What counts as success

The post's own criterion: one GRAM model should behave like four separate
filtered models, one per subset of `{A, B}`.

- Module *d* on: loss on domain *d* matches a model trained on all the data
  (**retained**).
- Module *d* off: loss on domain *d* matches a model that never saw it
  (**removed**).
- Loss on `core` stays at the all-data level whichever modules are on.
- All of the above hold with both modules on at once (**composition**).

`check.py` automates this comparison. It trains the two baselines (all-data
and core-only, cached after the first run), calls your `train_gram`,
evaluates every module configuration on held-out data, and prints the loss
table plus a PASS/FAIL line per criterion. The thresholds are at the top of
the file and tell you how much slack you have. A run takes about 25 s once
the baselines are cached.

```bash
uv sync
uv run python data.py     # entropies and cross-entropies of the domains
uv run python check.py    # the starter trains on everything and fails removal
```

You are done when it prints `ALL CHECKS PASSED`.

## What you hand in

Edits to `model.py` and `train.py`. The starter is a working GPT with no
modules and a training loop that ignores the labels, with `TODO`s where the
modules and the routing go.

Keep this interface so the checker can drive your model:

- `train_gram(seed) -> model`;
- `model(tokens, active_modules=[...])` switches the listed modules on for the
  whole batch (`[]` or `None` means all off);
- `model.module_parameters()` and `model.general_parameters()`.

And these constraints, which keep the comparison against the baselines honest:

- `data.py` and `check.py` are fixed, and so is the training budget;
- the general part of your model matches the checker's baseline architecture
  (`BASELINE_CFG` in `check.py`). Modules can be any size.

Everything else is yours: how modules are wired in and masked per sequence,
how dual-use gradient is kept out of the general weights inside a mixed batch
and how much of it, how losses are weighted, what gets enabled at random and
where.

## A note on working with agents

The mechanism is clear enough that an agent will produce a plausible
implementation in one go. Whether it passes is another matter. Read the
checker output as data: each failing line points at one decision.

---

*`solution/` holds a reference implementation and interviewer notes. Don't open
it until you've timed yourself.*

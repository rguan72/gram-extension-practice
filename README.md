# Can modular pretraining survive mixed batches?

*A 25-minute practice task. Use whatever coding agents you like. The output is
working code, not a writeup.*

## Why this question

Frontier models know things that are useful to a virologist and to someone
building a bioweapon. Today the main defences are refusal training and
classifiers, both of which sit on top of knowledge the model still has, so they
get jailbroken and have to be retuned for every release. The alternative, train
a separate model for each level of access, is far too expensive.

Anthropic's [modular pretraining post](https://alignment.anthropic.com/2026/modular-pretraining/)
proposes a way out. During pretraining, give every MLP layer a few extra
neurons per dual-use domain (an *auxiliary module*). When a batch of, say,
virology text comes through, switch the virology module on and keep most of
that batch's gradient out of the general-purpose weights. When general text
comes through, occasionally switch a random module on so the general weights
don't come to depend on any particular one. At deployment, deleting a module
deletes the capability. One training run then stands in for every
data-filtered model you might have wanted: in their experiments, GRAM matched
filtering (the usual gold standard for "never learned it") at about a fifth of
the compute, and unlike LoRA adapters the modules composed, so enabling
virology *and* cybersecurity gave you both.

The post leaves a lot unsaid, and that is what this task is about.

**The post routes per batch.** Each batch is labelled core or dual-use as a
unit, and the whole batch's gradient is either allowed into the general weights
or not. Real pretraining pipelines pack documents, so a batch is a mixture of
domains and the only labels you have are per sequence. Every optimizer step
then carries core and dual-use gradient into the same parameters. Does the
isolation still work when you have to route and mask *within* a batch? What
is the right way to do that?

**The post reports composability but does not say what produces it.** A module
that is only ever trained on its own domain has never seen another domain's
data with itself switched on. Why should it stay quiet there? What training
signal makes two modules coexist?

**The post names the dials but not their settings.** How much of the dual-use
gradient to block, how big the modules are, how often random modules are
enabled and on which data. These trade isolation against retention against
general performance, and the post says only that the freeze fraction "controls
how well the capability is isolated."

Your job is to answer these by building it.

## The setting

Everything is scaled down so a run takes seconds on a laptop CPU and you can
iterate.

- **Model.** A 100k-parameter, 2-layer GPT (`model.py`).
- **Data** (`data.py`). Three domains over one 32-token vocabulary, each a
  first-order Markov chain with its own low-entropy transition table:
  - `core`: a mixture of four such tables ("topics", standing in for the 44
    core topics the post used in SimpleStories), 80% of training sequences;
  - `A` and `B`: one table each, 10% of sequences each. These are the
    dual-use domains.

  The tables are independent, so a domain's statistics are knowledge that can
  only be learned from that domain's data. Nothing in the input says which
  domain or topic a sequence came from; the model has to tell from context.
  Batches are mixed, with one label per sequence.
- **Budget.** 1500 steps of 64 sequences, fixed for every model so comparisons
  are fair.

## What counts as success

The post's own criterion. A GRAM model should behave like four separate
filtered models, one per subset of `{A, B}`:

- with module *d* on, loss on domain *d* should match a model trained on all
  the data (**retained**);
- with module *d* off, loss on domain *d* should match a model that never saw
  domain *d* (**removed**);
- loss on `core` should stay at the all-data level whichever modules are on;
- both of the above should hold with both modules on at once (**composition**).

`check.py` automates exactly this comparison. It trains the two baselines
(all-data, core-only; cached after the first run), calls your `train_gram`,
evaluates every module configuration on held-out data, and prints a table of
losses plus a PASS/FAIL line per criterion with the margin it used. The
thresholds are at the top of the file; read them, they tell you how much
slack you have. Expect ~25 s per run once the baselines are cached.

```bash
uv sync
uv run python data.py     # entropies and cross-entropies of the domains
uv run python check.py    # the starter trains on everything and fails removal
```

You are done when it prints `ALL CHECKS PASSED`.

## What you hand in

Edits to `model.py` and `train.py`. The starter gives you a working GPT with
no modules and a plain training loop that ignores the labels, with `TODO`s
where the modules and the routing go.

Keep this interface so the checker can drive your model:

- `train_gram(seed) -> model`
- `model(tokens, active_modules=[...])` switches the listed modules on for the
  whole batch; `[]` or `None` means all off;
- `model.module_parameters()` and `model.general_parameters()`.

And these constraints, which keep the comparison against the baselines honest:

- `data.py` and `check.py` are fixed;
- the training budget in `data.py` is fixed;
- the general part of your model matches the checker's baseline architecture
  (`BASELINE_CFG` in `check.py`): same width, heads, layers, MLP size.
  Modules can be any size.

Everything else is yours to decide: how the modules are wired in, how you
mask their contribution per sequence, how you keep dual-use gradient out of
the general weights inside a mixed batch, how much of it, how you weight the
losses, what you enable at random and where.

## A note on working with agents

The mechanism above is clear enough that an agent will write a plausible
implementation in one go. Whether it passes is another matter: the natural
reading of the post is not sufficient for composition, and per-sequence
gradient masking is easy to get subtly wrong in ways that only show up in the
loss table. Read the checker output as data. Each failing line points at a
specific decision.

---

*`solution/` holds a reference implementation and interviewer notes. Don't open
it until you've timed yourself.*

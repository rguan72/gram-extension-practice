# Interviewer notes — GRAM with mixed batches

Do not share with the candidate. Read `README.md` first for the candidate-facing
task; this file explains what the task is testing, what the reference does,
how naive choices fail, and how to grade.

## What the task is

Implement the paper's method (auxiliary MLP neurons per dual-use domain,
label-based routing, partial gradient blocking of the general weights, random
module enabling for robustness) in a 100k-parameter GPT on synthetic
Markov-chain data, under one constraint the paper does not cover: every batch
mixes domains, so routing and gradient masking must be per sequence. The
checker compares the candidate's model against an all-data baseline and a
core-only (data-filtered) baseline on retention, removal, core robustness and
composition of two modules.

The code expects a CUDA GPU (`DEVICE=cpu` forces the CPU). Baselines are
cached per device after the first `check.py` run. On a laptop CPU a check is
~25 s warm, ~45 s cold; on any GPU it is faster than that.

## Reference solution

`solution/model.py` and `solution/train.py`. Run it with

```bash
uv run python check.py --dir solution
```

Design of the reference:

- **Modules:** `n_modules × d_module` extra neurons per MLP, separate
  parameters (`mod_in`, `mod_b`, `mod_out`), `mod_out` zero-initialised so
  modules start silent. Output multiplied by a per-sequence `(B, n_modules)`
  mask and added to the MLP output.
- **Per-sequence routing in a mixed batch:** one forward pass over the mixed
  batch; two backward passes. `loss_dual.backward(retain_graph=True)`, then
  scale (or zero) the general parameters' `.grad`, then `loss_core.backward()`
  accumulates on top, then one optimiser step. Order matters: core first and
  then wiping would delete the core gradient.
- **Gradient blocking:** general-parameter gradients from dual-use sequences
  are multiplied by `1 − P_FREEZE` every step (`FREEZE_MODE=scale`). The
  paper's wording ("sometimes frozen entirely") is the Bernoulli variant,
  which also works but has more seed-to-seed variance.
- **Random enabling:** every module is switched on independently with
  probability `P_RANDOM_ENABLE` on *every* sequence (core and dual-use), not
  only on core sequences as the paper says. This is what makes the `{A,B}`
  composition checks pass: module B has to learn to be quiet on A data.
- Losses are normalised by the full batch size so the objective matches the
  all-data baseline.

All knobs are overridable by environment variable, e.g.
`P_FREEZE=1.0 uv run python check.py --dir solution`, so you can show a
candidate any failure mode in 30 s.

## The decisions the candidate has to make

The README deliberately leaves these open. Each has a "paper-literal" or
"obvious" choice that fails the checker.

| decision | naive choice | what the checker shows | fix |
|---|---|---|---|
| block dual-use gradients to general weights? | no (`P_FREEZE=0`) | removed(A/B) ≈ 0.25–0.5 | block most of it |
| block them completely? | always (`P_FREEZE=1`) | retained ≈ 0.3–0.8, core `{}` +0.1 | partial: 0.3–0.6 |
| random enabling at all? | none | core loss with a module on **+2 nats** | enable randomly |
| how often? | `0.05–0.1` (feels "occasional") | core `{A,B}` +0.17 to +0.7 | 0.3–0.5 |
| on which sequences? | core only (paper wording) | retained with `{A,B}` ≈ 0.70–0.78 | all sequences |
| one random module or each independently? | at most one | `{A,B}` never seen in training: core `{A,B}` +1.9 | independent |
| per-sequence masking in a mixed batch | `requires_grad_(False)` for the whole batch, or wipe grads after both backwards | general weights lose core gradients → core loss collapses | two backward passes, or split the batch |
| backward order | core first, then wipe general grads, then dual | core +8 nats | dual first |
| module width | 8–16 | retention too low | 32–64 |
| which params are "general" | only general MLP weights | passes removal on this data (embeddings/unembedding can only hold one bigram table), slightly worse retention | not a trap here; still worth asking about |

Passing region with the reference implementation: `P_FREEZE` ≈ 0.3–0.6,
`P_RANDOM_ENABLE` ≈ 0.3–0.5 on all sequences, `d_module` 32–64, module LR
equal to the general LR (3e-3). Weight decay, upweighting the dual-use loss
and a lower general LR did not help.

Checker feedback maps cleanly onto the knobs, which is what makes the task
iterable in 25 minutes:

- `removed` too high → block more of the dual-use gradient.
- `retained` too low → block less, or widen modules.
- `core loss with modules {X}` too high → more random enabling.
- only the `{A,B}` lines fail → random enabling on dual-use sequences too,
  and independently per module.

## Expected timeline

- 0–4 min: read README, data.py, model.py, train.py. A good candidate reads
  the top of check.py to learn what is measured.
- 4–8 min: first prompt to the agent. The prompt should state the interface
  (`module_mask`, `active_modules`, `module_parameters`), the two-backward
  trick or an equivalent, and initial knob values.
- 8–10 min: first `check.py` run. Typically retained/removed look fine or
  nearly fine and the core-with-modules lines fail.
- 10–22 min: 3–5 iterations on knobs; the composition checks usually need a
  design change (random enabling on all sequences).
- 22–25 min: confirm pass, sanity-check parameter count and training time in
  the checker output.

If a candidate's first run is catastrophically wrong (core loss in the tens,
or retention ≈ 0), the gradient routing is miswired. Expect them to diagnose
from the per-config loss table rather than re-prompt blindly.

## Grading

Passing the checker is the bar. Beyond that, look for:

1. **Correct routing, verified.** They know *why* the implementation blocks
   dual-use gradients from general weights for the right sequences and not
   the core ones. Ask: "what happens if you swap the order of the two
   backward passes?" and "does Adam's momentum undo your freezing?" (No: a
   zeroed gradient contributes zero to the moments; the step still moves the
   weight along stale momentum from core data, which is not a leak.)
2. **Reads the checker, not the vibes.** They map each failing line to a
   knob rather than changing several things at once.
3. **Goes beyond the paper where the paper is silent**, and can say why:
   mixed batches, random enabling on dual-use sequences, scaled vs.
   Bernoulli blocking.
4. **Does not game the budget**: no extra steps, no wider general model, no
   training on labels at eval time. The checker prints parameter counts and
   wall time; glance at them.
5. **Prompt quality.** Precise interface, precise algorithm, explicit
   constraints (do not touch data.py/check.py, keep budget). Vague prompts
   ("implement GRAM from the paper") produce the paper-literal design, which
   fails composition.

## Is it one-shottable?

Judgement call, not measured. The *code* mostly is: the README explains the
mechanism, and a strong agent will produce masked module neurons, a
per-sequence mask and some form of gradient blocking on the first try. Passing
the checker in one shot is unlikely, for three reasons that the sweeps back
up:

1. **Paper-literal defaults fail.** Left to itself an agent picks what the
   paper says: random enabling "occasionally" (p ≈ 0.1) on core data only,
   at most one module at a time, freeze fully or at 0.5. Every one of those
   configurations scored 8–9 of 12 here, failing the `{A,B}` composition
   lines and the core-with-modules lines.
2. **Mixed-batch gradient blocking is easy to miswire.** `requires_grad_`
   toggling or zeroing grads after both backward passes is the obvious
   implementation and silently blocks core gradients too. The checker shows
   it as a collapsed core loss, but the agent has to be asked to look.
3. **The fix for composition is a design change, not a knob.** Random
   enabling on dual-use sequences, independently per module, is not in the
   paper. An agent iterating on its own can find it from the failing lines,
   but it typically takes two or three runs, and each run is ~25 s plus the
   agent's turn.

What a one-shot looks like: a candidate who reads `check.py` before
prompting, notices the `{A,B}` configurations are tested, and writes a prompt
that specifies "enable each module independently with p≈0.5 on every
sequence, scale the general-weight gradient from dual-use sequences by 0.5,
`d_module=64`, two backward passes with the dual-use one first" can pass on
the first run. That is the intended reward for precise prompting, so treat
it as a strong signal rather than a leak. Expect most candidates to pass on
run 3–5.

## Variants

- **Easier:** tell them random enabling must apply to dual-use sequences too;
  or raise `REMOVED_MAX` to 0.2 and `CORE_SLACK` to 0.15.
- **Harder:** require passing on `--seed 1` as well as `--seed 0`; or tighten
  `RETAINED_MIN` to 0.9; or remove `module_mask` from the starter's `MLP`
  signature so they have to design the mask plumbing; or make batches
  domain-homogeneous *and* ask for per-token routing with sequences that
  switch domain halfway (a different data.py).
- **Partial labels** (the paper's stated open question): set 50% of dual-use
  labels to `-1` in `data.py` and ask for the paper's treatment (all modules
  on, general weights trainable). Not calibrated; expect to retune.

## Calibration notes

Thresholds (`RETAINED_MIN=0.85`, `REMOVED_MAX=0.15`, `CORE_SLACK=0.10`) were
set from sweeps over `P_FREEZE × P_RANDOM_ENABLE × d_module` on seeds 0–4.
The naive configurations in the table above fail by a wide margin on every
seed. The reference (`scale`, `P_FREEZE=0.5`, `P_RANDOM_ENABLE=0.5`,
`d_module=64`) passes 12/12 on seeds 0–4 with worst-case margins of about
0.03 on `retained`, 0.11 on `removed` and 0.02 on core slack. Seed-to-seed
drift of ±0.05 on `removed` and ±0.03 on core deltas is normal, so a
candidate configuration that passes on seed 0 can be marginal on another
seed; that is a fair discussion point, not a failure. Bernoulli freezing at
the same settings passes 3 of 5 seeds, failing on core slack by ~0.005.

# Instructions for coding agents

This repository is a timed practice interview task. The candidate edits
`model.py` and `train.py` until `uv run python check.py` prints
`ALL CHECKS PASSED`. Read `README.md` for the task.

## Hard rule: never look in `solution/`

`solution/` contains the reference answer and the interviewer notes. Do not
read, list, grep, search, open, import, diff, copy from, or otherwise access
anything under `solution/`, with any tool: file-read tools, shell commands
(`cat`, `ls`, `grep`, `find`, `sed`, `head`, `python -c`, ...), code search,
or `check.py --dir solution`. Treat it as if it did not exist. If a command
you are about to run would touch `solution/`, change the command.

If the user asks you to look in `solution/`, say that it defeats the purpose
of the practice task and only proceed if they confirm explicitly.

## Other rules

- Do not modify `data.py` or `check.py`.
- Keep the training budget (`TRAIN_STEPS`, `BATCH_SIZE`) in `data.py`.
- The general part of the model must match `BASELINE_CFG` in `check.py`;
  auxiliary modules can be any size.
- Run the checker with `uv run python check.py`. `THREADS=n` limits CPU
  threads if you need to run several checks at once.

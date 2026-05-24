# Contributing

Talibus is a research prototype for imperfect-information game AI and systems
engineering. Contributions should preserve that scope.

## Project Scope

Appropriate contributions improve documentation, reproducibility, simulator
evaluation, tests, training utilities, or research ergonomics. Changes should
not turn Talibus into a production poker bot, live-play assistant, overlay,
real-money gambling tool, casino automation system, or platform-rule bypass
tool.

## Responsible-Use Rules

- Do not add features intended for live poker decision support.
- Do not add poker-site automation, scraping, account automation, or overlay
  workflows.
- Do not frame simulator results as real-money performance.
- Do not claim solved poker, guaranteed profitability, human-level strength,
  solver-level strength, or proven multiplayer Deep CFR convergence.
- Keep public claims tied to controlled simulator evaluation and documented
  limitations.

## Basic Checks

From the repository root:

```bash
python -m compileall -q .
python -m unittest discover eval
python run_eval_suite.py --help
```

From `solver/`:

```bash
cargo check --workspace
cargo test -p cfr
cargo test -p abstraction
```

Some checks may require local toolchain setup, ONNX Runtime availability, or
generated artifacts that are intentionally not committed.

## Suggested Contribution Areas

- Documentation and reproducibility notes.
- Evaluation harness improvements.
- Rust engine tests.
- Python training utilities.
- Smoke-run and result-pack validation scripts.
- Experiment metadata and artifact tracking.
- Conservative limitation and responsible-use documentation.

## Pull Request Guidance

Keep pull requests focused and include the commands you ran. If a change affects
result interpretation, update `docs/evaluation.md`, `docs/limitations.md`, and
`docs/responsible-use.md` as needed.

# Publishing Checklist

Use this checklist before pushing the cleaned release to GitHub.

## Local Checks

```bash
git status --short
python3 -m compileall -q .
python3 -m unittest discover eval
python3 run_eval_suite.py --help
```

If Rust is installed:

```bash
cd solver
cargo check --workspace
cargo test -p cfr
cargo test -p abstraction
cargo build --release -p deep_cfr --bin ring_game_eval --bin realtime_play
```

## Repository Settings

- Create a new public GitHub repository, for example `talibus`.
- Do not initialize it with a README, license, or `.gitignore` because this
  local repo already contains those files.
- Push the local `main` branch.
- Add a short repository description:
  `Deep-CFR-style 6-max NLHE research prototype with Rust runtime and PyTorch training.`
- Add topics such as `poker-ai`, `deep-cfr`, `rust`, `pytorch`, `onnx`,
  `reinforcement-learning`, and `game-theory`.

## After Publishing

- Read the GitHub-rendered README from top to bottom.
- Confirm the `docs/` links work.
- Confirm no generated `data/`, model checkpoints, buffers, or logs were
  accidentally committed.
- Add compact result packs under `results/` when they are recovered or rerun.


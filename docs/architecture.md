# Architecture

Talibus is organized around a training-time Deep-CFR-style pipeline and a
runtime decision pipeline.

## Game And Abstraction

The Rust workspace in `solver/` contains the game and solver core.

- `solver/game` implements NLHE game mechanics: seats, blinds, betting rounds,
  stacks, pots, board cards, terminal states, and showdown.
- `solver/cfr/src/nlhe_game.rs` wraps the game as an imperfect-information
  model. It builds information sets from the acting player, street, card
  bucket, board bucket, and betting history.
- The action space is abstracted into fixed policy slots rather than arbitrary
  continuous bet sizing. Slots include fold/check/call, multiple bet/raise
  sizes, and all-in.
- Postflop representation uses cluster assets from `checkpoints/nlhe_clusters`.

## Deep CFR-Style Training

The traversal code in `solver/deep_cfr/src/traverse.rs` samples game states and
produces training samples.

- Traverser nodes enumerate legal abstract actions and record advantage-like
  samples.
- Opponent nodes sample actions from the current neural policy estimate.
- Samples are serialized for Python training.

The Python stack in `training/deep_cfr` trains neural networks from those
samples.

- `model.py` defines the dense PyTorch model.
- `train.py` loads binary sample files and trains advantage/strategy heads.
- `reservoir.py` implements in-memory and disk-backed sample management.
- `run_deep_cfr.py` orchestrates traversal, training, checkpoints, diagnostics,
  panel evaluation, and long-run state.

## Feature Encoding

`solver/deep_cfr/src/encoding.rs` converts poker states into fixed-size model
features. The encoded state includes private cards, board cards, street,
position, pot and stack information, legal-action masks, player activity, and
recent action history.

## Model Deployment

Trained PyTorch models are exported to ONNX. The Rust runtime loads these ONNX
models for inference so evaluation and real-time search can run without Python
in the decision loop.

`solver/deep_cfr/src/onnx_policy.rs` handles ONNX inference and legal-action
normalization.

## Runtime Search

`solver/deep_cfr/src/realtime_search.rs` adds depth-limited local search from a
current game state. The neural policy acts as the baseline/continuation model,
while the search refines the root decision within a budget.

The relevant binaries are:

- `ring_game_eval`: model-only ring evaluation against scripted opponents.
- `realtime_play`: real-time-search evaluation and interactive/runtime modes.


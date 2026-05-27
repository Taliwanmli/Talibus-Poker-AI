# Trained Model Release

Talibus includes a trained ONNX model release for the 6-max long-run result
pack:

`artifacts/models/talibus-6max-longrun-opt-v1/`

The release contains the strategy model used for the published mixed-table
evaluation and the paired advantage model from the same best-ring promotion
snapshot.

## Released Files

| File | Purpose | Size | SHA-256 |
| --- | --- | ---: | --- |
| `strategy_shared_best_ring.onnx` | Default released strategy policy. This is the model used for the published mixed-table evaluation. | 16,854,867 bytes | `b3473f08c0dda9611babaa14f646b2ec5bb0d28b225fc10f599a45b472842af2` |
| `advantage_shared_best_ring.onnx` | Paired advantage model from the same best-ring promotion snapshot. | 16,854,771 bytes | `70a3b7d4b314be22e02fa5f47127d0fa832cd899f6a3fbee3c8b1a7ec5fe4875` |
| `model_spec.json` | Machine-readable model, training, abstraction, ONNX, and evaluation metadata. | text | see file |
| `SHA256SUMS` | Checksum file for the ONNX artefacts. | text | see file |

The normal PyTorch checkpoints, raw traversal buffers, fp16 alternates, and raw
logs are not included in Git. They are larger generated artefacts and are not
required to run the public strategy model.

## Model Scope

- Game: 6-max No-Limit Texas Hold'em.
- Training approach: Deep-CFR-style traversal with neural advantage/strategy
  models.
- Runtime target: Rust inference through ONNX Runtime via `ort` 2.0.0-rc.11.
- Public default policy: `strategy_shared_best_ring.onnx` with `--policy strategy`.
- Result pack: `results/2026-04-03-6max-longrun-opt/`.

## Game And Abstraction Assumptions

The released model is tied to the abstraction and feature encoding in this
repository.

- Input feature dimension: 510.
- Maximum action slots: 13.
- Legal actions are masked by the runtime before normalization.
- Preflop open-raise buckets: 2.0x, 2.5x, 3.0x, 3.5x.
- Preflop re-raise multiplier buckets: 2.0x, 2.3x, 2.7x, 3.2x, 4.0x.
- Flop bet buckets: 0.33, 0.67, 1.0, 1.5 pot.
- Turn bet buckets: 0.5, 0.75, 1.0, 1.5 pot.
- River bet buckets: 0.33, 0.75, 1.25, 2.0 pot.
- Raise multiplier buckets are street-specific; see `model_spec.json` and
  `solver/abstraction/src/action_abstraction.rs`.
- Card and board abstraction assets are under `checkpoints/nlhe_clusters`.

## ONNX Specification

Both released models use the same ONNX interface:

| Tensor | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `input` | `float32` | `[batch, 510]` | Encoded poker state features. |
| `action_mask` | `float32` | `[batch, 13]` | Legal-action mask for abstract action slots. |
| `output` | `float32` | `[batch, 13]` | Strategy probabilities or advantage values, depending on model. |

The models were exported with ONNX opset 17 by
`training/deep_cfr/train.py::export_onnx`. The strategy export applies masked
softmax in the model wrapper. The advantage export emits masked advantage-like
action values that the Rust runtime can convert into a policy with regret
matching.

The Rust binaries should be run with ONNX Runtime 1.23.x or newer. If the
runtime shared library is not already discoverable on the platform library
path, set `ORT_DYLIB_PATH` to the ONNX Runtime shared library before starting
`ring_game_eval` or `realtime_play`.

## Verifying Artefacts

From the model directory:

```bash
sha256sum -c SHA256SUMS
```

On Windows PowerShell:

```powershell
Get-FileHash .\*.onnx -Algorithm SHA256
```

Compare the output with `SHA256SUMS` or `model_spec.json`.

## Running The Released Strategy Model

Build the Rust runtime binaries:

```bash
cd solver
cargo build --release -p deep_cfr --bin ring_game_eval --bin realtime_play
cd ..
```

Run a short model-only evaluation against the TAG scripted baseline:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
solver/target/release/ring_game_eval \
  --model artifacts/models/talibus-6max-longrun-opt-v1/strategy_shared_best_ring.onnx \
  --policy strategy \
  --cluster-dir checkpoints/nlhe_clusters \
  --num-players 6 \
  --hands 100 \
  --opponent tag
```

Run a short depth-limited-search evaluation:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
solver/target/release/realtime_play \
  --model artifacts/models/talibus-6max-longrun-opt-v1/strategy_shared_best_ring.onnx \
  --mode ring-eval \
  --policy strategy \
  --cluster-dir checkpoints/nlhe_clusters \
  --num-players 6 \
  --hands 100 \
  --opponents tag,calling-station,lag,nit,tag \
  --time-budget-ms 500
```

On Windows, use the `.exe` binaries under `solver\target\release`.

## Published Metrics

The included mixed-table result pack used:

- 1,000 hands per seat.
- 6 seat-rotation runs.
- 200 deck samples.
- 2,000 ms depth-limited-search budget.
- Scripted opponents: TAG, calling station, LAG, nit, TAG.
- Blinds: 10/20.
- Starting stack: 2,000 chips.

Observed seat bb/100 values:

| Seat | bb/100 | Avg decision ms | Avg iterations/decision |
| ---: | -----: | --------------: | ----------------------: |
| 0 | 4539.885 | 2694.191 | 136825.822 |
| 1 | 5498.015 | 2581.715 | 193268.254 |
| 2 | 3664.615 | 3324.101 | 128664.083 |
| 3 | 5205.470 | 2810.123 | 121001.381 |
| 4 | 4923.275 | 2757.157 | 151768.604 |
| 5 | 6222.160 | 2689.477 | 158766.520 |

Average across seats: 5008.903 bb/100.

The 500 ms budget-floor run from seat 0 reported 4747.525 bb/100 with an
average decision time of 1152.724 ms.

## Interpretation Limits

These numbers are strong inside the included scripted simulator setup. They are
not a claim of real-world profitability, human-level strength, commercial
solver quality, or superhuman play. The result pack does not include confidence
intervals, and the scripted opponents are not strong poker agents. Treat the
release as a reproducible research artefact and regression reference for this
codebase.

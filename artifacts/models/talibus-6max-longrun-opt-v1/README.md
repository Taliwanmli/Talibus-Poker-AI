# Talibus 6-Max Longrun Opt v1 Model

This directory contains the trained ONNX artefacts released with the Talibus
6-max long-run result pack.

## Artefacts

| File | Role | Size | SHA-256 |
| --- | --- | ---: | --- |
| `strategy_shared_best_ring.onnx` | Strategy model used for the published mixed-table evaluation. The output is a masked action-probability vector. | 16,854,867 bytes | `b3473f08c0dda9611babaa14f646b2ec5bb0d28b225fc10f599a45b472842af2` |
| `advantage_shared_best_ring.onnx` | Paired advantage model from the same best-ring promotion snapshot. The output is an action-value/advantage vector intended for regret-matching style normalization. | 16,854,771 bytes | `70a3b7d4b314be22e02fa5f47127d0fa832cd899f6a3fbee3c8b1a7ec5fe4875` |

The strategy model is the default model for public runtime examples and for
the published controlled simulator metrics. The advantage model is included for
research completeness and for code paths that evaluate advantage-style policy
outputs.

## Setting

- Game: 6-max No-Limit Texas Hold'em.
- Blinds used in the published evaluation: 10/20.
- Starting stack used in the published evaluation: 2,000 chips.
- Action space: fixed abstract action slots with legal-action masking.
- Input tensor: `input`, `float32`, shape `[batch, 510]`.
- Action mask tensor: `action_mask`, `float32`, shape `[batch, 13]`.
- Output tensor: `output`, `float32`, shape `[batch, 13]`.
- ONNX opset: 17.
- Rust runtime: `ort` 2.0.0-rc.11 with ONNX Runtime 1.23.x or newer.

## Verifying Hashes

From this directory on Unix-like systems:

```bash
sha256sum -c SHA256SUMS
```

On Windows PowerShell:

```powershell
Get-FileHash *.onnx -Algorithm SHA256
```

Compare the printed hashes with `SHA256SUMS` or `model_spec.json`.

## Rust Runtime Example

Build the runtime binaries from the repository root:

```bash
cd solver
cargo build --release -p deep_cfr --bin ring_game_eval --bin realtime_play
cd ..
```

Run a short model-only scripted-opponent evaluation:

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

On Windows PowerShell, use:

```powershell
$env:ORT_DYLIB_PATH = "<path-to-onnxruntime.dll>"
.\solver\target\release\ring_game_eval.exe `
  --model .\artifacts\models\talibus-6max-longrun-opt-v1\strategy_shared_best_ring.onnx `
  --policy strategy `
  --cluster-dir .\checkpoints\nlhe_clusters `
  --num-players 6 `
  --hands 100 `
  --opponent tag
```

## Interpretation

These artefacts support research review and controlled simulator reproduction.
They do not imply solved poker, superhuman play, commercial solver quality,
profitability, or performance against strong human players.

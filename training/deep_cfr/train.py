from __future__ import annotations

import argparse
import math
import struct
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from model import DeepCfrNet, ModelConfig, INPUT_DIM, MAX_ACTIONS
from reservoir import ReservoirBuffer

ADVANTAGE_SAMPLE_MAGIC = b"DCFR"
STRATEGY_SAMPLE_MAGIC = b"DCSG"
SAMPLE_MAGIC = ADVANTAGE_SAMPLE_MAGIC
SAMPLE_VERSION = 2
HEADER_STRUCT = struct.Struct("<4sIII")


def log(message: str) -> None:
    print(message, flush=True)


def configure_warning_filters() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp\..*")
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*isinstance\(treespec,\s*LeafSpec\).*",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*legacy TorchScript-based ONNX export.*",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Deep CFR neural networks from binary samples.")
    parser.add_argument("--samples", type=Path, default=None, help="Binary sample file path")
    parser.add_argument("--model-in", type=Path, default=None, help="Optional model checkpoint/onnx input path")
    parser.add_argument("--model-out", type=Path, required=True, help="ONNX output path")
    parser.add_argument("--type", choices=["advantage", "strategy"], required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout-p", type=float, default=0.10)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--max-sample-reuse-per-iter", type=float, default=6.0)
    parser.add_argument("--adv-huber-delta", type=float, default=20.0)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--buffer-path", type=Path, default=None)
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--onnx-opset", type=int, default=17)
    parser.add_argument(
        "--export-fp16-onnx",
        action="store_true",
        help="Export an additional fp16-optimized ONNX model for inference.",
    )
    parser.add_argument(
        "--fp16-model-out",
        type=Path,
        default=None,
        help="Optional FP16 ONNX output path (default: <model-out>.fp16.onnx).",
    )
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Initialize a random model, export ONNX and checkpoint, then exit",
    )
    args = parser.parse_args()
    if not args.init_only and args.samples is None:
        parser.error("--samples is required unless --init-only is set")
    if not (0.0 <= args.dropout_p < 1.0):
        parser.error("--dropout-p must be in [0, 1)")
    if args.max_sample_reuse_per_iter <= 0.0:
        parser.error("--max-sample-reuse-per-iter must be > 0")
    if args.adv_huber_delta <= 0.0:
        parser.error("--adv-huber-delta must be > 0")
    if args.hidden_dim <= 0:
        parser.error("--hidden-dim must be > 0")
    if args.bottleneck_dim <= 0:
        parser.error("--bottleneck-dim must be > 0")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def derive_buffer_path(args: argparse.Namespace) -> Path:
    if args.buffer_path is not None:
        return args.buffer_path
    return Path("training/deep_cfr/buffers") / f"{args.type}.pkl"


def derive_state_path(args: argparse.Namespace) -> Path:
    if args.state_path is not None:
        return args.state_path
    if args.model_in is not None:
        if args.model_in.suffix == ".pt":
            return args.model_in
        if args.model_in.suffix == ".onnx":
            return args.model_in.with_suffix(".pt")
    if args.model_out.suffix == ".onnx":
        return args.model_out.with_suffix(".pt")
    return Path("training/deep_cfr/checkpoints") / f"{args.type}.pt"


def load_binary_samples(
    path: Path,
    *,
    expected_magic: bytes = ADVANTAGE_SAMPLE_MAGIC,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) < HEADER_STRUCT.size:
        raise ValueError(f"sample file too small: {path}")

    magic, version, input_dim, max_actions = HEADER_STRUCT.unpack_from(data, 0)
    if magic != expected_magic:
        raise ValueError(f"invalid sample magic {magic!r}, expected {expected_magic!r}")
    if version == 0 or version > SAMPLE_VERSION:
        raise ValueError(f"unsupported sample version {version}, expected <= {SAMPLE_VERSION}")
    if input_dim != INPUT_DIM:
        raise ValueError(f"input_dim mismatch: file={input_dim}, expected={INPUT_DIM}")
    if max_actions != MAX_ACTIONS:
        raise ValueError(f"max_actions mismatch: file={max_actions}, expected={MAX_ACTIONS}")

    payload = memoryview(data)[HEADER_STRUCT.size :]
    if version == 1:
        record_size = input_dim * 4 + max_actions * 4 + 1 + 4
        if len(payload) % record_size != 0:
            raise ValueError(
                f"sample payload size {len(payload)} is not divisible by record_size {record_size}"
            )
        sample_count = len(payload) // record_size
        record_dtype = np.dtype(
            [
                ("features", ("<f4", input_dim)),
                ("targets", ("<f4", max_actions)),
                ("valid_actions", "u1"),
                ("iteration", "<u4"),
            ],
            align=False,
        )
        if record_dtype.itemsize != record_size:
            raise ValueError(
                f"sample dtype size mismatch: dtype={record_dtype.itemsize}, record={record_size}"
            )
        records = np.frombuffer(payload, dtype=record_dtype, count=sample_count)
        features = np.asarray(records["features"], dtype=np.float32)
        targets = np.asarray(records["targets"], dtype=np.float32)
        valid_actions = np.asarray(records["valid_actions"], dtype=np.int64)
        action_masks = np.zeros((sample_count, max_actions), dtype=np.uint8)
        clipped = np.clip(valid_actions, 0, max_actions)
        for idx, count in enumerate(clipped):
            action_masks[idx, :count] = 1
        iterations = np.asarray(records["iteration"], dtype=np.int64)
        return features, targets, action_masks, iterations

    record_size = input_dim * 4 + max_actions * 4 + max_actions + 4
    if len(payload) % record_size != 0:
        raise ValueError(
            f"sample payload size {len(payload)} is not divisible by record_size {record_size}"
        )
    sample_count = len(payload) // record_size
    record_dtype = np.dtype(
        [
            ("features", ("<f4", input_dim)),
            ("targets", ("<f4", max_actions)),
            ("action_mask", ("u1", max_actions)),
            ("iteration", "<u4"),
        ],
        align=False,
    )
    if record_dtype.itemsize != record_size:
        raise ValueError(
            f"sample dtype size mismatch: dtype={record_dtype.itemsize}, record={record_size}"
        )
    records = np.frombuffer(payload, dtype=record_dtype, count=sample_count)
    features = np.asarray(records["features"], dtype=np.float32)
    targets = np.asarray(records["targets"], dtype=np.float32)
    action_masks = np.asarray(records["action_mask"], dtype=np.uint8)
    action_masks = np.where(action_masks > 0, 1, 0).astype(np.uint8, copy=False)
    iterations = np.asarray(records["iteration"], dtype=np.int64)
    return features, targets, action_masks, iterations


def load_model_weights(model: torch.nn.Module, state_path: Path, device: torch.device) -> bool:
    if not state_path.exists():
        return False
    checkpoint = torch.load(state_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return True


def train_advantage_step(
    model: DeepCfrNet,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    action_masks: torch.Tensor,
    sample_weights: torch.Tensor,
    huber_delta: float = 20.0,
) -> torch.Tensor:
    mask = action_masks.float()
    pred = model(batch_x, action_mask=mask, strategy_mode=False)
    error = pred - batch_y
    abs_error = error.abs()
    delta = float(huber_delta)
    huber = torch.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta),
    )
    per_sample = (huber * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return (per_sample * sample_weights).mean()


def train_strategy_step(
    model: DeepCfrNet,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    action_masks: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    mask = action_masks.float()
    probs = model(batch_x, action_mask=mask, strategy_mode=True)
    target = torch.clamp(batch_y, min=0.0) * mask
    target_mass = target.sum(dim=1, keepdim=True)
    uniform_target = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    normalized_target = target / target_mass.clamp(min=1e-8)
    target = torch.where(target_mass > 1e-8, normalized_target, uniform_target)
    per_sample = -(target * torch.log(torch.clamp(probs, min=1e-8)) * mask).sum(dim=1)
    return (per_sample * sample_weights).mean()


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, model: DeepCfrNet, strategy_mode: bool):
        super().__init__()
        self.model = model
        self.strategy_mode = strategy_mode

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        return self.model(x, action_mask=action_mask, strategy_mode=self.strategy_mode)


def export_fp16_onnx_from_fp32(model_out: Path, fp16_out: Path) -> None:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load_model(str(model_out))
    fp16_model = convert_float_to_float16(model, keep_io_types=True)
    onnx.save_model(fp16_model, str(fp16_out))


def export_onnx(
    model: DeepCfrNet,
    model_out: Path,
    network_type: str,
    opset: int,
    device: torch.device,
    export_fp16: bool = False,
    fp16_out: Path | None = None,
) -> None:
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    wrapper = OnnxExportWrapper(model, strategy_mode=(network_type == "strategy")).to(device)
    dummy = torch.randn(1, INPUT_DIM, device=device)
    dummy_mask = torch.ones((1, MAX_ACTIONS), dtype=torch.float32, device=device)
    torch.onnx.export(
        wrapper,
        (dummy, dummy_mask),
        model_out,
        input_names=["input", "action_mask"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "action_mask": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    if export_fp16:
        fp16_path = fp16_out or model_out.with_name(f"{model_out.stem}.fp16{model_out.suffix}")
        fp16_path.parent.mkdir(parents=True, exist_ok=True)
        export_fp16_onnx_from_fp32(model_out, fp16_path)


def main() -> None:
    configure_warning_filters()
    args = parse_args()
    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    buffer_path = derive_buffer_path(args)
    state_path = derive_state_path(args)

    cfg = ModelConfig(
        input_dim=INPUT_DIM,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        max_actions=MAX_ACTIONS,
        dropout_p=args.dropout_p,
    )
    model = DeepCfrNet(cfg).to(device)

    if args.init_only:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "network_type": args.type,
                "input_dim": INPUT_DIM,
                "max_actions": MAX_ACTIONS,
                "hidden_dim": args.hidden_dim,
                "bottleneck_dim": args.bottleneck_dim,
            },
            state_path,
        )
        export_onnx(
            model,
            args.model_out,
            args.type,
            args.onnx_opset,
            device,
            export_fp16=bool(args.export_fp16_onnx),
            fp16_out=args.fp16_model_out,
        )
        log(f"[deep-cfr] init-only: created random {args.type} model")
        log(f"[deep-cfr] saved state={state_path}")
        log(f"[deep-cfr] exported onnx={args.model_out}")
        if args.export_fp16_onnx:
            fp16_path = args.fp16_model_out or args.model_out.with_name(
                f"{args.model_out.stem}.fp16{args.model_out.suffix}"
            )
            log(f"[deep-cfr] exported onnx fp16={fp16_path}")
        return

    expected_magic = ADVANTAGE_SAMPLE_MAGIC if args.type == "advantage" else STRATEGY_SAMPLE_MAGIC
    features, targets, action_masks, iterations = load_binary_samples(
        args.samples,
        expected_magic=expected_magic,
    )
    log(
        f"[deep-cfr] loaded {len(features)} samples from {args.samples} "
        f"(input={INPUT_DIM}, actions={MAX_ACTIONS})"
    )

    reservoir = ReservoirBuffer.load(buffer_path, default_max_size=args.buffer_size)
    reservoir.add_many(features, targets, action_masks, iterations, rng)
    log(f"[deep-cfr] reservoir size={len(reservoir)} / {args.buffer_size}")

    loaded = load_model_weights(model, state_path, device)
    log(f"[deep-cfr] model init={'checkpoint' if loaded else 'random'} ({state_path})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_cuda = device.type == "cuda"
    use_amp = use_cuda and not args.disable_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    if len(reservoir) == 0:
        raise ValueError("reservoir is empty after loading samples")
    batch_size = min(args.batch_size, len(reservoir))
    max_steps_by_reuse = max(
        1,
        int(math.ceil((len(reservoir) * args.max_sample_reuse_per_iter) / batch_size)),
    )
    effective_steps = min(args.steps, max_steps_by_reuse)
    if effective_steps < args.steps:
        log(
            "[deep-cfr] capped training steps to "
            f"{effective_steps}/{args.steps} "
            f"(max reuse {args.max_sample_reuse_per_iter:.2f}x)"
        )
    # Pre-sample all batch indices once to reduce per-step CPU overhead
    # in the hot training loop.
    batch_indices = rng.integers(0, len(reservoir), size=(effective_steps, batch_size), dtype=np.int64)
    log(f"[deep-cfr] pre-sampled {effective_steps} batches (batch_size={batch_size})")

    for step in range(1, effective_steps + 1):
        batch = reservoir.sample_from_indices(batch_indices[step - 1])
        batch_x = torch.from_numpy(batch.features)
        batch_y = torch.from_numpy(batch.targets)
        batch_mask = torch.from_numpy(batch.action_masks)
        batch_itr = torch.from_numpy(batch.iterations)
        if use_cuda:
            batch_x = batch_x.pin_memory()
            batch_y = batch_y.pin_memory()
            batch_mask = batch_mask.pin_memory()
            batch_itr = batch_itr.pin_memory()

        batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=use_cuda)
        batch_y = batch_y.to(device=device, dtype=torch.float32, non_blocking=use_cuda)
        batch_mask = batch_mask.to(device=device, dtype=torch.float32, non_blocking=use_cuda)
        batch_itr = batch_itr.to(device=device, dtype=torch.float32, non_blocking=use_cuda)
        weights = (batch_itr / batch_itr.max().clamp(min=1.0)).clamp(min=1e-4)
        weights = weights / weights.mean().clamp(min=1e-8)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            if args.type == "advantage":
                loss = train_advantage_step(
                    model,
                    batch_x,
                    batch_y,
                    batch_mask,
                    weights,
                    huber_delta=args.adv_huber_delta,
                )
            else:
                loss = train_strategy_step(model, batch_x, batch_y, batch_mask, weights)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0 or step == effective_steps:
            log(f"[deep-cfr] step={step}/{effective_steps} loss={loss.item():.6f}")

    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "network_type": args.type,
            "input_dim": INPUT_DIM,
            "max_actions": MAX_ACTIONS,
            "hidden_dim": args.hidden_dim,
            "bottleneck_dim": args.bottleneck_dim,
        },
        state_path,
    )
    reservoir.save(buffer_path)
    export_onnx(
        model,
        args.model_out,
        args.type,
        args.onnx_opset,
        device,
        export_fp16=bool(args.export_fp16_onnx),
        fp16_out=args.fp16_model_out,
    )
    log(f"[deep-cfr] saved state={state_path}")
    log(f"[deep-cfr] saved buffer={buffer_path}")
    log(f"[deep-cfr] exported onnx={args.model_out}")
    if args.export_fp16_onnx:
        fp16_path = args.fp16_model_out or args.model_out.with_name(
            f"{args.model_out.stem}.fp16{args.model_out.suffix}"
        )
        log(f"[deep-cfr] exported onnx fp16={fp16_path}")


if __name__ == "__main__":
    main()

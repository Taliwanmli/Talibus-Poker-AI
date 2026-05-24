from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SampleBatch:
    features: np.ndarray
    targets: np.ndarray
    action_masks: np.ndarray
    iterations: np.ndarray


class ReservoirBuffer:
    """Reservoir sampling buffer (Algorithm R)."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max(0, int(max_size))
        self.seen = 0
        self._size = 0
        self._input_dim: int | None = None
        self._max_actions: int | None = None
        self._features: np.ndarray | None = None
        self._targets: np.ndarray | None = None
        self._action_masks: np.ndarray | None = None
        self._iterations: np.ndarray | None = None

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def _ensure_storage(self, input_dim: int, max_actions: int) -> None:
        if self._features is None:
            self._input_dim = int(input_dim)
            self._max_actions = int(max_actions)
            self._features = np.zeros((self.max_size, input_dim), dtype=np.float32)
            self._targets = np.zeros((self.max_size, max_actions), dtype=np.float32)
            self._action_masks = np.zeros((self.max_size, max_actions), dtype=np.uint8)
            self._iterations = np.zeros((self.max_size,), dtype=np.int64)
            return

        assert self._input_dim is not None
        assert self._max_actions is not None
        if self._input_dim != input_dim or self._max_actions != max_actions:
            raise ValueError(
                "sample shape mismatch: "
                f"expected features={self._input_dim}, actions={self._max_actions}, "
                f"got features={input_dim}, actions={max_actions}"
            )

    def _set_dense_payload(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        action_masks: np.ndarray,
        iterations: np.ndarray,
    ) -> None:
        if features.ndim != 2 or targets.ndim != 2:
            raise ValueError("features and targets must be rank-2 arrays")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must have matching row counts")
        if action_masks.ndim != 2 or action_masks.shape[1] != targets.shape[1]:
            raise ValueError("action_masks must be rank-2 and match target action dimension")
        if action_masks.shape[0] != features.shape[0] or iterations.shape[0] != features.shape[0]:
            raise ValueError("action_masks and iterations must match features row count")

        if features.shape[0] == 0:
            # Keep storage lazy for empty payloads so the first add_many call can
            # establish the correct feature/action dimensions.
            self._size = 0
            return

        self._ensure_storage(features.shape[1], targets.shape[1])
        assert self._features is not None
        assert self._targets is not None
        assert self._action_masks is not None
        assert self._iterations is not None

        dense_action_masks = np.asarray(action_masks, dtype=np.uint8)
        dense_action_masks = np.where(dense_action_masks > 0, 1, 0).astype(np.uint8, copy=False)
        flat_iterations = np.asarray(iterations, dtype=np.int64).reshape(-1)
        n = min(features.shape[0], self.max_size)
        if n > 0 and features.shape[0] > n:
            # If the on-disk reservoir is larger than the configured capacity,
            # keep the most recent samples by iteration index.
            keep_idx = np.argpartition(flat_iterations, -n)[-n:]
            keep_idx = keep_idx[np.argsort(flat_iterations[keep_idx], kind="stable")]
            selected_features = np.asarray(features[keep_idx], dtype=np.float32)
            selected_targets = np.asarray(targets[keep_idx], dtype=np.float32)
            selected_action_masks = np.asarray(dense_action_masks[keep_idx], dtype=np.uint8)
            selected_iterations = np.asarray(flat_iterations[keep_idx], dtype=np.int64)
        else:
            selected_features = np.asarray(features[:n], dtype=np.float32)
            selected_targets = np.asarray(targets[:n], dtype=np.float32)
            selected_action_masks = np.asarray(dense_action_masks[:n], dtype=np.uint8)
            selected_iterations = np.asarray(flat_iterations[:n], dtype=np.int64)
        self._features[:n] = selected_features
        self._targets[:n] = selected_targets
        self._action_masks[:n] = selected_action_masks
        self._iterations[:n] = selected_iterations
        self._size = n
        self.seen = max(self.seen, n)

    def add_one(
        self,
        feature: np.ndarray,
        target: np.ndarray,
        action_mask: np.ndarray,
        iteration: int,
        rng: np.random.Generator,
    ) -> None:
        features = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        targets = np.asarray(target, dtype=np.float32).reshape(1, -1)
        masks = np.asarray(action_mask, dtype=np.uint8).reshape(1, -1)
        itrs = np.asarray([iteration], dtype=np.int64)
        self.add_many(features, targets, masks, itrs, rng)

    def add_many(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        action_masks: np.ndarray,
        iterations: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        action_masks = np.asarray(action_masks, dtype=np.uint8)
        if action_masks.ndim == 1:
            action_masks = action_masks.reshape(-1, 1)
        iterations = np.asarray(iterations, dtype=np.int64).reshape(-1)

        if features.ndim != 2 or targets.ndim != 2:
            raise ValueError("features and targets must be rank-2 arrays")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must have matching row counts")
        if action_masks.ndim != 2 or action_masks.shape[1] != targets.shape[1]:
            raise ValueError("action_masks must be rank-2 and match target action dimension")
        if action_masks.shape[0] != features.shape[0] or iterations.shape[0] != features.shape[0]:
            raise ValueError("action_masks and iterations must match features row count")
        if features.shape[0] == 0:
            return

        self._ensure_storage(features.shape[1], targets.shape[1])
        assert self._features is not None
        assert self._targets is not None
        assert self._action_masks is not None
        assert self._iterations is not None

        count = features.shape[0]
        seen_before = self.seen
        self.seen += count

        if self.max_size <= 0:
            return

        start_idx = 0
        if self._size < self.max_size:
            fill = min(self.max_size - self._size, count)
            write_slice = slice(self._size, self._size + fill)
            self._features[write_slice] = features[:fill]
            self._targets[write_slice] = targets[:fill]
            self._action_masks[write_slice] = np.where(action_masks[:fill] > 0, 1, 0)
            self._iterations[write_slice] = iterations[:fill]
            self._size += fill
            start_idx = fill

        for local_idx in range(start_idx, count):
            seen_idx = seen_before + local_idx + 1
            slot = int(rng.integers(0, seen_idx))
            if slot < self.max_size:
                self._features[slot] = features[local_idx]
                self._targets[slot] = targets[local_idx]
                self._action_masks[slot] = np.where(action_masks[local_idx] > 0, 1, 0)
                self._iterations[slot] = int(iterations[local_idx])

    def sample(self, batch_size: int, rng: np.random.Generator) -> SampleBatch:
        if self._size == 0:
            raise ValueError("reservoir is empty")
        assert self._features is not None
        assert self._targets is not None
        assert self._action_masks is not None
        assert self._iterations is not None

        n = min(batch_size, self._size)
        idxs = rng.integers(0, self._size, size=n)
        feats = self._features[idxs]
        targs = self._targets[idxs]
        masks = self._action_masks[idxs]
        iters = self._iterations[idxs]
        return SampleBatch(feats, targs, masks, iters)

    def sample_from_indices(self, indices: np.ndarray) -> SampleBatch:
        if self._size == 0:
            raise ValueError("reservoir is empty")
        assert self._features is not None
        assert self._targets is not None
        assert self._action_masks is not None
        assert self._iterations is not None

        idxs = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idxs.size == 0:
            raise ValueError("indices must be non-empty")
        feats = self._features[idxs]
        targs = self._targets[idxs]
        masks = self._action_masks[idxs]
        iters = self._iterations[idxs]
        return SampleBatch(feats, targs, masks, iters)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._features is None:
            features = np.zeros((0, 0), dtype=np.float32)
            targets = np.zeros((0, 0), dtype=np.float32)
            action_masks = np.zeros((0, 0), dtype=np.uint8)
            iterations = np.zeros((0,), dtype=np.int64)
            input_dim = 0
            max_actions = 0
        else:
            assert self._targets is not None
            assert self._action_masks is not None
            assert self._iterations is not None
            assert self._input_dim is not None
            assert self._max_actions is not None
            features = self._features[: self._size]
            targets = self._targets[: self._size]
            action_masks = self._action_masks[: self._size]
            iterations = self._iterations[: self._size]
            input_dim = self._input_dim
            max_actions = self._max_actions

        payload = {
            "version": 2,
            "max_size": self.max_size,
            "seen": self.seen,
            "size": self._size,
            "input_dim": input_dim,
            "max_actions": max_actions,
            "features": features,
            "targets": targets,
            "action_masks": action_masks,
            "iterations": iterations,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path, default_max_size: int) -> "ReservoirBuffer":
        if not path.exists():
            return cls(default_max_size)
        with path.open("rb") as f:
            payload: Any = pickle.load(f)

        out = cls(default_max_size)
        if not isinstance(payload, dict):
            return out

        out.seen = int(payload.get("seen", 0))

        version = int(payload.get("version", 1))
        if version >= 2 and "features" in payload:
            features = np.asarray(payload.get("features", np.zeros((0, 0), dtype=np.float32)))
            targets = np.asarray(payload.get("targets", np.zeros((0, 0), dtype=np.float32)))
            action_masks = np.asarray(
                payload.get("action_masks", np.zeros((0, 0), dtype=np.uint8))
            )
            iterations = np.asarray(payload.get("iterations", np.zeros((0,), dtype=np.int64)))
            if action_masks.size == 0 and "valid_actions" in payload:
                # Backward compatibility for v2 payloads saved before action-mask upgrade.
                valid_actions = np.asarray(payload.get("valid_actions", np.zeros((0,), dtype=np.int64)))
                if features.ndim == 2 and targets.ndim == 2:
                    action_masks = np.zeros((features.shape[0], targets.shape[1]), dtype=np.uint8)
                    clipped = np.asarray(valid_actions, dtype=np.int64).reshape(-1)
                    clipped = np.clip(clipped, 0, targets.shape[1])
                    for idx, count in enumerate(clipped):
                        action_masks[idx, :count] = 1
            if features.ndim == 2 and targets.ndim == 2:
                out._set_dense_payload(features, targets, action_masks, iterations)
            return out

        # Backward compatibility: legacy payload saved as list of tuples.
        items = payload.get("items", [])
        if not items:
            return out
        features = np.asarray([item[0] for item in items], dtype=np.float32)
        targets = np.asarray([item[1] for item in items], dtype=np.float32)
        valid_actions = np.asarray([int(item[2]) for item in items], dtype=np.int64)
        action_masks = np.zeros((len(items), targets.shape[1]), dtype=np.uint8)
        clipped = np.clip(valid_actions, 0, targets.shape[1])
        for idx, count in enumerate(clipped):
            action_masks[idx, :count] = 1
        iterations = np.asarray([int(item[3]) for item in items], dtype=np.int64)
        if features.ndim == 2 and targets.ndim == 2:
            out._set_dense_payload(features, targets, action_masks, iterations)
        return out


class DiskBackedReservoirBuffer:
    """Reservoir sampling buffer backed by on-disk memmap arrays."""

    METADATA_FILE = "metadata.json"
    FEATURES_FILE = "features.bin"
    TARGETS_FILE = "targets.bin"
    ACTION_MASKS_FILE = "action_masks.bin"
    ITERATIONS_FILE = "iterations.bin"

    def __init__(self, max_size: int, disk_dir: Path) -> None:
        self.max_size = max(0, int(max_size))
        self.disk_dir = Path(disk_dir)
        self.seen = 0
        self._size = 0
        self._input_dim: int | None = None
        self._max_actions: int | None = None
        self._features: np.memmap | None = None
        self._targets: np.memmap | None = None
        self._action_masks: np.memmap | None = None
        self._iterations: np.memmap | None = None

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def _metadata_path(self) -> Path:
        return self.disk_dir / self.METADATA_FILE

    def _features_path(self) -> Path:
        return self.disk_dir / self.FEATURES_FILE

    def _targets_path(self) -> Path:
        return self.disk_dir / self.TARGETS_FILE

    def _action_masks_path(self) -> Path:
        return self.disk_dir / self.ACTION_MASKS_FILE

    def _iterations_path(self) -> Path:
        return self.disk_dir / self.ITERATIONS_FILE

    def _open_existing_storage(self, input_dim: int, max_actions: int) -> None:
        self._input_dim = int(input_dim)
        self._max_actions = int(max_actions)
        if self.max_size <= 0:
            return
        self._features = np.memmap(
            self._features_path(),
            mode="r+",
            dtype=np.float32,
            shape=(self.max_size, self._input_dim),
        )
        self._targets = np.memmap(
            self._targets_path(),
            mode="r+",
            dtype=np.float32,
            shape=(self.max_size, self._max_actions),
        )
        self._action_masks = np.memmap(
            self._action_masks_path(),
            mode="r+",
            dtype=np.uint8,
            shape=(self.max_size, self._max_actions),
        )
        self._iterations = np.memmap(
            self._iterations_path(),
            mode="r+",
            dtype=np.int64,
            shape=(self.max_size,),
        )

    def _ensure_storage(self, input_dim: int, max_actions: int) -> None:
        if self._features is None:
            self._input_dim = int(input_dim)
            self._max_actions = int(max_actions)
            if self.max_size <= 0:
                return
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            self._features = np.memmap(
                self._features_path(),
                mode="w+",
                dtype=np.float32,
                shape=(self.max_size, self._input_dim),
            )
            self._targets = np.memmap(
                self._targets_path(),
                mode="w+",
                dtype=np.float32,
                shape=(self.max_size, self._max_actions),
            )
            self._action_masks = np.memmap(
                self._action_masks_path(),
                mode="w+",
                dtype=np.uint8,
                shape=(self.max_size, self._max_actions),
            )
            self._iterations = np.memmap(
                self._iterations_path(),
                mode="w+",
                dtype=np.int64,
                shape=(self.max_size,),
            )
            return

        assert self._input_dim is not None
        assert self._max_actions is not None
        if self._input_dim != input_dim or self._max_actions != max_actions:
            raise ValueError(
                "sample shape mismatch: "
                f"expected features={self._input_dim}, actions={self._max_actions}, "
                f"got features={input_dim}, actions={max_actions}"
            )

    def _flush_memmaps(self) -> None:
        for arr in (self._features, self._targets, self._action_masks, self._iterations):
            if arr is not None:
                arr.flush()

    def add_one(
        self,
        feature: np.ndarray,
        target: np.ndarray,
        action_mask: np.ndarray,
        iteration: int,
        rng: np.random.Generator,
    ) -> None:
        features = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        targets = np.asarray(target, dtype=np.float32).reshape(1, -1)
        masks = np.asarray(action_mask, dtype=np.uint8).reshape(1, -1)
        itrs = np.asarray([iteration], dtype=np.int64)
        self.add_many(features, targets, masks, itrs, rng)

    def add_many(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        action_masks: np.ndarray,
        iterations: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        action_masks = np.asarray(action_masks, dtype=np.uint8)
        if action_masks.ndim == 1:
            action_masks = action_masks.reshape(-1, 1)
        iterations = np.asarray(iterations, dtype=np.int64).reshape(-1)

        if features.ndim != 2 or targets.ndim != 2:
            raise ValueError("features and targets must be rank-2 arrays")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must have matching row counts")
        if action_masks.ndim != 2 or action_masks.shape[1] != targets.shape[1]:
            raise ValueError("action_masks must be rank-2 and match target action dimension")
        if action_masks.shape[0] != features.shape[0] or iterations.shape[0] != features.shape[0]:
            raise ValueError("action_masks and iterations must match features row count")
        if features.shape[0] == 0:
            return

        binary_action_masks = np.where(action_masks > 0, 1, 0).astype(np.uint8, copy=False)
        self._ensure_storage(features.shape[1], targets.shape[1])
        assert self._features is not None or self.max_size <= 0
        assert self._targets is not None or self.max_size <= 0
        assert self._action_masks is not None or self.max_size <= 0
        assert self._iterations is not None or self.max_size <= 0

        count = features.shape[0]
        seen_before = self.seen
        self.seen += count

        if self.max_size <= 0:
            return

        start_idx = 0
        if self._size < self.max_size:
            fill = min(self.max_size - self._size, count)
            write_slice = slice(self._size, self._size + fill)
            self._features[write_slice] = features[:fill]
            self._targets[write_slice] = targets[:fill]
            self._action_masks[write_slice] = binary_action_masks[:fill]
            self._iterations[write_slice] = iterations[:fill]
            self._size += fill
            start_idx = fill

        for local_idx in range(start_idx, count):
            seen_idx = seen_before + local_idx + 1
            slot = int(rng.integers(0, seen_idx))
            if slot < self.max_size:
                self._features[slot] = features[local_idx]
                self._targets[slot] = targets[local_idx]
                self._action_masks[slot] = binary_action_masks[local_idx]
                self._iterations[slot] = int(iterations[local_idx])

    def sample(self, batch_size: int, rng: np.random.Generator) -> SampleBatch:
        if self._size == 0:
            raise ValueError("reservoir is empty")
        n = min(batch_size, self._size)
        idxs = rng.integers(0, self._size, size=n, dtype=np.int64)
        return self.sample_from_indices(idxs)

    def sample_from_indices(self, indices: np.ndarray) -> SampleBatch:
        if self._size == 0:
            raise ValueError("reservoir is empty")
        assert self._features is not None
        assert self._targets is not None
        assert self._action_masks is not None
        assert self._iterations is not None

        idxs = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idxs.size == 0:
            raise ValueError("indices must be non-empty")
        if np.min(idxs) < 0 or np.max(idxs) >= self._size:
            raise ValueError("sample indices are out of range")

        # Sorting converts random gather into near-sequential reads on disk.
        sorted_idxs = np.sort(idxs, kind="stable")
        feats = np.asarray(self._features[sorted_idxs], dtype=np.float32)
        targs = np.asarray(self._targets[sorted_idxs], dtype=np.float32)
        masks = np.asarray(self._action_masks[sorted_idxs], dtype=np.uint8)
        iters = np.asarray(self._iterations[sorted_idxs], dtype=np.int64)
        return SampleBatch(feats, targs, masks, iters)

    def bulk_load_unique(self, sorted_unique_indices: np.ndarray) -> SampleBatch:
        if self._size == 0:
            raise ValueError("reservoir is empty")
        assert self._features is not None
        assert self._targets is not None
        assert self._action_masks is not None
        assert self._iterations is not None

        idxs = np.asarray(sorted_unique_indices, dtype=np.int64).reshape(-1)
        if idxs.size == 0:
            raise ValueError("sorted_unique_indices must be non-empty")
        if np.min(idxs) < 0 or np.max(idxs) >= self._size:
            raise ValueError("bulk load indices are out of range")
        if np.any(idxs[1:] < idxs[:-1]):
            raise ValueError("bulk load indices must be sorted in non-decreasing order")
        if idxs.size > 1 and np.any(idxs[1:] == idxs[:-1]):
            raise ValueError("bulk load indices must be unique")

        feats = np.asarray(self._features[idxs], dtype=np.float32)
        targs = np.asarray(self._targets[idxs], dtype=np.float32)
        masks = np.asarray(self._action_masks[idxs], dtype=np.uint8)
        iters = np.asarray(self._iterations[idxs], dtype=np.int64)
        return SampleBatch(feats, targs, masks, iters)

    def save(self, _path: Path | None = None) -> None:
        self.disk_dir.mkdir(parents=True, exist_ok=True)
        self._flush_memmaps()
        payload = {
            "version": 1,
            "max_size": self.max_size,
            "seen": int(self.seen),
            "size": int(self._size),
            "input_dim": int(self._input_dim or 0),
            "max_actions": int(self._max_actions or 0),
        }
        metadata_path = self._metadata_path()
        tmp_path = metadata_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(metadata_path)

    @classmethod
    def load(cls, disk_dir: Path, default_max_size: int) -> "DiskBackedReservoirBuffer":
        out = cls(default_max_size, disk_dir)
        metadata_path = out._metadata_path()
        if not metadata_path.exists():
            return out

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return out

        stored_max_size = int(payload.get("max_size", out.max_size))
        if stored_max_size != out.max_size:
            raise ValueError(
                "disk buffer max_size mismatch: "
                f"metadata={stored_max_size}, configured={out.max_size}"
            )

        out.seen = max(0, int(payload.get("seen", 0)))
        out._size = max(0, int(payload.get("size", 0)))
        if out._size > out.max_size:
            raise ValueError(
                f"disk buffer size out of range: size={out._size}, max_size={out.max_size}"
            )
        input_dim = int(payload.get("input_dim", 0))
        max_actions = int(payload.get("max_actions", 0))
        if input_dim > 0 and max_actions > 0:
            try:
                out._open_existing_storage(input_dim, max_actions)
            except FileNotFoundError:
                if out._size > 0:
                    raise
                out._input_dim = input_dim
                out._max_actions = max_actions
        return out

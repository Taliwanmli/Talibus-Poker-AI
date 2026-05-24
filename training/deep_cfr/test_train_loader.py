from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from model import INPUT_DIM, MAX_ACTIONS
from train import SAMPLE_VERSION, STRATEGY_SAMPLE_MAGIC, load_binary_samples


class TrainLoaderHeaderTests(unittest.TestCase):
    def test_rejects_stale_max_actions_header(self) -> None:
        header = struct.pack(
            "<4sIII",
            STRATEGY_SAMPLE_MAGIC,
            int(SAMPLE_VERSION),
            int(INPUT_DIM),
            int(MAX_ACTIONS - 1),
        )
        with tempfile.TemporaryDirectory(prefix="talibus_train_loader_") as tmpdir:
            path = Path(tmpdir) / "stale_max_actions.bin"
            path.write_bytes(header)
            with self.assertRaisesRegex(ValueError, "max_actions mismatch"):
                load_binary_samples(path, expected_magic=STRATEGY_SAMPLE_MAGIC)


if __name__ == "__main__":
    unittest.main()

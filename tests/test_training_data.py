import tempfile
import unittest
from pathlib import Path

import numpy as np

from bobail_config import POLICY_SIZE
from bobail_metrics import MetricsLogger
from bobail_training_data import (
    archive_training_data_to_buffer,
    complete_training_data_from_buffer,
    npz_sample_count,
    preprocess,
)
from bobail_training_data import next_episode


def write_dataset(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = len(values)
    x = np.zeros((sample_count, 3, 5, 5), dtype=np.float32)
    for index, value in enumerate(values):
        x[index, 0, 0, 0] = value
    y_pi = np.zeros((sample_count, POLICY_SIZE), dtype=np.float32)
    y_pi[:, 0] = 1.0
    y_v = np.array([[value] for value in values], dtype=np.float32)
    np.savez(path, x=x, y_pi=y_pi, y_v=y_v)


def dataset_values(path: Path) -> list[float]:
    with np.load(path) as data:
        return [float(value) for value in data["x"][:, 0, 0, 0]]


class TrainingDataTests(unittest.TestCase):
    def test_next_episode_ignores_self_play_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = MetricsLogger("training_counter", directory=temp_dir)
            logger.append("self_play_generation", {"episode": 14, "games": 100})
            logger.append("self_play_generation", {"episode": 23, "games": 100})

            self.assertEqual(next_episode(logger), 1)

    def test_next_episode_follows_trained_network_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = MetricsLogger("training_counter", directory=temp_dir)
            logger.append("self_play_generation", {"episode": 14, "games": 100})
            logger.append("training_epoch", {"episode": 14, "epoch": 1, "train_loss": 1.0})
            logger.append("accuracy", {"episode": 15, "train_top1": 0.2})

            self.assertEqual(next_episode(logger), 16)

    def test_complete_training_data_uses_recent_buffer_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            training_path = temp_path / "merged_data.npz"
            buffer_path = temp_path / "training_buffer" / "replay_buffer.npz"
            write_dataset(training_path, [10.0, 11.0])
            write_dataset(buffer_path, [1.0, 2.0, 3.0, 4.0, 5.0])

            completed_count = complete_training_data_from_buffer(training_path, buffer_path, max_raw_samples=5)

            self.assertEqual(completed_count, 5)
            self.assertEqual(dataset_values(training_path), [3.0, 4.0, 5.0, 10.0, 11.0])

    def test_complete_training_data_skips_buffer_when_current_is_large_enough(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            training_path = temp_path / "merged_data.npz"
            buffer_path = temp_path / "training_buffer" / "replay_buffer.npz"
            write_dataset(training_path, [10.0, 11.0, 12.0])
            write_dataset(buffer_path, [1.0, 2.0, 3.0])

            completed_count = complete_training_data_from_buffer(training_path, buffer_path, max_raw_samples=3)

            self.assertEqual(completed_count, 3)
            self.assertEqual(dataset_values(training_path), [10.0, 11.0, 12.0])

    def test_archive_training_data_keeps_only_recent_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            training_path = temp_path / "merged_data.npz"
            buffer_path = temp_path / "training_buffer" / "replay_buffer.npz"
            write_dataset(training_path, [10.0, 11.0, 12.0])
            write_dataset(buffer_path, [1.0, 2.0, 3.0, 4.0])

            kept_count = archive_training_data_to_buffer(training_path, buffer_path, max_raw_samples=5)

            self.assertEqual(kept_count, 5)
            self.assertEqual(npz_sample_count(buffer_path), 5)
            self.assertEqual(dataset_values(buffer_path), [3.0, 4.0, 10.0, 11.0, 12.0])

    def test_preprocess_dedup_prevents_train_val_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "merged_data.npz"
            write_dataset(data_path, [float(index) for index in range(12)] + [1.0, 2.0, 3.0])

            x_train, _, x_val, _, _, _ = preprocess(str(data_path))

            train_keys = {tuple(item.flatten().tolist()) for item in x_train}
            val_keys = {tuple(item.flatten().tolist()) for item in x_val}
            self.assertFalse(train_keys & val_keys)


if __name__ == "__main__":
    unittest.main()

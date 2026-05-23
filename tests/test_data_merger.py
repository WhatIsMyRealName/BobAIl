import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataMerger import merge_npz_files


class DataMergerTests(unittest.TestCase):
    def test_merge_keeps_existing_output_and_deletes_only_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "merged_data.npz"
            source_path = temp_path / "advanced_data.npz"
            np.savez(
                output_path,
                x=np.array([[1.0]], dtype=np.float32),
                y_pi=np.array([[0.25, 0.75]], dtype=np.float32),
                y_v=np.array([[1.0]], dtype=np.float32),
            )
            np.savez(
                source_path,
                x=np.array([[2.0], [3.0]], dtype=np.float32),
                y_pi=np.array([[0.5, 0.5], [1.0, 0.0]], dtype=np.float32),
                y_v=np.array([[0.0], [-1.0]], dtype=np.float32),
            )

            merge_npz_files(directory=temp_dir, output_file="merged_data.npz")

            self.assertFalse(source_path.exists())
            self.assertTrue(output_path.exists())
            with np.load(output_path) as merged:
                np.testing.assert_array_equal(
                    merged["x"],
                    np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
                )
                np.testing.assert_array_equal(
                    merged["y_v"],
                    np.array([[1.0], [0.0], [-1.0]], dtype=np.float32),
                )

    def test_merge_ignores_training_buffer_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "merged_data.npz"
            source_path = temp_path / "advanced_data.npz"
            buffer_path = temp_path / "training_buffer"
            buffer_path.mkdir()
            np.savez(
                source_path,
                x=np.array([[1.0]], dtype=np.float32),
                y_pi=np.array([[1.0, 0.0]], dtype=np.float32),
                y_v=np.array([[1.0]], dtype=np.float32),
            )
            np.savez(
                buffer_path / "replay_buffer.npz",
                x=np.array([[99.0]], dtype=np.float32),
                y_pi=np.array([[0.0, 1.0]], dtype=np.float32),
                y_v=np.array([[-1.0]], dtype=np.float32),
            )

            merge_npz_files(directory=temp_dir, output_file="merged_data.npz")

            self.assertTrue((buffer_path / "replay_buffer.npz").exists())
            with np.load(output_path) as merged:
                np.testing.assert_array_equal(merged["x"], np.array([[1.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

import numpy as np
from pathlib import Path

from bobail_config import TRAINING_BUFFER_DIR


def merge_npz_files(directory: str = ".", output_file: str = "merged_data.npz") -> None:
    directory_path = Path(directory)
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = directory_path / output_path
    training_buffer_path = Path(TRAINING_BUFFER_DIR)
    if not training_buffer_path.is_absolute():
        training_buffer_path = directory_path / training_buffer_path
    training_buffer_path = training_buffer_path.resolve()
    source_files = sorted(
        path
        for path in directory_path.glob("*.npz")
        if path.resolve() != output_path.resolve()
        and training_buffer_path not in path.resolve().parents
    )
    npz_files = ([output_path] if output_path.exists() else []) + source_files

    if not npz_files:
        print("No .npz files found")
        return

    merged_data: dict[str, np.ndarray] = {}
    for npz_file in npz_files:
        with np.load(npz_file) as data:
            for key in data.files:
                values = data[key]
                if key in merged_data:
                    merged_data[key] = np.concatenate([merged_data[key], values])
                else:
                    merged_data[key] = values.copy()

    np.savez(output_path, **merged_data)
    print(f"Merged data saved to {output_path}")

    for npz_file in source_files:
        npz_file.unlink()
        print(f"Deleted {npz_file}")


if __name__ == "__main__":
    merge_npz_files()


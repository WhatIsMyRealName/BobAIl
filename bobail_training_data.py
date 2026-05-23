import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from bobail_config import (
    BOARD_SIZE,
    DIRS,
    DIR_TO_IDX,
    ITERATION_CHECKPOINT_DIR,
    MERGED_TRAINING_DATA_PATH,
    PENDING_SELF_PLAY_METRICS_PATH,
    POLICY_SHAPE,
    POLICY_SIZE,
    RAW_SELF_PLAY_DATA_PATH,
    RNG,
    SAVED_NETWORK_PATH,
    TRAINING_BUFFER_DIR,
    TRAINING_BUFFER_MAX_RAW_SAMPLES,
)
from bobail_metrics import MetricsLogger, aggregate_numeric
from bobail_model_zero import _Network


NPZ_ARRAY_KEYS = ("x", "y_pi", "y_v")


@dataclass(frozen=True)
class PreprocessStats:
    raw_samples: int
    augmented_samples: int
    dedup_samples: int
    train_samples: int
    val_samples: int
    y_pi_entropy_mean: float
    y_v_mean: float


@dataclass(frozen=True)
class PreprocessedData:
    x_train: np.ndarray
    y_train: tuple[np.ndarray, np.ndarray]
    x_val: np.ndarray
    y_val: tuple[np.ndarray, np.ndarray]
    stats: PreprocessStats


def build_bobail_network() -> _Network:
    return _Network()


def load_network() -> _Network:
    if os.path.exists(SAVED_NETWORK_PATH):
        return _Network.load(SAVED_NETWORK_PATH)
    print("No saved network found. Building a new network.")
    return build_bobail_network()


def preprocess_with_stats(path: str) -> PreprocessedData:
    with np.load(path) as data:
        x_orig = data["x"].copy()
        y_pi_orig = data["y_pi"].copy()
        y_v_orig = data["y_v"].copy()
    num_samples = x_orig.shape[0]

    dir_map_cols = [DIR_TO_IDX[(dx, -dy)] for dx, dy in DIRS]
    x_aug = np.flip(x_orig.copy(), axis=3)
    y_v_aug = y_v_orig.copy()
    policy_aug = np.flip(y_pi_orig.reshape(num_samples, *POLICY_SHAPE), axis=3)
    policy_aug = policy_aug[:, dir_map_cols, :, :, :]
    policy_aug = policy_aug[:, :, :, :, dir_map_cols]
    y_pi_aug = policy_aug.reshape(num_samples, -1)

    x_all = np.concatenate([x_orig, x_aug], axis=0)
    y_v_all = np.concatenate([y_v_orig, y_v_aug], axis=0)
    y_pi_all = np.concatenate([y_pi_orig, y_pi_aug], axis=0)

    aggregated: dict[tuple[float, ...], list[Any]] = {}
    for x_item, policy_item, value_item in zip(x_all, y_pi_all, y_v_all):
        key = tuple(x_item.flatten().tolist())
        if key in aggregated:
            aggregated[key][0] += policy_item
            aggregated[key][1] += value_item
            aggregated[key][2] += 1
        else:
            aggregated[key] = [policy_item, value_item, 1, x_item.copy()]

    x_dedup = []
    y_pi_dedup = []
    y_v_dedup = []
    for policy_sum, value_sum, count, x_example in aggregated.values():
        x_dedup.append(x_example)
        y_pi_dedup.append(policy_sum / count)
        y_v_dedup.append(value_sum / count)

    x_dedup_array = np.stack(x_dedup).astype(np.float32)
    y_pi_dedup_array = np.stack(y_pi_dedup, dtype=np.float32)
    y_v_dedup_array = np.stack(y_v_dedup, dtype=np.float32)

    split_size = int(0.8 * len(x_dedup_array))
    indices = RNG.permutation(len(x_dedup_array))
    train_idx = indices[:split_size]
    val_idx = indices[split_size:]
    x_train = x_dedup_array[train_idx]
    y_train = (y_pi_dedup_array[train_idx], y_v_dedup_array[train_idx])
    x_val = x_dedup_array[val_idx]
    y_val = (y_pi_dedup_array[val_idx], y_v_dedup_array[val_idx])

    policy_probs = y_pi_dedup_array.astype(np.float64)
    policy_probs = np.clip(policy_probs, 1e-12, None)
    policy_probs /= policy_probs.sum(axis=1, keepdims=True)
    policy_entropy = -(policy_probs * np.log(policy_probs)).sum(axis=1)
    stats = PreprocessStats(
        raw_samples=int(num_samples),
        augmented_samples=int(len(x_all)),
        dedup_samples=int(len(x_dedup_array)),
        train_samples=int(len(x_train)),
        val_samples=int(len(x_val)),
        y_pi_entropy_mean=float(policy_entropy.mean()),
        y_v_mean=float(y_v_dedup_array.mean()),
    )
    return PreprocessedData(x_train, y_train, x_val, y_val, stats)


def preprocess(path: str) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray, tuple[np.ndarray, np.ndarray], int, int]:
    preprocessed = preprocess_with_stats(path)
    return (
        preprocessed.x_train,
        preprocessed.y_train,
        preprocessed.x_val,
        preprocessed.y_val,
        preprocessed.stats.raw_samples,
        preprocessed.stats.train_samples,
    )


def replay_buffer_path(directory: str | os.PathLike[str] = TRAINING_BUFFER_DIR) -> str:
    return os.path.join(os.fspath(directory), "replay_buffer.npz")


def load_npz_arrays(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].copy() for key in NPZ_ARRAY_KEYS}


def save_npz_arrays(path: str | os.PathLike[str], arrays: dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    np.savez(path, **{key: arrays[key] for key in NPZ_ARRAY_KEYS})


def npz_sample_count(path: str | os.PathLike[str]) -> int:
    with np.load(path) as data:
        return int(data["x"].shape[0])


def concatenate_npz_arrays(datasets: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([dataset[key] for dataset in datasets], axis=0)
        for key in NPZ_ARRAY_KEYS
    }


def tail_npz_arrays(arrays: dict[str, np.ndarray], sample_count: int) -> dict[str, np.ndarray]:
    if sample_count == 0:
        return {key: arrays[key][:0].copy() for key in NPZ_ARRAY_KEYS}
    return {key: arrays[key][-sample_count:].copy() for key in NPZ_ARRAY_KEYS}


def complete_training_data_from_buffer(
    training_data_path: str | os.PathLike[str] = MERGED_TRAINING_DATA_PATH,
    buffer_path: str | os.PathLike[str] = replay_buffer_path(),
    max_raw_samples: int = TRAINING_BUFFER_MAX_RAW_SAMPLES,
) -> int:
    current_count = npz_sample_count(training_data_path)
    if current_count >= max_raw_samples:
        print(f"[Buffer] Current training data already has {current_count} samples.")
        return current_count
    if not os.path.exists(buffer_path):
        print(f"[Buffer] No replay buffer found. Training with {current_count} samples.")
        return current_count

    needed_count = max_raw_samples - current_count
    current_arrays = load_npz_arrays(training_data_path)
    buffer_arrays = load_npz_arrays(buffer_path)
    buffer_count = int(buffer_arrays["x"].shape[0])
    selected_buffer_count = min(needed_count, buffer_count)
    selected_buffer_arrays = tail_npz_arrays(buffer_arrays, selected_buffer_count)
    completed_arrays = concatenate_npz_arrays([selected_buffer_arrays, current_arrays])
    save_npz_arrays(training_data_path, completed_arrays)
    completed_count = int(completed_arrays["x"].shape[0])
    print(
        f"[Buffer] Added {selected_buffer_count} replay samples to training data "
        f"({current_count} -> {completed_count})."
    )
    return completed_count


def archive_training_data_to_buffer(
    training_data_path: str | os.PathLike[str] = MERGED_TRAINING_DATA_PATH,
    buffer_path: str | os.PathLike[str] = replay_buffer_path(),
    max_raw_samples: int = TRAINING_BUFFER_MAX_RAW_SAMPLES,
) -> int:
    training_arrays = load_npz_arrays(training_data_path)
    if os.path.exists(buffer_path):
        buffer_arrays = load_npz_arrays(buffer_path)
        combined_arrays = concatenate_npz_arrays([buffer_arrays, training_arrays])
    else:
        combined_arrays = training_arrays

    combined_count = int(combined_arrays["x"].shape[0])
    kept_count = min(combined_count, max_raw_samples)
    replay_arrays = tail_npz_arrays(combined_arrays, kept_count)
    save_npz_arrays(buffer_path, replay_arrays)
    print(f"[Buffer] Archived training data. Replay buffer samples: {kept_count}.")
    return kept_count


def evaluate_accuracy(
    network: _Network,
    x_data: np.ndarray,
    y_pi_data: np.ndarray,
    batch_size: int = 64,
) -> tuple[float, float]:
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    network.eval()

    with torch.inference_mode():
        for start in range(0, len(x_data), batch_size):
            end = min(start + batch_size, len(x_data))
            batch_size_actual = end - start
            x_batch = torch.from_numpy(x_data[start:end]).float().to("cpu")
            policy_logits, _ = network(x_batch)
            policy_probs = F.softmax(policy_logits, dim=1).detach().cpu().numpy()

            predicted_top1 = np.argmax(policy_probs.reshape(batch_size_actual, -1), axis=1)
            predicted_top5 = np.argsort(policy_probs.reshape(batch_size_actual, -1), axis=1)[:, -5:]
            true_moves = np.argmax(y_pi_data[start:end].reshape(batch_size_actual, -1), axis=1)
            correct_top1 += int((predicted_top1 == true_moves).sum())
            correct_top5 += sum(1 for row, true_move in zip(predicted_top5, true_moves) if true_move in row)
            total += batch_size_actual

    return correct_top1 / total, correct_top5 / total


def next_episode(metrics_logger: MetricsLogger) -> int:
    training_phases = {"training_epoch", "accuracy"}
    episodes = [
        int(record["episode"])
        for record in metrics_logger.load()
        if record.get("phase") in training_phases and isinstance(record.get("episode"), (int, float))
    ]
    return max(episodes) + 1 if episodes else 1


def save_iteration_checkpoint(network: _Network, episode: int) -> None:
    os.makedirs(ITERATION_CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(ITERATION_CHECKPOINT_DIR, f"bobail_zero_iter_{episode:04d}.pth")
    network.save(checkpoint_path)


def clear_training_data() -> None:
    for path in (RAW_SELF_PLAY_DATA_PATH, MERGED_TRAINING_DATA_PATH):
        if os.path.exists(path):
            os.remove(path)
            print(f"[Dataset] Deleted {path}")


def json_metric_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def normalize_metric_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: json_metric_value(value) for key, value in record.items()}


def load_pending_self_play_metrics() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not os.path.exists(PENDING_SELF_PLAY_METRICS_PATH):
        return [], []
    with open(PENDING_SELF_PLAY_METRICS_PATH, "r", encoding="utf-8") as fh:
        data: dict[str, list[dict[str, Any]]] = json.load(fh)
    return data["segments"], data["games"]


def save_pending_self_play_metrics(segments: list[dict[str, Any]], games: list[dict[str, Any]]) -> None:
    with open(PENDING_SELF_PLAY_METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"segments": segments, "games": games}, fh)


def append_pending_self_play_metrics(generation_metrics: dict[str, Any], game_metrics: list[dict[str, Any]]) -> None:
    segments, games = load_pending_self_play_metrics()
    segments.append(normalize_metric_record(generation_metrics))
    games.extend(normalize_metric_record(game_metric) for game_metric in game_metrics)
    save_pending_self_play_metrics(segments, games)
    print(f"[Metrics] Pending self-play segments: {len(segments)} | games: {len(games)}")


def clear_pending_self_play_metrics() -> None:
    if os.path.exists(PENDING_SELF_PLAY_METRICS_PATH):
        os.remove(PENDING_SELF_PLAY_METRICS_PATH)
        print(f"[Metrics] Deleted {PENDING_SELF_PLAY_METRICS_PATH}")


def publish_pending_self_play_metrics(metrics_logger: MetricsLogger, episode: int) -> None:
    segments, games = load_pending_self_play_metrics()
    if not segments:
        return

    total_games = sum(int(segment["games"]) for segment in segments)
    weighted_samples = sum(float(segment["samples_per_game"]) * int(segment["games"]) for segment in segments)
    generation_metrics: dict[str, Any] = {
        "timestamp": time.time(),
        "phase": "self_play_generation",
        "episode": int(episode),
        "generation_segments": int(len(segments)),
        "games": int(total_games),
        "workers": int(max(int(segment["workers"]) for segment in segments)),
        "raw_samples": int(round(weighted_samples)),
        "avg_moves_per_game": float(weighted_samples / total_games),
        "samples_per_game": float(weighted_samples / total_games),
    }
    if games:
        generation_metrics.update(aggregate_numeric(games, prefix="game_"))
    metrics_logger.append_many([generation_metrics])
    print(f"[Metrics] Published pending self-play as episode {episode}: {len(segments)} segments, {total_games} games.")

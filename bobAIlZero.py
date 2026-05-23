import time

import numpy as np
import torch
import torch.nn.functional as F

from bobail_config import (
    GENERATE_SELF_PLAY_DATA,
    MERGED_TRAINING_DATA_PATH,
    SAVED_NETWORK_PATH,
    SELF_PLAY_GAMES_PER_ITERATION,
    SELF_PLAY_FIRST_MOVE_TAU,
    SELF_PLAY_N_SIMULATIONS,
    SELF_PLAY_RANDOM_START_RATE,
    SELF_PLAY_TAU,
    SELF_PLAY_TD_LAMBDA,
    TRAINING_BATCH_SIZE,
    TRAINING_EARLY_STOPPING_PATIENCE,
    TRAINING_EPOCHS,
    TRAINING_LEARNING_RATE,
    TRAINING_LR_SCHEDULER_FACTOR,
    TRAINING_LR_SCHEDULER_PATIENCE,
    TRAINING_MIN_LEARNING_RATE,
    UPDATE_NETWORK,
)
from bobail_metrics import MetricsLogger
from bobail_model_zero import _Network
from bobail_selfplay import generate_dataset_parallel
from bobail_training_data import (
    append_pending_self_play_metrics,
    archive_training_data_to_buffer,
    clear_pending_self_play_metrics,
    clear_training_data,
    complete_training_data_from_buffer,
    evaluate_accuracy,
    load_network,
    next_episode,
    preprocess_with_stats,
    publish_pending_self_play_metrics,
    save_iteration_checkpoint,
)
from dataMerger import merge_npz_files

def smoke_inference_position() -> np.ndarray:
    return np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )

def print_smoke_inference(network: _Network) -> None:
    network.eval()
    with torch.inference_mode():
        x = torch.from_numpy(smoke_inference_position()).float().to("cpu").unsqueeze(0)
        policy_logits, value = network(x)
        policy_probs = F.softmax(policy_logits, dim=1).detach().cpu().numpy()
        value_np = value.detach().cpu().numpy()

    print("Test inference:")
    print(
        "Policy choice: "
        f"{np.argmax(policy_probs)}, sum={policy_probs.sum()}, "
        f"max={policy_probs.max()}, min={policy_probs.min()}, var={policy_probs.var()}"
    )
    print("Value:", value_np)
    print()

def run_training_iteration(metrics_logger: MetricsLogger, episode: int) -> bool:
    print("\nItération n°", episode)
    network = load_network()
    print_smoke_inference(network)

    network.tau = SELF_PLAY_TAU  # type: ignore[attr-defined]
    network.learning_rate = TRAINING_LEARNING_RATE  # type: ignore[attr-defined]
    network.logs = []
    network.logs_test = []

    if GENERATE_SELF_PLAY_DATA:
        generation_metrics, game_metrics = generate_dataset_parallel(
            evolving_network=network,
            ppe=SELF_PLAY_GAMES_PER_ITERATION,
            tau=network.tau,  # type: ignore[attr-defined]
            first_move_tau=SELF_PLAY_FIRST_MOVE_TAU,
            n_simulations=SELF_PLAY_N_SIMULATIONS,
            td_lambda=SELF_PLAY_TD_LAMBDA,
            random_start_rate=SELF_PLAY_RANDOM_START_RATE,
            episode=episode,
        )
        append_pending_self_play_metrics(generation_metrics, game_metrics)
        merge_npz_files(output_file=MERGED_TRAINING_DATA_PATH)

    if not UPDATE_NETWORK:
        return False

    complete_training_data_from_buffer(MERGED_TRAINING_DATA_PATH)
    preprocessed = preprocess_with_stats(MERGED_TRAINING_DATA_PATH)
    x_train = preprocessed.x_train
    y_train = preprocessed.y_train
    x_val = preprocessed.x_val
    y_val = preprocessed.y_val
    stats = preprocessed.stats
    print(
        f"[TRAINING] Training with {stats.train_samples} data points after preprocess "
        f"(started with {stats.raw_samples} original samples)."
    )
    metrics_logger.append(
        "dataset_stats",
        {
            "episode": episode,
            "raw_samples": float(stats.raw_samples),
            "augmented_samples": float(stats.augmented_samples),
            "dedup_samples": float(stats.dedup_samples),
            "train_samples": float(stats.train_samples),
            "val_samples": float(stats.val_samples),
            "y_pi_entropy_mean": float(stats.y_pi_entropy_mean),
            "y_v_mean": float(stats.y_v_mean),
        },
    )
    network.fit3(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=TRAINING_EPOCHS,
        learning_rate=network.learning_rate,  # type: ignore[attr-defined]
        batch_size=TRAINING_BATCH_SIZE,
        metrics_logger=metrics_logger,
        episode=episode,
        early_stopping_patience=TRAINING_EARLY_STOPPING_PATIENCE,
        lr_scheduler_factor=TRAINING_LR_SCHEDULER_FACTOR,
        lr_scheduler_patience=TRAINING_LR_SCHEDULER_PATIENCE,
        min_learning_rate=TRAINING_MIN_LEARNING_RATE,
    )
    network.save(SAVED_NETWORK_PATH)
    save_iteration_checkpoint(network, episode)

    print(
        f"y_v stats - train: mean={y_train[1].mean():.4f}, std={y_train[1].std():.4f}, "
        f"min={y_train[1].min():.4f}, max={y_train[1].max():.4f}"
    )
    print(
        f"y_pi stats - train: mean={y_train[0].mean():.4f}, std={y_train[0].std():.4f}, "
        f"min={y_train[0].min():.4f}, max={y_train[0].max():.4f}"
    )

    start_time = time.time()
    train_top1, train_top5 = evaluate_accuracy(network, x_train, y_train[0], batch_size=TRAINING_BATCH_SIZE)
    train_seconds = time.time() - start_time
    print(f"[EVALUATION] Accuracy on training set - Top1: {train_top1:.4f} | Top5: {train_top5:.4f} ({train_seconds:.2f}s)")

    start_time = time.time()
    val_top1, val_top5 = evaluate_accuracy(network, x_val, y_val[0], batch_size=TRAINING_BATCH_SIZE)
    val_seconds = time.time() - start_time
    print(f"[EVALUATION] Accuracy on validation set - Top1: {val_top1:.4f} | Top5: {val_top5:.4f} ({val_seconds:.2f}s)")

    metrics_logger.append(
        "accuracy",
        {
            "episode": episode,
            "train_top1": float(train_top1),
            "train_top5": float(train_top5),
            "val_top1": float(val_top1),
            "val_top5": float(val_top5),
        },
    )
    publish_pending_self_play_metrics(metrics_logger, episode)
    metrics_logger.plot_training("bobail_zero_training.png")
    metrics_logger.plot_selfplay("bobail_zero_selfplay.png")
    clear_pending_self_play_metrics()
    archive_training_data_to_buffer(MERGED_TRAINING_DATA_PATH)
    clear_training_data()
    return True

if __name__ == "__main__":
    logger = MetricsLogger("bobail_zero")
    episode = next_episode(logger)
    for _ in range(1):
        network_was_trained = run_training_iteration(logger, episode)
        if network_was_trained:
            episode += 1

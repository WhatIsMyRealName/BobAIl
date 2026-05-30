import copy
import multiprocessing
import os
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

from bobail_config import (
    BOARD_SIZE,
    FloatLike,
    IntLike,
    MCTS_C_PUCT,
    MCTS_PUCT_PARENT_EXPONENT,
    POLICY_SIZE,
    RAW_SELF_PLAY_DATA_PATH,
    RNG,
    SAVED_NETWORK_PATH,
    SELF_PLAY_MAX_PLIES,
    SELF_PLAY_FIRST_MOVE_TAU,
    SELF_PLAY_N_SIMULATIONS,
    SELF_PLAY_RANDOM_START_RATE,
    SELF_PLAY_TAU,
    SELF_PLAY_TD_LAMBDA,
)
from bobail_encoding import transform_input_data
from bobail_metrics import MetricsLogger, aggregate_numeric
from bobail_mcts import MCTS
from bobail_model_zero import _Network, configure_torch_inference_threads
from bobailgame import BobailGame
from utils import progress_bar

INTERESTING_SELF_PLAY_GAME_METRICS = {
    "mcts_max_depth_mean",
}

def compact_self_play_game_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key in INTERESTING_SELF_PLAY_GAME_METRICS}

def AIvsAIgame(
    evolving_network: _Network,
    tau: FloatLike,
    n_simulations: IntLike = SELF_PLAY_N_SIMULATIONS,
    first_move_tau: FloatLike = SELF_PLAY_FIRST_MOVE_TAU,
    l: float = SELF_PLAY_TD_LAMBDA,
    random_start_rate: float = SELF_PLAY_RANDOM_START_RATE,
    return_metrics: bool = False,
) -> tuple[list[list[Any]], int] | tuple[list[list[Any]], int, dict[str, Any]]:
    """Joue une partie de self-play et renvoie les exemples réseau générés."""
    if not 0.0 <= random_start_rate <= 1.0:
        raise ValueError("random_start_rate must be between 0 and 1")
    if first_move_tau < 0:
        raise ValueError("first_move_tau must be positive or zero")

    random_start = bool(RNG.random() < random_start_rate)
    game = BobailGame.generate_random_game(BOARD_SIZE, BOARD_SIZE) if random_start else BobailGame(BOARD_SIZE)
    balanced_first_move = not random_start
    tree = MCTS(
        evolving_network,
        n_simulations,
        tau,
        max_depth=SELF_PLAY_MAX_PLIES,
        c_puct=MCTS_C_PUCT,
        puct_parent_exponent=MCTS_PUCT_PARENT_EXPONENT,
        device="cpu",
    )
    game_data_collector_1: list[list[Any]] = []
    game_data_collector_2: list[list[Any]] = []
    move_metrics: list[dict[str, Any]] = []
    total_moves = 0
    winner = 0

    for _ in range(SELF_PLAY_MAX_PLIES):
        outcome = game.check_victory()
        if isinstance(outcome, int):
            winner = outcome
            break

        legal_moves = outcome
        search_game = game
        search_legal_moves = legal_moves
        is_balanced_first_move = balanced_first_move and total_moves == 0
        if is_balanced_first_move:
            search_game, search_legal_moves = game.balanced_first_move_legal_moves()

        tree.tau = float(first_move_tau if total_moves == 0 else tau)
        legal_moves, probs, target, targets = tree.run(search_game, search_legal_moves)
        move_metrics.append({
            "max_depth": tree.last_run_metrics["max_depth"],
        })

        move = legal_moves[RNG.choice(len(probs), p=probs)]
        total_moves += 1
        sample = [transform_input_data(search_game), targets, [target * l]]
        if game.current_player == 1:
            game_data_collector_1.append(sample)
        else:
            game_data_collector_2.append(sample)
        game.apply_move(move)
        game.current_player = 3 - game.current_player

    z = 1.0 if winner == 1 else -1.0 if winner == 2 else 0.0
    if game_data_collector_1:
        game_data_collector_1[-1][2][0] = z
        for index in range(len(game_data_collector_1) - 2, -1, -1):
            game_data_collector_1[index][2][0] += game_data_collector_1[index + 1][2][0] * (1.0 - l)
    if game_data_collector_2:
        game_data_collector_2[-1][2][0] = -z
        for index in range(len(game_data_collector_2) - 2, -1, -1):
            game_data_collector_2[index][2][0] += game_data_collector_2[index + 1][2][0] * (1.0 - l)

    data = game_data_collector_1 + game_data_collector_2
    if not return_metrics:
        return data, total_moves

    game_metrics: dict[str, Any] = {}
    if move_metrics:
        game_metrics.update(aggregate_numeric(move_metrics, prefix="mcts_"))
    return data, total_moves, compact_self_play_game_metrics(game_metrics)


def run_games_in_worker(
    n_games: IntLike,
    worker_id: int,
    progress_list: Iterable,
    evolving_network: _Network,
    tau: FloatLike,
    n_simulations: IntLike,
    first_move_tau: FloatLike,
    td_lambda: float,
    random_start_rate: float,
) -> tuple[list[list[Any]], int, list[dict[str, Any]]]:
    configure_torch_inference_threads()
    worker_data: list[list[Any]] = []
    worker_metrics: list[dict[str, Any]] = []
    total_moves = 0

    for game_index in range(1, int(n_games) + 1):
        game_data, moves, game_metrics = AIvsAIgame(
            evolving_network,
            tau,
            first_move_tau=first_move_tau,
            n_simulations=n_simulations,
            l=td_lambda,
            random_start_rate=random_start_rate,
            return_metrics=True,
        )
        total_moves += moves
        worker_data.extend(game_data)
        worker_metrics.append(game_metrics)
        progress_list[worker_id] = game_index  # type: ignore[index]
    return worker_data, total_moves, worker_metrics

def generate_dataset_parallel(
    evolving_network: _Network,
    ppe: int = 100,
    tau: FloatLike = SELF_PLAY_TAU,
    n_simulations: IntLike = SELF_PLAY_N_SIMULATIONS,
    first_move_tau: FloatLike = SELF_PLAY_FIRST_MOVE_TAU,
    td_lambda: float = SELF_PLAY_TD_LAMBDA,
    random_start_rate: float = SELF_PLAY_RANDOM_START_RATE,
    num_workers: int | None = None,
    metrics_logger: MetricsLogger | None = None,
    episode: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if ppe <= 0:
        raise ValueError("ppe must be strictly positive")
    if num_workers is None:
        num_workers = max(1, min(ppe, (os.cpu_count() or 1) - 2))
    if num_workers <= 0:
        raise ValueError("num_workers must be strictly positive")

    all_data: list[list[Any]] = []
    all_game_metrics: list[dict[str, Any]] = []
    total_moves = 0
    manager = multiprocessing.Manager()
    progress_list = manager.list([0] * num_workers)
    stop_monitor = threading.Event()

    def monitor_progress() -> None:
        while not stop_monitor.is_set():
            current_done = sum(progress_list)
            progress_bar(current_done, ppe)
            if current_done >= ppe:
                break
            time.sleep(5)

    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        base = ppe // num_workers
        remainder = ppe % num_workers
        for worker_id in range(num_workers):
            n_games_worker = base + (1 if worker_id < remainder else 0)
            if n_games_worker <= 0:
                continue
            network_copy = copy.deepcopy(evolving_network)
            network_copy.export()
            futures.append(
                executor.submit(
                    run_games_in_worker,
                    n_games_worker,
                    worker_id,
                    progress_list,
                    network_copy,
                    tau,
                    n_simulations,
                    first_move_tau,
                    td_lambda,
                    random_start_rate,
                )
            )

        for future in as_completed(futures):
            worker_data, moves, worker_metrics = future.result()
            total_moves += moves
            all_data.extend(worker_data)
            all_game_metrics.extend(worker_metrics)

    stop_monitor.set()
    monitor_thread.join(timeout=1.0)
    progress_bar(ppe, ppe)

    generation_seconds = time.time() - start_time
    print(f"[GENERATION] Parties: {ppe} | Workers: {num_workers} | Temps: {generation_seconds:.2f}s")
    print(f"[GAMES] - Coups moyens estimés / partie: {total_moves / ppe:.2f}")

    generation_metrics: dict[str, Any] = {
        "episode": int(episode),
        "games": int(ppe),
        "workers": int(num_workers),
        "move_tau": float(tau),
        "first_move_tau": float(first_move_tau),
        "raw_samples": int(len(all_data)),
        "avg_moves_per_game": float(total_moves / ppe),
        "samples_per_game": float(len(all_data) / ppe),
    }
    if all_game_metrics:
        generation_metrics.update(aggregate_numeric(all_game_metrics, prefix="game_"))
    if metrics_logger is not None:
        metrics_logger.append("self_play_generation", generation_metrics)

    x_data = np.array([item[0] for item in all_data], dtype=np.float32)
    y_pi = np.array([item[1] for item in all_data], dtype=np.float32)
    y_v = np.array([item[2] for item in all_data], dtype=np.float32)

    if os.path.exists(RAW_SELF_PLAY_DATA_PATH):
        with np.load(RAW_SELF_PLAY_DATA_PATH) as existing:
            prev_x = existing["x"].copy()
            prev_y_pi = existing["y_pi"].copy()
            prev_y_v = existing["y_v"].copy()
    else:
        prev_x = np.zeros((0, 3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        prev_y_pi = np.zeros((0, POLICY_SIZE), dtype=np.float32)
        prev_y_v = np.zeros((0, 1), dtype=np.float32)

    np.savez(
        RAW_SELF_PLAY_DATA_PATH,
        x=np.concatenate((prev_x, x_data)),
        y_pi=np.concatenate((prev_y_pi, y_pi)),
        y_v=np.concatenate((prev_y_v, y_v)),
    )
    print(f"[Dataset] Données ajoutées : x: {x_data.shape}, y_v: {y_v.shape}, y_pi: {y_pi.shape}")
    print(f"[Dataset] Taille totale : x: {prev_x.shape[0] + x_data.shape[0]}")
    evolving_network.save(SAVED_NETWORK_PATH)
    return generation_metrics, all_game_metrics

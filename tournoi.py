import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch

from bobail_metrics import MetricsLogger, aggregate_numeric
from bobail_model_zero import _Network, configure_torch_inference_threads
from bobail_config import MCTS_C_PUCT, RNG
from bobail_encoding import adapt_moves_pov, transform_input_data
from bobail_mcts import MCTS
from bobailgame import BobailGame, Move

MetricValue: TypeAlias = str | int | float
NumericMetric: TypeAlias = int | float
TournamentResult: TypeAlias = dict[str, MetricValue]
SummaryRow: TypeAlias = dict[str, MetricValue]

DEFAULT_CHECKPOINT_DIR = "./checkpoints"
TOURNAMENT_GAMES_PER_PAIR = 30
TOURNAMENT_MCTS_SIMULATIONS = 256
TOURNAMENT_MAX_PLIES = 80
TOURNAMENT_FIRST_MOVE_TAU = 1.0
TOURNAMENT_MOVE_TAU = 0.75
TOURNAMENT_NUM_WORKERS: int | None = None

@dataclass(frozen=True)
class Agent:
    name: str
    network: _Network

TOURNAMENT_WORKER_AGENTS: dict[str, Agent] = {}

@dataclass
class Standing:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    moves: list[int] = field(default_factory=list)

def load_agent(path: str) -> Agent:
    network = _Network.load(path)
    network.export()
    return Agent(name=Path(path).stem, network=network)

def checkpoint_paths(checkpoint_dir: str) -> list[str]:
    paths = sorted(Path(checkpoint_dir).glob("*.pth"))
    if not paths:
        raise ValueError(f"No .pth checkpoints found in {checkpoint_dir}")
    return [str(path) for path in paths]

def sample_index(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    probs = probabilities.astype(np.float64)
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("Move probabilities must have a strictly positive sum")
    probs /= total
    return int(rng.choice(len(probs), p=probs))

def temperature_policy(probs: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 1e-10:
        selected = np.zeros_like(probs, dtype=np.float64)
        selected[int(np.argmax(probs))] = 1.0
        return selected
    adjusted = probs.astype(np.float64) ** (1.0 / tau)
    total = float(adjusted.sum())
    if total <= 0.0:
        raise ValueError("Temperature-adjusted probabilities must have a strictly positive sum")
    adjusted /= total
    return adjusted

def policy_move(agent: Agent, game: BobailGame, legal_moves: list[Move], tau: float, rng: np.random.Generator) -> Move:
    x = torch.from_numpy(transform_input_data(game)).float().unsqueeze(0)
    with torch.inference_mode():
        policy_logits, _ = agent.network(x)
    policy = torch.softmax(policy_logits, dim=1).cpu().numpy().reshape(8, game.BOARD_SIZE, game.BOARD_SIZE, 8)
    scores = np.zeros(len(legal_moves), dtype=np.float32)
    for i, i0, i1, i2, i3 in adapt_moves_pov(game, legal_moves):
        scores[i] = policy[i0, i1, i2, i3]
    probs = temperature_policy(scores, tau)
    return legal_moves[sample_index(probs, rng)]

def mcts_move(
    tree: MCTS,
    game: BobailGame,
    legal_moves: list[Move],
    rng: np.random.Generator,
) -> tuple[Move, dict[str, NumericMetric]]:
    ordered_moves, probs, _, _ = tree.run(game, legal_moves)
    return ordered_moves[sample_index(probs, rng)], dict(tree.last_run_metrics)

def play_game(
    green: Agent,
    red: Agent,
    simulations: int,
    max_plies: int,
    use_mcts: bool,
    move_tau: float,
    use_dirichlet_noise: bool,
    rng: np.random.Generator,
    first_move_tau: float = TOURNAMENT_FIRST_MOVE_TAU,
) -> TournamentResult:
    if first_move_tau < 0.0:
        raise ValueError("first_move_tau must be positive or zero")
    game = BobailGame()
    start_time = time.time()
    mcts_metrics: list[dict[str, NumericMetric]] = []
    move_count = 0
    winner = 0
    initial_player = int(game.current_player)
    initial_bobail_row = int(game.bobail_position[0])
    initial_bobail_col = int(game.bobail_position[1])
    green_tree: MCTS | None = None
    red_tree: MCTS | None = None
    if use_mcts:
        green_tree = MCTS(
            green.network,
            n_simulations=simulations,
            tau=move_tau,
            max_depth=max_plies,
            c_puct=MCTS_C_PUCT,
            device="cpu",
            use_dirichlet_noise=use_dirichlet_noise,
        )
        red_tree = MCTS(
            red.network,
            n_simulations=simulations,
            tau=move_tau,
            max_depth=max_plies,
            c_puct=MCTS_C_PUCT,
            device="cpu",
            use_dirichlet_noise=use_dirichlet_noise,
        )

    for ply in range(1, max_plies + 1):
        outcome = game.check_victory()
        if isinstance(outcome, int):
            winner = outcome
            break

        search_game = game
        search_legal_moves = outcome
        is_balanced_first_move = move_count == 0
        if is_balanced_first_move:
            search_game, search_legal_moves = game.balanced_first_move_legal_moves()

        agent = green if game.current_player == 1 else red
        current_tau = first_move_tau if move_count == 0 else move_tau
        if use_mcts:
            tree = green_tree if game.current_player == 1 else red_tree
            assert tree is not None
            tree.tau = current_tau
            move, move_mcts_metrics = mcts_move(
                tree,
                search_game,
                search_legal_moves,
                rng,
            )
            mcts_metrics.append(move_mcts_metrics)
        else:
            move = policy_move(agent, search_game, search_legal_moves, current_tau, rng)

        game.apply_move(move)
        game.current_player = 3 - game.current_player
        move_count += 1

    if winner == 0:
        outcome = game.check_victory()
        if isinstance(outcome, int):
            winner = outcome

    result: TournamentResult = {
        "green": green.name,
        "red": red.name,
        "winner": int(winner),
        "green_win": int(winner == 1),
        "red_win": int(winner == 2),
        "draw": int(winner == 0),
        "moves": int(move_count),
        "seconds": float(time.time() - start_time),
        "balanced_first_move": 1,
        "random_start": 0,
        "initial_player": int(initial_player),
        "initial_bobail_row": int(initial_bobail_row),
        "initial_bobail_col": int(initial_bobail_col),
        "use_mcts": int(use_mcts),
        "simulations": int(simulations),
        "move_tau": float(move_tau),
        "first_move_tau": float(first_move_tau),
        "dirichlet_noise": int(use_dirichlet_noise),
    }
    if mcts_metrics:
        result.update(aggregate_numeric(mcts_metrics, prefix="mcts_"))
    return result

def initialize_tournament_worker(agents: list[Agent]) -> None:
    configure_torch_inference_threads()
    for agent in agents:
        agent.network.export()
    global TOURNAMENT_WORKER_AGENTS
    TOURNAMENT_WORKER_AGENTS = {agent.name: agent for agent in agents}

def seed_tournament_rng(seed: int) -> None:
    RNG.seed(seed % (2**32 - 1))

def play_tournament_game_worker(
    game_index: int,
    green_name: str,
    red_name: str,
    simulations: int,
    max_plies: int,
    use_mcts: bool,
    move_tau: float,
    first_move_tau: float,
    use_dirichlet_noise: bool,
    seed: int,
) -> TournamentResult:
    seed_tournament_rng(seed)
    result = play_game(
        TOURNAMENT_WORKER_AGENTS[green_name],
        TOURNAMENT_WORKER_AGENTS[red_name],
        simulations,
        max_plies,
        use_mcts,
        move_tau,
        use_dirichlet_noise,
        np.random.default_rng(seed),
        first_move_tau=first_move_tau,
    )
    result["game_index"] = game_index
    return result

def update_elo(elo: dict[str, float], green: str, red: str, winner: int, k_factor: float = 20.0) -> None:
    green_score = 1.0 if winner == 1 else 0.0 if winner == 2 else 0.5
    red_score = 1.0 - green_score
    green_expected = 1.0 / (1.0 + 10.0 ** ((elo[red] - elo[green]) / 400.0))
    red_expected = 1.0 - green_expected
    elo[green] += k_factor * (green_score - green_expected)
    elo[red] += k_factor * (red_score - red_expected)

def run_tournament(
    agents: list[Agent],
    games_per_pair: int,
    simulations: int,
    max_plies: int,
    use_mcts: bool,
    run_name: str,
    move_tau: float,
    use_dirichlet_noise: bool,
    seed: int | None,
    first_move_tau: float = TOURNAMENT_FIRST_MOVE_TAU,
    num_workers: int | None = TOURNAMENT_NUM_WORKERS,
) -> dict[str, str | int | dict[str, SummaryRow]]:
    if len(agents) < 2:
        raise ValueError("Tournament needs at least two agents")
    agent_names = [agent.name for agent in agents]
    if len(set(agent_names)) != len(agent_names):
        raise ValueError("Checkpoint names must be unique")
    if games_per_pair <= 0:
        raise ValueError("games_per_pair must be strictly positive")
    if max_plies <= 0:
        raise ValueError("max_plies must be strictly positive")
    if move_tau < 0.0:
        raise ValueError("move_tau must be positive or zero")
    if first_move_tau < 0.0:
        raise ValueError("first_move_tau must be positive or zero")
    if use_mcts and simulations <= 0:
        raise ValueError("simulations must be strictly positive when MCTS is enabled")

    configure_torch_inference_threads()
    rng = np.random.default_rng(seed)
    logger = MetricsLogger(run_name)
    results: list[TournamentResult] = []
    standings = {agent.name: Standing() for agent in agents}
    elo = {agent.name: 1000.0 for agent in agents}

    pairings = list(itertools.product(agents, repeat=2))
    scheduled_games = [
        (pair_index, green, red, game_in_pair)
        for pair_index, (green, red) in enumerate(pairings, start=1)
        for game_in_pair in range(1, games_per_pair + 1)
    ]
    rng.shuffle(scheduled_games)
    schedule = [
        (
            game_index,
            pair_index,
            green,
            red,
            game_in_pair,
            int(rng.integers(0, np.iinfo(np.int64).max)),
        )
        for game_index, (pair_index, green, red, game_in_pair) in enumerate(scheduled_games, start=1)
    ]
    total_games = len(schedule)
    worker_count = num_workers if num_workers is not None else max(1, min(total_games, (os.cpu_count() or 1) - 1))
    if worker_count <= 0:
        raise ValueError("num_workers must be strictly positive")
    worker_count = min(worker_count, total_games)
    progress_step = max(1, total_games // 10)
    tournament_start_time = time.time()
    print(
        f"[TOURNOI] run={run_name} | agents={len(agents)} | parties={total_games} | "
        f"games_per_pair={games_per_pair} | simulations={simulations} | "
        f"max_plies={max_plies} | first_tau={first_move_tau} | tau={move_tau} | "
        f"mcts={int(use_mcts)} | dirichlet_noise={int(use_dirichlet_noise)} | "
        f"workers={worker_count}",
        flush=True,
    )

    def print_progress(completed_games: int) -> None:
        if completed_games == total_games or completed_games % progress_step == 0:
            elapsed = time.time() - tournament_start_time
            games_per_second = completed_games / elapsed if elapsed > 0.0 else 0.0
            remaining = (total_games - completed_games) / games_per_second if games_per_second > 0.0 else 0.0
            print(
                f"[TOURNOI] {completed_games}/{total_games} parties | "
                f"elapsed={elapsed:.1f}s | remaining~{remaining:.1f}s",
                flush=True,
            )

    if worker_count == 1:
        for game_index, _, green, red, _, game_seed in schedule:
            seed_tournament_rng(game_seed)
            result = play_game(
                green,
                red,
                simulations,
                max_plies,
                use_mcts,
                move_tau,
                use_dirichlet_noise,
                np.random.default_rng(game_seed),
                first_move_tau=first_move_tau,
            )
            result["game_index"] = game_index
            results.append(result)
            print_progress(len(results))
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=initialize_tournament_worker,
            initargs=(agents,),
        ) as executor:
            futures = [
                executor.submit(
                    play_tournament_game_worker,
                    game_index,
                    green.name,
                    red.name,
                    simulations,
                    max_plies,
                    use_mcts,
                    move_tau,
                    first_move_tau,
                    use_dirichlet_noise,
                    game_seed,
                )
                for game_index, _, green, red, _, game_seed in schedule
            ]
            for future in as_completed(futures):
                results.append(future.result())
                print_progress(len(results))

    results.sort(key=lambda result: int(result["game_index"]))
    for result in results:
        logger.append("tournament_game", result)
        winner = int(result["winner"])
        green_name = str(result["green"])
        red_name = str(result["red"])
        if green_name != red_name:
            green_stats = standings[green_name]
            red_stats = standings[red_name]
            moves = int(result["moves"])
            green_stats.games += 1
            red_stats.games += 1
            green_stats.moves.append(moves)
            red_stats.moves.append(moves)
            if winner == 1:
                green_stats.wins += 1
                red_stats.losses += 1
            elif winner == 2:
                red_stats.wins += 1
                green_stats.losses += 1
            else:
                green_stats.draws += 1
                red_stats.draws += 1
            update_elo(elo, green_name, red_name, winner)

    summary: dict[str, str | int | dict[str, SummaryRow]] = {"run_name": run_name, "games": len(results), "standings": {}}
    summary_rows: list[SummaryRow] = []
    for name, stats in standings.items():
        row: SummaryRow = {
            "agent": name,
            "games": stats.games,
            "wins": stats.wins,
            "losses": stats.losses,
            "draws": stats.draws,
            "score": stats.wins + 0.5 * stats.draws,
            "win_rate": stats.wins / stats.games,
            "avg_moves": float(np.mean(stats.moves)),
            "elo": float(elo[name]),
        }
        summary_rows.append(row)
        standings_summary = summary["standings"]
        assert isinstance(standings_summary, dict)
        standings_summary[name] = row

    matrix_rows, win_counts, win_rates, score_rates, matrix_table = build_green_win_matrix(agents, results)
    agent_names = [agent.name for agent in agents]
    logger.save_table("tournoi_summary.csv", summary_rows)
    logger.save_table("tournoi_pair_details.csv", matrix_table)
    save_matrix_csv(logger, "tournoi_matrix.csv", "green\\red", matrix_rows, agent_names, win_counts)
    save_matrix_csv(logger, "tournoi_win_rate_matrix.csv", "green\\red", matrix_rows, agent_names, win_rates)
    save_matrix_csv(logger, "tournoi_score_rate_matrix.csv", "green\\red", matrix_rows, agent_names, score_rates)
    logger.plot_tournament_matrix(matrix_rows, agent_names, score_rates)
    return summary

def build_green_win_matrix(
    agents: list[Agent],
    results: list[TournamentResult],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[SummaryRow]]:
    agent_names = [agent.name for agent in agents]
    win_counts = np.full((len(agent_names), len(agent_names)), np.nan, dtype=np.float64)
    win_rates = np.full((len(agent_names), len(agent_names)), np.nan, dtype=np.float64)
    score_rates = np.full((len(agent_names), len(agent_names)), np.nan, dtype=np.float64)
    table: list[SummaryRow] = []
    for row_idx, green in enumerate(agent_names):
        for col_idx, red in enumerate(agent_names):
            games = [result for result in results if result["green"] == green and result["red"] == red]
            if not games:
                table.append({"green": green, "red": red, "games": 0, "green_wins": "", "green_win_rate": ""})
                continue
            green_wins = int(sum(int(game["green_win"]) for game in games))
            draws = int(sum(int(game["draw"]) for game in games))
            green_win_rate = green_wins / len(games)
            green_score_rate = (green_wins + 0.5 * draws) / len(games)
            win_counts[row_idx, col_idx] = green_wins
            win_rates[row_idx, col_idx] = green_win_rate
            score_rates[row_idx, col_idx] = green_score_rate
            table.append({
                "green": green,
                "red": red,
                "games": len(games),
                "green_wins": green_wins,
                "green_win_rate": green_win_rate,
                "green_score_rate": green_score_rate,
                "red_wins": int(sum(int(game["red_win"]) for game in games)),
                "red_win_rate": float(np.mean([int(game["red_win"]) for game in games])),
                "draw_rate": float(np.mean([game["draw"] for game in games])),
                "avg_moves": float(np.mean([game["moves"] for game in games])),
            })
    return agent_names, win_counts, win_rates, score_rates, table

def save_matrix_csv(
    logger: MetricsLogger,
    filename: str,
    row_header: str,
    rows: list[str],
    columns: list[str],
    matrix: np.ndarray,
) -> Path:
    path = logger.directory / filename
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([row_header, *columns])
        for row_index, row_name in enumerate(rows):
            values: list[str | int | float] = []
            for column_index in range(len(columns)):
                value = matrix[row_index, column_index]
                if np.isnan(value):
                    values.append("")
                elif float(value).is_integer():
                    values.append(int(value))
                else:
                    values.append(float(value))
            writer.writerow([row_name, *values])
    return path

def run_default_tournament() -> dict[str, str | int | dict[str, SummaryRow]]:
    configure_torch_inference_threads()
    agents = [load_agent(path) for path in checkpoint_paths(DEFAULT_CHECKPOINT_DIR)]
    summary = run_tournament(
        agents=agents,
        games_per_pair=TOURNAMENT_GAMES_PER_PAIR,
        simulations=TOURNAMENT_MCTS_SIMULATIONS,
        max_plies=TOURNAMENT_MAX_PLIES,
        use_mcts=True,
        run_name=f"tournoi_{int(time.time())}",
        move_tau=TOURNAMENT_MOVE_TAU,
        use_dirichlet_noise=False,
        seed=None,
        first_move_tau=TOURNAMENT_FIRST_MOVE_TAU,
        num_workers=TOURNAMENT_NUM_WORKERS,
    )
    standings = summary["standings"]
    assert isinstance(standings, dict)
    for name, stats in sorted(standings.items(), key=lambda item: item[1]["score"], reverse=True):
        print(
            f"{name}: score={stats['score']:.1f}, "
            f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}, "
            f"avg_moves={stats['avg_moves']:.1f}, "
            f"elo={stats['elo']:.1f}"
        )
    return summary


if __name__ == "__main__":
    run_default_tournament()

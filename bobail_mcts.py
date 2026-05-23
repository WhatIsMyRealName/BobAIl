from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
import torch

from MCTS_visu import MCTSVisualizer3D
from bobail_config import (
    FloatLike,
    IntLike,
    MCTS_C_PUCT,
    MCTS_DIRICHLET_ALPHA_0,
    MCTS_DIRICHLET_NOISE_WEIGHT,
    MCTS_PUCT_PARENT_EXPONENT,
    NDarrayFloat32,
    NDarrayFloat32_1D,
    RNG,
    SELF_PLAY_MAX_PLIES,
    SELF_PLAY_TAU,
)
from bobail_encoding import adapt_moves_pov, transform_input_data
from bobail_model_zero import _Network
from bobailgame import BobailGame, Move, Position

NodeKey: TypeAlias = tuple[tuple[Position, ...], tuple[Position, ...], Position, int]
MCTSMetricValue: TypeAlias = int | float
MCTSRunMetrics: TypeAlias = dict[str, MCTSMetricValue]

class MCTS:
    """Monte Carlo Graph Search guidé par le réseau et filtré par tactique immédiate."""

    def __init__(
        self,
        network: _Network,
        n_simulations: IntLike,
        tau: FloatLike = SELF_PLAY_TAU,
        max_depth: int = SELF_PLAY_MAX_PLIES,
        c_puct: FloatLike = MCTS_C_PUCT,
        puct_parent_exponent: FloatLike = MCTS_PUCT_PARENT_EXPONENT,
        device: str = "cpu",
        use_dirichlet_noise: bool = True,
    ) -> None:
        if n_simulations <= 0:
            raise ValueError("n_simulations must be strictly positive")
        if tau < 0:
            raise ValueError("tau must be positive or zero")
        if c_puct <= 0:
            raise ValueError("c_puct must be strictly positive")
        if puct_parent_exponent <= 0:
            raise ValueError("puct_parent_exponent must be strictly positive")

        self.nodes: dict[NodeKey, dict[str, Any]] = {}
        self.network = network
        self.n_simulations = int(n_simulations)
        self.tau = float(tau)
        self.max_depth = max_depth
        self.c_puct = float(c_puct)
        self.puct_parent_exponent = float(puct_parent_exponent)
        self.device = device
        self.use_dirichlet_noise = use_dirichlet_noise
        self.alpha_0 = MCTS_DIRICHLET_ALPHA_0
        self.noise_weight = MCTS_DIRICHLET_NOISE_WEIGHT
        self.tactical_cache: dict[NodeKey, tuple[tuple[Move, ...], tuple[Move, ...]]] = {}
        self.current_run_metrics: MCTSRunMetrics | None = None
        self.last_run_metrics: MCTSRunMetrics = {}

    def get_node_key(self, game: BobailGame) -> NodeKey:
        pawns_1 = tuple(sorted(game.tokens[1]))
        pawns_2 = tuple(sorted(game.tokens[2]))
        return pawns_1, pawns_2, game.bobail_position, game.current_player

    def raw_evaluate_state(self, game: BobailGame) -> tuple[np.ndarray, float]:
        input_data = transform_input_data(game)
        x = torch.from_numpy(input_data).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            policy_logits, value = self.network(x)

        policy = policy_logits.detach().cpu().numpy().reshape(8, game.BOARD_SIZE, game.BOARD_SIZE, 8)
        return policy, float(value.detach().cpu().numpy().reshape(-1)[0])

    def legal_priors(self, logits: np.ndarray, legal_moves: Sequence[Move], game: BobailGame) -> NDarrayFloat32_1D:
        legal_logits = np.empty(len(legal_moves), dtype=np.float64)
        for output_index, (_, dir_bobail, old_row, old_col, dir_token) in enumerate(adapt_moves_pov(game, legal_moves)):
            legal_logits[output_index] = logits[dir_bobail, old_row, old_col, dir_token]

        legal_logits -= np.max(legal_logits)
        np.exp(legal_logits, out=legal_logits)
        legal_logits /= legal_logits.sum()
        return legal_logits.astype(np.float32)

    def inflate(self, logits: NDarrayFloat32_1D, legal_moves: Sequence[Move], game: BobailGame) -> NDarrayFloat32:
        updated = np.zeros((8, game.BOARD_SIZE, game.BOARD_SIZE, 8), dtype=np.float32)
        for move_index, dir_bobail, old_row, old_col, dir_token in adapt_moves_pov(game, legal_moves):
            updated[dir_bobail, old_row, old_col, dir_token] = logits[move_index]
        return updated

    def add_dirichlet_noise(self, priors: NDarrayFloat32_1D) -> NDarrayFloat32_1D:
        alpha = np.full(len(priors), self.alpha_0 / len(priors), dtype=np.float32)
        noise = RNG.dirichlet(alpha).astype(np.float32)
        mixed = (1.0 - self.noise_weight) * priors + self.noise_weight * noise
        mixed /= mixed.sum()
        return mixed

    @staticmethod
    def winning_row(player: int, board_size: int) -> int:
        return board_size - 1 if player == 1 else 0

    def losing_row(self, player: int, board_size: int) -> int:
        return self.winning_row(3 - player, board_size)

    def move_reaches_winning_row(self, game: BobailGame, move: Move) -> bool:
        return move[0][1][0] == self.winning_row(game.current_player, game.BOARD_SIZE)

    def move_reaches_losing_row(self, game: BobailGame, move: Move) -> bool:
        return move[0][1][0] == self.losing_row(game.current_player, game.BOARD_SIZE)

    def is_winning_move(self, game: BobailGame, move: Move) -> bool:
        if self.move_reaches_winning_row(game, move):
            return True
        if self.move_reaches_losing_row(game, move):
            return False

        reverse = game.apply_move(move, sync_board=False)
        game.current_player = 3 - game.current_player
        opponent_has_move = game.has_legal_move(game.tokens, game.bobail_position, game.current_player)
        game.current_player = 3 - game.current_player
        game.apply_move(reverse, sync_board=False)
        return not opponent_has_move

    def is_losing_move(self, game: BobailGame, move: Move) -> bool:
        return self.move_reaches_losing_row(game, move)

    def classify_tactical_moves(self, game: BobailGame, legal_moves: Sequence[Move]) -> tuple[list[int], list[int]]:
        node_key = self.get_node_key(game)
        if node_key in self.tactical_cache:
            winning_moves, losing_moves = self.tactical_cache[node_key]
        else:
            winning = [move for move in legal_moves if self.is_winning_move(game, move)]
            if winning:
                winning_moves = tuple(winning)
                losing_moves = ()
            else:
                winning_moves = ()
                losing_moves = tuple(move for move in legal_moves if self.is_losing_move(game, move))
            self.tactical_cache[node_key] = (winning_moves, losing_moves)

        move_to_index = {move: i for i, move in enumerate(legal_moves)}
        winning_indices = [move_to_index[move] for move in winning_moves if move in move_to_index]
        losing_indices = [move_to_index[move] for move in losing_moves if move in move_to_index]
        return winning_indices, losing_indices

    def tactical_search_state(
        self,
        game: BobailGame,
        legal_moves: Sequence[Move],
    ) -> tuple[float | None, list[Move], list[int], list[int]]:
        legal_moves = list(legal_moves)
        winning_indices, losing_indices = self.classify_tactical_moves(game, legal_moves)

        if winning_indices:
            return 1.0, legal_moves, winning_indices, losing_indices
        if len(losing_indices) == len(legal_moves):
            return -1.0, legal_moves, winning_indices, losing_indices
        if losing_indices:
            losing_set = set(losing_indices)
            legal_moves = [move for i, move in enumerate(legal_moves) if i not in losing_set]
        return None, legal_moves, winning_indices, losing_indices

    def PUCT(
        self,
        total_N: int,
        N: int | NDarrayFloat32_1D,
        Q: int | NDarrayFloat32_1D,
        P: int | NDarrayFloat32_1D,
    ) -> float | NDarrayFloat32_1D:
        return self.exploration_bonus(total_N, N, P) + Q  # type: ignore[return-value]

    def exploration_bonus(
        self,
        total_N: int,
        N: int | NDarrayFloat32_1D,
        P: int | NDarrayFloat32_1D,
    ) -> float | NDarrayFloat32_1D:
        coefficient = self.c_puct * ((total_N + 1) ** self.puct_parent_exponent)
        return coefficient * P / (1.0 + N)  # type: ignore[return-value]

    def inversion_de_PUCT(
        self,
        c_star_final_puct_value: FloatLike,
        total_N: IntLike,
        Q: NDarrayFloat32_1D,
        P: NDarrayFloat32_1D,
        eps: float = 1e-50,
    ) -> np.ndarray:
        coefficient = self.c_puct * ((int(total_N) + 1) ** self.puct_parent_exponent)
        puct_margin = c_star_final_puct_value - Q
        desired = np.full(Q.shape, np.inf, dtype=np.float64)
        prunable = puct_margin > eps
        desired[prunable] = np.ceil((coefficient * P[prunable]) / puct_margin[prunable]) - 1.0
        desired[prunable] = np.maximum(desired[prunable], 0.0)
        return desired

    def target_policy_pruning(
        self,
        c_star: IntLike,
        c_star_final_puct_value: FloatLike,
        visits: NDarrayFloat32_1D,
        total_N: IntLike,
        Q: NDarrayFloat32_1D,
        P: NDarrayFloat32_1D,
    ) -> NDarrayFloat32_1D:
        desired = self.inversion_de_PUCT(c_star_final_puct_value, total_N, Q, P)
        desired[int(c_star)] = visits[int(c_star)]
        pruned = np.minimum(visits.astype(np.float64), desired)
        pruned = np.maximum(pruned, 0.0)
        return pruned.astype(np.float32)

    def create_node_from_evaluation(
        self,
        game: BobailGame,
        legal_moves: Sequence[Move],
        policy_logits: np.ndarray,
        value: FloatLike,
    ) -> float:
        legal_moves = list(legal_moves)
        priors = self.legal_priors(policy_logits, legal_moves, game)
        order = np.argsort(-priors)
        ordered_priors = priors[order]
        ordered_moves = [legal_moves[i] for i in order]
        self.nodes[self.get_node_key(game)] = {
            "moves": ordered_moves,
            "P": (ordered_priors / ordered_priors.sum()).astype(np.float32),
            "N": np.zeros(len(legal_moves), dtype=np.int32),
            "W": np.zeros(len(legal_moves), dtype=np.float32),
            "Q": np.zeros(len(legal_moves), dtype=np.float32),
            "move_set": frozenset(legal_moves),
        }
        return float(value)

    def create_node(self, game: BobailGame, legal_moves: Sequence[Move]) -> float:
        policy_logits, value = self.raw_evaluate_state(game)
        return self.create_node_from_evaluation(game, legal_moves, policy_logits, value)

    def ensure_node(self, game: BobailGame, legal_moves: Sequence[Move]) -> float | None:
        legal_moves = list(legal_moves)
        node_key = self.get_node_key(game)
        legal_move_set = frozenset(legal_moves)
        if node_key in self.nodes and self.nodes[node_key]["move_set"] == legal_move_set:
            return None
        return self.create_node(game, legal_moves)

    def finalize_run_metrics(self) -> MCTSRunMetrics:
        metrics = self.current_run_metrics
        if metrics is None:
            raise ValueError("No current run metrics to finalize")

        self.last_run_metrics = dict(metrics)
        self.current_run_metrics = None
        return self.last_run_metrics

    def step(
        self,
        playout: int,
        node_game: BobailGame,
        depth: int,
        root_key: NodeKey | str,
        show: MCTSVisualizer3D | None = None,
    ) -> FloatLike:
        if self.current_run_metrics is not None:
            self.current_run_metrics["max_depth"] = max(self.current_run_metrics["max_depth"], depth)

        outcome = node_game.check_victory()
        if isinstance(outcome, int):
            return 1.0 if outcome == node_game.current_player else -1.0

        legal_moves = list(outcome)
        if depth > 1:
            tactical_value, legal_moves, _, _ = self.tactical_search_state(node_game, legal_moves)
            if tactical_value is not None:
                return tactical_value

        if depth >= self.max_depth:
            _, value = self.raw_evaluate_state(node_game)
            return value

        node_key = self.get_node_key(node_game)
        is_search_root = depth == 1 and node_key == root_key
        if not is_search_root:
            node_value = self.ensure_node(node_game, legal_moves)
            if node_value is not None:
                return node_value
        elif node_key not in self.nodes or "P_search" not in self.nodes[node_key]:
            raise ValueError("Root node must be initialized by run before step")

        stats = self.nodes[node_key]
        visits = stats["N"]
        q_values = stats["Q"]
        priors = stats["P_search"] if is_search_root else stats["P"]
        moves = stats["moves"]
        total_visits = visits.sum()
        if visits.size == 0:
            raise ValueError("Empty moves")

        puct_values = self.PUCT(total_visits, visits, q_values, priors)
        if depth == 1:
            forced_playouts = np.sqrt(priors * playout).astype(np.int32) # k = 1 ? Is it enough ? Maybe TODO : set as parameter
            forced_mask = (visits >= 1) & (visits <= forced_playouts)
            puct_values[forced_mask] = np.inf  # type: ignore[index]

        best_index = int(np.argmax(puct_values))
        selected_move = moves[best_index]
        previous_hash = hash(node_game) if show is not None else None
        reverse = node_game.apply_move(selected_move, sync_board=show is not None)
        node_game.current_player = 3 - node_game.current_player
        if show is not None and previous_hash is not None:
            show.add_transition(previous_hash, hash(node_game))

        child_value = self.step(playout, node_game, depth + 1, root_key=root_key, show=show)
        value_for_parent = -child_value
        node_game.current_player = 3 - node_game.current_player
        node_game.apply_move(reverse, sync_board=show is not None)

        stats["N"][best_index] += 1
        stats["W"][best_index] += value_for_parent
        stats["Q"][best_index] = stats["W"][best_index] / stats["N"][best_index]
        return value_for_parent

    def run(
        self,
        root_game: BobailGame,
        root_legal_moves: Sequence[Move],
        show: MCTSVisualizer3D | None = None,
    ) -> tuple[list[Move], NDarrayFloat32_1D, FloatLike, np.ndarray]:
        root_key = self.get_node_key(root_game)
        root_legal_moves = list(root_legal_moves)
        self.current_run_metrics = {
            "max_depth": 0,
        }

        tactical_value, root_search_moves, winning_indices, losing_indices = self.tactical_search_state(
            root_game,
            root_legal_moves,
        )

        if tactical_value == 1.0:
            visits = np.zeros(len(root_legal_moves), dtype=np.float32)
            visits[winning_indices] = 1.0
            probs = visits / visits.sum()
            self.finalize_run_metrics()
            return root_legal_moves, probs, 1.0, self.inflate(probs, root_legal_moves, root_game).flatten()

        if tactical_value == -1.0:
            visits = np.ones(len(root_legal_moves), dtype=np.float32)
            probs = visits / visits.sum()
            self.finalize_run_metrics()
            return root_legal_moves, probs, -1.0, self.inflate(probs, root_legal_moves, root_game).flatten()

        root_legal_moves = root_search_moves
        self.ensure_node(root_game, root_legal_moves)
        root_stats = self.nodes[root_key]
        root_stats["P_search"] = self.add_dirichlet_noise(root_stats["P"]) if self.use_dirichlet_noise else root_stats["P"]

        for playout in range(self.n_simulations):
            self.step(playout, root_game, depth=1, root_key=root_key, show=show)
        if show is not None:
            show.show(min_width=0.2, max_width=4.0)

        moves = root_stats["moves"]
        search_priors = root_stats["P_search"]
        visits = root_stats["N"]
        q_values = root_stats["Q"]
        del root_stats["P_search"]

        total_visits = np.sum(visits)
        best_child = int(np.argmax(visits))
        best_final_puct = self.PUCT(total_visits, visits[best_child], q_values[best_child], search_priors[best_child])
        pruned_visits = self.target_policy_pruning(best_child, best_final_puct, visits, total_visits, q_values, search_priors)

        if self.tau <= 1e-10:
            probs = np.zeros_like(pruned_visits)
            probs[np.argmax(pruned_visits)] = 1.0
        else:
            probs = pruned_visits ** (1.0 / self.tau)
            probs /= probs.sum()

        visit_policy = pruned_visits / pruned_visits.sum()
        selected_q = q_values[np.argmax(pruned_visits)]
        self.finalize_run_metrics()
        return moves, probs, selected_q, self.inflate(visit_policy, moves, root_game).flatten()

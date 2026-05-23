import random
import unittest
from typing import Any

import numpy as np
import torch

import tournoi
from bobailgame import DIRECTIONS, BobailGame, Move, Position
from bobail_config import POLICY_SIZE
from bobail_mcts import MCTS

class DummyNetwork(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        policy = torch.zeros((batch_size, POLICY_SIZE), dtype=torch.float32, device=x.device)
        value = torch.zeros((batch_size, 1), dtype=torch.float32, device=x.device)
        return policy, value


class CountingNetwork(DummyNetwork):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        return super().forward(x)


def make_game(tokens: dict[int, list[Position]], bobail: Position, current_player: int) -> BobailGame:
    game = BobailGame(5)
    game.tokens = {1: sorted(tokens[1]), 2: sorted(tokens[2])}
    game.bobail_position = bobail
    game.current_player = current_player
    game._sync_board()
    return game


def reference_valid_moves_from_blockers(
    board_size: int,
    position: Position,
    blockers: set[Position],
    is_bobail: bool = False,
) -> list[Position]:
    valid_moves: list[Position] = []
    for dr, dc in DIRECTIONS:
        current_row, current_col = position
        last_valid: Position | None = None
        while True:
            next_row, next_col = current_row + dr, current_col + dc
            next_position = (next_row, next_col)
            if not (0 <= next_row < board_size and 0 <= next_col < board_size):
                break
            if next_position in blockers:
                break
            if is_bobail:
                valid_moves.append(next_position)
                break
            last_valid = next_position
            current_row, current_col = next_row, next_col
        if last_valid is not None:
            valid_moves.append(last_valid)
    return valid_moves


def reference_generate_moves(game: BobailGame) -> list[Move]:
    token_blockers = set(game.tokens[1]) | set(game.tokens[2])
    moves: list[Move] = []
    for bobail_move in reference_valid_moves_from_blockers(
        game.BOARD_SIZE,
        game.bobail_position,
        token_blockers,
        is_bobail=True,
    ):
        blockers = token_blockers | {bobail_move}
        for token in game.tokens[game.current_player]:
            for token_move in reference_valid_moves_from_blockers(game.BOARD_SIZE, token, blockers):
                moves.append((("bobail", bobail_move), ("token", token, token_move)))
    return sorted(moves)


class BobailFastMoveTests(unittest.TestCase):
    def test_generate_moves_keeps_exact_sorted_order(self) -> None:
        random.seed(13)
        for _ in range(300):
            tokens, bobail, current_player = BobailGame.generate_random_position(5, 5)
            game = make_game(tokens, bobail, current_player)

            self.assertEqual(
                game.generate_moves(game.tokens, game.bobail_position, game.current_player),
                reference_generate_moves(game),
            )

    def test_has_legal_move_matches_generate_moves(self) -> None:
        random.seed(7)
        for _ in range(200):
            tokens, bobail, current_player = BobailGame.generate_random_position(5, 5)
            game = make_game(tokens, bobail, current_player)
            generated = game.generate_moves(game.tokens, game.bobail_position, game.current_player)
            self.assertEqual(
                game.has_legal_move(game.tokens, game.bobail_position, game.current_player),
                len(generated) > 0,
            )

    def test_random_position_is_not_terminal_and_starts_on_middle_row(self) -> None:
        random.seed(11)
        for _ in range(100):
            tokens, bobail, current_player = BobailGame.generate_random_position(5, 5)
            game = make_game(tokens, bobail, current_player)

            self.assertEqual(bobail[0], 2)
            self.assertNotIsInstance(game.check_victory(), int)

    def test_balanced_first_move_returns_bobail_to_center(self) -> None:
        game = BobailGame(5)
        search_game, legal_moves = game.balanced_first_move_legal_moves()

        self.assertEqual(search_game.bobail_position, (2, 1))
        self.assertTrue(legal_moves)
        self.assertTrue(all(move[0][1] == game.center_position() for move in legal_moves))
        game.apply_move(legal_moves[0])
        self.assertEqual(game.bobail_position, game.center_position())

    def test_check_victory_matches_line_and_no_move_victories(self) -> None:
        line_top = make_game(
            {1: [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)], 2: [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]},
            (0, 2),
            1,
        )
        self.assertEqual(line_top.check_victory(), 2)

        line_bottom = make_game(
            {1: [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)], 2: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]},
            (4, 2),
            2,
        )
        self.assertEqual(line_bottom.check_victory(), 1)

        blocked = make_game(
            {
                1: [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3)],
                2: [(3, 1), (3, 2), (3, 3), (0, 0), (4, 4)],
            },
            (2, 2),
            1,
        )
        self.assertFalse(blocked.has_legal_move(blocked.tokens, blocked.bobail_position, blocked.current_player))
        self.assertEqual(blocked.check_victory(), 2)

    def test_apply_move_keeps_player_tokens_sorted_and_hash_uses_state(self) -> None:
        game = BobailGame(5)
        initial_hash = hash(game)
        move = game.generate_moves(game.tokens, game.bobail_position, game.current_player)[0]
        reverse = game.apply_move(move, sync_board=False)

        self.assertEqual(game.tokens[1], sorted(game.tokens[1]))
        self.assertNotEqual(hash(game), initial_hash)

        game.apply_move(reverse, sync_board=False)
        self.assertEqual(game.tokens[1], sorted(game.tokens[1]))
        self.assertEqual(hash(game), initial_hash)


class MCTSTacticalTests(unittest.TestCase):
    def make_mcts(self, n_simulations: int = 8) -> MCTS:
        return MCTS(DummyNetwork(), n_simulations=n_simulations, tau=1.0, use_dirichlet_noise=False)

    def test_direct_winning_moves_short_circuit_to_uniform_target(self) -> None:
        game = make_game(
            {
                1: [(0, 0), (0, 4), (1, 0), (1, 4), (2, 0)],
                2: [(2, 4), (3, 0), (3, 4), (4, 0), (4, 4)],
            },
            (3, 2),
            1,
        )
        legal_moves = game.generate_moves(game.tokens, game.bobail_position, game.current_player)
        mcts = self.make_mcts()
        winning_indices, losing_indices = mcts.classify_tactical_moves(game, legal_moves)
        self.assertGreater(len(winning_indices), 1)
        self.assertEqual(losing_indices, [])

        returned_moves, probs, value, target = mcts.run(game, legal_moves)

        self.assertEqual(returned_moves, legal_moves)
        self.assertEqual(value, 1.0)
        expected = np.zeros(len(legal_moves), dtype=np.float32)
        expected[winning_indices] = 1.0 / len(winning_indices)
        np.testing.assert_allclose(probs, expected)
        self.assertEqual(int((target > 0).sum()), len(winning_indices))

    def test_losing_moves_are_masked_when_not_all_moves_lose(self) -> None:
        game = make_game(
            {
                1: [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
                2: [(0, 0), (0, 4), (3, 0), (3, 4), (2, 2)],
            },
            (1, 2),
            1,
        )
        legal_moves = game.generate_moves(game.tokens, game.bobail_position, game.current_player)
        mcts = self.make_mcts(n_simulations=4)
        winning_indices, losing_indices = mcts.classify_tactical_moves(game, legal_moves)
        self.assertEqual(winning_indices, [])
        self.assertGreater(len(losing_indices), 0)
        self.assertLess(len(losing_indices), len(legal_moves))

        returned_moves, probs, value, target = mcts.run(game, legal_moves)
        losing_moves = {legal_moves[i] for i in losing_indices}

        self.assertTrue(all(mcts.move_reaches_losing_row(game, move) for move in losing_moves))
        self.assertTrue(set(returned_moves).isdisjoint(losing_moves))
        self.assertEqual(len(returned_moves), len(legal_moves) - len(losing_indices))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)
        self.assertEqual(int((target > 0).sum()), int((probs > 0).sum()))

    def test_all_losing_moves_short_circuit_to_uniform_loss(self) -> None:
        game = make_game(
            {
                1: [(1, 1), (1, 3), (2, 1), (2, 2), (2, 3)],
                2: [(0, 0), (0, 4), (3, 0), (3, 4), (4, 2)],
            },
            (1, 2),
            1,
        )
        legal_moves = game.generate_moves(game.tokens, game.bobail_position, game.current_player)
        mcts = self.make_mcts()
        winning_indices, losing_indices = mcts.classify_tactical_moves(game, legal_moves)
        self.assertEqual(winning_indices, [])
        self.assertEqual(len(losing_indices), len(legal_moves))
        self.assertTrue(all(mcts.move_reaches_losing_row(game, move) for move in legal_moves))

        returned_moves, probs, value, target = mcts.run(game, legal_moves)

        self.assertEqual(returned_moves, legal_moves)
        self.assertEqual(value, -1.0)
        np.testing.assert_allclose(probs, np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32))
        self.assertEqual(int((target > 0).sum()), len(legal_moves))

    def test_losing_move_does_not_mean_opponent_can_win_next(self) -> None:
        game = make_game(
            {
                1: [(0, 0), (2, 0), (2, 4), (0, 3), (0, 1)],
                2: [(0, 2), (0, 4), (3, 3), (4, 4), (1, 4)],
            },
            (1, 1),
            1,
        )
        legal_moves = game.generate_moves(game.tokens, game.bobail_position, game.current_player)
        mcts = self.make_mcts()
        winning_indices, losing_indices = mcts.classify_tactical_moves(game, legal_moves)

        self.assertEqual(winning_indices, [])
        self.assertEqual(losing_indices, [])

    def test_internal_winning_short_circuit_skips_network_and_node_creation(self) -> None:
        game = make_game(
            {
                1: [(0, 0), (0, 4), (1, 0), (1, 4), (2, 0)],
                2: [(2, 4), (3, 0), (3, 4), (4, 0), (4, 4)],
            },
            (3, 2),
            1,
        )
        network = CountingNetwork()
        mcts = MCTS(network, n_simulations=1, tau=1.0, use_dirichlet_noise=False)

        value = mcts.step(0, game, depth=2, root_key="root")

        self.assertEqual(value, 1.0)
        self.assertEqual(network.calls, 0)
        self.assertNotIn(mcts.get_node_key(game), mcts.nodes)

    def test_internal_losing_moves_are_masked_before_expansion(self) -> None:
        game = make_game(
            {
                1: [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
                2: [(0, 0), (0, 4), (3, 0), (3, 4), (2, 2)],
            },
            (1, 2),
            1,
        )
        network = CountingNetwork()
        mcts = MCTS(network, n_simulations=1, tau=1.0, use_dirichlet_noise=False)

        value = mcts.step(0, game, depth=2, root_key="root")
        stats = mcts.nodes[mcts.get_node_key(game)]

        self.assertEqual(value, 0.0)
        self.assertEqual(network.calls, 1)
        self.assertTrue(all(not mcts.is_losing_move(game, move) for move in stats["moves"]))

    def test_internal_all_losing_short_circuit_skips_network_and_node_creation(self) -> None:
        game = make_game(
            {
                1: [(1, 1), (1, 3), (2, 1), (2, 2), (2, 3)],
                2: [(0, 0), (0, 4), (3, 0), (3, 4), (4, 2)],
            },
            (1, 2),
            1,
        )
        network = CountingNetwork()
        mcts = MCTS(network, n_simulations=1, tau=1.0, use_dirichlet_noise=False)

        value = mcts.step(0, game, depth=2, root_key="root")

        self.assertEqual(value, -1.0)
        self.assertEqual(network.calls, 0)
        self.assertNotIn(mcts.get_node_key(game), mcts.nodes)

    def test_inversion_de_puct_preserves_non_prunable_and_bounds_pruned_children(self) -> None:
        mcts = self.make_mcts()
        visits = np.array([5.0, 1000.0, 1000.0], dtype=np.float32)
        q_values = np.array([0.2, 0.1, 0.7], dtype=np.float32)
        priors = np.array([0.4, 0.3, 0.3], dtype=np.float32)
        total_n = 100
        c_star = 0
        c_star_final_puct = 0.6

        pruned = mcts.target_policy_pruning(c_star, c_star_final_puct, visits, total_n, q_values, priors)

        self.assertEqual(pruned[c_star], visits[c_star])
        self.assertEqual(pruned[2], visits[2])
        self.assertLessEqual(pruned[1], visits[1])
        self.assertGreaterEqual(float(pruned.min()), 0.0)
        self.assertLessEqual(
            float(mcts.PUCT(total_n, pruned[1], q_values[1], priors[1])),
            c_star_final_puct + 1e-6,
        )

    def test_tournament_mcts_play_game_creates_one_tree_per_agent(self) -> None:
        class TrackingMCTS(MCTS):
            created = 0
            run_taus: list[float] = []

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                TrackingMCTS.created += 1
                super().__init__(*args, **kwargs)

            def run(self, *args: Any, **kwargs: Any) -> Any:
                TrackingMCTS.run_taus.append(self.tau)
                return super().run(*args, **kwargs)

        original_mcts = tournoi.MCTS
        tournoi.MCTS = TrackingMCTS
        try:
            agent = tournoi.Agent("dummy", CountingNetwork())
            result = tournoi.play_game(
                agent,
                agent,
                simulations=1,
                max_plies=2,
                use_mcts=True,
                move_tau=0.75,
                use_dirichlet_noise=False,
                rng=np.random.default_rng(123),
                first_move_tau=1.0,
            )
        finally:
            tournoi.MCTS = original_mcts

        self.assertEqual(TrackingMCTS.created, 2)
        self.assertEqual(TrackingMCTS.run_taus[0], 1.0)
        self.assertIn(0.75, TrackingMCTS.run_taus)
        self.assertEqual(result["use_mcts"], 1)
        self.assertEqual(result["first_move_tau"], 1.0)
        self.assertEqual(result["move_tau"], 0.75)
        self.assertLessEqual(result["moves"], 2)


if __name__ == "__main__":
    unittest.main(argv=["test_mcts_tactical"])

from pathlib import Path

import numpy as np

from bobailgame import BobailGame, Position
from bobail_config import DIRS, DIR_TO_IDX, POLICY_SHAPE
from bobail_encoding import adapt_moves_pov

def make_game_from_planes(planes: np.ndarray) -> BobailGame:
    game = BobailGame(5)
    current_tokens: list[Position] = [tuple(map(int, pos)) for pos in np.argwhere(planes[0] > 0.5)]
    opponent_tokens: list[Position] = [tuple(map(int, pos)) for pos in np.argwhere(planes[2] > 0.5)]
    bobail_positions = np.argwhere(planes[1] > 0.5)
    if len(current_tokens) != 5 or len(opponent_tokens) != 5 or len(bobail_positions) != 1:
        raise ValueError("Invalid encoded board planes")
    game.tokens = {1: current_tokens, 2: opponent_tokens}
    game.bobail_position = tuple(map(int, bobail_positions[0]))
    game.current_player = 1
    game._sync_board()
    return game

def legal_policy_indices(game: BobailGame) -> set[tuple[int, int, int, int]]:
    outcome = game.check_victory()
    if isinstance(outcome, int):
        return set()
    return {(i0, i1, i2, i3) for _, i0, i1, i2, i3 in adapt_moves_pov(game, outcome)}

def mirrored_policy_indices(policy: np.ndarray) -> list[tuple[int, int, int, int]]:
    dir_map_cols = [DIR_TO_IDX[(dx, -dy)] for (dx, dy) in DIRS]
    nonzero = np.argwhere(policy.reshape(POLICY_SHAPE) > 1e-8)
    mirrored: list[tuple[int, int, int, int]] = []
    for dir1, row, col, dir2 in nonzero:
        mirrored.append((dir_map_cols[int(dir1)], int(row), POLICY_SHAPE[2] - 1 - int(col), dir_map_cols[int(dir2)]))
    return mirrored

def assert_policy_legal(planes: np.ndarray, policy: np.ndarray, mirrored: bool) -> None:
    game = make_game_from_planes(planes[:, :, ::-1] if mirrored else planes)
    legal_indices = legal_policy_indices(game)
    if mirrored:
        policy_indices = mirrored_policy_indices(policy)
    else:
        policy_indices = [tuple(map(int, idx)) for idx in np.argwhere(policy.reshape(POLICY_SHAPE) > 1e-8)]
    illegal_indices = [idx for idx in policy_indices if idx not in legal_indices]
    if illegal_indices:
        kind = "mirrored" if mirrored else "original"
        raise AssertionError(f"Illegal {kind} policy indices: {illegal_indices[:5]}")

def audit_dataset(path: Path, limit: int | None) -> None:
    with np.load(path) as data:
        x = data["x"].copy()
        y_pi = data["y_pi"].copy()
        y_v = data["y_v"].copy()

    if x.shape[1:] != (3, 5, 5):
        raise AssertionError(f"Invalid x shape: {x.shape}")
    if y_pi.shape[1] != int(np.prod(POLICY_SHAPE)):
        raise AssertionError(f"Invalid policy shape: {y_pi.shape}")
    if y_v.shape[1:] != (1,):
        raise AssertionError(f"Invalid value shape: {y_v.shape}")

    plane_sums = x.sum(axis=(2, 3))
    if not np.all(plane_sums[:, 0] == 5) or not np.all(plane_sums[:, 1] == 1) or not np.all(plane_sums[:, 2] == 5):
        raise AssertionError("Invalid encoded plane sums")
    if np.any(x.sum(axis=1) > 1):
        raise AssertionError("Overlapping encoded pieces")
    policy_sums = y_pi.sum(axis=1)
    if not np.allclose(policy_sums, 1.0, atol=1e-5):
        raise AssertionError("Policy targets are not normalized")

    sample_count = len(x) if limit is None else min(limit, len(x))
    for i in range(sample_count):
        assert_policy_legal(x[i], y_pi[i], mirrored=False)
        assert_policy_legal(x[i], y_pi[i], mirrored=True)

    nonzero_counts = (y_pi > 1e-8).sum(axis=1)
    print(f"samples={len(x)} audited={sample_count}")
    print(f"policy_sum_min={policy_sums.min():.8f} policy_sum_mean={policy_sums.mean():.8f} policy_sum_max={policy_sums.max():.8f}")
    print(f"policy_nonzero_mean={nonzero_counts.mean():.2f} policy_nonzero_min={nonzero_counts.min()} policy_nonzero_max={nonzero_counts.max()}")
    print(f"value_min={y_v.min():.4f} value_mean={y_v.mean():.4f} value_max={y_v.max():.4f} value_std={y_v.std():.4f}")


if __name__ == "__main__":
    audit_dataset(Path("merged_data.npz"), None)

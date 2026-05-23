from collections.abc import Iterator, Sequence
from typing import TypeAlias

import numpy as np

from bobail_config import BOARD_SIZE, DIRS, DIR_TO_IDX, POLICY_SHAPE
from bobailgame import BobailGame, Move

PolicyIndex: TypeAlias = tuple[int, int, int, int, int]


def transform_input_data(
    game: BobailGame,
    legal_moves: Sequence[Move] | None = None,
    flip_perspective: bool = True,
) -> np.ndarray:
    """Encode une position, ou les successeurs de coups légaux, pour le réseau."""
    board_size = game.BOARD_SIZE

    def encode_one_state(position: BobailGame) -> np.ndarray:
        current_player = position.current_player
        board = np.zeros((3, board_size, board_size), dtype=np.float32)
        for row, col in position.tokens[current_player]:
            board[0, row, col] = 1.0
        for row, col in position.tokens[3 - current_player]:
            board[2, row, col] = 1.0
        bobail_row, bobail_col = position.bobail_position
        board[1, bobail_row, bobail_col] = 1.0

        if flip_perspective and current_player == 2:
            board = np.flip(board, axis=(1, 2)).copy()
        return board

    if legal_moves is None:
        return encode_one_state(game)

    input_data = np.zeros((len(legal_moves), 3, board_size, board_size), dtype=np.float32)
    for index, move in enumerate(legal_moves):
        reverse = game.apply_move(move, sync_board=False)
        game.current_player = 3 - game.current_player
        input_data[index] = encode_one_state(game)
        game.current_player = 3 - game.current_player
        game.apply_move(reverse, sync_board=False)
    return input_data

def adapt_moves_pov(game: BobailGame, legal_moves: Sequence[Move]) -> Iterator[PolicyIndex]:
    """Mappe les coups légaux vers les indices de policy dans le point de vue réseau."""
    bobail_row, bobail_col = game.bobail_position
    for move_index, ((_, bobail_dest), (_, token_origin, token_dest)) in enumerate(legal_moves):
        dest_row, dest_col = bobail_dest
        origin_row, origin_col = token_origin
        token_row, token_col = token_dest

        if game.current_player == 2:
            board_limit = game.BOARD_SIZE - 1
            bobail_row_2 = board_limit - bobail_row
            bobail_col_2 = board_limit - bobail_col
            dest_row_2 = board_limit - dest_row
            dest_col_2 = board_limit - dest_col
            origin_row_2 = board_limit - origin_row
            origin_col_2 = board_limit - origin_col
            token_row_2 = board_limit - token_row
            token_col_2 = board_limit - token_col
        else:
            bobail_row_2 = bobail_row
            bobail_col_2 = bobail_col
            dest_row_2 = dest_row
            dest_col_2 = dest_col
            origin_row_2 = origin_row
            origin_col_2 = origin_col
            token_row_2 = token_row
            token_col_2 = token_col

        distance = max(abs(origin_row_2 - token_row_2), abs(origin_col_2 - token_col_2))
        bobail_direction = DIR_TO_IDX[(dest_row_2 - bobail_row_2, dest_col_2 - bobail_col_2)]
        token_direction = DIR_TO_IDX[
            ((token_row_2 - origin_row_2) // distance, (token_col_2 - origin_col_2) // distance)
        ]
        yield move_index, bobail_direction, origin_row_2, origin_col_2, token_direction

__all__ = [
    "BOARD_SIZE",
    "DIRS",
    "DIR_TO_IDX",
    "POLICY_SHAPE",
    "PolicyIndex",
    "adapt_moves_pov",
    "transform_input_data",
]
import random
from bisect import bisect_left
from functools import lru_cache
from typing import Iterator, TypeAlias

import numpy as np

Position: TypeAlias = tuple[int, int]
BobailMove: TypeAlias = tuple[str, Position]
TokenMove: TypeAlias = tuple[str, Position, Position]
Move: TypeAlias = tuple[BobailMove, TokenMove]
GameStateKey: TypeAlias = tuple[int, tuple[Position, ...], tuple[Position, ...], Position, int]
BlockerMask: TypeAlias = int
DIRECTIONS: tuple[Position, ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
)

MOVE_CACHE_SIZE = 25_000
LEGAL_MOVE_CACHE_SIZE = 100_000
RAY_CACHE_SIZE = 100_000

def _state_key(
    board_size: int,
    tokens: dict[int, list[Position]],
    bobail_position: Position,
    current_player: int,
) -> GameStateKey:
    return (
        board_size,
        tuple(tokens[1]),
        tuple(tokens[2]),
        bobail_position,
        current_player,
    )

def _blocker_mask(board_size: int, blockers: tuple[Position, ...]) -> BlockerMask:
    mask = 0
    for row, col in blockers:
        mask |= 1 << (row * board_size + col)
    return mask

@lru_cache(maxsize=RAY_CACHE_SIZE)
def _valid_moves_from_blockers_cached(
    board_size: int,
    position: Position,
    blocker_mask: BlockerMask,
    is_bobail: bool,
) -> tuple[Position, ...]:
    valid_moves: list[Position] = []
    for dr, dc in DIRECTIONS:
        current_row, current_col = position
        last_valid: Position | None = None
        while True:
            next_row, next_col = current_row + dr, current_col + dc
            next_position = (next_row, next_col)
            if not (0 <= next_row < board_size and 0 <= next_col < board_size):
                break
            if blocker_mask & (1 << (next_row * board_size + next_col)):
                break
            if is_bobail:
                valid_moves.append(next_position)
                break
            last_valid = next_position
            current_row, current_col = next_row, next_col
        if last_valid is not None:
            valid_moves.append(last_valid)
    return tuple(sorted(valid_moves))

@lru_cache(maxsize=RAY_CACHE_SIZE)
def _has_valid_move_from_blockers_cached(
    board_size: int,
    position: Position,
    blocker_mask: BlockerMask,
) -> bool:
    for dr, dc in DIRECTIONS:
        current_row, current_col = position
        while True:
            next_row, next_col = current_row + dr, current_col + dc
            next_position = (next_row, next_col)
            if not (0 <= next_row < board_size and 0 <= next_col < board_size):
                break
            if blocker_mask & (1 << (next_row * board_size + next_col)):
                break
            return True
    return False

@lru_cache(maxsize=MOVE_CACHE_SIZE)
def _generate_moves_from_key(state_key: GameStateKey) -> tuple[Move, ...]:
    board_size, player_1_tokens, player_2_tokens, bobail_position, current_player = state_key
    token_blocker_mask = _blocker_mask(board_size, player_1_tokens + player_2_tokens)
    current_tokens = player_1_tokens if current_player == 1 else player_2_tokens
    moves: list[Move] = []

    bobail_moves = _valid_moves_from_blockers_cached(board_size, bobail_position, token_blocker_mask, True)
    for bobail_move in bobail_moves:
        bobail_row, bobail_col = bobail_move
        blocker_mask = token_blocker_mask | (1 << (bobail_row * board_size + bobail_col))
        for token in current_tokens:
            token_moves = _valid_moves_from_blockers_cached(board_size, token, blocker_mask, False)
            for token_move in token_moves:
                moves.append((("bobail", bobail_move), ("token", token, token_move)))
    return tuple(moves)

@lru_cache(maxsize=LEGAL_MOVE_CACHE_SIZE)
def _has_legal_move_from_key(state_key: GameStateKey) -> bool:
    board_size, player_1_tokens, player_2_tokens, bobail_position, current_player = state_key
    token_blocker_mask = _blocker_mask(board_size, player_1_tokens + player_2_tokens)
    current_tokens = player_1_tokens if current_player == 1 else player_2_tokens
    bobail_moves = _valid_moves_from_blockers_cached(board_size, bobail_position, token_blocker_mask, True)
    for bobail_move in bobail_moves:
        bobail_row, bobail_col = bobail_move
        blocker_mask = token_blocker_mask | (1 << (bobail_row * board_size + bobail_col))
        for token in current_tokens:
            if _has_valid_move_from_blockers_cached(board_size, token, blocker_mask):
                return True
    return False

class BobailGame:
    def __init__(self, board_size: int = 5) -> None:
        self.BOARD_SIZE = board_size
        self.tokens: dict[int, list[Position]] = {
            1: [(board_size - 1, i) for i in range(board_size)],
            2: [(0, i) for i in range(board_size)],
        }
        self.bobail_position = (board_size // 2, board_size // 2)
        self.board = np.zeros((board_size, board_size), dtype=int)
        self._sync_board()
        self.current_player = 1

    def _sync_board(self) -> None:
        self.board.fill(0)
        for player, tokens in self.tokens.items():
            for token in tokens:
                self.board[token] = player
        self.board[self.bobail_position] = 3

    def state_key(self) -> GameStateKey:
        return _state_key(self.BOARD_SIZE, self.tokens, self.bobail_position, self.current_player)

    def _valid_moves_from_blockers(self, position: Position, blockers: set[Position], is_bobail: bool = False) -> list[Position]:
        blocker_mask = _blocker_mask(self.BOARD_SIZE, tuple(blockers))
        return list(_valid_moves_from_blockers_cached(self.BOARD_SIZE, position, blocker_mask, is_bobail))

    def _has_valid_move_from_blockers(self, position: Position, blockers: set[Position]) -> bool:
        blocker_mask = _blocker_mask(self.BOARD_SIZE, tuple(blockers))
        return _has_valid_move_from_blockers_cached(self.BOARD_SIZE, position, blocker_mask)

    def get_valid_moves(self, position: Position, tokens: list[Position], is_bobail: bool = False) -> list[Position]:
        return self._valid_moves_from_blockers(position, set(tokens), is_bobail=is_bobail)

    def iter_moves(self, tokens: dict[int, list[Position]], bobail_position: Position, current_player: int) -> Iterator[Move]:
        yield from _generate_moves_from_key(_state_key(self.BOARD_SIZE, tokens, bobail_position, current_player))

    def has_legal_move(self, tokens: dict[int, list[Position]], bobail_position: Position, current_player: int) -> bool:
        return _has_legal_move_from_key(_state_key(self.BOARD_SIZE, tokens, bobail_position, current_player))

    def generate_moves(self, tokens: dict[int, list[Position]], bobail_position: Position, current_player: int) -> list[Move]:
        return list(_generate_moves_from_key(_state_key(self.BOARD_SIZE, tokens, bobail_position, current_player)))

    def center_position(self) -> Position:
        return self.BOARD_SIZE // 2, self.BOARD_SIZE // 2

    def balanced_first_move_bobail_position(self) -> Position:
        if self.BOARD_SIZE < 3:
            raise ValueError("Balanced first move needs a board of size at least 3")
        middle = self.BOARD_SIZE // 2
        return middle, middle - 1

    def balanced_first_move_search_game(self) -> "BobailGame":
        search_game = BobailGame(self.BOARD_SIZE)
        search_game.tokens = {1: list(self.tokens[1]), 2: list(self.tokens[2])}
        search_game.bobail_position = self.balanced_first_move_bobail_position()
        search_game.current_player = self.current_player
        occupied = set(search_game.tokens[1]) | set(search_game.tokens[2])
        if search_game.bobail_position in occupied:
            raise ValueError("Balanced first move bobail offset is occupied")
        search_game._sync_board()
        return search_game

    def balanced_first_move_legal_moves(self) -> tuple["BobailGame", list[Move]]:
        search_game = self.balanced_first_move_search_game()
        center = self.center_position()
        legal_moves = search_game.generate_moves(search_game.tokens, search_game.bobail_position, search_game.current_player)
        centered_moves = [move for move in legal_moves if move[0][1] == center]
        if not centered_moves:
            raise ValueError("No balanced first moves return the bobail to the center")
        return search_game, centered_moves

    def apply_move(self, move: Move, sync_board: bool = True) -> Move:
        bobail_move, token_move = move
        _, new_bobail_pos = bobail_move
        _, token_pos, new_token_pos = token_move
        old_bobail_pos = self.bobail_position
        self.bobail_position = new_bobail_pos
        player_tokens = self.tokens[self.current_player]
        token_index = player_tokens.index(token_pos)
        del player_tokens[token_index]
        player_tokens.insert(bisect_left(player_tokens, new_token_pos), new_token_pos)

        if sync_board:
            self._sync_board()

        return ("bobail", old_bobail_pos), ("token", new_token_pos, token_pos)

    def check_victory(self) -> int | list[Move]:
        if self.bobail_position[0] == 0 or self.bobail_position[0] == self.BOARD_SIZE - 1:
            return 1 + int(self.bobail_position[0] == 0)
        moves = self.generate_moves(self.tokens, self.bobail_position, self.current_player)
        if not moves:
            return 3 - self.current_player
        return moves

    @staticmethod
    def generate_random_position(board_size: int = 5, num_tokens_per_player: int = 5) -> tuple[dict[int, list[Position]], Position, int]:
        if board_size < 3:
            raise ValueError("Random starts need a board of size at least 3")
        if 2 * num_tokens_per_player + 1 > board_size * board_size:
            raise ValueError("Too many pieces for the board size")

        middle_row = board_size // 2
        while True:
            bobail_position = (middle_row, random.randrange(board_size))
            all_positions = [(r, c) for r in range(board_size) for c in range(board_size) if (r, c) != bobail_position]
            random.shuffle(all_positions)
            player_1_tokens = sorted(all_positions[:num_tokens_per_player])
            player_2_tokens = sorted(all_positions[num_tokens_per_player:2 * num_tokens_per_player])
            current_player = random.choice([1, 2])
            tokens = {1: player_1_tokens, 2: player_2_tokens}

            game = BobailGame(board_size)
            game.tokens = {1: list(tokens[1]), 2: list(tokens[2])}
            game.bobail_position = bobail_position
            game.current_player = current_player
            game._sync_board()
            if not isinstance(game.check_victory(), int):
                return tokens, bobail_position, current_player

    @staticmethod
    def generate_random_game(board_size: int = 5, num_tokens_per_player: int = 5) -> "BobailGame":
        tokens, bobail_position, current_player = BobailGame.generate_random_position(board_size, num_tokens_per_player)
        game = BobailGame(board_size)
        game.tokens = {1: list(tokens[1]), 2: list(tokens[2])}
        game.bobail_position = bobail_position
        game.current_player = current_player
        game._sync_board()
        return game

    def __str__(self) -> str:
        symbols = {0: "  ", 1: "V1", 2: "R2", 3: "BB"}
        header = "   " + " ".join(f"{c}" for c in range(self.BOARD_SIZE))
        lines = [f"Joueur courant : {self.current_player}", header]
        for r in range(self.BOARD_SIZE):
            row = " ".join(symbols[int(self.board[r, c])] for c in range(self.BOARD_SIZE))
            lines.append(f"{r}  {row}")
        return "\n".join(lines)

    def __hash__(self) -> int:
        return hash(self.state_key())

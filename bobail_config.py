import math
import os
import time
from typing import TypeAlias

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

NDarrayFloat32: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
NDarrayFloat32_1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float32]]
IntLike: TypeAlias = int | np.int64
FloatLike: TypeAlias = float | np.float32

BOARD_SIZE = 5
DIRS = [(-1, 1), (-1, 0), (-1, -1), (0, 1), (0, -1), (1, -1), (1, 0), (1, 1)]
DIR_TO_IDX = {direction: i for i, direction in enumerate(DIRS)}
POLICY_SHAPE = (8, BOARD_SIZE, BOARD_SIZE, 8)
POLICY_SIZE = math.prod(POLICY_SHAPE)

SELF_PLAY_N_SIMULATIONS = 1024
SELF_PLAY_MAX_PLIES = 80
SELF_PLAY_TAU = 0.75
SELF_PLAY_FIRST_MOVE_TAU = 1.0
SELF_PLAY_TD_LAMBDA = 0.20
SELF_PLAY_GAMES_PER_ITERATION = 1000
SELF_PLAY_RANDOM_START_RATE = 0.20

MCTS_C_PUCT = 2.0 # 3
MCTS_PUCT_PARENT_EXPONENT = 0.5
MCTS_DIRICHLET_ALPHA_0 = 10.0
MCTS_DIRICHLET_NOISE_WEIGHT = 0.20

TORCH_INFERENCE_THREADS = 1
TORCH_INTEROP_THREADS = 1

TRAINING_EPOCHS = 20
TRAINING_EARLY_STOPPING_PATIENCE = 2
TRAINING_BATCH_SIZE = 128
TRAINING_LEARNING_RATE = 1e-4
TRAINING_MIN_LEARNING_RATE = 3e-5
TRAINING_LR_SCHEDULER_FACTOR = 0.5
TRAINING_LR_SCHEDULER_PATIENCE = 1
TRAINING_BUFFER_DIR = "./training_buffer"
TRAINING_BUFFER_MAX_RAW_SAMPLES = 150_000

SAVED_NETWORK_PATH = "./bobail_zero.pth"
ITERATION_CHECKPOINT_DIR = "./checkpoints"
RAW_SELF_PLAY_DATA_PATH = "./advanced_data.npz"
MERGED_TRAINING_DATA_PATH = "./merged_data.npz"
PENDING_SELF_PLAY_METRICS_PATH = "./pending_selfplay_metrics.json"

GENERATE_SELF_PLAY_DATA = True
UPDATE_NETWORK = False

RNG = RandomState(MT19937(SeedSequence(int(time.time() * 1e6) + os.getpid())))

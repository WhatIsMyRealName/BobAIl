import tkinter as tk
from tkinter import messagebox, Tk
import copy
from typing import Literal, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F

from MCTS_visu import MCTSVisualizer3D
from bobail_config import MCTS_C_PUCT, SELF_PLAY_N_SIMULATIONS
from bobail_encoding import adapt_moves_pov, transform_input_data
from bobailgame import BobailGame
from bobail_mcts import MCTS
from bobail_model_zero import _Network
CELL_SIZE = 80
MARGIN = 20
BOARD_COLOR = "#A85C10"
ALT_BOARD_COLOR = "#B06820"
LEGAL_MOVE_COLOR = "#081e9b"
HIGHLIGHT_BOBAIL = "#d9b800"
GREEN_COLOR = "#1b891b"
RED_COLOR = "#cc0000"
SELECTED_TOKEN_BORDER = "#f0b324"
DEST_BORDER = "#02ba02"
MODEL_PATH = "./bobail_zero.pth"

ColorName: TypeAlias = Literal["green", "red"]
GoalSide: TypeAlias = Literal["top", "bottom"]

COLOR_LABELS: dict[ColorName, str] = {
    "green": "Vert",
    "red": "Rouge",
}
COLOR_FILLS: dict[ColorName, str] = {
    "green": GREEN_COLOR,
    "red": RED_COLOR,
}

class GuiPlay:
    def __init__(
        self,
        root: Tk,
        network: _Network,
        game: BobailGame | None = None,
        device: str = "cpu",
        human_color: ColorName = "green",
        green_goal: GoalSide = "top",
        starting_color: ColorName = "green",
        flip_board: bool = False,
    ) -> None:
        human_color = self.parse_color(human_color)
        green_goal = self.parse_goal(green_goal)
        starting_color = self.parse_color(starting_color)
        self.root = root
        self.device = device
        self.network = network
        self.human_color: ColorName = human_color
        self.network_color: ColorName = self.other_color(human_color)
        self.green_goal: GoalSide = green_goal
        self.starting_color: ColorName = starting_color
        self.flip_board = bool(flip_board)
        self.human_player = self.player_for_color(self.human_color)
        network.export()
        self.tree = MCTS(self.network, n_simulations=SELF_PLAY_N_SIMULATIONS, tau=0, max_depth=80, c_puct=MCTS_C_PUCT, device=self.device, use_dirichlet_noise=False)

        self.game = game if game is not None else BobailGame()
        self.game.current_player = self.player_for_color(self.starting_color)
        self.board_size = self.game.BOARD_SIZE

        self.history = []
        self.selection = {"phase": None, "bobail_dest": None, "token_origin": None}
        self.temp_bobail = None
        self.first_move_token_only = True

        self.eval_height = 22
        self.eval_color_left = self.player_fill(1)
        self.eval_color_right = self.player_fill(2)

        self.eval_canvas = tk.Canvas(root, height=self.eval_height, bg="white", highlightthickness=0)
        self.eval_canvas.pack(side=tk.TOP, fill=tk.X)
        self.eval_rect_left  = self.eval_canvas.create_rectangle(0, 0, 0, self.eval_height, fill=self.eval_color_left, width=0)
        self.eval_rect_right = self.eval_canvas.create_rectangle(0, 0, 0, self.eval_height, fill=self.eval_color_right, width=0)
        self.eval_text_left  = self.eval_canvas.create_text(6, self.eval_height/2, anchor="w", text="", font=("TkDefaultFont", 9), fill="white")
        self.eval_text_right = self.eval_canvas.create_text(0, self.eval_height/2, anchor="e", text="", font=("TkDefaultFont", 9), fill="white")
        self.eval_canvas.bind("<Configure>", lambda ev: self.update_eval_bar())

        main_frame = tk.Frame(root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        canvas_size = 2*MARGIN + CELL_SIZE * self.board_size
        self.canvas = tk.Canvas(main_frame, width=canvas_size, height=canvas_size, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        self.mode_var = tk.StringVar(value="mcts")
        self.show_mcts_var = tk.BooleanVar(value=False)
        self.human_color_var = tk.StringVar(value=self.human_color)
        self.network_color_var = tk.StringVar(value=self.network_color)
        self.green_goal_var = tk.StringVar(value=self.green_goal)
        self.starting_color_var = tk.StringVar(value=self.starting_color)
        self.flip_board_var = tk.BooleanVar(value=self.flip_board)
        self.setup_mode_var = tk.BooleanVar(value=False)
        self.setup_piece_var = tk.StringVar(value="green")
        self.setup_current_color_var = tk.StringVar(value=self.color_for_player(self.game.current_player))

        tk.Label(left_frame, text="Bobail").pack(pady=(0, 8))

        actions_frame = tk.LabelFrame(left_frame, text="Actions", padx=6, pady=6)
        actions_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Button(actions_frame, text="New Game", command=self.new_game).pack(fill=tk.X)
        tk.Button(actions_frame, text="Undo last move", command=self.undo).pack(fill=tk.X, pady=(4, 0))
        tk.Button(actions_frame, text="Cancel selection", command=self.cancel_selection).pack(fill=tk.X, pady=(4, 0))
        tk.Button(actions_frame, text="Need help?", command=self.help_play).pack(fill=tk.X, pady=(4, 0))

        player_frame = tk.LabelFrame(left_frame, text="Joueur", padx=6, pady=6)
        player_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Label(player_frame, text="Humain").pack(anchor=tk.W)
        tk.Radiobutton(player_frame, text="Vert", variable=self.human_color_var, value="green", command=self.change_human_color).pack(anchor=tk.W)
        tk.Radiobutton(player_frame, text="Rouge", variable=self.human_color_var, value="red", command=self.change_human_color).pack(anchor=tk.W)
        tk.Label(player_frame, text="Réseau").pack(anchor=tk.W, pady=(6, 0))
        tk.Radiobutton(player_frame, text="Vert", variable=self.network_color_var, value="green", command=self.change_network_color).pack(anchor=tk.W)
        tk.Radiobutton(player_frame, text="Rouge", variable=self.network_color_var, value="red", command=self.change_network_color).pack(anchor=tk.W)
        tk.Label(player_frame, text="Objectif du vert").pack(anchor=tk.W, pady=(6, 0))
        tk.Radiobutton(player_frame, text="Haut", variable=self.green_goal_var, value="top", command=self.change_green_goal).pack(anchor=tk.W)
        tk.Radiobutton(player_frame, text="Bas", variable=self.green_goal_var, value="bottom", command=self.change_green_goal).pack(anchor=tk.W)
        tk.Label(player_frame, text="Trait initial").pack(anchor=tk.W, pady=(6, 0))
        tk.Radiobutton(player_frame, text="Vert", variable=self.starting_color_var, value="green", command=self.change_starting_color).pack(anchor=tk.W)
        tk.Radiobutton(player_frame, text="Rouge", variable=self.starting_color_var, value="red", command=self.change_starting_color).pack(anchor=tk.W)
        tk.Checkbutton(player_frame, text="Retourner le plateau", variable=self.flip_board_var, command=self.change_board_orientation).pack(anchor=tk.W, pady=(4, 0))

        network_frame = tk.LabelFrame(right_frame, text="Réseau", padx=6, pady=6)
        network_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Radiobutton(network_frame, text="MCTS", variable=self.mode_var, value="mcts").pack(anchor=tk.W)
        tk.Radiobutton(network_frame, text="Policy", variable=self.mode_var, value="policy").pack(anchor=tk.W)
        tk.Radiobutton(network_frame, text="Value", variable=self.mode_var, value="value").pack(anchor=tk.W)
        tk.Checkbutton(network_frame, text="Visualiser MCTS", variable=self.show_mcts_var).pack(anchor=tk.W, pady=(4, 0))

        custom_frame = tk.LabelFrame(right_frame, text="Position custom", padx=6, pady=6)
        custom_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Checkbutton(custom_frame, text="Editer", variable=self.setup_mode_var, command=self.toggle_setup_mode).pack(anchor=tk.W)
        tk.Label(custom_frame, text="Piece").pack(anchor=tk.W, pady=(6, 0))
        tk.Radiobutton(custom_frame, text="Bobail", variable=self.setup_piece_var, value="bobail").pack(anchor=tk.W)
        tk.Radiobutton(custom_frame, text="Pion vert", variable=self.setup_piece_var, value="green").pack(anchor=tk.W)
        tk.Radiobutton(custom_frame, text="Pion rouge", variable=self.setup_piece_var, value="red").pack(anchor=tk.W)
        tk.Radiobutton(custom_frame, text="Vide", variable=self.setup_piece_var, value="empty").pack(anchor=tk.W)
        tk.Label(custom_frame, text="Trait").pack(anchor=tk.W, pady=(6, 0))
        tk.Radiobutton(custom_frame, text="Vert", variable=self.setup_current_color_var, value="green").pack(anchor=tk.W)
        tk.Radiobutton(custom_frame, text="Rouge", variable=self.setup_current_color_var, value="red").pack(anchor=tk.W)
        tk.Button(custom_frame, text="Position aleatoire", command=self.apply_random_custom_position).pack(fill=tk.X, pady=(6, 0))
        tk.Button(custom_frame, text="Appliquer", command=self.apply_custom_position).pack(fill=tk.X, pady=(6, 0))

        self.status_label = tk.Label(right_frame, text="", wraplength=200, justify=tk.LEFT)
        self.status_label.pack(pady=8)

        self.redraw()
        self.root.after(100, self.play_network_turn_if_needed)

    @staticmethod
    def parse_color(color: str) -> ColorName:
        if color not in COLOR_LABELS:
            raise ValueError("color must be 'green' or 'red'")
        return "green" if color == "green" else "red"

    @staticmethod
    def parse_goal(goal: str) -> GoalSide:
        if goal not in ("top", "bottom"):
            raise ValueError("goal must be 'top' or 'bottom'")
        return "top" if goal == "top" else "bottom"

    @staticmethod
    def validate_color(color: str) -> None:
        GuiPlay.parse_color(color)

    @staticmethod
    def validate_goal(goal: str) -> None:
        GuiPlay.parse_goal(goal)

    @staticmethod
    def other_color(color: ColorName) -> ColorName:
        return "red" if color == "green" else "green"

    def player_for_color(self, color: ColorName) -> int:
        green_player = 2 if self.green_goal == "top" else 1
        return green_player if color == "green" else 3 - green_player

    def color_for_player(self, player: int) -> ColorName:
        if player == self.player_for_color("green"):
            return "green"
        return "red"

    def player_color_name(self, player: int) -> str:
        return COLOR_LABELS[self.color_for_player(player)]

    def player_fill(self, player: int) -> str:
        return COLOR_FILLS[self.color_for_player(player)]

    def color_label(self, color: ColorName) -> str:
        return COLOR_LABELS[color]

    def color_goal_label(self, color: ColorName) -> str:
        if color == "green":
            return "haut" if self.green_goal == "top" else "bas"
        return "bas" if self.green_goal == "top" else "haut"

    def is_human_turn(self) -> bool:
        return self.game.current_player == self.human_player

    def is_first_human_starting_move(self) -> bool:
        return (
            self.first_move_token_only
            and self.game.current_player == self.player_for_color(self.starting_color)
            and self.is_human_turn()
        )

    def push_history(self) -> None:
        self.history.append((copy.deepcopy(self.game), self.first_move_token_only))

    def change_human_color(self) -> None:
        color = self.parse_color(self.human_color_var.get())
        self.human_color = color
        self.network_color = self.other_color(self.human_color)
        self.network_color_var.set(self.network_color)
        self.human_player = self.player_for_color(self.human_color)
        self.new_game(confirm=False)

    def change_network_color(self) -> None:
        color = self.parse_color(self.network_color_var.get())
        self.network_color = color
        self.human_color = self.other_color(self.network_color)
        self.human_color_var.set(self.human_color)
        self.human_player = self.player_for_color(self.human_color)
        self.new_game(confirm=False)

    def change_green_goal(self) -> None:
        goal = self.parse_goal(self.green_goal_var.get())
        self.green_goal = goal
        self.human_player = self.player_for_color(self.human_color)
        self.new_game(confirm=False)

    def change_starting_color(self) -> None:
        color = self.parse_color(self.starting_color_var.get())
        self.starting_color = color
        self.new_game(confirm=False)

    def change_board_orientation(self) -> None:
        self.flip_board = bool(self.flip_board_var.get())
        self.redraw()

    def play_network_turn_if_needed(self) -> None:
        if self.network is not None and not self.is_human_turn() and not self.setup_mode_var.get():
            self.push_history()
            self.status_label.configure(text=f"Network thinking as {self.player_color_name(self.game.current_player)}...")
            self.redraw()
            self.network_move()

    def toggle_setup_mode(self) -> None:
        self.selection = {"phase": None, "bobail_dest": None, "token_origin": None}
        self.temp_bobail = None
        self.setup_current_color_var.set(self.color_for_player(self.game.current_player))
        self.redraw()

    def apply_custom_position(self) -> None:
        green_player = self.player_for_color("green")
        red_player = self.player_for_color("red")
        if len(self.game.tokens[green_player]) != self.board_size or len(self.game.tokens[red_player]) != self.board_size:
            messagebox.showinfo("Custom position", f"Il faut exactement {self.board_size} pions verts et {self.board_size} pions rouges.")
            return
        occupied = set(self.game.tokens[1]) | set(self.game.tokens[2])
        if self.game.bobail_position in occupied:
            messagebox.showinfo("Custom position", "Le bobail ne peut pas etre sur un pion.")
            return
        color = self.parse_color(self.setup_current_color_var.get())
        self.game.current_player = self.player_for_color(color)
        self.game._sync_board()
        self.history = []
        self.selection = {"phase": None, "bobail_dest": None, "token_origin": None}
        self.temp_bobail = None
        self.first_move_token_only = False
        self.setup_mode_var.set(False)
        self.tree = MCTS(self.network, n_simulations=SELF_PLAY_N_SIMULATIONS, tau=0, max_depth=80, c_puct=MCTS_C_PUCT, device=self.device, use_dirichlet_noise=False)
        self.redraw()
        self.root.after(100, self.play_network_turn_if_needed)

    def apply_random_custom_position(self) -> None:
        self.game = BobailGame.generate_random_game(self.board_size, self.board_size)
        self.setup_current_color_var.set(self.color_for_player(self.game.current_player))
        self.history = []
        self.selection = {"phase": None, "bobail_dest": None, "token_origin": None}
        self.temp_bobail = None
        self.first_move_token_only = False
        self.setup_mode_var.set(False)
        self.tree = MCTS(self.network, n_simulations=SELF_PLAY_N_SIMULATIONS, tau=0, max_depth=80, c_puct=MCTS_C_PUCT, device=self.device, use_dirichlet_noise=False)
        self.redraw()
        self.root.after(100, self.play_network_turn_if_needed)
    
    def get_win_probability(self):
        x2 = transform_input_data(self.game)
        x2_t = torch.from_numpy(x2).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out2 = self.network(x2_t)
        v2 = out2[1]
        v2 = float(v2.cpu().numpy()[0][0]) if hasattr(v2, "cpu") else float(v2) #type: ignore
        if self.game.current_player == 1:
            return (v2+1)/2
        else:
            return 1 - (v2+1)/2
    
    def update_eval_bar(self):
        """
        Met à jour la barre selon la probabilité retournée par self.get_win_probability().
        On s'attend à une valeur entre 0.0 et 1.0 correspondant à la probabilité du premier joueur.
        """
        prob = float(self.get_win_probability())
        w = max(1, self.eval_canvas.winfo_width())
        h = self.eval_height

        left_w = int(round(w * prob))

        self.eval_canvas.coords(self.eval_rect_left, 0, 0, left_w, h)
        self.eval_canvas.coords(self.eval_rect_right, left_w, 0, w, h)
        self.eval_canvas.itemconfig(self.eval_rect_left, fill=self.player_fill(1))
        self.eval_canvas.itemconfig(self.eval_rect_right, fill=self.player_fill(2))

        left_text = f"{int(round(prob*100))}%"
        right_text = f"{int(round((1-prob)*100))}%"

        self.eval_canvas.coords(self.eval_text_left, 6, h/2)
        self.eval_canvas.itemconfig(self.eval_text_left, text=f"{left_text} {self.player_color_name(1)}")

        self.eval_canvas.coords(self.eval_text_right, w-6, h/2)
        self.eval_canvas.itemconfig(self.eval_text_right, text=f"{self.player_color_name(2)} {right_text}")

        self.eval_canvas.itemconfig(self.eval_text_left, fill="white")
        self.eval_canvas.itemconfig(self.eval_text_right, fill="white")

    def board_to_view(self, r, c):
        if self.flip_board:
            return self.board_size - 1 - r, self.board_size - 1 - c
        return r, c

    def view_to_board(self, r, c):
        if self.flip_board:
            return self.board_size - 1 - r, self.board_size - 1 - c
        return r, c

    def board_to_pixel(self, r, c):
        view_r, view_c = self.board_to_view(r, c)
        x = MARGIN + view_c * CELL_SIZE
        y = MARGIN + view_r * CELL_SIZE
        return x, y, x + CELL_SIZE, y + CELL_SIZE

    def pixel_to_board(self, px, py):
        cx = px - MARGIN
        cy = py - MARGIN
        if cx < 0 or cy < 0:
            return None
        c = int(cx // CELL_SIZE)
        r = int(cy // CELL_SIZE)
        if 0 <= r < self.board_size and 0 <= c < self.board_size:
            return self.view_to_board(r, c)
        return None

    def draw_board(self):
        self.canvas.delete("all")
        for r in range(self.board_size):
            for c in range(self.board_size):
                x1, y1, x2, y2 = self.board_to_pixel(r, c)
                color = BOARD_COLOR if (r + c) % 2 == 0 else ALT_BOARD_COLOR
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black", tags=f"cell_{r}_{c}")

    def redraw(self):
        self.draw_board()
        self.update_eval_bar()
        # highlight depending on selection phase
        if not self.setup_mode_var.get():
            if self.is_first_human_starting_move():
                if self.selection["phase"] is None:
                    self.highlight_token_only_origins()
                elif self.selection["phase"] == "token_only_origin":
                    self.highlight_token_only_dests(self.selection["token_origin"])
            elif self.selection["phase"] is None:
                self.highlight_bobail_dests()
            elif self.selection["phase"] == "bobail":
                self.highlight_token_origins(self.selection["bobail_dest"])
            elif self.selection["phase"] == "token_origin":
                self.highlight_token_dests(self.selection["bobail_dest"], self.selection["token_origin"])

        # draw bobail: if temp_bobail exists, show bobail at new position
        if self.temp_bobail is not None:
            _, new_b = self.temp_bobail
            br, bc = new_b
        else:
            br, bc = self.game.bobail_position

        x1, y1, x2, y2 = self.board_to_pixel(br, bc)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        r = CELL_SIZE * 0.28
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=HIGHLIGHT_BOBAIL, outline="black", width=2)

        # draw tokens
        for player, coords in self.game.tokens.items():
            color = self.player_fill(player)
            for (r0, c0) in coords:
                x1, y1, x2, y2 = self.board_to_pixel(r0, c0)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                rr = CELL_SIZE * 0.22
                self.canvas.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, fill=color, outline="black", width=1)

        if self.setup_mode_var.get():
            green_player = self.player_for_color("green")
            red_player = self.player_for_color("red")
            status = (
                f"Custom edit - green={len(self.game.tokens[green_player])}/{self.board_size}, "
                f"red={len(self.game.tokens[red_player])}/{self.board_size}, "
                f"trait={self.color_label(self.parse_color(self.setup_current_color_var.get()))}"
            )
        else:
            turn_owner = "Human" if self.is_human_turn() else "Network"
            current_color = self.player_color_name(self.game.current_player)
            suffix = " - first move: token only" if self.is_first_human_starting_move() else ""
            status = (
                f"{turn_owner} turn: {current_color} (player {self.game.current_player}) - "
                f"target={self.color_goal_label(self.color_for_player(self.game.current_player))} - "
                f"Mode: {self.mode_var.get().upper()}{suffix}"
            )
        self.status_label.configure(text=status)
        self.root.update()

    def get_legal_moves(self):
        return self.game.generate_moves(self.game.tokens, self.game.bobail_position, self.game.current_player)

    def highlight_bobail_dests(self):
        legal = self.get_legal_moves()
        dests = set()
        for m in legal:
            d = self._get_bobail_dest_from_move(m)
            if d is not None:
                dests.add(d)
        for (r, c) in dests:
            x1, y1, x2, y2 = self.board_to_pixel(r, c)
            self.canvas.create_rectangle(x1+4, y1+4, x2-4, y2-4, outline=LEGAL_MOVE_COLOR, width=3)

    def highlight_token_origins(self, bobail_dest):
        legal = self.get_legal_moves()
        origins = set()
        for m in legal:
            d = self._get_bobail_dest_from_move(m)
            if d != bobail_dest:
                continue
            old, new = self._get_token_old_new_from_move(m)
            if old is not None:
                origins.add(old)
        for (r, c) in origins:
            x1, y1, x2, y2 = self.board_to_pixel(r, c)
            self.canvas.create_rectangle(x1+6, y1+6, x2-6, y2-6, outline=SELECTED_TOKEN_BORDER, width=3)

    def highlight_token_dests(self, bobail_dest, token_origin):
        legal = self.get_legal_moves()
        dests = set()
        for m in legal:
            d = self._get_bobail_dest_from_move(m)
            if d != bobail_dest:
                continue
            old, new = self._get_token_old_new_from_move(m)
            if old == token_origin and new is not None:
                dests.add(new)
        for (r, c) in dests:
            x1, y1, x2, y2 = self.board_to_pixel(r, c)
            self.canvas.create_rectangle(x1+8, y1+8, x2-8, y2-8, outline=DEST_BORDER, width=3)

    def token_only_moves(self):
        blockers = set(self.game.tokens[1]) | set(self.game.tokens[2]) | {self.game.bobail_position}
        moves = []
        for token in self.game.tokens[self.game.current_player]:
            for dest in self.game._valid_moves_from_blockers(token, blockers, is_bobail=False):
                moves.append((("bobail", self.game.bobail_position), ("token", token, dest)))
        return sorted(moves)

    def highlight_token_only_origins(self):
        origins = {move[1][1] for move in self.token_only_moves()}
        for (r, c) in origins:
            x1, y1, x2, y2 = self.board_to_pixel(r, c)
            self.canvas.create_rectangle(x1+6, y1+6, x2-6, y2-6, outline=SELECTED_TOKEN_BORDER, width=3)

    def highlight_token_only_dests(self, token_origin):
        dests = {move[1][2] for move in self.token_only_moves() if move[1][1] == token_origin}
        for (r, c) in dests:
            x1, y1, x2, y2 = self.board_to_pixel(r, c)
            self.canvas.create_rectangle(x1+8, y1+8, x2-8, y2-8, outline=DEST_BORDER, width=3)

    def _is_token_only_origin_legal(self, origin):
        return any(move[1][1] == origin for move in self.token_only_moves())

    def _is_token_only_dest_legal(self, origin, dest):
        return any(move[1][1] == origin and move[1][2] == dest for move in self.token_only_moves())

    def _find_token_only_move(self, origin, dest):
        for move in self.token_only_moves():
            if move[1][1] == origin and move[1][2] == dest:
                return move
        return None

    def remove_piece_at(self, position) -> None:
        self.game.tokens[1] = [token for token in self.game.tokens[1] if token != position]
        self.game.tokens[2] = [token for token in self.game.tokens[2] if token != position]

    def edit_custom_cell(self, event) -> None:
        pos = self.pixel_to_board(event.x, event.y)
        if pos is None:
            return
        piece = self.setup_piece_var.get()
        self.remove_piece_at(pos)
        if piece == "bobail":
            self.game.bobail_position = pos
        elif piece == "green":
            player = self.player_for_color("green")
            if pos == self.game.bobail_position:
                self.status_label.configure(text="Cannot place a token on the bobail.")
                self.game._sync_board()
                self.redraw()
                return
            if len(self.game.tokens[player]) >= self.board_size:
                self.status_label.configure(text=f"Already {self.board_size} green tokens.")
                self.game._sync_board()
                self.redraw()
                return
            self.game.tokens[player].append(pos)
        elif piece == "red":
            player = self.player_for_color("red")
            if pos == self.game.bobail_position:
                self.status_label.configure(text="Cannot place a token on the bobail.")
                self.game._sync_board()
                self.redraw()
                return
            if len(self.game.tokens[player]) >= self.board_size:
                self.status_label.configure(text=f"Already {self.board_size} red tokens.")
                self.game._sync_board()
                self.redraw()
                return
            self.game.tokens[player].append(pos)
        elif piece == "empty" and pos == self.game.bobail_position:
            self.status_label.configure(text="The bobail must stay on the board.")
        self.game.tokens[1].sort()
        self.game.tokens[2].sort()
        self.game._sync_board()
        self.redraw()

    def _find_bobail_dest_for_origin(self, origin):
        legal = self.get_legal_moves()
        for m in legal:
            old, new = self._get_token_old_new_from_move(m)
            d = self._get_bobail_dest_from_move(m)
            if old == origin and d is not None:
                return d
        return None

    def on_canvas_click(self, event):
        if self.setup_mode_var.get():
            self.edit_custom_cell(event)
            return

        if not self.is_human_turn():
            self.status_label.configure(text="Wait for the network to play.")
            return

        pos = self.pixel_to_board(event.x, event.y)
        if pos is None:
            return
        r, c = pos

        if self.is_first_human_starting_move():
            if self.selection["phase"] is None:
                if self._is_token_only_origin_legal((r, c)):
                    self.push_history()
                    self.selection["phase"] = "token_only_origin" # type: ignore
                    self.selection["token_origin"] = (r, c) # type: ignore
                    self.selection["bobail_dest"] = self.game.bobail_position # type: ignore
                    self.status_label.configure(text=f"Token origin {(r,c)} selected. Choose destination.")
                    self.redraw()
                    return
                self.status_label.configure(text="Select a highlighted token origin for the first move.")
                return

            if self.selection["phase"] == "token_only_origin":
                origin = self.selection["token_origin"]
                if self._is_token_only_origin_legal((r, c)):
                    self.selection["token_origin"] = (r, c) # type: ignore
                    self.status_label.configure(text=f"Token origin {(r,c)} selected. Choose destination.")
                    self.redraw()
                    return
                if self._is_token_only_dest_legal(origin, (r, c)):
                    move = self._find_token_only_move(origin, (r, c))
                    if move is None:
                        self.status_label.configure(text="Move not found (unexpected). Selection cancelled.")
                        self.cancel_selection()
                        return
                    self.game.apply_move(move)
                    self.game.current_player = 3 - self.game.current_player
                    self.first_move_token_only = False
                    self.selection = {"phase": None, "bobail_dest": None, "token_origin": None}
                    self.temp_bobail = None
                    self.status_label.configure(text="First token-only move applied. Network will respond if available.")
                    self.redraw()

                    winner = self.game.check_victory()
                    if type(winner) == int:
                        messagebox.showinfo("Game over", f"Winner: {self.player_color_name(winner)} (Player {winner}).\nNombre de noeuds visites par le MCTS: {len(self.tree.nodes)}.")
                        return

                    if self.network is not None and not self.is_human_turn():
                        self.push_history()
                        self.network_move()
                    return
                self.status_label.configure(text=f"{(r,c)} is not a legal destination for the selected origin.")
                return

        # Phase None: select bobail dest or click a token origin (inferred bobail)
        if self.selection["phase"] is None:
            if self._is_bobail_dest_legal((r, c)):
                self.push_history()
                old_b = self.game.bobail_position
                new_b = (r, c)
                self.selection["phase"] = "bobail" # type: ignore
                self.selection["bobail_dest"] = new_b # type: ignore
                self.selection["token_origin"] = None
                self.temp_bobail = (old_b, new_b)
                self.status_label.configure(text=f"Bobail destination {new_b} selected. Choose a token origin.")
                self.redraw()
                return

            inferred = self._find_bobail_dest_for_origin((r, c))
            if inferred is not None:
                self.push_history()
                old_b = self.game.bobail_position
                new_b = inferred
                self.selection["phase"] = "token_origin" # type: ignore
                self.selection["bobail_dest"] = new_b # type: ignore
                self.selection["token_origin"] = (r, c) # type: ignore
                self.temp_bobail = (old_b, new_b)
                self.status_label.configure(text=f"Inferred bobail dest {new_b}; token origin {(r,c)} selected. Choose destination.")
                self.redraw()
                return

            self.status_label.configure(text="Select a highlighted bobail destination or a token origin associated with one.")
            return

        # Phase bobail: choose token origin or change bobail dest
        if self.selection["phase"] == "bobail":
            if self._is_token_origin_legal(self.selection["bobail_dest"], (r, c)):
                self.selection["phase"] = "token_origin" # type: ignore
                self.selection["token_origin"] = (r, c) # type: ignore
                self.status_label.configure(text=f"Token origin {(r,c)} selected. Now choose token destination.")
                self.redraw()
            else:
                if self._is_bobail_dest_legal((r, c)):
                    self.selection["bobail_dest"] = (r, c) # type: ignore
                    old_b = self.temp_bobail[0] if self.temp_bobail is not None else self.game.bobail_position
                    self.temp_bobail = (old_b, (r, c))
                    self.status_label.configure(text=f"Bobail dest changed to {(r,c)}. Choose token origin.")
                    self.redraw()
                else:
                    self.status_label.configure(text=f"{(r,c)} is not a legal token origin for that bobail destination.")
            return

        # Phase token_origin: change origin or pick destination to finalize
        if self.selection["phase"] == "token_origin":
            if self._is_token_origin_legal(self.selection["bobail_dest"], (r, c)):
                self.selection["token_origin"] = (r, c) # type: ignore
                self.status_label.configure(text=f"Token origin {(r,c)} selected. Now choose destination.")
                self.redraw()
                return
            if self._is_token_dest_legal(self.selection["bobail_dest"], self.selection["token_origin"], (r, c)):
                move = self._find_move(self.selection["bobail_dest"], self.selection["token_origin"], (r, c))
                if move is None:
                    self.status_label.configure(text="Move not found (unexpected). Selection cancelled.")
                    self.cancel_selection()
                    return
                # apply full move (user move)
                self.game.apply_move(move)
                self.game.current_player = 3 - self.game.current_player
                self.first_move_token_only = False
                # clear selection & temp visual
                self.selection = {"phase": None, "bobail_dest": None, "token_origin": None}
                self.temp_bobail = None
                self.status_label.configure(text="Move applied. Network will respond if available.")
                self.redraw()

                # check for winner after human move
                winner = self.game.check_victory()
                if type(winner) == int:
                    messagebox.showinfo("Game over", f"Winner: {self.player_color_name(winner)} (Player {winner}).\nNombre de noeuds visités par le MCTS: {len(self.tree.nodes)}.")
                    return

                # automatically let the network play the next ply (if network present)
                if self.network is not None and not self.is_human_turn():
                    self.push_history()
                    self.network_move()
                return
            self.status_label.configure(text=f"{(r,c)} is not a legal destination for the selected origin.")
            return

    def _get_bobail_dest_from_move(self, move):
        return move[0][1]

    def _get_token_old_new_from_move(self, move):
        return move[1][1], move[1][2]

    def _is_bobail_dest_legal(self, dest):
        legal = self.get_legal_moves()
        for m in legal:
            d = self._get_bobail_dest_from_move(m)
            if d == dest:
                return True
        return False

    def _is_token_origin_legal(self, bobail_dest, origin):
        legal = self.get_legal_moves()
        for m in legal:
            d = self._get_bobail_dest_from_move(m)
            if d != bobail_dest:
                continue
            old, _ = self._get_token_old_new_from_move(m)
            if old == origin:
                return True
        return False

    def _is_token_dest_legal(self, bobail_dest, origin, dest):
        legal = self.get_legal_moves()
        for m in legal:
            d = self._get_bobail_dest_from_move(m)
            if d != bobail_dest:
                continue
            old, new = self._get_token_old_new_from_move(m)
            if old == origin and new == dest:
                return True
        return False

    def _find_move(self, bobail_dest, origin, dest):
        legal = self.get_legal_moves()
        for m in legal:
            d = self._get_bobail_dest_from_move(m)
            if d != bobail_dest:
                continue
            old, new = self._get_token_old_new_from_move(m)
            if old == origin and new == dest:
                return m
        return None

    def choose_move(self, legal_moves, search_game: BobailGame | None = None):
        mode = self.mode_var.get()
        if not self.network or not legal_moves:
            return None, None
        game = search_game if search_game is not None else self.game

        inp = transform_input_data(game)
        x = torch.from_numpy(inp).float().to(self.device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            out = self.network(x)
        pi_logits = out[0]
        pi = F.softmax(pi_logits, dim=1)
        pi = pi.reshape(-1, 8, self.board_size, self.board_size, 8)[0].cpu().numpy()

        # MCTS mode: single canonical call/signature
        if mode == "mcts":
            visualizer = MCTSVisualizer3D(hash(game)) if self.show_mcts_var.get() else None
            legal_moves, probs, _, _ = self.tree.run(game, legal_moves, show=visualizer)
            if isinstance(probs, np.ndarray) and probs.size == len(legal_moves):
                idx = int(np.argmax(probs))
                return idx, legal_moves[idx]

        if mode == "policy":
            probs = np.zeros(len(legal_moves), dtype=np.float32)
            for (i, i0, i1, i2, i3) in adapt_moves_pov(game, legal_moves):
                probs[i] = pi[i0, i1, i2, i3]
            if probs.sum() > 0:
                idx = int(np.argmax(probs))
                return idx, legal_moves[idx]

        # value mode: evaluate successor states and pick best v
        if mode == "value":
            best_v = -float("inf")
            best_idx = None
            for i, m in enumerate(legal_moves):
                rev = game.apply_move(m)
                game.current_player = 3 - game.current_player
                x2 = transform_input_data(game)
                x2_t = torch.from_numpy(x2).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out2 = self.network(x2_t)
                v2 = out2[1]
                v2 = float(v2.cpu().numpy()[0][0]) if hasattr(v2, "cpu") else float(v2) #type: ignore
                v2 = -v2
                if v2 > best_v:
                    best_v = v2
                    best_idx = i
                game.current_player = 3 - game.current_player
                game.apply_move(rev)
            if best_idx is not None:
                return best_idx, legal_moves[best_idx]

        idx = int(np.random.randint(0, len(legal_moves)))
        return idx, legal_moves[idx]

    def network_move(self):
        search_game = self.game
        is_balanced_first_move = (
            self.first_move_token_only
            and self.game.current_player == self.player_for_color(self.starting_color)
        )
        if is_balanced_first_move:
            search_game, legal_moves = self.game.balanced_first_move_legal_moves()
        else:
            legal_moves = self.get_legal_moves()
        if not legal_moves:
            # nothing to play
            return
        idx, move = self.choose_move(legal_moves, search_game)
        if move is None:
            return
        # apply move
        self.game.apply_move(move)
        self.game.current_player = 3 - self.game.current_player
        self.first_move_token_only = False
        self.selection = {"phase": None, "bobail_dest": None, "token_origin": None}
        self.temp_bobail = None
        self.status_label.configure(text="Network played. Your turn.")
        self.redraw()
        winner = self.game.check_victory()
        if type(winner) == int:
            messagebox.showinfo("Game over", f"Winner: {self.player_color_name(winner)} (Player {winner})")

    def help_play(self):
        if self.is_first_human_starting_move():
            self.status_label.configure(text="Help is disabled for the first token-only move.")
            return
        self.push_history()
        self.status_label.configure(text=f"Network thinking as {self.player_color_name(self.game.current_player)}...")
        self.redraw()
        self.network_move()

    def undo(self):
        if not self.history:
            messagebox.showinfo("Undo", "No previous move to undo.")
            return
        self.game, self.first_move_token_only = self.history.pop()
        self.selection = {"phase": None, "bobail_dest": None, "token_origin": None}
        self.temp_bobail = None
        self.redraw()

    def cancel_selection(self):
        if self.selection["phase"] in ("bobail", "token_origin", "token_only_origin") and self.history:
            self.game, self.first_move_token_only = self.history.pop()
        self.selection = {"phase": None, "bobail_dest": None, "token_origin": None}
        self.temp_bobail = None
        self.redraw()

    def new_game(self, confirm: bool = True):
        if confirm and not messagebox.askyesno("New game", "Start new game?"):
            return
        self.game = BobailGame()
        self.game.current_player = self.player_for_color(self.starting_color)
        self.history = []
        self.selection = {"phase": None, "bobail_dest": None, "token_origin": None}
        self.temp_bobail = None
        self.first_move_token_only = True
        self.setup_current_color_var.set(self.color_for_player(self.game.current_player))
        self.tree = MCTS(self.network, n_simulations=SELF_PLAY_N_SIMULATIONS, tau=0, max_depth=80, c_puct=MCTS_C_PUCT, device=self.device, use_dirichlet_noise=False)
        self.redraw()
        self.root.after(100, self.play_network_turn_if_needed)

if __name__ == "__main__":
    root = Tk()
    root.title("Bobail")
    network = _Network.load(MODEL_PATH)

    app = GuiPlay(root, game=BobailGame(), network=network, device="cpu")
    root.mainloop()

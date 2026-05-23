import csv
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


METRICS_DIR = Path("metrics")


def now_seconds() -> float:
    return time.time()


def entropy(values: np.ndarray) -> float:
    probs = values.astype(np.float64)
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs = probs / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def numeric_summary(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def aggregate_numeric(records: list[dict[str, Any]], prefix: str = "") -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                grouped.setdefault(key, []).append(float(value))

    summary: dict[str, float] = {}
    for key, values in grouped.items():
        stats = numeric_summary(values)
        for stat_name, stat_value in stats.items():
            summary[f"{prefix}{key}_{stat_name}"] = stat_value
    return summary


class MetricsLogger:
    def __init__(self, run_name: str, directory: str | os.PathLike[str] = METRICS_DIR) -> None:
        self.run_name = run_name
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.directory / f"{run_name}.csv"
        self._records: list[dict[str, Any]] | None = None
        self._fieldnames: set[str] = set()

    def append(self, phase: str, metrics: dict[str, Any]) -> dict[str, Any]:
        record = {
            "timestamp": time.time(),
            "phase": phase,
            **metrics,
        }
        self.append_many([record])
        return record

    def append_many(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []
        current_records = self.load()
        old_fieldnames = set(self._fieldnames)
        new_fieldnames = old_fieldnames | {key for record in records for key in record.keys()}
        current_records.extend(records)
        self._records = current_records
        self._fieldnames = new_fieldnames

        if not self.csv_path.exists() or new_fieldnames != old_fieldnames:
            self._rewrite_csv(current_records)
            return records

        fieldnames = sorted(new_fieldnames)
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writerows(records)
        return records

    def load(self) -> list[dict[str, Any]]:
        if self._records is not None:
            return self._records
        if not self.csv_path.exists():
            self._records = []
            self._fieldnames = set()
            return []
        with self.csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            self._fieldnames = set(reader.fieldnames or [])
            self._records = [self._parse_record(row) for row in reader]
            return self._records

    def save_table(self, filename: str, rows: list[dict[str, Any]]) -> Path:
        path = self.directory / filename
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def plot_training(self, filename: str | None = None) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        all_records = self.load()
        records = [record for record in all_records if record.get("phase") == "training_epoch"]
        accuracy_records = [record for record in all_records if record.get("phase") == "accuracy"]
        filename = filename or f"{self.run_name}_training.png"
        output_path = self.directory / filename
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        if not records and not accuracy_records:
            axes[0, 0].text(0.5, 0.5, "No training metrics", ha="center", va="center")
            for ax in axes.flatten():
                ax.axis("off")
            fig.savefig(output_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            return output_path

        if records:
            x = np.arange(1, len(records) + 1)

            ax = axes[0, 0]
            self._plot_metric(ax, x, records, "train_loss", "Train")
            self._plot_metric(ax, x, records, "val_loss", "Validation")
            self._draw_episode_boundaries(ax, records)
            ax.set_title("Loss totale")
            ax.set_xlabel("Epoch globale")
            ax.set_ylabel("Loss")
            self._start_y_axis_at_zero(ax)
            self._start_x_axis_at_zero(ax)
            ax.grid(True, alpha=0.25)
            self._show_legend_if_present(ax)

            ax = axes[0, 1]
            self._plot_metric(ax, x, records, "train_policy_loss", "Train policy")
            self._plot_metric(ax, x, records, "val_policy_loss", "Val policy")
            self._plot_metric(ax, x, records, "train_value_loss", "Train value")
            self._plot_metric(ax, x, records, "val_value_loss", "Val value")
            self._draw_episode_boundaries(ax, records)
            ax.set_title("Loss policy / value")
            ax.set_xlabel("Epoch globale")
            ax.set_ylabel("Loss")
            self._start_y_axis_at_zero(ax)
            self._start_x_axis_at_zero(ax)
            ax.grid(True, alpha=0.25)
            self._show_legend_if_present(ax)

            ax = axes[1, 0]
            ax.axis("off")
        else:
            for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
                ax.axis("off")

        ax = axes[1, 1]
        if accuracy_records:
            x_acc = np.array([float(record["episode"]) for record in accuracy_records], dtype=np.float64)
            self._plot_metric(ax, x_acc, accuracy_records, "train_top1", "Train top1")
            self._plot_metric(ax, x_acc, accuracy_records, "val_top1", "Val top1")
            self._plot_metric(ax, x_acc, accuracy_records, "train_top5", "Train top5")
            self._plot_metric(ax, x_acc, accuracy_records, "val_top5", "Val top5")
            ax.set_title("Accuracy policy")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0.0, 1.02)
            self._start_x_axis_at_zero(ax)
            ax.grid(True, alpha=0.25)
            self._show_legend_if_present(ax)
        else:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        return output_path

    def plot_selfplay(self, filename: str | None = None) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        all_records = self.load()
        records = [record for record in all_records if record.get("phase") == "self_play_generation"]
        dataset_records = [record for record in all_records if record.get("phase") == "dataset_stats"]
        filename = filename or f"{self.run_name}_selfplay.png"
        output_path = self.directory / filename
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        if not records and not dataset_records:
            for ax in axes.flatten():
                ax.text(0.5, 0.5, "No self-play metrics", ha="center", va="center")
                ax.axis("off")
            fig.savefig(output_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            return output_path

        if dataset_records:
            x_dataset = np.array([float(record["episode"]) for record in dataset_records], dtype=np.float64)
            ax = axes[0, 0]
            self._plot_metric(ax, x_dataset, dataset_records, "raw_samples", "Bruts")
            self._plot_metric(ax, x_dataset, dataset_records, "dedup_samples", "Après dédup")
            ax.set_title("Samples")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Samples")
            self._start_y_axis_at_zero(ax)
            self._start_x_axis_at_zero(ax)
            ax.grid(True, alpha=0.25)
            self._show_legend_if_present(ax)

            ax = axes[1, 0]
            self._plot_metric(ax, x_dataset, dataset_records, "y_pi_entropy_mean", "Entropie y_pi")
            ax.set_title("Entropie policy")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Entropie moyenne")
            self._start_y_axis_at_zero(ax)
            self._start_x_axis_at_zero(ax)
            ax.grid(True, alpha=0.25)
            self._show_legend_if_present(ax)

            ax = axes[1, 1]
            self._plot_metric(ax, x_dataset, dataset_records, "y_v_mean", "Moyenne y_v")
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
            ax.set_title("Valeur cible")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Moyenne")
            ax.set_ylim(-1.0, 1.0)
            self._start_x_axis_at_zero(ax)
            ax.grid(True, alpha=0.25)
            self._show_legend_if_present(ax)
        else:
            axes[0, 0].axis("off")
            axes[1, 0].axis("off")
            axes[1, 1].axis("off")

        if records:
            x = np.array([float(record["episode"]) for record in records], dtype=np.float64)
            ax = axes[0, 1]
            self._plot_metric(ax, x, records, "avg_moves_per_game", "Coups moyens")
            ax.set_title("Longueur des parties")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Coups")
            self._start_y_axis_at_zero(ax)
            self._start_x_axis_at_zero(ax)
            ax.grid(True, alpha=0.25)
            self._show_legend_if_present(ax)

            ax_depth = ax.twinx()
            self._plot_metric(ax_depth, x, records, "game_mcts_max_depth_mean_mean", "Profondeur MCTS", linestyle=":")
            ax_depth.set_ylabel("Profondeur")
            self._start_x_axis_at_zero(ax_depth)
            self._show_legend_if_present(ax_depth)
        else:
            axes[0, 1].axis("off")

        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        return output_path

    def plot_tournament_matrix(
        self,
        rows: list[str],
        columns: list[str],
        matrix: np.ndarray,
        filename: str = "tournoi_matrix.png",
        cell_labels: np.ndarray | None = None,
    ) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.pyplot as plt

        output_path = self.directory / filename
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(columns)), max(5, 1.0 * len(rows))))
        cmap = LinearSegmentedColormap.from_list("green_score", ["#c62828", "#ffffff", "#2e7d32"])
        cmap.set_bad("#e0e0e0")
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(columns)), labels=columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(rows)), labels=rows)
        ax.set_xlabel("Rouge (joueur 2)")
        ax.set_ylabel("Vert (joueur 1, commence)")
        ax.set_title("Tournoi - victoires du vert")
        for i in range(len(rows)):
            for j in range(len(columns)):
                value = matrix[i, j]
                if np.isnan(value):
                    label = ""
                elif cell_labels is not None:
                    label_value = cell_labels[i, j]
                    label = "" if np.isnan(label_value) else f"{int(label_value)}"
                else:
                    label = f"{value:.2f}"
                ax.text(j, i, label, ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, label="Score relatif vert")
        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        return output_path

    def plot(self, filename: str | None = None) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        records = self.load()
        filename = filename or f"{self.run_name}.png"
        output_path = self.directory / filename
        numeric_keys = sorted({
            key
            for record in records
            for key, value in record.items()
            if isinstance(value, (int, float)) and key != "timestamp"
        })
        if not numeric_keys:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "No numeric metrics", ha="center", va="center")
            ax.axis("off")
            fig.savefig(output_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            return output_path
        cols = 2
        rows = math.ceil(len(numeric_keys) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(12, max(3, 2.6 * rows)), squeeze=False)
        for ax, key in zip(axes.flatten(), numeric_keys):
            xs = []
            ys = []
            for i, record in enumerate(records):
                value = record.get(key)
                if isinstance(value, (int, float)):
                    xs.append(i)
                    ys.append(value)
            ax.plot(xs, ys, marker=".", linewidth=1)
            ax.set_title(key)
            self._start_y_axis_at_zero(ax)
            ax.grid(True, alpha=0.25)
        for ax in axes.flatten()[len(numeric_keys):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        return output_path

    def _rewrite_csv(self, records: list[dict[str, Any]] | None = None) -> None:
        records = self.load() if records is None else records
        if not records:
            return
        fieldnames = sorted({key for record in records for key in record.keys()})
        self._records = records
        self._fieldnames = set(fieldnames)
        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record)

    @staticmethod
    def _parse_record(row: dict[str, str]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                parsed[key] = ""
                continue
            try:
                number = float(value)
            except ValueError:
                parsed[key] = value
                continue
            parsed[key] = int(number) if number.is_integer() else number
        return parsed

    @staticmethod
    def _plot_metric(
        ax: Any,
        x: np.ndarray,
        records: list[dict[str, Any]],
        key: str,
        label: str,
        linestyle: str = "-",
    ) -> None:
        values = [record.get(key) for record in records]
        points = [(xi, value) for xi, value in zip(x, values) if isinstance(value, (int, float))]
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker=".", linewidth=1.5, linestyle=linestyle, label=label)

    @staticmethod
    def _start_y_axis_at_zero(ax: Any) -> None:
        _, top = ax.get_ylim()
        ax.set_ylim(bottom=0.0, top=top if top > 0.0 else 1.0)

    @staticmethod
    def _start_x_axis_at_zero(ax: Any) -> None:
        _, right = ax.get_xlim()
        ax.set_xlim(left=0.0, right=right if right > 0.0 else 1.0)

    @staticmethod
    def _show_legend_if_present(ax: Any) -> None:
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend()

    @staticmethod
    def _draw_episode_boundaries(ax: Any, records: list[dict[str, Any]]) -> None:
        previous_episode = records[0].get("episode")
        for index, record in enumerate(records[1:], start=2):
            episode = record.get("episode")
            if episode != previous_episode:
                ax.axvline(index - 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.45)
                ax.text(index - 0.45, ax.get_ylim()[1], f"it. {episode}", rotation=90, va="top", fontsize=8)
            previous_episode = episode

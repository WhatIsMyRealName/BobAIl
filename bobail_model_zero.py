import json
import math
import os
import tempfile
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from bobail_config import TORCH_INFERENCE_THREADS, TORCH_INTEROP_THREADS
from bobail_metrics import MetricsLogger

_TORCH_INFERENCE_THREADS_CONFIGURED = False


def configure_torch_inference_threads() -> None:
    global _TORCH_INFERENCE_THREADS_CONFIGURED
    if _TORCH_INFERENCE_THREADS_CONFIGURED:
        return
    torch.set_num_threads(TORCH_INFERENCE_THREADS)
    torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    _TORCH_INFERENCE_THREADS_CONFIGURED = True


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.activation(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.activation(y + residual)


class PolicyHead(nn.Module):
    def __init__(self, channels: int, board_size: int) -> None:
        super().__init__()
        self.board_size = board_size
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 8 * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        logits = self.head(x)
        logits = logits.view(batch_size, 8, 8, self.board_size, self.board_size)
        logits = logits.permute(0, 1, 3, 4, 2).contiguous()
        return logits.view(batch_size, 8 * self.board_size * self.board_size * 8)


class ValueHead(nn.Module):
    def __init__(self, channels: int, board_size: int) -> None:
        super().__init__()
        hidden_size = 128
        self.head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(32 * board_size * board_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class _Network(nn.Module):
    def __init__(
        self,
        board_size: int = 5,
        num_channels: int = 64,
        num_res_blocks: int = 8,
        value_loss_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.value_loss_weight = value_loss_weight
        self.logs: list[float] = []
        self.logs_test: list[float] = []
        self.training_metrics: list[dict[str, float]] = []
        self.device_str = "cpu"

        self.stem = ConvBlock(3, num_channels)
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )
        self.pi_head = PolicyHead(num_channels, board_size)
        self.value_head = ValueHead(num_channels, board_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

        final_policy = self.pi_head.head[-1]
        final_value = self.value_head.head[-2]
        if isinstance(final_policy, nn.Conv2d):
            nn.init.zeros_(final_policy.weight)
            nn.init.zeros_(final_policy.bias)
        if isinstance(final_value, nn.Linear):
            nn.init.zeros_(final_value.weight)
            nn.init.zeros_(final_value.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        features = self.stem(x)
        features = self.residual_tower(features)
        return self.pi_head(features), self.value_head(features)

    @staticmethod
    def load(base_path: str) -> "_Network":
        if not os.path.exists(base_path):
            raise FileNotFoundError(base_path)
        model = _Network()
        model.load_state_dict(torch.load(base_path, map_location="cpu"))

        meta_path = base_path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta: dict[str, Any] = json.load(fh)
            for key in ["tau", "learning_rate", "logs", "logs_test", "training_metrics", "value_loss_weight"]:
                if key in meta:
                    setattr(model, key, meta[key])
        return model

    def save(self, base_path: str) -> None:
        os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
        torch.save(self.state_dict(), base_path)
        meta = {
            "logs": self.logs,
            "logs_test": self.logs_test,
            "training_metrics": self.training_metrics,
            "value_loss_weight": self.value_loss_weight,
        }
        for key in ["tau", "learning_rate"]:
            if hasattr(self, key):
                meta[key] = getattr(self, key)
        with open(base_path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    def export(self) -> None:
        self.eval()

    def summary(self, theory: float | None = None) -> None:
        import matplotlib.pyplot as plt

        plt.plot(self.logs, marker=".", linestyle="-", color="b", label="Erreur train")
        plt.plot(self.logs_test, marker=".", linestyle="-", color="r", label="Erreur validation")
        if theory is not None:
            plt.hlines(theory, 0, len(self.logs) - 1, colors="g", linestyles="dashed", label="Erreur theorique aleatoire")
        plt.title("Erreur au cours du temps (epochs)", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Erreur (UA)", fontsize=12)
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.legend(loc="best")
        plt.show()

    def fit3(
        self,
        x_train: np.ndarray | torch.Tensor,
        y_train: tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor],
        x_val: np.ndarray | torch.Tensor | None = None,
        y_val: tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor] | None = None,
        epochs: int = 20,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        device: str = "cpu",
        metrics_logger: MetricsLogger | None = None,
        episode: int = 0,
        early_stopping_patience: int = 2,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 1,
        min_learning_rate: float = 3e-5,
    ) -> None:
        if early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be strictly positive")
        if not (0.0 < lr_scheduler_factor < 1.0):
            raise ValueError("lr_scheduler_factor must be between 0 and 1")
        if lr_scheduler_patience < 0:
            raise ValueError("lr_scheduler_patience must be positive or zero")
        if min_learning_rate <= 0:
            raise ValueError("min_learning_rate must be strictly positive")

        self.device_str = device
        self.to(device)

        y_pi_train, y_v_train = y_train
        train_loader = self._make_loader(x_train, y_pi_train, y_v_train, batch_size, shuffle=True)
        val_loader = None
        if x_val is not None and y_val is not None:
            y_pi_val, y_v_val = y_val
            val_loader = self._make_loader(x_val, y_pi_val, y_v_val, batch_size, shuffle=False)

        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=min_learning_rate,
        )
        temp_best = tempfile.NamedTemporaryFile(
            prefix="bobail_fit3_best_",
            suffix=".pth",
            dir=os.getcwd(),
            delete=False,
        )
        best_model_path = temp_best.name
        temp_best.close()
        best_val_loss = math.inf
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.train()
            running_loss = 0.0
            running_policy_loss = 0.0
            running_value_loss = 0.0
            n_samples = 0

            for xb, ypi, yv in train_loader:
                xb = xb.to(device)
                ypi = ypi.to(device)
                yv = yv.to(device)

                optimizer.zero_grad(set_to_none=True)
                pi_logits, v = self(xb)
                policy_loss = F.kl_div(F.log_softmax(pi_logits, dim=1), ypi, reduction="batchmean")
                value_loss = F.mse_loss(v, yv)
                loss = policy_loss + self.value_loss_weight * value_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                batch_size_actual = pi_logits.size(0)
                running_loss += loss.item() * batch_size_actual
                running_policy_loss += policy_loss.item() * batch_size_actual
                running_value_loss += value_loss.item() * batch_size_actual
                n_samples += batch_size_actual

            train_loss = running_loss / n_samples
            train_policy_loss = running_policy_loss / n_samples
            train_value_loss = running_value_loss / n_samples
            self.logs.append(train_loss)

            val_loss = None
            val_policy_loss = None
            val_value_loss = None
            improved_validation = False
            if val_loader is not None:
                val_loss, val_policy_loss, val_value_loss = self._evaluate_loss(val_loader, device)
                self.logs_test.append(val_loss)
                improved_validation = val_loss < best_val_loss - 1e-6
                if improved_validation:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    torch.save(self.state_dict(), best_model_path)
                else:
                    epochs_without_improvement += 1

            elapsed = time.time() - t0
            metrics = {
                "episode": int(episode),
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_policy_loss": float(train_policy_loss),
                "train_value_loss": float(train_value_loss),
            }
            if val_loss is not None and val_policy_loss is not None and val_value_loss is not None:
                metrics.update({
                    "val_loss": float(val_loss),
                    "val_policy_loss": float(val_policy_loss),
                    "val_value_loss": float(val_value_loss),
                })
            self.training_metrics.append(metrics)
            if metrics_logger is not None:
                metrics_logger.append("training_epoch", metrics)

            if val_loss is None:
                print(
                    f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.6f} "
                    f"- policy: {train_policy_loss:.6f} - value: {train_value_loss:.6f} "
                    f"- time: {elapsed:.2f}s"
                )
            else:
                print(
                    f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.6f} "
                    f"- val_loss: {val_loss:.6f} - time: {elapsed:.2f}s"
                )
                print(
                    f"  Train policy/value: {train_policy_loss:.6f}/{train_value_loss:.6f} "
                    f"| Val policy/value: {val_policy_loss:.6f}/{val_value_loss:.6f}"
                )
                scheduler.step(val_loss)
                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"Early stopping: no validation improvement for "
                        f"{early_stopping_patience} epochs. Restoring epoch {best_epoch}."
                    )
                    break

        if val_loader is not None:
            self.load_state_dict(torch.load(best_model_path, map_location=device))
        os.remove(best_model_path)

    def _evaluate_loss(self, loader: DataLoader, device: str) -> tuple[float, float, float]:
        self.eval()
        running_loss = 0.0
        running_policy_loss = 0.0
        running_value_loss = 0.0
        n_samples = 0

        with torch.inference_mode():
            for xb, ypi, yv in loader:
                xb = xb.to(device)
                ypi = ypi.to(device)
                yv = yv.to(device)
                pi_logits, v = self(xb)
                policy_loss = F.kl_div(F.log_softmax(pi_logits, dim=1), ypi, reduction="batchmean")
                value_loss = F.mse_loss(v, yv)
                loss = policy_loss + self.value_loss_weight * value_loss
                batch_size_actual = pi_logits.size(0)
                running_loss += loss.item() * batch_size_actual
                running_policy_loss += policy_loss.item() * batch_size_actual
                running_value_loss += value_loss.item() * batch_size_actual
                n_samples += batch_size_actual

        return (
            running_loss / n_samples,
            running_policy_loss / n_samples,
            running_value_loss / n_samples,
        )

    @staticmethod
    def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        return value.detach().cpu().numpy()

    @staticmethod
    def _make_loader(
        x: np.ndarray | torch.Tensor,
        y_pi: np.ndarray | torch.Tensor,
        y_v: np.ndarray | torch.Tensor,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        x_t = torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x.float()
        y_pi_t = torch.from_numpy(y_pi).float() if isinstance(y_pi, np.ndarray) else y_pi.float()
        y_v_t = torch.from_numpy(y_v).float() if isinstance(y_v, np.ndarray) else y_v.float()
        return DataLoader(TensorDataset(x_t, y_pi_t, y_v_t), batch_size=batch_size, shuffle=shuffle)


def debug_network_shapes() -> None:
    test_data = torch.zeros((1, 3, 5, 5), dtype=torch.float32)
    test_data[0, 0, 4, :] = 1.0
    test_data[0, 1, 2, 2] = 1.0
    test_data[0, 2, 0, :] = 1.0

    net = _Network()
    net.eval()
    with torch.no_grad():
        pi, v = net(test_data)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Policy shape: {tuple(pi.shape)}")
    print(f"Value shape: {tuple(v.shape)}")
    print(f"Initial value: {v.item():.4f}")


if __name__ == "__main__":
    debug_network_shapes()

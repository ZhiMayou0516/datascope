from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None


@dataclass
class DeepTrainConfig:
    model_type: str = "CNN"
    task_type: str = "classification"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-3
    test_size: float = 0.2
    random_state: int = 42
    max_samples: int = 512


class SimpleCNN2D(nn.Module):
    def __init__(self, in_channels: int, output_dim: int, task_type: str):
        super().__init__()
        self.task_type = task_type
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, x):
        return self.head(self.features(x))


class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, task_type: str, cell: str = "RNN"):
        super().__init__()
        self.task_type = task_type
        rnn_cls = nn.LSTM if cell.upper() == "LSTM" else nn.RNN
        self.rnn = rnn_cls(input_dim, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_dim))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class TinyViT(nn.Module):
    def __init__(self, in_channels: int, output_dim: int, task_type: str, patch_size: int = 4, embed_dim: int = 64):
        super().__init__()
        self.task_type = task_type
        self.patch = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, output_dim))

    def forward(self, x):
        patches = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])


def torch_available() -> bool:
    return torch is not None


def _prepare_image_tensor(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X).astype("float32")
    if X.ndim == 3:  # N,H,W
        X = X[:, None, :, :]
    elif X.ndim == 4:
        # N,H,W,C -> N,C,H,W if channel-last
        if X.shape[-1] in (1, 3, 4):
            X = np.transpose(X, (0, 3, 1, 2))
        elif X.shape[1] not in (1, 3, 4):
            X = X[:, :1, :, :]
    else:
        raise ValueError("CNN/ViT 需要 X 为 N×H×W 或 N×H×W×C/N×C×H×W。")
    mean, std = np.nanmean(X), np.nanstd(X)
    X = (X - mean) / (std + 1e-6)
    return np.nan_to_num(X)


def _prepare_sequence_tensor(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X).astype("float32")
    if X.ndim != 3:
        raise ValueError("RNN/LSTM 需要 X 为 N×T×F。")
    mean, std = np.nanmean(X), np.nanstd(X)
    X = (X - mean) / (std + 1e-6)
    return np.nan_to_num(X)


def model_summary(model_type: str, input_shape: tuple[int, ...], task_type: str, output_dim: int) -> pd.DataFrame:
    if not torch_available():
        return pd.DataFrame([{"item": "status", "value": "torch not installed"}])
    if model_type in {"CNN", "ViT"}:
        channels = 1 if len(input_shape) == 3 else (input_shape[-1] if input_shape[-1] in (1, 3, 4) else input_shape[1])
        model = SimpleCNN2D(channels, output_dim, task_type) if model_type == "CNN" else TinyViT(channels, output_dim, task_type)
    else:
        input_dim = input_shape[-1]
        model = SimpleRNNModel(input_dim, output_dim, task_type, cell=model_type)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pd.DataFrame(
        [
            {"item": "model_type", "value": model_type},
            {"item": "input_shape", "value": str(input_shape)},
            {"item": "task_type", "value": task_type},
            {"item": "output_dim", "value": output_dim},
            {"item": "total_params", "value": int(total_params)},
            {"item": "trainable_params", "value": int(trainable_params)},
        ]
    )


def train_deep_baseline(X: np.ndarray, y: np.ndarray, config: DeepTrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not torch_available():
        raise RuntimeError("torch 未安装，无法训练深度学习模型。请运行 pip install torch。")
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if len(X) != len(y):
        raise ValueError("X 与 y 的样本数不一致。")
    if len(X) < 10:
        raise ValueError("深度学习 demo 至少需要 10 个样本。")
    if len(X) > config.max_samples:
        rng = np.random.default_rng(config.random_state)
        idx = rng.choice(len(X), size=config.max_samples, replace=False)
        X, y = X[idx], y[idx]

    if config.model_type in {"CNN", "ViT"}:
        X_tensor_np = _prepare_image_tensor(X)
        in_channels = X_tensor_np.shape[1]
        model = SimpleCNN2D(in_channels, 1, config.task_type) if config.model_type == "CNN" else TinyViT(in_channels, 1, config.task_type)
    else:
        X_tensor_np = _prepare_sequence_tensor(X)
        model = SimpleRNNModel(X_tensor_np.shape[-1], 1, config.task_type, cell=config.model_type)

    label_encoder = None
    if config.task_type == "classification":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y.astype(str))
        output_dim = len(label_encoder.classes_)
        if config.model_type in {"CNN", "ViT"}:
            model = SimpleCNN2D(X_tensor_np.shape[1], output_dim, config.task_type) if config.model_type == "CNN" else TinyViT(X_tensor_np.shape[1], output_dim, config.task_type)
        else:
            model = SimpleRNNModel(X_tensor_np.shape[-1], output_dim, config.task_type, cell=config.model_type)
        y_tensor_np = y_encoded.astype("int64")
        stratify = y_encoded if pd.Series(y_encoded).value_counts().min() >= 2 else None
    else:
        y_tensor_np = y.astype("float32").reshape(-1, 1)
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor_np, y_tensor_np, test_size=config.test_size, random_state=config.random_state, stratify=stratify
    )
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss() if config.task_type == "classification" else nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    history = []
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb.long() if config.task_type == "classification" else yb.float())
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else None})

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test).to(device)).cpu().numpy()
    if config.task_type == "classification":
        pred_idx = logits.argmax(axis=1)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred_idx)),
            "f1_weighted": float(f1_score(y_test, pred_idx, average="weighted", zero_division=0)),
        }
        if label_encoder is not None:
            pred_label = label_encoder.inverse_transform(pred_idx)
            true_label = label_encoder.inverse_transform(y_test.astype(int))
        else:
            pred_label, true_label = pred_idx, y_test
        pred_df = pd.DataFrame({"y_true": true_label, "y_pred": pred_label})
    else:
        pred = logits.reshape(-1)
        truth = y_test.reshape(-1)
        metrics = {"mae": float(mean_absolute_error(truth, pred)), "r2": float(r2_score(truth, pred))}
        pred_df = pd.DataFrame({"y_true": truth, "y_pred": pred})
    if history:
        metrics["final_train_loss"] = history[-1].get("train_loss")
    metrics_df = pd.DataFrame([metrics])
    return metrics_df, pred_df

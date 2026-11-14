"""定义用于 MNIST 分类的 MLP 和 CNN 模型。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from mnist_nn.constants import INPUT_SIZE, NUM_CLASSES

MODEL_NAMES = ("mlp", "cnn")


class MLP(nn.Module):
    """两层隐藏层的全连接网络。

    结构：Flatten -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear
    输入为 [batch, 1, 28, 28]，输出为 [batch, 10] 的 logits。
    """

    def __init__(self, hidden_size: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        if hidden_size < 2:
            raise ValueError("hidden_size 至少为 2。")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 区间。")

        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.flatten(x))


class CNN(nn.Module):
    """两层卷积 + 两次最大池化的简单卷积网络。

    结构：
        Conv(1->32, 3x3) -> ReLU -> MaxPool(2)
        Conv(32->64, 3x3) -> ReLU -> MaxPool(2)
        Flatten -> Linear(64*7*7 -> 128) -> ReLU -> Dropout -> Linear(128 -> 10)

    输入为 [batch, 1, 28, 28]，输出为 [batch, 10] 的 logits。
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 区间。")

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(
    name: str = "mlp", hidden_size: int = 256, dropout: float = 0.2
) -> nn.Module:
    """根据名字创建模型；检查点里保存的名字会用于恢复对应结构。"""
    if name == "mlp":
        return MLP(hidden_size=hidden_size, dropout=dropout)
    if name == "cnn":
        return CNN(dropout=dropout)
    raise ValueError(f"未知模型：{name}，可选值为 {', '.join(MODEL_NAMES)}。")


def load_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """从检查点恢复模型，返回处于 eval 模式的模型和检查点元数据。"""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型文件：{path}，请先运行训练脚本。")

    payload = torch.load(path, map_location=device, weights_only=True)
    model_name = payload.get("model_name", "mlp")
    model = build_model(model_name, **payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload

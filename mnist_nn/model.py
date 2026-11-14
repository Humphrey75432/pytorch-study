"""定义用于 MNIST 分类的简单多层感知机（MLP）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from mnist_nn.constants import INPUT_SIZE, NUM_CLASSES


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


def load_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[MLP, dict[str, Any]]:
    """从检查点恢复模型，返回处于 eval 模式的模型和检查点元数据。"""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型文件：{path}，请先运行训练脚本。")

    payload = torch.load(path, map_location=device, weights_only=True)
    model = MLP(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload

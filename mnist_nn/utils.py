"""设备选择、随机种子等通用工具。"""

from __future__ import annotations

import random

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """把命令行里的设备名解析成 torch.device。

    auto 的优先级为 CUDA -> Apple MPS -> CPU。
    """
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "当前环境没有可用的 CUDA，请改用 --device cpu 或 --device auto。"
            )
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError(
                "当前环境没有可用的 MPS，请改用 --device cpu 或 --device auto。"
            )
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """设置 Python 与 PyTorch 的随机种子，让每次运行尽量可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型的可训练参数数量。"""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

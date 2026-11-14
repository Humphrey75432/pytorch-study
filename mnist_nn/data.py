"""下载并加载 MNIST 数据集。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from mnist_nn.constants import MNIST_MEAN, MNIST_STD


@dataclass(frozen=True)
class DataConfig:
    """数据加载相关的配置。"""

    data_dir: Path = Path("data")
    batch_size: int = 64
    num_workers: int = 0
    val_size: int = 5_000
    seed: int = 42
    download: bool = True


@dataclass(frozen=True)
class TensorBatches:
    """一次性加载到内存/显存中的训练、验证和测试张量。"""

    train_images: torch.Tensor
    train_labels: torch.Tensor
    val_images: torch.Tensor
    val_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor


def build_transform() -> transforms.Compose:
    """把 PIL 图片转成张量，并按 MNIST 的均值/标准差标准化。"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


def build_datasets(config: DataConfig) -> tuple[Dataset, Dataset, Dataset]:
    """构建训练集、验证集和测试集。

    官方训练集有 60000 张图片。这里按固定随机种子切出 val_size 张作为验证集，
    测试集（10000 张）只在训练结束后使用，避免在训练过程中“偷看”测试集。
    """
    if not 0 < config.val_size < 60_000:
        raise ValueError("val_size 必须位于 1 到 59999 之间。")

    transform = build_transform()
    train_full = datasets.MNIST(
        root=config.data_dir,
        train=True,
        download=config.download,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=config.data_dir,
        train=False,
        download=config.download,
        transform=transform,
    )

    generator = torch.Generator().manual_seed(config.seed)
    train_set, val_set = random_split(
        train_full,
        [len(train_full) - config.val_size, config.val_size],
        generator=generator,
    )
    return train_set, val_set, test_set


def build_dataloaders(
    config: DataConfig, device: torch.device
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练 / 验证 / 测试 DataLoader。"""
    train_set, val_set, test_set = build_datasets(config)
    pin_memory = device.type == "cuda"
    common_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": config.num_workers > 0,
    }

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        **common_kwargs,
    )
    val_loader = DataLoader(val_set, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **common_kwargs)
    return train_loader, val_loader, test_loader


def build_test_loader(config: DataConfig, device: torch.device) -> DataLoader:
    """只加载测试集，供预测脚本使用。"""
    test_set = datasets.MNIST(
        root=config.data_dir,
        train=False,
        download=config.download,
        transform=build_transform(),
    )
    return DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )


def build_tensor_data(config: DataConfig, device: torch.device) -> TensorBatches:
    """把整个 MNIST 一次性搬到目标设备。

    MNIST 很小，float32 下全部数据约 240 MB，完全可以常驻显存。
    这样可以避免每个 batch 都经过 DataLoader、Python 预处理和 CPU->GPU 拷贝，
    配合较大的 batch size 能显著提高 GPU 利用率。
    """
    train_set, val_set, test_set = build_datasets(config)

    def materialize(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=0)
        images = torch.cat([batch[0] for batch in loader])
        labels = torch.cat([batch[1] for batch in loader])
        return images.to(device), labels.to(device)

    train_images, train_labels = materialize(train_set)
    val_images, val_labels = materialize(val_set)
    test_images, test_labels = materialize(test_set)
    return TensorBatches(
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

"""训练 MNIST 手写数字分类模型。"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from mnist_nn.data import (
    DataConfig,
    TensorBatches,
    build_dataloaders,
    build_tensor_data,
)
from mnist_nn.model import MODEL_NAMES, build_model, load_model
from mnist_nn.utils import count_parameters, resolve_device, set_seed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """训练一轮，返回平均损失和准确率。"""
    model.train()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), dtype=torch.long, device=device)
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.detach() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum()
        total_samples += batch_size

    return (total_loss / total_samples).item(), (total_correct / total_samples).item()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """在验证集或测试集上评估，返回平均损失和准确率。"""
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), dtype=torch.long, device=device)
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.detach() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum()
        total_samples += batch_size

    return (total_loss / total_samples).item(), (total_correct / total_samples).item()


def train_one_epoch_tensors(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    """在已经放到设备上的整块张量上训练一轮。"""
    model.train()
    num_samples = images.size(0)
    permutation = torch.randperm(num_samples, device=device)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), dtype=torch.long, device=device)

    for start in range(0, num_samples, batch_size):
        indices = permutation[start : start + batch_size]
        batch_images = images[indices]
        batch_labels = labels[indices]

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_images)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        batch = batch_labels.size(0)
        total_loss += loss.detach() * batch
        total_correct += (logits.argmax(dim=1) == batch_labels).sum()

    return (total_loss / num_samples).item(), (total_correct / num_samples).item()


@torch.inference_mode()
def evaluate_tensors(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[float, float]:
    """在已经放到设备上的整块张量上评估。"""
    model.eval()
    num_samples = images.size(0)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), dtype=torch.long, device=device)

    for start in range(0, num_samples, batch_size):
        batch_images = images[start : start + batch_size]
        batch_labels = labels[start : start + batch_size]

        logits = model(batch_images)
        loss = criterion(logits, batch_labels)

        batch = batch_labels.size(0)
        total_loss += loss.detach() * batch
        total_correct += (logits.argmax(dim=1) == batch_labels).sum()

    return (total_loss / num_samples).item(), (total_correct / num_samples).item()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    model_config: dict[str, float | int],
    metrics: dict[str, float | int],
) -> None:
    """保存模型参数和重建模型所需的配置。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "metrics": metrics,
        },
        path,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 MNIST 上训练一个简单的 PyTorch 神经网络。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="mlp",
        help="模型结构：mlp 或 cnn",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=256, help="MLP 第一层隐藏单元数"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout 概率")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="MNIST 数据目录"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="模型与训练记录输出目录",
    )
    parser.add_argument(
        "--val-size", type=int, default=5_000, help="从训练集切出的验证集大小"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader 工作进程数"
    )
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="把整个数据集一次性预加载到内存/显存（GPU 大 batch 训练更快）",
    )
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否自动下载 MNIST 数据集",
    )

    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs 必须大于等于 1")
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于等于 1")
    if args.lr <= 0:
        parser.error("--lr 必须大于 0")
    if args.hidden_size < 2:
        parser.error("--hidden-size 必须大于等于 2")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout 必须位于 [0, 1) 区间")
    if not 0 < args.val_size < 60_000:
        parser.error("--val-size 必须位于 1 到 59999 之间")
    if args.num_workers < 0:
        parser.error("--num-workers 不能为负数")
    return args


def _jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    try:
        device = resolve_device(args.device)
    except ValueError as exc:
        print(f"错误：{exc}")
        return 2

    data_config = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_size=args.val_size,
        seed=args.seed,
        download=args.download,
    )

    tensors: TensorBatches | None = None
    try:
        if args.preload:
            print(f"正在把 MNIST 预加载到 {device}（约 240 MB）...")
            preload_start = time.perf_counter()
            tensors = build_tensor_data(data_config, device)
            print(f"预加载完成，用时 {time.perf_counter() - preload_start:.2f}s")
            train_count = tensors.train_images.size(0)
            val_count = tensors.val_images.size(0)
            test_count = tensors.test_images.size(0)
        else:
            train_loader, val_loader, test_loader = build_dataloaders(
                data_config, device
            )
            train_count = len(train_loader.dataset)
            val_count = len(val_loader.dataset)
            test_count = len(test_loader.dataset)
    except RuntimeError as exc:
        print(f"错误：加载 MNIST 失败。{exc}")
        print("提示：首次运行需要联网下载数据集；也可以检查 --data-dir 是否已有数据。")
        return 1

    model = build_model(
        args.model, hidden_size=args.hidden_size, dropout=args.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_config: dict[str, float | int] = {"dropout": args.dropout}
    if args.model == "mlp":
        model_config["hidden_size"] = args.hidden_size

    checkpoint_path = args.output_dir / f"mnist_{args.model}.pt"
    history_path = args.output_dir / f"history_{args.model}.json"

    if args.model == "mlp":
        model_description = (
            f"MLP(hidden_size={args.hidden_size}, dropout={args.dropout})"
        )
    else:
        model_description = f"CNN(dropout={args.dropout})"

    print(f"设备        : {device}")
    print(
        "数据集      : "
        f"{args.data_dir}（训练 {train_count} / 验证 {val_count} / 测试 {test_count}）"
    )
    print(f"模型        : {model_description}")
    print(f"参数量      : {count_parameters(model):,}")
    print(
        f"训练配置    : epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}"
    )
    if args.preload and device.type == "cuda" and args.batch_size < 512:
        print("提示        : 预加载模式下，GPU 使用 --batch-size 1024 或更大通常更快。")
    print()

    best_val_accuracy = -1.0
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        if tensors is not None:
            train_loss, train_accuracy = train_one_epoch_tensors(
                model,
                tensors.train_images,
                tensors.train_labels,
                optimizer,
                criterion,
                device,
                args.batch_size,
            )
            val_loss, val_accuracy = evaluate_tensors(
                model, tensors.val_images, tensors.val_labels, criterion, device
            )
        else:
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - epoch_start

        improved = val_accuracy > best_val_accuracy
        if improved:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model,
                model_name=args.model,
                model_config=model_config,
                metrics={
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "seconds": elapsed,
            }
        )
        marker = "  <- best" if improved else ""
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.2%} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.2%} | "
            f"{elapsed:.1f}s{marker}"
        )

    total_seconds = time.perf_counter() - training_start
    print()
    print(
        f"训练完成，用时 {total_seconds:.1f}s。最佳验证准确率 {best_val_accuracy:.2%}（epoch {best_epoch}）"
    )

    # 用验证集上表现最好的检查点做最终测试，避免报告最后一轮的结果。
    model, _ = load_model(checkpoint_path, device)
    if tensors is not None:
        test_loss, test_accuracy = evaluate_tensors(
            model, tensors.test_images, tensors.test_labels, criterion, device
        )
    else:
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"测试集      : loss {test_loss:.4f}, accuracy {test_accuracy:.2%}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(
            {
                "args": _jsonable_args(args),
                "epochs": history,
                "best": {"epoch": best_epoch, "val_accuracy": best_val_accuracy},
                "test": {"loss": test_loss, "accuracy": test_accuracy},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"模型        : {checkpoint_path}")
    print(f"训练记录    : {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

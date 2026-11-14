"""可视化训练曲线、预测错误样本和第一层权重。"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import PercentFormatter
from torch import nn
from torch.utils.data import DataLoader

from mnist_nn.constants import MNIST_MEAN, MNIST_STD
from mnist_nn.data import DataConfig, build_test_loader
from mnist_nn.model import CNN, MLP, load_model
from mnist_nn.utils import resolve_device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="可视化 MNIST 训练曲线、错误样本和第一层权重。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/mnist_cnn.pt"),
        help="模型检查点路径",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("artifacts/history_cnn.json"),
        help="训练历史 JSON 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/plots"),
        help="可视化图片输出目录",
    )
    parser.add_argument("--count", type=int, default=16, help="最多展示多少张错误样本")
    parser.add_argument(
        "--max-filters", type=int, default=32, help="最多展示多少个第一层卷积核"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="MNIST 数据目录"
    )
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否自动下载 MNIST 数据集",
    )

    args = parser.parse_args(argv)
    if not 1 <= args.count <= 64:
        parser.error("--count 必须位于 1 到 64 之间")
    if not 1 <= args.max_filters <= 64:
        parser.error("--max-filters 必须位于 1 到 64 之间")
    return args


def plot_training_curves(history_path: str | Path, output_path: str | Path) -> None:
    """从训练历史 JSON 画 loss 和 accuracy 曲线。"""
    payload = json.loads(Path(history_path).read_text(encoding="utf-8"))
    records = payload.get("epochs", [])
    if not records:
        raise ValueError("训练历史里没有 epochs 记录。")

    epoch_numbers = [int(record["epoch"]) for record in records]
    train_loss = [float(record["train_loss"]) for record in records]
    val_loss = [float(record["val_loss"]) for record in records]
    train_accuracy = [float(record["train_accuracy"]) for record in records]
    val_accuracy = [float(record["val_accuracy"]) for record in records]

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))

    loss_axis.plot(epoch_numbers, train_loss, marker="o", label="train")
    loss_axis.plot(epoch_numbers, val_loss, marker="o", label="validation")
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()
    loss_axis.grid(alpha=0.3)

    accuracy_axis.plot(epoch_numbers, train_accuracy, marker="o", label="train")
    accuracy_axis.plot(epoch_numbers, val_accuracy, marker="o", label="validation")
    best_epoch = payload.get("best", {}).get("epoch")
    if best_epoch is not None:
        accuracy_axis.axvline(
            int(best_epoch),
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"best epoch {int(best_epoch)}",
        )
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    accuracy_axis.legend()
    accuracy_axis.grid(alpha=0.3)

    test_accuracy = payload.get("test", {}).get("accuracy")
    title = "Training history"
    if test_accuracy is not None:
        title += f" (test accuracy {float(test_accuracy):.2%})"
    figure.suptitle(title)
    figure.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _denormalize(image: torch.Tensor) -> torch.Tensor:
    return image.squeeze(0).mul(MNIST_STD).add(MNIST_MEAN).clamp(0.0, 1.0)


def plot_misclassified(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: str | Path,
    max_items: int = 16,
) -> int:
    """找出测试集中预测错误的样本并画成网格，返回实际展示数量。"""
    model.eval()
    images_list: list[torch.Tensor] = []
    labels_list: list[int] = []
    predictions_list: list[int] = []
    confidences_list: list[float] = []

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = logits.softmax(dim=1)
            confidences, predictions = probabilities.max(dim=1)
            predictions = predictions.cpu()
            confidences = confidences.cpu()
            wrong = predictions != labels

            if not wrong.any():
                continue

            indices = wrong.nonzero(as_tuple=False).flatten()
            for index in indices.tolist():
                if len(images_list) >= max_items:
                    break
                images_list.append(images[index].clone())
                labels_list.append(int(labels[index]))
                predictions_list.append(int(predictions[index]))
                confidences_list.append(float(confidences[index]))

            if len(images_list) >= max_items:
                break

    shown = len(images_list)
    if shown == 0:
        return 0

    columns = min(4, shown)
    rows = math.ceil(shown / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.4, rows * 2.8))
    axes_list = [axes] if shown == 1 else list(axes.flat)

    for axis, image, truth, prediction, confidence in zip(
        axes_list, images_list, labels_list, predictions_list, confidences_list
    ):
        axis.imshow(_denormalize(image), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(
            f"true: {truth}  pred: {prediction}\nconfidence: {confidence:.1%}",
            color="darkred",
            fontsize=9,
        )
        axis.axis("off")

    for axis in axes_list[shown:]:
        axis.axis("off")

    figure.suptitle(f"Misclassified test samples (showing {shown})")
    figure.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return shown


def plot_first_layer_weights(
    model: nn.Module, output_path: str | Path, max_filters: int = 32
) -> int:
    """可视化第一层权重：CNN 是卷积核，MLP 是像素模板。返回展示数量。"""
    if isinstance(model, CNN):
        weight = model.features[0].weight.detach().cpu()
        count = min(weight.size(0), max_filters)
        filters = [weight[index, 0].numpy() for index in range(count)]
        title = "CNN first convolution filters"
    elif isinstance(model, MLP):
        weight = model.layers[0].weight.detach().cpu()
        count = min(weight.size(0), max_filters)
        filters = [weight[index].reshape(28, 28).numpy() for index in range(count)]
        title = "MLP first layer weights (pixel templates)"
    else:
        raise TypeError(f"不支持的模型类型：{type(model).__name__}")

    columns = min(8, count)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.6, rows * 1.7))
    axes_list = [axes] if count == 1 else list(axes.flat)

    for index, (axis, weights) in enumerate(zip(axes_list, filters)):
        limit = max(float(abs(weights).max()), 1e-8)
        axis.imshow(
            weights,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(f"#{index}", fontsize=8)
        axis.axis("off")

    for axis in axes_list[count:]:
        axis.axis("off")

    figure.suptitle(f"{title} (showing {count})")
    figure.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return count


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    if args.history.is_file():
        curves_path = args.output_dir / "training_curves.png"
        plot_training_curves(args.history, curves_path)
        print(f"训练曲线    : {curves_path}")
        generated += 1
    else:
        print(f"跳过训练曲线：找不到 {args.history}")

    if args.checkpoint.is_file():
        try:
            device = resolve_device(args.device)
            model, _ = load_model(args.checkpoint, device)
        except (ValueError, RuntimeError) as exc:
            print(f"错误：加载模型失败。{exc}")
            return 1

        data_config = DataConfig(
            data_dir=args.data_dir,
            batch_size=256,
            download=args.download,
        )
        test_loader = build_test_loader(data_config, device)
        misclassified_path = args.output_dir / "misclassified.png"
        shown = plot_misclassified(
            model, test_loader, device, misclassified_path, max_items=args.count
        )
        if shown:
            print(f"错误样本    : {misclassified_path}（{shown} 张）")
            generated += 1
        else:
            print("错误样本    : 测试集中没有找到错误样本")

        weights_path = args.output_dir / "first_layer_weights.png"
        count = plot_first_layer_weights(
            model, weights_path, max_filters=args.max_filters
        )
        print(f"第一层权重  : {weights_path}（{count} 个）")
        generated += 1
    else:
        print(f"跳过模型可视化：找不到 {args.checkpoint}")

    if generated == 0:
        print("没有生成任何图片，请检查 --history 和 --checkpoint 路径。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

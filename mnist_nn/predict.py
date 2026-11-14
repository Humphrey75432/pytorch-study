"""加载训练好的模型，对 MNIST 测试集图片进行预测并保存结果图。"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from mnist_nn.constants import MNIST_MEAN, MNIST_STD
from mnist_nn.data import DataConfig, build_test_loader
from mnist_nn.model import load_model
from mnist_nn.utils import resolve_device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用训练好的 MLP 预测 MNIST 测试图片。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/mnist_mlp.pt"),
        help="模型检查点路径",
    )
    parser.add_argument("--count", type=int, default=16, help="预测图片数量（1-64）")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/predictions.png"),
        help="预测结果图保存路径",
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
    return args


def _to_pil(image: torch.Tensor) -> Image.Image:
    """把标准化后的单通道图片还原成 0-255 的 PIL 图片。"""
    array = (
        image.squeeze(0)
        .mul(MNIST_STD)
        .add(MNIST_MEAN)
        .clamp(0.0, 1.0)
        .mul(255)
        .to(torch.uint8)
        .numpy()
    )
    return Image.fromarray(array, mode="L")


def save_prediction_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    path: Path,
) -> None:
    """把图片、真实标签和预测标签拼成一张网格图。"""
    count = len(images)
    columns = min(4, count)
    rows = math.ceil(count / columns)
    scale = 4
    cell = 28 * scale
    padding = 10
    label_height = 26

    width = columns * cell + (columns + 1) * padding
    height = rows * (cell + label_height) + (rows + 1) * padding
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for index in range(count):
        row, column = divmod(index, columns)
        x = padding + column * (cell + padding)
        y = padding + row * (cell + label_height + padding)

        image = _to_pil(images[index]).resize((cell, cell), Image.Resampling.NEAREST)
        canvas.paste(image.convert("RGB"), (x, y))

        truth = int(labels[index])
        prediction = int(predictions[index])
        correct = truth == prediction
        text = f"T:{truth} P:{prediction}"
        draw.text((x, y + cell + 3), text, fill="green" if correct else "red")
        draw.text((x, y + cell + 14), f"{float(confidences[index]):.0%}", fill="gray")

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        device = resolve_device(args.device)
    except ValueError as exc:
        print(f"错误：{exc}")
        return 2

    try:
        model, payload = load_model(args.checkpoint, device)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"错误：{exc}")
        return 1

    config = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.count,
        download=args.download,
    )
    test_loader = build_test_loader(config, device)
    images, labels = next(iter(test_loader))

    with torch.inference_mode():
        logits = model(images.to(device))
        probabilities = logits.softmax(dim=1)
        confidences, predictions = probabilities.max(dim=1)

    predictions = predictions.cpu()
    confidences = confidences.cpu()

    metrics = payload.get("metrics", {})
    if metrics:
        print(
            f"模型：{args.checkpoint}（epoch {metrics.get('epoch')}，"
            f"验证准确率 {float(metrics.get('val_accuracy', 0.0)):.2%}）"
        )
    print(f"设备：{device}\n")

    correct = 0
    for index in range(len(images)):
        truth = int(labels[index])
        prediction = int(predictions[index])
        confidence = float(confidences[index])
        matched = truth == prediction
        correct += int(matched)
        print(
            f"[{index:02d}] 真实 {truth} | 预测 {prediction} | "
            f"置信度 {confidence:.2%} | {'正确' if matched else '错误'}"
        )

    print(f"\n本批准确率：{correct}/{len(images)} = {correct / len(images):.2%}")
    save_prediction_grid(images, labels, predictions, confidences, args.output)
    print(f"结果图：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

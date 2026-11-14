"""对比不同 Dropout 概率对 MNIST 模型准确率的影响。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mnist_nn.model import MODEL_NAMES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练多个不同 Dropout 概率的模型并对比准确率。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="cnn",
        help="要对比的模型结构",
    )
    parser.add_argument(
        "--dropouts",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.5],
        help="要对比的 Dropout 概率列表",
    )
    parser.add_argument("--epochs", type=int, default=5, help="每个模型的训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="MNIST 数据目录"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/dropout_compare"),
        help="实验结果输出目录",
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
    args.dropouts = list(dict.fromkeys(args.dropouts))
    if not args.dropouts:
        parser.error("--dropouts 至少需要一个值")
    if any(not 0.0 <= dropout < 1.0 for dropout in args.dropouts):
        parser.error("--dropouts 中的值必须位于 [0, 1) 区间")
    if args.epochs < 1:
        parser.error("--epochs 必须大于等于 1")
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于等于 1")
    if args.lr <= 0:
        parser.error("--lr 必须大于 0")
    return args


def plot_dropout_results(
    rows: list[dict[str, float]],
    output_path: str | Path,
    model: str,
    epochs: int,
) -> None:
    """画 Dropout 与验证/测试准确率的关系图。"""
    dropouts = [row["dropout"] for row in rows]
    validation = [row["best_val_accuracy"] * 100 for row in rows]
    test = [row["test_accuracy"] * 100 for row in rows]

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(dropouts, validation, marker="o", label="best validation")
    axis.plot(dropouts, test, marker="s", label="test")
    axis.margins(y=0.15)
    for x, y in zip(dropouts, test):
        axis.annotate(
            f"{y:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    axis.set_title(f"{model.upper()} Dropout comparison ({epochs} epochs)")
    axis.set_xlabel("Dropout")
    axis.set_ylabel("Accuracy (%)")
    axis.grid(alpha=0.3)
    axis.legend()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    total = len(args.dropouts)

    for index, dropout in enumerate(args.dropouts, start=1):
        run_dir = args.output_dir / f"dropout_{str(dropout).replace('.', '_')}"
        print(f"\n[{index}/{total}] 训练 {args.model.upper()}，dropout={dropout}")

        command = [
            sys.executable,
            "-m",
            "mnist_nn.train",
            "--model",
            args.model,
            "--dropout",
            str(dropout),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(run_dir),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        if not args.download:
            command.append("--no-download")

        start = time.perf_counter()
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"训练失败，退出码：{exc.returncode}")
            return exc.returncode or 1
        elapsed = time.perf_counter() - start

        history_path = run_dir / f"history_{args.model}.json"
        if not history_path.is_file():
            print(f"错误：找不到训练记录 {history_path}")
            return 1
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        best_val = float(payload.get("best", {}).get("val_accuracy", 0.0))
        test_accuracy = float(payload.get("test", {}).get("accuracy", 0.0))
        epochs_completed = int(payload.get("epochs_completed", len(payload["epochs"])))

        rows.append(
            {
                "dropout": dropout,
                "best_val_accuracy": best_val,
                "test_accuracy": test_accuracy,
                "epochs_completed": float(epochs_completed),
                "total_seconds": float(elapsed),
            }
        )
        print(
            f"  验证准确率 {best_val:.2%}，测试准确率 {test_accuracy:.2%}，用时 {elapsed:.1f}s"
        )

    csv_path = args.output_dir / "dropout_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = args.output_dir / "dropout_comparison.png"
    plot_dropout_results(rows, plot_path, args.model, args.epochs)

    print("\n结果汇总：")
    print(f"{'dropout':>8} {'best val':>10} {'test':>10} {'seconds':>10}")
    for row in rows:
        print(
            f"{row['dropout']:>8.2f} {row['best_val_accuracy']:>9.2%} "
            f"{row['test_accuracy']:>9.2%} {row['total_seconds']:>9.1f}s"
        )
    print(f"\nCSV     : {csv_path}")
    print(f"对比图  : {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

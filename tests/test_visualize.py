"""可视化函数的单元测试，不需要下载 MNIST。"""

import json
from pathlib import Path

from mnist_nn.model import CNN, MLP
from mnist_nn.visualize import plot_first_layer_weights, plot_training_curves


def test_plot_training_curves(tmp_path: Path) -> None:
    history = {
        "epochs": [
            {
                "epoch": 1,
                "train_loss": 0.5,
                "val_loss": 0.4,
                "train_accuracy": 0.80,
                "val_accuracy": 0.85,
            },
            {
                "epoch": 2,
                "train_loss": 0.3,
                "val_loss": 0.25,
                "train_accuracy": 0.90,
                "val_accuracy": 0.92,
            },
        ],
        "best": {"epoch": 2, "val_accuracy": 0.92},
        "test": {"loss": 0.2, "accuracy": 0.93},
    }
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps(history), encoding="utf-8")
    output_path = tmp_path / "curves.png"

    plot_training_curves(history_path, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_cnn_first_layer_weights(tmp_path: Path) -> None:
    output_path = tmp_path / "cnn_weights.png"

    shown = plot_first_layer_weights(CNN(dropout=0.0), output_path, max_filters=4)

    assert shown == 4
    assert output_path.is_file()


def test_plot_mlp_first_layer_weights(tmp_path: Path) -> None:
    output_path = tmp_path / "mlp_weights.png"

    shown = plot_first_layer_weights(
        MLP(hidden_size=8, dropout=0.0), output_path, max_filters=3
    )

    assert shown == 3
    assert output_path.is_file()

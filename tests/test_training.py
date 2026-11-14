"""训练技巧相关函数的单元测试。"""

import pytest
import torch
from torch import nn
from torchvision import transforms

from mnist_nn.compare_dropout import plot_dropout_results
from mnist_nn.data import build_transform
from mnist_nn.train import EarlyStopping, build_scheduler, parse_args


def test_early_stopping_triggers_after_patience() -> None:
    stopper = EarlyStopping(patience=2)

    assert stopper.update(improved=True) is False
    assert stopper.update(improved=False) is False
    assert stopper.update(improved=False) is True


def test_early_stopping_resets_on_improvement() -> None:
    stopper = EarlyStopping(patience=2)

    stopper.update(improved=False)
    stopper.update(improved=False)
    stopper.update(improved=True)

    assert stopper.counter == 0
    assert stopper.should_stop is False


def test_build_transform_without_augmentation() -> None:
    transform = build_transform(augment=False)

    assert not any(
        isinstance(step, transforms.RandomAffine) for step in transform.transforms
    )


def test_build_transform_with_augmentation() -> None:
    transform = build_transform(augment=True)

    assert any(
        isinstance(step, transforms.RandomAffine) for step in transform.transforms
    )


def test_preload_and_augment_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--preload", "--augment"])


def test_build_scheduler_none() -> None:
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)

    assert build_scheduler("none", optimizer, epochs=5) is None


def test_build_scheduler_cosine() -> None:
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)

    scheduler = build_scheduler("cosine", optimizer, epochs=5)

    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_plot_dropout_results(tmp_path) -> None:
    rows = [
        {
            "dropout": 0.0,
            "best_val_accuracy": 0.98,
            "test_accuracy": 0.981,
        },
        {
            "dropout": 0.2,
            "best_val_accuracy": 0.99,
            "test_accuracy": 0.991,
        },
    ]
    output_path = tmp_path / "dropout.png"

    plot_dropout_results(rows, output_path, model="cnn", epochs=5)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0

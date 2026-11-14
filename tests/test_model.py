"""模型的基本单元测试，不依赖 MNIST 数据下载。"""

import pytest
import torch

from mnist_nn.constants import INPUT_SIZE, NUM_CLASSES
from mnist_nn.model import MLP


def test_output_shape() -> None:
    model = MLP(hidden_size=32, dropout=0.0)
    logits = model(torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, NUM_CLASSES)


def test_forward_and_backward() -> None:
    model = MLP(hidden_size=16, dropout=0.0)
    logits = model(torch.randn(2, 1, 28, 28))
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (2, NUM_CLASSES)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_output_size_matches_input_definition() -> None:
    assert INPUT_SIZE == 28 * 28


def test_invalid_hidden_size() -> None:
    with pytest.raises(ValueError):
        MLP(hidden_size=1)


def test_invalid_dropout() -> None:
    with pytest.raises(ValueError):
        MLP(dropout=1.0)

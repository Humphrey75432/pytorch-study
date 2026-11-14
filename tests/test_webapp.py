"""Web 应用预处理和推理的单元测试。"""

import base64
import io

import pytest
import torch
from PIL import Image, ImageDraw

from mnist_nn.webapp import predict_tensor, preprocess_data_url


def _to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," + encoded


def _digit_image() -> Image.Image:
    image = Image.new("L", (280, 280), 0)
    draw = ImageDraw.Draw(image)
    draw.line([(90, 220), (140, 60), (195, 220)], fill=255, width=18)
    return image


class _FixedModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(images.size(0), 10, device=images.device)
        logits[:, 3] = 10.0
        return logits


def test_preprocess_data_url() -> None:
    tensor, preview = preprocess_data_url(_to_data_url(_digit_image()))

    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype == torch.float32
    assert tensor.abs().sum() > 0
    assert preview.startswith("data:image/png;base64,")


def test_blank_canvas_is_rejected() -> None:
    blank = Image.new("L", (280, 280), 0)

    with pytest.raises(ValueError, match="画板是空的"):
        preprocess_data_url(_to_data_url(blank))


def test_predict_tensor_returns_top3() -> None:
    tensor = torch.zeros(1, 1, 28, 28)
    result = predict_tensor(_FixedModel(), tensor, torch.device("cpu"))

    assert result["digit"] == 3
    assert result["confidence"] > 0.99
    top3 = result["top3"]
    assert isinstance(top3, list)
    assert len(top3) == 3
    assert top3[0]["digit"] == 3

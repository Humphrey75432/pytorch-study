"""FastAPI + HTML Canvas 手写数字识别页面。"""

from __future__ import annotations

import argparse
import base64
import html
import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps
from pydantic import BaseModel
from torchvision import transforms

from mnist_nn.constants import MNIST_MEAN, MNIST_STD
from mnist_nn.model import load_model
from mnist_nn.utils import resolve_device

INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MNIST 手写数字识别</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #141b31;
      --panel-2: #1b2440;
      --text: #eef2ff;
      --muted: #96a2c6;
      --accent: #64d2ff;
      --accent-2: #7c9cff;
      --border: #2a3658;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(900px 600px at 8% -10%, rgba(100, 210, 255, .18), transparent 60%),
        radial-gradient(800px 600px at 100% 10%, rgba(124, 156, 255, .16), transparent 55%),
        var(--bg);
      font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 28px 18px;
    }
    .shell { width: min(980px, 100%); }
    header { margin-bottom: 20px; }
    .eyebrow {
      letter-spacing: .18em;
      text-transform: uppercase;
      font-size: 12px;
      color: var(--accent);
      font-weight: 700;
    }
    h1 {
      font-size: clamp(26px, 4vw, 42px);
      line-height: 1.15;
      margin: 8px 0 6px;
    }
    .subtitle { margin: 0; color: var(--muted); line-height: 1.6; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-top: 14px;
      padding: 7px 11px;
      border: 1px solid var(--border);
      border-radius: 999px;
      color: #c8d5ff;
      background: rgba(20, 27, 49, .72);
      font-size: 12px;
    }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: #50e3a4; box-shadow: 0 0 12px #50e3a4; }
    .card {
      background: linear-gradient(180deg, rgba(20, 27, 49, .96), rgba(15, 22, 40, .96));
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 20px;
      box-shadow: 0 24px 80px rgba(0, 0, 0, .35);
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) minmax(280px, .9fr);
      gap: 20px;
      align-items: stretch;
    }
    .canvas-panel, .result-panel {
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--panel);
      padding: 16px;
    }
    .canvas-wrap {
      position: relative;
      width: 100%;
      max-width: 360px;
      margin: 0 auto;
      aspect-ratio: 1;
      border-radius: 16px;
      overflow: hidden;
      background: #000;
      box-shadow: inset 0 0 0 1px #2a3658;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
      cursor: crosshair;
      touch-action: none;
    }
    .canvas-hint {
      text-align: center;
      color: var(--muted);
      font-size: 13px;
      margin: 12px 0 0;
    }
    .result-panel { display: flex; flex-direction: column; gap: 14px; }
    .result-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
    .result-head h2 { margin: 0; font-size: 15px; color: var(--muted); font-weight: 600; }
    .result-head span { color: var(--accent); font-size: 12px; }
    .digit-row { display: flex; align-items: center; gap: 16px; }
    .digit {
      width: 92px;
      height: 92px;
      display: grid;
      place-items: center;
      border-radius: 20px;
      background: linear-gradient(145deg, #1d2a4d, #11182c);
      border: 1px solid #34436b;
      color: var(--accent);
      font-size: 52px;
      font-weight: 800;
      line-height: 1;
    }
    .confidence { color: var(--muted); font-size: 13px; line-height: 1.7; }
    .confidence strong { display: block; color: var(--text); font-size: 22px; }
    .preview {
      width: 96px;
      height: 96px;
      border-radius: 12px;
      border: 1px solid var(--border);
      image-rendering: pixelated;
      align-self: center;
      display: none;
      background: #000;
    }
    .bars { display: flex; flex-direction: column; gap: 8px; min-height: 96px; }
    .bar-row { display: grid; grid-template-columns: 18px 1fr 48px; gap: 8px; align-items: center; font-size: 12px; color: var(--muted); }
    .bar-row .num { color: var(--text); font-weight: 700; }
    .bar-track { height: 8px; border-radius: 999px; background: #0d1426; overflow: hidden; }
    .bar-fill { height: 100%; width: 0; border-radius: inherit; background: linear-gradient(90deg, var(--accent-2), var(--accent)); transition: width .25s ease; }
    .buttons { display: flex; gap: 10px; margin-top: auto; }
    button {
      flex: 1;
      border: 0;
      border-radius: 12px;
      padding: 12px 14px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      color: #06111f;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      transition: transform .15s ease, opacity .15s ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: .55; cursor: wait; transform: none; }
    button.secondary {
      background: #1b2440;
      color: var(--text);
      border: 1px solid var(--border);
    }
    .status { min-height: 18px; font-size: 12px; color: var(--muted); }
    .status.error { color: #ff8a9b; }
    footer { margin-top: 14px; text-align: center; color: var(--muted); font-size: 12px; }
    @media (max-width: 760px) {
      .layout { grid-template-columns: 1fr; }
      .canvas-wrap { max-width: 320px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="eyebrow">PyTorch · MNIST · FastAPI</div>
      <h1>手写数字识别</h1>
      <p class="subtitle">在黑色画板上写一个 0～9 的数字，模型会实时给出预测和置信度。</p>
      <div class="badge"><span class="dot"></span><span>__MODEL_INFO__</span></div>
    </header>

    <main class="card">
      <div class="layout">
        <section class="canvas-panel">
          <div class="canvas-wrap">
            <canvas id="canvas" width="280" height="280" aria-label="手写数字画板"></canvas>
          </div>
          <p class="canvas-hint">按住鼠标或手指书写，尽量把数字写在中间</p>
        </section>

        <section class="result-panel">
          <div class="result-head">
            <h2>模型判断</h2>
            <span>Top 3</span>
          </div>
          <div class="digit-row">
            <div id="digit" class="digit">?</div>
            <div class="confidence">
              置信度
              <strong id="confidence">—</strong>
            </div>
            <img id="preview" class="preview" alt="28×28 模型输入预览">
          </div>
          <div id="bars" class="bars"></div>
          <div class="buttons">
            <button id="clear" class="secondary" type="button">清空画板</button>
            <button id="predict" type="button">识别数字</button>
          </div>
          <div id="status" class="status"></div>
        </section>
      </div>
    </main>
    <footer>PyTorch 模型 + FastAPI 后端 · 开发学习项目</footer>
  </div>

  <script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const digitEl = document.getElementById('digit');
    const confidenceEl = document.getElementById('confidence');
    const barsEl = document.getElementById('bars');
    const previewEl = document.getElementById('preview');
    const statusEl = document.getElementById('status');
    const predictButton = document.getElementById('predict');

    function paintBackground() {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    function resetResult() {
      digitEl.textContent = '?';
      confidenceEl.textContent = '—';
      barsEl.innerHTML = '';
      previewEl.style.display = 'none';
      statusEl.textContent = '';
      statusEl.classList.remove('error');
    }

    function resetCanvas() {
      paintBackground();
      resetResult();
    }

    paintBackground();
    ctx.strokeStyle = '#fff';
    ctx.fillStyle = '#fff';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    let drawing = false;
    let lastPoint = null;

    function pointFromEvent(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * (canvas.width / rect.width),
        y: (event.clientY - rect.top) * (canvas.height / rect.height),
      };
    }

    function startStroke(point) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 9, 0, Math.PI * 2);
      ctx.fill();
      lastPoint = point;
    }

    canvas.addEventListener('pointerdown', (event) => {
      event.preventDefault();
      canvas.setPointerCapture(event.pointerId);
      drawing = true;
      resetResult();
      const point = pointFromEvent(event);
      startStroke(point);
    });

    canvas.addEventListener('pointermove', (event) => {
      if (!drawing || !lastPoint) return;
      event.preventDefault();
      const point = pointFromEvent(event);
      ctx.beginPath();
      ctx.moveTo(lastPoint.x, lastPoint.y);
      ctx.lineTo(point.x, point.y);
      ctx.stroke();
      lastPoint = point;
    });

    function endStroke(event) {
      if (!drawing) return;
      drawing = false;
      lastPoint = null;
      if (event && canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    }

    canvas.addEventListener('pointerup', endStroke);
    canvas.addEventListener('pointercancel', endStroke);

    function showPrediction(data) {
      digitEl.textContent = data.digit;
      confidenceEl.textContent = (data.confidence * 100).toFixed(1) + '%';
      previewEl.src = data.preview;
      previewEl.style.display = 'block';
      barsEl.innerHTML = '';

      data.top3.forEach((item) => {
        const row = document.createElement('div');
        row.className = 'bar-row';

        const number = document.createElement('span');
        number.className = 'num';
        number.textContent = item.digit;

        const track = document.createElement('div');
        track.className = 'bar-track';
        const fill = document.createElement('div');
        fill.className = 'bar-fill';
        fill.style.width = Math.max(2, item.probability * 100) + '%';
        track.appendChild(fill);

        const percent = document.createElement('span');
        percent.textContent = (item.probability * 100).toFixed(1) + '%';

        row.append(number, track, percent);
        barsEl.appendChild(row);
      });
    }

    async function predict() {
      predictButton.disabled = true;
      statusEl.textContent = '识别中…';
      statusEl.classList.remove('error');

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: canvas.toDataURL('image/png') }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || '识别失败');
        }
        showPrediction(data);
        statusEl.textContent = '识别完成';
      } catch (error) {
        statusEl.textContent = error.message || '识别失败，请重试';
        statusEl.classList.add('error');
      } finally {
        predictButton.disabled = false;
      }
    }

    predictButton.addEventListener('click', predict);
    document.getElementById('clear').addEventListener('click', resetCanvas);
  </script>
</body>
</html>
"""


class PredictRequest(BaseModel):
    image: str


def preprocess_data_url(data_url: str) -> tuple[torch.Tensor, str]:
    """把画板导出的 PNG data URL 转成模型输入和 28x28 预览图。"""
    if not data_url or not data_url.startswith("data:image/"):
        raise ValueError("无效的图片数据。")

    try:
        header, encoded = data_url.split(",", 1)
        if ";base64" not in header:
            raise ValueError("图片不是 base64 编码。")
        raw = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(raw)).convert("L")
    except Exception as exc:
        raise ValueError("无法解析画板图片。") from exc

    array = np.asarray(image, dtype=np.uint8)
    if array.mean() > 127:
        image = ImageOps.invert(image)
        array = 255 - array

    mask = array > 20
    if not mask.any():
        raise ValueError("画板是空的，请先写一个数字。")

    rows, columns = np.where(mask)
    image = image.crop(
        (
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        )
    )

    width, height = image.size
    scale = 20.0 / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(image, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))

    tensor = transforms.ToTensor()(canvas).unsqueeze(0)
    tensor = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))(tensor)

    preview = canvas.resize((112, 112), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    preview_url = (
        "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    )
    return tensor, preview_url


def predict_tensor(
    model: torch.nn.Module, tensor: torch.Tensor, device: torch.device
) -> dict[str, object]:
    """对单张 28x28 图片推理，返回 top-1 和 top-3 结果。"""
    with torch.inference_mode():
        logits = model(tensor.to(device))
        probabilities = logits.softmax(dim=1)[0]

    top_probabilities, top_digits = probabilities.topk(k=min(3, probabilities.numel()))
    return {
        "digit": int(top_digits[0].item()),
        "confidence": float(top_probabilities[0].item()),
        "top3": [
            {"digit": int(digit.item()), "probability": float(probability.item())}
            for digit, probability in zip(top_digits, top_probabilities)
        ],
    }


def create_app(checkpoint: str | Path, device_name: str = "auto") -> FastAPI:
    """加载检查点并创建 FastAPI 应用。"""
    checkpoint_path = Path(checkpoint)
    device = resolve_device(device_name)
    model, _ = load_model(checkpoint_path, device)
    model.eval()

    app = FastAPI(title="MNIST 手写数字识别")
    model_info = f"{model.__class__.__name__} · {device} · {checkpoint_path.name}"

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(
            INDEX_HTML.replace("__MODEL_INFO__", html.escape(model_info))
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model_info}

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, object]:
        try:
            tensor, preview = preprocess_data_url(request.image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = predict_tensor(model, tensor, device)
        result["preview"] = preview
        return result

    return app


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 MNIST 手写数字识别页面。")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/mnist_cnn.pt"),
        help="模型检查点路径",
    )
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")

    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须位于 1 到 65535 之间")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        app = create_app(args.checkpoint, args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}")
        print("提示：请先运行训练脚本生成 artifacts/mnist_cnn.pt。")
        return 1

    print(f"页面地址：http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

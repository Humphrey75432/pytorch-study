"""便捷入口：直接运行 `uv run main.py` 即可用默认参数训练模型。"""

from mnist_nn.train import main

if __name__ == "__main__":
    raise SystemExit(main())

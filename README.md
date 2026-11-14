# PyTorch MNIST 入门项目

一个使用 **PyTorch** 在 **MNIST** 手写数字数据集上训练 **MLP** 和 **CNN** 的小项目，包含：

- 支持 MLP（全连接）和 CNN（卷积）两种模型
- 自动下载并预处理 MNIST
- 训练集 / 验证集 / 测试集划分
- 训练、验证、测试完整流程
- 自动选择 CUDA / MPS / CPU
- 保存最佳模型与训练记录
- 加载模型预测，并生成带标签的预测结果图
- 基础单元测试

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 解释器、虚拟环境和依赖。

## 环境要求

- 已安装 `uv`（当前环境为 uv 0.13.6）
- Python 3.15（已由 uv 安装在 `.venv` 中）
- NVIDIA GPU（本机为 RTX 5080，驱动 617.42）

项目在 `pyproject.toml` 中把 `torch` / `torchvision` 指向 PyTorch 的 **cu132** 索引，首次 `uv sync` 会下载约 2 GB 的 CUDA 版本。如果换到没有 NVIDIA GPU 的机器，可以改用 CPU 版（见文末常见问题）。

## 快速开始

```powershell
# 1. 安装/同步依赖（首次执行，会下载 CUDA 版 PyTorch）
uv sync

# 2. 训练 MLP（默认 5 轮，自动选择 GPU）
uv run python -m mnist_nn.train

# 或者训练 CNN（推荐，准确率更高）
uv run python -m mnist_nn.train --model cnn --epochs 5 --batch-size 128
```

也可以直接运行便捷入口，效果与默认 MLP 相同：

```powershell
uv run main.py
```

首次运行会自动把 MNIST 下载到 `data/` 目录，约 11 MB。训练结束后会得到：

| 文件 | 说明 |
| --- | --- |
| `artifacts/mnist_mlp.pt` | MLP 在验证集上表现最好的检查点 |
| `artifacts/mnist_cnn.pt` | CNN 在验证集上表现最好的检查点 |
| `artifacts/history_mlp.json` | MLP 每轮损失、准确率、耗时及最终测试结果 |
| `artifacts/history_cnn.json` | CNN 每轮损失、准确率、耗时及最终测试结果 |

当前环境实测：

- MLP：默认 5 轮，测试集准确率约 **97.8%**
- CNN：5 轮、batch size 128，测试集准确率 **99.15%**

### 预测并生成结果图

```powershell
# MLP
uv run python -m mnist_nn.predict --count 16

# CNN
uv run python -m mnist_nn.predict --checkpoint artifacts/mnist_cnn.pt --count 16 --output artifacts/predictions_cnn.png
```

脚本会读取测试集前 16 张图片，打印真实标签、预测标签和置信度，并生成对应的 PNG 图片。

## GPU 说明

当前环境：

- GPU：NVIDIA GeForce RTX 5080（16 GB，compute capability 12.0）
- PyTorch：`2.15.0+cu132`
- CUDA 运行时：13.2
- 显卡驱动：617.42

验证 GPU 是否可用：

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

训练脚本默认 `--device auto`，检测到 CUDA 就会优先使用 GPU；也可以显式指定：

```powershell
uv run python -m mnist_nn.train --device cuda
```

**默认配置下为什么 MLP 的 GPU 没有更快？**

- 模型只有 235,146 个参数，batch size 为 64 时每一步的计算量很小
- 每个 batch 都要经过 DataLoader、Python 预处理和 CPU→GPU 拷贝，GPU 经常在等数据
- 对这个小型 MLP，CPU 的 3～4 秒/轮已经很快，默认 GPU 配置反而略慢
- 换成 CNN、更大的全连接层、更大的 batch，或者更复杂的数据集后，GPU 的优势才会明显

CNN 有 421,642 个参数和卷积层，GPU 利用率比 MLP 高一些。当前实测 CNN 5 轮、batch size 128 在 RTX 5080 上约 23.7 秒，测试准确率 **99.15%**。

**让 RTX 5080 跑满：预加载 + 大 batch**

项目提供了 `--preload` 选项：一次性把整个 MNIST（float32 约 240 MB）搬到显存，之后直接在 GPU 上切 batch，不再经过 DataLoader 和逐批拷贝。

```powershell
uv run python -m mnist_nn.train --preload --batch-size 1024 --epochs 10 --lr 0.002
```

实测结果：

| 配置 | 总耗时 | 测试准确率 |
| --- | --- | --- |
| MLP，CPU，默认 5 轮 | 约 18 秒 | 97.80% |
| MLP，GPU，默认 5 轮 | 约 29 秒 | 97.80% |
| MLP，GPU，`--preload` + batch 1024 + 10 轮 | 约 12.5 秒 | 97.89% |
| CNN，GPU，batch 128 + 5 轮 | 约 23.7 秒 | **99.15%** |

这 12.5 秒里，预加载约 5.8 秒，第一轮包含 CUDA 初始化约 6.2 秒；从第二轮开始每轮不到 0.1 秒。因此训练轮数越多，GPU 的优势越明显。若只是跑 5 轮，CPU 反而更省事。

为什么这样能快：

- 数据已经在显存里，GPU 不需要等 CPU 喂数据
- batch 从 64 提到 1024，每轮更新次数从约 860 次降到约 54 次，内核启动和 Python 循环开销大幅减少
- 为了弥补更新次数变少，把训练轮数从 5 提到 10，并适当提高学习率

TF32 和混合精度（AMP）对这个模型帮助有限，因为矩阵太小，收益会被数据搬运和启动开销抵消；`torch.compile` 对 10 轮这种短训练也不划算，编译时间可能超过训练本身。继续加大隐藏层或换更大的模型后，GPU 的优势会更明显。

## 项目结构

```text
pytorch-study/
├── main.py                  # 便捷训练入口
├── pyproject.toml           # 项目元数据与依赖
├── uv.lock                  # 依赖锁定文件
├── mnist_nn/
│   ├── __init__.py
│   ├── constants.py         # 图片尺寸、类别数、归一化参数
│   ├── data.py              # 下载、预处理、划分、DataLoader
│   ├── model.py             # MLP / CNN 模型与检查点加载
│   ├── train.py             # 训练 / 验证 / 测试流程
│   ├── predict.py           # 推理与预测结果可视化
│   └── utils.py             # 设备选择、随机种子等
└── tests/
    └── test_model.py        # 不依赖数据下载的单元测试
```

## 模型说明

项目通过 `--model` 选择模型，默认是 `mlp`。

### MLP：多层感知机

```text
输入 [B, 1, 28, 28]
  -> Flatten          -> [B, 784]
  -> Linear(784, 256) -> ReLU -> Dropout(0.2)
  -> Linear(256, 128) -> ReLU
  -> Linear(128, 10)  -> logits
```

MLP 会把整张图拍平成 784 个数字，因此丢失了“上下左右相邻”的空间结构。

### CNN：卷积神经网络

```text
输入 [B, 1, 28, 28]
  -> Conv2d(1, 32, 3x3, padding=1) -> ReLU -> MaxPool2d(2)   # 28x28 -> 14x14
  -> Conv2d(32, 64, 3x3, padding=1) -> ReLU -> MaxPool2d(2)  # 14x14 -> 7x7
  -> Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout(0.2)
  -> Linear(128, 10) -> logits
```

CNN 参数量为 **421,642**。卷积核在图像上滑动，能保留局部空间关系；池化层降低分辨率、扩大感受野。对图像任务来说，这通常比直接 Flatten 更合理，所以准确率也更高。

训练细节：

- 损失函数：`CrossEntropyLoss`
- 优化器：`Adam`，默认学习率 `1e-3`
- MLP 默认批大小：`64`
- CNN 建议批大小：`128`
- 输入标准化：均值 `0.1307`，标准差 `0.3081`
- 数据划分：训练集 55000 / 验证集 5000 / 测试集 10000
- 每轮在验证集上评估，只保存验证准确率最高的检查点；训练结束后再用该检查点测试一次

当前实测：MLP 5 轮约 **97.8%**；CNN 5 轮、batch size 128 约 **99.15%**。

## 常用参数

### 训练

```powershell
uv run python -m mnist_nn.train --help
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `mlp` | 模型结构：`mlp` / `cnn` |
| `--epochs` | `5` | 训练轮数 |
| `--batch-size` | `64` | 批大小 |
| `--lr` | `0.001` | Adam 学习率 |
| `--hidden-size` | `256` | MLP 第一层隐藏单元数（CNN 忽略） |
| `--dropout` | `0.2` | Dropout 概率 |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `--data-dir` | `data` | 数据集目录 |
| `--output-dir` | `artifacts` | 模型和记录输出目录 |
| `--val-size` | `5000` | 从训练集切出的验证集大小 |
| `--num-workers` | `0` | DataLoader 工作进程数 |
| `--preload` | 关闭 | 一次性把整个数据集加载到内存/显存，配合大 batch 加速 GPU |
| `--seed` | `42` | 随机种子 |
| `--no-download` | - | 不自动下载，仅使用本地已有数据 |

训练 MLP：

```powershell
uv run python -m mnist_nn.train --epochs 8 --hidden-size 512 --batch-size 128 --lr 0.0005
```

训练 CNN：

```powershell
uv run python -m mnist_nn.train --model cnn --epochs 5 --batch-size 128
```

### 预测

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--checkpoint` | `artifacts/mnist_mlp.pt` | 模型检查点，CNN 使用 `artifacts/mnist_cnn.pt` |
| `--count` | `16` | 预测图片数量（1～64） |
| `--output` | `artifacts/predictions.png` | 结果图路径 |
| `--device` | `auto` | 推理设备 |
| `--no-download` | - | 不自动下载数据 |

## 运行测试

测试只构造随机张量验证模型结构，不会下载 MNIST：

```powershell
uv run pytest -q
```

## 常见问题

**下载数据集失败**

训练脚本会通过 torchvision 自动下载数据。若网络受限，可手动准备 `data/MNIST/raw/` 下的四个原始文件：

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

然后加上 `--no-download` 运行。

**`--device cuda` 报错**

先检查当前 PyTorch 是否能识别显卡：

```powershell
uv run python -c "import torch; print(torch.cuda.is_available())"
```

输出 `True` 说明 GPU 可用。若为 `False`，通常是装成了 CPU 版 PyTorch，或者驱动版本过旧。本项目依赖的是 cu132 版本；更换机器时需要根据显卡驱动选择匹配的 CUDA 版本。

**换到没有 NVIDIA 显卡的机器**

先删除 `pyproject.toml` 中的 `[tool.uv.sources]` 段和 `[[tool.uv.index]]` 段，然后重新解析依赖：

```powershell
uv lock --upgrade-package torch --upgrade-package torchvision
uv sync
```

这样会从 PyPI 安装 CPU 版 PyTorch。不要在没有 GPU 的机器上强制 `--device cuda`。

**想清空训练结果重新开始**

删除 `artifacts/` 目录即可。MNIST 数据会保留在 `data/`，下次训练无需重新下载。

<div align="center">

<img src="https://dummyimage.com/820x160/111/fff&text=MyTorch" alt="MyTorch Banner" />

<h1>MyTorch</h1>
<p><b>一个基于 <code>NumPy</code> 从零实现的可阅读 / 可实验 / 可扩展的极简深度学习 & 自动求导框架</b></p>

<p>
<a href="#"> <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python"/> </a>
<a href="#license"> <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/> </a>
<a href="#roadmap"> <img src="https://img.shields.io/badge/Status-Experimental-orange" alt="Status"/> </a>
<a href="#faq"> <img src="https://img.shields.io/badge/Docs-Progress-blueviolet" alt="Docs"/> </a>
<a href="#benchmarks"> <img src="https://img.shields.io/badge/Perf-Edu%20Priority-lightgrey" alt="Perf"/> </a>
<a href="#-开发--贡献"> <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" alt="PRs"/> </a>
</p>

<p>
<b>已包含核心源码与实验脚本；可直接安装 & 复现实验。</b>
</p>

<sub>English summary at the bottom · 英文简介见文末</sub>

</div>

> 提示：本仓库定位教学与实验复现，优先保证“可读性 + 简洁性”，非性能竞速。欢迎基于此继续扩展。

---

## 🔗 快速导航 (Table of Contents)

- [✨ 特性概览](#-特性概览)
- [📂 目录结构](#-目录结构精简视图)
- [⚙️ 安装](#️-安装)
- [🚀 快速上手](#-快速上手)
- [🧩 核心设计](#-核心设计说明)
- [🏗️ 架构示意](#-架构示意)
- [📈 实验与结果](#-实验与结果概要)
- [🧪 Benchmarks](#benchmarks)
- [🛠️ 开发 / 贡献](#-开发--贡献)
- [🗺️ 路线图 / Roadmap](#-路线图--todo)
- [❓ FAQ](#-faq)
- [📜 许可](#-许可)
- [🙌 致谢](#-致谢)
- [👤 作者](#-作者)
- [English Brief](#english-brief)

---

## ✨ 特性概览

- 🔢 张量类 `MyTensor`：封装 `numpy.ndarray`，记录父节点与生成算子，支持梯度累积
- 🔄 自动求导：显式构建有向无环计算图 + 拓扑排序反向传播
- 🧮 常用算子：加 / 减 / 乘 / 矩阵乘 / Reshape / ReLU / Softmax / CrossEntropyLoss 等
- 🧠 模型模块化：`Module` / `Linear` / `MLP` / （可扩展 CNN 层）
- 🛠️ 优化器：SGD、Momentum、Nesterov、Adagrad、RMSProp、Adam（统一参数更新接口）
- 📦 数据封装：`NumpyDataset` + `DataLoader`（批生成、打乱、标准化）
- 🧪 实验脚本：学习率 / 批大小 / 优化器对比 / Hidden Size / 权重衰减
- 📈 日志记录：CSV + JSON + 最优权重持久化（`weights.npz` / `best_model.npz`）
- 🧾 报告产出：LaTeX 生成实验报告（`report/main.pdf`）

---

## 📂 目录结构（精简视图）

```text
mytorch/
├─ pyproject.toml        # 打包 & 依赖
├─ train.py              # 训练脚本（MLP on CIFAR-10）
├─ dataprocess.py        # 数据集读取与划分
├─ model/                # 额外模型（如 MLP）
├─ src/mytorch/          # 核心库源码
│  ├─ __init__.py        # 导出 MyTensor, DataLoader, NumpyDataset
│  ├─ mytensor.py        # 计算图与自动求导核心
│  ├─ nn/                # 模块、层、激活与损失
│  ├─ operation/         # 原子算子 / 反向规则
│  └─ mydata.py          # 数据管道
├─ logs/                 # 训练日志 (CSV/JSON)
├─ save/                 # 不同实验配置保存的权重/日志
└─ report/main.pdf       # 实验分析与可视化
```

---

## ⚙️ 安装

支持两种方式：源代码开发模式（推荐学习/调试）与构建后普通安装。

1. 克隆源码并以“可编辑”模式安装（便于实时修改调试）
```bash
git clone https://github.com/yourname/mytorch.git
cd mytorch
python -m pip install -e .
```
2. （可选）仅作为依赖使用（未来可发布到 PyPI 后替换为 `pip install mytorch`）
```bash
python -m pip install .
```
3. 开发依赖（静态检查 / 测试）
```bash
python -m pip install -e .[dev]
```

核心运行依赖仅 `numpy>=1.23`。其余在 `pyproject.toml` 的可选分组 `dev`。

数据集获取：
```bash
# 方法A: 使用脚本（若已提供）
python scripts/download.py

# 方法B: 手动下载 CIFAR-10 Python 版本 (cifar-10-python.tar.gz) 放到仓库根目录或 data/ 目录
```

环境最小需求：Python 3.9+，>=8GB 内存（大批量/多实验时更舒适）。GPU 非必需。

---

## 🚀 快速上手

### 1. 最小例子：张量前向与反向
创建一个最小的线性分类器并执行前向与反向：

```python
import numpy as np
from mytorch import MyTensor
from mytorch.nn.func import relu, matmul

x = MyTensor(np.random.randn(4, 3), requires_grad=True)
w = MyTensor(np.random.randn(3, 2), requires_grad=True)
b = MyTensor(np.zeros((2,)), requires_grad=True)

logits = matmul(x, w) + b  # 线性
act = relu(logits)         # 激活
loss = act.sum()           # 标量损失
loss.backward()            # 触发反向传播

print(w.grad)              # 查看梯度
```

### 2. 使用内置 `MLP` 训练 CIFAR‑10（摘自 `train.py`）

```python
from mytorch import DataLoader, NumpyDataset
from model.mlp import MLP
from dataprocess import get_data
from mytorch.nn import CrossEntropyLoss
from mytorch.nn.optim import SGD  # 若你的 optim 模块路径不同请调整

(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(path='cifar-10-python.tar.gz')
train_ds = NumpyDataset(X_train, y_train, dtype=np.float32, normalize=True)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, to_tensor=True)

model = MLP(in_dim=3*32*32, hidden=512, out_dim=10)
criterion = CrossEntropyLoss()
optim = SGD(model.parameters(), lr=0.1)

for xb, yb in train_loader:
    # 假设 yb 已是 MyTensor，内部用 one-hot（示例略）
    logits = model(xb)
    # 这里应将 yb 处理成 one-hot；训练脚本中已有封装
    # ...
    # loss = criterion(logits, y_one_hot)
    # loss.backward(); optim.step(); model.zero_grad()
    break  # 演示一次 batch
```

### 3. 命令行训练 (MLP)

```bash
python train.py --lr 0.01 --batch-size 128 --epochs 100 --optimizer momentum --hidden 512 \
  --weight-decay 0 --seed 42
```

### 4. 训练内置 CNN
脚本：`train_cnn.py`（包含日志、最佳权重、微批梯度累积、卷积分块等参数）。
```bash
python train_cnn.py --epochs 100 --batch-size 128 --lr 0.01 \
  --optimizer momentum --log-dir ./logs_cnn --save-best ./best_cnn.npz
```
可选内存/稳定性参数：
```bash
  --micro-batch 32            # 将一个 batch 拆成更小 micro-batch 逐次反向
  --conv-chunk-size 16         # 卷积前向分块处理样本数
  --conv-max-bytes 40000000    # 自动依据上限字节数切块
  --conv-method loop|im2col    # 回退低内存 loop 实现
  --mem-profile                # 打印卷积内存估算
```

输出内容：
```
logs_cnn/
  train_log.csv      # (epoch, phase, loss, acc)
  history.json       # 累积历史（便于绘图）
  checkpoints/       # 若 --save-all 开启
best_cnn.npz         # 验证集最佳模型权重
```

### 5. 生成/更新 LaTeX 报告
报告源文件：`report/main.tex`，资源图：`report/images/`。
推荐使用 XeLaTeX：
```bash
cd report
xelatex main.tex
xelatex main.tex   # 运行两次保证目录/引用
```
生成 `report/main.pdf`。

---

## 🧩 核心设计说明

| 组件 | 说明 | 关键点 |
|------|------|--------|
| `MyTensor` | 数据 + grad + 计算图元信息 | `parents` / `op` / 惰性构图 |
| 拓扑排序 | 反向传播顺序控制 | 检测环 + 后序 DFS |
| 原子算子 | `forward` + `backward` 封装 | 保持最小可扩展接口 |
| `Module` | 参数容器与层抽象 | 递归参数收集 / `zero_grad()` |
| 优化器 | 逐参数状态更新 | 统一 `step()` + `weight_decay` |
| 数据加载 | 简化版 `DataLoader` | 洗牌 / Batch 生成 / 转张量 |

附加机制：

| 主题 | 说明 |
|------|------|
| 初始化策略 | Conv2d 采用 Kaiming Normal (fan_out, ReLU)，Linear 可扩展 Xavier/Kaiming |
| 内存控制 | 卷积前向按样本/字节阈值切块，支持 micro-batch 梯度累积 |
| 日志系统 | CSV + JSON 双格式，支持恢复/可视化；最佳权重单独保存 |
| 可视化 | Notebook + Matplotlib：loss/acc 曲线、t-SNE 隐层特征分布 |
| 可扩展性 | 新增算子：实现 OP_base.forward/backward + 注册；新增层：继承 Module |

---

## 🏗️ 架构示意

```text
       ┌───────────────────────────────┐
       │           User Code          │
       │  (train.py / examples / …)   │
       └──────────────┬────────────────┘
            │ forward() 调用
          ┌──────▼───────┐
          │  Modules     │  (Linear / MLP / Future CNN ...)
          └──────┬───────┘
            │ 创建算子节点 (Operation)
          ┌──────▼────────┐
          │  MyTensor     │  (data, grad, parents, op)
          └──────┬────────┘
            │ 组织 DAG (计算图)
          ┌──────▼────────┐
          │ Topological   │  (后序遍历 -> 反向顺序)
          │   Sort        │
          └──────┬────────┘
            │ 触发 backward()
          ┌──────▼────────┐
          │   Ops.backward │  (局部梯度规则)
          └──────┬────────┘
            │ 传递 / 累加 grad
          ┌──────▼────────┐
          │ Optimizers    │  (SGD / Adam ... 更新参数)
          └───────────────┘
```

> 设计目标：用最少层次传达“**张量即节点 + 运算即边 + backward 即沿边传播梯度**”。

---

## 📈 实验与结果概要

已完成：

1. 学习率 (0.001 / 0.01 / 0.1 / 1.0) × BatchSize (16 / 64 / 128 / 256)
2. 优化器对比：SGD / Momentum / Nesterov / RMSProp / Adam / Adagrad
3. 隐藏层宽度：8 → 512 对欠拟合与过拟合的影响
4. 正则化（权重衰减）与泛化
5. 最佳模型保存 (Val Acc Max) 与测试集评估

核心实验（MLP + CNN）均提供统一日志格式，可复用绘图脚本或 notebook：

| 实验 | 维度 | 目标 |
|------|------|------|
| 学习率 × BatchSize | 4×4 组合 | 收敛特征 & 过拟合观察 |
| 优化器对比 | 6 种 | 收敛速度与稳定性 |
| 模型宽度 | 8→512 | 表达能力 vs 过拟合 |
| L2 正则 | 多权重衰减 | 泛化影响 |
| 简易 CNN | 与 MLP 对比 | 模型先验作用 |

CNN 默认实验设置（与 `train_cnn.py` 同步）：

| 项 | 配置 |
|----|------|
| 数据 | CIFAR-10 40k/5k/5k (train/val/test, seed=42, 归一化[0,1]) |
| 模型 | Conv(3→32,3×3,pad1)-ReLU-Pool2 → Conv(32→64,3×3,pad1)-ReLU-Pool2 → Flatten → FC(4096→10) |
| 损失 | CrossEntropy (one-hot) |
| 优化器 | Momentum（对比含 SGD/Adam/Adagrad/RMSProp/Nestrov） |
| 学习率 | 0.01 |
| 批大小 | train 128 / eval 256 |
| 轮数 | 100 (epoch 0 记录初始) |
| 初始化 | Conv: Kaiming Normal；FC: Xavier；bias=0 |
| 其它 | 支持卷积分块 / micro-batch；保存最佳权重 |

训练曲线与可视化（详见 `report/images/`）：

<p align="center">
  <img src="report/images/loss_grid.png" width="40%" />
  <img src="report/images/optimizer_val_acc.png" width="40%" />
</p>

> 更全面图表请打开：`report/main.pdf`

---

## 🧪 Benchmarks

> 本项目定位“教学清晰”优先，性能非主要优化方向；以下结果仅展示数量级供参考。

| 任务 | 模型 | Epochs | 时间 (CPU i5) | 最终 Val Acc | 备注 |
|------|------|--------|--------------|--------------|------|
| CIFAR‑10 | MLP (512 hidden) | 20 | ~X 分钟 | ~Y% | 单线程 NumPy |

（注：你可在本地运行后填写 X / Y；或增加 CNN 后再扩充表格。）

---

## 🛠️ 开发 / 贡献

运行静态检查与测试（若已安装可选依赖）：

```bash
ruff check .
mypy src/mytorch
pytest -q
```

建议贡献流程：

1. Fork & 新建分支：`feat/op-softmax` / `fix/conv-backward` 等
2. 添加算子：继承 OP_base，实现 forward/backward；必要中间量用 `self.saved_tensors`
3. 补充最小单元测试：梯度正确性可采用数值梯度对比
4. 更新 README / API 文档 / 示例脚本
5. 提交 PR：说明动机、API 兼容性、是否影响现有实验复现

---

## 🗺️ 路线图 / TODO

- [ ] 更完善的 CNN / 卷积与池化层（Padding/Stride 泛化 & loop fallback 优化）
- [ ] 自动广播与更多张量形状操作（matmul / view / expand）
- [ ] GPU (CuPy) / 后端抽象层
- [ ] 更多损失（L1 / SmoothL1 / Label Smoothing / Focal）
- [ ] 学习率调度器 & 梯度裁剪
- [ ] Mixup / Cutout 等轻量数据增强
- [ ] 简易 profiler（记录每算子耗时 / 内存峰值）
- [ ] Gradient Check 工具脚本

查看 `TODO.md` 获取最新计划。

---

## ❓ FAQ

**Q: 为什么不直接用 PyTorch？**  
教学成本：PyTorch 底层包含大量 C++ / Kernel / JIT / Dispatcher 机制，不利于初学者完整追溯。MyTorch 让你在 < 2k 行代码里走通核心链路。

**Q: 支持动态图还是静态图？**  
属于“即时构图”（eager）方式：每次前向都会构建当前计算图，并在 `backward()` 后释放引用。

**Q: 能扩展到 GPU 吗？**  
理论上可以把内部 `numpy` 抽象换成 `cupy` 或写一个后端适配层；当前 Roadmap 中有列出。

**Q: 可以商用吗？**  
遵循 MIT License，可自由使用，但请注意它不是为生产性能与稳定性设计。

**Q: 如何调试梯度是否正确？**  
可做：数值梯度对比（finite difference）；后续可添加 `gradient_check.py` 脚本。

---

## 📜 许可

本项目采用 MIT License（若 `LICENSE` 未更新，请补充作者与年份）。

---

## 🙌 致谢

- 经典框架设计启发：PyTorch / TinyGrad / MicroGrad
- 数据集：CIFAR‑10
- 指导课程 / 助教讨论

---

## 👤 作者

**张子路 (Zilu Zhang)**  
Email: <zhangzilu@bupt.edu.cn>  
时间：2025-10  

欢迎交流：实现、重构建议、课程讨论或教学用途引用。

---

## English Brief

MyTorch is a minimal educational deep learning framework built on top of NumPy. It implements a dynamic computation graph, reverse‑mode autodiff, modular layers (Linear / simple CNN), classic optimizers, logging, and a reproducible experiment suite on CIFAR‑10 (optimizers, learning rates, width scaling, regularization, and a lightweight CNN). A LaTeX report (report/main.tex) documents design insights. Emphasis: clarity & extensibility > raw performance.

### Citation (Optional)

If you reference this project in teaching material / reports:

```text
@misc{mytorch2025,
  title  = {MyTorch: A Minimal NumPy-based Autograd and Neural Network Framework},
  author = {Zilu Zhang},
  year   = {2025},
  url    = {https://github.com/yourname/mytorch}
}
```

---

<div align="center">
<sub>“Build your own tools to really understand the ones you use.”</sub>
</div>


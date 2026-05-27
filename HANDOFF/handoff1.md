# Handoff 1 — KNN-MMD + CSI-BERT 项目状态交接

> 创建时间：2026-05-27  
> 目的：新模型切换后能无缝接续当前工作状态

---

## 0. 项目一句话描述

用 WiFi CSI 信号做**跨域人体活动识别**。核心论文是 KNN-MMD（IEEE TMC 2025），用 Few-Shot + 局部 MMD 对齐解决 domain shift。数据预处理用 CSI-BERT（IEEE INFOCOM 2024 Workshop）恢复丢包。数据集是 WiFall（自建，5 动作 × 10 人，ESP32-S3 采集，2.4GHz，52 子载波，100Hz）。

---

## 1. 目录结构（全路径）

```
/home/sanguin1us/ScientificResearch/Documents/WIFI_CSI_HAR/KNN_MMD/codes/KNN-MMD-main(1)/
├── KNN-MMD-main/          ← KNN-MMD 核心代码 + 训练入口 + 实验归档
│   ├── train_fall.py       ← WiFall 训练入口（5类动作识别）
│   ├── train.py            ← WiGesture 训练入口（未使用）
│   ├── model.py            ← ResNet18 + MLP 分类头
│   ├── dataset.py          ← 数据加载 + 跨域划分（load_zero_shot）
│   ├── func.py             ← MK-MMD / 核函数 / KNN距离计算
│   ├── data/fall/          ← ★ 当前训练数据（CSI-BERT恢复 + 类均衡）
│   ├── data/fall_before_rebalance/  ← 均衡前的CSI-BERT数据备份
│   ├── data/fall_linear_interp_backup/  ← 原始线性插值数据备份
│   ├── experiments/        ← ★ 训练产物自动归档
│   │   ├── history/        ← 每次训练独立子文件夹（永不覆盖）
│   │   └── best/           ← 全局最佳（被更优结果自动覆盖）
│   ├── runs/               ← TensorBoard 事件文件
│   ├── knowledge.md        ← 38000字项目知识手册（必读）
│   ├── TRAINING_GUIDE.md   ← 训练操作指南
│   ├── run_1shot_x3.sh     ← 1-shot × 3 连续训练脚本
│   └── rebalance_data.py   ← 训练集类均衡脚本（我们写的）
│
├── CSI-BERT-main/          ← CSI-BERT 丢包恢复模型
│   ├── pretrain.py          ← ★ 预训练入口（已添加TensorBoard + 归档）
│   ├── recover.py           ← 丢包恢复入口
│   ├── model.py             ← CSIBERT / Token_Classifier / Sequence_Classifier
│   ├── dataset.py           ← CSI-BERT的数据加载（load_all）
│   ├── finetune.py          ← 下游微调（未使用）
│   ├── data/                ← 含 -1000 哨兵值的原始 magnitude/phase/timestamp
│   ├── experiments/         ← ★ 预训练产物自动归档（我们新增的）
│   │   ├── history/         ← pretrain_NTAR_* 和 pretrain_NTaR_*
│   │   └── best/            ← 当前最佳 = NTaR（无对抗，MAPE=12.26%）
│   ├── runs/                ← TensorBoard 事件文件
│   ├── recover.npy          ← ★ CSI-BERT 恢复后的 magnitude（已搬到 KNN-MMD）
│   └── replace.npy          ← 全帧重建版本（未使用）
│
└── HANDOFF/                 ← 交接文档目录
    └── handoff1.md          ← 本文档
```

---

## 2. 训练环境

```bash
source /home/sanguin1us/miniconda3/etc/profile.d/conda.sh
conda activate myenv
python -c "import torch; print(torch.cuda.is_available())"  # True → RTX 2050
```

**已知环境问题**：`transformers==4.57.6` 已移除 `AdamW`，`pretrain.py` 已改为 `from torch.optim import AdamW`。

**CUDA 驱动**：需重启后 `nvidia-smi` 才能正常工作（NVML 版本不匹配）。

---

## 3. 已完成的实验全记录

### 3.1 KNN-MMD 分类训练

**任务**：WiFall 动作识别，1-shot，Person 0 目标域

| 数据版本 | 训练集 | 次数 | Best Acc 均值 | Best Acc 范围 | 备注 |
|---|---|---|---|---|---|
| 线性插值 · 未均衡 | 1793 | 4 | **55.9%** | 50.9–61.0% | 当前最高单次 61.0% |
| CSI-BERT · 未均衡 | 1793 | 4 | **53.4%** | 50.9–55.9% | 与线性无统计差异 |
| CSI-BERT · 均衡 | 1100 | 3 | **46.9%** | 45.8–49.2% | 均衡起反效果 |

**统一超参**：`--k 1 --n 1 --p 0.5 --d 32 --mode 1 --lr 0.0005 --batch_size 256 --cuda 0`

**关键发现**：
- CSI-BERT 恢复 vs 线性插值 → 精度无显著差异（53.4% vs 55.9%，在1-shot波动范围内）
- 类均衡使精度下降 ~6.5pp（丢弃了 38% 训练数据）
- 全部结果远低于论文的 75.3%

### 3.2 CSI-BERT 预训练

| 实验 | 配置 | Best MAPE | Epochs | 备注 |
|---|---|---|---|---|
| NTAR | Normal + TimeEmb + **Adversarial** + RandomMask | 13.40% | 318 | 判别器后期崩塌 |
| **NTaR** | Normal + TimeEmb + RandomMask（**无对抗**） | **12.26%** | 470 | ★ 当前最佳，已自动覆盖 best/ |

**预训练命令**：
```bash
cd CSI-BERT-main/
python pretrain.py --normal --time_embedding --random_mask_percent --data_path ./data/magnitude.npy --cuda 0
```

**关键发现**：去掉 `--adversarial` 后 MAPE 从 13.4% 降到 12.3%，训练更稳定，无判别器崩塌问题。1857 样本对对抗训练来说太小。

### 3.3 CSI-BERT 丢包恢复

```bash
cd CSI-BERT-main/
python recover.py --normal --time_embedding --path ./experiments/best/pretrain.pth --data_path ./data/magnitude.npy --cuda 0
```

产出 `recover.npy` (1857, 100, 52)，已搬到 KNN-MMD 的 `data/fall/magnitude_linear.npy`。

---

## 4. 核心未解决问题：75.3% 复现失败

### 4.1 论文声称的条件

- 数据集：WiFall（和我们是同一个 GitHub 仓库）
- 训练集：People 1-9，540 样本（108/类，已丢 34 个 fall）
- 测试集：Person 0，60 样本
- UMAP d=128（但 60 样本根本跑不了 d=128——这是硬矛盾）
- 其他超参和我们一致

### 4.2 已排除的原因

| 嫌疑 | 结论 |
|---|---|
| 超参数不同 | ❌ 排除——完全一致 |
| 数据管线不同 | ❌ 排除——同一 process1.py + process2-split.py |
| 代码bug | ❌ 排除——train_fall.py 无逻辑错误 |
| d=32太小 | ❌ 排除——论文 Table VI 验证 d=32 在1-shot下精度不差 |
| CSI-BERT vs 线性插值 | ❌ 排除——两者精度无差异 |
| 类不均衡 | ❌ 排除——均衡后反而更差 |

### 4.3 仍可能的原因

1. **CSV 源数据被扩充**——928 个 CSV 产出 1857 样本，论文描述的 ~600 样本 → 数据采集量在论文发表后增加了。分布变化导致模型行为不同
2. **支撑集选取无随机性**——代码按 npy 顺序取前 k 个，不是随机采样。不同的数据排列导致不同的支撑集质量
3. **未跑满 10 折交叉验证**——论文可能报告的是 10 个 Person ID 各做目标域后的平均，而非单 ID
4. **论文实际用的 d 值未公开**——60 样本跑不了 d=128，论文实际 d 值未知

---

## 5. 我们修改过的文件

### 5.1 CSI-BERT-main/pretrain.py（有 git 记录）

| 修改 | 内容 |
|---|---|
| 新增 import | `SummaryWriter`, `datetime`, `os`, `json`, `shutil`, `re` |
| 修复 import | `from torch.optim import AdamW`（原 `from transformers import BertConfig,AdamW` 在 4.57 不可用） |
| 新增 TensorBoard | run_name 生成 + SummaryWriter + 每 epoch scalar 日志 |
| 新增归档系统 | `experiments/history/<run>/` + `experiments/best/` 自动覆盖，含 meta.json |

### 5.2 KNN-MMD-main/data/fall/（npy 文件替换）

| 文件 | 当前版本 |
|---|---|
| magnitude_linear.npy | **CSI-BERT recover.npy**（1857×100×52，后类均衡到 1164） |
| phase_linear.npy | 原始线性插值版本（复用） |
| action.npy | 原始 |
| people.npy | 原始 |

### 5.3 KNN-MMD-main/rebalance_data.py（我们新建的）

类均衡脚本，按非 fall 类的最小样本数统一削减所有类。备份到 `data/fall_before_rebalance/`。

---

## 6. 新模型接手后的推荐方向

### 优先级 1：解决 75.3% 精度差距

- 检查 Git 历史中 WiFall CSV 的提交记录，看数据何时被扩充
- 尝试找到论文实验时的确切 npy 文件（可能在早期 commit 中）
- 跑 10 折交叉验证（所有 Person ID 各做一次目标域取平均），而非只看 Person 0

### 优先级 2：让 d=128 可行

- Person 0 只有 64 样本 → 把 Person 0 + Person 1 的部分样本合并为目标域（人工增加样本数）
- 或换用 PCA 替代 UMAP（PCA 无样本数维度约束）
- 或修改 `dimension_reducation` 的 UMAP 参数（n_neighbors 调小）

### 优先级 3：数据管线优化

- 当前 process2-split.py 固定 gap=1, length=100 → 可尝试不同窗口参数
- 丢弃 CSI-BERT 改用线性插值（精度无差异但简单得多）

---

## 7. 关键文件快速索引

| 想知道什么 | 看哪个文件 |
|---|---|
| 项目完整知识 | KNN-MMD-main/knowledge.md |
| 怎么跑训练 | KNN-MMD-main/TRAINING_GUIDE.md |
| 所有历史实验精度 | KNN-MMD-main/experiments/history/*/meta.json |
| 当前最佳 KNN-MMD 权重 | KNN-MMD-main/experiments/best/ |
| 当前最佳 CSI-BERT 权重 | CSI-BERT-main/experiments/best/ |
| KNN-MMD 论文原文 | ../documents/KNN_MMD Cross Domain Wireless Sensing Via Local Distribution Alignment.pdf |
| 数据预处理脚本 | KNN-MMD-main/WiFall/data_process_example/ |
| TensorBoard 监控 | `tensorboard --logdir <path>/runs --port 6006` |

---

## 8. Conda 环境

```bash
source /home/sanguin1us/miniconda3/etc/profile.d/conda.sh
conda activate myenv
```

关键包：`torch`, `torchvision`, `numpy`, `umap-learn`, `scikit-learn`, `scipy`, `tqdm`, `pandas`, `transformers==4.57.6`, `tensorboard`

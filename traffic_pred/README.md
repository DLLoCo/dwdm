# 城市出行需求预测 — 代码结构说明

## 项目概览

**任务**：给定纽约市 200 个网格区域过去一段时间的出租车上下车数据，预测下一个时间步（30分钟）各区域的需求量。

**数据格式**：输入 (B, 35, 200, 2)，输出 (B, 1, 200, 2)。35 步 = 3天前同时段9步 + 2天前9步 + 1天前9步 + 最近4小时8步。2个通道 = 上车(pickup) + 下车(dropoff)。

**最终模型 MAE = 17.14**，比 LSTM(22.55) 好 24%，比 CCRNN(26.29) 好 35%。

---

## 目录结构

```
traffic_pred/
├── configs/                    # 配置文件（超参数、数据路径等）
│   ├── nyctaxi.yaml            # 基础配置
│   ├── nyctaxi_cluster.yaml    # v6: 带聚类的配置
│   ├── nyctaxi_memory.yaml     # v7: 最终模型配置 ← 当前使用
│   └── ablation/               # 消融实验配置
│       ├── full.yaml
│       ├── no_residual.yaml
│       ├── no_spatial.yaml
│       ├── no_temporal.yaml
│       └── no_cluster.yaml
│
├── data/                       # 数据目录
│   ├── raw/                    # 原始 NYC TLC 数据（CSV/Parquet）
│   └── processed/
│       ├── NYCTaxi/            # ST-SSL 格式的预处理数据 ← 主要数据源
│       │   ├── train.npz       # 训练集 (1912, 35, 200, 2)
│       │   ├── val.npz         # 验证集 (274, 35, 200, 2)
│       │   ├── test.npz        # 测试集 (546, 35, 200, 2)
│       │   └── adj_mx.npz     # 邻接矩阵 (200, 200)
│       └── NYCTaxi_OD/         # OD 流量数据（Flow Gate 用，最终未使用）
│           └── od_hourly.npz   # (48, 200, 200) 每半小时的 OD 矩阵
│
├── model/                      # 模型核心代码
│   ├── net.py                  # ★ 主模型 TrafficPredNet（整合所有模块）
│   ├── st_block.py             # ST Block: TC + CGC + 残差连接
│   ├── temporal_conv.py        # 门控时间卷积（GLU）
│   ├── cgc.py                  # 耦合图卷积 + 邻接矩阵学习器
│   ├── flow_gate.py            # Flow Gate（OD 注意力，最终未使用）
│   └── ccgru_cell.py           # CCGRU 单元（GRU+图卷积，参考用）
│
├── lib/                        # 工具库
│   ├── utils.py                # 通用工具：配置加载、随机种子、StandardScaler
│   ├── dataloader.py           # 数据加载：读 npz → 归一化 → DataLoader
│   ├── metrics.py              # 评估指标：MAE、RMSE、MAPE、MaskedMAELoss
│   ├── adj_builder.py          # 网格邻接矩阵构建（当 adj_mx.npz 不存在时）
│   └── cluster_builder.py      # 空间/时间聚类（K-Means 预处理）
│
├── train.py                    # ★ 训练脚本
├── evaluate.py                 # ★ 评估脚本（输出分时段/分区域指标）
├── visualize.py                # 可视化（热力图、对比图、训练曲线等）
├── baselines.py                # 基线模型：HA、MLP、LSTM、TCN、Transformer、STGCN、CCRNN
├── run_ablation.py             # 消融实验一键脚本
├── diagnose_night.py           # 深夜误差诊断脚本
├── explore_data.py             # 数据探索与可视化
├── preprocess_raw.py           # 原始数据预处理（生成 OD 矩阵）
│
├── checkpoints/                # 模型权重保存目录
│   └── v7_memory/
│       └── best_model.pt       # 最终最佳模型
│
└── figures/                    # 输出图片
    ├── test_predictions.npz    # evaluate.py 的预测结果
    ├── heatmap_*.png           # 空间热力图
    ├── time_series.png         # 时序对比图
    ├── comparison_*.png        # baseline 对比柱状图
    └── night_diagnosis/        # 深夜诊断图
```

---

## 核心代码详解

### 1. model/net.py — 主模型（最重要的文件）

**作用**：定义完整的 `TrafficPredNet`，整合所有子模块。

**前向传播流程**：

```
输入 x (B,35,200,2)
  ↓
Linear(2→64)                    # 特征投影到隐藏维度
  ↓
+ Spatial Cluster Embedding     # 加上区域类型先验（5类：冷区/热点/...）
  ↓
ST Block × 3                    # 3层时空编码（每层用不同的学习邻接矩阵）
  ↓
h = h[:, -1, :, :]             # 取最后时间步的表示
  ↓
Memory Bank                     # 注意力查询20个可学习prototype，增强表示
  ↓
MLP(h) → correction            # 预测修正量
  ↓
pred = x[:,-1,:,:] + correction # 残差预测：最后一步观测 + 修正
  ↓
输出 (B,1,200,2)
```

**关键类**：
- `ClusterEmbedding`：用 nn.Embedding 把聚类标签映射为向量
- `MemoryBank`：K=20 个可学习 prototype，通过 softmax 注意力检索
- `TrafficPredNet`：主模型，`from_config(config)` 从 yaml 创建

### 2. model/st_block.py — 时空编码块

**作用**：单个 ST Block，堆叠 3 层构成编码器。

**流程**：
```
输入 (B, T, N, D)
  → GatedTemporalConv    沿时间维做 1D 卷积，GLU 门控
  → GraphConv             对每个时间步做图卷积（扩散过程）
  → Dropout + LayerNorm + Residual
输出 (B, T, N, D)       shape 不变
```

### 3. model/temporal_conv.py — 门控时间卷积

**作用**：沿时间维度提取局部时间模式。

**原理**：把 (B,T,N,D) reshape 成 (B×N, D, T) 做 Conv1d，输出通道分两半——一半是值，一半过 sigmoid 做门控（GLU），两者逐元素相乘。kernel_size=3 表示看前后各 1 步。

### 4. model/cgc.py — 耦合图卷积

**作用**：学习区域间的空间依赖关系。

**两个核心组件**：

**GraphConv**（图卷积）：`output = Σ_{k=0}^{K} A^k × X × θ_k`。K=3 表示信息扩散 3 跳——每个节点能感知到 3 步以内的邻居。

**AdjacencyLearner**（邻接矩阵学习）：
- 初始邻接矩阵 A^(0) 由两组可学习嵌入 E1, E2 通过 `softmax(ReLU(E1·E2^T))` 生成（非对称，能捕捉"A→B 多 但 B→A 少"的模式）
- 上层邻接矩阵通过线性映射耦合演化：A^(1) = ψ(A^(0))，A^(2) = ψ(A^(1))
- 每层 ST Block 使用不同的邻接矩阵 → 多层次空间依赖

### 5. lib/cluster_builder.py — 聚类预处理

**作用**：对 200 个节点做空间聚类（K-Means），发现区域类型。

**做法**：
- 从训练集计算每个节点的 4 个统计特征：pickup 均值/标准差 + dropoff 均值/标准差
- StandardScaler 归一化后做 K-Means (K=5)
- 按均值排序：Cluster 0 = 冷区(156节点)，Cluster 4 = 超热点(6节点)
- 结果作为 nn.Embedding 的输入索引

### 6. lib/dataloader.py — 数据加载

**作用**：读取 ST-SSL 格式的 npz 数据，归一化，创建 DataLoader。

**流程**：
1. 加载 train/val/test.npz 和 adj_mx.npz
2. 如果开启聚类：在归一化之前做 K-Means（用原始尺度计算更准）
3. 用 StandardScaler 做 Z-score 归一化（训练集 fit，所有集 transform）
4. 估算时间索引（用于分时段评估，但模型本身不依赖绝对时间）
5. 创建 TensorDataset → DataLoader
6. 返回 6 个值：(train_loader, val_loader, test_loader, scaler, adj, cluster_info)

### 7. lib/metrics.py — 评估指标

**作用**：计算 MAE、RMSE、MAPE（都忽略零值区域）。

- `masked_mae(pred, true)`：平均绝对误差，忽略 true=0 的点
- `masked_rmse`：均方根误差
- `masked_mape`：平均百分比误差
- `MaskedMAELoss`：训练用的 loss 函数

### 8. train.py — 训练脚本

**作用**：读配置 → 加载数据 → 创建模型 → 训练循环 → 保存最佳模型。

**关键流程**：
```python
python train.py --config configs/nyctaxi_memory.yaml
```
- 每个 epoch：前向 → 算 loss → 反向 → 梯度裁剪 → 更新
- 每个 epoch 结束：在验证集上评估，如果 val_loss 更好就保存 checkpoint
- 早停：连续 15 个 epoch 验证集不改善就停止
- 输出：`checkpoints/v7_memory/best_model.pt` + `train_history.json`

### 9. evaluate.py — 评估脚本

**作用**：加载最佳模型，在测试集上评估，输出详细指标。

```python
python evaluate.py --config configs/nyctaxi_memory.yaml --checkpoint checkpoints/v7_memory/best_model.pt
```

**输出的分析维度**：
- 整体 MAE/RMSE/MAPE
- 分流量类型（pickup vs dropoff）
- 分时段（早高峰/午间/晚高峰/深夜/其他）
- 工作日 vs 周末
- 热点节点 vs 冷区节点
- 分空间聚类（Cluster 0~4 各自的误差）
- 分时间聚类（如果开启）

### 10. baselines.py — 基线模型

**作用**：实现 7 个对比模型，用同样的数据和评估方式训练评估。

| 模型 | 类型 | 核心 | MAE |
|------|------|------|-----|
| HA | 统计 | 历史同时段均值 | 36.14 |
| Last-value | 统计 | 直接用上一步 | 11.44 |
| MLP | 深度学习 | 全连接网络 | 23.64 |
| LSTM | 深度学习 | 循环神经网络 | 22.55 |
| TCN | 深度学习 | 时间卷积 | 25.05 |
| Transformer | 深度学习 | 自注意力 | 23.84 |
| STGCN | 图网络 | 固定图 + 时间卷积 | 30.98 |
| CCRNN | 图网络 | 学习图 + GRU | 26.29 |

### 11. visualize.py — 可视化

**作用**：从 `test_predictions.npz` 生成报告用图表。

生成的图：
1. 空间热力图（真实值 vs 预测值 vs 误差）
2. Baseline 对比柱状图
3. 训练损失曲线
4. 时序对比图（选几个代表节点）
5. 按需求量级分段的误差分析
6. 空间误差热力图

### 12. explore_data.py — 数据探索

**作用**：分析原始数据的时空分布特征，生成报告第二章用的图。

生成的图包括：日内需求曲线、工作日vs周末对比、空间热力图（不同时段）、需求分布直方图、Top-20%节点分析、OD 流量分析。

### 13. diagnose_night.py — 深夜诊断

**作用**：分析为什么深夜预测特别差。

诊断发现：模型在深夜系统性高估（92% 高估），热点节点白天需求 ~500 深夜降到 ~80，但模型预测 ~400。这直接促使我们设计了残差预测策略。

---

## 运行顺序

```bash
# 1. 训练最终模型
python train.py --config configs/nyctaxi_memory.yaml

# 2. 评估
python evaluate.py --config configs/nyctaxi_memory.yaml --checkpoint checkpoints/v7_memory/best_model.pt

# 3. 生成可视化
python visualize.py --config configs/nyctaxi_memory.yaml

# 4. 消融实验（可选）
python run_ablation.py

# 5. 跑 baseline 对比（可选，已有结果）
python baselines.py
```

---

## 模型演化历程

| 版本 | 改动 | MAE | 关键发现 |
|------|------|-----|---------|
| v1 | GRU baseline | 35.96 | GRU 太慢 |
| v2 | TC + CGC | 21.95 | 架构有效，但深夜 MAE=55 |
| v3~v5 | +FlowGate/TimeEmbed | 22~24 | 都没改善 |
| v6 | +聚类+残差 | 17.83 | 残差预测解决深夜问题 |
| **v7** | **+Memory Bank** | **17.14** | **最终版本** |

---

## 配置文件说明 (nyctaxi_memory.yaml)

```yaml
model:
  hidden_dim: 64          # 隐藏维度
  num_layers: 3           # ST Block 层数
  cheb_k: 3               # 图卷积扩散步数
  embed_dim: 50           # 节点嵌入维度（用于学习邻接矩阵）
  kernel_size: 3          # 时间卷积核大小
  use_spatial_cluster: true   # 开启空间聚类嵌入
  use_residual: true          # 开启残差预测
  use_memory_bank: true       # 开启模式记忆库
  n_prototypes: 20            # 记忆库中 prototype 数量
```

---

## 汇报时的重点

1. **数据处理**：采用 ST-SSL 的多周期窗口格式（不是简单的连续序列）
2. **模型架构**：TC+CGC 并行时空编码 → 聚类先验 → Memory Bank 模式检索 → 残差预测
3. **深夜问题的发现和解决**：这是最有故事性的部分，从诊断到解决的完整链条
4. **消融实验**：每加一个模块都有明确的数字提升
5. **与前沿方法的联系**：Memory Bank 参考 GEnSHIN/MegaCRN，CGC 来自 CCRNN
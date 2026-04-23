# scLGAE: 单细胞低秩图自编码器

## 项目简介

scLGAE (Single-cell Low-rank Graph AutoEncoder) 是一个专门用于单细胞RNA测序数据聚类分析的深度学习框架。该项目结合了图自编码器 (Graph AutoEncoder, GAE) 和低秩表示 (Low-Rank Representation, LRR) 技术，旨在提高单细胞数据的聚类性能和细胞类型识别准确性。

### 主要特性

- **图自编码器 (GAE)**: 使用图注意力网络 (GAT) 作为编码器，学习细胞的低维嵌入表示
- **低秩表示 (LRR)**: 通过低秩约束和块对角正则化优化细胞相似性矩阵
- **多模态融合**: 支持多种数据融合策略和权重学习
- **超参数优化**: 集成 Optuna 框架进行自动化参数调优
- **可视化分析**: 提供全面的聚类结果可视化和分析工具
- **消融实验**: 支持多种模型变体的对比实验

## 技术架构

### 核心模块

1. **GAE 模块** (`gae_model.py`)
   - GAT 编码器：多头注意力机制学习细胞嵌入
   - ZINB 解码器：零膨胀负二项分布建模基因表达
   - 支持 ZINB 损失和流形损失

2. **LRR 模块** (`lrr_training.py`)
   - 低秩约束：促进全局结构学习
   - 块对角正则化：增强细胞类型分离
   - 自适应权重融合

3. **数据预处理** (`data_preprocessing.py`)
   - 基因过滤和细胞质控
   - 归一化和对数变换
   - KNN 图构建

4. **聚类分析** (`clustering.py`)
   - 谱聚类
   - K-means 聚类
   - 多种评估指标 (ARI, NMI, AMI, AOM)

## 安装要求

### 系统要求
- Python == 3.12
- CUDA (可选，用于GPU加速)

### 依赖包
项目依赖见 requirements.txt
torch 相关包默认使用cuda118加速版本, 可根据系统自行修改



## 使用方法

### 基本使用

1. **配置参数** (`config.py`)
```python
# 设置数据路径和工作目录
DATASET_DIR = "path/to/your/data"
WORK_DIR = "path/to/scLGAE"
```

2. **运行默认模型**
```python
from experiments import default_model
default_model()
```

3. **单个数据集分析**
```python
from experiments import visualization_experiment
# 运行可视化实验
visualization_experiment('Muraro')
```

### 高级功能

#### 超参数优化
```python
from experiments import opt_ab2_model
# 对多个数据集进行参数优化
opt_ab2_model()
```

#### 消融实验
```python
from experiments import ablation_experiment
# 运行消融实验
ablation_experiment(2)  # 运行第2号消融实验
```

#### 单数据集优化
```python
from experiments import opt_ab2_single
# 对单个数据集进行优化
best_params, results = opt_ab2_single('Muraro', n_trials=50)
```

## 项目结构

```
scLGAE/
├── README.md                      # 项目文档
├── requirements.txt               # 依赖包列表
│
├── core/                          # 核心模块
│   ├── experiments.py             # 实验函数入口
│   ├── config.py                  # 配置文件
│   ├── gae_model.py              # GAE模型定义
│   ├── lrr_training.py           # 训练和LRR模块
│   ├── data_preprocessing.py      # 数据预处理
│   ├── clustering.py             # 聚类算法
│   ├── utilities.py              # 工具函数
│   └── visualization.py          # 可视化模块
│
├── supplementary/                 # 辅助模块
│   ├── optimization_new.py       # 超参数优化
│   ├── multiview.py              # 多视图学习
│   ├── SNF.py                    # 相似性网络融合
│   ├── GAEval.py                 # GAE评估模块
│   ├── symmetricNMF.py           # 对称非负矩阵分解
│   ├── test_optimization_new.py  # 优化测试
│   ├── optimzation.py            # 旧版本优化
│   └── clustering_results.csv    # 聚类结果数据
│
├── docs/                          # 文档
│   └── scLGAE.pdf               # 项目详细文档
│
└── .git/                          # Git版本控制
```

## 支持的数据集

项目已在以下单细胞数据集上进行测试：

- **Muraro**: 胰腺细胞数据集
- **Adam**: 肺部细胞数据集  
- **Romanov**: 下丘脑细胞数据集
- **Young**: 肾脏细胞数据集
- **Quake系列**: 
  - 10x Genomics: Bladder, Limb_Muscle, Spleen
  - Smart-seq2: Diaphragm, Limb_Muscle, Lung, Trachea

## 实验结果

### 性能指标
- **ARI** (Adjusted Rand Index): 调整兰德指数
- **NMI** (Normalized Mutual Information): 归一化互信息
- **AMI** (Adjusted Mutual Information): 调整互信息  
- **AOM** (Average Overall Metric): 平均综合指标

### 消融实验
1. **Ablation 1**: X→LRR 直接低秩表示
2. **Ablation 2**: GAE重构 + LRR (主要框架)
3. **Ablation 3**: 仅GAE + K-means聚类
4. **Ablation 4**: LRR + K-means聚类
5. **Ablation 5**: 多视图融合框架

## 可视化功能

项目提供丰富的可视化分析：

- **UMAP降维可视化**: 展示细胞嵌入和聚类结果
- **相似性矩阵热图**: 可视化细胞间相似性
- **优化历史曲线**: 超参数优化过程可视化
- **聚类性能对比**: 不同方法的性能比较

## 配置说明

### 主要参数类别

1. **KNN参数** (`knn_params`)
   - `k_knn`: 最近邻数量
   - `metric`: 距离度量方法
   - `threshold`: 相似性阈值

2. **预处理参数** (`preprocess_params`)
   - `min_genes`: 最小基因数
   - `min_cells`: 最小细胞数
   - `target_sum`: 目标总读数
   - `n_top_genes`: 高变基因数量

3. **GAE参数** (`gae_params`)
   - `gat_hidden_channels`: GAT隐藏层维度
   - `head_num`: 注意力头数
   - `lr`: 学习率
   - `alpha`, `beta`: 损失函数权重

4. **LRR参数** (`lrr_params`)
   - `lrr_lambda`: 低秩正则化权重
   - `lrr_gamma`: 块对角正则化权重
   - `lrr_weight`: LRR融合权重

## 开发和贡献

### 添加新数据集
1. 将数据放置在 `DATASET_DIR` 目录下
2. 在 `data.py` 中添加数据加载函数
3. 更新 `main.py` 中的数据集列表

### 自定义模型
1. 在 `model.py` 中定义新的编码器/解码器
2. 在 `newtrain.py` 中实现训练逻辑
3. 在 `main.py` 中添加运行接口


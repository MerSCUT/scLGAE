import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.decomposition import NMF
import numpy as np
import h5py
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.neighbors import NearestNeighbors
import optuna
import pandas as pd
import os
import time
from sklearn.metrics import adjusted_rand_score as ARI
# 在导入部分添加AMI
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# 还需要添加一些自定义的组件


from config import *



def set_random_seed(seed=42):
    """设置随机种子以确保实验可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子为 {seed}")

def check_nan(tensor, name="Tensor"):
    """检查张量是否包含NaN值"""
    if torch.isnan(tensor).any():
        print(f"警告: {name} 包含NaN值")
        return True
    return False

def average_overlap_measure(labels_true, labels_pred):
    """
    计算平均重叠度量(Average Overlap Measure)
    AOM衡量的是每个真实类别与预测簇之间的最大重叠比例的平均值。
    值越接近1表示聚类结果与真实标签的一致性越高。
    参数:
        labels_true: 真实标签
        labels_pred: 预测标签
    返回:
        AOM值
    """
    unique_true = np.unique(labels_true)
    n_true = len(unique_true)
    aom = 0.0
    for true_label in unique_true:
        # 获取当前真实类别的样本
        mask = (labels_true == true_label)
        pred_labels = labels_pred[mask]
        # 计算这些样本在各个预测簇中的分布
        unique_pred, counts = np.unique(pred_labels, return_counts=True)
        # 计算最大重叠比例
        if len(counts) > 0:
            max_overlap = np.max(counts) / np.sum(mask)
            aom += max_overlap
    return aom / n_true

def reweighted_frobenius_loss(X, eps=1e-4):
    # X: (batch, dim) 或 (dim, dim)
    U, S, V = torch.svd(X)  # 奇异值分解
    loss = torch.sum(S**2 / (S + eps))
    return loss


import torch
from typing import Optional

@torch.no_grad()
def _fix_sparse_coalesced(x: torch.Tensor) -> torch.Tensor:
    # 保守处理：COO 格式建议 coalesce；CSR 不需要
    if x.is_sparse and x.layout == torch.sparse_coo:
        return x.coalesce()
    return x

def nuclear_norm_randomized(
    X_sparse: torch.Tensor,
    rank: int = 20,
    oversample: int = 10,
    n_power_iter: int = 1,
    seed: Optional[int] = None,
):
    """
    用随机化 SVD 近似 ||X||_*，适用于稀疏张量 X_sparse (m x n)。
    只用 sparse@dense 乘法 + 小矩阵 SVD，支持 autograd。

    Args:
        X_sparse: 稀疏矩阵，形状 (m, n)，支持 COO/CSR。
        rank: 目标秩（近似的奇异值个数）。
        oversample: 过采样维度，通常 5~20。
        n_power_iter: 功率迭代次数（1~3 一般足够）。
        seed: 随机种子（可选）。

    Returns:
        approx_nucnorm: 近似核范数 (标量张量)
        S: 近似获得的奇异值（长度 ≤ rank+oversample）
    """
    assert X_sparse.is_sparse, "X_sparse 必须是稀疏张量 (COO/CSR)。"
    X_sparse = _fix_sparse_coalesced(X_sparse)

    m, n = X_sparse.shape
    l = min(rank + oversample, n)  # 子空间维度

    if seed is not None:
        gen = torch.Generator(device=X_sparse.device).manual_seed(seed)
        Omega = torch.randn(n, l, dtype=X_sparse.dtype, device=X_sparse.device, generator=gen)
    else:
        Omega = torch.randn(n, l, dtype=X_sparse.dtype, device=X_sparse.device)

    # 1) 采样列空间：Y = X * Omega （稀疏@稠密）
    Y = torch.sparse.mm(X_sparse, Omega)  # (m, l)

    # 2) 可选功率迭代，增强谱分离：Y = (X X^T)^q X Omega
    #   使用交替的 sparse@dense 乘法：Z = X^T Y, Y = X Z
    for _ in range(max(0, n_power_iter)):
        Z = torch.sparse.mm(X_sparse.transpose(0, 1), Y)  # (n, l)
        Y = torch.sparse.mm(X_sparse, Z)                  # (m, l)

    # 3) 正交化：Y = Q R
    #   torch.linalg.qr 支持梯度，选择 reduced 模式减少开销
    Q, _ = torch.linalg.qr(Y, mode="reduced")  # (m, l)

    # 4) 小矩阵 B = Q^T X （避免把 X 展开为 dense）
    #    用 (Q^T X) = (X^T Q)^T，并计算 XtQ via 稀疏@稠密
    XtQ = torch.sparse.mm(X_sparse.transpose(0, 1), Q)  # (n, l)
    B = XtQ.transpose(0, 1)                             # (l, n)

    # 5) 对小矩阵做 SVD（dense），得到奇异值近似
    #    注意：这是对 X 的奇异值的近似。
    U_hat, S, Vh_hat = torch.linalg.svd(B, full_matrices=False)

    # 6) 近似核范数：奇异值求和
    #    也可以只取前 rank 个（通常 S 已经是按降序）。
    k_eff = min(rank, S.numel())
    approx_nucnorm = S[:k_eff].sum()

    return approx_nucnorm, S


from typing import Literal, Tuple
def build_affinity(
    Z: torch.Tensor,
    k0: int = 10,                 # 自适应带宽用的第 k0 近邻
    k: int = 20,                  # 互为 kNN 的 k
    metric: Literal["euclidean", "cosine"] = "euclidean",
    return_sparse: bool = False,  # 是否返回稀疏 COO
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    从嵌入 Z 构造亲和矩阵 W（对称、非负、去自环、互为 kNN 稀疏化）。

    参数
    ----
    Z : (n, d) torch.Tensor
        每行一个细胞的嵌入表示。可在 CPU/GPU 上。
    k0 : int
        自适应带宽的近邻次序（建议 7~15）。仅当 metric="euclidean" 时使用。
    k : int
        互为 kNN 的近邻个数（建议 10~30）。
    metric : {"euclidean", "cosine"}
        - "euclidean": 自适应高斯核,  W_ij = exp(-||zi-zj||^2 / (sigma_i*sigma_j))
        - "cosine":    先单位化，W_ij = relu(zi^T zj)
    return_sparse : bool
        True 则返回稀疏 COO 张量；False 返回致密张量。
    eps : float
        数值稳定项。

    返回
    ----
    W : (n, n) torch.Tensor
        稀疏或致密的亲和矩阵（对称、非负、无自环）。
    """
    assert Z.dim() == 2, "Z must be (n, d)"
    n = Z.size(0)
    if n <= 1:
        return torch.zeros((n, n), device=Z.device, dtype=Z.dtype)

    k0 = max(1, min(k0, n - 1))
    k  = max(1, min(k,  n - 1))

    if metric == "euclidean":
        # pairwise Euclidean distances
        # torch.cdist 返回 L2 距离；再平方得到 ||·||^2
        D = torch.cdist(Z, Z, p=2) ** 2                  # (n, n)
        # 自环距离置无穷，避免被选为近邻
        D.fill_diagonal_(float("inf"))

        # 第 k0 近邻距离作为自适应带宽 sigma_i
        # topk 对距离做升序 ⇒ 取 indices 直接 gather
        # 先取前 k0 个最近邻的距离（含第1,2,...,k0）
        knn_dists, _ = torch.topk(-D, k=k0, dim=1)       # 负号变最大化 ⇒ 得到最小的 k0 个
        sigma = (-knn_dists[:, -1]).clamp_min(eps)       # 第 k0 小的距离
        # 计算高斯核：exp( -D_ij / (sigma_i * sigma_j) )
        # 外积得到 sigma_i*sigma_j
        Sigma = torch.outer(sigma, sigma)
        W = torch.exp(-(D / (Sigma + eps)))

    elif metric == "cosine":
        # 归一化并计算 ReLU( dot )
        Z_norm = Z / (Z.norm(dim=1, keepdim=True) + eps)
        W = (Z_norm @ Z_norm.T).clamp(min=0.0)
        # 自环先不处理，留到最后统一去掉
        W = W.contiguous()
        # 为了后续 kNN 基于“相似度大为近邻”，我们把对角线临时置 -inf 以避免选到自己
        W.fill_diagonal_(float("-inf"))
    else:
        raise ValueError("metric must be 'euclidean' or 'cosine'")

    # ---- 互为 kNN 稀疏化 ----
    # 对每一行，选出相似度最大的 k 个（欧氏核里“越大越近”已经满足）
    # 注意：若 metric == "euclidean"，对角线不是 -inf，此处先行屏蔽自环
    if metric == "euclidean":
        W_no_self = W.clone()
        W_no_self.fill_diagonal_(float("-inf"))
        vals, idx = torch.topk(W_no_self, k=k, dim=1)
    else:
        # metric == "cosine" 已将对角线设为 -inf
        vals, idx = torch.topk(W, k=k, dim=1)

    # 用稀疏掩码表示“i 的 kNN 包含 j”
    knn_mask = torch.zeros((n, n), device=Z.device, dtype=torch.bool)
    row_idx = torch.arange(n, device=Z.device).unsqueeze(1).expand_as(idx)
    knn_mask[row_idx, idx] = True

    # 互为 kNN（mutual）：既是 i 的邻居又是 j 的邻居
    mutual = knn_mask & knn_mask.T

    # 保留互为 kNN 的边，其它置 0；随后对称化、去自环
    W = torch.where(mutual, W, torch.zeros((), device=W.device, dtype=W.dtype))
    W = 0.5 * (W + W.T)
    W = torch.where(torch.eye(n, device=W.device, dtype=torch.bool), torch.zeros((), device=W.device, dtype=W.dtype), W)

    # 数值裁剪，确保非负
    W.clamp_(min=0.0)

    if return_sparse:
        # 转为稀疏 COO，去掉 0
        idxs = torch.nonzero(W > 0, as_tuple=False).T
        vals = W[idxs[0], idxs[1]]
        W_sparse = torch.sparse_coo_tensor(idxs, vals, size=W.shape, device=W.device, dtype=W.dtype)
        return W_sparse.coalesce()

    return W
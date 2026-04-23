import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import random
from copy import deepcopy

def get_undirected_edges_from_adj(A_np):
    """从对称邻接矩阵中取出无重复无自环的边 (i<j)"""
    assert (A_np.T == A_np).all(), "A 需为对称邻接矩阵"
    n = A_np.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    mask = (A_np[iu, ju] > 0)
    iu, ju = iu[mask], ju[mask]
    edges = np.vstack([iu, ju]).T
    return edges  # shape: [E, 2], 每条边 i<j

def sample_negative_edges(n, pos_edges, num_neg):
    """从上三角随机采样负边（图中不存在的边），数量=num_neg"""
    pos_set = set((int(i), int(j)) for i, j in pos_edges)
    neg = []
    trials = 0
    max_trials = num_neg * 10 + 1000
    while len(neg) < num_neg and trials < max_trials:
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            trials += 1
            continue
        u, v = (i, j) if i < j else (j, i)
        if (u, v) in pos_set:
            trials += 1
            continue
        neg.append((u, v))
        trials += 1
    if len(neg) < num_neg:
        # 兜底：全枚举补齐
        all_set = set((min(i,j), max(i,j)) for i in range(n) for j in range(i+1, n))
        remain = list(all_set - pos_set)
        random.shuffle(remain)
        neg += remain[:(num_neg - len(neg))]
    return np.array(neg, dtype=int)

def split_edges(pos_edges, val_frac=0.1, seed=0):
    """将正边划分为训练/验证"""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pos_edges))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return pos_edges[train_idx], pos_edges[val_idx]

def build_sparse_adj_from_edges(n, edges):
    """用边列表构造稀疏邻接（对称），用于编码器输入（训练时不含val边）"""
    if len(edges) == 0:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty((0,), dtype=torch.float32)
    else:
        u = torch.tensor(edges[:,0], dtype=torch.long)
        v = torch.tensor(edges[:,1], dtype=torch.long)
        # 无向图：补对称
        indices = torch.vstack([torch.cat([u, v]), torch.cat([v, u])])
        values = torch.ones(indices.shape[1], dtype=torch.float32)
    A_sp = torch.sparse_coo_tensor(indices, values, size=(n, n))
    return A_sp

def score_edges(Z, edges):
    """用内积解码边分数（logits），edges: np[int,int]"""
    if len(edges) == 0:
        return np.array([])
    with torch.no_grad():
        u = torch.tensor(edges[:,0], dtype=torch.long, device=Z.device)
        v = torch.tensor(edges[:,1], dtype=torch.long, device=Z.device)
        logits = (Z[u] * Z[v]).sum(dim=1)  # 未过sigmoid
        return logits.detach().cpu().numpy()

def linkpred_metrics_from_logits(pos_logits, neg_logits):
    """由正负样本的logits计算 AUC/AP（基于sigmoid概率）"""
    if len(pos_logits) == 0 or len(neg_logits) == 0:
        return 0.0, 0.0
    y_true = np.concatenate([np.ones_like(pos_logits), np.zeros_like(neg_logits)])
    y_score = 1.0 / (1.0 + np.exp(-np.concatenate([pos_logits, neg_logits])))
    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    return auc, ap
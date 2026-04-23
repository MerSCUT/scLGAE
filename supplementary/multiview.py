# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from data import *
from model import *
from utils import *
from config import *
from clustering import *
import numpy as np
import pandas as pd
# ============ 工具函数 ============

import torch
import torch.nn.functional as F

@torch.no_grad()
def knn_self_tuning_graph(
    Z: torch.Tensor,
    k: int = 15,
    metric: str = "euclidean",   # 'euclidean' 或 'cosine'
    batch_size: int = None,      # 大数据集用分批，默认一次算完
    sym_mode: str = "average",   # 'average' 或 'max'
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    返回: S 形状 [n, n] 的对称稀疏相似度 (sparse_coo_tensor)
    S_ij = exp( - ||z_i - z_j||^2 / (sigma_i * sigma_j) ) 仅在 kNN 上非零
    其中 sigma_i 是 i 的第 k 个近邻距离（自适应尺度）
    """
    device = Z.device
    n = Z.size(0)
    assert k < n, "k 必须小于样本数 n"

    if metric not in ("euclidean", "cosine"):
        raise ValueError("metric 仅支持 'euclidean' 或 'cosine'")

    # 规范化（若用余弦距离）
    if metric == "cosine":
        Zn = F.normalize(Z, p=2, dim=1)
    else:
        Zn = Z

    # 预分配
    knn_idx = torch.empty((n, k), dtype=torch.long, device=device)
    knn_dist = torch.empty((n, k), dtype=Z.dtype, device=device)
    sigma = torch.empty((n,), dtype=Z.dtype, device=device)

    if batch_size is None:
        # 经验：小几千细胞可一次性；更大用分批
        batch_size = n

    # 分批计算到全集的距离 -> top-k
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        B = end - start
        Zb = Zn[start:end]  # [B, d]

        if metric == "euclidean":
            # cdist: [B, n]
            D = torch.cdist(Zb, Zn)  # 欧氏距离
        else:
            # 余弦距离 = 1 - 余弦相似度
            D = 1.0 - (Zb @ Zn.T).clamp(-1, 1)

        # 排除自身：把对角所在位置替换成 +inf，确保不会被选为近邻
        ar = torch.arange(B, device=device)
        D[ar, start + ar] = float("inf")

        # 先取 k+1 个（防极端情况），再去掉可能的自身；这里已置 inf，可直接取 k
        vals, idx = torch.topk(D, k=k, largest=False, sorted=True)  # [B, k]

        knn_idx[start:end] = idx
        knn_dist[start:end] = vals
        # 第 k 个近邻的距离作为 sigma_i（按升序，索引 k-1）
        sigma[start:end] = vals[:, -1]

    # 计算权重：w_ij = exp( - d_ij^2 / (sigma_i * sigma_j) )
    # sigma_j 按邻居索引 gather
    sigma_j = sigma[knn_idx]                 # [n, k]
    sigma_i = sigma.unsqueeze(1).expand_as(sigma_j)
    denom = sigma_i * sigma_j + eps
    w = torch.exp(-(knn_dist ** 2) / denom)  # [n, k]

    # 构造有向稀疏图（每行保留 k 个近邻）
    row = torch.arange(n, device=device).repeat_interleave(k)  # [n*k]
    col = knn_idx.reshape(-1)                                  # [n*k]
    val = w.reshape(-1)

    S = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0),
        val,
        size=(n, n),
        device=device
    ).coalesce()

    # 对称化：S <- (S + S^T)/2  或  max(S, S^T)
    ST = S.transpose(0, 1).coalesce()
    if sym_mode == "max":
        # elementwise max: 将两个稀疏张量拼接后，同坐标取最大
        idx = torch.cat([S.indices(), ST.indices()], dim=1)
        val = torch.cat([S.values(), ST.values()], dim=0)
        # 合并同坐标 -> 取最大
        # 做法：先按坐标排序，再在同坐标组内取最大
        # 生成唯一键以便排序（行主序）
        key = idx[0] * n + idx[1]
        order = torch.argsort(key)
        idx = idx[:, order]
        val = val[order]
        # 找到每个唯一坐标的起始位置
        uniq, inv_idx, counts = torch.unique_consecutive(key[order], return_inverse=False, return_counts=True)
        # 组内最大：先用 -inf init，再 scatter_reduce
        out_idx = idx[:, torch.cumsum(torch.cat([torch.tensor([0], device=device), counts[:-1]]), dim=0)]
        # scatter_reduce 需要 2.0+，若不可用，退而求次：逐组取最大（简单但略慢）
        # 这里用 segment max 的简易实现
        max_vals = []
        offset = 0
        for c in counts.tolist():
            max_vals.append(val[offset:offset+c].max())
            offset += c
        max_vals = torch.stack(max_vals, dim=0)
        S_sym = torch.sparse_coo_tensor(out_idx, max_vals, (n, n), device=device).coalesce()
    else:
        S_sym = (S + ST) * 0.5
        S_sym = S_sym.coalesce()

    # 保证无自环
    if S_sym._nnz() > 0:
        ii = S_sym.indices()
        vv = S_sym.values()
        mask = ii[0] != ii[1]
        S_sym = torch.sparse_coo_tensor(ii[:, mask], vv[mask], (n, n), device=device).coalesce()

    return S_sym

def normalize_adj_sym(A, eps=1e-8):
    """A: (n,n) 对称非负邻接，返回 D^{-1/2} A D^{-1/2}"""
    deg = A.sum(dim=1)
    D_inv_sqrt = torch.pow(deg + eps, -0.5)
    D_inv_sqrt = torch.diag(D_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt

def knn_affinity_cosine(Z, k=20):
    """
    基于余弦相似度的 kNN 亲和图 (n,n)，对称化 + 行归一化。
    Z: (n, d) 节点嵌入
    """
    Z_norm = F.normalize(Z, p=2, dim=1)      # 余弦
    S = Z_norm @ Z_norm.t()                  # (n,n) 相似度 ∈[-1,1]
    n = S.size(0)

    # 对每行取 topk（排除自己）
    topk_val, topk_idx = torch.topk(S, k=k+1, dim=1)  # 含自环
    mask = torch.ones_like(S, dtype=torch.bool)
    mask.scatter_(1, topk_idx, False)  # topk 位置置 False，其余 True
    S_masked = S.masked_fill(mask, 0.0)

    # 去掉自环
    S_masked.fill_diagonal_(0.0)

    # 对称化
    S_sym = 0.5 * (S_masked + S_masked.t())

    # 非负&行归一化
    S_sym = torch.clamp(S_sym, min=0)
    row_sum = S_sym.sum(dim=1, keepdim=True) + 1e-8
    S_sym = S_sym / row_sum
    return S_sym  # (n,n)

def laplacian_from_affinity(S):
    """无向图拉普拉斯 L = D - S"""
    d = S.sum(dim=1)
    D = torch.diag(d)
    return D - S

def trace_xtAx(X, A):
    """返回 Tr(X^T A X)"""
    return torch.trace(X.T @ A @ X)

# ============ 简易 GCN/GAE ============

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, X, A_hat):
        # A_hat 需是已归一化的对称邻接（D^{-1/2} A D^{-1/2}）
        return F.relu(self.lin(A_hat @ X))

class SimpleGAE(nn.Module):
    """
    图分支：编码器得到 Z_g
    可选重构：属性 X 与 邻接 A（内积解码）
    """
    def __init__(self, in_dim, hid_dim=256, out_dim=64):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hid_dim)
        self.gcn2 = GCNLayer(hid_dim, out_dim)
        # 属性重构头（可选）
        self.dec_attr = nn.Linear(out_dim, in_dim, bias=True)

    def encode(self, X, A_hat):
        h = self.gcn1(X, A_hat)
        Z = self.gcn2(h, A_hat)              # (n, out_dim)
        return Z

    def decode_adj(self, Z):
        # 内积解码（sigmoid）
        return torch.sigmoid(Z @ Z.t())

    def decode_attr(self, Z):
        return self.dec_attr(Z)

    def forward(self, X, A_hat):
        Z = self.encode(X, A_hat)
        A_pred = self.decode_adj(Z)
        X_pred = self.decode_attr(Z)
        return Z, A_pred, X_pred

# ============ 子空间分支：自表达（因子分解核范近似） ============

class SelfExpressiveFactorized(nn.Module):
    """
    自表达层：给定 Z_s (n,d)，学习 C = U V^T ∈ R^{n×n}，并优化 ||Z^T C - Z^T||_F^2
    核范近似：0.5*beta*(||U||_F^2 + ||V||_F^2)
    支持项：
      - 可选 mask 支持（例如仅在 KNN 支持上保留系数）
      - diag(C)=0
    """
    def __init__(self, n_samples, rank=64, beta=1.0, use_mask=False):
        super().__init__()
        self.n = n_samples
        self.r = rank
        self.beta = beta
        self.use_mask = use_mask
        self.U = nn.Parameter(torch.randn(self.n, self.r) * 0.01)
        self.V = nn.Parameter(torch.randn(self.n, self.r) * 0.01)

    def forward(self, Z_s, support_mask=None):
        """
        Z_s: (n, d)
        support_mask: (n,n) in {0,1}，可选
        """
        C = self.U @ self.V.t()  # (n,n)
        # 可选：仅保留支持上的系数（例如 knn 图），不改变可微性
        if self.use_mask and support_mask is not None:
            C = C * support_mask

        # diag(C)=0
        C = C - torch.diag(torch.diag(C))

        # 自表达重构（列视角：Z^T @ C ≈ Z^T）
        Zt = Z_s.t()                  # (d, n)
        Zt_recon = Zt @ C             # (d, n)
        recon_loss = F.mse_loss(Zt_recon, Zt)

        # 核范近似（因子分解的 Fro 范）
        nuc_surrogate = 0.5 * self.beta * (self.U.pow(2).sum() + self.V.pow(2).sum())

        return C, recon_loss, nuc_surrogate

# ============ 主模型：双分支 + 共识 + 可微谱损失 ============

class MultiViewSubspaceClustering(nn.Module):
    """
    结构：
      - 图分支：GAE -> Z_g
      - 子空间分支：MLP 编码 -> Z_s -> 自表达 C_s
      - 亲和：
          S_g = kNN(Z_g) 或 softmax 相似
          S_s = |C_s| + |C_s|^T
          S = α S_s + (1-α) S_g
      - 可微谱：Tr(H^T L(S) H) + 正交约束 ||H^T H - I||_F^2
    """
    def __init__(self, n_samples, in_dim, hid_dim=256, zg_dim=64, zs_dim=64,
                 k_for_knn=20, rank_c=64, beta_nuc=1.0, use_support_for_C=True,
                 use_knn_for_Sg=True, n_clusters=10):
        super().__init__()
        self.n = n_samples
        self.k = k_for_knn
        self.use_knn_for_Sg = use_knn_for_Sg

        # 图分支
        self.gae = SimpleGAE(in_dim, hid_dim, zg_dim)

        # 子空间分支编码器（可换成更强的 MLP/Transformer）
        self.enc_s = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, zs_dim)
        )
        self.selfexpr = SelfExpressiveFactorized(n_samples, rank=rank_c,
                                                 beta=beta_nuc, use_mask=use_support_for_C)

        # 共识权重 α （用 sigmoid 把实数映射到 (0,1)）
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))  # 初始化为 0 => α≈0.5

        # 可微谱聚类：H ∈ R^{n×k}
        self.n_clusters = n_clusters
        self.H = nn.Parameter(torch.randn(self.n, n_clusters) * 0.01)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def build_Sg(self, Z_g):
        if self.use_knn_for_Sg:
            return knn_affinity_cosine(Z_g, k=self.k)   # (n,n)
        else:
            # softmax(sim) 版本
            sim = (Z_g @ Z_g.t())
            sim = sim - sim.max(dim=1, keepdim=True).values
            S = torch.softmax(sim, dim=1)
            S = 0.5 * (S + S.t())
            return S

    def forward(self, X, A_norm, support_mask=None):
        """
        X: (n,d)
        A_norm: (n,n) 已对称归一化邻接，用于 GCN
        support_mask: (n,n) in {0,1}，用于限制 C_s 支持（例如 KNN 图）
        """
        # ---- Graph view ----
        Z_g, A_pred, X_pred = self.gae(X, A_norm)  # Z_g: (n,zg_dim)

        # ---- Subspace view ----
        Z_s = self.enc_s(X)                        # (n,zs_dim)
        C_s, recon_s, nuc_s = self.selfexpr(Z_s, support_mask=support_mask)  # (n,n), scalar, scalar

        # 亲和
        S_g = self.build_Sg(Z_g)                   # (n,n)
        S_s = torch.abs(C_s) + torch.abs(C_s.t())  # (n,n)

        # 共识
        S = self.alpha * S_s + (1.0 - self.alpha) * S_g  # (n,n)

        return {
            "Z_g": Z_g, "A_pred": A_pred, "X_pred": X_pred,
            "Z_s": Z_s, "C_s": C_s,
            "S_g": S_g, "S_s": S_s, "S": S,
            "recon_s": recon_s, "nuc_s": nuc_s
        }

# ============ 损失组合（含可微谱项） ============

def gae_losses(A_true, X_true, A_pred, X_pred, w_adj=1.0, w_attr=1.0):
    """
    A_true: (n,n) 0/1 或权重
    X_true: (n,d)
    A_pred: (n,n) sigmoid 内积
    X_pred: (n,d)
    """
    # 邻接 BCE（注意正负样本不平衡，可按需加权/采样改进）
    adj_loss = F.binary_cross_entropy(A_pred, torch.clamp(A_true, 0, 1))
    # 属性重构 MSE
    attr_loss = F.mse_loss(X_pred, X_true)
    return w_adj * adj_loss + w_attr * attr_loss, adj_loss, attr_loss

def consensus_and_spectral_losses(S_g, S_s, S, H, lambda_cons=1.0, lambda_spec=1.0, ortho_w=0.1):
    """
    - 一致性：||S_g - S_s||_F^2
    - 可微谱：Tr(H^T L(S) H) + ortho_w * ||H^T H - I||_F^2
    """
    cons_loss = lambda_cons * torch.norm(S_g - S_s, p='fro')**2

    L = laplacian_from_affinity(S)
    spec_loss = lambda_spec * trace_xtAx(H, L)

    # 近似正交约束
    k = H.size(1)
    I = torch.eye(k, device=H.device, dtype=H.dtype)
    ortho_loss = ortho_w * torch.norm(H.t() @ H - I, p='fro')**2

    return cons_loss + spec_loss + ortho_loss, cons_loss, spec_loss, ortho_loss

# ============ 训练脚手架（预热 + 联合） ============

@torch.no_grad()
def build_support_mask_from_knn(X_or_Z, k=20):
    """基于输入构建 KNN 支持掩码 (n,n) in {0,1}；可用于限制 C 的支持"""
    S = knn_affinity_cosine(X_or_Z, k=k)   # 行归一和对称化后的稀疏权重（密存储）
    M = (S > 0).float()
    M.fill_diagonal_(0.0)
    return M

def train_model(
    X, A, n_clusters,
    hid_dim=256, zg_dim=64, zs_dim=64,
    k_for_knn=20, rank_c=64,
    lambda1=5.0, beta_nuc=1.0,       # 自表达项权重、核范近似权重
    lambda2=1.0, lambda3=0.5,        # 一致性、谱项权重
    lr=1e-3, weight_decay=1e-5,
    warmup_g=10, warmup_s=10, joint_epochs=100,
    device='cuda'
):
    """
    X: (n,d) float
    A: (n,n) 0/1 或权重
    返回：训练好的模型与最后的共识亲和 S、谱嵌入 H
    """
    n, d = X.shape
    X = X.to(device)
    A = A.to(device)

    # 预处理：图归一化
    A_norm = normalize_adj_sym(A)

    # 支持掩码（可用基于 X 的 knn 或基于预热后的 Z_s 再更新）
    support_mask = build_support_mask_from_knn(X, k=k_for_knn).to(device)

    model = MultiViewSubspaceClustering(
        n_samples=n, in_dim=d, hid_dim=hid_dim, zg_dim=zg_dim, zs_dim=zs_dim,
        k_for_knn=k_for_knn, rank_c=rank_c, beta_nuc=beta_nuc,
        use_support_for_C=True, use_knn_for_Sg=True, n_clusters=n_clusters
    ).to(device)

    # 优化器（全部参数先放一起，预热通过损失开关控制）
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ===== 预热：图分支 =====
    for ep in range(warmup_g):
        out = model(X, A_norm, support_mask)
        Z_g, A_pred, X_pred = out["Z_g"], out["A_pred"], out["X_pred"]
        loss_g, adj_l, attr_l = gae_losses(A, X, A_pred, X_pred, w_adj=1.0, w_attr=1.0)

        opt.zero_grad()
        loss_g.backward()
        opt.step()

    # ===== 预热：子空间分支 =====
    for ep in range(warmup_s):
        out = model(X, A_norm, support_mask)
        recon_s = out["recon_s"]
        nuc_s   = out["nuc_s"]
        loss_s = lambda1 * (recon_s + nuc_s)

        opt.zero_grad()
        loss_s.backward()
        opt.step()

    # ===== 联合训练 =====
    for ep in range(joint_epochs):
        out = model(X, A_norm, support_mask)

        # 图分支损失
        loss_gae, adj_l, attr_l = gae_losses(A, X, out["A_pred"], out["X_pred"], w_adj=1.0, w_attr=1.0)

        # 自表达损失
        recon_s = out["recon_s"]
        nuc_s   = out["nuc_s"]
        loss_selfexpr = lambda1 * (recon_s + nuc_s)

        # 一致性 + 可微谱
        loss_spec_all, loss_cons, loss_spec, loss_ortho = consensus_and_spectral_losses(
            out["S_g"], out["S_s"], out["S"], model.H,
            lambda_cons=lambda2, lambda_spec=lambda3, ortho_w=0.1
        )

        loss = loss_gae + loss_selfexpr + loss_spec_all

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep + 1) % 10 == 0:
            print(f"[Joint {ep+1:03d}] "
                  f"GAE={loss_gae.item():.4f} | SelfExpr={loss_selfexpr.item():.4f} | "
                  f"Cons={loss_cons.item():.4f} | Spec={loss_spec.item():.4f} | Ortho={loss_ortho.item():.4f} | "
                  f"alpha={model.alpha.item():.3f}")

        # 可选：每隔 T 个 epoch，更新 support_mask（例如用最新 Z_s 或 S 的 KNN）
        # if (ep + 1) % 20 == 0:
        #     with torch.no_grad():
        #         support_mask = build_support_mask_from_knn(out["Z_s"], k=k_for_knn).to(device)

    with torch.no_grad():
        final = model(X, A_norm, support_mask)
        S = final["S"]
        H = F.normalize(model.H, p=2, dim=1)

    return model, S, H





def main():
    os.chdir(WORK_DIR)
    dataset_list = [
                        'Muraro',
                        'Quake_10x_Bladder',
                        'Quake_10x_Limb_Muscle',
                        'Quake_10x_Spleen',
                        'Quake_Smart-seq2_Diaphragm',
                        'Quake_Smart-seq2_Limb_Muscle',
                        'Quake_Smart-seq2_Lung',
                        'Quake_Smart-seq2_Trachea',
                        'Adam',
                        'Romanov',
                        'Young'
                    ]
    for dataset_name in dataset_list:
        print(f"\n===== 开始实验 (数据集: {dataset_name}) =====")
        set_random_seed(SEED)
        device = DEVICE
        adata, A, A_sp, L_norm = Load_Data(dataset_name)

        label = adata.obs['cell_type1']
        print(f"已加载 {dataset_name} 数据")

        
        X = torch.tensor(adata.X, dtype=torch.float32)
        A = torch.tensor(A, dtype=torch.float32)

        model, S_consensus, H = train_model(
            X, A, n_clusters=8,
            hid_dim=256, zg_dim=64, zs_dim=64,
            k_for_knn=20, rank_c=64,
            lambda1=5.0, beta_nuc=1.0,
            lambda2=1.0, lambda3=0.5,
            lr=1e-3, weight_decay=1e-5,
            warmup_g=10, warmup_s=10, joint_epochs=100,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        from sklearn.cluster import KMeans
        y = KMeans(n_clusters=8, n_init=20).fit_predict(H.cpu().numpy())
        ari = adjusted_rand_score(label, y)
        nmi = normalized_mutual_info_score(label, y)
        ami = adjusted_mutual_info_score(label, y)
        aom = average_overlap_measure(label, y)
        print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}, AMI: {ami:.4f}, AOM: {aom:.4f}")

if __name__ == "__main__":
    main()

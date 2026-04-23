import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from sklearn.decomposition import NMF
import numpy as np
import h5py
import scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import anndata

class GAE(nn.Module):
    # Decoder
    def __init__(self, encoder, decoder):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        
        pi, mu, theta = self.decoder(z)
        
        return z, pi, mu, theta


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads = 2, dropout = 0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads = heads, concat = True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads = 1, concat = False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = self.dropout, training = self.training)
        z = self.conv2(x, edge_index)
        return z

class MLP(nn.Module):
    def __init__(self, input_feats, active):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_feats, 1)
        self.active = active

    def forward(self, z):
        z = self.linear(z)
        if self.active == 'sigmoid':
            z = torch.sigmoid(z)
        elif self.active == 'exp':
            z = torch.exp(z)
        else: # active = None
            z = torch.relu(z)
        return z

class ZINBdecoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(ZINBdecoder, self).__init__()
        """
        参数说明:
        in_feats: 输入特征的维度
        hidden_feats: 隐藏层特征的维度
        out_feats: 输出特征的维度
        heads: 注意力头的数量，默认为1
        dropout: Dropout比率，用于防止过拟合，默认为0.1
        """
        self.mlp1 = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )
        self.pi = MLP(out_feats, 'sigmoid')
        self.mu = MLP(out_feats, 'exp')
        self.theta = MLP(out_feats, 'exp')
        
        

    def forward(self, z):
        # 定义两个MLP全连接层
        
        z = self.mlp1(z)
        pi = self.pi(z)
        mu = self.mu(z)
        theta = self.theta(z)
        
        return pi, mu, theta

class LRRLayer(nn.Module):
    """改进的低秩表示(LRR)自表达层 - 无监督实现"""
    def __init__(self, n_samples, knn_graph, lambda_reg=1.0, gamma=0.1):
        """
        knn_graph: 稠密KNN邻接矩阵 (n_samples x n_samples)
        lambda_reg: 低秩正则化强度
        gamma: 块对角约束强度
        """
        super(LRRLayer, self).__init__()
        self.n_samples = n_samples
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        
        # 获取KNN图中的非零索引 (排除自环)
        nonzero_idx = np.array(np.nonzero(knn_graph))
        mask = nonzero_idx[0] != nonzero_idx[1]  # 排除自环
        self.nonzero_idx = torch.tensor(nonzero_idx[:, mask], dtype=torch.long)
        
        # 仅初始化KNN图中的边 (稀疏表示)
        num_edges = self.nonzero_idx.shape[1]
        self.C_nonzero = nn.Parameter(torch.randn(num_edges) * 0.01)
        
        # 为数值稳定性添加小常数
        self.eps = 1e-8
        
        # 创建初始块结构估计 (基于KNN图)
        self._create_initial_block_mask(knn_graph)
    
    def _create_initial_block_mask(self, knn_graph):
        """创建基于KNN图的初始块结构掩码 (无监督)"""
        # 使用KNN图作为初始相似度
        self.block_mask = torch.ones(self.n_samples, self.n_samples)
        
        # 在KNN图外的区域设置为需要惩罚
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                if knn_graph[i, j] == 0 and i != j:
                    self.block_mask[i, j] = 1.0
                else:
                    self.block_mask[i, j] = 0.0
    
    def update_block_mask(self, C):
        """动态更新块结构掩码 (基于当前C矩阵)"""
        # 使用软阈值创建块结构估计
        C_abs = torch.abs(C)
        threshold = torch.median(C_abs)
        block_mask = (C_abs < threshold).float()
        
        # 保留对角线 (自环)
        diag_mask = torch.eye(self.n_samples, device=C.device)
        block_mask = block_mask * (1 - diag_mask)  # 仅保留非对角线元素
        
        return block_mask
    
    def forward(self, Z):
        Z= Z.t()
        # 创建完整的C矩阵 (初始为0)
        C = torch.zeros(self.n_samples, self.n_samples, device=Z.device)
        C[self.nonzero_idx[0], self.nonzero_idx[1]] = self.C_nonzero
        
        # 规范化C矩阵 (每列和为1)
        col_sums = C.sum(dim=0, keepdim=True) + self.eps
        C = C / col_sums
        
        # 动态更新块结构掩码
        block_mask = self.update_block_mask(C)
        
        # 计算LRR损失
        # 1. 自表达重构损失
        Z_recon = torch.mm(Z, C)
        recon_loss = F.mse_loss(Z_recon, Z)
        
        # 2. 低秩正则化 (用Frobenius范数近似核范数)
        reg_loss = self.lambda_reg * torch.norm(C, p='fro') ** 2
        
        # 3. 块对角约束 (关键! 惩罚非块对角元素)
        offdiag_C = C * block_mask
        block_loss = self.gamma * torch.norm(offdiag_C, p='fro') ** 2
        
        return C, recon_loss, reg_loss, block_loss

def ZINB_loss(X: torch.Tensor,
            pi: torch.Tensor,
            mu: torch.Tensor,
            theta: torch.Tensor,
            eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the Zero-Inflated Negative Binomial (ZINB) loss.

    Args:
        X: Observed count matrix (N x G).
        pi: Dropout probability matrix (N x G), values in (0,1).
        mu: Mean of NB distribution matrix (N x G), >0.
        theta: Dispersion parameter matrix (N x G), >0.
        eps: Small constant for numerical stability.

    Returns:
        Scalar ZINB loss (negative log-likelihood) averaged over all entries.
    """
    # Ensure positive
    theta = torch.clamp(theta, min=eps)
    mu = torch.clamp(mu, min=eps)
    
    # Compute NB log-likelihood components
    t1 = torch.lgamma(theta + X) - torch.lgamma(theta) - torch.lgamma(X + 1)
    t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    t3 = X * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    nb_log_prob = t1 + t2 + t3

    # NB probability for zero counts
    nb_zero_log_prob = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    
    # Case X == 0: log(pi + (1-pi) * exp(nb_zero_log_prob))
    log_prob_zero = torch.log(pi + (1.0 - pi) * torch.exp(nb_zero_log_prob) + eps)
    
    # Case X > 0: log((1-pi) * exp(nb_log_prob)) = log(1-pi) + nb_log_prob
    log_prob_nonzero = torch.log(1.0 - pi + eps) + nb_log_prob
    
    # Mask for zero / non-zero
    mask_zero = (X < 0.5).type(torch.float32)
    mask_nonzero = (X >= 0.5).type(torch.float32)
    
    # Combine
    log_prob = mask_zero * log_prob_zero + mask_nonzero * log_prob_nonzero
    
    # Negative log-likelihood
    loss = -torch.sum(log_prob)
    # Optionally average
    loss = loss / X.numel()
    
    return loss

def compute_lrr_coefficient(Z, lambd=0.1, max_iter=50, tol=1e-8, rho=1.9, mu_init=1e-6, max_mu=1e6, max_time = 180):
    """
    Compute the low-rank representation coefficient matrix C for Z = Z C.
    Z: n_cells x embedding_dim feature matrix.
    Returns C: n_cells x n_cells coefficient matrix.
    """
    
    print("Compute LRR coefficient...")
    X = Z.T  # Transpose to features x cells (d x n)
    d, n = X.shape
    
    # Precompute fixed matrices
    XXt = X @ X.T  # d x d
    XXt = XXt.detach().cpu().numpy()
    X = X.detach().cpu().numpy()


    I_d = np.eye(d)
    
    # Use np.add for explicit matrix addition and handle inversion safely
    try:
        inv_I_XXt = np.linalg.inv(np.add(I_d, XXt))  # d x d
    except np.linalg.LinAlgError:
        # Add small regularization to avoid singular matrix
        inv_I_XXt = np.linalg.inv(I_d + XXt + 1e-6 * np.eye(d))
    
    # Initialize variables
    Y1 = np.zeros((d, n))
    Y2 = np.zeros((n, n))
    E = np.zeros((d, n))
    J = np.zeros((n, n))
    C = np.zeros((n, n))
    mu = mu_init
    import time
    start_time = time.time()
    for iteration in range(max_iter):
        # Update C using Woodbury formula
        print(f"Iteration {iteration}...")
        term1 = X.T @ (X - E)  # n x n
        term2 = (1 / mu) * (X.T @ Y1)  # n x n
        term3 = J  # n x n
        term4 = - (1 / mu) * Y2  # n x n
        RHS = term1 + np.add(term2, np.add(term3, term4))  # n x n
        
        XR = X @ RHS  # d x n
        tmp = inv_I_XXt @ XR  # d x n
        C = RHS - (X.T @ tmp)  # n x n
        
        # Update E
        A = X - (X @ C) + (Y1 / mu)  # d x n
        for i in range(n):
            a = A[:, i]
            norm_a = np.linalg.norm(a)
            if norm_a > lambd / mu:
                E[:, i] = (1 - (lambd / mu) / norm_a) * a
            else:
                E[:, i] = 0
        
        # Update J with singular value thresholding
        M = C + (Y2 / mu)  # n x n
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        S_thresh = np.maximum(S - (1 / mu), 0)
        J = U @ np.diag(S_thresh) @ Vt
        
        # Update multipliers
        Y1 = Y1 + mu * (X - (X @ C) - E)
        Y2 = Y2 + mu * (C - J)
        
        # Update mu
        mu = min(max_mu, rho * mu)
        
        # Check convergence
        res1 = np.max(np.abs(X - (X @ C) - E))
        res2 = np.max(np.abs(C - J))
        if res1 < tol and res2 < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break
        else:
            print(f"res1 = {res1} , res2 = {res2}")

        if time.time() - start_time > max_time:
            print(f"Time limit reached after {max_time} seconds.")
            break
    else:
        print(f"Did not converge after {max_iter} iterations.")
    
    return C


class SparseLRRLayer(nn.Module):
    """改进的低秩表示(LRR)自表达层 - 稀疏C实现"""
    def __init__(self,  knn_graph, lambda_reg=1.0, gamma=0.1, eps=1e-8):
        super().__init__()
        self.n_samples = knn_graph.shape[0]
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.eps = eps

        # 取 KNN 图边（排除自环）
        if isinstance(knn_graph, torch.Tensor):
            nz = torch.nonzero(knn_graph, as_tuple=False).t()  # (2, E)
            mask = nz[0] != nz[1]
            row_idx = nz[0][mask]
            col_idx = nz[1][mask]
        else:
            nonzero_idx = np.array(np.nonzero(knn_graph))
            mask = nonzero_idx[0] != nonzero_idx[1]
            row_idx = torch.tensor(nonzero_idx[0, mask], dtype=torch.long)
            col_idx = torch.tensor(nonzero_idx[1, mask], dtype=torch.long)

        # 注册成 buffer，方便随模型移动到对应 device/dtype
        self.register_buffer("row_idx", row_idx)   # 源 i
        self.register_buffer("col_idx", col_idx)   # 目标 j
        self.num_edges = self.row_idx.numel()

        # 仅在KNN边上定义可训练参数
        self.C_nonzero = nn.Parameter(torch.randn(self.num_edges) * 0.01)

    @torch.no_grad()
    def _median_abs_on_edges(self, vals):
        """仅基于非零边上的系数计算阈值（稀疏友好）"""
        # 与原实现的“对稠密C取中位数”相比，这里对‘非零边’取中位数，更符合直觉
        return vals.abs().median()

    def _build_sparse_C(self, values, device):
        """根据边索引与边值构造稀疏 COO 的 C（n×n）"""
        idx = torch.stack([self.row_idx.to(device), self.col_idx.to(device)], dim=0)  # (2, E)
        C_sp = torch.sparse_coo_tensor(idx, values, size=(self.n_samples, self.n_samples), device=device)
        return C_sp.coalesce()

    def forward(self, Z):
        # 与原逻辑一致：在层内做一次转置以使用 Z @ C
        Z = Z.t()  # 之后形状为 (feat_dim, n_samples)
        device = Z.device

        # 1) 按列归一化：计算每列的和（仅基于边）
        #    col_sums[j] = sum_i C_ij
        col_sums = torch.zeros(self.n_samples, device=device, dtype=Z.dtype)
        col_sums.index_add_(0, self.col_idx, self.C_nonzero)  # 仅累加边上的值
        col_sums = col_sums + self.eps  # 避免除零

        # 每条边的归一化系数：C_ij <- C_ij / col_sums[j]
        norm_vals = self.C_nonzero / col_sums[self.col_idx]

        # 2) 稀疏 C（COO）
        C_sp = self._build_sparse_C(norm_vals, device=device)  # 稀疏、仅边位置非零

        # 3) 自表达重构：Z_recon = Z @ C   （把右乘稀疏转成左乘稀疏，以用稀疏mm）
        #    (Z @ C) = (C^T @ Z^T)^T
        Z_recon_T = torch.sparse.mm(C_sp.transpose(0, 1), Z.transpose(0, 1))  # (n, feat_dim)
        Z_recon = Z_recon_T.transpose(0, 1)  # (feat_dim, n)

        recon_loss = F.mse_loss(Z_recon, Z)

        # 4) 低秩近似正则（与原代码一致：Fro 范数近似核范数）
        #    稀疏 Fro^2 就是非零边上平方和
        reg_loss = self.lambda_reg * (norm_vals.pow(2).sum())

        # 5) 块对角约束（稀疏版）：用当前非零边的 |C_ij| 与中位数比较，惩罚较小值
        #    注：原版对“稠密C”的中位数做阈值，这里改为只在边上做（避免密化）
        threshold = self._median_abs_on_edges(norm_vals.detach())
        edge_mask = (norm_vals.abs() < threshold).to(Z.dtype)  # E 维
        block_loss = self.gamma * ( (norm_vals * edge_mask).pow(2).sum() )

        return C_sp, recon_loss, reg_loss, block_loss




class SparseLRRLayer_laplace(nn.Module):
    """低秩表示(LRR)自表达层 - 稀疏C + 拉普拉斯正则 (替代块对角正则)"""
    def __init__(self, knn_graph, lambda_reg=1.0, gamma=0.1, eta=1.0, eps=1e-8):
        """
        参数:
            knn_graph: (n x n) KNN邻接 (numpy或tensor, 非零即视为边; 可非对称)
            lambda_reg: Frobenius(L2) 正则系数 (原 lrr_lambda)
            gamma:      (保留接口, 如需兼容原代码; 不再用于 block 正则)
            eta:        拉普拉斯正则系数 (R_lap = eta * tr(C^T L C))
            eps:        列归一时的数值稳定项
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.eta = gamma
        self.eps = eps

        # ---- 读取图并对称化 ----
        if isinstance(knn_graph, torch.Tensor):
            A = (knn_graph != 0).to(torch.float32)  # 0/1
        else:
            A = torch.tensor((knn_graph != 0).astype(np.float32))
        n = A.shape[0]
        self.n_samples = n

        # 对称化: A_sym = 1[(A + A^T) > 0]
        A_sym = ((A + A.t()) > 0).to(torch.float32)

        # 缓存邻接的非零索引 (不含自环)
        nz = torch.nonzero(A_sym, as_tuple=False).t()
        mask = nz[0] != nz[1]
        row_idx = nz[0][mask]
        col_idx = nz[1][mask]

        # 注册 buffer: KNN 边 (用于参数化 C)
        self.register_buffer("row_idx", row_idx)   # 源 i
        self.register_buffer("col_idx", col_idx)   # 目标 j
        self.num_edges = self.row_idx.numel()

        # 仅在KNN边上定义可训练参数 (初始化小随机数)
        self.C_nonzero = nn.Parameter(torch.randn(self.num_edges) * 0.01)

        # ---- 构造稀疏拉普拉斯 L = D - A_sym (作为 buffer 存) ----
        deg = A_sym.sum(dim=1)  # (n,)
        # 稀疏 A_sym
        A_nz = torch.nonzero(A_sym, as_tuple=False).t()
        A_vals = torch.ones(A_nz.shape[1], dtype=torch.float32)
        A_sp = torch.sparse_coo_tensor(A_nz, A_vals, size=(n, n)).coalesce()

        # 稀疏 D (对角)
        D_idx = torch.arange(n, dtype=torch.long)
        D_sp = torch.sparse_coo_tensor(
            torch.stack([D_idx, D_idx], dim=0),
            deg.to(torch.float32),
            size=(n, n)
        ).coalesce()

        L_sp = (D_sp - A_sp).coalesce()
        # 缓存为 buffer，便于随模型移动到对应 device
        self.register_buffer("L_indices", L_sp.indices())
        self.register_buffer("L_values",  L_sp.values())
        self.register_buffer("L_size",    torch.tensor(L_sp.size()))

    @torch.no_grad()
    def _median_abs_on_edges(self, vals):
        # 兼容保留（如果外部还引用）, 实际不再用于正则
        return vals.abs().median()

    def _build_sparse_C(self, values, device):
        """根据边索引与边值构造稀疏 COO 的 C（n×n）"""
        idx = torch.stack([self.row_idx.to(device), self.col_idx.to(device)], dim=0)  # (2, E)
        C_sp = torch.sparse_coo_tensor(idx, values, size=(self.n_samples, self.n_samples), device=device)
        return C_sp.coalesce()

    def forward(self, Z):
        """
        返回:
            C_sp:        稀疏自表达矩阵 (COO)
            recon_loss:  自表达重构 MSE
            reg_loss:    Fro(L2) 正则 (仅边上)
            lap_loss:    拉普拉斯正则 eta * tr(C^T L C)
        """
        # 与原逻辑一致：在层内做一次转置以使用 Z @ C
        Z = Z.t()  # (feat_dim, n)
        device = Z.device

        # ---- 1) 列归一 (仅基于边) ----
        col_sums = torch.zeros(self.n_samples, device=device, dtype=Z.dtype)
        col_sums.index_add_(0, self.col_idx, self.C_nonzero)  # sum_i C_ij
        col_sums = col_sums + self.eps

        norm_vals = self.C_nonzero / col_sums[self.col_idx]

        # ---- 2) 稀疏 C (COO) ----
        C_sp = self._build_sparse_C(norm_vals, device=device)  # (n, n) sparse

        # ---- 3) 自表达重构: Z @ C （用稀疏mm的等价式）----
        Z_recon_T = torch.sparse.mm(C_sp.transpose(0, 1), Z.transpose(0, 1))  # (n, d)
        Z_recon = Z_recon_T.transpose(0, 1)  # (d, n)
        recon_loss = F.mse_loss(Z_recon, Z)

        # ---- 4) Fro(L2) 正则 (仅边上) ----
        reg_loss = self.lambda_reg * (norm_vals.pow(2).sum())

        # ---- 5) 拉普拉斯正则: R_lap = eta * tr(C^T L C) ----
        # 构造稀疏 L
        L_sp = torch.sparse_coo_tensor(
            self.L_indices.to(device),
            self.L_values.to(device),
            size=tuple(self.L_size.tolist())
        ).coalesce()

        # 计算 tr(C^T L C) = <C, L C>
        # 注意: torch.sparse.mm 支持 稀疏 @ 稠密，故将 C 转成稠密一次性计算（规模较大时可做分块/近似）
        C_den = C_sp.to_dense()                       # (n, n)
        LC_den = torch.sparse.mm(L_sp, C_den)         # (n, n)
        lap_loss = self.eta * (C_den * LC_den).sum()  # Frobenius 内积 = 元素乘积求和


        return C_sp, recon_loss, reg_loss, lap_loss

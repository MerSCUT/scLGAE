import torch

def _ensure_dense(M: torch.Tensor) -> torch.Tensor:
    return M.to_dense() if M.is_sparse else M

def _symmetrize_zero_diag(W: torch.Tensor) -> torch.Tensor:
    W = (W + W.T) * 0.5
    i = torch.arange(W.size(0), device=W.device)
    W[i, i] = 0
    return torch.clamp(W, min=0)

def _row_normalize(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    rs = M.sum(dim=1, keepdim=True).clamp_min(eps)
    return M / rs

def _knn_stochastic(W: torch.Tensor, K: int, diag_self: float) -> torch.Tensor:
    """
    从相似度 W 生成“随机游走”矩阵。
    - 若 diag_self=0.5：对角占 0.5，其余权重分给 KNN 并归一到 0.5；
    - 若 diag_self=0.0：仅 KNN，整行和为 1，且对角为 0（用于 S）。
    """
    n = W.size(0)
    device, dtype = W.device, W.dtype
    assert 0.0 <= diag_self < 1.0
    k_eff = min(K, n - 1)

    # 排除自身拿 top-k
    W2 = W.clone()
    idx_self = torch.arange(n, device=device)
    W2[idx_self, idx_self] = float("-inf")
    vals, idx = torch.topk(W2, k=k_eff, dim=1)  # [n, k]

    # 仅保留 top-k
    S = torch.zeros_like(W)
    S.scatter_(1, idx, torch.clamp(vals, min=0))

    # 行归一
    S = _row_normalize(S)

    if diag_self > 0:
        # 将非对角权重缩放到 (1 - diag_self)，对角填 diag_self
        off = S.clone()
        off[idx_self, idx_self] = 0
        off_sum = off.sum(dim=1, keepdim=True).clamp_min(1e-12)
        off = off * ((1.0 - diag_self) / off_sum)
        S = off
        S[idx_self, idx_self] = diag_self
    else:
        # 保持对角为 0，总和为 1
        S[idx_self, idx_self] = 0

    return S

def snf(sim_list, K: int = 20, T: int = 15, self_weight: float = 0.5,
        clip_negative: bool = True):
    """
    sim_list: [W1, W2, ...] 每个为 n×n 相似度 (dense/sparse 均可传入)，要求非负、基本对称
              若仅有 L_sym，可传 [I-L1, I-L2, ...] 作为近似。
    返回: fused (n×n dense), P_list (长度为视图数的列表)
    """
    assert len(sim_list) >= 2, "至少需要两个视图做融合"
    sims = []
    for W in sim_list:
        Wd = _ensure_dense(W).clone()
        if clip_negative:
            Wd = torch.clamp(Wd, min=0)
        Wd = _symmetrize_zero_diag(Wd)
        sims.append(Wd)

    # 初始 P^{(v)} 和 S^{(v)}
    P = []
    S = []
    for W in sims:
        P.append(_knn_stochastic(W, K=K, diag_self=self_weight))  # 行和=1，含自环
        S.append(_knn_stochastic(W, K=K, diag_self=0.0))          # 行和=1，无自环

    m = len(sims)
    for _ in range(T):
        P_sum = sum(P)
        P_next = []
        for v in range(m):
            others = (P_sum - P[v]) * (1.0 / (m - 1))
            Mv = S[v] @ others @ S[v].T          # 信息扩散
            Mv = _row_normalize(Mv)              # 行归一
            Mv = _knn_stochastic(Mv, K=K, diag_self=self_weight)  # 重新稀疏化+自环
            P_next.append(Mv)
        P = P_next

    fused = sum(P) * (1.0 / m)
    fused = _symmetrize_zero_diag(fused) + torch.eye(fused.size(0), device=fused.device, dtype=fused.dtype) * 1e-12
    return fused, P

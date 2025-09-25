import numpy as np

def _prep_affinity(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    规范化亲和矩阵：对称化、去自环、非负裁剪，并以 Frobenius 范数归一化到 ~1。
    """
    W = np.asarray(W, dtype=float)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    W = np.maximum(W, 0.0)
    fro = np.linalg.norm(W, 'fro')
    if fro > eps:
        W = W / fro
    return W

def _eigengap_sym_laplacian(W: np.ndarray, k: int, eps: float = 1e-12):
    """
    计算对称归一化拉普拉斯 L_sym = I - D^{-1/2} W D^{-1/2} 的第 k 与第 k+1
    个最小特征值之差（0-based：取 vals[k] - vals[k-1]）。
    返回 eigengap 与全部特征值。
    """
    n = W.shape[0]
    # 度与防零
    d = np.clip(W.sum(axis=1), eps, None)
    inv_sqrt_d = 1.0 / np.sqrt(d)
    Dm = inv_sqrt_d[:, None] * W * inv_sqrt_d[None, :]
    Lsym = np.eye(n) - Dm

    # 对称实矩阵 => eigh
    vals = np.linalg.eigh(Lsym)[0]
    vals = np.clip(vals, 0.0, None)  # 数值稳定
    vals.sort()

    if k < 1 or k >= len(vals):
        raise ValueError(f"k must be in [1, n-1], got k={k}, n={len(vals)}")
    gap = float(vals[k] - vals[k-1])
    return gap, vals

def fusion_weight_from_eigengap(W_Z: np.ndarray, W_C: np.ndarray, k: int, eps: float = 1e-12):
    """
    基于谱间隙自适应计算融合权重 alpha：
        alpha = gap_Z / (gap_Z + gap_C)
    参数
    ----
    W_Z, W_C : 亲和矩阵 (n, n)，不要求已对称或归一化
    k        : 目标簇数（用于 eigengap 的位置）
    返回
    ----
    alpha : float ∈ [0,1]
    info  : dict（包含各自 gap 与特征值，便于调试/可视化）
    """
    WZ = _prep_affinity(W_Z, eps=eps)
    WC = _prep_affinity(W_C, eps=eps)

    gapZ, valsZ = _eigengap_sym_laplacian(WZ, k=k, eps=eps)
    gapC, valsC = _eigengap_sym_laplacian(WC, k=k, eps=eps)

    denom = gapZ + gapC
    if denom <= eps or not np.isfinite(denom):
        alpha = 0.5  # 退化情况给等权
    else:
        alpha = float(np.clip(gapZ / denom, 0.0, 1.0))

    info = {
        "gap_Z": gapZ,
        "gap_C": gapC,
        "eigvals_Z": valsZ,
        "eigvals_C": valsC,
        "alpha": alpha,
    }
    return alpha, info

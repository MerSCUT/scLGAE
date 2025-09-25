import torch
import torch.nn.functional as F

@torch.no_grad()
def _symmetrize_and_clip(C):
    # 保证非负与对称性（输入若已满足，这步等价于 no-op）
    C = torch.clamp(C, min=0)
    C = 0.5 * (C + C.T)
    return C

def symmetric_nmf(C, n_clusters, iters=500, tol=1e-5, device=None,
                  seed=42, normalize_rows_every=0, eps=1e-9, verbose=False):
    """
    对称 NMF:  min_{H >= 0} || C - H H^T ||_F^2
    输入:
        C: [n, n] 非负对称张量（相似度/亲和矩阵）
        n_clusters: 目标簇数 k
        iters: 最大迭代次数
        tol: 相对改变量阈值（早停）
        device: 'cuda' 或 'cpu'（None 则自动）
        seed: 随机种子
        normalize_rows_every: 每隔多少步做一次行归一（0 关闭）
        eps: 数值稳定项，避免除零
        verbose: 是否打印收敛日志
    返回:
        H: [n, k] 非负因子
        labels: [n]，按行 argmax 得到的簇标签
        losses: 训练过程中目标值轨迹
    """
    if device is None:
        device = C.device if isinstance(C, torch.Tensor) else ('cuda' if torch.cuda.is_available() else 'cpu')
    if not isinstance(C, torch.Tensor):
        C = torch.tensor(C, dtype=torch.float32, device=device)
    else:
        C = C.to(device=device, dtype=torch.float32)

    torch.manual_seed(seed)
    C = _symmetrize_and_clip(C)

    n = C.shape[0]
    k = int(n_clusters)

    # 初始化 H >= 0（非负随机 + 行归一）
    H = torch.rand((n, k), device=device)
    H = H / (H.sum(dim=1, keepdim=True) + eps)

    losses = []
    prev_loss = None

    for t in range(1, iters + 1):
        # 乘法更新: H <- H * (CH) / (H H^T H)
        CH = C @ H                                       # [n,k]
        HHTH = (H @ H.T @ H)                             # [n,k]
        H = H * (CH / (HHTH + eps))
        H = torch.clamp(H, min=0)

        # 可选：周期性按行归一，提升稳定性和可解释性（行向量视作软簇权重）
        if normalize_rows_every and (t % normalize_rows_every == 0):
            H = H / (H.sum(dim=1, keepdim=True) + eps)

        # 计算目标值（可用于早停）
        if t == 1 or t % 5 == 0 or t == iters:
            R = H @ H.T
            loss = torch.norm(C - R, p='fro')**2
            losses.append(loss.item())

            if verbose and (t % 50 == 0 or t in (1, iters)):
                print(f"[SymNMF] iter={t:4d}  loss={loss.item():.6f}")

            # 早停：相对改变量足够小
            if prev_loss is not None:
                rel_improve = abs(prev_loss - loss.item()) / (prev_loss + eps)
                if rel_improve < tol:
                    if verbose:
                        print(f"[SymNMF] early-stop at iter={t}, Δrel={rel_improve:.2e}")
                    break
            prev_loss = loss.item()

    # 行归一（最终一次，便于阐释/聚类）
    H = H / (H.sum(dim=1, keepdim=True) + eps)

    # 基于 H 的软→硬分配：按行 argmax
    labels = torch.argmax(H, dim=1).detach().cpu().numpy()
    return H.detach(), labels, losses
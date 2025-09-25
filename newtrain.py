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
from sklearn.metrics import explained_variance_score, normalized_mutual_info_score as NMI
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
import anndata

from data import *
from model import *
from utils import *
from config import *
from clustering import *
from multiview import knn_self_tuning_graph
from SNF import snf
from GAEval import *
from symmetricNMF import symmetric_nmf
from FusionWeight import fusion_weight_from_eigengap

def train_gae_model_new(adata, A, L_norm, device, params = default_params): # 修改为字典传参
    """
    训练GAT-GAE模型 (修改2: 移除流形损失)
    参数:
        X: 表达矩阵
        A: 邻接矩阵
        L_norm: 归一化拉普拉斯矩阵 (为兼容性保留，但不再用于损失计算)
        device: 计算设备
        params: 包含所有模型超参数的字典
    返回:
        model: 训练好的模型
        Z: 细胞嵌入
        train_loss: 训练损失记录
    """
    # 从params字典中提取参数
    gat_hidden_channels = params['gae_params']['gat_hidden_channels']
    gat_out_channels = params['gae_params']['gat_out_channels']
    head_num = params['gae_params']['head_num']
    zinb_hidden_channels = params['gae_params']['zinb_hidden_channels']
    alpha = params['gae_params']['alpha']
    lr = params['gae_params']['lr']
    epochs = params['gae_params']['epochs']
    beta = params['gae_params']['beta']

    ''' 
    训练损失函数:
    重构损失 + alpha * ZINB损失 + beta * 流形损失
    '''
    
    print("\n===== 开始训练GAT-GAE模型 =====")
    # 转换为PyTorch张量
    X = adata.X
    X = torch.tensor(X, dtype=torch.float32).to(device)
    A = torch.tensor(A, dtype=torch.float32).to(device)
    # L_norm 不再用于损失计算
    L_norm = torch.tensor(L_norm, dtype=torch.float32).to(device)
    
    # 将邻接矩阵转换为稀疏张量
    A_sp = torch.sparse_coo_tensor(
        indices=torch.nonzero(A).t(),
        values=A[A != 0],
        size=A.size()
    )
    # 初始化模型
    encoder = GAT(X.shape[1], gat_hidden_channels, gat_out_channels,
                 heads=head_num, dropout=0.1).to(device)
    decoder = ZINBdecoder(gat_out_channels, zinb_hidden_channels, out_feats=16).to(device)
    model = GAE(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 训练过程
    train_loss = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        Z, pi, mu, theta = model(X, A_sp)
        # 检查输出是否包含NaN
        if check_nan(Z, "嵌入Z") or check_nan(pi, "pi") or check_nan(mu, "mu") or check_nan(theta, "theta"):
            print("检测到NaN值，重新初始化模型并使用更小的学习率")
            # 重新初始化模型
            encoder = GAT(X.shape[1], gat_hidden_channels, gat_out_channels,
                         heads=head_num, dropout=0.1).to(device)
            decoder = ZINBdecoder(gat_out_channels, zinb_hidden_channels, out_feats=16).to(device)
            model = GAE(encoder, decoder).to(device)
            # 使用更小的学习率
            optimizer = optim.Adam(model.parameters(), lr=lr/2, weight_decay=1e-5)
            # 重新前向传播
            Z, pi, mu, theta = model(X, A_sp)


        hatA = F.relu(Z @ Z.t())
        # 1. 重构损失 (邻接矩阵重构)
        bceloss = torch.nn.BCEWithLogitsLoss()
        reconstruction_loss = bceloss(hatA, A)
        # 2. ZINB 损失 (表达矩阵重构)
        if params['ablation_params']['zinb_loss']:
            zinb_loss = ZINB_loss(X, pi, mu, theta)
        else:
            zinb_loss = torch.tensor(0.0)


        
        
        L_norm_Z = torch.matmul(L_norm, Z)
        if params['ablation_params']['manifold_loss']:
            manifold_loss = (Z * L_norm_Z).sum()
        else:
            manifold_loss = torch.tensor(0.0)

        '''
            # 检查损失是否为NaN
        if torch.isnan(reconstruction_loss) or torch.isnan(zinb_loss) or torch.isnan(manifold_loss): # 修改检查条件
            print("检测到NaN损失，跳过此批次")
            continue
        '''

        loss = reconstruction_loss  + alpha * zinb_loss + beta * manifold_loss 
        L_Z = alpha * zinb_loss 
        L_M = beta * manifold_loss
        # 检查总损失是否为NaN
        if torch.isnan(loss):
            print("总损失为NaN，终止训练")
            break
        # 反向传播和优化
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss.append(loss.item())
        
        # 打印损失
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'Epoch: {epoch+1:03d}, '
                  f'Re_Loss: {reconstruction_loss.item():.4f}, '
                  f'ZINB_Loss: {L_Z.item():.4f}, ' 
                  f'Manifold_Loss: {L_M.item():.4f}, '
                  f'Total_Loss: {loss.item():.4f}')
    # 确保最终输出没有NaN
    model.eval()
    with torch.no_grad():
        Z, pi, mu, theta = model(X, A_sp)
        if check_nan(Z, "最终嵌入Z"):
            print("警告: 最终嵌入包含NaN值，用零替换")
            Z = torch.where(torch.isnan(Z), torch.zeros_like(Z), Z)
    print("GAT-GAE模型训练完成")
    return model, Z, train_loss, A_sp

def train_lrr_layer_new(Z, A_sp, device, params):
    """
    单独训练LRR层，保持GAE参数冻结
    参数:
        
        params: 包含所有超参数的字典
    返回:
        lrr_layer: 训练好的LRR层
        C: 自表达系数矩阵
        train_loss: 训练损失记录
    """
    # 从params字典中提取参数
    lrr_lambda = params['lrr_params']['lrr_lambda']
    lrr_gamma = params['lrr_params']['lrr_gamma']
    lr = params['lrr_params']['lr']
    epochs = params['lrr_params']['epochs']
    
    print("\n===== 开始训练LRR层 =====")
    

        
    # 初始化LRR层
    n_cell = Z.shape[0]
    lrr_layer = SparseLRRLayer_laplace( A_sp.to_dense().cpu().numpy(),
                        lambda_reg=lrr_lambda, gamma=lrr_gamma).to(device)
                        
    # 只优化LRR层参数
    optimizer = optim.Adam(lrr_layer.parameters(), lr=lr, weight_decay=1e-5)
    
    # 训练历史记录
    train_loss = []
    lrr_recon_losses = []
    lrr_reg_losses = []
    lrr_block_losses = []

    
    for epoch in range(epochs):
        lrr_layer.train()
        optimizer.zero_grad()
        
        # 前向传播: LRR
        C, lrr_recon_loss, lrr_reg_loss, lrr_block_loss = lrr_layer(Z)
        
        # 检查LRR损失是否为NaN
        if torch.isnan(lrr_recon_loss) or torch.isnan(lrr_reg_loss) or torch.isnan(lrr_block_loss):
            print("LRR损失包含NaN，跳过此批次")
            continue
            
        # 总LRR损失
        if params['ablation_params']['low_rank_reg']:
            lrr_reg_loss = lrr_reg_loss
        else:
            lrr_reg_loss = torch.tensor(0.0)
        if params['ablation_params']['block_reg']:
            lrr_block_loss = lrr_block_loss
        else:
            lrr_block_loss = torch.tensor(0.0)

        new_loss, _ = nuclear_norm_randomized(C, rank = 40, n_power_iter = 2)
        new_reg = 1e-1
        loss = lrr_recon_loss + lrr_reg_loss + lrr_block_loss + new_reg * new_loss
        New_L = new_reg * new_loss
        # 检查总损失是否为NaN
        if torch.isnan(loss):
            print("总损失为NaN，终止训练")
            break
            
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(lrr_layer.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 记录损失
        train_loss.append(loss.item())
        lrr_recon_losses.append(lrr_recon_loss.item())
        lrr_reg_losses.append(lrr_reg_loss.item())
        lrr_block_losses.append(lrr_block_loss.item())
        
        # 打印损失
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'Epoch: {epoch+1:03d}, '
                  f'LRR_Recon: {lrr_recon_loss.item():.4f}, '
                  f'LRR_Reg: {lrr_reg_loss.item():.4f}, '
                  f'LRR_Block: {lrr_block_loss.item():.4f}, '
                  f'New_Loss: {New_L.item():.4f}, '
                  f'Total_Loss: {loss.item():.4f}')
    
    # 获取最终的自表达系数矩阵
    lrr_layer.eval()
    with torch.no_grad():
        C, _, _, _ = lrr_layer(Z)
        C = C.to_dense().cpu().numpy()
        # 检查C矩阵是否包含NaN
        if np.isnan(C).any():
            print("警告: 最终C矩阵包含NaN值，用零替换")
            C = np.nan_to_num(C, nan=0.0)
            
    
        
    return lrr_layer, C, train_loss, lrr_recon_losses, lrr_reg_losses, lrr_block_losses


def run_experiment_new(dataset_name, dataset_dir, params = default_params):
    '''
    '''
    print(f"\n===== 开始实验 (数据集: {dataset_name}) =====")
    set_random_seed(SEED)
    device = DEVICE
    adata, A, A_sp, L_norm = Load_Data(dataset_name)
    
    label = adata.obs['cell_type1']
    print(f"已加载 {dataset_name} 数据")
    
    
    # train GAE
    GAE_model, Z, train_loss, A_sp_gae = train_gae_model_new(adata, A, L_norm, device, params)
    GAE_model.eval()
    for param in GAE_model.parameters():
        param.requires_grad = False

    lrr_layer, C, train_loss, lrr_recon_losses, lrr_reg_losses, lrr_block_losses = train_lrr_layer_new(Z, A_sp_gae, device, params)
    
    n_clusters = len(np.unique(label))
    spec_labels = safe_spectral_clustering(C, n_clusters, random_state=42)
    ari, nmi, ami, aom = evaluate_clustering(spec_labels, label)
    # 设置输出颜色为黄色
    print("\033[93m", end="")  # 设置黄色输出
    print(f"数据集 {dataset_name} - ARI: {ari:.4f}, NMI: {nmi:.4f}, AMI: {ami:.4f}, AOM: {aom:.4f}")
    print("\033[0m", end="")  # 重置默认颜色

    return ari, nmi, ami, aom

def abl1_XtoLRR(dataset_name, dataset_dir, params = default_params):
    '''
    '''
    print(f"\n===== 开始实验 (数据集: {dataset_name}) =====")
    set_random_seed(SEED)
    device = DEVICE
    adata, A, A_sp, L_norm = Load_Data(dataset_name)
    label = adata.obs['cell_type1']
    print(f"已加载 {dataset_name} 数据")

    Z = torch.tensor(adata.X, dtype=torch.float32).to(device)
    A_sp = torch.sparse_coo_tensor(
        indices=torch.nonzero(torch.tensor(A)).t(),
        values=torch.tensor(A)[torch.tensor(A) != 0],
        size=torch.tensor(A).size()
    )
    
    lrr_layer, C, train_loss, lrr_recon_losses, lrr_reg_losses, lrr_block_losses = train_lrr_layer_new(Z, A_sp, device, params)

    n_clusters = len(np.unique(label))
    spec_labels = safe_spectral_clustering(C, n_clusters, random_state=42)
    ari, nmi, ami, aom = evaluate_clustering(spec_labels, label)
    # 设置输出颜色为黄色
    print("\033[93m", end="")  # 设置黄色输出
    print(f"数据集 {dataset_name} - ARI: {ari:.4f}, NMI: {nmi:.4f}, AMI: {ami:.4f}, AOM: {aom:.4f}")
    print("\033[0m", end="")  # 重置默认颜色

    return ari, nmi, ami, aom
    
def ab2_GAEreconAfoLRR(dataset_name, dataset_dir, params = default_params):
    '''
    '''
    print(f"\n===== 开始ab2实验 (数据集: {dataset_name}) =====")
    set_random_seed(SEED)
    device = DEVICE
    adata, A, A_sp, L_norm = Load_Data(dataset_name)
    
    label = adata.obs['cell_type1']
    print(f"已加载 {dataset_name} 数据")
    
    
    # train GAE
    GAE_model, Z, train_loss, A_sp_gae = train_gae_model_new_with_val(adata, A, L_norm, device, params, patience = 5)
    GAE_model.eval()
    for param in GAE_model.parameters():
        param.requires_grad = False
    # 用 Z 重构新的邻接矩阵
    # 用 Z 的内积相似度 + 阈值过滤得到新的邻接矩阵
    with torch.no_grad():
        # 计算 Z 的内积相似度矩阵
        Z_norm = F.normalize(Z, p=2, dim=1)  # L2 归一化
        similarity_matrix = torch.mm(Z_norm, Z_norm.t())  # 内积相似度
        
        # 设置阈值过滤（可以根据需要调整阈值）
        # 动态设置阈值：找到使得没有全0行的最小阈值
        n_nodes = similarity_matrix.size(0)
        
        # 从高到低尝试不同的阈值，找到临界点
        thresholds = torch.linspace(0.9, 0.1, 81)  # 从0.9到0.1，步长0.01
        
        for thresh in thresholds:
            A_temp = (similarity_matrix > thresh).float()
            A_temp.fill_diagonal_(0)  # 移除自环
            
            # 检查是否有全0行
            row_sums = A_temp.sum(dim=1)
            if (row_sums == 0).sum() == 0:  # 没有全0行
                threshold = thresh.item()
                print(f"为数据集 {dataset_name} 设置阈值: {threshold:.3f}")
                break
        else:
            # 如果所有阈值都会产生全0行，使用最小阈值
            threshold = 0.1
            print(f"警告：数据集 {dataset_name} 使用最小阈值: {threshold}")
        #threshold = 0.5  # 相似度阈值
        A_new = (similarity_matrix > threshold).float()
        
        # 移除自环
        A_new.fill_diagonal_(0)
        
        
        # 转换为稀疏张量
        indices = torch.nonzero(A_new, as_tuple=False).t()
        values = A_new[indices[0], indices[1]]
        A_sp_gae = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=A_new.size(),
            device=device
        ).coalesce()
    
    lrr_layer, C, train_loss, lrr_recon_losses, lrr_reg_losses, lrr_block_losses = train_lrr_layer_new(Z, A_sp_gae, device, params)
    
    n_clusters = len(np.unique(label))
    spec_labels = safe_spectral_clustering(C, n_clusters, random_state=42)
    ari, nmi, ami, aom = evaluate_clustering(spec_labels, label)
    # 设置输出颜色为黄色
    print("\033[93m", end="")  # 设置黄色输出
    print(f"数据集 {dataset_name} - ARI: {ari:.4f}, NMI: {nmi:.4f}, AMI: {ami:.4f}, AOM: {aom:.4f}")
    print("\033[0m", end="")  # 重置默认颜色

    '''
    # NMF Clustering
    H, nmf_labels, _ = symmetric_nmf(C, n_clusters)
    ari_nmf, nmi_nmf, ami_nmf, aom_nmf = evaluate_clustering(nmf_labels, label)
    print("\033[95m", end="")  # 设置紫色输出
    print(f"NMF聚类结果 - 数据集 {dataset_name} - ARI: {ari_nmf:.4f}, NMI: {nmi_nmf:.4f}, AMI: {ami_nmf:.4f}, AOM: {aom_nmf:.4f}")
    print("\033[0m", end="")  # 重置默认颜色

    # GAE + UMAP + Kmeans 结果:
    # 对 Z 进行 UMAP 降维
    import umap
    
    # 使用 UMAP 进行降维
    umap_reducer = umap.UMAP(n_components=50, random_state=42, n_neighbors=15, min_dist=0.1)
    Z_umap = umap_reducer.fit_transform(Z.cpu().numpy())
    
    print(f"UMAP降维完成: {Z.shape} -> {Z_umap.shape}")
    
    # 对 Z 进行 K-means 聚类
    n_clusters = len(np.unique(label))
    
    # 使用 K-means 进行聚类
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(Z_umap)
    
    # 评估聚类结果
    ari_gae, nmi_gae, ami_gae, aom_gae = evaluate_clustering(kmeans_labels, label)
    print("\033[94m", end="")  # 设置蓝色输出
    print(f"GAE+UMAP+K-means聚类结果 - 数据集 {dataset_name} - ARI: {ari_gae:.4f}, NMI: {nmi_gae:.4f}, AMI: {ami_gae:.4f}, AOM: {aom_gae:.4f}")
    print("\033[0m", end="")  # 重置默认颜色
    '''
    return ari, nmi, ami, aom

def Framework2(dataset_name, dataset_dir, params = default_params):
    '''
    '''
    print(f"\n===== 开始Framework2实验 (数据集: {dataset_name}) =====")
    set_random_seed(SEED)
    device = DEVICE
    adata, A, A_sp, L_norm = Load_Data(dataset_name)
    
    label = adata.obs['cell_type1']
    print(f"已加载 {dataset_name} 数据")
    
    # train GAE
    GAE_model, Z, train_loss, A_sp_gae = train_gae_model_new_with_val(adata, A, L_norm, device, params, patience = 5)
    GAE_model.eval()
    for param in GAE_model.parameters():
        param.requires_grad = False

    # 对 adata.X 进行 PCA 降维
    from sklearn.decomposition import PCA
    
    # 获取原始数据
    X_original = adata.X
    print(f"原始数据形状: {X_original.shape}")
    
    # 使用 PCA 降维，保留90%的解释方差
    pca = PCA(n_components=0.90, random_state=42)  # 保留90%的方差
    X_pca = pca.fit_transform(X_original)
    
    print(f"保留的解释方差比例: {pca.explained_variance_ratio_.sum():.4f}")
    

    hatX = torch.tensor(X_pca, dtype=torch.float32).to(device)

    lrr_layer, C, train_loss, lrr_recon_losses, lrr_reg_losses, lrr_block_losses = train_lrr_layer_new(hatX, A_sp_gae, device, params)
    
    # 对 Z 进行 UMAP 降维
    import umap
    
    # 使用 UMAP 进行降维
    umap_reducer = umap.UMAP(n_components=50, random_state=42, n_neighbors=15, min_dist=0.1)
    Z_umap = umap_reducer.fit_transform(Z.cpu().numpy())
    
    
    # 将 Z_umap 和 C 进行拼接作为总特征
    # Z_umap_tensor: (n_samples, 50)
    # C: (n_samples, n_samples) - 需要取第i列作为第i个样本的特征
    
    # 获取 C 矩阵的每一列作为特征
    C_features = C.T  # 转置后每行对应一个样本的C特征
    Z_umap
    # 拼接 Z_umap 和 C 特征
    # 将tensor转换为numpy数组进行拼接
    '''
    combined_features = np.concatenate([Z_umap, C_features], axis=1)
    
    
    # 使用拼接后的特征进行 K-means 聚类
    n_clusters = len(np.unique(label))
    
    # 将特征转换为 numpy 数组
    combined_features_np = combined_features
    
    # 对融合特征进行 PCA 降维
    from sklearn.decomposition import PCA
    
    print(f"融合特征原始维度: {combined_features_np.shape}")
    
    # 使用 PCA 降维，保留95%的解释方差
    pca_fusion = PCA(n_components=0.95, random_state=42)
    combined_features_pca = pca_fusion.fit_transform(combined_features_np)
    
    print(f"PCA降维后维度: {combined_features_pca.shape}")
    print(f"保留的解释方差比例: {pca_fusion.explained_variance_ratio_.sum():.4f}")
    
    # 更新特征数组
    combined_features_np = combined_features_pca
    # 使用 K-means 聚类
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(combined_features_np)
    
    # 评估聚类结果
    ari_kmeans, nmi_kmeans, ami_kmeans, aom_kmeans = evaluate_clustering(kmeans_labels, label)
    print("\033[93m", end="")  # 设置黄色输出
    print(f"K-means聚类结果 (拼接特征) - 数据集 {dataset_name} - ARI: {ari_kmeans:.4f}, NMI: {nmi_kmeans:.4f}, AMI: {ami_kmeans:.4f}, AOM: {aom_kmeans:.4f}")
    print("\033[0m", end="")  # 重置默认颜色
    '''







    W = build_affinity(Z,k0=10,k=20,metric='cosine',return_sparse=False,eps=1e-12)
    from sklearn.cluster import SpectralClustering
    # 对亲和矩阵 W 进行谱聚类
    n_clusters = len(np.unique(label))
    spectral_clustering_W = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed',
        random_state=42
    )
    
    # 将 W 转换为 numpy 数组（如果是 tensor）
    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().numpy()
    else:
        W_np = W
    
    # 进行谱聚类
    spectral_labels_W = spectral_clustering_W.fit_predict(W_np)
    
    # 评估基于 W 的谱聚类结果
    ari_W, nmi_W, ami_W, aom_W = evaluate_clustering(spectral_labels_W, label)
    print("\033[95m", end="")  # 设置紫色输出
    print(f"谱聚类结果 (基于亲和矩阵W) - 数据集 {dataset_name} - ARI: {ari_W:.4f}, NMI: {nmi_W:.4f}, AMI: {ami_W:.4f}, AOM: {aom_W:.4f}")
    print("\033[0m", end="")  # 重置默认颜色
    
    

    spec_labels, C_S = safe_spectral_clustering(C, n_clusters, random_state=42, return_S=True)

    ari_C, nmi_C, ami_C, aom_C = evaluate_clustering(spec_labels, label)
    print("\033[96m", end="")  # 设置青色输出
    print(f"谱聚类结果 (基于C矩阵) - 数据集 {dataset_name} - ARI: {ari_C:.4f}, NMI: {nmi_C:.4f}, AMI: {ami_C:.4f}, AOM: {aom_C:.4f}")
    print("\033[0m", end="")  # 重置默认颜色
    
    # 视图融合：将C_S和W按一定比例融合
    
    
    print("\n===== 视图融合谱聚类结果 =====")
    # 输出C_S和W的最大最小值差异
    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().numpy()
    else:
        W_np = W
    
    C_S_max = np.max(C_S)
    C_S_min = np.min(C_S)
    W_max = np.max(W_np)
    W_min = np.min(W_np)
    # 将C_S和W的最大值都缩放到1（已检查最小值都是0）
    C_S_normalized = C_S / C_S_max if C_S_max > 0 else C_S
    W_normalized = W_np / W_max if W_max > 0 else W_np
    
    

    weight, _ = fusion_weight_from_eigengap(W_normalized, C_S_normalized, k=n_clusters, eps=1e-12)
    print(f"融合权重: {weight:.4f}")
    # 融合矩阵：weight * C_S + (1 - weight) * W
    fused_matrix = weight * C_S + (1 - weight) * W_np
    
    # 对融合矩阵进行谱聚类
    spectral_clustering_fused = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed',
        random_state=42
    )
    
    fused_labels = spectral_clustering_fused.fit_predict(fused_matrix)
    
    # 评估融合后的谱聚类结果
    ari_fused, nmi_fused, ami_fused, aom_fused = evaluate_clustering(fused_labels, label)
    
    print("\033[92m", end="")  # 设置绿色输出
    print(f"融合谱聚类 (权重C_S:{weight:.1f}, W:{1-weight:.1f}) - 数据集 {dataset_name} - ARI: {ari_fused:.4f}, NMI: {nmi_fused:.4f}, AMI: {ami_fused:.4f}, AOM: {aom_fused:.4f}")
    print("\033[0m", end="")  # 重置默认颜色
    
    
    
    
    
    
    

    return ari_C, nmi_C, ami_C, aom_C



def train_gae_model_new_with_val(
    adata, A, L_norm, device, params,
    val_frac=0.2, patience=30, monitor="ap", seed=0, verbose=True
    ):
    """
    训练带“链路预测验证+早停”的 GAT-GAE。
    - 在训练图上优化：重构损失(边级 BCE) + α*ZINB + β*流形
    - 在验证集上评估：AUC/AP；基于 monitor 指标早停（默认 AP）
    返回：
        best_model, best_Z, train_losses, val_hist, A_train_sp
    """
    # 读取超参
    gat_hidden_channels = params['gae_params']['gat_hidden_channels']
    gat_out_channels    = params['gae_params']['gat_out_channels']
    head_num           = params['gae_params']['head_num']
    zinb_hidden_channels = params['gae_params']['zinb_hidden_channels']
    alpha              = params['gae_params']['alpha']
    lr                 = params['gae_params']['lr']
    epochs             = params['gae_params']['epochs']
    beta               = params['gae_params']['beta']

    if verbose:
        print("\n===== 开始训练 GAT-GAE（含链路预测验证与早停） =====")

    # 准备张量
    X = adata.X
    X = torch.tensor(X, dtype=torch.float32, device=device)
    A_np = np.asarray(A)
    L_norm = torch.tensor(L_norm, dtype=torch.float32, device=device)

    n = A_np.shape[0]

    # ---- 1) 边划分（正边） + 负边采样 ----
    pos_edges_all = get_undirected_edges_from_adj(A_np)
    train_pos, val_pos = split_edges(pos_edges_all, val_frac=val_frac, seed=seed)
    train_neg = sample_negative_edges(n, pos_edges_all, num_neg=len(train_pos))
    val_neg   = sample_negative_edges(n, pos_edges_all, num_neg=len(val_pos))

    # 训练图 = 移除验证正边（避免信息泄露）
    # 构造仅含训练正边的稀疏邻接，供 GAT 编码器使用
    A_train_sp = build_sparse_adj_from_edges(n, train_pos).coalesce().to(device)

    # ---- 2) 初始化模型 ----
    encoder = GAT(X.shape[1], gat_hidden_channels, gat_out_channels,
                  heads=head_num, dropout=0.1).to(device)
    decoder = ZINBdecoder(gat_out_channels, zinb_hidden_channels, out_feats=16).to(device)
    model = GAE(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    bceloss = torch.nn.BCEWithLogitsLoss()

    # ---- 3) 训练循环（边级重构 + ZINB + 流形）并进行验证与早停 ----
    train_losses = []
    val_hist = []  # [(epoch, auc, ap)]
    best_state = None
    best_Z = None
    best_score = -1e9
    patience_ctr = 0
    monitor_key = "ap" if monitor.lower() not in ["auc", "ap"] else monitor.lower()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 前向
        Z, pi, mu, theta = model(X, A_train_sp)

        # NaN 防御
        if check_nan(Z, "嵌入Z") or check_nan(pi, "pi") or check_nan(mu, "mu") or check_nan(theta, "theta"):
            if verbose:
                print("检测到NaN值，重置模型并降低学习率")
            encoder = GAT(X.shape[1], gat_hidden_channels, gat_out_channels,
                          heads=head_num, dropout=0.1).to(device)
            decoder = ZINBdecoder(gat_out_channels, zinb_hidden_channels, out_feats=16).to(device)
            model = GAE(encoder, decoder).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr/2, weight_decay=1e-5)
            Z, pi, mu, theta = model(X, A_train_sp)

        # ---- 边级重构损失：仅在训练集的正/负边上计算 ----
        # logits = z_i^T z_j
        def edge_logits(edges_np):
            if len(edges_np) == 0:
                return torch.empty(0, device=device)
            u = torch.tensor(edges_np[:,0], dtype=torch.long, device=device)
            v = torch.tensor(edges_np[:,1], dtype=torch.long, device=device)
            return (Z[u] * Z[v]).sum(dim=1)

        logits_pos = edge_logits(train_pos)
        logits_neg = edge_logits(train_neg)

        # 标签
        y_pos = torch.ones_like(logits_pos)
        y_neg = torch.zeros_like(logits_neg)
        logits = torch.cat([logits_pos, logits_neg], dim=0)
        labels = torch.cat([y_pos, y_neg], dim=0)

        reconstruction_loss = bceloss(logits, labels)

        # ZINB损失
        if params['ablation_params']['zinb_loss']:
            zinb_loss = ZINB_loss(X, pi, mu, theta)
        else:
            zinb_loss = torch.tensor(0.0, device=device)

        # 流形损失（可关）
        if params['ablation_params']['manifold_loss']:
            L_norm_Z = torch.matmul(L_norm, Z)
            manifold_loss = (Z * L_norm_Z).sum()
        else:
            manifold_loss = torch.tensor(0.0, device=device)

        loss = reconstruction_loss + alpha * zinb_loss + beta * manifold_loss

        if torch.isnan(loss):
            if verbose:
                print("总损失为NaN，终止训练")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

        # ---- 验证（链路预测 AUC/AP）----
        model.eval()
        with torch.no_grad():
            Z_eval, _, _, _ = model(X, A_train_sp)  # 编码器仍只看训练图
            pos_logits_val = score_edges(Z_eval, val_pos)
            neg_logits_val = score_edges(Z_eval, val_neg)
            val_auc, val_ap = linkpred_metrics_from_logits(pos_logits_val, neg_logits_val)
            val_hist.append((epoch, val_auc, val_ap))

        # 打印
        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"Epoch {epoch:03d} | "
                  f"Re: {reconstruction_loss.item():.4f} | "
                  f"ZINB: {(alpha*zinb_loss).item():.4f} | "
                  f"Mani: {(beta*manifold_loss).item():.4f} | "
                  f"Total: {loss.item():.4f} | "
                  f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

        # ---- 早停逻辑（默认监控 AP，更稳定）----
        current = val_ap if monitor_key == "ap" else val_auc
        if current > best_score + 1e-6:
            best_score = current
            best_state = deepcopy(model.state_dict())
            best_Z = Z_eval.detach().clone()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if verbose:
                    print(f"早停触发（monitor={monitor_key}, best={best_score:.4f}）于 epoch {epoch}")
                break

    # 回滚到最佳
    if best_state is not None:
        model.load_state_dict(best_state)
        Z_final = best_Z
    else:
        model.eval()
        with torch.no_grad():
            Z_final, _, _, _ = model(X, A_train_sp)

    if verbose:
        print("GAT-GAE模型训练完成（含链路预测验证与早停）")

    return model, Z_final, train_losses,  A_train_sp 





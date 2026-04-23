import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# 设置中文字体和绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set_style("whitegrid")

def visualize_preprocessed_data(adata, save_path=None, title="预处理后基因表达矩阵UMAP可视化"):
    """
    对预处理后的基因表达矩阵进行UMAP可视化
    
    参数:
        adata: AnnData对象，包含预处理后的基因表达数据
        save_path: 图片保存路径，如果为None则不保存
        title: 图表标题
    
    返回:
        embedding: UMAP嵌入坐标
        reducer: 训练好的UMAP模型
    """
    print(f"开始对预处理数据进行UMAP降维...")
    
    # 获取表达矩阵
    X = adata.X
    if hasattr(X, 'toarray'):  # 如果是稀疏矩阵
        X = X.toarray()
    
    # UMAP降维
    reducer = umap.UMAP(
        n_components=2, 
        random_state=42, 
        n_neighbors=15, 
        min_dist=0.1,
        metric='cosine'
    )
    embedding = reducer.fit_transform(X)
    
    # 获取细胞类型标签
    if 'cell_type1' in adata.obs:
        labels = adata.obs['cell_type1']
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        unique_labels = label_encoder.classes_
    else:
        labels = None
        print("警告: 未找到 'cell_type1' 标签，将生成无标签的可视化")
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    if labels is not None:
        # 为每种细胞类型分配颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embedding[mask, 0], 
                embedding[mask, 1], 
                c=[colors[i]], 
                label=label, 
                alpha=0.7, 
                s=20
            )
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=20)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('UMAP1', fontsize=12)
    plt.ylabel('UMAP2', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预处理数据UMAP可视化已保存到: {save_path}")
    
    plt.show()
    
    return embedding, reducer

def visualize_gae_embedding(Z, labels=None, save_path=None, title="GAE嵌入特征矩阵UMAP可视化"):
    """
    对GAE嵌入特征矩阵Z进行UMAP可视化
    
    参数:
        Z: GAE嵌入特征矩阵 (tensor或numpy array)
        labels: 细胞类型标签 (可选)
        save_path: 图片保存路径，如果为None则不保存
        title: 图表标题
    
    返回:
        embedding: UMAP嵌入坐标
        reducer: 训练好的UMAP模型
    """
    print(f"开始对GAE嵌入特征进行UMAP降维...")
    
    # 转换为numpy数组
    if isinstance(Z, torch.Tensor):
        Z_np = Z.detach().cpu().numpy()
    else:
        Z_np = Z
    
    # UMAP降维
    reducer = umap.UMAP(
        n_components=2, 
        random_state=42, 
        n_neighbors=15, 
        min_dist=0.1,
        metric='cosine'
    )
    embedding = reducer.fit_transform(Z_np)
    
    # 处理标签
    if labels is not None:
        if hasattr(labels, 'values'):  # pandas Series
            labels = labels.values
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        unique_labels = label_encoder.classes_
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    if labels is not None:
        # 为每种细胞类型分配颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embedding[mask, 0], 
                embedding[mask, 1], 
                c=[colors[i]], 
                label=label, 
                alpha=0.7, 
                s=20
            )
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=20)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('UMAP1', fontsize=12)
    plt.ylabel('UMAP2', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"GAE嵌入UMAP可视化已保存到: {save_path}")
    
    plt.show()
    
    return embedding, reducer

def visualize_affinity_matrix(W, labels=None, save_path=None, title="亲和矩阵UMAP可视化"):
    """
    对用于谱聚类的亲和矩阵W进行UMAP可视化
    
    参数:
        W: 亲和矩阵 (numpy array)
        labels: 细胞类型标签 (可选)
        save_path: 图片保存路径，如果为None则不保存
        title: 图表标题
    
    返回:
        embedding: UMAP嵌入坐标
        reducer: 训练好的UMAP模型
    """
    print(f"开始对亲和矩阵进行UMAP降维...")
    
    # 确保W是numpy数组
    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().numpy()
    else:
        W_np = W
    
    # 确保矩阵是对称的
    W_np = (W_np + W_np.T) / 2
    
    # 使用预计算距离进行UMAP降维
    # 将相似度矩阵转换为距离矩阵
    # 距离 = 1 - 相似度，但要确保非负
    max_val = np.max(W_np)
    if max_val > 0:
        W_normalized = W_np / max_val
    else:
        W_normalized = W_np
    
    # 转换为距离矩阵
    distance_matrix = 1.0 - W_normalized
    distance_matrix = np.maximum(distance_matrix, 0)  # 确保非负
    
    # 确保对角线为0（自己与自己的距离为0）
    np.fill_diagonal(distance_matrix, 0)
    
    # UMAP降维，使用预计算的距离矩阵
    reducer = umap.UMAP(
        n_components=2, 
        random_state=42, 
        metric='precomputed'
    )
    embedding = reducer.fit_transform(distance_matrix)
    
    # 处理标签
    if labels is not None:
        if hasattr(labels, 'values'):  # pandas Series
            labels = labels.values
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        unique_labels = label_encoder.classes_
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    if labels is not None:
        # 为每种细胞类型分配颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embedding[mask, 0], 
                embedding[mask, 1], 
                c=[colors[i]], 
                label=label, 
                alpha=0.7, 
                s=20
            )
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=20)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('UMAP1', fontsize=12)
    plt.ylabel('UMAP2', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"亲和矩阵UMAP可视化已保存到: {save_path}")
    
    plt.show()
    
    return embedding, reducer

def ab2_GAEreconAfoLRR_with_visualization(dataset_name, dataset_dir, params, figures_dir="figures"):
    """
    ab2框架的GAE重构邻接矩阵用于LRR，带三阶段可视化
    
    参数:
        dataset_name: 数据集名称
        dataset_dir: 数据集目录
        params: 参数字典
        figures_dir: 图片保存目录
    
    返回:
        ari, nmi, ami, aom: 聚类评估指标
    """
    import sys
    sys.path.append('.')
    from data import Load_Data
    from config import DEVICE, SEED
    from utils import set_random_seed, build_affinity
    from newtrain import train_gae_model_new_with_val, train_lrr_layer_new
    from clustering import safe_spectral_clustering, evaluate_clustering
    import torch.nn.functional as F
    
    print(f"\n===== 开始ab2可视化实验 (数据集: {dataset_name}) =====")
    set_random_seed(SEED)
    device = DEVICE
    
    # 1. 加载和预处理数据
    adata, A, A_sp, L_norm = Load_Data(dataset_name)
    label = adata.obs['cell_type1']
    print(f"已加载 {dataset_name} 数据")
    
    # 创建保存目录
    save_dir = os.path.join(figures_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # ===== 阶段1: 预处理数据可视化 =====
    print("\n----- 阶段1: 预处理数据UMAP可视化 -----")
    preprocessing_save_path = os.path.join(save_dir, f"{dataset_name}_preprocessing_umap.png")
    preprocessing_embedding, _ = visualize_preprocessed_data(
        adata, 
        save_path=preprocessing_save_path,
        title=f"{dataset_name} - 预处理后基因表达矩阵UMAP可视化"
    )
    
    # 2. 训练GAE模型
    GAE_model, Z, train_loss, A_sp_gae = train_gae_model_new_with_val(
        adata, A, L_norm, device, params, patience=5
    )
    GAE_model.eval()
    for param in GAE_model.parameters():
        param.requires_grad = False
    
    # ===== 阶段2: GAE嵌入特征可视化 =====
    print("\n----- 阶段2: GAE嵌入特征UMAP可视化 -----")
    gae_embedding_save_path = os.path.join(save_dir, f"{dataset_name}_gae_embedding_umap.png")
    gae_embedding, _ = visualize_gae_embedding(
        Z, 
        labels=label,
        save_path=gae_embedding_save_path,
        title=f"{dataset_name} - GAE嵌入特征矩阵UMAP可视化"
    )
    
    # 3. 用Z重构新的邻接矩阵
    with torch.no_grad():
        Z_norm = F.normalize(Z, p=2, dim=1)
        similarity_matrix = torch.mm(Z_norm, Z_norm.t())
        
        # 动态设置阈值
        n_nodes = similarity_matrix.size(0)
        thresholds = torch.linspace(0.9, 0.1, 81)
        
        for thresh in thresholds:
            A_temp = (similarity_matrix > thresh).float()
            A_temp.fill_diagonal_(0)
            row_sums = A_temp.sum(dim=1)
            if (row_sums == 0).sum() == 0:
                threshold = thresh.item()
                print(f"为数据集 {dataset_name} 设置阈值: {threshold:.3f}")
                break
        else:
            threshold = 0.1
            print(f"警告：数据集 {dataset_name} 使用最小阈值: {threshold}")
        
        A_new = (similarity_matrix > threshold).float()
        A_new.fill_diagonal_(0)
        
        indices = torch.nonzero(A_new, as_tuple=False).t()
        values = A_new[indices[0], indices[1]]
        A_sp_gae = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=A_new.size(),
            device=device
        ).coalesce()
    
    # 4. 训练LRR层
    lrr_layer, C, train_loss, lrr_recon_losses, lrr_reg_losses, lrr_block_losses = train_lrr_layer_new(
        Z, A_sp_gae, device, params
    )
    
    # 5. 构建亲和矩阵W用于最终的谱聚类可视化
    W = build_affinity(Z, k0=10, k=20, metric='cosine', return_sparse=False, eps=1e-12)
    
    # ===== 阶段3: 亲和矩阵可视化 =====
    print("\n----- 阶段3: 谱聚类亲和矩阵UMAP可视化 -----")
    affinity_save_path = os.path.join(save_dir, f"{dataset_name}_affinity_matrix_umap.png")
    affinity_embedding, _ = visualize_affinity_matrix(
        W, 
        labels=label,
        save_path=affinity_save_path,
        title=f"{dataset_name} - 谱聚类亲和矩阵UMAP可视化"
    )
    
    # 6. 最终聚类评估
    n_clusters = len(np.unique(label))
    spec_labels = safe_spectral_clustering(C, n_clusters, random_state=42)
    ari, nmi, ami, aom = evaluate_clustering(spec_labels, label)

    affinity_embedding, _ = visualize_affinity_matrix(
        W, 
        labels=spec_labels,
        save_path=os.path.join(save_dir, f"{dataset_name}_affinity_matrix_umap_spec.png"),
        title=f"{dataset_name} - 谱聚类亲和矩阵UMAP可视化"
    )
    
    # 输出结果
    print("\033[93m", end="")
    print(f"数据集 {dataset_name} - ARI: {ari:.4f}, NMI: {nmi:.4f}, AMI: {ami:.4f}, AOM: {aom:.4f}")
    print("\033[0m", end="")
    
    # 7. 创建实验总结
    preprocessing_info = {
        "数据集名称": dataset_name,
        "细胞数量": adata.shape[0],
        "基因数量": adata.shape[1],
        "细胞类型数": len(np.unique(label))
    }
    
    gae_info = {
        "GAT隐藏维度": params['gae_params']['gat_hidden_channels'],
        "GAT输出维度": params['gae_params']['gat_out_channels'],
        "注意力头数": params['gae_params']['head_num'],
        "学习率": params['gae_params']['lr'],
        "训练轮数": params['gae_params']['epochs']
    }
    
    clustering_info = {
        "聚类算法": "谱聚类",
        "聚类数": n_clusters,
        "ARI": f"{ari:.4f}",
        "NMI": f"{nmi:.4f}",
        "AMI": f"{ami:.4f}",
        "AOM": f"{aom:.4f}"
    }
    
    create_visualization_summary(dataset_name, preprocessing_info, gae_info, clustering_info, save_dir)
    
    print(f"\n===== {dataset_name} 可视化实验完成 =====")
    print(f"所有图片已保存到: {save_dir}")
    
    return ari, nmi, ami, aom

def create_visualization_summary(dataset_name, preprocessing_info, gae_info, clustering_info, save_dir):
    """
    创建可视化实验的总结报告
    
    参数:
        dataset_name: 数据集名称
        preprocessing_info: 预处理信息字典
        gae_info: GAE训练信息字典
        clustering_info: 聚类信息字典
        save_dir: 保存目录
    """
    summary_path = os.path.join(save_dir, f"{dataset_name}_visualization_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"数据集 {dataset_name} 可视化实验总结\n")
        f.write("=" * 50 + "\n\n")
        
        # 预处理信息
        f.write("1. 预处理信息:\n")
        for key, value in preprocessing_info.items():
            f.write(f"   {key}: {value}\n")
        f.write("\n")
        
        # GAE训练信息
        f.write("2. GAE训练信息:\n")
        for key, value in gae_info.items():
            f.write(f"   {key}: {value}\n")
        f.write("\n")
        
        # 聚类信息
        f.write("3. 聚类信息:\n")
        for key, value in clustering_info.items():
            f.write(f"   {key}: {value}\n")
        f.write("\n")
        
        f.write("可视化文件说明:\n")
        f.write("- {}_preprocessing_umap.png: 预处理后基因表达矩阵的UMAP可视化\n".format(dataset_name))
        f.write("- {}_gae_embedding_umap.png: GAE嵌入特征矩阵的UMAP可视化\n".format(dataset_name))
        f.write("- {}_affinity_matrix_umap.png: 谱聚类亲和矩阵的UMAP可视化\n".format(dataset_name))
    
    print(f"可视化总结报告已保存到: {summary_path}")

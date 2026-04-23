import os
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

def Preprocessed(dataset_list, dataset_dir,  knn_params = None, preprocess_params = None):
    # DataPreprocessed Module
    # 检查预处理数据目录是否存在以及是否包含预处理后的数据
    preprocessed_dir = os.path.join(os.getcwd(), 'preprocessed_data')
    if os.path.exists(preprocessed_dir):
        
        # 检查dataset_list中的数据集是否在preprocessed_dir下都有对应名字的文件夹
        h5_files = []
        missing_datasets = []
        
        for dataset_name in dataset_list:
            dataset_folder = os.path.join(preprocessed_dir, dataset_name)
            if os.path.exists(dataset_folder) and os.path.isdir(dataset_folder):
                # 检查文件夹内是否包含必要的文件
                adata_file = os.path.join(dataset_folder, 'adata.h5ad')
                adj_file = os.path.join(dataset_folder, 'adj.npz')
                if os.path.exists(adata_file) and os.path.exists(adj_file):
                    h5_files.append(dataset_name)
                else:
                    missing_datasets.append(dataset_name)
            else:
                missing_datasets.append(dataset_name)
        
        if len(missing_datasets) > 0:
            print(f"以下数据集没有预处理完成: {missing_datasets}")
            
        else:
            print(f"所有数据集都已预处理完成")
            return
                
    
    for dataset_name in missing_datasets:
        print(f"正在预处理 {dataset_name} 数据")
        adata = load_dataset(dataset_name, dataset_dir)    
        adata, A = preprocess_data(adata, knn_params, preprocess_params)
        # 创建预处理数据目录
        os.makedirs(preprocessed_dir, exist_ok=True)
        os.makedirs(os.path.join(preprocessed_dir, dataset_name), exist_ok=True)
        # 保存预处理后的adata
        adata_path = os.path.join(preprocessed_dir, f"{dataset_name}/adata.h5ad")
        adata.write(adata_path)
        
        # 将邻接矩阵转换为稀疏矩阵并保存
        import scipy.sparse as sp
        A_sparse = sp.csr_matrix(A)
        adj_path = os.path.join(preprocessed_dir, f"{dataset_name}/adj.npz")
        sp.save_npz(adj_path, A_sparse)
        
        print(f"已保存 {dataset_name} 的预处理数据:")
    return

def Load_Data(dataset_name):
    preprocessed_dir = os.path.join(os.getcwd(), 'preprocessed_data')
    adata_path = os.path.join(preprocessed_dir, f"{dataset_name}/adata.h5ad")
    adj_path = os.path.join(preprocessed_dir, f"{dataset_name}/adj.npz")

    # 加载预处理后的adata
    adata = anndata.read_h5ad(adata_path)
    
    # 加载邻接矩阵
    import scipy.sparse as sp
    A_sparse = sp.load_npz(adj_path)
    A = A_sparse.toarray()  # 转换回密集矩阵

    
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    D_inv_sqrt = np.linalg.inv(np.sqrt(D + 1e-10))  # 添加小值防止除零
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    return adata, A, A_sparse, L_norm





def load_h5ad(h5_path):
    '''
    加载h5ad数据, 其中exprs为表达矩阵, label为cell_type.
    '''
    return anndata.read_h5ad(h5_path)
def load_dataset(dataset_name, dataset_dir):
    '''
    加载h5ad数据集
    '''
    return load_h5ad(os.path.join(dataset_dir, f"{dataset_name}.h5ad"))
def preprocess_pipeline(adata, min_genes=200, min_cells=3, 
                      target_sum=1e4, n_top_genes=500, 
                      cut_max_value=10):
    """
    单细胞RNA-seq数据预处理流水线
    """
    # 基本过滤
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # 计算基因的均值和方差
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    
    # 选择高变基因
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='cell_ranger')
    adata = adata[:, adata.var['highly_variable']]
    # 降维前的预处理
    sc.pp.scale(adata, max_value=cut_max_value)
    return adata

def preprocess_data(adata, knn_params = None, preprocess_params = None):
    
    if preprocess_params is None:
        adata = preprocess_pipeline(adata, min_genes=200, min_cells=3,
                           target_sum=1e4, n_top_genes=500,
                           cut_max_value=10)
    else:
        adata = preprocess_pipeline(adata, **preprocess_params)

    A = build_knn_adj_matrix(adata.X, k=knn_params['k_knn'], metric=knn_params['metric'], threshold=knn_params['threshold'])

    
    return adata, A


def build_knn_adj_matrix(X, k=15, metric='cosine', threshold=0.05, min_edges=5):
    """
    构建k-NN邻接矩阵，并根据距离阈值过滤边，同时确保每个细胞至少保留min_edges条边
    参数:
        X: cell by genes的基因表达谱矩阵
        k: 近邻数量
        metric: 距离度量方法，默认为'cosine'
        threshold: 距离阈值比例（0到1），保留距离小于该比例的边
        min_edges: 每个细胞最少保留的边数
    返回:
        adj_matrix: 邻接矩阵 (N x N)
    """
    # 初始化NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X)
    
    # 找到k近邻
    distances, indices = nbrs.kneighbors(X)
    # distances: 形状为(n_cells, k+1)，每一行包含k+1个距离值（包括自身）
    # indices: 形状为(n_cells, k+1)，每一行包含k+1个整数索引（包括自身）
    
    # 获取所有距离值（排除自身距离）
    all_distances = distances[:, 1:].flatten()  # 排除自身距离（第一列）
    if len(all_distances) == 0:
        raise ValueError("No valid distances found. Check input data or k value.")
    
    # 计算距离阈值
    distance_threshold = np.percentile(all_distances, threshold * 100)
    
    # 构建邻接矩阵
    n_cells = X.shape[0]
    adj_matrix = np.zeros((n_cells, n_cells))
    
    # 第一次填充：根据距离阈值初步确定边
    edges_to_add = []
    for i in range(n_cells):
        for j_idx, dist in zip(indices[i][1:], distances[i][1:]):  # 跳过自身
            if dist <= distance_threshold:
                edges_to_add.append((i, j_idx))
    
    # 计算每个细胞的初步边数
    edge_counts = np.zeros(n_cells)
    for i, j in edges_to_add:
        edge_counts[i] += 1
        edge_counts[j] += 1  # 因为边是对称的
    
    # 第二次填充：确保每个细胞至少有min_edges条边
    for i in range(n_cells):
        for j_idx, dist in zip(indices[i][1:], distances[i][1:]):  # 跳过自身
            if (i, j_idx) in edges_to_add or (j_idx, i) in edges_to_add:
                adj_matrix[i, j_idx] = 1
                adj_matrix[j_idx, i] = 1  # 确保矩阵对称
            elif edge_counts[i] < min_edges or edge_counts[j_idx] < min_edges:
                # 如果删除边会导致任一细胞边数少于min_edges，则保留
                adj_matrix[i, j_idx] = 1
                adj_matrix[j_idx, i] = 1  # 确保矩阵对称
                edge_counts[i] += 1
                edge_counts[j_idx] += 1
    
    return adj_matrix

def build_threshold_adj_matrix(X, threshold, metric='cosine', k=5):
    """
    构建基于阈值的邻接矩阵
    参数:
        X: cell by genes的基因表达谱矩阵
        threshold: 0到1之间的比例，用于选择前threshold比例的最小距离作为硬边
        metric: 距离度量，默认为'cosine'
        k: 对于孤立细胞，添加top-k最近邻
    返回:
        adj_matrix: 邻接矩阵 (N x N)
    """
    n_cells = X.shape[0]
    
    # 计算所有细胞对的距离矩阵
    dist_matrix = pairwise_distances(X, metric=metric)
    
    # 提取非对角线距离（上三角部分，避免重复和自环）
    all_distances = dist_matrix[np.triu_indices(n_cells, k=1)]
    
    # 排序距离
    all_distances_sorted = np.sort(all_distances)
    
    # 找到阈值：前threshold比例的最小距离的边界
    if len(all_distances_sorted) == 0:
        raise ValueError("No distances available.")
    thresh_idx = int(threshold * len(all_distances_sorted))
    if thresh_idx >= len(all_distances_sorted):
        thresh_idx = len(all_distances_sorted) - 1
    distance_threshold = all_distances_sorted[thresh_idx]
    
    # 构建初始邻接矩阵：距离 <= 阈值的设为1（排除自环）
    adj_matrix = (dist_matrix <= distance_threshold).astype(float)
    np.fill_diagonal(adj_matrix, 0)  # 移除自环
    
    # 检查孤立细胞（度为0）
    degrees = np.sum(adj_matrix, axis=1)
    isolated_cells = np.where(degrees == 0)[0]
    
    if len(isolated_cells) > 0:
        # 使用NearestNeighbors找到每个孤立细胞的top-k最近邻
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X)
        distances, indices = nbrs.kneighbors(X[isolated_cells])
        
        # 为每个孤立细胞添加边
        for idx, i in enumerate(isolated_cells):
            # 跳过第一个（自身）
            neighbors = indices[idx][1:]
            for j in neighbors:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # 确保对称
    
    return adj_matrix


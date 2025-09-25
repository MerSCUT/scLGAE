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
from utils import *

from model import *


from config import *



def safe_spectral_clustering(C, n_clusters, random_state=42, return_S = False):
    """
    安全的谱聚类，处理NaN和数值稳定性问题
    """
    # 检查C矩阵中的NaN
    if np.isnan(C).any():
        print("警告: C矩阵包含NaN值，尝试修复...")
        # 用0替换NaN值
        C = np.nan_to_num(C, nan=0.0)
    # 检查是否有全零行（可能导致除零错误）
    row_sums = np.sum(np.abs(C), axis=1)
    zero_rows = np.where(row_sums == 0)[0]
    if len(zero_rows) > 0:
        print(f"警告: 检测到 {len(zero_rows)} 个全零行，添加小扰动...")
        # 为全零行添加小扰动
        for idx in zero_rows:
            C[idx, idx] = 1e-5
    # 确保矩阵对称（谱聚类需要）
    C = (C + C.T) / 2
    C[C<0] = 0
    
    try:
        # 尝试进行谱聚类
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                              random_state=random_state)
        #print(C)
        labels = sc.fit_predict(C)
        if return_S:
            return labels, C
        return labels
    except Exception as e:
        print(f"谱聚类失败: {e}")
        print("safe_spectral_clustering尝试使用KMeans作为备用方案...")
        # 如果谱聚类失败，使用KMeans作为备用
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        if return_S:
            return kmeans.fit_predict(C), C
        return kmeans.fit_predict(C)

def nmf_clustering(C, n_clusters, random_state=42, max_iter=1000):
    """
    使用NMF进行聚类
    参数:
        C: 自表达系数矩阵
        n_clusters: 聚类数量
        random_state: 随机种子
        max_iter: 最大迭代次数
    返回:
        labels: 聚类标签
    """
    # 检查C矩阵中的NaN
    if np.isnan(C).any():
        print("警告: C矩阵包含NaN值，用零替换")
        C = np.nan_to_num(C, nan=0.0)
    
    # 确保矩阵非负（NMF要求）
    C_abs = np.abs(C)
    
    # 归一化处理
    #row_sums = np.sum(C_abs, axis=1, keepdims=True)
    #C_normalized = np.divide(C_abs, row_sums, where=row_sums!=0)
    C_normalized = C_abs
    try:
        # 应用NMF
        model = NMF(n_components=n_clusters, random_state=random_state, 
                   init='nndsvda', max_iter=max_iter)
        W = model.fit_transform(C_normalized)
        
        # 获取聚类标签
        labels = np.argmax(W, axis=1)
        return labels
    except Exception as e:
        print(f"NMF聚类失败: {e}")
        print("尝试使用KMeans作为备用方案...")
        # 如果NMF失败，使用KMeans作为备用
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        return kmeans.fit_predict(C_normalized)

def evaluate_clustering(predict_labels, true_labels):
    ari = ARI(true_labels, predict_labels)
    nmi = NMI(true_labels, predict_labels)
    ami = AMI(true_labels, predict_labels)
    aom = average_overlap_measure(true_labels, predict_labels)
    return ari, nmi, ami, aom
import optuna
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from config import *
from newtrain import run_experiment_new
from data import Preprocessed
from utils import set_random_seed


def objective_single_dataset(trial, dataset_name, dataset_dir):
    """
    针对单个数据集的Optuna优化目标函数
    参数:
        trial: optuna trial对象
        dataset_name: 单个数据集名称
        dataset_dir: 数据集目录
    返回:
        ARI + NMI的平均值
    """
    
    # 建议GAE参数
    gae_params = {
        'gat_hidden_channels': trial.suggest_categorical('gat_hidden_channels', [64, 128, 256]),
        'gat_out_channels': trial.suggest_categorical('gat_out_channels', [64, 128, 144, 256]),
        'head_num': trial.suggest_categorical('head_num', [2, 4, 8]),
        'zinb_hidden_channels': trial.suggest_categorical('zinb_hidden_channels', [8, 16, 32]),
        'lr': trial.suggest_float('gae_lr', 1e-6, 1e-3, log=True),
        'alpha': trial.suggest_float('alpha', 0.001, 1e1, log=True),
        'beta': trial.suggest_float('beta', 1e-5, 1e1, log=True),
        'epochs': 50  # 固定epoch数以节省时间
    }
    
    # 建议LRR参数
    lrr_params = {
        'lrr_lambda': trial.suggest_float('lrr_lambda', 1e-6, 1e-3, log=True),
        'lrr_gamma': trial.suggest_float('lrr_gamma', 1e-3, 1e1, log=True),
        'lrr_weight': trial.suggest_float('lrr_weight', 0.01, 1.0),
        'lrr_epochs': 100,  # 固定epoch数
        'lr': trial.suggest_float('lrr_lr', 1e-4, 1e-2, log=True),
        'epochs': 100
    }
    
    # 构建完整参数字典
    trial_params = {
        'knn_params': default_params['knn_params'],
        'preprocess_params': default_params['preprocess_params'],
        'gae_params': gae_params,
        'lrr_params': lrr_params,
        'ablation_params': default_params['ablation_params']
    }
    
    # 在单个数据集上运行实验
    try:
        ari, nmi, ami, aom = run_experiment_new(dataset_name, dataset_dir, trial_params)
        # 目标函数：ARI + NMI的平均值
        objective_value = (ari + nmi) / 2
        print(f"Trial {trial.number}, Dataset {dataset_name}: ARI={ari:.4f}, NMI={nmi:.4f}, Objective={objective_value:.4f}")
        return objective_value
    except Exception as e:
        print(f"Trial {trial.number}, Dataset {dataset_name} failed: {e}")
        return 0.0  # 失败时返回0分


def objective(trial, dataset_list, dataset_dir):
    """
    多数据集的Optuna优化目标函数（保留用于向后兼容）
    参数:
        trial: optuna trial对象
        dataset_list: 数据集列表
        dataset_dir: 数据集目录
    返回:
        ARI + NMI的平均值
    """
    
    # 建议GAE参数
    gae_params = {
        'gat_hidden_channels': trial.suggest_categorical('gat_hidden_channels', [64, 128, 256]),
        'gat_out_channels': trial.suggest_categorical('gat_out_channels', [64, 128, 144, 256]),
        'head_num': trial.suggest_categorical('head_num', [2, 4, 8]),
        'zinb_hidden_channels': trial.suggest_categorical('zinb_hidden_channels', [8, 16, 32]),
        'lr': trial.suggest_float('gae_lr', 1e-6, 1e-3, log=True),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e1, log=True),
        'beta': trial.suggest_float('beta', 1e-5, 1e1, log=True),
        'epochs': 50  # 固定epoch数以节省时间
    }
    
    # 建议LRR参数
    lrr_params = {
        'lrr_lambda': trial.suggest_float('lrr_lambda', 1e-6, 1e-3, log=True),
        'lrr_gamma': trial.suggest_float('lrr_gamma', 1e-3, 1e1, log=True),
        'lrr_weight': trial.suggest_float('lrr_weight', 0.01, 1.0),
        'lrr_epochs': 100,  # 固定epoch数
        'lr': trial.suggest_float('lrr_lr', 1e-4, 1e-2, log=True),
        'epochs': 100
    }
    
    # 构建完整参数字典
    trial_params = {
        'knn_params': default_params['knn_params'],
        'preprocess_params': default_params['preprocess_params'],
        'gae_params': gae_params,
        'lrr_params': lrr_params,
        'ablation_params': default_params['ablation_params']
    }
    
    # 在多个数据集上运行实验并计算平均性能
    ari_scores = []
    nmi_scores = []
    
    for dataset_name in dataset_list:
        try:
            ari, nmi, ami, aom = run_experiment_new(dataset_name, dataset_dir, trial_params)
            ari_scores.append(ari)
            nmi_scores.append(nmi)
            print(f"Trial {trial.number}, Dataset {dataset_name}: ARI={ari:.4f}, NMI={nmi:.4f}")
        except Exception as e:
            print(f"Trial {trial.number}, Dataset {dataset_name} failed: {e}")
            # 如果某个数据集失败，给予较低分数
            ari_scores.append(0.0)
            nmi_scores.append(0.0)
    
    # 计算平均ARI和NMI
    avg_ari = np.mean(ari_scores)
    avg_nmi = np.mean(nmi_scores)
    
    # 目标函数：ARI + NMI的平均值
    objective_value = (avg_ari + avg_nmi) / 2
    
    print(f"Trial {trial.number}: Avg ARI={avg_ari:.4f}, Avg NMI={avg_nmi:.4f}, Objective={objective_value:.4f}")
    
    return objective_value


def optimize_parameters(n_trials=50, dataset_list=None):
    """
    使用Optuna优化参数
    参数:
        n_trials: 优化试验次数
        dataset_list: 数据集列表，如果为None则使用默认列表
    """
    
    if dataset_list is None:
        dataset_list = [
            'Muraro',
            'Quake_10x_Bladder', 
            'Quake_10x_Limb_Muscle',
            'Quake_10x_Spleen'
        ]  # 使用较少数据集以节省时间
    
    dataset_dir = DATASET_DIR
    
    print(f"开始参数优化，试验次数: {n_trials}")
    print(f"使用数据集: {dataset_list}")
    
    # 确保预处理数据存在
    knn_params = default_params['knn_params']
    preprocess_params = default_params['preprocess_params']
    Preprocessed(dataset_list, dataset_dir, knn_params, preprocess_params)
    
    # 创建optuna study
    study = optuna.create_study(direction='maximize')
    
    # 运行优化
    study.optimize(lambda trial: objective(trial, dataset_list, dataset_dir), n_trials=n_trials)
    
    # 输出最佳参数
    print("\n===== 优化完成 =====")
    print("最佳参数:")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\n最佳目标值: {study.best_value:.4f}")
    
    # 构建完整的最佳参数字典
    best_gae_params = {
        'gat_hidden_channels': best_params['gat_hidden_channels'],
        'gat_out_channels': best_params['gat_out_channels'],
        'head_num': best_params['head_num'],
        'zinb_hidden_channels': best_params['zinb_hidden_channels'],
        'lr': best_params['gae_lr'],
        'alpha': best_params['alpha'],
        'beta': best_params['beta'],
        'epochs': 50
    }
    
    best_lrr_params = {
        'lrr_lambda': best_params['lrr_lambda'],
        'lrr_gamma': best_params['lrr_gamma'],
        'lrr_weight': best_params['lrr_weight'],
        'lrr_epochs': 100,
        'lr': best_params['lrr_lr'],
        'epochs': 100
    }
    
    best_full_params = {
        'knn_params': default_params['knn_params'],
        'preprocess_params': default_params['preprocess_params'],
        'gae_params': best_gae_params,
        'lrr_params': best_lrr_params,
        'ablation_params': default_params['ablation_params']
    }
    
    # 保存最佳参数
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存最佳参数到JSON文件
    best_params_file = os.path.join(results_dir, f'best_params_{timestamp}.json')
    with open(best_params_file, 'w', encoding='utf-8') as f:
        json.dump(best_full_params, f, indent=2, ensure_ascii=False)
    
    print(f"\n最佳参数已保存到: {best_params_file}")
    
    # 保存优化历史
    trials_df = study.trials_dataframe()
    trials_file = os.path.join(results_dir, f'optimization_trials_{timestamp}.csv')
    trials_df.to_csv(trials_file, index=False)
    
    print(f"优化历史已保存到: {trials_file}")
    
    return best_full_params, study


def optimize_single_dataset(dataset_name, n_trials=50, dataset_dir=None):
    """
    为单个数据集优化参数
    参数:
        dataset_name: 数据集名称
        n_trials: 优化试验次数
        dataset_dir: 数据集目录
    返回:
        best_params: 最佳参数
        study: optuna study对象
    """
    
    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    
    print(f"\n===== 开始优化数据集: {dataset_name} =====")
    print(f"试验次数: {n_trials}")
    
    # 确保预处理数据存在
    knn_params = default_params['knn_params']
    preprocess_params = default_params['preprocess_params']
    Preprocessed([dataset_name], dataset_dir, knn_params, preprocess_params)
    
    # 创建optuna study
    study = optuna.create_study(direction='maximize')
    
    # 运行优化
    study.optimize(lambda trial: objective_single_dataset(trial, dataset_name, dataset_dir), n_trials=n_trials)
    
    # 输出最佳参数
    print(f"\n===== 数据集 {dataset_name} 优化完成 =====")
    print("最佳参数:")
    best_params = study.best_params
    for key, value in best_params.items():
        pass
    
    print(f"最佳目标值: {study.best_value:.4f}")
    
    # 构建完整的最佳参数字典
    best_gae_params = {
        'gat_hidden_channels': best_params['gat_hidden_channels'],
        'gat_out_channels': best_params['gat_out_channels'],
        'head_num': best_params['head_num'],
        'zinb_hidden_channels': best_params['zinb_hidden_channels'],
        'lr': best_params['gae_lr'],
        'alpha': best_params['alpha'],
        'beta': best_params['beta'],
        'epochs': 50
    }
    
    best_lrr_params = {
        'lrr_lambda': best_params['lrr_lambda'],
        'lrr_gamma': best_params['lrr_gamma'],
        'lrr_weight': best_params['lrr_weight'],
        'lrr_epochs': 100,
        'lr': best_params['lrr_lr'],
        'epochs': 100
    }
    
    best_full_params = {
        'dataset_name': dataset_name,
        'best_objective': study.best_value,
        'knn_params': default_params['knn_params'],
        'preprocess_params': default_params['preprocess_params'],
        'gae_params': best_gae_params,
        'lrr_params': best_lrr_params,
        'ablation_params': default_params['ablation_params']
    }
    
    return best_full_params, study


def optimize_all_datasets(n_trials=50, dataset_list=None):
    """
    为所有数据集分别进行优化
    参数:
        n_trials: 每个数据集的优化试验次数
        dataset_list: 数据集列表，如果为None则使用所有数据集
    返回:
        all_best_params: 所有数据集的最佳参数字典
        all_studies: 所有optuna study对象字典
    """
    
    if dataset_list is None:
        # 使用所有可用的数据集
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
    
    print(f"===== 开始为所有数据集进行参数优化 =====")
    print(f"数据集数量: {len(dataset_list)}")
    print(f"每个数据集试验次数: {n_trials}")
    print(f"数据集列表: {dataset_list}")
    
    all_best_params = {}
    all_studies = {}
    optimization_summary = []
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.getcwd(), 'results')
    dataset_results_dir = os.path.join(results_dir, f'dataset_optimization_{timestamp}')
    os.makedirs(dataset_results_dir, exist_ok=True)
    
    # 为每个数据集单独优化
    for i, dataset_name in enumerate(dataset_list):
        print(f"\n【{i+1}/{len(dataset_list)}】正在优化数据集: {dataset_name}")
        
        try:
            best_params, study = optimize_single_dataset(dataset_name, n_trials)
            all_best_params[dataset_name] = best_params
            all_studies[dataset_name] = study
            
            # 保存单个数据集的最佳参数
            dataset_params_file = os.path.join(dataset_results_dir, f'best_params_{dataset_name}.json')
            with open(dataset_params_file, 'w', encoding='utf-8') as f:
                json.dump(best_params, f, indent=2, ensure_ascii=False)
            
            # 保存单个数据集的优化历史
            trials_df = study.trials_dataframe()
            trials_file = os.path.join(dataset_results_dir, f'optimization_trials_{dataset_name}.csv')
            trials_df.to_csv(trials_file, index=False)
            
            # 记录汇总信息
            optimization_summary.append({
                'dataset': dataset_name,
                'best_objective': study.best_value,
                'n_trials': len(study.trials),
                'best_gat_hidden': best_params['gae_params']['gat_hidden_channels'],
                'best_gat_out': best_params['gae_params']['gat_out_channels'],
                'best_head_num': best_params['gae_params']['head_num'],
                'best_zinb_hidden': best_params['gae_params']['zinb_hidden_channels'],
                'best_gae_lr': best_params['gae_params']['lr'],
                'best_alpha': best_params['gae_params']['alpha'],
                'best_beta': best_params['gae_params']['beta'],
                'best_lrr_lambda': best_params['lrr_params']['lrr_lambda'],
                'best_lrr_gamma': best_params['lrr_params']['lrr_gamma'],
                'best_lrr_weight': best_params['lrr_params']['lrr_weight'],
                'best_lrr_lr': best_params['lrr_params']['lr']
            })
            
            print(f"✓ 数据集 {dataset_name} 优化完成，最佳目标值: {study.best_value:.4f}")
            
        except Exception as e:
            print(f"✗ 数据集 {dataset_name} 优化失败: {e}")
            continue
    
    # 保存汇总结果
    summary_df = pd.DataFrame(optimization_summary)
    summary_file = os.path.join(dataset_results_dir, 'optimization_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    # 保存所有最佳参数到一个文件
    all_params_file = os.path.join(dataset_results_dir, 'all_best_params.json')
    with open(all_params_file, 'w', encoding='utf-8') as f:
        json.dump(all_best_params, f, indent=2, ensure_ascii=False)
    
    print(f"\n===== 所有数据集优化完成 =====")
    print(f"成功优化数据集数量: {len(all_best_params)}")
    print(f"结果保存目录: {dataset_results_dir}")
    print(f"汇总文件: {summary_file}")
    
    # 显示汇总统计
    if len(optimization_summary) > 0:
        print(f"\n===== 优化结果汇总 =====")
        print(f"平均最佳目标值: {summary_df['best_objective'].mean():.4f} ± {summary_df['best_objective'].std():.4f}")
        print(f"最佳目标值范围: [{summary_df['best_objective'].min():.4f}, {summary_df['best_objective'].max():.4f}]")
        
        # 显示参数统计
        print(f"\n参数统计:")
        print(f"GAT隐藏层通道数分布: {summary_df['best_gat_hidden'].value_counts().to_dict()}")
        print(f"GAT输出通道数分布: {summary_df['best_gat_out'].value_counts().to_dict()}")
        print(f"注意力头数分布: {summary_df['best_head_num'].value_counts().to_dict()}")
    
    return all_best_params, all_studies


if __name__ == "__main__":
    # 设置随机种子
    set_random_seed(SEED)
    
    # 切换到工作目录
    os.chdir(WORK_DIR)
    
    # 运行所有数据集的优化
    all_best_params, all_studies = optimize_all_datasets(n_trials=30)
    
    print("\n===== 所有优化完成 =====")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ab2_GAEreconAfoLRR 模型的超参数优化模块
专门针对 ab2_GAEreconAfoLRR 函数进行参数优化
"""

import optuna
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from newtrain import ab2_GAEreconAfoLRR
from data import Preprocessed
from config import DATASET_DIR, abl2_default_params
from utils import set_random_seed

def objective_ab2(trial, dataset='Muraro', dataset_dir=DATASET_DIR):
    """
    ab2_GAEreconAfoLRR 模型的 Optuna 目标函数
    """

    
    try:
        # 定义 ab2_GAEreconAfoLRR 的超参数搜索空间
        params = {
            'knn_params': {
                'k_knn': 20,
                'metric': 'cosine',
                'threshold': 0.05
            },
            'preprocess_params': {
                'min_genes': 200,
                'min_cells': 3,
                'target_sum': 1e4,
                'n_top_genes': 2000,
                'cut_max_value': 10
            },
            'gae_params': {
                'gat_hidden_channels': trial.suggest_int('gat_hidden_channels', 64, 256, step=32),
                'gat_out_channels': trial.suggest_int('gat_out_channels', 32, 192, step=16),
                'head_num': trial.suggest_categorical('head_num', [1, 2, 4, 8]),
                'zinb_hidden_channels': 16,
                'lr': trial.suggest_loguniform('gae_lr', 1e-8, 1e-3),
                'alpha': trial.suggest_loguniform('alpha', 1e-6, 1.0),
                'beta': trial.suggest_loguniform('beta', 1e-6, 10.0),
                'epochs': trial.suggest_int('gae_epochs', 150, 300, step=25)
            },
            'lrr_params': {
                'lrr_lambda': trial.suggest_loguniform('lrr_lambda', 1e-6, 1e-3),
                'lrr_gamma': trial.suggest_loguniform('lrr_gamma', 1e-6, 1e-2),
                'lrr_weight': trial.suggest_uniform('lrr_weight', 0.1, 0.9),
                'lr': trial.suggest_loguniform('lrr_lr', 1e-6, 1e-3),
                'epochs': trial.suggest_int('lrr_epochs_alt', 80, 150, step=10)
            },
            'ablation_params': {
                'zinb_loss': False,
                'manifold_loss': trial.suggest_categorical('manifold_loss', [True, False]),
                'low_rank_reg': trial.suggest_categorical('low_rank_reg', [True, False]),
                'block_reg': trial.suggest_categorical('block_reg', [True, False])
            }
        }
        
        
        
        # 运行 ab2_GAEreconAfoLRR 实验
        ari, nmi, ami, aom = ab2_GAEreconAfoLRR(dataset, dataset_dir, params)
        
        # 计算目标值：主要优化 ARI 和 NMI，也考虑 AMI
        objective_value = 0.4 * ari + 0.4 * nmi + 0.2 * ami
        
        print(f"试验 {trial.number} 结果:")
        print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}, AMI: {ami:.4f}, AOM: {aom:.4f}")
        print(f"目标值: {objective_value:.4f}")
        
        # 存储结果用于后续分析
        trial.set_user_attr('ari', ari)
        trial.set_user_attr('nmi', nmi)
        trial.set_user_attr('ami', ami)
        trial.set_user_attr('aom', aom)
        
        return objective_value
        
    except Exception as e:
        print(f"试验 {trial.number} 失败: {str(e)}")
        # 返回一个很低的值，让 Optuna 知道这次试验失败了
        return -1.0

def optimize_ab2_single_dataset(dataset='Muraro', n_trials=50, dataset_dir=DATASET_DIR):
    """
    对单个数据集运行 ab2_GAEreconAfoLRR 的超参数优化
    """
    print(f"\n===== 开始为数据集 {dataset} 优化 ab2_GAEreconAfoLRR 模型 =====")
    
    # 数据预处理
    print("开始数据预处理...")
    try:
        knn_params = abl2_default_params['knn_params']
        preprocess_params = abl2_default_params['preprocess_params']
        Preprocessed([dataset], dataset_dir, knn_params, preprocess_params)
        print(f"数据集 {dataset} 预处理完成")
    except Exception as e:
        print(f"数据预处理失败: {e}")
        return None, None
    
    # 创建 Optuna study
    study_name = f"ab2_GAEreconAfoLRR_{dataset}_optimization"
    storage_name = f"sqlite:///ab2_optimization_{dataset}.db"
    
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        print(f"加载现有study，已完成 {len(study.trials)} 次试验")
    except:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        print("创建新study")
    
    # 运行优化
    print(f"开始运行 {n_trials} 次优化试验...")
    start_time = time.time()
    
    study.optimize(
        lambda trial: objective_ab2(trial, dataset, dataset_dir), 
        n_trials=n_trials,
        timeout=None  # 不设置时间限制
    )
    
    end_time = time.time()
    print(f"优化完成，总用时: {(end_time - start_time) / 60:.2f} 分钟")
    
    # 分析结果
    if len(study.trials) == 0:
        print("没有成功的试验")
        return None, None
    
    # 获取最佳试验
    best_trial = study.best_trial
    print(f"\n===== 数据集 {dataset} 的最佳参数 =====")
    print(f"最佳目标值: {best_trial.value:.4f}")
    print(f"最佳 ARI: {best_trial.user_attrs.get('ari', 'N/A'):.4f}")
    print(f"最佳 NMI: {best_trial.user_attrs.get('nmi', 'N/A'):.4f}")
    print(f"最佳 AMI: {best_trial.user_attrs.get('ami', 'N/A'):.4f}")
    print(f"最佳 AOM: {best_trial.user_attrs.get('aom', 'N/A'):.4f}")
    
    print("\n最佳超参数:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # 保存结果
    results_dir = f"ab2_optimization_results/{dataset}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存最佳参数
    best_params = reconstruct_params_from_trial(best_trial)
    best_params['best_objective'] = best_trial.value
    best_params['best_metrics'] = {
        'ari': best_trial.user_attrs.get('ari', 0),
        'nmi': best_trial.user_attrs.get('nmi', 0),
        'ami': best_trial.user_attrs.get('ami', 0),
        'aom': best_trial.user_attrs.get('aom', 0)
    }
    
    # 保存到文件
    import json
    with open(f"{results_dir}/best_params.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # 生成优化历史图
    plt.figure(figsize=(12, 5))
    
    # 目标值历史
    plt.subplot(1, 2, 1)
    objective_values = [t.value for t in study.trials if t.value is not None]
    plt.plot(objective_values, 'b-o', markersize=3)
    plt.title(f'{dataset} - 优化历史')
    plt.xlabel('试验次数')
    plt.ylabel('目标值')
    plt.grid(True)
    
    # ARI 历史
    plt.subplot(1, 2, 2)
    ari_values = [t.user_attrs.get('ari', 0) for t in study.trials if t.value is not None]
    plt.plot(ari_values, 'r-o', markersize=3)
    plt.title(f'{dataset} - ARI 历史')
    plt.xlabel('试验次数')
    plt.ylabel('ARI')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/optimization_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存试验数据
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"{results_dir}/all_trials.csv", index=False)
    
    print(f"结果已保存到 {results_dir}/")
    
    return best_params, study

def reconstruct_params_from_trial(trial):
    """从 Optuna trial 重构完整的参数字典"""
    params = {
        'knn_params': {
            'k_knn': 20,
            'metric': 'cosine',
            'threshold': 0.05
        },
        'preprocess_params': {
            'min_genes': 200,
            'min_cells': 3,
            'target_sum': 1e4,
            'n_top_genes': 2000,
            'cut_max_value': 10
        },
        'gae_params': {
            'gat_hidden_channels': trial.params['gat_hidden_channels'],
            'gat_out_channels': trial.params['gat_out_channels'],
            'head_num': trial.params['head_num'],
            'zinb_hidden_channels': trial.params['zinb_hidden_channels'],
            'lr': trial.params['gae_lr'],
            'alpha': trial.params['alpha'],
            'beta': trial.params['beta'],
            'epochs': trial.params['gae_epochs']
        },
        'lrr_params': {
            'lrr_lambda': trial.params['lrr_lambda'],
            'lrr_gamma': trial.params['lrr_gamma'],
            'lrr_weight': trial.params['lrr_weight'],
            'lrr_epochs': trial.params['lrr_epochs'],
            'lr': trial.params['lrr_lr'],
            'epochs': trial.params['lrr_epochs_alt']
        },
        'ablation_params': {
            'zinb_loss': False,
            'manifold_loss': trial.params['manifold_loss'],
            'low_rank_reg': trial.params['low_rank_reg'],
            'block_reg': trial.params['block_reg']
        }
    }
    return params

def optimize_ab2_multiple_datasets(datasets=None, n_trials=50, dataset_dir=DATASET_DIR):
    """
    对多个数据集分别运行 ab2_GAEreconAfoLRR 的超参数优化
    """
    if datasets is None:
        datasets = [
            'Muraro',
            'Quake_10x_Spleen',
            'Quake_Smart-seq2_Lung',
            'Adam',
            'Young'
        ]
    
    print(f"===== 开始为 {len(datasets)} 个数据集优化 ab2_GAEreconAfoLRR 模型 =====")
    
    all_best_params = {}
    all_studies = {}
    summary_results = []
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{'='*60}")
        print(f"正在优化数据集 {i}/{len(datasets)}: {dataset}")
        print(f"{'='*60}")
        
        try:
            best_params, study = optimize_ab2_single_dataset(
                dataset=dataset, 
                n_trials=n_trials, 
                dataset_dir=dataset_dir
            )
            
            if best_params is not None:
                all_best_params[dataset] = best_params
                all_studies[dataset] = study
                
                summary_results.append({
                    'dataset': dataset,
                    'best_objective': best_params['best_objective'],
                    'best_ari': best_params['best_metrics']['ari'],
                    'best_nmi': best_params['best_metrics']['nmi'],
                    'best_ami': best_params['best_metrics']['ami'],
                    'best_aom': best_params['best_metrics']['aom'],
                    'total_trials': len(study.trials)
                })
                print(f"数据集 {dataset} 优化成功")
            else:
                print(f"数据集 {dataset} 优化失败")
                
        except Exception as e:
            print(f"数据集 {dataset} 优化出错: {e}")
            continue
    
    # 生成汇总报告
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv('ab2_optimization_summary.csv', index=False)
        
        print(f"\n===== ab2_GAEreconAfoLRR 优化汇总 =====")
        print(f"成功优化的数据集: {len(summary_results)}/{len(datasets)}")
        print(f"平均最佳目标值: {summary_df['best_objective'].mean():.4f} ± {summary_df['best_objective'].std():.4f}")
        print(f"平均最佳 ARI: {summary_df['best_ari'].mean():.4f} ± {summary_df['best_ari'].std():.4f}")
        print(f"平均最佳 NMI: {summary_df['best_nmi'].mean():.4f} ± {summary_df['best_nmi'].std():.4f}")
        
        # 显示最佳数据集
        best_dataset_idx = summary_df['best_objective'].idxmax()
        best_dataset_info = summary_df.iloc[best_dataset_idx]
        print(f"\n表现最佳的数据集: {best_dataset_info['dataset']}")
        print(f"最佳目标值: {best_dataset_info['best_objective']:.4f}")
        print(f"最佳 ARI: {best_dataset_info['best_ari']:.4f}")
        print(f"最佳 NMI: {best_dataset_info['best_nmi']:.4f}")
    
    return all_best_params, all_studies, summary_results

def validate_ab2_best_params(best_params_dict, datasets=None, dataset_dir=DATASET_DIR):
    """
    使用最佳参数验证 ab2_GAEreconAfoLRR 模型性能
    """
    if datasets is None:
        datasets = list(best_params_dict.keys())
    
    print(f"===== 使用最佳参数验证 ab2_GAEreconAfoLRR 模型 =====")
    
    validation_results = []
    
    for dataset in datasets:
        if dataset not in best_params_dict:
            print(f"数据集 {dataset} 没有最佳参数，跳过")
            continue
            
        print(f"\n验证数据集: {dataset}")
        
        try:
            best_params = best_params_dict[dataset]
            
            # 移除优化相关的键
            params_for_validation = {k: v for k, v in best_params.items() 
                                   if k not in ['best_objective', 'best_metrics']}
            
            # 运行验证
            ari, nmi, ami, aom = ab2_GAEreconAfoLRR(dataset, dataset_dir, params_for_validation)
            
            validation_results.append({
                'dataset': dataset,
                'ari': ari,
                'nmi': nmi,
                'ami': ami,
                'aom': aom,
                'objective': 0.4 * ari + 0.4 * nmi + 0.2 * ami,
                'optuna_predicted_objective': best_params.get('best_objective', 0)
            })
            
            print(f"验证结果 - ARI: {ari:.4f}, NMI: {nmi:.4f}, AMI: {ami:.4f}, AOM: {aom:.4f}")
            
        except Exception as e:
            print(f"数据集 {dataset} 验证失败: {e}")
            continue
    
    # 保存验证结果
    if validation_results:
        validation_df = pd.DataFrame(validation_results)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        validation_file = f'ab2_validation_results_{timestamp}.csv'
        validation_df.to_csv(validation_file, index=False)
        
        print(f"\n===== ab2_GAEreconAfoLRR 验证汇总 =====")
        print(f"验证的数据集数量: {len(validation_results)}")
        print(f"平均 ARI: {validation_df['ari'].mean():.4f} ± {validation_df['ari'].std():.4f}")
        print(f"平均 NMI: {validation_df['nmi'].mean():.4f} ± {validation_df['nmi'].std():.4f}")
        print(f"平均 AMI: {validation_df['ami'].mean():.4f} ± {validation_df['ami'].std():.4f}")
        print(f"平均 AOM: {validation_df['aom'].mean():.4f} ± {validation_df['aom'].std():.4f}")
        print(f"验证结果已保存到: {validation_file}")
        
        # 计算预测准确性
        if 'optuna_predicted_objective' in validation_df.columns:
            prediction_error = validation_df['objective'] - validation_df['optuna_predicted_objective']
            print(f"\nOptuna 预测准确性:")
            print(f"预测误差 (实际 - 预测): {prediction_error.mean():.4f} ± {prediction_error.std():.4f}")
    
    return validation_results

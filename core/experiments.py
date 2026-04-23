
import anndata
import os



from data import Preprocessed, Load_Data
from config import *
from newtrain import *
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def default_model():
    params = Young_best_params
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
    # dataset_list = ['Muraro']
    knn_params = params['knn_params']
    preprocess_params = params['preprocess_params']
    dataset_dir = DATASET_DIR
    Preprocessed(dataset_list,dataset_dir, knn_params, preprocess_params)
    # 导入数据
    ari_list = []
    nmi_list = []
    ami_list = []
    aom_list = []
    for dataset_name in dataset_list:
        ari, nmi, ami, aom = run_experiment_new(dataset_name, dataset_dir, params)
        print(f"LRR谱聚类 - ARI: {ari:.4f}, NMI: {nmi:.4f}, AMI: {ami:.4f}, AOM: {aom:.4f}")
        ari_list.append(ari)
        nmi_list.append(nmi)
        ami_list.append(ami)
        aom_list.append(aom)
    # 保存结果到CSV文件
    import pandas as pd
    
    results_df = pd.DataFrame({
        'dataset_name': dataset_list,
        'ari': ari_list,
        'nmi': nmi_list,
        'ami': ami_list,
        'aom': aom_list
    })
    
    # 保存到CSV文件
    # 生成带时间戳的文件名
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/clustering_results_{timestamp}.csv'
    
    # 确保results目录存在
    os.makedirs('results', exist_ok=True)

    results_df.to_csv(filename, index=False)
    print(f"\n结果已保存到 {filename}")
    
    # 打印汇总统计
    
    return

def opt_model():
    """
    使用Optuna为每个数据集分别优化模型参数
    """
    print("===== 开始为所有数据集分别进行参数优化 =====")
    
    # 导入优化模块
    from optimzation import optimize_all_datasets
    
    # 设置优化参数
    n_trials = 80  # 每个数据集的优化试验次数
    
    # 运行所有数据集的参数优化
    try:
        all_best_params, all_studies = optimize_all_datasets(n_trials=n_trials)
        
        print(f"\n===== 所有数据集优化完成，开始使用各自最佳参数运行验证 =====")
        
        # 使用每个数据集的最佳参数运行验证实验
        validation_results = []
        
        for dataset_name, best_params in all_best_params.items():
            try:
                print(f"\n使用最佳参数验证数据集: {dataset_name}")
                ari, nmi, ami, aom = run_experiment_new(dataset_name, DATASET_DIR, best_params)
                
                validation_results.append({
                    'Dataset': dataset_name,
                    'ARI': ari,
                    'NMI': nmi,
                    'AMI': ami,
                    'AOM': aom,
                    'Objective': (ari + nmi) / 2,
                    'Best_Objective_From_Optuna': best_params['best_objective']
                })
                
                print(f"{dataset_name}: ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, AOM={aom:.4f}")
                print(f"目标值: {(ari + nmi) / 2:.4f} (Optuna最佳: {best_params['best_objective']:.4f})")
                
            except Exception as e:
                print(f"数据集 {dataset_name} 验证失败: {e}")
                continue
        
        # 保存验证结果
        if validation_results:
            import pandas as pd
            from datetime import datetime
            
            results_df = pd.DataFrame(validation_results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 确保results目录存在
            os.makedirs('results', exist_ok=True)
            
            validation_file = f'results/dataset_optimization_validation_{timestamp}.csv'
            results_df.to_csv(validation_file, index=False)
            print(f"\n验证结果已保存到: {validation_file}")
            
            # 计算统计信息
            print(f"\n===== 分别优化后的汇总统计 =====")
            print(f"成功优化的数据集数量: {len(validation_results)}")
            print(f"平均 ARI: {results_df['ARI'].mean():.4f} ± {results_df['ARI'].std():.4f}")
            print(f"平均 NMI: {results_df['NMI'].mean():.4f} ± {results_df['NMI'].std():.4f}")
            print(f"平均 AMI: {results_df['AMI'].mean():.4f} ± {results_df['AMI'].std():.4f}")
            print(f"平均 AOM: {results_df['AOM'].mean():.4f} ± {results_df['AOM'].std():.4f}")
            print(f"平均目标值 (ARI+NMI)/2: {results_df['Objective'].mean():.4f} ± {results_df['Objective'].std():.4f}")
            
            # 比较Optuna预测和实际结果
            print(f"\n===== Optuna预测准确性 =====")
            objective_diff = results_df['Objective'] - results_df['Best_Objective_From_Optuna']
            print(f"预测误差 (实际 - 预测): {objective_diff.mean():.4f} ± {objective_diff.std():.4f}")
            print(f"预测误差范围: [{objective_diff.min():.4f}, {objective_diff.max():.4f}]")
            
            # 显示最佳和最差数据集
            best_dataset = results_df.loc[results_df['Objective'].idxmax()]
            worst_dataset = results_df.loc[results_df['Objective'].idxmin()]
            print(f"\n最佳数据集: {best_dataset['Dataset']} (目标值: {best_dataset['Objective']:.4f})")
            print(f"最差数据集: {worst_dataset['Dataset']} (目标值: {worst_dataset['Objective']:.4f})")
        
        return all_best_params, validation_results
        
    except Exception as e:
        print(f"优化过程出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def parameters_analysis():
    pass

def visualization_experiment(dataset_name='Muraro'):
    """
    进行ab2框架的可视化实验
    
    参数:
        dataset_name: 要可视化的数据集名称，默认为'Muraro'
    """
    print(f"===== 开始 {dataset_name} 数据集的可视化实验 =====")
    
    # 导入可视化模块
    from visualize import ab2_GAEreconAfoLRR_with_visualization
    from config import abl2_default_params, DATASET_DIR
    
    # 确保数据预处理
    params = abl2_default_params
    knn_params = params['knn_params']
    preprocess_params = params['preprocess_params']
    
    # 预处理单个数据集
    from data import Preprocessed
    Preprocessed([dataset_name], DATASET_DIR, knn_params, preprocess_params)
    
    try:
        # 运行带可视化的ab2框架
        ari, nmi, ami, aom = ab2_GAEreconAfoLRR_with_visualization(
            dataset_name=dataset_name,
            dataset_dir=DATASET_DIR, 
            params=params,
            figures_dir="figures"
        )
        
        print(f"\n===== {dataset_name} 可视化实验完成 =====")
        print(f"最终聚类性能:")
        print(f"  ARI: {ari:.4f}")
        print(f"  NMI: {nmi:.4f}")
        print(f"  AMI: {ami:.4f}")
        print(f"  AOM: {aom:.4f}")
        print(f"  平均性能: {(ari + nmi + ami + aom) / 4:.4f}")
        
        return ari, nmi, ami, aom
        
    except Exception as e:
        print(f"可视化实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def opt_ab2_model():
    """
    使用 optimization_new.py 为 ab2_GAEreconAfoLRR 模型进行超参数优化
    """
    print("===== 开始 ab2_GAEreconAfoLRR 模型的超参数优化 =====")
    
    # 导入新的优化模块
    from optimization_new import optimize_ab2_multiple_datasets, validate_ab2_best_params
    
    # 设置要优化的数据集列表
    datasets_to_optimize = [
                        'Muraro',
                        'Quake_10x_Bladder',
                        'Quake_10x_Limb_Muscle',
                        #'Quake_10x_Spleen',
                        'Quake_Smart-seq2_Diaphragm',
                        'Quake_Smart-seq2_Limb_Muscle',
                        'Quake_Smart-seq2_Lung',
                        #'Quake_Smart-seq2_Trachea',
                        'Adam',
                        'Romanov',
                        #'Young'
                    ]
    
    # 设置优化参数
    n_trials = 80  # 每个数据集的优化试验次数
    
    try:
        print(f"将对 {len(datasets_to_optimize)} 个数据集分别进行优化，每个数据集 {n_trials} 次试验")
        
        # 运行多数据集优化
        all_best_params, all_studies, summary_results = optimize_ab2_multiple_datasets(
            datasets=datasets_to_optimize,
            n_trials=n_trials,
            dataset_dir=DATASET_DIR
        )
        
        if all_best_params:
            print(f"\n===== 优化完成，开始验证最佳参数 =====")
            
            # 使用最佳参数进行验证
            validation_results = validate_ab2_best_params(
                best_params_dict=all_best_params,
                datasets=datasets_to_optimize,
                dataset_dir=DATASET_DIR
            )
            
            print(f"\n===== ab2_GAEreconAfoLRR 优化和验证完成 =====")
            print(f"成功优化的数据集: {len(all_best_params)}/{len(datasets_to_optimize)}")
            
            if validation_results:
                # 计算总体统计
                import pandas as pd
                val_df = pd.DataFrame(validation_results)
                print(f"验证平均性能:")
                print(f"  ARI: {val_df['ari'].mean():.4f} ± {val_df['ari'].std():.4f}")
                print(f"  NMI: {val_df['nmi'].mean():.4f} ± {val_df['nmi'].std():.4f}")
                print(f"  AMI: {val_df['ami'].mean():.4f} ± {val_df['ami'].std():.4f}")
                print(f"  AOM: {val_df['aom'].mean():.4f} ± {val_df['aom'].std():.4f}")
                
                # 找出表现最好的数据集
                best_dataset_idx = val_df['objective'].idxmax()
                best_result = val_df.iloc[best_dataset_idx]
                print(f"\n表现最佳的数据集: {best_result['dataset']}")
                print(f"  ARI: {best_result['ari']:.4f}")
                print(f"  NMI: {best_result['nmi']:.4f}")
                print(f"  目标值: {best_result['objective']:.4f}")
            
            return all_best_params, validation_results
        else:
            print("没有成功的优化结果")
            return None, None
            
    except Exception as e:
        print(f"ab2 模型优化过程出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def opt_ab2_single(dataset_name='Muraro', n_trials=20):
    """
    对单个数据集运行 ab2_GAEreconAfoLRR 的超参数优化
    
    Args:
        dataset_name: 要优化的数据集名称
        n_trials: 优化试验次数
    """
    print(f"===== 开始对数据集 {dataset_name} 进行 ab2_GAEreconAfoLRR 模型优化 =====")
    
    from optimization_new import optimize_ab2_single_dataset
    
    try:
        best_params, study = optimize_ab2_single_dataset(
            dataset=dataset_name,
            n_trials=n_trials,
            dataset_dir=DATASET_DIR
        )
        
        if best_params:
            print(f"\n===== 数据集 {dataset_name} 优化完成 =====")
            print(f"最佳目标值: {best_params['best_objective']:.4f}")
            print("最佳性能指标:")
            for metric, value in best_params['best_metrics'].items():
                print(f"  {metric.upper()}: {value:.4f}")
            
            # 运行验证
            print(f"\n开始验证数据集 {dataset_name} 的最佳参数...")
            params_for_validation = {k: v for k, v in best_params.items() 
                                   if k not in ['best_objective', 'best_metrics']}
            
            ari, nmi, ami, aom = ab2_GAEreconAfoLRR(dataset_name, DATASET_DIR, params_for_validation)
            
            print(f"验证结果:")
            print(f"  ARI: {ari:.4f}")
            print(f"  NMI: {nmi:.4f}")
            print(f"  AMI: {ami:.4f}")
            print(f"  AOM: {aom:.4f}")
            print(f"  目标值: {0.4 * ari + 0.4 * nmi + 0.2 * ami:.4f}")
            
            return best_params, (ari, nmi, ami, aom)
        else:
            print(f"数据集 {dataset_name} 优化失败")
            return None, None
            
    except Exception as e:
        print(f"数据集 {dataset_name} 优化出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def ablation_experiment(ablation_number):
    
    params = Young_best_params
    dataset_list = [
                        #'Muraro',
                        #'Quake_10x_Bladder',
                        #'Quake_10x_Limb_Muscle',
                        'Quake_10x_Spleen',
                        #'Quake_Smart-seq2_Diaphragm',
                        #'Quake_Smart-seq2_Limb_Muscle',
                        'Quake_Smart-seq2_Lung',
                        'Quake_Smart-seq2_Trachea',
                        'Adam',
                        'Romanov',
                        'Young'
                    ]
    # dataset_list = ['Muraro']
    knn_params = params['knn_params']
    preprocess_params = params['preprocess_params']
    dataset_dir = DATASET_DIR
    Preprocessed(dataset_list,dataset_dir, knn_params, preprocess_params)
    # 导入数据
    ari_list = []
    nmi_list = []
    ami_list = []
    aom_list = []
    for dataset_name in dataset_list:
        if ablation_number == 1:
            ari, nmi, ami, aom = abl1_XtoLRR(dataset_name, DATASET_DIR, params)
        elif ablation_number == 2:
            ari, nmi, ami, aom = ab2_GAEreconAfoLRR(dataset_name, DATASET_DIR, abl2_default_params)
        elif ablation_number == 3:
            ari, nmi, ami, aom = abl3_GAEclustering(dataset_name, DATASET_DIR, default_params)
        elif ablation_number == 4:
            ari, nmi, ami, aom = abl4_LRRKmeans(dataset_name, DATASET_DIR, default_params)
        elif ablation_number == 5:
            ari, nmi, ami, aom = Framework2(dataset_name, DATASET_DIR, default_params)
        ari_list.append(ari)
        nmi_list.append(nmi)
        ami_list.append(ami)
        aom_list.append(aom)
    # 保存结果到CSV文件
    import pandas as pd
    results_df = pd.DataFrame({
        'dataset_name': dataset_list,
        'ari': ari_list,
        'nmi': nmi_list,
        'ami': ami_list,
        'aom': aom_list
    })
    # 生成带时间戳的文件名
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'ablation_{ablation_number}_results_{timestamp}.csv'
    results_df.to_csv('results/'+filename, index=False)
    return


def main():
    os.chdir(WORK_DIR)
    
    print("===== scLGAE 单细胞聚类模型主程序 =====")
    print("可用的运行模式:")
    print("1. default_model() - 运行默认模型")
    print("2. opt_model() - 原始优化模块")
    print("3. opt_ab2_model() - ab2_GAEreconAfoLRR 模型超参数优化")
    print("4. opt_ab2_single() - 单个数据集的 ab2 模型优化")
    print("5. ablation_experiment() - 消融实验")
    print("6. parameters_analysis() - 参数分析")
    print("7. visualization_experiment() - ab2框架可视化实验")
    
    # 根据需要选择运行模式
    # default_model()
    # opt_model()
    # opt_ab2_model()  # 新的 ab2 模型优化
    # opt_ab2_single('Muraro', 30)  # 单个数据集优化示例
    # ablation_experiment(5)  # 当前运行的是消融实验5
    
    # 运行可视化实验 - 在固定参数下对Muraro数据集进行ab2框架聚类并可视化三个阶段
    visualization_experiment('Young')
    

if __name__ == "__main__":
    main()
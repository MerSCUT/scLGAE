#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 optimization_new.py 模块的集成和基本功能
"""

import sys
import os
import traceback

def test_import():
    """测试模块导入"""
    print("===== 测试模块导入 =====")
    try:
        from optimization_new import (
            objective_ab2, 
            optimize_ab2_single_dataset,
            optimize_ab2_multiple_datasets,
            validate_ab2_best_params,
            reconstruct_params_from_trial
        )
        print("✓ optimization_new 模块导入成功")
        return True
    except Exception as e:
        print(f"✗ optimization_new 模块导入失败: {e}")
        traceback.print_exc()
        return False

def test_newtrain_import():
    """测试 newtrain 模块中 ab2_GAEreconAfoLRR 函数的导入"""
    print("\n===== 测试 newtrain 导入 =====")
    try:
        from newtrain import ab2_GAEreconAfoLRR
        print("✓ ab2_GAEreconAfoLRR 函数导入成功")
        return True
    except Exception as e:
        print(f"✗ ab2_GAEreconAfoLRR 函数导入失败: {e}")
        traceback.print_exc()
        return False

def test_config_import():
    """测试配置文件导入"""
    print("\n===== 测试配置导入 =====")
    try:
        from config import DATASET_DIR, abl2_default_params
        print(f"✓ 配置导入成功")
        print(f"  DATASET_DIR: {DATASET_DIR}")
        print(f"  abl2_default_params 包含键: {list(abl2_default_params.keys())}")
        return True
    except Exception as e:
        print(f"✗ 配置导入失败: {e}")
        traceback.print_exc()
        return False

def test_param_reconstruction():
    """测试参数重构功能"""
    print("\n===== 测试参数重构功能 =====")
    try:
        from optimization_new import reconstruct_params_from_trial
        
        # 创建一个模拟的 trial 对象
        class MockTrial:
            def __init__(self):
                self.params = {
                    'k_knn': 30,
                    'metric': 'cosine',
                    'threshold': 0.05,
                    'target_sum': 10000.0,
                    'n_top_genes': 2000,
                    'cut_max_value': 10,
                    'gat_hidden_channels': 128,
                    'gat_out_channels': 64,
                    'head_num': 4,
                    'zinb_hidden_channels': 16,
                    'gae_lr': 0.0001,
                    'alpha': 0.01,
                    'beta': 0.001,
                    'gae_epochs': 200,
                    'lrr_lambda': 0.00001,
                    'lrr_gamma': 0.0001,
                    'lrr_weight': 0.5,
                    'lrr_epochs': 100,
                    'lrr_lr': 0.00001,
                    'lrr_epochs_alt': 100,
                    'zinb_loss': False,
                    'manifold_loss': True,
                    'low_rank_reg': False,
                    'block_reg': True
                }
        
        mock_trial = MockTrial()
        reconstructed_params = reconstruct_params_from_trial(mock_trial)
        
        print("✓ 参数重构功能测试成功")
        print(f"  重构的参数包含键: {list(reconstructed_params.keys())}")
        print(f"  GAE 参数: {list(reconstructed_params['gae_params'].keys())}")
        print(f"  LRR 参数: {list(reconstructed_params['lrr_params'].keys())}")
        return True
        
    except Exception as e:
        print(f"✗ 参数重构功能测试失败: {e}")
        traceback.print_exc()
        return False

def test_main_interface():
    """测试 main.py 中的新接口"""
    print("\n===== 测试 main.py 接口 =====")
    try:
        from main import opt_ab2_model, opt_ab2_single
        print("✓ main.py 中的 ab2 优化接口导入成功")
        print("  - opt_ab2_model: 多数据集优化")
        print("  - opt_ab2_single: 单数据集优化")
        return True
    except Exception as e:
        print(f"✗ main.py 接口测试失败: {e}")
        traceback.print_exc()
        return False

def test_optuna_availability():
    """测试 Optuna 是否可用"""
    print("\n===== 测试 Optuna 可用性 =====")
    try:
        import optuna
        print(f"✓ Optuna 可用，版本: {optuna.__version__}")
        
        # 测试创建一个简单的 study
        study = optuna.create_study(direction='maximize')
        print("✓ Optuna study 创建成功")
        return True
    except Exception as e:
        print(f"✗ Optuna 不可用: {e}")
        return False

def test_data_dependencies():
    """测试数据相关依赖"""
    print("\n===== 测试数据相关依赖 =====")
    try:
        from data import Preprocessed
        from utils import set_random_seed
        print("✓ 数据处理模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 数据相关依赖测试失败: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("开始运行 optimization_new.py 集成测试...\n")
    
    tests = [
        test_import,
        test_newtrain_import,
        test_config_import,
        test_param_reconstruction,
        test_main_interface,
        test_optuna_availability,
        test_data_dependencies
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print(f"\n===== 测试汇总 =====")
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！optimization_new.py 模块集成成功")
        print("\n使用说明:")
        print("1. 在 main.py 中调用 opt_ab2_model() 进行多数据集优化")
        print("2. 在 main.py 中调用 opt_ab2_single('数据集名称', 试验次数) 进行单数据集优化")
        print("3. 优化结果将保存在 ab2_optimization_results/ 目录中")
        print("4. 每个数据集的最佳参数将保存为 JSON 文件")
    else:
        print(f"❌ 有 {total - passed} 个测试失败，请检查相关依赖和模块")
    
    return passed == total

if __name__ == "__main__":
    # 设置工作目录
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)

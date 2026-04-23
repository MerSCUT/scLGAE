import torch
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42

DATASET_DIR = r"G:/GSingleCell/data"

WORK_DIR = r"G:/GSingleCell/scLGAE"


default_params = {
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
        'gat_hidden_channels':128,
        'gat_out_channels':144,
        'head_num':4,
        'zinb_hidden_channels':16,
        'lr':0.00005871151865254813,
        'alpha':0.01,
        'beta':0.0002561432962688087,
        'epochs': 150
    },
    'lrr_params': {
        'lrr_lambda':0.000007507452161005634,
        'lrr_gamma':0.00045129085613869995,
        'lrr_weight':0.5811554384821028,
        'lrr_epochs':100,
        'lr':0.00005871151865254813,
        'epochs': 100
    },
    'ablation_params': {
        'zinb_loss': False,
        'manifold_loss': True,
        'low_rank_reg': False,
        'block_reg' : True
    }
}

Young_best_params = {
    "knn_params": {
    "k_knn": 20,
    "metric": "cosine",
    "threshold": 0.05
  },
  "preprocess_params": {
    "min_genes": 200,
    "min_cells": 3,
    "target_sum": 10000.0,
    "n_top_genes": 2000,
    "cut_max_value": 10
  },
  "gae_params": {
    "gat_hidden_channels": 64,
    "gat_out_channels": 64,
    "head_num": 8,
    "zinb_hidden_channels": 16,
    "lr": 5.351169399982244e-05,
    "alpha": 0.2710814977320114,
    "beta": 6.521270374720974,
    "epochs": 100
  },
  "lrr_params": {
    "lrr_lambda": 4.449644872934596e-05,
    "lrr_gamma": 1.7630502084296922,
    "lrr_weight": 0.6243528037126993,
    "lrr_epochs": 100,
    "lr": 0.00004834591653781593,
    "epochs": 100
  },
  "ablation_params": {
    "zinb_loss": False,
    "manifold_loss": False,
    'low_rank_reg': True,
    'block_reg' : True
  }
}

abl2_default_params = {
        'knn_params': {
            'k_knn': 30,
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
            'gat_hidden_channels':128,
            'gat_out_channels':144,
            'head_num':4,
            'zinb_hidden_channels':16,
            'lr':0.00005871151865254813,
            'alpha':0.01,
            'beta':0.0002561432962688087,
            'epochs': 200
        },
        'lrr_params': {
            'lrr_lambda':0.000007507452161005634,
            'lrr_gamma':0.0000045129085613869995,
            'lrr_weight':0.5811554384821028,
            'lrr_epochs':100,
            'lr':0.000005871151865254813,
            'epochs': 100
        },
        'ablation_params': {
            'zinb_loss': False,
            'manifold_loss': False,
            'low_rank_reg': True,
            'block_reg' : True
        }
    }
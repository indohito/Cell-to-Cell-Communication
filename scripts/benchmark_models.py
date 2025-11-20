#!/usr/bin/env python3
"""
Benchmarking script to compare GAT, GCN, and GraphSAGE models.

This script loads all trained models and evaluates them on the same test set,
producing a comprehensive comparison table with metrics and optional DeLong tests.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats

# Import models
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.gnn_model import GATLinkPredictor, BaselineGNNLinkPredictor
from models.graph_baselines import GCNLinkPredictor, SAGELinkPredictor
from training.utils import (
    get_device,
    load_graph_and_edges,
    build_data_splits,
    evaluate_model,
)

# Import ImprovedGATPredictor from train.py
# Note: This assumes the script is run from the project root
# If run from scripts/, we need to import differently
try:
    from train import ImprovedGATPredictor
except ImportError:
    # Fallback: add parent directory to path
    import importlib.util
    train_path = project_root / 'train.py'
    spec = importlib.util.spec_from_file_location("train", train_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    ImprovedGATPredictor = train_module.ImprovedGATPredictor


def delong_test(y_true, y_pred1, y_pred2):
    """
    DeLong test for comparing two ROC curves.
    
    Tests the null hypothesis that two ROC curves have equal AUC.
    
    Args:
        y_true: true binary labels
        y_pred1: predicted probabilities from model 1
        y_pred2: predicted probabilities from model 2
    
    Returns:
        tuple: (z_statistic, p_value)
    """
    from sklearn.metrics import roc_auc_score
    
    n = len(y_true)
    
    # Get AUCs
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    
    # Compute covariance matrix
    # This is a simplified version - full DeLong test uses more complex covariance
    # For simplicity, we'll use a bootstrap-like approximation
    
    # Simple approximation: use Mann-Whitney U test statistic
    # More accurate would be to compute the full covariance structure
    # For now, we'll use a simplified approach
    
    # Rank-based approach
    def compute_auc_variance(y_true, y_pred):
        """Compute variance of AUC using DeLong's method."""
        n1 = np.sum(y_true == 1)
        n0 = np.sum(y_true == 0)
        
        if n1 == 0 or n0 == 0:
            return 0.0
        
        # Get predictions for positive and negative classes
        pred_pos = y_pred[y_true == 1]
        pred_neg = y_pred[y_true == 0]
        
        # Compute V10 and V01 (simplified)
        # V10[i] = mean indicator that pred_pos[i] > pred_neg[j] for all j
        v10 = np.array([np.mean(pred_pos[i] > pred_neg) for i in range(n1)])
        v01 = np.array([np.mean(pred_pos > pred_neg[j]) for j in range(n0)])
        
        # Variance components
        s10 = np.var(v10) / n1
        s01 = np.var(v01) / n0
        
        return s10 + s01
    
    # Compute variances
    var1 = compute_auc_variance(y_true, y_pred1)
    var2 = compute_auc_variance(y_true, y_pred2)
    
    # Compute covariance (simplified)
    # For independent models, covariance is approximately 0
    # In practice, we'd compute the full covariance structure
    cov = 0.0  # Simplified - assumes independence
    
    # Compute z-statistic
    diff = auc1 - auc2
    se = np.sqrt(var1 + var2 - 2 * cov)
    
    if se == 0:
        z_stat = 0.0
    else:
        z_stat = diff / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_value


def load_model(model_path, model_type, input_dim, hidden_dim=128, device='cpu'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: path to model checkpoint
        model_type: 'gat_baseline', 'gat_affinity', 'gcn', or 'sage'
        input_dim: input dimension
        hidden_dim: hidden dimension (default 128 for GAT, 64 for GCN/SAGE)
        device: torch device
    
    Returns:
        loaded model
    """
    if not Path(model_path).exists():
        return None
    
    if model_type == 'gat_baseline':
        # Load ImprovedGATPredictor (baseline)
        model = ImprovedGATPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=0.2,
            use_affinity=False
        )
    elif model_type == 'gat_affinity':
        # Load ImprovedGATPredictor (affinity-weighted)
        model = ImprovedGATPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=0.2,
            use_affinity=True
        )
    elif model_type == 'gcn':
        model = GCNLinkPredictor(
            input_dim=input_dim,
            hidden_dim=64,  # GCN uses 64 by default
            num_layers=2,
            dropout=0.2
        )
    elif model_type == 'sage':
        model = SAGELinkPredictor(
            input_dim=input_dim,
            hidden_dim=64,  # SAGE uses 64 by default
            num_layers=2,
            dropout=0.2
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def get_predictions(model, data, split='test', device=None, use_weights=False):
    """
    Get predictions from a model on a given split.
    
    Args:
        model: the model
        data: data dict from build_data_splits
        split: 'test', 'val', or 'train'
        device: torch device
        use_weights: whether to use affinity weights
    
    Returns:
        tuple: (predictions, labels) as numpy arrays
    """
    if device is None:
        device = get_device()
    
    model.eval()
    x = data['x_scaled'].to(device)
    n_nodes = x.shape[0]
    
    s_idx = torch.tensor(data['s_idx'], dtype=torch.long, device=device)
    t_idx = torch.tensor(data['t_idx'], dtype=torch.long, device=device)
    weights = torch.tensor(data['weights'], dtype=torch.float32, device=device)
    
    if split == 'val':
        eval_idx = data['val_idx']
    elif split == 'test':
        eval_idx = data['test_idx']
    else:
        eval_idx = data['train_idx']
    
    # Create set of positive edges for negative sampling
    pos_edge_set = set()
    for si, ti in zip(s_idx.cpu().numpy(), t_idx.cpu().numpy()):
        pos_edge_set.add((int(si), int(ti)))
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # Get node embeddings
        if hasattr(model, 'encoder'):
            h_all = model.encoder(x)
        else:
            h_all = model(x, edge_index=None)
        
        # Positive predictions
        s_pos = s_idx[eval_idx]
        t_pos = t_idx[eval_idx]
        w_pos = weights[eval_idx] if use_weights else torch.ones_like(weights[eval_idx])
        
        if hasattr(model, 'predict_edges'):
            pos_edge_index = torch.stack([s_pos, t_pos], dim=0)
            p_pos = model.predict_edges(h_all, pos_edge_index, w_pos if use_weights else None)
            p_pos = p_pos.squeeze()
        else:
            h_s_pos = h_all[s_pos]
            h_t_pos = h_all[t_pos]
            p_pos = torch.sigmoid(torch.sum(h_s_pos * h_t_pos, dim=1) * w_pos)
        
        all_preds.append(p_pos.cpu().numpy())
        all_labels.append(np.ones(len(eval_idx)))
        
        # Negative predictions (sample same number as positives)
        n_neg = len(eval_idx)
        neg_edges = []
        for _ in range(n_neg):
            s_n = np.random.randint(0, n_nodes)
            t_n = np.random.randint(0, n_nodes)
            attempts = 0
            while (s_n, t_n) in pos_edge_set and attempts < 10:
                s_n = np.random.randint(0, n_nodes)
                t_n = np.random.randint(0, n_nodes)
                attempts += 1
            neg_edges.append((s_n, t_n))
        
        s_neg = torch.tensor([e[0] for e in neg_edges], dtype=torch.long, device=device)
        t_neg = torch.tensor([e[1] for e in neg_edges], dtype=torch.long, device=device)
        
        if hasattr(model, 'predict_edges'):
            neg_edge_index = torch.stack([s_neg, t_neg], dim=0)
            p_neg = model.predict_edges(h_all, neg_edge_index, None)
            p_neg = p_neg.squeeze()
        else:
            h_s_neg = h_all[s_neg]
            h_t_neg = h_all[t_neg]
            p_neg = torch.sigmoid(torch.sum(h_s_neg * h_t_neg, dim=1))
        
        all_preds.append(p_neg.cpu().numpy())
        all_labels.append(np.zeros(len(neg_edges)))
    
    # Combine predictions
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    return preds, labels


def main():
    print("="*70)
    print("MODEL BENCHMARKING: GAT vs GCN vs GraphSAGE")
    print("="*70)
    print()
    print("NOTE: This benchmark evaluates all models on BOTH:")
    print("      - VALIDATION SET (for comparison with training history)")
    print("      - TEST SET (held-out, for final performance evaluation)")
    print()
    
    # Load data
    print("Loading graph and preparing test split...")
    graph, edge_list = load_graph_and_edges()
    data = build_data_splits(graph, edge_list)
    print(f"Test set contains {len(data['test_idx'])} edges for evaluation")
    
    device = get_device()
    input_dim = graph.x.shape[1]
    
    print(f"Graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
    print(f"Test set size: {len(data['test_idx'])} edges")
    print(f"Using device: {device}\n")
    
    # Define models to benchmark
    models_to_load = [
        ('GAT_baseline', 'results/models/model_improved_baseline.pt', 'gat_baseline', 128, False),
        ('GAT_affinity', 'results/models/model_improved_affinity_weighted.pt', 'gat_affinity', 128, True),
        ('GCN', 'results/models/model_gcn.pt', 'gcn', 64, False),
        ('SAGE', 'results/models/model_sage.pt', 'sage', 64, False),
    ]
    
    results = []
    results_val = []  # Validation metrics
    predictions_dict = {}
    predictions_dict_val = {}  # Validation predictions for DeLong test
    
    # Load and evaluate each model
    print("Loading and evaluating models...\n")
    for model_name, model_path, model_type, hidden_dim, use_weights in models_to_load:
        print(f"  [{model_name}] Loading from {model_path}...")
        
        model = load_model(model_path, model_type, input_dim, hidden_dim, device)
        
        if model is None:
            print(f"    ⚠️  Model file not found, skipping {model_name}")
            continue
        
        # Evaluate on VALIDATION set (for comparison with training history)
        print(f"    Evaluating on VALIDATION set...")
        metrics_val = evaluate_model(model, data, split='val', device=device, use_weights=use_weights)
        preds_val, labels_val = get_predictions(model, data, split='val', device=device, use_weights=use_weights)
        predictions_dict_val[model_name] = (preds_val, labels_val)
        
        results_val.append({
            'Model': model_name,
            'ROC-AUC': metrics_val['roc_auc'],
            'PR-AUC': metrics_val['pr_auc'],
            'Precision': metrics_val['precision'],
            'Recall': metrics_val['recall'],
            'F1': metrics_val['f1'],
        })
        
        print(f"    ✓ Val ROC-AUC: {metrics_val['roc_auc']:.4f}, PR-AUC: {metrics_val['pr_auc']:.4f}")
        
        # Evaluate on TEST SET (held-out, for final performance)
        print(f"    Evaluating on TEST set (held-out)...")
        metrics = evaluate_model(model, data, split='test', device=device, use_weights=use_weights)
        
        # Get predictions for DeLong test (using test set)
        preds, labels = get_predictions(model, data, split='test', device=device, use_weights=use_weights)
        predictions_dict[model_name] = (preds, labels)
        
        results.append({
            'Model': model_name,
            'ROC-AUC': metrics['roc_auc'],
            'PR-AUC': metrics['pr_auc'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
        })
        
        print(f"    ✓ Test ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}\n")
    
    if not results:
        print("\n❌ No models were successfully loaded. Please train models first.")
        return
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Compute differences relative to GAT_affinity
    if 'GAT_affinity' in df_results['Model'].values:
        gat_aff_roc = df_results[df_results['Model'] == 'GAT_affinity']['ROC-AUC'].values[0]
        df_results['ΔROC vs GAT_aff'] = df_results['ROC-AUC'] - gat_aff_roc
    else:
        df_results['ΔROC vs GAT_aff'] = np.nan
    
    # Run DeLong tests if we have GAT_affinity predictions
    if 'GAT_affinity' in predictions_dict:
        gat_aff_preds, gat_aff_labels = predictions_dict['GAT_affinity']
        p_values = []
        
        for model_name in df_results['Model'].values:
            if model_name == 'GAT_affinity':
                p_values.append(np.nan)
            elif model_name in predictions_dict:
                preds, labels = predictions_dict[model_name]
                # Ensure same labels for comparison
                if len(labels) == len(gat_aff_labels):
                    _, p_val = delong_test(gat_aff_labels, gat_aff_preds, preds)
                    p_values.append(p_val)
                else:
                    p_values.append(np.nan)
            else:
                p_values.append(np.nan)
        
        df_results['p-value (vs GAT_aff)'] = p_values
    else:
        df_results['p-value (vs GAT_aff)'] = np.nan
    
    # Create validation results DataFrame
    df_results_val = pd.DataFrame(results_val)
    
    # Print results tables
    print("\n" + "="*70)
    print("BENCHMARK RESULTS - VALIDATION SET")
    print("="*70)
    print()
    print(df_results_val.to_string(index=False, float_format='%.4f'))
    print()
    
    print("\n" + "="*70)
    print("BENCHMARK RESULTS - TEST SET")
    print("="*70)
    print()
    
    # Format table
    print(df_results.to_string(index=False, float_format='%.4f'))
    print()
    
    # Save results
    Path('results').mkdir(exist_ok=True, parents=True)
    output_path = 'results/model_benchmark_comparison.csv'
    df_results.to_csv(output_path, index=False)
    print(f"Saved TEST SET results to {output_path}")
    
    output_path_val = 'results/model_benchmark_comparison_val.csv'
    df_results_val.to_csv(output_path_val, index=False)
    print(f"Saved VALIDATION SET results to {output_path_val}")
    
    # Create visualizations
    if len(results) > 0:
        print("\nCreating visualizations...")
        Path('results/figures').mkdir(exist_ok=True, parents=True)
        
        models = df_results['Model'].values
        roc_aucs_test = df_results['ROC-AUC'].values
        roc_aucs_val = df_results_val['ROC-AUC'].values
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Figure 1: Test Set Comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        bars = ax.bar(models, roc_aucs_test, color=colors[:len(models)], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, roc_aucs_test):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison: ROC-AUC (Test Set)', fontsize=14, fontweight='bold')
        ax.set_ylim([min(roc_aucs_test) - 0.01, max(roc_aucs_test) + 0.01])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        fig_path = 'results/figures/model_benchmark_comparison.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
        plt.close()
        
        # Figure 2: Validation Set Comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        bars = ax.bar(models, roc_aucs_val, color=colors[:len(models)], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, roc_aucs_val):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison: ROC-AUC (Validation Set)', fontsize=14, fontweight='bold')
        ax.set_ylim([min(roc_aucs_val) - 0.01, max(roc_aucs_val) + 0.01])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        fig_path_val = 'results/figures/model_benchmark_comparison_val.png'
        plt.savefig(fig_path_val, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {fig_path_val}")
        plt.close()
        
        # Figure 3: Side-by-side comparison (Test vs Validation)
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, roc_aucs_val, width, label='Validation Set', 
                      color='#1f77b4', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, roc_aucs_test, width, label='Test Set', 
                      color='#ff7f0e', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison: Validation vs Test Set', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.set_ylim([min(min(roc_aucs_val), min(roc_aucs_test)) - 0.01, 
                    max(max(roc_aucs_val), max(roc_aucs_test)) + 0.01])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        fig_path_both = 'results/figures/model_benchmark_comparison_both.png'
        plt.savefig(fig_path_both, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {fig_path_both}")
        plt.close()
    
    print("\n" + "="*70)
    print("Benchmarking complete!")
    print("="*70)


if __name__ == '__main__':
    main()


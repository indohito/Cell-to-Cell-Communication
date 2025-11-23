#!/usr/bin/env python3
"""
AffinityCCI Experiment V2: Advanced Metrics
Tracks ROC-AUC, PR-AUC, F1, and MSE for Baseline vs Weighted Models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import copy
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.utils import negative_sampling

try:
    from models.egnn_model import create_egnn_model, GRAPELoss
except ImportError:
    sys.exit("ERROR: models/egnn_model.py not found")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    work_dir = Path(".")
    graph_path = work_dir / "precision_graph.pt"
    out_dir = work_dir / "results" / "experiment_v2"
    
    hidden_dim = 64
    edge_dim = 3
    num_layers = 2
    num_heads = 4
    dropout = 0.3
    lr = 0.001
    epochs = 100

config = Config()
config.out_dir.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# TRAINING LOGIC
# ==============================================================================
def train_model(train_graph, gt_graph, mode_name, lambda_affinity):
    print(f"\n>>> STARTING RUN: {mode_name} (Lambda={lambda_affinity})")
    
    # 1. Setup
    model_config = copy.copy(config)
    model_config.input_dim = train_graph.x.shape[1]
    model = create_egnn_model(model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    criterion = GRAPELoss(lambda_affinity=lambda_affinity)
    
    # 2. Masks
    num_edges = train_graph.num_edges
    indices = torch.randperm(num_edges, device=device)
    split = int(0.8 * num_edges)
    train_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    train_mask[indices[:split]] = True
    test_mask[indices[split:]] = True
    
    # 3. Labels
    link_labels = torch.ones(num_edges, device=device)
    train_aff_true = train_graph.edge_attr[:, 1]
    train_aff_mask = train_graph.edge_attr[:, 2]
    val_aff_true = gt_graph.edge_attr[:, 1] # Ground Truth for Val
    val_aff_mask = gt_graph.edge_attr[:, 2] # Ground Truth for Val
    
    history = []
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass (Positive Edges)
        l_pred, a_pred, _ = model(train_graph.x, train_graph.edge_index, train_graph.edge_attr)
        
        loss = criterion(
            l_pred[train_mask], link_labels[train_mask],
            a_pred[train_mask], train_aff_true[train_mask], train_aff_mask[train_mask]
        )
        
        loss.backward()
        optimizer.step()
        
        # --- VALIDATION & METRICS ---
        model.eval()
        with torch.no_grad():
            # A. Positive Samples (From Test Set)
            pos_logits, pos_aff, _ = model(train_graph.x, train_graph.edge_index, train_graph.edge_attr)
            pos_logits_val = pos_logits[test_mask]
            pos_probs = torch.sigmoid(pos_logits_val).cpu().numpy()
            
            # B. Negative Samples (Generate Fake Edges)
            # Sample same number of negatives as we have test positives
            num_neg = test_mask.sum().item()
            neg_edge_index = negative_sampling(
                edge_index=train_graph.edge_index, 
                num_nodes=train_graph.num_nodes,
                num_neg_samples=num_neg,
                method='sparse'
            ).to(device)
            
            # Create dummy attributes for negatives [Conf=0, Aff=0, Mask=0]
            neg_attr = torch.zeros((num_neg, 3), device=device)
            
            # Run Model on Negatives
            neg_logits, _, _ = model(train_graph.x, neg_edge_index, neg_attr)
            neg_probs = torch.sigmoid(neg_logits).cpu().numpy()
            
            # C. Combine for Classification Metrics
            y_true = np.concatenate([np.ones(num_neg), np.zeros(num_neg)])
            y_scores = np.concatenate([pos_probs, neg_probs])
            
            try:
                roc_auc = roc_auc_score(y_true, y_scores)
                pr_auc = average_precision_score(y_true, y_scores)
                f1 = f1_score(y_true, y_scores > 0.5)
            except:
                roc_auc, pr_auc, f1 = 0.5, 0.0, 0.0

            # D. Affinity MSE (Only on measured POSITIVE edges)
            measured_in_test = val_aff_mask[test_mask].bool()
            if measured_in_test.sum() > 0:
                mse = ((pos_aff[test_mask][measured_in_test] - val_aff_true[test_mask][measured_in_test])**2).mean().item()
            else:
                mse = 0.0
                
        # Log
        history.append({
            'Epoch': epoch, 'Mode': mode_name,
            'Loss': loss.item(), 'Affinity_MSE': mse,
            'ROC_AUC': roc_auc, 'PR_AUC': pr_auc, 'F1_Score': f1
        })
        
        if epoch == 1 or epoch % 10 == 0:
            print(f"  Ep {epoch:03d} | MSE: {mse:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    # Save
    torch.save(model.state_dict(), config.out_dir / f"model_{mode_name}.pt")
    return history

# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print(f"Loading Graph from {config.graph_path}...")
    try:
        graph_orig = torch.load(config.graph_path, weights_only=False).to(device)
        graph_orig.x = graph_orig.x.float()
        graph_orig.edge_attr = graph_orig.edge_attr.float()
        print(f"Nodes: {graph_orig.num_nodes}, Edges: {graph_orig.num_edges}")
    except Exception as e: sys.exit(f"✗ Error: {e}")

    # 1. Baseline (Unweighted)
    print("\n" + "="*40 + "\n  PHASE A: BASELINE (UNWEIGHTED)\n" + "="*40)
    graph_base = copy.deepcopy(graph_orig)
    graph_base.edge_attr[:, 1] = 0.0 
    graph_base.edge_attr[:, 2] = 0.0 
    hist_base = train_model(graph_base, graph_orig, "Baseline", lambda_affinity=0.0)

    # 2. AffinityCCI (Weighted)
    print("\n" + "="*40 + "\n  PHASE B: AFFINITYCCI (WEIGHTED)\n" + "="*40)
    hist_weight = train_model(graph_orig, graph_orig, "AffinityCCI", lambda_affinity=0.5)

    # Save Results
    df = pd.DataFrame(hist_base + hist_weight)
    out_csv = config.out_dir / "metrics_v2.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nExperiment Complete. Saved to {out_csv}")

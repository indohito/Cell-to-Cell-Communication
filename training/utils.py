#!/usr/bin/env python3
"""
Reusable training utilities extracted from train.py.

This module provides shared functions for data loading, preparation,
training, and evaluation that can be used by multiple training scripts.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                            precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler


def get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_graph_and_edges(graph_path='results/graph.pt', edge_list_path='results/edge_list.csv'):
    """
    Load pre-computed graph and edge list.
    
    Args:
        graph_path: path to graph.pt file
        edge_list_path: path to edge_list.csv file
    
    Returns:
        tuple: (graph, edge_list DataFrame)
    """
    from torch_geometric.data import Data
    try:
        torch.serialization.add_safe_globals([Data])
    except AttributeError:
        # Older PyTorch versions don't have add_safe_globals
        pass
    
    # Load graph with compatibility for older PyTorch versions
    try:
        graph = torch.load(graph_path, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't support weights_only
        graph = torch.load(graph_path)
    
    edge_list = pd.read_csv(edge_list_path)
    return graph, edge_list


def normalize_weights(weights):
    """
    Normalize edge weights to [0.5, 1.5] range to avoid extreme scaling.
    
    Args:
        weights: array-like of edge weights
    
    Returns:
        normalized weights as numpy array
    """
    if len(weights) == 0:
        return weights
    
    weights = np.asarray(weights, dtype=np.float32)
    
    # Handle edge cases
    if np.max(weights) == np.min(weights):
        return np.ones_like(weights, dtype=np.float32)
    
    # Min-max normalization to [0.5, 1.5]
    w_min, w_max = np.min(weights), np.max(weights)
    normalized = (weights - w_min) / (w_max - w_min)
    normalized = 0.5 + normalized  # Scale to [0.5, 1.5]
    
    return normalized.astype(np.float32)


def build_data_splits(graph, edge_list, random_seed=42):
    """
    Build train/val/test splits from graph and edge list.
    
    This function prepares data in the same format as prepare_data_improved
    from train.py, ensuring consistent splits across all training scripts.
    
    Args:
        graph: PyTorch Geometric Data object
        edge_list: DataFrame with edge information
        random_seed: random seed for reproducibility
    
    Returns:
        dict with keys:
            - 'graph': the graph object
            - 'x_scaled': scaled node features tensor
            - 's_idx': source node indices (numpy array)
            - 't_idx': target node indices (numpy array)
            - 'weights': normalized edge weights (numpy array)
            - 'train_idx': training edge indices
            - 'val_idx': validation edge indices
            - 'test_idx': test edge indices
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    print("Preparing data (FAST mode - mini-batching)...")
    
    s_idx, t_idx = graph.edge_index.numpy()
    n_pos = len(s_idx)
    
    # Get weights from graph edge_attr
    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
        all_weights_pos = graph.edge_attr.squeeze().numpy().astype(np.float32)
    else:
        all_weights_pos = np.ones(n_pos, dtype=np.float32)
    
    assert len(all_weights_pos) == n_pos, f"Weight size {len(all_weights_pos)} != {n_pos}"
    all_weights_pos = normalize_weights(all_weights_pos)
    
    print(f"  Using {n_pos} positive edges")
    print(f"  Will sample negatives during training (no pre-generation)")
    
    # Split indices for train/val/test (70/15/15)
    perm = np.random.permutation(n_pos)
    n_tr = int(0.7 * n_pos)
    n_va = int(0.15 * n_pos)
    
    train_idx = perm[:n_tr]
    val_idx = perm[n_tr:n_tr + n_va]
    test_idx = perm[n_tr + n_va:]
    
    print(f"  Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    
    # Scale node features ONLY ONCE
    print("  Scaling node features...")
    x_scaled = graph.x.numpy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_scaled)
    x_scaled = torch.tensor(x_scaled, dtype=torch.float32)
    
    print(f"Data preparation complete (no heavy copying)!")
    
    return {
        'graph': graph,
        'x_scaled': x_scaled,
        's_idx': s_idx,
        't_idx': t_idx,
        'weights': all_weights_pos,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
    }


def train_one_epoch(model, data, optimizer, criterion, device, use_weights=False, batch_size=512):
    """
    Train model for one epoch.
    
    Args:
        model: the model to train
        data: data dict from build_data_splits
        optimizer: optimizer
        criterion: loss function
        device: torch device
        use_weights: whether to use affinity weights (default False)
        batch_size: batch size for training (default 512)
    
    Returns:
        average training loss for the epoch
    """
    model.train()
    x = data['x_scaled'].to(device)
    n_nodes = x.shape[0]
    
    s_idx = torch.tensor(data['s_idx'], dtype=torch.long, device=device)
    t_idx = torch.tensor(data['t_idx'], dtype=torch.long, device=device)
    weights = torch.tensor(data['weights'], dtype=torch.float32, device=device)
    train_idx = data['train_idx']
    
    # Create set of positive edges for negative sampling
    pos_edge_set = set()
    for si, ti in zip(s_idx.cpu().numpy(), t_idx.cpu().numpy()):
        pos_edge_set.add((int(si), int(ti)))
    
    loss_tot = 0
    count = 0
    
    # Train on positive edges with negative sampling (mini-batches)
    for i in range(0, len(train_idx), batch_size):
        batch_idx = train_idx[i:min(i+batch_size, len(train_idx))]
        
        # Positive edges
        s_pos = s_idx[batch_idx]
        t_pos = t_idx[batch_idx]
        w_pos = weights[batch_idx] if use_weights else torch.ones_like(weights[batch_idx])
        
        # Negative samples (same size as positive batch)
        n_neg = len(batch_idx)
        s_neg = torch.randint(0, n_nodes, (n_neg,), device=device)
        t_neg = torch.randint(0, n_nodes, (n_neg,), device=device)
        
        # Filter out accidental positive edges from negatives
        neg_edges = []
        for j in range(n_neg):
            s_n, t_n = s_neg[j].item(), t_neg[j].item()
            attempts = 0
            while (s_n, t_n) in pos_edge_set and attempts < 10:
                s_n = np.random.randint(0, n_nodes)
                t_n = np.random.randint(0, n_nodes)
                attempts += 1
            neg_edges.append((s_n, t_n))
        
        if neg_edges:
            s_neg = torch.tensor([e[0] for e in neg_edges], dtype=torch.long, device=device)
            t_neg = torch.tensor([e[1] for e in neg_edges], dtype=torch.long, device=device)
        
        # Compute embeddings once per batch
        # For models that match train.py interface, use encoder method
        if hasattr(model, 'encoder'):
            h_all = model.encoder(x)
        else:
            # Fallback: try forward pass
            h_all = model(x, edge_index=None)
        
        # Positive edge predictions
        h_s_pos = h_all[s_pos]
        h_t_pos = h_all[t_pos]
        
        # Use predict_edges if available, otherwise use dot product
        if hasattr(model, 'predict_edges'):
            # Build edge_index for positive edges
            pos_edge_index = torch.stack([s_pos, t_pos], dim=0)
            p_pos = model.predict_edges(h_all, pos_edge_index, w_pos if use_weights else None)
            p_pos = p_pos.squeeze()
        else:
            # Fallback: dot product (matching train.py style)
            p_pos = torch.sigmoid(torch.sum(h_s_pos * h_t_pos, dim=1) * w_pos)
        
        # Negative edge predictions
        h_s_neg = h_all[s_neg]
        h_t_neg = h_all[t_neg]
        
        if hasattr(model, 'predict_edges'):
            neg_edge_index = torch.stack([s_neg, t_neg], dim=0)
            p_neg = model.predict_edges(h_all, neg_edge_index, None)
            p_neg = p_neg.squeeze()
        else:
            p_neg = torch.sigmoid(torch.sum(h_s_neg * h_t_neg, dim=1))
        
        # Combined loss
        if use_weights:
            # Weight positive loss by affinity strength
            loss_pos = criterion(p_pos, torch.ones_like(p_pos))
            w_normalized = 0.5 + 1.5 * (w_pos - w_pos.min()) / (w_pos.max() - w_pos.min() + 1e-8)
            loss_pos = (loss_pos * w_normalized).mean()
            loss_neg = criterion(p_neg, torch.zeros_like(p_neg)).mean()
            loss = (loss_pos + loss_neg) / 2
        else:
            loss_pos = criterion(p_pos, torch.ones_like(p_pos))
            loss_neg = criterion(p_neg, torch.zeros_like(p_neg))
            loss = (loss_pos + loss_neg) / 2
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_tot += loss.item() * len(batch_idx)
        count += len(batch_idx)
    
    return loss_tot / count if count > 0 else 0.0


def evaluate_model(model, data, split='val', device=None, use_weights=False):
    """
    Evaluate model on a given split and compute metrics.
    
    Args:
        model: the model to evaluate
        data: data dict from build_data_splits
        split: 'train', 'val', or 'test'
        device: torch device (if None, will use get_device())
        use_weights: whether to use affinity weights (default False)
    
    Returns:
        dict with keys: 'roc_auc', 'pr_auc', 'precision', 'recall', 'f1'
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
    
    # Compute metrics
    roc_auc = roc_auc_score(labels, preds)
    
    # Precision-recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall_vals, precision_vals)
    
    # F1 score (at optimal threshold)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    f1 = np.max(f1_scores)
    
    # Best threshold metrics
    best_threshold = np.mean(preds[labels == 1]) if np.any(labels == 1) else 0.5
    preds_binary = (preds > best_threshold).astype(int)
    precision_best = precision_score(labels, preds_binary, zero_division=0)
    recall_best = recall_score(labels, preds_binary, zero_division=0)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': f1,
        'precision': precision_best,
        'recall': recall_best,
    }


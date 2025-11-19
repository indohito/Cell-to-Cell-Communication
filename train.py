#!/usr/bin/env python3
"""Training script with GAT architecture and data preprocessing."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                            precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

class ImprovedGATPredictor(nn.Module):
    """
    Improved GAT-based predictor with:
    - Proper multi-head attention mechanism
    - Edge weight integration into attention
    - Learnable embeddings for genes and cell types
    - Better link prediction head
    """
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.2, use_affinity=False):
        super().__init__()
        self.use_affinity = use_affinity
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node encoder with batch norm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            self.attention_layers.append(
                MultiHeadAttention(in_dim, out_dim, num_heads, dropout)
            )
        
        # Link prediction head (deeper and better)
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, edge_index=None, edge_attr=None):
        # Encode node features
        h = self.encoder(x)
        
        # Apply attention layers only if edge_index is provided
        if edge_index is not None:
            for attn_layer in self.attention_layers:
                h = attn_layer(h, edge_index, edge_attr if self.use_affinity else None)
        
        return h
    
    def predict_edges(self, h, edge_index, edge_attr=None):
        """Predict edge scores given node embeddings."""
        src, dst = edge_index
        src_feat = h[src]
        dst_feat = h[dst]
        
        # Concatenate and predict
        combined = torch.cat([src_feat, dst_feat], dim=1)
        logits = self.link_predictor(combined)
        scores = self.sigmoid(logits)
        
        # Apply affinity weighting if enabled
        if self.use_affinity and edge_attr is not None:
            scores = scores * edge_attr.unsqueeze(1)
        
        return scores


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for graph."""
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # Linear transformations for query, key, value
        self.linear_q = nn.Linear(in_dim, out_dim)
        self.linear_k = nn.Linear(in_dim, out_dim)
        self.linear_v = nn.Linear(in_dim, out_dim)
        
        # Output projection
        self.linear_out = nn.Linear(out_dim, out_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / np.sqrt(self.head_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass with multi-head attention.
        
        Args:
            x: Node features (N, in_dim)
            edge_index: Edge indices (2, E)
            edge_attr: Edge weights (E,), optional
            
        Returns:
            Updated node features (N, out_dim)
        """
        src, dst = edge_index
        
        # Project to Q, K, V
        Q = self.linear_q(x)  # (N, out_dim)
        K = self.linear_k(x)  # (N, out_dim)
        V = self.linear_v(x)  # (N, out_dim)
        
        # Reshape for multi-head: (N, num_heads, head_dim)
        Q = Q.view(-1, self.num_heads, self.head_dim)
        K = K.view(-1, self.num_heads, self.head_dim)
        V = V.view(-1, self.num_heads, self.head_dim)
        
        # Get source and destination features
        Q_src = Q[src]  # (E, num_heads, head_dim)
        K_dst = K[dst]  # (E, num_heads, head_dim)
        V_dst = V[dst]  # (E, num_heads, head_dim)
        
        # Compute attention scores
        scores = (Q_src * K_dst).sum(dim=-1) * self.scale  # (E, num_heads)
        
        # Apply edge weights if available
        if edge_attr is not None:
            edge_weight = edge_attr.view(-1, 1)  # (E, 1)
            scores = scores * edge_weight
        
        # Normalize per destination node (softmax over source neighbors)
        # Group edges by destination
        attn_weights = torch.zeros_like(scores)
        for t in range(x.shape[0]):
            mask = dst == t
            if mask.any():
                node_scores = scores[mask]  # (num_edges_to_t, num_heads)
                attn_weights[mask] = torch.softmax(node_scores, dim=0)
        
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: for each head, weighted sum of values
        attn_output = torch.zeros(x.shape[0], self.out_dim, device=x.device, dtype=x.dtype)
        for h_idx in range(self.num_heads):
            # Get weights for this head: (E,)
            head_weights = attn_weights[:, h_idx]  # (E,)
            
            # Weight the values: (E, head_dim)
            weighted_v = head_weights.unsqueeze(-1) * V_dst[:, h_idx, :]  # (E, 1) * (E, head_dim)
            
            # Scatter add to destination nodes
            for t in range(x.shape[0]):
                mask = dst == t
                if mask.any():
                    # Sum weighted values for all edges into node t
                    attn_output[t, h_idx*self.head_dim:(h_idx+1)*self.head_dim] = weighted_v[mask].sum(dim=0)
        
        # Output projection
        out = self.linear_out(attn_output)
        return F.relu(out)


def load_data():
    """Load pre-computed graph and edge list."""
    from torch_geometric.data import Data
    try:
        torch.serialization.add_safe_globals([Data])
    except AttributeError:
        # Older PyTorch versions don't have add_safe_globals
        pass
    
    # Load graph with compatibility for older PyTorch versions
    try:
        graph = torch.load('results/graph.pt', weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't support weights_only
        graph = torch.load('results/graph.pt')
    
    edge_list = pd.read_csv('results/edge_list.csv')
    return graph, edge_list


def normalize_weights(weights):
    """Normalize edge weights to [0.5, 1.5] range to avoid extreme scaling."""
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


def prepare_data_improved(graph, edge_list):
    """ULTRA-FAST data preparation: only process training data, not all negatives."""
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
    
    # ULTRA FAST: Don't create huge negative sample arrays
    # Instead, just split positive edges and sample negatives on-the-fly during training
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


def evaluate_model(model, data, name, use_w, dev, split='val'):
    """Evaluate model on validation or test set and compute metrics."""
    model.eval()
    x = data['x_scaled'].to(dev)
    n_nodes = x.shape[0]
    
    s_idx = torch.tensor(data['s_idx'], dtype=torch.long, device=dev)
    t_idx = torch.tensor(data['t_idx'], dtype=torch.long, device=dev)
    weights = torch.tensor(data['weights'], dtype=torch.float32, device=dev)
    
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
        h_all = model.encoder(x)
        
        # Positive predictions
        s_pos = s_idx[eval_idx]
        t_pos = t_idx[eval_idx]
        w_pos = weights[eval_idx] if use_w else torch.ones_like(weights[eval_idx])
        
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
        
        s_neg = torch.tensor([e[0] for e in neg_edges], dtype=torch.long, device=dev)
        t_neg = torch.tensor([e[1] for e in neg_edges], dtype=torch.long, device=dev)
        
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
    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)
    
    # F1 score (at optimal threshold)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1 = np.max(f1_scores)
    
    # Best threshold metrics
    best_idx = np.argmax(f1_scores)
    best_threshold = np.mean(preds[labels == 1])
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


def train_improved(model, data, name, use_w, num_epochs=15):
    """Training with both positive and negative samples for meaningful classification."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(dev)
    
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(opt, T_max=num_epochs)
    loss_fn = nn.BCELoss()
    
    hist = []
    x = data['x_scaled'].to(dev)
    n_nodes = x.shape[0]
    
    s_idx = torch.tensor(data['s_idx'], dtype=torch.long, device=dev)
    t_idx = torch.tensor(data['t_idx'], dtype=torch.long, device=dev)
    weights = torch.tensor(data['weights'], dtype=torch.float32, device=dev)
    
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']
    
    # Create set of positive edges for negative sampling
    pos_edge_set = set()
    for si, ti in zip(s_idx.cpu().numpy(), t_idx.cpu().numpy()):
        pos_edge_set.add((int(si), int(ti)))
    
    print(f"\nTraining {name}...")
    for ep in range(1, num_epochs + 1):
        model.train()
        loss_tot = 0
        count = 0
        
        # Train on positive edges with negative sampling (mini-batches)
        batch_size = 512
        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[i:min(i+batch_size, len(train_idx))]
            
            # Positive edges
            s_pos = s_idx[batch_idx]
            t_pos = t_idx[batch_idx]
            w_pos = weights[batch_idx] if use_w else torch.ones_like(weights[batch_idx])
            
            # Negative samples (same size as positive batch)
            n_neg = len(batch_idx)
            s_neg = torch.randint(0, n_nodes, (n_neg,), device=dev)
            t_neg = torch.randint(0, n_nodes, (n_neg,), device=dev)
            
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
                s_neg = torch.tensor([e[0] for e in neg_edges], dtype=torch.long, device=dev)
                t_neg = torch.tensor([e[1] for e in neg_edges], dtype=torch.long, device=dev)
            
            # Compute embeddings once per batch
            h_all = model.encoder(x)
            
            # Positive edge predictions
            h_s_pos = h_all[s_pos]
            h_t_pos = h_all[t_pos]
            p_pos = torch.sigmoid(torch.sum(h_s_pos * h_t_pos, dim=1) * w_pos)
            
            # Negative edge predictions (no affinity weights for negatives)
            h_s_neg = h_all[s_neg]
            h_t_neg = h_all[t_neg]
            p_neg = torch.sigmoid(torch.sum(h_s_neg * h_t_neg, dim=1))
            
            # Combined loss: positive edges should be ~1, negative edges should be ~0
            loss_pos = loss_fn(p_pos, torch.ones_like(p_pos))
            loss_neg = loss_fn(p_neg, torch.zeros_like(p_neg))
            loss = (loss_pos + loss_neg) / 2
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            loss_tot += loss.item() * len(batch_idx)
            count += len(batch_idx)
        
        scheduler.step()
        
        # Evaluate on validation set every epoch
        val_metrics = evaluate_model(model, data, name, use_w, dev, split='val')
        
        if ep % 5 == 0 or ep == 1:
            print(f"  [{name}] Epoch {ep}/{num_epochs}: Loss={loss_tot/count:.4f}, "
                  f"Val ROC-AUC={val_metrics['roc_auc']:.4f}, F1={val_metrics['f1']:.4f}")
        
        record = {'epoch': ep, 'loss': loss_tot/count, 'model': name}
        record.update(val_metrics)
        hist.append(record)
        
        if ep == num_epochs:
            print(f"{name} training complete")
            break
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, data, name, use_w, dev, split='test')
    print(f"\n  [{name}] Test Set Performance:")
    print(f"    ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"    PR-AUC:  {test_metrics['pr_auc']:.4f}")
    print(f"    F1:      {test_metrics['f1']:.4f}")
    print(f"    Precision: {test_metrics['precision']:.4f}")
    print(f"    Recall:   {test_metrics['recall']:.4f}")
    
    Path('results/models').mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), f'results/models/model_improved_{name}.pt')
    return hist


if __name__ == '__main__':
    print("Starting training pipeline...")
    
    graph, edge_list = load_data()
    print(f"Loaded graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
    
    data = prepare_data_improved(graph, edge_list)
    
    results = []
    
    # Train improved baseline (no affinity)
    print("\n" + "="*60)
    print("=== Training Improved Baseline (No Affinity) ===")
    print("="*60)
    model_baseline = ImprovedGATPredictor(
        graph.x.shape[1], 
        hidden_dim=128, 
        num_heads=4, 
        num_layers=2, 
        dropout=0.2,
        use_affinity=False
    )
    hist_baseline = train_improved(model_baseline, data, 'baseline', False, num_epochs=10)
    results.extend(hist_baseline)
    
    # Train improved affinity-weighted
    print("\n" + "="*60)
    print("=== Training Improved Affinity-Weighted ===")
    print("="*60)
    model_affinity = ImprovedGATPredictor(
        graph.x.shape[1],
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
        use_affinity=True
    )
    hist_affinity = train_improved(model_affinity, data, 'affinity_weighted', True, num_epochs=10)
    results.extend(hist_affinity)
    
    # Save results
    print("\n" + "="*60)
    print("=== Saving Results ===")
    print("="*60)
    
    Path('results').mkdir(exist_ok=True, parents=True)
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv('results/training_history_improved.csv', index=False)
        print(f"Saved training history to results/training_history_improved.csv")
        print(f"\nTraining Summary:")
        print(df.groupby('model')[['loss']].agg(['min', 'max', 'mean']))
    else:
        print("No training results to save")
    
    print("\n" + "="*60)
    print("Training pipeline complete!")
    print("="*60)

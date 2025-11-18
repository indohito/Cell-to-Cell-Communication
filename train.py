#!/usr/bin/env python3
"""Training script for GNN-based CCC prediction."""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                            precision_score, recall_score, f1_score)

torch.manual_seed(42)
np.random.seed(42)

class GATPredictor(nn.Module):
    def __init__(self, input_dim, use_affinity=False):
        super().__init__()
        self.use_affinity = use_affinity
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.sig = nn.Sigmoid()
    
    def forward(self, x, edge_index, weight=None):
        h = self.mlp(x)
        s, t = edge_index
        score = (h[s] * h[t]).sum(1, keepdim=True)
        if self.use_affinity and weight is not None:
            score = score * weight.unsqueeze(1)
        return self.sig(score)

def load_data():
    from torch_geometric.data import Data
    torch.serialization.add_safe_globals([Data])
    graph = torch.load('results/graph.pt', weights_only=False)
    edge_list = pd.read_csv('results/edge_list.csv')
    return graph, edge_list

def prepare_data(graph, edge_list):
    """Prepare training data with positive and negative edges and affinity weights."""
    pos_edges = set()
    edge_weights_dict = {}
    for _, row in edge_list.iterrows():
        key = (row['source_cell_type'], row['ligand'], row['target_cell_type'], row['receptor'])
        pos_edges.add(key)
        weight = row.get('weight', 1.0) if 'weight' in row else 1.0
        edge_weights_dict[key] = weight
    
    s_idx, t_idx = graph.edge_index.numpy()
    n_pos = len(s_idx)
    
    n_neg = n_pos
    neg_s, neg_t = [], []
    while len(neg_s) < n_neg:
        s = np.random.randint(0, graph.num_nodes)
        t = np.random.randint(0, graph.num_nodes)
        if s != t and (s, t) not in zip(s_idx, t_idx):
            neg_s.append(s)
            neg_t.append(t)
    
    all_s = np.concatenate([s_idx, neg_s])
    all_t = np.concatenate([t_idx, neg_t])
    all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)], dtype=np.float32)
    
    all_weights = np.ones(len(all_s), dtype=np.float32)
    for i in range(len(all_s)):
        if i < n_pos:
            key_tuple = tuple(edge_list.iloc[i][['source_cell_type', 'ligand', 'target_cell_type', 'receptor']])
            all_weights[i] = edge_weights_dict.get(key_tuple, 1.0)
    
    perm = np.random.permutation(len(all_labels))
    all_s = all_s[perm]
    all_t = all_t[perm]
    all_labels = all_labels[perm]
    all_weights = all_weights[perm]
    
    n_tr = int(0.7 * len(perm))
    n_va = int(0.15 * len(perm))
    
    return {
        'graph': graph,
        'edge_index': torch.tensor([all_s, all_t], dtype=torch.long),
        'labels': torch.tensor(all_labels, dtype=torch.float32),
        'weights': torch.tensor(all_weights, dtype=torch.float32),
        'train': np.arange(n_tr),
        'val': np.arange(n_tr, n_tr + n_va),
        'test': np.arange(n_tr + n_va, len(all_labels)),
    }

def train(model, data, name, use_w):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(dev)
    opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.BCELoss()
    
    hist = []
    x = data['graph'].x.to(dev)
    ei = data['edge_index'].to(dev)
    labels_all = data['labels'].to(dev)
    weights_all = data['weights'].to(dev)
    
    for ep in range(1, 16):
        model.train()
        loss_tot = 0
        for i in range(0, len(data['train']), 256):
            idx = data['train'][i:i+256]
            ei_batch = ei[:, idx]
            l_batch = labels_all[idx]
            w_batch = weights_all[idx] if use_w else None
            p = model(x, ei_batch, w_batch)
            loss = loss_fn(p.squeeze(), l_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_tot += loss.item() * len(idx)
        
        model.eval()
        with torch.no_grad():
            test_idx = data['test']
            ei_test = ei[:, test_idx]
            l_test = labels_all[test_idx]
            w_test = weights_all[test_idx] if use_w else None
            p = model(x, ei_test, w_test).squeeze().cpu().numpy()
            l = l_test.cpu().numpy()
            
            if len(np.unique(l)) > 1:  # Both classes present
                roc = roc_auc_score(l, p)
                pr, re, _ = precision_recall_curve(l, p)
                pr_a = auc(re, pr)
                prec = precision_score(l, p > 0.5)
                rec = recall_score(l, p > 0.5)
                f = f1_score(l, p > 0.5)
            else:
                roc = pr_a = prec = rec = f = 0.0
            
            hist.append({'epoch': ep, 'loss': loss_tot/len(data['train']), 'roc': roc, 'pr_auc': pr_a, 'prec': prec, 'rec': rec, 'f1': f, 'model': name})
            
            if ep % 5 == 0 or ep == 1:
                print(f"{name} | Epoch {ep:2d}: loss={hist[-1]['loss']:.4f} roc={roc:.4f} f1={f:.4f}")
    
    Path('results/models').mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), f'results/models/model_{name}.pt')
    return hist

if __name__ == '__main__':
    graph, edge_list = load_data()
    data = prepare_data(graph, edge_list)

    h_all = []
    h_all.extend(train(GATPredictor(graph.x.shape[1], use_affinity=False), data, 'baseline', False))
    h_all.extend(train(GATPredictor(graph.x.shape[1], use_affinity=True), data, 'affinity_weighted', True))

    df = pd.DataFrame(h_all)
    df.to_csv('results/training_history.csv', index=False)
    df.groupby('model').tail(1)[['model', 'roc', 'pr_auc', 'prec', 'rec', 'f1']].to_csv('results/training_metrics.csv', index=False)

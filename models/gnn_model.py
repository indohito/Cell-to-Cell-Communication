#!/usr/bin/env python3
"""
Graph Attention Network (GAT) for link prediction in cell-to-cell communication.

Model architecture:
  - Node encoder: embeds node features
  - GAT layers: learns node representations via multi-head attention
  - Link predictor: predicts edge weight/presence between node pairs
  - Supports both affinity-weighted and unweighted (baseline) variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool
from torch_geometric.data import Data

class NodeEncoder(nn.Module):
    """Encodes node features to embedding dimension."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        return F.relu(self.fc(x))

class GATLinkPredictor(nn.Module):
    """
    GAT-based link predictor for CCC prediction.
    
    Predicts edge weights or presence given node pairs and graph structure.
    """
    def __init__(self, 
                 input_dim,
                 hidden_dim=64,
                 num_layers=2,
                 num_heads=4,
                 dropout=0.1,
                 use_affinity=True):
        """
        Args:
            input_dim: dimension of input node features
            hidden_dim: dimension of hidden layers
            num_layers: number of GAT layers
            num_heads: number of attention heads
            dropout: dropout rate
            use_affinity: whether to use affinity weights (True) or uniform (False)
        """
        super().__init__()
        self.use_affinity = use_affinity
        self.hidden_dim = hidden_dim
        
        # Node encoder
        self.encoder = NodeEncoder(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=num_heads, 
                                       dropout=dropout, add_self_loops=False))
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                           heads=num_heads, dropout=dropout, add_self_loops=False))
        
        # Link prediction head
        final_dim = hidden_dim * num_heads
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * final_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None, edge_src=None, edge_dst=None):
        """
        Forward pass.
        
        Args:
            x: node features (num_nodes, input_dim)
            edge_index: edge indices (2, num_edges)
            edge_attr: edge weights (num_edges, 1) - optional
            edge_src: source node indices for link prediction (num_eval_edges,)
            edge_dst: destination node indices for link prediction (num_eval_edges,)
        
        Returns:
            edge_scores: predicted edge weights (num_eval_edges, 1)
        """
        # Encode nodes
        x = self.encoder(x)
        
        # Apply GAT layers
        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = self.dropout(x)
        
        # Link prediction: concatenate node embeddings
        if edge_src is not None and edge_dst is not None:
            src_feat = x[edge_src]
            dst_feat = x[edge_dst]
            scores = self.link_predictor(torch.cat([src_feat, dst_feat], dim=1))
            return scores
        else:
            # If no edge indices provided, return all node embeddings
            return x

class BaselineGNNLinkPredictor(nn.Module):
    """
    Baseline GNN without using edge weights (affinity).
    Same architecture as GATLinkPredictor but trained without edge_attr.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.encoder = NodeEncoder(input_dim, hidden_dim)
        
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=num_heads, 
                                       dropout=dropout, add_self_loops=False))
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                           heads=num_heads, dropout=dropout, add_self_loops=False))
        
        final_dim = hidden_dim * num_heads
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * final_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None, edge_src=None, edge_dst=None):
        """Forward pass (no edge_attr used, even if provided)."""
        x = self.encoder(x)
        
        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = self.dropout(x)
        
        if edge_src is not None and edge_dst is not None:
            src_feat = x[edge_src]
            dst_feat = x[edge_dst]
            scores = self.link_predictor(torch.cat([src_feat, dst_feat], dim=1))
            return scores
        else:
            return x

def create_model(graph_input_dim, hidden_dim=64, num_layers=2, num_heads=4, 
                 dropout=0.1, use_affinity=True, device='cpu'):
    """
    Factory function to create GNN model.
    
    Args:
        graph_input_dim: input dimension from graph
        hidden_dim: hidden dimension
        num_layers: number of layers
        num_heads: number of attention heads
        dropout: dropout rate
        use_affinity: if True, use GATLinkPredictor; if False, use Baseline
        device: torch device
    
    Returns:
        model on specified device
    """
    if use_affinity:
        model = GATLinkPredictor(graph_input_dim, hidden_dim, num_layers, 
                                 num_heads, dropout, use_affinity=True)
    else:
        model = BaselineGNNLinkPredictor(graph_input_dim, hidden_dim, num_layers, 
                                        num_heads, dropout)
    
    return model.to(device)

if __name__ == "__main__":
    # Test instantiation
    print("Testing GNN models...")
    model_affinity = create_model(graph_input_dim=34, use_affinity=True)
    model_baseline = create_model(graph_input_dim=34, use_affinity=False)
    
    print(f"Affinity model: {sum(p.numel() for p in model_affinity.parameters())} parameters")
    print(f"Baseline model: {sum(p.numel() for p in model_baseline.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(100, 34)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 1)
    
    out = model_affinity(x, edge_index, edge_attr, torch.arange(100), torch.arange(100))
    print(f"Affinity output shape: {out.shape}")
    
    out_baseline = model_baseline(x, edge_index, edge_src=torch.arange(100), edge_dst=torch.arange(100))
    print(f"Baseline output shape: {out_baseline.shape}")
    
    print("\n✓ Model instantiation successful!")

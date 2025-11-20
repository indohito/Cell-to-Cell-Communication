#!/usr/bin/env python3
"""
Graph baseline models for link prediction: GCN and GraphSAGE.

These models provide alternative GNN architectures to compare against
the existing GAT-based models (baseline and affinity-weighted).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GCNLinkPredictor(nn.Module):
    """
    GCN-based link predictor for cell-to-cell communication.
    
    Uses Graph Convolutional Network layers for message passing,
    followed by a link prediction head that scores edge pairs.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        """
        Args:
            input_dim: dimension of input node features
            hidden_dim: dimension of hidden layers (default 64)
            num_layers: number of GCN layers (default 2)
            dropout: dropout rate (default 0.2)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
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
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.gcn_layers.append(GCNConv(in_dim, hidden_dim))
        
        # Link prediction head
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
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x, edge_index):
        """
        Encode node features using GCN layers.
        
        Args:
            x: node features (num_nodes, input_dim)
            edge_index: edge indices (2, num_edges)
        
        Returns:
            node embeddings (num_nodes, hidden_dim)
        """
        # Initial encoding
        h = self.encoder(x)
        
        # Apply GCN layers
        for i, gcn_layer in enumerate(self.gcn_layers):
            h = gcn_layer(h, edge_index)
            h = F.relu(h)
            if i < len(self.gcn_layers) - 1:  # No dropout after last layer
                h = self.dropout(h)
        
        return h
    
    def decode(self, h, edge_index_to_score):
        """
        Decode edge scores from node embeddings.
        
        Args:
            h: node embeddings (num_nodes, hidden_dim)
            edge_index_to_score: edges to score (2, num_edges_to_score)
        
        Returns:
            edge scores (num_edges_to_score, 1)
        """
        src, dst = edge_index_to_score
        src_feat = h[src]
        dst_feat = h[dst]
        
        # Concatenate embeddings
        combined = torch.cat([src_feat, dst_feat], dim=1)
        
        # Predict through MLP
        logits = self.link_predictor(combined)
        scores = self.sigmoid(logits)
        
        return scores
    
    def forward(self, x, edge_index=None, edge_attr=None):
        """
        Forward pass - returns node embeddings.
        
        Args:
            x: node features (num_nodes, input_dim)
            edge_index: edge indices for message passing (2, num_edges)
            edge_attr: edge attributes (ignored for GCN, kept for compatibility)
        
        Returns:
            node embeddings (num_nodes, hidden_dim)
        """
        if edge_index is None:
            # If no edge_index, just return encoded features
            return self.encoder(x)
        
        return self.encode(x, edge_index)
    
    def predict_edges(self, h, edge_index, edge_attr=None):
        """
        Predict edge scores given node embeddings.
        
        This method matches the interface used in train.py for compatibility.
        
        Args:
            h: node embeddings (num_nodes, hidden_dim)
            edge_index: edge indices to score (2, num_edges)
            edge_attr: edge attributes (ignored for GCN, kept for compatibility)
        
        Returns:
            edge scores (num_edges, 1)
        """
        return self.decode(h, edge_index)


class SAGELinkPredictor(nn.Module):
    """
    GraphSAGE-based link predictor for cell-to-cell communication.
    
    Uses GraphSAGE layers for message passing with neighbor sampling,
    followed by a link prediction head.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        """
        Args:
            input_dim: dimension of input node features
            hidden_dim: dimension of hidden layers (default 64)
            num_layers: number of SAGE layers (default 2)
            dropout: dropout rate (default 0.2)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
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
        
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.sage_layers.append(SAGEConv(in_dim, hidden_dim))
        
        # Link prediction head
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
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x, edge_index):
        """
        Encode node features using GraphSAGE layers.
        
        Args:
            x: node features (num_nodes, input_dim)
            edge_index: edge indices (2, num_edges)
        
        Returns:
            node embeddings (num_nodes, hidden_dim)
        """
        # Initial encoding
        h = self.encoder(x)
        
        # Apply SAGE layers
        for i, sage_layer in enumerate(self.sage_layers):
            h = sage_layer(h, edge_index)
            h = F.relu(h)
            if i < len(self.sage_layers) - 1:  # No dropout after last layer
                h = self.dropout(h)
        
        return h
    
    def decode(self, h, edge_index_to_score):
        """
        Decode edge scores from node embeddings.
        
        Args:
            h: node embeddings (num_nodes, hidden_dim)
            edge_index_to_score: edges to score (2, num_edges_to_score)
        
        Returns:
            edge scores (num_edges_to_score, 1)
        """
        src, dst = edge_index_to_score
        src_feat = h[src]
        dst_feat = h[dst]
        
        # Concatenate embeddings
        combined = torch.cat([src_feat, dst_feat], dim=1)
        
        # Predict through MLP
        logits = self.link_predictor(combined)
        scores = self.sigmoid(logits)
        
        return scores
    
    def forward(self, x, edge_index=None, edge_attr=None):
        """
        Forward pass - returns node embeddings.
        
        Args:
            x: node features (num_nodes, input_dim)
            edge_index: edge indices for message passing (2, num_edges)
            edge_attr: edge attributes (ignored for SAGE, kept for compatibility)
        
        Returns:
            node embeddings (num_nodes, hidden_dim)
        """
        if edge_index is None:
            # If no edge_index, just return encoded features
            return self.encoder(x)
        
        return self.encode(x, edge_index)
    
    def predict_edges(self, h, edge_index, edge_attr=None):
        """
        Predict edge scores given node embeddings.
        
        This method matches the interface used in train.py for compatibility.
        
        Args:
            h: node embeddings (num_nodes, hidden_dim)
            edge_index: edge indices to score (2, num_edges)
            edge_attr: edge attributes (ignored for SAGE, kept for compatibility)
        
        Returns:
            edge scores (num_edges, 1)
        """
        return self.decode(h, edge_index)


__all__ = ['GCNLinkPredictor', 'SAGELinkPredictor']


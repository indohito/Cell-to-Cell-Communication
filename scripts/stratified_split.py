#!/usr/bin/env python3
"""Stratified edge splitting to prevent data leakage."""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def load_graph_metadata(integration_table_path='results/integration_table_weighted.csv'):
    """Load metadata for edges (cell types, genes, etc.)."""
    df = pd.read_csv(integration_table_path)
    return df

def build_celltype_mapping(graph_path='results/graph.pt'):
    """Build mapping from node ID to cell type."""
    graph = torch.load(graph_path, weights_only=False)
    
    # Reconstruct node names from graph construction logic
    # This assumes graph construction follows: first num_celltype nodes are cell types, rest are genes
    # We'll load from the integration table to get exact mapping
    
    df = pd.read_csv('results/integration_table_weighted.csv')
    
    # Get unique cell types
    unique_cell_types = sorted(set(df['cell_type_A'].unique()) | set(df['cell_type_B'].unique()))
    celltype_to_id = {ct: i for i, ct in enumerate(unique_cell_types)}
    
    # All other nodes are genes
    unique_genes = sorted(set(df['ligand_gene'].unique()) | set(df['receptor_gene'].unique()))
    gene_to_id = {gene: i + len(celltype_to_id) for i, gene in enumerate(unique_genes)}
    
    # Reverse mapping
    id_to_celltype = {v: k for k, v in celltype_to_id.items()}
    id_to_gene = {v: k for k, v in gene_to_id.items()}
    
    node_id_to_type = {}
    node_id_to_name = {}
    
    for node_id in range(graph.num_nodes):
        if node_id in id_to_celltype:
            node_id_to_type[node_id] = 'celltype'
            node_id_to_name[node_id] = id_to_celltype[node_id]
        elif node_id in id_to_gene:
            node_id_to_type[node_id] = 'gene'
            node_id_to_name[node_id] = id_to_gene[node_id]
        else:
            node_id_to_type[node_id] = 'unknown'
            node_id_to_name[node_id] = f'node_{node_id}'
    
    return node_id_to_type, node_id_to_name

def stratified_edge_split(edge_index, node_id_to_name, num_folds=5, 
                         train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split edges stratified by (source, target) node pairs to avoid leakage.
    
    Args:
        edge_index: (2, num_edges) tensor
        node_id_to_name: {node_id -> node_name} mapping
        num_folds: number of folds for stratification
        train_ratio, val_ratio: proportions for train/val/test
        seed: random seed
    
    Returns:
        train_idx, val_idx, test_idx (torch tensors of edge indices)
    """
    src_nodes = edge_index[0].cpu().numpy()
    dst_nodes = edge_index[1].cpu().numpy()
    
    # Create stratification groups: (source_name, target_name) pairs
    groups = []
    for s, d in zip(src_nodes, dst_nodes):
        src_name = node_id_to_name.get(int(s), f'node_{s}')
        dst_name = node_id_to_name.get(int(d), f'node_{d}')
        group = f"{src_name}_{dst_name}"
        groups.append(group)
    
    groups = np.array(groups)
    unique_groups = np.unique(groups)
    group_to_id = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_id[g] for g in groups])
    
    print(f"Stratifying edges with {len(unique_groups)} unique (source, target) pairs...")
    
    # Use StratifiedGroupKFold for splitting
    splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.arange(len(groups)), group_ids, group_ids))
    
    # Use first fold: train_val vs test
    train_val_indices, test_indices = splits[0]
    
    # Further split train_val into train and val
    n_train_val = len(train_val_indices)
    n_train = int(n_train_val * train_ratio / (train_ratio + val_ratio))
    
    # Need to further stratify within train_val
    train_val_subset_groups = groups[train_val_indices]
    train_val_group_ids = group_ids[train_val_indices]
    
    # Create a second stratification for train/val split
    splitter2 = StratifiedGroupKFold(n_splits=int(1 / (val_ratio / (train_ratio + val_ratio))), 
                                     shuffle=True, random_state=seed + 1)
    splits2 = list(splitter2.split(np.arange(len(train_val_indices)), 
                                    train_val_group_ids, train_val_group_ids))
    
    train_subset_indices, val_subset_indices = splits2[0]
    
    train_indices = train_val_indices[train_subset_indices]
    val_indices = train_val_indices[val_subset_indices]
    
    train_idx = torch.tensor(train_indices, dtype=torch.long)
    val_idx = torch.tensor(val_indices, dtype=torch.long)
    test_idx = torch.tensor(test_indices, dtype=torch.long)
    
    print(f"  Train: {len(train_idx)} edges")
    print(f"  Val:   {len(val_idx)} edges")
    print(f"  Test:  {len(test_idx)} edges")
    
    return train_idx, val_idx, test_idx

if __name__ == '__main__':
    # Test
    graph = torch.load('results/graph.pt', weights_only=False)
    node_id_to_type, node_id_to_name = build_celltype_mapping()
    
    train_idx, val_idx, test_idx = stratified_edge_split(
        graph.edge_index, node_id_to_name, 
        train_ratio=0.7, val_ratio=0.15, seed=42
    )
    
    print(f"\nStratified split complete:")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

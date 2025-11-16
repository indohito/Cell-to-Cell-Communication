#!/usr/bin/env python3
"""
Construct a PyTorch Geometric graph from the integration table.

Graph structure:
- Nodes: cell types + genes (ligand/receptor)
- Edges: (ligand_gene -> cell_type_B) with weight = combined_weight_norm
         optionally: (cell_type_A -> receptor_gene) with the same weight

The graph models cell-to-cell communication as a bipartite interaction where:
  - Sender cell type A expresses ligand L
  - Receiver cell type B expresses receptor R
  - Edge weight incorporates affinity(L-R) × expr(L in A) × expr(R in B)

Output:
  - results/graph.pt: PyG Data object
  - results/graph_stats.txt: Summary statistics
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import json

def build_graph(integration_table_path="results/integration_table_weighted.csv", 
                output_path="results/graph.pt",
                stats_path="results/graph_stats.txt",
                weight_col="weight",
                min_weight=0.0):
    """
    Build PyG graph from integration table.
    
    Args:
        integration_table_path: path to weighted integration table
        output_path: where to save the graph
        stats_path: where to save statistics
        weight_col: column to use as edge weights
        min_weight: minimum weight threshold (0.0 = all edges)
    """
    print("[1/5] Loading integration table...")
    df = pd.read_csv(integration_table_path)
    print(f"  Loaded {len(df)} rows")
    
    # Filter by minimum weight
    if min_weight > 0:
        df = df[df[weight_col] >= min_weight]
        print(f"  Filtered to {len(df)} rows (min_weight={min_weight})")
    
    # Create node mappings
    print("[2/5] Creating node mappings...")
    cell_types = sorted(set(df['cell_type_A']) | set(df['cell_type_B']))
    genes = sorted(set(df['ligand_gene']) | set(df['receptor_gene']))
    
    cell_type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    gene_to_idx = {g: len(cell_types) + i for i, g in enumerate(genes)}
    
    num_nodes = len(cell_types) + len(genes)
    print(f"  Cell types: {len(cell_types)}")
    print(f"  Genes: {len(genes)}")
    print(f"  Total nodes: {num_nodes}")
    
    # Create node features (one-hot encoding by type)
    print("[3/5] Creating node features...")
    x = torch.zeros(num_nodes, 2)  # [is_cell_type, is_gene]
    x[:len(cell_types), 0] = 1  # cell types
    x[len(cell_types):, 1] = 1   # genes
    
    # Optional: add embeddings
    np.random.seed(42)
    embeddings = torch.randn(num_nodes, 32)
    x = torch.cat([x, embeddings], dim=1)  # (num_nodes, 34)
    
    # Create edges and edge attributes
    print("[4/5] Creating edges and edge weights...")
    edge_src = []
    edge_dst = []
    edge_weights = []
    edge_types = []  # 'ligand_to_celltype' or 'celltype_to_receptor'
    
    for _, row in df.iterrows():
        ligand_idx = gene_to_idx[row['ligand_gene']]
        celltype_B_idx = cell_type_to_idx[row['cell_type_B']]
        celltype_A_idx = cell_type_to_idx[row['cell_type_A']]
        receptor_idx = gene_to_idx[row['receptor_gene']]
        weight = row[weight_col]
        
        # Edge 1: ligand_gene -> cell_type_B (receiver expresses receptor)
        edge_src.append(ligand_idx)
        edge_dst.append(celltype_B_idx)
        edge_weights.append(weight)
        edge_types.append(0)  # ligand_to_celltype
        
        # Edge 2: cell_type_A -> receptor_gene (sender expresses ligand)
        edge_src.append(celltype_A_idx)
        edge_dst.append(receptor_idx)
        edge_weights.append(weight)
        edge_types.append(1)  # celltype_to_receptor
    
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
    edge_type = torch.tensor(edge_types, dtype=torch.long).unsqueeze(1)
    
    num_edges = edge_index.shape[1]
    print(f"  Total edges: {num_edges}")
    print(f"  Edge weight range: [{edge_attr.min():.4f}, {edge_attr.max():.4f}]")
    
    # Create PyG Data object
    print("[5/5] Creating PyG Data object...")
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
    )
    
    # Add metadata
    graph.cell_types = cell_types
    graph.genes = genes
    graph.cell_type_to_idx = cell_type_to_idx
    graph.gene_to_idx = gene_to_idx
    graph.num_cell_types = len(cell_types)
    graph.num_genes = len(genes)
    
    # Save graph
    torch.save(graph, output_path)
    print(f"\nGraph saved to {output_path}")
    print(f"  Nodes: {graph.x.shape[0]}")
    print(f"  Node features: {graph.x.shape[1]}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Edge attributes: {graph.edge_attr.shape}")
    
    # Save statistics
    stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_cell_types": len(cell_types),
        "num_genes": len(genes),
        "num_node_features": graph.x.shape[1],
        "edge_weight_min": float(edge_attr.min()),
        "edge_weight_max": float(edge_attr.max()),
        "edge_weight_mean": float(edge_attr.mean()),
        "edge_weight_std": float(edge_attr.std()),
        "cell_types": cell_types,
        "num_interactions": len(df)
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")
    
    return graph

if __name__ == "__main__":
    graph = build_graph()

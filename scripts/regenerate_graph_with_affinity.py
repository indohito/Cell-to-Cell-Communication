#!/usr/bin/env python3
"""
Regenerate graph.pt from edge_list.csv with affinity weights.
Converts edge_list format to integration_table format and rebuilds graph.
"""

import pandas as pd
import torch
from torch_geometric.data import Data
import json
from pathlib import Path

def regenerate_graph_from_edge_list():
    """Regenerate graph.pt from affinity-weighted edge_list.csv"""
    
    results_dir = Path("results")
    
    print("=" * 70)
    print("Regenerating Graph with Affinity Weights")
    print("=" * 70)
    
    # Load edge list
    print("\n[1/4] Loading edge_list.csv with affinity weights...")
    edge_df = pd.read_csv(results_dir / "edge_list.csv")
    print(f"  Loaded {len(edge_df)} edges")
    print(f"  Columns: {list(edge_df.columns)}")
    
    # Rename columns to match integration_table format
    print("\n[2/4] Converting edge list format...")
    df = edge_df.copy()
    df['ligand_gene'] = df['ligand']
    df['receptor_gene'] = df['receptor']
    df['cell_type_A'] = df['source_cell_type']
    df['cell_type_B'] = df['target_cell_type']
    
    # Use weight column for edge weights
    print(f"  Using 'weight' column for edge weights")
    print(f"  Weight range: [{df['weight'].min():.4f}, {df['weight'].max():.4f}]")
    
    # Create node mappings
    print("\n[3/4] Building graph structure...")
    cell_types = sorted(set(df['cell_type_A']) | set(df['cell_type_B']))
    genes = sorted(set(df['ligand_gene']) | set(df['receptor_gene']))
    
    cell_type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    gene_to_idx = {g: len(cell_types) + i for i, g in enumerate(genes)}
    
    num_nodes = len(cell_types) + len(genes)
    print(f"  Cell types: {len(cell_types)}")
    print(f"  Genes: {len(genes)}")
    print(f"  Total nodes: {num_nodes}")
    
    # Create node features
    x = torch.zeros(num_nodes, 2)  # [is_cell_type, is_gene]
    x[:len(cell_types), 0] = 1     # cell types
    x[len(cell_types):, 1] = 1      # genes
    
    # Add random embeddings
    torch.manual_seed(42)
    embeddings = torch.randn(num_nodes, 32)
    x = torch.cat([x, embeddings], dim=1)  # (num_nodes, 34)
    
    # Create edges
    edge_src = []
    edge_dst = []
    edge_weights = []
    edge_types = []
    
    for _, row in df.iterrows():
        ligand_idx = gene_to_idx[row['ligand_gene']]
        celltype_B_idx = cell_type_to_idx[row['cell_type_B']]
        celltype_A_idx = cell_type_to_idx[row['cell_type_A']]
        receptor_idx = gene_to_idx[row['receptor_gene']]
        weight = row['weight']
        
        # Edge 1: ligand_gene -> cell_type_B
        edge_src.append(ligand_idx)
        edge_dst.append(celltype_B_idx)
        edge_weights.append(weight)
        edge_types.append(0)
        
        # Edge 2: cell_type_A -> receptor_gene
        edge_src.append(celltype_A_idx)
        edge_dst.append(receptor_idx)
        edge_weights.append(weight)
        edge_types.append(1)
    
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
    edge_type = torch.tensor(edge_types, dtype=torch.long).unsqueeze(1)
    
    print(f"  Total edges: {edge_index.shape[1]}")
    print(f"  Edge weight range: [{edge_attr.min():.4f}, {edge_attr.max():.4f}]")
    print(f"  Edge weight mean: {edge_attr.mean():.4f}")
    
    # Create PyG Data object
    print("\n[4/4] Creating and saving PyG Data object...")
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
    output_path = results_dir / "graph.pt"
    torch.save(graph, str(output_path))
    print(f"\nGraph saved to: {output_path}")
    print(f"  Nodes: {graph.x.shape[0]}")
    print(f"  Node features: {graph.x.shape[1]}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Edge attributes: {graph.edge_attr.shape}")
    
    # Save statistics
    stats = {
        "num_nodes": num_nodes,
        "num_edges": edge_index.shape[1],
        "num_cell_types": len(cell_types),
        "num_genes": len(genes),
        "num_node_features": graph.x.shape[1],
        "edge_weight_min": float(edge_attr.min()),
        "edge_weight_max": float(edge_attr.max()),
        "edge_weight_mean": float(edge_attr.mean()),
        "edge_weight_std": float(edge_attr.std()),
        "cell_types": cell_types[:5] + (['...'] if len(cell_types) > 5 else []),
        "num_interactions": len(df),
        "affinity_weighted": True
    }
    
    stats_path = results_dir / "graph_stats.txt"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_path}")
    
    print("\n" + "=" * 70)
    print("Graph regeneration complete!")
    print("=" * 70)
    
    return graph

if __name__ == "__main__":
    regenerate_graph_from_edge_list()

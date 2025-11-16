#!/usr/bin/env python3
"""Generate improved node embeddings using multiple strategies."""

import torch
import numpy as np
import pandas as pd
import os
from scipy import sparse

def generate_laplacian_embeddings(graph_path='results/graph.pt', embedding_dim=32):
    """
    Generate embeddings using graph Laplacian eigenvectors.
    This captures global graph structure without external dependencies.
    """
    print("Loading graph...")
    graph = torch.load(graph_path, weights_only=False)
    num_nodes = graph.x.shape[0]
    num_edges = graph.edge_index.shape[1]
    
    print(f"Graph: {num_nodes} nodes, {num_edges} edges")
    
    # Build adjacency matrix
    print("\nBuilding adjacency matrix...")
    src = graph.edge_index[0].cpu().numpy()
    dst = graph.edge_index[1].cpu().numpy()
    
    # Use edge weights if available
    if graph.edge_attr is not None:
        weights = graph.edge_attr.cpu().numpy().flatten()
    else:
        weights = np.ones(len(src))
    
    # Build symmetric adjacency matrix
    A = sparse.coo_matrix((weights, (src, dst)), shape=(num_nodes, num_nodes))
    A_sym = A + A.T  # Make symmetric (undirected)
    A_sym = A_sym.tocsr()
    
    print(f"  Sparsity: {1 - A_sym.nnz / (num_nodes * num_nodes):.4f}")
    
    # Compute degree matrix
    print("Computing degree matrix...")
    degree = np.array(A_sym.sum(axis=1)).flatten()
    degree_sqrt_inv = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
    D_sqrt_inv = sparse.diags(degree_sqrt_inv)
    
    # Normalized Laplacian: L = I - D^-1/2 A D^-1/2
    # Using: L = I - D^-1/2 A D^-1/2
    print("Computing normalized Laplacian...")
    I = sparse.eye(num_nodes, format='csr')
    L_norm = I - D_sqrt_inv @ A_sym @ D_sqrt_inv
    
    # Get smallest eigenvectors (except first which is all ones)
    print(f"Computing top {embedding_dim} eigenvectors...")
    try:
        from scipy import sparse as sp_sparse
        from scipy.sparse.linalg import eigsh
        
        # Compute smallest eigenvalues/eigenvectors
        # We want eigenvectors corresponding to smallest eigenvalues
        eigenvals, eigenvecs = eigsh(L_norm, k=embedding_dim, which='SM', maxiter=1000)
        
        print(f"  Eigenvalues (first 5): {eigenvals[:5]}")
        
        # Use eigenvectors as embeddings
        embeddings = eigenvecs.astype(np.float32)
        
    except Exception as e:
        print(f"  Warning: Eigenvector computation failed ({e}). Using random embeddings as fallback.")
        embeddings = np.random.randn(num_nodes, embedding_dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    print(f"  Min: {embeddings.min():.6f}")
    print(f"  Max: {embeddings.max():.6f}")
    
    return embeddings

def generate_degree_weighted_embeddings(graph_path='results/graph.pt', embedding_dim=32):
    """
    Generate embeddings based on node degree and neighborhood features.
    """
    print("Loading graph...")
    graph = torch.load(graph_path, weights_only=False)
    num_nodes = graph.x.shape[0]
    
    # Get node degrees
    src = graph.edge_index[0].cpu().numpy()
    dst = graph.edge_index[1].cpu().numpy()
    
    in_degree = np.bincount(dst, minlength=num_nodes)
    out_degree = np.bincount(src, minlength=num_nodes)
    total_degree = in_degree + out_degree
    
    # Get edge weights
    if graph.edge_attr is not None:
        weights = graph.edge_attr.cpu().numpy().flatten()
    else:
        weights = np.ones(len(src))
    
    weighted_in_degree = np.bincount(dst, weights=weights, minlength=num_nodes)
    weighted_out_degree = np.bincount(src, weights=weights, minlength=num_nodes)
    
    # Build embedding
    embeddings = []
    
    # Degree features
    embeddings.append(np.log1p(total_degree)[:, None])  # Log degree
    embeddings.append(np.log1p(weighted_in_degree + weighted_out_degree)[:, None])  # Log weighted degree
    
    # Normalize features
    for i in range(len(embeddings)):
        mean = embeddings[i].mean()
        std = embeddings[i].std() + 1e-6
        embeddings[i] = (embeddings[i] - mean) / std
    
    # Add random components for diversity
    random_part = np.random.randn(num_nodes, embedding_dim - len(embeddings))
    random_part = random_part / np.linalg.norm(random_part, axis=1, keepdims=True)
    
    embeddings.append(random_part)
    
    embeddings = np.concatenate(embeddings, axis=1).astype(np.float32)
    
    # Ensure correct size
    if embeddings.shape[1] > embedding_dim:
        embeddings = embeddings[:, :embedding_dim]
    elif embeddings.shape[1] < embedding_dim:
        pad = np.random.randn(num_nodes, embedding_dim - embeddings.shape[1]).astype(np.float32)
        embeddings = np.concatenate([embeddings, pad], axis=1)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    
    return embeddings

def replace_node_features(graph_path, embeddings, method_name='laplacian'):
    """Replace node features in graph with new embeddings."""
    print(f"\nUpdating graph with {method_name} embeddings...")
    
    graph = torch.load(graph_path, weights_only=False)
    old_features = graph.x.clone()
    
    # Replace node features
    graph.x = torch.tensor(embeddings, dtype=torch.float32)
    
    print(f"  Old features shape: {old_features.shape}")
    print(f"  New features shape: {graph.x.shape}")
    
    # Create backup before overwriting
    backup_path = graph_path.replace('.pt', '_with_random_features.pt')
    if not os.path.exists(backup_path):
        # Only backup if not already done
        graph_backup = torch.load(graph_path, weights_only=False)
        torch.save(graph_backup, backup_path)
        print(f"Original graph backed up to {backup_path}")
    
    # Save updated graph
    torch.save(graph, graph_path)
    print(f"Graph updated with {method_name} embeddings: {graph_path}")
    
    return graph

def main():
    """Generate and integrate improved embeddings."""
    
    graph_path = 'results/graph.pt'
    embedding_dim = 32
    method = 'laplacian'  # Options: 'laplacian', 'degree'
    
    if method == 'laplacian':
        embeddings = generate_laplacian_embeddings(graph_path, embedding_dim)
    elif method == 'degree':
        embeddings = generate_degree_weighted_embeddings(graph_path, embedding_dim)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Save embeddings
    os.makedirs('results/embeddings', exist_ok=True)
    np.save(f'results/embeddings/{method}_embeddings.npy', embeddings)
    print(f"Saved embeddings to results/embeddings/{method}_embeddings.npy")
    
    # Replace graph features
    graph = replace_node_features(graph_path, embeddings, method)
    
    print(f"Improved embeddings generated using {method}")

if __name__ == "__main__":
    main()

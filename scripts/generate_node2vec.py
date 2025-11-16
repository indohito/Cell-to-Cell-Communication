#!/usr/bin/env python3
"""Generate Node2Vec embeddings for graph nodes."""

import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import Node2Vec
import os

def generate_node2vec_embeddings(graph_path='results/graph.pt', embedding_dim=32, 
                                 walk_length=20, context_size=10, walks_per_node=10,
                                 epochs=40, batch_size=128, num_workers=4):
    """
    Generate Node2Vec embeddings for all nodes in the graph.
    
    Args:
        graph_path: Path to saved graph
        embedding_dim: Dimensionality of embeddings
        walk_length: Length of each random walk
        context_size: Context window size
        walks_per_node: Number of walks per node
        epochs: Training epochs
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        embeddings: (num_nodes, embedding_dim) numpy array
    """
    print("Loading graph...")
    graph = torch.load(graph_path, weights_only=False)
    num_nodes = graph.x.shape[0]
    
    print(f"Graph loaded: {num_nodes} nodes, {graph.edge_index.shape[1]} edges")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\nInitializing Node2Vec...")
    node2vec = Node2Vec(
        graph.edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=1.0,  # Return parameter
        q=1.0,  # In-out parameter
        num_negative_samples=1,
        sparse=True
    ).to(device)
    
    print("\nPreparing data loader...")
    loader = node2vec.loader(batch_size=batch_size, shuffle=True, num_workers=0)
    
    print("Training Node2Vec...")
    optimizer = torch.optim.SparseAdam(node2vec.parameters(), lr=0.01)
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        count = 0
        
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        
        avg_loss = total_loss / max(1, count)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.6f}")
    
    # Extract final embeddings
    print("\nExtracting embeddings...")
    with torch.no_grad():
        all_nodes = torch.arange(num_nodes, device=device)
        # Process in batches to avoid OOM
        embeddings_list = []
        batch_size_extract = 512
        for i in range(0, num_nodes, batch_size_extract):
            batch_nodes = all_nodes[i:i+batch_size_extract]
            batch_emb = node2vec(batch_nodes).detach().cpu()
            embeddings_list.append(batch_emb)
        
        embeddings = torch.cat(embeddings_list, dim=0).numpy()
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding mean: {embeddings.mean():.6f}, std: {embeddings.std():.6f}")
    
    return embeddings, node2vec

def replace_node_features(graph_path, embeddings):
    """Replace node features in graph with Node2Vec embeddings."""
    print("\nUpdating graph with Node2Vec embeddings...")
    
    graph = torch.load(graph_path, weights_only=False)
    old_features = graph.x.clone()
    
    # Replace node features
    graph.x = torch.tensor(embeddings, dtype=torch.float32)
    
    print(f"  Old features shape: {old_features.shape}")
    print(f"  New features shape: {graph.x.shape}")
    
    # Save updated graph
    torch.save(graph, graph_path)
    print(f"Graph saved with Node2Vec embeddings to {graph_path}")
    
    # Save old features for ablation studies
    torch.save(old_features, graph_path.replace('.pt', '_old_features.pt'))
    print(f"Old features saved to {graph_path.replace('.pt', '_old_features.pt')}")
    
    return graph

def main():
    """Generate and integrate Node2Vec embeddings."""
    
    # Generate embeddings
    embeddings, model = generate_node2vec_embeddings(
        graph_path='results/graph.pt',
        embedding_dim=32,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        epochs=40,
        batch_size=128
    )
    
    # Save embeddings
    os.makedirs('results/embeddings', exist_ok=True)
    np.save('results/embeddings/node2vec_embeddings.npy', embeddings)
    print(f"\nSaved embeddings to results/embeddings/node2vec_embeddings.npy")
    
    # Replace graph features
    graph = replace_node_features('results/graph.pt', embeddings)
    
    print("\nNode2Vec embedding generation complete")

if __name__ == "__main__":
    main()

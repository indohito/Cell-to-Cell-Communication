#!/usr/bin/env python3
"""
Training script for GCN and GraphSAGE models.

This script trains alternative GNN architectures (GCN, GraphSAGE) using
the same data pipeline and evaluation metrics as the existing GAT models.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from models.graph_baselines import GCNLinkPredictor, SAGELinkPredictor
from training.utils import (
    get_device,
    load_graph_and_edges,
    build_data_splits,
    train_one_epoch,
    evaluate_model,
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def train_model(model_type, epochs=15, hidden_dim=64, num_layers=2, dropout=0.2, lr=0.001):
    """
    Train a GCN or GraphSAGE model.
    
    Args:
        model_type: 'gcn' or 'sage'
        epochs: number of training epochs
        hidden_dim: hidden dimension
        num_layers: number of GNN layers
        dropout: dropout rate
        lr: learning rate
    
    Returns:
        tuple: (training_history, test_metrics)
    """
    print(f"\n{'='*60}")
    print(f"=== Training {model_type.upper()} Model ===")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading graph and edge list...")
    graph, edge_list = load_graph_and_edges()
    print(f"Loaded graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
    
    # Build data splits
    data = build_data_splits(graph, edge_list)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Build model
    in_channels = graph.x.shape[1]
    if model_type.lower() == 'gcn':
        model = GCNLinkPredictor(
            input_dim=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type.lower() == 'sage':
        model = SAGELinkPredictor(
            input_dim=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'gcn' or 'sage'")
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type.upper()}")
    print(f"  Input dimension: {in_channels}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Total parameters: {num_params:,}\n")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()
    
    # Training history
    history = []
    
    print(f"Training for {epochs} epochs...\n")
    for epoch in range(1, epochs + 1):
        # Train one epoch
        train_loss = train_one_epoch(
            model, data, optimizer, criterion, device,
            use_weights=False, batch_size=512
        )
        
        # Evaluate on validation set
        val_metrics = evaluate_model(model, data, split='val', device=device, use_weights=False)
        
        # Update learning rate
        scheduler.step()
        
        # Log progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{model_type.upper()}] Epoch {epoch:02d}/{epochs} "
                  f"loss={train_loss:.4f} "
                  f"ROC={val_metrics['roc_auc']:.4f} "
                  f"PR={val_metrics['pr_auc']:.4f} "
                  f"F1={val_metrics['f1']:.4f}")
        
        # Record history
        record = {
            'model': model_type,
            'epoch': epoch,
            'loss': train_loss,
            'roc_auc': val_metrics['roc_auc'],
            'pr_auc': val_metrics['pr_auc'],
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'f1': val_metrics['f1'],
        }
        history.append(record)
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_metrics = evaluate_model(model, data, split='test', device=device, use_weights=False)
    
    print(f"\n  [{model_type.upper()}] Test Set Performance:")
    print(f"    ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"    PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"    Precision: {test_metrics['precision']:.4f}")
    print(f"    Recall:    {test_metrics['recall']:.4f}")
    print(f"    F1:        {test_metrics['f1']:.4f}")
    
    # Save model
    Path('results/models').mkdir(exist_ok=True, parents=True)
    model_path = f'results/models/model_{model_type}.pt'
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to {model_path}")
    
    return history, test_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train GCN or GraphSAGE models for link prediction'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['gcn', 'sage'],
        required=True,
        help='Model type: gcn or sage'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Number of training epochs (default: 15)'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=64,
        help='Hidden dimension (default: 64)'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=2,
        help='Number of GNN layers (default: 2)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate (default: 0.2)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    args = parser.parse_args()
    
    # Train model
    history, test_metrics = train_model(
        model_type=args.model_type,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr
    )
    
    # Save results
    print(f"\n{'='*60}")
    print("=== Saving Results ===")
    print(f"{'='*60}\n")
    
    Path('results').mkdir(exist_ok=True, parents=True)
    
    # Save training history
    df_history = pd.DataFrame(history)
    history_path = f'results/{args.model_type}_history.csv'
    df_history.to_csv(history_path, index=False)
    print(f"Saved training history to {history_path}")
    
    # Save final test metrics
    # Check if training_metrics.csv exists and what format it uses
    metrics_path = 'results/training_metrics.csv'
    if Path(metrics_path).exists():
        # Try to append to existing file
        try:
            df_existing = pd.read_csv(metrics_path)
            # Add new row
            new_row = {
                'model': args.model_type,
                'roc_auc': test_metrics['roc_auc'],
                'pr_auc': test_metrics['pr_auc'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1': test_metrics['f1'],
            }
            # Add date if column exists
            if 'date' in df_existing.columns:
                new_row['date'] = datetime.now().strftime('%Y-%m-%d')
            df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
            df_existing.to_csv(metrics_path, index=False)
            print(f"Appended metrics to {metrics_path}")
        except Exception as e:
            print(f"Warning: Could not append to {metrics_path}: {e}")
            # Create new file
            new_metrics_path = 'results/new_training_metrics.csv'
            df_new = pd.DataFrame([{
                'model': args.model_type,
                'roc_auc': test_metrics['roc_auc'],
                'pr_auc': test_metrics['pr_auc'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1': test_metrics['f1'],
                'date': datetime.now().strftime('%Y-%m-%d'),
            }])
            df_new.to_csv(new_metrics_path, index=False)
            print(f"Saved metrics to {new_metrics_path}")
    else:
        # Create new metrics file
        df_metrics = pd.DataFrame([{
            'model': args.model_type,
            'roc_auc': test_metrics['roc_auc'],
            'pr_auc': test_metrics['pr_auc'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'date': datetime.now().strftime('%Y-%m-%d'),
        }])
        df_metrics.to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()


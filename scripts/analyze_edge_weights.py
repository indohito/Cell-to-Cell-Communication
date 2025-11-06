#!/usr/bin/env python3
"""Analyze edge weight distribution and suggest improvements."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_weights(csv_path='results/integration_table_weighted.csv', weight_col='combined_weight_norm'):
    """Analyze edge weight distribution."""
    print(f"Loading weights from {csv_path}...")
    df = pd.read_csv(csv_path)
    weights = df[weight_col].values
    
    print(f"\nWeight statistics for '{weight_col}':")
    print(f"  Mean: {weights.mean():.6f}")
    print(f"  Std: {weights.std():.6f}")
    print(f"  Min: {weights.min():.6f}, Max: {weights.max():.6f}")
    print(f"  Percentiles:")
    print(f"    1%={np.percentile(weights, 1):.6f}")
    print(f"    5%={np.percentile(weights, 5):.6f}")
    print(f"    25%={np.percentile(weights, 25):.6f}")
    print(f"    50%={np.percentile(weights, 50):.6f}")
    print(f"    75%={np.percentile(weights, 75):.6f}")
    print(f"    95%={np.percentile(weights, 95):.6f}")
    print(f"    99%={np.percentile(weights, 99):.6f}")
    
    num_zeros = (weights == 0).sum()
    print(f"  Zeros: {num_zeros} / {len(weights)} ({100 * num_zeros / len(weights):.1f}%)")
    
    # Plot distribution
    os.makedirs('results/figures', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Raw histogram
    axes[0, 0].hist(weights, bins=100, edgecolor='k', alpha=0.7)
    axes[0, 0].set_xlabel(weight_col)
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Raw weight distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log scale (non-zero only)
    weights_nonzero = weights[weights > 0]
    axes[0, 1].hist(np.log1p(weights_nonzero), bins=100, edgecolor='k', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel(f'log1p({weight_col})')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Log-scale distribution (non-zero only)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_weights = np.sort(weights)
    cumsum = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    axes[1, 0].plot(np.linspace(0, 100, len(cumsum)), cumsum, linewidth=2)
    axes[1, 0].set_xlabel('Percentile (%)')
    axes[1, 0].set_ylabel('Cumulative weight fraction')
    axes[1, 0].set_title('Cumulative weight distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot with log scale
    axes[1, 1].boxplot([weights_nonzero, np.log1p(weights_nonzero)], 
                        labels=['Raw', 'Log1p'],
                        patch_artist=True)
    axes[1, 1].set_ylabel('Weight value')
    axes[1, 1].set_title('Weight distributions (Box plot)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'results/figures/edge_weight_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved weight analysis plot to {output_path}")
    plt.close()
    
    # Recommendation
    print("\nRecommendations:")
    if weights_nonzero.std() / weights_nonzero.mean() > 2:
        print("  - High variance detected (CV > 2). Consider log-scaling for better numerical stability.")
    if num_zeros / len(weights) > 0.5:
        print("  - High sparsity detected (>50% zeros). Ensure loss function handles sparse weights.")
    print("  - Consider edge weight normalization: robust scaling or quantile normalization.")
    
    return {
        'mean': float(weights.mean()),
        'std': float(weights.std()),
        'min': float(weights.min()),
        'max': float(weights.max()),
        'num_zeros': int(num_zeros),
        'pct_zeros': float(100 * num_zeros / len(weights))
    }

if __name__ == '__main__':
    stats = analyze_weights()

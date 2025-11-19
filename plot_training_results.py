#!/usr/bin/env python3
"""Plot training results from training_history_improved.csv"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
df = pd.read_csv('training_history_improved.csv')

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss curves for both models
ax = axes[0]
for model in df['model'].unique():
    model_data = df[df['model'] == model]
    ax.plot(model_data['epoch'], model_data['loss'], 
            marker='o', label=model, linewidth=2, markersize=6)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss (BCELoss)', fontsize=12, fontweight='bold')
ax.set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, 11))

# Plot 2: Comparison of final losses with error bars
ax = axes[1]
models = df['model'].unique()
final_losses = []
min_losses = []
mean_losses = []

for model in models:
    model_data = df[df['model'] == model]
    final_losses.append(model_data['loss'].iloc[-1])
    min_losses.append(model_data['loss'].min())
    mean_losses.append(model_data['loss'].mean())

x_pos = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x_pos - width, final_losses, width, label='Final Loss', alpha=0.8)
bars2 = ax.bar(x_pos, min_losses, width, label='Best Loss', alpha=0.8)
bars3 = ax.bar(x_pos + width, mean_losses, width, label='Mean Loss', alpha=0.8)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Loss Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/training_results_plots.png', dpi=300, bbox_inches='tight')
print("✓ Saved plot to results/training_results_plots.png")

# Print summary statistics
print("\n" + "="*60)
print("Training Summary Statistics")
print("="*60)
summary = df.groupby('model')[['loss']].agg(['min', 'max', 'mean', 'std']).round(6)
print(summary)

print("\n" + "="*60)
print("Final Loss Comparison")
print("="*60)
for model in models:
    model_data = df[df['model'] == model]
    final = model_data['loss'].iloc[-1]
    best = model_data['loss'].min()
    improvement = ((model_data['loss'].iloc[0] - final) / model_data['loss'].iloc[0]) * 100
    print(f"{model:20s}: Final={final:.6f}, Best={best:.6f}, Improvement={improvement:.2f}%")

plt.show()

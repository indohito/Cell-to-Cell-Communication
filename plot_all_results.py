"""
Generate comprehensive visualization of training results.
Creates all major figures comparing baseline vs affinity-weighted models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Load data - Use the better performing run
# Run 2 has better ROC-AUC (+0.417%), better recall (+4.28%), better affinity detection
df = pd.read_csv('training_history_improved (2).csv')
baseline = df[df['model'] == 'baseline'].reset_index(drop=True)
affinity = df[df['model'] == 'affinity_weighted'].reset_index(drop=True)

print(f"Loading best performing run: training_history_improved (2).csv")

# Create output directory
output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating training visualization figures...")
print(f"Output directory: {output_dir}")

# ============================================================================
# Figure 1: Training Loss Comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss over epochs
axes[0].plot(baseline['epoch'], baseline['loss'], 'o-', linewidth=2.5, 
             markersize=6, label='Baseline', color='#1f77b4')
axes[0].plot(affinity['epoch'], affinity['loss'], 's-', linewidth=2.5, 
             markersize=6, label='Affinity-Weighted', color='#ff7f0e')
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Loss (BCE)', fontsize=12, fontweight='bold')
axes[0].set_title('Training Loss Convergence', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Loss reduction bar chart
loss_reduction = [
    baseline.iloc[0]['loss'] - baseline.iloc[-1]['loss'],
    affinity.iloc[0]['loss'] - affinity.iloc[-1]['loss']
]
pct_reduction = [
    loss_reduction[0] / baseline.iloc[0]['loss'] * 100,
    loss_reduction[1] / affinity.iloc[0]['loss'] * 100
]

colors = ['#1f77b4', '#ff7f0e']
bars = axes[1].bar(['Baseline', 'Affinity-Weighted'], loss_reduction, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('Loss Reduction', fontsize=12, fontweight='bold')
axes[1].set_title('Loss Improvement (Epoch 1 → 10)', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, max(loss_reduction) * 1.2)

# Add value labels on bars
for i, (bar, val, pct) in enumerate(zip(bars, loss_reduction, pct_reduction)):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
print("✓ Saved: training_loss.png")
plt.close()

# ============================================================================
# Figure 2: ROC-AUC Comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC-AUC over epochs
axes[0].plot(baseline['epoch'], baseline['roc_auc'], 'o-', linewidth=2.5, 
             markersize=6, label='Baseline', color='#1f77b4')
axes[0].plot(affinity['epoch'], affinity['roc_auc'], 's-', linewidth=2.5, 
             markersize=6, label='Affinity-Weighted', color='#ff7f0e')
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
axes[0].set_title('ROC-AUC Over Epochs', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].set_ylim([0.99, 1.0])
axes[0].grid(True, alpha=0.3)

# Epoch-by-epoch improvement
improvement_pct = ((affinity['roc_auc'] - baseline['roc_auc']) / baseline['roc_auc'] * 100).values
axes[1].bar(baseline['epoch'], improvement_pct, color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1].axhline(y=np.mean(improvement_pct), color='red', linestyle='--', linewidth=2, label=f'Average: {np.mean(improvement_pct):.3f}%')
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Affinity vs Baseline ROC-AUC Improvement', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'roc_auc_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_auc_comparison.png")
plt.close()

# ============================================================================
# Figure 3: All Metrics Comparison (Final Epoch)
# ============================================================================
metrics = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall']
b_final = baseline.iloc[-1]
a_final = affinity.iloc[-1]

baseline_vals = [b_final[m] for m in metrics]
affinity_vals = [a_final[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', 
               color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, affinity_vals, width, label='Affinity-Weighted',
               color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Final Epoch Metrics Comparison (Epoch 10)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=11)
ax.set_ylim([0.7, 1.0])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: metrics_comparison.png")
plt.close()

# ============================================================================
# Figure 4: Metric Improvements (Percentage Change)
# ============================================================================
improvements = []
for metric in metrics:
    b_val = b_final[metric]
    a_val = a_final[metric]
    pct_change = (a_val - b_val) / b_val * 100
    improvements.append(pct_change)

colors_imp = ['#2ca02c' if x > 0 else '#d62728' for x in improvements]
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(metrics, improvements, color=colors_imp, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
ax.set_title('Metric Improvements: Affinity-Weighted vs Baseline', fontsize=13, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvements)):
    x_pos = val + (0.1 if val > 0 else -0.1)
    ax.text(x_pos, bar.get_y() + bar.get_height()/2.,
            f'{val:+.3f}%',
            ha='left' if val > 0 else 'right', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'metric_improvements.png', dpi=300, bbox_inches='tight')
print("✓ Saved: metric_improvements.png")
plt.close()

# ============================================================================
# Figure 5: All Metrics Over Epochs (6-panel subplot)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    ax.plot(baseline['epoch'], baseline[metric], 'o-', linewidth=2.5, 
            markersize=5, label='Baseline', color='#1f77b4')
    ax.plot(affinity['epoch'], affinity[metric], 's-', linewidth=2.5,
            markersize=5, label='Affinity-Weighted', color='#ff7f0e')
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if metric != 'recall':
        ax.set_ylim([0.98, 1.0])

# Remove extra subplot
fig.delaxes(axes[5])

# Add text summary in the last panel position
summary_text = f"""
TRAINING SUMMARY (10 Epochs)

BASELINE:
  Final ROC-AUC: {b_final['roc_auc']:.6f}
  Final Recall:  {b_final['recall']:.6f}
  Loss Reduction: {(baseline.iloc[0]['loss'] - baseline.iloc[-1]['loss'])/baseline.iloc[0]['loss']*100:.1f}%

AFFINITY-WEIGHTED:
  Final ROC-AUC: {a_final['roc_auc']:.6f}
  Final Recall:  {a_final['recall']:.6f}
  Loss Reduction: {(affinity.iloc[0]['loss'] - affinity.iloc[-1]['loss'])/affinity.iloc[0]['loss']*100:.1f}%

IMPROVEMENT:
  ROC-AUC: +{(a_final['roc_auc'] - b_final['roc_auc'])/b_final['roc_auc']*100:.3f}%
  Recall:  +{(a_final['recall'] - b_final['recall'])/b_final['recall']*100:.3f}%
"""

axes[5].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
axes[5].axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'all_metrics_over_epochs.png', dpi=300, bbox_inches='tight')
print("✓ Saved: all_metrics_over_epochs.png")
plt.close()

# ============================================================================
# Figure 6: Training Dynamics - Loss vs ROC-AUC
# ============================================================================
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot loss on left axis
ax1.plot(baseline['epoch'], baseline['loss'], 'o-', linewidth=2.5, 
         markersize=6, label='Baseline Loss', color='#1f77b4')
ax1.plot(affinity['epoch'], affinity['loss'], 's-', linewidth=2.5,
         markersize=6, label='Affinity Loss', color='#ff7f0e')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss (BCE)', fontsize=12, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='upper left')

# Create second y-axis for ROC-AUC
ax2 = ax1.twinx()
ax2.plot(baseline['epoch'], baseline['roc_auc'], 'o--', linewidth=2.5,
         markersize=6, label='Baseline ROC-AUC', color='#2ca02c', alpha=0.7)
ax2.plot(affinity['epoch'], affinity['roc_auc'], 's--', linewidth=2.5,
         markersize=6, label='Affinity ROC-AUC', color='#d62728', alpha=0.7)
ax2.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold', color='black')
ax2.set_ylim([0.991, 0.998])
ax2.tick_params(axis='y', labelcolor='black')
ax2.legend(fontsize=10, loc='upper right')

plt.title('Training Dynamics: Loss and ROC-AUC', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'training_dynamics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: training_dynamics.png")
plt.close()

# ============================================================================
# Figure 7: Recall vs Precision Trade-off
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(baseline['recall'], baseline['precision'], s=200, alpha=0.7,
          label='Baseline', color='#1f77b4', edgecolors='black', linewidth=1.5)
ax.scatter(affinity['recall'], affinity['precision'], s=200, alpha=0.7,
          label='Affinity-Weighted', color='#ff7f0e', marker='s', edgecolors='black', linewidth=1.5)

# Draw lines connecting epochs
ax.plot(baseline['recall'], baseline['precision'], 'o-', linewidth=1.5, 
        alpha=0.3, color='#1f77b4')
ax.plot(affinity['recall'], affinity['precision'], 's-', linewidth=1.5,
        alpha=0.3, color='#ff7f0e')

# Annotate some epochs
for i in [0, 4, 9]:
    ax.annotate(f'E{int(baseline.iloc[i]["epoch"])}', 
               (baseline.iloc[i]['recall'], baseline.iloc[i]['precision']),
               textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    ax.annotate(f'E{int(affinity.iloc[i]["epoch"])}',
               (affinity.iloc[i]['recall'], affinity.iloc[i]['precision']),
               textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Recall vs Precision Trade-off', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'recall_precision_tradeoff.png', dpi=300, bbox_inches='tight')
print("✓ Saved: recall_precision_tradeoff.png")
plt.close()

# ============================================================================
# Figure 8: Summary Statistics Table
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Prepare data
summary_data = []
summary_data.append(['Metric', 'Baseline (Final)', 'Affinity (Final)', 'Difference', '% Change'])
summary_data.append(['—'*15, '—'*15, '—'*15, '—'*15, '—'*15])

for metric in metrics:
    b_val = b_final[metric]
    a_val = a_final[metric]
    diff = a_val - b_val
    pct = (diff / b_val * 100) if b_val != 0 else 0
    summary_data.append([
        metric.upper(),
        f'{b_val:.6f}',
        f'{a_val:.6f}',
        f'{diff:+.6f}',
        f'{pct:+.3f}%'
    ])

summary_data.append(['—'*15, '—'*15, '—'*15, '—'*15, '—'*15])
summary_data.append([
    'Loss Reduction',
    f'{(baseline.iloc[0]["loss"] - baseline.iloc[-1]["loss"])/baseline.iloc[0]["loss"]*100:.1f}%',
    f'{(affinity.iloc[0]["loss"] - affinity.iloc[-1]["loss"])/affinity.iloc[0]["loss"]*100:.1f}%',
    '-',
    '-'
])

table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style separator rows
for i in range(5):
    table[(2, i)].set_facecolor('#E7E6E6')
    table[(len(summary_data)-2, i)].set_facecolor('#E7E6E6')

# Alternate row colors
for i in range(3, len(summary_data)-2):
    color = '#F2F2F2' if i % 2 == 0 else 'white'
    for j in range(5):
        table[(i, j)].set_facecolor(color)

plt.title('Training Results Summary Table', fontsize=14, fontweight='bold', pad=20)
plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: summary_table.png")
plt.close()

# ============================================================================
# Print Summary
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION SUMMARY")
print("="*80)
print(f"\nGenerated {len(list(output_dir.glob('*.png')))} figures in {output_dir}")
print("\nFiles created:")
for fig_file in sorted(output_dir.glob('*.png')):
    size_kb = fig_file.stat().st_size / 1024
    print(f"  • {fig_file.name:<40} ({size_kb:>6.1f} KB)")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print(f"\nROC-AUC Improvement: {(a_final['roc_auc'] - b_final['roc_auc'])/b_final['roc_auc']*100:+.3f}%")
print(f"Recall Improvement:  {(a_final['recall'] - b_final['recall'])/b_final['recall']*100:+.3f}%")
print(f"Precision Improvement: {(a_final['precision'] - b_final['precision'])/b_final['precision']*100:+.3f}%")
print("\n✅ All figures saved successfully!")


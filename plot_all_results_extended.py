"""
Generate comprehensive visualization of training results for ALL models.
Creates all major figures comparing GAT baseline, GAT affinity, GCN, and GraphSAGE models.
Similar to plot_all_results.py but includes all model variants.
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

# Create output directory
output_dir = Path('results/figures_extended')
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("GENERATING EXTENDED VISUALIZATIONS FOR ALL MODELS")
print("="*80)
print(f"Output directory: {output_dir}\n")

# Load training history data for all models
# Try to load from different possible file names
models_data = {}
model_configs = [
    ('baseline', 'training_history_improved (2).csv', 'Baseline GAT'),
    ('affinity_weighted', 'training_history_improved (2).csv', 'Affinity-Weighted GAT'),
    ('gcn', 'results/gcn_history.csv', 'GCN'),
    ('sage', 'results/sage_history.csv', 'GraphSAGE'),
]

print("Loading training history data...")
for model_key, filename, display_name in model_configs:
    # Try both root and results directory
    filepath = Path(filename)
    if not filepath.exists():
        # Try in results directory if not in root
        alt_path = Path('results') / Path(filename).name
        if alt_path.exists():
            filepath = alt_path
        else:
            print(f"  ⚠️  {filename} not found, skipping {display_name}")
            continue
    
    try:
        df = pd.read_csv(filepath)
        # Filter by model name if the CSV contains multiple models
        if 'model' in df.columns:
            model_df = df[df['model'] == model_key].reset_index(drop=True)
        else:
            # If no model column, assume the whole file is for this model
            model_df = df.copy()
        
        if len(model_df) > 0:
            models_data[model_key] = {
                'data': model_df,
                'name': display_name,
                'color': None  # Will assign colors below
            }
            print(f"  ✓ Loaded {display_name}: {len(model_df)} epochs")
        else:
            print(f"  ⚠️  No data found for {model_key} in {filename}")
    except Exception as e:
        print(f"  ⚠️  Error loading {filename}: {e}")

if len(models_data) == 0:
    print("\n❌ No training history data found. Please train models first.")
    exit(1)

# Assign colors to models
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for i, (key, _) in enumerate(models_data.items()):
    models_data[key]['color'] = colors[i % len(colors)]

print(f"\nLoaded {len(models_data)} models for visualization\n")

# Get model keys in consistent order
model_keys = list(models_data.keys())
if 'baseline' in model_keys:
    model_keys.remove('baseline')
    model_keys.insert(0, 'baseline')
if 'affinity_weighted' in model_keys:
    if 'baseline' in model_keys:
        idx = model_keys.index('baseline') + 1
    else:
        idx = 0
    if 'affinity_weighted' in model_keys:
        model_keys.remove('affinity_weighted')
    model_keys.insert(idx, 'affinity_weighted')

# ============================================================================
# Figure 1: Training Loss Comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss over epochs
for key in model_keys:
    model_info = models_data[key]
    df = model_info['data']
    if 'epoch' in df.columns and 'loss' in df.columns:
        axes[0].plot(df['epoch'], df['loss'], 'o-', linewidth=2.5, 
                    markersize=6, label=model_info['name'], color=model_info['color'])

axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Loss (BCE)', fontsize=12, fontweight='bold')
axes[0].set_title('Training Loss Convergence', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Loss reduction bar chart
loss_reductions = []
model_names = []
for key in model_keys:
    model_info = models_data[key]
    df = model_info['data']
    if 'loss' in df.columns and len(df) > 0:
        loss_reduction = df.iloc[0]['loss'] - df.iloc[-1]['loss']
        loss_reductions.append(loss_reduction)
        model_names.append(model_info['name'])

if loss_reductions:
    colors_bar = [models_data[key]['color'] for key in model_keys if key in models_data and 'loss' in models_data[key]['data'].columns]
    bars = axes[1].bar(model_names, loss_reductions, 
                      color=colors_bar[:len(loss_reductions)], 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Loss Reduction', fontsize=12, fontweight='bold')
    axes[1].set_title('Loss Improvement (First → Last Epoch)', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, max(loss_reductions) * 1.2)
    
    # Add value labels
    for bar, val in zip(bars, loss_reductions):
        height = bar.get_height()
        pct = (val / (val + df.iloc[-1]['loss'])) * 100 if val > 0 else 0
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(output_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
print("✓ Saved: training_loss.png")
plt.close()

# ============================================================================
# Figure 2: ROC-AUC Comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC-AUC over epochs
for key in model_keys:
    model_info = models_data[key]
    df = model_info['data']
    if 'epoch' in df.columns and 'roc_auc' in df.columns:
        axes[0].plot(df['epoch'], df['roc_auc'], 'o-', linewidth=2.5, 
                    markersize=6, label=model_info['name'], color=model_info['color'])

axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
axes[0].set_title('ROC-AUC Over Epochs', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].set_ylim([0.98, 1.0])
axes[0].grid(True, alpha=0.3)

# Epoch-by-epoch improvement (relative to baseline)
if 'baseline' in models_data:
    baseline_df = models_data['baseline']['data']
    if 'roc_auc' in baseline_df.columns and 'epoch' in baseline_df.columns:
        for key in model_keys:
            if key == 'baseline':
                continue
            model_info = models_data[key]
            df = model_info['data']
            if 'roc_auc' in df.columns and 'epoch' in df.columns:
                # Align epochs
                improvement_pct = []
                epochs = []
                for epoch in df['epoch']:
                    baseline_roc = baseline_df[baseline_df['epoch'] == epoch]['roc_auc'].values
                    model_roc = df[df['epoch'] == epoch]['roc_auc'].values
                    if len(baseline_roc) > 0 and len(model_roc) > 0:
                        improvement_pct.append((model_roc[0] - baseline_roc[0]) / baseline_roc[0] * 100)
                        epochs.append(epoch)
                
                if improvement_pct:
                    axes[1].plot(epochs, improvement_pct, 'o-', linewidth=2, 
                               markersize=5, label=f"{model_info['name']} vs Baseline",
                               color=model_info['color'], alpha=0.8)

axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
axes[1].set_title('ROC-AUC Improvement vs Baseline', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'roc_auc_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_auc_comparison.png")
plt.close()

# ============================================================================
# Figure 3: All Metrics Comparison (Final Epoch)
# ============================================================================
metrics = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall']
x = np.arange(len(metrics))
width = 0.8 / len(model_keys)  # Adjust width based on number of models

fig, ax = plt.subplots(figsize=(14, 6))

# Get final epoch values for each model
final_values = {}
for key in model_keys:
    model_info = models_data[key]
    df = model_info['data']
    if len(df) > 0:
        final_epoch = df.iloc[-1]
        final_values[key] = [final_epoch.get(m, 0) for m in metrics]

# Plot bars
for i, key in enumerate(model_keys):
    if key in final_values:
        offset = (i - len(model_keys)/2 + 0.5) * width
        bars = ax.bar(x + offset, final_values[key], width, 
                     label=models_data[key]['name'],
                     color=models_data[key]['color'], 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, final_values[key]):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Final Epoch Metrics Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=10)
ax.set_ylim([0.7, 1.0])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: metrics_comparison.png")
plt.close()

# ============================================================================
# Figure 4: Metric Improvements (Percentage Change vs Baseline)
# ============================================================================
if 'baseline' in models_data:
    baseline_final = models_data['baseline']['data'].iloc[-1]
    
    fig, axes = plt.subplots(len(model_keys) - 1, 1, figsize=(12, 4 * (len(model_keys) - 1)))
    if len(model_keys) - 1 == 1:
        axes = [axes]
    
    subplot_idx = 0
    for key in model_keys:
        if key == 'baseline':
            continue
        
        model_info = models_data[key]
        df = model_info['data']
        if len(df) == 0:
            continue
        
        model_final = df.iloc[-1]
        improvements = []
        
        for metric in metrics:
            b_val = baseline_final.get(metric, 0)
            m_val = model_final.get(metric, 0)
            if b_val > 0:
                pct_change = (m_val - b_val) / b_val * 100
                improvements.append(pct_change)
            else:
                improvements.append(0)
        
        colors_imp = ['#2ca02c' if x > 0 else '#d62728' for x in improvements]
        ax = axes[subplot_idx]
        bars = ax.barh(metrics, improvements, color=colors_imp, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_info["name"]} vs Baseline', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            x_pos = val + (0.1 if val > 0 else -0.1)
            ax.text(x_pos, bar.get_y() + bar.get_height()/2.,
                   f'{val:+.3f}%',
                   ha='left' if val > 0 else 'right', va='center', 
                   fontsize=10, fontweight='bold')
        
        subplot_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_improvements.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: metric_improvements.png")
    plt.close()

# ============================================================================
# Figure 5: All Metrics Over Epochs (Multi-panel subplot)
# ============================================================================
n_metrics = len(metrics)
n_cols = 3
n_rows = (n_metrics + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    for key in model_keys:
        model_info = models_data[key]
        df = model_info['data']
        if 'epoch' in df.columns and metric in df.columns:
            marker = 'o' if key == 'baseline' else 's' if key == 'affinity_weighted' else '^' if key == 'gcn' else 'D'
            ax.plot(df['epoch'], df[metric], marker=marker, linewidth=2.5, 
                   markersize=5, label=model_info['name'], color=model_info['color'])
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if metric != 'recall':
        ax.set_ylim([0.95, 1.0])

# Remove extra subplots
for idx in range(n_metrics, len(axes)):
    fig.delaxes(axes[idx])

# Add text summary in the last panel if there's space
if n_metrics < len(axes):
    summary_text = "TRAINING SUMMARY\n\n"
    for key in model_keys:
        model_info = models_data[key]
        df = model_info['data']
        if len(df) > 0:
            final = df.iloc[-1]
            summary_text += f"{model_info['name']}:\n"
            summary_text += f"  Final ROC-AUC: {final.get('roc_auc', 0):.6f}\n"
            summary_text += f"  Final F1: {final.get('f1', 0):.6f}\n\n"
    
    axes[n_metrics].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                        verticalalignment='center', bbox=dict(boxstyle='round', 
                        facecolor='wheat', alpha=0.5))
    axes[n_metrics].axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'all_metrics_over_epochs.png', dpi=300, bbox_inches='tight')
print("✓ Saved: all_metrics_over_epochs.png")
plt.close()

# ============================================================================
# Figure 6: Training Dynamics - Loss vs ROC-AUC
# ============================================================================
fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot loss on left axis
for key in model_keys:
    model_info = models_data[key]
    df = model_info['data']
    if 'epoch' in df.columns and 'loss' in df.columns:
        marker = 'o' if key == 'baseline' else 's' if key == 'affinity_weighted' else '^' if key == 'gcn' else 'D'
        ax1.plot(df['epoch'], df['loss'], marker=marker, linewidth=2.5, 
                markersize=6, label=f'{model_info["name"]} Loss', 
                color=model_info['color'])

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss (BCE)', fontsize=12, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, loc='upper left')

# Create second y-axis for ROC-AUC
ax2 = ax1.twinx()
for key in model_keys:
    model_info = models_data[key]
    df = model_info['data']
    if 'epoch' in df.columns and 'roc_auc' in df.columns:
        marker = 'o' if key == 'baseline' else 's' if key == 'affinity_weighted' else '^' if key == 'gcn' else 'D'
        ax2.plot(df['epoch'], df['roc_auc'], marker=marker, linestyle='--', 
                linewidth=2.5, markersize=6, label=f'{model_info["name"]} ROC-AUC',
                color=model_info['color'], alpha=0.7)

ax2.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold', color='black')
ax2.set_ylim([0.98, 1.0])
ax2.tick_params(axis='y', labelcolor='black')
ax2.legend(fontsize=9, loc='upper right')

plt.title('Training Dynamics: Loss and ROC-AUC', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'training_dynamics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: training_dynamics.png")
plt.close()

# ============================================================================
# Figure 7: Recall vs Precision Trade-off
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

markers = {'baseline': 'o', 'affinity_weighted': 's', 'gcn': '^', 'sage': 'D'}
for key in model_keys:
    model_info = models_data[key]
    df = model_info['data']
    if 'recall' in df.columns and 'precision' in df.columns:
        marker = markers.get(key, 'o')
        ax.scatter(df['recall'], df['precision'], s=150, alpha=0.7,
                  label=model_info['name'], color=model_info['color'], 
                  marker=marker, edgecolors='black', linewidth=1.5)
        
        # Draw lines connecting epochs
        ax.plot(df['recall'], df['precision'], linestyle='-', linewidth=1.5,
               alpha=0.3, color=model_info['color'])

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Recall vs Precision Trade-off', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'recall_precision_tradeoff.png', dpi=300, bbox_inches='tight')
print("✓ Saved: recall_precision_tradeoff.png")
plt.close()

# ============================================================================
# Figure 8: Summary Statistics Table
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('tight')
ax.axis('off')

# Prepare data
summary_data = []
header = ['Metric'] + [models_data[key]['name'] for key in model_keys]
summary_data.append(header)
summary_data.append(['—'*15] * len(header))

for metric in metrics:
    row = [metric.upper()]
    for key in model_keys:
        df = models_data[key]['data']
        if len(df) > 0:
            val = df.iloc[-1].get(metric, 0)
            row.append(f'{val:.6f}')
        else:
            row.append('N/A')
    summary_data.append(row)

summary_data.append(['—'*15] * len(header))

# Add loss reduction row
row = ['Loss Reduction']
for key in model_keys:
    df = models_data[key]['data']
    if len(df) > 0 and 'loss' in df.columns:
        reduction = (df.iloc[0]['loss'] - df.iloc[-1]['loss']) / df.iloc[0]['loss'] * 100
        row.append(f'{reduction:.1f}%')
    else:
        row.append('N/A')
summary_data.append(row)

table = ax.table(cellText=summary_data, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(len(header)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style separator rows
for i in range(len(header)):
    table[(2, i)].set_facecolor('#E7E6E6')
    table[(len(summary_data)-2, i)].set_facecolor('#E7E6E6')

# Alternate row colors
for i in range(3, len(summary_data)-2):
    color = '#F2F2F2' if i % 2 == 0 else 'white'
    for j in range(len(header)):
        table[(i, j)].set_facecolor(color)

plt.title('Training Results Summary Table (All Models)', fontsize=14, fontweight='bold', pad=20)
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

# Print final metrics for each model
for key in model_keys:
    model_info = models_data[key]
    df = model_info['data']
    if len(df) > 0:
        final = df.iloc[-1]
        print(f"\n{model_info['name']}:")
        print(f"  Final ROC-AUC: {final.get('roc_auc', 0):.6f}")
        print(f"  Final F1: {final.get('f1', 0):.6f}")
        print(f"  Final Precision: {final.get('precision', 0):.6f}")
        print(f"  Final Recall: {final.get('recall', 0):.6f}")

# Compare to baseline if available
if 'baseline' in models_data:
    baseline_final = models_data['baseline']['data'].iloc[-1]
    baseline_roc = baseline_final.get('roc_auc', 0)
    
    print("\n" + "-"*80)
    print("IMPROVEMENTS vs BASELINE:")
    print("-"*80)
    for key in model_keys:
        if key == 'baseline':
            continue
        model_info = models_data[key]
        df = model_info['data']
        if len(df) > 0:
            model_final = df.iloc[-1]
            model_roc = model_final.get('roc_auc', 0)
            if baseline_roc > 0:
                improvement = (model_roc - baseline_roc) / baseline_roc * 100
                print(f"  {model_info['name']}: {improvement:+.3f}% ROC-AUC improvement")

print("\n✅ All figures saved successfully!")


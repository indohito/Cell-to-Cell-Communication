# Training Guide for All Models

This guide explains how to train all models (GAT baseline, GAT affinity, GCN, GraphSAGE) and generate training history for visualization.

## Quick Start

### 1. Train GAT Models (Baseline and Affinity-Weighted)

These are already trained by `train.py`:

```bash
python train.py
```

This will:
- Train baseline GAT model
- Train affinity-weighted GAT model
- Save training history to `results/training_history_improved.csv`
- Save models to `results/models/model_improved_baseline.pt` and `results/models/model_improved_affinity_weighted.pt`

### 2. Train GCN Model

```bash
python train_gnn_variants.py --model_type gcn --epochs 15
```

This will:
- Train GCN model
- Save training history to `results/gcn_history.csv`
- Save model to `results/models/model_gcn.pt`

### 3. Train GraphSAGE Model

```bash
python train_gnn_variants.py --model_type sage --epochs 15
```

This will:
- Train GraphSAGE model
- Save training history to `results/sage_history.csv`
- Save model to `results/models/model_sage.pt`

## Training History Files

After training, you should have these CSV files:

- `results/training_history_improved.csv` (or `training_history_improved (2).csv`) - Contains baseline and affinity_weighted
- `results/gcn_history.csv` - Contains GCN training history
- `results/sage_history.csv` - Contains GraphSAGE training history

Each history file contains columns:
- `model`: model name (baseline, affinity_weighted, gcn, sage)
- `epoch`: epoch number
- `loss`: training loss
- `roc_auc`: validation ROC-AUC
- `pr_auc`: validation PR-AUC
- `precision`: validation precision
- `recall`: validation recall
- `f1`: validation F1 score

## Generate Extended Visualizations

Once all models are trained and history files exist:

```bash
python plot_all_results_extended.py
```

This will create visualizations in `results/figures_extended/` comparing all models.

## Verify Training History

To check if training history files exist:

```bash
# Windows
dir results\*history*.csv

# Linux/Mac
ls results/*history*.csv
```

You should see:
- `training_history_improved.csv` (or similar)
- `gcn_history.csv`
- `sage_history.csv`

## Troubleshooting

### Missing History Files

If `plot_all_results_extended.py` says files are missing:

1. **Check file locations**: History files should be in `results/` directory
2. **Check file names**: 
   - GCN: `results/gcn_history.csv`
   - SAGE: `results/sage_history.csv`
   - GAT: `results/training_history_improved.csv` or `training_history_improved (2).csv` in root

### Retrain Models

If you need to retrain any model:

```bash
# Retrain GCN
python train_gnn_variants.py --model_type gcn --epochs 15

# Retrain SAGE
python train_gnn_variants.py --model_type sage --epochs 15

# Retrain GAT models
python train.py
```

## Expected Output

After training all models, you should have:

**Models:**
- `results/models/model_improved_baseline.pt`
- `results/models/model_improved_affinity_weighted.pt`
- `results/models/model_gcn.pt`
- `results/models/model_sage.pt`

**Training History:**
- `results/training_history_improved.csv` (or root: `training_history_improved (2).csv`)
- `results/gcn_history.csv`
- `results/sage_history.csv`

**Visualizations:**
- `results/figures_extended/*.png` (8 visualization files)


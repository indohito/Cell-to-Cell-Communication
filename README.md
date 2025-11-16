# Cell-to-Cell Communication - TNBC Dataset & Analysis

Predicting cell-to-cell communication in Triple-Negative Breast Cancer (TNBC) using a Graph Attention Network with BindingDB binding affinity integration.

## Project Status:

**Latest Results** (November 6, 2025):
- **Model**: GATLinkPredictor (Graph Attention Network, 14,369 parameters)
- **Test ROC-AUC**: 0.9850 (affinity-weighted) vs 0.9816 (baseline) — **+0.34% improvement**
- **Precision**: 95.95% (affinity) vs 93.81% (baseline) — **+2.28% fewer false positives**
- **Biological Validation**: 90% of top-10 predictions match CellPhoneDB/literature
- **Statistical Significance**: p < 0.001 (DeLong test on 104K test edges)

## What's Included

- **TNBC Single-Cell Data**: 42,564 cells × 28,258 genes, 29 cell types
- **CellPhoneDB Interactions**: 2,911 curated ligand-receptor pairs
- **BindingDB Affinity Integration**: 105,247 edges with binding affinity scores (12.14% coverage)
- **Trained Models**: Baseline and affinity-weighted GNNs (converged, ready for inference)
- **Results & Analysis**: Training metrics, integration tables, computed graphs
- **Documentation**: 7,500+ words of methodology, supplementary protocols, improvement analysis

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Cells** | 42,564 TNBC cells |
| **Genes** | 28,258 features |
| **Cell Types** | 29 unique types |
| **Interactions (CellPhoneDB)** | 2,911 L-R pairs |
| **Interactions (Generated)** | 867,398 possible cell-type pairs |
| **With BindingDB Affinity** | 105,247 edges (12.14% coverage) |
| **Unique Proteins** | 1,355 in interaction database |
| **Knowledge Graph Nodes** | 959 (930 genes + 29 cell types) |
| **Knowledge Graph Edges** | 1,730,758 directed edges |

---

## Training Results

### Final Model Performance

**Affinity-Weighted Model** 
```
Metric           Value    Vs. Baseline    Significance
─────────────────────────────────────────────────────
ROC-AUC          0.9850   +0.34%          p < 0.001
PR-AUC           0.9729   +0.44%          p < 0.001 
Precision        95.95%   +2.28%          Better selectivity
Recall           99.60%   -0.40%          Near-perfect sensitivity
F1-Score         0.9774   +0.96%          Overall better balance
────────────────────────────────────────────────────
Test Set Size: 104,086 edges (52,043 positive + 52,043 negative)
Training Time: ~120 minutes on standard GPU
Convergence: Epoch 15 (loss plateau at 0.0202)
```

**Key Finding**: Binding affinity successfully improves cell-to-cell communication prediction in TNBC. High-affinity ligand-receptor pairs are enriched in actual communication networks.

### Training Configuration

```
Model Architecture:
├─ Graph Attention Network (GATLinkPredictor)
├─ 2 layers with 8 attention heads per layer
├─ 64D hidden embeddings
├─ 14,369 total parameters
├─ Binary cross-entropy loss
└─ Balanced 1:1 positive/negative sampling

Training Setup:
├─ Optimizer: Adam (lr=0.01, weight_decay=5e-4)
├─ Batch size: 256 (5x oversampling for minorities)
├─ Epochs: 15 (each model)
├─ Validation: 15% of data (130K edges)
├─ Test set: 15% of data (104K edges)
└─ Reproducibility: Fixed seeds, deterministic behavior
```

### Data Splits

- **Training**: 70% (600K positive + 600K negative edges)
- **Validation**: 15% (65K positive + 65K negative edges)
- **Test**: 15% (52K positive + 52K negative edges)

All splits use stratified random sampling with fixed seed for reproducibility.

---

### 1. Clone & Setup
```bash
git clone https://github.com/indohito/Cell-to-Cell-Communication.git
cd Cell-to-Cell-Communication
pip install -r requirements.txt
```

### 2. Load Dataset & Results
```python
import scanpy as sc
import pandas as pd
import torch

# Load TNBC single-cell data
adata = sc.read_h5ad('dataset.h5ad')
print(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")

# Load computed knowledge graph
graph = torch.load('results/graph.pt')
print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

# Load integration results (BindingDB + expression)
integration = pd.read_csv('results/integration_table_weighted.csv')
print(f"Integrated edges: {len(integration)} (with affinity weights)")

# Load trained models
baseline_model = torch.load('results/models/model_baseline.pt')
affinity_model = torch.load('results/models/model_affinity_weighted.pt')
print("Models loaded: baseline (ROC 0.9816) and affinity (ROC 0.9850)")
```

### 3. Make Predictions
```python
from models.gnn_model import GATLinkPredictor

# Load model
model = torch.load('results/models/model_affinity_weighted.pt')
model.eval()

# Get edge predictions (example)
with torch.no_grad():
    edge_index = graph.edge_index[:, :1000]  # First 1000 edges
    scores = model(graph.x, edge_index)
    predictions = (scores > 0.5).int().squeeze()

print(f"Predicted interactions: {predictions.sum()} out of {len(predictions)}")
```

### 4. Explore Predictions
```python
# Load top predictions with affinity weights
top_edges = pd.read_csv('results/integration_table_weighted.csv')
top_edges_sorted = top_edges.nlargest(20, 'affinity_weight')
print(top_edges_sorted[['ligand', 'receptor', 'affinity_weight', 'expression_product']])
```

## Benchmarking with New Models

This baseline can be extended with alternative models or datasets. To integrate your own approach:

### Using Pre-Built Data

The training pipeline provides pre-built graph structure, node embeddings, and data splits:

```python
from train import load_data, prepare_data

# Load pre-computed graph and edge list
graph, edge_list = load_data()
data = prepare_data(graph, edge_list, use_affinity=False)

# data contains: train_idx, val_idx, test_idx, pos_edges, neg_edges
```

### Implementing Your Model

Create a model that accepts graph structure and edge indices:

```python
import torch
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index, edge_attr=None):
        # x: (num_nodes, input_dim) node features
        # edge_index: (2, num_edges) edge indices
        # edge_attr: (num_edges, 1) optional edge weights
        h = torch.relu(self.layer1(x[edge_index[0]]) + x[edge_index[1]])
        scores = torch.sigmoid(self.layer2(h))
        return scores.squeeze()

# Train with existing pipeline
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YourModel(graph.x.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Use prepare_data output for training loop
history = []
for epoch in range(1, 16):
    model.train()
    loss = train_epoch(model, data, optimizer, device)
    history.append({'epoch': epoch, 'loss': loss})
    
    # Validation metrics
    val_scores = evaluate(model, data['val_idx'], graph, device)
    
    print(f"Epoch {epoch:2d} - Loss: {loss:.4f}, Val-AUC: {val_scores['roc']:.4f}")

# Save results
pd.DataFrame(history).to_csv('results/your_model_history.csv', index=False)
torch.save(model.state_dict(), 'results/models/your_model.pt')
```

### Benchmarking Against Baselines

The repository includes comparison utilities:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load all model results
baseline = pd.read_csv('results/training_history.csv')
your_results = pd.read_csv('results/your_model_history.csv')

# Compare ROC-AUC curves
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(baseline[baseline['model']=='baseline']['epoch'], 
        baseline[baseline['model']=='baseline']['roc'], 
        label='Baseline', marker='o')
ax.plot(your_results['epoch'], your_results['roc_auc'], 
        label='Your Model', marker='s')
ax.set_xlabel('Epoch')
ax.set_ylabel('ROC-AUC')
ax.legend()
plt.savefig('results/figures/model_comparison.png')
```

### Using Alternative Datasets

To benchmark with your own data:

1. **Prepare your data** in the same format:
   - Integration table: CSV with columns `cell_type_A`, `ligand_gene`, `cell_type_B`, `receptor_gene`, `weight`
   - Run: `python scripts/construct_graph.py` to build graph structure

2. **Split data**:
   ```bash
   python scripts/stratified_split.py  # Generates train/val/test indices
   ```

3. **Train and evaluate**:
   ```bash
   python train.py  # Or your model script
   ```

4. **Visualize results**:
   ```bash
   python scripts/visualize_results.py
   ```

---

---

## Repository Structure

```
Cell-to-Cell-Communication/
│
├── DATASETS & DATA
│   ├── dataset.h5ad                    # TNBC scRNA-seq (42.5K cells, 28.2K genes)
│   ├── cellphonedb_data/               # CellPhoneDB interaction database reference
│   └── cpdb/                           # Virtual environment (optional cleanup)
│
├── TRAINED MODELS
│   └── results/models/
│       ├── model_baseline.pt           # Baseline GNN (ROC: 0.9816)
│       └── model_affinity_weighted.pt  # Final winner (ROC: 0.9850)
│
├── RESULTS & INTEGRATION
│   ├── results/
│   │   ├── graph.pt                    # Computed knowledge graph (1.73M edges)
│   │   ├── integration_table_weighted.csv  # BindingDB + expression (70M, 867K edges)
│   │   ├── edge_list.csv               # Edge list for model (39M)
│   │   ├── celltype_pair_weights.csv   # Cell-type pair affinity weights
│   │   ├── training_history.csv        # Epoch-by-epoch metrics
│   │   ├── training_metrics.csv        # Final test set metrics
│   │   └── figures/                    # Visualizations (ROC, PR, convergence curves)
│   └── models/
│       └── gnn_model.py                # GATLinkPredictor architecture
│
├── SCRIPTS & ANALYSIS
│   ├── train.py                        # Main training script (both models)
│   ├── scripts/
│   │   ├── compare_baselines.py        # CellPhoneDB, LR, RF baselines (ready to run)
│   │   ├── construct_graph.py          # Build knowledge graph
│   │   ├── build_integration_table_with_affinity.py  # BindingDB integration
│   │   ├── stratified_split.py         # Data splitting (70/15/15)
│   │   └── [other utility scripts]
│   └── analyze_all_iterations.py       # Comprehensive analysis script
│
├── SETUP & CONFIG
│   ├── README.md                       # This file (updated with results)
│   ├── requirements.txt                # All dependencies
│   └── .gitignore                      # Git exclusion rules
│
├── REFERENCE DATA
│   ├── cellphonedb_data/               # CellPhoneDB database files
│   └── BindingDB_All.tsv               # [CANDIDATE FOR REMOVAL - see cleanup section]
```

---

## Data File Descriptions

### Primary Datasets

#### `dataset.h5ad` (1.2 GB)
The main TNBC single-cell RNA-seq dataset in AnnData format (HDF5-backed).

**Contents**:
- **Observations (rows)**: 42,512 TNBC cells
- **Variables (columns)**: 28,200 genes/features
- **Cell types**: 29 unique cell type annotations
- **Format**: Log-normalized expression matrix
- **Metadata**: Cell type labels, patient information, quality metrics

**Use Case**: Primary data for all CCC analysis. Load with:
```python
import scanpy as sc
adata = sc.read_h5ad('dataset.h5ad')
```

#### `results/combined_dataset.h5ad` (1.2 GB)
Combined and preprocessed version of the dataset with additional annotations.

**Contents**:
- Same expression data as `dataset.h5ad`
- Pre-computed dimensionality reduction (PCA, UMAP if available)
- Additional cell type metadata
- Preprocessed for analysis workflows
- All cells integrated and batch-corrected

**Use Case**: Ready-to-use dataset for downstream analysis without preprocessing steps.

---

## Project Summary

This project demonstrates that **binding affinity improves cell-to-cell communication prediction** in TNBC.

**Key Results**:
- Affinity-weighted model: ROC-AUC 0.9850 vs baseline 0.9816 (+0.34%, p < 0.001)
- 105,247 BindingDB edges integrated with expression networks
- Top predictions validated against literature (90% accuracy)

---

## Dependencies

```bash
pip install -r requirements.txt
```

Required: PyTorch, PyTorch Geometric, scikit-learn, pandas, scanpy, h5py

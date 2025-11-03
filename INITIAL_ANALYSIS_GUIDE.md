
# Combined Dataset - Initial Analysis Guide

## Files Included

### 1. **combined_dataset.h5ad** (Main file)
- **Format**: AnnData object (scanpy standard)
- **Size**: ~1.2 GB (same as original)
- **Cells**: 42,512 (TNBC patient cells)
- **Genes**: 28,200 (features)
- **Cell Types**: 29 unique types

**How to use:**
```python
import scanpy as sc
adata = sc.read_h5ad('results/combined_dataset.h5ad')

# Access cell metadata
print(adata.obs['cell_type'].value_counts())

# Access gene info
print(adata.var[['gene_name', 'uniprot', 'hvg']])

# Access metadata
print(adata.uns['cells'])
```

### 2. **interaction_summary.csv**
- **Rows**: 2,911 unique ligand-receptor interactions
- **Columns**: partner_a, partner_b, protein_name_a, protein_name_b, pair_id
- **Use**: Understand what interactions are possible between cells

**Sample:**
```
partner_a | partner_b | protein_name_a | protein_name_b
P12830    | ITGA2B1   | CADH1_HUMAN   | ITGA2B1_HUMAN
...
```

### 3. **protein_summary.csv**
- **Rows**: 1,355 unique proteins
- **Columns**: uniprot, protein_name, transmembrane, receptor, secreted
- **Use**: Understand protein properties for filtering

**Distribution:**
- Transmembrane proteins: 357
- Receptor proteins: 289
- Secreted proteins: 221

### 4. **dataset_metadata.json**
- **Content**: Structured metadata about the combined dataset
- **Keys**: cells, genes, interactions, proteins
- **Use**: Quick reference for data dimensions

**Example:**
```json
{
  "cells": {
    "total": 42512,
    "n_cell_types": 29,
    "cell_types": ["T.cells.CD4", "T.cells.CD8", ...]
  },
  "interactions": {
    "total": 2911,
    "unique_pairs": 2711
  }
}
```

### 5. **data_summary_statistics.csv**
- **Quick reference** for key metrics
- Open in Excel or view with `cat`

## Key Insights

### Cell Composition
- **Most abundant**: T.cells.CD4 (6,890 cells, 16.2%)
- **Second**: T.cells.CD8 (5,299 cells, 12.5%)
- **Third**: Cancer.Cycling (4,372 cells, 10.3%)
- **Rarest**: Cycling.PVL (20 cells, 0.05%)

### Gene Information
- **Total features**: 28,200
- **Highly variable**: ~5,000 (identified by scRNA-seq analysis)
- **Mapped to UniProt**: ~15,000 (53%)

### Interaction Network
- **Total interactions**: 2,911
- **Unique pairs**: 2,711 (some pairs have multiple interaction types)
- **Ligand-receptor concept**: Cell A (expressing ligand) → Cell B (expressing receptor)

### Protein Types
- **Transmembrane**: 357 (membrane-spanning proteins)
- **Receptors**: 289 (receive signals)
- **Secreted**: 221 (extracellular signaling)

## Getting Started

### 1. Load and Explore
```python
import scanpy as sc
import pandas as pd

# Load dataset
adata = sc.read_h5ad('results/combined_dataset.h5ad')

# Explore cells
print(f"Shape: {adata.shape}")
print(f"Cell types: {adata.obs['cell_type'].nunique()}")

# Explore genes
print(adata.var[['gene_name', 'uniprot', 'hvg']].head(10))

# Check for interactions
interactions = pd.read_csv('results/interaction_summary.csv')
print(f"Total interactions: {len(interactions)}")
```

### 2. Analyze Cell Type Interactions
```python
# What cell types are present?
cell_types = adata.obs['cell_type'].unique()
print(f"All {len(cell_types)} cell types:")
for ct in sorted(cell_types):
    count = (adata.obs['cell_type'] == ct).sum()
    print(f"  - {ct}: {count} cells")

# Get expression of specific genes
genes_of_interest = ['TNF', 'IL6', 'IFNG']
for gene in genes_of_interest:
    if gene in adata.var['gene_name'].values:
        expr = adata[:, adata.var['gene_name'] == gene].X
        print(f"{gene}: mean={expr.mean():.3f}, max={expr.max():.3f}")
```

### 3. Link Genes to Interactions
```python
# Find which genes in dataset participate in interactions
proteins = pd.read_csv('results/protein_summary.csv')
interactions = pd.read_csv('results/interaction_summary.csv')

# Get all UniProts involved in interactions
interacting_uniprots = set(
    interactions['partner_a'].dropna().unique().tolist() +
    interactions['partner_b'].dropna().unique().tolist()
)

# Find overlaps with dataset
gene_uniprots = adata.var['uniprot'].dropna().unique()
overlap = gene_uniprots.intersection(interacting_uniprots)
print(f"Genes in interactions: {len(overlap)} / {len(gene_uniprots)}")

# Filter for genes with interactions
adata_filtered = adata[:, adata.var['uniprot'].isin(overlap)]
print(f"Filtered dataset shape: {adata_filtered.shape}")
```

### 4. Exploratory Data Analysis (EDA)**
```python
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os

adata = sc.read_h5ad('results/combined_dataset.h5ad')
counts = adata.obs['cell_type'].value_counts().sort_values(ascending=True)
os.makedirs('results/figures', exist_ok=True)

# Save file (optional)
counts.to_csv('results/cell_type_counts.csv', header=['count'])

# Draw a bar chart (horizontal)
plt.figure(figsize=(8, max(4, len(counts) * 0.25)))
counts.plot(kind='barh', color='C0')
plt.xlabel('Cell count')
plt.title('Cell counts per cell_type')
plt.tight_layout()
plt.savefig('results/figures/cell_type_counts.png', dpi=150)
print('Wrote results/cell_type_counts.csv and results/figures/cell_type_counts.png')
```

## Next Steps

1. **Exploratory Data Analysis (EDA)**
   - Distribution of cell types （completed）
   - Gene expression patterns per cell type
   - PCA/UMAP visualization

2. **Build Interaction Network**
   - Match expressed genes to interactions
   - Create cell-type-level communication graph
   - Weight edges by expression levels

3. **Train Predictive Models**
   - Use datasets in training scripts
   - Compare different interaction definitions
   - Validate predictions against CCC benchmarks

4. **Validate Results**
   - Compare to known biological pathways
   - Check against experimental data (if available)
   - Benchmark against other methods

## Data Quality Notes

- 42,512 cells (high quality, filtered)
- 28,200 genes (raw count space, no filtering applied yet)
- Cell type labels: curated, consistent annotation system
- CellPhoneDB: v4.0+ (latest resource)
- Gene-to-UniProt mapping: ~53% coverage (normal for scRNA-seq)
- Not all interactions present in dataset (genes may not be expressed)

# Cell-to-Cell Communication - TNBC Dataset & Analysis

This repository contains datasets and analysis resources for studying cell-to-cell communication (CCC) in Triple-Negative Breast Cancer (TNBC).

## What's Included

- **TNBC Single-Cell Data**: 42.5k cells, 28.2k genes, 29 cell types
- **CellPhoneDB Interactions**: 2.9k ligand-receptor pairs
- **Analysis Guides**: Getting started with CCC research

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Cells** | 42,512 TNBC cells |
| **Genes** | 28,200 features |
| **Cell Types** | 29 unique types |
| **Interactions** | 2,911 L-R pairs |
| **Proteins** | 1,355 unique proteins |

---

### 1. Clone & Setup
```bash
git clone <repo-url>
cd Cell-to-Cell-Communication
pip install -r requirements.txt
```

### 2. Load Dataset
```python
import scanpy as sc
adata = sc.read_h5ad('dataset.h5ad')
print(adata)  # View data structure
print(adata.obs['cell_type'].value_counts())  # Cell type distribution
```

### 3. Explore Interactions
```python
import pandas as pd
interactions = pd.read_csv('results/interaction_summary.csv')
print(f"Total interactions: {len(interactions)}")
print(interactions.head())
```

---

## Repository Structure

```
Cell-to-Cell-Communication/
├── Datasets
│   ├── dataset.h5ad                 # TNBC single-cell data
│   └── cellphonedb_data/            # Ligand-receptor interactions
│
├── Documentation
│   ├── README.md                 
│   └── INITIAL_ANALYSIS_GUIDE.md    # Data exploration guide
│
├── Analysis Tools
│   └── scripts/                     # Exploration & visualization utilities
│
├── Configuration
│   ├── .gitignore
│   └── requirements.txt
│
└── Results (Analysis Data)
    ├── combined_dataset.h5ad
    ├── interaction_summary.csv
    ├── protein_summary.csv
    ├── dataset_metadata.json
    └── data_summary_statistics.csv
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

### Reference Data: CellPhoneDB Ligand-Receptor Interactions

#### `cellphonedb_data/` Directory
Complete CellPhoneDB database files for ligand-receptor interaction analysis.

**Files in this directory**:

##### `interaction_input.csv`
**Columns**: id_cp_interaction, partner_a, partner_b, protein_name_a, protein_name_b, annotation_strategy, source, is_ppi, curator, reactome_complex, reactome_reaction, reactome_pathway, complexPortal_complex, comments, version, interactors, classification, directionality, modulatory_effect

**Description**: 
- Complete ligand-receptor pair definitions
- 2,911 curated human L-R interactions
- Each row is one interaction pair
- Includes PPI (protein-protein interaction) flags
- Source and curation information for each interaction
- Classification (e.g., Adhesion, Signaling, ECM-Receptor)
- Directionality indicates paracrine communication direction

**Example**:
```
CDH1 (partner_a) + ITGA2+ITGB1 (partner_b) = Adhesion-type interaction
```

##### `protein_input.csv`
**Columns**: uniprot, protein_name, transmembrane, peripheral, secreted, secreted_desc, receptor, receptor_desc, integrin, biosynthetic_pathway_desc, other, tags

**Description**:
- 1,355 unique proteins involved in L-R interactions
- Protein properties for functional classification:
  - **transmembrane**: T/F - Is protein embedded in cell membrane?
  - **secreted**: T/F - Is protein secreted into extracellular space?
  - **receptor**: T/F - Is protein a known receptor?
  - **integrin**: T/F - Is protein an integrin family member?
  - **peripheral**: T/F - Is protein peripherally associated with membrane?
- UniProt accession and standardized protein names

**Use Case**: Filter interactions by protein type (e.g., secreted ligand → membrane receptor)

##### `gene_input.csv`
**Columns**: gene_name, uniprot, hgnc_symbol, ensembl

**Description**:
- Gene ID mapping for the 28,200 genes in the dataset
- Columns:
  - **gene_name**: Original gene identifier
  - **uniprot**: UniProt protein accession (maps to proteins in interaction database)
  - **hgnc_symbol**: HGNC gene symbol (standardized human gene nomenclature)
  - **ensembl**: Ensembl gene ID (Ensembl database identifier)
- Used to link genes in your data to proteins in CellPhoneDB

**Use Case**: Convert gene expression data to protein identifiers for interaction analysis

##### `complex_input.csv`
**Description**:
- Protein complexes (multi-subunit proteins treated as single entities)
- Used when ligand-receptor pairs involve multi-subunit complexes
- Example: "integrin_a2b1_complex" is a 2-subunit complex

##### `sources/` Subdirectory
Additional reference files:
- **transcription_factor_input.csv**: Transcription factors and their properties
- **uniprot_synonyms.tsv**: UniProt accession ID synonyms and mappings

---

### Analysis Summary Files

#### `results/interaction_summary.csv` (188 KB)
**Columns**: partner_a, partner_b, protein_name_a, protein_name_b, pair_id

**Description**:
- Summary of all L-R interaction pairs
- Each row is one unique interaction
- Columns:
  - **partner_a**: UniProt ID of ligand (interaction initiator)
  - **partner_b**: UniProt ID of receptor (interaction receiver)
  - **protein_name_a**: Standardized name of ligand protein
  - **protein_name_b**: Standardized name of receptor protein
  - **pair_id**: Unique pair identifier (e.g., "P12830_integrin_a2b1_complex")
- Used for filtering and querying specific interactions

**Use Case**: Quick lookup of interaction partners or filtering by specific proteins

**Example**:
```python
interactions = pd.read_csv('results/interaction_summary.csv')
# Find all interactions involving CDH1
cad_interactions = interactions[interactions['protein_name_a'] == 'CADH1_HUMAN']
```

#### `results/protein_summary.csv` (46 KB)
**Columns**: uniprot, protein_name, transmembrane, receptor, secreted

**Description**:
- Summary of protein functional properties
- Each row is one unique protein in the interaction database
- Boolean columns (True/False) for protein classification:
  - **transmembrane**: Integral membrane protein?
  - **receptor**: Capable of binding signals?
  - **secreted**: Released into extracellular environment?
- Simplified version of `cellphonedb_data/protein_input.csv` with key properties only

**Use Case**: Filter interactions by protein type (e.g., "secreted ligands" or "membrane receptors")

**Example**:
```python
proteins = pd.read_csv('results/protein_summary.csv')
# Find all secreted proteins
secreted = proteins[proteins['secreted'] == True]
# Find all membrane receptors
receptors = proteins[proteins['receptor'] == True]
```

#### `results/dataset_metadata.json` (8.3 KB)
**Format**: JSON (JavaScript Object Notation)

**Contents**:
- Metadata about the TNBC dataset
- Typical fields:
  - Cell counts by type
  - Gene statistics
  - Data preprocessing information
  - Cell type composition
  - Data quality metrics

**Use Case**: Quick reference for dataset characteristics

**Example**:
```python
import json
with open('results/dataset_metadata.json') as f:
    metadata = json.load(f)
    print(f"Total cells: {metadata['total_cells']}")
    print(f"Cell types: {metadata['n_cell_types']}")
```

#### `results/data_summary_statistics.csv` (varies)
**Columns**: Metric, Value

**Description**:
- Quick reference statistics table
- Human-readable format for reporting
- Typical metrics:
  - Total Cells: 42,512
  - Cell Types: 29
  - Total Genes: 28,200
  - L-R Interaction Pairs: 2,911
  - Unique Proteins: 1,355

**Use Case**: Quick facts for presentations or documentation

**Example**:
```
Metric,Value
Total Cells,42512
Cell Types,29
Total Genes,28200
Interaction Pairs,2911
Unique Proteins,1355
```

---

## How to Explore the Data

### Load & Inspect TNBC Data
```python
import scanpy as sc
adata = sc.read_h5ad('dataset.h5ad')

# Basic info
print(adata)
print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")

# Cell type distribution
print(adata.obs['cell_type'].value_counts())

# Gene info
print(adata.var.head())
```

### Explore Ligand-Receptor Interactions
```python
import pandas as pd

# Load interactions
interactions = pd.read_csv('results/interaction_summary.csv')
print(f"Total interactions: {len(interactions)}")

# Load protein info
proteins = pd.read_csv('results/protein_summary.csv')
print(proteins.head())
```

### Load Metadata
```python
import json
with open('results/dataset_metadata.json') as f:
    metadata = json.load(f)
    print(metadata)
```

---

## Analysis Guide

For detailed exploration instructions, see: **INITIAL_ANALYSIS_GUIDE.md**

Topics covered:
- Combined dataset structure
- Cell type composition
- Gene information & mapping
- Interaction networks
- Protein properties

---

## Data Sources & References

- **TNBC Dataset**: Single-cell RNA-seq from OpenProblems project
- **CellPhoneDB**: Efremova et al., 2020 - Curated human L-R interactions
- **Gene IDs**: UniProt mapping for consistency

---

## Next Steps

1. **Explore**: Use scripts in `scripts/` for visualization
2. **Analyze**: Examine cell-cell communication patterns
3. **Plan**: Design validation approach or machine learning pipeline

---

<!-- ## Next Steps (Prioritized)

### High Priority
1. **Log-scaling edge weights** - Handle sparsity
2. **OpenProblems format** - Convert output
3. **Hard negative mining** - Better training signal

### Medium Priority
4. **5-fold cross-validation** - Statistical rigor
5. **Bayesian hyperparameter optimization**
6. **Some ablation studies**

### Low Priority
7. Attention weight analysis for interpretability
8. Test on official OpenProblems TNBC dataset
9. Submit -->

---

### First Time Setup
```bash
# Clone repo
git clone <url>
cd Cell-to-Cell-Communication

# Install dependencies
pip install -r requirements.txt

# Run quick validation
python quick_test.py

# Explore results
python analyze_all_iterations.py

---

## Dependencies

- **Python**: 3.11+
- **PyTorch**: 2.0+
- **PyG (PyTorch Geometric)**: 2.3+
- **scikit-learn**: For metrics and splitting
- **pandas**: Data manipulation
- **scipy**: Spectral decomposition (Laplacian)
- **h5py**: Reading .h5ad files

Install all:
```bash
pip install -r requirements.txt
```
# AffinityCCI: Biochemical Prioritization of Cell-Cell Communication

A Graph Attention Network that integrates transcriptomic potential with proteomic reality to infer functional signaling in Triple-Negative Breast Cancer.

## Project Overview

Standard methods for inferring Cell-Cell Communication (CCC) rely heavily on gene expression abundance (e.g., Ligand × Receptor counts). However, high expression does not always equal functional signaling. Many "sticky" structural proteins are highly expressed but functionally static, while potent signaling cytokines often have low transcript counts.

AffinityCCI bridges this gap by integrating Binding Affinity (Kd) priors directly into a Graph Neural Network. By weighting the graph attention mechanism with biochemical data, the model acts as a functional filter, "rescuing" high-affinity drivers that are otherwise obscured by single-cell dropout noise.

## Key Results

We compared AffinityCCI against a standard unweighted GAT (Baseline) on a precision graph of the TNBC microenvironment.

| Metric | Baseline (Topology Only) | AffinityCCI (Ours) | Impact |
|--------|--------------------------|-------------------|--------|
| Affinity Learning (MSE) | 0.30 → 1.97 (Divergent) | 0.0372 (Convergent) | 87% Reduction in Error. Prevents semantic drift. |
| PD-L1/PD-1 Rank | Rank 347 | Rank 90 | +257 Positions. Successful rescue of immune checkpoint. |
| Structural Noise | High Priority | Suppressed | Demotes high-expression/low-affinity adhesion molecules. |

**Biological Validation:** The model autonomously identified the PD-L1 (CD274) -- PD-1 (PDCD1) axis—a Nobel-winning immunotherapy target—as a top 100 driver, correcting the false negative produced by standard expression analysis.

## Repository Structure

```
AffinityCCI/
├── src/
│   ├── build_precision_graph.py   # Complex explosion + Hybrid Injection
│   ├── train_experiment_v2.py     # Training Loop (Baseline vs. AffinityCCI comparison)
│   ├── biological_validation.py   # Rank Shift Analysis (The "So What?" script)
│   └── models/
│       └── egnn_model.py          # Affinity-Weighted GAT Architecture
├── scripts/
│   └── *.sh                       # Slurm job scripts
├── results/
│   ├── biological_ranking.csv     # Final ranked list of L-R interactions
│   ├── top_drivers.csv            # Top 50 predicted high-affinity pairs
│   └── plots/                     # Visualization of results
└── dataset.h5ad                   # (Excluded via .gitignore) TNBC Single-Cell Atlas
```

## Methodology

### 1. Hybrid Graph Construction & Data Integration

We constructed a Precision Graph by integrating heterogeneous biological databases. Crucially, we utilized idmapping_2025_11_20.tsv to bridge the nomenclature gap between genomic data (Gene Symbols) and proteomic data (UniProt IDs).

**Topological Backbone:** We aggregated physical interactions from multiple sources to ensure high coverage. The core signaling topology is derived from CellPhoneDB, supported by broad protein-protein interaction (PPI) networks from BioGRID (BIOGRID-ORGANISM-Homo_sapiens-5.0.251.tab3.txt) and IntAct (intact-micluster.txt).

**Node Resolution:** We implemented a "Complex Explosion" algorithm to resolve heteromeric complexes (e.g., TGFBR1/2) into their constituent physical protein nodes using the complex_input.csv definitions.

**Affinity Injection:** We populated edge attributes using a hybrid strategy:
- **Gold Standard:** Experimental dissociation constants (Kd) parsed from BindingDB_All.tsv
- **Silver Standard:** For edges without exact Kd matches, we utilized confidence scores from intact-micluster.txt as a proxy for binding probability

**Node Features:** 1,337 Protein nodes initialized with mean expression values from the TNBC scRNA-seq atlas (dataset.h5ad).

### 2. Model Architecture

**Affinity-Weighted GAT:** Unlike standard GNNs that treat edges as binary links, our architecture injects the edge attribute vector into the attention mechanism directly into learned attention weights. This allows the model to dynamically upweight neighbors with strong biochemical binding potential.

The architecture consists of:
- Input encoder: Gene expression to hidden dimension (128)
- 3-layer GAT with affinity-weighted edge embeddings
- Dual prediction heads: link prediction (BCE) and affinity prediction (MSE)
- Learnable attention weights that incorporate biochemical priors

### 3. Evaluation Strategy

Instead of standard classification metrics (ROC-AUC) which are ill-suited for intensity prediction, we used:

- **Affinity MSE:** To quantify the learning of chemical rules
- **Biological Rank Shift:** ΔR = Rank_Baseline - Rank_Affinity. This metric isolates the specific contribution of the affinity prior to the biological prioritization of pathways

## Results Visualization

### The "Rescue" Effect

(See `results/plots/rank_shift_dumbbell.png`)

The dumbbell plot demonstrates the shift in interaction priority. Red lines indicate pathways "rescued" by their high affinity (e.g., Chemokines, Checkpoints), while blue lines indicate "suppressed" structural noise (e.g., Collagens).

### Preventing Semantic Drift

(See `results/plots/mse_comparison.png`)

Without affinity priors, the Baseline model suffers from feature over-smoothing, causing affinity prediction error to explode over time. AffinityCCI remains chemically grounded, converging to a stable low error.

## Usage

**1. Environment Setup**

```bash
# Requires Python 3.11 + PyTorch 2.5.1 + CUDA 12.1
bash setup_conda.sh
```

**2. Build Graph**

```bash
python src/build_precision_graph.py
```

**3. Run Experiment**

```bash
python src/train_experiment_v2.py
```

**4. Generate Biological Ranks**

```bash
python src/biological_validation.py
```

## Contributors

- **Project Team:** Akash Shetty, Troy Zhou, Jinho Lee, Oviyan Anabarasu, Kangyu Sun
- **Course:** BIOINF 593 (Machine Learning in Computational Biology)
- **Data Sources:** Open Problems (TNBC), CellPhoneDB, BindingDB, IntAct
